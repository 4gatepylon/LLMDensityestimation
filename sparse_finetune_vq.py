#!/usr/bin/env python3
"""
Fine-tune an already-trained Residual-VQ model with a progressive ℓ¹
penalty, saving checkpoints and PCA plots at the requested frequency.

Assumes the *base* implementation lives in coder.py.

Example
-------
python sparse_finetune_vq.py \
       --checkpoint residual_vq_model.pt \
       --data_glob './gpt_activations/*.npy' \
       --epochs 20 --l1_init 0.0 --l1_final 1e-3 --l1_warm 8 \
       --save_every 2 --plot_every 2 --run_name sparse-v2
"""

import argparse, glob, random, gc, os, pathlib
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb

# ←–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# IMPORT THE UTILITIES FROM coder.py  (same folder)
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––→
from vq_scripts.coder import (
    ResidualVQ,      # stacked VQ module
    NPZDataset,      # helper dataset that reads *.npy rows
    ProjectedVQ      # Need this for type hint in plot function
    # _plot_residual_vq_codebooks   # PCA / analysis figure <-- Don't import this, use local copy
)
# Import plotting libraries needed for the local copy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.gridspec as gridspec
import numpy as np
import math # For kaiming_uniform_ init

# Helper function to infer structure from old state_dict
def infer_hyperparameters(state_dict):
    print("Inferring model structure from old state_dict...")
    inferred_h = {}
    max_stage = -1
    book_keys_stage0 = []

    # Find max stage index and keys for stage 0
    for key in state_dict.keys():
        if key.startswith('blocks.'):
            try:
                parts = key.split('.')
                stage_idx = int(parts[1])
                max_stage = max(max_stage, stage_idx)
                if stage_idx == 0:
                    book_keys_stage0.append(key)
            except (IndexError, ValueError):
                print(f"Warning: Could not parse stage index from key: {key}")
                continue

    if max_stage == -1:
        raise ValueError("Could not determine number of stages from state_dict keys.")
    inferred_h['stages'] = max_stage + 1
    print(f"  Inferred stages: {inferred_h['stages']}")

    # Infer other params from stage 0 keys/shapes
    try:
        # Infer dim and num_books from biases or projector
        if 'blocks.0.biases' in state_dict:
            bias_shape = state_dict['blocks.0.biases'].shape
            inferred_h['num_books'] = bias_shape[0]
            inferred_h['dim'] = bias_shape[1]
        elif 'blocks.0.projector' in state_dict:
            proj_shape = state_dict['blocks.0.projector'].shape
            inferred_h['num_books'] = proj_shape[0]
            inferred_h['dim'] = proj_shape[2]
        else:
             raise KeyError("Could not find 'blocks.0.biases' or 'blocks.0.projector' to infer dim/books.")
        print(f"  Inferred num_books: {inferred_h['num_books']}")
        print(f"  Inferred dim: {inferred_h['dim']}")

        # Infer num_codes and rank from codebook
        if 'blocks.0.codebook' in state_dict:
            cb_shape = state_dict['blocks.0.codebook'].shape
            if cb_shape[0] != inferred_h['num_books']:
                 print(f"Warning: num_books mismatch between codebook ({cb_shape[0]}) and bias/projector ({inferred_h['num_books']}). Using codebook shape.")
                 inferred_h['num_books'] = cb_shape[0]
            inferred_h['num_codes'] = cb_shape[1]
            inferred_h['rank'] = cb_shape[2]
        else:
            raise KeyError("Could not find 'blocks.0.codebook' to infer codes/rank.")
        print(f"  Inferred num_codes: {inferred_h['num_codes']}")
        print(f"  Inferred rank: {inferred_h['rank']}")

        # Make reasonable guesses for other params not inferrable from state_dict
        inferred_h['k_init'] = 0.1 # Cannot infer initial value
        inferred_h['gamma'] = 0.03 # Cannot infer reliably
        inferred_h['orth_lambda'] = 0.0 # Cannot infer reliably
        inferred_h['normalize'] = False # Cannot infer reliably
        inferred_h['l1_threshold'] = 1e-3 # Default, will be overridden by CLI arg anyway
        inferred_h['temp_target'] = 0.1 # Cannot infer reliably
        inferred_h['temp_factor'] = 0.001 # Cannot infer reliably
        print("  Set defaults for non-inferrable hyperparameters (gamma, orth, normalize, etc.)")

    except Exception as e:
        raise ValueError(f"Failed to infer hyperparameters from state_dict: {e}")

    return inferred_h

# ─────────────────────────── CLI ────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description="Fine-tune a ResidualVQ model with sparsity, inferring structure from checkpoint.")
    # --- Required ---
    p.add_argument('--checkpoint', default='/workspace/final_sparse_7err.pt', help='Path to the *.pt model checkpoint (can be old state_dict only or new dict format).')
    p.add_argument('--data_glob', default='./gpt_activations/*.npy', help='Path glob to training data files (*.npy).')

    # --- Fine-tuning specific (Structure is inferred) ---
    p.add_argument('--l1_init',  type=float, default=0.0, help='Initial L1 penalty weight.')
    p.add_argument('--l1_final', type=float, default=0, help='Final L1 penalty weight.') # 1e-4 is a lot, gives 17% error, while 1e-5 is decent. 
    p.add_argument('--l1_warm',  type=int,   default=10, help='Epochs to ramp up L1 penalty.')
    p.add_argument('--l1_threshold', type=float, default=1e-3, help='L1 norm threshold for declaring codes/books dead during this fine-tuning run.')
    p.add_argument('--freeze_temp', default=True, action='store_true', help='Freeze temperature (k) parameters during finetuning.')

    # bookkeeping
    p.add_argument('--epochs', type=int, default=20, help='Number of fine-tuning epochs.')
    p.add_argument('--batch',  type=int, default=128, help='Batch size.')
    p.add_argument('--lr',     type=float, default=5e-4, help='Learning rate.')
    p.add_argument('--device', default='cuda', help='Device to use (e.g., cuda, cpu).')
    p.add_argument('--save_every', type=int, default=2, help='Save checkpoint frequency (epochs).')
    p.add_argument('--plot_every', type=int, default=2, help='Plot PCA frequency (epochs).')
    p.add_argument('--ckpt_dir', default='./checkpoints_sparse', help='Directory for saving fine-tuned checkpoints.')
    p.add_argument('--pca_dir',  default='./codebook_pca_sparse', help='Directory for saving PCA plots.')
    p.add_argument('--project',  default='projected-vq', help='WandB project name.')
    p.add_argument('--run_name', default='sparse-finetune', help='WandB run name.')

    # --- Removed structural args (now inferred) ---
    # p.add_argument('--stages', type=int,  default=3)
    # p.add_argument('--books\',  type=int, default=200)
    # ... etc ...

    return p.parse_args()

# ─────────────────────── forward patch ──────────────────────
# Add a wrapper so we can pass l1_weight into every stage
def add_l1_forward():
    def forward_with_l1(self, x, *, return_prob=False):
        flat = x.view(-1, x.size(-1))
        total_recon, total_loss = torch.zeros_like(flat), 0.0
        residual, cum, probs = flat, [], []
        E = flat.size(0) # Get batch size (E) for normalization
        for blk in self.blocks:
            # Get standard loss, recon, and probs from the block
            loss, recon, prob = blk(residual, return_prob=True)
            
            total_recon += recon
            residual = residual - recon.detach()
            total_loss  += loss # Accumulate the original block loss
            cum.append(total_recon.view_as(x))
            probs.append(prob)
        out = (total_loss, total_recon.view_as(x), cum)
        if return_prob: out += (probs,)
        return out
    ResidualVQ.forward_with_l1 = forward_with_l1

# ─────────────────── LOCAL PLOTTING FUNCTION (MODIFIED) ────────────────────
def _plot_residual_vq_codebooks_sparse(vq_stage: ProjectedVQ, usage: torch.Tensor, ep: int, stage_idx: int, args):
    """Generates PCA plots, analysis heatmaps, and L1 norm plots for the specified VQ stage during sparse finetuning."""
    os.makedirs(args.pca_dir, exist_ok=True)
    B, C, r, d = vq_stage.B, vq_stage.C, vq_stage.r, vq_stage.d
    num_plot_books = min(B, 20) # Limit PCA plots and bar chart to first 20 books

    # --- Get Data ---
    probs = (usage / usage.sum(1, keepdim=True).clamp_min(1e-9)).cpu().numpy()
    codes = vq_stage.codebook.detach().cpu().numpy() # [B, C, r]
    k_values = vq_stage.k.detach().cpu().numpy() # [B]
    biases = vq_stage.biases.detach().cpu().numpy() # [B, d]

    # Calculate Bias Norms
    bias_norms = np.linalg.norm(biases, axis=1)

    # --- Calculate Code L1 Norm Statistics --- 
    book_l1_means = []
    book_l1_q25 = []
    book_l1_q75 = []
    for b in range(B):
        book_codes = codes[b] # Shape (C, r)
        if C < 1:
            book_l1_means.append(0)
            book_l1_q25.append(0)
            book_l1_q75.append(0)
            continue
        
        # Calculate L1 norm for each code in the book
        code_l1_norms = np.linalg.norm(book_codes, ord=1, axis=1) # L1 norm along the rank dim 'r'
        
        if len(code_l1_norms) == 0:
            book_l1_means.append(0)
            book_l1_q25.append(0)
            book_l1_q75.append(0)
        else:
            book_l1_means.append(np.mean(code_l1_norms))
            q25, q75 = np.percentile(code_l1_norms, [25, 75])
            book_l1_q25.append(q25)
            book_l1_q75.append(q75)
            
    book_l1_means = np.array(book_l1_means)
    book_l1_q25 = np.array(book_l1_q25)
    book_l1_q75 = np.array(book_l1_q75)

    # --- Setup Figure using GridSpec ---
    fig = plt.figure(figsize=(22, 20)) # Adjusted figure size for 2x2 bottom grid
    fig.suptitle(f"Epoch {ep+1} Sparse Finetune VQ Analysis (Stage {stage_idx})", fontsize=16, y=0.99)

    # Main grid: 6 rows (4 for PCA, 2 for bottom plots), 5 columns for PCA + 1 narrow for PCA colorbar
    gs_main = gridspec.GridSpec(6, 6, figure=fig,
                                height_ratios=[1, 1, 1, 1, 0.5, 0.5], # Give bottom rows appropriate height
                                width_ratios=[1, 1, 1, 1, 1, 0.15], # 5 cols for plots, 1 for PCA cbar
                                hspace=0.6, wspace=0.4, # Adjust spacing
                                left=0.05, right=0.95, top=0.95, bottom=0.05) # Adjust margins

    pca_axes = []
    for i in range(4):
        for j in range(5):
            pca_axes.append(fig.add_subplot(gs_main[i, j]))

    plotted_sm_pca = None
    # all_book_means_pca = [] # Store means just for the PCA plot - REPLACED BELOW
    # plotted_count = 0 - REPLACED BELOW
    # books_checked = 0 - REMOVED

    # --- Book Selection Logic ---
    is_book_dead_np = vq_stage.is_book_dead.cpu().numpy()
    is_code_dead_np = vq_stage.is_code_dead.cpu().numpy()

    alive_book_indices = [i for i, dead in enumerate(is_book_dead_np) if not dead]
    if not alive_book_indices:
        print("No alive codebooks to plot.")
        books_to_plot_indices = []
    else:
        # Calculate active codes for each alive book
        alive_books_with_counts = []
        for b_idx in alive_book_indices:
            active_count = (~is_code_dead_np[b_idx]).sum()
            alive_books_with_counts.append((b_idx, active_count))

        # Sort by active count (descending)
        alive_books_with_counts.sort(key=lambda item: item[1], reverse=True)

        # Select top 10
        top_10_indices = [item[0] for item in alive_books_with_counts[:10]]

        # Select random 10 from the rest
        remaining_alive_indices = [item[0] for item in alive_books_with_counts[10:]]
        random.shuffle(remaining_alive_indices)
        random_10_indices = remaining_alive_indices[:10]

        books_to_plot_indices = top_10_indices + random_10_indices
        print(f"Selected {len(books_to_plot_indices)} books to plot based on activity: {books_to_plot_indices}")

    # --- Initialize Plotting Variables ---
    # Calculate vmin/vmax for probability color scale based on selected books
    valid_probs_selected = []
    if books_to_plot_indices:
        probs_selected = probs[books_to_plot_indices].flatten()
        valid_probs_selected = probs_selected[np.isfinite(probs_selected)]

    vmin_prob = 0
    vmax_prob = np.percentile(valid_probs_selected, 99) if len(valid_probs_selected) > 0 else 1
    if not np.isfinite(vmax_prob) or vmax_prob == 0: vmax_prob = 1

    all_book_means_pca = [np.full(r, np.nan)] * num_plot_books # Init with NaNs, size=num_plot_books (20)
    successful_plot_indices = [] # Store indices `b` of plotted books
    plotted_count = 0

    # --- Plotting Loop (using selected books) ---
    for b in books_to_plot_indices:
        if plotted_count >= num_plot_books:
            break # Stop if we have already filled the plot slots

        # Target axis for the next successful plot
        ax = pca_axes[plotted_count]
        ax.set_facecolor('lightgray')
        book_codes = codes[b]

        # Get active code count for the title
        active_code_count_this_book = (~is_code_dead_np[b]).sum()

        # --- Check 1: Input NaNs ---
        if np.isnan(book_codes).any():
            print(f"Skipping plot for Book {b}: Contains NaN")
            # Don't increment plotted_count, axis will be filled later
            continue

        # --- Check 2: Can we even attempt PCA? ---
        min_codes_for_pca = 2
        min_rank_for_1d = 1
        if book_codes.shape[0] < min_codes_for_pca or r < min_rank_for_1d:
             print(f"Skipping plot for Book {b}: Not enough codes ({book_codes.shape[0]} < {min_codes_for_pca}) or rank ({r} < {min_rank_for_1d}) for PCA")
             # Don't increment plotted_count
             continue

        # Calculate mean
        current_mean = np.mean(book_codes, axis=0)
        if np.isnan(current_mean).any():
            print(f"Warning: Mean calculation for book {b} resulted in NaN.")
            ax.text(0.5, 0.5, 'Mean is NaN', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Book {b} ({active_code_count_this_book}/{C} active)\n(Mean NaN)")
            # Don't increment plotted_count, effectively skipping this slot but axis is marked
            # We need to increment plotted_count here to move to the next axis slot
            plotted_count += 1
            continue

        # --- Attempt PCA & Plotting --- 
        try:
            n_components_attempt = min(2, r, book_codes.shape[1])
            if n_components_attempt == 0:
                 print(f"Skipping plot for Book {b}: Cannot attempt PCA with 0 components.")
                 continue

            pca = PCA(n_components=n_components_attempt)
            pts = pca.fit_transform(book_codes)
            effective_components = pca.n_components_
            explained_variance = pca.explained_variance_

            # Check 3: Collapsed Codebook (Zero Variance)
            if effective_components == 0 or np.all(np.isclose(explained_variance, 0)):
                print(f"Skipping plot for Book {b}: Collapsed (Zero Variance Detected)")
                continue # Skip to next try

            # Plotting Logic (Check 4)
            ax.grid(True)
            ax.set_aspect('equal', adjustable='box') # Set equal aspect ratio
            plot_made = False
            if effective_components == 1 or (effective_components == 2 and np.isclose(explained_variance[1], 0)):
                # Plot as 1D
                variance_1d = explained_variance[0]
                sc = ax.scatter(pts[:, 0], np.zeros_like(pts[:, 0]), c=probs[b], cmap="magma", vmin=vmin_prob, vmax=vmax_prob, s=25)
                ax.set_title(f"Book {b} ({active_code_count_this_book}/{C} act) (1D)\nVar={variance_1d:.2e}")
                plot_made = True
            elif effective_components == 2:
                # Plot as 2D
                variance_2d = explained_variance.sum()
                ratio_2d = explained_variance / variance_2d
                sc = ax.scatter(pts[:, 0], pts[:, 1], c=probs[b], cmap="magma", vmin=vmin_prob, vmax=vmax_prob, s=25)
                ax.set_title(f"Book {b} ({active_code_count_this_book}/{C} act) (2D)\nEVR={ratio_2d.sum():.1%}")
                plot_made = True

            if plot_made:
                if plotted_sm_pca is None: plotted_sm_pca = sc
                all_book_means_pca[plotted_count] = current_mean # Store mean at plot position
                successful_plot_indices.append(b)
                plotted_count += 1 # INCREMENT SUCCESSFUL PLOT COUNT
            else:
                 print(f"Skipping plot for Book {b}: Unexpected PCA components ({effective_components})")
                 # Don't increment plotted_count
                 continue

        except ValueError as e:
            print(f"PCA Error for Book {b}: {e}")
            # Don't increment plotted_count
            continue
        except Exception as e:
            print(f"Plotting Error for Book {b}: {e}")
            # Don't increment plotted_count
            continue

    # --- Fill remaining axes ---
    if plotted_count < num_plot_books:
        print(f"Plotted {plotted_count} codebooks. Filling remaining {num_plot_books - plotted_count} axes.")
        for i in range(plotted_count, num_plot_books):
            ax = pca_axes[i]
            ax.set_facecolor('lightgray')
            ax.text(0.5, 0.5, 'No more valid\nbooks to plot', ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"Axis {i+1}")

    # --- PCA Colorbar ---
    if plotted_sm_pca:
        cbar_pca_ax = fig.add_subplot(gs_main[:4, 5])
        fig.colorbar(plotted_sm_pca, cax=cbar_pca_ax, label="Usage Prob")

    # --- Nested GridSpec for Bottom Row (2x2) ---
    gs_bottom = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_main[4:, :5],
                                                 hspace=0.4, wspace=0.3) # Space between bottom plots

    # --- Top-Left: Book Mean Similarity Heatmap --- 
    heatmap_ax = fig.add_subplot(gs_bottom[0, 0])
    gs_bottom_heatmap = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_bottom[0,0], width_ratios=[0.9, 0.1], wspace=0.1)
    heatmap_ax = fig.add_subplot(gs_bottom_heatmap[0,0])
    cbar_heatmap_ax = fig.add_subplot(gs_bottom_heatmap[0,1])

    # Filter means for successfully plotted books, removing NaNs
    # Use the `all_book_means_pca` list which was populated at index `plotted_count`
    valid_means_for_heatmap = [m for i, m in enumerate(all_book_means_pca[:plotted_count]) if not np.isnan(m).any()]

    if len(valid_means_for_heatmap) >= 2:
        mean_vectors_array = np.array(valid_means_for_heatmap)
        # Check shape just in case
        if mean_vectors_array.shape[0] >= 2 and mean_vectors_array.shape[1] > 0:
            cos_sim = np.abs(cosine_similarity(mean_vectors_array))
            im = heatmap_ax.imshow(cos_sim, cmap='magma', vmin=0, vmax=1)
            # Adjust labels for successfully plotted books
            # Use the indices stored in successful_plot_indices
            heatmap_labels = [f"P{i}(B{b})" for i, b in enumerate(successful_plot_indices)]
            heatmap_ax.set_xticks(np.arange(len(heatmap_labels)))
            heatmap_ax.set_yticks(np.arange(len(heatmap_labels)))
            heatmap_ax.set_xticklabels(heatmap_labels, fontsize=6) # Smaller font size
            heatmap_ax.set_yticklabels(heatmap_labels, fontsize=6)
            plt.setp(heatmap_ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor") # Adjust rotation
            heatmap_ax.set_title(f"Book Mean Abs Cos Sim (Plotted {plotted_count})") # Updated title
            fig.colorbar(im, cax=cbar_heatmap_ax, label='Abs Cos Sim')
        else:
            heatmap_ax.text(0.5, 0.5, 'Mean Sim N/A\n(<2 valid)', ha='center', va='center', transform=heatmap_ax.transAxes)
            heatmap_ax.axis('off'); cbar_heatmap_ax.axis('off')
    else:
        heatmap_ax.text(0.5, 0.5, 'Mean Sim N/A\n(<2 plotted)', ha='center', va='center', transform=heatmap_ax.transAxes)
        heatmap_ax.axis('off'); cbar_heatmap_ax.axis('off')

    # --- Top-Right: K-Value Histogram ---
    hist_k_ax = fig.add_subplot(gs_bottom[0, 1])
    hist_k_ax.hist(k_values, bins=max(10, B // 10))
    hist_k_ax.set_title(f"K Values (N={B})", fontsize=10)
    hist_k_ax.set_xlabel("Temperature (k)", fontsize=8)
    hist_k_ax.set_ylabel("Frequency", fontsize=8)
    hist_k_ax.tick_params(axis='both', which='major', labelsize=8)
    hist_k_ax.grid(axis='y', linestyle='--', alpha=0.7)

    # --- Bottom-Left: L1 Norm Bar Chart ---
    bar_ax = fig.add_subplot(gs_bottom[1, 0])
    book_indices = np.arange(num_plot_books)
    means_to_plot = book_l1_means[:num_plot_books]
    q25_to_plot = book_l1_q25[:num_plot_books]
    q75_to_plot = book_l1_q75[:num_plot_books]
    # Calculate error bar lengths relative to the mean
    lower_error = means_to_plot - q25_to_plot
    upper_error = q75_to_plot - means_to_plot
    # Clip errors at 0 to avoid negative values for matplotlib
    lower_error = np.clip(lower_error, 0, None)
    upper_error = np.clip(upper_error, 0, None) # Also clip upper error
    error_bars = [lower_error, upper_error]
    
    bar_ax.bar(book_indices, means_to_plot, yerr=error_bars, capsize=4)
    bar_ax.set_title(f"Avg Code L1 Norm (Top {num_plot_books} Books, 25-75 Percentile)", fontsize=10)
    bar_ax.set_xlabel("Book Index", fontsize=8)
    bar_ax.set_ylabel("Avg L1 Norm", fontsize=8)
    bar_ax.set_xticks(book_indices)
    bar_ax.set_xticklabels([f"{b}" for b in book_indices], fontsize=7)
    bar_ax.tick_params(axis='y', which='major', labelsize=8)
    bar_ax.grid(axis='y', linestyle='--', alpha=0.7)

    # --- Bottom-Right: Bias Norm Histogram ---
    hist_bias_ax = fig.add_subplot(gs_bottom[1, 1])
    hist_bias_ax.hist(bias_norms, bins=max(10, B // 10))
    hist_bias_ax.set_title(f"Bias Norms (N={B})", fontsize=10)
    hist_bias_ax.set_xlabel("L2 Norm of Bias", fontsize=8)
    hist_bias_ax.set_ylabel("Frequency", fontsize=8)
    hist_bias_ax.tick_params(axis='both', which='major', labelsize=8)
    hist_bias_ax.grid(axis='y', linestyle='--', alpha=0.7)

    # --- Save Figure ---
    path = os.path.join(args.pca_dir, f"epoch_{ep+1:02d}_stage_{stage_idx}_analysis_sparse.png") # Add _sparse suffix
    plt.savefig(path);
    plt.close(fig)

# ───────────────────────── main loop ─────────────────────────
def main():
    A = get_args()
    add_l1_forward() # patch once
    torch.autograd.set_detect_anomaly(True)

    # data ----------------------------------------------------
    files = sorted(glob.glob(A.data_glob))
    if not files:
        raise FileNotFoundError('no data files match glob')
    random.shuffle(files)
    split = int(0.1 * len(files))
    train_dl = DataLoader(NPZDataset(files[split:]), batch_size=A.batch,
                          shuffle=True,  num_workers=4, pin_memory=True)
    val_dl   = DataLoader(NPZDataset(files[:split]),  batch_size=A.batch,
                          shuffle=False, num_workers=2, pin_memory=True)

    # model ---------------------------------------------------
    # --- Load Checkpoint and Determine Hyperparameters ---
    print(f"Loading checkpoint: {A.checkpoint}")
    # Load to CPU first for inspection without occupying GPU memory
    loaded_data = torch.load(A.checkpoint, map_location='cpu')
    H = None
    state_dict = None

    if isinstance(loaded_data, dict) and 'hyperparameters' in loaded_data and 'state_dict' in loaded_data:
        # New format checkpoint
        print("Detected new checkpoint format with saved hyperparameters.")
        H = loaded_data['hyperparameters']
        state_dict = loaded_data['state_dict']
        print("Loaded hyperparameters:", H)
    elif isinstance(loaded_data, dict): # Check if it's a dict (potential state_dict)
         # Assume old format (state_dict only)
        print("Detected old checkpoint format (state_dict only).")
        state_dict = loaded_data
        H = infer_hyperparameters(state_dict) # Infer H from the state_dict
    else:
        raise TypeError(f"Loaded checkpoint is of unexpected type: {type(loaded_data)}. Expected dict.")

    if H is None or state_dict is None:
         raise ValueError("Could not determine hyperparameters or state_dict from checkpoint.")

    # --- Initialize Model using Determined Hyperparameters ---
    print("Initializing model architecture...")
    # Use determined H, but override threshold with current CLI arg for this run
    # Also, allow potential overrides from H if keys are missing using .get()
    vq = ResidualVQ(
        stages=H.get('stages', 3), # Default fallback if somehow missing
        dim=H.get('dim', 768),
        num_books=H.get('num_books', 200),
        num_codes=H.get('num_codes', 20),
        rank=H.get('rank', 2),
        k_init=H.get('k_init', 0.1), # Loaded k will override this anyway
        gamma=H.get('gamma', 0.03),
        orth_lambda=H.get('orth_lambda', 0.0),
        normalize=H.get('normalize', False),
        device=A.device, # Use current device arg
        l1_threshold=A.l1_threshold # *** Use the l1_threshold for *this* run ***
        # temp_target/factor not included as they are loss terms usually set per-run
    ).to(A.device) # Move model to target device *before* loading state_dict

    # --- Load State Dict ---
    print(f"Loading state_dict onto device {A.device}...")
    # Remap state_dict tensors to the target device if they aren't already
    state_dict = {k: v.to(A.device) for k, v in state_dict.items()}
    vq.load_state_dict(state_dict)
    print("Model state loaded successfully.")

    opt   = torch.optim.AdamW(vq.parameters(), lr=A.lr)
    if A.freeze_temp:
        print("Freezing temperature parameters (k).")
        params_to_optimize = [
            p for name, p in vq.named_parameters()
            if p.requires_grad and '.k' not in name
        ]
        opt = torch.optim.AdamW(params_to_optimize, lr=A.lr)
    else:
        print("Optimizing all parameters including temperature (k).")
        opt = torch.optim.AdamW(vq.parameters(), lr=A.lr)

    sched = CosineAnnealingLR(opt, T_max=A.epochs, eta_min=0)

    wandb.init(project=A.project, name=A.run_name, config=A)
    pathlib.Path(A.ckpt_dir).mkdir(exist_ok=True, parents=True)
    pathlib.Path(A.pca_dir).mkdir(exist_ok=True, parents=True)

    # training -----------------------------------------------
    for ep in range(A.epochs):
        # linear λ schedule
        λ = A.l1_init + (A.l1_final - A.l1_init) * min(1.0, ep / max(1, A.l1_warm))
        wandb.log({'lambda_l1': λ, 'epoch': ep}, commit=False)

        # ---- train ----
        vq.train(); running = 0.0
        for xb in tqdm(train_dl, desc=f'train {ep+1}'):
            xb = xb.to(A.device)
            loss, _, _ = vq.forward_with_l1(xb)

            # --- Add L1 penalty on codebook parameters of *active* books --- 
            if λ > 0:
                codebook_l1_penalty = 0.0
                for blk in vq.blocks:
                    # Calculate L1 only for non-dead codes within non-dead books
                    active_codes_mask = ~blk.is_code_dead # Shape [B, C]
                    # Ensure we only consider books that are not dead
                    active_books_mask = ~blk.is_book_dead # Shape [B]
                    # Combine masks: needs shape [B, C, r] or apply sequentially
                    # Mask codebook: zero out dead codes
                    masked_codebook = blk.codebook * active_codes_mask.unsqueeze(-1)
                    # Further zero out entire books that are dead (redundant if codes already zero, but safer)
                    masked_codebook = masked_codebook * active_books_mask.view(blk.B, 1, 1)
                    # Sum the absolute values of the masked codebook
                    codebook_l1_penalty += masked_codebook.abs().sum()
                loss = loss + λ * codebook_l1_penalty
            # --- End L1 Penalty ---

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(vq.parameters(), 1.0)
            opt.step(); running += loss.item()
            del xb, loss; gc.collect()
        wandb.log({'train_loss_epoch': running/len(train_dl), 'epoch': ep})

        # ---- validate ----
        vq.eval(); val_rel_err_tot = 0.0; n_val = 0
        with torch.no_grad():
            for vb in tqdm(val_dl, desc=f'val {ep+1}', leave=False):
                vb = vb.to(A.device)
                # Note: We don't strictly need intermediate recons or probs for basic val metrics
                _, final_recon, _ = vq.forward_with_l1(vb)
                # step_mse = F.mse_loss(final_recon, vb).item()
                # val_mse_tot += step_mse
                
                # Calculate Relative L2 Error
                error_norm = torch.linalg.norm(final_recon - vb, dim=-1)
                batch_norm = torch.linalg.norm(vb, dim=-1)
                step_rel_err = (error_norm / batch_norm.clamp(min=1e-9)).mean().item()
                val_rel_err_tot += step_rel_err

                n_val += 1
                del vb, final_recon; gc.collect()
        
        # avg_val_mse = val_mse_tot / n_val if n_val > 0 else 0
        avg_val_rel_err = val_rel_err_tot / n_val if n_val > 0 else 0
        # print(f"Epoch {ep+1}/{A.epochs} | λ: {λ:.2e} | Val MSE: {avg_val_mse:.4f}")
        print(f"Epoch {ep+1}/{A.epochs} | λ: {λ:.2e} | Val Rel Err: {avg_val_rel_err:.4f}")
        # wandb.log({'val_mse_epoch': avg_val_mse, 'epoch': ep})
        wandb.log({'val_rel_err_epoch': avg_val_rel_err, 'epoch': ep})

        # ---- Update and Print Dead Status ----
        if hasattr(vq, 'update_dead_status') and callable(getattr(vq, 'update_dead_status')):
            vq.update_dead_status() # Update based on current norms
            # Check if print_status exists before calling
            if hasattr(vq, 'print_status') and callable(getattr(vq, 'print_status')):
                 vq.print_status() # Print the counts of active elements
        # -----------------------------------

        # ---- save checkpoint ----
        if (ep+1) % A.save_every == 0:
            ck = f"{A.ckpt_dir}/epoch_{ep+1:03d}.pt"
            intermediate_save_data = {
                'hyperparameters': H, 
                'state_dict': vq.state_dict()
            }
            torch.save(intermediate_save_data, ck)
            wandb.save(ck)

        # ---- plots ----
        if (ep+1) % A.plot_every == 0:
            usages = [torch.zeros(H['num_books'], H['num_codes'], device=A.device) for _ in range(H['stages'])]
            vq.eval()
            with torch.no_grad():
                for vb in val_dl:
                    vb = vb.to(A.device)
                    _, _, _, pr = vq.forward_with_l1(vb, return_prob=True)
                    for s in range(H['stages']):
                        usages[s] += pr[s].sum(0)
            for s in range(H['stages']):
                # Call the local, sparse-specific plotting function
                _plot_residual_vq_codebooks_sparse(
                    vq.blocks[s], usages[s], ep, s,
                    argparse.Namespace(pca_dir=A.pca_dir, codes=H['num_codes'])
                )

        sched.step()

    torch.save(vq.state_dict(), f"{A.ckpt_dir}/final_sparse.pt")
    print("Fine-tuning complete → checkpoints written to", A.ckpt_dir)

    # --- Save Final Model (always in the NEW format) ---
    final_save_path = f"{A.ckpt_dir}/final_sparse.pt"
    # H was determined earlier either from new checkpoint or inferred from old one
    final_save_data = {
        'hyperparameters': H,
        'state_dict': vq.state_dict()
    }
    torch.save(final_save_data, final_save_path)
    print(f"Final fine-tuned model saved with hyperparameters to {final_save_path}")

if __name__ == '__main__':
    main()
