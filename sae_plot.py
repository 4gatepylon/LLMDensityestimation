#!/usr/bin/env python3
"""
sae_highfreq_heatmap.py  (v4 – fixed)
=====================================
• Encoder *and* decoder span heat‑maps (side‑by‑side)
• Histogram of %‑active features
• Top‑50‑by‑activity CSV  +  latent_stats.csv
• Bug‑fixes:
    – decoder rows indexed with W_dec[i]
    – .detach() before .cpu().numpy() to avoid requires‑grad error
    – copy() NumPy mmap slice to avoid "not writable" warning

TODO please stop using np.load/np.save with pickling
"""
import argparse, os, glob, csv, sys, pathlib, time
import numpy as np
import torch, matplotlib.pyplot as plt
from tqdm import tqdm
import itertools # Import itertools for combinations

# ─────────────────── defaults ────────────────────
DEF_RELEASE = "callummcdougall/arena-demos-transcoder"
DEF_SAE_ID  = "gpt2-small-layer-8-mlp-transcoder-folded-b_dec_out"
DEF_LAYER   = 8
DEF_HOOK    = "ln2"     # activation hook used during collection

# ─────────────────── SAE loader ──────────────────
def load_sae(release, sae_id, device):
    from sae_lens import SAE
    print(f"Loading SAE  {release}  /  {sae_id}")
    sae, cfg, _ = SAE.from_pretrained(release=release,
                                      sae_id=sae_id,
                                      device=device)
    return sae, cfg

# ─────────────────── plotting helpers ────────────
def plot_pct_hist(pct, out_path):
    plt.figure(figsize=(8,6))
    plt.hist(pct, bins=50, log=True)
    plt.xlabel("% of tokens where feature fires")
    plt.ylabel("Number of features (log scale)")
    plt.title("SAE activity percentage")
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()

def plot_side_by_side(H_enc, H_dec, H_coact, edges_proj, edges_coact,
                      feat_labels, out_path, cos_sim_enc, cos_sim_dec):
    # Create figure with 1 row for projections, 1 row for coactivation
    fig = plt.figure(figsize=(11, 8), constrained_layout=True) # Use constrained_layout
    gs = fig.add_gridspec(2, 2) # 2 rows, 2 cols grid
    ax_enc = fig.add_subplot(gs[0, 0]) # Top-left
    ax_dec = fig.add_subplot(gs[0, 1], sharey=ax_enc) # Top-right, share Y with top-left
    ax_coact = fig.add_subplot(gs[1, :]) # Bottom row, spanning both columns

    # Encoder plot (axes[0])
    H = H_enc
    im_top = ax_enc.imshow(np.log1p(H).T, origin="lower", aspect="auto",
                           extent=[edges_proj[0][0], edges_proj[0][-1], edges_proj[1][0], edges_proj[1][-1]],
                           cmap="viridis")
    ax_enc.set_title(f"Encoder projection (cos(θ)={cos_sim_enc:.2f})")
    ax_enc.set_xlabel(f"Proj. onto Enc {feat_labels[0]}")
    ax_enc.set_ylabel(f"Proj. onto Enc {feat_labels[1]}")

    # Decoder plot (axes[1])
    H = H_dec
    im_top = ax_dec.imshow(np.log1p(H).T, origin="lower", aspect="auto",
                           extent=[edges_proj[0][0], edges_proj[0][-1], edges_proj[1][0], edges_proj[1][-1]],
                           cmap="viridis")
    ax_dec.set_title(f"Decoder projection (cos(θ)={cos_sim_dec:.2f})")
    ax_dec.set_xlabel(f"Proj. onto norm Dec {feat_labels[0]}")
    plt.setp(ax_dec.get_yticklabels(), visible=False) # Hide y-tick labels for shared axis

    # Add colorbar for top row plots - associate with ax_dec
    fig.colorbar(im_top, ax=ax_dec, shrink=0.8, label="log(count) - Projections")

    # Coactivation plot (bottom row)
    H = H_coact
    im_coact = ax_coact.imshow(np.log1p(H).T, origin="lower", aspect="auto",
                               extent=[edges_coact[0][0], edges_coact[0][-1], edges_coact[1][0], edges_coact[1][-1]],
                               cmap="viridis")
    ax_coact.set_title("Coactivation (ReLU output)")
    ax_coact.set_xlabel(f"Coactivation {feat_labels[0]}")
    ax_coact.set_ylabel(f"Coactivation {feat_labels[1]}")

    # Add colorbar for bottom plot
    fig.colorbar(im_coact, ax=ax_coact, shrink=0.8, label="log(count) - Coactivations")

    plt.savefig(out_path, dpi=300); plt.close()

# ─────────────────── main ─────────────────────────
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--run",     default="/workspace/runs/gpt2_L8_ln2_de41c410")
    pa.add_argument("--release", default=DEF_RELEASE)
    pa.add_argument("--sae_id",  default=DEF_SAE_ID)
    pa.add_argument("--topk",    type=int, default=100)
    pa.add_argument("--bins",    type=int, default=150)
    pa.add_argument("--mbatch",  type=int, default=8)
    pa.add_argument("--vec_batch", type=int, default=32768)
    pa.add_argument("--device",  default="cuda")
    args = pa.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sae, _  = load_sae(args.release, args.sae_id, str(device))
    W_enc, b_enc, W_dec = sae.W_enc, sae.b_enc, sae.W_dec
    d_sae = W_enc.shape[1]

    # ── activation files ──
    chunks = sorted(glob.glob(os.path.join(args.run, "act_*.npy")))
    if not chunks:
        sys.exit(f"No act_*.npy files found in {args.run}")

    fires = np.zeros(d_sae, np.int64); total = 0

    # ── pass 0: firing counts ──
    for f in tqdm(chunks, desc="stats"):
        X = torch.from_numpy(np.load(f, mmap_mode="r").astype(np.float32, copy=False)).to(device)
        with torch.no_grad():
            A = torch.relu(X @ W_enc + b_enc)
            fires += (A > 0).sum(0).cpu().numpy()
        total += X.size(0); del X, A; torch.cuda.empty_cache()

    pct = fires / total * 100.0
    outdir = pathlib.Path(args.run) / "sae_plots"; outdir.mkdir(parents=True, exist_ok=True)
    plot_pct_hist(pct, outdir / "activity_pct_hist.png")

    top_idx = np.argsort(pct)[-50:][::-1]
    with open(outdir / "top50_by_activity.csv", "w", newline="") as fp:
        w = csv.writer(fp); w.writerow(["rank","latent","fires","pct"])
        for r, idx in enumerate(top_idx, 1):
            w.writerow([r, idx, fires[idx], f"{pct[idx]:.4f}"])
    print(f"✓ Top-{len(top_idx)} features saved to top50_by_activity.csv")

    # === Plotting loop for top 5 pairs ===
    top_n_plot = 5
    top_plot_indices = top_idx[:top_n_plot]
    print(f"\nGenerating plots for top {top_n_plot} feature pairs...")

    # Determine global min/max across a sample for potentially shared axes (optional)
    # sample_X = torch.from_numpy(np.load(chunks[0], mmap_mode="r")[:10_000].copy().astype(np.float32)).to(device)

    for idx1, idx2 in itertools.combinations(range(top_n_plot), 2):
        i = top_plot_indices[idx1]
        j = top_plot_indices[idx2]
        print(f"\nProcessing pair: HF{i} vs HF{j}")

        # projectors
        proj_W, proj_b = W_enc[:, [i,j]], b_enc[[i,j]]
        with torch.no_grad():
            W_dec_i_norm = W_dec[i].norm()
            W_dec_j_norm = W_dec[j].norm()
            # Handle potential zero norm vectors
            norm_W_dec_i = W_dec[i]/W_dec_i_norm if W_dec_i_norm > 1e-6 else W_dec[i]
            norm_W_dec_j = W_dec[j]/W_dec_j_norm if W_dec_j_norm > 1e-6 else W_dec[j]
            D = torch.stack([norm_W_dec_i, norm_W_dec_j], 1)
            # Calculate cosine similarities for axis annotation *for this pair*
            cos_sim_enc = torch.cosine_similarity(W_enc[:, i], W_enc[:, j], dim=0).item()
            cos_sim_dec = torch.cosine_similarity(norm_W_dec_i, norm_W_dec_j, dim=0).item()

        # Define projection functions for the current pair (i, j)
        def enc_proj(arr):
            with torch.no_grad():
                out = arr @ proj_W + proj_b
            return out.detach().cpu().numpy()

        def dec_proj(arr):
            with torch.no_grad():
                out = arr @ D
            return out.detach().cpu().numpy()

        # Define coactivation function for the current pair (i, j)
        def coact_vals(arr, feat_indices):
            with torch.no_grad():
                # Calculate hidden pre-activation
                hidden_pre = arr @ W_enc + b_enc
                # Apply ReLU
                activations = torch.relu(hidden_pre)
                # Select only the features of interest
                coacts = activations[:, feat_indices]
            # Use a small threshold to account for float precision near zero
            coacts_np = coacts.detach().cpu().numpy()
            return coacts_np

        # Calculate histogram range for this pair using a sample, considering *both* projections
        sample_X = torch.from_numpy(np.load(chunks[0], mmap_mode="r")[:10_000].copy().astype(np.float32)).to(device)
        P_sample_enc = enc_proj(sample_X)
        P_sample_dec = dec_proj(sample_X)
        P_sample_coact = coact_vals(sample_X, [i, j])
        # Filter coactivation sample to exclude points where *both* activations are effectively zero
        coact_threshold = 1e-6 # Increased threshold
        P_sample_coact_filtered = P_sample_coact[(P_sample_coact[:, 0] > coact_threshold) | (P_sample_coact[:, 1] > coact_threshold)]

        # Find overall min/max across both projections for axis 0
        min_enc0, max_enc0 = P_sample_enc[:,0].min(), P_sample_enc[:,0].max()
        min_dec0, max_dec0 = P_sample_dec[:,0].min(), P_sample_dec[:,0].max()
        global_min0 = min(min_enc0, min_dec0)
        global_max0 = max(max_enc0, max_dec0)

        # Find overall min/max across both projections for axis 1
        min_enc1, max_enc1 = P_sample_enc[:,1].min(), P_sample_enc[:,1].max()
        min_dec1, max_dec1 = P_sample_dec[:,1].min(), P_sample_dec[:,1].max()
        global_min1 = min(min_enc1, min_dec1)
        global_max1 = max(max_enc1, max_dec1)

        # Define bins using the global ranges for projections
        e0_proj = np.linspace(global_min0, global_max0, args.bins+1)
        e1_proj = np.linspace(global_min1, global_max1, args.bins+1)

        # Define bins for coactivations (often start from 0 due to ReLU)
        # Calculate ranges based on the *filtered* sample, starting bins at 0
        if P_sample_coact_filtered.size > 0:
            max_coact0 = P_sample_coact_filtered[:,0].max()
            max_coact1 = P_sample_coact_filtered[:,1].max()
        else: # Handle case where sample has no co-activations > threshold
            max_coact0, max_coact1 = 1.0, 1.0 # Default max if no data
        # Start bins at 0, go up to max observed in filtered sample
        e0_coact = np.linspace(0, max_coact0, args.bins+1)
        e1_coact = np.linspace(0, max_coact1, args.bins+1)

        del sample_X, P_sample_enc, P_sample_dec, P_sample_coact, P_sample_coact_filtered; torch.cuda.empty_cache()

        # Initialize histograms for this pair
        H_enc = np.zeros((args.bins,args.bins), np.int64)
        H_dec = np.zeros_like(H_enc)
        H_coact = np.zeros_like(H_enc)

        # Accumulate histograms for this pair across all data
        for f in tqdm(chunks, desc=f"hist {i}v{j}", leave=False):
            X = torch.from_numpy(np.load(f, mmap_mode="r").astype(np.float32, copy=False)).to(device)
            # Calculate projections using pair-specific functions
            P1 = enc_proj(X)
            P2 = dec_proj(X)
            P_coact = coact_vals(X, [i, j])
            del X; torch.cuda.empty_cache() # Del X early

            # Filter coactivations to exclude points where both are effectively zero
            P_coact_filtered = P_coact[(P_coact[:, 0] > coact_threshold) | (P_coact[:, 1] > coact_threshold)]

            h1, *_ = np.histogram2d(P1[:,0], P1[:,1], bins=(e0_proj, e1_proj))
            h2, *_ = np.histogram2d(P2[:,0], P2[:,1], bins=(e0_proj, e1_proj))
            # Calculate coactivation histogram only if there's filtered data
            if P_coact_filtered.size > 0:
                h_coact, *_ = np.histogram2d(P_coact_filtered[:,0], P_coact_filtered[:,1], bins=(e0_coact, e1_coact))
            else:
                h_coact = np.zeros((args.bins, args.bins)) # Empty histogram

            H_enc += h1.astype(np.int64); H_dec += h2.astype(np.int64)
            H_coact += h_coact.astype(np.int64)
            del P1, P2, P_coact, P_coact_filtered, h1, h2, h_coact; torch.cuda.empty_cache()

        # Plot for the current pair
        plot_side_by_side(H_enc, H_dec, H_coact, (e0_proj, e1_proj), (e0_coact, e1_coact),
                          (f"HF{i}", f"HF{j}"),
                          outdir / f"HF{i}_HF{j}_enc_vs_dec.png",
                          cos_sim_enc, cos_sim_dec)

    print(f"\n✓ All plots generated in {outdir}")

if __name__ == "__main__":
    main()
