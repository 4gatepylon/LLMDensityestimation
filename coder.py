#!/usr/bin/env python3
"""
Projected / Residual VQ package  (now with optional projector‑row normalisation and multi‑stage residual depth)
============================================================================================================
Key new CLI flags
-----------------
--stages N      : how many residual *layers* (1 ⇒ original single stage)
--normalize     : if passed, rows of each projector P_b are re‑normalised every fwd pass (default **off**)

Quick start (no renorm, 2 residual stages):
    python projected_vq_package.py train \
           --data_glob './runs/*.npy' --stages 2

TODO please stop using np.load/np.save with pickling
"""
from __future__ import annotations
import argparse, glob, random, gc, math, os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.gridspec as gridspec # Import gridspec
from checkpoint_fs import save_checkpoint

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Single‑stage Projected VQ
# ──────────────────────────────────────────────────────────────────────────────
class ProjectedVQ(nn.Module):
    """
    A *single* PQ-style codebook block with optional bias and support for dead codes/books.
    
    NOTE this ENTIRE thing is written by o3 with very minimal supervision.

    It looks like it will first project things onto rank self.r subspaces
    and do this in self.B num books each with self.C codes. In other words,
    we project for each book onto a self.r dimensional subspace and then we
    perform vector quantization on these subspaces using self.C codes (obviously
    of lower dimensionality). Then obviously each of these lower rank codes
    corresponds to a higher rank vector in the original space using the basis
    from the projection.
    """

    def __init__(
        self,
        dim: int = 768,
        num_books: int = 40,
        num_codes: int = 50,
        rank: int | None = None,
        k_init: float = 0.1,
        beta: float = 0.25,
        gamma: float = 0.03,
        orth_lambda: float = 1.0,
        lru_refresh: int = 1000,
        normalize: bool = False,
        device: str = "cuda",
        l1_threshold: float = 1e-3, # Threshold for declaring codes dead
        temp_target: float = 0.1,   # Target temperature
        temp_factor: float = 0.0    # Factor for temperature penalty
    ) -> None:
        super().__init__()
        self.d, self.B, self.C = dim, num_books, num_codes
        self.r = rank or dim // num_books # XXX rank should be passed ngl wtf
        self.normalize = normalize
        self.l1_threshold = l1_threshold # Store threshold
        self.temp_target = temp_target   # Store temp target
        self.temp_factor = temp_factor   # Store temp factor

        # parameters -----------------------------------------------------------
        self.projector = nn.Parameter(torch.randn(self.B, self.r, self.d))
        nn.init.orthogonal_(self.projector.view(-1, self.d))
        self.codebook = nn.Parameter(torch.randn(self.B, self.C, self.r)) # Represents displacements in r-dim space
        nn.init.kaiming_uniform_(self.codebook, a=math.sqrt(5))
        self.k = nn.Parameter(torch.full((self.B,), k_init))
        self.biases = nn.Parameter(torch.zeros(self.B, self.d)) # Bias for each book's affine subspace

        # misc bookkeeping -----------------------------------------------------
        self.beta, self.gamma, self.orth_lambda = beta, gamma, orth_lambda
        self.register_buffer("usage", torch.zeros(self.B, self.C))
        self.lru_refresh, self._step = lru_refresh, 0

        # Buffers for dead code/book tracking
        self.register_buffer("is_code_dead", torch.zeros(self.B, self.C, dtype=torch.bool))
        self.register_buffer("is_book_dead", torch.zeros(self.B, dtype=torch.bool))

    # ---------------------------------------------------------------------
    @torch.no_grad()
    # TODO(Adriano) seems to be dead code (only called by `ResidualVQ` update_dead_status
    # which in turn is dead code... is this an o3 issue?)
    def update_dead_status(self):
        """Updates the dead status of codes and books based on L1 norm."""
        if self.l1_threshold <= 0: # Skip if threshold is non-positive
             print("Skipping dead status update: l1_threshold <= 0")
             return

        # Calculate L1 norm for each code vector
        code_l1_norms = torch.linalg.norm(self.codebook, ord=1, dim=-1) # Shape [B, C]

        # Update code dead status
        new_is_code_dead = code_l1_norms < self.l1_threshold
        # Ensure already dead codes remain dead (monotonicity)
        self.is_code_dead.copy_(self.is_code_dead | new_is_code_dead)

        # Update book dead status: a book is dead if ALL its codes are dead
        new_is_book_dead = self.is_code_dead.all(dim=1)
        # Ensure already dead books remain dead
        self.is_book_dead.copy_(self.is_book_dead | new_is_book_dead)

        # Zero out parameters of dead codes/books
        # Unsqueeze needed to match dimensions for broadcasting
        self.codebook.masked_fill_(self.is_code_dead.unsqueeze(-1), 0)
        self.biases.masked_fill_(self.is_book_dead.unsqueeze(-1), 0)
        # Only zero projector if not normalizing (otherwise norm keeps it non-zero)
        if not self.normalize:
            self.projector.masked_fill_(self.is_book_dead.view(self.B, 1, 1), 0)
        # We don't zero out 'k', gradients should handle it implicitly via forward pass masking

    # ---------------------------------------------------------------------
    def forward(
            self,
            x: torch.Tensor,
            *args,
            return_prob: bool = False,
            **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        E, d = x.shape
        P = F.normalize(self.projector, dim=-1) if self.normalize else self.projector

        # --- Mask active projectors and biases for calculations ---
        # Use ~is_book_dead to get a mask of *active* books
        active_book_mask_proj = (~self.is_book_dead).view(self.B, 1, 1).float()
        active_book_mask_bias = (~self.is_book_dead).view(self.B, 1).float()
        active_book_mask_local = (~self.is_book_dead).view(1, self.B, 1).float() # For E,B,r tensor

        P_masked = P * active_book_mask_proj
        biases_masked = self.biases * active_book_mask_bias

        # 1. Calculate projection of centered input efficiently using masked parameters
        z_proj = torch.einsum("ed,brd->ebr", x, P_masked)
        # TODO why is this bias in higher dimensional space?
        bias_proj = torch.einsum("bd,brd->br", biases_masked, P_masked) # Use masked P here too
        z = z_proj - bias_proj.unsqueeze(0) # Shape (E, B, r).

        # 2. Find nearest code displacement, masking dead codes
        dist2 = (z.unsqueeze(2) - self.codebook.unsqueeze(0)).pow(2).sum(-1) # Shape (E, B, C)
        # Set distance to infinity for dead codes before softmax
        dist2 = dist2.masked_fill(self.is_code_dead.unsqueeze(0), 1e20) # Use a large float instead of inf
        # Handle case where all codes in a book might be dead (avoid NaN in softmax)
        # If all codes are dead (inf distance), probs should be uniform over dead codes (effectively zero later)
        # A simpler approach: if a book is dead, its contribution is zeroed later anyway.
        # Clamp k to avoid issues if k becomes zero or negative for some reason (though it shouldn't)
        prob = F.softmax(-self.k.clamp(min=1e-9).view(1, self.B, 1) * dist2, -1) # Shape (E, B, C)
        # Explicitly zero probabilities for dead codes (handles edge cases/numerical stability)
        prob = prob.masked_fill(self.is_code_dead.unsqueeze(0), 0)

        # 3. Calculate expected code displacement
        recon_local_disp = (prob.unsqueeze(-1) * self.codebook.unsqueeze(0)).sum(2) # Shape (E, B, r)
        # Mask contribution from dead books
        recon_local_disp = recon_local_disp * active_book_mask_local

        # 4. Map expected displacements back to d-dim space using masked projector
        recon_disp_summed = torch.einsum("ebr,brd->ed", recon_local_disp, P_masked) # Shape (E, d)

        # 5. Add the sum of biases (only from active books)
        biases_sum = biases_masked.sum(dim=0) # Sums over B, already masked
        recon = recon_disp_summed + biases_sum.unsqueeze(0) # Shape (E, d)

        # --- Loss Calculation ---
        mse = F.mse_loss(recon, x)
        commit = F.mse_loss(recon.detach(), x) # Commitment loss still makes sense

        # Entropy calculation: - sum(p * log(p))
        # Ensure prob is non-negative and mask out zero probabilities before log to prevent NaN
        prob_for_log = prob.clamp_min(0.) # Ensure non-negative
        # Mask dead codes before log to avoid log(0) -> NaN
        safe_log_prob = (prob_for_log + 1e-20).log() # Add small epsilon before log
        entropy = -(prob * safe_log_prob).sum(-1) # Sum over C
        # Average entropy only over active books
        num_active_books = (~self.is_book_dead).sum().clamp(min=1) # Avoid division by zero
        avg_entropy_per_active_book = entropy.sum(dim=0) / E # Sum over E -> shape [B]
        masked_entropy = avg_entropy_per_active_book * (~self.is_book_dead).float()
        final_entropy = masked_entropy.sum() / num_active_books # Average over active books
        loss = mse + self.beta * commit - self.gamma * final_entropy

        # Add Temperature Penalty (averaged over active books)
        if self.temp_factor > 0:
            active_k = self.k[~self.is_book_dead] # Select k for active books
            if active_k.numel() > 0: # Only add penalty if there are active books
                temp_penalty = self.temp_factor * ((active_k - self.temp_target)**2).mean()
                loss += temp_penalty

        # Orthogonality constraint only on active projectors
        if self.orth_lambda > 0:
            active_indices = torch.where(~self.is_book_dead)[0]
            if len(active_indices) > 1:
                P_active = P[active_indices] # Use original P or P_masked? P makes more sense
                P_norm_active = F.normalize(P_active, dim=-1)
                dot_active = torch.einsum("brd,crd->bc", P_norm_active, P_norm_active)
                identity_active = torch.eye(len(active_indices), device=x.device)
                orth_loss = self.orth_lambda * ((dot_active - identity_active) ** 2).mean()
                loss += orth_loss

        # --- Bookkeeping ---
        with torch.no_grad():
            # Usage should only accumulate for non-dead codes
            self.usage += prob.sum(0) * (~self.is_code_dead).float()
        self._step += 1

        # --- Return ---
        out = (loss, recon)
        if return_prob:
            # Return probabilities, masking dead codes explicitly just in case
            out += (prob.masked_fill(self.is_code_dead.unsqueeze(0), 0),)
        return out

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _lru_refresh(self):
        for b in range(self.B):
            if self.is_book_dead[b]: # Skip dead books
                continue

            active_indices = torch.where(~self.is_code_dead[b])[0]
            if len(active_indices) < 2: # Need at least 2 active codes to refresh
                continue

            # Find least/most used among *active* codes
            usage_active = self.usage[b][active_indices]
            least_idx_in_active = usage_active.argmin()
            most_idx_in_active = usage_active.argmax()

            # Map back to original C indices
            least_original_idx = active_indices[least_idx_in_active]
            most_original_idx = active_indices[most_idx_in_active]

            if least_original_idx == most_original_idx:
                continue

            # Refresh least used active code based on most used active code
            self.codebook[b, least_original_idx].copy_(
                self.codebook[b, most_original_idx] + 0.05 * torch.randn_like(self.codebook[b, most_original_idx])
            )
            # Reset usage only for the refreshed code? Or all active? Resetting all is simpler.
            # self.usage[b, least_original_idx] = self.usage[b, most_original_idx] # Alternative: copy usage
            # Let's stick to original logic of resetting whole book usage after refresh check
        self.usage.zero_() # Reset usage for all books after checking all

    # --- Status Methods ---
    def num_active_codewords(self) -> int:
        """Returns the total number of active (non-dead) codewords in this stage."""
        return (~self.is_code_dead).sum().item()

    def num_active_codebooks(self) -> int:
        """Returns the number of active (non-dead) codebooks in this stage."""
        return (~self.is_book_dead).sum().item()

    def print_status(self, stage_idx: int | None = None):
        """Prints the number of active codes and books."""
        prefix = f"Stage {stage_idx}: " if stage_idx is not None else ""
        total_codes = self.B * self.C
        total_books = self.B
        active_codes = self.num_active_codewords()
        active_books = self.num_active_codebooks()
        print(f"{prefix}Active Books: {active_books}/{total_books}, Active Codes: {active_codes}/{total_codes}")
    

    # ---------- 1) ENCODER ----------
    def encode(
            self,
            x: torch.Tensor,
            *args,
            hard: bool = False,
            flatten: bool = True,
            **kwargs
    ) -> torch.Tensor:
        """
        Args
        ----
        x           : (E, d) input vectors
        hard        : if True → straight-through argmax → one-hot codes
        flatten     : if True → output shape (E, B*C); else (E, B, C)
        return_prob : also return softmax probabilities (useful for analysis)

        Returns
        -------
        probs       : (E, B, C) tensor of probabilities (of each code; each INDEX corresponds to a code)
        """
        if hard:
            raise NotImplementedError("Hard encoding not implemented")
        # NOTE: this is basically copied from `forward` above but modified in a few places...
        E, d = x.shape
        P = F.normalize(self.projector, dim=-1) if self.normalize else self.projector # norm 1 (euclidean)
        active_book_mask_proj = (~self.is_book_dead).view(self.B, 1, 1).float()
        active_book_mask_bias = (~self.is_book_dead).view(self.B, 1).float()
        P_masked = P * active_book_mask_proj
        biases_masked = self.biases * active_book_mask_bias
        z_proj = torch.einsum("ed,brd->ebr", x, P_masked)
        bias_proj = torch.einsum("bd,brd->br", biases_masked, P_masked) # Use masked P here too
        # NOTE these are the projected points!
        # NOTE now we merely need to suck them into each of the cluster centers
        z = z_proj - bias_proj.unsqueeze(0) # Shape (E, B, r)

        # 2. Find nearest code displacement, masking dead codes (copied also from `forward`)
        dist2 = (z.unsqueeze(2) - self.codebook.unsqueeze(0)).pow(2).sum(-1) # Shape (E, B, C)
        dist2 = dist2.masked_fill(self.is_code_dead.unsqueeze(0), 1e20)
        prob = F.softmax(-self.k.clamp(min=1e-9).view(1, self.B, 1) * dist2, -1) # Shape (E, B, C)
        prob = prob.masked_fill(self.is_code_dead.unsqueeze(0), 0)
        # NOTE for every vector we have now in each codebook given it a probability of being that code
        # This is the latent
        assert prob.shape == (E, self.B, self.C) 
        if flatten:
            return prob.view(E, -1) # B*C is our number of codes
        return prob

    # ---------- 2) DECODER ----------
    def decode(self, probs: torch.Tensor, *args, **kwargs):
        """
        Reconstruct from code-usage vectors.

        Args
        ----
        probs      : (E, B*C) if flattened else (E, B, C)
        """
        active_book_mask_local = (~self.is_book_dead).view(1, self.B, 1).float() # For E,B,r tensor

        # 1. Unflatten this stuff
        E = probs.shape[0]
        if probs.dim() == 2:
            probs = probs.view(E, self.B, self.C)
        assert probs.shape == (E, self.B, self.C)

        # Expected displacement in r-space
        # TODO(Adriano) what is this?
        recon_local_disp = (probs.unsqueeze(-1) * self.codebook.unsqueeze(0)).sum(2)  # (E,B,r)
        recon_local_disp = recon_local_disp * active_book_mask_local

        # Back-project + add biases (same lines as old forward 4–5)
        P = F.normalize(self.projector, dim=-1) if self.normalize else self.projector
        # We are we summing over book dimension? TODO(Adriano) I think this is because we take on code per book?
        # TODO(Adriano) I guess we sum over each book and rank... hmmm...
        recon_disp = torch.einsum("ebr,brd->ed", recon_local_disp, P)
        assert recon_disp.shape == (E, self.d)
        recon      = recon_disp + (self.biases * (~self.is_book_dead).view(self.B, 1).float()).sum(0)
        assert recon.shape == (E, self.d)
        return recon

    # ---------- 3) TRAINING FORWARD ----------
    def enc_dec_forward(self, x: torch.Tensor, *, hard: bool = False, return_prob: bool = False) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This should be roughyl the same as `forward` but it uses encode and decode
        instead. We unit test this in the test_coder.py file.
        """
        E = x.shape[0]
        prob = self.encode(x, hard=hard, flatten=False, return_prob=True)
        recon = self.decode(prob, unflatten=False)

        # --- identical loss computation but now using `prob` & `recon` ---
        P = F.normalize(self.projector, dim=-1) if self.normalize else self.projector
        mse     = F.mse_loss(recon, x)
        commit  = F.mse_loss(recon.detach(), x)
        prob_for_log = prob.clamp_min(0.)
        safe_log_prob = (prob_for_log + 1e-20).log()
        entropy = -(prob * safe_log_prob).sum(-1) # Sum over C
        # Average entropy only over active books
        num_active_books = (~self.is_book_dead).sum().clamp(min=1) # Avoid division by zero
        avg_entropy_per_active_book = entropy.sum(dim=0) / E # Sum over E -> shape [B]
        masked_entropy = avg_entropy_per_active_book * (~self.is_book_dead).float()
        final_entropy = masked_entropy.sum() / num_active_books # Average over active books
        loss = mse + self.beta * commit - self.gamma * final_entropy
        # Add Temperature Penalty (averaged over active books)
        if self.temp_factor > 0:
            active_k = self.k[~self.is_book_dead] # Select k for active books
            if active_k.numel() > 0: # Only add penalty if there are active books
                temp_penalty = self.temp_factor * ((active_k - self.temp_target)**2).mean()
                loss += temp_penalty

        # Orthogonality constraint only on active projectors
        if self.orth_lambda > 0:
            active_indices = torch.where(~self.is_book_dead)[0]
            if len(active_indices) > 1:
                P_active = P[active_indices] # Use original P or P_masked? P makes more sense
                P_norm_active = F.normalize(P_active, dim=-1)
                dot_active = torch.einsum("brd,crd->bc", P_norm_active, P_norm_active)
                identity_active = torch.eye(len(active_indices), device=x.device)
                orth_loss = self.orth_lambda * ((dot_active - identity_active) ** 2).mean()
                loss += orth_loss

        # --- Bookkeeping ---
        with torch.no_grad():
            # Usage should only accumulate for non-dead codes
            self.usage += prob.sum(0) * (~self.is_code_dead).float()
        self._step += 1

        # --- Return ---
        out = (loss, recon)
        if return_prob:
            # Return probabilities, masking dead codes explicitly just in case
            out += (prob.masked_fill(self.is_code_dead.unsqueeze(0), 0),)
        return out

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Residual stack wrapper
# ──────────────────────────────────────────────────────────────────────────────
class ResidualVQ(nn.Module):
    """Stack *n* `ProjectedVQ`s; each encodes residual of predecessors."""

    def __init__(self, stages: int = 1, **vq_kwargs):
        super().__init__()
        # Pass the l1_threshold down to each ProjectedVQ block
        self.stages = stages
        self.blocks = nn.ModuleList(ProjectedVQ(**vq_kwargs) for _ in range(stages))

    def forward(self, x, *, return_prob=False):
        flat = x.view(-1, x.size(-1))
        total_recon = torch.zeros_like(flat)
        total_loss = 0.0
        probs = []
        intermediate_recons = []
        residual = flat
        for s, blk in enumerate(self.blocks):
            loss, recon, prob = blk(residual, return_prob=True)
            current_stage_recon = recon
            total_recon = total_recon + current_stage_recon
            intermediate_recons.append(total_recon.view_as(x))
            total_loss += loss # Accumulate loss (already handles dead codes/books internally)
            residual = residual - current_stage_recon.detach()
            if return_prob:
                 probs.append(prob) # Probs are already masked internally

        out = (total_loss, total_recon.view_as(x), intermediate_recons)
        if return_prob:
            out += (probs,)
        return out

    @torch.no_grad()
    # TODO(Adriano) seems to be dead code
    def update_dead_status(self):
        """Calls update_dead_status on all blocks."""
        print("Updating dead status across all stages...")
        for s, blk in enumerate(self.blocks):
             blk.update_dead_status()

    def num_active_codewords(self) -> int:
        """Returns the total number of active codewords across all stages."""
        return sum(blk.num_active_codewords() for blk in self.blocks)

    def num_active_codebooks(self) -> int:
        """Returns the total number of active codebooks across all stages."""
        return sum(blk.num_active_codebooks() for blk in self.blocks)

    def print_status(self):
        """Prints the status of active codes and books for each stage."""
        print("--- VQ Status ---")
        total_active_codes = 0
        total_codes = 0
        total_active_books = 0
        total_books = 0
        for s, blk in enumerate(self.blocks):
            blk.print_status(stage_idx=s)
            total_active_codes += blk.num_active_codewords()
            total_codes += blk.B * blk.C
            total_active_books += blk.num_active_codebooks()
            total_books += blk.B
        print(f"Overall: Active Books: {total_active_books}/{total_books}, Active Codes: {total_active_codes}/{total_codes}")
        print("-----------------")

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Dataset helper
# ──────────────────────────────────────────────────────────────────────────────
class NPZDataset(torch.utils.data.Dataset):
    def __init__(self, files: List[str]):
        self.files = files
        self.cache = {}
        self.rows = [(fi, i) for fi, f in enumerate(files) for i in range(np.load(f, mmap_mode='r').shape[0])]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        fi, ri = self.rows[idx]
        if fi not in self.cache:
            self.cache[fi] = np.load(self.files[fi], mmap_mode='r')
        return torch.tensor(self.cache[fi][ri], dtype=torch.float32)

# ──────────────────────────────────────────────────────────────────────────────
# 4.  Training CLI
# ──────────────────────────────────────────────────────────────────────────────

def train_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_glob', default='./gpt_activations/*.npy')
    ap.add_argument('--books', type=int, default=50)
    ap.add_argument('--codes', type=int, default=100)
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--k', type=float, default=0.1, help="Initial k value")
    ap.add_argument('--gamma', type=float, default=0.03)
    ap.add_argument('--orth', type=float, default=0)        # not 0.03
    ap.add_argument('--stages', type=int, default=1, help='residual depth')
    ap.add_argument('--normalize', default=False, action='store_true', help='row‑normalise projectors each forward pass')
    # Temperature penalty args
    ap.add_argument('--temp_target', type=float, default=0.0, help='Target temperature for penalty term')
    ap.add_argument('--temp_factor', type=float, default=0.01, help='Factor for temperature penalty term (default 0=off)')
    # Other args
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--project', default='projected-vq')
    ap.add_argument('--run_name', default=None)
    ap.add_argument('--pca_dir', default='./codebook_pca', help='Directory to save PCA plots (set to None to disable)')
    args = ap.parse_args()

    files = sorted(glob.glob(args.data_glob))
    if not files:
        raise FileNotFoundError('no data files match glob')
    random.shuffle(files)
    split = int(0.1 * len(files))
    train_dl = DataLoader(NPZDataset(files[split:]), batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(NPZDataset(files[:split]), batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    vq = ResidualVQ(
        stages=args.stages,
        dim=768, num_books=args.books, num_codes=args.codes, rank=args.rank,
        k_init=args.k, gamma=args.gamma, orth_lambda=args.orth, normalize=args.normalize, device=args.device,
        temp_target=args.temp_target, temp_factor=args.temp_factor # Pass new args here
    ).to(args.device)

    opt = torch.optim.AdamW(vq.parameters(), lr=args.lr)
    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0)

    wandb.init(project=args.project, name=args.run_name, config=args)

    for ep in range(args.epochs):
        # -------- train --------
        vq.train(); tot=0
        pbar = tqdm(train_dl, desc=f'train {ep+1}')
        for step,batch in enumerate(pbar):
            batch=batch.to(args.device)
            # vq returns: total_loss, final_recon, intermediate_recons
            loss, recon, _ = vq(batch) # Unpack 3 values, ignore intermediate_recons here
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(vq.parameters(),1.);
            opt.step();
            
            # Call LRU refresh for each block after optimizer step
            for blk in vq.blocks:
                if blk.training and blk._step % blk.lru_refresh == 0:
                    blk._lru_refresh()

            # Calculate metrics for tqdm
            with torch.no_grad(): # No need for gradients here
                step_mse = F.mse_loss(recon, batch).item()
                error_norm = torch.linalg.norm(recon - batch, dim=-1)
                batch_norm = torch.linalg.norm(batch, dim=-1)
                step_rel_err = (error_norm / batch_norm.clamp(min=1e-9)).mean().item()
            
            current_loss = loss.item()
            tot += current_loss
            pbar.set_postfix(loss=current_loss, mse=step_mse, rel_err=step_rel_err) # Update postfix
            if step%100==0:
                # Log train_loss_step (the VQ loss), could add mse/rel_err here too if desired
                wandb.log({'train_loss_step':current_loss,'epoch':ep,'step':step}) 
            del batch,recon,loss; gc.collect()
        avg_epoch_loss = tot / len(train_dl) if len(train_dl) > 0 else 0
        wandb.log({'train_loss_epoch':avg_epoch_loss,'epoch':ep})

        # -------- val ----------
        vq.eval(); 
        mse_tot=n=0; 
        rel_err_tot_stages = [0.0] * args.stages # Accumulator list for stage-wise rel err
        
        # Initialize usage tensors for each stage for PCA plotting
        val_usages = [torch.zeros(args.books, args.codes, device=args.device) for _ in range(args.stages)]
        
        with torch.no_grad():
            val_pbar = tqdm(val_dl, desc=f'val {ep+1}', leave=False) # Add tqdm wrapper
            for batch in val_pbar:
                batch = batch.to(args.device)
                # Get loss, final_recon, intermediate_recons (list), and probabilities (list)
                _, final_recon, intermediate_recons, probs = vq(batch, return_prob=True)
                
                # --- Calculate Metrics ---
                # 1. Final Validation MSE (using final reconstruction)
                step_mse = F.mse_loss(final_recon, batch).item()
                mse_tot += step_mse
                
                # 2. Relative L2 Error (Vector-wise) for each stage
                step_rel_err_stages = [] # Store rel err for each stage for this batch
                batch_norm = torch.linalg.norm(batch, dim=-1).clamp(min=1e-9)
                for stage_idx in range(args.stages):
                    stage_recon = intermediate_recons[stage_idx] # Cumulative recon up to this stage
                    error_norm = torch.linalg.norm(stage_recon - batch, dim=-1)
                    relative_error_vec = error_norm / batch_norm
                    step_rel_err = relative_error_vec.mean().item()
                    step_rel_err_stages.append(step_rel_err)
                    rel_err_tot_stages[stage_idx] += step_rel_err # Accumulate per stage
                # --- End Metrics ---

                n += 1 # Increment batch counter
                # Update postfix with final MSE and stage-wise relative errors
                postfix_dict = {'mse': step_mse}
                # Add stage-wise relative errors
                postfix_dict.update({f'rel_s{i}': err for i, err in enumerate(step_rel_err_stages)})
                # Explicitly add the final stage's relative error as final_rel_err
                postfix_dict['final_rel_err'] = step_rel_err_stages[-1]
                val_pbar.set_postfix(**postfix_dict)

                # Accumulate usage for each stage (for PCA plotting)
                if probs and len(probs) == args.stages:
                    for s in range(args.stages):
                        current_prob = probs[s]
                        # Prob shape check (handle flattened or seq inputs)
                        if current_prob.dim() == 3: # E, B, C
                            val_usages[s] += current_prob.sum(dim=0)
                        elif current_prob.dim() == 4: # B, S, B, C ? (Unlikely based on VQ code)
                            val_usages[s] += current_prob.sum(dim=(0, 1))
                        else: # Fallback: B, C
                             val_usages[s] += current_prob.sum(dim=0)

        # Calculate average val metrics safely
        avg_val_mse = mse_tot / n if n > 0 else 0
        avg_rel_err_stages = [(tot / n if n > 0 else 0) for tot in rel_err_tot_stages]
        
        # Log final MSE and stage-wise relative errors
        log_data = {'val_mse': avg_val_mse, 'epoch': ep}
        log_data.update({f'val_rel_err_s{i}': avg for i, avg in enumerate(avg_rel_err_stages)})
        wandb.log(log_data)
        
        # Print final MSE and final stage relative error
        print(f'epoch {ep+1}: val_mse={avg_val_mse:.4f}  final_rel_err={avg_rel_err_stages[-1]:.4f}')

        # ---- Print VQ Status ----
        # Check if the vq object has the print_status method before calling
        if hasattr(vq, 'print_status') and callable(getattr(vq, 'print_status')):
            vq.print_status()
        # --------------------------

        sched.step()

        # --- PCA Plotting (All Stages) ---
        if args.pca_dir and args.pca_dir.lower() != 'none':
            print(f"Generating PCA plots for {args.stages} stages...")
            for s in range(args.stages):
                vq_stage = vq.blocks[s]
                stage_usage = val_usages[s] # Get usage for this stage
                if vq_stage.r < 2:
                    print(f"Skipping PCA plot for stage {s}: Projector rank ({vq_stage.r}) < 2.")
                    continue
                if args.codes < 2:
                    print(f"Skipping PCA plot for stage {s}: Number of codes ({args.codes}) < 2.")
                    continue
                # Call plot function for the current stage
                _plot_residual_vq_codebooks(vq_stage, stage_usage, ep, s, args)
            print("PCA plots generation finished.")

    # --- Save Model --- 
    model_save_path = "residual_vq_model.pt"
    # Save hyperparameters along with the state dict
    save_data = {
        'hyperparameters': {
            'stages': args.stages,
            'dim': 768, # Assuming hardcoded or retrieved if made dynamic
            'num_books': args.books,
            'num_codes': args.codes,
            'rank': args.rank,
            'k_init': args.k,
            'gamma': args.gamma,
            'orth_lambda': args.orth,
            'normalize': args.normalize,
            'l1_threshold': getattr(args, 'l1_threshold', 1e-3), # Handle if arg doesn't exist
            'temp_target': getattr(args, 'temp_target', 0.1),
            'temp_factor': getattr(args, 'temp_factor', 0.0)
        },
        'state_dict': vq.state_dict()
    }
    save_checkpoint(model_save_path, save_data['hyperparameters'], save_data['state_dict'])
    print(f"Training finished. Model saved with hyperparameters to {model_save_path}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Plotting Function (Adapted for ResidualVQ - Plots specified stage)
# ──────────────────────────────────────────────────────────────────────────────
def _plot_residual_vq_codebooks(vq_stage: ProjectedVQ, usage: torch.Tensor, ep: int, stage_idx: int, args):
    """Generates PCA plots, analysis heatmaps, and histograms for the specified VQ stage."""
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

    # Calculate Average Internal Codebook Similarities
    avg_internal_similarities = []
    for b in range(B):
        book_codes = codes[b] # Shape (C, r)
        if C < 2:
            avg_internal_similarities.append(0)
            continue
        sim_matrix = cosine_similarity(book_codes) # Shape (C, C)
        abs_sim_matrix = np.abs(sim_matrix)
        upper_triangle_indices = np.triu_indices(C, k=1)
        sim_values = abs_sim_matrix[upper_triangle_indices]
        if len(sim_values) == 0:
             avg_internal_similarities.append(0) # Or handle as NaN/None if preferred
        else:
             avg_internal_similarities.append(np.mean(sim_values))
    avg_internal_similarities = np.array(avg_internal_similarities)

    # --- Setup Figure using GridSpec ---
    fig = plt.figure(figsize=(22, 20)) # Adjusted figure size for 2x2 bottom grid
    fig.suptitle(f"Epoch {ep+1} VQ Analysis (Stage {stage_idx})", fontsize=16, y=0.99)

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

    # --- PCA Plotting (remains largely the same) ---
    valid_probs = probs[:num_plot_books].flatten()
    valid_probs = valid_probs[np.isfinite(valid_probs)]
    vmin_prob = 0
    vmax_prob = np.percentile(valid_probs, 99) if len(valid_probs) > 0 else 1
    if not np.isfinite(vmax_prob) or vmax_prob == 0: vmax_prob = 1

    plotted_sm_pca = None
    all_book_means = []

    for b in range(num_plot_books):
        ax = pca_axes[b]
        ax.set_facecolor('lightgray')
        book_codes = codes[b]
        all_book_means.append(np.mean(book_codes, axis=0))
        n_components_pca = min(2, book_codes.shape[0], book_codes.shape[1])
        if n_components_pca < 2:
            ax.text(0.5, 0.5, 'PCA N/A', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Book {b} (PCA N/A)")
        else:
            pca = PCA(n_components=n_components_pca)
            pts = pca.fit_transform(book_codes)
            explained_variance = pca.explained_variance_ratio_.sum()
            sc = ax.scatter(pts[:,0], pts[:,1], c=probs[b], cmap="magma", vmin=vmin_prob, vmax=vmax_prob, s=25)
            if plotted_sm_pca is None: plotted_sm_pca = sc
            ax.set_title(f"Book {b}\\nEVR={explained_variance:.1%}")
            ax.grid(True)

    for i in range(num_plot_books, 20):
        ax = pca_axes[i]
        ax.set_facecolor('lightgray')
        ax.axis("off")

    # --- PCA Colorbar ---
    if plotted_sm_pca:
        cbar_pca_ax = fig.add_subplot(gs_main[:4, 5])
        fig.colorbar(plotted_sm_pca, cax=cbar_pca_ax, label="Usage Prob")

    # --- Nested GridSpec for Bottom Row (2x2) ---
    # Use the first 5 columns of the last *two* rows of the main grid
    gs_bottom = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_main[4:, :5],
                                                 hspace=0.4, wspace=0.3) # Space between bottom plots

    # --- Top-Left: Book Mean Similarity Heatmap ---
    heatmap_ax = fig.add_subplot(gs_bottom[0, 0])
    # Create a dummy axes for the heatmap colorbar, positioned manually relative to heatmap_ax
    # This requires careful calculation or use a helper library like matplotlib-axesgrid1 for robust placement.
    # For simplicity here, we'll slightly adjust the gridspec/figure or accept imperfect placement.
    # Let's try allocating space within the 2x2 grid itself.
    gs_bottom_heatmap = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_bottom[0,0], width_ratios=[0.9, 0.1], wspace=0.1)
    heatmap_ax = fig.add_subplot(gs_bottom_heatmap[0,0])
    cbar_heatmap_ax = fig.add_subplot(gs_bottom_heatmap[0,1])

    if all_book_means:
        mean_vectors_array = np.array(all_book_means)
        if mean_vectors_array.shape[0] >= 2 and mean_vectors_array.shape[1] > 0:
            cos_sim = np.abs(cosine_similarity(mean_vectors_array))
            im = heatmap_ax.imshow(cos_sim, cmap='magma', vmin=0, vmax=1)
            heatmap_labels = [f"B{b}" for b in range(num_plot_books)]
            heatmap_ax.set_xticks(np.arange(len(heatmap_labels)))
            heatmap_ax.set_yticks(np.arange(len(heatmap_labels)))
            heatmap_ax.set_xticklabels(heatmap_labels, fontsize=8)
            heatmap_ax.set_yticklabels(heatmap_labels, fontsize=8)
            plt.setp(heatmap_ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            heatmap_ax.set_title(f"Book Mean Abs Cos Sim (Top {num_plot_books})", fontsize=10)
            fig.colorbar(im, cax=cbar_heatmap_ax, label='Abs Cos Sim')
        else:
            heatmap_ax.text(0.5, 0.5, 'Mean Sim N/A', ha='center', va='center', transform=heatmap_ax.transAxes)
            heatmap_ax.axis('off')
            cbar_heatmap_ax.axis('off')
    else:
        heatmap_ax.text(0.5, 0.5, 'Mean Sim N/A', ha='center', va='center', transform=heatmap_ax.transAxes)
        heatmap_ax.axis('off')
        cbar_heatmap_ax.axis('off')

    # --- Top-Right: K-Value Histogram ---
    hist_k_ax = fig.add_subplot(gs_bottom[0, 1])
    hist_k_ax.hist(k_values, bins=max(10, B // 10))
    hist_k_ax.set_title(f"K Values (N={B})", fontsize=10)
    hist_k_ax.set_xlabel("Temperature (k)", fontsize=8)
    hist_k_ax.set_ylabel("Frequency", fontsize=8)
    hist_k_ax.tick_params(axis='both', which='major', labelsize=8)
    hist_k_ax.grid(axis='y', linestyle='--', alpha=0.7)

    # --- Bottom-Left: Internal Codebook Similarity Bar Chart ---
    bar_ax = fig.add_subplot(gs_bottom[1, 0])
    book_indices = np.arange(num_plot_books)
    bar_ax.bar(book_indices, avg_internal_similarities[:num_plot_books])
    bar_ax.set_title(f"Avg Internal Code Sim (Top {num_plot_books} Books)", fontsize=10)
    bar_ax.set_xlabel("Book Index", fontsize=8)
    bar_ax.set_ylabel("Avg Abs Cos Sim", fontsize=8)
    bar_ax.set_xticks(book_indices)
    bar_ax.set_xticklabels([f"{b}" for b in book_indices], fontsize=7)
    bar_ax.tick_params(axis='y', which='major', labelsize=8)
    bar_ax.set_ylim(0, 1)
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
    path = os.path.join(args.pca_dir, f"epoch_{ep+1:02d}_stage_{stage_idx}_analysis.png")
    plt.savefig(path);
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# 6.  Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__=='__main__':
    train_cli() # Directly call train_cli to run training by default
