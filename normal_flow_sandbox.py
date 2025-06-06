#!/usr/bin/env python3
"""
train_nf_halfspace.py
---------------------
Train a Neural Spline Flow (NSF) on a toy mixture distribution consisting of
  • 50 % uniform points on the positive x₀ hemisphere of the unit hypersphere
  • 50 % standard normal points truncated to the negative x₀ half‑space.

This version **fixes the transform name** to match the public `nflows` API
(`PiecewiseRationalQuadraticCouplingTransform`) and wires up a ResidualNet
conditioner via `nflows.nn.nets`. It also toggles the binary mask every layer
so that all coordinates are updated.

The script includes experiment tracking with **Weights & Biases**.
"""

import argparse
from typing import Callable
import math

import torch
from torch import nn
from tqdm import tqdm

# Import the scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR 

from nflows import distributions, flows, transforms
from nflows.nn import nets as nnets
from nflows.utils import create_alternating_binary_mask
import wandb

# -----------------------------------------------------------------------------
# Data generation
# -----------------------------------------------------------------------------

def sample_hemisphere(n: int, d: int, device) -> torch.Tensor:
    # Sample directions uniformly from sphere surface
    z = torch.randn(n, d, device=device)
    sphere_points = z / z.norm(dim=1, keepdim=True)

    # Sample radii to ensure uniform volume density
    u = torch.rand(n, 1, device=device)  # Uniform(0, 1)
    radii = u.pow(1.0 / d)

    # Scale sphere points by radii
    x = radii * sphere_points

    # Ensure points are in the positive x0 hemisphere
    x[:, 0] = x[:, 0].abs()
    return x

def sample_trunc_normal(n: int, d: int, device) -> torch.Tensor:
    x = torch.randn(n, d, device=device)
    x[:, 0] = -x[:, 0].abs()
    return x

def sample_mixture(n: int, d: int, device) -> torch.Tensor:
    a = sample_hemisphere(n // 2, d, device)
    b = sample_trunc_normal(n - n // 2, d, device)
    return torch.cat([a, b], 0)

# -----------------------------------------------------------------------------
# True Log Probability (for validation)
# -----------------------------------------------------------------------------

def true_log_prob(x: torch.Tensor, d: int) -> torch.Tensor:
    """Calculates the true log probability density of the mixture model using logsumexp."""
    # Constants
    log_05 = math.log(0.5)
    log_2 = math.log(2.0)
    log_pi = math.log(math.pi)
    log_gamma_half_d_plus_1 = torch.lgamma(torch.tensor(d / 2.0 + 1.0, device=x.device)).item()
    log_Vd = (d / 2.0) * log_pi - log_gamma_half_d_plus_1  # Log volume of d-ball

    # Log density of the uniform hemisphere component (p_hemi = 2/Vd)
    log_density_hemi = log_2 - log_Vd
    # Log density of the truncated normal component (p_trunc = 2 * N(x|0,I))
    log_density_normal_const = log_2 - (d / 2.0) * (log_2 + log_pi) # log(2 * (2pi)^(-d/2))

    # Conditions
    x_norm_sq = x.pow(2).sum(dim=1)
    in_ball = x_norm_sq <= 1.0
    non_neg_hemi = x[:, 0] >= 0.0 # x0 >= 0 for hemisphere
    non_pos_hemi = x[:, 0] <= 0.0 # x0 <= 0 for truncated normal

    # --- Calculate log(0.5 * p_hemi(x)) --- 
    log_p_comp1 = torch.full_like(x[:, 0], -float('inf'))
    mask_A = in_ball & non_neg_hemi
    log_p_comp1[mask_A] = log_05 + log_density_hemi # log(0.5 * 2 / Vd) = log(1/Vd)

    # --- Calculate log(0.5 * p_trunc(x)) --- 
    log_p_comp2 = torch.full_like(x[:, 0], -float('inf'))
    mask_B = non_pos_hemi
    log_N_density_points = log_density_normal_const - 0.5 * x_norm_sq[mask_B] # log(2*N(x)) for points in region
    log_p_comp2[mask_B] = log_05 + log_N_density_points # log(0.5 * 2 * N(x)) = log(N(x))

    # --- Combine using logsumexp --- 
    # log( exp(log_p_comp1) + exp(log_p_comp2) )
    log_probs_both = torch.stack([log_p_comp1, log_p_comp2], dim=0)
    log_prob_mix = torch.logsumexp(log_probs_both, dim=0)

    # --- Check for invalid points --- 
    # If log_prob_mix is -inf, the point fell outside both regions' support.
    # This shouldn't happen for points generated by sample_mixture.
    invalid_mask = torch.isinf(log_prob_mix) & (log_prob_mix < 0)
    if invalid_mask.any():
        problematic_points = x[invalid_mask]
        print(f"Error: {invalid_mask.sum()} points found with zero true probability density!")
        print("Problematic points (first 10):\n", problematic_points[:10])
        raise ValueError("Point(s) found with zero true probability density according to mixture definition.")

    return log_prob_mix

# -----------------------------------------------------------------------------
# Flow construction
# -----------------------------------------------------------------------------

def make_conditioner(hidden: int) -> Callable[[int, int], nn.Module]:
    def factory(in_f: int, out_f: int):
        return nnets.ResidualNet(
            in_features=in_f,
            out_features=out_f,
            hidden_features=hidden,
            num_blocks=2,
            activation=nn.ReLU(),
        )
    return factory


def build_flow(dim: int, hidden: int, layers: int, bins: int, tail: float) -> flows.Flow:
    transforms_list = []
    for i in range(layers):
        transforms_list.append(
            transforms.coupling.PiecewiseRationalQuadraticCouplingTransform(
                mask=create_alternating_binary_mask(
                    features=dim, even=(i % 2 == 0)
                ),
                transform_net_create_fn=make_conditioner(hidden),
                num_bins=bins,
                tails="linear",
                tail_bound=tail,
            )
        )
        transforms_list.append(transforms.RandomPermutation(features=dim))

    transform = transforms.CompositeTransform(transforms_list)
    base = distributions.StandardNormal([dim])
    return flows.Flow(transform, base)

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    flow = build_flow(args.dim, args.hidden, args.layers, args.bins, args.tail).to(device)
    # Use AdamW instead of Adam
    opt = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=args.wd) 
    # Add the Cosine Annealing scheduler
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs * args.steps_per_epoch) 

    wandb.init(project=args.project, config=vars(args))
    wandb.watch(flow, log="parameters", log_freq=1000)

    for epoch in range(1, args.epochs + 1):
        flow.train()
        loss_epoch = 0.0
        for _ in tqdm(range(args.steps_per_epoch)):
            x = sample_mixture(args.batch, args.dim, device)
            loss = -flow.log_prob(x).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(flow.parameters(), args.clip); opt.step()
            # Step the scheduler after each optimizer step
            scheduler.step() 
            loss_epoch += loss.item()
        loss_epoch /= args.steps_per_epoch

        # Validation step
        with torch.no_grad():
            val_x = sample_mixture(args.val_batch, args.dim, device)
            model_log_probs = flow.log_prob(val_x)
            true_log_probs = true_log_prob(val_x, args.dim)

            val_NLL = -model_log_probs.mean().item()
            true_NLL = -true_log_probs.mean().item()
            log_prob_error_var = (model_log_probs - true_log_probs).var().item()
            log_prob_error_std = log_prob_error_var ** 0.5 # Calculate std dev

        wandb.log({
            "epoch": epoch,
            "train_NLL": loss_epoch,
            "val_NLL": val_NLL,
            "true_NLL": true_NLL,
            "log_prob_error_var": log_prob_error_var,
            "log_prob_error_std": log_prob_error_std # Log std dev
        })
        print(f"Epoch {epoch:03d} | train NLL {loss_epoch:.3f} | val NLL {val_NLL:.3f} | true NLL {true_NLL:.3f} | err std {log_prob_error_std:.3f}") # Print std dev

    # final samples
    flow.eval()
    with torch.no_grad():
        samples = flow.sample(2048).cpu().numpy()
    wandb.log({"samples": wandb.Histogram(samples)})

    # Save the trained model state
    save_path = f"trained_flow_dim{args.dim}.pth"
    torch.save(flow.state_dict(), save_path)
    print(f"Trained model state dict saved to {save_path}")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser("Train NSF on a hemisphere/normal mixture")
    p.add_argument("--dim", type=int, default=100)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--bins", type=int, default=16)
    p.add_argument("--tail", type=float, default=5.0)
    p.add_argument("--batch", type=int, default=4*8192)
    p.add_argument("--val_batch", type=int, default=32768)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--steps_per_epoch", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-6)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--project", type=str, default="nf-hemisphere-mix")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    train(get_args())
