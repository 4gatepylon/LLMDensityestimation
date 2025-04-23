#!/usr/bin/env python3
"""
pca_heatmaps.py  –  quick CPU heat‑maps for GPT‑2 activation PCs
===============================================================

Core capabilities
-----------------
1. Any PC pair heat‑map               (--pairs "1,2;1,3;700,701")
2. Extra random orthonormal pairs     (--rand 5)
3. PC‑1 × PC‑2 coloured by mean PC‑3  (--pc3map)
4. N×N grid of PC‑1 × PC‑2 at N² PC‑3 quantiles  (--pc3grid N)

Output names are self‑documenting:  e.g.  PC1_PC2.png, PC1_PC2_meanPC3.png
"""

import argparse, os, glob, re, sys, pathlib, random
import numpy as np, joblib, matplotlib.pyplot as plt
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# small helpers
# ──────────────────────────────────────────────────────────────────────────────
def parse_pairs(s: str):
    """Return list of 0‑indexed PC index pairs."""
    out = []
    for part in re.split(r"[; ]+", s.strip()):
        if not part:
            continue
        try:
            i, j = map(int, part.split(","))
        except ValueError:
            sys.exit(f'Bad --pairs element: "{part}"')
        out.append((i - 1, j - 1))
    return out


def random_pairs(dim: int, n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n):
        v1 = rng.standard_normal(dim).astype(np.float32)
        v1 /= np.linalg.norm(v1)
        v2 = rng.standard_normal(dim).astype(np.float32)
        v2 -= v1 * (v1 @ v2)          # orthogonalise
        v2 /= np.linalg.norm(v2)
        pairs.append((v1, v2))
    return pairs


def mkdir(pathlike):
    p = pathlib.Path(pathlike); p.mkdir(parents=True, exist_ok=True); return p


def log_hist(H):  # avoid log(0)
    return np.log1p(H, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/workspace/runs/gpt2_L8_ln2_7c1b91f2")
    ap.add_argument("--out", default="heatmaps",
                    help="output directory (created if missing)")
    ap.add_argument("--pairs", default="1,2;1,3;1,4;1,5;1,6;1,7;1,8;1,9;1,10;2,3;2,4;2,5;2,6;2,7;2,8;2,9;2,10;3,4;3,5;3,6;3,7;3,8;3,9;3,10;4,5;4,6;4,7;4,8;4,9;4,10;5,6;5,7;5,8;5,9;5,10;6,7;6,8;6,9;6,10;7,8;7,9;7,10;8,9;8,10;9,10",
                    help='PC pairs, 1‑indexed: e.g. "1,2;1,3;700,701"')
    ap.add_argument("--rand", type=int, default=0,
                    help="# extra random direction pairs")
    ap.add_argument("--bins", type=int, default=200,
                    help="# histogram bins per axis")
    ap.add_argument("--mbatch", type=int, default=10,
                    help="# .npy chunks per mini‑batch")
    ap.add_argument("--sample", type=int, default=3,
                    help="# chunks for quick range (0 = robust)")
    ap.add_argument("--pc3map", action="store_true",
                    help="add PC1×PC2 heat‑map coloured by mean PC3")
    ap.add_argument("--pc3grid", type=int, default=0,
                    help="NxN PC3‑conditioned grid (0 = off)")
    args = ap.parse_args()

    # ---------- load PCA ----------
    pca_path = os.path.join(args.run, "pca.joblib")
    if not os.path.isfile(pca_path):
        sys.exit(f"Missing {pca_path}")
    pca = joblib.load(pca_path)
    mu = pca.mean_.astype(np.float32)
    d = mu.shape[0]

    # ---------- build plane list ----------
    planes, labels = [], []
    for (i, j) in parse_pairs(args.pairs):
        for idx in (i, j):
            if idx >= pca.components_.shape[0]:
                sys.exit(f"PC index {idx+1} in --pairs exceeds PCA dimension")
        planes.append(np.stack([pca.components_[i], pca.components_[j]]))
        labels.append(f"PC{i+1}_PC{j+1}")

    for k, (v1, v2) in enumerate(random_pairs(d, args.rand)):
        planes.append(np.stack([v1, v2]))
        labels.append(f"RND{k}_RND{k}")

    # ensure PC1×PC2 first if we need PC3 features
    if args.pc3map or args.pc3grid:
        if labels[0] != "PC1_PC2":
            planes.insert(0, np.stack([pca.components_[0], pca.components_[1]]))
            labels.insert(0, "PC1_PC2")

    # ---------- activation chunks ----------
    chunks = sorted(glob.glob(os.path.join(args.run, "act_*.npy")))
    if not chunks:
        sys.exit("No act_*.npy files in run directory")

    # ---------- containers ----------
    bins = args.bins
    edges = [None] * len(planes)
    H2d = [None] * len(planes)

    # PC3‑specific
    want_pc3map = args.pc3map
    want_grid = args.pc3grid > 0
    pc3_vec = pca.components_[2].astype(np.float32)
    pc3_cnt = pc3_sum = None
    if want_pc3map:
        pc3_cnt = np.zeros((bins, bins), dtype=np.int64)
        pc3_sum = np.zeros((bins, bins), dtype=np.float64)

    grid_bins = max(args.pc3grid, 1)
    grid3d = None
    pc3_edges = None
    if want_grid:
        grid3d = np.zeros((grid_bins, bins, bins), dtype=np.int64)

    # ---------- quick‑range (optional) ----------
    if args.sample > 0:
        for idx, W in tqdm(enumerate(planes), total=len(planes), desc="Quick range"):
            xs, ys = [], []
            for f in chunks[:args.sample]:
                X = np.load(f, mmap_mode='r', allow_pickle=False).astype(np.float32) - mu
                P = X @ W.T
                xs += [P[:,0].min(), P[:,0].max()]
                ys += [P[:,1].min(), P[:,1].max()]
            edges[idx] = (np.linspace(min(xs), max(xs), bins+1),
                          np.linspace(min(ys), max(ys), bins+1))
            H2d[idx] = np.zeros((bins, bins), dtype=np.int64)

    # ---------- streaming pass ----------
    for s in tqdm(range(0, len(chunks), args.mbatch), desc="stream"):
        batch = chunks[s:s+args.mbatch]
        X = np.concatenate(
            [np.load(f, mmap_mode='r', allow_pickle=False).astype(np.float32)
             for f in batch])
        X -= mu

        # fill per‑plane histograms
        for idx, W in tqdm(enumerate(planes), total=len(planes), desc="Planes", leave=False):
            P = X @ W.T
            if edges[idx] is None:  # first sight – derive range
                edges[idx] = (np.linspace(P[:,0].min(), P[:,0].max(), bins+1),
                              np.linspace(P[:,1].min(), P[:,1].max(), bins+1))
                H2d[idx] = np.zeros((bins, bins), dtype=np.int64)
            h, _, _ = np.histogram2d(P[:,0], P[:,1], bins=edges[idx])
            H2d[idx] += h.astype(np.int64)

        # PC3‑derived maps use first plane (PC1×PC2)
        if want_pc3map or want_grid:
            P12 = X @ planes[0].T
            pc3_vals = X @ pc3_vec

            # universal PC1/PC2 edges
            if 'gex' not in locals():
                gex = np.linspace(P12[:,0].min(), P12[:,0].max(), bins+1)
                gey = np.linspace(P12[:,1].min(), P12[:,1].max(), bins+1)

            ix = np.searchsorted(gex, P12[:,0], side='right')-1
            iy = np.searchsorted(gey, P12[:,1], side='right')-1
            good = (0<=ix)&(ix<bins)&(0<=iy)&(iy<bins)

            if want_pc3map:
                np.add.at(pc3_cnt, (iy[good], ix[good]), 1)
                np.add.at(pc3_sum, (iy[good], ix[good]), pc3_vals[good])

            if want_grid:
                if pc3_edges is None:
                    pc3_edges = np.quantile(pc3_vals,
                                            np.linspace(0,1,grid_bins+1))
                pc3_bin = np.searchsorted(pc3_edges, pc3_vals, side='right')-1
                mask = good & (0<=pc3_bin)&(pc3_bin<grid_bins)
                np.add.at(grid3d, (pc3_bin[mask], iy[mask], ix[mask]), 1)

    # ---------- save plots ----------
    outdir = mkdir(args.out)

    for idx, H in tqdm(enumerate(H2d), total=len(H2d), desc="Saving plots"):
        plt.figure(figsize=(6,5))
        plt.imshow(log_hist(H).T, origin="lower", aspect="auto",
                   extent=[edges[idx][0][0], edges[idx][0][-1],
                           edges[idx][1][0], edges[idx][1][-1]],
                   cmap="viridis")
        plt.xlabel(labels[idx].split('_')[0])
        plt.ylabel(labels[idx].split('_')[1])
        plt.title(labels[idx])
        plt.colorbar(label="log(count)")
        plt.tight_layout()
        plt.savefig(outdir / f"{labels[idx]}.png", dpi=300)
        plt.close()

    # mean PC3 coloured map
    if want_pc3map:
        mean_pc3 = np.divide(pc3_sum, pc3_cnt, where=pc3_cnt>0)
        plt.figure(figsize=(6,5))
        plt.imshow(mean_pc3.T, origin="lower", aspect="auto",
                   extent=[gex[0], gex[-1], gey[0], gey[-1]],
                   cmap="coolwarm")
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.title("PC1_PC2_meanPC3")
        plt.colorbar(label="mean PC3")
        plt.tight_layout()
        plt.savefig(outdir / "PC1_PC2_meanPC3.png", dpi=300)
        plt.close()

    # PC3‑conditioned grid
    if want_grid:
        g = grid_bins
        fig, axes = plt.subplots(g, g, figsize=(3.5*g, 3.5*g), sharex=True, sharey=True)
        for b in range(g):
            data = log_hist(grid3d[b])
            for a in range(g):
                ax = axes[b,a]
                im = ax.imshow(data.T, origin="lower", aspect="auto",
                               extent=[gex[0], gex[-1], gey[0], gey[-1]],
                               cmap="viridis")
                ax.set_title(f"{pc3_edges[b]:.2f} ≤ PC3 < {pc3_edges[b+1]:.2f}",
                             fontsize=8)
                if b==g-1: ax.set_xlabel("PC1")
                if a==0:   ax.set_ylabel("PC2")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="log(count)")
        fig.tight_layout()
        fig.savefig(outdir / f"PC1_PC2_gridPC3_{g}x{g}.png", dpi=300)
        plt.close()

    print("✓ Plots saved in", outdir)


if __name__ == "__main__":
    main()
