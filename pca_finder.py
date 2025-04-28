#!/usr/bin/env python3
"""
mini_pca_debug.py
=================
Collect activations from any hook‐point in a GPT‑2 block and perform PCA
with ample sanity logging.  Each run goes into its own directory so
results never mix.

Hook choices
------------
    pre   = input to ln_1  (classic  "resid_pre")
    ln1   = output of ln_1 (default)
    ln2   = output of ln_2 (pre‑MLP)
    post  = output of MLP  ("resid_post")

Example:
    python mini_pca_debug.py --layer 8 --hook ln2 --tokens 3_000_000

TODO please stop using np.load/np.save with pickling
"""
import argparse, os, json, hashlib, gc, random, re, glob, sys
from typing import List
import numpy as np
import torch, matplotlib.pyplot as plt, joblib
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA as CPU_IPCA

# ───────────────────────── helpers ────────────────────────────
def cfg_hash(d: dict) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]

def run_dir(cfg: dict) -> str:
    tag = f"L{cfg['layer']}_{cfg['hook']}_{cfg_hash(cfg)}"
    p   = os.path.join("runs", f"{cfg['model']}_{tag}")
    os.makedirs(p, exist_ok=True)
    meta = os.path.join(p, "meta.json")
    if os.path.exists(meta) and json.load(open(meta)) != cfg:
        sys.exit(f"⚠️  {p} already exists with different parameters – delete or use --fresh")
    json.dump(cfg, open(meta, "w"), indent=2)
    return p

# ───────────────────────── hook machinery ─────────────────────
_buf: List[torch.Tensor] = []
def pre_ln(_, inputs):      _buf.append(inputs[0].detach())
def post_ln(_, __, output): _buf.append(output.detach())
def resid_post(_, __, output): _buf.append(output.detach())

def attach(block, mode: str):
    if mode == "pre":   # before ln_1
        return block.ln_1.register_forward_pre_hook(pre_ln)
    if mode == "ln1":   # after  ln_1
        return block.ln_1.register_forward_hook(post_ln)
    if mode == "ln2":   # after  ln_2  (pre‑MLP)
        return block.ln_2.register_forward_hook(post_ln)
    if mode == "post":  # after MLP (resid_post)
        return block.register_forward_hook(resid_post)
    raise ValueError(f"Unknown hook mode {mode}")

# ───────────────────────── collection ─────────────────────────
def collect(cfg, model, tok, ds, device, outdir):
    d, saved, idx, buf, txts = model.config.hidden_size, 0, 0, [], []
    pat  = os.path.join(outdir, "act_{:04d}.npy")
    hook = attach(model.transformer.h[cfg["layer"]], cfg["hook"])
    bar  = tqdm(total=cfg["tokens"], desc="COLLECT", unit="tok")

    for step, ex in enumerate(ds):
        if saved >= cfg["tokens"]:
            break
        t = ex.get("text", "").strip()
        if not t or re.match(r"^=+.*=+$", t):
            continue
        txts.append(t)
        if len(txts) < cfg["batch"]:
            continue

        inp = tok(txts, return_tensors="pt",
                  padding=True, truncation=True, max_length=cfg["seq"])
        txts.clear()
        inp = {k: v.to(device) for k, v in inp.items()}

        global _buf; _buf = []
        with torch.inference_mode():
            model(**inp)

        acts  = torch.cat(_buf, 0)
        mask  = inp["attention_mask"].view(-1).bool()
        pad   = tok.pad_token_id or tok.eos_token_id
        mask &= inp["input_ids"].view(-1).ne(pad)
        pos   = torch.arange(mask.size(0), device=device) % cfg["seq"]
        mask &= pos.ne(0)
        sel   = acts.view(-1, d)[mask].cpu().float()

        keep = sel.size(0)
        if keep:
            print(f"[BATCH {step:05d}] keep={keep:5d}  "
                  f"μ∞={sel.mean(0).abs().max():7.3e}  "
                  f"σ∞={sel.std(0).max():7.3e}")
        take = min(keep, cfg["tokens"] - saved)
        if take:
            buf.append(sel[:take])
            saved += take
            bar.update(take)

        if sum(b.size(0) for b in buf) >= 10*d:
            np.save(pat.format(idx), torch.cat(buf, 0).numpy())
            buf.clear(); idx += 1
        torch.cuda.empty_cache(); gc.collect()

    if buf:
        np.save(pat.format(idx), torch.cat(buf, 0).numpy())
    bar.close(); hook.remove()
    return saved, d

# ───────────────────────── PCA helpers ────────────────────────
def mean_and_drift(files, limit=None):
    tot, cnt = None, 0
    for f in files:
        X = np.load(f, mmap_mode="r")
        cnt += len(X); tot = X.sum(0) if tot is None else tot + X.sum(0)
    mu = tot / cnt
    drift = max(abs(np.load(f).mean(0) - mu).max() for f in files[:3])
    print(f"[DRIFT] max residual drift ≈ {drift:.3e} (≪1e‑3 ideal)")
    return mu

def incremental_pca(files, mu, k, limit=None):
    """
    Perform Incremental PCA on activation data stored in multiple files,
    potentially using only a subset of the data.

    Args:
        files: List of paths to .npy files containing activation batches.
        mu: The pre-calculated mean vector of the activations.
        k: The number of principal components to compute.
        limit: Maximum number of activation vectors to consider. If None, use all.

    Returns:
        The fitted IncrementalPCA object.
    """
    ipca = CPU_IPCA(n_components=k)
    mu32 = mu.astype(np.float32)
    processed_count = 0

    print(f"Running IPCA for {k} components on {limit if limit else 'all'} tokens...")
    bar = tqdm(total=limit if limit else None, desc="IPCA", unit="tok")

    for f in files:
        if limit is not None and processed_count >= limit:
            break

        X = np.load(f).astype(np.float32)
        take = len(X)
        if limit is not None:
            remaining_limit = limit - processed_count
            take = min(take, remaining_limit)

        if take <= 0:
            continue

        X_chunk = X[:take] - mu32
        norm_mask = np.linalg.norm(X_chunk, axis=1) > 1e-6
        X_filtered = X_chunk[norm_mask]

        if X_filtered.shape[0] > 0:
            ipca.partial_fit(X_filtered)
            update_count = X_chunk.shape[0] # Update bar based on original chunk size before filtering
            bar.update(update_count)
            processed_count += update_count

        # Clean up
        del X, X_chunk, X_filtered, norm_mask
        gc.collect()

    bar.close()
    print(f"IPCA finished after processing {processed_count:,} tokens.")
    return ipca

# ───────────────────────── main ───────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--hook", choices=["pre", "ln1", "ln2", "post"], default="ln2")
    ap.add_argument("--dataset", default="openwebtext")
    ap.add_argument("--config",  default=None)
    ap.add_argument("--tokens",  type=int, default=200_000_0)
    ap.add_argument("--batch",   type=int, default=256)
    ap.add_argument("--seq",     type=int, default=512)
    ap.add_argument("--k",       type=int, default=None)
    ap.add_argument("--pca-tokens", type=int, default=200_000, help="Max tokens for PCA (default: all collected)")
    ap.add_argument("--cpu",     action="store_true")
    ap.add_argument("--fresh",   action="store_true")
    cfg = vars(ap.parse_args())

    outdir = run_dir(cfg)
    if cfg["fresh"]:
        for f in glob.glob(os.path.join(outdir, "act_*.npy")):
            os.remove(f)

    random.seed(42); np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg["cpu"] else "cpu")

    model = AutoModelForCausalLM.from_pretrained(cfg["model"]).eval().to(device)
    tok   = AutoTokenizer.from_pretrained(cfg["model"])
    tok.pad_token = tok.eos_token

    ds = load_dataset(cfg["dataset"], cfg["config"], split="train", streaming=True, trust_remote_code=True)
    ds = ds.shuffle(buffer_size=100_000, seed=42)

    saved, d = collect(cfg, model, tok, ds, device, outdir)
    print(f"Collected {saved:,} tokens (hidden dim = {d})")

    files = sorted(glob.glob(os.path.join(outdir, "act_*.npy")))
    # Ensure pca_tokens doesn't exceed collected tokens
    pca_token_limit = cfg["pca_tokens"]
    if pca_token_limit is not None:
        if pca_token_limit > saved:
            print(f"Warning: --pca-tokens ({pca_token_limit:,}) > collected tokens ({saved:,}). Using {saved:,} instead.")
            pca_token_limit = saved
        elif pca_token_limit <= 0:
             print(f"Warning: --pca-tokens ({pca_token_limit:,}) is non-positive. Using all collected tokens ({saved:,}).")
             pca_token_limit = None # Use all collected

    mu    = mean_and_drift(files, limit=pca_token_limit)

    k   = min(d, cfg["k"] or d)
    pca = incremental_pca(files, mu, k, limit=pca_token_limit)
    evr = np.cumsum(pca.explained_variance_ratio_)
    k95 = int(np.searchsorted(evr, 0.95) + 1)
    print(f"95 % variance at k = {k95}/{d}")

    plt.plot(range(1, len(evr)+1), evr); plt.axhline(.95, ls="--")
    plt.xlabel("# PCs"); plt.ylabel("cumulative EVR"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "evr.png"))

    joblib.dump(pca, os.path.join(outdir, "pca.joblib"))
    np.save(os.path.join(outdir, "components.npy"), pca.components_)

if __name__ == "__main__":
    main()
