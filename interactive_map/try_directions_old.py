#!/usr/bin/env python3
"""
try_directions.py
─────────────────
Write one PCA "direction" file inside an ActivationDB run directory
based on activations associated with a specific tag.

    directions/dirs_<NAME>.npy  – PCA on rows with the specified tag

The matrix has shape (k, d) and a sibling JSON with provenance.
NAME defaults to the tag name if --name is not provided.
"""

from __future__ import annotations
import argparse, json, pathlib, hashlib, gc
from typing import Any, Dict

import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.decomposition import IncrementalPCA as IPCA

from actbd_gpu import ActivationDB     # ← your DB toolkit

# ───────────────────────── helpers ───────────────────────────────────────
def sha8(obj: Any) -> str:
    return hashlib.sha1(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]

def np2py(x: Any):          # universal NumPy-scalar → Python cast
    return x.item() if isinstance(x, np.generic) else x

def tag_keys_cuda(db: ActivationDB, device):
    return {n: t.to(device) for n, t in db._tag_tokens.items()}

def chunk_tag_masks(tag_keys: Dict[str, torch.Tensor],
                    tok_keys_chunk: torch.Tensor) -> Dict[str, np.ndarray]:
    return {n: torch.isin(tok_keys_chunk, v).cpu().numpy()
            for n, v in tag_keys.items()}

def stream_ipca(db: ActivationDB,
                filt_expr: str,
                hook: str | None,
                max_rows: int,
                n_components: int,
                batch_size: int,
                device: torch.device) -> IPCA:
    hi_filter = db._hook2i.get(hook) if hook else None
    tag_keys  = tag_keys_cuda(db, device)

    ipca  = IPCA(n_components=n_components, batch_size=batch_size)
    rows  = 0
    rng   = np.random.default_rng(0)
    code  = compile(filt_expr, "<filter>", "eval")

    for d in tqdm(db._mem, desc="IPCA-stream"):
        hook_mask = (d["src"] == hi_filter) if hi_filter is not None else slice(None)
        if isinstance(hook_mask, np.ndarray) and not hook_mask.any():
            continue

        keys_np = ((d["seq"].astype(np.int64) << 32) |
                   d["seq_ofs"].astype(np.int64))[hook_mask]
        keys_t  = torch.from_numpy(keys_np).to(device)
        tags    = chunk_tag_masks(tag_keys, keys_t)

        loc = {"d": {k: v[hook_mask] for k, v in d.items()}, "tags": tags}
        mask = eval(code, {"np": np}, loc)
        mask = mask.astype(bool) if mask.dtype != bool else mask
        if not mask.any():
            continue

        X = d["h"][hook_mask][mask].astype(np.float32)
        if rows + len(X) > max_rows:
            keep = max_rows - rows
            X    = X[rng.choice(len(X), keep, replace=False)]
        ipca.partial_fit(X)
        rows += len(X)
        if rows >= max_rows:
            break
        gc.collect()

    if rows < n_components:
        raise RuntimeError(f"Only {rows} rows matching filter → need ≥ {n_components}.")
    return ipca

# ─────────────────────────── main ────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Calculate PCA directions for activations matching a specific tag.")
    ap.add_argument("--run",  default="/workspace/gpt_act_distribution/runs/gpt2_L6:ln1,L7:ln2_91979bad")
    ap.add_argument("--hook", help="Single hook (e.g. L6:ln1); default = first hook found")
    ap.add_argument("--tag", default="token_290", help="Tag name to filter activations for PCA.")
    ap.add_argument("--name", help="Optional name for output files (dirs_<name>.npy/json). Defaults to the tag name.")
    ap.add_argument("--k",            type=int, default=50, help="Number of principal components.")
    ap.add_argument("--max_tokens",   type=int, default=100_000, help="Maximum number of tokens (rows) to use for PCA.")
    ap.add_argument("--batch_size",   type=int, default=8192, help="Batch size for IncrementalPCA.")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run).expanduser()
    device  = torch.device(args.device)
    db      = ActivationDB(str(run_dir), verbose=True)

    hook = args.hook or db.hooks[0]
    print(f"Using hook: {hook}")

    if not db._tag_tokens:
        raise RuntimeError("No tags found in this run directory.")
    if args.tag not in db._tag_tokens:
        raise ValueError(f"Tag '{args.tag}' not found in the run. Available tags: {list(db._tag_tokens.keys())}")

    tag_name = args.tag
    label = args.name or tag_name # Use provided name or default to tag name
    filt = f"tags['{tag_name}']"

    out_dir = run_dir / "directions"; out_dir.mkdir(exist_ok=True)

    print(f"\n=== Calculating PCA for tag '{tag_name}' (output label: {label}) ===")
    ipca = stream_ipca(db, filt, hook,
                       max_rows=args.max_tokens,
                       n_components=args.k,
                       batch_size=args.batch_size,
                       device=device)

    dirs = ipca.components_.astype(np.float32)
    out_npy_path = out_dir / f"dirs_{label}.npy"
    out_json_path = out_dir / f"dirs_{label}.json"
    np.save(out_npy_path, dirs)

    meta = {k: np2py(v) for k, v in dict(
        hook       = hook,
        tag        = tag_name, # Record the specific tag used
        filter     = filt,
        k          = args.k,
        max_tokens = args.max_tokens,
        rows       = ipca.n_samples_seen_
    ).items()}
    out_json_path.write_text(json.dumps(meta, indent=2))

    print(f"✓ PCA directions saved → {out_npy_path}  (rows used: {meta['rows']:,})")
    print(f"✓ Metadata saved      → {out_json_path}")

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
