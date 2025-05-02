#!/usr/bin/env python3
"""
make_pca_from_tags.py
─────────────────────
Fit a PCA *inside* an ActivationDB run, but only on rows that satisfy a
Boolean filter (tags, token IDs, positions …).

Typical use
-----------
python make_pca_from_tags.py \
    --run   runs/gpt2_L6:ln1,L6:ln2_abcd1234 \
    --hook  L6:ln1 \
    --filter "(tags['NOUN'] & (d['pos'] < 3))" \
    --max_tokens 200_000 \
    --n_components 128
"""

from __future__ import annotations
import argparse, json, pathlib, hashlib, gc, random

from typing import Dict, Any
import numpy as np
import torch
from tqdm.auto import tqdm
import joblib
from sklearn.decomposition import IncrementalPCA as IPCA

from actdb_gpu import ActivationDB               # ← same module as your DB

# ───────────────────────── helpers ───────────────────────────────────────
def sha8(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True,
                                   default=str).encode()).hexdigest()[:8]

def prepare_tag_keys(db: ActivationDB, device):
    return {n: t.to(device) for n, t in db._tag_tokens.items()}

def chunk_tag_masks(tag_keys: Dict[str, torch.Tensor],
                    tok_keys_chunk: torch.Tensor) -> Dict[str, np.ndarray]:
    return {n: torch.isin(tok_keys_chunk, tk).cpu().numpy()
            for n, tk in tag_keys.items()}

# ─────────────────────────── main ────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/workspace/gpt_act_distribution/runs/gpt2_L6:ln1,L6:ln2_run_1", help="ActivationDB run dir")
    ap.add_argument("--hook", help="single hook (e.g. L6:ln1); default = all")
    ap.add_argument("--filter", required=True,
                    help="Python expr → Boolean mask (see doc-string)")
    ap.add_argument("--max_tokens", type=int, default=20_000,
                    help="stop after this many rows matched")
    ap.add_argument("--n_components", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8192,
                    help="IPCA internal batch size")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run).expanduser()
    device  = torch.device(args.device)

    # ---------------------------------------------------------------------
    db = ActivationDB(str(run_dir), verbose=True)
    tag_keys = prepare_tag_keys(db, device)

    hi_filter = db._hook2i.get(args.hook) if args.hook else None
    ipca = IPCA(n_components=args.n_components, batch_size=args.batch_size)

    filt_code = compile(args.filter, "<filter>", "eval")

    processed = 0
    rng = np.random.default_rng(0)

    for d in tqdm(db._mem, desc="chunks"):
        # hook selection
        h_mask = (d["src"] == hi_filter) if hi_filter is not None else slice(None)
        if isinstance(h_mask, np.ndarray) and not h_mask.any():
            continue

        keys_np = ((d["seq"].astype(np.int64) << 32) |
                   d["seq_ofs"].astype(np.int64))[h_mask]
        keys_t  = torch.from_numpy(keys_np).to(device)
        tags    = chunk_tag_masks(tag_keys, keys_t)

        # evaluate Boolean filter
        loc_dict = {"d": {k: v[h_mask] for k, v in d.items()},
                    "tags": tags}
        mask = eval(filt_code, {"np": np, "torch": torch}, loc_dict)
        if mask.dtype != bool:
            mask = mask.astype(bool)
        if not mask.any():
            continue

        X = d["h"][h_mask][mask].astype(np.float32)
        if processed + len(X) > args.max_tokens:
            surplus = processed + len(X) - args.max_tokens
            keep    = len(X) - surplus
            idx     = rng.choice(len(X), keep, replace=False)
            X = X[idx]
        ipca.partial_fit(X)
        processed += len(X)

        if processed >= args.max_tokens:
            break
        gc.collect()

    if processed < args.n_components:
        raise RuntimeError(f"Only {processed} rows matched; "
                           f"need ≥ n_components ({args.n_components}).")

    # ---------------------------------------------------------------------
    meta = dict(run=str(run_dir), hook=args.hook, filter=args.filter,
                max_tokens=args.max_tokens, n_components=args.n_components)
    out_path = run_dir / f"pca_{args.hook or 'ALL'}_{sha8(meta)}.joblib"
    joblib.dump({"pca": ipca, "meta": meta}, out_path)
    print(f"✓ PCA saved → {out_path}  (rows used: {processed:,})")

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
