#!/usr/bin/env python3
"""
try_tag_from_directions.py
─────────────────────────
Load an ActivationDB run and a directions file (.npy). Create a new tag
containing tokens whose activations (at a specific hook) project onto
two specified directions (e.g., PC1, PC2) within given bounds.

Example:
python try_tag_from_directions.py \
    --run runs/gpt2_L6:ln1,L7:ln2_91979bad \
    --dirs runs/gpt2_L6:ln1,L7:ln2_91979bad/directions/dirs_ALL.npy \
    --hook L6:ln1 \
    --tag_name PC1_PC2_box \
    --pc_indices 0 1 \
    --bounds 0 1 0 1
"""

from __future__ import annotations
import argparse
import pathlib
import gc
import numpy as np
import torch
from tqdm.auto import tqdm
import joblib  # Used by ActivationDB implicitly

from actdb_gpu import ActivationDB

def main() -> None:
    ap = argparse.ArgumentParser(description="Add tag based on activation projections onto directions.")
    ap.add_argument("--run", default="/workspace/gpt_act_distribution/runs/gpt2_L6:ln1,L7:ln2_91979bad",
                    help="ActivationDB run directory")
    ap.add_argument("--dirs", default="/workspace/gpt_act_distribution/runs/gpt2_L6:ln1,L7:ln2_91979bad/directions/dirs_ALL.npy",
                    help="Path to the directions file (.npy, shape KxD)")
    ap.add_argument("--hook", default="L6:ln1",
                    help="Single hook name (e.g., L6:ln1) to filter activations")
    ap.add_argument("--tag_name", default="chunk_in_6_vs_7",
                    help="Name for the new tag")
    ap.add_argument("--pc_indices", type=int, nargs=2, default=[5, 6],
                    help="Indices of the two directions (PCs) to use")
    ap.add_argument("--bounds", type=float, nargs=4, default=[-3, -1.5, -1.0, 1.0],
                    metavar=("MIN1", "MAX1", "MIN2", "MAX2"),
                    help="Bounds for the projections [min1, max1] and [min2, max2]")
    ap.add_argument("--centered", default=False,
                    help="Interpret bounds relative to the mean projection for the hook")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to use for potential torch operations (though mostly numpy here)")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run).expanduser()
    dirs_fp = pathlib.Path(args.dirs).expanduser()
    device = torch.device(args.device) # Keep for consistency, though not heavily used

    print(f"Loading ActivationDB from: {run_dir}")
    db = ActivationDB(str(run_dir), verbose=True) # Enable verbosity for random_windows

    print(f"Loading directions from: {dirs_fp}")
    if not dirs_fp.is_file():
        raise FileNotFoundError(f"Directions file not found: {dirs_fp}")
    dirs_all = np.load(dirs_fp) # K x D

    if dirs_all.shape[1] != db.d:
        raise ValueError(f"Direction dimension ({dirs_all.shape[1]}) does not match DB activation dimension ({db.d})")

    # --- Select directions and bounds ---
    idx1, idx2 = args.pc_indices
    if not (0 <= idx1 < dirs_all.shape[0] and 0 <= idx2 < dirs_all.shape[0]):
        raise ValueError(f"PC indices {args.pc_indices} out of bounds for {dirs_all.shape[0]} directions")
    if idx1 == idx2:
        raise ValueError("PC indices must be different")

    dir1 = dirs_all[idx1].astype(np.float32) # Shape (D,)
    dir2 = dirs_all[idx2].astype(np.float32) # Shape (D,)
    user_min1, user_max1, user_min2, user_max2 = args.bounds
    print(f"Using directions {idx1} and {idx2}.")

    # --- Find hook index ---
    hook_name = args.hook
    if hook_name not in db._hook2i:
        raise ValueError(f"Hook '{hook_name}' not found in DB hooks: {db.hooks}")
    hi = db._hook2i[hook_name]
    print(f"Filtering for hook: {hook_name} (index {hi})")

    # --- Calculate Mean Projections (if centered) ---
    mean_proj1, mean_proj2 = 0.0, 0.0
    fmin1, fmax1, fmin2, fmax2 = user_min1, user_max1, user_min2, user_max2

    if args.centered:
        print("Calculating mean projections (Pass 1)...")
        sum_proj1, sum_proj2 = 0.0, 0.0
        total_count = 0
        for d in tqdm(db._mem, desc="Mean Calc (Pass 1)"):
            mask_hook = (d["src"] == hi)
            if not mask_hook.any(): continue
            X = d['h'][mask_hook].astype(np.float32)
            chunk_proj1 = X @ dir1
            chunk_proj2 = X @ dir2
            sum_proj1 += np.sum(chunk_proj1)
            sum_proj2 += np.sum(chunk_proj2)
            total_count += len(X)
            gc.collect()

        if total_count == 0:
            raise ValueError(f"No activations found for hook '{hook_name}'. Cannot calculate mean.")

        mean_proj1 = sum_proj1 / total_count
        mean_proj2 = sum_proj2 / total_count
        fmin1 = mean_proj1 + user_min1
        fmax1 = mean_proj1 + user_max1
        fmin2 = mean_proj2 + user_min2
        fmax2 = mean_proj2 + user_max2
        print(f"  Mean Proj PC{idx1}: {mean_proj1:.4f}, Mean Proj PC{idx2}: {mean_proj2:.4f}")
        print(f"  Effective Bounds (Centered): [{fmin1:.4f}, {fmax1:.4f}] x [{fmin2:.4f}, {fmax2:.4f}]")
    else:
        print(f"Using Absolute Bounds: [{fmin1}, {fmax1}] x [{fmin2}, {fmax2}]")

    # --- Iterate through chunks and collect keys (Pass 2 or only pass) ---
    tagged_token_keys = set()
    tagged_token_positions = []
    processed_activations = 0
    pass_desc = "Tagging (Pass 2)" if args.centered else "Tagging"

    for i, d in enumerate(tqdm(db._mem, desc=pass_desc)):
        # 1. Filter by hook
        mask_hook = (d["src"] == hi)
        if not mask_hook.any():
            continue

        # 2. Get activations for the hook
        X = d['h'][mask_hook].astype(np.float32) # N_hook x D
        processed_activations += X.shape[0]

        # 3. Calculate projections
        proj1 = X @ dir1 # N_hook
        proj2 = X @ dir2 # N_hook

        # 4. Apply bounds condition
        mask_bounds = (proj1 >= fmin1) & (proj1 <= fmax1) & (proj2 >= fmin2) & (proj2 <= fmax2)

        if not mask_bounds.any():
            continue

        # 5. Get the token keys for activations satisfying both hook and bounds
        seq_nums = d['seq'][mask_hook][mask_bounds]
        seq_offsets = d['seq_ofs'][mask_hook][mask_bounds]
        positions = d['pos'][mask_hook][mask_bounds]
        keys = (seq_nums.astype(np.int64) << 32) | seq_offsets.astype(np.int64)

        # 6. Add unique keys to the set
        tagged_token_keys.update(keys)
        tagged_token_positions.extend(positions.tolist())
        gc.collect() # Maybe helpful with large datasets

    # --- Save the tag ---
    tag_array = np.array(sorted(list(tagged_token_keys)), dtype=np.int64)

    tags_dir = run_dir / "tags"
    tags_dir.mkdir(exist_ok=True) # Ensure the tags directory exists
    tag_file_path = tags_dir / f"tag_{args.tag_name}.npy"

    np.save(tag_file_path, tag_array)

    print(f"\nProcessed {processed_activations:,} activations for hook '{hook_name}'.")
    print(f"Found {len(tag_array):,} tokens satisfying the projection bounds.")
    print(f"✓ Tag '{args.tag_name}' saved to: {tag_file_path}")

    # --- Print position statistics ---
    if tagged_token_positions:
        pos_array = np.array(tagged_token_positions)
        print("\nStatistics for token positions within the tag:")
        print(f"  Min: {np.min(pos_array)}")
        print(f"  Max: {np.max(pos_array)}")
        print(f"  Mean: {np.mean(pos_array):.2f}")
        print(f"  Median: {np.median(pos_array):.0f}")

    # --- Sample and print windows for the new tag ---
    if len(tag_array) > 0:
        print(f"\nSampling {min(10, len(tag_array))} windows for tag '{args.tag_name}':")
        # Manually add the tag to the loaded DB instance
        db._tag_tokens[args.tag_name] = torch.from_numpy(tag_array).to(db.device)
        # Use the existing random_windows method
        windows = db.random_windows(tag=args.tag_name, k=10, colour=True)
        for i, (window_str, trigger_id) in enumerate(windows):
            print(f"{i+1: >3}: {window_str}  | Trigger ID: {trigger_id}")
    else:
        print(f"\nNo tokens found for tag '{args.tag_name}', cannot sample windows.")

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high") # Consistent with other scripts
    main()
