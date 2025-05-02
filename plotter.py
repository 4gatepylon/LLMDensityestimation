#!/usr/bin/env python3
"""
plot_directions.py   ·   v0.2
────────────────────────────────────────────────────────────────────────────
Visualise (at most) the first K principal directions stored in a dirs_*.npy
file that lives inside an ActivationDB run.

For every unordered pair (i,j) among those K directions the script
creates one figure with (1 + n_colour_funcs) sub-plots:

 • panel 0      – log₁₀ density 2-D histogram
 • panel m>0    – mean( colour_fn[m-1] ) over the same 2-D bins
"""
from __future__ import annotations
import argparse, pathlib, gc, json, hashlib, re
from typing import Any, List, Dict, Tuple
import time

import numpy as np
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from actdb_gpu import ActivationDB
# ───────────────────────── helpers ───────────────────────────────────────
def sha8(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True,
                                   default=str).encode()).hexdigest()[:8]

def compile_exprs(exprs: List[str]):
    """Return list[(src, code)]."""
    return [ (e, compile(e, f"<expr:{i}>", "eval"))
             for i, e in enumerate(exprs) ]

def tag_keys_cuda(db: ActivationDB, device):
    return {n: t.to(device) for n, t in db._tag_tokens.items()}

def chunk_tag_masks(tag_keys, tok_keys):
    return {n: torch.isin(tok_keys, v).cpu().numpy()
            for n, v in tag_keys.items()}

def _get_next_run_subdir(base_dir: pathlib.Path) -> pathlib.Path:
    """Finds the next available run_### directory."""
    base_dir.mkdir(parents=True, exist_ok=True)
    existing_runs = [int(p.name.split('_')[-1]) for p in base_dir.glob('run_*') if p.name.split('_')[-1].isdigit()]
    next_run_num = max(existing_runs, default=-1) + 1
    run_subdir = base_dir / f"run_{next_run_num:03d}"
    run_subdir.mkdir()
    return run_subdir

# ─────────────────────────── main ────────────────────────────────────────
def main() -> None:
    pa = argparse.ArgumentParser()
    pa.add_argument("--run",     default="/workspace/gpt_act_distribution/runs/gpt2_L6:ln1,L7:ln2_random")
    pa.add_argument("--dirs",    default="/workspace/gpt_act_distribution/runs/gpt2_L6:ln1,L7:ln2_random/directions/dirs_ALL.npy")# default="/workspace/gpt_act_distribution/runs/gpt2_L6:ln1,L7:ln2_91979bad/directions/dirs_ALL.npy", help="k×d .npy file")
    pa.add_argument("--hook", type=str, default="L6:ln1",
                    help="Hook name (e.g., 'L6:ln1') to source activations from. Defaults to the first hook found in the run.")
    pa.add_argument("--k",         type=int, default=9,
                    help="number of directions to plot (≤ rows in dirs)")
    pa.add_argument("--filter",
                    default="np.ones_like(d['id'], dtype=bool)", # default="tags['chunk_in_6_vs_7']", # default="np.ones_like(d['id'], dtype=bool)",
                    help="Python expr → Boolean mask")
    pa.add_argument("--colors", nargs="*", default=[],
                    help="zero or more Python exprs; empty → only density")
    pa.add_argument("--bins",     type=int, default=200)
    pa.add_argument("--max_tok",  type=int, default=400_000)
    pa.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    pa.add_argument("--out_dir")
    pa.add_argument("--plot_pos", default=True, action="store_true",
                    help="Add a panel showing mean token position per bin")
    pa.add_argument("--plot_tag_avg_color", default=True,
                    help="Add a panel showing the average color based on token tags (default: True).")
    pa.add_argument("--tag_colors", type=str, default="{}",
                    help="JSON string mapping tag names to colors (e.g., '{\"NOUN\": \"blue\", \"token_the\": \"#FF8C00\"}'). "
                         "Unspecified tags get random colors.")
    args = pa.parse_args()

    run_dir  = pathlib.Path(args.run).expanduser()
    dirs_fp  = pathlib.Path(args.dirs).expanduser()
    device   = torch.device(args.device)

    # ────────────────── open DB & load directions
    db    = ActivationDB(str(run_dir), verbose=True)
    hook  = args.hook or db.hooks[0]
    hi    = db._hook2i.get(hook)
    if hi is None:
        raise ValueError(f"hook '{hook}' not found in run")

    dirs   = np.load(dirs_fp)                     # k_total × d
    k_total, d = dirs.shape
    if d != db.d:
        raise ValueError(f"direction dim {d} ≠ activation dim {db.d}")
    k_plot = min(args.k, k_total)
    dirs   = dirs[:k_plot]                        # k_plot × d

    print(f"Using first {k_plot}/{k_total} directions from {dirs_fp.name}")

    # ────────────────── compile filter / colour fns
    filt_code           = compile(args.filter, "<filter>", "eval")
    colour_exprs = compile_exprs(args.colors)
    tag_keys      = tag_keys_cuda(db, device)
    all_tag_names = list(db._tag_tokens.keys()) # Get all available tag names

    # --- Generate Base Tag Color Map (Always) --- START
    print("DEBUG: Generating base color map for tags...")
    start_time_color_gen = time.time()
    tag_color_map = {}
    # Assign random colors to all tags initially
    # Use a deterministic but varied color generation
    rng_color = np.random.default_rng(123) # Seed for reproducibility
    hue_step = 0.61803398875 # Golden ratio conjugate for hue distribution
    current_hue = rng_color.random()
    assigned_colors = set()

    num_total_tags = len(all_tag_names)
    print(f"  DEBUG: Found {num_total_tags} total tags in the database.")
    token_tags_processed = 0

    for tag_idx, tag in enumerate(all_tag_names):
        if "token" in tag: # <-- Only process tags with "token" in their name
            # --- REMOVED distinctness check loop --- START
            # Generate color with good separation
            current_hue = (current_hue + hue_step) % 1.0
            saturation = rng_color.uniform(0.6, 1.0)
            value = rng_color.uniform(0.7, 1.0) # Value > 0.7 avoids dark colors/black
            rgb_color = mcolors.hsv_to_rgb((current_hue, saturation, value))
            rgb_tuple = tuple(rgb_color)

            # Explicit check to avoid near-black colors
            if np.linalg.norm(np.array(rgb_tuple)) < 0.2: # Check magnitude (distance from black)
                # If too dark, just assign a default medium gray instead of looping
                print(f"    DEBUG: Generated near-black color for tag '{tag}', assigning gray.")
                rgb_tuple = (0.6, 0.6, 0.6)

            tag_color_map[tag] = rgb_tuple
            # assigned_colors.add(rgb_tuple) # No longer needed as we don't check
            token_tags_processed += 1
            if token_tags_processed % 500 == 0: # PRINT PROGRESS EVERY 500 TOKEN TAGS
                 print(f"    DEBUG: Processed {token_tags_processed}/{num_total_tags} tags for color map...")
            # --- REMOVED distinctness check loop --- END
        # else: tag name does not contain "token", so we skip color assignment for it

    end_time_color_gen = time.time() # ADDED TIMING
    print(f"DEBUG: Generated base color map for {len(tag_color_map)} token tags. (Took {end_time_color_gen - start_time_color_gen:.2f} seconds)")
    # --- Generate Base Tag Color Map (Always) --- END

    # --- Override with User-Specified Colors (if plotting tag avg) --- START
    if args.plot_tag_avg_color:
        print("DEBUG: Applying user-specified tag colors...")
        user_token_tag_colors = {}
        try:
            user_token_tag_colors = json.loads(args.tag_colors)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse --tag_colors JSON: {e}. Using random colors for all tags.")

        # Keep track of colors overridden by user to ensure final map is consistent
        final_assigned_colors = set(tag_color_map.values())

        # Convert provided colors to RGB tuples (0-1 range)
        for tag, color_str in user_token_tag_colors.items():
            if "token" not in tag:
                 print(f"Warning: Skipping user color for tag '{tag}' as it does not contain 'token'.")
                 continue
            try:
                user_rgb = mcolors.to_rgb(color_str)
                # Check if user color is black/too dark
                if np.linalg.norm(np.array(user_rgb)) < 0.1:
                    print(f"Warning: User-specified color {color_str} for tag '{tag}' is black or very dark. Keeping random color.")
                else:
                    # Remove old random color, add new user color
                    if tag in tag_color_map:
                        final_assigned_colors.discard(tag_color_map[tag])
                    tag_color_map[tag] = user_rgb
                    final_assigned_colors.add(user_rgb)
            except ValueError:
                print(f"Warning: Invalid color '{color_str}' for tag '{tag}'. Assigning random color.")
                # If invalid, it keeps the original random color which is already assigned

        print("DEBUG: Finished applying user colors. Final Tag Colors for Plotting:", {tag: f"{tuple(round(c, 2) for c in color)}" for tag, color in tag_color_map.items()})
    # --- Override with User-Specified Colors --- END

    # ────────────────── stream once → gather arrays
    token_tags_to_use = set(tag_color_map.keys()) # Get the set of token tags we assigned colors to

    coords   = []
    colours  = [ [] for _ in colour_exprs ]
    positions = []
    avg_tag_colors_list = [] # To store average tag colors per token
    rows_done = 0
    rng = np.random.default_rng(0)

    print("DEBUG: Starting stream processing over database chunks...")
    for chunk_idx, ch in enumerate(tqdm(db._mem, desc="stream")):
        print(f"  DEBUG: Processing chunk {chunk_idx}...")
        mask_hook = (ch["src"] == hi)
        if not mask_hook.any():
            print(f"    DEBUG: Skipping chunk {chunk_idx}: No activations for hook '{hook}'.")
            continue

        keys_np = ((ch["seq"].astype(np.int64) << 32) |
                   ch["seq_ofs"].astype(np.int64))[mask_hook]
        keys_t  = torch.from_numpy(keys_np).to(device)
        tags    = chunk_tag_masks(tag_keys, keys_t)

        loc = {"d": {k: v[mask_hook] for k, v in ch.items()},
               "tags": tags}

        keep = eval(filt_code, {"np": np, "torch": torch}, loc)
        if keep.dtype != bool:
            keep = keep.astype(bool)
        if not keep.any():
            continue

        X_all = ch["h"][mask_hook][keep].astype(np.float32)   # M × d

        # sub-sample if exceeding --max_tok
        if rows_done + len(X_all) > args.max_tok:
            wanted = args.max_tok - rows_done
            idx    = rng.choice(len(X_all), wanted, replace=False)
            X_all  = X_all[idx]
            subslice = idx
        else:
            subslice = slice(None)

        rows_done += len(X_all)
        coords.append(X_all @ dirs.T)                         # M × k_plot

        # --- Calculate Average Tag Color --- START
        if args.plot_tag_avg_color:
            batch_avg_colors = np.zeros((len(X_all), 3), dtype=np.float32) # Default to black
            # Get boolean masks for this chunk, subsetted by 'keep' and 'subslice'
            chunk_tags_bool = {
                tag: mask[keep][subslice]
                for tag, mask in tags.items() if tag in token_tags_to_use # Use only the relevant token tags
            }

            for i in range(len(X_all)): # Iterate through each token in the subsampled batch
                active_tags_for_token = [
                    tag_name for tag_name, mask in chunk_tags_bool.items() if mask[i]
                ]
                if active_tags_for_token:
                    colors_for_token = [tag_color_map[tag] for tag in active_tags_for_token]
                    # Calculate the mean RGB values
                    batch_avg_colors[i] = np.mean(colors_for_token, axis=0)

            avg_tag_colors_list.append(batch_avg_colors)
        # --- Calculate Average Tag Color --- END

        # Collect positions corresponding to filtered activations
        pos_all = ch['pos'][mask_hook][keep]
        pos_all = pos_all[subslice]
        positions.append(pos_all)

        for (expr, code), bucket in zip(colour_exprs, colours):
            val = eval(code, {"np": np, "torch": torch}, loc)
            val = val[keep] if isinstance(val, np.ndarray) else np.full(len(loc["d"]["id"][keep]), val)
            val = val[subslice]
            bucket.append(val.astype(np.float32))

        print(f"    DEBUG: Finished processing chunk {chunk_idx}. Total rows processed so far: {rows_done}")
        if rows_done >= args.max_tok:
            print(f"  DEBUG: Reached max_tok ({args.max_tok}). Stopping stream processing.")
            break
        gc.collect()

    if rows_done == 0:
        print("DEBUG: No rows collected after stream processing.")
        raise RuntimeError("Filter removed all rows; nothing to plot.")
    print(f"DEBUG: Finished stream processing. Total rows collected: {rows_done}")

    coords    = np.concatenate(coords)            # N × k_plot
    positions = np.concatenate(positions)         # N
    colour_vs = [ np.concatenate(v) for v in colours ]
    if args.plot_tag_avg_color:
        avg_tag_colors = np.concatenate(avg_tag_colors_list) # N x 3
    else:
        avg_tag_colors = None # Ensure it exists but is None if not used

    # === DEBUG: Check coords array ===
    print(f"DEBUG: coords shape: {coords.shape}")
    if coords.shape[1] > 1:
        print(f"DEBUG: coords std dev per column: {np.std(coords, axis=0)}")
    # === END DEBUG ===

    # ────────────────── plotting
    base_plot_dir = pathlib.Path(args.out_dir or (run_dir / "plots")).expanduser()
    plot_run_dir = _get_next_run_subdir(base_plot_dir) # Get the specific dir for this run
    print(f"Saving plots to: {plot_run_dir}")

    bins = args.bins
    for i in range(k_plot):
        for j in range(i+1, k_plot):
            x, y = coords[:, i], coords[:, j]
            # === DEBUG: Check x, y std dev inside loop ===
            # print(f"DEBUG: Plotting i={i}, j={j} | x_std={np.std(x):.4f}, y_std={np.std(y):.4f}") # Keep commented out
            # === END DEBUG ===

            H, xedges, yedges = np.histogram2d(x, y, bins=bins, density=False)
            # always compute bin-indices for **this** pair
            xi = np.digitize(x, xedges) - 1
            yi = np.digitize(y, yedges) - 1
            inside = (xi >= 0) & (xi < bins) & (yi >= 0) & (yi < bins)

            # --- Calculate square limits --- START
            x_range = xedges[-1] - xedges[0]
            y_range = yedges[-1] - yedges[0]
            max_range = max(x_range, y_range)
            x_center = (xedges[-1] + xedges[0]) / 2
            y_center = (yedges[-1] + yedges[0]) / 2
            square_xmin = x_center - max_range / 2
            square_xmax = x_center + max_range / 2
            square_ymin = y_center - max_range / 2
            square_ymax = y_center + max_range / 2
            square_extent = [square_xmin, square_xmax, square_ymin, square_ymax]
            # --- Calculate square limits --- END

            n_panels = 1 + len(colour_exprs) + args.plot_pos + args.plot_tag_avg_color # Add 1 if plotting tag avg color
            # Adjust figure height slightly more for bottom legend
            fig, axes = plt.subplots(1, n_panels, figsize=(5*n_panels, 6.0))
                                      # constrained_layout=True) # Disable constrained_layout
            axes = np.atleast_1d(axes) # Ensure axes is always an array

            # --- Prepare shared colormaps --- START
            cmap_viridis_bg = plt.get_cmap('viridis').copy()
            cmap_viridis_bg.set_bad('lightgray')

            cmap_plasma_bg = plt.get_cmap('plasma').copy()
            cmap_plasma_bg.set_bad(color='lightgray')
            # --- Prepare shared colormaps --- END

            ax_idx = 1  # first panel reserved for density

            # panel 0: density
            im0 = axes[0].imshow(
                np.log10(H.T + 1e-3), origin='lower',
                extent=square_extent, # Use square extent
                aspect='auto', # Aspect handled by limits now
                cmap=cmap_viridis_bg) # Use viridis with gray background
            axes[0].set_title(f"log₁₀ density · dir{i} vs dir{j}")
            fig.colorbar(im0, ax=axes[0])

            # -----------------------------------------------------------------
            # Figure-wide caption (suptitle) – shows the PC pair and filter used
            # -----------------------------------------------------------------
            caption = (
                f"PC{i} vs PC{j}   •   run: {run_dir.name}   "
                f"•   filter: {args.filter}"
            )
            fig.suptitle(caption, y=0.96, fontsize=14)

            # colour panels
            if colour_exprs:
                # xi, yi, inside already computed above
                # inside = (xi >= 0) & (xi < bins) & (yi >= 0) & (yi < bins)

                for (expr, _), vals in zip(colour_exprs, colour_vs):
                    ax = axes[ax_idx]
                    mean_color = np.zeros_like(H, dtype=np.float32)
                    cnt_color  = np.zeros_like(H, dtype=np.int32)

                    np.add.at(mean_color, (xi[inside], yi[inside]), vals[inside])
                    np.add.at(cnt_color,  (xi[inside], yi[inside]), 1)
                    mean_color[cnt_color == 0] = np.nan
                    mean_color = np.divide(mean_color, cnt_color, where=cnt_color>0)

                    im = ax.imshow(
                        mean_color.T, origin='lower',
                        extent=square_extent, # Use square extent
                        aspect='auto', # Aspect handled by limits now
                        cmap=cmap_viridis_bg) # Use viridis with gray background
                    ax.set_title(expr)
                    fig.colorbar(im, ax=ax)
                    ax_idx += 1 # Increment axis index

            # position panel
            if args.plot_pos:
                ax_pos = axes[ax_idx] # Use current axis index
                # indices already valid for this pair
                # if 'inside' not in locals():
                #      xi = np.digitize(x, xedges) - 1
                #      yi = np.digitize(y, yedges) - 1
                #      inside = (xi >= 0) & (xi < bins) & (yi >= 0) & (yi < bins)

                mean_pos = np.zeros_like(H, dtype=np.float32)
                cnt_pos  = np.zeros_like(H, dtype=np.int32)

                np.add.at(mean_pos, (xi[inside], yi[inside]), positions[inside])
                np.add.at(cnt_pos,  (xi[inside], yi[inside]), 1)
                mean_pos[cnt_pos == 0] = np.nan
                mean_pos = np.divide(mean_pos, cnt_pos, where=cnt_pos > 0)

                # Determine vmin/vmax for this specific plot, ignoring NaNs
                actual_vmin = np.nanmin(mean_pos)
                actual_vmax = np.nanmax(mean_pos)

                # Check if limits are valid before plotting position heatmap
                if np.isnan(actual_vmin) or np.isnan(actual_vmax):
                    print(f"      Skipping position plot for PC{i} vs PC{j}: No data points in bins.")
                    # Optionally hide the axis if it exists but won't be used
                    ax_pos.set_visible(False)
                else:
                    # Calculate shifted vmin for color mapping to lift minimum off the bottom
                    data_range = actual_vmax - actual_vmin
                    # Increase offset from 0.05 to 0.1 for stronger separation
                    plot_vmin = actual_vmin - data_range * 0.2 if data_range > 0 else actual_vmin

                    # --- Transform data and set up colorbar --- START
                    plot_target_min = 0.2
                    plot_target_max = 1.0

                    if data_range > 0:
                        scale = (plot_target_max - plot_target_min) / data_range
                        offset = plot_target_min - actual_vmin * scale
                        transformed_data = mean_pos * scale + offset
                    else: # Handle case of single value
                        scale = 1.0
                        offset = 0.0
                        transformed_data = np.full_like(mean_pos, plot_target_min) # Assign target min
                        transformed_data[np.isnan(mean_pos)] = np.nan # Keep NaNs

                    # Prepare the plasma colormap
                    cmap_plasma_bg = plt.get_cmap('plasma').copy()
                    cmap_plasma_bg.set_bad(color='lightgray')

                    # Plot transformed data
                    im_pos = ax_pos.imshow(
                        transformed_data.T, origin='lower', # Use transformed data
                        extent=square_extent,
                        aspect='auto',
                        cmap=cmap_plasma_bg,
                        vmin=plot_target_min, vmax=plot_target_max)
                    ax_pos.set_facecolor('lightgray')
                    ax_pos.set_title("Mean Token Position")

                    # Customize colorbar to show original values
                    cbar = fig.colorbar(im_pos, ax=ax_pos)
                    num_ticks = 5
                    original_ticks = np.linspace(actual_vmin, actual_vmax, num_ticks)
                    if data_range > 0:
                        transformed_ticks = original_ticks * scale + offset
                    else: # Handle single value case for ticks
                        transformed_ticks = np.linspace(plot_target_min, plot_target_min, num_ticks)

                    cbar.set_ticks(transformed_ticks)
                    cbar.set_ticklabels([f"{t:.1f}" for t in original_ticks]) # Format original values
                    # --- Transform data and set up colorbar --- END

                ax_idx += 1

            # --- Tag Average Color Panel --- START
            if args.plot_tag_avg_color:
                ax_tag = axes[ax_idx] # Use current axis index

                # indices already valid for this pair
                # if 'inside' not in locals():
                #      xi = np.digitize(x, xedges) - 1
                #      yi = np.digitize(y, yedges) - 1
                #      inside = (xi >= 0) & (xi < bins) & (yi >= 0) & (yi < bins)

                # sum_rgb = np.zeros((*H.shape, 3), dtype=np.float64) # Use float64 for summation - Incorrect, calc per channel
                count_rgb = np.zeros_like(H, dtype=np.int32)

                # Sum the RGB values for points inside valid bins
                r_sum = np.zeros_like(H, dtype=np.float64)
                g_sum = np.zeros_like(H, dtype=np.float64)
                b_sum = np.zeros_like(H, dtype=np.float64)

                np.add.at(r_sum, (xi[inside], yi[inside]), avg_tag_colors[inside, 0])
                np.add.at(g_sum, (xi[inside], yi[inside]), avg_tag_colors[inside, 1])
                np.add.at(b_sum, (xi[inside], yi[inside]), avg_tag_colors[inside, 2])
                np.add.at(count_rgb, (xi[inside], yi[inside]), 1)

                # Calculate average color, handle division by zero
                avg_rgb = np.full((*H.shape, 3), fill_value=[0.8, 0.8, 0.8], dtype=np.float32) # Default to light gray
                mask_nonzero = count_rgb > 0
                avg_rgb[mask_nonzero, 0] = r_sum[mask_nonzero] / count_rgb[mask_nonzero]
                avg_rgb[mask_nonzero, 1] = g_sum[mask_nonzero] / count_rgb[mask_nonzero]
                avg_rgb[mask_nonzero, 2] = b_sum[mask_nonzero] / count_rgb[mask_nonzero]

                # Display the image - TRANSPOSE IS CRUCIAL for imshow with origin='lower'
                im_tag = ax_tag.imshow(
                    np.transpose(avg_rgb, (1, 0, 2)), # Transpose (bins_x, bins_y, 3) -> (bins_y, bins_x, 3)
                    origin='lower',
                    extent=square_extent,
                    aspect='auto',
                    interpolation='nearest') # Use nearest to avoid blurring colors

                ax_tag.set_title("Avg Freq Token Color")
                ax_tag.set_facecolor('lightgray') # Set background for the axes area
                ax_idx += 1 # Increment axis index
            # --- Tag Average Color Panel --- END

            # --- Add Legend/Caption --- START
            if args.plot_tag_avg_color and tag_color_map:
                # Filter for token tags and sort them
                token_tags = {tag: color for tag, color in tag_color_map.items() if tag.startswith("token_")}
                sorted_tags = sorted(token_tags.items())

                patches = []
                for tag, color in sorted_tags:
                    # --- MODIFIED LEGEND LABEL LOGIC (ID BASED) --- START
                    label = f"<{tag}?>" # Default label in case of error
                    print(f"DEBUG: Processing tag for legend: '{tag}'") # ADDED DEBUG PRINT
                    try:
                        # --- MORE DEBUGGING --- START
                        print(f"  DEBUG: Type of tag variable: {type(tag)}")
                        print(f"  DEBUG: repr(tag): {repr(tag)}")
                        # --- Explicitly convert to standard Python string --- START
                        tag_as_py_str = str(tag)
                        print(f"  DEBUG: Type of str(tag): {type(tag_as_py_str)}")
                        print(f"  DEBUG: repr(str(tag)): {repr(tag_as_py_str)}")
                        # --- Explicitly convert to standard Python string --- END
                        tag_stripped = tag_as_py_str.strip() # Strip the standard string
                        print(f"  DEBUG: repr(str(tag).strip()): {repr(tag_stripped)}") # DEBUG STRIPPED
                        regex_pattern = r"token_(\\d+)"
                        print(f"  DEBUG: Regex pattern: '{regex_pattern}'")
                        match_result = re.match(regex_pattern, tag_stripped) # USE STRIPPED STANDARD STRING
                        print(f"  DEBUG: re.match(pattern, tag_stripped) result: {match_result}")
                        search_result = re.search(regex_pattern, tag_stripped) # TRY SEARCH on standard string
                        print(f"  DEBUG: re.search(pattern, tag_stripped) result: {search_result}") # DEBUG SEARCH
                        # --- MORE DEBUGGING --- END

                        # Use the match result if it worked
                        match = match_result
                        if match:
                            extracted_id_str = match.group(1)
                            print(f"  DEBUG: Extracted ID string: '{extracted_id_str}'")
                            token_id = int(extracted_id_str)
                            print(f"  DEBUG: Converted token ID (int): {token_id}")
                            # Decode the token ID to get the actual string representation
                            original_token_string = db.tok.decode([token_id])
                            print(f"  DEBUG: Decoded token string: {repr(original_token_string)}")

                            # Escape backslashes and single quotes *before* the f-string
                            escaped_token_string = original_token_string.replace('\\', '\\\\').replace("'", "\\'")
                            # Add quotes for clarity in the legend
                            label = f"'{escaped_token_string}' (ID:{token_id})"
                            patches.append(mpatches.Patch(color=color, label=label))
                        else:
                             print(f"Warning: Could not extract token ID from tag '{tag}'. Skipping in legend.")
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Error processing tag '{tag}' for legend: {e}. Skipping.")
                    except Exception as e:
                         # Catch potential errors during escaping/label creation
                         print(f"Warning: Unexpected error creating legend label for tag '{tag}': {e}")
                    # --- MODIFIED LEGEND LABEL LOGIC (ID BASED) --- END

                if patches:
                    # Let matplotlib arrange, but cap columns to avoid excessive width
                    num_cols = min(len(patches), 8) # Adjust max columns as needed
                    fig.legend(handles=patches, title="Frequent Token Tags (Decoded)",
                               loc='lower center', bbox_to_anchor=(0.5, 0.01), # Place below plots
                               ncol=num_cols, fontsize='small', frameon=False)

            # --- Add Legend/Caption --- END

            # give the extra space requested for the dropped-down legend/caption
            # Adjust bottom margin in tight_layout if legend overlaps x-axis labels
            # fig.subplots_adjust(bottom=0.27) # Might conflict with tight_layout

            # Common axis settings
            # ax_idx_offset = 1 + len(colour_exprs) # No longer needed due to dynamic ax_idx
            for idx, ax in enumerate(axes):
                if not ax.get_visible(): continue # Skip hidden axes
                ax.set_xlabel(f"dir {i}")
                ax.set_ylabel(f"dir {j}")
                ax.set_xlim(square_xmin, square_xmax) # Force square limits
                ax.set_ylim(square_ymin, square_ymax) # Force square limits
                # Ensure background fills square, skip if it's the tag color plot which handles its own bg
                # Check if this axis is the tag average color plot (handled separately)
                is_tag_avg_plot = args.plot_tag_avg_color and ax_idx > (1 + len(colour_exprs)) and idx == (1 + len(colour_exprs))
                if not is_tag_avg_plot:
                     ax.set_facecolor('lightgray')

            # Use descriptive filename and save in the run-specific directory
            fn = plot_run_dir / f"PC{i}_vs_PC{j}.png"
            # keep 12 % at the top for the caption and 14 % at the bottom for legend
            # Adjust bottom percentage if legend needs more space
            plt.tight_layout(rect=[0, 0.14, 1, 0.88]) # y_min=0.14 reserves 14% bottom margin
            fig.savefig(fn, dpi=150)
            plt.close(fig)
            print("✓ saved", fn.relative_to(base_plot_dir)) # Print relative path for clarity

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
