from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from floodcastbench_utils import write_questions_report, build_manifest_rows, ensure_data_dir, ensure_output_dir, load_csv, natural_time_key


def parse_args():
    p = argparse.ArgumentParser(description="Create FloodCastBench sanity-check figures.")
    p.add_argument("--data_dir", type=Path, default=Path("data/FloodCastBench"))
    p.add_argument("--output_dir", type=Path, default=Path("outputs/dataset_profile"))
    p.add_argument("--max_files_per_group", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--input_window", type=int, default=5)
    p.add_argument("--horizons", type=int, nargs="+", default=[20, 72, 144])
    return p.parse_args()


def read_band(path):
    import rasterio
    with rasterio.open(path) as src:
        return src.read(1, masked=True).astype(np.float64).filled(np.nan)


def stretch(arr):
    valid = np.isfinite(arr)
    if not valid.any(): return np.zeros_like(arr, dtype=float)
    lo, hi = np.nanpercentile(arr, [2, 98])
    if hi <= lo: hi = lo + 1.0
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def save_single(arr, path, title, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(6, 5)); im = ax.imshow(stretch(arr), cmap=cmap)
    ax.set_title(title); ax.axis("off"); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def save_grid(items, path, title, cmap="viridis"):
    fig, axes = plt.subplots(1, len(items), figsize=(4 * len(items), 4)); axes = np.atleast_1d(axes)
    for ax, (arr, subtitle) in zip(axes, items):
        ax.imshow(stretch(arr), cmap=cmap); ax.set_title(subtitle); ax.axis("off")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def temporal_rows(rows, event, resolution, variable):
    selected = [r for r in rows if r["event"] == event and r["resolution"] == resolution and r["variable_guess"] == variable and r["is_raster"] in {True, "True", "true"}]
    return sorted(selected, key=lambda r: natural_time_key(r["time_index_guess"]))


def first_row(rows, variable):
    return next((r for r in rows if r["variable_guess"] == variable and r["is_raster"] in {True, "True", "true"}), None)


def main():
    args = parse_args(); data_dir = ensure_data_dir(args.data_dir); output_dir = ensure_output_dir(args.output_dir); figures = output_dir / "figures"
    manifest = output_dir / "floodcastbench_manifest.csv"
    rows = load_csv(manifest) if manifest.exists() else build_manifest_rows(data_dir)
    groups = []
    for r in rows:
        key = (r["event"], r["resolution"])
        if r["variable_guess"] == "water_depth_forecast" and key not in groups:
            groups.append(key)
    for event, resolution in groups:
        seq = temporal_rows(rows, event, resolution, "water_depth_forecast")
        if not seq: continue
        arr = read_band(Path(seq[0]["file_path"]))
        save_single(arr, figures / f"water_depth_{event.replace(' ', '_')}_{resolution}.png", f"Water depth: {event} {resolution}")
    if groups:
        event, resolution = groups[0]; seq = temporal_rows(rows, event, resolution, "water_depth_forecast")
        items = []
        for pos in [0, 1, 5, 20]:
            if pos < len(seq): items.append((read_band(Path(seq[pos]["file_path"])), f"t+{pos}\n{seq[pos]['time_index_guess']}"))
        if items: save_grid(items, figures / "temporal_sequence_preview.png", f"Temporal sequence: {event} {resolution}")
        current = read_band(Path(seq[0]["file_path"])); future = read_band(Path(seq[min(20, len(seq)-1)]["file_path"]))
        cmask = current > args.threshold; fmask = future > args.threshold; newpix = fmask & ~cmask
        save_single(cmask.astype(float), figures / "derived_flood_mask_threshold_0p01.png", f"Flood mask threshold={args.threshold}", cmap="Blues")
        save_grid([(cmask.astype(float), "current mask"), (fmask.astype(float), "future mask"), (newpix.astype(float), "newly flooded")], figures / "propagation_path_preview.png", f"Propagation path threshold={args.threshold}", cmap="Blues")
    for variable, name, cmap in [("dem", "dem_preview.png", "terrain"), ("rainfall", "rainfall_preview.png", "Blues"), ("lulc", "lulc_preview.png", "tab20")]:
        row = first_row(rows, variable)
        if row: save_single(read_band(Path(row["file_path"])), figures / name, f"{variable}: {row['event']}", cmap=cmap)
        else: print(f"WARNING: no raster found for {variable}")
    write_questions_report(output_dir)
    print(f"Wrote figures to {figures}")

if __name__ == "__main__":
    main()
