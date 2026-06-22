from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
from floodcastbench_utils import build_manifest_rows, ensure_data_dir, ensure_output_dir, load_csv, read_raster_metadata, sample_group, write_csv

FIELDS = ["path", "event", "resolution", "variable_guess", "shape_height", "shape_width", "count_bands", "dtype", "crs", "transform", "pixel_size_x", "pixel_size_y", "nodata_value", "min", "max", "mean", "std", "nan_count", "negative_count", "zero_count", "positive_count"]

def parse_args():
    p = argparse.ArgumentParser(description="Profile sampled FloodCastBench rasters.")
    p.add_argument("--data_dir", type=Path, default=Path("data/FloodCastBench"))
    p.add_argument("--output_dir", type=Path, default=Path("outputs/dataset_profile"))
    p.add_argument("--max_files_per_group", type=int, default=5)
    return p.parse_args()

def main():
    args = parse_args(); data_dir = ensure_data_dir(args.data_dir); output_dir = ensure_output_dir(args.output_dir)
    manifest = output_dir / "floodcastbench_manifest.csv"
    rows = load_csv(manifest) if manifest.exists() else build_manifest_rows(data_dir)
    groups = defaultdict(list)
    for r in rows:
        if r["is_raster"] in {True, "True", "true"}:
            groups[(r["event"], r["resolution"], r["variable_guess"])].append(r)
    out = []
    for _, group in sorted(groups.items()):
        for row in sample_group(group, args.max_files_per_group):
            try:
                meta = read_raster_metadata(Path(row["file_path"]))
            except Exception as e:
                print(f"WARNING: failed to profile {row['relative_path']}: {e}")
                continue
            out.append({"path": row["relative_path"], "event": row["event"], "resolution": row["resolution"], "variable_guess": row["variable_guess"], **meta})
    write_csv(output_dir / "raster_statistics.csv", out, FIELDS)
    print(f"Profiled rasters: {len(out)}")

if __name__ == "__main__":
    main()
