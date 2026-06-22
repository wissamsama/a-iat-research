from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
from floodcastbench_utils import build_manifest_rows, ensure_data_dir, ensure_output_dir, write_csv

FIELDS = ["file_path", "relative_path", "file_name", "extension", "top_level_folder", "event", "resolution", "variable_guess", "time_index_guess", "is_raster", "file_size_mb"]

def parse_args():
    p = argparse.ArgumentParser(description="Build FloodCastBench full file manifest.")
    p.add_argument("--data_dir", type=Path, default=Path("data/FloodCastBench"))
    p.add_argument("--output_dir", type=Path, default=Path("outputs/dataset_profile"))
    p.add_argument("--max_files_per_group", type=int, default=5)
    return p.parse_args()

def main():
    args = parse_args(); data_dir = ensure_data_dir(args.data_dir); output_dir = ensure_output_dir(args.output_dir)
    rows = build_manifest_rows(data_dir)
    write_csv(output_dir / "floodcastbench_manifest.csv", rows, FIELDS)
    print(f"Files: {len(rows)}")
    print("Extensions:", dict(sorted(Counter(r["extension"] for r in rows).items())))
    print(f"Wrote {output_dir / 'floodcastbench_manifest.csv'}")

if __name__ == "__main__":
    main()
