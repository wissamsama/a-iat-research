from __future__ import annotations
import argparse
from pathlib import Path
from floodcastbench_utils import write_questions_report, build_manifest_rows, ensure_data_dir, ensure_output_dir, load_csv, supervised_preview_rows, temporal_summary_rows, write_csv

def parse_args():
    p = argparse.ArgumentParser(description="Check FloodCastBench temporal windows.")
    p.add_argument("--data_dir", type=Path, default=Path("data/FloodCastBench"))
    p.add_argument("--output_dir", type=Path, default=Path("outputs/dataset_profile"))
    p.add_argument("--max_files_per_group", type=int, default=5)
    p.add_argument("--input_window", type=int, default=5)
    p.add_argument("--horizons", type=int, nargs="+", default=[20, 72, 144])
    return p.parse_args()

def main():
    args = parse_args(); data_dir = ensure_data_dir(args.data_dir); output_dir = ensure_output_dir(args.output_dir)
    manifest = output_dir / "floodcastbench_manifest.csv"
    rows = load_csv(manifest) if manifest.exists() else build_manifest_rows(data_dir)
    temporal = temporal_summary_rows(rows)
    preview = supervised_preview_rows(rows, args.input_window, args.horizons)
    write_csv(output_dir / "temporal_index_summary.csv", temporal)
    write_csv(output_dir / "supervised_learning_samples_preview.csv", preview)
    write_questions_report(output_dir)
    print(f"Temporal groups: {len(temporal)}")
    print(f"Supervised preview rows: {len(preview)}")

if __name__ == "__main__":
    main()
