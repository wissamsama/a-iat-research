from __future__ import annotations

"""Train-only statistics of consecutive-frame water-depth differences.

DIFF-SPARSE V2's delta prediction mode diffuses x0 = (next_frame - base) /
delta_scale instead of the absolute field. At 300 s frame spacing consecutive
frames are nearly identical (oracle persistence RMSE ~0.004 normalized), so
the absolute target wastes model capacity re-encoding the last frame; the
frame-to-frame change is the actual information. This tool computes the std
of those changes over the train frame range (physical units and relative to
the water std) so the delta target can be normalized to ~unit variance.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from datasets.floodcastbench_diff_sparse_v1_dataset import (  # noqa: E402
    _load_frames,
    _read_raster,
    split_frame_ranges,
)
from tools.train_floodcastbench_diff_sparse_v1 import load_config, path_from_config  # noqa: E402


def compute_delta_stats(config: dict) -> dict:
    dataset_config = config.get("dataset", {})
    root = path_from_config(config, "dataset_root")
    fidelity = str(dataset_config.get("fidelity", "high")).lower()
    resolution = str(dataset_config.get("resolution", "60m")).lower()
    event = str(dataset_config.get("event", "australia")).capitalize()
    family = "High-fidelity flood forecasting" if fidelity == "high" else "Low-fidelity flood forecasting"
    water_dir = root / family / resolution / event
    frames = _load_frames(water_dir)
    ranges = split_frame_ranges(len(frames), dataset_config.get("split_counts"))
    train_start, train_end = ranges["train"]

    total = 0.0
    total_sq = 0.0
    count = 0.0
    abs_max = 0.0
    previous = _read_raster(frames[train_start].path).astype(np.float64)
    for index in range(train_start + 1, train_end):
        current = _read_raster(frames[index].path).astype(np.float64)
        delta = current - previous
        total += float(delta.sum())
        total_sq += float((delta * delta).sum())
        count += float(delta.size)
        abs_max = max(abs_max, float(np.abs(delta).max()))
        previous = current

    mean = total / count
    variance = max(total_sq / count - mean * mean, 0.0)
    std = math.sqrt(variance)
    return {
        "version": "diff_sparse_v2_train_delta_stats",
        "train_frame_range": [train_start, train_end],
        "num_deltas": int(train_end - train_start - 1),
        "delta_mean_physical": mean,
        "delta_std_physical": std,
        "delta_abs_max_physical": abs_max,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stats = compute_delta_stats(load_config(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
