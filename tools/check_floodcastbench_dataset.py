from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from datasets import FloodCastBenchWaterDepthDataset


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def split_ratios_from_config(config: dict) -> tuple[float, float, float]:
    ratios = config.get("split_ratios", {})
    return (
        float(ratios.get("train", 0.70)),
        float(ratios.get("val", 0.15)),
        float(ratios.get("test", 0.15)),
    )


def build_dataset(config: dict) -> FloodCastBenchWaterDepthDataset:
    dataset_config = config["dataset"]
    return FloodCastBenchWaterDepthDataset(
        root=PROJECT_DIR / dataset_config.get("root", "data/FloodCastBench"),
        event=dataset_config.get("event", "Australia flood"),
        fidelity=dataset_config.get("fidelity", "high"),
        resolution=dataset_config.get("resolution", "30m"),
        input_window=int(dataset_config.get("input_window", 5)),
        horizon=int(dataset_config.get("horizon", 20)),
        split=dataset_config.get("split", "train"),
        split_ratios=split_ratios_from_config(dataset_config),
        normalization=dataset_config.get("normalization", "none"),
    )


def tensor_stats(name: str, tensor: torch.Tensor) -> None:
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        print(f"{name}: shape={tuple(tensor.shape)} min=nan max=nan mean=nan")
        return
    print(
        f"{name}: shape={tuple(tensor.shape)} "
        f"min={finite.min().item():.6g} "
        f"max={finite.max().item():.6g} "
        f"mean={finite.mean().item():.6g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight FloodCastBench Dataset sanity check.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs" / "floodcastbench_water_depth.yaml",
        help="Path to FloodCastBench water-depth dataset config.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_DIR / args.config
    config = load_config(config_path)
    dataset = build_dataset(config)

    root_mtime_before = dataset.root.stat().st_mtime
    x, y, meta = dataset[args.sample_index]
    root_mtime_after = dataset.root.stat().st_mtime

    print("FloodCastBench water-depth Dataset sanity check")
    print(f"config: {config_path}")
    print(f"root: {dataset.root}")
    print(f"source_dir: {dataset.source_dir}")
    print(f"split: {dataset.split}")
    print(f"samples: {len(dataset)}")
    print(f"frames: {len(dataset.frames)}")
    print(f"frame_shape: {dataset.height}x{dataset.width}")
    tensor_stats("x", x)
    tensor_stats("y", y)
    print(f"input_timestamps: {meta['input_timestamps']}")
    print(f"target_timestamp: {meta['target_timestamp']}")
    print(f"first_input_path: {meta['input_paths'][0]}")
    print(f"target_path: {meta['target_path']}")
    print(f"raw_root_mtime_unchanged: {root_mtime_before == root_mtime_after}")


if __name__ == "__main__":
    main()
