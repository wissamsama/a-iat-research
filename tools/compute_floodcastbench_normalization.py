from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import torch
import yaml
from rasterio.errors import NotGeoreferencedWarning

PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from datasets import FloodCastBenchWaterDepthDataset


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def split_ratios_from_config(dataset_config: dict) -> tuple[float, float, float]:
    ratios = dataset_config.get("split_ratios", {})
    return (
        float(ratios.get("train", 0.70)),
        float(ratios.get("val", 0.15)),
        float(ratios.get("test", 0.15)),
    )


def build_dataset(config: dict, args) -> FloodCastBenchWaterDepthDataset:
    dataset_config = config["dataset"]
    return FloodCastBenchWaterDepthDataset(
        root=PROJECT_DIR / dataset_config.get("root", "data/FloodCastBench"),
        event=args.event or dataset_config.get("event", "Australia flood"),
        fidelity=args.fidelity or dataset_config.get("fidelity", "high"),
        resolution=args.resolution or dataset_config.get("resolution", "30m"),
        input_window=int(args.input_window or dataset_config.get("input_window", 5)),
        horizon=int(args.horizon or dataset_config.get("horizon", 20)),
        split=args.split,
        split_ratios=split_ratios_from_config(dataset_config),
        normalization={"mode": "none"},
    )


def update_stream(values: torch.Tensor, state: dict) -> None:
    finite = values.detach().float().reshape(-1)
    finite = finite[torch.isfinite(finite)]
    if finite.numel() == 0:
        return
    state["count"] += int(finite.numel())
    state["sum"] += float(finite.sum().item())
    state["sum_sq"] += float((finite * finite).sum().item())
    state["min"] = min(state["min"], float(finite.min().item()))
    state["max"] = max(state["max"], float(finite.max().item()))


def compute_stats(dataset, max_samples: int | None, percentile_sample_limit: int) -> dict:
    state = {"count": 0, "sum": 0.0, "sum_sq": 0.0, "min": math.inf, "max": -math.inf}
    samples = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    percentile_values = []
    percentile_count = 0

    for index in range(samples):
        x, y, _ = dataset[index]
        update_stream(x, state)
        update_stream(y, state)
        if percentile_count < percentile_sample_limit:
            combined = torch.cat([x.reshape(-1), y.reshape(-1)]).float()
            combined = combined[torch.isfinite(combined)]
            remaining = percentile_sample_limit - percentile_count
            if combined.numel() > remaining:
                combined = combined[:remaining]
            percentile_values.append(combined.cpu())
            percentile_count += int(combined.numel())

    if state["count"] == 0:
        raise RuntimeError("No finite water-depth values found for normalization statistics.")
    mean = state["sum"] / state["count"]
    variance = max((state["sum_sq"] / state["count"]) - (mean * mean), 0.0)
    std = math.sqrt(variance)

    p01 = None
    p99 = None
    if percentile_values:
        values = torch.cat(percentile_values)
        p01 = float(torch.quantile(values, 0.01).item())
        p99 = float(torch.quantile(values, 0.99).item())

    return {
        "num_samples_used": samples,
        "num_pixels_used": state["count"],
        "mean": mean,
        "std": std,
        "min": state["min"],
        "max": state["max"],
        "p01": p01,
        "p99": p99,
        "percentiles_approximate": True,
        "percentile_pixels_used": percentile_count,
    }


def output_name(dataset) -> str:
    event = dataset.event.lower().replace(" flood", "").replace(" ", "_")
    return f"water_depth_{event}_{dataset.resolution}_{dataset.split}_input{dataset.input_window}_h{dataset.horizon}_stats.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute train-only FloodCastBench water-depth normalization statistics.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "floodcastbench_cnn_baseline.yaml")
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--event", default=None)
    parser.add_argument("--fidelity", default=None)
    parser.add_argument("--resolution", default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--input-window", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--percentile-sample-limit", type=int, default=1000000)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / "floodcastbench_normalization")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    config_path = args.config if args.config.is_absolute() else PROJECT_DIR / args.config
    config = load_config(config_path)
    dataset = build_dataset(config, args)
    raw_mtime_before = dataset.root.stat().st_mtime
    start = time.perf_counter()
    stats = compute_stats(dataset, args.max_samples, args.percentile_sample_limit)
    raw_mtime_after = dataset.root.stat().st_mtime

    result = {
        "dataset": "FloodCastBench",
        "event": dataset.event,
        "fidelity": dataset.fidelity,
        "resolution": dataset.resolution,
        "split": dataset.split,
        "input_window": dataset.input_window,
        "horizon": dataset.horizon,
        "num_available_samples": len(dataset),
        "computed_from_train_only": dataset.split == "train",
        "max_samples": args.max_samples,
        "is_smoke_stats": args.max_samples is not None,
        "raw_root_mtime_unchanged": raw_mtime_before == raw_mtime_after,
        "elapsed_seconds": time.perf_counter() - start,
        **stats,
    }

    print(json.dumps(result, indent=2))
    if args.save:
        output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_DIR / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / output_name(dataset)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
