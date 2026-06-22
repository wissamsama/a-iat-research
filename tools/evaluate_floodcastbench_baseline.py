from __future__ import annotations

import argparse
import csv
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

from baselines.floodcastbench import available_baselines, predict_floodcastbench_baseline, prediction_rule_for_baseline
from datasets import FloodCastBenchWaterDepthDataset
from metrics import BinaryMetricAccumulator, WaterDepthMetricAccumulator

MASK_THRESHOLDS = (0.001, 0.01)


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


def build_dataset(config: dict, split: str | None, horizon: int | None) -> FloodCastBenchWaterDepthDataset:
    dataset_config = config["dataset"]
    return FloodCastBenchWaterDepthDataset(
        root=PROJECT_DIR / dataset_config.get("root", "data/FloodCastBench"),
        event=dataset_config.get("event", "Australia flood"),
        fidelity=dataset_config.get("fidelity", "high"),
        resolution=dataset_config.get("resolution", "30m"),
        input_window=int(dataset_config.get("input_window", 5)),
        horizon=int(horizon if horizon is not None else dataset_config.get("horizon", 20)),
        split=split or dataset_config.get("split", "train"),
        split_ratios=split_ratios_from_config(dataset_config),
        normalization=dataset_config.get("normalization", "none"),
    )


def format_float(value: float) -> str:
    if value is None or math.isnan(value):
        return "nan"
    return f"{value:.6g}"


def print_metric_block(title: str, metrics: dict[str, float], keys: list[str]) -> None:
    print(f"\n{title}:")
    for key in keys:
        print(f"{key}: {format_float(metrics[key])}")


def evaluate(
    dataset: FloodCastBenchWaterDepthDataset,
    baseline: str = "persistence",
    max_samples: int | None = None,
    progress_every: int = 50,
) -> dict:
    baseline = baseline.lower()
    if baseline not in available_baselines():
        available = ", ".join(available_baselines())
        raise ValueError(f"Unsupported baseline '{baseline}'. Available baselines: {available}")

    water_acc = WaterDepthMetricAccumulator()
    mask_accs = {gamma: BinaryMetricAccumulator() for gamma in MASK_THRESHOLDS}
    path_accs = {gamma: BinaryMetricAccumulator() for gamma in MASK_THRESHOLDS}

    sample_count = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    start_time = time.perf_counter()

    with torch.no_grad():
        for index in range(sample_count):
            x, y, _ = dataset[index]
            current = x[-1]
            prediction = predict_floodcastbench_baseline(x, horizon=dataset.horizon, baseline=baseline)
            target = y

            water_acc.update(prediction, target)
            for gamma in MASK_THRESHOLDS:
                current_mask = current > gamma
                pred_future_mask = prediction > gamma
                target_future_mask = target > gamma
                mask_accs[gamma].update(pred_future_mask, target_future_mask)

                pred_path = pred_future_mask & (~current_mask)
                target_path = target_future_mask & (~current_mask)
                path_accs[gamma].update(pred_path, target_path)

            if progress_every and (index + 1) % progress_every == 0:
                print(f"progress: {index + 1}/{sample_count}")

    elapsed = time.perf_counter() - start_time
    return {
        "metadata": {
            "baseline": baseline,
            "dataset": "FloodCastBench",
            "event": dataset.event,
            "fidelity": dataset.fidelity,
            "resolution": dataset.resolution,
            "split": dataset.split,
            "input_window": dataset.input_window,
            "horizon": dataset.horizon,
            "num_samples": sample_count,
            "num_available_samples": len(dataset),
            "prediction_rule": prediction_rule_for_baseline(baseline),
            "elapsed_seconds": elapsed,
        },
        "water_depth": water_acc.compute(),
        "mask": {str(gamma): acc.compute() for gamma, acc in mask_accs.items()},
        "propagation_path": {str(gamma): acc.compute() for gamma, acc in path_accs.items()},
    }


def print_summary(results: dict) -> None:
    metadata = results["metadata"]
    print("Dataset: FloodCastBench")
    print(f"Baseline: {metadata['baseline']}")
    print(f"Event: {metadata['event']}")
    print(f"Resolution: {metadata['resolution']}")
    print(f"Split: {metadata['split']}")
    print(f"Input window: {metadata['input_window']}")
    print(f"Horizon: {metadata['horizon']}")
    print(f"Number of samples: {metadata['num_samples']}")
    print(f"Prediction rule: {metadata['prediction_rule']}")
    print_metric_block("Water-depth metrics", results["water_depth"], ["mae", "mse", "rmse", "nse", "pearson_r"])
    for gamma in MASK_THRESHOLDS:
        print_metric_block(
            f"Mask metrics gamma={gamma}",
            results["mask"][str(gamma)],
            ["csi", "iou", "f1", "precision", "recall"],
        )
    for gamma in MASK_THRESHOLDS:
        print_metric_block(
            f"Propagation-path metrics gamma={gamma}",
            results["propagation_path"][str(gamma)],
            ["iou", "f1", "precision", "recall"],
        )
    print(f"\nElapsed seconds: {format_float(metadata['elapsed_seconds'])}")


def save_results(results: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = results["metadata"]
    event_slug = metadata["event"].lower().replace(" flood", "").replace(" ", "_")
    base_name = f"{metadata['baseline']}_{event_slug}_{metadata['resolution']}_h{metadata['horizon']}_{metadata['split']}"
    json_path = output_dir / f"{base_name}.json"
    csv_path = output_dir / f"{base_name}.csv"

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    rows = []
    for key, value in metadata.items():
        rows.append({"section": "metadata", "metric": key, "value": value})
    for metric, value in results["water_depth"].items():
        rows.append({"section": "water_depth", "metric": metric, "value": value})
    for gamma, metrics in results["mask"].items():
        for metric, value in metrics.items():
            rows.append({"section": f"mask_gamma_{gamma}", "metric": metric, "value": value})
    for gamma, metrics in results["propagation_path"].items():
        for metric, value in metrics.items():
            rows.append({"section": f"propagation_path_gamma_{gamma}", "metric": metric, "value": value})

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["section", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deterministic FloodCastBench forecasting baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs" / "floodcastbench_water_depth.yaml",
        help="Path to FloodCastBench water-depth config.",
    )
    parser.add_argument("--baseline", choices=available_baselines(), default="persistence")
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional debug limit; omitted means full split.")
    parser.add_argument("--progress-every", type=int, default=50, help="Print progress every N samples; use 0 to disable.")
    parser.add_argument("--save", action="store_true", help="Save scalar metrics under outputs/floodcastbench_baselines/.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / "floodcastbench_baselines",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_DIR / args.config
    config = load_config(config_path)

    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    dataset = build_dataset(config, split=args.split, horizon=args.horizon)
    root_mtime_before = dataset.root.stat().st_mtime
    results = evaluate(
        dataset,
        baseline=args.baseline,
        max_samples=args.max_samples,
        progress_every=args.progress_every,
    )
    root_mtime_after = dataset.root.stat().st_mtime
    results["metadata"]["raw_root_mtime_unchanged"] = root_mtime_before == root_mtime_after

    print_summary(results)
    print(f"Raw root mtime unchanged: {results['metadata']['raw_root_mtime_unchanged']}")

    if args.save:
        json_path, csv_path = save_results(results, args.output_dir)
        print(f"Saved JSON: {json_path}")
        print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
