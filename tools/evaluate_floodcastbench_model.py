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
from torch.utils.data import DataLoader
from rasterio.errors import NotGeoreferencedWarning

PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from baselines.floodcastbench import predict_floodcastbench_baseline
from datasets import FloodCastBenchWaterDepthDataset
from metrics.floodcastbench_eval import ForecastMetricBundle, RawClampedMetricBundle, flatten_metrics
from models.flood_cnn import FloodCNNBaseline
from models.flood_latent_temporal import FloodLatentTemporalModel

COLUMNS = [
    "method",
    "split",
    "horizon",
    "mae",
    "rmse",
    "mse",
    "nse",
    "csi_gamma_0_001",
    "csi_gamma_0_01",
    "path_iou_gamma_0_001",
    "path_iou_gamma_0_01",
    "negative_prediction_ratio",
]


def save_json(data: dict, path: Path) -> None:
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    path.write_text(json.dumps(clean(data), indent=2), encoding="utf-8")


def build_dataset_from_checkpoint(checkpoint: dict, split: str, horizon: int | None) -> FloodCastBenchWaterDepthDataset:
    config = checkpoint.get("config", {})
    dataset_config = config.get("dataset", {})
    split_ratios = dataset_config.get("split_ratios", {})
    return FloodCastBenchWaterDepthDataset(
        root=PROJECT_DIR / dataset_config.get("root", "data/FloodCastBench"),
        event=dataset_config.get("event", "Australia flood"),
        fidelity=dataset_config.get("fidelity", "high"),
        resolution=dataset_config.get("resolution", "30m"),
        input_window=int(dataset_config.get("input_window", 5)),
        horizon=int(horizon if horizon is not None else dataset_config.get("horizon", 20)),
        split=split,
        split_ratios=(
            float(split_ratios.get("train", 0.70)),
            float(split_ratios.get("val", 0.15)),
            float(split_ratios.get("test", 0.15)),
        ),
        normalization=dataset_config.get("normalization", {"mode": "none"}),
    )


def model_name_from_checkpoint(checkpoint: dict) -> str:
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})
    return str(checkpoint.get("model_name") or model_config.get("name", "flood_cnn_baseline"))


def build_model_from_checkpoint(checkpoint: dict, dataset):
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})
    model_name = model_name_from_checkpoint(checkpoint).lower()
    if model_name == "flood_cnn_baseline":
        model = FloodCNNBaseline(
            input_window=int(model_config.get("input_window", dataset.input_window)),
            base_channels=int(model_config.get("base_channels", 16)),
            output_activation=model_config.get("output_activation", "identity"),
            final_bias_init=model_config.get("final_bias_init"),
        )
    elif model_name == "flood_latent_temporal":
        model = FloodLatentTemporalModel(
            input_window=int(model_config.get("input_window", dataset.input_window)),
            base_channels=int(model_config.get("base_channels", 16)),
            latent_channels=int(model_config.get("latent_channels", 64)),
            temporal_module=model_config.get("temporal_module", "temporal_conv"),
            residual_prediction=bool(model_config.get("residual_prediction", True)),
            output_activation=model_config.get("output_activation", "identity"),
            final_bias_init=model_config.get("final_bias_init"),
        )
    else:
        raise ValueError("checkpoint model_name must be one of: flood_cnn_baseline, flood_latent_temporal")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def maybe_resize(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[-2:] != target.shape[-2:]:
        return torch.nn.functional.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
    return pred


def evaluate(checkpoint_path: Path, split: str, horizon: int | None, max_batches: int | None, compare_baselines: bool, device: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    dataset = build_dataset_from_checkpoint(checkpoint, split, horizon)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    torch_device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model_name = model_name_from_checkpoint(checkpoint)
    model = build_model_from_checkpoint(checkpoint, dataset).to(torch_device)
    gammas = tuple(float(value) for value in checkpoint.get("config", {}).get("evaluation", {}).get("gammas", [0.001, 0.01]))

    model_bundle = RawClampedMetricBundle(gammas)
    baseline_bundles = {name: ForecastMetricBundle(gammas) for name in ("persistence", "linear_delta")}
    raw_mtime_before = dataset.root.stat().st_mtime
    start = time.perf_counter()
    batches = 0

    with torch.no_grad():
        for batch_index, (x, y, _) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            x = x.to(torch_device)
            y = y.to(torch_device)
            pred = maybe_resize(model(x), y)
            pred_physical = dataset.denormalize_water_depth(pred.detach())
            y_physical = dataset.denormalize_water_depth(y.detach())
            current_physical = dataset.denormalize_water_depth(x[:, -1].detach())
            model_bundle.update(pred_physical, y_physical, current_physical)

            if compare_baselines:
                x_physical = dataset.denormalize_water_depth(x.detach()).cpu()
                target = y_physical.cpu()
                current = x_physical[:, -1]
                for baseline, bundle in baseline_bundles.items():
                    baseline_preds = []
                    for sample_index in range(x_physical.shape[0]):
                        baseline_preds.append(predict_floodcastbench_baseline(x_physical[sample_index], dataset.horizon, baseline))
                    baseline_pred = torch.stack(baseline_preds, dim=0)
                    bundle.update(baseline_pred, target.cpu(), current.cpu())
            batches += 1

    raw_mtime_after = dataset.root.stat().st_mtime
    rows = []
    model_metrics = model_bundle.compute()
    rows.append(row_from_metrics(f"{model_name}_raw", model_metrics, "raw", dataset))
    rows.append(row_from_metrics(f"{model_name}_clamped", model_metrics, "clamped", dataset))
    if compare_baselines:
        for baseline, bundle in baseline_bundles.items():
            rows.append(row_from_plain_metrics(baseline, bundle.compute(), dataset))

    return {
        "metadata": {
            "checkpoint": str(checkpoint_path),
            "split": dataset.split,
            "horizon": dataset.horizon,
            "input_window": dataset.input_window,
            "event": dataset.event,
            "resolution": dataset.resolution,
            "model": model_name,
            "normalization_mode": dataset.normalization_mode,
            "normalization_stats_path": dataset.normalization_stats_path,
            "metrics_are_denormalized": True,
            "num_batches": batches,
            "num_available_samples": len(dataset),
            "raw_root_mtime_unchanged": raw_mtime_before == raw_mtime_after,
            "elapsed_seconds": time.perf_counter() - start,
        },
        "rows": rows,
    }


def row_from_metrics(method: str, metrics: dict, variant: str, dataset) -> dict:
    flat = flatten_metrics(metrics, variant)
    return {
        "method": method,
        "split": dataset.split,
        "horizon": dataset.horizon,
        "mae": flat.get("mae"),
        "rmse": flat.get("rmse"),
        "mse": flat.get("mse"),
        "nse": flat.get("nse"),
        "csi_gamma_0_001": flat.get("csi_gamma_0_001"),
        "csi_gamma_0_01": flat.get("csi_gamma_0_01"),
        "path_iou_gamma_0_001": flat.get("path_iou_gamma_0_001"),
        "path_iou_gamma_0_01": flat.get("path_iou_gamma_0_01"),
        "negative_prediction_ratio": metrics.get("negative_prediction_ratio"),
        "pred_min": metrics.get("pred_min"),
        "pred_max": metrics.get("pred_max"),
        "pred_mean": metrics.get("pred_mean"),
    }


def row_from_plain_metrics(method: str, metrics: dict, dataset) -> dict:
    return {
        "method": method,
        "split": dataset.split,
        "horizon": dataset.horizon,
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "mse": metrics.get("mse"),
        "nse": metrics.get("nse"),
        "csi_gamma_0_001": metrics.get("csi_gamma_0_001"),
        "csi_gamma_0_01": metrics.get("csi_gamma_0_01"),
        "path_iou_gamma_0_001": metrics.get("path_iou_gamma_0_001"),
        "path_iou_gamma_0_01": metrics.get("path_iou_gamma_0_01"),
        "negative_prediction_ratio": None,
    }


def print_table(rows: list[dict]) -> None:
    widths = {column: max(len(column), *(len(format_value(row.get(column))) for row in rows)) for column in COLUMNS}
    print(" | ".join(column.ljust(widths[column]) for column in COLUMNS))
    print("-+-".join("-" * widths[column] for column in COLUMNS))
    for row in rows:
        print(" | ".join(format_value(row.get(column)).ljust(widths[column]) for column in COLUMNS))


def format_value(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6g}"
    return str(value)


def save_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS + ["pred_min", "pred_max", "pred_mean"])
        writer.writeheader()
        writer.writerows(rows)


def output_stem(checkpoint_path: Path, model_name: str, split: str, horizon: int) -> str:
    run_id = checkpoint_path.parent.name
    safe_model = "".join(char if char.isalnum() or char in "-_" else "_" for char in model_name)
    return f"{safe_model}_{run_id}_{split}_h{horizon}_same_sample_comparison"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a FloodCastBench CNN checkpoint and optional same-sample baselines.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--compare-baselines", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / "floodcastbench_model_evaluations")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else PROJECT_DIR / args.checkpoint
    results = evaluate(checkpoint_path, args.split, args.horizon, args.max_batches, args.compare_baselines, args.device)
    print_table(results["rows"])
    print(f"raw_root_mtime_unchanged: {results['metadata']['raw_root_mtime_unchanged']}")

    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(checkpoint_path, results["metadata"]["model"], results["metadata"]["split"], results["metadata"]["horizon"])
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    save_json(results, json_path)
    save_csv(results["rows"], csv_path)
    print(f"saved_json: {json_path}")
    print(f"saved_csv: {csv_path}")


if __name__ == "__main__":
    main()
