from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
from rasterio.errors import NotGeoreferencedWarning
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from baselines.floodcastbench import predict_floodcastbench_baseline
from datasets import FloodCastBenchWaterDepthDataset
from metrics import BinaryMetricAccumulator, WaterDepthMetricAccumulator
from metrics.floodcastbench_eval import gamma_suffix
from models.flood_cnn import FloodCNNBaseline
from models.flood_latent_temporal import FloodLatentTemporalModel


COLUMNS = [
    "method",
    "split",
    "horizon",
    "samples",
    "mae",
    "rmse",
    "mse",
    "nse",
    "pearson_r",
    "csi_gamma_0_001",
    "iou_gamma_0_001",
    "f1_gamma_0_001",
    "precision_gamma_0_001",
    "recall_gamma_0_001",
    "csi_gamma_0_01",
    "iou_gamma_0_01",
    "f1_gamma_0_01",
    "precision_gamma_0_01",
    "recall_gamma_0_01",
    "path_iou_gamma_0_001",
    "path_iou_gamma_0_01",
    "negative_prediction_ratio",
    "pred_min",
    "pred_max",
    "pred_mean",
    "target_mean",
]


@dataclass
class TensorStatsAccumulator:
    count: int = 0
    negative_count: int = 0
    pred_sum: float = 0.0
    target_sum: float = 0.0
    pred_min: float = math.inf
    pred_max: float = -math.inf

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_flat = pred.detach().float().reshape(-1)
        target_flat = target.detach().float().reshape(-1)
        valid = torch.isfinite(pred_flat) & torch.isfinite(target_flat)
        if not bool(valid.any()):
            return
        pred_values = pred_flat[valid]
        target_values = target_flat[valid]
        self.count += int(pred_values.numel())
        self.negative_count += int((pred_values < 0).sum().item())
        self.pred_sum += float(pred_values.sum().item())
        self.target_sum += float(target_values.sum().item())
        self.pred_min = min(self.pred_min, float(pred_values.min().item()))
        self.pred_max = max(self.pred_max, float(pred_values.max().item()))

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            return {
                "negative_prediction_ratio": math.nan,
                "pred_min": math.nan,
                "pred_max": math.nan,
                "pred_mean": math.nan,
                "target_mean": math.nan,
            }
        return {
            "negative_prediction_ratio": self.negative_count / self.count,
            "pred_min": self.pred_min,
            "pred_max": self.pred_max,
            "pred_mean": self.pred_sum / self.count,
            "target_mean": self.target_sum / self.count,
        }


class FullForecastMetricBundle:
    def __init__(self, gammas: tuple[float, ...]) -> None:
        self.gammas = gammas
        self.water = WaterDepthMetricAccumulator()
        self.mask = {gamma: BinaryMetricAccumulator() for gamma in gammas}
        self.path = {gamma: BinaryMetricAccumulator() for gamma in gammas}
        self.stats = TensorStatsAccumulator()

    def update(self, pred: torch.Tensor, target: torch.Tensor, current: torch.Tensor) -> None:
        self.water.update(pred, target)
        self.stats.update(pred, target)
        for gamma in self.gammas:
            current_mask = current > gamma
            pred_mask = pred > gamma
            target_mask = target > gamma
            self.mask[gamma].update(pred_mask, target_mask)
            self.path[gamma].update(pred_mask & (~current_mask), target_mask & (~current_mask))

    def compute(self) -> dict[str, float]:
        metrics = dict(self.water.compute())
        metrics.update(self.stats.compute())
        for gamma, acc in self.mask.items():
            suffix = gamma_suffix(gamma)
            binary = acc.compute()
            metrics[f"csi_gamma_{suffix}"] = binary["csi"]
            metrics[f"iou_gamma_{suffix}"] = binary["iou"]
            metrics[f"f1_gamma_{suffix}"] = binary["f1"]
            metrics[f"precision_gamma_{suffix}"] = binary["precision"]
            metrics[f"recall_gamma_{suffix}"] = binary["recall"]
        for gamma, acc in self.path.items():
            metrics[f"path_iou_gamma_{gamma_suffix(gamma)}"] = acc.compute()["iou"]
        return metrics


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


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


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


def build_model_from_checkpoint(checkpoint: dict, dataset: FloodCastBenchWaterDepthDataset) -> torch.nn.Module:
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
        raise ValueError(f"Unsupported checkpoint model_name: {model_name}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def maybe_resize(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[-2:] != target.shape[-2:]:
        return torch.nn.functional.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
    return pred


def row_from_metrics(method: str, split: str, horizon: int, samples: int, metrics: dict[str, float]) -> dict[str, object]:
    row: dict[str, object] = {"method": method, "split": split, "horizon": horizon, "samples": samples}
    for column in COLUMNS:
        if column not in row:
            row[column] = metrics.get(column)
    return row


def evaluate_split(
    *,
    split: str,
    horizon: int | None,
    baselines: tuple[str, ...],
    checkpoints: dict[str, Path],
    reference_checkpoint: Path,
    device: str,
) -> dict:
    reference = load_checkpoint(reference_checkpoint)
    dataset = build_dataset_from_checkpoint(reference, split, horizon)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    gammas = tuple(float(value) for value in reference.get("config", {}).get("evaluation", {}).get("gammas", [0.001, 0.01]))
    torch_device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    bundles = {baseline: FullForecastMetricBundle(gammas) for baseline in baselines}
    models: dict[str, torch.nn.Module] = {}
    missing_checkpoints: dict[str, str] = {}
    model_bundles: dict[str, FullForecastMetricBundle] = {}

    for method, path in checkpoints.items():
        if not path.exists():
            missing_checkpoints[method] = str(path)
            continue
        checkpoint = load_checkpoint(path)
        model = build_model_from_checkpoint(checkpoint, dataset).to(torch_device)
        models[method] = model
        model_bundles[f"{method}_raw"] = FullForecastMetricBundle(gammas)
        model_bundles[f"{method}_clamped"] = FullForecastMetricBundle(gammas)

    raw_mtime_before = dataset.root.stat().st_mtime
    start = time.perf_counter()
    samples = 0

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(torch_device)
            y = y.to(torch_device)
            y_physical = dataset.denormalize_water_depth(y.detach()).cpu()
            x_physical = dataset.denormalize_water_depth(x.detach()).cpu()
            current_physical = x_physical[:, -1]

            for method, model in models.items():
                pred_normalized = maybe_resize(model(x), y)
                pred_physical = dataset.denormalize_water_depth(pred_normalized.detach()).cpu()
                model_bundles[f"{method}_raw"].update(pred_physical, y_physical, current_physical)
                model_bundles[f"{method}_clamped"].update(torch.clamp(pred_physical, min=0.0), y_physical, current_physical)

            for baseline, bundle in bundles.items():
                preds = [
                    predict_floodcastbench_baseline(x_physical[index], dataset.horizon, baseline)
                    for index in range(x_physical.shape[0])
                ]
                bundle.update(torch.stack(preds, dim=0), y_physical, current_physical)

            samples += int(x.shape[0])

    raw_mtime_after = dataset.root.stat().st_mtime
    rows = []
    for baseline, bundle in bundles.items():
        rows.append(row_from_metrics(baseline, dataset.split, dataset.horizon, samples, bundle.compute()))
    for method, bundle in model_bundles.items():
        rows.append(row_from_metrics(method, dataset.split, dataset.horizon, samples, bundle.compute()))

    return {
        "metadata": {
            "split": dataset.split,
            "horizon": dataset.horizon,
            "input_window": dataset.input_window,
            "event": dataset.event,
            "resolution": dataset.resolution,
            "fidelity": dataset.fidelity,
            "normalization_mode": dataset.normalization_mode,
            "normalization_stats_path": dataset.normalization_stats_path,
            "metrics_are_denormalized": True,
            "same_sample_comparison": True,
            "sample_order": "DataLoader(shuffle=False, batch_size=1), all methods updated inside the same sample loop",
            "num_samples": samples,
            "num_available_samples": len(dataset),
            "device": str(torch_device),
            "gammas": list(gammas),
            "reference_checkpoint": str(reference_checkpoint),
            "checkpoints": {method: str(path) for method, path in checkpoints.items()},
            "missing_checkpoints": missing_checkpoints,
            "raw_root": str(dataset.root),
            "raw_root_mtime_before": raw_mtime_before,
            "raw_root_mtime_after": raw_mtime_after,
            "raw_root_mtime_unchanged": raw_mtime_before == raw_mtime_after,
            "elapsed_seconds": time.perf_counter() - start,
        },
        "rows": rows,
    }


def format_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6g}"
    return str(value)


def print_table(rows: list[dict[str, object]]) -> None:
    visible = [
        "method",
        "samples",
        "rmse",
        "mae",
        "nse",
        "csi_gamma_0_001",
        "csi_gamma_0_01",
        "path_iou_gamma_0_001",
        "path_iou_gamma_0_01",
        "negative_prediction_ratio",
    ]
    widths = {column: max(len(column), *(len(format_value(row.get(column))) for row in rows)) for column in visible}
    print(" | ".join(column.ljust(widths[column]) for column in visible))
    print("-+-".join("-" * widths[column] for column in visible))
    for row in rows:
        print(" | ".join(format_value(row.get(column)).ljust(widths[column]) for column in visible))


def save_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare FloodCastBench baselines and trained forecasting models on the same samples.")
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument(
        "--cnn-checkpoint",
        type=Path,
        default=PROJECT_DIR / "train_runs" / "21-06-2026_16-37-25_fcb_cnn_norm_h20" / "checkpoint_best.pth",
    )
    parser.add_argument(
        "--latent-checkpoint",
        type=Path,
        default=PROJECT_DIR / "train_runs" / "22-06-2026_15-25-53_fcb_latent_conv_h20" / "checkpoint_best.pth",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / "floodcastbench_model_evaluations")
    parser.add_argument("--compare-baselines", action="store_true", help="Accepted for CLI compatibility; baselines are always included.")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

    cnn_checkpoint = args.cnn_checkpoint if args.cnn_checkpoint.is_absolute() else PROJECT_DIR / args.cnn_checkpoint
    latent_checkpoint = args.latent_checkpoint if args.latent_checkpoint.is_absolute() else PROJECT_DIR / args.latent_checkpoint
    reference_checkpoint = latent_checkpoint if latent_checkpoint.exists() else cnn_checkpoint

    results = evaluate_split(
        split=args.split,
        horizon=args.horizon,
        baselines=("persistence", "linear_delta"),
        checkpoints={
            "flood_cnn_baseline": cnn_checkpoint,
            "flood_latent_temporal": latent_checkpoint,
        },
        reference_checkpoint=reference_checkpoint,
        device=args.device,
    )

    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = latent_checkpoint.parent.name if latent_checkpoint.exists() else reference_checkpoint.parent.name
    stem = f"floodcastbench_models_{args.split}_h{args.horizon}_{run_id}_full_comparison"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    save_json(results, json_path)
    save_csv(results["rows"], csv_path)

    print_table(results["rows"])
    print(f"raw_root_mtime_unchanged: {results['metadata']['raw_root_mtime_unchanged']}")
    if results["metadata"]["missing_checkpoints"]:
        print(f"missing_checkpoints: {results['metadata']['missing_checkpoints']}")
    print(f"saved_json: {json_path}")
    print(f"saved_csv: {csv_path}")


if __name__ == "__main__":
    main()
