from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from datasets.floodcastbench_fno_plus_official_v1_dataset import (  # noqa: E402
    compute_train_normalization_stats,
    build_fno_plus_official_v1_dataset,
)
from models.fno_plus_official import FNOPlusOfficial3d  # noqa: E402
from training.utils import set_seed  # noqa: E402


DEFAULT_PATHS = {
    "dataset_root": Path("/home/wissam/utem-workspace/data/FloodCastBench"),
    "experiment_root": Path("/home/wissam/utem-workspace/experiments/FloodCastBench"),
    "checkpoint_root": Path("/home/wissam/utem-workspace/checkpoints/FloodCastBench"),
    "log_root": Path("/home/wissam/utem-workspace/logs/FloodCastBench"),
}

METRIC_FIELDS = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_current_relative_rmse",
    "val_current_relative_rmse",
    "train_classical_rmse",
    "val_classical_rmse",
    "train_nse",
    "val_nse",
    "train_pearson_r",
    "val_pearson_r",
    "train_csi_gamma_0_001",
    "val_csi_gamma_0_001",
    "train_csi_gamma_0_01",
    "val_csi_gamma_0_01",
    "train_precision_gamma_0_001",
    "val_precision_gamma_0_001",
    "train_recall_gamma_0_001",
    "val_recall_gamma_0_001",
    "train_flooded_area_ratio_gamma_0_001",
    "val_flooded_area_ratio_gamma_0_001",
    "train_negative_prediction_ratio",
    "val_negative_prediction_ratio",
    "train_pred_min",
    "val_pred_min",
    "train_pred_max",
    "val_pred_max",
    "train_pred_mean",
    "val_pred_mean",
    "train_target_min",
    "val_target_min",
    "train_target_max",
    "val_target_max",
    "train_target_mean",
    "val_target_mean",
    "learning_rate",
    "epoch_time_sec",
]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_yaml(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def resolve_path(value: str | Path | None, fallback: Path) -> Path:
    selected = Path(value) if value not in (None, "") else fallback
    return selected if selected.is_absolute() else PROJECT_DIR / selected


def path_from_config(config: dict, key: str) -> Path:
    return resolve_path(config.get("paths", {}).get(key), DEFAULT_PATHS[key])


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in str(value)).strip("_")


def resolve_device(value: str) -> torch.device:
    value = str(value).lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    config = json.loads(json.dumps(config))
    paths = config.setdefault("paths", {})
    for attr in ("dataset_root", "experiment_root", "checkpoint_root", "log_root"):
        value = getattr(args, attr)
        if value is not None:
            paths[attr] = str(value)
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        config.setdefault("loader", {})["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        config.setdefault("loader", {})["num_workers"] = int(args.num_workers)
    if args.device is not None:
        config.setdefault("training", {})["device"] = args.device
    return config


def run_suffix(config: dict) -> str:
    experiment_name = str(config.get("experiment", {}).get("name", "")).strip()
    if experiment_name:
        return safe_name(experiment_name)
    dataset = config["dataset"]
    suffix = "_".join(
        [
            "fcb",
            "fno_plus_official_v1_normalized_pilot",
            str(dataset.get("fidelity", "high")).replace("high", "highfid").replace("low", "lowfid"),
            str(dataset.get("resolution", "60m")),
        ]
    )
    if int(config.get("training", {}).get("epochs", 100)) != 5:
        suffix = suffix.replace("_pilot", "")
    return suffix


def build_model(config: dict) -> FNOPlusOfficial3d:
    model_config = config["model"]
    return FNOPlusOfficial3d(
        input_channels=int(model_config.get("input_channels", 6)),
        output_steps=int(model_config.get("output_steps", 19)),
        modes=int(model_config.get("modes", 12)),
        width=int(model_config.get("width", 20)),
        fourier_layers=int(model_config.get("fourier_layers", 4)),
        output_offset=int(model_config.get("output_offset", 1)),
    )


def build_loader(dataset, config: dict, shuffle: bool) -> DataLoader:
    loader_config = config["loader"]
    return DataLoader(
        dataset,
        batch_size=int(loader_config.get("batch_size", 1)),
        shuffle=shuffle,
        num_workers=int(loader_config.get("num_workers", 0)),
        pin_memory=bool(loader_config.get("pin_memory", False)),
    )


class PhysicalMetricAccumulator:
    def __init__(self, gammas: tuple[float, ...] = (0.001, 0.01)) -> None:
        self.gammas = tuple(float(gamma) for gamma in gammas)
        self.sse = 0.0
        self.count = 0
        self.target_sum = 0.0
        self.target_sq_sum = 0.0
        self.pred_sum = 0.0
        self.pred_sq_sum = 0.0
        self.cross_sum = 0.0
        self.pred_min = math.inf
        self.pred_max = -math.inf
        self.target_min = math.inf
        self.target_max = -math.inf
        self.negative_count = 0
        self.gamma_counts = {
            gamma: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "pred": 0.0, "target": 0.0}
            for gamma in self.gammas
        }

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.detach().float().cpu()
        target = target.detach().float().cpu()
        error = pred - target
        self.sse += float(torch.sum(error**2).item())
        self.count += int(target.numel())
        self.target_sum += float(torch.sum(target).item())
        self.target_sq_sum += float(torch.sum(target**2).item())
        self.pred_sum += float(torch.sum(pred).item())
        self.pred_sq_sum += float(torch.sum(pred**2).item())
        self.cross_sum += float(torch.sum(pred * target).item())
        self.pred_min = min(self.pred_min, float(torch.min(pred).item()))
        self.pred_max = max(self.pred_max, float(torch.max(pred).item()))
        self.target_min = min(self.target_min, float(torch.min(target).item()))
        self.target_max = max(self.target_max, float(torch.max(target).item()))
        self.negative_count += int((pred < 0).sum().item())
        for gamma in self.gammas:
            pred_mask = pred > gamma
            target_mask = target > gamma
            counts = self.gamma_counts[gamma]
            counts["tp"] += float(torch.logical_and(pred_mask, target_mask).sum().item())
            counts["fp"] += float(torch.logical_and(pred_mask, ~target_mask).sum().item())
            counts["fn"] += float(torch.logical_and(~pred_mask, target_mask).sum().item())
            counts["pred"] += float(pred_mask.sum().item())
            counts["target"] += float(target_mask.sum().item())

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        eps = 1e-12
        target_mean = self.target_sum / self.count
        pred_mean = self.pred_sum / self.count
        target_var_sum = self.target_sq_sum - self.count * target_mean * target_mean
        pred_var_sum = self.pred_sq_sum - self.count * pred_mean * pred_mean
        covariance_sum = self.cross_sum - self.count * pred_mean * target_mean
        result = {
            "current_relative_rmse": math.sqrt(self.sse / (self.target_sq_sum + eps)),
            "classical_rmse": math.sqrt(self.sse / self.count),
            "nse": 1.0 - self.sse / (target_var_sum + eps),
            "pearson_r": covariance_sum / math.sqrt(max(pred_var_sum * target_var_sum, 0.0) + eps),
            "negative_prediction_ratio": self.negative_count / self.count,
            "pred_min": self.pred_min,
            "pred_max": self.pred_max,
            "pred_mean": pred_mean,
            "target_min": self.target_min,
            "target_max": self.target_max,
            "target_mean": target_mean,
        }
        for gamma, counts in self.gamma_counts.items():
            suffix = str(gamma).replace(".", "_")
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]
            result[f"csi_gamma_{suffix}"] = tp / (tp + fp + fn + eps)
            result[f"precision_gamma_{suffix}"] = tp / (tp + fp + eps)
            result[f"recall_gamma_{suffix}"] = tp / (tp + fn + eps)
            result[f"flooded_area_ratio_gamma_{suffix}"] = counts["pred"] / (counts["target"] + eps)
        return result


def run_epoch(
    model,
    loader,
    criterion,
    device,
    inverse_target,
    optimizer=None,
    max_batches: int | None = None,
) -> dict:
    train = optimizer is not None
    model.train() if train else model.eval()
    accumulator = PhysicalMetricAccumulator((0.001, 0.01))
    loss_sum = 0.0
    batches = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_index, (x, target_norm, _) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            target_norm = target_norm.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            pred_norm = model(x)
            loss = criterion(pred_norm, target_norm)
            if train:
                loss.backward()
                optimizer.step()
            pred_physical = inverse_target(pred_norm)
            target_physical = inverse_target(target_norm)
            accumulator.update(pred_physical, target_physical)
            loss_sum += float(loss.item())
            batches += 1
    metrics = accumulator.compute()
    metrics["loss"] = loss_sum / batches if batches else math.nan
    metrics["batches"] = batches
    return metrics


def write_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=METRIC_FIELDS).writeheader()


def append_metrics(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writerow({field: row.get(field, math.nan) for field in METRIC_FIELDS})


def create_run_dirs(config: dict) -> tuple[Path, Path, Path]:
    name = f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}_{safe_name(run_suffix(config))}"
    experiment_dir = path_from_config(config, "experiment_root") / name
    checkpoint_dir = path_from_config(config, "checkpoint_root") / name
    log_dir = path_from_config(config, "log_root") / name
    experiment_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir, checkpoint_dir, log_dir


def build_dataset(config: dict, split: str, stats: dict):
    return build_fno_plus_official_v1_dataset(
        path_from_config(config, "dataset_root"),
        config,
        split=split,
        normalization_stats=stats,
    )


def dry_run(config: dict) -> None:
    stats = compute_train_normalization_stats(
        path_from_config(config, "dataset_root"),
        config,
        min_std=float(config.get("normalization", {}).get("min_std", 1e-6)),
    )
    train_dataset = build_dataset(config, "train", stats)
    val_dataset = build_dataset(config, "val", stats)
    print("=== RESOLVED OFFICIAL FNO+ V1 NORMALIZED CONFIG ===")
    print(f"run_suffix: {run_suffix(config)}")
    print(f"dataset_root: {path_from_config(config, 'dataset_root')}")
    print(f"experiment_root: {path_from_config(config, 'experiment_root')}")
    print(f"checkpoint_root: {path_from_config(config, 'checkpoint_root')}")
    print(f"train_samples: {len(train_dataset)}")
    print(f"val_samples: {len(val_dataset)}")
    print(f"context_length: {train_dataset.context_length}")
    print(f"input_shape: [6, {train_dataset.height}, {train_dataset.width}, {train_dataset.window_length}]")
    print(f"target_shape: [1, {train_dataset.height}, {train_dataset.width}, 19]")
    print("normalization_stats:")
    print(json.dumps(stats["channels"], indent=2))
    print("run_dir: DRY_RUN_NO_RUN_DIR")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train official FNO+ v1 normalized reproduction experiment.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device")
    parser.add_argument("--dry-run-config", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args)
    set_seed(int(config["training"].get("seed", 42)))
    if args.dry_run_config:
        dry_run(config)
        return 0

    experiment_dir, checkpoint_dir, log_dir = create_run_dirs(config)
    stats = compute_train_normalization_stats(
        path_from_config(config, "dataset_root"),
        config,
        min_std=float(config.get("normalization", {}).get("min_std", 1e-6)),
    )
    save_yaml(config, experiment_dir / "config.yaml")
    save_json(stats, experiment_dir / "normalization_stats.json")

    metrics_path = experiment_dir / "metrics.csv"
    write_header(metrics_path)

    train_dataset = build_dataset(config, "train", stats)
    val_dataset = build_dataset(config, "val", stats)
    train_loader = build_loader(train_dataset, config, shuffle=True)
    val_loader = build_loader(val_dataset, config, shuffle=False)
    device = resolve_device(config["training"].get("device", "auto"))
    model = build_model(config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 0.001)),
        betas=tuple(float(x) for x in config["training"].get("betas", [0.9, 0.999])),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["training"].get("epochs", 100)),
        eta_min=float(config["training"].get("min_learning_rate", 0.0)),
    )
    selection_metric = config["training"].get("selection_metric", "val_current_relative_rmse")
    best_value = math.inf
    best_epoch = None

    inverse_target = train_dataset.inverse_transform_target
    for epoch in range(1, int(config["training"].get("epochs", 100)) + 1):
        start = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            inverse_target,
            optimizer,
            args.max_train_batches,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            inverse_target,
            None,
            args.max_val_batches,
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": time.perf_counter() - start,
        }
        for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
            for key in (
                "current_relative_rmse",
                "classical_rmse",
                "nse",
                "pearson_r",
                "csi_gamma_0_001",
                "csi_gamma_0_01",
                "precision_gamma_0_001",
                "recall_gamma_0_001",
                "flooded_area_ratio_gamma_0_001",
                "negative_prediction_ratio",
                "pred_min",
                "pred_max",
                "pred_mean",
                "target_min",
                "target_max",
                "target_mean",
            ):
                row[f"{prefix}_{key}"] = metrics[key]
        append_metrics(metrics_path, row)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "normalization_stats": stats,
            "metrics": row,
            "selection_metric": selection_metric,
            "best_selection_metric": best_value,
        }
        torch.save(checkpoint, checkpoint_dir / "checkpoint_last.pth")
        value = float(row[selection_metric])
        if value < best_value:
            best_value = value
            best_epoch = epoch
            checkpoint["best_selection_metric"] = best_value
            torch.save(checkpoint, checkpoint_dir / "checkpoint_best.pth")
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} {selection_metric}={value:.6f}"
        )

    save_json(
        {
            "experiment_dir": str(experiment_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "log_dir": str(log_dir),
            "epochs": int(config["training"].get("epochs", 100)),
            "selection_metric": selection_metric,
            "best_epoch": best_epoch,
            "best_selection_metric": best_value,
            "normalization_stats": str(experiment_dir / "normalization_stats.json"),
        },
        experiment_dir / "summary.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
