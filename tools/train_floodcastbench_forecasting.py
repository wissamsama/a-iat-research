from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml
from rasterio.errors import NotGeoreferencedWarning

PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from datasets import FloodCastBenchWaterDepthDataset
from metrics.floodcastbench_eval import RawClampedMetricBundle, flatten_metrics
from models.flood_cnn import FloodCNNBaseline
from models.flood_latent_temporal import FloodLatentTemporalModel
from training.utils import set_seed

METRIC_FIELDS = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_rmse_raw",
    "val_rmse_raw",
    "train_rmse_clamped",
    "val_rmse_clamped",
    "train_mae_raw",
    "val_mae_raw",
    "train_mae_clamped",
    "val_mae_clamped",
    "train_mse_raw",
    "val_mse_raw",
    "train_nse_raw",
    "val_nse_raw",
    "val_csi_gamma_0_001_raw",
    "val_csi_gamma_0_001_clamped",
    "val_csi_gamma_0_01_raw",
    "val_csi_gamma_0_01_clamped",
    "val_path_iou_gamma_0_001_raw",
    "val_path_iou_gamma_0_001_clamped",
    "val_path_iou_gamma_0_01_raw",
    "val_path_iou_gamma_0_01_clamped",
    "train_pred_min",
    "val_pred_min",
    "train_pred_max",
    "val_pred_max",
    "train_pred_mean",
    "val_pred_mean",
    "train_negative_prediction_ratio",
    "val_negative_prediction_ratio",
    "train_target_mean",
    "val_target_mean",
    "learning_rate",
    "epoch_time_sec",
]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_yaml(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def sanitize_json(value):
    if isinstance(value, dict):
        return {key: sanitize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(sanitize_json(data), file, indent=2)


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in str(value))


def create_run_dir(output_dir: str | Path, experiment_name: str) -> Path:
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    run_dir = PROJECT_DIR / output_dir / f"{timestamp}_{safe_name(experiment_name)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_metrics_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writeheader()


def append_metrics(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writerow({field: row.get(field, math.nan) for field in METRIC_FIELDS})


def split_ratios_from_config(config: dict) -> tuple[float, float, float]:
    ratios = config["dataset"].get("split_ratios", {})
    return (
        float(ratios.get("train", 0.70)),
        float(ratios.get("val", 0.15)),
        float(ratios.get("test", 0.15)),
    )


def normalization_from_config(dataset_config: dict):
    return dataset_config.get("normalization", {"mode": "none"})


def build_dataset(config: dict, split: str, horizon_override: int | None = None) -> FloodCastBenchWaterDepthDataset:
    dataset_config = config["dataset"]
    horizon = int(horizon_override if horizon_override is not None else dataset_config.get("horizon", 20))
    return FloodCastBenchWaterDepthDataset(
        root=PROJECT_DIR / dataset_config.get("root", "data/FloodCastBench"),
        event=dataset_config.get("event", "Australia flood"),
        fidelity=dataset_config.get("fidelity", "high"),
        resolution=dataset_config.get("resolution", "30m"),
        input_window=int(dataset_config.get("input_window", 5)),
        horizon=horizon,
        split=split,
        split_ratios=split_ratios_from_config(config),
        normalization=normalization_from_config(dataset_config),
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


def resolve_device(value: str) -> torch.device:
    value = str(value).lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def build_model(config: dict, dataset) -> nn.Module:
    model_config = config["model"]
    model_name = str(model_config.get("name", "flood_cnn_baseline")).lower()
    if model_name == "flood_cnn_baseline":
        return FloodCNNBaseline(
            input_window=int(model_config.get("input_window", dataset.input_window)),
            base_channels=int(model_config.get("base_channels", 16)),
            output_activation=model_config.get("output_activation", "identity"),
            final_bias_init=model_config.get("final_bias_init"),
        )
    if model_name == "flood_latent_temporal":
        return FloodLatentTemporalModel(
            input_window=int(model_config.get("input_window", dataset.input_window)),
            base_channels=int(model_config.get("base_channels", 16)),
            latent_channels=int(model_config.get("latent_channels", 64)),
            temporal_module=model_config.get("temporal_module", "temporal_conv"),
            residual_prediction=bool(model_config.get("residual_prediction", True)),
            output_activation=model_config.get("output_activation", "identity"),
            final_bias_init=model_config.get("final_bias_init"),
        )
    raise ValueError("model.name must be one of: flood_cnn_baseline, flood_latent_temporal")

def build_loss(name: str):
    name = str(name).lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    if name == "huber":
        return nn.HuberLoss()
    raise ValueError("training.loss must be one of: mse, mae, huber")


def maybe_resize(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[-2:] != target.shape[-2:]:
        return torch.nn.functional.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
    return pred


def run_epoch(model, loader, criterion, device, gammas, dataset, optimizer=None, max_batches: int | None = None) -> dict:
    train = optimizer is not None
    model.train() if train else model.eval()
    bundle = RawClampedMetricBundle(gammas)

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_index, (x, y, _) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = maybe_resize(model(x), y)
            loss = criterion(pred, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            pred_physical = dataset.denormalize_water_depth(pred.detach())
            y_physical = dataset.denormalize_water_depth(y.detach())
            current_physical = dataset.denormalize_water_depth(x[:, -1].detach())
            bundle.update(pred_physical, y_physical, current_physical, loss_value=float(loss.detach().item()))

    return bundle.compute()


def checkpoint_payload(model, optimizer, epoch, config, best_val_rmse, dataset, model_config) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "config": config,
        "best_val_rmse": best_val_rmse,
        "model_name": model_config.get("name", "flood_cnn_baseline"),
        "dataset_metadata": {
            "dataset": "FloodCastBench",
            "task": "water_depth_forecasting",
            "event": dataset.event,
            "fidelity": dataset.fidelity,
            "resolution": dataset.resolution,
            "input_window": dataset.input_window,
            "horizon": dataset.horizon,
            "height": dataset.height,
            "width": dataset.width,
            "normalization_mode": dataset.normalization_mode,
            "normalization_stats_path": dataset.normalization_stats_path,
        },
    }


def print_debug_stats(label: str, metrics: dict) -> None:
    values = []
    for key in ("pred_min", "pred_max", "pred_mean", "negative_prediction_ratio", "target_mean"):
        if key in metrics:
            values.append(f"{key}={metrics[key]:.6g}")
    if values:
        print(f"  {label}: " + " | ".join(values))


def row_from_metrics(epoch, train_metrics, val_metrics, lr, epoch_time) -> dict:
    train_raw = flatten_metrics(train_metrics, "raw")
    train_clamped = flatten_metrics(train_metrics, "clamped")
    val_raw = flatten_metrics(val_metrics, "raw")
    val_clamped = flatten_metrics(val_metrics, "clamped")
    return {
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "val_loss": val_metrics["loss"],
        "train_rmse_raw": train_raw["rmse"],
        "val_rmse_raw": val_raw["rmse"],
        "train_rmse_clamped": train_clamped["rmse"],
        "val_rmse_clamped": val_clamped["rmse"],
        "train_mae_raw": train_raw["mae"],
        "val_mae_raw": val_raw["mae"],
        "train_mae_clamped": train_clamped["mae"],
        "val_mae_clamped": val_clamped["mae"],
        "train_mse_raw": train_raw["mse"],
        "val_mse_raw": val_raw["mse"],
        "train_nse_raw": train_raw["nse"],
        "val_nse_raw": val_raw["nse"],
        "val_csi_gamma_0_001_raw": val_raw["csi_gamma_0_001"],
        "val_csi_gamma_0_001_clamped": val_clamped["csi_gamma_0_001"],
        "val_csi_gamma_0_01_raw": val_raw["csi_gamma_0_01"],
        "val_csi_gamma_0_01_clamped": val_clamped["csi_gamma_0_01"],
        "val_path_iou_gamma_0_001_raw": val_raw["path_iou_gamma_0_001"],
        "val_path_iou_gamma_0_001_clamped": val_clamped["path_iou_gamma_0_001"],
        "val_path_iou_gamma_0_01_raw": val_raw["path_iou_gamma_0_01"],
        "val_path_iou_gamma_0_01_clamped": val_clamped["path_iou_gamma_0_01"],
        "train_pred_min": train_metrics.get("pred_min", math.nan),
        "val_pred_min": val_metrics.get("pred_min", math.nan),
        "train_pred_max": train_metrics.get("pred_max", math.nan),
        "val_pred_max": val_metrics.get("pred_max", math.nan),
        "train_pred_mean": train_metrics.get("pred_mean", math.nan),
        "val_pred_mean": val_metrics.get("pred_mean", math.nan),
        "train_negative_prediction_ratio": train_metrics.get("negative_prediction_ratio", math.nan),
        "val_negative_prediction_ratio": val_metrics.get("negative_prediction_ratio", math.nan),
        "train_target_mean": train_metrics.get("target_mean", math.nan),
        "val_target_mean": val_metrics.get("target_mean", math.nan),
        "learning_rate": lr,
        "epoch_time_sec": epoch_time,
    }


def run_normal_training(config, train_dataset, val_dataset, test_dataset, model, criterion, optimizer, device, gammas, run_dir, metrics_path, args) -> dict:
    train_loader = build_loader(train_dataset, config, shuffle=True)
    val_loader = build_loader(val_dataset, config, shuffle=False)
    test_loader = build_loader(test_dataset, config, shuffle=False) if test_dataset is not None else None

    raw_mtime_before = train_dataset.root.stat().st_mtime
    best_val_rmse = math.inf
    best_epoch = None
    best_val_metrics = None
    patience = int(config["training"].get("early_stopping_patience", 3))
    epochs_without_improvement = 0
    epochs = int(config["training"].get("epochs", 5))
    start_time = time.perf_counter()

    print(f"train_samples: {len(train_dataset)} | val_samples: {len(val_dataset)}")

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, criterion, device, gammas, train_dataset, optimizer=optimizer, max_batches=args.max_train_batches)
        val_metrics = run_epoch(model, val_loader, criterion, device, gammas, val_dataset, optimizer=None, max_batches=args.max_val_batches)
        val_raw = flatten_metrics(val_metrics, "raw")
        epoch_time = time.perf_counter() - epoch_start
        lr = optimizer.param_groups[0]["lr"]
        append_metrics(metrics_path, row_from_metrics(epoch, train_metrics, val_metrics, lr, epoch_time))
        print(f"epoch {epoch}: train_loss={train_metrics['loss']:.6g} val_rmse_raw={val_raw['rmse']:.6g} val_mae_raw={val_raw['mae']:.6g}")
        print_debug_stats("train first batch", train_metrics)
        print_debug_stats("val first batch", val_metrics)

        payload = checkpoint_payload(model, optimizer, epoch, config, best_val_rmse, train_dataset, config["model"])
        torch.save(payload, run_dir / "checkpoint_last.pth")
        if val_raw["rmse"] < best_val_rmse:
            best_val_rmse = val_raw["rmse"]
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            epochs_without_improvement = 0
            payload = checkpoint_payload(model, optimizer, epoch, config, best_val_rmse, train_dataset, config["model"])
            torch.save(payload, run_dir / "checkpoint_best.pth")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"early stopping after epoch {epoch}")
                break

    test_metrics = None
    evaluate_test = bool(config["evaluation"].get("evaluate_test_after_training", True)) and not args.skip_test and test_loader is not None
    effective_max_test = args.max_test_batches
    if effective_max_test is None and (args.max_train_batches is not None or args.max_val_batches is not None):
        effective_max_test = args.max_val_batches if args.max_val_batches is not None else args.max_train_batches
    if evaluate_test:
        best_path = run_dir / "checkpoint_best.pth"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = run_epoch(model, test_loader, criterion, device, gammas, test_dataset, optimizer=None, max_batches=effective_max_test)
        test_raw = flatten_metrics(test_metrics, "raw")
        print(f"test_rmse_raw={test_raw['rmse']:.6g} test_mae_raw={test_raw['mae']:.6g}")
        print_debug_stats("test first batch", test_metrics)

    raw_mtime_after = train_dataset.root.stat().st_mtime
    summary = {
        "experiment_name": config["experiment"].get("name"),
        "mode": "normal_training",
        "dataset": config["dataset"].get("name"),
        "event": train_dataset.event,
        "resolution": train_dataset.resolution,
        "input_window": train_dataset.input_window,
        "horizon": train_dataset.horizon,
        "model": config["model"].get("name", "flood_cnn_baseline"),
        "output_activation": config["model"].get("output_activation", "identity"),
        "final_bias_init": config["model"].get("final_bias_init"),
        "loss": config["training"].get("loss", "mse"),
        "device": str(device),
        "normalization_mode": train_dataset.normalization_mode,
        "normalization_stats_path": train_dataset.normalization_stats_path,
        "metrics_are_denormalized": True,
        "best_epoch": best_epoch,
        "best_val_rmse_raw": best_val_rmse,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "raw_root_mtime_unchanged": raw_mtime_before == raw_mtime_after,
        "total_time_sec": time.perf_counter() - start_time,
        "run_dir": str(run_dir),
    }
    save_json(summary, run_dir / "summary.json")
    return summary

def run_overfit_one_batch(config, train_dataset, model, criterion, optimizer, device, gammas, run_dir, metrics_path, args) -> dict:
    loader = build_loader(train_dataset, config, shuffle=False)
    x, y, _ = next(iter(loader))
    x = x.to(device)
    y = y.to(device)
    raw_mtime_before = train_dataset.root.stat().st_mtime
    start_time = time.perf_counter()
    best_rmse = math.inf
    best_step = None
    initial_metrics = None
    final_metrics = None

    print(f"overfit_one_batch: steps={args.overfit_steps} log_every={args.overfit_log_every}")
    for step in range(1, args.overfit_steps + 1):
        step_start = time.perf_counter()
        model.train()
        pred = maybe_resize(model(x), y)
        loss = criterion(pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        should_log = step == 1 or step % args.overfit_log_every == 0 or step == args.overfit_steps
        if should_log:
            model.eval()
            with torch.no_grad():
                pred_eval = maybe_resize(model(x), y)
                pred_physical = train_dataset.denormalize_water_depth(pred_eval.detach())
                y_physical = train_dataset.denormalize_water_depth(y.detach())
                current_physical = train_dataset.denormalize_water_depth(x[:, -1].detach())
                bundle = RawClampedMetricBundle(gammas)
                bundle.update(pred_physical, y_physical, current_physical, loss_value=float(criterion(pred_eval, y).detach().item()))
                metrics = bundle.compute()
            raw_metrics = flatten_metrics(metrics, "raw")
            if initial_metrics is None:
                initial_metrics = metrics
            final_metrics = metrics
            append_metrics(metrics_path, row_from_metrics(step, metrics, metrics, optimizer.param_groups[0]["lr"], time.perf_counter() - step_start))
            print(
                f"step {step}: loss={metrics['loss']:.6g} rmse_raw={raw_metrics['rmse']:.6g} "
                f"pred_mean={metrics.get('pred_mean', math.nan):.6g} target_mean={metrics.get('target_mean', math.nan):.6g}"
            )
            payload = checkpoint_payload(model, optimizer, step, config, raw_metrics["rmse"], train_dataset, config["model"])
            torch.save(payload, run_dir / "checkpoint_last.pth")
            if raw_metrics["rmse"] < best_rmse:
                best_rmse = raw_metrics["rmse"]
                best_step = step
                torch.save(payload, run_dir / "checkpoint_best.pth")

    raw_mtime_after = train_dataset.root.stat().st_mtime
    initial_loss = initial_metrics["loss"] if initial_metrics is not None else math.nan
    final_loss = final_metrics["loss"] if final_metrics is not None else math.nan
    summary = {
        "experiment_name": config["experiment"].get("name"),
        "mode": "overfit_one_batch",
        "dataset": config["dataset"].get("name"),
        "event": train_dataset.event,
        "resolution": train_dataset.resolution,
        "input_window": train_dataset.input_window,
        "horizon": train_dataset.horizon,
        "model": config["model"].get("name", "flood_cnn_baseline"),
        "output_activation": config["model"].get("output_activation", "identity"),
        "final_bias_init": config["model"].get("final_bias_init"),
        "loss": config["training"].get("loss", "mse"),
        "device": str(device),
        "normalization_mode": train_dataset.normalization_mode,
        "normalization_stats_path": train_dataset.normalization_stats_path,
        "metrics_are_denormalized": True,
        "overfit_steps": args.overfit_steps,
        "overfit_log_every": args.overfit_log_every,
        "best_step": best_step,
        "best_rmse_raw": best_rmse,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_decreased": final_loss < initial_loss,
        "final_metrics": final_metrics,
        "raw_root_mtime_unchanged": raw_mtime_before == raw_mtime_after,
        "total_time_sec": time.perf_counter() - start_time,
        "run_dir": str(run_dir),
    }
    save_json(summary, run_dir / "summary.json")
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight FloodCastBench CNN forecasting baseline.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "floodcastbench_cnn_baseline.yaml")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--overfit-one-batch", action="store_true")
    parser.add_argument("--overfit-steps", type=int, default=200)
    parser.add_argument("--overfit-log-every", type=int, default=10)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_DIR / args.config
    config = load_config(config_path)
    if args.horizon is not None:
        config["dataset"]["horizon"] = int(args.horizon)
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)

    set_seed(int(config["experiment"].get("seed", 42)))
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

    train_dataset = build_dataset(config, "train")
    val_dataset = None if args.overfit_one_batch else build_dataset(config, "val")
    test_dataset = (
        build_dataset(config, "test")
        if not args.overfit_one_batch and config["evaluation"].get("evaluate_test_after_training", True)
        else None
    )

    device = resolve_device(config["training"].get("device", "auto"))
    model = build_model(config, train_dataset).to(device)
    criterion = build_loss(config["training"].get("loss", "mse"))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 0.0001)),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    gammas = tuple(float(value) for value in config["evaluation"].get("gammas", [0.001, 0.01]))

    run_dir = create_run_dir(config["experiment"].get("output_dir", "train_runs"), config["experiment"].get("name", "floodcastbench_cnn_baseline"))
    metrics_path = run_dir / "metrics.csv"
    write_metrics_header(metrics_path)
    save_yaml(config, run_dir / "config.yaml")

    print(f"run_dir: {run_dir}")
    print(f"device: {device}")
    print(
        "model: "
        f"name={config['model'].get('name', 'flood_cnn_baseline')} "
        f"activation={config['model'].get('output_activation', 'identity')} "
        f"final_bias_init={config['model'].get('final_bias_init')}"
    )
    print(
        "normalization: "
        f"mode={train_dataset.normalization_mode} stats_path={train_dataset.normalization_stats_path}"
    )

    if args.overfit_one_batch:
        summary = run_overfit_one_batch(config, train_dataset, model, criterion, optimizer, device, gammas, run_dir, metrics_path, args)
    else:
        summary = run_normal_training(config, train_dataset, val_dataset, test_dataset, model, criterion, optimizer, device, gammas, run_dir, metrics_path, args)
    print(f"summary: {run_dir / 'summary.json'}")
    print(f"raw_root_mtime_unchanged: {summary['raw_root_mtime_unchanged']}")


if __name__ == "__main__":
    main()
