from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
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

VALID_TOP_LEVEL_CONFIG_KEYS = {"experiment", "paths", "dataset", "loader", "model", "training", "evaluation"}
REQUIRED_TOP_LEVEL_CONFIG_KEYS = {"experiment", "dataset", "loader", "model", "training", "evaluation"}
VALID_TEMPORAL_MODULES = {"identity", "temporal_conv", "gru", "mamba"}
DATASET_NAME_MAP = {"floodcastbench": "fcb"}
MODEL_NAME_MAP = {
    "flood_latent_temporal": "latent",
    "flood_cnn_baseline": "cnn_baseline",
}
TEMPORAL_MODULE_NAME_MAP = {
    "temporal_conv": "conv",
    "mamba": "mamba",
    "gru": "gru",
    "identity": "identity",
}
FUTURE_WSL_PATHS = {
    "dataset_root": Path("/home/wissam/utem-workspace/data/FloodCastBench"),
    "experiment_root": Path("/home/wissam/utem-workspace/experiments/FloodCastBench"),
    "checkpoint_root": Path("/home/wissam/utem-workspace/checkpoints/FloodCastBench"),
    "log_root": Path("/home/wissam/utem-workspace/logs/FloodCastBench"),
}


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return config


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


def is_wsl_like_environment() -> bool:
    if Path("/proc/sys/fs/binfmt_misc/WSLInterop").exists():
        return True
    proc_version = Path("/proc/version")
    if not proc_version.exists():
        return False
    return "microsoft" in proc_version.read_text(encoding="utf-8", errors="ignore").lower()


def resolve_path(value: str | Path | None, fallback: str | Path | None = None, *, base_dir: Path = PROJECT_DIR) -> Path:
    selected = value if value not in (None, "") else fallback
    if selected in (None, ""):
        raise ValueError("Cannot resolve an empty path without a fallback")
    path = Path(selected)
    return path if path.is_absolute() else base_dir / path


def configured_path(config: dict, key: str) -> Path | None:
    paths = config.get("paths", {})
    value = paths.get(key)
    if value in (None, ""):
        return None
    return resolve_path(value)


def dataset_root_from_config(config: dict) -> Path:
    root = configured_path(config, "dataset_root")
    if root is not None:
        return root
    dataset_config = config.get("dataset", {})
    configured_root = dataset_config.get("root")
    if is_wsl_like_environment() and str(configured_root).replace("\\", "/") in {"data/FloodCastBench", "None", ""}:
        return FUTURE_WSL_PATHS["dataset_root"]
    if configured_root not in (None, ""):
        return resolve_path(configured_root)
    if is_wsl_like_environment():
        return FUTURE_WSL_PATHS["dataset_root"]
    return PROJECT_DIR / "data" / "FloodCastBench"


def experiment_root_from_config(config: dict) -> Path:
    root = configured_path(config, "experiment_root")
    if root is not None:
        return root
    if is_wsl_like_environment():
        return FUTURE_WSL_PATHS["experiment_root"]
    return resolve_path(config.get("experiment", {}).get("output_dir", "train_runs"))


def checkpoint_root_from_config(config: dict) -> Path | None:
    root = configured_path(config, "checkpoint_root")
    if root is not None:
        return root
    if is_wsl_like_environment():
        return FUTURE_WSL_PATHS["checkpoint_root"]
    return None


def log_root_from_config(config: dict) -> Path | None:
    root = configured_path(config, "log_root")
    if root is not None:
        return root
    if is_wsl_like_environment():
        return FUTURE_WSL_PATHS["log_root"]
    return None


def require_positive_int(value, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer, got {parsed}")
    return parsed


def cli_args_for_json(args: argparse.Namespace) -> dict:
    result = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def safe_tag(value: str) -> str:
    cleaned = str(value).strip().lower().replace(" ", "_")
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in cleaned).strip("_")


def auto_run_suffix(config: dict, run_tag: str | None = None) -> str:
    dataset_name = str(config["dataset"].get("name", "floodcastbench")).lower()
    model_name = str(config["model"].get("name", "flood_latent_temporal")).lower()
    temporal_module = str(config["model"].get("temporal_module", "temporal_conv")).lower()
    horizon = require_positive_int(config["dataset"].get("horizon", 20), "dataset.horizon")

    dataset_part = DATASET_NAME_MAP.get(dataset_name, safe_tag(dataset_name))
    model_part = MODEL_NAME_MAP.get(model_name, safe_tag(model_name))

    if model_name == "flood_latent_temporal":
        module_part = TEMPORAL_MODULE_NAME_MAP.get(temporal_module, safe_tag(temporal_module))
        parts = [dataset_part, model_part, module_part, f"h{horizon}"]
    else:
        parts = [dataset_part, model_part, f"h{horizon}"]

    if run_tag:
        tag = safe_tag(run_tag)
        if tag:
            parts.append(tag)
    return "_".join(parts)


def update_normalization_stats_path(config: dict) -> None:
    dataset_config = config.get("dataset", {})
    normalization = dataset_config.get("normalization", {})
    if normalization.get("mode", "none") != "standard":
        return

    event = str(dataset_config.get("event", "australia")).lower().replace(" flood", "")
    event = safe_name(event).lower()
    resolution = str(dataset_config.get("resolution", "30m")).lower()
    input_window = require_positive_int(dataset_config.get("input_window", 5), "dataset.input_window")
    horizon = require_positive_int(dataset_config.get("horizon", 20), "dataset.horizon")
    stats_path = (
        PROJECT_DIR
        / "outputs"
        / "floodcastbench_normalization"
        / f"water_depth_{event}_{resolution}_train_input{input_window}_h{horizon}_stats.json"
    )
    normalization["stats_path"] = str(stats_path.relative_to(PROJECT_DIR)).replace("\\", "/")
    dataset_config["normalization"] = normalization


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    resolved = copy.deepcopy(config)
    missing_keys = sorted(REQUIRED_TOP_LEVEL_CONFIG_KEYS - set(resolved))
    if missing_keys:
        raise ValueError(f"Missing top-level config keys before applying CLI overrides: {', '.join(missing_keys)}")

    if args.experiment_name is not None:
        resolved["experiment"]["name"] = args.experiment_name
    paths = resolved.setdefault("paths", {})
    if args.dataset_root is not None:
        paths["dataset_root"] = str(args.dataset_root)
    if args.experiment_root is not None:
        paths["experiment_root"] = str(args.experiment_root)
    if args.checkpoint_root is not None:
        paths["checkpoint_root"] = str(args.checkpoint_root)
    if args.log_root is not None:
        paths["log_root"] = str(args.log_root)
    if args.horizon is not None:
        resolved["dataset"]["horizon"] = require_positive_int(args.horizon, "--horizon")
    if args.input_window is not None:
        input_window = require_positive_int(args.input_window, "--input-window")
        resolved["dataset"]["input_window"] = input_window
        resolved["model"]["input_window"] = input_window
    if args.temporal_module is not None:
        resolved["model"]["temporal_module"] = args.temporal_module
    if args.batch_size is not None:
        resolved["loader"]["batch_size"] = require_positive_int(args.batch_size, "--batch-size")
    if args.num_workers is not None:
        num_workers = int(args.num_workers)
        if num_workers < 0:
            raise ValueError("--num-workers must be >= 0")
        resolved["loader"]["num_workers"] = num_workers
    if args.pin_memory is not None:
        resolved["loader"]["pin_memory"] = bool(args.pin_memory)
    if args.epochs is not None:
        resolved["training"]["epochs"] = require_positive_int(args.epochs, "--epochs")
    if args.learning_rate is not None:
        learning_rate = float(args.learning_rate)
        if learning_rate <= 0:
            raise ValueError("--learning-rate must be > 0")
        resolved["training"]["learning_rate"] = learning_rate
    if args.weight_decay is not None:
        weight_decay = float(args.weight_decay)
        if weight_decay < 0:
            raise ValueError("--weight-decay must be >= 0")
        resolved["training"]["weight_decay"] = weight_decay
    if args.early_stopping_patience is not None:
        resolved["training"]["early_stopping_patience"] = require_positive_int(
            args.early_stopping_patience, "--early-stopping-patience"
        )
    if args.device is not None:
        resolved["training"]["device"] = args.device
    if args.evaluate_test_after_training is not None:
        resolved["evaluation"]["evaluate_test_after_training"] = bool(args.evaluate_test_after_training)
    if args.skip_test:
        resolved["evaluation"]["evaluate_test_after_training"] = False

    if "input_window" not in resolved["model"]:
        resolved["model"]["input_window"] = resolved["dataset"].get("input_window", 5)
    update_normalization_stats_path(resolved)
    automatic_name = auto_run_suffix(resolved, args.run_tag)
    if args.experiment_name is None:
        resolved["experiment"]["name"] = automatic_name
    return resolved


def validate_config(config: dict, config_path: Path) -> None:
    top_level_keys = set(config)
    unknown_keys = sorted(top_level_keys - VALID_TOP_LEVEL_CONFIG_KEYS)
    missing_keys = sorted(REQUIRED_TOP_LEVEL_CONFIG_KEYS - top_level_keys)
    if unknown_keys:
        raise ValueError(f"Unknown top-level config keys in {config_path}: {', '.join(unknown_keys)}")
    if missing_keys:
        raise ValueError(f"Missing top-level config keys in {config_path}: {', '.join(missing_keys)}")

    dataset_config = config["dataset"]
    model_config = config["model"]
    loader_config = config["loader"]
    training_config = config["training"]

    dataset_config["input_window"] = require_positive_int(dataset_config.get("input_window"), "dataset.input_window")
    dataset_config["horizon"] = require_positive_int(dataset_config.get("horizon"), "dataset.horizon")
    model_config["input_window"] = require_positive_int(model_config.get("input_window"), "model.input_window")
    loader_config["batch_size"] = require_positive_int(loader_config.get("batch_size"), "loader.batch_size")
    loader_config["num_workers"] = int(loader_config.get("num_workers", 0))
    if loader_config["num_workers"] < 0:
        raise ValueError("loader.num_workers must be >= 0")
    training_config["epochs"] = require_positive_int(training_config.get("epochs"), "training.epochs")

    temporal_module = str(model_config.get("temporal_module", "temporal_conv"))
    if temporal_module not in VALID_TEMPORAL_MODULES:
        raise ValueError(f"model.temporal_module must be one of {sorted(VALID_TEMPORAL_MODULES)}, got {temporal_module!r}")
    if temporal_module == "mamba" and importlib.util.find_spec("mamba_ssm") is None:
        raise RuntimeError(
            "model.temporal_module=mamba requires the official mamba-ssm package. "
            "Use the WSL CUDA environment or choose --temporal-module temporal_conv."
        )


def print_resolved_config_summary(
    config_path: Path,
    run_dir: Path | None,
    config: dict,
    run_suffix: str,
    saved_effective_config: Path | None,
    saved_cli_args: Path | None,
) -> None:
    dataset_config = config["dataset"]
    model_config = config["model"]
    loader_config = config["loader"]
    training_config = config["training"]
    evaluation_config = config["evaluation"]
    print("=== RESOLVED TRAINING CONFIG ===")
    print(f"config_file: {config_path}")
    print(f"run_dir: {run_dir if run_dir is not None else 'DRY_RUN_NO_RUN_DIR'}")
    print(f"experiment_name: {config['experiment'].get('name')}")
    print(f"run_suffix: {run_suffix}")
    print(f"dataset_root: {dataset_root_from_config(config)}")
    print(f"experiment_root: {experiment_root_from_config(config)}")
    print(f"checkpoint_root: {checkpoint_root_from_config(config) or 'RUN_DIR'}")
    print(f"log_root: {log_root_from_config(config) or 'NOT_CONFIGURED'}")
    print(f"dataset: {dataset_config.get('name')}")
    print(f"event: {dataset_config.get('event')}")
    print(f"resolution: {dataset_config.get('resolution')}")
    print(f"input_window: {dataset_config.get('input_window')}")
    print(f"horizon: {dataset_config.get('horizon')}")
    print(f"temporal_module: {model_config.get('temporal_module')}")
    print(f"batch_size: {loader_config.get('batch_size')}")
    print(f"epochs: {training_config.get('epochs')}")
    print(f"learning_rate: {training_config.get('learning_rate')}")
    print(f"weight_decay: {training_config.get('weight_decay')}")
    print(f"device: {training_config.get('device')}")
    print(f"num_workers: {loader_config.get('num_workers')}")
    print(f"pin_memory: {loader_config.get('pin_memory')}")
    print(f"evaluate_test_after_training: {evaluation_config.get('evaluate_test_after_training')}")
    print(f"saved_effective_config: {saved_effective_config if saved_effective_config is not None else 'DRY_RUN_NOT_SAVED'}")
    print(f"saved_cli_args: {saved_cli_args if saved_cli_args is not None else 'DRY_RUN_NOT_SAVED'}")


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in str(value))


def create_run_dir(root: str | Path, experiment_name: str) -> Path:
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    run_dir = resolve_path(root) / f"{timestamp}_{safe_name(experiment_name)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def create_checkpoint_dir(config: dict, run_dir: Path) -> Path:
    root = checkpoint_root_from_config(config)
    if root is None:
        return run_dir
    checkpoint_dir = root / run_dir.name
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    return checkpoint_dir


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
        root=dataset_root_from_config(config),
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


def checkpoint_payload(model, optimizer, scheduler, epoch, config, best_val_rmse, dataset, model_config) -> dict:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
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
    return payload


def build_scheduler(config: dict, optimizer):
    scheduler_config = config["training"].get("scheduler")
    if not scheduler_config:
        return None
    name = str(scheduler_config.get("name", "none")).lower()
    if name in {"none", "null"}:
        return None
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=require_positive_int(scheduler_config.get("step_size", 10), "training.scheduler.step_size"),
            gamma=float(scheduler_config.get("gamma", 0.1)),
        )
    raise ValueError("training.scheduler.name must be one of: none, step")


def load_resume_checkpoint(resume_from: Path, model, optimizer, scheduler, device) -> dict:
    if not resume_from.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
    checkpoint = torch.load(resume_from, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


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


def run_normal_training(
    config,
    train_dataset,
    val_dataset,
    test_dataset,
    model,
    criterion,
    optimizer,
    scheduler,
    device,
    gammas,
    run_dir,
    checkpoint_dir,
    metrics_path,
    args,
    resume_checkpoint: dict | None = None,
) -> dict:
    train_loader = build_loader(train_dataset, config, shuffle=True)
    val_loader = build_loader(val_dataset, config, shuffle=False)
    test_loader = build_loader(test_dataset, config, shuffle=False) if test_dataset is not None else None

    raw_mtime_before = train_dataset.root.stat().st_mtime
    best_val_rmse = float(resume_checkpoint.get("best_val_rmse", math.inf)) if resume_checkpoint else math.inf
    best_epoch = None
    best_val_metrics = None
    patience = int(config["training"].get("early_stopping_patience", 3))
    epochs_without_improvement = 0
    epochs = int(config["training"].get("epochs", 5))
    start_time = time.perf_counter()

    start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1 if resume_checkpoint else 1
    print(f"train_samples: {len(train_dataset)} | val_samples: {len(val_dataset)}")
    if resume_checkpoint:
        print(f"resuming_from_epoch: {start_epoch - 1} | next_epoch: {start_epoch} | best_val_rmse_raw: {best_val_rmse:.6g}")

    for epoch in range(start_epoch, epochs + 1):
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

        if scheduler is not None:
            scheduler.step()

        payload = checkpoint_payload(model, optimizer, scheduler, epoch, config, best_val_rmse, train_dataset, config["model"])
        torch.save(payload, checkpoint_dir / "checkpoint_last.pth")
        if val_raw["rmse"] < best_val_rmse:
            best_val_rmse = val_raw["rmse"]
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            epochs_without_improvement = 0
            payload = checkpoint_payload(model, optimizer, scheduler, epoch, config, best_val_rmse, train_dataset, config["model"])
            torch.save(payload, checkpoint_dir / "checkpoint_best.pth")
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
        best_path = checkpoint_dir / "checkpoint_best.pth"
        eval_checkpoint_path = best_path if best_path.exists() else checkpoint_dir / "checkpoint_last.pth"
        if eval_checkpoint_path.exists():
            checkpoint = torch.load(eval_checkpoint_path, map_location=device)
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
        "checkpoint_dir": str(checkpoint_dir),
        "resumed_from": str(args.resume_from) if args.resume_from else None,
        "start_epoch": start_epoch,
        "last_epoch": epoch if "epoch" in locals() else start_epoch - 1,
    }
    save_json(summary, run_dir / "summary.json")
    return summary

def run_overfit_one_batch(config, train_dataset, model, criterion, optimizer, scheduler, device, gammas, run_dir, checkpoint_dir, metrics_path, args) -> dict:
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
            payload = checkpoint_payload(model, optimizer, scheduler, step, config, raw_metrics["rmse"], train_dataset, config["model"])
            torch.save(payload, checkpoint_dir / "checkpoint_last.pth")
            if raw_metrics["rmse"] < best_rmse:
                best_rmse = raw_metrics["rmse"]
                best_step = step
                torch.save(payload, checkpoint_dir / "checkpoint_best.pth")

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
        "checkpoint_dir": str(checkpoint_dir),
    }
    save_json(summary, run_dir / "summary.json")
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight FloodCastBench CNN forecasting baseline.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "floodcastbench_latent_temporal.yaml")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--input-window", type=int, default=None)
    parser.add_argument("--temporal-module", choices=sorted(VALID_TEMPORAL_MODULES), default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--experiment-root", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--log-root", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    pin_memory_group = parser.add_mutually_exclusive_group()
    pin_memory_group.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=None)
    pin_memory_group.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    evaluation_group = parser.add_mutually_exclusive_group()
    evaluation_group.add_argument("--evaluate-test-after-training", dest="evaluate_test_after_training", action="store_true", default=None)
    evaluation_group.add_argument("--no-evaluate-test-after-training", dest="evaluate_test_after_training", action="store_false")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--dry-run-config", action="store_true")
    parser.add_argument("--save-cli-args", action="store_true")
    parser.add_argument("--overfit-one-batch", action="store_true")
    parser.add_argument("--overfit-steps", type=int, default=200)
    parser.add_argument("--overfit-log-every", type=int, default=10)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_DIR / args.config
    config = apply_cli_overrides(load_config(config_path), args)
    validate_config(config, config_path)
    run_suffix = str(config["experiment"].get("name"))
    if args.experiment_name is not None:
        print("WARNING: Manual experiment name override used; automatic naming from resolved config was bypassed.")

    if args.dry_run_config:
        print_resolved_config_summary(config_path, None, config, run_suffix, None, None)
        return

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
    scheduler = build_scheduler(config, optimizer)
    resume_checkpoint = None
    if args.resume_from is not None:
        resume_path = args.resume_from if args.resume_from.is_absolute() else PROJECT_DIR / args.resume_from
        resume_checkpoint = load_resume_checkpoint(resume_path, model, optimizer, scheduler, device)
    gammas = tuple(float(value) for value in config["evaluation"].get("gammas", [0.001, 0.01]))

    run_dir = create_run_dir(experiment_root_from_config(config), config["experiment"].get("name", "floodcastbench_cnn_baseline"))
    checkpoint_dir = create_checkpoint_dir(config, run_dir)
    log_dir = log_root_from_config(config)
    if log_dir is not None:
        (log_dir / run_dir.name).mkdir(parents=True, exist_ok=False)
    metrics_path = run_dir / "metrics.csv"
    write_metrics_header(metrics_path)
    saved_effective_config = run_dir / "config.yaml"
    saved_cli_args = run_dir / "cli_args.json" if args.save_cli_args else None
    save_yaml(config, saved_effective_config)
    if saved_cli_args is not None:
        save_json(cli_args_for_json(args), saved_cli_args)
    print_resolved_config_summary(config_path, run_dir, config, run_suffix, saved_effective_config, saved_cli_args)

    print(f"run_dir: {run_dir}")
    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"log_dir: {log_dir / run_dir.name if log_dir is not None else 'NOT_CONFIGURED'}")
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
        summary = run_overfit_one_batch(config, train_dataset, model, criterion, optimizer, scheduler, device, gammas, run_dir, checkpoint_dir, metrics_path, args)
    else:
        summary = run_normal_training(
            config,
            train_dataset,
            val_dataset,
            test_dataset,
            model,
            criterion,
            optimizer,
            scheduler,
            device,
            gammas,
            run_dir,
            checkpoint_dir,
            metrics_path,
            args,
            resume_checkpoint=resume_checkpoint,
        )
    print(f"summary: {run_dir / 'summary.json'}")
    print(f"raw_root_mtime_unchanged: {summary['raw_root_mtime_unchanged']}")


if __name__ == "__main__":
    main()
