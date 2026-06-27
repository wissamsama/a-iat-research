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

from datasets.floodcastbench_fno_dataset import FloodCastBenchFNODataset
from evaluation.floodcastbench_official_metrics import OfficialFloodMetricAccumulator
from models.fno_plus import FNOPlus2d
from training.utils import set_seed


FUTURE_WSL_PATHS = {
    "dataset_root": Path("/home/wissam/utem-workspace/data/FloodCastBench"),
    "experiment_root": Path("/home/wissam/utem-workspace/experiments/FloodCastBench"),
    "checkpoint_root": Path("/home/wissam/utem-workspace/checkpoints/FloodCastBench"),
    "log_root": Path("/home/wissam/utem-workspace/logs/FloodCastBench"),
}

METRIC_FIELDS = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_relative_rmse",
    "val_relative_rmse",
    "train_nse",
    "val_nse",
    "train_pearson_r",
    "val_pearson_r",
    "train_csi_gamma_0_001",
    "val_csi_gamma_0_001",
    "train_csi_gamma_0_01",
    "val_csi_gamma_0_01",
    "learning_rate",
    "epoch_time_sec",
]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def save_yaml(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def save_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def resolve_path(value: str | Path | None, fallback: Path) -> Path:
    selected = Path(value) if value not in (None, "") else fallback
    return selected if selected.is_absolute() else PROJECT_DIR / selected


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
    if args.dataset_root is not None:
        paths["dataset_root"] = str(args.dataset_root)
    if args.experiment_root is not None:
        paths["experiment_root"] = str(args.experiment_root)
    if args.checkpoint_root is not None:
        paths["checkpoint_root"] = str(args.checkpoint_root)
    if args.log_root is not None:
        paths["log_root"] = str(args.log_root)
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        config.setdefault("loader", {})["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        config.setdefault("loader", {})["num_workers"] = int(args.num_workers)
    if args.device is not None:
        config.setdefault("training", {})["device"] = args.device
    return config


def path_from_config(config: dict, key: str) -> Path:
    return resolve_path(config.get("paths", {}).get(key), FUTURE_WSL_PATHS[key])


def run_suffix(config: dict) -> str:
    dataset = config["dataset"]
    model = config["model"]
    return "_".join(
        [
            "fcb",
            str(model.get("name", "fno_plus")),
            str(dataset.get("fidelity", "highfid")).replace("high", "highfid").replace("low", "lowfid"),
            str(dataset.get("resolution", "60m")),
        ]
    )


def create_run_dirs(config: dict) -> tuple[Path, Path, Path]:
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    name = f"{timestamp}_{safe_name(run_suffix(config))}"
    experiment_dir = path_from_config(config, "experiment_root") / name
    checkpoint_dir = path_from_config(config, "checkpoint_root") / name
    log_dir = path_from_config(config, "log_root") / name
    experiment_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir, checkpoint_dir, log_dir


def build_dataset(config: dict, split: str) -> FloodCastBenchFNODataset:
    dataset_config = config["dataset"]
    return FloodCastBenchFNODataset(
        root=path_from_config(config, "dataset_root"),
        event=dataset_config.get("event", "australia"),
        fidelity=dataset_config.get("fidelity", "high"),
        resolution=dataset_config.get("resolution", "60m"),
        split=split,
        sample_length=int(dataset_config.get("sample_length", 20)),
        stride=int(dataset_config.get("stride", 20)),
        include_dem=bool(dataset_config.get("include_dem", True)),
        include_rainfall=bool(dataset_config.get("include_rainfall", True)),
        include_time=bool(dataset_config.get("include_time", True)),
        split_counts=dataset_config.get("split_counts"),
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


def build_model(config: dict, dataset: FloodCastBenchFNODataset) -> nn.Module:
    model_config = config["model"]
    return FNOPlus2d(
        input_channels=dataset.input_channels,
        output_steps=int(model_config.get("output_steps", 19)),
        modes=int(model_config.get("modes", 12)),
        width=int(model_config.get("width", 20)),
        fourier_layers=int(model_config.get("fourier_layers", 4)),
    )


def write_metrics_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=METRIC_FIELDS).writeheader()


def append_metrics(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writerow({field: row.get(field, math.nan) for field in METRIC_FIELDS})


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    gammas: tuple[float, ...],
    optimizer=None,
    max_batches: int | None = None,
) -> dict:
    train = optimizer is not None
    model.train() if train else model.eval()
    accumulator = OfficialFloodMetricAccumulator(gammas)
    loss_sum = 0.0
    batches = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_index, (x, target, _) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, target)
            if train:
                loss.backward()
                optimizer.step()
            accumulator.update(pred, target)
            loss_sum += float(loss.item())
            batches += 1
    metrics = accumulator.compute()
    metrics["loss"] = loss_sum / batches if batches else math.nan
    metrics["batches"] = batches
    return metrics


def print_dry_run(config: dict, train_dataset: FloodCastBenchFNODataset, val_dataset: FloodCastBenchFNODataset) -> None:
    print("=== RESOLVED FNO+ CONFIG ===")
    print(f"run_suffix: {run_suffix(config)}")
    print(f"dataset_root: {path_from_config(config, 'dataset_root')}")
    print(f"experiment_root: {path_from_config(config, 'experiment_root')}")
    print(f"checkpoint_root: {path_from_config(config, 'checkpoint_root')}")
    print(f"log_root: {path_from_config(config, 'log_root')}")
    print(f"event: {config['dataset'].get('event')}")
    print(f"resolution: {config['dataset'].get('resolution')}")
    print(f"train_samples: {len(train_dataset)}")
    print(f"val_samples: {len(val_dataset)}")
    print(f"input_channels: {train_dataset.input_channels}")
    print(f"output_steps: {train_dataset.output_steps}")
    print(f"spatial_shape: {train_dataset.height}x{train_dataset.width}")
    print("run_dir: DRY_RUN_NO_RUN_DIR")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the FloodCastBench FNO+ reproduction scaffold.")
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
    training_config = config["training"]
    loader_config = config["loader"]
    set_seed(int(training_config.get("seed", 42)))

    train_dataset = build_dataset(config, "train")
    val_dataset = build_dataset(config, "val")
    if args.dry_run_config:
        print_dry_run(config, train_dataset, val_dataset)
        return 0

    experiment_dir, checkpoint_dir, log_dir = create_run_dirs(config)
    save_yaml(config, experiment_dir / "config.yaml")
    metrics_path = experiment_dir / "metrics.csv"
    write_metrics_header(metrics_path)

    train_loader = build_loader(train_dataset, config, shuffle=True)
    val_loader = build_loader(val_dataset, config, shuffle=False)
    device = resolve_device(training_config.get("device", "auto"))
    model = build_model(config, train_dataset).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 0.001)),
        betas=tuple(float(x) for x in training_config.get("betas", [0.9, 0.999])),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(training_config.get("epochs", 100)),
        eta_min=float(training_config.get("min_learning_rate", 0.0)),
    )
    gammas = tuple(float(x) for x in config.get("evaluation", {}).get("gammas", [0.001, 0.01]))

    best_val = math.inf
    epochs = int(training_config.get("epochs", 100))
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            gammas,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            gammas,
            max_batches=args.max_val_batches,
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
            for key in ("relative_rmse", "nse", "pearson_r", "csi_gamma_0_001", "csi_gamma_0_01"):
                row[f"{prefix}_{key}"] = metrics[key]
        append_metrics(metrics_path, row)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "metrics": row,
        }
        torch.save(checkpoint, checkpoint_dir / "checkpoint_last.pth")
        if val_metrics["relative_rmse"] < best_val:
            best_val = val_metrics["relative_rmse"]
            torch.save(checkpoint, checkpoint_dir / "checkpoint_best.pth")
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.6f} "
            f"val_relative_rmse={val_metrics['relative_rmse']:.6f}"
        )

    save_json(
        {
            "experiment_dir": str(experiment_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "log_dir": str(log_dir),
            "best_val_relative_rmse": best_val,
            "epochs": epochs,
        },
        experiment_dir / "summary.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
