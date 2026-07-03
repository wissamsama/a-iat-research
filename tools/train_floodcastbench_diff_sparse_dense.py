from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from datasets.floodcastbench_diff_sparse_dataset import build_diff_sparse_dense_dataset  # noqa: E402
from datasets.floodcastbench_fno_plus_official_v1_dataset import compute_train_normalization_stats  # noqa: E402
from models.diff_sparse import DenseDiffSparseModel  # noqa: E402
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
    "val_x0_rmse",
    "learning_rate",
    "elapsed_seconds",
]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_yaml(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def resolve_path(value: str | Path | None, fallback: Path) -> Path:
    selected = Path(value) if value not in (None, "") else fallback
    return selected if selected.is_absolute() else PROJECT_DIR / selected


def path_from_config(config: dict[str, Any], key: str) -> Path:
    return resolve_path(config.get("paths", {}).get(key), DEFAULT_PATHS[key])


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in str(value)).strip("_")


def run_suffix(config: dict[str, Any]) -> str:
    return safe_name(config.get("experiment", {}).get("name", "fcb_diff_sparse_dense_missing0_highfid_60m"))


def resolve_device(value: str) -> torch.device:
    value = str(value).lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
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


def cli_args_for_summary(args: argparse.Namespace) -> dict[str, Any]:
    summary_args: dict[str, Any] = {}
    for key, value in vars(args).items():
        summary_args[key] = str(value) if isinstance(value, Path) else value
    return summary_args


def command_reconstruction() -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])


def git_status_short() -> list[str]:
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return [f"git status --short failed: {exc!r}"]

    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or f"returncode={completed.returncode}"
        return [f"git status --short failed: {message}"]

    return [line for line in completed.stdout.splitlines() if line.strip()]


def build_loader(dataset, config: dict[str, Any], shuffle: bool) -> DataLoader:
    loader_config = config["loader"]
    return DataLoader(
        dataset,
        batch_size=int(loader_config.get("batch_size", 1)),
        shuffle=shuffle,
        num_workers=int(loader_config.get("num_workers", 0)),
        pin_memory=bool(loader_config.get("pin_memory", False)),
    )


def write_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=METRIC_FIELDS).writeheader()


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writerow({field: row.get(field, math.nan) for field in METRIC_FIELDS})


def create_run_dirs(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    base_name = f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}_{run_suffix(config)}"
    suffix = ""
    attempt = 1
    while True:
        name = f"{base_name}{suffix}"
        experiment_dir = path_from_config(config, "experiment_root") / name
        checkpoint_dir = path_from_config(config, "checkpoint_root") / name
        log_dir = path_from_config(config, "log_root") / name
        if not experiment_dir.exists() and not checkpoint_dir.exists() and not log_dir.exists():
            experiment_dir.mkdir(parents=True, exist_ok=False)
            checkpoint_dir.mkdir(parents=True, exist_ok=False)
            log_dir.mkdir(parents=True, exist_ok=False)
            return experiment_dir, checkpoint_dir, log_dir
        attempt += 1
        suffix = f"_r{attempt}"


def build_datasets(config: dict[str, Any], stats: dict[str, Any]):
    root = path_from_config(config, "dataset_root")
    return (
        build_diff_sparse_dense_dataset(root, config, split="train", normalization_stats=stats),
        build_diff_sparse_dense_dataset(root, config, split="val", normalization_stats=stats),
    )


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def tensor_shape(value: torch.Tensor) -> list[int]:
    return [int(dim) for dim in value.shape]


def first_batch_shape_report(model: DenseDiffSparseModel, batch: dict[str, Any]) -> dict[str, Any]:
    context = batch["context"]
    context_mask = batch["context_mask"]
    target = batch["target"]
    timesteps = torch.zeros(target.shape[0], dtype=torch.long, device=target.device)
    noise = torch.randn_like(target)
    x_noisy = model.q_sample(target, timesteps, noise=noise)
    pred = model(x_noisy, timesteps, context, context_mask)
    return {
        "context": tensor_shape(context),
        "context_mask": tensor_shape(context_mask),
        "target": tensor_shape(target),
        "noisy_target": tensor_shape(x_noisy),
        "timesteps": tensor_shape(timesteps),
        "prediction": tensor_shape(pred),
        "context_mask_min": float(context_mask.min().item()),
        "context_mask_max": float(context_mask.max().item()),
        "prediction_finite": bool(torch.isfinite(pred).all().item()),
    }


def run_epoch(
    model: DenseDiffSparseModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
    grad_clip_norm: float | None = None,
) -> dict[str, float]:
    train = optimizer is not None
    model.train() if train else model.eval()
    loss_sum = 0.0
    rmse_sum = 0.0
    batches = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            loss, diagnostics = model.training_step_loss(batch)
            if train:
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                optimizer.step()
            loss_sum += float(loss.detach().item())
            rmse_sum += float(diagnostics["x0_rmse"])
            batches += 1
    return {
        "loss": loss_sum / batches if batches else math.nan,
        "x0_rmse": rmse_sum / batches if batches else math.nan,
        "batches": float(batches),
    }


def dry_run(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    stats = compute_train_normalization_stats(
        path_from_config(config, "dataset_root"),
        config,
        min_std=float(config.get("normalization", {}).get("min_std", 1e-6)),
    )
    train_dataset, val_dataset = build_datasets(config, stats)
    train_loader = build_loader(train_dataset, config, shuffle=False)
    device = resolve_device(config["training"].get("device", "auto"))
    model = DenseDiffSparseModel(config).to(device)
    batch = move_batch_to_device(next(iter(train_loader)), device)
    with torch.no_grad():
        shapes = first_batch_shape_report(model, batch)
        loss, diagnostics = model.training_step_loss(batch)

    report = {
        "code_root": str(PROJECT_DIR),
        "dataset_root": str(path_from_config(config, "dataset_root")),
        "experiment_root": str(path_from_config(config, "experiment_root")),
        "checkpoint_root": str(path_from_config(config, "checkpoint_root")),
        "log_root": str(path_from_config(config, "log_root")),
        "device": str(device),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "missing_rate": float(config.get("masking", {}).get("missing_rate", 0.0)),
        "diffusion_steps": int(config.get("diffusion", {}).get("steps", 20)),
        "prediction_type": str(config.get("diffusion", {}).get("prediction_type", "x0")),
        "first_batch_shapes": shapes,
        "dry_run_loss": float(loss.item()),
        "dry_run_diagnostics": diagnostics,
        "writes": "none",
    }
    print("=== DENSE DIFF-SPARSE DRY RUN ===")
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Train dense missing-rate-zero DIFF-SPARSE-style FloodCastBench baseline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--device")
    parser.add_argument("--dry-run-config", action="store_true")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args)
    seed = int(config.get("training", {}).get("seed", config.get("experiment", {}).get("seed", 42)))
    set_seed(seed)

    print(f"code_root: {PROJECT_DIR}")
    print(f"config_path: {args.config}")
    print(f"dataset_root: {path_from_config(config, 'dataset_root')}")
    print(f"experiment_root: {path_from_config(config, 'experiment_root')}")
    print(f"checkpoint_root: {path_from_config(config, 'checkpoint_root')}")
    print(f"log_root: {path_from_config(config, 'log_root')}")
    print(f"missing_rate: {config.get('masking', {}).get('missing_rate', 0.0)}")
    print(f"diffusion_steps: {config.get('diffusion', {}).get('steps', 20)}")
    print(f"prediction_type: {config.get('diffusion', {}).get('prediction_type', 'x0')}")

    if args.dry_run_config:
        dry_run(config, args)
        return 0

    experiment_dir, checkpoint_dir, log_dir = create_run_dirs(config)
    print(f"experiment_dir: {experiment_dir}")
    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"log_dir: {log_dir}")

    stats = compute_train_normalization_stats(
        path_from_config(config, "dataset_root"),
        config,
        min_std=float(config.get("normalization", {}).get("min_std", 1e-6)),
    )
    save_yaml(config, experiment_dir / "config.yaml")
    save_json(stats, experiment_dir / "normalization_stats.json")

    train_dataset, val_dataset = build_datasets(config, stats)
    train_loader = build_loader(train_dataset, config, shuffle=True)
    val_loader = build_loader(val_dataset, config, shuffle=False)
    device = resolve_device(config["training"].get("device", "auto"))
    print(f"device: {device}")

    model = DenseDiffSparseModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 1e-4)),
        weight_decay=float(config["training"].get("weight_decay", 1e-5)),
    )
    metrics_path = experiment_dir / "metrics.csv"
    write_header(metrics_path)

    first_batch = move_batch_to_device(next(iter(build_loader(train_dataset, config, shuffle=False))), device)
    with torch.no_grad():
        shapes = first_batch_shape_report(model, first_batch)
    print("first_batch_shapes:")
    print(json.dumps(shapes, indent=2))

    best_value = math.inf
    best_epoch = None
    grad_clip_norm = config["training"].get("grad_clip_norm")
    epochs = int(config["training"].get("epochs", 1))
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
            grad_clip_norm=float(grad_clip_norm) if grad_clip_norm is not None else None,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            max_batches=args.max_val_batches,
        )
        elapsed = time.perf_counter() - start
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_x0_rmse": val_metrics["x0_rmse"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": elapsed,
        }
        append_metrics(metrics_path, row)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "normalization_stats": stats,
            "metrics": row,
            "selection_metric": "val_loss",
            "scientific_status": "dense_missing0_sanity_baseline",
        }
        torch.save(checkpoint, checkpoint_dir / "checkpoint_last.pth")
        if float(row["val_loss"]) < best_value:
            best_value = float(row["val_loss"])
            best_epoch = epoch
            torch.save(checkpoint, checkpoint_dir / "checkpoint_best.pth")
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.6f} "
            f"val_loss={row['val_loss']:.6f} val_x0_rmse={row['val_x0_rmse']:.6f}"
        )

    save_json(
        {
            "experiment_dir": str(experiment_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "log_dir": str(log_dir),
            "epochs": epochs,
            "selection_metric": "val_loss",
            "best_epoch": best_epoch,
            "best_selection_metric": best_value,
            "metrics_csv": str(metrics_path),
            "normalization_stats": str(experiment_dir / "normalization_stats.json"),
            "metric_units": "normalized_for_initial_sanity",
            "first_batch_shapes": shapes,
            "cli_args": cli_args_for_summary(args),
            "command_reconstruction": command_reconstruction(),
            "git_status_short": git_status_short(),
            "scientific_status": "dense_missing0_sanity_baseline",
            "does_not_claim": [
                "full sparse-sensor DIFF-SPARSE reproduction",
                "superiority over FNO+",
                "long-horizon performance",
                "uncertainty calibration",
            ],
        },
        experiment_dir / "summary.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
