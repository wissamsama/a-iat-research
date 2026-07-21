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

from datasets.floodcastbench_diff_sparse_v1_dataset import (  # noqa: E402
    build_diff_sparse_v1_dataset,
    compute_v1_normalization_stats,
)
from models.diff_sparse_v1 import DiffSparseV1Model  # noqa: E402
from training.utils import set_seed  # noqa: E402


SCIENTIFIC_STATUS = "diff_sparse_v1_floodcastbench"
DOES_NOT_CLAIM = [
    "official FloodCastBench benchmark performance",
    "official DIFF-SPARSE TideWatch reproduction",
    "superiority over persistence or FNO+ until evaluated",
    "uncertainty calibration",
]

DEFAULT_PATHS = {
    "dataset_root": Path("/home/wissam/utem-workspace/data/FloodCastBench"),
    "experiment_root": Path("/home/wissam/utem-workspace/experiments/FloodCastBench"),
    "checkpoint_root": Path("/home/wissam/utem-workspace/checkpoints/FloodCastBench"),
    "log_root": Path("/home/wissam/utem-workspace/logs/FloodCastBench"),
}

METRIC_FIELDS = [
    "epoch",
    "train_loss",
    "train_x0_rmse",
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


def resolve_device(value: str) -> torch.device:
    value = str(value).lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


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


def cli_args_for_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


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
    if args.missing_rate is not None:
        config.setdefault("masking", {})["missing_rate"] = float(args.missing_rate)
    return config


def create_run_dirs(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    run_name = safe_name(config.get("experiment", {}).get("name", "fcb_diff_sparse_v1"))
    base_name = f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}_{run_name}"
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


def load_or_compute_stats(config: dict[str, Any], stats_json: Path | None) -> dict[str, Any]:
    if stats_json is not None:
        with Path(stats_json).open("r", encoding="utf-8") as file:
            stats = json.load(file)
        if stats.get("version") != "diff_sparse_v1_train_only_standardization":
            raise ValueError(f"Unexpected stats version in {stats_json}: {stats.get('version')!r}")
        return stats
    return compute_v1_normalization_stats(
        path_from_config(config, "dataset_root"),
        config,
        min_std=float(config.get("normalization", {}).get("min_std", 1e-6)),
    )


def build_loader(dataset, config: dict[str, Any], shuffle: bool, num_workers: int | None = None) -> DataLoader:
    loader_config = config.get("loader", {})
    resolved_num_workers = int(loader_config.get("num_workers", 0)) if num_workers is None else num_workers
    # Opt-in (loader.persistent_workers, default False -- every existing
    # config keeps PyTorch's own default, byte-identical behavior). With
    # workers >0 and this unset, DataLoader tears down and respawns worker
    # processes every epoch, discarding any in-process state (e.g.
    # datasets/floodcastbench_diff_sparse_v2_dataset.py's
    # cache_frames_in_memory) they built up -- diagnosed on the DGX Spark
    # after cache_frames_in_memory alone showed no epoch-to-epoch speedup:
    # the cache was being silently thrown away and rebuilt from scratch
    # every single epoch. persistent_workers=True keeps the worker
    # processes (and their caches) alive across epochs; only meaningful
    # when num_workers > 0, per PyTorch's own constraint.
    persistent_workers = bool(loader_config.get("persistent_workers", False)) and resolved_num_workers > 0
    return DataLoader(
        dataset,
        batch_size=int(loader_config.get("batch_size", 32)),
        shuffle=shuffle,
        num_workers=resolved_num_workers,
        pin_memory=bool(loader_config.get("pin_memory", False)),
        persistent_workers=persistent_workers,
    )


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def prepare_model_batch(batch: dict[str, Any], context_length: int) -> dict[str, torch.Tensor]:
    """Slice the window sample into the model's training contract (one-step target)."""

    return {
        "context_water_masked": batch["context_water_masked"],
        "sensor_mask": batch["sensor_mask"],
        "dem": batch["dem"],
        "rainfall_context": batch["rainfall"][:, :context_length],
        "timestamps_context": batch["timestamps"][:, :context_length],
        "target": batch["target"][:, 0:1],
    }


def run_epoch(
    model: DiffSparseV1Model,
    loader: DataLoader,
    device: torch.device,
    context_length: int,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
    grad_clip_norm: float | None = None,
) -> dict[str, float]:
    train = optimizer is not None
    model.train() if train else model.eval()
    loss_sum = 0.0
    rmse_sum = 0.0
    batches = 0
    grad_context = torch.enable_grad() if train else torch.no_grad()
    with grad_context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            model_batch = prepare_model_batch(batch, context_length)
            if train:
                optimizer.zero_grad(set_to_none=True)
            loss, diagnostics = model.training_step_loss(model_batch)
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


def run_deterministic_validation(
    model: DiffSparseV1Model,
    loader: DataLoader,
    device: torch.device,
    context_length: int,
    val_seed: int,
    max_batches: int | None,
) -> dict[str, float]:
    """Validation with fixed RNG so val_loss is comparable across epochs.

    The val loader must use num_workers=0: dataset-side randomness (crops, masks,
    noise masking) then runs in this process and is covered by the reseed. Global
    RNG state is saved and restored so training randomness is unaffected.
    """

    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(val_seed)
    try:
        return run_epoch(model, loader, device, context_length, optimizer=None, max_batches=max_batches)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def first_batch_shape_report(model: DiffSparseV1Model, batch: dict[str, Any], context_length: int) -> dict[str, Any]:
    model_batch = prepare_model_batch(batch, context_length)
    target = model_batch["target"]
    timesteps = torch.zeros(target.shape[0], dtype=torch.long, device=target.device)
    x_noisy = model.q_sample(target, timesteps)
    context_embedding = model.encode_context(model_batch)
    prediction = model.denoise(x_noisy, timesteps, context_embedding)
    return {
        "context_water_masked": list(model_batch["context_water_masked"].shape),
        "sensor_mask": list(model_batch["sensor_mask"].shape),
        "dem": list(model_batch["dem"].shape),
        "rainfall_context": list(model_batch["rainfall_context"].shape),
        "timestamps_context": list(model_batch["timestamps_context"].shape),
        "target": list(target.shape),
        "context_embedding": list(context_embedding.shape),
        "prediction": list(prediction.shape),
        "sensor_mask_mean": float(model_batch["sensor_mask"].mean().item()),
        "prediction_finite": bool(torch.isfinite(prediction).all().item()),
        "terminal_sqrt_alpha_cumprod": float(model.sqrt_alpha_cumprod[-1].item()),
    }


def dry_run(config: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    root = path_from_config(config, "dataset_root")
    train_dataset = build_diff_sparse_v1_dataset(root, config, split="train", normalization_stats=stats)
    val_dataset = build_diff_sparse_v1_dataset(
        root, config, split="val", normalization_stats=stats, patch_mode="random"
    )
    device = resolve_device(config.get("training", {}).get("device", "auto"))
    model = DiffSparseV1Model(config).to(device)
    loader = build_loader(train_dataset, config, shuffle=False, num_workers=0)
    batch = move_batch_to_device(next(iter(loader)), device)
    with torch.no_grad():
        shapes = first_batch_shape_report(model, batch, train_dataset.context_length)
        loss, diagnostics = model.training_step_loss(prepare_model_batch(batch, train_dataset.context_length))
    report = {
        "device": str(device),
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "context_length": train_dataset.context_length,
        "prediction_length": train_dataset.prediction_length,
        "patch_size": train_dataset.patch_size,
        "missing_rate": train_dataset.missing_rate,
        "mask_mode": train_dataset.mask_mode,
        "diffusion_steps": model.diffusion_steps,
        "beta_end": float(model.betas[-1].item()),
        "terminal_sqrt_alpha_cumprod": float(model.sqrt_alpha_cumprod[-1].item()),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "first_batch_shapes": shapes,
        "dry_run_loss": float(loss.item()),
        "dry_run_diagnostics": diagnostics,
        "normalization_channels": {name: stats["channels"][name] for name in ("water", "dem", "rainfall")},
        "writes": "none",
        "scientific_status": SCIENTIFIC_STATUS,
    }
    print("=== DIFF-SPARSE V1 DRY RUN ===")
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Train DIFF-SPARSE v1 on FloodCastBench.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--missing-rate", type=float)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--stats-json", type=Path, help="Reuse precomputed normalization stats")
    parser.add_argument("--device")
    parser.add_argument("--dry-run-config", action="store_true")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args)
    seed = int(config.get("training", {}).get("seed", config.get("experiment", {}).get("seed", 42)))
    set_seed(seed)

    print(f"code_root: {PROJECT_DIR}")
    print(f"config_path: {args.config}")
    print(f"dataset_root: {path_from_config(config, 'dataset_root')}")
    print(f"missing_rate: {config.get('masking', {}).get('missing_rate', 0.0)}")
    print(f"mask_mode: {config.get('masking', {}).get('mask_mode', 'noise')}")
    print(f"diffusion: steps={config.get('diffusion', {}).get('steps', 20)} "
          f"beta=[{config.get('diffusion', {}).get('beta_start', 1e-4)}, "
          f"{config.get('diffusion', {}).get('beta_end', 1.0)}]")

    stats = load_or_compute_stats(config, args.stats_json)

    if args.dry_run_config:
        dry_run(config, stats)
        return 0

    experiment_dir, checkpoint_dir, log_dir = create_run_dirs(config)
    print(f"experiment_dir: {experiment_dir}")
    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"log_dir: {log_dir}")
    save_yaml(config, experiment_dir / "config.yaml")
    save_json(stats, experiment_dir / "normalization_stats.json")

    root = path_from_config(config, "dataset_root")
    train_dataset = build_diff_sparse_v1_dataset(root, config, split="train", normalization_stats=stats)
    val_dataset = build_diff_sparse_v1_dataset(
        root, config, split="val", normalization_stats=stats, patch_mode="random"
    )
    train_loader = build_loader(train_dataset, config, shuffle=True)
    val_loader = build_loader(val_dataset, config, shuffle=False, num_workers=0)
    device = resolve_device(config.get("training", {}).get("device", "auto"))
    print(f"device: {device}")
    print(f"train_windows: {len(train_dataset)} val_windows: {len(val_dataset)}")

    model = DiffSparseV1Model(config).to(device)
    print(f"model_parameters: {sum(parameter.numel() for parameter in model.parameters())}")
    training_config = config.get("training", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training_config.get("lr_factor", 0.5)),
        patience=int(training_config.get("lr_patience", 3)),
    )
    val_seed = int(training_config.get("val_seed", 1234))
    grad_clip_norm = training_config.get("grad_clip_norm")
    epochs = int(training_config.get("epochs", 40))

    metrics_path = experiment_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=METRIC_FIELDS).writeheader()

    first_batch = move_batch_to_device(
        next(iter(build_loader(train_dataset, config, shuffle=False, num_workers=0))), device
    )
    with torch.no_grad():
        shapes = first_batch_shape_report(model, first_batch, train_dataset.context_length)
    print("first_batch_shapes:")
    print(json.dumps(shapes, indent=2))

    best_value = math.inf
    best_epoch = None
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            train_dataset.context_length,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
            grad_clip_norm=float(grad_clip_norm) if grad_clip_norm is not None else None,
        )
        val_metrics = run_deterministic_validation(
            model,
            val_loader,
            device,
            train_dataset.context_length,
            val_seed=val_seed,
            max_batches=args.max_val_batches,
        )
        scheduler.step(val_metrics["loss"])
        elapsed = time.perf_counter() - start
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_x0_rmse": train_metrics["x0_rmse"],
            "val_loss": val_metrics["loss"],
            "val_x0_rmse": val_metrics["x0_rmse"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": elapsed,
        }
        with metrics_path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=METRIC_FIELDS).writerow(row)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "normalization_stats": stats,
            "metrics": row,
            "selection_metric": "val_loss",
            "scientific_status": SCIENTIFIC_STATUS,
        }
        torch.save(checkpoint, checkpoint_dir / "checkpoint_last.pth")
        if float(row["val_loss"]) < best_value:
            best_value = float(row["val_loss"])
            best_epoch = epoch
            torch.save(checkpoint, checkpoint_dir / "checkpoint_best.pth")
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.6f} val_loss={row['val_loss']:.6f} "
            f"val_x0_rmse={row['val_x0_rmse']:.6f} lr={row['learning_rate']:.2e} "
            f"elapsed={elapsed:.1f}s"
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
            "missing_rate": float(config.get("masking", {}).get("missing_rate", 0.0)),
            "mask_mode": str(config.get("masking", {}).get("mask_mode", "noise")),
            "diffusion_beta_end": float(config.get("diffusion", {}).get("beta_end", 1.0)),
            "terminal_sqrt_alpha_cumprod": shapes["terminal_sqrt_alpha_cumprod"],
            "first_batch_shapes": shapes,
            "cli_args": cli_args_for_summary(args),
            "command_reconstruction": command_reconstruction(),
            "git_status_short": git_status_short(),
            "scientific_status": SCIENTIFIC_STATUS,
            "does_not_claim": DOES_NOT_CLAIM,
        },
        experiment_dir / "summary.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
