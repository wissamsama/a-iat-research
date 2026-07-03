from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from datasets.floodcastbench_diff_sparse_high_horizon_dataset import (  # noqa: E402
    build_diff_sparse_high_horizon_dataset,
    compute_high_horizon_normalization_stats,
)
from models.diff_sparse import DenseDiffSparseModel  # noqa: E402
from tools import train_floodcastbench_diff_sparse_dense as base_train  # noqa: E402
from training.utils import set_seed  # noqa: E402


SCIENTIFIC_STATUS = "dense_missing0_direct_h100_sanity_baseline"
DOES_NOT_CLAIM = [
    "official FloodCastBench benchmark performance",
    "physical-unit forecast skill",
    "sparse-sensor robustness",
    "uncertainty calibration",
    "autoregressive rollout",
    "superiority over persistence or FNO+",
]


def build_datasets(config: dict[str, Any], stats: dict[str, Any]):
    root = base_train.path_from_config(config, "dataset_root")
    return (
        build_diff_sparse_high_horizon_dataset(root, config, split="train", normalization_stats=stats),
        build_diff_sparse_high_horizon_dataset(root, config, split="val", normalization_stats=stats),
    )


def dry_run(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    stats = compute_high_horizon_normalization_stats(
        base_train.path_from_config(config, "dataset_root"),
        config,
        min_std=float(config.get("normalization", {}).get("min_std", 1e-6)),
    )
    train_dataset, val_dataset = build_datasets(config, stats)
    train_loader = base_train.build_loader(train_dataset, config, shuffle=False)
    device = base_train.resolve_device(config["training"].get("device", "auto"))
    model = DenseDiffSparseModel(config).to(device)
    batch = base_train.move_batch_to_device(next(iter(train_loader)), device)
    with torch.no_grad():
        shapes = base_train.first_batch_shape_report(model, batch)
        loss, diagnostics = model.training_step_loss(batch)

    report = {
        "code_root": str(PROJECT_DIR),
        "dataset_root": str(base_train.path_from_config(config, "dataset_root")),
        "experiment_root": str(base_train.path_from_config(config, "experiment_root")),
        "checkpoint_root": str(base_train.path_from_config(config, "checkpoint_root")),
        "log_root": str(base_train.path_from_config(config, "log_root")),
        "device": str(device),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "target_horizon_label": config.get("dataset", {}).get("target_horizon_label", "h100"),
        "target_horizon_index_from_h1": int(config.get("dataset", {}).get("target_horizon_index_from_h1", 99)),
        "target_normalization_key": stats.get("target_normalization_key"),
        "target_normalization_stats": stats["channels"][stats["target_normalization_key"]],
        "context_normalization_version": "official_v1_train_only_standardization_for_initial_depth_dem_rainfall",
        "missing_rate": float(config.get("masking", {}).get("missing_rate", 0.0)),
        "diffusion_steps": int(config.get("diffusion", {}).get("steps", 20)),
        "prediction_type": str(config.get("diffusion", {}).get("prediction_type", "x0")),
        "first_batch_shapes": shapes,
        "dry_run_loss": float(loss.item()),
        "dry_run_diagnostics": diagnostics,
        "writes": "none",
        "scientific_status": SCIENTIFIC_STATUS,
    }
    print("=== DENSE DIFF-SPARSE H100 DRY RUN ===")
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train direct h100 dense missing-rate-zero DIFF-SPARSE-style FloodCastBench baseline."
    )
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

    config = base_train.apply_overrides(base_train.load_config(args.config), args)
    seed = int(config.get("training", {}).get("seed", config.get("experiment", {}).get("seed", 42)))
    set_seed(seed)

    print(f"code_root: {PROJECT_DIR}")
    print(f"config_path: {args.config}")
    print(f"dataset_root: {base_train.path_from_config(config, 'dataset_root')}")
    print(f"experiment_root: {base_train.path_from_config(config, 'experiment_root')}")
    print(f"checkpoint_root: {base_train.path_from_config(config, 'checkpoint_root')}")
    print(f"log_root: {base_train.path_from_config(config, 'log_root')}")
    print(f"target_horizon_label: {config.get('dataset', {}).get('target_horizon_label', 'h100')}")
    print(f"target_horizon_index_from_h1: {config.get('dataset', {}).get('target_horizon_index_from_h1', 99)}")
    print(f"target_normalization: {config.get('target_normalization', {}).get('key', 'target_depth_h100_direct')}")
    print(f"missing_rate: {config.get('masking', {}).get('missing_rate', 0.0)}")
    print(f"diffusion_steps: {config.get('diffusion', {}).get('steps', 20)}")
    print(f"prediction_type: {config.get('diffusion', {}).get('prediction_type', 'x0')}")

    if args.dry_run_config:
        dry_run(config, args)
        return 0

    experiment_dir, checkpoint_dir, log_dir = base_train.create_run_dirs(config)
    print(f"experiment_dir: {experiment_dir}")
    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"log_dir: {log_dir}")

    stats = compute_high_horizon_normalization_stats(
        base_train.path_from_config(config, "dataset_root"),
        config,
        min_std=float(config.get("normalization", {}).get("min_std", 1e-6)),
    )
    base_train.save_yaml(config, experiment_dir / "config.yaml")
    base_train.save_json(stats, experiment_dir / "normalization_stats.json")

    train_dataset, val_dataset = build_datasets(config, stats)
    train_loader = base_train.build_loader(train_dataset, config, shuffle=True)
    val_loader = base_train.build_loader(val_dataset, config, shuffle=False)
    device = base_train.resolve_device(config["training"].get("device", "auto"))
    print(f"device: {device}")
    print(f"train_eligible_samples: {len(train_dataset)}/{train_dataset.configured_sample_count}")
    print(f"val_eligible_samples: {len(val_dataset)}/{val_dataset.configured_sample_count}")

    model = DenseDiffSparseModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 1e-4)),
        weight_decay=float(config["training"].get("weight_decay", 1e-5)),
    )
    metrics_path = experiment_dir / "metrics.csv"
    base_train.write_header(metrics_path)

    first_batch = base_train.move_batch_to_device(
        next(iter(base_train.build_loader(train_dataset, config, shuffle=False))),
        device,
    )
    with torch.no_grad():
        shapes = base_train.first_batch_shape_report(model, first_batch)
    print("first_batch_shapes:")
    print(json.dumps(shapes, indent=2))

    best_value = math.inf
    best_epoch = None
    grad_clip_norm = config["training"].get("grad_clip_norm")
    epochs = int(config["training"].get("epochs", 20))
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_metrics = base_train.run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
            grad_clip_norm=float(grad_clip_norm) if grad_clip_norm is not None else None,
        )
        val_metrics = base_train.run_epoch(
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
        base_train.append_metrics(metrics_path, row)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "normalization_stats": stats,
            "metrics": row,
            "selection_metric": "val_loss",
            "target_horizon_label": config.get("dataset", {}).get("target_horizon_label", "h100"),
            "target_horizon_index_from_h1": int(config.get("dataset", {}).get("target_horizon_index_from_h1", 99)),
            "target_normalization_key": stats.get("target_normalization_key"),
            "metric_units": "normalized_h100_direct_training_sanity",
            "scientific_status": SCIENTIFIC_STATUS,
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

    base_train.save_json(
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
            "metric_units": "normalized_h100_direct_training_sanity",
            "target_horizon_label": config.get("dataset", {}).get("target_horizon_label", "h100"),
            "target_horizon_index_from_h1": int(config.get("dataset", {}).get("target_horizon_index_from_h1", 99)),
            "target_normalization_key": stats.get("target_normalization_key"),
            "target_normalization_stats": stats["channels"][stats["target_normalization_key"]],
            "context_normalization_version": "official_v1_train_only_standardization_for_initial_depth_dem_rainfall",
            "first_batch_shapes": shapes,
            "eligible_samples": {
                "train": {"eligible": len(train_dataset), "configured": train_dataset.configured_sample_count},
                "val": {"eligible": len(val_dataset), "configured": val_dataset.configured_sample_count},
            },
            "cli_args": base_train.cli_args_for_summary(args),
            "command_reconstruction": base_train.command_reconstruction(),
            "git_status_short": base_train.git_status_short(),
            "scientific_status": SCIENTIFIC_STATUS,
            "does_not_claim": DOES_NOT_CLAIM,
        },
        experiment_dir / "summary.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
