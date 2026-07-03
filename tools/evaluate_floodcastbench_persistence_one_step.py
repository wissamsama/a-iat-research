from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from datasets.floodcastbench_diff_sparse_dataset import (  # noqa: E402
    WATER_DEPTH_CONTEXT_CHANNEL,
    build_diff_sparse_dense_dataset,
)
from datasets.floodcastbench_fno_plus_official_v1_dataset import compute_train_normalization_stats  # noqa: E402
from training.utils import set_seed  # noqa: E402


DEFAULT_PATHS = {
    "dataset_root": Path("/home/wissam/utem-workspace/data/FloodCastBench"),
    "experiment_root": Path("/home/wissam/utem-workspace/experiments/FloodCastBench"),
    "checkpoint_root": Path("/home/wissam/utem-workspace/checkpoints/FloodCastBench"),
    "log_root": Path("/home/wissam/utem-workspace/logs/FloodCastBench"),
}

METRIC_FIELDS = [
    "split",
    "batch_index",
    "baseline",
    "batch_size",
    "normalized_persistence_mse",
    "normalized_persistence_rmse",
    "normalized_persistence_mae",
    "count",
]

DIFF_SPARSE_REFERENCE = {
    "val": {
        "normalized_sample_rmse_mean": 0.05467267563059386,
        "normalized_sample_mae_mean": 0.033615678415766785,
    },
    "test": {
        "normalized_sample_rmse_mean": 0.05897360276690722,
        "normalized_sample_mae_mean": 0.0359387162274548,
    },
}

DOES_NOT_CLAIM = [
    "official FloodCastBench benchmark performance",
    "physical-unit forecast skill",
    "sparse-sensor robustness",
    "long-horizon performance",
    "uncertainty calibration",
    "superiority over FNO+",
    "full sparse-sensor DIFF-SPARSE reproduction",
]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def resolve_path(value: str | Path | None, fallback: Path) -> Path:
    selected = Path(value) if value not in (None, "") else fallback
    return selected if selected.is_absolute() else PROJECT_DIR / selected


def path_from_config(config: dict[str, Any], key: str) -> Path:
    return resolve_path(config.get("paths", {}).get(key), DEFAULT_PATHS[key])


def run_suffix(config: dict[str, Any]) -> str:
    value = str(config.get("experiment", {}).get("name", "fcb_diff_sparse_dense_missing0_highfid_60m"))
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value).strip("_")


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
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def latest_experiment_dir(config: dict[str, Any]) -> Path:
    experiment_root = path_from_config(config, "experiment_root")
    suffix = run_suffix(config)
    candidates = sorted(
        [
            path
            for path in experiment_root.glob(f"*_{suffix}")
            if path.is_dir() and (path / "normalization_stats.json").exists()
        ],
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No experiment directory with normalization_stats.json found under {experiment_root} for suffix {suffix!r}"
        )
    return candidates[0]


def unique_dir(path: Path) -> Path:
    candidate = path
    attempt = 1
    while candidate.exists():
        attempt += 1
        candidate = path.with_name(f"{path.name}_r{attempt}")
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def default_output_dir(experiment_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    return experiment_dir / f"persistence_baseline_{timestamp}"


def assert_dense_missing0_config(config: dict[str, Any]) -> None:
    masking = config.get("masking", {})
    missing_rate = float(masking.get("missing_rate", 0.0))
    mask_mode = str(masking.get("mask_mode", "all_ones")).lower()
    if missing_rate != 0.0:
        raise ValueError(f"Expected missing_rate=0.0 for this persistence sanity baseline, got {missing_rate}")
    if mask_mode != "all_ones":
        raise ValueError(f"Expected mask_mode='all_ones', got {mask_mode!r}")


def load_or_compute_normalization_stats(
    config: dict[str, Any],
    experiment_dir: Path,
) -> tuple[dict[str, Any], str]:
    stats_path = experiment_dir / "normalization_stats.json"
    if stats_path.exists():
        return load_json(stats_path), str(stats_path)

    stats = compute_train_normalization_stats(
        path_from_config(config, "dataset_root"),
        config,
        min_std=float(config.get("normalization", {}).get("min_std", 1e-6)),
    )
    return stats, "computed_from_train_split_not_written"


def build_loader(dataset, config: dict[str, Any]) -> DataLoader:
    loader_config = config.get("loader", {})
    return DataLoader(
        dataset,
        batch_size=int(loader_config.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(loader_config.get("num_workers", 0)),
        pin_memory=bool(loader_config.get("pin_memory", False)),
    )


def move_batch_to_cpu(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.cpu() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def tensor_shape(value: torch.Tensor) -> list[int]:
    return [int(dim) for dim in value.shape]


def extract_h1_prediction(context: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if context.ndim != 5:
        raise ValueError(f"Expected context [B, 6, H, W, 20], got {tuple(context.shape)}")
    if context.shape[1] <= WATER_DEPTH_CONTEXT_CHANNEL:
        raise ValueError(
            f"Expected water-depth channel index {WATER_DEPTH_CONTEXT_CHANNEL}, got {context.shape[1]} channels"
        )
    if context.shape[-1] < 1:
        raise ValueError(f"Expected at least one context timestep, got {context.shape[-1]}")
    if target.ndim != 4:
        raise ValueError(f"Expected target [B, 1, H, W], got {tuple(target.shape)}")

    h1 = context[:, WATER_DEPTH_CONTEXT_CHANNEL : WATER_DEPTH_CONTEXT_CHANNEL + 1, :, :, 0].contiguous()
    if h1.shape != target.shape:
        raise ValueError(f"h1 prediction shape {tuple(h1.shape)} does not match target {tuple(target.shape)}")
    return h1


def normalization_audit(normalization_stats: dict[str, Any]) -> dict[str, Any]:
    channels = normalization_stats["channels"]
    initial = channels["initial_depth"]
    target = channels["target_depth"]
    keys = ("mean", "std", "min", "max")
    return {
        "initial_depth_stats_key": "channels.initial_depth",
        "target_depth_stats_key": "channels.target_depth",
        "initial_depth": {key: float(initial[key]) for key in keys},
        "target_depth": {key: float(target[key]) for key in keys},
        "stats_identical": all(float(initial[key]) == float(target[key]) for key in keys),
        "mean_delta_target_minus_initial": float(target["mean"]) - float(initial["mean"]),
        "std_delta_target_minus_initial": float(target["std"]) - float(initial["std"]),
    }


def retarget_h1_to_target_normalization(h1_initial_norm: torch.Tensor, normalization_stats: dict[str, Any]) -> torch.Tensor:
    channels = normalization_stats["channels"]
    initial_stats = channels["initial_depth"]
    target_stats = channels["target_depth"]
    h1_physical = h1_initial_norm * float(initial_stats["std"]) + float(initial_stats["mean"])
    return ((h1_physical - float(target_stats["mean"])) / float(target_stats["std"])).contiguous()


def batch_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Non-finite values found in h1 persistence prediction")
    if not torch.isfinite(target).all():
        raise FloatingPointError("Non-finite values found in h2 target")

    diff = prediction - target
    mse = float(diff.square().mean().item())
    rmse = float(math.sqrt(max(mse, 0.0)))
    mae = float(diff.abs().mean().item())
    if not all(math.isfinite(value) for value in (mse, rmse, mae)):
        raise FloatingPointError(f"Non-finite persistence metrics: mse={mse}, rmse={rmse}, mae={mae}")
    return {
        "normalized_persistence_mse": mse,
        "normalized_persistence_rmse": rmse,
        "normalized_persistence_mae": mae,
    }


def empty_totals() -> dict[str, float]:
    return {"sq_error": 0.0, "abs_error": 0.0, "count": 0.0}


def update_totals(totals: dict[str, float], prediction: torch.Tensor, target: torch.Tensor) -> None:
    diff = prediction - target
    totals["sq_error"] += float(diff.square().sum().item())
    totals["abs_error"] += float(diff.abs().sum().item())
    totals["count"] += float(diff.numel())


def finalize_totals(totals: dict[str, float]) -> dict[str, float]:
    count = totals["count"]
    if count <= 0:
        raise RuntimeError("Cannot finalize persistence metrics from zero pixels")
    mse = totals["sq_error"] / count
    metrics = {
        "normalized_persistence_mse": float(mse),
        "normalized_persistence_rmse": float(math.sqrt(max(mse, 0.0))),
        "normalized_persistence_mae": float(totals["abs_error"] / count),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError(f"Non-finite aggregate persistence metrics: {metrics}")
    return metrics


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in METRIC_FIELDS})


def diff_sparse_comparison(
    split: str,
    persistence_metrics_by_baseline: dict[str, dict[str, float]],
) -> dict[str, dict[str, float | None]]:
    reference = DIFF_SPARSE_REFERENCE.get(split)
    if reference is None:
        return {}

    diff_rmse = reference["normalized_sample_rmse_mean"]
    diff_mae = reference["normalized_sample_mae_mean"]
    comparison: dict[str, dict[str, float | None]] = {}
    for baseline, persistence_metrics in persistence_metrics_by_baseline.items():
        persistence_rmse = persistence_metrics["normalized_persistence_rmse"]
        persistence_mae = persistence_metrics["normalized_persistence_mae"]
        comparison[baseline] = {
            "diff_sparse_normalized_sample_rmse_mean": diff_rmse,
            "diff_sparse_normalized_sample_mae_mean": diff_mae,
            "persistence_normalized_rmse": persistence_rmse,
            "persistence_normalized_mae": persistence_mae,
            "diff_sparse_rmse_improvement_percent_vs_persistence": (
                100.0 * (persistence_rmse - diff_rmse) / persistence_rmse if persistence_rmse != 0 else None
            ),
            "diff_sparse_mae_improvement_percent_vs_persistence": (
                100.0 * (persistence_mae - diff_mae) / persistence_mae if persistence_mae != 0 else None
            ),
        }
    return comparison


def evaluate_split(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    experiment_dir = args.experiment_dir if args.experiment_dir is not None else latest_experiment_dir(config)
    experiment_dir = resolve_path(experiment_dir, PROJECT_DIR)
    normalization_stats, normalization_stats_source = load_or_compute_normalization_stats(config, experiment_dir)

    dataset = build_diff_sparse_dense_dataset(
        path_from_config(config, "dataset_root"),
        config,
        split=args.split,
        normalization_stats=normalization_stats,
    )
    loader = build_loader(dataset, config)
    output_base = args.output_dir if args.output_dir is not None else default_output_dir(experiment_dir)
    output_dir = unique_dir(resolve_path(output_base, PROJECT_DIR))
    audit = normalization_audit(normalization_stats)

    baselines = ("raw_context_normalized_persistence", "retargeted_persistence")
    totals = {baseline: empty_totals() for baseline in baselines}
    rows: list[dict[str, Any]] = []
    first_batch_shapes: dict[str, Any] | None = None
    processed_batches = 0
    mask_min = math.inf
    mask_max = -math.inf

    for batch_index, batch in enumerate(loader):
        if args.num_batches is not None and batch_index >= args.num_batches:
            break
        batch = move_batch_to_cpu(batch)
        context = batch["context"]
        target = batch["target"]
        context_mask = batch["context_mask"]
        raw_prediction = extract_h1_prediction(context, target)
        retargeted_prediction = retarget_h1_to_target_normalization(raw_prediction, normalization_stats)
        if retargeted_prediction.shape != target.shape:
            raise ValueError(
                f"Retargeted h1 shape {tuple(retargeted_prediction.shape)} does not match target {tuple(target.shape)}"
            )

        current_mask_min = float(context_mask.min().item())
        current_mask_max = float(context_mask.max().item())
        mask_min = min(mask_min, current_mask_min)
        mask_max = max(mask_max, current_mask_max)
        if abs(current_mask_min - 1.0) > 1e-6 or abs(current_mask_max - 1.0) > 1e-6:
            raise ValueError(
                f"Expected dense all-ones mask for split={args.split}, got min={current_mask_min}, max={current_mask_max}"
            )

        batch_predictions = {
            "raw_context_normalized_persistence": raw_prediction,
            "retargeted_persistence": retargeted_prediction,
        }
        for baseline, prediction in batch_predictions.items():
            metrics = batch_metrics(prediction, target)
            update_totals(totals[baseline], prediction, target)
            rows.append(
                {
                    "split": args.split,
                    "batch_index": batch_index,
                    "baseline": baseline,
                    "batch_size": int(target.shape[0]),
                    **metrics,
                    "count": int(target.numel()),
                }
            )
        if first_batch_shapes is None:
            first_batch_shapes = {
                "context": tensor_shape(context),
                "context_mask": tensor_shape(context_mask),
                "raw_h1_prediction": tensor_shape(raw_prediction),
                "retargeted_h1_prediction": tensor_shape(retargeted_prediction),
                "target_h2": tensor_shape(target),
                "context_mask_min": current_mask_min,
                "context_mask_max": current_mask_max,
                "raw_prediction_finite": bool(torch.isfinite(raw_prediction).all().item()),
                "retargeted_prediction_finite": bool(torch.isfinite(retargeted_prediction).all().item()),
                "target_finite": bool(torch.isfinite(target).all().item()),
            }
        processed_batches += 1

    if processed_batches == 0:
        raise RuntimeError(f"No samples were processed for split={args.split!r}")

    aggregate_metrics = {baseline: finalize_totals(totals[baseline]) for baseline in baselines}

    metrics_path = output_dir / "persistence_metrics.csv"
    write_metrics_csv(metrics_path, rows)
    comparison = diff_sparse_comparison(args.split, aggregate_metrics)
    summary = {
        "config_path": str(args.config),
        "split": args.split,
        "num_batches_requested": args.num_batches,
        "num_batches_processed": processed_batches,
        "dataset_samples": len(dataset),
        "experiment_dir": str(experiment_dir),
        "output_dir": str(output_dir),
        "normalization_stats_source": normalization_stats_source,
        "normalization_version": normalization_stats.get("version"),
        "normalization_audit": audit,
        "persistence_formulas": {
            "raw_context_normalized_persistence": "prediction = context[:, 3:4, :, :, 0]",
            "retargeted_persistence": (
                "h1_physical = h1_initial_norm * initial_depth.std + initial_depth.mean; "
                "prediction = (h1_physical - target_depth.mean) / target_depth.std"
            ),
        },
        "h1_extraction": "context[:, 3:4, :, :, 0]",
        "h1_channel_index": WATER_DEPTH_CONTEXT_CHANNEL,
        "target": "dataset wrapper one-step target h2, shape [B, 1, H, W]",
        "first_batch_shapes": first_batch_shapes,
        "mask_min": mask_min,
        "mask_max": mask_max,
        "metrics_by_baseline": aggregate_metrics,
        "metrics": aggregate_metrics["retargeted_persistence"],
        "diff_sparse_reference_comparison": comparison,
        "persistence_metrics_csv": str(metrics_path),
        "metric_units": "normalized_one_step_sanity",
        "scientific_status": "persistence_one_step_normalized_sanity_baseline",
        "does_not_claim": DOES_NOT_CLAIM,
        "command_reconstruction": command_reconstruction(),
        "git_status_short": git_status_short(),
        "cli_args": cli_args_for_summary(args),
    }
    save_json(summary, output_dir / "persistence_summary.json")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate one-step normalized persistence baseline h1 -> h2 for FloodCastBench."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--num-batches", type=int)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if args.num_batches is not None and args.num_batches < 1:
        raise ValueError("--num-batches must be >= 1 when provided")
    if not args.config.exists():
        raise FileNotFoundError(f"Config does not exist: {args.config}")

    config = load_config(args.config)
    assert_dense_missing0_config(config)
    seed = int(config.get("training", {}).get("seed", config.get("experiment", {}).get("seed", 42)))
    set_seed(seed)

    summary = evaluate_split(config, args)
    print("=== FLOODCASTBENCH ONE-STEP PERSISTENCE SANITY BASELINE ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
