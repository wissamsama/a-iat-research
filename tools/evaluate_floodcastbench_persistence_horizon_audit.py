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

from datasets.floodcastbench_fno_plus_official_v1_dataset import (  # noqa: E402
    build_fno_plus_official_v1_dataset,
    compute_train_normalization_stats,
)
from training.utils import set_seed  # noqa: E402


WATER_DEPTH_CONTEXT_CHANNEL = 3
DEFAULT_PATHS = {
    "dataset_root": Path("/home/wissam/utem-workspace/data/FloodCastBench"),
    "experiment_root": Path("/home/wissam/utem-workspace/experiments/FloodCastBench"),
    "checkpoint_root": Path("/home/wissam/utem-workspace/checkpoints/FloodCastBench"),
    "log_root": Path("/home/wissam/utem-workspace/logs/FloodCastBench"),
}

HORIZON_FIELDS = [
    "horizon_index",
    "horizon_label",
    "num_batches_processed",
    "normalized_persistence_mse_mean",
    "normalized_persistence_rmse_mean",
    "normalized_persistence_mae_mean",
    "physical_persistence_mse_mean_if_available",
    "physical_persistence_rmse_mean_if_available",
    "physical_persistence_mae_mean_if_available",
    "count",
]

SAMPLE_FIELDS = [
    "split",
    "batch_index",
    "sample_index",
    "horizon_index",
    "horizon_label",
    "normalized_persistence_mse",
    "normalized_persistence_rmse",
    "normalized_persistence_mae",
    "physical_persistence_mse_if_available",
    "physical_persistence_rmse_if_available",
    "physical_persistence_mae_if_available",
    "count",
]

DOES_NOT_CLAIM = [
    "DIFF-SPARSE rollout evaluation",
    "official FloodCastBench benchmark performance",
    "physical-unit forecast skill",
    "sparse-sensor robustness",
    "long-horizon model validation",
    "uncertainty calibration",
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
        raise FileNotFoundError(f"No matching experiment directory found under {experiment_root}")
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
    return experiment_dir / f"persistence_horizon_audit_{timestamp}"


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


def load_or_compute_normalization_stats(config: dict[str, Any], experiment_dir: Path) -> tuple[dict[str, Any], str]:
    stats_path = experiment_dir / "normalization_stats.json"
    if stats_path.exists():
        return load_json(stats_path), str(stats_path)
    stats = compute_train_normalization_stats(
        path_from_config(config, "dataset_root"),
        config,
        min_std=float(config.get("normalization", {}).get("min_std", 1e-6)),
    )
    return stats, "computed_from_train_split_not_written"


def assert_dense_missing0_config(config: dict[str, Any]) -> None:
    masking = config.get("masking", {})
    missing_rate = float(masking.get("missing_rate", 0.0))
    mask_mode = str(masking.get("mask_mode", "all_ones")).lower()
    if missing_rate != 0.0:
        raise ValueError(f"Expected missing_rate=0.0, got {missing_rate}")
    if mask_mode != "all_ones":
        raise ValueError(f"Expected mask_mode='all_ones', got {mask_mode!r}")


def build_loader(dataset, config: dict[str, Any]) -> DataLoader:
    loader_config = config.get("loader", {})
    return DataLoader(
        dataset,
        batch_size=int(loader_config.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(loader_config.get("num_workers", 0)),
        pin_memory=bool(loader_config.get("pin_memory", False)),
    )


def move_batch_to_cpu(batch: Any) -> tuple[torch.Tensor, torch.Tensor, Any]:
    context, target_sequence, meta = batch
    return context.cpu(), target_sequence.cpu(), meta


def tensor_shape(value: torch.Tensor) -> list[int]:
    return [int(dim) for dim in value.shape]


def horizon_label(horizon_index: int) -> str:
    return f"h{horizon_index + 2:02d}"


def stats_audit(normalization_stats: dict[str, Any]) -> dict[str, Any]:
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
    }


def extract_h1_target_norm(context: torch.Tensor, normalization_stats: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if context.ndim != 5:
        raise ValueError(f"Expected context [B, 6, H, W, 20], got {tuple(context.shape)}")
    if context.shape[1] <= WATER_DEPTH_CONTEXT_CHANNEL:
        raise ValueError(f"Expected water-depth channel index 3 in context, got shape {tuple(context.shape)}")
    h1_initial_norm = context[:, WATER_DEPTH_CONTEXT_CHANNEL : WATER_DEPTH_CONTEXT_CHANNEL + 1, :, :, 0].contiguous()
    channels = normalization_stats["channels"]
    initial_stats = channels["initial_depth"]
    target_stats = channels["target_depth"]
    h1_physical = h1_initial_norm * float(initial_stats["std"]) + float(initial_stats["mean"])
    h1_target_norm = ((h1_physical - float(target_stats["mean"])) / float(target_stats["std"])).contiguous()
    return h1_target_norm, h1_physical.contiguous()


def target_physical(target_norm: torch.Tensor, normalization_stats: dict[str, Any]) -> torch.Tensor:
    stats = normalization_stats["channels"]["target_depth"]
    return target_norm * float(stats["std"]) + float(stats["mean"])


def empty_totals() -> dict[str, float]:
    return {"sq_error": 0.0, "abs_error": 0.0, "count": 0.0}


def update_totals(totals: dict[str, float], prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction shape {tuple(prediction.shape)} does not match target {tuple(target.shape)}")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise FloatingPointError("Non-finite prediction or target in persistence horizon audit")
    diff = prediction - target
    sq_error = float(diff.square().sum().item())
    abs_error = float(diff.abs().sum().item())
    count = float(diff.numel())
    totals["sq_error"] += sq_error
    totals["abs_error"] += abs_error
    totals["count"] += count
    mse = sq_error / count
    return {
        "mse": float(mse),
        "rmse": float(math.sqrt(max(mse, 0.0))),
        "mae": float(abs_error / count),
        "count": int(count),
    }


def finalize_totals(totals: dict[str, float]) -> dict[str, float]:
    if totals["count"] <= 0:
        raise RuntimeError("Cannot finalize metrics from zero pixels")
    mse = totals["sq_error"] / totals["count"]
    return {
        "mse": float(mse),
        "rmse": float(math.sqrt(max(mse, 0.0))),
        "mae": float(totals["abs_error"] / totals["count"]),
        "count": int(totals["count"]),
    }


def save_map(path: Path, array: torch.Tensor, title: str, cmap: str = "viridis", vmin: float | None = None, vmax: float | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = array.detach().float().cpu().squeeze().numpy()
    fig, axis = plt.subplots(figsize=(6, 5))
    artist = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title)
    axis.axis("off")
    fig.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def maybe_save_maps(
    maps_dir: Path,
    batch_index: int,
    horizon_index: int,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> list[str]:
    if batch_index != 0 or horizon_index not in (0, 8, 18):
        return []
    label = horizon_label(horizon_index)
    target_map = target[0]
    prediction_map = prediction[0]
    error_map = (prediction_map - target_map).abs()
    scale_values = torch.cat([target_map.flatten(), prediction_map.flatten()])
    vmin = float(scale_values.min().item())
    vmax = float(scale_values.max().item())
    files = [
        (maps_dir / f"batch000_{label}_target.png", target_map, f"{label} normalized target", "viridis", vmin, vmax),
        (
            maps_dir / f"batch000_{label}_persistence.png",
            prediction_map,
            f"{label} normalized persistence",
            "viridis",
            vmin,
            vmax,
        ),
        (maps_dir / f"batch000_{label}_abs_error.png", error_map, f"{label} normalized abs error", "magma", None, None),
    ]
    saved = []
    for path, array, title, cmap, map_vmin, map_vmax in files:
        save_map(path, array, title, cmap=cmap, vmin=map_vmin, vmax=map_vmax)
        saved.append(str(path))
    return saved


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def selected_horizon_summary(rows_by_horizon: list[dict[str, Any]]) -> dict[str, Any]:
    by_label = {row["horizon_label"]: row for row in rows_by_horizon}
    selected = {}
    for label in ("h02", "h05", "h10", "h15", "h20"):
        row = by_label[label]
        selected[label] = {
            "normalized_persistence_rmse_mean": row["normalized_persistence_rmse_mean"],
            "normalized_persistence_mae_mean": row["normalized_persistence_mae_mean"],
            "physical_persistence_rmse_mean_if_available": row["physical_persistence_rmse_mean_if_available"],
            "physical_persistence_mae_mean_if_available": row["physical_persistence_mae_mean_if_available"],
        }
    return selected


def evaluate(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    experiment_dir = latest_experiment_dir(config)
    normalization_stats, normalization_stats_source = load_or_compute_normalization_stats(config, experiment_dir)
    dataset = build_fno_plus_official_v1_dataset(
        path_from_config(config, "dataset_root"),
        config,
        split=args.split,
        normalization_stats=normalization_stats,
    )
    loader = build_loader(dataset, config)
    output_base = args.output_dir if args.output_dir is not None else default_output_dir(experiment_dir)
    output_dir = unique_dir(resolve_path(output_base, PROJECT_DIR))
    maps_dir = output_dir / "maps"
    if args.save_maps:
        maps_dir.mkdir(parents=True, exist_ok=False)

    horizon_count = 19
    normalized_totals = [empty_totals() for _ in range(horizon_count)]
    physical_totals = [empty_totals() for _ in range(horizon_count)]
    per_sample_rows: list[dict[str, Any]] = []
    map_files: list[str] = []
    first_batch_shapes: dict[str, Any] | None = None
    processed_batches = 0

    for batch_index, batch in enumerate(loader):
        if args.num_batches is not None and batch_index >= args.num_batches:
            break
        context, target_sequence, _meta = move_batch_to_cpu(batch)
        if target_sequence.ndim != 5:
            raise ValueError(f"Expected full target [B, 1, H, W, 19], got {tuple(target_sequence.shape)}")
        if target_sequence.shape[1] != 1 or target_sequence.shape[-1] != horizon_count:
            raise ValueError(f"Expected target channel=1 and 19 horizons, got {tuple(target_sequence.shape)}")
        h1_target_norm, h1_physical = extract_h1_target_norm(context, normalization_stats)

        if first_batch_shapes is None:
            first_target = target_sequence[..., 0].contiguous()
            first_batch_shapes = {
                "context": tensor_shape(context),
                "full_target_sequence": tensor_shape(target_sequence),
                "h1_target_norm": tensor_shape(h1_target_norm),
                "h1_physical": tensor_shape(h1_physical),
                "target_horizon": tensor_shape(first_target),
                "h1_finite": bool(torch.isfinite(h1_target_norm).all().item()),
                "target_sequence_finite": bool(torch.isfinite(target_sequence).all().item()),
            }

        for horizon_index in range(horizon_count):
            target_norm = target_sequence[..., horizon_index].contiguous()
            if h1_target_norm.shape != target_norm.shape:
                raise ValueError(
                    f"h1_target_norm shape {tuple(h1_target_norm.shape)} does not match horizon target "
                    f"{tuple(target_norm.shape)}"
                )
            target_phys = target_physical(target_norm, normalization_stats)
            normal = update_totals(normalized_totals[horizon_index], h1_target_norm, target_norm)
            physical = update_totals(physical_totals[horizon_index], h1_physical, target_phys)
            per_sample_rows.append(
                {
                    "split": args.split,
                    "batch_index": batch_index,
                    "sample_index": 0,
                    "horizon_index": horizon_index,
                    "horizon_label": horizon_label(horizon_index),
                    "normalized_persistence_mse": normal["mse"],
                    "normalized_persistence_rmse": normal["rmse"],
                    "normalized_persistence_mae": normal["mae"],
                    "physical_persistence_mse_if_available": physical["mse"],
                    "physical_persistence_rmse_if_available": physical["rmse"],
                    "physical_persistence_mae_if_available": physical["mae"],
                    "count": normal["count"],
                }
            )
            if args.save_maps:
                map_files.extend(maybe_save_maps(maps_dir, batch_index, horizon_index, h1_target_norm, target_norm))

        processed_batches += 1

    if processed_batches == 0:
        raise RuntimeError(f"No batches processed for split={args.split!r}")

    rows_by_horizon: list[dict[str, Any]] = []
    for horizon_index in range(horizon_count):
        normal = finalize_totals(normalized_totals[horizon_index])
        physical = finalize_totals(physical_totals[horizon_index])
        rows_by_horizon.append(
            {
                "horizon_index": horizon_index,
                "horizon_label": horizon_label(horizon_index),
                "num_batches_processed": processed_batches,
                "normalized_persistence_mse_mean": normal["mse"],
                "normalized_persistence_rmse_mean": normal["rmse"],
                "normalized_persistence_mae_mean": normal["mae"],
                "physical_persistence_mse_mean_if_available": physical["mse"],
                "physical_persistence_rmse_mean_if_available": physical["rmse"],
                "physical_persistence_mae_mean_if_available": physical["mae"],
                "count": normal["count"],
            }
        )

    by_horizon_path = output_dir / "persistence_horizon_metrics_by_horizon.csv"
    per_sample_path = output_dir / "persistence_horizon_metrics_per_sample.csv"
    write_csv(by_horizon_path, rows_by_horizon, HORIZON_FIELDS)
    write_csv(per_sample_path, per_sample_rows, SAMPLE_FIELDS)

    best = min(rows_by_horizon, key=lambda row: float(row["normalized_persistence_rmse_mean"]))
    worst = max(rows_by_horizon, key=lambda row: float(row["normalized_persistence_rmse_mean"]))
    average_rmse = sum(float(row["normalized_persistence_rmse_mean"]) for row in rows_by_horizon) / len(rows_by_horizon)
    summary = {
        "config_path": str(args.config),
        "split": args.split,
        "num_batches_requested": args.num_batches,
        "num_batches_processed": processed_batches,
        "dataset_samples": len(dataset),
        "experiment_dir": str(experiment_dir),
        "output_dir": str(output_dir),
        "normalization_stats_source": normalization_stats_source,
        "normalization_audit": stats_audit(normalization_stats),
        "h1_extraction": "context[:, 3:4, :, :, 0]",
        "retargeting_formula": (
            "h1_physical = h1_initial_norm * initial_depth.std + initial_depth.mean; "
            "h1_target_norm = (h1_physical - target_depth.mean) / target_depth.std"
        ),
        "target_sequence": "official-v1 target h2:h20, expected [B, 1, H, W, 19]",
        "first_batch_shapes": first_batch_shapes,
        "best_horizon": {
            "horizon_label": best["horizon_label"],
            "normalized_persistence_rmse_mean": best["normalized_persistence_rmse_mean"],
        },
        "worst_horizon": {
            "horizon_label": worst["horizon_label"],
            "normalized_persistence_rmse_mean": worst["normalized_persistence_rmse_mean"],
        },
        "selected_horizons": selected_horizon_summary(rows_by_horizon),
        "average_normalized_persistence_rmse_h02_to_h20": average_rmse,
        "metrics_by_horizon_csv": str(by_horizon_path),
        "metrics_per_sample_csv": str(per_sample_path),
        "map_files": map_files,
        "metric_units": {
            "normalized": "target_depth_standardized_train_stats",
            "physical": "meters_diagnostic_not_official",
        },
        "scientific_status": "horizon_wise_persistence_difficulty_audit",
        "does_not_claim": DOES_NOT_CLAIM,
        "command_reconstruction": command_reconstruction(),
        "git_status_short": git_status_short(),
        "cli_args": cli_args_for_summary(args),
    }
    save_json(summary, output_dir / "persistence_horizon_summary.json")
    print("=== FLOODCASTBENCH HORIZON-WISE PERSISTENCE DIFFICULTY AUDIT ===")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit horizon-wise h1 persistence difficulty for FloodCastBench h2:h20.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--num-batches", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--save-maps", action="store_true")
    args = parser.parse_args()

    if args.num_batches is not None and args.num_batches < 1:
        raise ValueError("--num-batches must be >= 1 when provided")
    if not args.config.exists():
        raise FileNotFoundError(f"Config does not exist: {args.config}")
    config = load_config(args.config)
    assert_dense_missing0_config(config)
    seed = int(config.get("training", {}).get("seed", config.get("experiment", {}).get("seed", 42)))
    set_seed(seed)
    evaluate(config, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
