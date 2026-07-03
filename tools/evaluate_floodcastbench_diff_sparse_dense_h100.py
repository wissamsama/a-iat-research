from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from datasets.floodcastbench_diff_sparse_high_horizon_dataset import build_diff_sparse_high_horizon_dataset  # noqa: E402
from models.diff_sparse import DenseDiffSparseModel  # noqa: E402
from tools import evaluate_floodcastbench_diff_sparse_dense as base_eval  # noqa: E402
from training.utils import set_seed  # noqa: E402


SCIENTIFIC_STATUS = "dense_missing0_direct_h100_sampling_sanity"
CHECKPOINT_STATUS = "dense_missing0_direct_h100_sanity_baseline"
METRIC_UNITS = "normalized_h100_direct_sampling_sanity"
DOES_NOT_CLAIM = [
    "official FloodCastBench benchmark performance",
    "physical-unit forecast skill",
    "sparse-sensor robustness",
    "uncertainty calibration",
    "autoregressive rollout",
    "full sparse-sensor DIFF-SPARSE reproduction",
    "superiority over FNO+",
]


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint).__name__}")
    status = checkpoint.get("scientific_status")
    if status != CHECKPOINT_STATUS:
        raise ValueError(f"Checkpoint scientific_status is not compatible with h100 sampling: {status!r}")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing model_state_dict")
    if not isinstance(checkpoint.get("normalization_stats"), dict):
        raise KeyError("Checkpoint is missing normalization_stats required to rebuild the h100 dataset")
    return checkpoint


def default_output_dir(config: dict[str, Any], checkpoint_path: Path) -> Path:
    experiment_root = base_eval.path_from_config(config, "experiment_root")
    run_dir = experiment_root / checkpoint_path.parent.name
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    return run_dir / f"eval_sampling_h100_{timestamp}"


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=base_eval.METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in base_eval.METRIC_FIELDS})


def persistence_baseline(config: dict[str, Any], split: str) -> dict[str, float | None]:
    baseline = config.get("evaluation", {}).get("persistence_h100_direct", {})
    return {
        "rmse": baseline.get(f"{split}_rmse"),
        "mae": baseline.get(f"{split}_mae"),
    }


def compare_to_persistence(metrics: dict[str, float], baseline: dict[str, float | None]) -> dict[str, float | None]:
    rmse = baseline.get("rmse")
    mae = baseline.get("mae")
    comparison: dict[str, float | None] = {
        "persistence_rmse": float(rmse) if rmse is not None else None,
        "persistence_mae": float(mae) if mae is not None else None,
        "diff_sparse_rmse": float(metrics["normalized_sample_rmse_mean"]),
        "diff_sparse_mae": float(metrics["normalized_sample_mae_mean"]),
        "rmse_improvement_percent_vs_persistence": None,
        "mae_improvement_percent_vs_persistence": None,
    }
    if rmse is not None and float(rmse) != 0.0:
        comparison["rmse_improvement_percent_vs_persistence"] = 100.0 * (
            float(rmse) - float(metrics["normalized_sample_rmse_mean"])
        ) / float(rmse)
    if mae is not None and float(mae) != 0.0:
        comparison["mae_improvement_percent_vs_persistence"] = 100.0 * (
            float(mae) - float(metrics["normalized_sample_mae_mean"])
        ) / float(mae)
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate direct h100 dense missing-rate-zero DIFF-SPARSE-style sampling sanity."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--save-maps", action="store_true")
    args = parser.parse_args()

    if args.num_batches < 1:
        raise ValueError("--num-batches must be >= 1")
    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")
    config = base_eval.load_config(args.config)
    base_eval.assert_dense_missing0_config(config)
    seed = int(config.get("training", {}).get("seed", config.get("experiment", {}).get("seed", 42)))
    set_seed(seed)

    checkpoint = load_checkpoint(args.checkpoint)
    stats = checkpoint["normalization_stats"]
    device = base_eval.resolve_device(args.device)
    dataset = build_diff_sparse_high_horizon_dataset(
        base_eval.path_from_config(config, "dataset_root"),
        config,
        split=args.split,
        normalization_stats=stats,
    )
    loader = base_eval.build_loader(dataset, config)
    model = DenseDiffSparseModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    output_base = args.output_dir if args.output_dir is not None else default_output_dir(config, args.checkpoint)
    output_dir = base_eval.unique_dir(base_eval.resolve_path(output_base, PROJECT_DIR))
    maps_dir = output_dir / "maps"
    if args.save_maps:
        maps_dir.mkdir(parents=True, exist_ok=False)

    print(f"code_root: {PROJECT_DIR}")
    print(f"config_path: {args.config}")
    print(f"checkpoint_path: {args.checkpoint}")
    print(f"run_dir: {base_eval.checkpoint_run_dir(config, args.checkpoint)}")
    print(f"output_dir: {output_dir}")
    print(f"split: {args.split}")
    print(f"num_batches: {args.num_batches}")
    print(f"num_samples: {args.num_samples}")
    print(f"device: {device}")
    print(f"target_horizon_label: {dataset.target_horizon_label}")
    print(f"eligible_samples: {len(dataset)}/{dataset.configured_sample_count}")
    print("sampling: gaussian_noise -> reverse_diffusion_steps -> predicted_h100_map")

    metric_rows: list[dict[str, Any]] = []
    map_files: list[str] = []
    map_value_ranges: dict[str, dict[str, dict[str, float]]] = {}
    first_batch_shapes: dict[str, Any] | None = None
    all_predictions_finite = True
    all_targets_finite = True
    mask_min = math.inf
    mask_max = -math.inf
    processed_batches = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.num_batches:
                break
            batch = base_eval.move_batch_to_device(batch, device)
            context = batch["context"]
            context_mask = batch["context_mask"]
            target = batch["target"]
            current_mask_min = float(context_mask.min().item())
            current_mask_max = float(context_mask.max().item())
            mask_min = min(mask_min, current_mask_min)
            mask_max = max(mask_max, current_mask_max)
            if abs(current_mask_min - 1.0) > 1e-6 or abs(current_mask_max - 1.0) > 1e-6:
                raise ValueError(f"Expected all-ones context_mask, got min={current_mask_min}, max={current_mask_max}")
            if not torch.isfinite(target).all():
                all_targets_finite = False
                raise FloatingPointError(f"Non-finite target values in batch {batch_index}")

            samples = []
            for sample_index in range(args.num_samples):
                prediction = base_eval.reverse_diffusion_sample(model, context, context_mask, target.shape)
                if not torch.isfinite(prediction).all():
                    all_predictions_finite = False
                    raise FloatingPointError(
                        f"Non-finite prediction values in batch {batch_index}, sample {sample_index}"
                    )
                samples.append(prediction)

            sample_stack = torch.stack(samples, dim=0)
            sample_std = sample_stack.std(dim=0, unbiased=False) if args.num_samples > 1 else torch.zeros_like(target)
            std_mean = float(sample_std.mean().item())
            std_max = float(sample_std.max().item())
            for sample_index, prediction in enumerate(samples):
                metric_rows.append(
                    {
                        "batch_index": batch_index,
                        "kind": "sample",
                        "sample_index": sample_index,
                        **base_eval.normalized_metrics(prediction, target),
                        "normalized_sample_std_mean": std_mean,
                        "normalized_sample_std_max": std_max,
                    }
                )

            mean_prediction = sample_stack.mean(dim=0)
            if args.num_samples > 1:
                metric_rows.append(
                    {
                        "batch_index": batch_index,
                        "kind": "sample_mean",
                        "sample_index": "mean",
                        **base_eval.normalized_metrics(mean_prediction, target),
                        "normalized_sample_std_mean": std_mean,
                        "normalized_sample_std_max": std_max,
                    }
                )

            if first_batch_shapes is None:
                first_batch_shapes = {
                    "context": base_eval.tensor_shape(context),
                    "context_mask": base_eval.tensor_shape(context_mask),
                    "target": base_eval.tensor_shape(target),
                    "sample_stack": base_eval.tensor_shape(sample_stack),
                    "sample_prediction": base_eval.tensor_shape(samples[0]),
                    "sample_mean_prediction": base_eval.tensor_shape(mean_prediction),
                    "context_mask_min": current_mask_min,
                    "context_mask_max": current_mask_max,
                    "prediction_finite": bool(torch.isfinite(samples[0]).all().item()),
                    "target_finite": bool(torch.isfinite(target).all().item()),
                }

            if args.save_maps and batch_index < 2:
                target_map = target[0]
                sample_maps = [sample[0] for sample in samples]
                sample_mean_map = mean_prediction[0]
                sample_std_map = sample_std[0]
                error_sample0 = (sample_maps[0] - target_map).abs()
                error_mean = (sample_mean_map - target_map).abs()
                batch_tag = f"h100_{args.split}_batch{batch_index:03d}"
                prediction_scale_values = torch.cat(
                    [target_map.flatten(), sample_maps[0].flatten(), sample_mean_map.flatten()]
                    + ([sample_maps[1].flatten()] if len(sample_maps) > 1 else [])
                )
                shared_vmin = float(prediction_scale_values.min().item())
                shared_vmax = float(prediction_scale_values.max().item())
                files: list[tuple[str, Path, torch.Tensor, str, str, float | None, float | None]] = [
                    ("target", maps_dir / f"{batch_tag}_target.png", target_map, "Normalized h100 target", "viridis", shared_vmin, shared_vmax),
                    ("sample000_prediction", maps_dir / f"{batch_tag}_sample000_prediction.png", sample_maps[0], "Normalized h100 sample000 prediction", "viridis", shared_vmin, shared_vmax),
                    ("abs_error_sample000", maps_dir / f"{batch_tag}_abs_error_sample000.png", error_sample0, "Normalized h100 sample000 absolute error", "magma", None, None),
                ]
                if len(sample_maps) > 1:
                    files.extend(
                        [
                            ("sample001_prediction", maps_dir / f"{batch_tag}_sample001_prediction.png", sample_maps[1], "Normalized h100 sample001 prediction", "viridis", shared_vmin, shared_vmax),
                            ("sample_mean_prediction", maps_dir / f"{batch_tag}_sample_mean_prediction.png", sample_mean_map, "Normalized h100 sample mean prediction", "viridis", shared_vmin, shared_vmax),
                            ("sample_std", maps_dir / f"{batch_tag}_sample_std.png", sample_std_map, "Normalized h100 sample standard deviation", "magma", None, None),
                            ("abs_error_mean", maps_dir / f"{batch_tag}_abs_error_mean.png", error_mean, "Normalized h100 sample mean absolute error", "magma", None, None),
                        ]
                    )
                batch_ranges: dict[str, dict[str, float]] = {}
                for range_key, path, array, title, cmap, vmin, vmax in files:
                    base_eval.save_map(path, array, title, cmap=cmap, vmin=vmin, vmax=vmax)
                    map_files.append(str(path))
                    batch_ranges[range_key] = base_eval.tensor_value_range(array)
                map_value_ranges[batch_tag] = batch_ranges

            processed_batches += 1

    if processed_batches == 0:
        raise RuntimeError(f"No batches were processed for split={args.split!r}")

    metrics_path = output_dir / "eval_metrics.csv"
    write_metrics_csv(metrics_path, metric_rows)
    aggregate_metrics = base_eval.aggregate_metric_rows(metric_rows)
    if not all(math.isfinite(value) for value in aggregate_metrics.values()):
        raise FloatingPointError(f"Non-finite aggregate metrics: {aggregate_metrics}")
    comparison = compare_to_persistence(aggregate_metrics, persistence_baseline(config, args.split))

    summary = {
        "config_path": str(args.config),
        "checkpoint_path": str(args.checkpoint),
        "split": args.split,
        "num_batches_requested": int(args.num_batches),
        "num_batches_processed": int(processed_batches),
        "num_samples": int(args.num_samples),
        "device": str(device),
        "command_reconstruction": base_eval.command_reconstruction(),
        "git_status_short": base_eval.git_status_short(),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_metrics": checkpoint.get("metrics"),
        "checkpoint_scientific_status": checkpoint.get("scientific_status"),
        "run_directory": str(base_eval.checkpoint_run_dir(config, args.checkpoint)),
        "output_directory": str(output_dir),
        "eval_metrics_csv": str(metrics_path),
        "map_files": map_files,
        "map_value_ranges": map_value_ranges,
        "first_batch_shapes": first_batch_shapes,
        "mask_min": mask_min,
        "mask_max": mask_max,
        "all_predictions_finite": bool(all_predictions_finite),
        "all_targets_finite": bool(all_targets_finite),
        "diffusion_steps": int(model.diffusion_steps),
        "prediction_type": str(model.prediction_type),
        "target_horizon_label": dataset.target_horizon_label,
        "target_horizon_index_from_h1": int(dataset.target_horizon_index_from_h1),
        "target_normalization_key": dataset.target_normalization_key,
        "eligible_sample_count": len(dataset),
        "configured_sample_count": dataset.configured_sample_count,
        "excluded_samples": dataset.excluded_samples,
        "test_subset_note": "test result is h100-capable subset only, N=10/14" if args.split == "test" else "",
        "metrics": aggregate_metrics,
        "comparison_against_h100_direct_persistence": comparison,
        "metric_units": METRIC_UNITS,
        "scientific_status": SCIENTIFIC_STATUS,
        "does_not_claim": DOES_NOT_CLAIM,
        "cli_args": base_eval.cli_args_for_summary(args),
    }
    base_eval.save_json(summary, output_dir / "eval_summary.json")
    print("=== DENSE DIFF-SPARSE H100 SAMPLING SANITY EVAL ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
