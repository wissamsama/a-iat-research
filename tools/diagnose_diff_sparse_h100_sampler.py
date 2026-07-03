from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from datasets.floodcastbench_diff_sparse_high_horizon_dataset import build_diff_sparse_high_horizon_dataset  # noqa: E402
from models.diff_sparse import DenseDiffSparseModel  # noqa: E402
from tools import evaluate_floodcastbench_diff_sparse_dense as base_eval  # noqa: E402
from training.utils import set_seed  # noqa: E402


RUN_DIR = Path(
    "/home/wissam/utem-workspace/experiments/FloodCastBench/"
    "03-07-2026_11-02-57_fcb_diff_sparse_dense_missing0_highfid_60m_h100"
)
CHECKPOINT_STATUS = "dense_missing0_direct_h100_sanity_baseline"
SCIENTIFIC_STATUS = "dense_missing0_direct_h100_sampler_diagnostic"
TEACHER_TIMESTEPS = [0, 1, 5, 10, 15, 19]

TEACHER_FIELDS = [
    "split",
    "timestep",
    "batches",
    "normalized_mse",
    "normalized_rmse",
    "normalized_mae",
    "persistence_normalized_mse",
    "persistence_normalized_rmse",
    "persistence_normalized_mae",
    "persistence_config_rmse",
    "persistence_config_mae",
]

REVERSE_FIELDS = [
    "split",
    "variant",
    "batch_index",
    "sample_index",
    "kind",
    "normalized_mse",
    "normalized_rmse",
    "normalized_mae",
    "sample_std_mean",
    "sample_std_max",
    "prediction_min",
    "prediction_max",
    "prediction_mean",
    "prediction_std",
    "target_min",
    "target_max",
    "target_mean",
    "target_std",
    "all_steps_finite",
    "values_exploded",
]

TRAJECTORY_FIELDS = [
    "split",
    "batch_index",
    "variant",
    "reverse_index",
    "timestep",
    "x_t_min",
    "x_t_max",
    "x_t_mean",
    "x_t_std",
    "predicted_x0_min",
    "predicted_x0_max",
    "predicted_x0_mean",
    "predicted_x0_std",
    "predicted_x0_finite",
    "x_previous_min",
    "x_previous_max",
    "x_previous_mean",
    "x_previous_std",
    "x_previous_finite",
]

DOES_NOT_CLAIM = [
    "official FloodCastBench benchmark performance",
    "physical-unit forecast skill",
    "sparse-sensor robustness",
    "uncertainty calibration",
    "new training result",
    "superiority over persistence or FNO+",
]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


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


def unique_dir(path: Path) -> Path:
    candidate = path
    attempt = 1
    while candidate.exists():
        attempt += 1
        candidate = path.with_name(f"{path.name}_r{attempt}")
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def resolve_device(value: str) -> torch.device:
    value = str(value).lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint).__name__}")
    if checkpoint.get("scientific_status") != CHECKPOINT_STATUS:
        raise ValueError(f"Unexpected checkpoint scientific_status: {checkpoint.get('scientific_status')!r}")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing model_state_dict")
    if not isinstance(checkpoint.get("normalization_stats"), dict):
        raise KeyError("Checkpoint is missing normalization_stats")
    return checkpoint


def build_loader(dataset, config: dict[str, Any]) -> DataLoader:
    loader_config = config.get("loader", {})
    return DataLoader(
        dataset,
        batch_size=int(loader_config.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(loader_config.get("num_workers", 0)),
        pin_memory=bool(loader_config.get("pin_memory", False)),
    )


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def tensor_shape(value: torch.Tensor) -> list[int]:
    return [int(dim) for dim in value.shape]


def tensor_stats(value: torch.Tensor) -> dict[str, float]:
    values = value.detach().float()
    return {
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
    }


def update_error(acc: dict[str, float], prediction: torch.Tensor, target: torch.Tensor) -> None:
    diff = (prediction - target).detach().double()
    acc["sq_sum"] += float(diff.square().sum().item())
    acc["abs_sum"] += float(diff.abs().sum().item())
    acc["count"] += float(diff.numel())
    acc["batches"] += 1.0


def finalize_error(acc: dict[str, float]) -> dict[str, float]:
    count = acc["count"]
    if count <= 0:
        raise RuntimeError("Cannot finalize empty error accumulator")
    mse = acc["sq_sum"] / count
    return {
        "normalized_mse": float(mse),
        "normalized_rmse": float(math.sqrt(max(mse, 0.0))),
        "normalized_mae": float(acc["abs_sum"] / count),
        "batches": int(acc["batches"]),
    }


def normalized_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    diff = (prediction - target).detach().double()
    mse = float(diff.square().mean().item())
    return {
        "normalized_mse": mse,
        "normalized_rmse": float(math.sqrt(max(mse, 0.0))),
        "normalized_mae": float(diff.abs().mean().item()),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    accumulators: dict[str, dict[str, float]] = defaultdict(lambda: {"sq_sum": 0.0, "abs_sum": 0.0, "count": 0.0})
    std_values: dict[str, list[float]] = defaultdict(list)
    finite_values: dict[str, list[bool]] = defaultdict(list)
    explosion_values: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        if row["kind"] != "sample":
            continue
        variant = str(row["variant"])
        mse = float(row["normalized_mse"])
        mae = float(row["normalized_mae"])
        count = 1.0
        accumulators[variant]["sq_sum"] += mse * count
        accumulators[variant]["abs_sum"] += mae * count
        accumulators[variant]["count"] += count
        std_values[variant].append(float(row["sample_std_mean"]))
        finite_values[variant].append(bool(row["all_steps_finite"]))
        explosion_values[variant].append(bool(row["values_exploded"]))
    result: dict[str, dict[str, float]] = {}
    for variant, acc in accumulators.items():
        mse = acc["sq_sum"] / acc["count"]
        result[variant] = {
            "normalized_sample_mse_mean": float(mse),
            "normalized_sample_rmse_mean": float(math.sqrt(max(mse, 0.0))),
            "normalized_sample_mae_mean": float(acc["abs_sum"] / acc["count"]),
            "normalized_sample_std_mean": float(sum(std_values[variant]) / len(std_values[variant])),
            "all_steps_finite": bool(all(finite_values[variant])),
            "values_exploded": bool(any(explosion_values[variant])),
        }
    return result


def target_clip_range(stats: dict[str, Any]) -> dict[str, float]:
    target_key = stats["target_normalization_key"]
    target_stats = stats["channels"][target_key]
    mean = float(target_stats["mean"])
    std = float(target_stats["std"])
    physical_min = float(target_stats["min"])
    physical_max = float(target_stats["max"])
    return {
        "physical_min": physical_min,
        "physical_max": physical_max,
        "mean": mean,
        "std": std,
        "normalized_min": float((physical_min - mean) / std),
        "normalized_max": float((physical_max - mean) / std),
    }


def persistence_prediction(batch: dict[str, torch.Tensor], stats: dict[str, Any]) -> torch.Tensor:
    context = batch["context"]
    channels = stats["channels"]
    initial_stats = channels["initial_depth"]
    target_stats = channels[stats["target_normalization_key"]]
    h1_initial_norm = context[:, 3:4, :, :, 0]
    h1_physical = h1_initial_norm * float(initial_stats["std"]) + float(initial_stats["mean"])
    return (h1_physical - float(target_stats["mean"])) / float(target_stats["std"])


@torch.no_grad()
def teacher_forced_diagnostics(
    model: DenseDiffSparseModel,
    loader: DataLoader,
    stats: dict[str, Any],
    device: torch.device,
    split: str,
    num_batches: int,
    persistence_config: dict[str, float | None],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accumulators = {t: {"sq_sum": 0.0, "abs_sum": 0.0, "count": 0.0, "batches": 0.0} for t in TEACHER_TIMESTEPS}
    persistence_acc = {"sq_sum": 0.0, "abs_sum": 0.0, "count": 0.0, "batches": 0.0}
    first_batch_shapes: dict[str, Any] | None = None
    for batch_index, batch in enumerate(loader):
        if batch_index >= num_batches:
            break
        batch = move_batch_to_device(batch, device)
        target = batch["target"]
        context = batch["context"]
        context_mask = batch["context_mask"]
        persistence = persistence_prediction(batch, stats)
        update_error(persistence_acc, persistence, target)
        for timestep in TEACHER_TIMESTEPS:
            t = torch.full((target.shape[0],), timestep, device=device, dtype=torch.long)
            x_t = model.q_sample(target, t)
            pred = model(x_t, t, context, context_mask)
            if not torch.isfinite(pred).all():
                raise FloatingPointError(f"Non-finite teacher-forced prediction at timestep {timestep}")
            update_error(accumulators[timestep], pred, target)
        if first_batch_shapes is None:
            first_batch_shapes = {
                "context": tensor_shape(context),
                "context_mask": tensor_shape(context_mask),
                "target": tensor_shape(target),
                "persistence_prediction": tensor_shape(persistence),
                "context_mask_min": float(context_mask.min().item()),
                "context_mask_max": float(context_mask.max().item()),
            }
    persistence_metrics = finalize_error(persistence_acc)
    rows = []
    for timestep in TEACHER_TIMESTEPS:
        metrics = finalize_error(accumulators[timestep])
        rows.append(
            {
                "split": split,
                "timestep": timestep,
                "batches": metrics["batches"],
                "normalized_mse": metrics["normalized_mse"],
                "normalized_rmse": metrics["normalized_rmse"],
                "normalized_mae": metrics["normalized_mae"],
                "persistence_normalized_mse": persistence_metrics["normalized_mse"],
                "persistence_normalized_rmse": persistence_metrics["normalized_rmse"],
                "persistence_normalized_mae": persistence_metrics["normalized_mae"],
                "persistence_config_rmse": persistence_config.get("rmse"),
                "persistence_config_mae": persistence_config.get("mae"),
            }
        )
    return rows, {
        "first_batch_shapes": first_batch_shapes,
        "persistence_metrics_from_dataset": persistence_metrics,
        "persistence_metrics_from_config": persistence_config,
    }


@torch.no_grad()
def reverse_sample(
    model: DenseDiffSparseModel,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    target_shape: torch.Size,
    variant: str,
    initial_noise: torch.Tensor | None,
    clip_range: dict[str, float],
    trace: bool = False,
) -> tuple[torch.Tensor, list[dict[str, Any]], bool, bool]:
    x_t = torch.randn(target_shape, device=context.device, dtype=context.dtype) if initial_noise is None else initial_noise.clone()
    betas = model.betas.to(device=context.device, dtype=context.dtype)
    alpha_bars = model.sqrt_alpha_cumprod.to(device=context.device, dtype=context.dtype).square()
    alphas = 1.0 - betas
    rows: list[dict[str, Any]] = []
    all_finite = bool(torch.isfinite(x_t).all().item())
    values_exploded = False
    for reverse_index, step in enumerate(reversed(range(model.diffusion_steps))):
        x_t_before = x_t
        timesteps = torch.full((target_shape[0],), step, device=context.device, dtype=torch.long)
        x0_hat = model(x_t_before, timesteps, context, context_mask)
        if variant == "ddim_x0_clipped":
            x0_for_update = torch.clamp(
                x0_hat,
                min=float(clip_range["normalized_min"]),
                max=float(clip_range["normalized_max"]),
            )
        else:
            x0_for_update = x0_hat

        beta_t = betas[step]
        alpha_t = alphas[step]
        alpha_bar_t = alpha_bars[step]
        alpha_bar_prev = torch.ones_like(alpha_bar_t) if step == 0 else alpha_bars[step - 1]
        denominator = torch.clamp(1.0 - alpha_bar_t, min=torch.finfo(context.dtype).eps)

        if variant == "ddpm_current":
            coef_x0 = beta_t * torch.sqrt(alpha_bar_prev) / denominator
            coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / denominator
            posterior_mean = coef_x0 * x0_for_update + coef_xt * x_t_before
            posterior_variance = beta_t * (1.0 - alpha_bar_prev) / denominator
            if step > 0:
                x_previous = posterior_mean + torch.sqrt(torch.clamp(posterior_variance, min=0.0)) * torch.randn_like(x_t_before)
            else:
                x_previous = posterior_mean
        elif variant in {"ddim_eta0", "ddim_x0_clipped"}:
            eps_hat = (x_t_before - torch.sqrt(alpha_bar_t) * x0_for_update) / torch.sqrt(denominator)
            x_previous = torch.sqrt(alpha_bar_prev) * x0_for_update + torch.sqrt(1.0 - alpha_bar_prev) * eps_hat
        else:
            raise ValueError(f"Unsupported sampler variant: {variant}")

        predicted_finite = bool(torch.isfinite(x0_hat).all().item())
        previous_finite = bool(torch.isfinite(x_previous).all().item())
        all_finite = all_finite and predicted_finite and previous_finite
        max_abs = max(float(x0_hat.detach().abs().max().item()), float(x_previous.detach().abs().max().item()))
        values_exploded = values_exploded or max_abs > 1.0e4
        if trace:
            x_t_stats = tensor_stats(x_t_before)
            x0_stats = tensor_stats(x0_hat)
            prev_stats = tensor_stats(x_previous)
            rows.append(
                {
                    "variant": variant,
                    "reverse_index": reverse_index,
                    "timestep": step,
                    "x_t_min": x_t_stats["min"],
                    "x_t_max": x_t_stats["max"],
                    "x_t_mean": x_t_stats["mean"],
                    "x_t_std": x_t_stats["std"],
                    "predicted_x0_min": x0_stats["min"],
                    "predicted_x0_max": x0_stats["max"],
                    "predicted_x0_mean": x0_stats["mean"],
                    "predicted_x0_std": x0_stats["std"],
                    "predicted_x0_finite": predicted_finite,
                    "x_previous_min": prev_stats["min"],
                    "x_previous_max": prev_stats["max"],
                    "x_previous_mean": prev_stats["mean"],
                    "x_previous_std": prev_stats["std"],
                    "x_previous_finite": previous_finite,
                }
            )
        x_t = x_previous
    return x_t, rows, all_finite, values_exploded


def save_map(path: Path, array: torch.Tensor, title: str, cmap: str = "viridis", vmin: float | None = None, vmax: float | None = None) -> None:
    base_eval.save_map(path, array, title, cmap=cmap, vmin=vmin, vmax=vmax)


@torch.no_grad()
def reverse_sampling_diagnostics(
    model: DenseDiffSparseModel,
    loader: DataLoader,
    stats: dict[str, Any],
    device: torch.device,
    split: str,
    num_batches: int,
    num_samples: int,
    output_dir: Path,
    save_maps: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[str]]:
    clip_range = target_clip_range(stats)
    variants = ["ddpm_current", "ddim_eta0", "ddim_x0_clipped"]
    metric_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    map_files: list[str] = []
    maps_dir = output_dir / "maps"
    if save_maps:
        maps_dir.mkdir(parents=True, exist_ok=False)

    for batch_index, batch in enumerate(loader):
        if batch_index >= num_batches:
            break
        batch = move_batch_to_device(batch, device)
        context = batch["context"]
        context_mask = batch["context_mask"]
        target = batch["target"]
        persistence = persistence_prediction(batch, stats)
        target_stats = tensor_stats(target)
        if not torch.isfinite(target).all():
            raise FloatingPointError(f"Non-finite target in batch {batch_index}")

        predictions_for_maps: dict[str, torch.Tensor] = {}
        for variant in variants:
            samples = []
            finite_flags = []
            exploded_flags = []
            for sample_index in range(num_samples):
                initial_noise = torch.randn_like(target)
                prediction, trace_rows, all_finite, exploded = reverse_sample(
                    model,
                    context,
                    context_mask,
                    target.shape,
                    variant=variant,
                    initial_noise=initial_noise,
                    clip_range=clip_range,
                    trace=(batch_index == 0 and sample_index == 0 and variant == "ddpm_current"),
                )
                samples.append(prediction)
                finite_flags.append(all_finite)
                exploded_flags.append(exploded)
                if trace_rows:
                    for row in trace_rows:
                        row.update({"split": split, "batch_index": batch_index})
                    trajectory_rows.extend(trace_rows)

            sample_stack = torch.stack(samples, dim=0)
            sample_std = sample_stack.std(dim=0, unbiased=False) if num_samples > 1 else torch.zeros_like(target)
            std_mean = float(sample_std.mean().item())
            std_max = float(sample_std.max().item())
            sample_mean = sample_stack.mean(dim=0)
            predictions_for_maps[variant] = sample_mean[0].detach().cpu()
            for sample_index, prediction in enumerate(samples):
                metrics = normalized_metrics(prediction, target)
                pred_stats = tensor_stats(prediction)
                metric_rows.append(
                    {
                        "split": split,
                        "variant": variant,
                        "batch_index": batch_index,
                        "sample_index": sample_index,
                        "kind": "sample",
                        "normalized_mse": metrics["normalized_mse"],
                        "normalized_rmse": metrics["normalized_rmse"],
                        "normalized_mae": metrics["normalized_mae"],
                        "sample_std_mean": std_mean,
                        "sample_std_max": std_max,
                        "prediction_min": pred_stats["min"],
                        "prediction_max": pred_stats["max"],
                        "prediction_mean": pred_stats["mean"],
                        "prediction_std": pred_stats["std"],
                        "target_min": target_stats["min"],
                        "target_max": target_stats["max"],
                        "target_mean": target_stats["mean"],
                        "target_std": target_stats["std"],
                        "all_steps_finite": finite_flags[sample_index],
                        "values_exploded": exploded_flags[sample_index],
                    }
                )
            mean_metrics = normalized_metrics(sample_mean, target)
            mean_stats = tensor_stats(sample_mean)
            metric_rows.append(
                {
                    "split": split,
                    "variant": variant,
                    "batch_index": batch_index,
                    "sample_index": "mean",
                    "kind": "sample_mean",
                    "normalized_mse": mean_metrics["normalized_mse"],
                    "normalized_rmse": mean_metrics["normalized_rmse"],
                    "normalized_mae": mean_metrics["normalized_mae"],
                    "sample_std_mean": std_mean,
                    "sample_std_max": std_max,
                    "prediction_min": mean_stats["min"],
                    "prediction_max": mean_stats["max"],
                    "prediction_mean": mean_stats["mean"],
                    "prediction_std": mean_stats["std"],
                    "target_min": target_stats["min"],
                    "target_max": target_stats["max"],
                    "target_mean": target_stats["mean"],
                    "target_std": target_stats["std"],
                    "all_steps_finite": all(finite_flags),
                    "values_exploded": any(exploded_flags),
                }
            )

        if save_maps and batch_index == 0:
            target_map = target[0].detach().cpu()
            persistence_map = persistence[0].detach().cpu()
            all_values = torch.cat(
                [target_map.flatten(), persistence_map.flatten()]
                + [prediction.flatten() for prediction in predictions_for_maps.values()]
            )
            vmin = float(all_values.min().item())
            vmax = float(all_values.max().item())
            map_specs = [
                ("target", target_map, "h100 target", "viridis", vmin, vmax),
                ("persistence_h100", persistence_map, "h100 persistence", "viridis", vmin, vmax),
            ]
            for variant, prediction in predictions_for_maps.items():
                map_specs.append((f"{variant}_prediction", prediction, f"{variant} prediction", "viridis", vmin, vmax))
                map_specs.append((f"{variant}_abs_error", (prediction - target_map).abs(), f"{variant} abs error", "magma", None, None))
            for name, array, title, cmap, map_vmin, map_vmax in map_specs:
                path = maps_dir / f"batch000_{name}.png"
                save_map(path, array, title, cmap=cmap, vmin=map_vmin, vmax=map_vmax)
                map_files.append(str(path))

    summary = {
        "clip_range": clip_range,
        "num_samples": num_samples,
        "variants": variants,
        "aggregates_by_variant": aggregate_rows(metric_rows),
    }
    return metric_rows, trajectory_rows, summary, map_files


def checkpoint_config_consistency(
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    model: DenseDiffSparseModel,
    load_result: torch.nn.modules.module._IncompatibleKeys,
) -> dict[str, Any]:
    state_dict = checkpoint["model_state_dict"]
    finite_weights = all(torch.isfinite(value).all().item() for value in state_dict.values() if torch.is_tensor(value))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "diffusion_steps_config": int(config.get("diffusion", {}).get("steps", -1)),
        "diffusion_steps_model": int(model.diffusion_steps),
        "prediction_type_config": str(config.get("diffusion", {}).get("prediction_type")),
        "prediction_type_model": str(model.prediction_type),
        "beta_schedule_config": str(config.get("diffusion", {}).get("beta_schedule")),
        "beta_start_config": float(config.get("diffusion", {}).get("beta_start")),
        "beta_end_config": float(config.get("diffusion", {}).get("beta_end")),
        "target_horizon_label_config": config.get("dataset", {}).get("target_horizon_label"),
        "target_horizon_label_checkpoint": checkpoint.get("target_horizon_label"),
        "target_horizon_index_from_h1_config": int(config.get("dataset", {}).get("target_horizon_index_from_h1")),
        "target_horizon_index_from_h1_checkpoint": checkpoint.get("target_horizon_index_from_h1"),
        "target_normalization_key_checkpoint": checkpoint.get("target_normalization_key"),
        "target_normalization_stats": checkpoint["normalization_stats"]["channels"][
            checkpoint["normalization_stats"]["target_normalization_key"]
        ],
        "model_parameter_count": int(parameter_count),
        "all_checkpoint_weights_finite": bool(finite_weights),
        "state_dict_missing_keys": list(load_result.missing_keys),
        "state_dict_unexpected_keys": list(load_result.unexpected_keys),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_metrics": checkpoint.get("metrics"),
        "checkpoint_scientific_status": checkpoint.get("scientific_status"),
    }


def persistence_config(config: dict[str, Any], split: str) -> dict[str, float | None]:
    baseline = config.get("evaluation", {}).get("persistence_h100_direct", {})
    return {
        "rmse": baseline.get(f"{split}_rmse"),
        "mae": baseline.get(f"{split}_mae"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose direct h100 Dense DIFF-SPARSE sampler behavior.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--num-batches", type=int, required=True)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-maps", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("training", {}).get("seed", config.get("experiment", {}).get("seed", 42)))
    set_seed(seed)
    checkpoint = load_checkpoint(args.checkpoint)
    stats = checkpoint["normalization_stats"]
    device = resolve_device(args.device)
    num_samples = int(args.num_samples or config.get("evaluation", {}).get("num_samples", 2))

    dataset = build_diff_sparse_high_horizon_dataset(
        Path(config["paths"]["dataset_root"]),
        config,
        split=args.split,
        normalization_stats=stats,
    )
    loader = build_loader(dataset, config)
    model = DenseDiffSparseModel(config).to(device)
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    output_base = args.output_dir if args.output_dir else RUN_DIR / f"sampler_diagnostic_{args.split}_{timestamp}"
    output_dir = unique_dir(output_base if output_base.is_absolute() else PROJECT_DIR / output_base)
    print("=== DENSE DIFF-SPARSE H100 SAMPLER DIAGNOSTIC STARTED ===", flush=True)
    print(f"config_path: {args.config}", flush=True)
    print(f"checkpoint_path: {args.checkpoint}", flush=True)
    print(f"split: {args.split}", flush=True)
    print(f"num_batches: {args.num_batches}", flush=True)
    print(f"num_samples: {num_samples}", flush=True)
    print(f"device: {device}", flush=True)
    print(f"output_dir: {output_dir}", flush=True)

    consistency = checkpoint_config_consistency(config, checkpoint, model, load_result)
    teacher_rows, teacher_summary = teacher_forced_diagnostics(
        model,
        loader,
        stats,
        device,
        args.split,
        args.num_batches,
        persistence_config(config, args.split),
    )
    # Rebuild loader so reverse sampling starts at batch000 after teacher-forced diagnostics.
    loader = build_loader(dataset, config)
    reverse_rows, trajectory_rows, reverse_summary, map_files = reverse_sampling_diagnostics(
        model,
        loader,
        stats,
        device,
        args.split,
        args.num_batches,
        num_samples,
        output_dir,
        args.save_maps,
    )

    teacher_path = output_dir / "teacher_forced_metrics_by_timestep.csv"
    reverse_path = output_dir / "reverse_sampling_metrics.csv"
    trajectory_path = output_dir / "sampler_trajectory_batch000.csv"
    write_csv(teacher_path, teacher_rows, TEACHER_FIELDS)
    write_csv(reverse_path, reverse_rows, REVERSE_FIELDS)
    write_csv(trajectory_path, trajectory_rows, TRAJECTORY_FIELDS)

    summary = {
        "config_path": str(args.config),
        "checkpoint_path": str(args.checkpoint),
        "split": args.split,
        "num_batches_requested": int(args.num_batches),
        "num_batches_processed": min(int(args.num_batches), len(dataset)),
        "num_samples": int(num_samples),
        "device": str(device),
        "dataset_eligible_sample_count": len(dataset),
        "dataset_configured_sample_count": dataset.configured_sample_count,
        "excluded_samples": dataset.excluded_samples,
        "checkpoint_config_consistency": consistency,
        "teacher_forced": teacher_summary,
        "teacher_forced_metrics_by_timestep_csv": str(teacher_path),
        "reverse_sampling": reverse_summary,
        "reverse_sampling_metrics_csv": str(reverse_path),
        "sampler_trajectory_batch000_csv": str(trajectory_path),
        "map_files": map_files,
        "output_dir": str(output_dir),
        "metric_units": "normalized_h100_direct_sampler_diagnostic",
        "scientific_status": SCIENTIFIC_STATUS,
        "does_not_claim": DOES_NOT_CLAIM,
        "command_reconstruction": command_reconstruction(),
        "git_status_short": git_status_short(),
    }
    save_json(summary, output_dir / "sampler_diagnostic_summary.json")
    print("=== DENSE DIFF-SPARSE H100 SAMPLER DIAGNOSTIC ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
