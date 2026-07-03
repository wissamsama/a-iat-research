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

from datasets.floodcastbench_diff_sparse_dataset import build_diff_sparse_dense_dataset  # noqa: E402
from models.diff_sparse import DenseDiffSparseModel  # noqa: E402
from training.utils import set_seed  # noqa: E402


DEFAULT_PATHS = {
    "dataset_root": Path("/home/wissam/utem-workspace/data/FloodCastBench"),
    "experiment_root": Path("/home/wissam/utem-workspace/experiments/FloodCastBench"),
    "checkpoint_root": Path("/home/wissam/utem-workspace/checkpoints/FloodCastBench"),
    "log_root": Path("/home/wissam/utem-workspace/logs/FloodCastBench"),
}

METRIC_FIELDS = [
    "batch_index",
    "kind",
    "sample_index",
    "normalized_sample_mse",
    "normalized_sample_rmse",
    "normalized_sample_mae",
    "normalized_sample_std_mean",
    "normalized_sample_std_max",
]

DOES_NOT_CLAIM = [
    "official FloodCastBench benchmark performance",
    "physical-unit forecast skill",
    "full sparse-sensor DIFF-SPARSE reproduction",
    "superiority over FNO+",
    "long-horizon performance",
    "uncertainty calibration",
]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def resolve_path(value: str | Path | None, fallback: Path) -> Path:
    selected = Path(value) if value not in (None, "") else fallback
    return selected if selected.is_absolute() else PROJECT_DIR / selected


def path_from_config(config: dict[str, Any], key: str) -> Path:
    return resolve_path(config.get("paths", {}).get(key), DEFAULT_PATHS[key])


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
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def unique_dir(path: Path) -> Path:
    candidate = path
    attempt = 1
    while candidate.exists():
        attempt += 1
        candidate = path.with_name(f"{path.name}_r{attempt}")
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def default_output_dir(config: dict[str, Any], checkpoint_path: Path) -> Path:
    experiment_root = path_from_config(config, "experiment_root")
    run_dir = experiment_root / checkpoint_path.parent.name
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    return run_dir / f"eval_sampling_{timestamp}"


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


def assert_dense_missing0_config(config: dict[str, Any]) -> None:
    masking = config.get("masking", {})
    missing_rate = float(masking.get("missing_rate", 0.0))
    mask_mode = str(masking.get("mask_mode", "all_ones")).lower()
    prediction_type = str(config.get("diffusion", {}).get("prediction_type", "x0")).lower()
    if missing_rate != 0.0:
        raise ValueError(f"Expected missing_rate=0.0 for dense sanity sampling, got {missing_rate}")
    if mask_mode != "all_ones":
        raise ValueError(f"Expected mask_mode='all_ones' for dense sanity sampling, got {mask_mode!r}")
    if prediction_type != "x0":
        raise ValueError(f"Expected diffusion.prediction_type='x0', got {prediction_type!r}")


def checkpoint_run_dir(config: dict[str, Any], checkpoint_path: Path) -> Path:
    return path_from_config(config, "experiment_root") / checkpoint_path.parent.name


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint).__name__}")
    status = checkpoint.get("scientific_status")
    if status != "dense_missing0_sanity_baseline":
        raise ValueError(
            "Checkpoint scientific_status is not compatible with dense missing0 sampling sanity: "
            f"{status!r}"
        )
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing model_state_dict")
    if not isinstance(checkpoint.get("normalization_stats"), dict):
        raise KeyError("Checkpoint is missing normalization_stats required to rebuild the dataset")
    return checkpoint


@torch.no_grad()
def reverse_diffusion_sample(
    model: DenseDiffSparseModel,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    target_shape: torch.Size,
) -> torch.Tensor:
    """Minimal DDPM-style x0-prediction reverse sampler for sanity evaluation.

    Sampling starts from Gaussian noise. At every reverse step, the model
    predicts x0, and the posterior mean is computed from the current x_t and
    predicted x0. This is intentionally minimal and is not an official
    DIFF-SPARSE benchmark sampler.
    """

    x_t = torch.randn(target_shape, device=context.device, dtype=context.dtype)
    betas = model.betas.to(device=context.device, dtype=context.dtype)
    alpha_bars = model.sqrt_alpha_cumprod.to(device=context.device, dtype=context.dtype).square()
    alphas = 1.0 - betas

    for step in reversed(range(model.diffusion_steps)):
        timesteps = torch.full((target_shape[0],), step, device=context.device, dtype=torch.long)
        x0_hat = model(x_t, timesteps, context, context_mask)
        if not torch.isfinite(x0_hat).all():
            raise FloatingPointError(f"Non-finite x0 prediction at reverse diffusion step {step}")

        beta_t = betas[step]
        alpha_t = alphas[step]
        alpha_bar_t = alpha_bars[step]
        alpha_bar_prev = torch.ones_like(alpha_bar_t) if step == 0 else alpha_bars[step - 1]
        denominator = torch.clamp(1.0 - alpha_bar_t, min=torch.finfo(context.dtype).eps)

        coef_x0 = beta_t * torch.sqrt(alpha_bar_prev) / denominator
        coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / denominator
        posterior_mean = coef_x0 * x0_hat + coef_xt * x_t
        posterior_variance = beta_t * (1.0 - alpha_bar_prev) / denominator

        if step > 0:
            noise = torch.randn_like(x_t)
            x_t = posterior_mean + torch.sqrt(torch.clamp(posterior_variance, min=0.0)) * noise
        else:
            x_t = posterior_mean
        if not torch.isfinite(x_t).all():
            raise FloatingPointError(f"Non-finite sampled tensor after reverse diffusion step {step}")

    return x_t


def normalized_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    diff = prediction - target
    mse = float(torch.mean(diff.square()).item())
    rmse = float(math.sqrt(max(mse, 0.0)))
    mae = float(torch.mean(diff.abs()).item())
    if not all(math.isfinite(value) for value in (mse, rmse, mae)):
        raise FloatingPointError(f"Non-finite normalized sampling metrics: mse={mse}, rmse={rmse}, mae={mae}")
    return {
        "normalized_sample_mse": mse,
        "normalized_sample_rmse": rmse,
        "normalized_sample_mae": mae,
    }


def tensor_value_range(array: torch.Tensor) -> dict[str, float]:
    values = array.detach().float()
    return {
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
    }


def save_map(
    path: Path,
    array: torch.Tensor,
    title: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
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


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in METRIC_FIELDS})


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    sample_rows = [row for row in rows if row["kind"] == "sample"]
    if not sample_rows:
        raise ValueError("No sample metric rows were produced")
    batch_std_mean = [float(row["normalized_sample_std_mean"]) for row in sample_rows]
    batch_std_max = [float(row["normalized_sample_std_max"]) for row in sample_rows]
    return {
        "normalized_sample_mse_mean": float(
            sum(float(row["normalized_sample_mse"]) for row in sample_rows) / len(sample_rows)
        ),
        "normalized_sample_rmse_mean": float(
            sum(float(row["normalized_sample_rmse"]) for row in sample_rows) / len(sample_rows)
        ),
        "normalized_sample_mae_mean": float(
            sum(float(row["normalized_sample_mae"]) for row in sample_rows) / len(sample_rows)
        ),
        "normalized_sample_std_mean": float(sum(batch_std_mean) / len(batch_std_mean)),
        "normalized_sample_std_max": float(max(batch_std_max)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate dense missing-rate-zero DIFF-SPARSE-style sampling sanity on FloodCastBench."
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
    if not args.config.exists():
        raise FileNotFoundError(f"Config does not exist: {args.config}")

    config = load_config(args.config)
    assert_dense_missing0_config(config)
    seed = int(config.get("training", {}).get("seed", config.get("experiment", {}).get("seed", 42)))
    set_seed(seed)

    checkpoint = load_checkpoint(args.checkpoint)
    stats = checkpoint["normalization_stats"]
    device = resolve_device(args.device)

    dataset = build_diff_sparse_dense_dataset(
        path_from_config(config, "dataset_root"),
        config,
        split=args.split,
        normalization_stats=stats,
    )
    loader = build_loader(dataset, config)
    model = DenseDiffSparseModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    output_base = args.output_dir if args.output_dir is not None else default_output_dir(config, args.checkpoint)
    output_dir = unique_dir(resolve_path(output_base, PROJECT_DIR))
    maps_dir = output_dir / "maps"
    if args.save_maps:
        maps_dir.mkdir(parents=True, exist_ok=False)

    print(f"code_root: {PROJECT_DIR}")
    print(f"config_path: {args.config}")
    print(f"checkpoint_path: {args.checkpoint}")
    print(f"run_dir: {checkpoint_run_dir(config, args.checkpoint)}")
    print(f"output_dir: {output_dir}")
    print(f"split: {args.split}")
    print(f"num_batches: {args.num_batches}")
    print(f"num_samples: {args.num_samples}")
    print(f"device: {device}")
    print("sampling: gaussian_noise -> reverse_diffusion_steps -> predicted_map")

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
            batch = move_batch_to_device(batch, device)
            context = batch["context"]
            context_mask = batch["context_mask"]
            target = batch["target"]

            current_mask_min = float(context_mask.min().item())
            current_mask_max = float(context_mask.max().item())
            mask_min = min(mask_min, current_mask_min)
            mask_max = max(mask_max, current_mask_max)
            if abs(current_mask_min - 1.0) > 1e-6 or abs(current_mask_max - 1.0) > 1e-6:
                raise ValueError(
                    "Dense missing0 sampling expected all-ones context_mask, "
                    f"got min={current_mask_min}, max={current_mask_max}"
                )
            if not torch.isfinite(target).all():
                all_targets_finite = False
                raise FloatingPointError(f"Non-finite target values in batch {batch_index}")

            samples = []
            for sample_index in range(args.num_samples):
                prediction = reverse_diffusion_sample(model, context, context_mask, target.shape)
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
                row = {
                    "batch_index": batch_index,
                    "kind": "sample",
                    "sample_index": sample_index,
                    **normalized_metrics(prediction, target),
                    "normalized_sample_std_mean": std_mean,
                    "normalized_sample_std_max": std_max,
                }
                metric_rows.append(row)

            mean_prediction = sample_stack.mean(dim=0)
            if args.num_samples > 1:
                metric_rows.append(
                    {
                        "batch_index": batch_index,
                        "kind": "sample_mean",
                        "sample_index": "mean",
                        **normalized_metrics(mean_prediction, target),
                        "normalized_sample_std_mean": std_mean,
                        "normalized_sample_std_max": std_max,
                    }
                )

            if first_batch_shapes is None:
                first_batch_shapes = {
                    "context": tensor_shape(context),
                    "context_mask": tensor_shape(context_mask),
                    "target": tensor_shape(target),
                    "sample_stack": tensor_shape(sample_stack),
                    "sample_prediction": tensor_shape(samples[0]),
                    "sample_mean_prediction": tensor_shape(mean_prediction),
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
                batch_tag = f"batch{batch_index:03d}"
                prediction_scale_values = torch.cat(
                    [target_map.flatten(), sample_maps[0].flatten(), sample_mean_map.flatten()]
                    + ([sample_maps[1].flatten()] if len(sample_maps) > 1 else [])
                )
                shared_vmin = float(prediction_scale_values.min().item())
                shared_vmax = float(prediction_scale_values.max().item())

                files: list[tuple[str, Path, torch.Tensor, str, str, float | None, float | None]] = [
                    (
                        "target",
                        maps_dir / f"{batch_tag}_target.png",
                        target_map,
                        "Normalized target",
                        "viridis",
                        shared_vmin,
                        shared_vmax,
                    ),
                    (
                        "sample000_prediction",
                        maps_dir / f"{batch_tag}_sample000_prediction.png",
                        sample_maps[0],
                        "Normalized sample000 prediction",
                        "viridis",
                        shared_vmin,
                        shared_vmax,
                    ),
                    (
                        "abs_error_sample000",
                        maps_dir / f"{batch_tag}_abs_error_sample000.png",
                        error_sample0,
                        "Normalized sample000 absolute error",
                        "magma",
                        None,
                        None,
                    ),
                ]

                if len(sample_maps) > 1:
                    files.extend(
                        [
                            (
                                "sample001_prediction",
                                maps_dir / f"{batch_tag}_sample001_prediction.png",
                                sample_maps[1],
                                "Normalized sample001 prediction",
                                "viridis",
                                shared_vmin,
                                shared_vmax,
                            ),
                            (
                                "sample_mean_prediction",
                                maps_dir / f"{batch_tag}_sample_mean_prediction.png",
                                sample_mean_map,
                                "Normalized sample mean prediction",
                                "viridis",
                                shared_vmin,
                                shared_vmax,
                            ),
                            (
                                "sample_std",
                                maps_dir / f"{batch_tag}_sample_std.png",
                                sample_std_map,
                                "Normalized sample standard deviation",
                                "magma",
                                None,
                                None,
                            ),
                            (
                                "abs_error_mean",
                                maps_dir / f"{batch_tag}_abs_error_mean.png",
                                error_mean,
                                "Normalized sample mean absolute error",
                                "magma",
                                None,
                                None,
                            ),
                        ]
                    )

                batch_ranges: dict[str, dict[str, float]] = {}
                for range_key, path, array, title, cmap, vmin, vmax in files:
                    save_map(path, array, title, cmap=cmap, vmin=vmin, vmax=vmax)
                    map_files.append(str(path))
                    batch_ranges[range_key] = tensor_value_range(array)
                map_value_ranges[batch_tag] = batch_ranges

            processed_batches += 1

    if processed_batches == 0:
        raise RuntimeError(f"No batches were processed for split={args.split!r}")

    metrics_path = output_dir / "eval_metrics.csv"
    write_metrics_csv(metrics_path, metric_rows)
    aggregate_metrics = aggregate_metric_rows(metric_rows)
    if not all(math.isfinite(value) for value in aggregate_metrics.values()):
        raise FloatingPointError(f"Non-finite aggregate metrics: {aggregate_metrics}")

    summary = {
        "config_path": str(args.config),
        "checkpoint_path": str(args.checkpoint),
        "split": args.split,
        "num_batches_requested": int(args.num_batches),
        "num_batches_processed": int(processed_batches),
        "num_samples": int(args.num_samples),
        "device": str(device),
        "command_reconstruction": command_reconstruction(),
        "git_status_short": git_status_short(),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_metrics": checkpoint.get("metrics"),
        "checkpoint_scientific_status": checkpoint.get("scientific_status"),
        "run_directory": str(checkpoint_run_dir(config, args.checkpoint)),
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
        "metrics": aggregate_metrics,
        "metric_units": "normalized_sampling_sanity",
        "scientific_status": "dense_missing0_sampling_sanity",
        "does_not_claim": DOES_NOT_CLAIM,
        "cli_args": cli_args_for_summary(args),
    }
    save_json(summary, output_dir / "eval_summary.json")

    print("=== DENSE DIFF-SPARSE SAMPLING SANITY EVAL ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
