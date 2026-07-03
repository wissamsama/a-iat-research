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

from datasets.floodcastbench_diff_sparse_v1_dataset import build_diff_sparse_v1_dataset  # noqa: E402
from models.diff_sparse_v1 import DiffSparseV1Model  # noqa: E402
from tools.evaluate_floodcastbench_fno_plus_official_v1_long_horizon_rollout import (  # noqa: E402
    MetricAccumulator as OfficialMetricAccumulator,
    gamma_key,
    write_csv as write_dynamic_csv,
)
from tools.train_floodcastbench_diff_sparse_v1 import (  # noqa: E402
    SCIENTIFIC_STATUS,
    cli_args_for_summary,
    command_reconstruction,
    git_status_short,
    load_config,
    path_from_config,
    resolve_device,
    resolve_path,
    save_json,
)
from training.utils import set_seed  # noqa: E402


EVAL_STATUS = "diff_sparse_v1_floodcastbench_rollout_eval"
DOES_NOT_CLAIM = [
    "official FloodCastBench benchmark performance",
    "official DIFF-SPARSE TideWatch reproduction",
    "uncertainty calibration",
]
OFFICIAL_GAMMAS = [0.001, 0.01]

STEP_FIELDS = [
    "split",
    "step",
    "horizon_label",
    "nrmse",
    "rmse_normalized",
    "mae_normalized",
    "rmse_physical_m",
    "mae_physical_m",
    "nacrps",
    "persistence_nrmse",
    "persistence_rmse_normalized",
    "persistence_mae_normalized",
    "persistence_rmse_physical_m",
]


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint).__name__}")
    if checkpoint.get("scientific_status") != SCIENTIFIC_STATUS:
        raise ValueError(
            f"Checkpoint scientific_status {checkpoint.get('scientific_status')!r} "
            f"is not a DIFF-SPARSE v1 checkpoint ({SCIENTIFIC_STATUS!r})"
        )
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing model_state_dict")
    if not isinstance(checkpoint.get("normalization_stats"), dict):
        raise KeyError("Checkpoint is missing normalization_stats")
    return checkpoint


def tile_positions(size: int, patch: int, stride: int | None = None) -> list[int]:
    if patch >= size:
        return [0]
    if stride is None:
        stride = patch
    stride = int(stride)
    if stride < 1:
        raise ValueError(f"tile stride must be >= 1, got {stride}")
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] + patch < size:
        starts.append(size - patch)
    return starts


def tile_blend_window(patch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Separable cell-centered Hann window for seam-suppressed tile blending.

    The window is strictly positive at tile edges because cell centers are used,
    so image-border pixels covered by a single tile remain well-defined.
    """

    if patch < 1:
        raise ValueError(f"patch must be >= 1, got {patch}")
    if patch == 1:
        return torch.ones(1, 1, device=device, dtype=dtype)
    coords = torch.arange(patch, device=device, dtype=dtype) + 0.5
    one_d = 0.5 - 0.5 * torch.cos(2.0 * math.pi * coords / float(patch))
    window = torch.outer(one_d, one_d)
    return window / torch.clamp(window.max(), min=torch.finfo(dtype).eps)


def to_physical(tensor: torch.Tensor, water_stats: dict[str, Any]) -> torch.Tensor:
    return tensor * float(water_stats["std"]) + float(water_stats["mean"])


class FinalHorizonPathAccumulator:
    def __init__(self, gammas: list[float]) -> None:
        self.gammas = tuple(float(gamma) for gamma in gammas)
        self.samples = 0
        self.counts = {
            gamma: {
                "tp": 0.0,
                "fp": 0.0,
                "fn": 0.0,
                "prop_tp": 0.0,
                "prop_fp": 0.0,
                "prop_fn": 0.0,
                "valid_propagation_steps": 0,
                "empty_true_propagation_steps": 0,
                "empty_predicted_propagation_steps": 0,
            }
            for gamma in self.gammas
        }

    def update(self, pred: torch.Tensor, target: torch.Tensor, initial: torch.Tensor) -> None:
        if pred.shape != target.shape:
            raise ValueError(f"pred and target path tensors must match, got {pred.shape} vs {target.shape}")
        if pred.ndim != 3:
            raise ValueError(f"Expected pred/target [L, H, W], got {tuple(pred.shape)}")
        if initial.shape != pred.shape[1:]:
            raise ValueError(f"Expected initial [H, W] matching pred, got {tuple(initial.shape)}")
        pred = pred.detach().float().cpu()
        target = target.detach().float().cpu()
        initial = initial.detach().float().cpu()
        self.samples += 1

        for gamma in self.gammas:
            counts = self.counts[gamma]
            initial_mask = initial > gamma
            pred_final = pred[-1] > gamma
            target_final = target[-1] > gamma
            pred_path = pred_final & ~initial_mask
            target_path = target_final & ~initial_mask
            counts["tp"] += float((pred_path & target_path).sum().item())
            counts["fp"] += float((pred_path & ~target_path).sum().item())
            counts["fn"] += float((~pred_path & target_path).sum().item())

            prev_pred = initial_mask
            prev_target = initial_mask
            for step in range(pred.shape[0]):
                pred_mask = pred[step] > gamma
                target_mask = target[step] > gamma
                pred_new = pred_mask & ~prev_pred
                target_new = target_mask & ~prev_target
                if int(pred_new.sum().item()) == 0:
                    counts["empty_predicted_propagation_steps"] += 1
                if int(target_new.sum().item()) == 0:
                    counts["empty_true_propagation_steps"] += 1
                if int(pred_new.sum().item()) > 0 or int(target_new.sum().item()) > 0:
                    counts["valid_propagation_steps"] += 1
                counts["prop_tp"] += float((pred_new & target_new).sum().item())
                counts["prop_fp"] += float((pred_new & ~target_new).sum().item())
                counts["prop_fn"] += float((~pred_new & target_new).sum().item())
                prev_pred = pred_mask
                prev_target = target_mask

    def compute(self, horizon_label: str, horizon_steps: int) -> dict[str, Any]:
        eps = 1e-12
        result: dict[str, Any] = {
            "horizon_label": horizon_label,
            "horizon_steps": horizon_steps,
            "samples": self.samples,
        }
        for gamma, counts in self.counts.items():
            key = gamma_key(gamma)
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]
            prop_tp = counts["prop_tp"]
            prop_fp = counts["prop_fp"]
            prop_fn = counts["prop_fn"]
            result.update(
                {
                    f"path_iou_gamma_{key}": tp / max(tp + fp + fn, eps),
                    f"path_tp_gamma_{key}": int(tp),
                    f"path_fp_gamma_{key}": int(fp),
                    f"path_fn_gamma_{key}": int(fn),
                    f"propagation_path_iou_gamma_{key}": prop_tp / max(prop_tp + prop_fp + prop_fn, eps),
                    f"propagation_path_tp_gamma_{key}": int(prop_tp),
                    f"propagation_path_fp_gamma_{key}": int(prop_fp),
                    f"propagation_path_fn_gamma_{key}": int(prop_fn),
                    f"valid_propagation_steps_gamma_{key}": counts["valid_propagation_steps"],
                    f"empty_true_propagation_steps_gamma_{key}": counts["empty_true_propagation_steps"],
                    f"empty_predicted_propagation_steps_gamma_{key}": counts[
                        "empty_predicted_propagation_steps"
                    ],
                }
            )
        return result


def unique_dir(path: Path) -> Path:
    candidate = path
    attempt = 1
    while candidate.exists():
        attempt += 1
        candidate = path.with_name(f"{path.name}_r{attempt}")
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


@torch.no_grad()
def rollout_window(
    model: DiffSparseV1Model,
    sample: dict[str, torch.Tensor],
    num_scenarios: int,
    patch_size: int,
    tile_stride: int,
    tile_chunk: int,
    remask: bool,
    mask_mode: str,
    clip_x0: tuple[float, float] | None,
    generator: torch.Generator | None,
    device: torch.device,
) -> torch.Tensor:
    """Autoregressive per-tile rollout (paper Algorithm 2 + inference section).

    Each 64x64 tile is rolled out independently for each scenario: sample the
    next frame from noise conditioned on the masked context, append it to the
    context (re-masked with the same static sensor mask when remask=True), and
    repeat. Overlapping tile regions are averaged. Returns [M, l, H, W].
    """

    context_masked = sample["context_water_masked"].to(device)
    sensor_mask = sample["sensor_mask"].to(device)
    dem = sample["dem"].to(device)
    rainfall = sample["rainfall"].to(device)
    timestamps = sample["timestamps"].to(device)
    context_length, height, width = context_masked.shape
    prediction_length = sample["target"].shape[0]

    ys = tile_positions(height, patch_size, tile_stride)
    xs = tile_positions(width, patch_size, tile_stride)
    tiles = [(y, x) for y in ys for x in xs]
    blend = tile_blend_window(patch_size, device=device, dtype=context_masked.dtype)

    output_sum = torch.zeros(num_scenarios, prediction_length, height, width, device=device)
    weight = torch.zeros(height, width, device=device)

    tiles_per_chunk = max(1, tile_chunk // max(num_scenarios, 1))
    for chunk_start in range(0, len(tiles), tiles_per_chunk):
        chunk = tiles[chunk_start : chunk_start + tiles_per_chunk]
        n_tiles = len(chunk)
        batch_size = n_tiles * num_scenarios

        def stack_tiles(tensor: torch.Tensor) -> torch.Tensor:
            crops = [tensor[..., y : y + patch_size, x : x + patch_size] for y, x in chunk]
            stacked = torch.stack(crops, dim=0)
            return stacked.repeat_interleave(num_scenarios, dim=0)

        context = stack_tiles(context_masked)
        mask = stack_tiles(sensor_mask)
        dem_tiles = stack_tiles(dem)
        rain_tiles = stack_tiles(rainfall)
        ts_batch = timestamps.unsqueeze(0).expand(batch_size, -1)

        for step in range(prediction_length):
            model_batch = {
                "context_water_masked": context,
                "sensor_mask": mask,
                "dem": dem_tiles,
                "rainfall_context": rain_tiles[:, step : step + context_length],
                "timestamps_context": ts_batch[:, step : step + context_length],
            }
            embedding = model.encode_context(model_batch)
            prediction = model.sample(
                embedding,
                (batch_size, 1, patch_size, patch_size),
                generator=generator,
                clip_x0=clip_x0,
            )
            for tile_index, (y, x) in enumerate(chunk):
                block = prediction[tile_index * num_scenarios : (tile_index + 1) * num_scenarios, 0]
                output_sum[:, step, y : y + patch_size, x : x + patch_size] += block * blend

            if remask:
                fill = torch.randn_like(prediction) if mask_mode == "noise" else torch.zeros_like(prediction)
                new_frame = prediction * mask + (1.0 - mask) * fill
            else:
                new_frame = prediction
            context = torch.cat([context[:, 1:], new_frame], dim=1)

        for y, x in chunk:
            weight[y : y + patch_size, x : x + patch_size] += blend

    return output_sum / weight.clamp(min=torch.finfo(output_sum.dtype).eps)


def persistence_forecast(
    sample: dict[str, torch.Tensor],
    prediction_length: int,
    device: torch.device,
    mode: str,
) -> torch.Tensor:
    """Persistence baseline in normalized space.

    oracle: dense last true context frame, the historical conservative baseline.
    sparse: observed cells from the last context frame, unobserved cells filled
    with the train water mean, which is 0.0 after standardization.
    """

    mode = str(mode).lower()
    last_true = sample["context_water_true"][-1].to(device)
    if mode == "oracle":
        base = last_true
    elif mode == "sparse":
        sensor_mask = sample["sensor_mask"][0].to(device)
        base = last_true * sensor_mask
    else:
        raise ValueError(f"Unsupported persistence mode {mode!r}; expected 'oracle' or 'sparse'")
    return base.unsqueeze(0).expand(prediction_length, -1, -1)


def crps_ensemble(samples: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
    """Empirical CRPS per pixel: mean|X-y| - (1/2M^2) sum_ij |X_i - X_j| (paper eq. 34)."""

    members = samples.shape[0]
    term1 = (samples - observation.unsqueeze(0)).abs().mean(dim=0)
    pairwise = (samples.unsqueeze(0) - samples.unsqueeze(1)).abs().sum(dim=(0, 1))
    return term1 - pairwise / (2.0 * members * members)


class MetricAccumulator:
    def __init__(self, prediction_length: int) -> None:
        self.length = prediction_length
        self.sq = torch.zeros(prediction_length, dtype=torch.float64)
        self.abs = torch.zeros(prediction_length, dtype=torch.float64)
        self.crps = torch.zeros(prediction_length, dtype=torch.float64)
        self.abs_obs = torch.zeros(prediction_length, dtype=torch.float64)
        self.count = torch.zeros(prediction_length, dtype=torch.float64)
        self.obs_min = math.inf
        self.obs_max = -math.inf

    def update(self, mean_forecast: torch.Tensor, samples: torch.Tensor | None, target: torch.Tensor) -> None:
        for step in range(self.length):
            observation = target[step]
            error = (mean_forecast[step] - observation).double()
            self.sq[step] += float(error.square().sum().item())
            self.abs[step] += float(error.abs().sum().item())
            self.abs_obs[step] += float(observation.double().abs().sum().item())
            self.count[step] += float(observation.numel())
            if samples is not None:
                self.crps[step] += float(crps_ensemble(samples[:, step], observation).double().sum().item())
            else:
                self.crps[step] += float(error.abs().sum().item())
        self.obs_min = min(self.obs_min, float(target.min().item()))
        self.obs_max = max(self.obs_max, float(target.max().item()))

    def finalize(self, water_std: float) -> dict[str, Any]:
        obs_range = max(self.obs_max - self.obs_min, 1e-12)
        per_step = []
        for step in range(self.length):
            count = max(float(self.count[step].item()), 1.0)
            mse = float(self.sq[step].item()) / count
            rmse = math.sqrt(max(mse, 0.0))
            mae = float(self.abs[step].item()) / count
            nacrps = float(self.crps[step].item()) / max(float(self.abs_obs[step].item()), 1e-12)
            per_step.append(
                {
                    "rmse_normalized": rmse,
                    "mae_normalized": mae,
                    "nrmse": rmse / obs_range,
                    "nacrps": nacrps,
                    "rmse_physical_m": rmse * water_std,
                    "mae_physical_m": mae * water_std,
                }
            )
        total_count = max(float(self.count.sum().item()), 1.0)
        overall_mse = float(self.sq.sum().item()) / total_count
        overall_rmse = math.sqrt(max(overall_mse, 0.0))
        overall_mae = float(self.abs.sum().item()) / total_count
        return {
            "per_step": per_step,
            "overall": {
                "rmse_normalized": overall_rmse,
                "mae_normalized": overall_mae,
                "nrmse": overall_rmse / obs_range,
                "nacrps": float(self.crps.sum().item()) / max(float(self.abs_obs.sum().item()), 1e-12),
                "rmse_physical_m": overall_rmse * water_std,
                "mae_physical_m": overall_mae * water_std,
                "observation_range_normalized": obs_range,
            },
        }


def save_maps(
    output_dir: Path,
    tag: str,
    target: torch.Tensor,
    mean_forecast: torch.Tensor,
    sample_std: torch.Tensor,
    persistence: torch.Tensor,
    step: int,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    maps_dir = output_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    entries = [
        ("target", target[step], "viridis"),
        ("mean_forecast", mean_forecast[step], "viridis"),
        ("abs_error", (mean_forecast[step] - target[step]).abs(), "magma"),
        ("sample_std", sample_std[step], "magma"),
        ("persistence_abs_error", (persistence[step] - target[step]).abs(), "magma"),
    ]
    shared = torch.cat([target[step].flatten(), mean_forecast[step].flatten()])
    vmin, vmax = float(shared.min().item()), float(shared.max().item())
    files = []
    for name, array, cmap in entries:
        figure, axis = plt.subplots(figsize=(6, 5))
        kwargs = {"vmin": vmin, "vmax": vmax} if cmap == "viridis" else {}
        artist = axis.imshow(array.cpu().numpy(), cmap=cmap, **kwargs)
        axis.set_title(f"{tag} step{step + 1} {name}")
        axis.axis("off")
        figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        path = maps_dir / f"{tag}_step{step + 1:02d}_{name}.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        files.append(str(path))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate DIFF-SPARSE v1 with autoregressive rollout.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--num-scenarios", type=int)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--tile-chunk", type=int, default=64, help="Max tiles*scenarios per model batch")
    parser.add_argument(
        "--tile-stride",
        type=int,
        help="Tile stride for rollout blending; default is 75% of patch size, e.g. 48 for 64x64 patches.",
    )
    parser.add_argument(
        "--persistence-mode",
        choices=["oracle", "sparse"],
        default="oracle",
        help="oracle uses dense last context frame; sparse uses observed mask cells and train-mean fill elsewhere.",
    )
    parser.add_argument("--missing-rate", type=float, help="Override eval sparsity (cross-sparsity evaluation)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--save-maps", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.missing_rate is not None:
        config.setdefault("masking", {})["missing_rate"] = float(args.missing_rate)
        print(f"NOTE: eval missing_rate overridden to {args.missing_rate}")
    seed = int(config.get("training", {}).get("seed", config.get("experiment", {}).get("seed", 42)))
    set_seed(seed)

    checkpoint = load_checkpoint(args.checkpoint)
    stats = checkpoint["normalization_stats"]
    water_stats = stats["channels"]["water"]
    device = resolve_device(args.device)

    dataset = build_diff_sparse_v1_dataset(
        path_from_config(config, "dataset_root"),
        config,
        split=args.split,
        normalization_stats=stats,
        patch_mode="full",
    )
    model = DiffSparseV1Model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    evaluation_config = config.get("evaluation", {})
    default_scenarios = int(
        evaluation_config.get("num_scenarios_test" if args.split == "test" else "num_scenarios_val", 2)
    )
    num_scenarios = int(args.num_scenarios or default_scenarios)
    remask = bool(evaluation_config.get("rollout_remask", True))
    clip_x0 = None
    if bool(evaluation_config.get("clip_x0", False)):
        mean, std = float(water_stats["mean"]), float(water_stats["std"])
        clip_x0 = ((float(water_stats["min"]) - mean) / std, (float(water_stats["max"]) - mean) / std)

    if args.output_dir is not None:
        output_base = resolve_path(args.output_dir, PROJECT_DIR)
    else:
        run_dir = path_from_config(config, "experiment_root") / args.checkpoint.parent.name
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        output_base = run_dir / f"eval_rollout_{args.split}_{timestamp}"
    output_dir = unique_dir(output_base)

    total_windows = len(dataset)
    windows = min(total_windows, args.max_windows) if args.max_windows else total_windows
    patch_size = dataset.patch_size
    tile_stride = int(args.tile_stride or max(1, round(0.75 * patch_size)))
    prediction_length = dataset.prediction_length

    print(f"code_root: {PROJECT_DIR}")
    print(f"checkpoint: {args.checkpoint} (epoch {checkpoint.get('epoch')})")
    print(f"output_dir: {output_dir}")
    print(f"split: {args.split} windows: {windows}/{total_windows}")
    print(f"num_scenarios: {num_scenarios} missing_rate: {dataset.missing_rate} mask_mode: {dataset.mask_mode}")
    print(
        f"patch_size: {patch_size} tile_stride: {tile_stride} "
        f"persistence_mode: {args.persistence_mode} rollout_remask: {remask} clip_x0: {clip_x0}"
    )

    model_metrics = MetricAccumulator(prediction_length)
    persistence_metrics = MetricAccumulator(prediction_length)
    official_overall_accumulator = OfficialMetricAccumulator(gammas=OFFICIAL_GAMMAS)
    official_step_accumulators = {
        step: OfficialMetricAccumulator(gammas=OFFICIAL_GAMMAS) for step in range(prediction_length)
    }
    official_path_accumulator = FinalHorizonPathAccumulator(gammas=OFFICIAL_GAMMAS)
    map_files: list[str] = []

    for window_index in range(windows):
        sample = dataset[window_index]
        target = sample["target"].to(device)
        generator = torch.Generator(device=device).manual_seed(seed * 1_000_003 + window_index)
        predictions = rollout_window(
            model,
            sample,
            num_scenarios=num_scenarios,
            patch_size=patch_size,
            tile_stride=tile_stride,
            tile_chunk=args.tile_chunk,
            remask=remask,
            mask_mode=dataset.mask_mode,
            clip_x0=clip_x0,
            generator=generator,
            device=device,
        )
        if not torch.isfinite(predictions).all():
            raise FloatingPointError(f"Non-finite rollout predictions in window {window_index}")
        mean_forecast = predictions.mean(dim=0)
        persistence = persistence_forecast(sample, prediction_length, device, args.persistence_mode)

        model_metrics.update(mean_forecast, predictions, target)
        persistence_metrics.update(persistence, None, target)
        mean_forecast_physical = to_physical(mean_forecast, water_stats)
        target_physical = to_physical(target, water_stats)
        official_overall_accumulator.update(mean_forecast_physical, target_physical)
        for step in range(prediction_length):
            official_step_accumulators[step].update(mean_forecast_physical[step], target_physical[step])
        initial_physical = to_physical(sample["context_water_true"][-1].to(device), water_stats)
        official_path_accumulator.update(mean_forecast_physical, target_physical, initial_physical)

        if args.save_maps and window_index == 0:
            sample_std = predictions.std(dim=0, unbiased=False) if num_scenarios > 1 else torch.zeros_like(mean_forecast)
            for step in (0, prediction_length - 1):
                map_files.extend(
                    save_maps(
                        output_dir,
                        f"{args.split}_window000",
                        target,
                        mean_forecast,
                        sample_std,
                        persistence,
                        step,
                    )
                )
        print(f"window {window_index + 1}/{windows} done", flush=True)

    water_std = float(water_stats["std"])
    model_result = model_metrics.finalize(water_std)
    persistence_result = persistence_metrics.finalize(water_std)

    rows = []
    for step in range(prediction_length):
        model_step = model_result["per_step"][step]
        persistence_step = persistence_result["per_step"][step]
        rows.append(
            {
                "split": args.split,
                "step": step + 1,
                "horizon_label": f"h{dataset.context_length + step + 1}",
                "nrmse": model_step["nrmse"],
                "rmse_normalized": model_step["rmse_normalized"],
                "mae_normalized": model_step["mae_normalized"],
                "rmse_physical_m": model_step["rmse_physical_m"],
                "mae_physical_m": model_step["mae_physical_m"],
                "nacrps": model_step["nacrps"],
                "persistence_nrmse": persistence_step["nrmse"],
                "persistence_rmse_normalized": persistence_step["rmse_normalized"],
                "persistence_mae_normalized": persistence_step["mae_normalized"],
                "persistence_rmse_physical_m": persistence_step["rmse_physical_m"],
            }
        )
    metrics_path = output_dir / "eval_metrics_per_step.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=STEP_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    official_rows = []
    for step in range(prediction_length):
        horizon_label = f"h{dataset.context_length + step + 1}"
        official_step = official_step_accumulators[step].compute()
        official_step.update(
            {
                "checkpoint_name": args.checkpoint.parent.name,
                "step": dataset.context_length + step + 1,
                "rollout_step": step + 1,
                "horizon_label": horizon_label,
                "rollout_samples": windows,
            }
        )
        official_rows.append(official_step)
    official_metrics_path = output_dir / "eval_metrics_official_per_step.csv"
    write_dynamic_csv(official_metrics_path, official_rows)

    official_overall = official_overall_accumulator.compute()
    final_horizon_label = f"h{dataset.context_length + prediction_length}"
    official_path_metrics = official_path_accumulator.compute(
        horizon_label=final_horizon_label,
        horizon_steps=prediction_length,
    )

    model_overall = model_result["overall"]
    persistence_overall = persistence_result["overall"]
    improvement = None
    if persistence_overall["rmse_normalized"] > 0:
        improvement = 100.0 * (
            persistence_overall["rmse_normalized"] - model_overall["rmse_normalized"]
        ) / persistence_overall["rmse_normalized"]

    summary = {
        "config_path": str(args.config),
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_metrics": checkpoint.get("metrics"),
        "split": args.split,
        "windows_evaluated": windows,
        "windows_total": total_windows,
        "num_scenarios": num_scenarios,
        "missing_rate": dataset.missing_rate,
        "mask_mode": dataset.mask_mode,
        "eval_mask_bank_size": dataset.eval_mask_bank_size,
        "context_length": dataset.context_length,
        "prediction_length": prediction_length,
        "patch_size": patch_size,
        "tile_stride": tile_stride,
        "tile_blending": "cell_centered_hann_distance_to_center",
        "persistence_mode": args.persistence_mode,
        "persistence_mode_definition": {
            "oracle": "dense last true context frame; historical conservative baseline",
            "sparse": "last true context frame at observed sensor cells, train water mean fill (0 normalized) elsewhere",
        },
        "rollout_remask": remask,
        "clip_x0": list(clip_x0) if clip_x0 else None,
        "device": str(device),
        "model": model_result,
        "persistence": persistence_result,
        "official_metrics_physical": {
            "units": "meters",
            "gammas_m": OFFICIAL_GAMMAS,
            "source": "mean DIFF-SPARSE rollout forecast and target inverse-transformed with shared train water stats",
            "overall": official_overall,
            "per_step": official_rows,
            "per_step_csv": str(official_metrics_path),
            f"path_{final_horizon_label}": official_path_metrics,
        },
        "rmse_improvement_percent_vs_persistence": improvement,
        "eval_metrics_per_step_csv": str(metrics_path),
        "eval_metrics_official_per_step_csv": str(official_metrics_path),
        "map_files": map_files,
        "output_directory": str(output_dir),
        "metric_definitions": {
            "nrmse": "paper eq. 15: RMSE / (max-min of observations over the evaluated set)",
            "nacrps": "paper eq. 16: sum of empirical CRPS / sum |observation| (persistence uses MAE as point-forecast CRPS)",
            "physical_units": "normalized errors scaled by train water std (meters)",
        },
        "command_reconstruction": command_reconstruction(),
        "git_status_short": git_status_short(),
        "cli_args": cli_args_for_summary(args),
        "scientific_status": EVAL_STATUS,
        "does_not_claim": DOES_NOT_CLAIM,
    }
    save_json(summary, output_dir / "eval_summary.json")
    print("=== DIFF-SPARSE V1 ROLLOUT EVAL ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
