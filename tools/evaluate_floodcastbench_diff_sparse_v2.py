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

from datasets.floodcastbench_diff_sparse_v2_dataset import build_diff_sparse_v2_dataset  # noqa: E402
from models.deterministic_twin import build_v2_family_model  # noqa: E402
from models.diff_sparse_v2 import DiffSparseV2Model  # noqa: E402
from tools.evaluate_floodcastbench_fno_plus_official_v1_long_horizon_rollout import (  # noqa: E402
    MetricAccumulator as OfficialMetricAccumulator,
    gamma_key,
    write_csv as write_dynamic_csv,
)
from tools.evaluate_floodcastbench_diff_sparse_v1 import (  # noqa: E402
    MetricAccumulator,
    persistence_forecast,
    save_maps,
    tile_blend_window,
    tile_positions,
    to_physical,
    unique_dir,
)
from tools.train_floodcastbench_diff_sparse_v1 import (  # noqa: E402
    cli_args_for_summary,
    command_reconstruction,
    git_status_short,
    load_config,
    path_from_config,
    resolve_device,
    resolve_path,
    save_json,
)
from tools.train_floodcastbench_diff_sparse_v2 import (  # noqa: E402
    SCIENTIFIC_STATUS,
    DeltaSpec,
    load_delta_stats,
)
from training.utils import set_seed  # noqa: E402


EVAL_STATUS = "diff_sparse_v2_floodcastbench_rollout_eval"
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
            f"is not a DIFF-SPARSE v2 checkpoint ({SCIENTIFIC_STATUS!r})"
        )
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing model_state_dict")
    if not isinstance(checkpoint.get("normalization_stats"), dict):
        raise KeyError("Checkpoint is missing normalization_stats")
    return checkpoint


class CalibrationAccumulator:
    """Probabilistic calibration of the scenario ensemble (master plan WP3).

    Everything is computed in PHYSICAL units (meters) from the M rollout
    scenarios vs the single deterministic ground truth. With a deterministic
    simulator target, what is measurable is the calibration of the model's
    RECONSTRUCTION/forecast ambiguity, not of physical aleatoric uncertainty
    -- state this in the paper (master plan §2 caveat a).

    Accumulated:
      - reliability, per flood threshold gamma: forecast probability of
        "wet" is the fraction k/M of scenarios >= gamma; per (step, k) we
        count pixels and how often the target was actually wet. M+1 discrete
        probability levels -- no binning artifacts.
      - central-interval coverage (50% and 90%) from per-pixel ensemble
        quantiles, per step.
      - rank histogram (target's rank among the M members, uniform random
        tie-breaking with a fixed seed), pooled over steps; an 'active'
        variant restricted to pixels where the target or any member is wet
        at the smallest gamma (dry-everywhere pixels otherwise dominate).
      - spread-skill: pixels bucketed by ensemble std (log-spaced physical
        bins); per bucket, mean spread vs MAE/RMSE of the ensemble mean.

    Requires M >= 2; the deterministic twin (M=1) skips calibration entirely.
    """

    def __init__(self, prediction_length: int, gammas: list[float], num_members: int) -> None:
        if num_members < 2:
            raise ValueError("CalibrationAccumulator requires >= 2 scenarios")
        self.length = int(prediction_length)
        self.gammas = [float(g) for g in gammas]
        self.members = int(num_members)
        self.reliability_counts = {
            g: torch.zeros(self.length, self.members + 1, dtype=torch.float64) for g in self.gammas
        }
        self.reliability_wet = {
            g: torch.zeros(self.length, self.members + 1, dtype=torch.float64) for g in self.gammas
        }
        self.intervals = {"50": (0.25, 0.75), "90": (0.05, 0.95)}
        self.coverage_inside = {k: torch.zeros(self.length, dtype=torch.float64) for k in self.intervals}
        self.coverage_total = torch.zeros(self.length, dtype=torch.float64)
        self.rank_counts = torch.zeros(self.members + 1, dtype=torch.float64)
        self.rank_counts_active = torch.zeros(self.members + 1, dtype=torch.float64)
        # First edge catches the exactly-collapsed (std == 0) population.
        self.spread_edges = torch.cat(
            [torch.tensor([0.0, 1e-6]), torch.logspace(-5, 0.5, steps=12, dtype=torch.float64)]
        )
        bins = self.spread_edges.numel() - 1
        self.spread_n = torch.zeros(bins, dtype=torch.float64)
        self.spread_sum_std = torch.zeros(bins, dtype=torch.float64)
        self.spread_sum_abs_err = torch.zeros(bins, dtype=torch.float64)
        self.spread_sum_sq_err = torch.zeros(bins, dtype=torch.float64)

    @torch.no_grad()
    def update(self, scenarios_physical: torch.Tensor, target_physical: torch.Tensor, window_index: int) -> None:
        if scenarios_physical.ndim != 4 or scenarios_physical.shape[0] != self.members:
            raise ValueError(f"Expected scenarios [M={self.members}, l, H, W], got {tuple(scenarios_physical.shape)}")
        if scenarios_physical.shape[1:] != target_physical.shape:
            raise ValueError("scenarios/target shape mismatch")
        scenarios = scenarios_physical.double()
        target = target_physical.double()

        for gamma in self.gammas:
            wet_members = (scenarios >= gamma).sum(dim=0)  # [l, H, W] in 0..M
            target_wet = target >= gamma
            for step in range(self.length):
                k = wet_members[step].flatten()
                obs = target_wet[step].flatten().double()
                self.reliability_counts[gamma][step] += torch.bincount(
                    k, minlength=self.members + 1
                ).double().cpu()
                self.reliability_wet[gamma][step] += torch.bincount(
                    k, weights=obs, minlength=self.members + 1
                ).double().cpu()

        quantile_points = torch.tensor(
            sorted({q for pair in self.intervals.values() for q in pair}),
            device=scenarios.device, dtype=scenarios.dtype,
        )
        quantiles = torch.quantile(scenarios, quantile_points, dim=0)  # [Q, l, H, W]
        q_index = {float(q): i for i, q in enumerate(quantile_points.tolist())}
        pixels_per_step = float(target[0].numel())
        for name, (lo, hi) in self.intervals.items():
            inside = (target >= quantiles[q_index[lo]]) & (target <= quantiles[q_index[hi]])
            self.coverage_inside[name] += inside.sum(dim=(1, 2)).double().cpu()
        self.coverage_total += pixels_per_step

        below = (scenarios < target.unsqueeze(0)).sum(dim=0)  # [l, H, W]
        ties = (scenarios == target.unsqueeze(0)).sum(dim=0)
        tie_generator = torch.Generator(device="cpu").manual_seed(97 + window_index)
        jitter = torch.rand(ties.shape, generator=tie_generator).to(ties.device)
        rank = (below + (jitter * (ties + 1).double()).floor().long()).clamp(0, self.members)
        self.rank_counts += torch.bincount(rank.flatten(), minlength=self.members + 1).double().cpu()
        smallest_gamma = min(self.gammas)
        active = (target >= smallest_gamma) | ((scenarios >= smallest_gamma).any(dim=0))
        if bool(active.any()):
            self.rank_counts_active += torch.bincount(
                rank[active].flatten(), minlength=self.members + 1
            ).double().cpu()

        spread = scenarios.std(dim=0, unbiased=False)
        error = (scenarios.mean(dim=0) - target).abs()
        bucket = torch.bucketize(spread.flatten().cpu(), self.spread_edges[1:-1])
        bins = self.spread_edges.numel() - 1
        flat_spread = spread.flatten().cpu()
        flat_error = error.flatten().cpu()
        self.spread_n += torch.bincount(bucket, minlength=bins).double()
        self.spread_sum_std += torch.bincount(bucket, weights=flat_spread, minlength=bins)
        self.spread_sum_abs_err += torch.bincount(bucket, weights=flat_error, minlength=bins)
        self.spread_sum_sq_err += torch.bincount(bucket, weights=flat_error.square(), minlength=bins)

    def summary(self) -> dict[str, Any]:
        reliability = {}
        for gamma in self.gammas:
            counts = self.reliability_counts[gamma]
            wet = self.reliability_wet[gamma]
            pooled_counts = counts.sum(dim=0)
            pooled_wet = wet.sum(dim=0)
            reliability[f"gamma_{gamma}"] = {
                "forecast_probability": [k / self.members for k in range(self.members + 1)],
                "pooled_count": pooled_counts.tolist(),
                "pooled_observed_frequency": (
                    pooled_wet / pooled_counts.clamp(min=1.0)
                ).tolist(),
                "per_step_count": counts.tolist(),
                "per_step_observed_frequency": (wet / counts.clamp(min=1.0)).tolist(),
            }
        coverage = {}
        for name, (lo, hi) in self.intervals.items():
            # Finite-ensemble bias: with M members and linearly-interpolated
            # empirical quantiles, a perfectly calibrated new draw falls inside
            # the (lo, hi) interval with probability (hi-lo)*(M-1)/(M+1), NOT
            # (hi-lo) -- e.g. the "90%" interval of an M=8 ensemble covers only
            # 70% in expectation. Compare 'pooled' against
            # 'nominal_finite_ensemble', not 'nominal'.
            coverage[name] = {
                "nominal": hi - lo,
                "nominal_finite_ensemble": (hi - lo) * (self.members - 1) / (self.members + 1),
                "per_step": (self.coverage_inside[name] / self.coverage_total.clamp(min=1.0)).tolist(),
                "pooled": float(self.coverage_inside[name].sum() / self.coverage_total.sum().clamp(min=1.0)),
            }
        rank_total = self.rank_counts.sum().clamp(min=1.0)
        rank_active_total = self.rank_counts_active.sum().clamp(min=1.0)
        spread_bins = []
        for i in range(self.spread_edges.numel() - 1):
            n = float(self.spread_n[i])
            spread_bins.append({
                "edge_low_m": float(self.spread_edges[i]),
                "edge_high_m": float(self.spread_edges[i + 1]),
                "count": n,
                "mean_spread_m": float(self.spread_sum_std[i] / max(n, 1.0)),
                "mae_m": float(self.spread_sum_abs_err[i] / max(n, 1.0)),
                "rmse_m": math.sqrt(float(self.spread_sum_sq_err[i] / max(n, 1.0))),
            })
        return {
            "num_members": self.members,
            "units": "meters",
            "caveat": (
                "single deterministic simulation target: this measures calibration of the "
                "model's reconstruction/forecast ambiguity, not physical aleatoric uncertainty"
            ),
            "reliability": reliability,
            "coverage": coverage,
            "rank_histogram": {
                "counts": self.rank_counts.tolist(),
                "frequency": (self.rank_counts / rank_total).tolist(),
                "active_counts": self.rank_counts_active.tolist(),
                "active_frequency": (self.rank_counts_active / rank_active_total).tolist(),
                "uniform_reference": 1.0 / (self.members + 1),
            },
            "spread_skill": spread_bins,
        }


class MultiHorizonPathAccumulator:
    """Path IoU and propagation-path IoU at EVERY rollout step.

    Per horizon step h (1-indexed within the rollout):
      - path IoU: IoU of (flooded at step h) MINUS (flooded initially), i.e.
        the cumulative newly-flooded-since-context area, pred vs target.
      - per-step propagation IoU: IoU of pixels newly flooded exactly at step
        h relative to step h-1 (step 0 = initial frame), pred vs target.
    Fixes V1's FinalHorizonPathAccumulator, which only reported the final
    horizon and pooled propagation counts over all steps.
    """

    def __init__(self, prediction_length: int, gammas: list[float]) -> None:
        self.length = int(prediction_length)
        self.gammas = tuple(float(gamma) for gamma in gammas)
        self.samples = 0
        self.counts = {
            gamma: {
                "tp": [0.0] * self.length,
                "fp": [0.0] * self.length,
                "fn": [0.0] * self.length,
                "prop_tp": [0.0] * self.length,
                "prop_fp": [0.0] * self.length,
                "prop_fn": [0.0] * self.length,
            }
            for gamma in self.gammas
        }

    def update(self, pred: torch.Tensor, target: torch.Tensor, initial: torch.Tensor) -> None:
        if pred.shape != target.shape:
            raise ValueError(f"pred and target must match, got {pred.shape} vs {target.shape}")
        if pred.ndim != 3 or pred.shape[0] != self.length:
            raise ValueError(f"Expected pred/target [{self.length}, H, W], got {tuple(pred.shape)}")
        if initial.shape != pred.shape[1:]:
            raise ValueError(f"Expected initial [H, W] matching pred, got {tuple(initial.shape)}")
        pred = pred.detach().float().cpu()
        target = target.detach().float().cpu()
        initial = initial.detach().float().cpu()
        self.samples += 1

        for gamma in self.gammas:
            counts = self.counts[gamma]
            initial_mask = initial > gamma
            prev_pred = initial_mask
            prev_target = initial_mask
            for step in range(self.length):
                pred_mask = pred[step] > gamma
                target_mask = target[step] > gamma
                pred_path = pred_mask & ~initial_mask
                target_path = target_mask & ~initial_mask
                counts["tp"][step] += float((pred_path & target_path).sum().item())
                counts["fp"][step] += float((pred_path & ~target_path).sum().item())
                counts["fn"][step] += float((~pred_path & target_path).sum().item())

                pred_new = pred_mask & ~prev_pred
                target_new = target_mask & ~prev_target
                counts["prop_tp"][step] += float((pred_new & target_new).sum().item())
                counts["prop_fp"][step] += float((pred_new & ~target_new).sum().item())
                counts["prop_fn"][step] += float((~pred_new & target_new).sum().item())
                prev_pred = pred_mask
                prev_target = target_mask

    def per_step_metrics(self) -> list[dict[str, Any]]:
        eps = 1e-12
        rows = []
        for step in range(self.length):
            row: dict[str, Any] = {"rollout_step": step + 1, "samples": self.samples}
            for gamma in self.gammas:
                key = gamma_key(gamma)
                counts = self.counts[gamma]
                tp, fp, fn = counts["tp"][step], counts["fp"][step], counts["fn"][step]
                ptp, pfp, pfn = counts["prop_tp"][step], counts["prop_fp"][step], counts["prop_fn"][step]
                row[f"path_iou_gamma_{key}"] = tp / max(tp + fp + fn, eps)
                row[f"propagation_path_iou_gamma_{key}"] = ptp / max(ptp + pfp + pfn, eps)
                row[f"path_tp_gamma_{key}"] = int(tp)
                row[f"path_fp_gamma_{key}"] = int(fp)
                row[f"path_fn_gamma_{key}"] = int(fn)
                row[f"propagation_tp_gamma_{key}"] = int(ptp)
                row[f"propagation_fp_gamma_{key}"] = int(pfp)
                row[f"propagation_fn_gamma_{key}"] = int(pfn)
            rows.append(row)
        return rows

    def pooled_propagation(self) -> dict[str, Any]:
        eps = 1e-12
        result: dict[str, Any] = {"samples": self.samples}
        for gamma in self.gammas:
            key = gamma_key(gamma)
            counts = self.counts[gamma]
            ptp, pfp, pfn = sum(counts["prop_tp"]), sum(counts["prop_fp"]), sum(counts["prop_fn"])
            result[f"propagation_path_iou_gamma_{key}"] = ptp / max(ptp + pfp + pfn, eps)
            final = self.length - 1
            tp, fp, fn = counts["tp"][final], counts["fp"][final], counts["fn"][final]
            result[f"final_path_iou_gamma_{key}"] = tp / max(tp + fp + fn, eps)
        return result


@torch.no_grad()
def rollout_window(
    model: DiffSparseV2Model,
    sample: dict[str, torch.Tensor],
    num_scenarios: int,
    patch_size: int,
    tile_stride: int,
    tile_chunk: int,
    remask: bool,
    mask_mode: str,
    delta: DeltaSpec,
    clamp: bool,
    generator: torch.Generator | None,
    device: torch.device,
) -> torch.Tensor:
    """V2 autoregressive per-tile rollout.

    Adds vs V1: sliding target-step rainfall conditioning, delta-space
    prediction with per-tile absolute reconstruction (base = observed last
    context frame at step 1, then the model's own dense previous prediction),
    and the physically-bounded x0 clamp. No re-masking by default (matches the
    reference's generate_multistep_scenarios). Returns ABSOLUTE-space
    predictions [M, l, H, W]."""

    context_masked = sample["context_water_masked"].to(device)
    context_true = sample["context_water_true"].to(device)
    sensor_mask = sample["sensor_mask"].to(device)
    dem = sample["dem"].to(device)
    rainfall = sample["rainfall"].to(device)
    timestamps = sample["timestamps"].to(device)
    manning = sample["manning"].to(device) if "manning" in sample else None
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
        context_true_tiles = stack_tiles(context_true)
        mask = stack_tiles(sensor_mask)
        dem_tiles = stack_tiles(dem)
        rain_tiles = stack_tiles(rainfall)
        manning_tiles = stack_tiles(manning) if manning is not None else None
        ts_batch = timestamps.unsqueeze(0).expand(batch_size, -1)
        base = delta.base_from_sample(context_true_tiles, mask)

        for step in range(prediction_length):
            # Step 0: base is the observed masked frame -> per-pixel scale.
            # Steps >= 1: base is the model's own dense prediction -> scalar
            # delta scale. Must mirror the training-time DeltaSpec contract.
            scale = delta.scale_for_observed_base(mask) if step == 0 else None
            model_batch = {
                "context_water_masked": context,
                "sensor_mask": mask,
                "dem": dem_tiles,
                "rainfall_context": rain_tiles[:, step : step + context_length],
                "rainfall_target": rain_tiles[:, step + context_length : step + context_length + 1],
                "timestamps_context": ts_batch[:, step : step + context_length],
            }
            if manning_tiles is not None:
                model_batch["manning"] = manning_tiles
            tokens, spatial = model.encode_context(model_batch)
            prediction = model.sample(
                tokens,
                spatial,
                (batch_size, 1, patch_size, patch_size),
                generator=generator,
                clip_x0=delta.clip_for_sampler(base, clamp, scale=scale),
            )
            absolute = delta.to_absolute(prediction, base, clamp=clamp, scale=scale)
            for tile_index, (y, x) in enumerate(chunk):
                block = absolute[tile_index * num_scenarios : (tile_index + 1) * num_scenarios, 0]
                output_sum[:, step, y : y + patch_size, x : x + patch_size] += block * blend

            if remask:
                fill = torch.randn_like(absolute) if mask_mode == "noise" else torch.zeros_like(absolute)
                new_frame = absolute * mask + (1.0 - mask) * fill
            else:
                new_frame = absolute
            context = torch.cat([context[:, 1:], new_frame], dim=1)
            base = absolute

        for y, x in chunk:
            weight[y : y + patch_size, x : x + patch_size] += blend

    return output_sum / weight.clamp(min=torch.finfo(output_sum.dtype).eps)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate DIFF-SPARSE v2 with autoregressive rollout.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--num-scenarios", type=int)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--tile-chunk", type=int, default=64, help="Max tiles*scenarios per model batch")
    parser.add_argument(
        "--tile-stride",
        type=int,
        help="Tile stride for rollout blending; V2 default is patch_size/2 (e.g. 32) for stronger seam suppression.",
    )
    parser.add_argument(
        "--persistence-mode",
        choices=["oracle", "sparse"],
        default="oracle",
    )
    parser.add_argument("--missing-rate", type=float, help="Override eval sparsity (cross-sparsity evaluation)")
    parser.add_argument(
        "--mask-structure",
        choices=["random", "gauge", "cluster"],
        help="Override masking.eval_mask_structure (WP7: structured sensor layouts, same budget)",
    )
    parser.add_argument(
        "--no-clamp-physical",
        action="store_true",
        help="Disable the >=0 physical depth clamp (V2 default: clamp during sampling AND on final forecasts)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--save-maps", action="store_true")
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Skip the WP3 calibration accumulator (on by default whenever num_scenarios >= 2)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.missing_rate is not None:
        config.setdefault("masking", {})["missing_rate"] = float(args.missing_rate)
        print(f"NOTE: eval missing_rate overridden to {args.missing_rate}")
    if args.mask_structure is not None:
        config.setdefault("masking", {})["eval_mask_structure"] = args.mask_structure
        print(f"NOTE: eval mask structure overridden to {args.mask_structure}")
    seed = int(config.get("training", {}).get("seed", config.get("experiment", {}).get("seed", 42)))
    set_seed(seed)

    checkpoint = load_checkpoint(args.checkpoint)
    stats = checkpoint["normalization_stats"]
    water_stats = stats["channels"]["water"]
    device = resolve_device(args.device)

    dataset = build_diff_sparse_v2_dataset(
        path_from_config(config, "dataset_root"),
        config,
        split=args.split,
        normalization_stats=stats,
        patch_mode="full",
    )
    model = build_v2_family_model(config).to(device)
    print(f"model class: {type(model).__name__}")
    use_ema = bool(config.get("evaluation", {}).get("use_ema", True)) and checkpoint.get("ema_state_dict")
    if use_ema:
        # EMA weights: keyed by parameter name; buffers come from the raw state dict.
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(checkpoint["ema_state_dict"][name])
    else:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    prediction_config = config.get("prediction", {})
    delta_stats = checkpoint.get("delta_stats") or load_delta_stats(
        Path(prediction_config["delta_stats_json"]) if prediction_config.get("delta_stats_json") else None
    )
    delta = DeltaSpec(str(prediction_config.get("target", "delta")), water_stats, delta_stats)

    evaluation_config = config.get("evaluation", {})
    default_scenarios = int(
        evaluation_config.get("num_scenarios_test" if args.split == "test" else "num_scenarios_val", 2)
    )
    num_scenarios = int(args.num_scenarios or default_scenarios)
    remask = bool(evaluation_config.get("rollout_remask", False))
    clamp_physical = bool(evaluation_config.get("clip_x0_physical", True)) and not args.no_clamp_physical

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
    tile_stride = int(args.tile_stride or max(1, patch_size // 2))
    prediction_length = dataset.prediction_length

    print(f"code_root: {PROJECT_DIR}")
    print(f"checkpoint: {args.checkpoint} (epoch {checkpoint.get('epoch')}) ema: {bool(use_ema)}")
    print(f"output_dir: {output_dir}")
    print(f"split: {args.split} windows: {windows}/{total_windows}")
    print(f"num_scenarios: {num_scenarios} missing_rate: {dataset.missing_rate} mask_mode: {dataset.mask_mode}")
    print(
        f"patch_size: {patch_size} tile_stride: {tile_stride} "
        f"persistence_mode: {args.persistence_mode} rollout_remask: {remask} "
        f"target_space: {delta.mode} (scale={delta.scale:.6g}) clamp_physical: {clamp_physical}"
    )

    model_metrics = MetricAccumulator(prediction_length)
    persistence_metrics = MetricAccumulator(prediction_length)
    official_overall_accumulator = OfficialMetricAccumulator(gammas=OFFICIAL_GAMMAS)
    official_step_accumulators = {
        step: OfficialMetricAccumulator(gammas=OFFICIAL_GAMMAS) for step in range(prediction_length)
    }
    path_accumulator = MultiHorizonPathAccumulator(prediction_length, gammas=OFFICIAL_GAMMAS)
    # Scenario-majority decision masks via the per-pixel MEDIAN field: for every
    # threshold gamma simultaneously, median > gamma <=> the majority of
    # scenarios exceed gamma -- the optimal mask decision rule under the model's
    # own uncertainty, vs thresholding the (front-smearing) mean.
    path_accumulator_median = MultiHorizonPathAccumulator(prediction_length, gammas=OFFICIAL_GAMMAS)
    official_step_accumulators_median = {
        step: OfficialMetricAccumulator(gammas=OFFICIAL_GAMMAS) for step in range(prediction_length)
    }
    calibration_accumulator = (
        CalibrationAccumulator(prediction_length, gammas=OFFICIAL_GAMMAS, num_members=num_scenarios)
        if num_scenarios >= 2 and not args.no_calibration
        else None
    )
    map_files: list[str] = []
    # Per-window metric log (paper master plan / limitations): the pooled
    # official_overall_accumulator above gives one relative RMSE for the
    # whole test split, which is all the pre-registered window x seed
    # paired significance test (n=39, Wilcoxon signed-rank / bootstrap)
    # needs to stay a seed-paired sign test (n=3) instead. A fresh,
    # single-window MetricAccumulator per window computes exactly the same
    # metrics the pooled one does, just scoped to one window instead of
    # accumulated across all of them -- purely additive, does not touch
    # official_overall_accumulator or any pooled result already relied on.
    per_window_metrics: list[dict[str, Any]] = []

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
            delta=delta,
            clamp=clamp_physical,
            generator=generator,
            device=device,
        )
        if not torch.isfinite(predictions).all():
            raise FloatingPointError(f"Non-finite rollout predictions in window {window_index}")
        mean_forecast = predictions.mean(dim=0)
        median_forecast = predictions.median(dim=0).values if num_scenarios > 1 else mean_forecast
        if clamp_physical:
            floor = delta.floor_absolute
            mean_forecast = mean_forecast.clamp(min=floor)
            median_forecast = median_forecast.clamp(min=floor)
            predictions = predictions.clamp(min=floor)
        persistence = persistence_forecast(sample, prediction_length, device, args.persistence_mode)

        model_metrics.update(mean_forecast, predictions, target)
        persistence_metrics.update(persistence, None, target)
        mean_forecast_physical = to_physical(mean_forecast, water_stats)
        median_forecast_physical = to_physical(median_forecast, water_stats)
        if clamp_physical:
            mean_forecast_physical = mean_forecast_physical.clamp(min=0.0)
            median_forecast_physical = median_forecast_physical.clamp(min=0.0)
        target_physical = to_physical(target, water_stats)
        official_overall_accumulator.update(mean_forecast_physical, target_physical)
        window_metric_accumulator = OfficialMetricAccumulator(gammas=OFFICIAL_GAMMAS)
        window_metric_accumulator.update(mean_forecast_physical, target_physical)
        per_window_metrics.append({"window_index": window_index, **window_metric_accumulator.compute()})
        for step in range(prediction_length):
            official_step_accumulators[step].update(mean_forecast_physical[step], target_physical[step])
            official_step_accumulators_median[step].update(median_forecast_physical[step], target_physical[step])
        initial_physical = to_physical(sample["context_water_true"][-1].to(device), water_stats)
        path_accumulator.update(mean_forecast_physical, target_physical, initial_physical)
        path_accumulator_median.update(median_forecast_physical, target_physical, initial_physical)
        if calibration_accumulator is not None:
            scenarios_physical = to_physical(predictions, water_stats)
            if clamp_physical:
                scenarios_physical = scenarios_physical.clamp(min=0.0)
            calibration_accumulator.update(scenarios_physical, target_physical, window_index)

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

    path_rows = path_accumulator.per_step_metrics()
    path_rows_median = path_accumulator_median.per_step_metrics()
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
        # Merge path/propagation IoU into the same per-step official CSV so the
        # dashboard can read everything from one artifact. _median columns are
        # the scenario-majority (median-field) decision-mask variants.
        official_step.update(
            {key: value for key, value in path_rows[step].items() if key not in ("rollout_step", "samples")}
        )
        official_step.update(
            {
                f"{key}_median": value
                for key, value in path_rows_median[step].items()
                if key not in ("rollout_step", "samples")
            }
        )
        median_step = official_step_accumulators_median[step].compute()
        official_step.update(
            {f"{key}_median": value for key, value in median_step.items() if key.startswith(("csi", "precision", "recall", "f1"))}
        )
        official_rows.append(official_step)
    official_metrics_path = output_dir / "eval_metrics_official_per_step.csv"
    write_dynamic_csv(official_metrics_path, official_rows)

    official_overall = official_overall_accumulator.compute()
    pooled_propagation = path_accumulator.pooled_propagation()
    pooled_propagation_median = path_accumulator_median.pooled_propagation()

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
        "rollout_remask": remask,
        "target_space": delta.mode,
        "delta_scale_normalized": delta.scale,
        "use_ema": bool(use_ema),
        "clamp_physical": clamp_physical,
        "clip_x0_floor_normalized": delta.floor_absolute if clamp_physical else None,
        "device": str(device),
        "model": model_result,
        "persistence": persistence_result,
        "official_metrics_physical": {
            "units": "meters",
            "gammas_m": OFFICIAL_GAMMAS,
            "source": "mean DIFF-SPARSE v2 rollout forecast and target inverse-transformed with shared train water stats",
            "overall": official_overall,
            "per_step": official_rows,
            "per_step_csv": str(official_metrics_path),
            "pooled_propagation": pooled_propagation,
            "pooled_propagation_median": pooled_propagation_median,
            "median_columns_definition": (
                "_median columns use the per-pixel median of scenarios: for every gamma, "
                "median > gamma iff the majority of scenarios exceed gamma -- the "
                "scenario-majority decision mask, vs thresholding the mean forecast"
            ),
        },
        "rmse_improvement_percent_vs_persistence": improvement,
        "eval_metrics_per_step_csv": str(metrics_path),
        "eval_metrics_official_per_step_csv": str(official_metrics_path),
        "eval_per_window_metrics_json": str(output_dir / "eval_per_window_metrics.json"),
        "map_files": map_files,
        "output_directory": str(output_dir),
        "metric_definitions": {
            "nrmse": "paper eq. 15: RMSE / (max-min of observations over the evaluated set)",
            "nacrps": "paper eq. 16: sum of empirical CRPS / sum |observation| (persistence uses MAE as point-forecast CRPS)",
            "physical_units": "normalized errors scaled by train water std (meters)",
            "path_iou": "IoU of cumulative newly-flooded-since-context area at each step, pred vs target",
            "propagation_path_iou": "IoU of pixels newly flooded exactly at each step (vs previous step), pred vs target",
        },
        "command_reconstruction": command_reconstruction(),
        "git_status_short": git_status_short(),
        "cli_args": cli_args_for_summary(args),
        "scientific_status": EVAL_STATUS,
        "does_not_claim": DOES_NOT_CLAIM,
    }
    if calibration_accumulator is not None:
        calibration = calibration_accumulator.summary()
        save_json(calibration, output_dir / "eval_calibration.json")
        summary["calibration"] = {
            "file": str(output_dir / "eval_calibration.json"),
            "num_members": calibration["num_members"],
            "coverage_pooled": {name: entry["pooled"] for name, entry in calibration["coverage"].items()},
            "caveat": calibration["caveat"],
        }
    else:
        summary["calibration"] = {
            "skipped": "num_scenarios < 2 (deterministic forecast)" if num_scenarios < 2 else "--no-calibration"
        }
    save_json(per_window_metrics, output_dir / "eval_per_window_metrics.json")
    save_json(summary, output_dir / "eval_summary.json")
    print("=== DIFF-SPARSE V2 ROLLOUT EVAL ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
