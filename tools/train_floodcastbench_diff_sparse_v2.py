from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from datasets.floodcastbench_diff_sparse_v2_dataset import build_diff_sparse_v2_dataset  # noqa: E402
from models.diff_sparse_v2 import ConsistencyLoss, DiffSparseV2Model  # noqa: E402
from tools.train_floodcastbench_diff_sparse_v1 import (  # noqa: E402
    apply_overrides,
    build_loader,
    cli_args_for_summary,
    command_reconstruction,
    create_run_dirs,
    git_status_short,
    load_config,
    load_or_compute_stats,
    move_batch_to_device,
    path_from_config,
    resolve_device,
    save_json,
    save_yaml,
)
from training.utils import set_seed  # noqa: E402


SCIENTIFIC_STATUS = "diff_sparse_v2_floodcastbench"
DOES_NOT_CLAIM = [
    "official FloodCastBench benchmark performance",
    "official DIFF-SPARSE TideWatch reproduction",
    "superiority over persistence, FNO+, or DIFF-SPARSE v1 until evaluated",
    "uncertainty calibration",
]

METRIC_FIELDS = [
    "epoch",
    "train_loss",
    "train_x0_rmse",
    "val_loss",
    "val_x0_rmse",
    "rollout_val_rmse",
    "learning_rate",
    "elapsed_seconds",
]


def normalized_zero_depth(water_stats: dict[str, Any]) -> float:
    """Normalized value of 0 physical water depth (the absolute-space clamp floor)."""

    return (0.0 - float(water_stats["mean"])) / float(water_stats["std"])


class DeltaSpec:
    """Everything needed to move between absolute and delta target spaces.

    Delta mode diffuses x0 = (next_frame - base) / scale where base is the
    last context frame as observed (true values at sensor cells, train-mean
    fill = 0 normalized elsewhere; under missing_rate=0 this is exactly the
    true last frame, and from rollout step 2 onward the base is the model's
    own dense previous prediction). scale = train delta std expressed in
    normalized units, so the delta target is ~unit variance. Measured on this
    dataset: delta std 0.0007 m vs water std ~0.29 m -- the actual per-step
    signal is ~400x smaller than the absolute field a non-delta model spends
    its capacity re-encoding.
    """

    def __init__(self, mode: str, water_stats: dict[str, Any], delta_stats: dict[str, Any] | None) -> None:
        self.mode = str(mode).lower()
        if self.mode not in {"delta", "absolute"}:
            raise ValueError(f"prediction.target must be 'delta' or 'absolute', got {mode!r}")
        self.floor_absolute = normalized_zero_depth(water_stats)
        water_mean = float(water_stats["mean"])
        water_std = float(water_stats["std"])
        # Generous physical ceiling (2x the observed train maximum, normalized)
        # used only to bound single-shot/no-grad reconstructions (pushforward's
        # terminal-step x0 guess) against runaway extrapolation -- not a
        # scientific clamp, a numerical safety net. A 2026-07-06 pilot
        # destabilized permanently at epoch 52 when an unclamped pushforward
        # reconstruction produced an extreme value that then poisoned the
        # training context; see reports/diff_sparse_v2_design.md.
        self.ceiling_absolute = (2.0 * float(water_stats["max"]) - water_mean) / water_std
        if self.mode == "delta":
            if delta_stats is None:
                raise ValueError(
                    "prediction.target=delta requires delta stats "
                    "(tools/compute_floodcastbench_diff_sparse_v2_delta_stats.py)"
                )
            self.scale = float(delta_stats["delta_std_physical"]) / water_std
            if self.scale <= 0:
                raise ValueError(f"Invalid delta scale {self.scale}")
        else:
            self.scale = 1.0

    def base_from_sample(self, context_water_true: torch.Tensor, sensor_mask: torch.Tensor) -> torch.Tensor:
        """Observed base frame: [.., H, W] -> [.., 1, H, W] (mean fill = 0 normalized)."""

        last_true = context_water_true[..., -1, :, :].unsqueeze(-3)
        return last_true * sensor_mask

    def to_target_space(self, absolute: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        if self.mode == "absolute":
            return absolute
        return (absolute - base) / self.scale

    def to_absolute(
        self, prediction: torch.Tensor, base: torch.Tensor, clamp: bool, bound_ceiling: bool = False
    ) -> torch.Tensor:
        if self.mode == "absolute":
            absolute = prediction
        else:
            absolute = base + self.scale * prediction
        if clamp:
            absolute = absolute.clamp(min=self.floor_absolute)
        if bound_ceiling:
            # Numerical safety net only (not a physical assumption): bounds
            # single-shot/no-grad reconstructions (pushforward) against
            # runaway extrapolation. See __init__ docstring.
            absolute = absolute.clamp(max=self.ceiling_absolute)
        return absolute

    def clip_for_sampler(self, base: torch.Tensor, enabled: bool):
        """clip_x0 tuple for model.sample(): scalar floor in absolute mode,
        per-pixel tensor floor in delta mode."""

        if not enabled:
            return None
        if self.mode == "absolute":
            return (self.floor_absolute, None)
        return ((self.floor_absolute - base) / self.scale, None)


def change_weight_map(
    target_absolute: torch.Tensor,
    base: torch.Tensor,
    wet_threshold_normalized: float,
    change_weight: float,
) -> torch.Tensor | None:
    """Per-pixel loss weights: pixels whose wet/dry state changes between the
    base frame and the target get weight (1 + change_weight). This is exactly
    the propagation-path population."""

    if change_weight <= 0:
        return None
    wet_target = target_absolute > wet_threshold_normalized
    wet_base = base > wet_threshold_normalized
    changed = (wet_target != wet_base).to(target_absolute.dtype)
    return 1.0 + change_weight * changed


def prepare_model_batch(
    batch: dict[str, Any],
    context_length: int,
    delta: DeltaSpec,
    wet_threshold_normalized: float,
    change_weight: float,
) -> dict[str, torch.Tensor]:
    """One-step training contract, V2: target-space transform + target rainfall
    + per-pixel change weights."""

    base = delta.base_from_sample(batch["context_water_true"], batch["sensor_mask"])
    target_absolute = batch["target"][:, 0:1]
    model_batch = {
        "context_water_masked": batch["context_water_masked"],
        "sensor_mask": batch["sensor_mask"],
        "dem": batch["dem"],
        "rainfall_context": batch["rainfall"][:, :context_length],
        "rainfall_target": batch["rainfall"][:, context_length : context_length + 1],
        "timestamps_context": batch["timestamps"][:, :context_length],
        "target": delta.to_target_space(target_absolute, base),
        "base": base,
        "target_absolute": target_absolute,
    }
    weights = change_weight_map(target_absolute, base, wet_threshold_normalized, change_weight)
    if weights is not None:
        model_batch["pixel_weights"] = weights
    return model_batch


def pushforward_batch(
    model: DiffSparseV2Model,
    batch: dict[str, Any],
    model_batch: dict[str, torch.Tensor],
    context_length: int,
    delta: DeltaSpec,
    wet_threshold_normalized: float,
    change_weight: float,
    clamp: bool,
) -> dict[str, torch.Tensor] | None:
    """Exposure-bias reduction (pushforward trick adapted to diffusion).

    One extra no-grad forward approximates the model's step-1 prediction (the
    terminal-step x0_hat from pure noise -- exactly the first reverse-sampling
    step), which is appended to the context; the gradient step then trains
    step 2 against the true frame, with the target expressed relative to the
    model's own (imperfect) base. The model learns to correct its own drift,
    which one-step teacher-forced training never exercises.
    """

    if batch["target"].shape[1] < 2:
        return None
    with torch.no_grad():
        tokens, spatial = model.encode_context(model_batch)
        shape = model_batch["target"].shape
        x_terminal = torch.randn(shape, device=model_batch["target"].device)
        terminal = torch.full((shape[0],), model.diffusion_steps - 1, dtype=torch.long, device=x_terminal.device)
        prediction = model.denoise(x_terminal, terminal, tokens, spatial)
        # bound_ceiling=True: this is a single-shot x0 guess from pure noise
        # (not full ancestral sampling), which can extrapolate to extreme
        # values early/mid training; an unclamped ceiling here previously
        # poisoned the training context and destabilized the run permanently
        # (2026-07-06 pilot, epoch 52).
        absolute_1 = delta.to_absolute(prediction, model_batch["base"], clamp=clamp, bound_ceiling=True)

    context_2 = torch.cat([batch["context_water_masked"][:, 1:], absolute_1], dim=1)
    target_absolute_2 = batch["target"][:, 1:2]
    step_batch = {
        "context_water_masked": context_2,
        "sensor_mask": batch["sensor_mask"],
        "dem": batch["dem"],
        "rainfall_context": batch["rainfall"][:, 1 : 1 + context_length],
        "rainfall_target": batch["rainfall"][:, context_length + 1 : context_length + 2],
        "timestamps_context": batch["timestamps"][:, 1 : 1 + context_length],
        "target": delta.to_target_space(target_absolute_2, absolute_1),
        "base": absolute_1,
        "target_absolute": target_absolute_2,
    }
    weights = change_weight_map(target_absolute_2, absolute_1, wet_threshold_normalized, change_weight)
    if weights is not None:
        step_batch["pixel_weights"] = weights
    return step_batch


class ExponentialMovingAverage:
    """EMA of model parameters; the standard free sample-quality win in diffusion."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {name: param.detach().clone() for name, param in model.named_parameters()}
        self._backup: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap_in(self, model: torch.nn.Module) -> None:
        self._backup = {name: param.detach().clone() for name, param in model.named_parameters()}
        for name, param in model.named_parameters():
            param.copy_(self.shadow[name])

    @torch.no_grad()
    def swap_out(self, model: torch.nn.Module) -> None:
        if self._backup is None:
            raise RuntimeError("swap_out called without a prior swap_in")
        for name, param in model.named_parameters():
            param.copy_(self._backup[name])
        self._backup = None

    def state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self.shadow)


def autocast_context(device: torch.device, mode: str):
    if mode == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def run_epoch(
    model: DiffSparseV2Model,
    loader: DataLoader,
    device: torch.device,
    context_length: int,
    delta: DeltaSpec,
    wet_threshold_normalized: float,
    change_weight: float,
    clamp: bool,
    amp_mode: str,
    optimizer: torch.optim.Optimizer | None = None,
    ema: ExponentialMovingAverage | None = None,
    max_batches: int | None = None,
    grad_clip_norm: float | None = None,
    pushforward_fraction: float = 0.0,
    consistency: ConsistencyLoss | None = None,
    consistency_weight: float = 0.0,
    water_stats: dict[str, Any] | None = None,
    dem_stats: dict[str, Any] | None = None,
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
            model_batch = prepare_model_batch(
                batch, context_length, delta, wet_threshold_normalized, change_weight
            )
            if train and pushforward_fraction > 0 and float(torch.rand(1).item()) < pushforward_fraction:
                pushed = pushforward_batch(
                    model, batch, model_batch, context_length, delta,
                    wet_threshold_normalized, change_weight, clamp,
                )
                if pushed is not None:
                    model_batch = pushed
            if train:
                optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_mode):
                loss, diagnostics = model.training_step_loss(model_batch)
                if consistency is not None and consistency_weight > 0.0 and train:
                    tokens, spatial = model.encode_context(model_batch)
                    t0 = torch.zeros(model_batch["target"].shape[0], dtype=torch.long, device=device)
                    x_noisy = model.q_sample(model_batch["target"], t0)
                    x0_hat = model.denoise(x_noisy, t0, tokens, spatial)
                    absolute_hat = delta.to_absolute(x0_hat, model_batch["base"], clamp=False)
                    water_mean = float(water_stats["mean"])
                    water_std = float(water_stats["std"])
                    dem_mean = float(dem_stats["mean"])
                    dem_std = float(dem_stats["std"])
                    inundation_physical = absolute_hat * water_std + water_mean
                    elevation_physical = model_batch["dem"] * dem_std + dem_mean
                    loss = loss + consistency_weight * consistency(inundation_physical, elevation_physical)
            if train:
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                optimizer.step()
                if ema is not None:
                    ema.update(model)
            loss_sum += float(loss.detach().item())
            rmse_sum += float(diagnostics["x0_rmse"])
            batches += 1
    return {
        "loss": loss_sum / batches if batches else math.nan,
        "x0_rmse": rmse_sum / batches if batches else math.nan,
        "batches": float(batches),
    }


def run_deterministic_validation(
    model: DiffSparseV2Model,
    loader: DataLoader,
    device: torch.device,
    context_length: int,
    delta: DeltaSpec,
    wet_threshold_normalized: float,
    change_weight: float,
    amp_mode: str,
    val_seed: int,
    max_batches: int | None,
) -> dict[str, float]:
    """One-step denoising val_loss with fixed RNG (drives the LR scheduler only)."""

    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(val_seed)
    try:
        return run_epoch(
            model, loader, device, context_length, delta, wet_threshold_normalized,
            change_weight, clamp=False, amp_mode=amp_mode, optimizer=None, max_batches=max_batches,
        )
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


class RolloutValidator:
    """Cheap generative rollout validation for checkpoint selection.

    Rolls out a small fixed set of val-window tiles autoregressively
    (1 scenario, fixed RNG, EMA weights when enabled) and reports pooled
    rollout RMSE in normalized ABSOLUTE space regardless of the training
    target space -- directly the metric family the final evaluation reports.
    """

    def __init__(
        self,
        dataset,
        context_length: int,
        prediction_length: int,
        patch_size: int,
        num_windows: int,
        tiles_per_window: int,
        seed: int,
        device: torch.device,
        delta: DeltaSpec,
        clamp: bool,
    ) -> None:
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.patch_size = patch_size
        self.seed = seed
        self.device = device
        self.delta = delta
        self.clamp = clamp
        self.samples: list[dict[str, torch.Tensor]] = []

        num_windows = min(num_windows, len(dataset))
        window_indices = [round(i * (len(dataset) - 1) / max(num_windows - 1, 1)) for i in range(num_windows)]
        generator = torch.Generator().manual_seed(seed)
        for window_index in sorted(set(window_indices)):
            sample = dataset[window_index]
            height, width = sample["context_water_masked"].shape[-2:]
            for _ in range(tiles_per_window):
                y0 = int(torch.randint(0, height - patch_size + 1, (1,), generator=generator).item())
                x0 = int(torch.randint(0, width - patch_size + 1, (1,), generator=generator).item())
                crop = {
                    "context_water_masked": sample["context_water_masked"][..., y0 : y0 + patch_size, x0 : x0 + patch_size].clone(),
                    "context_water_true": sample["context_water_true"][..., y0 : y0 + patch_size, x0 : x0 + patch_size].clone(),
                    "sensor_mask": sample["sensor_mask"][..., y0 : y0 + patch_size, x0 : x0 + patch_size].clone(),
                    "dem": sample["dem"][..., y0 : y0 + patch_size, x0 : x0 + patch_size].clone(),
                    "rainfall": sample["rainfall"][..., y0 : y0 + patch_size, x0 : x0 + patch_size].clone(),
                    "timestamps": sample["timestamps"].clone(),
                    "target": sample["target"][..., y0 : y0 + patch_size, x0 : x0 + patch_size].clone(),
                }
                self.samples.append(crop)

    @torch.no_grad()
    def evaluate(self, model: DiffSparseV2Model) -> float:
        model.eval()
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        sq_sum = 0.0
        count = 0.0
        context = torch.stack([s["context_water_masked"] for s in self.samples]).to(self.device)
        context_true = torch.stack([s["context_water_true"] for s in self.samples]).to(self.device)
        mask = torch.stack([s["sensor_mask"] for s in self.samples]).to(self.device)
        dem = torch.stack([s["dem"] for s in self.samples]).to(self.device)
        rain = torch.stack([s["rainfall"] for s in self.samples]).to(self.device)
        timestamps = torch.stack([s["timestamps"] for s in self.samples]).to(self.device)
        target = torch.stack([s["target"] for s in self.samples]).to(self.device)
        batch_size = context.shape[0]
        base = self.delta.base_from_sample(context_true, mask)

        for step in range(self.prediction_length):
            model_batch = {
                "context_water_masked": context,
                "sensor_mask": mask,
                "dem": dem,
                "rainfall_context": rain[:, step : step + self.context_length],
                "rainfall_target": rain[:, step + self.context_length : step + self.context_length + 1],
                "timestamps_context": timestamps[:, step : step + self.context_length],
            }
            tokens, spatial = model.encode_context(model_batch)
            prediction = model.sample(
                tokens,
                spatial,
                (batch_size, 1, self.patch_size, self.patch_size),
                generator=generator,
                clip_x0=self.delta.clip_for_sampler(base, self.clamp),
            )
            absolute = self.delta.to_absolute(prediction, base, clamp=self.clamp)
            error = absolute[:, 0] - target[:, step]
            sq_sum += float(error.double().square().sum().item())
            count += float(error.numel())
            context = torch.cat([context[:, 1:], absolute], dim=1)
            base = absolute
        return math.sqrt(sq_sum / max(count, 1.0))


def first_batch_shape_report(
    model: DiffSparseV2Model,
    batch: dict[str, Any],
    context_length: int,
    delta: DeltaSpec,
    wet_threshold_normalized: float,
    change_weight: float,
) -> dict[str, Any]:
    model_batch = prepare_model_batch(batch, context_length, delta, wet_threshold_normalized, change_weight)
    target = model_batch["target"]
    timesteps = torch.zeros(target.shape[0], dtype=torch.long, device=target.device)
    x_noisy = model.q_sample(target, timesteps)
    tokens, spatial = model.encode_context(model_batch)
    prediction = model.denoise(x_noisy, timesteps, tokens, spatial)
    return {
        "context_water_masked": list(model_batch["context_water_masked"].shape),
        "sensor_mask": list(model_batch["sensor_mask"].shape),
        "dem": list(model_batch["dem"].shape),
        "rainfall_context": list(model_batch["rainfall_context"].shape),
        "rainfall_target": list(model_batch["rainfall_target"].shape),
        "timestamps_context": list(model_batch["timestamps_context"].shape),
        "target": list(target.shape),
        "target_space": delta.mode,
        "delta_scale_normalized": delta.scale,
        "pixel_weights": list(model_batch["pixel_weights"].shape) if "pixel_weights" in model_batch else None,
        "context_tokens": list(tokens.shape),
        "spatial_features": list(spatial.shape),
        "prediction": list(prediction.shape),
        "sensor_mask_mean": float(model_batch["sensor_mask"].mean().item()),
        "prediction_finite": bool(torch.isfinite(prediction).all().item()),
        "terminal_sqrt_alpha_cumprod": float(model.sqrt_alpha_cumprod[-1].item()),
    }


def load_delta_stats(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as file:
        stats = json.load(file)
    if stats.get("version") != "diff_sparse_v2_train_delta_stats":
        raise ValueError(f"Unexpected delta stats version in {path}: {stats.get('version')!r}")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Train DIFF-SPARSE v2 on FloodCastBench.")
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
    parser.add_argument("--delta-stats-json", type=Path, help="Precomputed train delta stats (delta mode)")
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
    print(f"diffusion: steps={config.get('diffusion', {}).get('steps', 40)} "
          f"beta=[{config.get('diffusion', {}).get('beta_start', 1e-4)}, "
          f"{config.get('diffusion', {}).get('beta_end', 1.0)}]")

    stats = load_or_compute_stats(config, args.stats_json)
    water_stats = stats["channels"]["water"]
    dem_stats = stats["channels"]["dem"]

    prediction_config = config.get("prediction", {})
    target_space = str(prediction_config.get("target", "delta"))
    delta_stats_path = args.delta_stats_json or prediction_config.get("delta_stats_json")
    delta_stats = load_delta_stats(Path(delta_stats_path) if delta_stats_path else None)
    delta = DeltaSpec(target_space, water_stats, delta_stats)
    gamma_wet = float(prediction_config.get("wet_threshold_m", 0.001))
    wet_threshold_normalized = delta.floor_absolute + gamma_wet / float(water_stats["std"])
    change_weight = float(config.get("loss", {}).get("change_weight", 3.0))
    print(f"prediction target space: {delta.mode} (scale={delta.scale:.6g}) change_weight={change_weight}")

    root = path_from_config(config, "dataset_root")
    train_dataset = build_diff_sparse_v2_dataset(root, config, split="train", normalization_stats=stats)
    val_dataset = build_diff_sparse_v2_dataset(
        root, config, split="val", normalization_stats=stats, patch_mode="random"
    )
    device = resolve_device(config.get("training", {}).get("device", "auto"))
    model = DiffSparseV2Model(config).to(device)

    training_config = config.get("training", {})
    amp_mode = str(training_config.get("amp", "bf16")).lower()
    clamp = bool(config.get("evaluation", {}).get("clip_x0_physical", True))

    if args.dry_run_config:
        loader = build_loader(train_dataset, config, shuffle=False, num_workers=0)
        batch = move_batch_to_device(next(iter(loader)), device)
        with torch.no_grad():
            shapes = first_batch_shape_report(
                model, batch, train_dataset.context_length, delta, wet_threshold_normalized, change_weight
            )
            loss, diagnostics = model.training_step_loss(
                prepare_model_batch(
                    batch, train_dataset.context_length, delta, wet_threshold_normalized, change_weight
                )
            )
        report = {
            "device": str(device),
            "train_windows": len(train_dataset),
            "val_windows": len(val_dataset),
            "augmentation": train_dataset.augmentation,
            "missing_rate_range": train_dataset.missing_rate_range,
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "first_batch_shapes": shapes,
            "dry_run_loss": float(loss.item()),
            "dry_run_diagnostics": diagnostics,
            "clip_x0_floor_normalized": delta.floor_absolute,
            "amp": amp_mode,
            "writes": "none",
            "scientific_status": SCIENTIFIC_STATUS,
        }
        print("=== DIFF-SPARSE V2 DRY RUN ===")
        print(json.dumps(report, indent=2))
        return 0

    experiment_dir, checkpoint_dir, log_dir = create_run_dirs(config)
    print(f"experiment_dir: {experiment_dir}")
    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"log_dir: {log_dir}")
    save_yaml(config, experiment_dir / "config.yaml")
    save_json(stats, experiment_dir / "normalization_stats.json")
    if delta_stats is not None:
        save_json(delta_stats, experiment_dir / "delta_stats.json")

    train_loader = build_loader(train_dataset, config, shuffle=True)
    val_loader = build_loader(val_dataset, config, shuffle=False, num_workers=0)
    print(f"device: {device}")
    print(f"train_windows: {len(train_dataset)} val_windows: {len(val_dataset)}")
    print(f"model_parameters: {sum(parameter.numel() for parameter in model.parameters())}")
    print(f"augmentation: {train_dataset.augmentation} missing_rate_range: {train_dataset.missing_rate_range}")
    print(f"amp: {amp_mode}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training_config.get("lr_factor", 0.5)),
        patience=int(training_config.get("lr_patience", 15)),
    )
    ema_decay = float(training_config.get("ema_decay", 0.999))
    ema = ExponentialMovingAverage(model, ema_decay) if ema_decay > 0 else None
    val_seed = int(training_config.get("val_seed", 1234))
    grad_clip_norm = training_config.get("grad_clip_norm")
    epochs = int(training_config.get("epochs", 300))
    rollout_val_every = int(training_config.get("rollout_val_every", 5))
    early_stop_patience = training_config.get("early_stop_patience")
    early_stop_patience = int(early_stop_patience) if early_stop_patience is not None else None
    pushforward_fraction = float(training_config.get("pushforward_fraction", 0.25))
    consistency_weight = float(training_config.get("consistency_loss_weight", 0.0))
    consistency = ConsistencyLoss().to(device) if consistency_weight > 0.0 else None

    val_full_dataset = build_diff_sparse_v2_dataset(
        root, config, split="val", normalization_stats=stats, patch_mode="full"
    )
    rollout_validator = RolloutValidator(
        val_full_dataset,
        context_length=train_dataset.context_length,
        prediction_length=train_dataset.prediction_length,
        patch_size=train_dataset.patch_size,
        num_windows=int(training_config.get("rollout_val_windows", 2)),
        tiles_per_window=int(training_config.get("rollout_val_tiles_per_window", 4)),
        seed=val_seed,
        device=device,
        delta=delta,
        clamp=clamp,
    )
    print(f"rollout_validator tiles: {len(rollout_validator.samples)} (every {rollout_val_every} epochs)")
    print(f"clip floor (normalized 0m depth): {delta.floor_absolute:.6f} pushforward_fraction: {pushforward_fraction}")

    metrics_path = experiment_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=METRIC_FIELDS).writeheader()

    first_batch = move_batch_to_device(
        next(iter(build_loader(train_dataset, config, shuffle=False, num_workers=0))), device
    )
    with torch.no_grad():
        shapes = first_batch_shape_report(
            model, first_batch, train_dataset.context_length, delta, wet_threshold_normalized, change_weight
        )
    print("first_batch_shapes:")
    print(json.dumps(shapes, indent=2))

    best_value = math.inf
    best_epoch = None
    epochs_since_best = 0
    stopped_early_at = None
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            train_dataset.context_length,
            delta,
            wet_threshold_normalized,
            change_weight,
            clamp=clamp,
            amp_mode=amp_mode,
            optimizer=optimizer,
            ema=ema,
            max_batches=args.max_train_batches,
            grad_clip_norm=float(grad_clip_norm) if grad_clip_norm is not None else None,
            pushforward_fraction=pushforward_fraction,
            consistency=consistency,
            consistency_weight=consistency_weight,
            water_stats=water_stats,
            dem_stats=dem_stats,
        )
        val_metrics = run_deterministic_validation(
            model,
            val_loader,
            device,
            train_dataset.context_length,
            delta,
            wet_threshold_normalized,
            change_weight,
            amp_mode=amp_mode,
            val_seed=val_seed,
            max_batches=args.max_val_batches,
        )
        scheduler.step(val_metrics["loss"])

        rollout_val_rmse = None
        if epoch % rollout_val_every == 0 or epoch == epochs:
            if ema is not None:
                ema.swap_in(model)
            try:
                rollout_val_rmse = rollout_validator.evaluate(model)
            finally:
                if ema is not None:
                    ema.swap_out(model)

        elapsed = time.perf_counter() - start
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_x0_rmse": train_metrics["x0_rmse"],
            "val_loss": val_metrics["loss"],
            "val_x0_rmse": val_metrics["x0_rmse"],
            "rollout_val_rmse": rollout_val_rmse if rollout_val_rmse is not None else "",
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": elapsed,
        }
        with metrics_path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=METRIC_FIELDS).writerow(row)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict() if ema is not None else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "normalization_stats": stats,
            "delta_stats": delta_stats,
            "metrics": {key: value for key, value in row.items()},
            "selection_metric": "rollout_val_rmse",
            "scientific_status": SCIENTIFIC_STATUS,
        }
        torch.save(checkpoint, checkpoint_dir / "checkpoint_last.pth")
        if rollout_val_rmse is not None:
            if rollout_val_rmse < best_value:
                best_value = rollout_val_rmse
                best_epoch = epoch
                epochs_since_best = 0
                torch.save(checkpoint, checkpoint_dir / "checkpoint_best.pth")
            else:
                epochs_since_best += rollout_val_every
        rollout_text = f" rollout_val_rmse={rollout_val_rmse:.6f}" if rollout_val_rmse is not None else ""
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.6f} val_loss={row['val_loss']:.6f} "
            f"val_x0_rmse={row['val_x0_rmse']:.6f}{rollout_text} lr={row['learning_rate']:.2e} "
            f"elapsed={elapsed:.1f}s"
        )

        if early_stop_patience is not None and epochs_since_best >= early_stop_patience:
            stopped_early_at = epoch
            print(
                f"early stop at epoch {epoch}: no rollout_val_rmse improvement for "
                f">= {early_stop_patience} epochs (best epoch {best_epoch})"
            )
            break

    save_json(
        {
            "experiment_dir": str(experiment_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "log_dir": str(log_dir),
            "epochs": epochs,
            "stopped_early_at": stopped_early_at,
            "selection_metric": "rollout_val_rmse",
            "best_epoch": best_epoch,
            "best_selection_metric": best_value if best_value != math.inf else None,
            "metrics_csv": str(metrics_path),
            "missing_rate": float(config.get("masking", {}).get("missing_rate", 0.0)),
            "missing_rate_range": train_dataset.missing_rate_range,
            "augmentation": train_dataset.augmentation,
            "mask_mode": str(config.get("masking", {}).get("mask_mode", "noise")),
            "diffusion_beta_end": float(config.get("diffusion", {}).get("beta_end", 1.0)),
            "diffusion_steps": int(config.get("diffusion", {}).get("steps", 40)),
            "terminal_sqrt_alpha_cumprod": shapes["terminal_sqrt_alpha_cumprod"],
            "prediction_target": delta.mode,
            "delta_scale_normalized": delta.scale,
            "change_weight": change_weight,
            "pushforward_fraction": pushforward_fraction,
            "ema_decay": ema_decay,
            "amp": amp_mode,
            "clip_x0_floor_normalized": delta.floor_absolute,
            "consistency_loss_weight": consistency_weight,
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
