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
from models.deterministic_twin import build_v2_family_model  # noqa: E402
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
    own dense previous prediction).

    The scale is REGIME-AWARE (final design after the 2026-07-09 pilots):
      - base = OBSERVED last frame (standard training batches, rollout
        step 1): per-pixel scale via scale_for_observed_base(). Observed
        pixels carry the temporal-delta scale self.scale (~0.0024 normalized;
        physical delta std 0.0007 m vs water std 0.29 m, ~400x smaller than
        the absolute field); UNOBSERVED pixels carry the marginal field scale
        (exactly 1.0 by construction of standardization, since base there is
        the train-mean fill and the residual is the normalized absolute
        value). Per pixel this interpolates between V1's absolute-space
        prediction (unobserved) and V2's delta-space prediction (observed).
        Dense input (mask == 1) reduces exactly to the scalar delta scale.
      - base = the model's own dense prediction (pushforward step-2 batches,
        rollout steps >= 2): scalar delta scale. The pushforward LOSS is
        additionally restricted to observed pixels (see pushforward_batch).

    Evidence behind each choice (12-14-epoch pilots, m50/m95):
      - No per-pixel scale at step 1: O(1) reconstruction residuals at masked
        pixels divided by the tiny delta std -> O(400) targets, divergence
        (val_x0_rmse in the hundreds, NaN sampling) across seeds.
      - Per-pixel scale at rollout steps >= 2 as well ("uniform" variant):
        every rollout step becomes high-gain (O(1) moves) at masked pixels;
        the model double-counts its own reconstruction and the rollout RMSE
        GROWS across training (m50: 3.6 -> 11.3) while one-step val improves.
        The scalar delta scale keeps steps >= 2 low-gain and the rollout
        stable (m50 ~1.8 flat).
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
            # Bound for SYNTHETIC pushforward step-2 targets only (never real
            # targets): no correction larger than 1.5x the largest physical
            # frame-to-frame delta ever observed in training is a legitimate
            # training signal -- early-training pushforward bases can otherwise
            # produce arbitrarily large residuals in delta units.
            abs_max = delta_stats.get("delta_abs_max_physical")
            self.pushforward_target_bound = (
                1.5 * float(abs_max) / float(delta_stats["delta_std_physical"]) if abs_max else None
            )
        else:
            self.scale = 1.0
            self.pushforward_target_bound = None

    def base_from_sample(self, context_water_true: torch.Tensor, sensor_mask: torch.Tensor) -> torch.Tensor:
        """Observed base frame: [.., H, W] -> [.., 1, H, W] (mean fill = 0 normalized)."""

        last_true = context_water_true[..., -1, :, :].unsqueeze(-3)
        return last_true * sensor_mask

    def scale_for_observed_base(self, sensor_mask: torch.Tensor) -> torch.Tensor | None:
        """Per-pixel target scale when the base is the OBSERVED last context
        frame (standard training branch, rollout step 1). Observed pixels
        carry the temporal-delta scale; unobserved pixels carry the marginal
        field scale (1.0 normalized). None in absolute mode (scale unused)."""

        if self.mode == "absolute":
            return None
        return sensor_mask * self.scale + (1.0 - sensor_mask)

    def _scale(self, scale: torch.Tensor | float | None) -> torch.Tensor | float:
        return self.scale if scale is None else scale

    def to_target_space(
        self, absolute: torch.Tensor, base: torch.Tensor, scale: torch.Tensor | float | None = None
    ) -> torch.Tensor:
        if self.mode == "absolute":
            return absolute
        return (absolute - base) / self._scale(scale)

    def to_absolute(
        self,
        prediction: torch.Tensor,
        base: torch.Tensor,
        clamp: bool,
        bound_ceiling: bool = False,
        scale: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        if self.mode == "absolute":
            absolute = prediction
        else:
            absolute = base + self._scale(scale) * prediction
        if clamp:
            absolute = absolute.clamp(min=self.floor_absolute)
        if bound_ceiling:
            # Numerical safety net only (not a physical assumption): bounds
            # single-shot/no-grad reconstructions (pushforward) against
            # runaway extrapolation. See __init__ docstring.
            absolute = absolute.clamp(max=self.ceiling_absolute)
        return absolute

    def clip_for_sampler(self, base: torch.Tensor, enabled: bool, scale: torch.Tensor | float | None = None):
        """clip_x0 tuple for model.sample(): scalar floor in absolute mode,
        per-pixel tensor floor in delta mode."""

        if not enabled:
            return None
        if self.mode == "absolute":
            return (self.floor_absolute, None)
        return ((self.floor_absolute - base) / self._scale(scale), None)


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
    target_scale = delta.scale_for_observed_base(batch["sensor_mask"])
    target_absolute = batch["target"][:, 0:1]
    model_batch = {
        "context_water_masked": batch["context_water_masked"],
        "sensor_mask": batch["sensor_mask"],
        "dem": batch["dem"],
        "rainfall_context": batch["rainfall"][:, :context_length],
        "rainfall_target": batch["rainfall"][:, context_length : context_length + 1],
        "timestamps_context": batch["timestamps"][:, :context_length],
        "target": delta.to_target_space(target_absolute, base, scale=target_scale),
        "base": base,
        "target_absolute": target_absolute,
        "target_scale": target_scale,
    }
    if "manning" in batch:
        model_batch["manning"] = batch["manning"]
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
        # (2026-07-06 pilot, epoch 52). The step-1 prediction lives in the
        # observed-base target space, so it is reconstructed with the same
        # per-pixel scale.
        absolute_1 = delta.to_absolute(
            prediction, model_batch["base"], clamp=clamp, bound_ceiling=True,
            scale=model_batch.get("target_scale"),
        )

    context_2 = torch.cat([batch["context_water_masked"][:, 1:], absolute_1], dim=1)
    target_absolute_2 = batch["target"][:, 1:2]
    # Step-2 base is a model prediction -> scalar delta scale, matching
    # rollout steps >= 2 (see DeltaSpec docstring: the per-pixel scale at
    # steps >= 2 made the rollout high-gain at masked pixels and diverged,
    # 2026-07-09 pilot 2). The bound is a safety net for observed-pixel
    # outliers early in training.
    target_2 = delta.to_target_space(target_absolute_2, absolute_1)
    if delta.pushforward_target_bound is not None:
        bound = delta.pushforward_target_bound
        target_2 = target_2.clamp(min=-bound, max=bound)
    step_batch = {
        "context_water_masked": context_2,
        "sensor_mask": batch["sensor_mask"],
        "dem": batch["dem"],
        "rainfall_context": batch["rainfall"][:, 1 : 1 + context_length],
        "rainfall_target": batch["rainfall"][:, context_length + 1 : context_length + 2],
        "timestamps_context": batch["timestamps"][:, 1 : 1 + context_length],
        "target": target_2,
        "base": absolute_1,
        "target_absolute": target_absolute_2,
        "target_scale": None,
    }
    if "manning" in batch:
        step_batch["manning"] = batch["manning"]
    weights = change_weight_map(target_absolute_2, absolute_1, wet_threshold_normalized, change_weight)
    if weights is None:
        weights = torch.ones_like(target_2)
    # Pushforward trains only OBSERVED pixels: at masked pixels the residual
    # is the model's reconstruction error, which is not representable at the
    # delta scale (it saturated the bound and spiked train_loss ~10^3 at
    # m50/m95 the moment pushforward activated, 2026-07-09 pilot 1). At
    # observed pixels the residual is genuinely delta-scaled -- exactly the
    # exposure-bias signal pushforward exists to provide. Dense (mask == 1)
    # is unaffected.
    step_batch["pixel_weights"] = weights * batch["sensor_mask"]
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
    skipped = 0
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
                    absolute_hat = delta.to_absolute(
                        x0_hat, model_batch["base"], clamp=False, scale=model_batch.get("target_scale")
                    )
                    water_mean = float(water_stats["mean"])
                    water_std = float(water_stats["std"])
                    dem_mean = float(dem_stats["mean"])
                    dem_std = float(dem_stats["std"])
                    inundation_physical = absolute_hat * water_std + water_mean
                    elevation_physical = model_batch["dem"] * dem_std + dem_mean
                    loss = loss + consistency_weight * consistency(inundation_physical, elevation_physical)
            # Skip (never train on) non-finite batches: one tail-event batch
            # must not poison the weights and kill a 300-epoch run (observed
            # 2026-07-09: seed42 dense NaN from epoch 1). Counted and reported;
            # an epoch that skips too much fails loudly below.
            if not bool(torch.isfinite(loss.detach()).item()):
                skipped += 1
                if train:
                    optimizer.zero_grad(set_to_none=True)
                continue
            if train:
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                    if not bool(torch.isfinite(total_norm).item()):
                        skipped += 1
                        optimizer.zero_grad(set_to_none=True)
                        continue
                optimizer.step()
                if ema is not None:
                    ema.update(model)
            loss_sum += float(loss.detach().item())
            rmse_sum += float(diagnostics["x0_rmse"])
            batches += 1
    total = batches + skipped
    if skipped and total and skipped > 0.25 * total:
        raise RuntimeError(
            f"{skipped}/{total} batches skipped for non-finite loss/gradients; "
            "training is numerically unstable, aborting instead of silently degrading"
        )
    return {
        "loss": loss_sum / batches if batches else math.nan,
        "x0_rmse": rmse_sum / batches if batches else math.nan,
        "batches": float(batches),
        "skipped": float(skipped),
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
                if "manning" in sample:
                    crop["manning"] = sample["manning"][..., y0 : y0 + patch_size, x0 : x0 + patch_size].clone()
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
        manning = (
            torch.stack([s["manning"] for s in self.samples]).to(self.device)
            if "manning" in self.samples[0] else None
        )
        batch_size = context.shape[0]
        base = self.delta.base_from_sample(context_true, mask)

        for step in range(self.prediction_length):
            # Step 0: base is the observed masked frame -> per-pixel scale.
            # Steps >= 1: base is the model's own dense prediction -> scalar
            # delta scale (low-gain at masked pixels; the per-pixel scale at
            # steps >= 1 made the rollout diverge, 2026-07-09 pilot 2).
            scale = self.delta.scale_for_observed_base(mask) if step == 0 else None
            model_batch = {
                "context_water_masked": context,
                "sensor_mask": mask,
                "dem": dem,
                "rainfall_context": rain[:, step : step + self.context_length],
                "rainfall_target": rain[:, step + self.context_length : step + self.context_length + 1],
                "timestamps_context": timestamps[:, step : step + self.context_length],
            }
            if manning is not None:
                model_batch["manning"] = manning
            tokens, spatial = model.encode_context(model_batch)
            prediction = model.sample(
                tokens,
                spatial,
                (batch_size, 1, self.patch_size, self.patch_size),
                generator=generator,
                clip_x0=self.delta.clip_for_sampler(base, self.clamp, scale=scale),
            )
            absolute = self.delta.to_absolute(prediction, base, clamp=self.clamp, scale=scale)
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
    parser.add_argument("--early-stop-patience", type=int, help="Override training.early_stop_patience (WP6 extended-budget reruns)")
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
    if getattr(args, "early_stop_patience", None) is not None:
        config.setdefault("training", {})["early_stop_patience"] = int(args.early_stop_patience)
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
    model = build_v2_family_model(config).to(device)
    print(f"model class: {type(model).__name__}")

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
    # Pushforward needs a base worth correcting: with an untrained model the
    # step-1 reconstruction is garbage and the synthetic step-2 residuals are
    # training noise. Plain one-step training only for the first epochs.
    pushforward_warmup_epochs = int(training_config.get("pushforward_warmup_epochs", 10))
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
    print(
        f"clip floor (normalized 0m depth): {delta.floor_absolute:.6f} "
        f"pushforward_fraction: {pushforward_fraction} (warmup {pushforward_warmup_epochs} epochs)"
    )

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
    total_skipped_batches = 0
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
            pushforward_fraction=pushforward_fraction if epoch > pushforward_warmup_epochs else 0.0,
            consistency=consistency,
            consistency_weight=consistency_weight,
            water_stats=water_stats,
            dem_stats=dem_stats,
        )
        total_skipped_batches += int(train_metrics.get("skipped", 0))
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
            except FloatingPointError as error:
                # A non-finite sampling state at a validation step must not
                # kill a multi-day run: record inf (never selected as best;
                # early stopping will end a permanently broken run).
                print(f"WARNING epoch {epoch}: rollout validation non-finite ({error}); recording inf")
                rollout_val_rmse = math.inf
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
        skipped_epoch = int(train_metrics.get("skipped", 0)) + int(val_metrics.get("skipped", 0))
        skipped_text = f" skipped_batches={skipped_epoch}" if skipped_epoch else ""
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.6f} val_loss={row['val_loss']:.6f} "
            f"val_x0_rmse={row['val_x0_rmse']:.6f}{rollout_text} lr={row['learning_rate']:.2e} "
            f"elapsed={elapsed:.1f}s{skipped_text}"
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
            "pushforward_warmup_epochs": pushforward_warmup_epochs,
            "total_skipped_batches": total_skipped_batches,
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
