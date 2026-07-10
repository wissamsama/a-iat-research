from __future__ import annotations

"""Deterministic twin of DIFF-SPARSE V2 (paper master plan WP1).

THE controlled baseline for the paper's central question: does generative
(diffusion) modeling earn its keep under sparse observation, or does a
deterministic network of identical capacity suffice?

Design: the twin IS the V2 network — same TemporalContextEncoder, same
SpatialContextEncoder, same UNet2DConditionModel, same parameter count to the
last weight — evaluated as a deterministic function: the noise input channel
is fixed to zeros and the timestep to 0 (a constant bias through the timestep
embedding), so the output depends on the conditioning only. Training is plain
weighted regression (MSE) on the same regime-aware delta target, with the
same pixel weights, EMA, pushforward and clamps as V2. Inference is a single
forward pass; `sample()` ignores the RNG and returns the same field every
call, so the multi-scenario protocol degenerates to 1 scenario and the
empirical CRPS to the MAE (the evaluator's existing point-forecast
convention).

Everything that differs between twin and V2 in any experiment is therefore
exactly one bit: the presence of the diffusion process. This is the parity
the controlled comparison requires (master plan §4-WP1, rules R2/R6).
"""

from typing import Any, Callable

import torch
import torch.nn.functional as F

from models.diff_sparse_v2 import DiffSparseV2Model, _apply_clip


class DeterministicTwinModel(DiffSparseV2Model):
    def denoise(
        self,
        x_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        context_tokens: torch.Tensor,
        spatial_features: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic forward: the sample input and timestep are ignored
        (zeros / t=0), making the UNet a pure function of the conditioning.
        Keeping the signature lets the V2 trainer's pushforward branch and the
        V2 evaluator's rollout drive the twin unchanged."""

        zeros = torch.zeros_like(x_noisy)
        t0 = torch.zeros(x_noisy.shape[0], device=x_noisy.device, dtype=torch.long)
        return super().denoise(zeros, t0, context_tokens, spatial_features)

    def training_step_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """Plain weighted regression on the same target/pixel-weight contract
        as V2's Algorithm-1 loss, minus the diffusion (no noise, no timestep
        weighting)."""

        target = batch["target"]
        if target.ndim != 4 or target.shape[1] != 1:
            raise ValueError(f"Expected training target [B, 1, H, W], got {tuple(target.shape)}")
        tokens, spatial = self.encode_context(batch)
        prediction = self.denoise(torch.zeros_like(target), None, tokens, spatial)  # type: ignore[arg-type]
        squared_error = F.mse_loss(prediction, target, reduction="none")
        pixel_weights = batch.get("pixel_weights")
        if pixel_weights is not None:
            squared_error = squared_error * (pixel_weights / pixel_weights.mean().clamp(min=1e-12))
        loss = squared_error.mean()
        diagnostics = {
            "x0_rmse": float(torch.sqrt(F.mse_loss(prediction.detach(), target.detach())).item()),
            "timestep_mean": 0.0,
            "pred_finite": float(torch.isfinite(prediction).all().item()),
            "target_finite": float(torch.isfinite(target).all().item()),
        }
        return loss, diagnostics

    @torch.no_grad()
    def sample(
        self,
        context_tokens: torch.Tensor,
        spatial_features: torch.Tensor,
        shape: tuple[int, ...],
        generator: torch.Generator | None = None,
        denoiser: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        clip_x0: tuple[float | torch.Tensor | None, float | torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        """Single deterministic forward; `generator` is accepted for interface
        parity and ignored. Same physical clip contract as V2's sampler."""

        device = spatial_features.device
        zeros = torch.zeros(shape, device=device, dtype=spatial_features.dtype)
        if denoiser is not None:
            timesteps = torch.zeros(shape[0], device=device, dtype=torch.long)
            prediction = denoiser(zeros, timesteps)
        else:
            prediction = self.denoise(zeros, None, context_tokens, spatial_features)  # type: ignore[arg-type]
        if clip_x0 is not None:
            prediction = _apply_clip(prediction, clip_x0[0], clip_x0[1])
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Non-finite deterministic prediction")
        return prediction


def build_v2_family_model(config: dict[str, Any]) -> DiffSparseV2Model:
    """Factory shared by the V2 trainer/evaluator: dispatches on model.name so
    the exact same tools drive both the diffusion model and its twin."""

    name = str(config.get("model", {}).get("name", "diff_sparse_v2")).lower()
    if name == "diff_sparse_v2":
        return DiffSparseV2Model(config)
    if name == "deterministic_twin":
        return DeterministicTwinModel(config)
    raise ValueError(f"Unknown model.name {name!r} (expected diff_sparse_v2 or deterministic_twin)")
