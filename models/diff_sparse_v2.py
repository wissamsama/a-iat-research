from __future__ import annotations

import math
from typing import Any, Callable

import torch
from torch import nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel


DAY_SECONDS = 86400.0
HOUR_FREQUENCIES = (1.0, 2.0, 4.0)


def _groups(channels: int, preferred: int) -> int:
    for count in (preferred, 8, 4, 2):
        if channels % count == 0:
            return count
    return 1


def _apply_clip(
    x: torch.Tensor,
    floor: float | torch.Tensor | None,
    ceiling: float | torch.Tensor | None,
) -> torch.Tensor:
    """Clamp with scalar or per-pixel tensor bounds.

    Tensor bounds are needed by delta-prediction mode: the physical floor
    (absolute depth >= 0) maps to a per-pixel delta floor (0 - base)/scale.
    """

    if floor is not None:
        if torch.is_tensor(floor):
            x = torch.maximum(x, floor.to(dtype=x.dtype, device=x.device))
        else:
            x = x.clamp(min=float(floor))
    if ceiling is not None:
        if torch.is_tensor(ceiling):
            x = torch.minimum(x, ceiling.to(dtype=x.dtype, device=x.device))
        else:
            x = x.clamp(max=float(ceiling))
    return x


def _temporal_down_block_output_size(input_size: int, num_blocks: int) -> int:
    """Spatial size after `num_blocks` TemporalDownBlocks (two unpadded 3x3 convs
    then a 2x2 avgpool per block), matching hidden_state_net.py's DownSampleBlock
    exactly: each conv shrinks by 2, each pool halves via floor((size-2)/2)+1."""

    size = input_size
    for _ in range(num_blocks):
        size = size - 2 - 2
        size = (size - 2) // 2 + 1
    if size < 1:
        raise ValueError(
            f"patch size {input_size} is too small for {num_blocks} TemporalDownBlocks "
            f"(would shrink to {size})"
        )
    return size


class TemporalDownBlock(nn.Module):
    """Reference hidden_state_net.py DownSampleBlock: unpadded Conv3d pair + AvgPool3d(1,2,2).

    Kernel (1,3,3) convolves only spatially, never mixing across the context_length
    ("depth"/time) axis -- each of the T context frames is downsampled independently
    before flattening into a per-timestep token.
    """

    def __init__(self, in_channels: int, out_channels: int, groups: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3))
        self.norm1 = nn.GroupNorm(_groups(out_channels, groups), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3))
        self.norm2 = nn.GroupNorm(_groups(out_channels, groups), out_channels)
        self.pool = nn.AvgPool3d((1, 2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.norm1(self.conv1(x)))
        x = F.silu(self.norm2(self.conv2(x)))
        return self.pool(x)


class TemporalContextEncoder(nn.Module):
    """V1's reference-faithful temporal token encoder (hidden_state_net.py port).

    Produces one heavily-pooled spatial summary token per context timestep, fed as
    encoder_hidden_states to the UNet's cross-attention. Kept unchanged in V2 as
    the DIFF-SPARSE identity mechanism; V2 adds a *separate* pixel-aligned spatial
    pathway (SpatialContextEncoder) on top, it does not modify this one.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_config = config.get("model", {})
        dataset_config = config.get("dataset", {})
        self.context_length = int(dataset_config.get("context_length", 12))
        self.include_dem = bool(dataset_config.get("include_dem", True))
        self.include_rainfall = bool(dataset_config.get("include_rainfall", True))
        self.include_manning = bool(dataset_config.get("include_manning", False))
        self.include_covariates = bool(dataset_config.get("include_covariates", True))
        self.embedding_dim = int(model_config.get("context_embedding_dim", 64))
        groups = int(model_config.get("context_groupnorm_groups", 8))
        conv_channels = [int(c) for c in model_config.get("context_conv_channels", [16, 32, 64])]
        patch_size = int(dataset_config.get("patch_size", 64))

        in_channels = (
            1 + (1 if self.include_dem else 0) + 1 + (1 if self.include_rainfall else 0)
            + (1 if self.include_manning else 0)
        )
        blocks = []
        current = in_channels
        for out_channels in conv_channels:
            blocks.append(TemporalDownBlock(current, out_channels, groups))
            current = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.output_conv = nn.Conv3d(current, 1, kernel_size=1)

        self.covariate_features = len(HOUR_FREQUENCIES) * 2 + 1 if self.include_covariates else 0
        self.covariate_time_scale = float(dataset_config.get("covariate_time_scale", 864000.0))

        # Registered at construction time (never lazily on first forward): a
        # lazily-built layer is silently excluded from optimizers constructed
        # before the first forward pass -- see the V1 token_linear regression test.
        pooled_size = _temporal_down_block_output_size(patch_size, len(conv_channels))
        token_features = pooled_size * pooled_size + self.covariate_features
        self.token_linear = nn.Linear(token_features, self.embedding_dim)

    def encode_covariates(self, timestamps: torch.Tensor) -> torch.Tensor:
        """timestamps [B, T] seconds -> [B, T, covariate_features], raw (no learned projection)."""

        hour_fraction = (timestamps % DAY_SECONDS) / DAY_SECONDS
        features = [timestamps / self.covariate_time_scale]
        for frequency in HOUR_FREQUENCIES:
            angle = 2.0 * math.pi * frequency * hour_fraction
            features.extend([torch.sin(angle), torch.cos(angle)])
        return torch.stack(features, dim=-1)

    def forward(
        self,
        context_water_masked: torch.Tensor,
        sensor_mask: torch.Tensor,
        dem: torch.Tensor,
        rainfall_context: torch.Tensor | None,
        timestamps_context: torch.Tensor | None,
        manning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, height, width = context_water_masked.shape
        parts = [context_water_masked.unsqueeze(1)]
        if self.include_dem:
            parts.append(dem.unsqueeze(2).expand(batch, 1, length, height, width))
        parts.append(sensor_mask.unsqueeze(2).expand(batch, 1, length, height, width))
        if self.include_rainfall:
            if rainfall_context is None:
                raise ValueError("rainfall_context is required when include_rainfall is true")
            parts.append(rainfall_context.unsqueeze(1))
        if self.include_manning:
            if manning is None:
                raise ValueError("manning is required when include_manning is true")
            parts.append(manning.unsqueeze(2).expand(batch, 1, length, height, width))
        x = torch.cat(parts, dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.output_conv(x)
        tokens = x.permute(0, 2, 1, 3, 4).reshape(batch, length, -1)

        if self.include_covariates:
            if timestamps_context is None:
                raise ValueError("timestamps_context is required when include_covariates is true")
            tokens = torch.cat([tokens, self.encode_covariates(timestamps_context)], dim=-1)

        return self.token_linear(tokens)


class SpatialContextEncoder(nn.Module):
    """V2's pixel-aligned conditioning pathway (new vs the paper).

    A shallow padded 2D conv stack over the full per-pixel context stack
    [masked water history, sensor mask, DEM, context rainfall, target-step
    rainfall], producing `spatial_context_channels` feature maps at full
    resolution that are concatenated with x_noisy at the UNet input. Motivated
    by direct in-repo evidence: a concat conditioning pathway converges to
    dense one-step val_loss ~0.005 where attention-only conditioning stalls
    around ~3.7 for 70+ epochs -- pure temporal-token cross-attention gives the
    UNet no per-pixel access to the context at all, which caps dense accuracy
    and starves the flood-front localization that path/propagation IoU measures.

    Target-step rainfall is the rain falling *during* the predicted interval --
    exogenous forcing (same standing as FNO+'s rainfall input), and the direct
    causal driver of newly inundated pixels.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_config = config.get("model", {})
        dataset_config = config.get("dataset", {})
        self.context_length = int(dataset_config.get("context_length", 12))
        self.include_dem = bool(dataset_config.get("include_dem", True))
        self.include_rainfall = bool(dataset_config.get("include_rainfall", True))
        self.include_manning = bool(dataset_config.get("include_manning", False))
        self.include_target_rainfall = bool(model_config.get("include_target_rainfall", True))
        self.include_context_deltas = bool(model_config.get("include_context_deltas", True))
        self.out_channels = int(model_config.get("spatial_context_channels", 16))
        hidden = int(model_config.get("spatial_context_hidden_channels", 32))
        groups = int(model_config.get("context_groupnorm_groups", 8))

        in_channels = self.context_length + 1  # masked water history + sensor mask
        if self.include_dem:
            in_channels += 1
        if self.include_manning:
            in_channels += 1
        if self.include_rainfall:
            in_channels += self.context_length
        if self.include_target_rainfall:
            in_channels += 1
        if self.include_context_deltas:
            # Consecutive-frame water differences: the current inundation trend,
            # the most direct per-pixel predictor of next-step new inundation.
            in_channels += self.context_length - 1

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(hidden, groups), hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(hidden, groups), hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, self.out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        context_water_masked: torch.Tensor,
        sensor_mask: torch.Tensor,
        dem: torch.Tensor,
        rainfall_context: torch.Tensor | None,
        rainfall_target: torch.Tensor | None,
        manning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [context_water_masked, sensor_mask]
        if self.include_dem:
            parts.append(dem)
        if self.include_manning:
            if manning is None:
                raise ValueError("manning is required when include_manning is true")
            parts.append(manning)
        if self.include_rainfall:
            if rainfall_context is None:
                raise ValueError("rainfall_context is required when include_rainfall is true")
            parts.append(rainfall_context)
        if self.include_target_rainfall:
            if rainfall_target is None:
                raise ValueError("rainfall_target is required when include_target_rainfall is true")
            parts.append(rainfall_target)
        if self.include_context_deltas:
            parts.append(context_water_masked[:, 1:] - context_water_masked[:, :-1])
        return self.encoder(torch.cat(parts, dim=1))


class ConsistencyLoss(nn.Module):
    """Hydraulic spatial-coherence penalty, ported from the reference repo's
    consistency_loss.py (present there with weight 0; exposed here as an
    optional V2 knob, default weight 0.0).

    Penalizes neighbor pairs where predicted water level (elevation + depth)
    increases toward higher ground: water_level_diff * elevation_diff, ReLU'd,
    averaged over the 8-neighborhood. Operates in physical units.
    """

    def __init__(self) -> None:
        super().__init__()
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        kernels = torch.zeros(8, 1, 3, 3)
        for index, (dy, dx) in enumerate(offsets):
            kernels[index, 0, 1, 1] = 1.0
            kernels[index, 0, 1 + dy, 1 + dx] = -1.0
        self.register_buffer("kernels", kernels)

    def forward(self, inundation_physical: torch.Tensor, elevation_physical: torch.Tensor) -> torch.Tensor:
        if inundation_physical.ndim != 4 or inundation_physical.shape[1] != 1:
            raise ValueError(f"Expected inundation [B, 1, H, W], got {tuple(inundation_physical.shape)}")
        water_level = elevation_physical + inundation_physical
        water_level_diff = F.conv2d(water_level, self.kernels)
        elevation_diff = -F.conv2d(elevation_physical, self.kernels)
        penalty = F.relu(water_level_diff * elevation_diff)
        return penalty.mean()


class DiffSparseV2Model(nn.Module):
    """DIFF-SPARSE V2: the reference-faithful V1 core plus targeted upgrades.

    Kept from V1 (the DIFF-SPARSE identity): x0-parameterized conditional DDPM,
    N=20 steps, linear beta [1e-4, 1.0] (terminal SNR exactly 0), noise-masked
    sparse context (Algorithm 1), temporal-token cross-attention conditioning
    via diffusers.UNet2DConditionModel with CrossAttn at the 2 middle levels,
    raw-beta_t reverse variance, one-step training + autoregressive rollout.

    New in V2 (each motivated in reports/diff_sparse_v2_design.md):
      1. SpatialContextEncoder: pixel-aligned context features concatenated
         with x_noisy at the UNet input (hybrid conditioning).
      2. Target-step rainfall forcing (exogenous causal driver of new flooding).
      3. Physically-bounded sampling: x0_hat clamped to >= 0 physical depth
         at every reverse step (clip floor from normalization stats).
      4. Capacity knobs with moderately raised defaults
         (unet 32/64/64/128, context_embedding_dim 64).
      5. Optional Min-SNR loss weighting and hydraulic consistency loss
         (both off by default; ablation knobs).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_config = config.get("model", {})
        diffusion_config = config.get("diffusion", {})
        loss_config = config.get("loss", {})

        self.diffusion_steps = int(diffusion_config.get("steps", 20))
        self.prediction_type = str(diffusion_config.get("prediction_type", "x0")).lower()
        beta_schedule = str(diffusion_config.get("beta_schedule", "linear")).lower()
        if self.diffusion_steps < 2:
            raise ValueError("diffusion.steps must be >= 2")
        if self.prediction_type != "x0":
            raise ValueError("Only x0 prediction is implemented (paper parameterization)")
        if beta_schedule != "linear":
            raise ValueError("Only the linear beta schedule is implemented")

        beta_start = float(diffusion_config.get("beta_start", 1e-4))
        beta_end = float(diffusion_config.get("beta_end", 1.0))
        if not 0.0 < beta_start < 1.0 or not 0.0 < beta_end <= 1.0:
            raise ValueError(f"Invalid beta range [{beta_start}, {beta_end}]")
        betas = torch.linspace(beta_start, beta_end, self.diffusion_steps, dtype=torch.float64)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alpha_cumprod[:-1]])
        denominator = 1.0 - alpha_cumprod
        posterior_coef_x0 = betas * torch.sqrt(alpha_cumprod_prev) / denominator
        posterior_coef_xt = torch.sqrt(alphas) * (1.0 - alpha_cumprod_prev) / denominator
        # Raw beta_t reverse-step noise (reference diffusion.py "Option 1").
        posterior_variance = betas

        self.register_buffer("betas", betas.float())
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod).float())
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod).float())
        self.register_buffer("posterior_coef_x0", posterior_coef_x0.float())
        self.register_buffer("posterior_coef_xt", posterior_coef_xt.float())
        self.register_buffer("posterior_variance", posterior_variance.float())
        # Min-SNR-style loss weights for x0 prediction: w_t = min(SNR_t, gamma),
        # normalized to mean 1 so the loss scale stays comparable to plain MSE.
        # SNR at the terminal step is exactly 0 (beta_end=1.0), so a small floor
        # keeps a nonzero training signal there -- the sampler's first reverse
        # step consumes the terminal-step prediction.
        snr_gamma = loss_config.get("snr_gamma")
        self.snr_gamma = float(snr_gamma) if snr_gamma is not None else None
        if self.snr_gamma is not None:
            snr = alpha_cumprod / torch.clamp(1.0 - alpha_cumprod, min=1e-12)
            weights = torch.minimum(snr, torch.full_like(snr, self.snr_gamma))
            weights = torch.clamp(weights, min=float(loss_config.get("snr_weight_floor", 0.05)))
            weights = weights / weights.mean()
            self.register_buffer("loss_weights", weights.float())
        else:
            self.register_buffer("loss_weights", torch.ones(self.diffusion_steps))

        self.context_encoder = TemporalContextEncoder(config)
        self.spatial_encoder = SpatialContextEncoder(config)
        self.embedding_dim = self.context_encoder.embedding_dim
        # Ablation knob (paper master plan WP4-c): 0.0 zeroes the pixel-aligned
        # spatial features at the UNet input, reducing conditioning to the
        # temporal tokens alone (V1-style attention-only) while keeping the
        # architecture and parameter count strictly unchanged.
        self.spatial_features_scale = float(model_config.get("spatial_features_scale", 1.0))

        unet_channels = [int(c) for c in model_config.get("unet_channels", [32, 64, 64, 128])]
        if len(unet_channels) != 4:
            raise ValueError("unet_channels must have exactly 4 levels (matches the reference down/up_block_types)")
        levels = len(unet_channels)
        attention_levels = int(model_config.get("cross_attention_blocks", 2))
        outer = (levels - attention_levels) // 2
        down_block_types = tuple(
            "CrossAttnDownBlock2D" if outer <= i < levels - outer else "DownBlock2D" for i in range(levels)
        )
        up_block_types = tuple(
            "CrossAttnUpBlock2D" if outer <= i < levels - outer else "UpBlock2D" for i in range(levels)
        )
        self.unet = UNet2DConditionModel(
            in_channels=1 + self.spatial_encoder.out_channels,
            out_channels=1,
            layers_per_block=int(model_config.get("resnet_layers_per_block", 2)),
            norm_num_groups=int(model_config.get("groupnorm_groups", 16)),
            block_out_channels=tuple(unet_channels),
            cross_attention_dim=self.embedding_dim,
            dropout=float(model_config.get("dropout", 0.0)),
            down_block_types=down_block_types,
            up_block_types=up_block_types,
        )

    def encode_context(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (temporal token sequence [B, T, D], spatial feature maps [B, C, H, W])."""

        tokens = self.context_encoder(
            batch["context_water_masked"],
            batch["sensor_mask"],
            batch["dem"],
            batch.get("rainfall_context"),
            batch.get("timestamps_context"),
            manning=batch.get("manning"),
        )
        spatial = self.spatial_encoder(
            batch["context_water_masked"],
            batch["sensor_mask"],
            batch["dem"],
            batch.get("rainfall_context"),
            batch.get("rainfall_target"),
            manning=batch.get("manning"),
        )
        if self.spatial_features_scale != 1.0:
            spatial = spatial * self.spatial_features_scale
        return tokens, spatial

    def denoise(
        self,
        x_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        context_tokens: torch.Tensor,
        spatial_features: torch.Tensor,
    ) -> torch.Tensor:
        if x_noisy.ndim != 4 or x_noisy.shape[1] != 1:
            raise ValueError(f"Expected x_noisy [B, 1, H, W], got {tuple(x_noisy.shape)}")
        if spatial_features.shape[-2:] != x_noisy.shape[-2:]:
            raise ValueError(
                f"spatial_features spatial size {tuple(spatial_features.shape[-2:])} must match "
                f"x_noisy {tuple(x_noisy.shape[-2:])}"
            )
        unet_input = torch.cat([x_noisy, spatial_features], dim=1)
        return self.unet(unet_input, timestep=timesteps, encoder_hidden_states=context_tokens).sample

    def forward(self, x_noisy: torch.Tensor, timesteps: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens, spatial = self.encode_context(batch)
        return self.denoise(x_noisy, timesteps, tokens, spatial)

    def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if x0.ndim != 4:
            raise ValueError(f"Expected x0 [B, C, H, W], got {tuple(x0.shape)}")
        if timesteps.ndim != 1 or timesteps.shape[0] != x0.shape[0]:
            raise ValueError(f"Expected timesteps [B] matching x0 batch, got {tuple(timesteps.shape)}")
        if noise is None:
            noise = torch.randn_like(x0)
        timesteps = timesteps.long()
        sqrt_alpha = self.sqrt_alpha_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod[timesteps].view(-1, 1, 1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def training_step_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """Paper Algorithm 1 with optional per-timestep Min-SNR weighting."""

        target = batch["target"]
        if target.ndim != 4 or target.shape[1] != 1:
            raise ValueError(f"Expected training target [B, 1, H, W], got {tuple(target.shape)}")
        tokens, spatial = self.encode_context(batch)
        timesteps = torch.randint(0, self.diffusion_steps, (target.shape[0],), device=target.device)
        noise = torch.randn_like(target)
        x_noisy = self.q_sample(target, timesteps, noise=noise)
        prediction = self.denoise(x_noisy, timesteps, tokens, spatial)
        squared_error = F.mse_loss(prediction, target, reduction="none")
        pixel_weights = batch.get("pixel_weights")
        if pixel_weights is not None:
            # Per-pixel emphasis (V2: pixels whose wet/dry state changes between
            # base and target -- the propagation-path signal). Normalized so the
            # loss scale stays comparable to plain MSE.
            squared_error = squared_error * (pixel_weights / pixel_weights.mean().clamp(min=1e-12))
        per_sample = squared_error.mean(dim=(1, 2, 3))
        loss = (per_sample * self.loss_weights[timesteps]).mean()
        diagnostics = {
            "x0_rmse": float(torch.sqrt(F.mse_loss(prediction.detach(), target.detach())).item()),
            "timestep_mean": float(timesteps.float().mean().item()),
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
        """Paper Algorithm 2: DDPM reverse sampling from pure Gaussian noise.

        clip_x0 = (floor, ceiling); either side may be None, a scalar, or a
        per-pixel tensor (broadcastable to `shape`). V2's default eval protocol
        passes the normalized value of 0 physical water depth as the floor --
        in absolute mode a scalar, in delta mode the per-pixel tensor
        (0 - base)/delta_scale -- so every intermediate x0_hat respects
        depth >= 0, removing the sampling-noise oscillation around small
        flood-mask thresholds that dominates propagation-path false positives.
        """

        device = spatial_features.device
        x_t = torch.randn(shape, device=device, generator=generator, dtype=spatial_features.dtype)
        for step in reversed(range(self.diffusion_steps)):
            timesteps = torch.full((shape[0],), step, device=device, dtype=torch.long)
            if denoiser is not None:
                x0_hat = denoiser(x_t, timesteps)
            else:
                x0_hat = self.denoise(x_t, timesteps, context_tokens, spatial_features)
            if clip_x0 is not None:
                x0_hat = _apply_clip(x0_hat, clip_x0[0], clip_x0[1])
            mean = self.posterior_coef_x0[step] * x0_hat + self.posterior_coef_xt[step] * x_t
            if step > 0:
                noise = torch.randn(shape, device=device, generator=generator, dtype=x_t.dtype)
                x_t = mean + torch.sqrt(self.posterior_variance[step].clamp(min=0.0)) * noise
            else:
                x_t = mean
            if not torch.isfinite(x_t).all():
                raise FloatingPointError(f"Non-finite state after reverse diffusion step {step}")
        if clip_x0 is not None:
            x_t = _apply_clip(x_t, clip_x0[0], clip_x0[1])
        return x_t
