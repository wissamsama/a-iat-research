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
    """Official hidden_state_net.py DownSampleBlock: unpadded Conv3d pair + AvgPool3d(1,2,2).

    Deliberately unpadded (matching the reference exactly): each 3x3 spatial conv shrinks
    H/W by 2, then the (1,2,2) pool halves them. Kernel (1,3,3) convolves only spatially,
    never mixing across the context_length ("depth"/time) axis -- each of the T context
    frames is downsampled independently before flattening into a per-timestep token.
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
    """Official hidden_state_net.py HiddenStateNet: produces the cross-attention token
    sequence consumed by UNet2DConditionModel's encoder_hidden_states.

    Per-context-timestep channels [masked water, DEM, sensor mask, (rainfall)] are stacked
    as a [B, C, T, H, W] volume (DEM and sensor mask are static, broadcast across T; water
    and rainfall vary with T) and downsampled by 3 unpadded TemporalDownBlocks, producing one
    heavily-pooled spatial summary token per context timestep -- NOT a pixel-aligned spatial
    map. This is a temporal attention mechanism (context frames are "tokens", like words in
    text-conditioned diffusion), not a spatial one: the UNet's own convolutions do all
    per-pixel spatial reasoning, and cross-attention only injects "what history looked like,
    per past timestep, in coarse aggregate."

    Per-timestep raw sinusoidal covariate features are concatenated to each token before a
    single shared linear projection to context_embedding_dim (no separate covariate MLP,
    matching the reference's direct `torch.cat((x, covariate), dim=2)`).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_config = config.get("model", {})
        dataset_config = config.get("dataset", {})
        self.context_length = int(dataset_config.get("context_length", 12))
        self.include_dem = bool(dataset_config.get("include_dem", True))
        self.include_rainfall = bool(dataset_config.get("include_rainfall", True))
        self.include_covariates = bool(dataset_config.get("include_covariates", True))
        self.embedding_dim = int(model_config.get("context_embedding_dim", 32))
        groups = int(model_config.get("context_groupnorm_groups", 8))
        conv_channels = [int(c) for c in model_config.get("context_conv_channels", [16, 32, 64])]
        patch_size = int(dataset_config.get("patch_size", 64))

        in_channels = 1 + (1 if self.include_dem else 0) + 1 + (1 if self.include_rainfall else 0)
        blocks = []
        current = in_channels
        for out_channels in conv_channels:
            blocks.append(TemporalDownBlock(current, out_channels, groups))
            current = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.output_conv = nn.Conv3d(current, 1, kernel_size=1)

        self.covariate_features = len(HOUR_FREQUENCIES) * 2 + 1 if self.include_covariates else 0
        self.covariate_time_scale = float(dataset_config.get("covariate_time_scale", 864000.0))

        # Flattened per-timestep token size computed analytically (not lazily on
        # first forward) so token_linear is a real registered parameter from
        # __init__ -- building it lazily on first forward left it out of
        # model.parameters() whenever the optimizer is constructed before that
        # first forward pass, silently freezing the entire conditioning
        # projection at its random init and making training unable to progress
        # past a large loss plateau regardless of everything else being correct.
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


class DiffSparseV1Model(nn.Module):
    """DIFF-SPARSE adaptation for FloodCastBench (Islam et al., AAAI 2026).

    x0-parameterized conditional diffusion matching the official implementation
    (github.com/KAI10/Diff-Sparse): a diffusers.UNet2DConditionModel predicts the
    clean next-step water-depth field, conditioned via cross-attention on a
    temporal token sequence (TemporalContextEncoder) built from sparse (masked)
    water history, DEM, sensor mask, and (FloodCastBench-specific) rainfall. The
    linear beta schedule ends at beta_end=1.0 (paper Table 2), so the terminal
    step is pure noise and reverse sampling from N(0, I) matches the training
    distribution exactly.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_config = config.get("model", {})
        diffusion_config = config.get("diffusion", {})

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
        # Reference implementation's active reverse-step noise ("Option 1" in
        # diffusion.py: sigma_t^2 = beta_t), not the tighter beta-tilde posterior
        # variance (its "Option 2", present in the file only as a commented-out
        # alternative).
        posterior_variance = betas

        self.register_buffer("betas", betas.float())
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod).float())
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod).float())
        self.register_buffer("posterior_coef_x0", posterior_coef_x0.float())
        self.register_buffer("posterior_coef_xt", posterior_coef_xt.float())
        self.register_buffer("posterior_variance", posterior_variance.float())

        self.context_encoder = TemporalContextEncoder(config)
        self.embedding_dim = self.context_encoder.embedding_dim

        unet_channels = [int(c) for c in model_config.get("unet_channels", [16, 32, 32, 64])]
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
            in_channels=1,
            out_channels=1,
            layers_per_block=int(model_config.get("resnet_layers_per_block", 2)),
            norm_num_groups=int(model_config.get("groupnorm_groups", 16)),
            block_out_channels=tuple(unet_channels),
            cross_attention_dim=self.embedding_dim,
            dropout=float(model_config.get("dropout", 0.0)),
            down_block_types=down_block_types,
            up_block_types=up_block_types,
        )

    def encode_context(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.context_encoder(
            batch["context_water_masked"],
            batch["sensor_mask"],
            batch["dem"],
            batch.get("rainfall_context"),
            batch.get("timestamps_context"),
        )

    def denoise(self, x_noisy: torch.Tensor, timesteps: torch.Tensor, context_embedding: torch.Tensor) -> torch.Tensor:
        if x_noisy.ndim != 4 or x_noisy.shape[1] != 1:
            raise ValueError(f"Expected x_noisy [B, 1, H, W], got {tuple(x_noisy.shape)}")
        return self.unet(x_noisy, timestep=timesteps, encoder_hidden_states=context_embedding).sample

    def forward(self, x_noisy: torch.Tensor, timesteps: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.denoise(x_noisy, timesteps, self.encode_context(batch))

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
        """Paper Algorithm 1: masked context -> predict clean next-step target from noisy target."""

        target = batch["target"]
        if target.ndim != 4 or target.shape[1] != 1:
            raise ValueError(f"Expected training target [B, 1, H, W], got {tuple(target.shape)}")
        context_embedding = self.encode_context(batch)
        timesteps = torch.randint(0, self.diffusion_steps, (target.shape[0],), device=target.device)
        noise = torch.randn_like(target)
        x_noisy = self.q_sample(target, timesteps, noise=noise)
        prediction = self.denoise(x_noisy, timesteps, context_embedding)
        loss = F.mse_loss(prediction, target)
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
        context_embedding: torch.Tensor,
        shape: tuple[int, ...],
        generator: torch.Generator | None = None,
        denoiser: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        clip_x0: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Paper Algorithm 2: DDPM reverse sampling from pure Gaussian noise.

        Because beta_end=1.0, the first reverse step fully discards the initial
        noise (posterior_coef_xt[-1] == 0) and every later state is reconstructed
        from predicted x0 plus schedule-consistent noise.
        """

        device = context_embedding.device
        x_t = torch.randn(shape, device=device, generator=generator, dtype=context_embedding.dtype)
        for step in reversed(range(self.diffusion_steps)):
            timesteps = torch.full((shape[0],), step, device=device, dtype=torch.long)
            if denoiser is not None:
                x0_hat = denoiser(x_t, timesteps)
            else:
                x0_hat = self.denoise(x_t, timesteps, context_embedding)
            if clip_x0 is not None:
                x0_hat = x0_hat.clamp(min=float(clip_x0[0]), max=float(clip_x0[1]))
            mean = self.posterior_coef_x0[step] * x0_hat + self.posterior_coef_xt[step] * x_t
            if step > 0:
                noise = torch.randn(shape, device=device, generator=generator, dtype=x_t.dtype)
                x_t = mean + torch.sqrt(self.posterior_variance[step].clamp(min=0.0)) * noise
            else:
                x_t = mean
            if not torch.isfinite(x_t).all():
                raise FloatingPointError(f"Non-finite state after reverse diffusion step {step}")
        return x_t
