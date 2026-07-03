from __future__ import annotations

import math
from typing import Any, Callable

import torch
from torch import nn
import torch.nn.functional as F


DAY_SECONDS = 86400.0
HOUR_FREQUENCIES = (1.0, 2.0, 4.0)


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    if timesteps.ndim != 1:
        raise ValueError(f"Expected timesteps [B], got {tuple(timesteps.shape)}")
    half = dim // 2
    if half < 1:
        raise ValueError("embedding dim must be >= 2")
    scale = math.log(10000.0) / max(half - 1, 1)
    frequencies = torch.exp(torch.arange(half, device=timesteps.device, dtype=torch.float32) * -scale)
    args = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def _groups(channels: int, preferred: int) -> int:
    for count in (preferred, 8, 4, 2):
        if channels % count == 0:
            return count
    return 1


_POSENC_CACHE: dict[tuple, torch.Tensor] = {}


def positional_encoding_2d(dim: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Fixed 2D sine-cosine positional encoding [dim, H, W] (spatial alignment for cross-attention)."""

    key = (dim, height, width, str(device), dtype)
    cached = _POSENC_CACHE.get(key)
    if cached is not None:
        return cached
    if dim % 4 != 0:
        raise ValueError(f"positional encoding dim must be divisible by 4, got {dim}")
    quarter = dim // 4
    omega = torch.arange(quarter, device=device, dtype=torch.float32) / max(quarter - 1, 1)
    omega = 1.0 / (10000.0**omega)
    ys = torch.arange(height, device=device, dtype=torch.float32)
    xs = torch.arange(width, device=device, dtype=torch.float32)
    y_args = ys.unsqueeze(1) * omega.unsqueeze(0)
    x_args = xs.unsqueeze(1) * omega.unsqueeze(0)
    y_enc = torch.cat([torch.sin(y_args), torch.cos(y_args)], dim=1)
    x_enc = torch.cat([torch.sin(x_args), torch.cos(x_args)], dim=1)
    encoding = torch.cat(
        [
            y_enc.unsqueeze(1).expand(height, width, 2 * quarter),
            x_enc.unsqueeze(0).expand(height, width, 2 * quarter),
        ],
        dim=-1,
    ).permute(2, 0, 1)
    encoding = encoding.to(dtype=dtype)
    _POSENC_CACHE[key] = encoding
    return encoding


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, groups: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels, groups), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_groups(out_channels, groups), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(time_embedding)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class CrossAttentionBlock(nn.Module):
    """Cross-attention from UNet features to the context embedding (Rombach et al. style).

    Context tokens are the spatial context embedding pooled to the query resolution;
    fixed 2D positional encodings are added on both sides so attention can align
    context pixels with output pixels.
    """

    def __init__(self, query_dim: int, context_dim: int, heads: int, groups: int) -> None:
        super().__init__()
        if query_dim % heads != 0:
            raise ValueError(f"query_dim={query_dim} must be divisible by heads={heads}")
        self.heads = heads
        self.head_dim = query_dim // heads
        self.norm = nn.GroupNorm(_groups(query_dim, groups), query_dim)
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)
        self.context_dim = context_dim
        self.query_dim = query_dim

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        pooled = F.adaptive_avg_pool2d(context, (height, width))
        query_pos = positional_encoding_2d(channels, height, width, x.device, x.dtype)
        context_pos = positional_encoding_2d(self.context_dim, height, width, x.device, x.dtype)

        queries = (self.norm(x) + query_pos).flatten(2).transpose(1, 2)
        tokens = (pooled + context_pos).flatten(2).transpose(1, 2)

        q = self.to_q(queries).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(tokens).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(tokens).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.transpose(1, 2).reshape(batch, -1, channels)
        out = self.to_out(attended).transpose(1, 2).view(batch, channels, height, width)
        return x + out


class ContextEncoder(nn.Module):
    """Paper eq. (9)-(12): conv blocks over [masked water ⊕ DEM ⊕ mask (⊕ rainfall)],
    sinusoidal temporal covariates, then a 1x1 linear fusion to the context embedding."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_config = config.get("model", {})
        dataset_config = config.get("dataset", {})
        self.context_length = int(dataset_config.get("context_length", 12))
        self.include_dem = bool(dataset_config.get("include_dem", True))
        self.include_rainfall = bool(dataset_config.get("include_rainfall", True))
        self.include_covariates = bool(dataset_config.get("include_covariates", True))
        self.covariate_dim = int(model_config.get("covariate_dim", 16))
        self.embedding_dim = int(model_config.get("context_embedding_dim", 32))
        groups = int(model_config.get("groupnorm_groups", 8))
        conv_channels = [int(c) for c in model_config.get("context_conv_channels", [16, 32, 64])]

        in_channels = self.context_length + 1  # masked water history + sensor mask
        if self.include_dem:
            in_channels += 1
        if self.include_rainfall:
            in_channels += self.context_length

        layers: list[nn.Module] = []
        current = in_channels
        for out_channels in conv_channels:
            layers.extend(
                [
                    nn.Conv2d(current, out_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(_groups(out_channels, groups), out_channels),
                    nn.SiLU(inplace=True),
                ]
            )
            current = out_channels
        self.conv = nn.Sequential(*layers)

        covariate_features = len(HOUR_FREQUENCIES) * 2 + 1
        self.covariate_mlp = (
            nn.Sequential(
                nn.Linear(self.context_length * covariate_features, self.covariate_dim),
                nn.SiLU(inplace=True),
                nn.Linear(self.covariate_dim, self.covariate_dim),
            )
            if self.include_covariates
            else None
        )
        fusion_in = current + (self.covariate_dim if self.include_covariates else 0)
        self.fusion = nn.Conv2d(fusion_in, self.embedding_dim, kernel_size=1)
        self.covariate_time_scale = float(config.get("dataset", {}).get("covariate_time_scale", 864000.0))

    def encode_covariates(self, timestamps: torch.Tensor) -> torch.Tensor:
        """timestamps [B, c] in seconds -> [B, covariate_dim] (hour-of-day cycles + event fraction)."""

        hour_fraction = (timestamps % DAY_SECONDS) / DAY_SECONDS
        features = [timestamps / self.covariate_time_scale]
        for frequency in HOUR_FREQUENCIES:
            angle = 2.0 * math.pi * frequency * hour_fraction
            features.extend([torch.sin(angle), torch.cos(angle)])
        stacked = torch.stack(features, dim=-1).flatten(1)
        return self.covariate_mlp(stacked)

    def forward(
        self,
        context_water_masked: torch.Tensor,
        sensor_mask: torch.Tensor,
        dem: torch.Tensor,
        rainfall_context: torch.Tensor | None,
        timestamps_context: torch.Tensor | None,
    ) -> torch.Tensor:
        parts = [context_water_masked, sensor_mask]
        if self.include_dem:
            parts.append(dem)
        if self.include_rainfall:
            if rainfall_context is None:
                raise ValueError("rainfall_context is required when include_rainfall is true")
            parts.append(rainfall_context)
        features = self.conv(torch.cat(parts, dim=1))

        if self.include_covariates:
            if timestamps_context is None:
                raise ValueError("timestamps_context is required when include_covariates is true")
            covariates = self.encode_covariates(timestamps_context)
            covariates = covariates.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *features.shape[-2:])
            features = torch.cat([features, covariates], dim=1)
        return self.fusion(features)


class DiffSparseV1Model(nn.Module):
    """DIFF-SPARSE adaptation for FloodCastBench (Islam et al., AAAI 2026).

    x0-parameterized conditional diffusion: a UNet predicts the clean next-step
    water-depth field from a noisy field, the diffusion step, and a context
    embedding built from sparse (masked) water history, DEM, rainfall, and
    temporal covariates. The linear beta schedule ends at beta_end=1.0
    (paper Table 2), so the terminal step is pure noise and reverse sampling
    from N(0, I) matches the training distribution exactly.
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
        posterior_variance = betas * (1.0 - alpha_cumprod_prev) / denominator

        self.register_buffer("betas", betas.float())
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod).float())
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod).float())
        self.register_buffer("posterior_coef_x0", posterior_coef_x0.float())
        self.register_buffer("posterior_coef_xt", posterior_coef_xt.float())
        self.register_buffer("posterior_variance", posterior_variance.float())

        self.context_encoder = ContextEncoder(config)
        self.embedding_dim = self.context_encoder.embedding_dim
        self.conditioning = str(model_config.get("conditioning", "cross_attention_concat")).lower()
        if self.conditioning not in {"cross_attention", "cross_attention_concat"}:
            raise ValueError(f"Unsupported conditioning {self.conditioning!r}")

        groups = int(model_config.get("groupnorm_groups", 8))
        heads = int(model_config.get("attention_heads", 4))
        unet_channels = [int(c) for c in model_config.get("unet_channels", [16, 32, 32, 64])]
        res_layers = int(model_config.get("resnet_layers_per_block", 2))
        attention_levels = int(model_config.get("cross_attention_blocks", 2))
        levels = len(unet_channels)
        attention_from = levels - attention_levels
        time_dim = int(model_config.get("time_embedding_dim", 128))
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_dim, time_dim),
        )

        in_channels = 1 + (self.embedding_dim if self.conditioning == "cross_attention_concat" else 0)
        self.in_conv = nn.Conv2d(in_channels, unet_channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.down_attentions = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skip_channels = [unet_channels[0]]
        current = unet_channels[0]
        for level, channels in enumerate(unet_channels):
            blocks = nn.ModuleList()
            attentions = nn.ModuleList()
            for _ in range(res_layers):
                blocks.append(ResBlock(current, channels, time_dim, groups))
                attentions.append(
                    CrossAttentionBlock(channels, self.embedding_dim, heads, groups)
                    if level >= attention_from
                    else nn.Identity()
                )
                current = channels
                skip_channels.append(current)
            self.down_blocks.append(blocks)
            self.down_attentions.append(attentions)
            if level < levels - 1:
                self.downsamples.append(nn.Conv2d(current, current, kernel_size=3, stride=2, padding=1))
                skip_channels.append(current)
            else:
                self.downsamples.append(nn.Identity())

        self.mid_block1 = ResBlock(current, current, time_dim, groups)
        self.mid_attention = CrossAttentionBlock(current, self.embedding_dim, heads, groups)
        self.mid_block2 = ResBlock(current, current, time_dim, groups)

        self.up_blocks = nn.ModuleList()
        self.up_attentions = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level in reversed(range(levels)):
            channels = unet_channels[level]
            blocks = nn.ModuleList()
            attentions = nn.ModuleList()
            for _ in range(res_layers + 1):
                blocks.append(ResBlock(current + skip_channels.pop(), channels, time_dim, groups))
                attentions.append(
                    CrossAttentionBlock(channels, self.embedding_dim, heads, groups)
                    if level >= attention_from
                    else nn.Identity()
                )
                current = channels
            self.up_blocks.append(blocks)
            self.up_attentions.append(attentions)
            if level > 0:
                self.upsamples.append(nn.Conv2d(current, current, kernel_size=3, padding=1))
            else:
                self.upsamples.append(nn.Identity())
        if skip_channels:
            raise RuntimeError("UNet skip bookkeeping mismatch")

        self.out_norm = nn.GroupNorm(_groups(current, groups), current)
        self.out_conv = nn.Conv2d(current, 1, kernel_size=3, padding=1)

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
        if context_embedding.shape[-2:] != x_noisy.shape[-2:]:
            raise ValueError(
                f"context embedding spatial size {tuple(context_embedding.shape[-2:])} "
                f"must match x_noisy {tuple(x_noisy.shape[-2:])}"
            )
        time_embedding = self.time_mlp(sinusoidal_timestep_embedding(timesteps, self.time_dim))

        if self.conditioning == "cross_attention_concat":
            h = self.in_conv(torch.cat([x_noisy, context_embedding], dim=1))
        else:
            h = self.in_conv(x_noisy)

        skips = [h]
        for blocks, attentions, downsample in zip(self.down_blocks, self.down_attentions, self.downsamples):
            for block, attention in zip(blocks, attentions):
                h = block(h, time_embedding)
                if not isinstance(attention, nn.Identity):
                    h = attention(h, context_embedding)
                skips.append(h)
            if not isinstance(downsample, nn.Identity):
                h = downsample(h)
                skips.append(h)

        h = self.mid_block1(h, time_embedding)
        h = self.mid_attention(h, context_embedding)
        h = self.mid_block2(h, time_embedding)

        for blocks, attentions, upsample in zip(self.up_blocks, self.up_attentions, self.upsamples):
            for block, attention in zip(blocks, attentions):
                skip = skips.pop()
                if skip.shape[-2:] != h.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                h = block(torch.cat([h, skip], dim=1), time_embedding)
                if not isinstance(attention, nn.Identity):
                    h = attention(h, context_embedding)
            if not isinstance(upsample, nn.Identity):
                h = F.interpolate(h, scale_factor=2.0, mode="nearest")
                h = upsample(h)

        return self.out_conv(F.silu(self.out_norm(h)))

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
