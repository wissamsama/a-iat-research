from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal diffusion timestep embeddings."""

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


def _groups(channels: int) -> int:
    for group_count in (8, 4, 2):
        if channels % group_count == 0:
            return group_count
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseDiffSparseModel(nn.Module):
    """Compact dense DIFF-SPARSE-style conditional diffusion baseline.

    This is a local dense missing-rate-zero sanity baseline, not a full
    sparse-sensor DIFF-SPARSE reproduction. It uses x0 prediction:
    the denoiser predicts the clean one-step target from a noisy target,
    diffusion timestep, dense FloodCastBench context, and an observation mask.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_config = config.get("model", {})
        diffusion_config = config.get("diffusion", {})

        self.context_channels = int(model_config.get("context_channels", 6))
        self.target_channels = int(model_config.get("target_channels", 1))
        self.input_timesteps = int(model_config.get("input_timesteps", 20))
        self.base_channels = int(model_config.get("base_channels", 32))
        channel_mults = model_config.get("channel_mults", [1, 2, 4])
        if len(channel_mults) < 3:
            raise ValueError("model.channel_mults must contain at least three entries")
        c0 = int(self.base_channels * int(channel_mults[0]))
        c1 = int(self.base_channels * int(channel_mults[1]))
        c2 = int(self.base_channels * int(channel_mults[2]))

        self.diffusion_steps = int(diffusion_config.get("steps", 20))
        self.prediction_type = str(diffusion_config.get("prediction_type", "x0")).lower()
        beta_schedule = str(diffusion_config.get("beta_schedule", "linear")).lower()
        if self.diffusion_steps < 1:
            raise ValueError("diffusion.steps must be >= 1")
        if self.prediction_type != "x0":
            raise ValueError("Only x0 prediction is implemented for this dense sanity baseline")
        if beta_schedule != "linear":
            raise ValueError("Only a linear beta schedule is implemented")

        beta_start = float(diffusion_config.get("beta_start", 1e-4))
        beta_end = float(diffusion_config.get("beta_end", 2e-2))
        betas = torch.linspace(beta_start, beta_end, self.diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))

        context_in_channels = self.context_channels * self.input_timesteps + self.input_timesteps
        self.context_encoder = nn.Sequential(
            nn.Conv2d(context_in_channels, self.base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(self.base_channels), self.base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.base_channels, self.base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(self.base_channels), self.base_channels),
            nn.SiLU(inplace=True),
        )
        embedding_dim = self.base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_dim, self.base_channels),
            nn.SiLU(inplace=True),
            nn.Linear(self.base_channels, self.base_channels),
        )

        self.in_block = ConvBlock(self.target_channels + self.base_channels + self.base_channels, c0)
        self.down1 = ConvBlock(c0, c1)
        self.down2 = ConvBlock(c1, c2)
        self.mid = ConvBlock(c2, c2)
        self.up1 = ConvBlock(c2 + c1, c1)
        self.up2 = ConvBlock(c1 + c0, c0)
        self.out = nn.Conv2d(c0, self.target_channels, kernel_size=1)
        self.embedding_dim = embedding_dim

    def _flatten_context(self, context: torch.Tensor, context_mask: torch.Tensor | None) -> torch.Tensor:
        if context.ndim != 5:
            raise ValueError(f"Expected context [B, C, H, W, T], got {tuple(context.shape)}")
        batch, channels, height, width, timesteps = context.shape
        if channels != self.context_channels:
            raise ValueError(f"Expected {self.context_channels} context channels, got {channels}")
        if timesteps != self.input_timesteps:
            raise ValueError(f"Expected {self.input_timesteps} context timesteps, got {timesteps}")
        context_flat = context.permute(0, 1, 4, 2, 3).reshape(batch, channels * timesteps, height, width)

        if context_mask is None:
            context_mask = torch.ones(batch, 1, height, width, timesteps, device=context.device, dtype=context.dtype)
        if context_mask.ndim != 5:
            raise ValueError(f"Expected context_mask [B, 1, H, W, T], got {tuple(context_mask.shape)}")
        if context_mask.shape != (batch, 1, height, width, timesteps):
            raise ValueError(
                "context_mask must match [B, 1, H, W, T]; "
                f"expected {(batch, 1, height, width, timesteps)}, got {tuple(context_mask.shape)}"
            )
        mask_flat = context_mask.permute(0, 1, 4, 2, 3).reshape(batch, timesteps, height, width)
        return torch.cat([context_flat, mask_flat.to(dtype=context.dtype)], dim=1)

    def q_sample(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

    def forward(
        self,
        x_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_noisy.ndim != 4:
            raise ValueError(f"Expected x_noisy [B, C, H, W], got {tuple(x_noisy.shape)}")
        if x_noisy.shape[1] != self.target_channels:
            raise ValueError(f"Expected {self.target_channels} target channels, got {x_noisy.shape[1]}")

        context_features = self.context_encoder(self._flatten_context(context, context_mask))
        time_embedding = sinusoidal_timestep_embedding(timesteps, self.embedding_dim)
        time_features = self.time_mlp(time_embedding).view(x_noisy.shape[0], self.base_channels, 1, 1)
        time_features = time_features.expand(-1, -1, x_noisy.shape[-2], x_noisy.shape[-1])

        h0 = self.in_block(torch.cat([x_noisy, context_features, time_features], dim=1))
        h1 = self.down1(F.avg_pool2d(h0, kernel_size=2))
        h2 = self.down2(F.avg_pool2d(h1, kernel_size=2))
        mid = self.mid(h2)
        up1 = F.interpolate(mid, size=h1.shape[-2:], mode="bilinear", align_corners=False)
        up1 = self.up1(torch.cat([up1, h1], dim=1))
        up2 = F.interpolate(up1, size=h0.shape[-2:], mode="bilinear", align_corners=False)
        up2 = self.up2(torch.cat([up2, h0], dim=1))
        return self.out(up2)

    def training_step_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        target = batch["target"]
        context = batch["context"]
        context_mask = batch.get("context_mask")
        timesteps = torch.randint(0, self.diffusion_steps, (target.shape[0],), device=target.device)
        noise = torch.randn_like(target)
        x_noisy = self.q_sample(target, timesteps, noise=noise)
        pred = self.forward(x_noisy, timesteps, context, context_mask)
        loss = F.mse_loss(pred, target)
        diagnostics = {
            "x0_rmse": float(torch.sqrt(F.mse_loss(pred.detach(), target.detach())).item()),
            "timestep_mean": float(timesteps.float().mean().item()),
            "pred_finite": float(torch.isfinite(pred).all().item()),
            "target_finite": float(torch.isfinite(target).all().item()),
        }
        return loss, diagnostics
