from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class IdentityTemporalModule(nn.Module):
    """No temporal modeling: keep the last latent frame."""

    def forward(self, z_tokens: torch.Tensor) -> torch.Tensor:
        if z_tokens.ndim != 3:
            raise ValueError(f"Expected z_tokens [N, T, C], got {tuple(z_tokens.shape)}")
        return z_tokens[:, -1]


class TemporalConvModule(nn.Module):
    """Lightweight temporal convolution over Mamba-ready latent tokens."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, z_tokens: torch.Tensor) -> torch.Tensor:
        if z_tokens.ndim != 3:
            raise ValueError(f"Expected z_tokens [N, T, C], got {tuple(z_tokens.shape)}")
        z = z_tokens.transpose(1, 2)
        z = self.net(z)
        return z.transpose(1, 2)[:, -1]


class GRUTemporalModule(nn.Module):
    """Small GRU over latent tokens, matching the future Mamba interface."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=channels, hidden_size=channels, batch_first=True)

    def forward(self, z_tokens: torch.Tensor) -> torch.Tensor:
        if z_tokens.ndim != 3:
            raise ValueError(f"Expected z_tokens [N, T, C], got {tuple(z_tokens.shape)}")
        _, hidden = self.gru(z_tokens)
        return hidden[-1]


class MambaTemporalModule(nn.Module):
    """Official mamba-ssm backend over latent temporal tokens."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.backend = "mamba_ssm"
        try:
            from mamba_ssm import Mamba
        except ImportError as exc:
            raise ImportError(
                "temporal_module='mamba' requires the official mamba_ssm package. "
                "Run this model from the WSL environment where mamba-ssm is installed."
            ) from exc

        self.net = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)

    def forward(self, z_tokens: torch.Tensor) -> torch.Tensor:
        if z_tokens.ndim != 3:
            raise ValueError(f"Expected z_tokens [N, T, C], got {tuple(z_tokens.shape)}")
        return self.net(z_tokens)[:, -1]


def build_temporal_module(name: str, channels: int) -> nn.Module:
    name = str(name).lower()
    if name == "identity":
        return IdentityTemporalModule()
    if name == "temporal_conv":
        return TemporalConvModule(channels)
    if name == "gru":
        return GRUTemporalModule(channels)
    if name == "mamba":
        return MambaTemporalModule(channels)
    raise ValueError("temporal_module must be one of: identity, temporal_conv, gru, mamba")


class FloodLatentTemporalModel(nn.Module):
    """Mamba-ready latent temporal model for FloodCastBench water-depth forecasting.

    Input shape:  [B, input_window, 1, H, W]
    Output shape: [B, 1, H, W]

    The temporal module receives tokens shaped [B * H_lat * W_lat, T, C],
    which is the intended future interface for a Mamba backend.
    """

    VALID_OUTPUT_ACTIVATIONS = {"identity", "relu", "softplus"}

    def __init__(
        self,
        input_window: int = 5,
        base_channels: int = 16,
        latent_channels: int = 64,
        temporal_module: str = "temporal_conv",
        residual_prediction: bool = True,
        output_activation: str = "identity",
        final_bias_init: float | None = None,
    ) -> None:
        super().__init__()
        self.input_window = int(input_window)
        self.base_channels = int(base_channels)
        self.latent_channels = int(latent_channels)
        self.temporal_module_name = str(temporal_module).lower()
        self.residual_prediction = bool(residual_prediction)
        self.output_activation = str(output_activation).lower()

        if self.input_window < 1:
            raise ValueError("input_window must be >= 1")
        if self.base_channels < 1:
            raise ValueError("base_channels must be >= 1")
        if self.latent_channels < 1:
            raise ValueError("latent_channels must be >= 1")
        if self.output_activation not in self.VALID_OUTPUT_ACTIVATIONS:
            choices = ", ".join(sorted(self.VALID_OUTPUT_ACTIVATIONS))
            raise ValueError(f"output_activation must be one of: {choices}")

        mid_channels = max(self.base_channels, self.latent_channels // 2)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, self.base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.base_channels, mid_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, self.latent_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.temporal = build_temporal_module(self.temporal_module_name, self.latent_channels)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.latent_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, self.base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(self.base_channels, 1, kernel_size=3, padding=1)
        if final_bias_init is not None:
            nn.init.constant_(self.out.bias, float(final_bias_init))

    def apply_output_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "identity":
            return x
        if self.output_activation == "softplus":
            return F.softplus(x)
        return torch.relu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected x shape [B, input_window, 1, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.input_window:
            raise ValueError(f"Expected input_window={self.input_window}, got {x.shape[1]}")
        if x.shape[2] != 1:
            raise ValueError(f"Expected singleton water-depth channel at dim=2, got {x.shape[2]}")

        batch_size, input_window, _, height, width = x.shape
        target_size = (height, width)
        last_frame = x[:, -1]

        z = self.encoder(x.reshape(batch_size * input_window, 1, height, width))
        _, channels, latent_height, latent_width = z.shape
        z_seq = z.view(batch_size, input_window, channels, latent_height, latent_width)
        z_tokens = z_seq.permute(0, 3, 4, 1, 2).reshape(batch_size * latent_height * latent_width, input_window, channels)

        z_future_tokens = self.temporal(z_tokens)
        z_future = z_future_tokens.view(batch_size, latent_height, latent_width, channels).permute(0, 3, 1, 2)

        decoded = F.interpolate(z_future, scale_factor=2, mode="bilinear", align_corners=False)
        decoded = self.decoder(decoded)
        decoded = F.interpolate(decoded, size=target_size, mode="bilinear", align_corners=False)
        delta = self.out(decoded)

        prediction = last_frame + delta if self.residual_prediction else delta
        return self.apply_output_activation(prediction)
