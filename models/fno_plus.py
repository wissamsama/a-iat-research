from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """2D Fourier layer used by the FloodCastBench FNO+ scaffold."""

    def __init__(self, channels: int, modes: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.modes = int(modes)
        if self.channels < 1:
            raise ValueError("channels must be >= 1")
        if self.modes < 1:
            raise ValueError("modes must be >= 1")

        scale = 1.0 / (self.channels * self.channels)
        self.weights_pos = nn.Parameter(
            scale * torch.randn(self.channels, self.channels, self.modes, self.modes, dtype=torch.cfloat)
        )
        self.weights_neg = nn.Parameter(
            scale * torch.randn(self.channels, self.channels, self.modes, self.modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(x.shape)}")
        batch_size, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}")

        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batch_size,
            channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        modes_h = min(self.modes, height)
        modes_w = min(self.modes, width // 2 + 1)
        out_ft[:, :, :modes_h, :modes_w] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :modes_h, :modes_w],
            self.weights_pos[:, :, :modes_h, :modes_w],
        )
        out_ft[:, :, -modes_h:, :modes_w] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -modes_h:, :modes_w],
            self.weights_neg[:, :, :modes_h, :modes_w],
        )
        return torch.fft.irfft2(out_ft, s=(height, width))


class FNOPlus2d(nn.Module):
    """FNO+ reproduction scaffold for FloodCastBench Table 4.

    Input shape:
        [B, C, H, W]

    Output shape:
        [B, output_steps, H, W]

    The default architecture matches the paper settings requested for the first
    reproduction attempt: 4 Fourier layers, 12 modes, and width 20.
    """

    def __init__(
        self,
        input_channels: int,
        output_steps: int = 19,
        modes: int = 12,
        width: int = 20,
        fourier_layers: int = 4,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.output_steps = int(output_steps)
        self.modes = int(modes)
        self.width = int(width)
        self.fourier_layers = int(fourier_layers)

        if self.input_channels < 1:
            raise ValueError("input_channels must be >= 1")
        if self.output_steps < 1:
            raise ValueError("output_steps must be >= 1")
        if self.fourier_layers < 1:
            raise ValueError("fourier_layers must be >= 1")

        self.lift = nn.Conv2d(self.input_channels, self.width, kernel_size=1)
        self.spectral_layers = nn.ModuleList(
            SpectralConv2d(self.width, self.modes) for _ in range(self.fourier_layers)
        )
        self.pointwise_layers = nn.ModuleList(
            nn.Conv2d(self.width, self.width, kernel_size=1) for _ in range(self.fourier_layers)
        )
        self.proj1 = nn.Conv2d(self.width, self.width * 2, kernel_size=1)
        self.proj2 = nn.Conv2d(self.width * 2, self.output_steps, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {x.shape[1]}")

        x = self.lift(x)
        for spectral, pointwise in zip(self.spectral_layers, self.pointwise_layers):
            x = F.gelu(spectral(x) + pointwise(x))
        x = F.gelu(self.proj1(x))
        return self.proj2(x)
