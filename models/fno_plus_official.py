from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SpectralConv3d(nn.Module):
    """Space-time Fourier layer for the FNO+ reproduction attempt v0.

    Tensors use [B, C, H, W, T]. Fourier modes are applied over H, W, and T.
    """

    def __init__(self, channels: int, modes: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.modes = int(modes)
        if self.channels < 1:
            raise ValueError("channels must be >= 1")
        if self.modes < 1:
            raise ValueError("modes must be >= 1")

        scale = 1.0 / (self.channels * self.channels)
        shape = (self.channels, self.channels, self.modes, self.modes, self.modes)
        self.weights_pp = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weights_np = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weights_pn = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weights_nn = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    @staticmethod
    def _mul(input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixyt,ioxyt->boxyt", input_tensor, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B, C, H, W, T], got {tuple(x.shape)}")
        batch_size, channels, height, width, time_steps = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}")

        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out_ft = torch.zeros(
            batch_size,
            channels,
            height,
            width,
            time_steps // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        mh = min(self.modes, height)
        mw = min(self.modes, width)
        mt = min(self.modes, time_steps // 2 + 1)

        out_ft[:, :, :mh, :mw, :mt] = self._mul(
            x_ft[:, :, :mh, :mw, :mt],
            self.weights_pp[:, :, :mh, :mw, :mt],
        )
        out_ft[:, :, -mh:, :mw, :mt] = self._mul(
            x_ft[:, :, -mh:, :mw, :mt],
            self.weights_np[:, :, :mh, :mw, :mt],
        )
        out_ft[:, :, :mh, -mw:, :mt] = self._mul(
            x_ft[:, :, :mh, -mw:, :mt],
            self.weights_pn[:, :, :mh, :mw, :mt],
        )
        out_ft[:, :, -mh:, -mw:, :mt] = self._mul(
            x_ft[:, :, -mh:, -mw:, :mt],
            self.weights_nn[:, :, :mh, :mw, :mt],
        )
        return torch.fft.irfftn(out_ft, s=(height, width, time_steps), dim=(-3, -2, -1))


class FNOPlusOfficial3d(nn.Module):
    """FloodCastBench FNO+ official reproduction attempt v0.

    This is intentionally separate from the internal 2D FNO+ baseline.
    It implements a one-shot direct space-time FNO with input shape
    [B, C, H, W, 20] and output shape [B, 1, H, W, 19].
    """

    def __init__(
        self,
        input_channels: int = 6,
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

        self.lift = nn.Conv3d(self.input_channels, self.width, kernel_size=1)
        self.spectral_layers = nn.ModuleList(
            SpectralConv3d(self.width, self.modes) for _ in range(self.fourier_layers)
        )
        self.pointwise_layers = nn.ModuleList(
            nn.Conv3d(self.width, self.width, kernel_size=1) for _ in range(self.fourier_layers)
        )
        self.proj1 = nn.Conv3d(self.width, self.width * 2, kernel_size=1)
        self.proj2 = nn.Conv3d(self.width * 2, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B, C, H, W, T], got {tuple(x.shape)}")
        if x.shape[1] != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {x.shape[1]}")
        if x.shape[-1] < self.output_steps + 1:
            raise ValueError("Input time dimension must include t=1 plus requested output steps")

        x = self.lift(x)
        for spectral, pointwise in zip(self.spectral_layers, self.pointwise_layers):
            x = F.gelu(spectral(x) + pointwise(x))
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        return x[..., 1 : self.output_steps + 1]
