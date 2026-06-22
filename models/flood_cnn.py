from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FloodCNNBaseline(nn.Module):
    """Small CNN baseline for water-depth raster forecasting.

    Input shape:  [B, input_window, 1, H, W]
    Output shape: [B, 1, H, W]
    """

    VALID_OUTPUT_ACTIVATIONS = {"identity", "relu", "softplus"}

    def __init__(
        self,
        input_window: int = 5,
        base_channels: int = 16,
        output_activation: str = "identity",
        final_bias_init: float | None = None,
    ) -> None:
        super().__init__()
        self.input_window = int(input_window)
        self.output_activation = str(output_activation).lower()
        channels = int(base_channels)
        if self.input_window < 1:
            raise ValueError("input_window must be >= 1")
        if channels < 1:
            raise ValueError("base_channels must be >= 1")
        if self.output_activation not in self.VALID_OUTPUT_ACTIVATIONS:
            choices = ", ".join(sorted(self.VALID_OUTPUT_ACTIVATIONS))
            raise ValueError(f"output_activation must be one of: {choices}")

        self.enc1 = nn.Sequential(
            nn.Conv2d(self.input_window, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        if final_bias_init is not None:
            nn.init.constant_(self.out.bias, float(final_bias_init))

    def apply_output_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "identity":
            return x
        if self.output_activation == "softplus":
            return F.softplus(x)
        # ReLU enforces non-negativity, but negative raw outputs receive zero gradient.
        return torch.relu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected x shape [B, input_window, 1, H, W], got {tuple(x.shape)}")
        if x.shape[2] != 1:
            raise ValueError(f"Expected singleton water-depth channel at dim=2, got {x.shape[2]}")

        target_size = x.shape[-2:]
        x = x.squeeze(2)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec2(x)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        x = self.dec1(x)
        x = self.out(x)
        return self.apply_output_activation(x)
