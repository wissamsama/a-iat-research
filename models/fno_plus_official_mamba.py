from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except Exception:  # pragma: no cover - fallback depends on installed mamba-ssm layout.
    from mamba_ssm.modules.mamba_simple import Mamba

from models.fno_plus_official import SpectralConv3d


class TemporalMambaResidual(nn.Module):
    """Temporal Mamba residual block applied independently at each spatial cell.

    Input and output use latent layout [B, C, H, W, T]. Internally each spatial
    location is treated as a length-T sequence with C latent features:
    [B * H * W, T, C].
    """

    def __init__(
        self,
        width: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        layer_norm: bool = True,
        residual: bool = True,
        layer_scale_init: float | None = None,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.residual = bool(residual)
        self.layer_norm_enabled = bool(layer_norm)
        if self.width < 1:
            raise ValueError("width must be >= 1")

        self.norm = nn.LayerNorm(self.width) if self.layer_norm_enabled else nn.Identity()
        self.mamba = Mamba(
            d_model=self.width,
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
        )
        # LayerScale/ReZero-style gate (Touvron et al. 2021; Bachlechner et al.
        # 2020): mamba_ssm's default init gives the residual branch full
        # weight from step 1, i.e. an untrained transform perturbs an
        # already-trainable FNO latent at full strength from the first
        # gradient step. WPB3 (reports/fno_plus_beat_paper_plan.md) measured
        # this naive (layer_scale_init=None) config's val_rrmse curve to be
        # 2-3x rougher (spike count, peak, std) than the vanilla FNO+'s own
        # curve over the same 100 epochs -- a genuine optimization/stability
        # signature, not just a worse optimum. A learnable per-channel gate
        # initialized near zero makes the block start as an identity
        # function (forward pass unaffected) while remaining fully
        # trainable (the gate's gradient is update * upstream_grad, nonzero
        # even at gate=0), letting the network learn how much to trust the
        # Mamba branch instead of being forced to absorb it immediately.
        # None (default) preserves the exact original unscaled behavior --
        # backward compatible with the already-trained naive checkpoint.
        self.layer_scale_init = layer_scale_init
        if layer_scale_init is not None:
            self.layer_scale = nn.Parameter(torch.full((self.width,), float(layer_scale_init)))
        else:
            self.layer_scale = None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 5:
            raise ValueError(f"Expected [B, C, H, W, T], got {tuple(z.shape)}")
        batch, channels, height, width, time_steps = z.shape
        if channels != self.width:
            raise ValueError(f"Expected latent width {self.width}, got {channels}")

        z_seq = z.permute(0, 2, 3, 4, 1).reshape(batch * height * width, time_steps, channels)
        update = self.mamba(self.norm(z_seq))
        if update.shape != z_seq.shape:
            raise RuntimeError(f"Mamba changed sequence shape from {tuple(z_seq.shape)} to {tuple(update.shape)}")
        if self.layer_scale is not None:
            update = update * self.layer_scale
        z_seq = z_seq + update if self.residual else update
        return z_seq.reshape(batch, height, width, time_steps, channels).permute(0, 4, 1, 2, 3).contiguous()


class FNOPlusOfficial3dMamba(nn.Module):
    """Official-v1 FNO+ 3D baseline with one latent temporal Mamba residual block.

    The model preserves the official-v1 tensor convention:
    input [B, 6, H, W, 20] and output [B, 1, H, W, 19].
    Mamba is inserted after the Fourier backbone and before the projection head.
    """

    def __init__(
        self,
        input_channels: int = 6,
        output_steps: int = 19,
        modes: int = 12,
        width: int = 20,
        fourier_layers: int = 4,
        mamba_layers: int = 1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        layer_norm: bool = True,
        residual: bool = True,
        layer_scale_init: float | None = None,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.output_steps = int(output_steps)
        self.modes = int(modes)
        self.width = int(width)
        self.fourier_layers = int(fourier_layers)
        self.mamba_layers = int(mamba_layers)

        if self.input_channels < 1:
            raise ValueError("input_channels must be >= 1")
        if self.output_steps < 1:
            raise ValueError("output_steps must be >= 1")
        if self.fourier_layers < 1:
            raise ValueError("fourier_layers must be >= 1")
        if self.mamba_layers < 1:
            raise ValueError("mamba_layers must be >= 1")

        self.lift = nn.Conv3d(self.input_channels, self.width, kernel_size=1)
        self.spectral_layers = nn.ModuleList(
            SpectralConv3d(self.width, self.modes) for _ in range(self.fourier_layers)
        )
        self.pointwise_layers = nn.ModuleList(
            nn.Conv3d(self.width, self.width, kernel_size=1) for _ in range(self.fourier_layers)
        )
        self.temporal_mamba = nn.ModuleList(
            TemporalMambaResidual(
                width=self.width,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                layer_norm=layer_norm,
                residual=residual,
                layer_scale_init=layer_scale_init,
            )
            for _ in range(self.mamba_layers)
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

        z = self.lift(x)
        for spectral, pointwise in zip(self.spectral_layers, self.pointwise_layers):
            z = F.gelu(spectral(z) + pointwise(z))
        for block in self.temporal_mamba:
            z = block(z)
        z = F.gelu(self.proj1(z))
        z = self.proj2(z)
        return z[..., 1 : self.output_steps + 1]
