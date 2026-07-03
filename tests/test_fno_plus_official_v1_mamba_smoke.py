from __future__ import annotations

import pytest
import torch

from models.fno_plus_official import FNOPlusOfficial3d
from models.fno_plus_official_mamba import FNOPlusOfficial3dMamba, TemporalMambaResidual


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _count_params(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def test_mamba_import_path_works() -> None:
    try:
        from mamba_ssm import Mamba  # noqa: F401
    except Exception:
        from mamba_ssm.modules.mamba_simple import Mamba  # noqa: F401


def test_temporal_mamba_residual_preserves_latent_shape() -> None:
    device = _device()
    block = TemporalMambaResidual(width=8, d_state=8, d_conv=4, expand=2).to(device)
    z = torch.randn(1, 8, 6, 5, 20, device=device)
    out = block(z)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()


def test_fno_plus_mamba_forward_shape_and_finite_output() -> None:
    device = _device()
    model = FNOPlusOfficial3dMamba(
        input_channels=6,
        output_steps=19,
        modes=3,
        width=8,
        fourier_layers=2,
        mamba_layers=1,
        d_state=8,
        d_conv=4,
        expand=2,
    ).to(device)
    x = torch.randn(1, 6, 8, 8, 20, device=device)
    pred = model(x)
    assert pred.shape == (1, 1, 8, 8, 19)
    assert torch.isfinite(pred).all()


def test_fno_plus_mamba_tiny_train_eval_step_has_finite_loss() -> None:
    device = _device()
    model = FNOPlusOfficial3dMamba(
        input_channels=6,
        output_steps=19,
        modes=3,
        width=8,
        fourier_layers=2,
        mamba_layers=1,
        d_state=8,
        d_conv=4,
        expand=2,
    ).to(device)
    x = torch.randn(1, 6, 8, 8, 20, device=device)
    target = torch.randn(1, 1, 8, 8, 19, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, target)
    assert torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        eval_pred = model(x)
    assert torch.isfinite(eval_pred).all()


def test_mamba_parameter_count_is_larger_but_bounded() -> None:
    baseline = FNOPlusOfficial3d(input_channels=6, output_steps=19, modes=3, width=8, fourier_layers=2)
    mamba = FNOPlusOfficial3dMamba(
        input_channels=6,
        output_steps=19,
        modes=3,
        width=8,
        fourier_layers=2,
        mamba_layers=1,
        d_state=8,
        d_conv=4,
        expand=2,
    )
    baseline_params = _count_params(baseline)
    mamba_params = _count_params(mamba)
    assert mamba_params > baseline_params
    assert mamba_params < baseline_params * 2


@pytest.mark.parametrize("time_steps", [20, 24])
def test_mamba_model_supports_arbitrary_compatible_time_axis(time_steps: int) -> None:
    device = _device()
    model = FNOPlusOfficial3dMamba(
        input_channels=6,
        output_steps=19,
        modes=3,
        width=8,
        fourier_layers=2,
        mamba_layers=1,
        d_state=8,
        d_conv=4,
        expand=2,
    ).to(device)
    x = torch.randn(1, 6, 8, 8, time_steps, device=device)
    pred = model(x)
    assert pred.shape == (1, 1, 8, 8, 19)
