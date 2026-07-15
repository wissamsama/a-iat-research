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


# --- WPB4: LayerScale/ReZero gate on the Mamba residual branch ---
# (reports/fno_plus_beat_paper_plan.md WPB3 measured the naive/ungated
# variant's val_rrmse curve to be 2-3x rougher than vanilla FNO+'s own
# curve over 100 epochs -- these tests cover the fix's three required
# properties: unchanged default behavior, exact identity at gate=0, and
# that the gate still receives gradient despite starting at 0.)


def test_mamba_layer_scale_default_is_backward_compatible() -> None:
    """layer_scale_init=None (the default) must reproduce the original
    unscaled/ungated forward exactly -- no behavior change for the
    already-trained naive checkpoint."""
    device = _device()
    torch.manual_seed(0)
    block_default = TemporalMambaResidual(width=8, d_state=8, d_conv=4, expand=2).to(device)
    torch.manual_seed(0)
    block_explicit_none = TemporalMambaResidual(
        width=8, d_state=8, d_conv=4, expand=2, layer_scale_init=None
    ).to(device)
    assert block_explicit_none.layer_scale is None
    z = torch.randn(1, 8, 4, 4, 20, device=device)
    torch.manual_seed(1)
    out_default = block_default(z)
    torch.manual_seed(1)
    out_none = block_explicit_none(z)
    assert torch.equal(out_default, out_none)


def test_mamba_layer_scale_zero_init_is_exact_identity() -> None:
    """With layer_scale_init=0.0 and residual=True, the block must be an
    exact identity function at init (update * 0 = 0, z_seq + 0 = z_seq) --
    the whole point of the fix: an untrained Mamba branch must not perturb
    the (already trainable) FNO latent on the very first forward pass."""
    device = _device()
    block = TemporalMambaResidual(width=8, d_state=8, d_conv=4, expand=2, layer_scale_init=0.0).to(device)
    z = torch.randn(1, 8, 4, 4, 20, device=device)
    out = block(z)
    assert torch.equal(out, z)


def test_mamba_layer_scale_receives_gradient_despite_zero_init() -> None:
    """The gate must still be trainable from a zero start: d(loss)/d(gate)
    = update * upstream_grad, which is nonzero even though gate itself is
    0 at init (this is what makes it LayerScale/ReZero rather than a dead
    branch that never turns on)."""
    device = _device()
    block = TemporalMambaResidual(width=8, d_state=8, d_conv=4, expand=2, layer_scale_init=0.0).to(device)
    z = torch.randn(1, 8, 4, 4, 20, device=device)
    out = block(z)
    loss = out.pow(2).sum()
    loss.backward()
    assert block.layer_scale.grad is not None
    assert torch.isfinite(block.layer_scale.grad).all()
    assert block.layer_scale.grad.abs().sum().item() > 0


def test_fno_plus_mamba_model_threads_layer_scale_init() -> None:
    model = FNOPlusOfficial3dMamba(
        input_channels=6,
        output_steps=19,
        modes=3,
        width=8,
        fourier_layers=2,
        mamba_layers=2,
        d_state=8,
        d_conv=4,
        expand=2,
        layer_scale_init=0.0,
    )
    for block in model.temporal_mamba:
        assert block.layer_scale is not None
        assert torch.equal(block.layer_scale, torch.zeros(8))
