from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from models.deterministic_twin import DeterministicTwinModel, build_v2_family_model  # noqa: E402
from models.diff_sparse_v2 import DiffSparseV2Model  # noqa: E402
from tests.test_diff_sparse_v2_smoke import _delta_spec, _fake_model_batch, _small_config  # noqa: E402
from tools.train_floodcastbench_diff_sparse_v2 import prepare_model_batch, pushforward_batch  # noqa: E402


def _twin_config() -> dict:
    config = _small_config()
    config["model"]["name"] = "deterministic_twin"
    return config


def test_factory_dispatches_on_model_name() -> None:
    assert isinstance(build_v2_family_model(_small_config()), DiffSparseV2Model)
    assert isinstance(build_v2_family_model(_twin_config()), DeterministicTwinModel)
    bad = _small_config()
    bad["model"]["name"] = "nope"
    with pytest.raises(ValueError):
        build_v2_family_model(bad)


def test_twin_parameter_parity_with_v2() -> None:
    """The controlled comparison requires EXACT parameter parity (master plan
    R2): the twin must be the V2 network to the last weight."""

    v2 = DiffSparseV2Model(_small_config())
    twin = DeterministicTwinModel(_twin_config())
    v2_params = {name: tuple(p.shape) for name, p in v2.named_parameters()}
    twin_params = {name: tuple(p.shape) for name, p in twin.named_parameters()}
    assert v2_params == twin_params


def test_twin_is_deterministic_and_ignores_noise_input() -> None:
    torch.manual_seed(0)
    model = DeterministicTwinModel(_twin_config())
    model.eval()
    tokens = torch.randn(2, 4, 16)
    spatial = torch.randn(2, 8, 32, 32)
    a = model.sample(tokens, spatial, (2, 1, 32, 32), generator=torch.Generator().manual_seed(1))
    b = model.sample(tokens, spatial, (2, 1, 32, 32), generator=torch.Generator().manual_seed(999))
    assert torch.equal(a, b)  # different RNG, identical output
    # denoise ignores the sample input and the timestep entirely
    x1 = torch.randn(2, 1, 32, 32)
    x2 = torch.randn(2, 1, 32, 32)
    t1 = torch.full((2,), 3, dtype=torch.long)
    t2 = torch.full((2,), 17, dtype=torch.long)
    with torch.no_grad():
        assert torch.equal(model.denoise(x1, t1, tokens, spatial), model.denoise(x2, t2, tokens, spatial))


def test_twin_training_step_loss_and_gradients() -> None:
    config = _twin_config()
    model = DeterministicTwinModel(config)
    batch = _fake_model_batch(config)
    batch["pixel_weights"] = 1.0 + 3.0 * (torch.rand_like(batch["target"]) > 0.8).float()
    loss, diagnostics = model.training_step_loss(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert diagnostics["pred_finite"] == 1.0
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_twin_sample_respects_clip_floor() -> None:
    model = DeterministicTwinModel(_twin_config())
    tokens = torch.zeros(2, 4, 16)
    spatial = torch.zeros(2, 8, 8, 8)
    floor = torch.full((2, 1, 8, 8), 10.0)  # forces the clamp to bind
    result = model.sample(tokens, spatial, (2, 1, 8, 8), clip_x0=(floor, None))
    assert torch.all(result >= floor - 1e-6)


def test_twin_drives_pushforward_branch_unchanged() -> None:
    """The V2 trainer's pushforward branch must work on the twin as-is (same
    denoise signature; twin ignores the terminal noise/timestep)."""

    config = _twin_config()
    model = DeterministicTwinModel(config)
    c, l, size = config["dataset"]["context_length"], 2, 32
    torch.manual_seed(0)
    mask = (torch.rand(2, 1, size, size) > 0.5).float()
    context_true = torch.randn(2, c, size, size)
    batch = {
        "context_water_masked": context_true * mask,
        "context_water_true": context_true,
        "sensor_mask": mask,
        "dem": torch.randn(2, 1, size, size),
        "rainfall": torch.randn(2, c + l, size, size),
        "timestamps": torch.arange(c + l, dtype=torch.float32).repeat(2, 1) * 300.0,
        "target": context_true[:, -1:].repeat(1, l, 1, 1) + 0.004 * torch.randn(2, l, size, size),
    }
    delta = _delta_spec()
    model_batch = prepare_model_batch(batch, c, delta, wet_threshold_normalized=-0.396, change_weight=3.0)
    pushed = pushforward_batch(model, batch, model_batch, c, delta, -0.396, 3.0, clamp=True)
    assert pushed is not None
    assert torch.isfinite(pushed["target"]).all()
    loss, _ = model.training_step_loss(pushed)
    assert torch.isfinite(loss)
