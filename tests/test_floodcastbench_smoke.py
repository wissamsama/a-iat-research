from __future__ import annotations

import importlib.util

import pytest
import torch

from models.flood_latent_temporal import FloodLatentTemporalModel


def test_core_imports() -> None:
    import datasets  # noqa: F401
    import metrics.floodcastbench_eval  # noqa: F401
    import tools.train_floodcastbench_forecasting  # noqa: F401


def test_cuda_available_when_required() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this Python environment.")
    assert torch.cuda.get_device_name(0)


def test_official_mamba_import_when_installed() -> None:
    if importlib.util.find_spec("mamba_ssm") is None:
        pytest.skip("official mamba_ssm package is not installed in this environment.")
    from mamba_ssm import Mamba

    assert Mamba is not None


def test_latent_temporal_forward_cpu() -> None:
    model = FloodLatentTemporalModel(
        input_window=3,
        base_channels=4,
        latent_channels=8,
        temporal_module="temporal_conv",
    )
    x = torch.randn(2, 3, 1, 32, 32)
    y = model(x)
    assert y.shape == (2, 1, 32, 32)
    assert torch.isfinite(y).all()


def test_latent_mamba_forward_cuda_if_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this Python environment.")
    if importlib.util.find_spec("mamba_ssm") is None:
        pytest.skip("official mamba_ssm package is not installed in this environment.")

    model = FloodLatentTemporalModel(
        input_window=3,
        base_channels=4,
        latent_channels=8,
        temporal_module="mamba",
    ).cuda()
    x = torch.randn(1, 3, 1, 32, 32, device="cuda")
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 32, 32)
    assert torch.isfinite(y).all()


def test_minimal_checkpoint_save_load(tmp_path) -> None:
    model = FloodLatentTemporalModel(
        input_window=3,
        base_channels=4,
        latent_channels=8,
        temporal_module="temporal_conv",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint_path = tmp_path / "checkpoint_last.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None,
            "epoch": 1,
            "best_val_rmse": 0.123,
        },
        checkpoint_path,
    )

    loaded_model = FloodLatentTemporalModel(
        input_window=3,
        base_channels=4,
        latent_channels=8,
        temporal_module="temporal_conv",
    )
    loaded_optimizer = torch.optim.Adam(loaded_model.parameters(), lr=1e-4)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    assert checkpoint["epoch"] == 1
    assert checkpoint["best_val_rmse"] == pytest.approx(0.123)
