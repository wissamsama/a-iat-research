from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

rasterio = pytest.importorskip("rasterio")

from datasets.floodcastbench_fno_plus_official_v1_dataset import (  # noqa: E402
    compute_train_normalization_stats,
    build_fno_plus_official_v1_dataset,
)
from models.fno_plus_official import FNOPlusOfficial3d  # noqa: E402
from tools.train_floodcastbench_fno_plus_official_v1 import PhysicalMetricAccumulator  # noqa: E402


def _write_tif(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=str(array.dtype),
    ) as dataset:
        dataset.write(array, 1)


def _make_tiny_floodcastbench(root: Path) -> None:
    shape = (12, 12)
    for index in range(60):
        _write_tif(
            root / "High-fidelity flood forecasting" / "60m" / "Australia" / f"{index * 300}.tif",
            np.full(shape, 0.02 + index / 1000.0, dtype=np.float32),
        )
    _write_tif(root / "Relevant data" / "DEM" / "Australia_DEM.tif", np.full((24, 24), 12.0, dtype=np.float32))
    for index in range(10):
        _write_tif(
            root / "Relevant data" / "Rainfall" / "Australia flood" / f"rain_{index:03d}.tif",
            np.full((24, 24), 1.0 + index / 10.0, dtype=np.float32),
        )


def _config() -> dict:
    return {
        "dataset": {
            "event": "australia",
            "fidelity": "high",
            "resolution": "60m",
            "sample_length": 20,
            "stride": 20,
            "split_counts": {"train": 1, "val": 1, "test": 1},
        }
    }


def test_v1_dataset_normalization_and_inverse_transform(tmp_path: Path) -> None:
    _make_tiny_floodcastbench(tmp_path)
    config = _config()
    stats = compute_train_normalization_stats(tmp_path, config)
    dataset = build_fno_plus_official_v1_dataset(tmp_path, config, "train", stats)
    x_norm, target_norm, _ = dataset[0]
    assert x_norm.shape == (6, 12, 12, 20)
    assert target_norm.shape == (1, 12, 12, 19)
    assert torch.isfinite(x_norm).all()
    assert torch.isfinite(target_norm).all()
    assert abs(float(x_norm[3].mean())) < 1e-5
    restored = dataset.inverse_transform_target(target_norm)
    assert restored.mean().item() == pytest.approx(0.03, abs=0.02)


def test_v1_model_forward_shape() -> None:
    model = FNOPlusOfficial3d(input_channels=6, output_steps=19, modes=4, width=8, fourier_layers=2)
    x = torch.randn(1, 6, 12, 12, 20)
    pred = model(x)
    assert pred.shape == (1, 1, 12, 12, 19)


def test_v1_metrics_are_physical_units() -> None:
    target_physical = torch.full((1, 1, 8, 8, 19), 0.1)
    pred_physical = target_physical.clone()
    accumulator = PhysicalMetricAccumulator((0.001, 0.01))
    accumulator.update(pred_physical, target_physical)
    metrics = accumulator.compute()
    assert metrics["classical_rmse"] == pytest.approx(0.0)
    assert metrics["current_relative_rmse"] == pytest.approx(0.0)
    assert metrics["nse"] == pytest.approx(1.0)
    assert metrics["csi_gamma_0_001"] == pytest.approx(1.0)
    assert metrics["pred_mean"] == pytest.approx(0.1)
    assert metrics["target_mean"] == pytest.approx(0.1)
