from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

rasterio = pytest.importorskip("rasterio")

from datasets.floodcastbench_fno_dataset import FloodCastBenchFNODataset
from evaluation.floodcastbench_official_metrics import csi, nse, pearson_r, relative_rmse
from models.fno_plus import FNOPlus2d


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
    shape = (16, 16)
    for i in range(40):
        _write_tif(
            root / "High-fidelity flood forecasting" / "60m" / "Australia" / f"{i * 300}.tif",
            np.full(shape, i / 1000.0, dtype=np.float32),
        )
    _write_tif(root / "Relevant data" / "DEM" / "Australia_DEM.tif", np.ones((32, 32), dtype=np.float32))
    for i in range(7):
        _write_tif(
            root / "Relevant data" / "Rainfall" / "Australia flood" / f"rain_{i:03d}.tif",
            np.full((32, 32), i / 100.0, dtype=np.float32),
        )


def test_fno_plus_forward_shape() -> None:
    model = FNOPlus2d(input_channels=43, output_steps=19, modes=4, width=8, fourier_layers=2)
    x = torch.randn(2, 43, 16, 16)
    y = model(x)
    assert y.shape == (2, 19, 16, 16)


def test_official_metrics_perfect_prediction() -> None:
    target = torch.rand(2, 19, 8, 8)
    pred = target.clone()
    assert relative_rmse(pred, target).item() == pytest.approx(0.0)
    assert nse(pred, target).item() == pytest.approx(1.0)
    assert pearson_r(pred, target).item() == pytest.approx(1.0)
    assert csi(pred, target, gamma=0.001).item() == pytest.approx(1.0)


def test_fno_dataset_indexing_and_shapes(tmp_path: Path) -> None:
    _make_tiny_floodcastbench(tmp_path)
    dataset = FloodCastBenchFNODataset(
        root=tmp_path,
        event="australia",
        fidelity="high",
        resolution="60m",
        split="train",
        split_counts={"train": 1, "val": 1, "test": 0},
    )
    x, y, meta = dataset[0]
    assert len(dataset) == 1
    assert dataset.input_channels == 43
    assert x.shape == (43, 16, 16)
    assert y.shape == (19, 16, 16)
    assert meta["target_timestamps"][0] == 300
