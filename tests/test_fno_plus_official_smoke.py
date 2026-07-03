from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

rasterio = pytest.importorskip("rasterio")

from datasets.floodcastbench_fno_plus_official_dataset import FloodCastBenchFNOPlusOfficialDataset
from models.fno_plus_official import FNOPlusOfficial3d
from tools.recompute_fno_plus_official_metrics import MetricAccumulator, paper_formula_rmse


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
    for index in range(40):
        _write_tif(
            root / "High-fidelity flood forecasting" / "60m" / "Australia" / f"{index * 300}.tif",
            np.full(shape, 0.01 + index / 1000.0, dtype=np.float32),
        )
    _write_tif(root / "Relevant data" / "DEM" / "Australia_DEM.tif", np.ones((24, 24), dtype=np.float32))
    for index in range(7):
        _write_tif(
            root / "Relevant data" / "Rainfall" / "Australia flood" / f"rain_{index:03d}.tif",
            np.full((24, 24), index / 100.0, dtype=np.float32),
        )


def test_official_dataset_tensor_layout(tmp_path: Path) -> None:
    _make_tiny_floodcastbench(tmp_path)
    dataset = FloodCastBenchFNOPlusOfficialDataset(
        root=tmp_path,
        event="australia",
        fidelity="high",
        resolution="60m",
        split="train",
        split_counts={"train": 1, "val": 1, "test": 0},
    )
    x, target, meta = dataset[0]
    assert len(dataset) == 1
    assert x.shape == (6, 12, 12, 20)
    assert target.shape == (1, 12, 12, 19)
    assert meta["target_timestamps"][0] == 300


def test_official_model_forward_and_training_step() -> None:
    model = FNOPlusOfficial3d(input_channels=6, output_steps=19, modes=4, width=8, fourier_layers=2)
    x = torch.randn(1, 6, 12, 12, 20)
    target = torch.randn(1, 1, 12, 12, 19)
    pred = model(x)
    assert pred.shape == target.shape
    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()
    assert torch.isfinite(loss)


def test_official_metric_accumulator_perfect_prediction() -> None:
    target = torch.rand(1, 1, 8, 8, 19) + 0.01
    pred = target.clone()
    assert paper_formula_rmse(pred, target).item() == pytest.approx(0.0)
    accumulator = MetricAccumulator()
    accumulator.update(pred, target)
    metrics = accumulator.compute()
    assert metrics["paper_formula_rmse"] == pytest.approx(0.0)
    assert metrics["current_relative_rmse"] == pytest.approx(0.0)
    assert metrics["classical_rmse"] == pytest.approx(0.0)
    assert metrics["nse"] == pytest.approx(1.0)
    assert metrics["pearson_r"] == pytest.approx(1.0)
    assert metrics["csi_gamma_0_001"] == pytest.approx(1.0)
