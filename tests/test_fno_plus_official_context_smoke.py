from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

rasterio = pytest.importorskip("rasterio")

from datasets.floodcastbench_fno_plus_official_dataset import FloodCastBenchFNOPlusOfficialDataset
from models.fno_plus_official import FNOPlusOfficial3d


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


def _make_tiny_floodcastbench(root: Path, n_frames: int) -> None:
    shape = (12, 12)
    for index in range(n_frames):
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


def test_context_zero_matches_original_broadcast_behavior(tmp_path: Path) -> None:
    """context_length=0 must reproduce the original single-frame-broadcast
    tensor exactly (regression test for the WPB0 refactor)."""

    _make_tiny_floodcastbench(tmp_path, n_frames=40)
    baseline = FloodCastBenchFNOPlusOfficialDataset(
        root=tmp_path,
        split="train",
        split_counts={"train": 1, "val": 1, "test": 0},
    )
    explicit_zero = FloodCastBenchFNOPlusOfficialDataset(
        root=tmp_path,
        split="train",
        split_counts={"train": 1, "val": 1, "test": 0},
        context_length=0,
    )
    x0, target0, meta0 = baseline[0]
    x1, target1, meta1 = explicit_zero[0]
    assert torch.equal(x0, x1)
    assert torch.equal(target0, target1)
    assert meta0["target_timestamps"] == meta1["target_timestamps"]


def test_context_length_24_shapes_and_no_leakage(tmp_path: Path) -> None:
    n_frames = 90
    _make_tiny_floodcastbench(tmp_path, n_frames=n_frames)
    dataset = FloodCastBenchFNOPlusOfficialDataset(
        root=tmp_path,
        split="train",
        context_length=24,
        stride=44,
        split_counts={"train": 1, "val": 1, "test": 0},
    )
    x, target, meta = dataset[0]
    assert dataset.window_length == 44
    assert x.shape == (6, 12, 12, 44)
    assert target.shape == (1, 12, 12, 19)
    assert meta["context_length"] == 24
    assert len(meta["history_timestamps"]) == 24
    assert len(meta["target_timestamps"]) == 19
    # input_timestamp is the "current" frame (25th frame, index 24), not the
    # first history frame -- matches the original single-context convention.
    assert meta["input_timestamp"] == 24 * 300

    depth_channel = x[3]
    # The 24 history positions must hold their own real (distinct) values,
    # not a repeated constant -- this is the whole point of WPB0.
    history_values = depth_channel[0, 0, :24]
    assert len(torch.unique(history_values)) == 24
    # The 19 target positions must all equal the "current" frame's value
    # (broadcast forward, same convention as context_length=0), never a
    # target/future value -- i.e. no leakage of the prediction target into x.
    current_value = depth_channel[0, 0, 24]
    future_values = depth_channel[0, 0, 25:]
    assert torch.allclose(future_values, current_value.expand_as(future_values))
    assert not torch.allclose(future_values, target[0, 0, 0, :])


def test_context_model_forward_matches_output_offset() -> None:
    model = FNOPlusOfficial3d(
        input_channels=6,
        output_steps=19,
        modes=4,
        width=8,
        fourier_layers=2,
        output_offset=25,
    )
    x = torch.randn(1, 6, 10, 10, 44)
    target = torch.randn(1, 1, 10, 10, 19)
    pred = model(x)
    assert pred.shape == target.shape
    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()
    assert torch.isfinite(loss)


def test_context_model_output_offset_default_matches_vanilla() -> None:
    """output_offset defaults to 1, exactly reproducing the original
    x[..., 1:20] slicing for context_length=0 / T=20 inputs."""

    torch.manual_seed(0)
    model_default = FNOPlusOfficial3d(input_channels=6, output_steps=19, modes=4, width=8, fourier_layers=2)
    model_explicit = FNOPlusOfficial3d(
        input_channels=6, output_steps=19, modes=4, width=8, fourier_layers=2, output_offset=1
    )
    model_explicit.load_state_dict(model_default.state_dict())
    x = torch.randn(1, 6, 12, 12, 20)
    assert torch.equal(model_default(x), model_explicit(x))
