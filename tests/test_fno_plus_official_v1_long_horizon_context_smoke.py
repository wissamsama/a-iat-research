from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

rasterio = pytest.importorskip("rasterio")

from tools.evaluate_floodcastbench_fno_plus_official_v1_long_horizon_rollout import evaluate


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
    shape = (10, 10)
    for index in range(n_frames):
        _write_tif(
            root / "High-fidelity flood forecasting" / "60m" / "Australia" / f"{index * 300}.tif",
            np.full(shape, 0.01 + index / 1000.0, dtype=np.float32),
        )
    _write_tif(root / "Relevant data" / "DEM" / "Australia_DEM.tif", np.ones((24, 24), dtype=np.float32))
    for index in range(10):
        _write_tif(
            root / "Relevant data" / "Rainfall" / "Australia flood" / f"rain_{index:03d}.tif",
            np.full((24, 24), index / 100.0, dtype=np.float32),
        )


def _write_run_dir(run_dir: Path, dataset_root: Path, context_length: int, window_length: int) -> None:
    n_frames = 130
    train = n_frames // window_length - 2
    config = {
        "paths": {"dataset_root": str(dataset_root)},
        "dataset": {
            "event": "australia",
            "fidelity": "high",
            "resolution": "60m",
            "sample_length": 20,
            "context_length": context_length,
            "stride": window_length,
            "split_counts": {"train": train, "val": 1, "test": 1},
        },
        "model": {
            "input_channels": 6,
            "modes": 3,
            "width": 4,
            "fourier_layers": 1,
            "output_steps": 19,
            "output_offset": context_length + 1,
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    stats = {
        "channels": {
            "initial_depth": {"mean": 0.0, "std": 1.0},
            "dem": {"mean": 0.0, "std": 1.0},
            "rainfall": {"mean": 0.0, "std": 1.0},
            "target_depth": {"mean": 0.0, "std": 1.0},
        }
    }
    (run_dir / "normalization_stats.json").write_text(json.dumps(stats), encoding="utf-8")
    return n_frames


def _write_checkpoint(checkpoint_path: Path, context_length: int) -> None:
    from models.fno_plus_official import FNOPlusOfficial3d

    model = FNOPlusOfficial3d(
        input_channels=6, output_steps=19, modes=3, width=4, fourier_layers=1, output_offset=context_length + 1
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": 1, "model_state_dict": model.state_dict()}, checkpoint_path)


@pytest.mark.parametrize("context_length", [0, 24])
def test_long_horizon_rollout_runs_with_and_without_context(tmp_path: Path, context_length: int) -> None:
    dataset_root = tmp_path / "data"
    window_length = context_length + 20
    n_frames = 130
    _make_tiny_floodcastbench(dataset_root, n_frames=n_frames)

    run_dir = tmp_path / "run"
    _write_run_dir(run_dir, dataset_root, context_length, window_length)
    checkpoint_path = tmp_path / "checkpoint.pth"
    _write_checkpoint(checkpoint_path, context_length)

    output_dir = tmp_path / "out"
    args = argparse.Namespace(
        run_dir=str(run_dir),
        checkpoint=str(checkpoint_path),
        checkpoint_name=f"ctx{context_length}_smoke",
        output_dir=str(output_dir),
        horizons=[19],
        gammas=[0.001, 0.01],
        device="cpu",
        force=False,
    )
    summary = evaluate(args)
    row = summary["metrics_by_horizon"][0]
    assert row["rollout_samples"] >= 1
    assert np.isfinite(row["current_relative_rmse"])
    assert np.isfinite(row["nse"])
