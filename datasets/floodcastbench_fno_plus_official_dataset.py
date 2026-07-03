from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.floodcastbench_fno_dataset import (
    DEFAULT_SPLITS,
    EVENTS,
    RAINFALL_FOLDERS,
    _event_key,
    _load_frames,
    _read_raster,
    _resize_array,
)


class FloodCastBenchFNOPlusOfficialDataset(Dataset):
    """Direct space-time FNO+ samples for FloodCastBench.

    Returned tensors:
        x: [6, H, W, 20]
           channels are X, Y, T, initial water depth, DEM, rainfall.
        target: [1, H, W, 19]
           water depth for t=2..20.
    """

    input_channels = 6
    output_steps = 19

    def __init__(
        self,
        root: str | Path,
        event: str = "australia",
        fidelity: str = "high",
        resolution: str = "60m",
        split: str = "train",
        sample_length: int = 20,
        stride: int = 20,
        split_counts: dict[str, int] | None = None,
    ) -> None:
        self.root = Path(root)
        self.event_key = _event_key(event)
        self.event = EVENTS[self.event_key]
        self.fidelity = str(fidelity).lower()
        self.resolution = str(resolution).lower()
        self.split = str(split).lower()
        self.sample_length = int(sample_length)
        self.stride = int(stride)

        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")
        if self.sample_length != 20:
            raise ValueError("The official FNO+ attempt expects sample_length=20.")

        self.water_dir = self._water_dir()
        self.frames = _load_frames(self.water_dir)
        self.sample_starts = self._split_sample_starts(split_counts)
        if not self.sample_starts:
            raise ValueError(f"No {self.split} samples available")

        self.height, self.width = _read_raster(self.frames[0].path).shape
        self.target_shape = (self.height, self.width)
        self.dem = self._load_dem()
        self.rainfall_frames = self._load_rainfall_frames()
        self.xy = self._build_xy()
        self.time = torch.linspace(0.0, 1.0, self.sample_length).view(1, 1, self.sample_length)

    def __len__(self) -> int:
        return len(self.sample_starts)

    def __getitem__(self, index: int):
        start = self.sample_starts[index]
        frames = self.frames[start : start + self.sample_length]
        water = [_read_raster(frame.path) for frame in frames]
        initial = torch.from_numpy(water[0]).float()
        dem = torch.from_numpy(self.dem).float()
        rainfall = torch.from_numpy(
            np.stack([self._rainfall_for_timestamp(frame.timestamp) for frame in frames], axis=-1)
        ).float()
        target = torch.from_numpy(np.stack(water[1:], axis=-1)).float().unsqueeze(0)

        x_coord = self.xy[0].unsqueeze(-1).expand(self.height, self.width, self.sample_length)
        y_coord = self.xy[1].unsqueeze(-1).expand(self.height, self.width, self.sample_length)
        t_coord = self.time.expand(self.height, self.width, self.sample_length)
        initial = initial.unsqueeze(-1).expand(self.height, self.width, self.sample_length)
        dem = dem.unsqueeze(-1).expand(self.height, self.width, self.sample_length)
        x = torch.stack([x_coord, y_coord, t_coord, initial, dem, rainfall], dim=0).float()

        meta = {
            "event": self.event,
            "fidelity": self.fidelity,
            "resolution": self.resolution,
            "split": self.split,
            "sample_index": int(index),
            "global_sample_index": int(start // self.stride),
            "input_timestamp": int(frames[0].timestamp),
            "target_timestamps": [int(frame.timestamp) for frame in frames[1:]],
            "water_paths": [str(frame.path) for frame in frames],
        }
        return x, target, meta

    def _water_dir(self) -> Path:
        if self.fidelity == "high":
            family = "High-fidelity flood forecasting"
        elif self.fidelity == "low":
            family = "Low-fidelity flood forecasting"
        else:
            raise ValueError("fidelity must be 'high' or 'low'")
        path = self.root / family / self.resolution / self.event
        if not path.exists():
            raise FileNotFoundError(f"Water-depth folder not found: {path}")
        return path

    def _split_sample_starts(self, split_counts: dict[str, int] | None) -> list[int]:
        all_starts = [
            start
            for start in range(0, len(self.frames), self.stride)
            if start + self.sample_length <= len(self.frames)
        ]
        counts = split_counts or DEFAULT_SPLITS.get((self.fidelity, self.event_key, self.resolution))
        if counts is None:
            raise ValueError("No split counts available for this FNO+ official dataset")
        train_count = int(counts.get("train", 0))
        val_count = int(counts.get("val", 0))
        test_count = int(counts.get("test", 0))
        total = train_count + val_count + test_count
        if total > len(all_starts):
            raise ValueError(f"Requested {total} samples, but only {len(all_starts)} windows are available")
        ranges = {
            "train": (0, train_count),
            "val": (train_count, train_count + val_count),
            "test": (train_count + val_count, total),
        }
        start, end = ranges[self.split]
        return all_starts[start:end]

    def _load_dem(self) -> np.ndarray:
        path = self.root / "Relevant data" / "DEM" / f"{self.event}_DEM.tif"
        if not path.exists():
            raise FileNotFoundError(f"DEM not found: {path}")
        return _resize_array(_read_raster(path), self.target_shape, mode="bilinear")

    def _load_rainfall_frames(self) -> list[Path]:
        folder = self.root / "Relevant data" / "Rainfall" / RAINFALL_FOLDERS[self.event_key]
        files = sorted(folder.glob("*.tif"))
        if not files:
            raise FileNotFoundError(f"No rainfall TIFF files found in {folder}")
        return files

    def _rainfall_for_timestamp(self, water_timestamp: int) -> np.ndarray:
        rainfall_index = min(int(water_timestamp // 1800), len(self.rainfall_frames) - 1)
        return _resize_array(_read_raster(self.rainfall_frames[rainfall_index]), self.target_shape, mode="bilinear")

    def _build_xy(self) -> torch.Tensor:
        y = torch.linspace(0.0, 1.0, self.height).view(1, self.height, 1).expand(1, self.height, self.width)
        x = torch.linspace(0.0, 1.0, self.width).view(1, 1, self.width).expand(1, self.height, self.width)
        return torch.cat([x, y], dim=0)


def build_fno_plus_official_datasets(root: str | Path, config: dict[str, Any]):
    dataset_config = config.get("dataset", {})
    common = {
        "root": root,
        "event": dataset_config.get("event", "australia"),
        "fidelity": dataset_config.get("fidelity", "high"),
        "resolution": dataset_config.get("resolution", "60m"),
        "sample_length": int(dataset_config.get("sample_length", 20)),
        "stride": int(dataset_config.get("stride", 20)),
        "split_counts": dataset_config.get("split_counts"),
    }
    return {
        split: FloodCastBenchFNOPlusOfficialDataset(split=split, **common)
        for split in ("train", "val", "test")
    }
