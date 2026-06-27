from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import rasterio
except ImportError as error:  # pragma: no cover
    rasterio = None
    RASTERIO_IMPORT_ERROR = error
else:
    RASTERIO_IMPORT_ERROR = None


EVENTS = {
    "australia": "Australia",
    "uk": "UK",
    "pakistan": "Pakistan",
    "mozambique": "Mozambique",
}

RAINFALL_FOLDERS = {
    "australia": "Australia flood",
    "uk": "UK flood",
    "pakistan": "Pakistan flood",
    "mozambique": "Mozambique flood",
}

DEFAULT_SPLITS = {
    ("high", "australia", "60m"): {"train": 116, "val": 14, "test": 14},
    ("high", "australia", "30m"): {"train": 116, "val": 14, "test": 14},
    ("low", "pakistan", "480m"): {"train": 145, "val": 28, "test": 28},
}


@dataclass(frozen=True)
class FNOFrame:
    timestamp: int
    path: Path


def _event_key(event: str) -> str:
    key = str(event).strip().lower().replace(" flood", "")
    if key not in EVENTS:
        raise ValueError(f"Unsupported event {event!r}; expected one of {sorted(EVENTS)}")
    return key


def _parse_int_stem(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(f"Expected integer water-depth timestamp filename: {path}") from exc


def _load_frames(folder: Path) -> list[FNOFrame]:
    frames = [FNOFrame(timestamp=_parse_int_stem(path), path=path) for path in folder.glob("*.tif")]
    frames.sort(key=lambda frame: frame.timestamp)
    if not frames:
        raise FileNotFoundError(f"No TIFF files found in {folder}")
    return frames


def _read_raster(path: Path) -> np.ndarray:
    if rasterio is None:
        raise ImportError("rasterio is required for FloodCastBench FNO datasets.") from RASTERIO_IMPORT_ERROR
    with rasterio.open(path) as dataset:
        array = dataset.read(1).astype(np.float32)
        if dataset.nodata is not None:
            array = np.where(array == dataset.nodata, 0.0, array)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _resize_array(array: np.ndarray, shape: tuple[int, int], mode: str = "bilinear") -> np.ndarray:
    if array.shape == shape:
        return array.astype(np.float32)
    tensor = torch.from_numpy(array).float().unsqueeze(0).unsqueeze(0)
    if mode == "nearest":
        resized = F.interpolate(tensor, size=shape, mode=mode)
    else:
        resized = F.interpolate(tensor, size=shape, mode=mode, align_corners=False)
    return resized.squeeze(0).squeeze(0).numpy().astype(np.float32)


class FloodCastBenchFNODataset(Dataset):
    """Paper-style 20-step FloodCastBench samples for FNO/FNO+.

    Samples are non-overlapping 20-frame windows:
    frame[0] is the initial water depth t=1, targets are frames[1:20].
    For Australia 60m this gives 144 samples from 2881 frames.
    """

    def __init__(
        self,
        root: str | Path,
        event: str = "australia",
        fidelity: str = "high",
        resolution: str = "60m",
        split: str = "train",
        sample_length: int = 20,
        stride: int = 20,
        include_dem: bool = True,
        include_rainfall: bool = True,
        include_time: bool = True,
        split_counts: dict[str, int] | None = None,
    ) -> None:
        if rasterio is None:
            raise ImportError("rasterio is required for FloodCastBench FNO datasets.") from RASTERIO_IMPORT_ERROR
        self.root = Path(root)
        self.event_key = _event_key(event)
        self.event = EVENTS[self.event_key]
        self.fidelity = str(fidelity).lower()
        self.resolution = str(resolution).lower()
        self.split = str(split).lower()
        self.sample_length = int(sample_length)
        self.stride = int(stride)
        self.include_dem = bool(include_dem)
        self.include_rainfall = bool(include_rainfall)
        self.include_time = bool(include_time)

        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")
        if self.sample_length != 20:
            raise ValueError("This FNO+ reproduction scaffold currently expects sample_length=20.")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")

        self.water_dir = self._water_dir()
        self.frames = _load_frames(self.water_dir)
        self.sample_starts = self._split_sample_starts(split_counts)
        if not self.sample_starts:
            raise ValueError(f"No {self.split} samples available for {self.event} {self.resolution}")

        self.height, self.width = _read_raster(self.frames[0].path).shape
        self.target_shape = (self.height, self.width)
        self.dem = self._load_dem() if self.include_dem else np.zeros(self.target_shape, dtype=np.float32)
        self.rainfall_frames = self._load_rainfall_frames() if self.include_rainfall else []
        self.coord_channels = self._build_coord_channels()
        self.time_channels = self._build_time_channels() if self.include_time else torch.empty(0, self.height, self.width)
        self.input_channels = 2 + 1 + int(self.include_dem) + (20 if self.include_rainfall else 0) + (
            19 if self.include_time else 0
        )
        self.output_steps = 19

    def __len__(self) -> int:
        return len(self.sample_starts)

    def __getitem__(self, index: int):
        start = self.sample_starts[index]
        frames = self.frames[start : start + self.sample_length]
        water = [_read_raster(frame.path) for frame in frames]
        initial = torch.from_numpy(water[0]).unsqueeze(0)
        target = torch.from_numpy(np.stack(water[1:], axis=0)).float()

        channels = [self.coord_channels, initial]
        if self.include_dem:
            channels.append(torch.from_numpy(self.dem).unsqueeze(0))
        if self.include_rainfall:
            rainfall = [self._rainfall_for_timestamp(frame.timestamp) for frame in frames]
            channels.append(torch.from_numpy(np.stack(rainfall, axis=0)).float())
        if self.include_time:
            channels.append(self.time_channels)
        x = torch.cat(channels, dim=0).float()

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
            raise ValueError(
                "No default split counts for "
                f"fidelity={self.fidelity}, event={self.event_key}, resolution={self.resolution}"
            )
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
        # Rainfall files are every 1800 seconds while water-depth files are every
        # 300 seconds. Use the latest available rainfall frame at or before the
        # requested water-depth timestamp.
        rainfall_index = min(int(water_timestamp // 1800), len(self.rainfall_frames) - 1)
        return _resize_array(_read_raster(self.rainfall_frames[rainfall_index]), self.target_shape, mode="bilinear")

    def _build_coord_channels(self) -> torch.Tensor:
        y = torch.linspace(0.0, 1.0, self.height).view(1, self.height, 1).expand(1, self.height, self.width)
        x = torch.linspace(0.0, 1.0, self.width).view(1, 1, self.width).expand(1, self.height, self.width)
        return torch.cat([x, y], dim=0)

    def _build_time_channels(self) -> torch.Tensor:
        values = torch.linspace(1.0 / 19.0, 1.0, 19).view(19, 1, 1)
        return values.expand(19, self.height, self.width).clone()


def build_fno_datasets(root: str | Path, config: dict[str, Any]) -> dict[str, FloodCastBenchFNODataset]:
    dataset_config = config.get("dataset", {})
    common = {
        "root": root,
        "event": dataset_config.get("event", "australia"),
        "fidelity": dataset_config.get("fidelity", "high"),
        "resolution": dataset_config.get("resolution", "60m"),
        "sample_length": int(dataset_config.get("sample_length", 20)),
        "stride": int(dataset_config.get("stride", 20)),
        "include_dem": bool(dataset_config.get("include_dem", True)),
        "include_rainfall": bool(dataset_config.get("include_rainfall", True)),
        "include_time": bool(dataset_config.get("include_time", True)),
        "split_counts": dataset_config.get("split_counts"),
    }
    return {
        split: FloodCastBenchFNODataset(split=split, **common)
        for split in ("train", "val", "test")
    }
