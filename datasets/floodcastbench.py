from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import rasterio
except ImportError as error:  # pragma: no cover - exercised only when dependency is missing
    rasterio = None
    RASTERIO_IMPORT_ERROR = error
else:
    RASTERIO_IMPORT_ERROR = None


EVENT_ALIASES = {
    "australia": "Australia",
    "australia flood": "Australia",
}


@dataclass(frozen=True)
class FloodCastFrame:
    timestamp: int
    path: Path


@dataclass(frozen=True)
class FloodCastSample:
    input_indices: tuple[int, ...]
    target_index: int


def derive_flood_mask(water_depth: torch.Tensor, gamma: float = 0.001) -> torch.Tensor:
    """Derive a flood mask on the fly without writing derived mask files."""
    return water_depth > gamma


def derive_propagation_path(
    current_water_depth: torch.Tensor,
    future_water_depth: torch.Tensor,
    gamma: float = 0.001,
) -> torch.Tensor:
    """Return future flooded pixels that were not flooded at the current step."""
    current_mask = derive_flood_mask(current_water_depth, gamma=gamma)
    future_mask = derive_flood_mask(future_water_depth, gamma=gamma)
    return future_mask & (~current_mask)


class FloodCastBenchWaterDepthDataset(Dataset):
    """Lazy FloodCastBench water-depth forecasting dataset.

    First supported setup:
    [D(t-4), D(t-3), D(t-2), D(t-1), D(t)] -> D(t+horizon)
    using one event/resolution folder of raw water-depth TIFF files.
    """

    def __init__(
        self,
        root: str | Path = "data/FloodCastBench",
        event: str = "Australia flood",
        fidelity: str = "high",
        resolution: str = "30m",
        input_window: int = 5,
        horizon: int = 20,
        split: str = "train",
        split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
        normalization: str | dict[str, Any] | None = "none",
    ) -> None:
        if rasterio is None:
            raise ImportError("rasterio is required to read FloodCastBench TIFF files.") from RASTERIO_IMPORT_ERROR

        self.root = Path(root)
        self.event = event
        self.event_folder = _normalize_event_folder(event)
        self.fidelity = fidelity.lower()
        self.resolution = str(resolution)
        self.input_window = int(input_window)
        self.horizon = int(horizon)
        self.split = split.lower()
        self.split_ratios = split_ratios
        self.normalization = _parse_normalization(normalization)

        if self.input_window < 1:
            raise ValueError("input_window must be >= 1.")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1.")
        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test.")

        self.source_dir = self._resolve_source_dir()
        self.frames = self._load_frame_index()
        self.samples = self._build_samples()

        if not self.samples:
            raise ValueError(
                "No valid FloodCastBench samples found for "
                f"event={self.event!r}, fidelity={self.fidelity!r}, "
                f"resolution={self.resolution!r}, input_window={self.input_window}, "
                f"horizon={self.horizon}, split={self.split!r}."
            )

        self.height, self.width = self._read_shape(self.frames[0].path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        input_frames = [self.frames[position] for position in sample.input_indices]
        target_frame = self.frames[sample.target_index]

        x_arrays = [self._read_water_depth(frame.path) for frame in input_frames]
        y_array = self._read_water_depth(target_frame.path)

        x = torch.from_numpy(np.stack(x_arrays, axis=0)).unsqueeze(1).float()
        y = torch.from_numpy(y_array).unsqueeze(0).float()

        x = self.normalize_water_depth(x)
        y = self.normalize_water_depth(y)

        meta = {
            "dataset": "FloodCastBench",
            "task": "water_depth_forecasting",
            "event": self.event,
            "event_folder": self.event_folder,
            "fidelity": self.fidelity,
            "resolution": self.resolution,
            "split": self.split,
            "input_window": self.input_window,
            "horizon": self.horizon,
            "normalization": self.normalization,
            "input_timestamps": [frame.timestamp for frame in input_frames],
            "target_timestamp": target_frame.timestamp,
            "input_paths": [str(frame.path) for frame in input_frames],
            "target_path": str(target_frame.path),
        }
        return x, y, meta

    def normalize_water_depth(self, tensor: torch.Tensor) -> torch.Tensor:
        mode = self.normalization["mode"]
        if mode == "none":
            return tensor
        if mode == "standard":
            mean = float(self.normalization["mean"])
            std = float(self.normalization["std"])
            if std == 0:
                raise ValueError("normalization std must be non-zero.")
            return (tensor - mean) / std
        if mode == "min_max":
            min_value = float(self.normalization["min"])
            max_value = float(self.normalization["max"])
            if max_value == min_value:
                raise ValueError("normalization max must be different from min.")
            return (tensor - min_value) / (max_value - min_value)
        raise ValueError(f"Unsupported normalization mode: {mode}")

    def denormalize_water_depth(self, tensor: torch.Tensor) -> torch.Tensor:
        mode = self.normalization["mode"]
        if mode == "none":
            return tensor
        if mode == "standard":
            return tensor * float(self.normalization["std"]) + float(self.normalization["mean"])
        if mode == "min_max":
            min_value = float(self.normalization["min"])
            max_value = float(self.normalization["max"])
            return tensor * (max_value - min_value) + min_value
        raise ValueError(f"Unsupported normalization mode: {mode}")

    @property
    def normalization_mode(self) -> str:
        return str(self.normalization.get("mode", "none"))

    @property
    def normalization_stats_path(self) -> str | None:
        return self.normalization.get("stats_path")

    def _resolve_source_dir(self) -> Path:
        if not self.root.exists():
            raise FileNotFoundError(f"FloodCastBench root folder not found: {self.root}")

        if self.fidelity == "high":
            folder = "High-fidelity flood forecasting"
        elif self.fidelity == "low":
            folder = "Low-fidelity flood forecasting"
        else:
            raise ValueError("fidelity must be 'high' or 'low'.")

        source_dir = self.root / folder / self.resolution / self.event_folder
        if not source_dir.exists():
            raise FileNotFoundError(f"FloodCastBench water-depth folder not found: {source_dir}")
        return source_dir

    def _load_frame_index(self) -> list[FloodCastFrame]:
        frames = []
        for path in self.source_dir.glob("*.tif"):
            timestamp = _parse_water_depth_timestamp(path)
            frames.append(FloodCastFrame(timestamp=timestamp, path=path))

        if not frames:
            raise FileNotFoundError(f"No water-depth TIFF files found in: {self.source_dir}")

        frames.sort(key=lambda frame: frame.timestamp)
        _validate_strictly_increasing(frames)
        return frames

    def _build_samples(self) -> list[FloodCastSample]:
        split_start, split_end = _split_bounds(len(self.frames), self.split, self.split_ratios)
        last_start = split_end - self.input_window - self.horizon
        if last_start < split_start:
            return []

        samples = []
        # A sample is assigned to a split only if every input frame and the target
        # frame stay inside that split's contiguous temporal frame range.
        for start in range(split_start, last_start + 1):
            input_indices = tuple(range(start, start + self.input_window))
            target_index = start + self.input_window - 1 + self.horizon
            if target_index >= split_end:
                continue
            if not self._has_expected_timestamps(input_indices, target_index):
                continue
            samples.append(FloodCastSample(input_indices=input_indices, target_index=target_index))
        return samples

    def _has_expected_timestamps(self, input_indices: tuple[int, ...], target_index: int) -> bool:
        timestamps = [self.frames[index].timestamp for index in input_indices]
        frame_step = _infer_frame_step(self.frames)
        expected_inputs = [timestamps[0] + offset * frame_step for offset in range(self.input_window)]
        expected_target = timestamps[-1] + self.horizon * frame_step
        return timestamps == expected_inputs and self.frames[target_index].timestamp == expected_target

    def _read_water_depth(self, path: Path) -> np.ndarray:
        with rasterio.open(path) as dataset:
            array = dataset.read(1).astype(np.float32)
            if dataset.nodata is not None:
                array = np.where(array == dataset.nodata, np.nan, array)
        return array

    def _read_shape(self, path: Path) -> tuple[int, int]:
        with rasterio.open(path) as dataset:
            return int(dataset.height), int(dataset.width)


def _normalize_event_folder(event: str) -> str:
    event_key = str(event).strip().lower()
    return EVENT_ALIASES.get(event_key, event.replace(" flood", "").replace(" Flood", "").strip())


def _parse_water_depth_timestamp(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as error:
        raise ValueError(f"Water-depth filename must be an integer timestamp: {path}") from error


def _validate_strictly_increasing(frames: list[FloodCastFrame]) -> None:
    timestamps = [frame.timestamp for frame in frames]
    if len(timestamps) != len(set(timestamps)):
        raise ValueError("Duplicate water-depth timestamps found.")
    if timestamps != sorted(timestamps):
        raise ValueError("Water-depth timestamps are not sorted.")


def _infer_frame_step(frames: list[FloodCastFrame]) -> int:
    if len(frames) < 2:
        raise ValueError("At least two frames are required to infer temporal step.")
    deltas = [
        frames[index + 1].timestamp - frames[index].timestamp
        for index in range(len(frames) - 1)
    ]
    positive_deltas = [delta for delta in deltas if delta > 0]
    if not positive_deltas:
        raise ValueError("Cannot infer positive temporal step from frame timestamps.")
    return int(np.median(positive_deltas))


def _split_bounds(total_frames: int, split: str, ratios: tuple[float, float, float]) -> tuple[int, int]:
    if len(ratios) != 3:
        raise ValueError("split_ratios must contain train, val, and test ratios.")
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("split_ratios cannot contain negative values.")
    ratio_sum = sum(ratios)
    if ratio_sum <= 0:
        raise ValueError("split_ratios sum must be positive.")

    train_ratio, val_ratio, _ = [ratio / ratio_sum for ratio in ratios]
    train_end = int(total_frames * train_ratio)
    val_end = train_end + int(total_frames * val_ratio)

    bounds = {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, total_frames),
    }
    start, end = bounds[split]
    if start >= end:
        raise ValueError(f"Split {split!r} is empty for {total_frames} frames.")
    return start, end


def _parse_normalization(normalization: str | dict[str, Any] | None) -> dict[str, Any]:
    if normalization is None or normalization == "none":
        return {"mode": "none"}
    if isinstance(normalization, str):
        if normalization not in {"none"}:
            raise ValueError("String normalization currently supports only 'none'.")
        return {"mode": normalization}

    mode = normalization.get("mode", normalization.get("type", "none"))
    if mode == "none":
        return {"mode": "none"}
    if mode in {"standard", "mean_std"}:
        if "stats_path" in normalization and normalization["stats_path"]:
            stats_path = Path(normalization["stats_path"])
            if not stats_path.is_absolute():
                stats_path = Path(__file__).resolve().parents[1] / stats_path
            with stats_path.open("r", encoding="utf-8") as file:
                stats = json.load(file)
            return {
                "mode": "standard",
                "mean": stats["mean"],
                "std": stats["std"],
                "stats_path": str(normalization["stats_path"]),
            }
        return {
            "mode": "standard",
            "mean": normalization["mean"],
            "std": normalization["std"],
            "stats_path": normalization.get("stats_path"),
        }
    if mode == "min_max":
        return {"mode": mode, "min": normalization["min"], "max": normalization["max"]}
    raise ValueError(f"Unsupported normalization config: {normalization}")
