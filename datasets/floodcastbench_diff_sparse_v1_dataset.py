from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.floodcastbench_fno_dataset import (
    EVENTS,
    RAINFALL_FOLDERS,
    _event_key,
    _load_frames,
    _read_raster,
    _resize_array,
)

try:
    import rasterio
    from rasterio.windows import Window as RasterWindow
except ImportError as error:  # pragma: no cover
    rasterio = None
    RasterWindow = None
    RASTERIO_IMPORT_ERROR = error
else:
    RASTERIO_IMPORT_ERROR = None


CANONICAL_WINDOW_LENGTH = 20
CANONICAL_SPLIT_COUNTS = {"train": 116, "val": 14, "test": 14}
RAINFALL_STEP_SECONDS = 1800


def split_frame_ranges(
    frame_count: int,
    split_counts: dict[str, int] | None = None,
    canonical_window: int = CANONICAL_WINDOW_LENGTH,
) -> dict[str, tuple[int, int]]:
    """Frame-index ranges derived from the canonical non-overlapping 20-frame windows.

    Using frame ranges (instead of window counts) lets context/prediction lengths
    change without moving the split boundaries, so results stay comparable with
    the existing FNO+/persistence work.
    """

    counts = split_counts or CANONICAL_SPLIT_COUNTS
    train_end = int(counts["train"]) * canonical_window
    val_end = train_end + int(counts["val"]) * canonical_window
    if val_end > frame_count:
        raise ValueError(f"Split counts require {val_end} frames, only {frame_count} available")
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, frame_count),
    }


def window_starts_for_split(
    frame_range: tuple[int, int],
    window_length: int,
    stride: int,
) -> list[int]:
    start, end = frame_range
    last_start = end - window_length
    if last_start < start:
        return []
    return list(range(start, last_start + 1, stride))


def generate_sensor_mask(
    height: int,
    width: int,
    missing_rate: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Static binary sensor mask [1, H, W]; 1 = sensor cell (paper: ~(1-missing_rate) of cells)."""

    missing_rate = float(missing_rate)
    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError(f"missing_rate must be in [0, 1], got {missing_rate}")
    total = height * width
    sensor_count = int(round((1.0 - missing_rate) * total))
    if sensor_count >= total:
        return torch.ones(1, height, width)
    mask = torch.zeros(total)
    if sensor_count > 0:
        chosen = torch.randperm(total, generator=generator)[:sensor_count]
        mask[chosen] = 1.0
    return mask.view(1, height, width)


def apply_observation_masking(
    water_context: torch.Tensor,
    sensor_mask: torch.Tensor,
    mask_mode: str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """DIFF-SPARSE masking (paper Algorithm 1, lines 3-5) in normalized space.

    'noise': non-sensor cells replaced by standard Gaussian noise (paper default).
    'zeros': non-sensor cells replaced by 0 (paper's baseline masking ablation).
    """

    if water_context.ndim != 3:
        raise ValueError(f"Expected water_context [C, H, W], got {tuple(water_context.shape)}")
    if sensor_mask.shape != (1, *water_context.shape[1:]):
        raise ValueError(
            f"sensor_mask must be [1, H, W] matching water_context, got {tuple(sensor_mask.shape)}"
        )
    mask_mode = str(mask_mode).lower()
    if torch.all(sensor_mask == 1.0):
        return water_context
    if mask_mode == "noise":
        if generator is None:
            fill = torch.randn_like(water_context)
        else:
            fill = torch.randn(
                water_context.shape,
                generator=generator,
                device=water_context.device,
                dtype=water_context.dtype,
            )
    elif mask_mode == "zeros":
        fill = torch.zeros_like(water_context)
    else:
        raise ValueError(f"Unsupported mask_mode {mask_mode!r}; expected 'noise' or 'zeros'")
    return water_context * sensor_mask + (1.0 - sensor_mask) * fill


def _empty_accumulator() -> dict[str, float]:
    return {"sum": 0.0, "sum_sq": 0.0, "count": 0.0, "min": math.inf, "max": -math.inf}


def _update_accumulator(acc: dict[str, float], array: np.ndarray) -> None:
    values = array.astype(np.float64)
    acc["sum"] += float(values.sum())
    acc["sum_sq"] += float((values * values).sum())
    acc["count"] += float(values.size)
    acc["min"] = min(acc["min"], float(values.min()))
    acc["max"] = max(acc["max"], float(values.max()))


def _finalize_accumulator(acc: dict[str, float], min_std: float) -> dict[str, float]:
    count = acc["count"]
    if count <= 0:
        raise ValueError("Cannot finalize an empty statistics accumulator")
    mean = acc["sum"] / count
    variance = max(acc["sum_sq"] / count - mean * mean, 0.0)
    return {
        "mean": mean,
        "std": max(math.sqrt(variance), float(min_std)),
        "min": acc["min"],
        "max": acc["max"],
        "count": int(count),
    }


def compute_v1_normalization_stats(
    root: str | Path,
    config: dict[str, Any],
    min_std: float = 1e-6,
) -> dict[str, Any]:
    """Train-only standardization stats for water, DEM, and rainfall.

    A single shared water statistic is used for context and targets, so
    persistence needs no re-targeting between normalization spaces.
    """

    root = Path(root)
    dataset_config = config.get("dataset", {})
    event_key = _event_key(dataset_config.get("event", "australia"))
    event = EVENTS[event_key]
    fidelity = str(dataset_config.get("fidelity", "high")).lower()
    resolution = str(dataset_config.get("resolution", "60m")).lower()

    family = "High-fidelity flood forecasting" if fidelity == "high" else "Low-fidelity flood forecasting"
    water_dir = root / family / resolution / event
    frames = _load_frames(water_dir)
    ranges = split_frame_ranges(len(frames), dataset_config.get("split_counts"))
    train_start, train_end = ranges["train"]

    water_acc = _empty_accumulator()
    for frame in frames[train_start:train_end]:
        _update_accumulator(water_acc, _read_raster(frame.path))

    target_shape = _read_raster(frames[0].path).shape
    dem_path = root / "Relevant data" / "DEM" / f"{event}_DEM.tif"
    dem = _resize_array(_read_raster(dem_path), target_shape, mode="bilinear")
    dem_acc = _empty_accumulator()
    _update_accumulator(dem_acc, dem)

    rainfall_dir = root / "Relevant data" / "Rainfall" / RAINFALL_FOLDERS[event_key]
    rainfall_files = sorted(rainfall_dir.glob("*.tif"))
    if not rainfall_files:
        raise FileNotFoundError(f"No rainfall TIFF files found in {rainfall_dir}")
    max_train_timestamp = frames[train_end - 1].timestamp
    max_rain_index = min(int(max_train_timestamp // RAINFALL_STEP_SECONDS), len(rainfall_files) - 1)
    rain_acc = _empty_accumulator()
    for path in rainfall_files[: max_rain_index + 1]:
        _update_accumulator(rain_acc, _read_raster(path))

    return {
        "version": "diff_sparse_v1_train_only_standardization",
        "fit_split": "train",
        "train_frame_range": [train_start, train_end],
        "min_std": float(min_std),
        "channels": {
            "water": _finalize_accumulator(water_acc, min_std),
            "dem": _finalize_accumulator(dem_acc, min_std),
            "rainfall": _finalize_accumulator(rain_acc, min_std),
        },
    }


class FloodCastBenchDiffSparseV1Dataset(Dataset):
    """DIFF-SPARSE v1 dataset: masked context window -> next-step targets.

    Sample contract (all water/rain/dem values standardized when stats given):
        context_water_masked  [c, ph, pw]   sensor-masked normalized history
        context_water_true    [c, ph, pw]   unmasked history (persistence/diagnostics)
        sensor_mask           [1, ph, pw]   static per-sample sensor mask
        dem                   [1, ph, pw]
        rainfall              [c+l, ph, pw] context + future forcing (dense, exogenous)
        timestamps            [c+l]         frame timestamps in seconds
        target                [l, ph, pw]   future water depth
        meta                  dict

    patch_mode 'random' crops a random patch (training); 'full' returns the whole
    field (evaluation tiles it). Masks in 'full' mode come from a seeded bank of
    eval_mask_bank_size masks applied round-robin by window index (paper protocol).
    """

    def __init__(
        self,
        root: str | Path,
        config: dict[str, Any],
        split: str = "train",
        normalization_stats: dict[str, Any] | None = None,
        patch_mode: str | None = None,
    ) -> None:
        if rasterio is None:
            raise ImportError("rasterio is required for FloodCastBench datasets") from RASTERIO_IMPORT_ERROR

        self.root = Path(root)
        self.config = config
        self.split = str(split).lower()
        self.normalization_stats = normalization_stats
        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        dataset_config = config.get("dataset", {})
        self.event_key = _event_key(dataset_config.get("event", "australia"))
        self.event = EVENTS[self.event_key]
        self.fidelity = str(dataset_config.get("fidelity", "high")).lower()
        self.resolution = str(dataset_config.get("resolution", "60m")).lower()
        self.context_length = int(dataset_config.get("context_length", 12))
        self.prediction_length = int(dataset_config.get("prediction_length", 8))
        self.window_length = self.context_length + self.prediction_length
        self.patch_size = int(dataset_config.get("patch_size", 64))
        self.include_dem = bool(dataset_config.get("include_dem", True))
        self.include_rainfall = bool(dataset_config.get("include_rainfall", True))
        if patch_mode is None:
            patch_mode = "random" if self.split == "train" else "full"
        self.patch_mode = str(patch_mode).lower()
        if self.patch_mode not in {"random", "full"}:
            raise ValueError("patch_mode must be 'random' or 'full'")
        if self.context_length < 1 or self.prediction_length < 1:
            raise ValueError("context_length and prediction_length must be >= 1")

        masking_config = config.get("masking", {})
        self.missing_rate = float(masking_config.get("missing_rate", 0.0))
        self.mask_mode = str(masking_config.get("mask_mode", "noise")).lower()
        self.eval_mask_bank_size = int(masking_config.get("eval_mask_bank_size", 10))
        self.eval_mask_seed = int(masking_config.get("eval_mask_seed", 1234))

        family = "High-fidelity flood forecasting" if self.fidelity == "high" else "Low-fidelity flood forecasting"
        self.water_dir = self.root / family / self.resolution / self.event
        if not self.water_dir.exists():
            raise FileNotFoundError(f"Water-depth folder not found: {self.water_dir}")
        self.frames = _load_frames(self.water_dir)
        self._validate_uniform_timestamps()

        ranges = split_frame_ranges(len(self.frames), dataset_config.get("split_counts"))
        self.frame_range = ranges[self.split]
        stride_key = "train_stride" if self.split == "train" else "eval_stride"
        default_stride = 1 if self.split == "train" else CANONICAL_WINDOW_LENGTH
        self.stride = int(dataset_config.get(stride_key, default_stride))
        self.window_starts = window_starts_for_split(self.frame_range, self.window_length, self.stride)
        if not self.window_starts:
            raise ValueError(
                f"No {self.split} windows fit frame range {self.frame_range} "
                f"with window_length={self.window_length}, stride={self.stride}"
            )

        self.height, self.width = _read_raster(self.frames[0].path).shape
        if self.patch_mode == "random" and (self.patch_size > self.height or self.patch_size > self.width):
            raise ValueError(f"patch_size={self.patch_size} exceeds field size {self.height}x{self.width}")
        self.target_shape = (self.height, self.width)
        self.dem = self._load_dem()
        self.rainfall_files = self._load_rainfall_files()
        self._rainfall_cache: dict[int, np.ndarray] = {}
        self._eval_mask_bank: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.window_starts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        start = self.window_starts[index]
        frames = self.frames[start : start + self.window_length]

        if self.patch_mode == "random":
            y0 = int(torch.randint(0, self.height - self.patch_size + 1, (1,)).item())
            x0 = int(torch.randint(0, self.width - self.patch_size + 1, (1,)).item())
            ph = pw = self.patch_size
        else:
            y0 = x0 = 0
            ph, pw = self.height, self.width

        water = np.stack(
            [self._read_water_patch(frame.path, y0, x0, ph, pw) for frame in frames], axis=0
        )
        water = torch.from_numpy(water).float()
        rainfall = torch.from_numpy(
            np.stack([self._rainfall_for_timestamp(frame.timestamp)[y0 : y0 + ph, x0 : x0 + pw] for frame in frames], axis=0)
        ).float()
        dem = torch.from_numpy(self.dem[y0 : y0 + ph, x0 : x0 + pw]).float().unsqueeze(0)
        timestamps = torch.tensor([float(frame.timestamp) for frame in frames], dtype=torch.float32)

        if self.normalization_stats is not None:
            channels = self.normalization_stats["channels"]
            water = (water - float(channels["water"]["mean"])) / float(channels["water"]["std"])
            dem = (dem - float(channels["dem"]["mean"])) / float(channels["dem"]["std"])
            rainfall = (rainfall - float(channels["rainfall"]["mean"])) / float(channels["rainfall"]["std"])

        context_true = water[: self.context_length]
        target = water[self.context_length :]

        if self.patch_mode == "full":
            mask_generator = torch.Generator().manual_seed(self.eval_mask_seed * 100003 + index)
            sensor_mask = self._eval_mask(index, ph, pw)
        else:
            mask_generator = None
            sensor_mask = generate_sensor_mask(ph, pw, self.missing_rate)
        context_masked = apply_observation_masking(
            context_true, sensor_mask, self.mask_mode, generator=mask_generator
        )

        meta = {
            "event": self.event,
            "fidelity": self.fidelity,
            "resolution": self.resolution,
            "split": self.split,
            "window_index": int(index),
            "window_start": int(start),
            "patch_origin": [int(y0), int(x0)],
            "patch_size": [int(ph), int(pw)],
            "context_length": int(self.context_length),
            "prediction_length": int(self.prediction_length),
            "missing_rate": float(self.missing_rate),
            "mask_mode": self.mask_mode,
            "context_timestamps": [int(frame.timestamp) for frame in frames[: self.context_length]],
            "target_timestamps": [int(frame.timestamp) for frame in frames[self.context_length :]],
        }
        return {
            "context_water_masked": context_masked.contiguous(),
            "context_water_true": context_true.contiguous(),
            "sensor_mask": sensor_mask.contiguous(),
            "dem": dem.contiguous(),
            "rainfall": rainfall.contiguous(),
            "timestamps": timestamps,
            "target": target.contiguous(),
            "meta": meta,
        }

    def _eval_mask(self, window_index: int, height: int, width: int) -> torch.Tensor:
        bank_slot = window_index % max(self.eval_mask_bank_size, 1)
        cached = self._eval_mask_bank.get(bank_slot)
        if cached is not None and cached.shape[1:] == (height, width):
            return cached
        generator = torch.Generator().manual_seed(self.eval_mask_seed + bank_slot)
        mask = generate_sensor_mask(height, width, self.missing_rate, generator=generator)
        self._eval_mask_bank[bank_slot] = mask
        return mask

    def _validate_uniform_timestamps(self) -> None:
        timestamps = [frame.timestamp for frame in self.frames]
        deltas = {timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)}
        if len(deltas) != 1:
            raise ValueError(f"Water-depth timestamps are not uniformly spaced; deltas={sorted(deltas)}")
        self.frame_step_seconds = deltas.pop()

    def _read_water_patch(self, path: Path, y0: int, x0: int, height: int, width: int) -> np.ndarray:
        if height == self.height and width == self.width:
            return _read_raster(path)
        with rasterio.open(path) as dataset:
            array = dataset.read(1, window=RasterWindow(x0, y0, width, height)).astype(np.float32)
            if dataset.nodata is not None:
                array = np.where(array == dataset.nodata, 0.0, array)
        return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _load_dem(self) -> np.ndarray:
        if not self.include_dem:
            return np.zeros(self.target_shape, dtype=np.float32)
        path = self.root / "Relevant data" / "DEM" / f"{self.event}_DEM.tif"
        if not path.exists():
            raise FileNotFoundError(f"DEM not found: {path}")
        return _resize_array(_read_raster(path), self.target_shape, mode="bilinear")

    def _load_rainfall_files(self) -> list[Path]:
        if not self.include_rainfall:
            return []
        folder = self.root / "Relevant data" / "Rainfall" / RAINFALL_FOLDERS[self.event_key]
        files = sorted(folder.glob("*.tif"))
        if not files:
            raise FileNotFoundError(f"No rainfall TIFF files found in {folder}")
        return files

    def _rainfall_for_timestamp(self, water_timestamp: int) -> np.ndarray:
        if not self.include_rainfall:
            return np.zeros(self.target_shape, dtype=np.float32)
        rainfall_index = min(int(water_timestamp // RAINFALL_STEP_SECONDS), len(self.rainfall_files) - 1)
        cached = self._rainfall_cache.get(rainfall_index)
        if cached is None:
            cached = _resize_array(_read_raster(self.rainfall_files[rainfall_index]), self.target_shape, mode="bilinear")
            self._rainfall_cache[rainfall_index] = cached
        return cached


def build_diff_sparse_v1_dataset(
    root: str | Path,
    config: dict[str, Any],
    split: str,
    normalization_stats: dict[str, Any] | None,
    patch_mode: str | None = None,
) -> FloodCastBenchDiffSparseV1Dataset:
    return FloodCastBenchDiffSparseV1Dataset(
        root=root,
        config=config,
        split=split,
        normalization_stats=normalization_stats,
        patch_mode=patch_mode,
    )
