from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.floodcastbench_diff_sparse_dataset import make_context_mask
from datasets.floodcastbench_fno_dataset import (
    DEFAULT_SPLITS,
    EVENTS,
    RAINFALL_FOLDERS,
    _event_key,
    _load_frames,
    _read_raster,
    _resize_array,
)
from datasets.floodcastbench_fno_plus_official_v1_dataset import compute_train_normalization_stats


WATER_DEPTH_CONTEXT_CHANNEL = 3
DEFAULT_H100_DIRECT_STATS = {
    "mean": 0.1200550028330472,
    "std": 0.31688976890064807,
    "min": 2.1956322598271072e-05,
    "max": 15.00886058807373,
    "count": 33326336,
    "pixel_count": 33326336,
    "train_sample_count": 116,
    "fit_split": "train",
    "train_only": True,
    "horizon_range": "h100",
}


def target_horizon_index_from_config(config: dict[str, Any]) -> int:
    dataset_config = config.get("dataset", {})
    index = int(dataset_config.get("target_horizon_index_from_h1", 99))
    if index < 1:
        raise ValueError("target_horizon_index_from_h1 must be >= 1; h100 uses 99 from h1")
    return index


def target_horizon_label_from_config(config: dict[str, Any]) -> str:
    dataset_config = config.get("dataset", {})
    label = str(dataset_config.get("target_horizon_label", f"h{target_horizon_index_from_config(config) + 1:02d}"))
    if not label.startswith("h"):
        raise ValueError(f"target_horizon_label should look like 'h100', got {label!r}")
    return label


def split_sample_starts(
    frame_count: int,
    sample_length: int,
    stride: int,
    split: str,
    split_counts: dict[str, int] | None,
    split_key: tuple[str, str, str],
) -> list[int]:
    all_starts = [
        start
        for start in range(0, frame_count, stride)
        if start + sample_length <= frame_count
    ]
    counts = split_counts or DEFAULT_SPLITS.get(split_key)
    if counts is None:
        raise ValueError(f"No split counts available for {split_key}")
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
    start_index, end_index = ranges[split]
    return all_starts[start_index:end_index]


def eligible_starts(starts: list[int], frame_count: int, target_horizon_index_from_h1: int) -> list[int]:
    return [start for start in starts if start + target_horizon_index_from_h1 < frame_count]


def excluded_samples(
    starts: list[int],
    frame_count: int,
    target_horizon_index_from_h1: int,
    stride: int,
) -> list[dict[str, int]]:
    excluded: list[dict[str, int]] = []
    required_horizon_label_number = target_horizon_index_from_h1 + 1
    for sample_index, start in enumerate(starts):
        if start + target_horizon_index_from_h1 >= frame_count:
            excluded.append(
                {
                    "sample_index": int(sample_index),
                    "global_sample_index": int(start // stride),
                    "start_index": int(start),
                    "available_max_horizon": int(frame_count - start),
                    "required_horizon": int(required_horizon_label_number),
                }
            )
    return excluded


def target_normalization_key(config: dict[str, Any]) -> str:
    normalization_config = config.get("normalization", {})
    target_config = config.get("target_normalization", {})
    return str(
        target_config.get(
            "key",
            normalization_config.get("target_normalization", "target_depth_h100_direct"),
        )
    )


def target_normalization_stats_from_config(config: dict[str, Any]) -> dict[str, Any]:
    target_config = config.get("target_normalization", {})
    stats = deepcopy(target_config.get("stats", DEFAULT_H100_DIRECT_STATS))
    stats.setdefault("name", target_normalization_key(config))
    stats.setdefault("fit_split", "train")
    stats.setdefault("train_only", True)
    stats.setdefault("horizon_range", target_horizon_label_from_config(config))
    if float(stats["std"]) <= 0.0:
        raise ValueError(f"Target normalization std must be positive, got {stats['std']}")
    return stats


def compute_high_horizon_normalization_stats(
    root: str | Path,
    config: dict[str, Any],
    min_std: float = 1e-6,
) -> dict[str, Any]:
    """Return context official-v1 stats plus train-only direct h100 target stats.

    Initial depth, DEM, and rainfall use the same official-v1 train-only
    standardization path as the h2 Dense DIFF-SPARSE baseline. The target uses
    audited direct h100 train-only stats so prediction and target are in the
    h100 target-normalized space.
    """

    stats = compute_train_normalization_stats(root, config, min_std=min_std)
    stats = deepcopy(stats)
    key = target_normalization_key(config)
    target_stats = target_normalization_stats_from_config(config)
    stats["version"] = "diff_sparse_high_horizon_context_official_v1_target_direct"
    stats["target_normalization_key"] = key
    stats["target_horizon_label"] = target_horizon_label_from_config(config)
    stats["target_horizon_index_from_h1"] = target_horizon_index_from_config(config)
    stats["channels"][key] = target_stats
    stats["channels"]["target_depth"] = target_stats
    return stats


class FloodCastBenchDiffSparseHighHorizonDataset(Dataset):
    """Dense missing-rate-zero DIFF-SPARSE-style raw-frame direct-horizon dataset.

    The context contract intentionally matches the h2 wrapper:
        context: [6, H, W, 20]
        context_mask: [1, H, W, 20]
        target: [1, H, W]

    For h100, the target frame is start + 99 relative to h1. Ineligible split
    samples are skipped and recorded in ``excluded_samples``.
    """

    def __init__(
        self,
        root: str | Path,
        config: dict[str, Any],
        split: str = "train",
        normalization_stats: dict[str, Any] | None = None,
    ) -> None:
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
        self.sample_length = int(dataset_config.get("sample_length", 20))
        self.stride = int(dataset_config.get("stride", 20))
        self.target_horizon_index_from_h1 = target_horizon_index_from_config(config)
        self.target_horizon_label = target_horizon_label_from_config(config)
        self.target_normalization_key = target_normalization_key(config)
        masking_config = config.get("masking", {})
        self.missing_rate = float(masking_config.get("missing_rate", 0.0))
        self.mask_mode = str(masking_config.get("mask_mode", "all_ones"))

        if self.sample_length != 20:
            raise ValueError("The direct high-horizon dataset currently expects sample_length=20 for context")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")

        self.water_dir = self._water_dir()
        self.frames = _load_frames(self.water_dir)
        self.all_sample_starts = split_sample_starts(
            frame_count=len(self.frames),
            sample_length=self.sample_length,
            stride=self.stride,
            split=self.split,
            split_counts=dataset_config.get("split_counts"),
            split_key=(self.fidelity, self.event_key, self.resolution),
        )
        self.sample_starts = eligible_starts(
            self.all_sample_starts,
            len(self.frames),
            self.target_horizon_index_from_h1,
        )
        self.excluded_samples = excluded_samples(
            self.all_sample_starts,
            len(self.frames),
            self.target_horizon_index_from_h1,
            self.stride,
        )
        if not self.sample_starts:
            raise ValueError(
                f"No {self.split} samples are eligible for {self.target_horizon_label} "
                f"with target_horizon_index_from_h1={self.target_horizon_index_from_h1}"
            )

        self.height, self.width = _read_raster(self.frames[0].path).shape
        self.target_shape = (self.height, self.width)
        self.dem = self._load_dem()
        self.rainfall_frames = self._load_rainfall_frames()
        self.xy = self._build_xy()
        self.time = torch.linspace(0.0, 1.0, self.sample_length).view(1, 1, self.sample_length)

    def __len__(self) -> int:
        return len(self.sample_starts)

    @property
    def eligible_sample_count(self) -> int:
        return len(self.sample_starts)

    @property
    def configured_sample_count(self) -> int:
        return len(self.all_sample_starts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        start = self.sample_starts[index]
        frames = self.frames[start : start + self.sample_length]
        water = [_read_raster(frame.path) for frame in frames]
        initial = torch.from_numpy(water[0]).float()
        dem = torch.from_numpy(self.dem).float()
        rainfall = torch.from_numpy(
            np.stack([self._rainfall_for_timestamp(frame.timestamp) for frame in frames], axis=-1)
        ).float()
        target_frame_index = start + self.target_horizon_index_from_h1
        target = torch.from_numpy(_read_raster(self.frames[target_frame_index].path)).float().unsqueeze(0)

        x_coord = self.xy[0].unsqueeze(-1).expand(self.height, self.width, self.sample_length)
        y_coord = self.xy[1].unsqueeze(-1).expand(self.height, self.width, self.sample_length)
        t_coord = self.time.expand(self.height, self.width, self.sample_length)
        initial = initial.unsqueeze(-1).expand(self.height, self.width, self.sample_length)
        dem = dem.unsqueeze(-1).expand(self.height, self.width, self.sample_length)
        context = torch.stack([x_coord, y_coord, t_coord, initial, dem, rainfall], dim=0).float()

        if self.normalization_stats is not None:
            context, target = self._normalize(context, target)

        water_depth_context = context[WATER_DEPTH_CONTEXT_CHANNEL : WATER_DEPTH_CONTEXT_CHANNEL + 1]
        context_mask = make_context_mask(water_depth_context, self.missing_rate, self.mask_mode)
        meta = {
            "event": self.event,
            "fidelity": self.fidelity,
            "resolution": self.resolution,
            "split": self.split,
            "sample_index": int(index),
            "global_sample_index": int(start // self.stride),
            "start_index": int(start),
            "input_timestamp": int(frames[0].timestamp),
            "context_timestamps": [int(frame.timestamp) for frame in frames],
            "target_timestamp": int(self.frames[target_frame_index].timestamp),
            "target_frame_index": int(target_frame_index),
            "target_horizon_label": self.target_horizon_label,
            "target_horizon_index_from_h1": int(self.target_horizon_index_from_h1),
            "target_normalization_key": self.target_normalization_key,
            "diff_sparse_variant": "dense_missing0_direct_h100_sanity_baseline",
            "missing_rate": float(self.missing_rate),
            "mask_mode": self.mask_mode,
            "water_paths": [str(frame.path) for frame in frames],
            "target_path": str(self.frames[target_frame_index].path),
        }
        return {
            "context": context.contiguous(),
            "context_mask": context_mask.contiguous(),
            "target": target.contiguous(),
            "meta": meta,
        }

    def _normalize(self, context: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        channels = self.normalization_stats["channels"]
        context = context.clone()
        target = target.clone()
        for name, channel_index in (("initial_depth", 3), ("dem", 4), ("rainfall", 5)):
            stats = channels[name]
            context[channel_index] = (context[channel_index] - float(stats["mean"])) / float(stats["std"])
        target_stats = channels.get(self.target_normalization_key, channels["target_depth"])
        target = (target - float(target_stats["mean"])) / float(target_stats["std"])
        return context, target

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


def build_diff_sparse_high_horizon_dataset(
    root: str | Path,
    config: dict[str, Any],
    split: str,
    normalization_stats: dict[str, Any] | None,
) -> FloodCastBenchDiffSparseHighHorizonDataset:
    return FloodCastBenchDiffSparseHighHorizonDataset(
        root=root,
        config=config,
        split=split,
        normalization_stats=normalization_stats,
    )
