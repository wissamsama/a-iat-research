from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from datasets.floodcastbench_fno_plus_official_dataset import FloodCastBenchFNOPlusOfficialDataset


PHYSICAL_INPUT_CHANNELS = {
    "initial_depth": 3,
    "dem": 4,
    "rainfall": 5,
}
TARGET_KEY = "target_depth"


def _empty_accumulator() -> dict[str, float]:
    return {
        "sum": 0.0,
        "sum_sq": 0.0,
        "count": 0.0,
        "min": math.inf,
        "max": -math.inf,
    }


def _update_accumulator(accumulator: dict[str, float], tensor: torch.Tensor) -> None:
    values = tensor.detach().double()
    accumulator["sum"] += float(values.sum().item())
    accumulator["sum_sq"] += float((values * values).sum().item())
    accumulator["count"] += float(values.numel())
    accumulator["min"] = min(accumulator["min"], float(values.min().item()))
    accumulator["max"] = max(accumulator["max"], float(values.max().item()))


def _finalize_accumulator(accumulator: dict[str, float], min_std: float) -> dict[str, float]:
    count = accumulator["count"]
    if count <= 0:
        raise ValueError("Cannot compute normalization statistics from an empty accumulator")
    mean = accumulator["sum"] / count
    variance = max(accumulator["sum_sq"] / count - mean * mean, 0.0)
    std = max(math.sqrt(variance), float(min_std))
    return {
        "mean": mean,
        "std": std,
        "min": accumulator["min"],
        "max": accumulator["max"],
        "count": int(count),
    }


def compute_train_normalization_stats(
    root: str | Path,
    config: dict[str, Any],
    min_std: float = 1e-6,
) -> dict[str, Any]:
    """Compute official-v1 normalization statistics on the train split only."""

    dataset_config = config.get("dataset", {})
    train_dataset = FloodCastBenchFNOPlusOfficialDataset(
        root=root,
        event=dataset_config.get("event", "australia"),
        fidelity=dataset_config.get("fidelity", "high"),
        resolution=dataset_config.get("resolution", "60m"),
        split="train",
        sample_length=int(dataset_config.get("sample_length", 20)),
        stride=int(dataset_config.get("stride", 20)),
        split_counts=dataset_config.get("split_counts"),
        context_length=int(dataset_config.get("context_length", 0)),
    )
    accumulators = {
        "initial_depth": _empty_accumulator(),
        "dem": _empty_accumulator(),
        "rainfall": _empty_accumulator(),
        TARGET_KEY: _empty_accumulator(),
    }
    for index in range(len(train_dataset)):
        x, target, _ = train_dataset[index]
        for name, channel_index in PHYSICAL_INPUT_CHANNELS.items():
            _update_accumulator(accumulators[name], x[channel_index])
        _update_accumulator(accumulators[TARGET_KEY], target)

    return {
        "version": "official_v1_train_only_standardization",
        "fit_split": "train",
        "min_std": float(min_std),
        "sample_count": len(train_dataset),
        "channels": {
            name: _finalize_accumulator(accumulator, min_std)
            for name, accumulator in accumulators.items()
        },
    }


class FloodCastBenchFNOPlusOfficialV1Dataset(FloodCastBenchFNOPlusOfficialDataset):
    """Official-v1 FNO+ dataset with train-only physical-channel standardization.

    X, Y, and T channels are left unchanged. Initial water depth, DEM, rainfall,
    and target water depth are standardized with statistics computed on the
    train split only.
    """

    def __init__(
        self,
        *args,
        normalization_stats: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.normalization_stats = normalization_stats

    def __getitem__(self, index: int):
        x, target, meta = super().__getitem__(index)
        if self.normalization_stats is None:
            return x, target, meta

        x = x.clone()
        target = target.clone()
        channels = self.normalization_stats["channels"]
        for name, channel_index in PHYSICAL_INPUT_CHANNELS.items():
            stats = channels[name]
            x[channel_index] = (x[channel_index] - float(stats["mean"])) / float(stats["std"])
        target_stats = channels[TARGET_KEY]
        target = (target - float(target_stats["mean"])) / float(target_stats["std"])
        meta = dict(meta)
        meta["normalization_version"] = self.normalization_stats.get("version")
        return x, target, meta

    def inverse_transform_target(self, target: torch.Tensor) -> torch.Tensor:
        if self.normalization_stats is None:
            return target
        stats = self.normalization_stats["channels"][TARGET_KEY]
        return target * float(stats["std"]) + float(stats["mean"])


def build_fno_plus_official_v1_dataset(
    root: str | Path,
    config: dict[str, Any],
    split: str,
    normalization_stats: dict[str, Any] | None,
) -> FloodCastBenchFNOPlusOfficialV1Dataset:
    dataset_config = config.get("dataset", {})
    return FloodCastBenchFNOPlusOfficialV1Dataset(
        root=root,
        event=dataset_config.get("event", "australia"),
        fidelity=dataset_config.get("fidelity", "high"),
        resolution=dataset_config.get("resolution", "60m"),
        split=split,
        sample_length=int(dataset_config.get("sample_length", 20)),
        stride=int(dataset_config.get("stride", 20)),
        split_counts=dataset_config.get("split_counts"),
        context_length=int(dataset_config.get("context_length", 0)),
        normalization_stats=normalization_stats,
    )
