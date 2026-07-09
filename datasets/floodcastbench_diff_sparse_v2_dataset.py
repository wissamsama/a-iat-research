from __future__ import annotations

"""DIFF-SPARSE V2 dataset: a thin subclass of the (frozen) V1 dataset.

Adds, without touching any V1 file:
  1. Dihedral data augmentation (8 flips/rotations) for training samples,
     applied consistently to every spatial tensor. Physically consistent:
     gravity enters only through the DEM channel, which is transformed
     together with water/rain/masks. Relevant because the benchmark is a
     single flood event over a single region (2320 train frames) -- spatial
     augmentation is the main regularization available.
  2. Optional randomized training sparsity: masking.missing_rate_range
     samples a missing rate uniformly per training sample, producing one
     model robust across sparsity levels (the paper's own "different sensor
     configurations without retraining" claim, extended to levels). Off by
     default; evaluation always uses the fixed masking.missing_rate.
"""

from pathlib import Path
from typing import Any

import torch

from datasets.floodcastbench_diff_sparse_v1_dataset import FloodCastBenchDiffSparseV1Dataset


_SPATIAL_KEYS = ("context_water_masked", "context_water_true", "sensor_mask", "dem", "rainfall", "target")


def apply_dihedral(tensor: torch.Tensor, transform_index: int) -> torch.Tensor:
    """Apply one of the 8 dihedral-group transforms to the last two dims."""

    transform_index = int(transform_index) % 8
    rotations = transform_index % 4
    flip = transform_index >= 4
    if flip:
        tensor = torch.flip(tensor, dims=(-1,))
    if rotations:
        tensor = torch.rot90(tensor, k=rotations, dims=(-2, -1))
    return tensor.contiguous()


class FloodCastBenchDiffSparseV2Dataset(FloodCastBenchDiffSparseV1Dataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        dataset_config = self.config.get("dataset", {})
        masking_config = self.config.get("masking", {})
        self.augmentation = bool(dataset_config.get("augmentation", False)) and self.patch_mode == "random"
        rate_range = masking_config.get("missing_rate_range")
        if rate_range is not None and self.patch_mode == "random":
            low, high = float(rate_range[0]), float(rate_range[1])
            if not 0.0 <= low <= high <= 1.0:
                raise ValueError(f"missing_rate_range must satisfy 0 <= low <= high <= 1, got {rate_range}")
            self.missing_rate_range: tuple[float, float] | None = (low, high)
        else:
            self.missing_rate_range = None

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.missing_rate_range is not None:
            low, high = self.missing_rate_range
            self.missing_rate = low + (high - low) * float(torch.rand(1).item())
        sample = super().__getitem__(index)
        if self.augmentation:
            transform_index = int(torch.randint(0, 8, (1,)).item())
            if transform_index:
                for key in _SPATIAL_KEYS:
                    sample[key] = apply_dihedral(sample[key], transform_index)
            sample["meta"]["augmentation_dihedral"] = transform_index
        return sample


def build_diff_sparse_v2_dataset(
    root: str | Path,
    config: dict[str, Any],
    split: str = "train",
    normalization_stats: dict[str, Any] | None = None,
    patch_mode: str | None = None,
) -> FloodCastBenchDiffSparseV2Dataset:
    return FloodCastBenchDiffSparseV2Dataset(
        root,
        config,
        split=split,
        normalization_stats=normalization_stats,
        patch_mode=patch_mode,
    )
