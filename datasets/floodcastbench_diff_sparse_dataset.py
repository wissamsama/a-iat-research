from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from datasets.floodcastbench_fno_plus_official_v1_dataset import (
    build_fno_plus_official_v1_dataset,
)


WATER_DEPTH_CONTEXT_CHANNEL = 3


def target_step_to_index(config: dict[str, Any]) -> int:
    """Return a zero-based future-step index.

    The config field ``dataset.target_step`` is one-based: target_step=1 means
    the first future frame h2, stored at target[..., 0].
    """

    dataset_config = config.get("dataset", {})
    if "target_step_index" in dataset_config:
        index = int(dataset_config["target_step_index"])
    else:
        index = int(dataset_config.get("target_step", 1)) - 1
    if index < 0:
        raise ValueError("target_step/target_step_index must select a non-negative target index")
    return index


def extract_target_step(target_sequence: torch.Tensor, target_step_index: int = 0) -> torch.Tensor:
    """Extract one future water-depth map as [1, H, W] from [1, H, W, T]."""

    if target_sequence.ndim != 4:
        raise ValueError(f"Expected target_sequence [1, H, W, T], got {tuple(target_sequence.shape)}")
    if target_sequence.shape[0] != 1:
        raise ValueError(f"Expected singleton target channel, got {target_sequence.shape[0]}")
    if target_step_index >= target_sequence.shape[-1]:
        raise ValueError(
            f"target_step_index={target_step_index} is out of range for "
            f"{target_sequence.shape[-1]} available future steps"
        )
    return target_sequence[..., target_step_index].contiguous()


def make_context_mask(water_depth_context: torch.Tensor, missing_rate: float, mask_mode: str) -> torch.Tensor:
    """Build the DIFF-SPARSE observation mask for water-depth context.

    For the first dense sanity baseline, missing_rate=0.0 and mask_mode=all_ones
    produce an all-ones mask with the same shape as water_depth_context.
    """

    if water_depth_context.ndim != 4:
        raise ValueError(
            f"Expected water_depth_context [1, H, W, T], got {tuple(water_depth_context.shape)}"
        )
    missing_rate = float(missing_rate)
    mask_mode = str(mask_mode).lower()
    if missing_rate == 0.0 and mask_mode == "all_ones":
        return torch.ones_like(water_depth_context)
    raise NotImplementedError(
        "Only missing_rate=0.0 with mask_mode='all_ones' is implemented for the "
        "dense DIFF-SPARSE sanity baseline. Sparse masks are a later ablation."
    )


class FloodCastBenchDiffSparseDenseDataset(Dataset):
    """Dense missing-rate-zero DIFF-SPARSE-style wrapper for FloodCastBench.

    The wrapped official-v1 dataset keeps its existing normalization. This
    wrapper only changes the sample contract to a dictionary and extracts a
    one-step target for the first local conditional diffusion sanity baseline.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        config: dict[str, Any] | None = None,
        split: str = "train",
        normalization_stats: dict[str, Any] | None = None,
        base_dataset: Dataset | None = None,
    ) -> None:
        if config is None:
            config = {}
        self.config = config
        self.split = split
        self.target_step_index = target_step_to_index(config)
        masking_config = config.get("masking", {})
        self.missing_rate = float(masking_config.get("missing_rate", 0.0))
        self.mask_mode = str(masking_config.get("mask_mode", "all_ones"))

        if base_dataset is not None:
            self.base_dataset = base_dataset
        else:
            if root is None:
                raise ValueError("root is required when base_dataset is not provided")
            self.base_dataset = build_fno_plus_official_v1_dataset(
                root,
                config,
                split=split,
                normalization_stats=normalization_stats,
            )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        context, target_sequence, meta = self.base_dataset[index]
        if context.ndim != 4:
            raise ValueError(f"Expected context [6, H, W, 20], got {tuple(context.shape)}")
        if context.shape[0] != 6:
            raise ValueError(f"Expected 6 FNO+ context channels, got {context.shape[0]}")
        if target_sequence.ndim != 4:
            raise ValueError(f"Expected target [1, H, W, 19], got {tuple(target_sequence.shape)}")

        target = extract_target_step(target_sequence, self.target_step_index)
        water_depth_context = context[WATER_DEPTH_CONTEXT_CHANNEL : WATER_DEPTH_CONTEXT_CHANNEL + 1]
        context_mask = make_context_mask(water_depth_context, self.missing_rate, self.mask_mode)

        meta = dict(meta)
        meta.update(
            {
                "diff_sparse_variant": "dense_missing0_sanity_baseline",
                "target_step_index": int(self.target_step_index),
                "missing_rate": float(self.missing_rate),
                "mask_mode": self.mask_mode,
            }
        )
        return {
            "context": context.contiguous(),
            "context_mask": context_mask.contiguous(),
            "target": target,
            "meta": meta,
        }


def build_diff_sparse_dense_dataset(
    root: str | Path,
    config: dict[str, Any],
    split: str,
    normalization_stats: dict[str, Any] | None,
) -> FloodCastBenchDiffSparseDenseDataset:
    return FloodCastBenchDiffSparseDenseDataset(
        root=root,
        config=config,
        split=split,
        normalization_stats=normalization_stats,
    )
