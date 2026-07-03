from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from datasets.floodcastbench_diff_sparse_high_horizon_dataset import (
    FloodCastBenchDiffSparseHighHorizonDataset,
    compute_high_horizon_normalization_stats,
    target_horizon_index_from_config,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIR / "configs" / "floodcastbench_diff_sparse_dense_missing0_highfid_60m_h100.yaml"


def _config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _data_root(config: dict) -> Path:
    return Path(config["paths"]["dataset_root"])


def _require_data(root: Path) -> None:
    water_dir = root / "High-fidelity flood forecasting" / "60m" / "Australia"
    if not water_dir.exists():
        pytest.skip(f"FloodCastBench data unavailable: {water_dir}")


def test_h100_config_selects_expected_target_index() -> None:
    config = _config()
    assert config["dataset"]["target_horizon_label"] == "h100"
    assert target_horizon_index_from_config(config) == 99


def test_h100_split_eligibility_and_exclusions() -> None:
    config = _config()
    root = _data_root(config)
    _require_data(root)
    train = FloodCastBenchDiffSparseHighHorizonDataset(root, config, split="train")
    val = FloodCastBenchDiffSparseHighHorizonDataset(root, config, split="val")
    test = FloodCastBenchDiffSparseHighHorizonDataset(root, config, split="test")
    assert len(train) == 116
    assert len(val) == 14
    assert len(test) == 10
    assert test.configured_sample_count == 14
    assert [item["sample_index"] for item in test.excluded_samples] == [10, 11, 12, 13]
    assert [item["available_max_horizon"] for item in test.excluded_samples] == [81, 61, 41, 21]


def test_h100_dataset_sample_shapes_mask_and_finiteness() -> None:
    config = _config()
    root = _data_root(config)
    _require_data(root)
    stats = compute_high_horizon_normalization_stats(root, config)
    dataset = FloodCastBenchDiffSparseHighHorizonDataset(root, config, split="train", normalization_stats=stats)
    sample = dataset[0]
    assert tuple(sample["context"].shape) == (6, 536, 536, 20)
    assert tuple(sample["context_mask"].shape) == (1, 536, 536, 20)
    assert tuple(sample["target"].shape) == (1, 536, 536)
    assert float(sample["context_mask"].min().item()) == pytest.approx(1.0)
    assert float(sample["context_mask"].max().item()) == pytest.approx(1.0)
    assert torch.isfinite(sample["target"]).all()
    assert sample["meta"]["target_horizon_label"] == "h100"
    assert sample["meta"]["target_horizon_index_from_h1"] == 99
    assert sample["meta"]["target_normalization_key"] == "target_depth_h100_direct"


def test_h100_ineligible_test_index_is_not_part_of_dataset() -> None:
    config = _config()
    root = _data_root(config)
    _require_data(root)
    dataset = FloodCastBenchDiffSparseHighHorizonDataset(root, config, split="test")
    assert len(dataset) == 10
    with pytest.raises(IndexError):
        _ = dataset[10]
