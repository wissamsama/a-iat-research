from __future__ import annotations

import torch

from datasets.floodcastbench_diff_sparse_dataset import (
    FloodCastBenchDiffSparseDenseDataset,
    extract_target_step,
    make_context_mask,
)
from models.diff_sparse import DenseDiffSparseModel


class FakeOfficialV1Dataset:
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        context = torch.randn(6, 24, 24, 20)
        target = torch.randn(1, 24, 24, 19)
        return context, target, {"sample_index": index}


def _config() -> dict:
    return {
        "dataset": {"target_step": 1},
        "masking": {"missing_rate": 0.0, "mask_mode": "all_ones"},
        "model": {
            "name": "diff_sparse_dense",
            "context_channels": 6,
            "target_channels": 1,
            "input_timesteps": 20,
            "base_channels": 8,
            "channel_mults": [1, 2, 4],
        },
        "diffusion": {
            "steps": 10,
            "beta_schedule": "linear",
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "prediction_type": "x0",
        },
    }


def test_extract_target_step_shape() -> None:
    target_sequence = torch.randn(1, 32, 32, 19)
    target = extract_target_step(target_sequence, target_step_index=0)
    assert tuple(target.shape) == (1, 32, 32)
    assert torch.equal(target, target_sequence[..., 0])


def test_all_ones_mask_for_missing_zero() -> None:
    water_context = torch.randn(1, 32, 32, 20)
    mask = make_context_mask(water_context, missing_rate=0.0, mask_mode="all_ones")
    assert tuple(mask.shape) == tuple(water_context.shape)
    assert torch.all(mask == 1)


def test_dataset_wrapper_returns_dense_diff_sparse_dict() -> None:
    dataset = FloodCastBenchDiffSparseDenseDataset(config=_config(), base_dataset=FakeOfficialV1Dataset())
    sample = dataset[0]
    assert set(sample) == {"context", "context_mask", "target", "meta"}
    assert tuple(sample["context"].shape) == (6, 24, 24, 20)
    assert tuple(sample["context_mask"].shape) == (1, 24, 24, 20)
    assert tuple(sample["target"].shape) == (1, 24, 24)
    assert torch.all(sample["context_mask"] == 1)
    assert sample["meta"]["diff_sparse_variant"] == "dense_missing0_sanity_baseline"


def test_model_forward_with_fake_tensors() -> None:
    model = DenseDiffSparseModel(_config()).eval()
    context = torch.randn(2, 6, 32, 32, 20)
    context_mask = torch.ones(2, 1, 32, 32, 20)
    target = torch.randn(2, 1, 32, 32)
    timesteps = torch.tensor([0, 9], dtype=torch.long)
    noisy = model.q_sample(target, timesteps)
    with torch.no_grad():
        pred = model(noisy, timesteps, context, context_mask)
    assert tuple(pred.shape) == tuple(target.shape)
    assert torch.isfinite(pred).all()


def test_training_step_loss_returns_finite_scalar() -> None:
    model = DenseDiffSparseModel(_config())
    batch = {
        "context": torch.randn(2, 6, 32, 32, 20),
        "context_mask": torch.ones(2, 1, 32, 32, 20),
        "target": torch.randn(2, 1, 32, 32),
    }
    loss, diagnostics = model.training_step_loss(batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert diagnostics["pred_finite"] == 1.0
    assert diagnostics["target_finite"] == 1.0
