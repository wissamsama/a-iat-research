from __future__ import annotations

import math

import torch

from metrics import BinaryMetricAccumulator, WaterDepthMetricAccumulator


def gamma_suffix(gamma: float) -> str:
    return str(gamma).replace(".", "_")


def tensor_stats(tensor: torch.Tensor, prefix: str) -> dict[str, float]:
    values = tensor.detach().float()
    stats = {
        f"{prefix}_min": float(values.min().item()),
        f"{prefix}_max": float(values.max().item()),
        f"{prefix}_mean": float(values.mean().item()),
    }
    if prefix == "pred":
        stats["negative_prediction_ratio"] = float((values < 0).float().mean().item())
    return stats


class ForecastMetricBundle:
    def __init__(self, gammas: tuple[float, ...]):
        self.gammas = gammas
        self.water = WaterDepthMetricAccumulator()
        self.mask = {gamma: BinaryMetricAccumulator() for gamma in gammas}
        self.path = {gamma: BinaryMetricAccumulator() for gamma in gammas}

    def update(self, pred: torch.Tensor, target: torch.Tensor, current: torch.Tensor) -> None:
        self.water.update(pred, target)
        for gamma in self.gammas:
            current_mask = current > gamma
            pred_mask = pred > gamma
            target_mask = target > gamma
            self.mask[gamma].update(pred_mask, target_mask)
            self.path[gamma].update(pred_mask & (~current_mask), target_mask & (~current_mask))

    def compute(self) -> dict[str, float]:
        metrics = dict(self.water.compute())
        for gamma, acc in self.mask.items():
            metrics[f"csi_gamma_{gamma_suffix(gamma)}"] = acc.compute()["csi"]
        for gamma, acc in self.path.items():
            metrics[f"path_iou_gamma_{gamma_suffix(gamma)}"] = acc.compute()["iou"]
        return metrics


class RawClampedMetricBundle:
    def __init__(self, gammas: tuple[float, ...]):
        self.raw = ForecastMetricBundle(gammas)
        self.clamped = ForecastMetricBundle(gammas)
        self.loss_sum = 0.0
        self.batches = 0
        self.first_batch_stats: dict[str, float] = {}

    def update(
        self,
        pred_raw: torch.Tensor,
        target: torch.Tensor,
        current: torch.Tensor,
        loss_value: float | None = None,
    ) -> None:
        pred_clamped = torch.clamp(pred_raw, min=0.0)
        self.raw.update(pred_raw, target, current)
        self.clamped.update(pred_clamped, target, current)
        if not self.first_batch_stats:
            self.first_batch_stats.update(tensor_stats(pred_raw, "pred"))
            self.first_batch_stats.update(tensor_stats(target, "target"))
            self.first_batch_stats.update(tensor_stats(current, "input_current"))
        if loss_value is not None:
            self.loss_sum += float(loss_value)
        self.batches += 1

    def compute(self) -> dict[str, float | dict[str, float]]:
        raw = self.raw.compute()
        clamped = self.clamped.compute()
        metrics: dict[str, float | dict[str, float]] = {
            "raw": raw,
            "clamped": clamped,
            "loss": self.loss_sum / self.batches if self.batches else math.nan,
            "batches": self.batches,
        }
        metrics.update(self.first_batch_stats)
        return metrics


def flatten_metrics(metrics: dict, variant: str = "raw") -> dict[str, float]:
    selected = metrics.get(variant, {})
    flat = {
        "mae": selected.get("mae", math.nan),
        "mse": selected.get("mse", math.nan),
        "rmse": selected.get("rmse", math.nan),
        "nse": selected.get("nse", math.nan),
        "pearson_r": selected.get("pearson_r", math.nan),
        "csi_gamma_0_001": selected.get("csi_gamma_0_001", math.nan),
        "csi_gamma_0_01": selected.get("csi_gamma_0_01", math.nan),
        "path_iou_gamma_0_001": selected.get("path_iou_gamma_0_001", math.nan),
        "path_iou_gamma_0_01": selected.get("path_iou_gamma_0_01", math.nan),
        "loss": metrics.get("loss", math.nan),
        "batches": metrics.get("batches", 0),
    }
    for key in ("pred_min", "pred_max", "pred_mean", "negative_prediction_ratio", "target_min", "target_max", "target_mean", "input_current_min", "input_current_max", "input_current_mean"):
        if key in metrics:
            flat[key] = metrics[key]
    return flat
