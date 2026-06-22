from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def finite_pair(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flattened finite pred/target values sharing the same valid-pixel mask."""
    pred_flat = pred.detach().float().reshape(-1)
    target_flat = target.detach().float().reshape(-1)
    mask = torch.isfinite(pred_flat) & torch.isfinite(target_flat)
    return pred_flat[mask], target_flat[mask]


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return math.nan
    return numerator / denominator


def water_depth_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_values, target_values = finite_pair(pred, target)
    if pred_values.numel() == 0:
        return {"mae": math.nan, "mse": math.nan, "rmse": math.nan, "nse": math.nan, "pearson_r": math.nan}

    diff = pred_values - target_values
    abs_error = torch.abs(diff)
    sq_error = diff * diff
    mse = float(sq_error.mean().item())
    mae = float(abs_error.mean().item())
    rmse = math.sqrt(mse)

    target_mean = target_values.mean()
    nse_denominator = torch.sum((target_values - target_mean) ** 2).item()
    nse = 1.0 - _safe_divide(float(sq_error.sum().item()), float(nse_denominator))

    pred_centered = pred_values - pred_values.mean()
    target_centered = target_values - target_mean
    pearson_denominator = torch.sqrt(torch.sum(pred_centered ** 2) * torch.sum(target_centered ** 2)).item()
    pearson_r = _safe_divide(float(torch.sum(pred_centered * target_centered).item()), float(pearson_denominator))

    return {"mae": mae, "mse": mse, "rmse": rmse, "nse": nse, "pearson_r": pearson_r}




def region_error_metrics(pred: torch.Tensor, target: torch.Tensor, region_mask: torch.Tensor) -> dict[str, float | int]:
    """Compute MAE/RMSE over finite pixels inside a boolean region mask."""
    pred_flat = pred.detach().float().reshape(-1)
    target_flat = target.detach().float().reshape(-1)
    mask_flat = region_mask.detach().bool().reshape(-1)
    finite_mask = torch.isfinite(pred_flat) & torch.isfinite(target_flat) & mask_flat
    count = int(finite_mask.sum().item())
    if count == 0:
        return {"count": 0, "mae": math.nan, "rmse": math.nan}
    diff = pred_flat[finite_mask] - target_flat[finite_mask]
    mae = float(torch.abs(diff).mean().item())
    mse = float((diff * diff).mean().item())
    return {"count": count, "mae": mae, "rmse": math.sqrt(mse)}

def binary_mask_metrics(pred_mask: torch.Tensor, target_mask: torch.Tensor) -> dict[str, float]:
    pred = pred_mask.detach().bool().reshape(-1)
    target = target_mask.detach().bool().reshape(-1)
    tp = int((pred & target).sum().item())
    fp = int((pred & (~target)).sum().item())
    fn = int(((~pred) & target).sum().item())
    return binary_counts_to_metrics(tp=tp, fp=fp, fn=fn)


def binary_counts_to_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    csi = _safe_divide(tp, tp + fp + fn)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall) if not (math.isnan(precision) or math.isnan(recall)) else math.nan
    return {
        "csi": csi,
        "iou": csi,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


@dataclass
class WaterDepthMetricAccumulator:
    count: int = 0
    abs_error_sum: float = 0.0
    sq_error_sum: float = 0.0
    pred_sum: float = 0.0
    pred_sq_sum: float = 0.0
    target_sum: float = 0.0
    target_sq_sum: float = 0.0
    pred_target_sum: float = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_values, target_values = finite_pair(pred, target)
        if pred_values.numel() == 0:
            return
        diff = pred_values - target_values
        self.count += int(pred_values.numel())
        self.abs_error_sum += float(torch.abs(diff).sum().item())
        self.sq_error_sum += float((diff * diff).sum().item())
        self.pred_sum += float(pred_values.sum().item())
        self.pred_sq_sum += float((pred_values * pred_values).sum().item())
        self.target_sum += float(target_values.sum().item())
        self.target_sq_sum += float((target_values * target_values).sum().item())
        self.pred_target_sum += float((pred_values * target_values).sum().item())

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            return {"mae": math.nan, "mse": math.nan, "rmse": math.nan, "nse": math.nan, "pearson_r": math.nan}
        mae = self.abs_error_sum / self.count
        mse = self.sq_error_sum / self.count
        rmse = math.sqrt(mse)
        nse_denominator = self.target_sq_sum - (self.target_sum * self.target_sum / self.count)
        nse = 1.0 - _safe_divide(self.sq_error_sum, nse_denominator)
        pred_var_sum = self.pred_sq_sum - (self.pred_sum * self.pred_sum / self.count)
        target_var_sum = self.target_sq_sum - (self.target_sum * self.target_sum / self.count)
        covariance_sum = self.pred_target_sum - (self.pred_sum * self.target_sum / self.count)
        pearson_r = _safe_divide(covariance_sum, math.sqrt(max(pred_var_sum, 0.0) * max(target_var_sum, 0.0)))
        return {"mae": mae, "mse": mse, "rmse": rmse, "nse": nse, "pearson_r": pearson_r}


@dataclass
class BinaryMetricAccumulator:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def update(self, pred_mask: torch.Tensor, target_mask: torch.Tensor) -> None:
        pred = pred_mask.detach().bool().reshape(-1)
        target = target_mask.detach().bool().reshape(-1)
        self.tp += int((pred & target).sum().item())
        self.fp += int((pred & (~target)).sum().item())
        self.fn += int(((~pred) & target).sum().item())

    def compute(self) -> dict[str, float]:
        return binary_counts_to_metrics(tp=self.tp, fp=self.fp, fn=self.fn)
