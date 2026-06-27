from __future__ import annotations

import math

import torch


def relative_rmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Relative RMSE matching the paper-style normalized error.

    Computed as sqrt(sum((pred-target)^2) / (sum(target^2) + eps)).
    """

    pred = pred.float()
    target = target.float()
    return torch.sqrt(torch.sum((pred - target) ** 2) / (torch.sum(target**2) + eps))


def nse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    target_mean = torch.mean(target)
    return 1.0 - torch.sum((pred - target) ** 2) / (torch.sum((target - target_mean) ** 2) + eps)


def pearson_r(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    pred = pred.float().reshape(-1)
    target = target.float().reshape(-1)
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    numerator = torch.sum(pred_centered * target_centered)
    denominator = torch.sqrt(torch.sum(pred_centered**2) * torch.sum(target_centered**2) + eps)
    return numerator / denominator


def csi(pred: torch.Tensor, target: torch.Tensor, gamma: float, eps: float = 1e-12) -> torch.Tensor:
    pred_mask = pred > gamma
    target_mask = target > gamma
    hits = torch.logical_and(pred_mask, target_mask).sum().float()
    misses = torch.logical_and(~pred_mask, target_mask).sum().float()
    false_alarms = torch.logical_and(pred_mask, ~target_mask).sum().float()
    return hits / (hits + misses + false_alarms + eps)


class OfficialFloodMetricAccumulator:
    """Accumulates paper metrics over batches without storing full predictions."""

    def __init__(self, gammas: tuple[float, ...] = (0.001, 0.01)) -> None:
        self.gammas = tuple(float(gamma) for gamma in gammas)
        self.sse = 0.0
        self.target_sq = 0.0
        self.target_sum = 0.0
        self.target_sq_sum = 0.0
        self.pred_sum = 0.0
        self.pred_sq_sum = 0.0
        self.cross_sum = 0.0
        self.count = 0
        self.gamma_counts = {gamma: {"hits": 0.0, "misses": 0.0, "false_alarms": 0.0} for gamma in self.gammas}

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.detach().float().cpu()
        target = target.detach().float().cpu()
        error = pred - target
        self.sse += float(torch.sum(error**2).item())
        self.target_sq += float(torch.sum(target**2).item())
        self.target_sum += float(torch.sum(target).item())
        self.target_sq_sum += float(torch.sum(target**2).item())
        self.pred_sum += float(torch.sum(pred).item())
        self.pred_sq_sum += float(torch.sum(pred**2).item())
        self.cross_sum += float(torch.sum(pred * target).item())
        self.count += int(target.numel())

        for gamma in self.gammas:
            pred_mask = pred > gamma
            target_mask = target > gamma
            counts = self.gamma_counts[gamma]
            counts["hits"] += float(torch.logical_and(pred_mask, target_mask).sum().item())
            counts["misses"] += float(torch.logical_and(~pred_mask, target_mask).sum().item())
            counts["false_alarms"] += float(torch.logical_and(pred_mask, ~target_mask).sum().item())

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            return {
                "relative_rmse": math.nan,
                "nse": math.nan,
                "pearson_r": math.nan,
                "csi_gamma_0_001": math.nan,
                "csi_gamma_0_01": math.nan,
            }

        eps = 1e-12
        target_mean = self.target_sum / self.count
        pred_mean = self.pred_sum / self.count
        target_var_sum = self.target_sq_sum - self.count * target_mean * target_mean
        pred_var_sum = self.pred_sq_sum - self.count * pred_mean * pred_mean
        covariance_sum = self.cross_sum - self.count * pred_mean * target_mean

        result = {
            "relative_rmse": math.sqrt(self.sse / (self.target_sq + eps)),
            "nse": 1.0 - self.sse / (target_var_sum + eps),
            "pearson_r": covariance_sum / math.sqrt(max(pred_var_sum * target_var_sum, 0.0) + eps),
        }
        for gamma in self.gammas:
            counts = self.gamma_counts[gamma]
            suffix = str(gamma).replace(".", "_")
            denominator = counts["hits"] + counts["misses"] + counts["false_alarms"] + eps
            result[f"csi_gamma_{suffix}"] = counts["hits"] / denominator
        return result
