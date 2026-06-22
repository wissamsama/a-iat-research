from __future__ import annotations

import torch


BASELINE_RULES = {
    "persistence": "prediction = last input frame",
    "linear_delta": "prediction = clamp(last + horizon * (last - previous), min=0)",
}


def available_baselines() -> list[str]:
    return sorted(BASELINE_RULES)


def predict_floodcastbench_baseline(x: torch.Tensor, horizon: int, baseline: str) -> torch.Tensor:
    baseline = baseline.lower()
    if baseline == "persistence":
        return x[-1]
    if baseline == "linear_delta":
        if x.shape[0] < 2:
            raise ValueError("linear_delta baseline requires input_window >= 2.")
        last = x[-1]
        previous = x[-2]
        delta = last - previous
        return torch.clamp(last + int(horizon) * delta, min=0.0)
    available = ", ".join(available_baselines())
    raise ValueError(f"Unsupported FloodCastBench baseline '{baseline}'. Available baselines: {available}")


def prediction_rule_for_baseline(baseline: str) -> str:
    return BASELINE_RULES[baseline.lower()]
