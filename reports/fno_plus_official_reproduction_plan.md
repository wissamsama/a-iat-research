# FNO+ Official Reproduction Attempt Plan

## 1. Executive Summary

This plan creates a separate official FNO+ reproduction attempt v0 while preserving the existing internal FNO+ baseline. No plain FNO baseline was implemented, prepared, or trained.

The current internal baseline remains untouched:

```text
models/fno_plus.py
datasets/floodcastbench_fno_dataset.py
tools/train_floodcastbench_fno_plus.py
configs/floodcastbench_fno_plus_highfid_60m.yaml
```

New official-attempt files were created under separate names. The new model is a direct space-time FNO+ attempt using tensors shaped `[B, C, H, W, T]`, Fourier transforms over H/W/T, and one-shot output of `t=2..20`.

Important scientific result before training: the RMSE formula is not settled. The existing metric `relative_rmse` does **not** match the paper formula as written in the prompt.

## 2. Current Internal FNO+ Result Recap

Current best internal run:

```text
/home/wissam/utem-workspace/experiments/FloodCastBench/27-06-2026_14-00-18_fcb_fno_plus_highfid_60m
```

Current best checkpoint:

```text
/home/wissam/utem-workspace/checkpoints/FloodCastBench/27-06-2026_14-00-18_fcb_fno_plus_highfid_60m/checkpoint_best.pth
```

Current internal test metrics:

| Metric | Value |
|---|---:|
| current_relative_rmse | 0.012358298338 |
| classical_rmse | 0.008695371619 |
| paper_formula_rmse_literal_eps | 90.001318594995 |
| NSE | 0.999791164424 |
| Pearson r | 0.999898656296 |
| CSI@0.001 | 0.724176463561 |
| CSI@0.01 | 0.985721042228 |

Official FNO+ high-fidelity 60 m reference:

| Metric | Value |
|---|---:|
| RMSE / relative RMSE | 0.003941 |
| NSE | 0.999979 |
| Pearson r | 0.999990 |
| CSI@0.001 | 0.939638 |
| CSI@0.01 | 0.984588 |

## 3. Difference Between Current Internal FNO+ and Official FNO+

The internal baseline uses a 2D spatial FNO with temporal and physical variables encoded as channels:

```text
internal input: [B, 43, H, W]
internal output: [B, 19, H, W]
```

The official reproduction attempt v0 uses direct space-time tensors:

```text
official-v0 input: [B, 6, H, W, 20]
official-v0 output: [B, 1, H, W, 19]
```

The official-v0 channels are:

| Channel | Meaning |
|---|---|
| 0 | X coordinate |
| 1 | Y coordinate |
| 2 | T coordinate |
| 3 | initial water depth, repeated over T |
| 4 | DEM, repeated over T |
| 5 | rainfall at each of t=1..20 |

This is closer to “direct space-time convolutions” than the internal baseline, but it is still labeled `fno_plus_official_reproduction_attempt_v0` because exact paper code-level details remain ambiguous.

## 4. Official Metric Verification, Especially RMSE

The current code’s `relative_rmse` is:

```text
sqrt(sum((pred - target)^2) / sum(target^2))
```

The prompt gives the paper RMSE formula as:

```text
(1/N) * sum_i(|y_i - p_i|^2 / |y_i|^2)
```

These are not the same:

- current metric has a square root;
- current metric is a ratio of global sums;
- paper formula as written is a mean of per-pixel relative squared errors;
- paper formula as written has no square root;
- paper formula is extremely sensitive to dry cells where `y_i` is zero or near zero.

Evaluation-only recompute on the internal best checkpoint gives:

| Metric | Value |
|---|---:|
| paper_formula_rmse_literal_eps | 90.001318594995 |
| current_relative_rmse | 0.012358298338 |
| classical_rmse | 0.008695371619 |

Conclusion: `rmse_formula_match = false` for the formula as written. There is still uncertainty because the paper may describe the metric imprecisely or may apply masking/dry-cell handling not visible here.

## 5. Proposed Official FNO+ Tensor Layout

Chosen layout:

```text
input  x: [B, 6, H, W, 20]
target y: [B, 1, H, W, 19]
```

Reasons:

- compatible with PyTorch `Conv3d` and `torch.fft.rfftn`;
- keeps H, W, and T as operator dimensions;
- supports one-shot prediction;
- preserves official variables X/Y/T, initial depth, DEM, rainfall;
- avoids collapsing time into channels as in the internal baseline.

## 6. Proposed Official FNO+ Architecture

New model:

```text
models/fno_plus_official.py::FNOPlusOfficial3d
```

Properties:

| Property | Value |
|---|---|
| Architecture label | fno_plus_official_reproduction_attempt_v0 |
| Fourier type | 3D space-time spectral convolution |
| Tensor convention | [B, C, H, W, T] |
| Fourier layers | 4 |
| Lowest modes | 12 |
| Latent width | 20 |
| Strategy | one-shot |
| Output | t=2..20, 19 future frames |
| Output activation | identity |

## 7. New Files Created

| File | Purpose |
|---|---|
| `models/fno_plus_official.py` | 3D space-time FNO+ official attempt model |
| `datasets/floodcastbench_fno_plus_official_dataset.py` | FNO+ dataset returning `[6,H,W,20]` inputs and `[1,H,W,19]` targets |
| `tools/train_floodcastbench_fno_plus_official.py` | dedicated training entrypoint |
| `tools/evaluate_floodcastbench_fno_plus_official.py` | dedicated evaluation entrypoint |
| `tools/recompute_fno_plus_official_metrics.py` | evaluation-only metric recompute for paper formula/current metric/classical RMSE |
| `configs/floodcastbench_fno_plus_official_highfid_60m.yaml` | official-attempt config |
| `tests/test_fno_plus_official_smoke.py` | fast smoke tests |
| `reports/fno_plus_official_metric_recompute_internal_best.json` | recompute result on internal best checkpoint |

## 8. Smoke Test Result

Command:

```bash
/home/wissam/miniforge3/envs/floodcast-mamba/bin/python -m pytest -q tests/test_fno_plus_official_smoke.py
```

Result:

```text
3 passed, 44 warnings
```

Dry-run command:

```bash
/home/wissam/miniforge3/envs/floodcast-mamba/bin/python tools/train_floodcastbench_fno_plus_official.py --config configs/floodcastbench_fno_plus_official_highfid_60m.yaml --dry-run-config --num-workers 0
```

Dry-run result:

```text
train_samples: 116
val_samples: 14
input_shape: [6, 536, 536, 20]
target_shape: [1, 536, 536, 19]
run_dir: DRY_RUN_NO_RUN_DIR
```

## 9. Remaining Ambiguities in the Paper

- Exact RMSE implementation and dry-cell handling.
- Whether the official architecture uses this exact 3D FNO parameterization.
- Whether input or output normalization is applied.
- Whether inverse transforms are used before metric computation.
- Rainfall temporal alignment details.
- DEM/rainfall resampling method.
- Metric aggregation convention.
- Whether predictions are constrained non-negative.

## 10. Minimal Command to Run Official FNO+ Training Later

Do not run this until ready for a full experiment:

```bash
python tools/train_floodcastbench_fno_plus_official.py --config configs/floodcastbench_fno_plus_official_highfid_60m.yaml
```

For a tiny safety run only:

```bash
python tools/train_floodcastbench_fno_plus_official.py   --config configs/floodcastbench_fno_plus_official_highfid_60m.yaml   --epochs 1   --max-train-batches 1   --max-val-batches 1
```

## 11. Minimal Command to Evaluate Official FNO+ Later

```bash
python tools/evaluate_floodcastbench_fno_plus_official.py --run-dir <RUN_DIR> --checkpoint <CHECKPOINT_DIR>/checkpoint_best.pth --split test
```

## 12. Risks and Expected Failure Points

- Full-resolution 3D FNO is more memory-intensive than the internal 2D baseline.
- The literal paper formula may be unusable without a dry-cell mask.
- If official preprocessing used normalization, raw inputs may underperform again.
- If official metric aggregation differs, numbers may remain inconsistent.
- CSI@0.001 will remain highly sensitive to shallow water and threshold handling.

## 13. Claims Allowed vs Claims Not Allowed

Allowed:

- A separate official FNO+ reproduction attempt v0 has been implemented.
- The old internal FNO+ baseline remains preserved.
- The new model is closer to direct space-time FNO than the internal 2D baseline.
- Smoke tests pass.
- The RMSE formula remains unresolved and does not match current relative RMSE as written.

Not allowed:

- Do not claim official reproduction yet.
- Do not claim Table 4 is reproduced.
- Do not claim the literal paper RMSE formula is definitely what the authors computed in code.
- Do not compare official-v0 to Mamba until it has a complete trained/evaluated run.
- Do not claim FNO-only ablation exists; it has intentionally not been implemented.

