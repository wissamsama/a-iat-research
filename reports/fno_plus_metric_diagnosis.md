# FNO+ Metric Diagnosis

## 1. Executive Summary

The best and most complete FNO+ run is:

```text
/home/wissam/utem-workspace/experiments/FloodCastBench/27-06-2026_14-00-18_fcb_fno_plus_highfid_60m
```

It is the only detected FNO+ run with 100 completed epochs, `test_metrics_best.json`, and diagnostic outputs. The selected checkpoint is epoch 95:

```text
/home/wissam/utem-workspace/checkpoints/FloodCastBench/27-06-2026_14-00-18_fcb_fno_plus_highfid_60m/checkpoint_best.pth
```

Core conclusion: the run is useful as an internal FNO+ baseline, but it is **not directly valid as an official FloodCastBench Table 4 reproduction**. The local model is close to official FNO+ on CSI@0.01, but it is substantially worse on RMSE/relative RMSE and CSI@0.001.

The most important numerical result is:

| Split | Variant | RMSE / relative RMSE | NSE | Pearson r | CSI@0.001 | CSI@0.01 |
|---|---|---:|---:|---:|---:|---:|
| test | raw | 0.012358298338 | 0.999791164424 | 0.999898656296 | 0.724176463561 | 0.985721042228 |
| test | clamped `pred >= 0` | 0.012327009254 | 0.999792220557 | 0.999899608953 | 0.724176463561 | 0.985721042228 |

Negative predictions are not the main explanation: clamping barely changes RMSE/NSE/r and does not change CSI.

## 2. Exact Run Table

| Run | Status | Epochs | Best epoch | Train loss at best | Val loss at best | Best val relative RMSE | Final val relative RMSE | Test metrics | checkpoint_best | checkpoint_last |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| `27-06-2026_13-23-11_fcb_fno_plus_highfid_60m` | smoke/short run | 1 | 1 | 0.061683475971 | 0.505840122700 | 1.019133377139 | 1.019133377139 | not found | present | present |
| `27-06-2026_13-48-07_fcb_fno_plus_highfid_60m` | 5-epoch pilot | 5 | 5 | 0.052279939094 | 0.269567902599 | 0.741641996424 | 0.741641996424 | not found | present | present |
| `27-06-2026_14-00-18_fcb_fno_plus_highfid_60m` | complete 100-epoch run | 100 | 95 | 0.000072357976 | 0.000078356093 | 0.012644353209 | 0.012674069375 | found | present | present |

Checkpoint paths for best run:

```text
checkpoint_best: /home/wissam/utem-workspace/checkpoints/FloodCastBench/27-06-2026_14-00-18_fcb_fno_plus_highfid_60m/checkpoint_best.pth
checkpoint_last: /home/wissam/utem-workspace/checkpoints/FloodCastBench/27-06-2026_14-00-18_fcb_fno_plus_highfid_60m/checkpoint_last.pth
```

No PathIoU or flood-front metrics were found in the FNO+ scaffold outputs. Those metrics exist in older horizon-based FloodCastBench model evaluation files, but not for this FNO+ run.

## 3. Best Run Details

Best validation checkpoint, epoch 95:

| Metric | Value |
|---|---:|
| train_loss | 0.000072357976 |
| val_loss | 0.000078356093 |
| train_relative_rmse | 0.027437405668 |
| val_relative_rmse | 0.012644353209 |
| train_nse | 0.999147199998 |
| val_nse | 0.999780782018 |
| train_pearson_r | 0.999575760919 |
| val_pearson_r | 0.999894179203 |
| train_csi_gamma_0_001 | 0.721419670902 |
| val_csi_gamma_0_001 | 0.748698030640 |
| train_csi_gamma_0_01 | 0.878979388408 |
| val_csi_gamma_0_01 | 0.988005654902 |
| learning_rate | 0.000006155830 |

Final epoch 100 validation:

| Metric | Value |
|---|---:|
| train_loss | 0.000070574297 |
| val_loss | 0.000078724824 |
| train_relative_rmse | 0.027097119359 |
| val_relative_rmse | 0.012674069375 |
| train_nse | 0.999168222147 |
| val_nse | 0.999779750416 |
| train_pearson_r | 0.999586193886 |
| val_pearson_r | 0.999892511840 |
| train_csi_gamma_0_001 | 0.723075852538 |
| val_csi_gamma_0_001 | 0.752860082033 |
| train_csi_gamma_0_01 | 0.886104789984 |
| val_csi_gamma_0_01 | 0.987639056998 |

Best-checkpoint test metrics:

| Metric | Raw | Clamped `pred >= 0` |
|---|---:|---:|
| RMSE / relative RMSE | 0.012358298338 | 0.012327009254 |
| NSE | 0.999791164424 | 0.999792220557 |
| Pearson r | 0.999898656296 | 0.999899608953 |
| CSI@0.001 | 0.724176463561 | 0.724176463561 |
| CSI@0.005 | 0.902015476146 | 0.902015476146 |
| CSI@0.01 | 0.985721042228 | 0.985721042228 |
| CSI@0.05 | 0.985472077443 | 0.985472077443 |

Sanity diagnostics:

| Diagnostic | Value |
|---|---:|
| prediction_min | -0.009813625365 |
| prediction_max | 14.219116210938 |
| prediction_mean | 0.365123706411 |
| target_min | 0.000000000000 |
| target_max | 14.902684211731 |
| target_mean | 0.364703365655 |
| negative_prediction_ratio | 0.074474498649 |
| target_depth_lt_0_001_ratio | 0.354867074298 |
| target_depth_0_001_to_0_01_ratio | 0.057984458564 |

## 4. Official Table 4 Comparison

Official high-fidelity 60 m reference:

| Method | RMSE | NSE | r | CSI@0.001 | CSI@0.01 |
|---|---:|---:|---:|---:|---:|
| FNO | 0.004258 | 0.999975 | 0.999987 | 0.895553 | 0.980748 |
| FNO+ | 0.003941 | 0.999979 | 0.999990 | 0.939638 | 0.984588 |

Comparison against official FNO+:

| Metric | Official FNO+ | Our best test | Absolute diff | Relative diff | Validity |
|---|---:|---:|---:|---:|---|
| RMSE / relative RMSE | 0.003941 | 0.012358298338 | +0.008417298338 | +213.583% | partially valid |
| NSE | 0.999979 | 0.999791164424 | -0.000187835576 | -0.019% | partially valid |
| Pearson r | 0.999990 | 0.999898656296 | -0.000091343704 | -0.009% | partially valid |
| CSI@0.001 | 0.939638 | 0.724176463561 | -0.215461536439 | -22.931% | partially valid |
| CSI@0.01 | 0.984588 | 0.985721042228 | +0.001133042228 | +0.115% | partially valid |

Why only partially valid: the dataset setting and headline hyperparameters match the intended Table 4 target, but the implementation is not confirmed to match the official architecture, preprocessing, metric naming, or aggregation protocol.

## 5. Metric-by-Metric Discrepancy Analysis

### RMSE / Relative RMSE

The local metric is named `relative_rmse` and is implemented as:

```text
sqrt(sum((pred - target)^2) / sum(target^2))
```

The official table labels the metric as RMSE. If the official Table 4 number is absolute RMSE, the numeric comparison is not directly valid. If it is a normalized/relative RMSE reported as RMSE, the comparison is more meaningful, but still not fully confirmed.

Observed gap:

```text
ours:     0.012358298338
official: 0.003941
delta:   +0.008417298338
```

This is the strongest warning against claiming reproduction.

### NSE

Local NSE is high: `0.999791164424`, but below official FNO+ `0.999979`. Because NSE is close to 1, the absolute difference looks small but still corresponds to a meaningfully larger residual error.

### Pearson r

Local Pearson r is `0.999898656296`, close but below official `0.999990`. This indicates strong linear agreement but not exact official-level reproduction.

### CSI@0.001

This is the most scientifically revealing discrepancy:

```text
ours:     0.724176463561
official: 0.939638
delta:   -0.215461536439
```

The target-depth diagnostics show that shallow water dominates the threshold behavior:

- target depth `< 0.001`: `0.354867074298`
- target depth `0.001..0.01`: `0.057984458564`

This means small local errors near the flood boundary can heavily damage CSI@0.001.

### CSI@0.01

Local CSI@0.01 is `0.985721042228`, slightly above official FNO+ `0.984588`. This suggests larger/deeper flooded regions are captured well. It does not compensate for the CSI@0.001 gap, because Table 4 reports both thresholds and CSI@0.001 is much more sensitive to shallow flood extent.

## 6. Architecture Discrepancy Analysis

Confirmed local architecture:

- `FNOPlus2d`, not a 3D/space-time FNO.
- Input tensor: `[B, 43, H, W]`.
- Output tensor: `[B, 19, H, W]`.
- Future timesteps are predicted directly as output channels.
- Temporal and physical variables are encoded as channels.
- Fourier transform uses `torch.fft.rfft2`, so spectral mixing is spatial only.

This differs from a possible spatiotemporal FNO where time is an operator dimension. If the official FloodCastBench FNO+ used a true space-time FNO, recurrent rollout, or a different temporal operator, the local architecture is materially different.

Channel construction:

| Component | Channels |
|---|---:|
| X coordinate | 1 |
| Y coordinate | 1 |
| Initial depth t=1 | 1 |
| DEM | 1 |
| Rainfall t=1..20 | 20 |
| Time channels t=2..20 | 19 |
| Total | 43 |

DEM and rainfall are actually used, but as static/dynamic channels concatenated before a 2D FNO, not as separately modeled physical operators.

## 7. Metric Implementation Risks

Confirmed metric implementation uses global accumulation across all batches, samples, output timesteps, and pixels.

Risks:

- Official Table 4 may average metrics per sample/event rather than globally accumulating counts and sums.
- Official RMSE may be absolute RMSE, not relative RMSE.
- Official dry-cell treatment may mask dry cells or handle near-zero values differently.
- CSI uses `pred > gamma` and `target > gamma`; official code may use `>=`, masks, or postprocessing.
- Local output is raw and can be negative; clamping does not explain the gap, but official may enforce non-negative outputs during training.
- PathIoU/flood-front metrics were not produced for FNO+, so no flood-front comparison exists.

## 8. What Can Be Claimed Safely

Safe claims:

- A functional FNO+ scaffold exists for FloodCastBench high-fidelity Australia 60 m.
- The best local run completed 100 epochs.
- Best checkpoint was selected at epoch 95 using validation relative RMSE.
- Test evaluation exists for `checkpoint_best.pth`.
- Local test metrics are exactly:
  - relative RMSE: `0.01235829833769271`
  - NSE: `0.9997911644242803`
  - Pearson r: `0.9998986562960474`
  - CSI@0.001: `0.7241764635613173`
  - CSI@0.01: `0.9857210422280918`
- The current result is close to official FNO+ on CSI@0.01 only.
- The current result is not close to official FNO+ on RMSE/relative RMSE or CSI@0.001.
- The current implementation is best described as an internal FNO+ baseline inspired by FloodCastBench.

## 9. What Must Not Be Claimed

Do not claim:

- This is an official FloodCastBench FNO+ reproduction.
- This reproduces Table 4.
- The local FNO+ is better than official FNO+ because CSI@0.01 is slightly higher.
- DEM/rainfall improve the model in this codebase; no matched non-plus FNO baseline exists.
- This FNO+ result is fairly comparable to current Mamba runs.
- Mamba is better or worse than FNO+ based on current artifacts.

## 10. Minimal Next Actions

1. Add an internal FNO ablation without DEM/rainfall using the same split, architecture, loss, and metrics.
2. Add absolute RMSE alongside relative RMSE to remove metric ambiguity.
3. Run a preprocessing experiment with explicit normalization for water, DEM, and rainfall.
4. Investigate CSI@0.001 specifically with shallow-water masks and threshold-sensitive plots.
5. Verify whether the official FloodCastBench paper used global metric aggregation or per-sample averaging.
6. Only after the above, design a fair Mamba-vs-FNO+ comparison on the same resolution, target, split, and metric protocol.

