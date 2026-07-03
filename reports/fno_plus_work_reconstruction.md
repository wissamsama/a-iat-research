# FNO+ Work Reconstruction

## 1. Executive Summary

This report reconstructs the current FNO+ baseline work from repository files and external workspace artifacts. The current FNO+ work is functional and valuable, but it should be described cautiously: it is a **simplified internal FNO+ approximation**, not an official FloodCastBench reproduction.

Confirmed facts:

- The FNO+ scaffold was added in commit `f2c5235 Add FNO+ reproduction scaffold for FloodCastBench`.
- The implementation targets FloodCastBench high-fidelity Australia 60 m.
- The dataset produces 20-frame non-overlapping samples: input `t=1`, targets `t=2..20`.
- The real Australia 60 m dataset gives 116 train, 14 validation, and 14 test samples, consistent with the intended split counts.
- The model is a 2D spatial FNO (`FNOPlus2d`) with temporal information encoded as input channels, not a true 3D/space-time FNO.
- A full 100-epoch run exists: `27-06-2026_14-00-18_fcb_fno_plus_highfid_60m`.
- Best validation checkpoint was selected at epoch 95 using validation relative RMSE.
- Test metrics were computed after model selection using `checkpoint_best.pth`.
- The test result is close to official FNO+ on CSI@0.01 but not on relative RMSE or CSI@0.001.

Scientific interpretation: the current FNO+ result is a serious internal baseline and a useful diagnostic scaffold. It is not yet report-safe as an official reproduction because architecture, normalization, rainfall alignment, metric convention, and official training details remain uncertain.

## 2. Files and Implementation Status

| File | Status | Role |
|---|---:|---|
| `models/fno_plus.py` | exists | Defines `SpectralConv2d` and `FNOPlus2d`. |
| `datasets/floodcastbench_fno_dataset.py` | exists | Builds FloodCastBench FNO/FNO+ 20-step samples. |
| `evaluation/floodcastbench_official_metrics.py` | exists | Implements relative RMSE, NSE, Pearson r, CSI, and a global accumulator. |
| `tools/train_floodcastbench_fno_plus.py` | exists | Trains FNO+ scaffold, writes metrics, summaries, and checkpoints. |
| `configs/floodcastbench_fno_plus_highfid_60m.yaml` | exists | Main high-fidelity Australia 60 m FNO+ config. |
| `tests/test_fno_plus_smoke.py` | exists | Fast smoke tests for model shape, metrics, and dataset indexing. |

The implementation uses coordinates, initial water depth, DEM, rainfall, and time channels. There is no separate completed FNO baseline without DEM/rainfall found in the repository or workspace. The dataset class has switches for `include_dem`, `include_rainfall`, and `include_time`, but no separate non-plus FNO config/run was found.

## 3. Dataset and Split Verification

Dataset root inspected:

```text
/home/wissam/utem-workspace/data/FloodCastBench
```

Configured dataset:

```text
fidelity: high
event: australia
resolution: 60m
sample_length: 20
stride: 20
split_counts: train=116, val=14, test=14
```

Real dataset verification from `FloodCastBenchFNODataset`:

| Split | Samples | Start index range | First input timestamp | First target range | Last input timestamp | Last target range |
|---|---:|---:|---:|---|---:|---|
| train | 116 | 0 to 2300 | 0 | 300 to 5700 | 690000 | 690300 to 695700 |
| val | 14 | 2320 to 2580 | 696000 | 696300 to 701700 | 774000 | 774300 to 779700 |
| test | 14 | 2600 to 2860 | 780000 | 780300 to 785700 | 858000 | 858300 to 863700 |

Additional confirmed dataset properties:

- Water-depth frames: 2881 TIFFs.
- Spatial shape: `536 x 536`.
- Input tensor shape: `(43, 536, 536)`.
- Target tensor shape: `(19, 536, 536)`.
- Sample windows are non-overlapping: starts are `0, 20, 40, ...`.
- Input uses frame `t=1`, target uses frames `t=2..20`.
- DEM is static and resized to the water-depth shape.
- Rainfall count for Australia: 481 TIFFs.
- Rainfall is selected deterministically using latest frame at or before water timestamp: `water_timestamp // 1800`.

Input channel breakdown:

| Component | Channels |
|---|---:|
| X coordinate | 1 |
| Y coordinate | 1 |
| Initial water depth `t=1` | 1 |
| DEM | 1 |
| Rainfall `t=1..20` | 20 |
| Time channels for target steps `t=2..20` | 19 |
| Total | 43 |

Normalization status:

- Water depth: raw values, with nodata/nan/inf replaced by 0.
- DEM: raw values, bilinear-resized, with nodata/nan/inf replaced by 0.
- Rainfall: raw values, bilinear-resized, with nodata/nan/inf replaced by 0.

Uncertainty: the official paper may use a specific normalization or preprocessing protocol that is not reproduced here.

## 4. Model Architecture

Implemented model: `FNOPlus2d`.

Confirmed architecture from `models/fno_plus.py` and config:

| Property | Value |
|---|---|
| Model type | 2D spatial FNO |
| Input shape | `[B, 43, H, W]` |
| Output shape | `[B, 19, H, W]` |
| Fourier layers | 4 |
| Fourier modes | 12 |
| Width | 20 |
| Fourier transform | `torch.fft.rfft2`, spatial dimensions only |
| Future prediction | all 19 future frames directly as channels |
| Autoregressive? | no |
| Output activation | identity |
| Padding | none found |

Important interpretation: this is not a true spatiotemporal FNO operating over `(time, x, y)` as a 3D spectral domain. It is a 2D FNO where temporal context and output times are represented as channels. That may be a valid baseline design, but it is an approximation unless confirmed against the official baseline implementation.

## 5. Training Protocol

Main training script:

```text
tools/train_floodcastbench_fno_plus.py
```

Main config:

```text
configs/floodcastbench_fno_plus_highfid_60m.yaml
```

Configured hyperparameters:

| Hyperparameter | Value |
|---|---:|
| Epochs | 100 |
| Batch size | 1 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Betas | `(0.9, 0.999)` |
| Weight decay | 0.0001 |
| Scheduler | CosineAnnealingLR |
| Minimum LR | 0.0 |
| Loss | MSE |
| Seed | 42 |
| Device | auto |
| Validation metric for best checkpoint | lowest `val_relative_rmse` |

Output policy:

- Experiment artifacts are written under `/home/wissam/utem-workspace/experiments/FloodCastBench`.
- Checkpoints are written under `/home/wissam/utem-workspace/checkpoints/FloodCastBench`.
- Logs directory is created under `/home/wissam/utem-workspace/logs/FloodCastBench`.
- Run-local resolved config is saved as `config.yaml` in the experiment directory.
- `checkpoint_best.pth` and `checkpoint_last.pth` are both produced.
- `checkpoint_best.pth` is selected from validation metrics only.

No full training was relaunched during this reconstruction.

## 6. Completed/Incomplete Runs

Detected FNO+ experiment runs:

| Run | Status | Epochs | Best val relative RMSE | Final val relative RMSE | Test metrics | Checkpoints |
|---|---|---:|---:|---:|---|---|
| `27-06-2026_13-23-11_fcb_fno_plus_highfid_60m` | smoke/short run | 1 | 1.019133 | 1.019133 | not found | best + last present |
| `27-06-2026_13-48-07_fcb_fno_plus_highfid_60m` | 5-epoch pilot | 5 | 0.741642 | 0.741642 | not found | best + last present |
| `27-06-2026_14-00-18_fcb_fno_plus_highfid_60m` | completed full run | 100 | 0.012644 | 0.012674 | found | best + last present |

Full run paths:

```text
Experiment: /home/wissam/utem-workspace/experiments/FloodCastBench/27-06-2026_14-00-18_fcb_fno_plus_highfid_60m
Checkpoint: /home/wissam/utem-workspace/checkpoints/FloodCastBench/27-06-2026_14-00-18_fcb_fno_plus_highfid_60m/checkpoint_best.pth
Diagnostics: /home/wissam/utem-workspace/experiments/FloodCastBench/27-06-2026_14-00-18_fcb_fno_plus_highfid_60m/diagnostics_best_checkpoint
```

For the completed run:

- `summary.json` exists.
- `metrics.csv` contains 100 rows.
- `test_metrics_best.json` exists.
- Diagnostic JSON, per-timestep CSV, and three PNG diagnostic figures exist.
- Best checkpoint epoch: 95.

## 7. Metrics and Result Table

Metric implementation:

| Metric | Formula / implementation |
|---|---|
| Relative RMSE | `sqrt(sum((pred-target)^2) / (sum(target^2) + eps))` |
| NSE | `1 - SSE / sum((target - mean(target))^2)` |
| Pearson r | global centered covariance divided by product of global standard deviations |
| CSI | `hits / (hits + misses + false_alarms)` using `pred > gamma`, `target > gamma` |
| Aggregation | global accumulation over all pixels, timesteps, samples, and batches |

No PathIoU is implemented in the FNO+ official metrics file. PathIoU appears in older FloodCastBench model evaluation outputs, but not in this FNO+ training/evaluation scaffold.

Official high-fidelity 60 m Table 4 reference:

| Method | RMSE | NSE | r | CSI@0.001 | CSI@0.01 |
|---|---:|---:|---:|---:|---:|
| FNO | 0.004258 | 0.999975 | 0.999987 | 0.895553 | 0.980748 |
| FNO+ | 0.003941 | 0.999979 | 0.999990 | 0.939638 | 0.984588 |

Completed local FNO+ run, best validation checkpoint:

| Split / variant | RMSE / relative RMSE | NSE | r | CSI@0.001 | CSI@0.005 | CSI@0.01 | CSI@0.05 |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation best epoch 95 | 0.012644 | 0.999781 | 0.999894 | 0.748698 | not found | 0.988006 | not found |
| test raw | 0.012358 | 0.999791 | 0.999899 | 0.724176 | 0.902015 | 0.985721 | 0.985472 |
| test clamped `pred >= 0` | 0.012327 | 0.999792 | 0.999900 | 0.724176 | 0.902015 | 0.985721 | 0.985472 |

Delta vs official FNO+ target, test raw:

| Metric | Local | Official FNO+ | Delta local - official |
|---|---:|---:|---:|
| RMSE / relative RMSE | 0.012358 | 0.003941 | +0.008417 |
| NSE | 0.999791 | 0.999979 | -0.000188 |
| r | 0.999899 | 0.999990 | -0.000091 |
| CSI@0.001 | 0.724176 | 0.939638 | -0.215462 |
| CSI@0.01 | 0.985721 | 0.984588 | +0.001133 |

Diagnostics for the completed test run:

| Diagnostic | Value |
|---|---:|
| Prediction min | -0.009814 |
| Prediction max | 14.219116 |
| Prediction mean | 0.365124 |
| Target min | 0.000000 |
| Target max | 14.902684 |
| Target mean | 0.364703 |
| Negative prediction ratio | 0.074474 |
| Target depth `< 0.001` ratio | 0.354867 |
| Target depth `0.001..0.01` ratio | 0.057984 |
| MAE where target `< 0.01` | 0.002589 |
| MAE where target `>= 0.01` | 0.007036 |

Per-timestep trend from diagnostics:

- Timesteps evaluated: 19 (`t=2..20`).
- Raw relative RMSE starts at 0.015073 for `t=2` and ends at 0.014310 for `t=20`.
- Raw CSI@0.001 ranges from about 0.701003 to 0.742879.
- Raw CSI@0.01 ranges from about 0.981997 to 0.987500.
- Clamping improves relative RMSE only slightly and does not change CSI values, so negative predictions do not explain the main CSI@0.001 gap.

## 8. Interpretation of Results

The completed local run is plausible as an internal baseline:

- It trains stably over 100 epochs.
- Validation and test metrics are close to each other.
- `checkpoint_best` was selected by validation relative RMSE.
- Test evaluation was done after checkpoint selection.
- CSI@0.01 is slightly above the official FNO+ table value, which suggests the model captures larger/deeper flooded regions reasonably well.

However, the result is not close to the official FNO+ baseline on two critical axes:

- Relative RMSE is about 0.01236 versus the official FNO+ value 0.003941.
- CSI@0.001 is about 0.724 versus the official FNO+ value 0.939638.

The largest scientific warning is the shallow-water threshold. About 35.49% of target pixels are below 0.001, and another 5.80% are between 0.001 and 0.01. Small absolute errors around the shallow-flood boundary can heavily affect CSI@0.001. Since clamping predictions barely changes the metrics, the gap is not mainly caused by negative water-depth predictions.

Best current label: **simplified approximation / internal baseline**.

It should not be called an official reproduction yet.

## 9. Comparability With Official FloodCastBench FNO+

The current work partially matches the official setup:

| Item | Current scaffold | Official target alignment |
|---|---|---|
| Dataset | FloodCastBench high-fidelity Australia 60 m | matches target setting |
| Split counts | 116 / 14 / 14 | matches stated target counts |
| Input initial water depth | yes | matches |
| Coordinates | yes | matches |
| DEM | yes | matches FNO+ idea |
| Rainfall | yes | matches FNO+ idea |
| Output | `t=2..20` water depth | matches requested target |
| Fourier layers | 4 | matches requested setting |
| Modes | 12 | matches requested setting |
| Width | 20 | matches requested setting |
| Batch size | 1 | matches requested setting |
| Epochs | 100 | matches requested setting |
| Optimizer/scheduler | Adam + cosine | matches requested setting |

Main differences and uncertainties:

- The implementation is a 2D spatial FNO with temporal channels; the official implementation may be a different FNO variant.
- No official FloodCastBench FNO/FNO+ training code was found in the local official repository clone; only data-generation/hydraulic code was found.
- Normalization is not confirmed. Current inputs and targets are raw.
- Rainfall is aligned by latest frame at or before water-depth timestamp; this may or may not match the paper preprocessing.
- DEM/rainfall are resized by bilinear interpolation; official preprocessing is not confirmed.
- The paper table labels the metric as RMSE, while this scaffold uses relative RMSE by design. If the official metric convention differs, direct numeric comparison is unsafe.
- The metric accumulator is global over all pixels/timesteps/samples; the official aggregation convention is not independently verified here.
- There is no local FNO non-plus baseline to quantify the added value of DEM/rainfall.

Conclusion: the result is comparable as a **research approximation**, but not yet as a strict official reproduction.

## 10. Relation to Mamba Experiments, If Any

Existing latent/Mamba experiment directories were found in the workspace, and model-evaluation CSVs exist under:

```text
outputs/floodcastbench_model_evaluations/
```

Examples include:

- `floodcastbench_models_test_h20_full_comparison.csv`
- `floodcastbench_models_val_h20_full_comparison.csv`
- `floodcastbench_models_test_h72_22-06-2026_19-56-20_floodcastbench_latent_temporal_h72_full_comparison.csv`
- `floodcastbench_models_val_h72_22-06-2026_19-56-20_floodcastbench_latent_temporal_h72_full_comparison.csv`

These comparison files contain deterministic baselines and latent temporal models for Australia 30 m h20/h72 style tasks. No direct FNO+ versus Mamba comparison file was found.

Current FNO+ and Mamba results are not methodologically comparable yet because they differ in at least these dimensions:

- FNO+ uses high-fidelity Australia 60 m.
- Mamba/latent comparison files use Australia 30 m h20/h72 forecasting outputs.
- FNO+ predicts a 19-step sequence `t=2..20` from a 20-frame sample.
- Mamba comparisons use horizon-based forecast samples and include metrics such as PathIoU that are not in the FNO+ scaffold.
- Checkpoint selection, preprocessing, and metric aggregation are not yet harmonized between FNO+ and Mamba.

Safe claim: FNO+ has not yet been fairly compared against Mamba in the current repository artifacts.

Unsafe claim: Mamba is better or worse than FNO+ based on the current artifacts.

## 11. Missing Evidence

Missing or uncertain items before this becomes report-ready as a reproduction:

- Official FNO/FNO+ source implementation or exact architecture details.
- Confirmation whether official FNO+ uses 2D spatial FNO, 3D space-time FNO, recurrent rollout, or another formulation.
- Official normalization/preprocessing for water depth, DEM, and rainfall.
- Official rainfall temporal alignment rule.
- Official DEM/rainfall resampling rule.
- Confirmation that the paper's RMSE is exactly the scaffold's relative RMSE.
- Confirmation that official metrics are globally accumulated rather than averaged per sample/event.
- A non-plus FNO run without DEM/rainfall under the same scaffold.
- Seed sensitivity or repeated runs.
- Training curves as plots; only CSV curves were found.
- GPU metadata in run summaries; epoch times exist, but hardware is not written in `summary.json`.
- A direct Mamba/FNO+ comparison on the same dataset, resolution, split, horizon, metrics, and checkpoint-selection policy.

## 12. Recommended Next Steps

Priority 1: make the FNO+ result scientifically honest in the report.

- Describe it as a local FNO+ approximation, not official reproduction.
- Report both official Table 4 values and local values with clear caveats.
- Emphasize the CSI@0.001 gap and the relative RMSE gap.

Priority 2: add a matching non-plus FNO baseline.

- Use the same dataset/split/training script.
- Disable DEM and rainfall while preserving coordinates, time, and initial depth.
- Compare FNO vs FNO+ locally before discussing physical-variable benefit.

Priority 3: investigate preprocessing.

- Test normalized water depth / DEM / rainfall inputs.
- Keep raw-target metric evaluation.
- Record normalization stats and resolved config.

Priority 4: verify metric convention.

- Search paper/supplement/code for exact RMSE definition.
- If official RMSE is not relative RMSE, add both absolute RMSE and relative RMSE to diagnostics.

Priority 5: design a valid FNO+ vs Mamba comparison.

- Same event, same resolution, same split, same prediction target, same checkpoint-selection rule, same test-only evaluation.
- Until then, avoid superiority claims.

## 13. Claims Allowed vs Claims Not Allowed

Allowed claims:

- A functional FNO+ scaffold exists for FloodCastBench high-fidelity Australia 60 m.
- The scaffold uses coordinates, initial water depth, DEM, rainfall, and time channels.
- The scaffold trains a 2D FNO-style model with 4 Fourier layers, 12 modes, and width 20.
- The completed 100-epoch run selected epoch 95 as `checkpoint_best` by validation relative RMSE.
- Test metrics were computed for `checkpoint_best.pth`.
- The local test result is `relative_rmse=0.012358`, `NSE=0.999791`, `r=0.999899`, `CSI@0.001=0.724176`, `CSI@0.01=0.985721`.
- The result is close to the official FNO+ CSI@0.01 value but not close on relative RMSE or CSI@0.001.

Claims not allowed yet:

- This is an official FloodCastBench FNO+ reproduction.
- This reproduces Table 4 quantitatively.
- This proves FNO+ is better or worse than Mamba.
- DEM/rainfall improve performance in this codebase. A matched non-plus FNO baseline is missing.
- The metric implementation exactly matches the paper without remaining ambiguity.
- The preprocessing exactly matches the official baseline.

