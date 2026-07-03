# FNO+ Official-v0 Shallow-Water Diagnostic

## Objective

Diagnose why the official-v0 3D FNO+ checkpoint_best has strong global metrics but weak CSI at gamma=0.001 compared with the target reference. This is diagnostic-only: no training, model-code changes, or dataset-code changes were performed.

## Checkpoint and Run Used

- Run: `/home/wissam/utem-workspace/experiments/FloodCastBench/27-06-2026_16-28-24_fcb_fno_plus_official_v0_highfid_60m`
- Checkpoint: `/home/wissam/utem-workspace/checkpoints/FloodCastBench/27-06-2026_16-28-24_fcb_fno_plus_official_v0_highfid_60m/checkpoint_best.pth`
- Checkpoint epoch: `99`
- Output diagnostics folder: `/home/wissam/utem-workspace/experiments/FloodCastBench/27-06-2026_16-28-24_fcb_fno_plus_official_v0_highfid_60m/shallow_water_diagnostics_checkpoint_best`

## Global Test Metrics Recap

| Metric | Value |
|---|---:|
| `current_relative_rmse` | 0.008135040581762574 |
| `classical_rmse` | 0.0057238624709056735 |
| `nse` | 0.9999095087510275 |
| `pearson_r` | 0.9999564540343854 |
| `csi_gamma_0_001` | 0.7610737705248405 |
| `csi_gamma_0_01` | 0.990944324520436 |

## Threshold Sweep

| Threshold | CSI | Precision | Recall | FPR | FNR | Pred area | True area | Pred/true ratio | TP | FP | FN | TN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0005 | 0.774882 | 0.792407 | 0.972250 | 0.62560816 | 0.027750 | 66635336 | 54309426 | 1.226957 | 52802320 | 13833016 | 1507106 | 8278294 |
| 0.001 | 0.761074 | 0.765786 | 0.991980 | 0.55155965 | 0.008020 | 63863995 | 49301533 | 1.295375 | 48906137 | 14957858 | 395396 | 12161345 |
| 0.002 | 0.811968 | 0.813877 | 0.997120 | 0.37293275 | 0.002880 | 58101019 | 47423647 | 1.225149 | 47287055 | 10813964 | 136592 | 18183125 |
| 0.005 | 0.962162 | 0.963589 | 0.998464 | 0.05711355 | 0.001536 | 47685585 | 46019984 | 1.036193 | 45949290 | 1736295 | 70694 | 28664457 |
| 0.01 | 0.990944 | 0.992685 | 0.998234 | 0.01046151 | 0.001766 | 45121128 | 44870318 | 1.005590 | 44791063 | 330065 | 79255 | 31220353 |
| 0.02 | 0.994512 | 0.997599 | 0.996899 | 0.00314234 | 0.003101 | 43301001 | 43331405 | 0.999298 | 43197023 | 103978 | 134382 | 32985353 |
| 0.05 | 0.993429 | 0.999853 | 0.993574 | 0.00016369 | 0.006426 | 40100181 | 40353579 | 0.993721 | 40094277 | 5904 | 259302 | 36061253 |

## Depth-Bin Error Table

| Target bin | Pixels | Mean target | Mean pred | MAE | RMSE | Bias | Under % | Over % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| target = 0 | 2 | 0 | 0.016288687 | 0.016288687 | 0.016289545 | 0.016288687 | 0.000 | 100.000 |
| 0 < target < 0.001 | 27119201 | 0.00033367735 | 0.0012514112 | 0.0017306481 | 0.0021612441 | 0.00091773384 | 31.748 | 68.252 |
| 0.001 <= target < 0.005 | 3281548 | 0.0021624939 | 0.0036795711 | 0.0020312299 | 0.0024873128 | 0.0015170772 | 21.381 | 78.619 |
| 0.005 <= target < 0.01 | 1149667 | 0.0073312059 | 0.008617042 | 0.0018994611 | 0.0023327879 | 0.0012858361 | 23.763 | 76.237 |
| 0.01 <= target < 0.05 | 4516739 | 0.027251358 | 0.02631087 | 0.0020239 | 0.0026236423 | -0.00094048835 | 64.000 | 36.000 |
| target >= 0.05 | 40353579 | 0.68700818 | 0.68675841 | 0.0040870523 | 0.0075813855 | -0.000249763 | 60.178 | 39.821 |

## Per-Timestep CSI@0.001 Trend

| Forecast step | Rel RMSE | Classical RMSE | CSI@0.001 | Precision@0.001 | Recall@0.001 | Pred area | True area |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.008226 | 0.005787 | 0.879812 | 0.896331 | 0.979483 | 2833882 | 2593303 |
| 3 | 0.008077 | 0.005683 | 0.858044 | 0.870665 | 0.983387 | 2928835 | 2593113 |
| 4 | 0.007982 | 0.005616 | 0.840056 | 0.850573 | 0.985494 | 3004569 | 2593223 |
| 5 | 0.007921 | 0.005573 | 0.825040 | 0.833879 | 0.987315 | 3070589 | 2593398 |
| 6 | 0.007877 | 0.005542 | 0.811189 | 0.818833 | 0.988623 | 3131451 | 2593644 |
| 7 | 0.007847 | 0.005521 | 0.797986 | 0.804500 | 0.989955 | 3191955 | 2593985 |
| 8 | 0.007833 | 0.005511 | 0.786455 | 0.792072 | 0.991063 | 3246203 | 2594413 |
| 9 | 0.007839 | 0.005516 | 0.775679 | 0.780793 | 0.991626 | 3295738 | 2595020 |
| 10 | 0.007861 | 0.005531 | 0.765176 | 0.769622 | 0.992508 | 3346550 | 2595018 |
| 11 | 0.007908 | 0.005564 | 0.755447 | 0.759280 | 0.993362 | 3395064 | 2595029 |
| 12 | 0.007966 | 0.005605 | 0.746964 | 0.750303 | 0.994078 | 3438202 | 2595062 |
| 13 | 0.008034 | 0.005653 | 0.738956 | 0.741997 | 0.994485 | 3478126 | 2595073 |
| 14 | 0.008124 | 0.005716 | 0.731739 | 0.734393 | 0.995086 | 3516324 | 2595116 |
| 15 | 0.008216 | 0.005781 | 0.724245 | 0.726563 | 0.995615 | 3556168 | 2595158 |
| 16 | 0.008316 | 0.005852 | 0.717336 | 0.719326 | 0.996157 | 3594314 | 2595458 |
| 17 | 0.008424 | 0.005928 | 0.710053 | 0.711777 | 0.996599 | 3634487 | 2595773 |
| 18 | 0.008531 | 0.006003 | 0.702107 | 0.703581 | 0.997025 | 3678984 | 2596188 |
| 19 | 0.008661 | 0.006094 | 0.693363 | 0.694560 | 0.997520 | 3729136 | 2596549 |
| 20 | 0.008822 | 0.006208 | 0.682539 | 0.683379 | 0.998203 | 3793418 | 2597010 |

## Worst-Case Analysis

### Worst samples by CSI@0.001

- sample 8: CSI@0.001=0.758226, precision=0.763975, recall=0.990173, FN=34711, FP=1080504
- sample 12: CSI@0.001=0.758848, precision=0.763226, recall=0.992496, FN=26278, FP=1078281
- sample 11: CSI@0.001=0.759059, precision=0.763760, recall=0.991957, FN=28212, FP=1076292

### Worst timesteps by CSI@0.001

- forecast step t=20: CSI@0.001=0.682539, precision=0.683379, recall=0.998203
- forecast step t=19: CSI@0.001=0.693363, precision=0.694560, recall=0.997520
- forecast step t=18: CSI@0.001=0.702107, precision=0.703581, recall=0.997025

## Interpretation

- Main observed CSI@0.001 failure mode: **false positives / overprediction dominate at gamma=0.001**.
- At gamma=0.001, precision=0.765786, recall=0.991980, and predicted/true flooded-area ratio=1.295375.
- CSI@0.01 is much stronger than CSI@0.001, so the model captures deeper/larger flooded regions better than shallow-water extent.
- The shallowest bins and dry/near-dry zones are numerically sensitive: small absolute errors can flip binary masks at gamma=0.001.
- Per-timestep rows identify whether degradation is uniform or concentrated at particular forecast steps; see `per_timestep_metrics.csv` for exact values.
- The final-frame figures highlight boundary/front errors through TP/FP/FN maps at gamma=0.001 and gamma=0.01.

## Suggested Next Steps

- Inspect the saved worst-case figures before changing any training code.
- Verify whether rainfall alignment/order is chronologically correct, because shallow-water extent is sensitive to forcing timing.
- Consider a report-safe metric note explaining that CSI@0.001 is highly sensitive to shallow flood-front/boundary pixels.
- If code changes are later allowed, test post-processing or loss weighting only as a new controlled experiment, not as a replacement for this run.
