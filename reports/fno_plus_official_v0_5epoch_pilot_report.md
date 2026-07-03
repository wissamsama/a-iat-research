# FNO+ Official v0 5-Epoch Pilot Report

## 1. Executive Summary

The 5-epoch official FNO+ v0 pilot completed cleanly. It wrote metrics, summary, `checkpoint_best.pth`, and `checkpoint_last.pth`.

Learning is clearly detected in the engineering sense:

- train loss decreases from `0.082768` to `0.029409`;
- validation current relative RMSE improves from `0.910901` to best `0.480960` at epoch 4;
- validation NSE improves from `-0.137693` to best `0.682824` at epoch 4;
- validation Pearson r improves from `0.569638` to `0.861991` by epoch 5;
- CSI@0.01 improves from `0.625152` to best `0.664266` at epoch 4.

However, this is still a very early pilot. Metrics remain far from the internal 100-epoch FNO+ baseline and far from official Table 4. The model is learning, but not ready to be treated as a final reproduction.

Recommendation: **do not launch the final 100-epoch official reproduction yet as the definitive run**. Either run a longer controlled pilot, such as 20 epochs, or first address preprocessing/normalization and RMSE dry-cell ambiguity.

## 2. Run Metadata

| Item | Path / value |
|---|---|
| Run path | `/home/wissam/utem-workspace/experiments/FloodCastBench/27-06-2026_16-14-46_fcb_fno_plus_official_v0_highfid_60m` |
| Checkpoint dir | `/home/wissam/utem-workspace/checkpoints/FloodCastBench/27-06-2026_16-14-46_fcb_fno_plus_official_v0_highfid_60m` |
| Config copy | `/home/wissam/utem-workspace/experiments/FloodCastBench/27-06-2026_16-14-46_fcb_fno_plus_official_v0_highfid_60m/config.yaml` |
| Summary JSON | `/home/wissam/utem-workspace/experiments/FloodCastBench/27-06-2026_16-14-46_fcb_fno_plus_official_v0_highfid_60m/summary.json` |
| Metrics CSV | `/home/wissam/utem-workspace/experiments/FloodCastBench/27-06-2026_16-14-46_fcb_fno_plus_official_v0_highfid_60m/metrics.csv` |
| Prediction stats JSON | `/home/wissam/utem-workspace/experiments/FloodCastBench/27-06-2026_16-14-46_fcb_fno_plus_official_v0_highfid_60m/val_subset_prediction_stats_checkpoint_best.json` |
| checkpoint_best | `/home/wissam/utem-workspace/checkpoints/FloodCastBench/27-06-2026_16-14-46_fcb_fno_plus_official_v0_highfid_60m/checkpoint_best.pth` |
| checkpoint_last | `/home/wissam/utem-workspace/checkpoints/FloodCastBench/27-06-2026_16-14-46_fcb_fno_plus_official_v0_highfid_60m/checkpoint_last.pth` |
| Completed cleanly | yes |
| Completed epochs | 5 |
| Best epoch | 4 |
| Best selection metric | val_current_relative_rmse = 0.480959830833 |

Command run:

```bash
python tools/train_floodcastbench_fno_plus_official.py   --config configs/floodcastbench_fno_plus_official_highfid_60m.yaml   --epochs 5   --num-workers 2
```

## 3. Per-Epoch Metrics Table

| Epoch | Train loss | Val loss | Val current relative RMSE | Val NSE | Val Pearson r | Val CSI@0.001 | Val CSI@0.01 | LR | checkpoint_best update |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.082767623874 | 0.406650715641 | 0.910900641336 | -0.137692648954 | 0.569638244237 | 0.679442063955 | 0.625151987768 | 0.000904508497 | yes |
| 2 | 0.071268429273 | 0.376132809690 | 0.876053927948 | -0.052312232078 | 0.706697230789 | 0.661156668277 | 0.613255206450 | 0.000654508497 | yes |
| 3 | 0.055091338748 | 0.216961373176 | 0.665351956148 | 0.393004023467 | 0.813440138920 | 0.667639391043 | 0.638930772097 | 0.000345491503 | yes |
| 4 | 0.037502453325 | 0.113369739481 | 0.480959830833 | 0.682823831459 | 0.860318772861 | 0.675171179994 | 0.664265677566 | 0.000095491503 | yes |
| 5 | 0.029408665308 | 0.125346158764 | 0.505726531909 | 0.649317242003 | 0.861990785150 | 0.669725469967 | 0.659305992382 | 0.000000000000 | no |

Paper-formula RMSE values were also logged, but they remain extremely large because the literal formula divides by near-zero dry-cell targets. They should not be used for model selection until the official dry-cell convention is clarified.

## 4. Learning Dynamics Analysis

Confirmed learning signals:

- Train loss decreases monotonically over all 5 epochs.
- Validation current relative RMSE decreases strongly from epoch 1 to epoch 4.
- Validation loss decreases from epoch 1 to epoch 4, then rises slightly at epoch 5.
- Validation NSE improves from negative to positive.
- Validation Pearson r improves steadily.
- CSI@0.01 improves through epoch 4 and only slightly drops at epoch 5.

Caution:

- Epoch 5 is slightly worse than epoch 4 on validation relative RMSE, val loss, NSE, CSI@0.001, and CSI@0.01.
- This is not divergence; it is normal early-training fluctuation.
- Best checkpoint selection correctly keeps epoch 4.

No NaN or non-finite metric was detected in the CSV.

## 5. Prediction Statistics

Prediction statistics were computed on 3 validation samples using `checkpoint_best.pth`.

| Quantity | Min | Max | Mean | Std |
|---|---:|---:|---:|---:|
| prediction | -0.097396 | 5.663859 | 0.229004 | 0.462919 |
| target | 0.000027 | 14.938837 | 0.364019 | 0.595664 |
| error | -12.555689 | 0.919433 | -0.135015 | 0.309098 |

Rates:

| Rate | Value |
|---|---:|
| negative prediction rate | 0.121501 |
| near-zero prediction rate `<1e-6` | 0.000020 |
| target wet rate @0.001 | 0.662206 |
| prediction wet rate @0.001 | 0.868016 |
| target wet rate @0.01 | 0.595784 |
| prediction wet rate @0.01 | 0.762162 |

Interpretation:

- The pilot checkpoint overpredicts wet pixels: prediction wet rate is higher than target wet rate at both thresholds.
- There are still negative predictions: about `0.1215`.
- Prediction mean is below target mean on this subset, but wet-mask overprediction indicates spatial/threshold mismatch rather than a simple mean bias.

## 6. CSI@0.001 and CSI@0.01 Behavior

Validation CSI behavior:

| Metric | Epoch 1 | Best | Final epoch 5 |
|---|---:|---:|---:|
| CSI@0.001 | 0.679442 | 0.675171 | 0.669725 |
| CSI@0.01 | 0.625152 | 0.664266 | 0.659306 |

CSI@0.001 improves only modestly and is noisy. CSI@0.01 improves more visibly. This matches the broader project pattern: deeper-water threshold metrics are easier than shallow-water boundary metrics.

## 7. Comparison With Internal FNO+ v1

Internal FNO+ v1 best test metrics:

| Metric | Internal FNO+ v1 |
|---|---:|
| relative RMSE | 0.012358298338 |
| NSE | 0.999791164424 |
| Pearson r | 0.999898656296 |
| CSI@0.001 | 0.724176463561 |
| CSI@0.01 | 0.985721042228 |

Official-v0 5-epoch pilot best validation metrics:

| Metric | Official-v0 pilot best validation |
|---|---:|
| current relative RMSE | 0.480959830833 |
| NSE | 0.682823831459 |
| Pearson r | 0.860318772861 |
| CSI@0.001 | 0.675171179994 |
| CSI@0.01 | 0.664265677566 |

This comparison is only directional because it compares a 5-epoch validation pilot against a completed 100-epoch internal test run. The official-v0 pilot is far behind the internal completed baseline, as expected.

## 8. Distance From Official FNO+ Table 4

Official FloodCastBench FNO+ Table 4:

| Metric | Official FNO+ |
|---|---:|
| RMSE / relative RMSE | 0.003941 |
| NSE | 0.999979 |
| Pearson r | 0.999990 |
| CSI@0.001 | 0.939638 |
| CSI@0.01 | 0.984588 |

Official-v0 pilot best validation is far from Table 4. This is expected after 5 epochs and should not be overinterpreted.

## 9. Issues Detected

Engineering issues:

- No blocking engineering failure detected.
- Training, validation, checkpointing, summary writing, and metrics all work.

Scientific/modeling issues:

- Literal paper-formula RMSE remains unusable without dry-cell convention.
- Model still produces negative water-depth predictions.
- Wet-mask prediction rates are too high in the validation subset.
- Raw DEM scale is much larger than raw water depth; normalization may be important.
- No visualization system is currently supported in the official-v0 evaluation script; no qualitative PNGs were generated.

## 10. Recommendation

Recommendation: **fix or further pilot before final 100 epochs**.

The pipeline is technically ready to train, but the 5-epoch pilot is not enough to justify launching a final official reproduction run as-is. The best next move is either:

1. run a 20-epoch pilot to see whether validation RMSE/CSI continues improving, or
2. first add/try normalization and possibly non-negative output handling, then repeat a pilot.

If you still want to proceed technically, the full command is:

```bash
python tools/train_floodcastbench_fno_plus_official.py   --config configs/floodcastbench_fno_plus_official_highfid_60m.yaml   --epochs 100   --num-workers 2
```
