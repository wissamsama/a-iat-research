# FNO+ Official-v0 Post-Processing Sensitivity Diagnostic

This is a non-destructive diagnostic only. It is not a new official result and does not replace the raw `checkpoint_best` evaluation. Predictions were post-processed only in memory before metric computation.

## Run and Checkpoint

- Run: `/home/wissam/utem-workspace/experiments/FloodCastBench/27-06-2026_16-28-24_fcb_fno_plus_official_v0_highfid_60m`
- Checkpoint: `/home/wissam/utem-workspace/checkpoints/FloodCastBench/27-06-2026_16-28-24_fcb_fno_plus_official_v0_highfid_60m/checkpoint_best.pth`
- Checkpoint epoch: `99`
- Output folder: `/home/wissam/utem-workspace/experiments/FloodCastBench/27-06-2026_16-28-24_fcb_fno_plus_official_v0_highfid_60m/postprocess_sensitivity_checkpoint_best`

## Global Variant Metrics

| Variant | Rel RMSE | Classical RMSE | NSE | Pearson r | CSI@0.001 | CSI@0.01 | Precision@0.001 | Recall@0.001 | FP@0.001 | FN@0.001 | Pred/true area@0.001 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_prediction | 0.008135040478 | 0.005723862436 | 0.999909508752 | 0.999956485568 | 0.761073770525 | 0.990944324520 | 0.765785745160 | 0.991980046543 | 14957858 | 395396 | 1.295375439948 |
| clamp_min_0 | 0.008097201578 | 0.005697238762 | 0.999910348607 | 0.999956772879 | 0.761073770525 | 0.990944324520 | 0.765785745160 | 0.991980046543 | 14957858 | 395396 | 1.295375439948 |
| threshold_to_zero_0.0005 | 0.008097595713 | 0.005697516078 | 0.999910339879 | 0.999956779362 | 0.761073770525 | 0.990944324520 | 0.765785745160 | 0.991980046543 | 14957858 | 395396 | 1.295375439948 |
| threshold_to_zero_0.001 | 0.008097508101 | 0.005697454434 | 0.999910341819 | 0.999956797984 | 0.761073770525 | 0.990944324520 | 0.765785745160 | 0.991980046543 | 14957858 | 395396 | 1.295375439948 |
| threshold_to_zero_0.002 | 0.008087585536 | 0.005690472859 | 0.999910561416 | 0.999957004252 | 0.824417322726 | 0.990944324520 | 0.835321717851 | 0.984412452246 | 9567976 | 768490 | 1.178483009849 |
| threshold_to_zero_0.005 | 0.007979652418 | 0.005614530481 | 0.999912932697 | 0.999959130105 | 0.935508782929 | 0.990944324520 | 0.983059178156 | 0.950837613914 | 807833 | 2423781 | 0.967223169308 |
| threshold_to_zero_0.01 | 0.008011534942 | 0.005636963213 | 0.999912235556 | 0.999959531912 | 0.915192408636 | 0.990944324520 | 0.999991445249 | 0.915199574017 | 386 | 4180791 | 0.915207403388 |

## Answers

- Clamping negative values changes CSI@0.001 from `0.761073770525` to `0.761073770525` and relative RMSE from `0.008135040478` to `0.008097201578`.
- Best CSI@0.001 variant: `threshold_to_zero_0.005` with CSI@0.001 `0.935508782929`.
- RMSE tradeoff for best variant: relative RMSE delta `-0.000155388060`; classical RMSE delta `-0.000109331955`.
- Removing tiny positive predictions is diagnostic post-processing; it should not be used as an official result unless explicitly defined and justified as a separate post-processing experiment.

## Interpretation

- The sensitivity test supports the hypothesis that small positive predictions in near-dry pixels harm CSI@0.001.
- RMSE changed noticeably for the best CSI@0.001 post-processing variant and should be treated as a tradeoff.
- Because the raw model already has very high recall at gamma=0.001, the main sensitivity is precision/false positives rather than missed flooded pixels.

## Files

- `postprocess_global_metrics.csv`: `/home/wissam/utem-workspace/experiments/FloodCastBench/27-06-2026_16-28-24_fcb_fno_plus_official_v0_highfid_60m/postprocess_sensitivity_checkpoint_best/postprocess_global_metrics.csv`
- `postprocess_per_timestep_metrics.csv`: `/home/wissam/utem-workspace/experiments/FloodCastBench/27-06-2026_16-28-24_fcb_fno_plus_official_v0_highfid_60m/postprocess_sensitivity_checkpoint_best/postprocess_per_timestep_metrics.csv`
- `postprocess_summary.json`: `/home/wissam/utem-workspace/experiments/FloodCastBench/27-06-2026_16-28-24_fcb_fno_plus_official_v0_highfid_60m/postprocess_sensitivity_checkpoint_best/postprocess_summary.json`
