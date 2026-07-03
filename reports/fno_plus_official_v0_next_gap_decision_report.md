# FNO+ Official-v0 Next Gap Decision Report

Generated: 2026-06-28T15:07:46

## 1. Executive Summary

This report audits the current FloodCastBench FNO+ official-v0 reproduction attempt without retraining or modifying model/dataset/evaluation code.

Confirmed facts:

- The selected run is `/home/wissam/utem-workspace/experiments/FloodCastBench/27-06-2026_16-28-24_fcb_fno_plus_official_v0_highfid_60m`.
- The selected checkpoint is `/home/wissam/utem-workspace/checkpoints/FloodCastBench/27-06-2026_16-28-24_fcb_fno_plus_official_v0_highfid_60m/checkpoint_best.pth`.
- `checkpoint_best.pth` stores epoch `99`.
- The model is `FNOPlusOfficial3d`, a one-shot 3D space-time FNO over `[H, W, T]`, not the earlier internal 2D channel-stacked FNO baseline.
- The model has `11,061,901` trainable parameters.
- The output layer is identity: no ReLU, Softplus, clamp, wet/dry threshold, or post-processing is applied during training/evaluation.
- The biggest raw discrepancy is CSI@0.001, not CSI@0.01. Raw CSI@0.001 is `0.761073770525` while the official Table 4 FNO+ reference is `0.939638000000`.

Main decision:

The single best next controlled experiment is **not** another blind 100-epoch rerun. The best next training experiment is a separate **official-v1 normalized-input/output run**, with train-set normalization for physical channels and inverse-transform before metric computation. In parallel, the exact official evaluation convention should be clarified, because a purely diagnostic wet/dry threshold already moves CSI@0.001 close to the official number without retraining.

## 2. Output Activation / Physical Constraint Audit

Current implementation evidence:

- `models/fno_plus_official.py` ends with `self.proj2 = nn.Conv3d(..., 1, kernel_size=1)`.
- `forward()` returns `x[..., 1 : self.output_steps + 1]` directly.
- `tools/train_floodcastbench_fno_plus_official.py` computes `pred = model(x)` and `loss = nn.MSELoss()(pred, target)`.
- No output activation or clamp is present in the official-v0 path.

Interpretation:

- Negative and tiny positive water-depth predictions are mathematically allowed.
- This is important because CSI@0.001 is extremely sensitive to tiny positive predictions in near-dry cells.
- Existing post-processing diagnostics are therefore diagnostic only, not official raw model outputs.

## 3. FNO Architecture Fidelity Audit

|property|current official-v0 implementation|
|---|---|
|class|`FNOPlusOfficial3d`|
|input shape|`[B, 6, H, W, 20]`|
|output shape|`[B, 1, H, W, 19]`|
|operator dimensions|3D FFT over `H, W, T`|
|Fourier layers|4|
|width|20|
|configured modes|12|
|effective temporal rFFT modes for T=20|11, because `T//2 + 1 = 11`|
|spatial modes|12 in positive/negative H/W quadrants|
|padding|none|
|time handling|one-shot direct output t=2..20 from a 20-step latent field|
|autoregressive|no|
|output activation|identity|

The implementation is substantially more faithful than the earlier internal 2D FNO+ baseline because it treats time as an operator dimension. Remaining architecture uncertainty: the paper does not specify whether their implementation pads spatial/time dimensions, uses a particular FNO library convention, uses exact complex-weight quadrant handling, or applies an output physical constraint.

## 4. Dataset / Normalization Audit

Dataset facts from code:

- Event: Australia.
- Fidelity/resolution: high-fidelity 60m.
- Sample length: 20 frames.
- Stride: 20, so windows are non-overlapping.
- Splits: 116 train, 14 validation, 14 test.
- Input channels are exactly: X, Y, T, initial water depth, DEM, rainfall.
- Target is water depth t=2..20.
- X/Y/T are normalized to `[0, 1]` by construction.
- Initial water depth, DEM, rainfall, and target water depth are raw physical values.
- Rainfall index is `water_timestamp // 1800`, repeated for six 300-second water-depth frames. The local official data-generation clone supports this timing relationship.

Approximate deterministic train-set quantiles below are sampled from every train sample and should be read as distribution diagnostics, not exact persisted normalization constants.

|channel|sampled values|q0|q1|q5|q50|q95|q99|q100|sample mean|sample std|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|X|475136|0|0.00934579502791|0.0485981330276|0.499065428972|0.951401889324|0.990654230118|1|0.49969242851|0.289211872365|
|Y|475136|0|0.00934579502791|0.0485981330276|0.499065428972|0.949532687664|0.990654230118|1|0.499588383542|0.289046901824|
|T|475136|0|0|0|0.421052634716|0.947368443012|0.947368443012|0.947368443012|0.473375822919|0.302309456815|
|initial_depth|475136|3.05480098177e-05|9.69732107478e-05|0.000192297215108|0.00356572982855|0.536737620831|1.33174429536|9.72720432281|0.107178031183|0.304020796317|
|DEM|475136|0|0|0|6.19166827202|136.339324951|171.18170166|197.183303833|21.9009243413|39.4739808807|
|rainfall|475136|0|0|0|0.299999982119|8.25|12.8699998856|26.3299999237|1.87325992657|3.09261861387|
|target|475136|3.22744817822e-05|9.68113505223e-05|0.000187860132428|0.00324178126175|0.520801529288|1.28293463588|9.73070716858|0.101490146359|0.282881726162|

Scale concern:

- DEM is much larger in scale than water depth and rainfall.
- Targets are raw, and the training loss is raw MSE.
- This could make optimization and reported relative/RMSE behavior differ from an official implementation that normalizes physical channels internally.

## 5. Metric Convention Audit

Raw global checkpoint_best test metrics:

|variant|pixels|rel_rmse|classical_rmse|nse|pearson_r|csi001|csi01|
|---|---|---|---|---|---|---|---|
|raw_global|76420736|0.008135040371|0.005723862423|0.999909508755|0.999956454124|0.761073770525|0.990944324520|

Requested metric-convention variants:

|variant|pixels|rel_rmse|classical_rmse|nse|pearson_r|csi001|csi01|
|---|---|---|---|---|---|---|---|
|raw_global|76420736|0.008135040371|0.005723862423|0.999909508755|0.999956454124|0.761073770525|0.990944324520|
|clamp_min_0_global|76420736|0.008097201528|0.005697238790|0.999910348609|0.999956759256|0.761073770525|0.990944324520|
|threshold_to_zero_0_005_global|76420736|0.007979652301|0.005614530460|0.999912932701|0.999959125438|0.935508782929|0.990944324520|
|target_mask_y_gt_0|76420734|0.008135039509|0.005723861892|0.999909508774|0.999956454133|0.761073794212|0.990944368367|
|target_mask_y_gt_0_001|49301533|0.007926578824|0.006943694890|0.999892378651|0.999949960100|0.991980046543|0.990952787010|
|target_mask_y_gt_0_005|46019984|0.007892658630|0.007156242280|0.999887629254|0.999948511836|0.999964515416|0.991132440741|
|target_mask_y_gt_0_01|44870318|0.007882182272|0.007237715020|0.999885628569|0.999947893602|0.999999977714|0.998233687579|
|final_timestep_t20_global|4022144|0.008821924590|0.006207864911|0.999893587592|0.999954106915|0.682539047299|0.983995614209|
|mean_over_timesteps|4022144.0|0.008129830081|0.005720206616|0.999909509562|0.999958037669|0.765378191011|0.990956997513|
|mean_over_samples|5458624.0|0.008134939239|0.005723799650|0.999909509486|0.999956464294|0.761073874245|0.990944428735|
|valid_domain_mask_no_explicit_mask_found_same_as_global|76420736|0.008135040371|0.005723862423|0.999909508755|0.999956454124|0.761073770525|0.990944324520|

Additional paper-formula / epsilon-stabilized relative-error variants:

|variant|value|
|---|---:|
|mean_squared_relative_error_eps_1e-12|44.4652326896|
|mean_squared_relative_error_eps_1e-08|24.4315190871|
|mean_squared_relative_error_eps_1e-06|1.51960571354|
|mean_squared_relative_error_eps_0_0001|0.0204617100233|
|paper_formula_sqrt_sse_over_y2_y_gt_0|0.00813503950888|
|paper_formula_sqrt_sse_over_y2_y_gt_0_001|0.00792657882381|
|paper_formula_sqrt_sse_over_y2_y_gt_0_005|0.00789265863034|
|paper_formula_sqrt_sse_over_y2_y_gt_0_01|0.00788218227243|

Important interpretation:

- Classical global RMSE is `0.005723862423`.
- Current global relative RMSE is `0.008135040371`.
- The mean per-pixel squared relative error variants are huge and unstable when near-zero targets are included, so they are not plausible as the Table 4 scalar RMSE unless masking is used.
- Masking to deeper target cells can move relative error closer to the official RMSE scale, but this does not explain CSI@0.001 by itself.

## 6. Invalid-Domain / Outside-Domain Mask Audit

No explicit valid-domain mask is used by the current dataset or metrics path.

- `FloodCastBenchFNOPlusOfficialDataset` reads rasters into dense tensors.
- Metrics operate over all pixels in the tensor.
- No nodata mask, flood-domain mask, catchment mask, or outside-domain exclusion is applied.
- The `valid_domain_mask_no_explicit_mask_found_same_as_global` metric row is intentionally identical to raw global.

This remains a meaningful uncertainty: if the paper excludes outside-domain pixels or uses a wettable-domain mask, CSI and RMSE may not be directly comparable.

## 7. Local Official-Code Evidence Audit

- Local clone exists at /tmp/FloodCastBench_official_audit.
- Searches found data-generation code but no benchmark FNO/FNO+ training script in the local clone.
- Data_Generation_Code/main.py uses rainfall indexing consistent with current_time / 1800, supporting the repository rainfall alignment.
- Data_Generation_Code/main.py applies thresholding/clamping in hydraulic simulation state generation; this is not direct evidence of FNO+ prediction post-processing.

Conclusion from local official-code evidence: rainfall temporal alignment is no longer a leading suspect. There is no local benchmark FNO/FNO+ code proving the official model's normalization, output activation, mask, or exact metric aggregation convention.

## 8. Official Table 4 Comparison

|metric|official FNO+|official-v0 raw|absolute gap raw|threshold_to_zero_0.005 diagnostic|absolute gap diagnostic|
|---|---:|---:|---:|---:|---:|
|RMSE/classical_rmse|0.003941000000|0.005723862423|+0.001782862423|0.005614530460|+0.001673530460|
|NSE|0.999979000000|0.999909508755|-0.000069491245|0.999912932701|-0.000066067299|
|Pearson r|0.999990000000|0.999956454124|-0.000033545876|0.999959125438|-0.000030874562|
|CSI@0.001|0.939638000000|0.761073770525|-0.178564229475|0.935508782929|-0.004129217071|
|CSI@0.01|0.984588000000|0.990944324520|+0.006356324520|0.990944324520|+0.006356324520|

The diagnostic `threshold_to_zero_0.005` row should not be claimed as an official result. It is useful because it shows that the CSI@0.001 gap is largely a wet/dry threshold sensitivity problem rather than a global field-quality collapse.

## 9. Ranked Gaps

### Confirmed gaps

1. **No output physical constraint in official-v0.** The model can emit tiny positive depths in near-dry cells, directly harming CSI@0.001.
2. **No physical-channel normalization.** DEM/rain/depth/target are raw except X/Y/T, while DEM has a much larger scale.
3. **No explicit valid-domain mask.** Metrics use all dense pixels.
4. **Metric convention ambiguity.** Multiple RMSE definitions exist in the repo, and Table 4 names `RMSE` rather than specifying whether it is classical, relative, masked, or aggregated per sample/time.
5. **Architecture convention uncertainty.** Current model is a valid 3D FNO, but padding/library/operator details are not confirmed against official benchmark code.

### No longer leading suspects

- Rainfall timestamp alignment: current code and local HydroPML data generation both support 1800-second rainfall indexing.
- 2D-vs-3D mismatch for official-v0: official-v0 is now 3D space-time, unlike the earlier internal baseline.

## 10. At Most Three Next Controlled Experiments

### Experiment 1: official-v1 normalized physical channels

Exact change:

- Compute train-set mean/std or min/max for initial depth, DEM, rainfall, and target water depth.
- Train on normalized physical inputs and normalized target.
- Inverse-transform predictions before computing reported metrics.
- Preserve official-v0 unchanged and write outputs under a new run suffix.

Hypothesis tested:

- Whether raw physical scale imbalance, especially DEM scale and raw target MSE, explains the RMSE/NSE/Pearson gap.

Risk:

- Medium. This is scientifically reasonable, but must be documented as official-v1 unless paper authors confirm the same preprocessing.

### Experiment 2: declared wet/dry post-processing diagnostic

Exact change:

- Do not retrain.
- Evaluate raw checkpoint_best predictions after applying a declared tiny-depth threshold, e.g. `pred[pred < 0.005] = 0`.

Hypothesis tested:

- Whether CSI@0.001 is dominated by near-dry false positives.

Risk:

- High if reported as official. Low if clearly labelled as post-processing sensitivity.

### Experiment 3: canonical FNO padding ablation

Exact change:

- Add explicit padding before Fourier layers and crop afterward, in a separate model/config path.

Hypothesis tested:

- Whether non-periodic flood-domain boundaries and no-padding FNO behavior explain residual RMSE/front errors.

Risk:

- Medium. Common in FNO implementations, but not confirmed by available paper details.

## 11. Decision

Single best next training experiment:

**Run official-v1 with explicit train-set normalization/inverse-transform, not another unchanged official-v0 run.**

Code change recommended before retraining:

**Yes, but only in a separate versioned path/config.** Do not mutate official-v0. Add explicit normalization/inverse-transform support and keep the current raw official-v0 as a baseline.

Exact question to ask the paper authors:

> For Table 4 high-fidelity 60m FNO/FNO+, what exact preprocessing and evaluation were used: were water depth, DEM, and rainfall normalized; was the predicted water depth constrained or thresholded to be non-negative; was any wet/dry or valid-domain mask applied; and is the reported RMSE classical RMSE, relative RMSE, or a masked/averaged variant?

## 12. Machine-Readable Summary

A companion JSON summary is saved at `/home/wissam/utem-workspace/code/a-iat-research/reports/fno_plus_official_v0_next_gap_decision_summary.json`.
