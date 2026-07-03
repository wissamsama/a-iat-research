# FNO+ Official-v0 Protocol Gap Audit

Audit only. No training was launched and no model/dataset code was modified.

## Ranked Suspected Gaps

| Rank | Suspected gap | Severity | Evidence | Could explain RMSE gap? | Could explain CSI@0.001 gap? | Next action |
|---:|---|---|---|---|---|---|
| 1 | Metric convention/equivalence remains unconfirmed | critical | Local `current_relative_rmse` is `sqrt(SSE/sum(y^2))`; official Table 4 says RMSE but exact aggregation/masking is not in local code. Closest local scalar to official 0.003941 is classical RMSE 0.005724, still above official. | yes | indirect | Verify official metric code or reproduce Table metrics from original benchmark implementation if available. |
| 2 | Rainfall temporal interpretation is correct for current files but still coarse relative to model output | major | Local rainfall files are 481 TIFFs at 1800 s intervals; current loader repeats rainfall for six 300 s water-depth frames; official data-generation repo also uses `current_time / 1800`. | possible minor/unknown | possible minor/unknown | Do not change to 300 s matching unless new rainfall files exist; document repeated rainfall. |
| 3 | No output non-negativity or dry-pixel post-processing in raw result | major | Post-processing threshold-to-zero 0.005 raises CSI@0.001 to ~0.9355; raw predictions overpredict tiny positives. | small | yes, major | Treat as diagnostic only; if used, define as separate post-processing experiment. |
| 4 | No explicit invalid-domain mask applied | unknown/major | Raster masks report all valid for sampled rasters; no nodata invalid region found, but visual triangular/outside-domain question cannot be resolved from masks alone. | possible | possible | Inspect georeferenced files/domain metadata and official evaluation mask if available. |
| 5 | Architecture may still differ from paper implementation | unknown/major | `FNOPlusOfficial3d` is a plausible 3D FNO but exact official benchmark training code was not found in HydroPML repo. No padding. T modes truncated to 11 because T=20 rFFT. | possible | possible | Locate official benchmark training implementation or ask authors. |
| 6 | Raw-scale input normalization absent | major | DEM/rainfall/depth remain raw while X/Y/T are 0..1; channel scales differ strongly. | possible | possible | Test normalization only in a new controlled experiment, not by changing current run. |

## 1. Rainfall Alignment Audit

- Rainfall folder: `/home/wissam/utem-workspace/data/FloodCastBench/Relevant data/Rainfall/Australia flood`
- Number of rainfall TIFFs: `481`
- First 20 rainfall filenames: `['20220220-S190000.tif', '20220220-S193000.tif', '20220220-S200000.tif', '20220220-S203000.tif', '20220220-S210000.tif', '20220220-S213000.tif', '20220220-S220000.tif', '20220220-S223000.tif', '20220220-S230000.tif', '20220220-S233000.tif', '20220221-S000000.tif', '20220221-S003000.tif', '20220221-S010000.tif', '20220221-S013000.tif', '20220221-S020000.tif', '20220221-S023000.tif', '20220221-S030000.tif', '20220221-S033000.tif', '20220221-S040000.tif', '20220221-S043000.tif']`
- Last 20 rainfall filenames: `['20220302-S093000.tif', '20220302-S100000.tif', '20220302-S103000.tif', '20220302-S110000.tif', '20220302-S113000.tif', '20220302-S120000.tif', '20220302-S123000.tif', '20220302-S130000.tif', '20220302-S133000.tif', '20220302-S140000.tif', '20220302-S143000.tif', '20220302-S150000.tif', '20220302-S153000.tif', '20220302-S160000.tif', '20220302-S163000.tif', '20220302-S170000.tif', '20220302-S173000.tif', '20220302-S180000.tif', '20220302-S183000.tif', '20220302-S190000.tif']`
- Filename pattern: `YYYYMMDD-SHHMMSS.tif`, date-time encoded.
- Lexicographic sorting equals chronological sorting: `True`
- Inferred rainfall temporal step seconds: `{1800.0: 480}`
- Conclusion: local downloaded Australia rainfall files are 1800-second / 30-minute resolution, not 300-second resolution.
- Current loader uses `rainfall_index = min(int(water_timestamp // 1800), len(rainfall_frames) - 1)`.
- Therefore rainfall is intentionally repeated for six consecutive 300-second water-depth frames.

### Sample global index 0
- Water timestamps: `[0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600, 3900, 4200, 4500, 4800, 5100, 5400, 5700]`
- Rainfall files used: `['20220220-S190000.tif', '20220220-S190000.tif', '20220220-S190000.tif', '20220220-S190000.tif', '20220220-S190000.tif', '20220220-S190000.tif', '20220220-S193000.tif', '20220220-S193000.tif', '20220220-S193000.tif', '20220220-S193000.tif', '20220220-S193000.tif', '20220220-S193000.tif', '20220220-S200000.tif', '20220220-S200000.tif', '20220220-S200000.tif', '20220220-S200000.tif', '20220220-S200000.tif', '20220220-S200000.tif', '20220220-S203000.tif', '20220220-S203000.tif']`
- Unique rainfall sequence: `['20220220-S190000.tif', '20220220-S193000.tif', '20220220-S200000.tif', '20220220-S203000.tif']`

### Sample global index 1
- Water timestamps: `[6000, 6300, 6600, 6900, 7200, 7500, 7800, 8100, 8400, 8700, 9000, 9300, 9600, 9900, 10200, 10500, 10800, 11100, 11400, 11700]`
- Rainfall files used: `['20220220-S203000.tif', '20220220-S203000.tif', '20220220-S203000.tif', '20220220-S203000.tif', '20220220-S210000.tif', '20220220-S210000.tif', '20220220-S210000.tif', '20220220-S210000.tif', '20220220-S210000.tif', '20220220-S210000.tif', '20220220-S213000.tif', '20220220-S213000.tif', '20220220-S213000.tif', '20220220-S213000.tif', '20220220-S213000.tif', '20220220-S213000.tif', '20220220-S220000.tif', '20220220-S220000.tif', '20220220-S220000.tif', '20220220-S220000.tif']`
- Unique rainfall sequence: `['20220220-S203000.tif', '20220220-S210000.tif', '20220220-S213000.tif', '20220220-S220000.tif']`

### Sample global index 2
- Water timestamps: `[12000, 12300, 12600, 12900, 13200, 13500, 13800, 14100, 14400, 14700, 15000, 15300, 15600, 15900, 16200, 16500, 16800, 17100, 17400, 17700]`
- Rainfall files used: `['20220220-S220000.tif', '20220220-S220000.tif', '20220220-S223000.tif', '20220220-S223000.tif', '20220220-S223000.tif', '20220220-S223000.tif', '20220220-S223000.tif', '20220220-S223000.tif', '20220220-S230000.tif', '20220220-S230000.tif', '20220220-S230000.tif', '20220220-S230000.tif', '20220220-S230000.tif', '20220220-S230000.tif', '20220220-S233000.tif', '20220220-S233000.tif', '20220220-S233000.tif', '20220220-S233000.tif', '20220220-S233000.tif', '20220220-S233000.tif']`
- Unique rainfall sequence: `['20220220-S220000.tif', '20220220-S223000.tif', '20220220-S230000.tif', '20220220-S233000.tif']`

### Sample global index 143
- Water timestamps: `[858000, 858300, 858600, 858900, 859200, 859500, 859800, 860100, 860400, 860700, 861000, 861300, 861600, 861900, 862200, 862500, 862800, 863100, 863400, 863700]`
- Rainfall files used: `['20220302-S170000.tif', '20220302-S170000.tif', '20220302-S173000.tif', '20220302-S173000.tif', '20220302-S173000.tif', '20220302-S173000.tif', '20220302-S173000.tif', '20220302-S173000.tif', '20220302-S180000.tif', '20220302-S180000.tif', '20220302-S180000.tif', '20220302-S180000.tif', '20220302-S180000.tif', '20220302-S180000.tif', '20220302-S183000.tif', '20220302-S183000.tif', '20220302-S183000.tif', '20220302-S183000.tif', '20220302-S183000.tif', '20220302-S183000.tif']`
- Unique rainfall sequence: `['20220302-S170000.tif', '20220302-S173000.tif', '20220302-S180000.tif', '20220302-S183000.tif']`

Rainfall t=1..20 is deterministic and aligned according to 1800-second rainfall forcing blocks, not unique 300-second rainfall frames.

## 2. Dataset Version and Structure Audit

- Dataset root: `/home/wissam/utem-workspace/data/FloodCastBench`
- Australia high-fidelity 60 m water folder: `/home/wissam/utem-workspace/data/FloodCastBench/High-fidelity flood forecasting/60m/Australia`
- Water TIFF count: `2881`
- Water filename step seconds: `{300: 2880}`
- DEM path: `/home/wissam/utem-workspace/data/FloodCastBench/Relevant data/DEM/Australia_DEM.tif`
- Georeferenced files folder exists: `True` with files `['Australia.tfw', 'Mozambique.tfw', 'Pakistan.tfw', 'Spatial_References.txt', 'UK.tfw']`
- Initial conditions folder exists: `True`
- Raster `water_first` metadata: `{'path': '/home/wissam/utem-workspace/data/FloodCastBench/High-fidelity flood forecasting/60m/Australia/0.tif', 'shape': [536, 536], 'dtype': 'float32', 'crs': 'None', 'transform': '| 1.00, 0.00, 0.00|\n| 0.00, 1.00, 0.00|\n| 0.00, 0.00, 1.00|', 'nodata': None, 'mask_unique_values': [255], 'invalid_mask_pixels': 0, 'min': 2.8960159397684038e-05, 'max': 0.6093350648880005, 'mean': 0.002194788306951523, 'zero_count': 0, 'nan_count': 0}`
- Raster `water_last` metadata: `{'path': '/home/wissam/utem-workspace/data/FloodCastBench/High-fidelity flood forecasting/60m/Australia/864000.tif', 'shape': [536, 536], 'dtype': 'float32', 'crs': 'None', 'transform': '| 1.00, 0.00, 0.00|\n| 0.00, 1.00, 0.00|\n| 0.00, 0.00, 1.00|', 'nodata': None, 'mask_unique_values': [255], 'invalid_mask_pixels': 0, 'min': 7.740747969364747e-05, 'max': 14.897148132324219, 'mean': 0.3650911748409271, 'zero_count': 0, 'nan_count': 0}`
- Raster `rain_first` metadata: `{'path': '/home/wissam/utem-workspace/data/FloodCastBench/Relevant data/Rainfall/Australia flood/20220220-S190000.tif', 'shape': [1073, 1073], 'dtype': 'float32', 'crs': 'EPSG:32756', 'transform': '| 30.00, 0.00, 517437.19|\n| 0.00,-30.00, 6808613.86|\n| 0.00, 0.00, 1.00|', 'nodata': -3.4028230607370965e+38, 'mask_unique_values': [255], 'invalid_mask_pixels': 0, 'min': 0.0, 'max': 0.0, 'mean': 0.0, 'zero_count': 1151329, 'nan_count': 0}`
- Raster `dem` metadata: `{'path': '/home/wissam/utem-workspace/data/FloodCastBench/Relevant data/DEM/Australia_DEM.tif', 'shape': [1073, 1073], 'dtype': 'float32', 'crs': 'EPSG:32756', 'transform': '| 30.00, 0.00, 517437.19|\n| 0.00,-30.00, 6808613.86|\n| 0.00, 0.00, 1.00|', 'nodata': -3.4028234663852886e+38, 'mask_unique_values': [255], 'invalid_mask_pixels': 0, 'min': -3.676865816116333, 'max': 216.3242645263672, 'mean': 22.018657684326172, 'zero_count': 126223, 'nan_count': 0}`
- Sampled raster masks have no invalid mask pixels (`invalid_mask_pixels=0`) in the inspected water/rain/DEM rasters. No nodata-domain mask was found from these raster masks.
- The bottom-right triangular/outside-domain visual issue remains unresolved by raster masks; if it exists visually, it is not encoded as nodata in inspected rasters.

## 3. Metric Equivalence Audit

| Variant | Value | Abs diff vs official RMSE 0.003941 |
|---|---:|---:|
| `current_relative_rmse_global` | 0.00813504047843 | 0.00419404047843 |
| `classical_rmse_global` | 0.00572386243602 | 0.00178286243602 |
| `paper_formula_raw_no_epsilon_ignore_invalid_mean` | 37.5240625142 | 37.5201215142 |
| `paper_formula_eps_1e_12_global_mean` | 44.4652318449 | 44.4612908449 |
| `paper_formula_eps_1e_6_global_mean` | 1.51960570623 | 1.51566470623 |
| `mean_over_samples_current_relative_rmse` | 0.00813493934655 | 0.00419393934655 |
| `mean_over_samples_classical_rmse` | 0.00572379966303 | 0.00178279966303 |
| `mean_over_samples_paper_eps12` | 44.465231623 | 44.461290623 |
| `mean_over_timesteps_current_relative_rmse` | 0.00812983011482 | 0.00418883011482 |
| `mean_over_timesteps_classical_rmse` | 0.00572020661591 | 0.00177920661591 |
| `mean_over_timesteps_paper_eps12` | 44.4652314131 | 44.4612904131 |
| `paper_formula_only_y_gt_0_0` | 37.5240625142 | 37.5201215142 |
| `paper_formula_only_y_gt_0_001` | 0.162717369889 | 0.158776369889 |
| `paper_formula_only_y_gt_0_005` | 0.00472191477474 | 0.000780914774738 |
| `paper_formula_only_y_gt_0_01` | 0.00180864607231 | 0.00213235392769 |

Closest scalar variants to official RMSE:
- `paper_formula_only_y_gt_0_005` = `0.004721914774738134`; abs diff `0.0007809147747381336`
- `mean_over_timesteps_classical_rmse` = `0.005720206615907914`; abs diff `0.0017792066159079143`
- `mean_over_samples_classical_rmse` = `0.005723799663031973`; abs diff `0.001782799663031973`
- `classical_rmse_global` = `0.005723862436022205`; abs diff `0.001782862436022205`
- `paper_formula_only_y_gt_0_01` = `0.0018086460723113775`; abs diff `0.0021323539276886226`
- `mean_over_timesteps_current_relative_rmse` = `0.008129830114823522`; abs diff `0.0041888301148235215`
- `mean_over_samples_current_relative_rmse` = `0.008134939346545361`; abs diff `0.004193939346545361`
- `current_relative_rmse_global` = `0.008135040478428111`; abs diff `0.004194040478428111`

Metric equivalence is not confirmed. The closest local scalar is still above the official value, and several paper-formula variants are on a very different scale because dry/near-zero pixels dominate the denominator.

## 4. Wet/Dry and Clipping Convention Audit

Local search findings include these representative hits:

- `configs/floodcastbench_cnn_baseline.yaml:39:  # output_activation: softplus`
- `tests/test_fno_plus_smoke.py:11:from datasets.floodcastbench_fno_dataset import FloodCastBenchFNODataset`
- `tests/test_fno_plus_smoke.py:63:    dataset = FloodCastBenchFNODataset(`
- `README.md:78:Dry-run the current Mamba h72 configuration without launching training:`
- `README.md:93:  --dry-run-config`
- `metrics/floodcastbench_eval.py:30:        self.mask = {gamma: BinaryMetricAccumulator() for gamma in gammas}`
- `metrics/floodcastbench_eval.py:36:            current_mask = current > gamma`
- `metrics/floodcastbench_eval.py:37:            pred_mask = pred > gamma`
- `metrics/floodcastbench_eval.py:38:            target_mask = target > gamma`
- `metrics/floodcastbench_eval.py:39:            self.mask[gamma].update(pred_mask, target_mask)`
- `metrics/floodcastbench_eval.py:40:            self.path[gamma].update(pred_mask & (~current_mask), target_mask & (~current_mask))`
- `metrics/floodcastbench_eval.py:44:        for gamma, acc in self.mask.items():`
- `metrics/floodcastbench_eval.py:51:class RawClampedMetricBundle:`
- `metrics/floodcastbench_eval.py:54:        self.clamped = ForecastMetricBundle(gammas)`
- `metrics/floodcastbench_eval.py:66:        pred_clamped = torch.clamp(pred_raw, min=0.0)`
- `metrics/floodcastbench_eval.py:68:        self.clamped.update(pred_clamped, target, current)`
- `metrics/floodcastbench_eval.py:79:        clamped = self.clamped.compute()`
- `metrics/floodcastbench_eval.py:82:            "clamped": clamped,`
- `metrics/floodcastbench_eval.py:83:            "loss": self.loss_sum / self.batches if self.batches else math.nan,`
- `metrics/floodcastbench_eval.py:93:        "mae": selected.get("mae", math.nan),`
- `metrics/floodcastbench_eval.py:94:        "mse": selected.get("mse", math.nan),`
- `metrics/floodcastbench_eval.py:95:        "rmse": selected.get("rmse", math.nan),`
- `metrics/floodcastbench_eval.py:96:        "nse": selected.get("nse", math.nan),`
- `metrics/floodcastbench_eval.py:97:        "pearson_r": selected.get("pearson_r", math.nan),`
- `metrics/floodcastbench_eval.py:98:        "csi_gamma_0_001": selected.get("csi_gamma_0_001", math.nan),`
- `metrics/floodcastbench_eval.py:99:        "csi_gamma_0_01": selected.get("csi_gamma_0_01", math.nan),`
- `metrics/floodcastbench_eval.py:100:        "path_iou_gamma_0_001": selected.get("path_iou_gamma_0_001", math.nan),`
- `metrics/floodcastbench_eval.py:101:        "path_iou_gamma_0_01": selected.get("path_iou_gamma_0_01", math.nan),`
- `metrics/floodcastbench_eval.py:102:        "loss": metrics.get("loss", math.nan),`
- `metrics/__init__.py:4:    binary_mask_metrics,`
- `metrics/__init__.py:13:    "binary_mask_metrics",`
- `tools/evaluate_floodcastbench_baseline.py:25:MASK_THRESHOLDS = (0.001, 0.01)`
- `tools/evaluate_floodcastbench_baseline.py:58:    if value is None or math.isnan(value):`
- `tools/evaluate_floodcastbench_baseline.py:59:        return "nan"`
- `tools/evaluate_floodcastbench_baseline.py:81:    mask_accs = {gamma: BinaryMetricAccumulator() for gamma in MASK_THRESHOLDS}`
- `tools/evaluate_floodcastbench_baseline.py:82:    path_accs = {gamma: BinaryMetricAccumulator() for gamma in MASK_THRESHOLDS}`
- `tools/evaluate_floodcastbench_baseline.py:95:            for gamma in MASK_THRESHOLDS:`
- `tools/evaluate_floodcastbench_baseline.py:96:                current_mask = current > gamma`
- `tools/evaluate_floodcastbench_baseline.py:97:                pred_future_mask = prediction > gamma`
- `tools/evaluate_floodcastbench_baseline.py:98:                target_future_mask = target > gamma`
- `tools/evaluate_floodcastbench_baseline.py:99:                mask_accs[gamma].update(pred_future_mask, target_future_mask)`
- `tools/evaluate_floodcastbench_baseline.py:101:                pred_path = pred_future_mask & (~current_mask)`
- `tools/evaluate_floodcastbench_baseline.py:102:                target_path = target_future_mask & (~current_mask)`
- `tools/evaluate_floodcastbench_baseline.py:125:        "mask": {str(gamma): acc.compute() for gamma, acc in mask_accs.items()},`
- `tools/evaluate_floodcastbench_baseline.py:142:    for gamma in MASK_THRESHOLDS:`
- `tools/evaluate_floodcastbench_baseline.py:144:            f"Mask metrics gamma={gamma}",`
- `tools/evaluate_floodcastbench_baseline.py:145:            results["mask"][str(gamma)],`
- `tools/evaluate_floodcastbench_baseline.py:148:    for gamma in MASK_THRESHOLDS:`
- `tools/evaluate_floodcastbench_baseline.py:172:    for gamma, metrics in results["mask"].items():`
- `tools/evaluate_floodcastbench_baseline.py:174:            rows.append({"section": f"mask_gamma_{gamma}", "metric": metric, "value": value})`
- `attacks/fgsm.py:39:    adversarial_pixels = torch.clamp(pixel_images + perturbation, 0.0, 1.0)`
- `datasets/floodcastbench_fno_dataset.py:75:        if dataset.nodata is not None:`
- `datasets/floodcastbench_fno_dataset.py:76:            array = np.where(array == dataset.nodata, 0.0, array)`
- `datasets/floodcastbench_fno_dataset.py:77:    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)`
- `datasets/floodcastbench_fno_dataset.py:91:class FloodCastBenchFNODataset(Dataset):`
- `datasets/floodcastbench_fno_dataset.py:252:def build_fno_datasets(root: str | Path, config: dict[str, Any]) -> dict[str, FloodCastBenchFNODataset]:`
- `datasets/floodcastbench_fno_dataset.py:267:        split: FloodCastBenchFNODataset(split=split, **common)`
- `datasets/floodcastbench.py:39:def derive_flood_mask(water_depth: torch.Tensor, gamma: float = 0.001) -> torch.Tensor:`
- `datasets/floodcastbench.py:40:    """Derive a flood mask on the fly without writing derived mask files."""`
- `datasets/floodcastbench.py:50:    current_mask = derive_flood_mask(current_water_depth, gamma=gamma)`
- `datasets/floodcastbench.py:51:    future_mask = derive_flood_mask(future_water_depth, gamma=gamma)`
- `datasets/floodcastbench.py:52:    return future_mask & (~current_mask)`
- `datasets/floodcastbench.py:241:            if dataset.nodata is not None:`
- `datasets/floodcastbench.py:242:                array = np.where(array == dataset.nodata, np.nan, array)`
- `scripts/visualize_floodcastbench_samples.py:14:    p.add_argument("--threshold", type=float, default=0.01)`
- `scripts/visualize_floodcastbench_samples.py:23:        return src.read(1, masked=True).astype(np.float64).filled(np.nan)`
- `scripts/visualize_floodcastbench_samples.py:29:    lo, hi = np.nanpercentile(arr, [2, 98])`
- `scripts/visualize_floodcastbench_samples.py:31:    return np.clip((arr - lo) / (hi - lo), 0, 1)`
- `scripts/visualize_floodcastbench_samples.py:77:        cmask = current > args.threshold; fmask = future > args.threshold; newpix = fmask & ~cmask`
- `scripts/visualize_floodcastbench_samples.py:78:        save_single(cmask.astype(float), figures / "derived_flood_mask_threshold_0p01.png", f"Flood mask threshold={args.threshold}", cmap="Blues")`
- `scripts/visualize_floodcastbench_samples.py:79:        save_grid([(cmask.astype(float), "current mask"), (fmask.astype(float), "future mask"), (newpix.astype(float), "newly flooded")], figures / "propagation_path_preview.png", f"Propagation path threshold={args.threshold}", cmap="Blues")`
- `metrics/forecasting.py:10:    """Return flattened finite pred/target values sharing the same valid-pixel mask."""`
- `metrics/forecasting.py:13:    mask = torch.isfinite(pred_flat) & torch.isfinite(target_flat)`
- `metrics/forecasting.py:14:    return pred_flat[mask], target_flat[mask]`
- `metrics/forecasting.py:19:        return math.nan`
- `metrics/forecasting.py:26:        return {"mae": math.nan, "mse": math.nan, "rmse": math.nan, "nse": math.nan, "pearson_r": math.nan}`
- `metrics/forecasting.py:49:def region_error_metrics(pred: torch.Tensor, target: torch.Tensor, region_mask: torch.Tensor) -> dict[str, float | int]:`
- `metrics/forecasting.py:50:    """Compute MAE/RMSE over finite pixels inside a boolean region mask."""`
- `metrics/forecasting.py:53:    mask_flat = region_mask.detach().bool().reshape(-1)`
- `metrics/forecasting.py:54:    finite_mask = torch.isfinite(pred_flat) & torch.isfinite(target_flat) & mask_flat`

For FNO+ official-v0 specifically, no output clamp/ReLU/Softplus/post-threshold is applied in `models/fno_plus_official.py` or `tools/train_floodcastbench_fno_plus_official.py`; output activation is identity.

## 5. Model Fidelity Audit

- Tensor order entering model: `[B, C, H, W, T]`.
- `nn.Conv3d` interprets this as `[N, C, D, H, W]`; in this implementation those physical axes are `[H, W, T]`.
- FFT uses `torch.fft.rfftn(x, dim=(-3, -2, -1))`, therefore over `H,W,T`.
- Time length is 20; rFFT time bins are `20 // 2 + 1 = 11`, so `modes=12` is truncated to `mt=11` in time.
- Spatial modes use up to 12 in both H and W.
- Four spectral + pointwise residual blocks are used, followed by two projection Conv3d layers.
- No spatial/temporal padding is used.
- Coordinates X/Y are normalized 0..1. T is normalized `linspace(0,1,20)`.
- Output is `x[..., 1:20]`, corresponding to t=2..20 from a 20-time output volume.

## 6. Normalization Audit

| Variable/channel | Min | Max | Mean | Std |
|---|---:|---:|---:|---:|
| `X` | 0 | 1 | 0.5 | 0.289214206653 |
| `Y` | 0 | 1 | 0.5 | 0.289214206653 |
| `T` | 0 | 1 | 0.5 | 0.303488486685 |
| `initial_depth` | 2.19335106522e-05 | 15.0083646774 | 0.104619292097 | 0.288248768638 |
| `DEM` | -1.4188876152 | 214.345993042 | 22.019778904 | 39.5002153245 |
| `rainfall` | 0 | 26.3299999237 | 1.87318808458 | 3.09213886028 |
| `target_water_depth` | 2.16597545659e-05 | 15.013915062 | 0.106157868828 | 0.291286000404 |

The code normalizes only X/Y/T coordinate channels. DEM, rainfall, initial water depth, and target water depth are raw. Raw scales differ substantially, especially between coordinate channels, DEM, rainfall, and water depth. Whether the paper normalizes these variables is unknown from local code and the cloned repository.

## 7. Split and Sample Indexing Audit

- Total non-overlapping 20-frame windows: `144`
- Train/val/test counts: `116/14/14`
- Train start indices: `[0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 940, 960, 980, 1000, 1020, 1040, 1060, 1080, 1100, 1120, 1140, 1160, 1180, 1200, 1220, 1240, 1260, 1280, 1300, 1320, 1340, 1360, 1380, 1400, 1420, 1440, 1460, 1480, 1500, 1520, 1540, 1560, 1580, 1600, 1620, 1640, 1660, 1680, 1700, 1720, 1740, 1760, 1780, 1800, 1820, 1840, 1860, 1880, 1900, 1920, 1940, 1960, 1980, 2000, 2020, 2040, 2060, 2080, 2100, 2120, 2140, 2160, 2180, 2200, 2220, 2240, 2260, 2280, 2300]`
- Val start indices: `[2320, 2340, 2360, 2380, 2400, 2420, 2440, 2460, 2480, 2500, 2520, 2540, 2560, 2580]`
- Test start indices: `[2600, 2620, 2640, 2660, 2680, 2700, 2720, 2740, 2760, 2780, 2800, 2820, 2840, 2860]`
- Train input timestamps: `[0, 6000, 12000, 18000, 24000, 30000, 36000, 42000, 48000, 54000, 60000, 66000, 72000, 78000, 84000, 90000, 96000, 102000, 108000, 114000, 120000, 126000, 132000, 138000, 144000, 150000, 156000, 162000, 168000, 174000, 180000, 186000, 192000, 198000, 204000, 210000, 216000, 222000, 228000, 234000, 240000, 246000, 252000, 258000, 264000, 270000, 276000, 282000, 288000, 294000, 300000, 306000, 312000, 318000, 324000, 330000, 336000, 342000, 348000, 354000, 360000, 366000, 372000, 378000, 384000, 390000, 396000, 402000, 408000, 414000, 420000, 426000, 432000, 438000, 444000, 450000, 456000, 462000, 468000, 474000, 480000, 486000, 492000, 498000, 504000, 510000, 516000, 522000, 528000, 534000, 540000, 546000, 552000, 558000, 564000, 570000, 576000, 582000, 588000, 594000, 600000, 606000, 612000, 618000, 624000, 630000, 636000, 642000, 648000, 654000, 660000, 666000, 672000, 678000, 684000, 690000]`
- Val input timestamps: `[696000, 702000, 708000, 714000, 720000, 726000, 732000, 738000, 744000, 750000, 756000, 762000, 768000, 774000]`
- Test input timestamps: `[780000, 786000, 792000, 798000, 804000, 810000, 816000, 822000, 828000, 834000, 840000, 846000, 852000, 858000]`
- Windows are non-overlapping because stride is 20 frames and sample length is 20 frames.
- Target has 19 output frames: water-depth frames 2..20. The paper wording t=1 vs t=0 remains a naming ambiguity, but local code uses the first frame as initial input and the following 19 frames as target.

## 8. Official Repository Audit

- Clone path inspected: `/tmp/FloodCastBench_official_audit`
- The cloned HydroPML/FloodCastBench repository contains data-generation/hydraulic simulation code and README material.
- No FNO/FNO+/U-Net benchmark training scripts were found by filename/content search.
- No benchmark metric implementation for Table 4 was found.
- Rainfall/data-generation evidence for 1800-second forcing:
  - `Data_Generation_Code/main.py:100: mp = int(current_time / 1800)`

## Conclusion

- Rainfall alignment is not wrong for the downloaded local files: they are 1800-second rainfall TIFFs, and the official data-generation repository also uses `current_time / 1800`.
- Metric equivalence is not confirmed; this remains the most critical protocol uncertainty.
- The raw CSI@0.001 gap is strongly linked to tiny positive predictions in near-dry pixels, but using threshold-to-zero post-processing would be a separate experiment, not the raw official-v0 result.
- Before retraining, the most justified code change to consider is not rainfall alignment but an explicitly controlled normalization/post-processing/loss experiment, after clarifying the target metric convention.