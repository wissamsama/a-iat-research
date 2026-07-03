# FNO+ Status Report

Generated from local repository/workspace inspection only. No code was modified, no training was launched, and no artifact was deleted.

Snapshot note: two `tools/train_floodcastbench_fno_plus_official.py` parent processes were running during inspection. Metrics for the running `official_v0` runs may have advanced after the values recorded here.

## 1. Repository Structure Related to FNO+

### Code Files

| Path | Role |
|---|---|
| `models/fno_plus.py` | Defines the internal 2D FNO+ scaffold: `SpectralConv2d`, `FNOPlus2d`. |
| `models/fno_plus_official.py` | Defines the separate official-v0 3D space-time attempt: `SpectralConv3d`, `FNOPlusOfficial3d`. |
| `datasets/floodcastbench_fno_dataset.py` | Loads 20-frame FloodCastBench samples for the internal 2D FNO+ scaffold. |
| `datasets/floodcastbench_fno_plus_official_dataset.py` | Loads `[6,H,W,20]` samples for the official-v0 3D FNO+ attempt. |
| `evaluation/floodcastbench_official_metrics.py` | Implements relative RMSE, NSE, Pearson r, CSI, and `OfficialFloodMetricAccumulator`. |
| `tools/train_floodcastbench_fno_plus.py` | Trains the internal `FNOPlus2d` scaffold, writes `metrics.csv`, `summary.json`, and checkpoints. |
| `tools/train_floodcastbench_fno_plus_official.py` | Trains `FNOPlusOfficial3d`, writes `metrics.csv`, `summary.json`, and checkpoints. |
| `tools/evaluate_floodcastbench_fno_plus_official.py` | Evaluates official-v0 checkpoints on val/test and writes a JSON metrics file. |
| `tools/recompute_fno_plus_official_metrics.py` | Recomputes metric variants for the internal 2D FNO+ checkpoint; also defines `MetricAccumulator` used by official-v0 training/evaluation. |
| `tests/test_fno_plus_smoke.py` | Smoke tests for the internal 2D FNO+ scaffold. |
| `tests/test_fno_plus_official_smoke.py` | Smoke tests for the official-v0 3D FNO+ attempt. |

### Config Files

| Path | Purpose |
|---|---|
| `configs/floodcastbench_fno_plus_highfid_60m.yaml` | Internal 2D FNO+ Australia high-fidelity 60 m config. |
| `configs/floodcastbench_fno_plus_official_highfid_60m.yaml` | Official-v0 3D FNO+ Australia high-fidelity 60 m config. |

### Existing Reports

The repository already contains FNO+ reports and summaries under `reports/`, including:

- `reports/fno_plus_work_reconstruction.md`
- `reports/fno_plus_metric_diagnosis.md`
- `reports/fno_plus_official_reproduction_plan.md`
- `reports/fno_plus_official_v0_readiness_audit.md`
- `reports/fno_plus_official_v0_5epoch_pilot_report.md`

Those reports were treated as existing artifacts, not as assumptions.

## 2. Current FNO+ Implementations

### Internal 2D FNO+ Scaffold

Source: `models/fno_plus.py`

- Model class: `FNOPlus2d`
- Spectral layer: `SpectralConv2d`
- Expected input shape: `[B, C, H, W]`
- Output shape: `[B, output_steps, H, W]`
- Default output steps: `19`
- Fourier layers: configurable, config uses `4`
- Modes: configurable, config uses `12`
- Width: configurable, config uses `20`
- Output activation: identity; final layer is `nn.Conv2d`
- Temporal handling: temporal information is encoded as input channels, not as a Fourier/operator dimension.
- Prediction mode: one-shot prediction of all 19 target frames from one input tensor.

Input channels are built in `datasets/floodcastbench_fno_dataset.py`.

For the current config, `input_channels = 43`:

| Component | Channels |
|---|---:|
| X coordinate | 1 |
| Y coordinate | 1 |
| Initial water depth `t=1` | 1 |
| DEM | 1 |
| Rainfall `t=1..20` | 20 |
| Time channels for target steps | 19 |
| Total | 43 |

Target tensor:

- Shape: `[19,H,W]`
- Meaning: water depth frames `t=2..20`

### Official-v0 3D Space-Time FNO+ Attempt

Source: `models/fno_plus_official.py`

- Model class: `FNOPlusOfficial3d`
- Spectral layer: `SpectralConv3d`
- Expected input shape: `[B, C, H, W, T]`
- Configured input channels: `6`
- Input time length: `20`
- Output shape: `[B, 1, H, W, 19]`
- Fourier modes are applied over `H`, `W`, and `T` through `torch.fft.rfftn(..., dim=(-3,-2,-1))`.
- Fourier layers: config uses `4`
- Modes: config uses `12`
- Width: config uses `20`
- Output activation: identity; final layer is `nn.Conv3d`
- Temporal handling: direct space-time FNO over `[H,W,T]`
- Prediction mode: one-shot output for `t=2..20`

Input channels from `datasets/floodcastbench_fno_plus_official_dataset.py`:

| Channel | Meaning |
|---|---|
| 0 | X coordinate, broadcast across 20 time steps |
| 1 | Y coordinate, broadcast across 20 time steps |
| 2 | T coordinate, `torch.linspace(0.0, 1.0, 20)` |
| 3 | Initial water depth `t=1`, broadcast across 20 time steps |
| 4 | DEM, broadcast across 20 time steps |
| 5 | Rainfall aligned to each of the 20 water-depth timestamps |

Target tensor:

- Shape: `[1,H,W,19]`
- Meaning: water depth frames `t=2..20`

## 3. Dataset and Protocol Currently Used

Both FNO+ implementations use:

- Dataset root: `/home/wissam/utem-workspace/data/FloodCastBench`
- Water-depth folder: `/home/wissam/utem-workspace/data/FloodCastBench/High-fidelity flood forecasting/60m/Australia`
- Event: `Australia`
- Fidelity: `high`
- Resolution: `60m`
- Water-depth TIFF count observed: `2881`
- Spatial shape from code/config usage: `536 x 536`
- Sample length: `20`
- Stride: `20`
- Windowing: non-overlapping starts from `range(0, len(frames), stride)` where `start + sample_length <= len(frames)`
- Split counts from config: `train=116`, `val=14`, `test=14`
- Target: water depth for frames `t=2..20`

Water-depth file ordering:

- Water-depth file stems are parsed as integers.
- Frames are sorted by integer timestamp.

Rainfall ordering/alignment:

- Rainfall files are loaded with `sorted(folder.glob("*.tif"))`, i.e. lexicographic path order.
- Rainfall for a water-depth timestamp uses `rainfall_index = min(int(water_timestamp // 1800), len(rainfall_frames)-1)`.
- The code comments state rainfall is every `1800` seconds while water depth is every `300` seconds.
- This is deterministic in code, but whether the lexicographic rainfall ordering exactly matches chronological order was not proven here.

Static/dynamic variables:

| Variable | Internal 2D FNO+ | Official-v0 3D FNO+ |
|---|---|---|
| X coordinate | yes | yes |
| Y coordinate | yes | yes |
| Time coordinate/channel | yes, 19 target-time channels | yes, 20 time coordinates |
| Initial water depth | yes | yes |
| DEM | yes | yes |
| Rainfall | yes, 20 channels | yes, 20 time slices |

Normalization:

- DEM: loaded raw, resized by bilinear interpolation if needed; no normalization found.
- Rainfall: loaded raw, resized by bilinear interpolation if needed; no normalization found.
- Water depth: loaded raw; no normalization found.
- Nodata and NaN/inf handling: nodata values are set to `0.0`; NaN, positive infinity, and negative infinity are replaced with `0.0`.

## 4. Training Configuration

### Internal 2D FNO+ Config

Source: `configs/floodcastbench_fno_plus_highfid_60m.yaml`

| Field | Value |
|---|---|
| Model | `fno_plus` / `FNOPlus2d` |
| Modes | `12` |
| Width | `20` |
| Fourier layers | `4` |
| Output steps | `19` |
| Batch size | `1` |
| Num workers | `2` |
| Epochs | `100` unless overridden |
| Optimizer | Adam |
| Learning rate | `0.001` |
| Betas | `[0.9, 0.999]` |
| Weight decay | `0.0001` |
| Scheduler | CosineAnnealingLR, eta min `0.0` |
| Loss | MSELoss |
| Device | `auto` |
| Seed | `42` |
| Checkpoint selection | lowest validation `relative_rmse` in code |

### Official-v0 3D FNO+ Config

Source: `configs/floodcastbench_fno_plus_official_highfid_60m.yaml`

| Field | Value |
|---|---|
| Model | `fno_plus_official_v0` / `FNOPlusOfficial3d` |
| Input channels | `6` |
| Modes | `12` |
| Width | `20` |
| Fourier layers | `4` |
| Output steps | `19` |
| Batch size | `1` |
| Num workers | `2` |
| Epochs | `100` unless overridden |
| Optimizer | Adam |
| Learning rate | `0.001` |
| Betas | `[0.9, 0.999]` |
| Weight decay | `0.0001` |
| Scheduler | CosineAnnealingLR, eta min `0.0` |
| Loss | MSELoss |
| Device | `auto` |
| Seed | `42` |
| Checkpoint selection | `val_current_relative_rmse` |

No explicit GPU model is written into `metrics.csv` or `summary.json`. Device is resolved by code from `device: auto`.

## 5. FNO+ Runs and Results

Experiment root:

`/home/wissam/utem-workspace/experiments/FloodCastBench`

Checkpoint root:

`/home/wissam/utem-workspace/checkpoints/FloodCastBench`

### Run Table

| Run | Implementation | Status at inspection | Planned epochs | Logged epochs | Best validation epoch | Best validation loss | Best validation RMSE/relative RMSE | Best val NSE | Best val Pearson r | Best val CSI 0.001 | Best val CSI 0.01 | Final validation metrics summary |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `27-06-2026_13-23-11_fcb_fno_plus_highfid_60m` | Internal 2D | completed | 1 | 1 | 1 | 0.505840122700 | val_relative_rmse 1.019133377139 | -0.426823904743 | 0.001141369315 | 0.383158178228 | 0.332290904044 | same as best |
| `27-06-2026_13-48-07_fcb_fno_plus_highfid_60m` | Internal 2D | completed | 5 | 5 | 5 | 0.269567902599 | val_relative_rmse 0.741641996424 | 0.245825982890 | 0.839068904334 | 0.704730287811 | 0.676852331750 | same as best |
| `27-06-2026_14-00-18_fcb_fno_plus_highfid_60m` | Internal 2D | completed | 100 | 100 | 95 | 0.000078356093 | val_relative_rmse 0.012644353209 | 0.999780782018 | 0.999894179203 | 0.748698030640 | 0.988005654902 | epoch 100: val_relative_rmse 0.012674069375, val_nse 0.999779750416, val_csi_0.001 0.752860082033, val_csi_0.01 0.987639056998 |
| `27-06-2026_16-09-27_fcb_fno_plus_official_v0_highfid_60m` | Official-v0 3D | completed | 1 | 1 | 1 | 0.446424365044 | val_current_relative_rmse 0.957722704437 | -0.260333254947 | -0.083900045157 | 0.667097788747 | 0.596751122627 | same as best |
| `27-06-2026_16-14-46_fcb_fno_plus_official_v0_highfid_60m` | Official-v0 3D | completed | 5 | 5 | 4 | 0.113369739481 | val_current_relative_rmse 0.480959830833 | 0.682823831459 | 0.860318772861 | 0.675171179994 | 0.664265677566 | epoch 5: val_current_relative_rmse 0.505726531909, val_nse 0.649317242003, val_csi_0.001 0.669725469967, val_csi_0.01 0.659305992382 |
| `27-06-2026_16-28-12_fcb_fno_plus_official_v0_highfid_60m` | Official-v0 3D | running or interrupted | 100 | 59 | 58 | 0.000075973459 | val_current_relative_rmse 0.012450625844 | 0.999787447949 | 0.999923041895 | 0.872705855271 | 0.979772792526 | epoch 59: val_current_relative_rmse 0.022933923544, val_nse 0.999278826671, val_csi_0.001 0.666744420619, val_csi_0.01 0.714585316284 |
| `27-06-2026_16-28-24_fcb_fno_plus_official_v0_highfid_60m` | Official-v0 3D | running or interrupted | 100 | 59 | 58 | 0.000075973459 | val_current_relative_rmse 0.012450625844 | 0.999787447949 | 0.999923041895 | 0.872705855271 | 0.979772792526 | epoch 59: val_current_relative_rmse 0.022933923544, val_nse 0.999278826671, val_csi_0.001 0.666744420619, val_csi_0.01 0.714585316284 |

### Detailed Per-Run Notes

#### `27-06-2026_13-23-11_fcb_fno_plus_highfid_60m`

- Config: `.../experiments/FloodCastBench/27-06-2026_13-23-11_fcb_fno_plus_highfid_60m/config.yaml`
- Metrics: `.../metrics.csv`
- Summary: exists
- Checkpoint best: `.../checkpoints/FloodCastBench/27-06-2026_13-23-11_fcb_fno_plus_highfid_60m/checkpoint_best.pth`
- Checkpoint last: `.../checkpoint_last.pth`
- Best epoch: `1`
- Final epoch: `1`
- Anomaly: learning rate is `0.0` because this was a 1-epoch run with cosine scheduler.

#### `27-06-2026_13-48-07_fcb_fno_plus_highfid_60m`

- Config, metrics, summary all exist.
- Checkpoint best and last exist.
- Best epoch: `5`
- Final epoch: `5`
- Anomaly: learning rate is `0.0` at final epoch due 5-epoch cosine schedule.

#### `27-06-2026_14-00-18_fcb_fno_plus_highfid_60m`

- Config, metrics, summary all exist.
- Completed 100 epochs.
- Best epoch by validation relative RMSE: `95`
- Checkpoint best: `/home/wissam/utem-workspace/checkpoints/FloodCastBench/27-06-2026_14-00-18_fcb_fno_plus_highfid_60m/checkpoint_best.pth`
- Checkpoint last: `/home/wissam/utem-workspace/checkpoints/FloodCastBench/27-06-2026_14-00-18_fcb_fno_plus_highfid_60m/checkpoint_last.pth`
- Test metrics file: `test_metrics_best.json`
- Diagnostics folder: `diagnostics_best_checkpoint/`
- Test metrics from `test_metrics_best.json`:
  - relative RMSE: `0.01235829833769271`
  - NSE: `0.9997911644242803`
  - Pearson r: `0.9998986562960474`
  - CSI@0.001: `0.7241764635613173`
  - CSI@0.01: `0.9857210422280918`
  - loss: `7.560948506579734e-05`
  - samples: `14`

#### `27-06-2026_16-09-27_fcb_fno_plus_official_v0_highfid_60m`

- Config, metrics, summary all exist.
- Completed 1 epoch.
- Checkpoint best and last exist.
- Best epoch: `1`
- Final epoch: `1`
- Metric anomaly: `val_paper_formula_rmse = 54750.32929617428`, much larger than `val_current_relative_rmse = 0.9577227044366031`; this follows from the implemented paper-formula variant dividing by `target^2 + 1e-12` per pixel.

#### `27-06-2026_16-14-46_fcb_fno_plus_official_v0_highfid_60m`

- Config, metrics, summary all exist.
- Completed 5 epochs.
- Best epoch by `val_current_relative_rmse`: `4`
- Additional JSON: `val_subset_prediction_stats_checkpoint_best.json`
- Checkpoint best and last exist.
- Metric anomaly: `val_paper_formula_rmse` remains very large relative to other RMSE variants.

#### `27-06-2026_16-28-12_fcb_fno_plus_official_v0_highfid_60m`

- Config and metrics exist.
- No `summary.json` at inspection time.
- Logged epochs at inspection: `59`
- Planned epochs from config: `100`
- Best epoch at inspection: `58`
- Checkpoint best and last exist.
- Status: running or interrupted. A parent training process was observed with PID `68285`; two child processes were observed under it.
- Associated shell log likely: `/home/wissam/utem-workspace/logs/FloodCastBench/fno_plus_official_v0_full_100epoch_20260627_162809.log`
- Anomaly: this run appears duplicated with `16-28-24`; metrics are identical through the inspected epoch.

#### `27-06-2026_16-28-24_fcb_fno_plus_official_v0_highfid_60m`

- Config and metrics exist.
- No `summary.json` at inspection time.
- Logged epochs at inspection: `59`
- Planned epochs from config: `100`
- Best epoch at inspection: `58`
- Checkpoint best and last exist.
- Status: running or interrupted. A parent training process was observed with PID `68479`; two child processes were observed under it.
- Associated shell log likely: `/home/wissam/utem-workspace/logs/FloodCastBench/fno_plus_official_v0_full_100epoch_20260627_162821.log`
- Anomaly: this run appears duplicated with `16-28-12`; metrics are identical through the inspected epoch.

## 6. Logs

The run-local log directories under `/home/wissam/utem-workspace/logs/FloodCastBench/<run_name>` exist but contain no listed files.

Root-level shell logs found:

| Log file | Size | Notes |
|---|---:|---|
| `/home/wissam/utem-workspace/logs/FloodCastBench/fno_plus_official_v0_full_100epoch_20260627_162731.log` | 0 bytes | Empty launch artifact. |
| `/home/wissam/utem-workspace/logs/FloodCastBench/fno_plus_official_v0_full_100epoch_20260627_162809.log` | 4009 bytes at inspection | Contains epoch lines through epoch 58 at first tail; associated with running official-v0 training. |
| `/home/wissam/utem-workspace/logs/FloodCastBench/fno_plus_official_v0_full_100epoch_20260627_162821.log` | 4009 bytes at inspection | Contains epoch lines through epoch 58 at first tail; associated with running official-v0 training. |
| `/home/wissam/utem-workspace/logs/FloodCastBench/fno_plus_official_v0_full_100epoch.pid` | 6 bytes | Contains PID `68479`; it does not record PID `68285`. |

Observed training processes:

| PID | PPID | Status | Elapsed at inspection | Command |
|---:|---:|---|---|---|
| 68285 | 536 | Rsl | about 1h56m | `train_floodcastbench_fno_plus_official.py --config ... --epochs 100 --num-workers 2` |
| 68479 | 536 | Rsl | about 1h56m | `train_floodcastbench_fno_plus_official.py --config ... --epochs 100 --num-workers 2` |
| 105771, 105775 | 68285 | Sl | about 1m40s | child/worker processes |
| 105836, 105837 | 68479 | Sl | about 1m30s | child/worker processes |

## 7. Metric Implementation

Source: `evaluation/floodcastbench_official_metrics.py`

Implemented direct functions:

- `relative_rmse(pred, target, eps=1e-12)`
- `nse(pred, target, eps=1e-12)`
- `pearson_r(pred, target, eps=1e-12)`
- `csi(pred, target, gamma, eps=1e-12)`

`OfficialFloodMetricAccumulator` computes global aggregate metrics over all updated batches:

- SSE: `sum((pred-target)^2)`
- Relative RMSE: `sqrt(SSE / (sum(target^2) + eps))`
- NSE: `1 - SSE / sum((target - target_mean)^2)`
- Pearson r: computed from global sums/cross-sums
- CSI: global hits/misses/false alarms at configured thresholds

No mask excludes dry or near-zero target cells. Near-zero stabilization is only via `eps=1e-12` in denominators.

CSI thresholds in configs:

- `0.001`
- `0.01`

Additional official-v0 metric variants from `tools/recompute_fno_plus_official_metrics.py`:

- `paper_formula_rmse`: implemented as `mean((pred-target)^2 / (target^2 + 1e-12))`; despite the name, the function comment says no square root.
- `classical_rmse`: `sqrt(mean((pred-target)^2))`
- `current_relative_rmse`: from `OfficialFloodMetricAccumulator`

Suspicious or unstable metric behavior:

- `paper_formula_rmse` can become extremely large when many target pixels are zero or near zero because errors are divided by `target^2 + 1e-12` per pixel.
- No dry-cell masking is applied for `paper_formula_rmse`.
- Internal 2D FNO+ metrics do not include `classical_rmse` or `paper_formula_rmse`.
- No PathIoU/flood-front metric was found in the FNO+ training outputs.

Metric aggregation:

- Training/validation metrics are accumulated globally across the full loader for the epoch.
- They are not averaged per sample or per time step in the main `metrics.csv`.
- The internal diagnostic folder contains per-timestep metrics for the internal 100-epoch run.

## 8. Checkpoints and Artifacts

### Checkpoints

All detected FNO+ runs have `checkpoint_best.pth` and `checkpoint_last.pth`.

| Run | Checkpoint size | Completeness |
|---|---:|---|
| Internal 2D runs | 11,138,699 bytes each checkpoint | Complete files present. |
| Official-v0 3D runs | 265,493,247 bytes each checkpoint | Complete files present. |

Checkpoint contents inferred from training code:

- `epoch`
- `model_state_dict`
- `optimizer_state_dict`
- `scheduler_state_dict`
- `config`
- `metrics`

### CSV Logs

Each run has `metrics.csv`.

Internal 2D CSV fields:

- `epoch`, `train_loss`, `val_loss`
- train/val `relative_rmse`, `nse`, `pearson_r`
- train/val `csi_gamma_0_001`, `csi_gamma_0_01`
- `learning_rate`, `epoch_time_sec`

Official-v0 CSV fields:

- `epoch`, `train_loss`, `val_loss`
- train/val `paper_formula_rmse`
- train/val `current_relative_rmse`
- train/val `classical_rmse`
- train/val `nse`
- train/val `pearson_r`
- train/val `csi_gamma_0_001`, `csi_gamma_0_01`
- `learning_rate`, `epoch_time_sec`

### JSON Summaries and Evaluations

| Path | Contents | Completeness/usefulness |
|---|---|---|
| `.../27-06-2026_13-23-11.../summary.json` | experiment/checkpoint/log dirs, best val relative RMSE, epochs | Complete for 1-epoch smoke. |
| `.../27-06-2026_13-48-07.../summary.json` | same structure | Complete for 5-epoch pilot. |
| `.../27-06-2026_14-00-18.../summary.json` | same structure | Complete for 100-epoch internal run. |
| `.../27-06-2026_14-00-18.../test_metrics_best.json` | test metrics for checkpoint_best | Useful final test evaluation for internal 2D run. |
| `.../27-06-2026_16-09-27.../summary.json` | official-v0 1-epoch summary | Complete. |
| `.../27-06-2026_16-14-46.../summary.json` | official-v0 5-epoch summary | Complete. |
| `.../27-06-2026_16-14-46.../val_subset_prediction_stats_checkpoint_best.json` | prediction statistics for 3 validation samples | Useful diagnostic, not full test result. |
| `.../27-06-2026_16-28-12.../summary.json` | absent | Run not complete at inspection. |
| `.../27-06-2026_16-28-24.../summary.json` | absent | Run not complete at inspection. |

### Diagnostic Figures and Tables

For internal 2D run `27-06-2026_14-00-18_fcb_fno_plus_highfid_60m`:

Folder:

`/home/wissam/utem-workspace/experiments/FloodCastBench/27-06-2026_14-00-18_fcb_fno_plus_highfid_60m/diagnostics_best_checkpoint`

Files:

- `diagnostic_metrics.json`
- `per_timestep_metrics.csv`
- `prediction_sanity.json`
- `figures/test_sample_00_diagnostics.png`
- `figures/test_sample_07_diagnostics.png`
- `figures/test_sample_13_diagnostics.png`

These artifacts are useful for analysis of the internal 2D FNO+ checkpoint_best predictions.

## 9. Comparison Between FNO+ Runs

Based only on available logs at inspection:

1. Best completed internal 2D run:
   - `27-06-2026_14-00-18_fcb_fno_plus_highfid_60m`
   - Best validation relative RMSE: `0.01264435320861495`
   - Test relative RMSE: `0.01235829833769271`
   - Test CSI@0.001: `0.7241764635613173`
   - Test CSI@0.01: `0.9857210422280918`

2. Best official-v0 run at inspection:
   - Both `27-06-2026_16-28-12_fcb_fno_plus_official_v0_highfid_60m` and `27-06-2026_16-28-24_fcb_fno_plus_official_v0_highfid_60m`
   - Best logged epoch at inspection: `58`
   - Best validation current relative RMSE: `0.012450625843578986`
   - Best validation CSI@0.001: `0.8727058552710458`
   - Best validation CSI@0.01: `0.9797727925262533`
   - Status: not complete at inspection; no `summary.json`; no test metrics found.

3. The two official-v0 100-epoch runs appear duplicated:
   - Both were launched from the same command.
   - Both have identical metrics through inspected epochs.
   - Two parent training processes were observed.
   - The PID file records only `68479`, but PID `68285` also exists as a parent process.

## 10. Factual Conclusion

What has actually been implemented so far:

- An internal 2D FNO+ scaffold exists and is trained by `tools/train_floodcastbench_fno_plus.py`.
- A separate official-v0 3D space-time FNO+ attempt exists and is trained by `tools/train_floodcastbench_fno_plus_official.py`.
- Dataset loaders exist for both versions.
- Metric accumulation exists for global relative RMSE, NSE, Pearson r, and CSI at `0.001` and `0.01`.
- Additional official-v0 metric variants exist: `paper_formula_rmse`, `classical_rmse`, and `current_relative_rmse`.

What has actually been trained so far:

- Internal 2D FNO+:
  - 1-epoch run completed.
  - 5-epoch run completed.
  - 100-epoch run completed.
- Official-v0 3D FNO+:
  - 1-epoch run completed.
  - 5-epoch run completed.
  - Two apparent duplicate 100-epoch runs were running or incomplete at inspection, with 59 logged epochs.

Best results obtained so far:

- Best completed run with test metrics: internal 2D run `27-06-2026_14-00-18_fcb_fno_plus_highfid_60m`.
- Best completed test metrics found:
  - relative RMSE: `0.01235829833769271`
  - NSE: `0.9997911644242803`
  - Pearson r: `0.9998986562960474`
  - CSI@0.001: `0.7241764635613173`
  - CSI@0.01: `0.9857210422280918`
- Best validation metrics observed at inspection: official-v0 running runs at epoch `58`, with validation current relative RMSE `0.012450625843578986`, but these are not final and no test metrics were present.

Which run currently looks best based only on available logs:

- For completed runs with test evaluation: `27-06-2026_14-00-18_fcb_fno_plus_highfid_60m`.
- For validation metrics regardless of completion: the running official-v0 runs at epoch `58` have slightly lower validation current relative RMSE and much higher CSI@0.001 than the completed internal 2D run, but they are incomplete and duplicated at inspection.

What remains unclear from the repository:

- Whether the rainfall file lexicographic order matches true chronological order for all files.
- Whether the two official-v0 100-epoch runs were intentionally both launched.
- Which of the duplicate official-v0 runs should be retained if both finish.
- Whether official-v0 final test metrics will improve over the completed internal 2D test metrics.
- Whether `paper_formula_rmse` is scientifically intended, because its current implementation is very sensitive to near-zero target depths.
- Whether any plain non-plus FNO baseline exists; none was found in the inspected files/runs.

What should be checked next before making any scientific claim:

- Wait for exactly one official-v0 100-epoch run to complete and generate `summary.json`.
- Evaluate `checkpoint_best.pth` on the test split for the completed official-v0 run.
- Resolve the duplicate-running-run situation factually before reporting final numbers.
- Verify rainfall ordering/alignment against actual rainfall timestamps.
- Decide which RMSE variant is the report metric and document it explicitly.

