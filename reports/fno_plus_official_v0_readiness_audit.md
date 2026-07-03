# FNO+ Official v0 Readiness Audit

## 1. Executive Summary

The official FNO+ v0 implementation passes real-data forward, backward, optimizer, scheduler, metric, checkpoint, and summary-writing checks on the real FloodCastBench high-fidelity Australia 60 m dataset.

Technical readiness is good:

- real-data forward pass: OK;
- mini-training 1 epoch / 1 train batch / 1 val batch: OK;
- checkpoint writing: OK;
- summary JSON writing: OK;
- metric computation: OK;
- measured GPU peak for one backward step: about 6998 MB on NVIDIA RTX 6000 Ada Generation.

Scientific readiness is more cautious. I do **not** recommend launching the full 100-epoch official-v0 run immediately as the final official reproduction attempt. The safer next step is a 5-epoch pilot, because the RMSE/dry-cell handling remains unresolved and the official-v0 currently uses raw water depth, raw DEM, and raw rainfall.

## 2. Files Inspected

- `models/fno_plus_official.py`
- `datasets/floodcastbench_fno_plus_official_dataset.py`
- `tools/train_floodcastbench_fno_plus_official.py`
- `tools/evaluate_floodcastbench_fno_plus_official.py`
- `tools/recompute_fno_plus_official_metrics.py`
- `configs/floodcastbench_fno_plus_official_highfid_60m.yaml`
- `tests/test_fno_plus_official_smoke.py`

No internal FNO+ baseline file was modified.

## 3. Real-Data Tensor Shapes

Actual dataset sample from Australia 60 m train split:

| Tensor | Shape |
|---|---:|
| input sample | `[6, 536, 536, 20]` |
| target sample | `[1, 536, 536, 19]` |
| model batched input | `[1, 6, 536, 536, 20]` |
| model output | `[1, 1, 536, 536, 19]` |

Temporal metadata:

| Field | Value |
|---|---:|
| input timestamp | 0 |
| first target timestamp | 300 |
| last target timestamp | 5700 |

## 4. Real-Data Channel Statistics

| Channel | Min | Max | Mean | Std |
|---|---:|---:|---:|---:|
| X | 0.000000 | 1.000000 | 0.500000 | 0.289214 |
| Y | 0.000000 | 1.000000 | 0.500000 | 0.289214 |
| T | 0.000000 | 1.000000 | 0.500000 | 0.303489 |
| initial depth | 0.000029 | 0.609335 | 0.002195 | 0.005758 |
| DEM | -1.418888 | 214.345993 | 22.019779 | 39.500217 |
| rainfall | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| target depth | 0.000028 | 0.609087 | 0.002195 | 0.005804 |

The first training sample has zero rainfall over all selected rainfall frames. This is not necessarily a bug, but it means one-sample channel statistics do not prove rainfall variability across the full dataset.

## 5. Forward-Pass Result

Device: `cuda`.

Forward pass result:

| Field | Value |
|---|---:|
| forward_ok | true |
| output shape | `[1, 1, 536, 536, 19]` |
| output min | 0.134022 |
| output max | 0.202727 |
| output mean | 0.147663 |
| output std | 0.015312 |
| MSE loss | 0.021447 |
| forward no-grad time | 1.699 sec |
| no-grad peak GPU memory | 3050.964 MB |

The untrained output is much larger than the shallow initial target depth, which is expected for random initialization and not a shape bug.

## 6. Mini-Training Result

Command run:

```bash
/home/wissam/miniforge3/envs/floodcast-mamba/bin/python tools/train_floodcastbench_fno_plus_official.py   --config configs/floodcastbench_fno_plus_official_highfid_60m.yaml   --epochs 1   --num-workers 0   --max-train-batches 1   --max-val-batches 1   --device auto
```

Result:

```text
epoch=1 train_loss=0.008085 val_current_relative_rmse=0.957723
```

Mini-run path:

```text
/home/wissam/utem-workspace/experiments/FloodCastBench/27-06-2026_16-09-27_fcb_fno_plus_official_v0_highfid_60m
```

Mini-run checks:

| Check | Result |
|---|---|
| forward pass | OK |
| backward pass | OK |
| optimizer step | OK |
| scheduler step | OK |
| validation pass | OK |
| metric computation | OK |
| `metrics.csv` writing | OK |
| `summary.json` writing | OK |
| `checkpoint_best.pth` writing | OK |
| `checkpoint_last.pth` writing | OK |

Mini-run metrics:

| Metric | Value |
|---|---:|
| train_loss | 0.008085 |
| val_loss | 0.446424 |
| train_paper_formula_rmse | 195065.234644 |
| val_paper_formula_rmse | 54750.329296 |
| train_current_relative_rmse | 14.332514 |
| val_current_relative_rmse | 0.957723 |
| val_classical_rmse | 0.668150 |
| val_nse | -0.260333 |
| val_pearson_r | -0.083900 |
| val_csi_gamma_0_001 | 0.667098 |
| val_csi_gamma_0_01 | 0.596751 |

These numbers are not scientifically meaningful because this was a one-batch sanity run. They are useful only to verify the pipeline.

## 7. Memory and Runtime Risk

Model parameter count:

```text
11,061,901 parameters
```

Measured memory:

| Operation | Peak GPU memory |
|---|---:|
| no-grad forward | 3050.964 MB |
| one backward/optimizer step | 6998.401 MB |

Inference:

- Batch size 1 should fit comfortably on the RTX 6000 Ada 48 GB.
- Full training is likely feasible in VRAM.
- The main runtime cost is likely raster I/O plus full-resolution FFTs, not memory capacity.
- Mixed precision is not currently implemented in the official-v0 script.
- Gradient checkpointing is not currently implemented; it is not required for batch size 1 on this GPU, but it could be useful if width/modes increase.
- A crop/downsample debug mode is not currently supported in official-v0. Only `--max-train-batches` and `--max-val-batches` are available.

## 8. Temporal Alignment Verification

Confirmed:

- Input has 20 time steps.
- Initial depth is repeated across all 20 input time positions.
- DEM is repeated across all 20 input time positions.
- X/Y are repeated across all 20 input time positions.
- T is normalized from 0 to 1 over the 20 input positions.
- Rainfall is sampled for each of the 20 water timestamps.
- Target is water depth for frames `t=2..20`.
- Model output has 19 time steps.
- Loss compares output `[t=2..20]` directly with target `[t=2..20]`.

The alignment is internally consistent.

## 9. Normalization/Preprocessing Status

Current official-v0 preprocessing:

| Variable | Status |
|---|---|
| X/Y | normalized to `[0,1]` |
| T | normalized to `[0,1]` |
| water depth | raw |
| DEM | raw, bilinear-resized |
| rainfall | raw, bilinear-resized |
| nodata/nan/inf | replaced with 0 through inherited raster reader |

This matches the internal baseline in using raw depth/DEM/rainfall. It may differ from the official baseline if the paper/code applied normalization.

## 10. Metric Ambiguity Status

The metric ambiguity remains unresolved.

The literal paper-style formula recompute produced very large values because dry or near-zero target cells dominate division by `y^2`:

- internal best checkpoint literal formula: `90.001318594995`;
- mini-run train literal formula: `195065.234644`;
- mini-run val literal formula: `54750.329296`.

This is not compatible with Table 4 values unless the official implementation applies masking, thresholding, a different denominator, or the paper formula is imprecisely written.

## 11. Bugs Found and Fixed

No code bug was found during this readiness audit that required a fix.

Observed warnings:

- Rasterio `NotGeoreferencedWarning`; expected for these TIFF reads and not blocking.
- NumPy/rasterio deprecation warning inherited from raster reading; not blocking.

## 12. Remaining Risks Before Full Training

Blocking engineering issues: none found.

Non-blocking but important scientific risks:

- RMSE formula and dry-cell handling unresolved.
- Raw DEM scale is large relative to raw water depth.
- No normalization currently applied.
- Official architecture may still differ from this 3D FNO parameterization.
- Rainfall alignment may not match official preprocessing.
- No mixed precision support currently.
- No crop/downsample debug mode currently.

## 13. Recommendation

Recommendation: **do not launch the full 100-epoch official FNO+ training yet as the final reproduction run**.

The implementation is technically ready for a controlled pilot. The next safest command is:

```bash
python tools/train_floodcastbench_fno_plus_official.py   --config configs/floodcastbench_fno_plus_official_highfid_60m.yaml   --epochs 5   --num-workers 2
```

If that 5-epoch pilot shows stable loss, reasonable validation metrics, and no memory/runtime issue, then launching the full 100-epoch run is technically justified. For scientific reporting, RMSE/dry-cell handling and normalization should still be resolved before claiming official reproduction.
