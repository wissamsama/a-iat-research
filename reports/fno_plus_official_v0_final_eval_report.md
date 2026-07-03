# FNO+ Official-v0 Final Evaluation Report

This report was generated from existing repository/workspace artifacts plus test-only evaluation of saved checkpoints. No model code, dataset code, training metrics, or checkpoints were modified.

## 1. Selected Primary Run

- Selected primary official-v0 run: `/home/wissam/utem-workspace/experiments/FloodCastBench/27-06-2026_16-28-24_fcb_fno_plus_official_v0_highfid_60m`
- Primary checkpoint directory: `/home/wissam/utem-workspace/checkpoints/FloodCastBench/27-06-2026_16-28-24_fcb_fno_plus_official_v0_highfid_60m`
- Duplicate/parallel candidate inspected: `/home/wissam/utem-workspace/experiments/FloodCastBench/27-06-2026_16-28-12_fcb_fno_plus_official_v0_highfid_60m`
- Reason for selecting primary: both candidate runs completed 100 epochs and have the same scientific metrics at best/final epochs; `16-28-24` is the later run and matches the currently active IDE context. The duplicate is retained as an inspected duplicate, not deleted or altered.

## 2. Duplicate Run Status

| Run | metrics.csv | Epochs logged | Epoch 100 present | summary.json | checkpoint_best | checkpoint_last | Best epoch | Best val_current_relative_RMSE | Final epoch |
|---|---|---:|---|---|---|---|---:|---:|---:|
| `27-06-2026_16-28-12_fcb_fno_plus_official_v0_highfid_60m` | yes | 100 | yes | yes | yes | yes | 99 | 0.008475781122799973 | 100 |
| `27-06-2026_16-28-24_fcb_fno_plus_official_v0_highfid_60m` | yes | 100 | yes | yes | yes | yes | 99 | 0.008475781122799973 | 100 |

The two `metrics.csv` files are not byte-identical, mainly because runtime fields differ, but the inspected scientific metric values at best and final epochs match exactly.

## 3. Best Validation Epoch and Metrics

Best validation epoch is selected by `val_current_relative_rmse`, as stored in `summary.json` and implemented in `tools/train_floodcastbench_fno_plus_official.py`.

| Metric | Epoch 99 best validation | Epoch 100 final validation |
|---|---:|---:|
| `train_loss` | 4.884039399963387e-05 | 4.8761464423195914e-05 |
| `val_loss` | 3.520780878066684e-05 | 3.5257769403480257e-05 |
| `val_paper_formula_rmse` | 37.69482104961669 | 37.951260663074486 |
| `val_current_relative_rmse` | 0.008475781122799973 | 0.008481792592658457 |
| `val_classical_rmse` | 0.0059336169210293735 | 0.005937825354300096 |
| `val_nse` | 0.9999014986001897 | 0.9999013588258905 |
| `val_pearson_r` | 0.9999517755842645 | 0.9999517399309639 |
| `val_csi_gamma_0_001` | 0.7701415684868845 | 0.7706199863310764 |
| `val_csi_gamma_0_01` | 0.9906020657686124 | 0.9905851957408329 |
| `learning_rate` | 2.467198171342e-07 | 0.0 |
| `epoch_time_sec` | 121.72083294999902 | 118.86055011200006 |

## 4. Checkpoint Verification

- `checkpoint_best.pth`: `/home/wissam/utem-workspace/checkpoints/FloodCastBench/27-06-2026_16-28-24_fcb_fno_plus_official_v0_highfid_60m/checkpoint_best.pth`
- Stored checkpoint epoch: `99`
- Verification result: `checkpoint_best.pth` corresponds to the best logged validation epoch by `val_current_relative_rmse`.
- `checkpoint_last.pth`: `/home/wissam/utem-workspace/checkpoints/FloodCastBench/27-06-2026_16-28-24_fcb_fno_plus_official_v0_highfid_60m/checkpoint_last.pth`
- Stored checkpoint epoch: `100`

## 5. Test Evaluation Commands Used

Checkpoint best evaluation command:

```bash
python tools/evaluate_floodcastbench_fno_plus_official.py --run-dir /home/wissam/utem-workspace/experiments/FloodCastBench/27-06-2026_16-28-24_fcb_fno_plus_official_v0_highfid_60m --checkpoint /home/wissam/utem-workspace/checkpoints/FloodCastBench/27-06-2026_16-28-24_fcb_fno_plus_official_v0_highfid_60m/checkpoint_best.pth --split test --output /home/wissam/utem-workspace/experiments/FloodCastBench/27-06-2026_16-28-24_fcb_fno_plus_official_v0_highfid_60m/test_metrics_checkpoint_best_official_v0.json --device auto
```

Checkpoint last evaluation command:

```bash
python tools/evaluate_floodcastbench_fno_plus_official.py --run-dir /home/wissam/utem-workspace/experiments/FloodCastBench/27-06-2026_16-28-24_fcb_fno_plus_official_v0_highfid_60m --checkpoint /home/wissam/utem-workspace/checkpoints/FloodCastBench/27-06-2026_16-28-24_fcb_fno_plus_official_v0_highfid_60m/checkpoint_last.pth --split test --output /home/wissam/utem-workspace/experiments/FloodCastBench/27-06-2026_16-28-24_fcb_fno_plus_official_v0_highfid_60m/test_metrics_checkpoint_last_official_v0.json --device auto
```

- Generated checkpoint-best test metrics JSON: `/home/wissam/utem-workspace/experiments/FloodCastBench/27-06-2026_16-28-24_fcb_fno_plus_official_v0_highfid_60m/test_metrics_checkpoint_best_official_v0.json`
- Generated checkpoint-last test metrics JSON: `/home/wissam/utem-workspace/experiments/FloodCastBench/27-06-2026_16-28-24_fcb_fno_plus_official_v0_highfid_60m/test_metrics_checkpoint_last_official_v0.json`

## 6. Test Metrics

| Checkpoint | Split | Epoch | paper_formula_rmse | current_relative_rmse | classical_rmse | NSE | Pearson r | CSI 0.001 | CSI 0.01 | Samples |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| official-v0 checkpoint_best | test | 99 | 44.4652303794 | 0.00813504058176 | 0.00572386247091 | 0.999909508751 | 0.999956454034 | 0.761073770525 | 0.99094432452 | 14 |
| official-v0 checkpoint_last | test | 100 | 44.5769283353 | 0.00814562900961 | 0.00573131255111 | 0.999909273034 | 0.999956339055 | 0.76160836405 | 0.990929952339 | 14 |

## 7. Comparison Table

| Source | Split | Epoch | RMSE / relative RMSE label | RMSE / relative RMSE | Classical RMSE | NSE | Pearson r | CSI 0.001 | CSI 0.01 | Comparability comment |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Official FloodCastBench FNO+ Table 4 | test/reference | not specified | paper RMSE | 0.003941 | not available | 0.999979 | 0.99999 | 0.939638 | 0.984588 | Reference values; metric equivalence with local current_relative_rmse is not proven here. |
| Previous internal 2D FNO+ | test | 95 checkpoint_best | relative_rmse | 0.0123582983377 | not available | 0.999791164424 | 0.999898656296 | 0.724176463561 | 0.985721042228 | Completed internal 2D spatial FNO+; not identical architecture to official-v0 3D. |
| Official-v0 3D FNO+ best validation | validation | 99 | val_current_relative_rmse | 0.0084757811228 | 0.00593361692103 | 0.9999014986 | 0.999951775584 | 0.770141568487 | 0.990602065769 | Validation, not test; useful for checkpoint selection only. |
| Official-v0 3D FNO+ checkpoint_best | test | 99 | current_relative_rmse | 0.00813504058176 | 0.00572386247091 | 0.999909508751 | 0.999956454034 | 0.761073770525 | 0.99094432452 | Primary test result for selected checkpoint_best. |
| Official-v0 3D FNO+ checkpoint_last | test | 100 | current_relative_rmse | 0.00814562900961 | 0.00573131255111 | 0.999909273034 | 0.999956339055 | 0.76160836405 | 0.990929952339 | Optional final-epoch test result; slightly worse RMSE than checkpoint_best. |

## 8. Interpretation

### Confirmed

- The selected official-v0 run completed 100 epochs.
- The best validation epoch is epoch 99 by `val_current_relative_rmse`.
- `checkpoint_best.pth` stores epoch 99 and matches the best validation epoch.
- Test-only evaluation of `checkpoint_best.pth` completed on 14 test samples.
- Test-only evaluation of `checkpoint_last.pth` also completed on 14 test samples.
- Official-v0 3D checkpoint_best improves over the previous internal 2D FNO+ baseline on test current/relative RMSE, NSE, Pearson r, CSI@0.001, and CSI@0.01.

### Provisional

- The official-v0 result is closer to the official FloodCastBench FNO+ reference than the old 2D baseline on several metrics, especially `current_relative_rmse` and CSI@0.001.
- Metric equivalence between local `current_relative_rmse` and the official paper RMSE is not proven from code alone.
- `paper_formula_rmse` is not used as the main comparison metric here because the implemented formula divides by per-pixel `target^2 + eps` and is highly sensitive to dry or near-zero pixels.

### Cannot Be Claimed

- This cannot yet be claimed as a successful official reproduction of Table 4.
- The local official-v0 result does not match the official reference on RMSE/current_relative_RMSE or CSI@0.001.
- The official paper RMSE cannot be treated as identical to local `current_relative_rmse` unless the metric convention is verified.

### Remaining Gaps

- Official reference RMSE: `0.003941`; local official-v0 checkpoint_best test `current_relative_rmse`: `0.008135040581762574`, and local classical RMSE: `0.0057238624709056735`.
- Official reference CSI@0.001: `0.939638`; local official-v0 checkpoint_best test CSI@0.001: `0.7610737705248405`.
- Official reference CSI@0.01: `0.984588`; local official-v0 checkpoint_best test CSI@0.01: `0.990944324520436`, which is higher than the reference, but that does not offset the RMSE and CSI@0.001 gaps.

## 9. Next Recommended Step

The next step should be a targeted diagnostic of `checkpoint_best` test predictions focused on shallow-water pixels and CSI@0.001 errors, using non-overwriting outputs. This directly addresses the largest remaining discrepancy without relaunching training.
