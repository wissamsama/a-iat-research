# DIFF-SPARSE v1 paper-fidelity audit

Created: 2026-07-08

## Objective

This audit reframes the v1 goal correctly:

- Primary goal: make the FloodCastBench v1 implementation as close as practical to the official DIFF-SPARSE architecture.
- Non-goal: improve scores by adding shortcuts, tuning for FloodCastBench, or claiming superiority over persistence/FNO+.
- Scientific status: architecture-faithful FloodCastBench adaptation candidate, not a full official TideWatch reproduction.

The negative metric results from the 3-seed rewrite queue should therefore be read as performance diagnostics, not as proof that the architecture-matching work was pointless. They show that this FloodCastBench adaptation performs poorly under the current protocol; they do not by themselves invalidate the architectural fidelity work.

## Sources inspected

Official DIFF-SPARSE reference:

- `https://github.com/KAI10/Diff-Sparse`
- `training_config.py`
- `hidden_state_net.py`
- `diffusion.py`
- `lightning_training.py`
- `lightning_diffusion.py`
- `TidewatchDataset.py`
- `TidewatchPatchDataset.py`
- `generate_test_masks.py`

Local implementation:

- `models/diff_sparse_v1.py`
- `datasets/floodcastbench_diff_sparse_v1_dataset.py`
- `tools/train_floodcastbench_diff_sparse_v1.py`
- `tools/evaluate_floodcastbench_diff_sparse_v1.py`
- `configs/floodcastbench_diff_sparse_v1_highfid_60m.yaml`
- `reports/diff_sparse_v1_v2_technical_spec.md`
- `reports/diff_sparse_v1_rewrite_full_long_analysis.md`

## Executive verdict

The current v1 rewrite is substantially closer to the official DIFF-SPARSE architecture than the earlier local dense sanity prototype. The core model path now matches the official reference on the most important architectural mechanisms:

- `HiddenStateNet`-style temporal-token encoder.
- `diffusers.UNet2DConditionModel`.
- Cross-attention at the two middle UNet levels.
- x0-prediction diffusion objective.
- Linear DDPM beta schedule with `beta_end=1.0`.
- Reverse-process noise variance using raw `beta_t`.
- Gaussian sampling start.
- Sparse mask channel plus noise replacement for missing observations.
- Autoregressive rollout without re-masking generated predictions.

However, it is still not a strict official reproduction. The remaining differences are mostly caused by adapting TideWatch DIFF-SPARSE to FloodCastBench, plus a few protocol choices introduced for engineering stability.

## Fidelity table

| Component | Official DIFF-SPARSE | Local v1 implementation | Fidelity | Notes |
|---|---|---|---|---|
| Problem type | Conditional diffusion from sparse inundation observations | Conditional diffusion from masked FloodCastBench water-depth history | High adaptation | Same modeling idea, different dataset/domain. |
| Dataset domain | TideWatch Virginia coastal inundation | FloodCastBench Australia high-fidelity 60m | Required deviation | This prevents calling the result a full official reproduction. |
| Patch size | `patch_size=64` | `patch_size=64` | Exact | Config matches the paper/reference D=64 setup. |
| Train batch size | `train_batch_size=32` | `batch_size=32` | Exact | Matches reference Table/config. |
| Context length | `context_length=12` | `context_length=12` | Exact | Local rollout labels are h13:h24 for 12 future frames. |
| Training horizon | `training_horizon_length=1` | training target is `target[:, 0:1]` | Exact | One-step x0 training objective matches reference. |
| Val/test horizon | `validation_horizon_length=12`, `test_horizon_length=12` | `prediction_length=12` | Exact | Current rewrite evaluates 12-step rollout. |
| Water normalization | Shared train inundation mean/std for context and target | Shared train water mean/std for context and target | High | Local adds separate rainfall stats because FloodCastBench has rainfall. |
| Elevation normalization | Separate train elevation stats | Separate DEM stats | High | Same idea, different data source. |
| Covariate normalization/features | Reference uses 4 temporal covariates after processing: day/month sin/cos and hour sin/cos | Local uses timestamp scale plus 3 hour-frequency sin/cos pairs, 7 features | Medium deviation | This is a real conditioning difference. If strict architecture/protocol fidelity is desired, this should be aligned or explicitly justified. |
| Rainfall forcing | No raster rainfall channel in HiddenStateNet input | Adds dense rainfall as context channel | Required/domain deviation | FloodCastBench is rain-driven. This changes input channels from 3 to 4, but keeps the same encoder architecture pattern. |
| Context encoder | `HiddenStateNet`: 3 unpadded Conv3d blocks, kernel `(1,3,3)`, AvgPool3d `(1,2,2)`, channels 16/32/64, output conv, concat covariate, linear projection | `TemporalContextEncoder`: same unpadded Conv3d/AvgPool3d stack, channels 16/32/64, output conv, concat covariates, linear projection | Very high | Central architecture match. Input channel count differs because rainfall is included. |
| Token semantics | One heavily pooled token per context timestep | One heavily pooled token per context timestep | Exact | Important correction vs earlier spatial-map conditioning ideas. |
| UNet backbone | `UNet2DConditionModel`, in/out channels 1, block channels 16/32/32/64, layers per block 2, norm groups 16, dropout 0 | Same | Exact | Central architecture match. |
| Cross-attention placement | Down: Down, CrossAttn, CrossAttn, Down; Up mirror | Same generated from `cross_attention_blocks=2` | Exact | Matches two middle levels, not deepest-only or spatial concat. |
| Cross-attention type | diffusers default, not `only_cross_attention=True` | diffusers default | Exact | Self-attention plus cross-attention inside the relevant blocks. |
| Diffusion parameterization | x0 prediction | x0 prediction | Exact | Training loss is MSE between predicted clean field and target. |
| Diffusion steps | 20 | 20 | Exact | Matches reference. |
| Beta schedule | Linear, `min_beta=1e-4`, `max_beta=1.0` | Linear, `beta_start=1e-4`, `beta_end=1.0` | Exact | Fixes the earlier local `0.02` schedule issue. |
| Forward noising | `sqrt(alpha_bar) * x0 + sqrt(1-alpha_bar) * eta` | Same | Exact | Implemented in `q_sample`. |
| Reverse mean | x0-parameterized DDPM posterior mean | Same coefficient form | Exact | Matches reference formula. |
| Reverse variance | Raw `beta_t` option active in reference | Raw `beta_t` | Exact | Not beta-tilde. |
| Sampling start | Gaussian noise | Gaussian noise | Exact | Matches terminal SNR zero schedule. |
| Missing observation fill | Missing cells replaced by Gaussian noise; zero fill is ablation | `mask_mode=noise`, zero optional | High/exact behavior | Local dense m0 naturally produces all-ones masks. |
| Sensor mask channel | Added as channel in context | Added as channel in context | High | Static per sample, broadcast through context length. |
| Eval masks | 10 fixed masks loaded from `.pt` files | 10 deterministic generated masks in a bank | Close but not exact | Same fixed-mask idea; storage/generation path differs. |
| Autoregressive rollout | Generate one step, append prediction plus static additional channels, repeat | Same when `rollout_remask=false` | High | Current config resolved earlier ambiguity by disabling remask. |
| Rollout tiling | Official uses patch dataset over TideWatch patch origins | Local full 536x536 field tiled with overlap/blending | Required deviation | Needed for FloodCastBench full-field maps; not official TideWatch protocol. |
| Optimizer | Adam | AdamW with `weight_decay=0.0` | Small protocol deviation | With zero weight decay this is close, but not textually identical. |
| Epochs | 40 | 300 | Protocol deviation | This was chosen for optimization diagnostics, not paper fidelity. |
| LR scheduler | ReduceLROnPlateau factor 0.5, patience 3, monitor `val_masked_nrmse` | ReduceLROnPlateau factor 0.5, patience 15, monitor `val_loss` | Important protocol deviation | Not core architecture, but important if we call a run paper-like. |
| Validation metric used for checkpointing | Full rollout masked NRMSE | One-step denoising `val_loss` | Important protocol deviation | Cheaper and stable locally, but less faithful. |
| Grad clipping | Not in reference algorithm/config | `grad_clip_norm=1.0` | Protocol deviation | Stability guard, not paper-faithful. |
| Scenario counts | val 2, test 8 | val 2, test 8 | Exact | Matches reference config. |
| Scientific claims | Official TideWatch result | Local FloodCastBench adaptation only | Correct if labeled | Must not claim official reproduction or superiority. |

## What is now architecturally close

The most important correction is that v1 no longer treats conditioning as a pixel-aligned 2D map or a concat shortcut. It now follows the reference pattern:

1. Build sparse/masked historical context.
2. Encode the context sequence through a small 3D CNN that preserves the time axis.
3. Flatten the pooled spatial summary into one token per context timestep.
4. Concatenate temporal covariates to each token.
5. Feed the token sequence to `UNet2DConditionModel` as `encoder_hidden_states`.
6. Predict x0 from noisy target plus cross-attended context.

That is the central DIFF-SPARSE architectural idea. For the user's stated objective, this is the part that matters most.

## Remaining deviations ranked by importance

### A. Domain adaptation deviations that are hard to avoid

1. TideWatch versus FloodCastBench data.
2. Full-field 536x536 FloodCastBench evaluation via tiling instead of fixed TideWatch patch-origin evaluation.
3. Rainfall forcing as a raster channel. This is not in the official HiddenStateNet input, but FloodCastBench is a rainfall-driven benchmark.

These should be described as FloodCastBench adaptation choices, not as improvements.

### B. Architecture/conditioning deviations worth deciding explicitly

1. Covariate dimension and meaning differ: official processed covariates are 4 time features; local v1 currently uses 7 timestamp-derived features.
2. Eval mask bank is deterministic and fixed, but generated locally instead of loaded from precomputed `.pt` files.
3. Rainfall channel changes the first Conv3d input channel count from the official 3-channel setup to 4.

If the goal is maximum paper-architecture purity, the cleanest next question is whether rainfall should stay in v1 or move to a later FloodCastBench-specific v1-rain variant.

### C. Training/evaluation protocol deviations, not architecture

1. 300 epochs instead of 40.
2. LR patience 15 instead of 3.
3. `val_loss` one-step checkpoint selection instead of full rollout `val_masked_nrmse`.
4. Gradient clipping.
5. AdamW instead of Adam, although `weight_decay=0.0`.

These can strongly change results, but they are not the core model architecture.

## Performance result, interpreted correctly

The completed 3-seed rewrite queue is a negative performance result:

- Dense m0, sparse m50, and sparse m95 remain worse than persistence under the current h13:h24 rollout evaluation.
- The rewrite is also worse than the earlier local protocol-161 result on horizon-matched h20.
- This does not prove the official DIFF-SPARSE architecture is bad. It means this FloodCastBench adaptation/protocol did not produce useful forecast skill yet.

For the v1 objective stated here, the main success is architectural traceability. The performance result should be reported honestly as a limitation.

## Recommended wording

Use:

- "DIFF-SPARSE v1 FloodCastBench adaptation"
- "architecture-faithful local adaptation of the official DIFF-SPARSE reference"
- "reference-style temporal-token conditioning with `UNet2DConditionModel`"
- "negative performance result under current FloodCastBench protocol"
- "not optimized for FloodCastBench performance"

Avoid:

- "official DIFF-SPARSE reproduction"
- "state of the art"
- "beats FNO+"
- "validates sparse-sensor robustness"
- "validated uncertainty calibration"

## If we want an even more paper-faithful v1

These are the safest next engineering changes if the goal is fidelity, not score:

1. Create a separate `paper_fidelity` config rather than overwriting the long-run config.
2. Set training epochs to 40.
3. Set LR patience to 3.
4. Remove grad clipping or mark it off by default.
5. Use Adam instead of AdamW.
6. Add an optional validation path that checkpoints on rollout `val_masked_nrmse`, matching the reference monitor.
7. Decide whether v1 should exclude rainfall to preserve the exact 3-channel HiddenStateNet input. If rainfall stays, label it as a FloodCastBench adaptation.
8. Align temporal covariates to 4 features if strict reference matching is preferred.
9. Save/reuse eval mask banks as artifacts, mirroring the official `.pt` mask files.

Do not launch another 3-seed full queue until the definition of "paper-faithful v1" is frozen.

## Bottom line

The current v1 is the right direction for the user's corrected objective. It is much closer to the official DIFF-SPARSE architecture than the earlier local dense prototype. Its weak metrics should not push us toward ad hoc performance hacks if v1 is meant to be an architectural reproduction-style baseline.

The clean next step is not to tune. It is to freeze a paper-fidelity configuration and explicitly separate:

- `v1_paper_fidelity`: maximum architectural/protocol closeness.
- `v1_floodcastbench_adaptation`: same architecture, but with rainfall and pragmatic training choices.
- later `v2`: performance-oriented FloodCastBench improvements.
