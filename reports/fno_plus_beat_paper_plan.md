# Paper 2 Master Plan — "Beating FNO+" (dense, no sparsity)

Status: DRAFT v1 (2026-07-11). This is a **separate, independent paper** from
`reports/paper_master_plan.md` (the DIFF-SPARSE sparsity paper). Do not merge
the two. Any future decision, result, or scope change for *this* paper must be
written into *this* file — same governance discipline as the sparsity plan
(see §8).

## 0. Relationship to Paper 1

- Paper 1 (DIFF-SPARSE) keeps its own scope: sparsity study on FloodCastBench,
  V1 vs V2 vs deterministic twin, calibration. FNO+ appears there only as one
  more baseline point of reference, dense-only, already trained.
- Paper 2 (this file) **drops sparsity entirely**. Scope is 100% dense
  (`missing_rate=0.0` only, full-field observation), single question:

  > **Can FNO+'s own reported result on FloodCastBench (Australia, high-fidelity
  > 60m) be improved — either by tuning FNO+ itself, or by adding a temporal
  > Mamba block to it — beyond what this repo's current reproduction achieves?**

- No connection to diffusion, generative sampling, or ensembles. This is a
  plain supervised-regression improvement study.
- Rule reuse from Paper 1's rule set (`paper_master_plan.md` §8), where
  applicable to this scope:
  - **R1**: no single-seed claims for a final headline number — always ≥3
    seeds before calling a result established (already the norm here:
    `fno_plus_multiseed_results.md` used exactly 3 seeds for the vanilla
    baseline).
  - **R9 analogue**: any comparison must use the same evaluation protocol
    (pooled t=2..20, physical units after inverse-transform, `--split test`)
    on both sides. Never compare a training-time selection metric
    (`val_current_relative_rmse`) across model variants as a final number —
    only the final pooled test-split metric counts.

## 1. Verified starting point (code review, done 2026-07-11)

This section exists so future work does not re-litigate settled facts or
accidentally re-run experiments already completed. Everything below was
verified by reading code/configs/reports directly, not assumed.

### 1.1 Architecture — `models/fno_plus_official.py`

- `FNOPlusOfficial3d`: genuine 3D space-time FNO. `SpectralConv3d` does
  `torch.fft.rfftn`/`irfftn` over `(H, W, T)` with 4 separate complex weight
  tensors for the 4 sign-quadrant combinations of the truncated Fourier
  modes. 4 fourier+pointwise residual blocks (GELU), then `proj1`/`proj2`
  (Conv3d 1x1) to a single output channel.
- Single-shot (non-autoregressive): input `[B, 6, H, W, 20]` → output
  `[B, 1, H, W, 19]` (t=2..20) in one forward pass. No padding before the FFT.
  Time rFFT modes truncate to 11 effective (T=20 → T//2+1=11) even though
  `modes=12` is requested.
- Confirmed **conform to the paper's stated spec** point-by-point
  (`reports/fno_plus_dashboard_audit.md`): input channels, 4 layers, 12
  modes, width 20, Adam β=(0.9,0.999), weight_decay 1e-4, cosine LR
  0.001→0, 100 epochs, batch size 1, Australia-60m 116/14/14 non-overlapping
  20-frame-window splits. **This conformance table is the reason blind
  further hyperparameter matching against the paper is not the right next
  move — the obvious levers are already paper-conformant.** Any hyperparameter
  change explored in this paper is therefore explicitly a *departure* from
  the paper's protocol, done in the spirit of "can we do better than the
  paper's own choices," not "get closer to reproducing them."

### 1.2 Metric definition — RESOLVED, not an open question

`reports/fno_plus_dashboard_audit.md` proved algebraically (via the
NSE/relRMSE/TotVar identity) that Table 4's "RMSE" is exactly this repo's
pooled `current_relative_rmse = sqrt(SSE / sum(y^2))`, reproducing all 3
published (RMSE, NSE) pairs for U-Net/FNO/FNO+ to 2e-7. **Do not re-open the
metric-definition question in this paper** — it is settled. Use
`current_relative_rmse` (pooled test-split) as the primary headline metric,
exactly as Paper 1 does.

### 1.3 Vanilla FNO+ reproduction — current best, 3-seed

Source: `reports/fno_plus_multiseed_results.md`, config
`experiments/FloodCastBench/28-06-2026_15-59-18_.../config.yaml`.

| Seed | relRMSE | NSE | Pearson r | CSI@0.001 | CSI@0.01 |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.006694 | 0.999939 | 0.999971 | 0.909009 | 0.993807 |
| 7 | 0.006426 | 0.999944 | 0.999972 | 0.810893 | 0.996355 |
| 123 | 0.006530 | 0.999942 | 0.999972 | 0.887944 | 0.995453 |
| **mean±std** | **0.006550±0.000135** | 0.999941±0.000002 | 0.999972±0.000000 | 0.869±0.052 | 0.995±0.001 |

Official Table 4 FNO+: relRMSE 0.003941, NSE 0.999979, Pearson r 0.999990,
CSI@0.001 0.939638, CSI@0.01 0.984588. Gap ×1.66-1.70 on relRMSE, confirmed
**not** explained by seed noise (std is ~5% of the gap).

**This 3-seed mean (0.006550) is the reference baseline this paper must beat.**
Not the official 0.003941 (unreachable without unpublished
preprocessing/training details — see §1.4) and not any single seed.

Known already-tried, already-diagnosed levers on the vanilla-reproduction gap
(`fno_plus_official_v0_next_gap_decision_report.md`,
`fno_plus_dashboard_audit.md`) — **do not re-run these as if new**:

1. **Output non-negativity / normalization of physical channels**: already
   the difference between "v0" (raw physical channels, relRMSE 0.008135,
   CSI@0.001 0.724) and "v1" (train-set standardized h1/DEM/rain/target,
   inverse-transformed before metrics, relRMSE 0.006550 mean, CSI@0.001
   0.869 mean) — **already adopted**, this repo's "vanilla FNO+" baseline
   *is* v1, already includes this fix.
2. **Wet/dry post-threshold**: diagnostic only (moves CSI@0.001 from 0.724
   to 0.936 on v0 raw), explicitly not adopted as an official number because
   it's a labelled post-processing sensitivity check, not a trained
   improvement. Available as an optional evaluation-time diagnostic for this
   paper too, but any reported headline number must be the raw model output.
3. **Explicit valid-domain mask**: audited, none found in the local official
   data-generation clone; not applied. Open, unconfirmed as a paper detail.
4. **Canonical FNO spatial/temporal padding before FFT**: proposed but never
   run (`next_gap_decision_report.md` Experiment 3). **This is a real
   untried lever**, candidate for WPB1 below.
5. **Rainfall temporal alignment**: audited and ruled out — local Australia
   rainfall files are genuinely 1800s-resolution, matches official
   data-generation code (`current_time / 1800`). Not a gap; do not revisit.

### 1.4 Residual unexplained gap — genuinely open

Per `fno_plus_dashboard_audit.md` §"Honest current standing": relRMSE ×1.70
worse, NSE worse, Pearson r worse, CSI@0.001 worse, **CSI@0.01 actually
better** than official. Residual unexplained factors: exact official
preprocessing (not published in detail), CSI aggregation granularity,
official seed(s)/seed count. This paper does not aim to close this gap to
official numbers (unreachable without unpublished details) — it aims to beat
**this repo's own 0.006550 baseline**, which is the honest, controllable
target.

### 1.5 FNO+ + Mamba — code review

`models/fno_plus_official_mamba.py` (`FNOPlusOfficial3dMamba`,
`TemporalMambaResidual`): reuses `SpectralConv3d` from the vanilla model
unchanged. Design: full 4-layer Fourier backbone first (identical to
vanilla), then `mamba_layers` (default 1) `TemporalMambaResidual` blocks
applied on the *latent* `[B, width=20, H, W, T]` tensor — each block reshapes
to `[B*H*W, T, C]` (Mamba treats every spatial pixel as an independent length-20
sequence over the latent channel dimension), LayerNorm → `Mamba(d_model=20,
d_state=16, d_conv=4, expand=2)` → residual add, reshape back — then the same
`proj1`/`proj2` head as vanilla. `mamba_ssm` package confirmed installed and
importable.

**Existing single-seed result** (seed 42, 100 epochs, otherwise identical
protocol/config to vanilla, `29-06-2026_16-04-36_fcb_fno_plus_official_v1_mamba_latent_highfid_60m`,
best checkpoint epoch 87):

| Metric | Vanilla seed 42 | Mamba (1 layer) seed 42 | Delta |
|---|---:|---:|---:|
| relRMSE | 0.006694 | **0.009464** | **+41% worse** |
| NSE | 0.999939 | 0.999878 | worse |
| Pearson r | 0.999971 | 0.999940 | worse |
| CSI@0.001 | 0.909009 | 0.879792 | worse |
| CSI@0.01 | 0.993807 | 0.994427 | ~tie, slightly better |
| negative_prediction_ratio | not recorded | 0.196 | 19.6% of predicted pixels negative |

**This is the single most important fact for this paper's design.** The
existing, already-trained, naive Mamba insertion does not help — it clearly
hurts on the primary metric, on the one seed tested. This is not proof Mamba
*can't* help (1 seed, 1 placement, 1 hyperparameter set, no tuning), but it
means:
- The paper cannot start from "let's confirm Mamba helps" as a premise. It
  must start from "does *any* Mamba integration variant help, given that the
  first naive attempt clearly did not."
- `negative_prediction_ratio=0.196` on the Mamba run vs vanilla (not
  recorded, but v1 vanilla's CSI@0.001 of 0.909 implies far fewer
  spurious-positive dry cells) suggests the Mamba block may be destabilizing
  small-magnitude/dry-region predictions specifically — worth checking first
  before any larger sweep (cheap, single-run diagnostic, WPB3 below).

### 1.6 Compute budget context

Each 100-epoch training run (vanilla or Mamba) takes on the order of hours on
the P7 GPU (RTX 6000 Ada) — exact wall-clock not recorded in this review pass,
should be logged going forward per run in the results ledger (§3). Batch
size 1 (paper-conformant) means each run is comparatively cheap and many
short ablation runs are affordable; this is the basis for WPB1 (hyperparameter
sweep) being proposed as the first, cheapest set of experiments.

## 2. Paper positioning

**Type**: applied methods paper, not a sparsity/UQ paper. Narrower and more
conventional than Paper 1: "we take a recent operator-learning baseline on a
real benchmark, push its own hyperparameters and optionally augment it with a
sequence model, and report what does and doesn't help, honestly including
negative results (§1.5)."

**Explicitly NOT claimed**:
- Not a reproduction of the official paper's exact 0.003941 (see §1.4).
- Not a sparsity/missing-data study (that's Paper 1).
- Not a claim that Mamba helps FNO+ in general — only what is measured here.

**What would make this a solid, publishable result** (either branch is fine,
per Paper 1's precedent of accepting honest negative results):
- (a) A tuned/boosted vanilla FNO+ config that beats 0.006550 mean by a
  margin outside 3-seed noise (~0.000135 std) → a useful applied contribution
  even without Mamba.
- (b) A Mamba variant (corrected placement/hyperparameters/training) that
  beats the best vanilla config (tuned or not) outside seed noise → validates
  the "intelligently-integrated Mamba helps FNO+" thesis.
- (c) Neither works outside noise → still reportable as "we systematically
  tried N variants of tuning + Q Mamba placements, none improved on vanilla
  FNO+ beyond seed noise, here's why we think that is" — same honesty
  standard as Paper 1's WP1 negative result, still useful for the field
  (saves future authors from re-trying the same naive ideas: §1.5's finding
  already does part of this).

## 3. Results ledger (append rows as each run completes; keep in sync with WP tables below)

| WP | Variant | Seed | relRMSE | NSE | Pearson r | CSI@0.001 | CSI@0.01 | Run dir | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| — | vanilla v1 baseline | 42 | 0.006694 | 0.999939 | 0.999971 | 0.909009 | 0.993807 | `28-06-2026_15-59-18_...` | done (Paper 1 era) |
| — | vanilla v1 baseline | 7 | 0.006426 | 0.999944 | 0.999972 | 0.810893 | 0.996355 | `03-07-2026_17-43-06_...` | done |
| — | vanilla v1 baseline | 123 | 0.006530 | 0.999942 | 0.999972 | 0.887944 | 0.995453 | `03-07-2026_22-38-40_...` | done |
| — | mamba (1 layer, naive) | 42 | 0.009464 | 0.999878 | 0.999940 | 0.879792 | 0.994427 | `29-06-2026_16-04-36_...` | done, worse than vanilla |
| WPB0 | context=24 (matched to V2), seed 42 | 42 | 0.007822 | 0.999916 | 0.999960 | 0.896730 | 0.992231 | `12-07-2026_16-00-14_..._context24_...` | done |
| WPB0 | context=24 (matched to V2), seed 7 | 7 | 0.007591 | 0.999921 | 0.999962 | 0.875093 | 0.995100 | `13-07-2026_04-04-20_..._context24_seed7_...` | done |
| WPB0 | context=24 (matched to V2), seed 123 | 123 | 0.007175 | 0.999930 | 0.999968 | 0.888203 | 0.994551 | `15-07-2026_17-08-00_..._context24_seed123_...` | done |
| WPB0 | context=24 (matched to V2), 3-seed mean | 42/7/123 | **0.007529±0.000328** | 0.999922±0.000007 | 0.999963±0.000004 | 0.886676±0.010899 | 0.993961±0.001523 | — | **CONFIRMED at 3/3 seeds (R1) — worse than vanilla, not better** |
| WPB3 | mamba diagnostic (dry-region check) | 42 | | | | | | | pending |
| WPB1 | batch size sweep | 42 | | | | | | | pending |
| WPB1 | width/modes/layers sweep | 42 | | | | | | | pending |
| WPB1 | LR/schedule sweep | 42 | | | | | | | pending |
| WPB1 | padding ablation | 42 | | | | | | | pending |
| WPB2 | best-vanilla-config, 3 seeds | 42/7/123 | | | | | | | pending |
| WPB4 | mamba placement/size sweep | 42 | | | | | | | pending |
| WPB5 | best-mamba-config, 3 seeds | 42/7/123 | | | | | | | pending |

## 4. Work Packages

### WPB0 — Context-matched FNO+ (priority 1, serves both papers)

Rationale, discovered 2026-07-12 while trying to compare V2 (Paper 1) against
FNO+ fairly: FNO+'s current input is NOT single-frame-context by accident of
data availability, it's structural. `datasets/floodcastbench_fno_plus_official_dataset.py`
lines 79-90: the one observed depth frame (`water[0]`) is read once, then
`.unsqueeze(-1).expand(H, W, sample_length)` — **broadcast identically across
all 20 positions of the T axis**. FNO+ genuinely sees only 1 real historical
observation, no matter what its `[..., 20]` shape suggests. V2 (Paper 1) uses
`context_length: 24` — 24 real historical frames (2h) before predicting
anything. **Any "V2 beats FNO+" or "FNO+ beats V2" claim on the current setup
is confounded by a 24x information-budget mismatch, independent of
architecture.** This WP removes that confound and, as a side effect, is
itself a legitimate boost attempt for FNO+ (more real input info can only
help or be neutral, modulo optimization difficulty) — hence it fits both
Paper 1 (fair V2-vs-FNO+ reference point) and Paper 2 (this file, dense-only
"boost FNO+" scope) and is recorded here as the canonical version.

**Design**: extend the FNO+ input window from 20 frames (1 context + 19
target) to `K + 20` frames (K history + 1 "current" + 19 target), K=24 to
match V2 exactly. Replace the single broadcast `initial_depth` channel with a
genuine per-timestep depth channel: real observed depths at the first `K+1`
positions, target positions left as in the current design (predicted, not
fed back). `output_steps` stays 19 (unchanged head). `SpectralConv3d`'s time
FFT dimension grows from T=20 (11 effective rFFT modes after truncation) to
T=44 (23 effective rFFT modes) — `modes=12` now fits without truncation,
itself a minor incidental side-benefit worth noting in the writeup.
DEM/rainfall/X/Y/T channels extend the same way (rainfall needs K more real
per-timestep values, already supported by the existing timestamp-indexed
rainfall loader; DEM/X/Y are constant per pixel, trivial to extend).

**New files needed**: a context-extended dataset variant (subclass or
parallel to `FloodCastBenchFNOPlusOfficialDataset`, e.g.
`FloodCastBenchFNOPlusOfficialContextDataset`, `context_length` config
option), reusing `FNOPlusOfficial3d` unchanged (only `modes`/T-shape differ,
no architecture code change) — same conventions as the vanilla v1 training
tool otherwise (train-set normalization, same optimizer/schedule/seed
protocol, same eval tool with the pooled test-protocol convention).

**Runs**: seed 42 first (single run, ~100 epochs, same hyperparameters as
the vanilla baseline otherwise) to confirm the pipeline is correct and get a
first read. If it beats the vanilla 3-seed baseline (0.006550) by more than
one baseline-seed-std (0.006559 bar, same criterion as WPB1), confirm with
seeds 7/123 (R1). If it makes things worse or errors out, that's itself a
useful, reportable negative result (context alone doesn't help / is hard to
optimize with batch_size=1) — record either way, do not silently drop.

**Compute cost estimate (measured 2026-07-12, not guessed)**: read
`epoch_time_sec` directly from 3 already-completed 100-epoch runs' `metrics.csv`
on the P7 (RTX 6000 Ada):

| Run | Mean epoch time | Total (100 epochs) | Note |
|---|---:|---:|---|
| vanilla seed42 (`28-06-2026_15-59-18_...`) | 53.6s | 89 min | clean, uncontended |
| vanilla seed7 (`03-07-2026_17-43-06_...`) | 181.3s | 302 min | ran alongside other GPU jobs, ~3.4x slower — NOT a clean baseline |
| Mamba (1 layer) seed42 (`29-06-2026_16-04-36_...`) | 104.8s | 175 min | heavier model, clean run, ~2x vanilla |

Best-case (uncontended P7) vanilla is **~90 min/100-epoch run**. WPB0's T
roughly doubles (20→44); Conv3d/pointwise cost scales close to linearly with
T, FFT cost grows slightly faster (`T log T`) — a **~2-2.5x epoch-time
increase is the reasonable estimate**, i.e. **~3-3.5h/run on P7,
uncontended**. Contention (another job sharing the GPU, as happened to
seed7 above) can push this to 5h+; do not schedule this alongside another
GPU job on the same machine.

**On running this on the Dell**: **not recommended for the first
(seed 42, validate-the-pipeline) run, or as the primary venue for the 3-seed
confirmation.** Two independent reasons:
1. **Violates the established P7/Dell division of labor**
   (`experiments/FloodCastBench/coordination/status.md`,
   2026-07-10 rule): Dell gets short, bounded, ideally eval-only tasks so it
   is never blocking for multiple days; P7 keeps trainings. A ~3.5h+ training
   (worse if contended) is not a "small task" by that rule's own standard.
2. **Dell's GPU (A4000) is meaningfully slower than P7's (RTX 6000 Ada)** —
   no direct head-to-head benchmark exists yet in this repo for this exact
   workload, but the hardware gap (A4000 ~19 TFLOPS FP32 vs RTX 6000 Ada ~91
   TFLOPS FP32, roughly 4-5x) suggests a single WPB0 run could take
   **~14-18h on the Dell in the best case**, and a 3-seed confirmation run
   sequentially would push into the exact "2-3 days blocking" scenario
   explicitly ruled out earlier for the Dell.

**Recommendation (original)**: run the seed-42 validation on P7 first (cheap,
~3.5h, confirms the pipeline before committing more compute anywhere), Dell
only as a secondary/opportunistic role afterward.

**Actual decision, 2026-07-12**: P7 is shared with another student and had
to be handed over at midnight the same day for 4 days — it will not be
available at all during that window. The reasons against Dell above (slow
hardware, multi-day risk) no longer apply as objections once P7 is off the
table entirely: a slow, unattended, multi-day run on the Dell is exactly the
right use of that idle time, since nothing else is competing for the P7
during those 4 days anyway and no other work is being blocked by it. Code
implemented and tested (unit tests + real-data dry-run + short real-data
smoke run, see below) on 2026-07-12 while the P7 finished the long-horizon
WP0 queue; pushed to GitHub; a coordination instruction was created for the
Dell to pull the code and launch ONE seed (42) training standalone. 3-seed
confirmation (WPB0 row 2) stays explicitly conditional on this first result
and is not launched pre-emptively.

**Implementation notes (code, 2026-07-12)**:
- `models/fno_plus_official.py`: `FNOPlusOfficial3d` gained an
  `output_offset: int = 1` constructor arg (default reproduces the exact
  original `x[..., 1:20]` slicing bit-for-bit — regression-tested).
- `datasets/floodcastbench_fno_plus_official_dataset.py`:
  `FloodCastBenchFNOPlusOfficialDataset` gained `context_length: int = 0`.
  Window grows to `context_length + sample_length`; the depth channel now
  holds real per-position observed values for the first `context_length + 1`
  positions and broadcasts the last known value forward across the 19 target
  positions (identical convention to the old single-frame broadcast, just
  starting from a real value instead of always frame 0) — `context_length=0`
  reproduces the original tensors exactly, verified by a direct
  `torch.equal` regression test.
- `datasets/floodcastbench_fno_plus_official_v1_dataset.py` and both
  `tools/{train,evaluate}_floodcastbench_fno_plus_official_v1.py`:
  `context_length` / `output_offset` threaded through config reading
  (`dataset.context_length`, `model.output_offset`) — no other pipeline code
  changed (normalization, training loop, metric accumulator, checkpointing
  all reused unmodified).
- New config: `configs/floodcastbench_fno_plus_official_v1_context24_highfid_60m.yaml`
  (`context_length: 24`, `stride: 44` for non-overlapping windows at the new
  44-frame window length, `split_counts: {train: 52, val: 6, test: 7}` —
  computed from the real 2881-frame Australia-60m sequence, same
  train/val/test proportions as the vanilla split; `output_offset: 25`).
- New test file `tests/test_fno_plus_official_context_smoke.py` (4 tests):
  `context_length=0` exact-equality regression vs the original dataset,
  `context_length=24` shape/no-leakage check (24 distinct real history
  values, 19 target positions all equal to the broadcast "current" value,
  explicitly asserted `!=` the actual target to catch any accidental
  leakage), model forward/backward at the new `output_offset`, and a
  default-vs-explicit `output_offset=1` equality check. All pass, plus the
  full pre-existing `fno_plus`-tagged suite (20 tests) still passes.
- Validated end-to-end against the REAL dataset (not just synthetic test
  fixtures): `--dry-run-config` against the real config produced
  `train_samples: 52, val_samples: 6, context_length: 24, input_shape: [6,
  536, 536, 44]` as expected, and a 1-epoch/1-batch CPU smoke run (real data,
  `/tmp` scratch output dirs) completed a full train+val+checkpoint+eval
  cycle with no errors.

**Result, 2026-07-13 (Dell, coordination instruction 0006, seed 42, 100/100
epochs, clean run, best checkpoint epoch 70)**: relRMSE 0.007822 vs vanilla
seed42 0.006694 — **context24 is WORSE, not better**, by +0.00113 (~8.4x the
vanilla 3-seed std of 0.000135 — a real effect, not noise on one seed). NSE,
Pearson r, CSI@0.001, CSI@0.01 all slightly worse too; only CSI@0.01 stays
close.

**Interpretation — the initial WPB0 hypothesis (giving FNO+ V2's context
would help it, since it directly removes a known information-budget
confound) is refuted, at least for this naive implementation.** Plausible
reasons, not yet distinguished: (a) FNO+ vanilla was already near-saturated
(NSE 0.999939, i.e. already explaining ~99.994% of target variance) with 1
frame of context on this benchmark — the marginal value of more input
history may simply be near zero regardless of architecture, so there was no
real headroom to unlock; (b) T growing from 20 to 44 with the same
`modes=12`/`width=20`/`fourier_layers=4` and unchanged `batch_size=1` /
learning-rate schedule may just be a harder optimization problem (more
parameters exposed to the FFT operator, same budget) rather than a
capacity-vs-information story — i.e. this may be confounded with WPB1's
untested hyperparameter axes, not a clean read on "does context help." Not
distinguished here; flagged as a caveat, not resolved.

**Consequence for Paper 1 (the sparsity paper)**: this result is actually
good news for that paper's honesty, not bad news for this one — it
strengthens (does not weaken) the "V2 clearly beats FNO+" comparison
reported there. V2 dense (3-seed mean relRMSE 0.001576, `context_length=24`)
already beat FNO+ vanilla (0.006550) by ×4.2 even before this result; now
that FNO+ has ALSO been given the exact same context budget and still came
out worse than its own 1-frame version, the "V2 wins because it just gets
more input information" explanation is directly weakened — V2's advantage
looks more architectural than context-driven. **Action**: add this number to
`reports/paper_master_plan.md`'s V2-vs-FNO+ discussion as a control point.

**Update 2026-07-13**: the user asked to confirm at 3 seeds anyway and also
to produce the long-horizon rollout curves (not just the native-protocol
number), so both were launched. Coordination instruction 0007 (Dell) chains:
seed42 long-horizon eval (the one thing missing for the already-trained seed)
→ train seed7 → native+long-horizon eval seed7 → train seed123 →
native+long-horizon eval seed123, via `scripts/run_wpb0_context24_remaining_seeds.sh`.
This required extending `tools/evaluate_floodcastbench_fno_plus_official_v1_long_horizon_rollout.py`
(previously hardcoded for context_length=0) to support the rolling
context-window convention: a history buffer of the last `context_length`
frames, real at first and blending into the model's own past predictions as
the rollout advances past its original start (never re-reading ground truth
beyond that point, which would leak information a real deployment wouldn't
have) — regression-tested (`context_length=0` unchanged bit-for-bit) and
sanity-checked against the real seed-42 checkpoint (horizon=19 CSI@0.001
matched the already-known native eval to 4 decimal places: 0.8968 vs
0.8967). Dashboard gained a dedicated `FNO_CONTEXT24_SEED_RUN_DIRS` curve,
guarded to skip until each seed's long-horizon output exists. Commits
`f13ef51`, `3ff3a35`.

**Interruption and recovery, 2026-07-13 → 2026-07-15**: instruction 0007
partially failed on the Dell — seed7 trained cleanly but both its evals hit
a relative-path bug in the orchestrator script (fixed, commit `55e364a`),
and seed123's training then died silently at epoch 26/100. Root cause
confirmed empirically: the Dell's `experiments`/`checkpoints`/`logs` are
NFS-mounted from the P7 (`/etc/exports`), and the P7 was shut down (loaned
to another student) at 2026-07-13 12:10 — the exact minute of seed123's
last checkpoint write. A process blocked on a dead NFS write hangs silently
rather than crashing, hence 2+ days with no error and no progress. Recorded
as PROTOCOL.md rule 7 (never write a long unsupervised job's live state
directly to an NFS path the P7 itself might unmount). Once the P7 returned
(2026-07-15), the remaining work (seed7's 2 evals against its already-good
checkpoint, seed123 full retrain + its 2 evals) was redone on the P7 itself
via `scripts/run_wpb0_context24_recovery.sh` — clean run, no further
issues.

**Final 3-seed result, 2026-07-15 (R1 satisfied)**: mean relRMSE
0.007529±0.000328 vs vanilla's 0.006550±0.000135 — **context24 confirmed
worse at 3/3 seeds**, mean gap ~7.3x the vanilla std, individually
consistent (0.007822 / 0.007591 / 0.007175, all above vanilla's highest
single seed 0.006694). NSE/Pearson r also consistently worse; CSI@0.01
stays close (0.994 vs vanilla's 0.995). **This closes WPB0**: the "more
context helps FNO+" hypothesis is refuted with the same rigor as the rest
of this project's negative results (§8's R1/R8 analogues), not just a
single-seed read. Long-horizon rollout data now exists for all 3 seeds
(`long_horizon_rollout_eval_dense_v2/checkpoint_best/` under each seed's run
dir) — dashboard curve ready to regenerate.

### WPB1 — Vanilla FNO+ hyperparameter boost (cheapest, do first)

Rationale: the current vanilla config is paper-conformant but was never
tuned for *our* best result — it was tuned for *fidelity to the paper's
stated numbers*. Since batch size 1 / 100 epochs / this width are all
paper-matched, not necessarily optimal, and GPU capacity is available, a
systematic sweep is cheap and has not been done.

Single-seed (42) screening runs, each a full 100-epoch training, changing
ONE axis at a time from the current baseline (`modes=12, width=20,
fourier_layers=4, batch_size=1, lr=0.001 cosine, epochs=100`):

1. **Batch size**: {1 (baseline), 2, 4, 8} — batch_size=1 is unusually small;
   larger batches may stabilize gradient estimates. Note: may require LR
   re-scaling; if batch>1 changes convergence epoch count, allow early
   stopping instead of forcing exactly 100 epochs.
2. **Width/modes/layers**: {width 32, modes 16, fourier_layers 6} tried
   independently (3 separate runs, not a full grid — budget discipline) —
   more capacity than paper-conformant setting, since GPU allows it.
3. **LR/schedule**: {lr 3e-4 cosine, lr 1e-3 with warmup+cosine, lr 1e-3
   step-decay} vs baseline cosine 1e-3→0.
4. **Padding ablation**: add explicit spatial+temporal padding before FFT,
   crop after (`next_gap_decision_report.md` Experiment 3, never run before).
5. **Longer training**: 200 epochs with early-stop patience, to check if 100
   epochs (paper-matched) is actually under-converged for this repo's setup
   (checkpoint selection was already best-epoch-in-run around 71-87/100 across
   the 3 baseline seeds — mild evidence of not being at the very end, worth
   checking if it keeps improving past 100).

Decision criterion (pre-registered): a variant is a genuine improvement only
if its seed-42 relRMSE beats 0.006694 (seed-42 baseline) by more than one
baseline 3-seed std (0.000135), i.e. relRMSE < 0.006559. Any variant clearing
this bar moves to WPB2 (3-seed confirmation). Variants that don't clear it
are recorded in the ledger and dropped, not re-tried with minor tweaks
(avoid unbounded fishing).

### WPB2 — Best vanilla config, 3-seed confirmation

Take the single best-performing WPB1 variant (if any cleared the bar,
otherwise skip this WP and record "no vanilla hyperparameter change beat
baseline outside noise" as the WPB1/WPB2 conclusion), rerun with seeds 7 and
123, confirm the improvement holds on the mean (R1: no single-seed claims).
This becomes the new "best vanilla FNO+" reference for WPB4/WPB5's Mamba
comparison.

### WPB3 — Mamba diagnostic (cheap, do before any Mamba sweep)

Single diagnostic pass on the *existing* Mamba checkpoint
(`29-06-2026_16-04-36_.../checkpoint_best.pth`) before spending any new GPU
time: no retraining, just deeper analysis of why relRMSE is worse.
- Recompute `negative_prediction_ratio` and a dry/wet-stratified relRMSE
  breakdown (dry pixels vs wet pixels, using the same threshold convention as
  `csi_gamma_0_001`) for both the Mamba run and the seed-42 vanilla baseline,
  side by side. §1.5 flagged `negative_prediction_ratio=0.196` on Mamba as a
  lead — confirm whether the RMSE gap is dominated by dry-region noise (fixable
  by e.g. removing/reducing residual scale, adding a light output constraint)
  or is spread across wet pixels too (would suggest the temporal-mixing
  placement itself is the problem, not a magnitude/stability issue).
- Check training curves (`metrics.csv` in the Mamba run dir) for
  instability signatures (loss spikes, non-monotone val curve) vs the vanilla
  baseline's curve.

Decision criterion: this diagnostic determines whether WPB4 starts from "fix
a stability/magnitude issue" (e.g. smaller Mamba residual init, added
LayerNorm/dropout, lower effective LR on the Mamba block) or "try a
different placement" (e.g. Mamba before the Fourier backbone instead of
after, Mamba interleaved between Fourier layers instead of as one block at
the end, Mamba only on a subset of latent channels).

**Training-curve instability check, done 2026-07-15 (zero GPU, pure
`metrics.csv` analysis)** — direct evidence for the "stability issue, not
just placement" branch:

| | Vanilla seed42 | Mamba seed42 | Ratio |
|---|---:|---:|---:|
| val_rrmse spikes (>3x median epoch-to-epoch jump) | 18 | **37** | ×2.1 |
| max val_rrmse reached (100 epochs) | 0.171 | **0.505** | ×3.0 |
| val_rrmse std across 100 epochs | 0.020 | **0.057** | ×2.9 |
| best epoch | 71 | 87 | — |

The Mamba run isn't just converging to a worse optimum — its validation
curve is measurably rougher throughout training (2-3x more spikes, 3x
higher peak error, 3x higher variance), consistent with a genuine
optimization/stability problem introduced by the Mamba branch, not merely a
placement or capacity limitation. This favors starting WPB4 from the
stability-fix branch (smaller residual init, lower Mamba-branch LR, added
regularization) before trying alternative placements.

**Wet/dry stratified breakdown, run 2026-07-15** — `tools/analyze_fno_plus_mamba_wet_dry.py`,
full result: `experiments/FloodCastBench/wpb3_mamba_wet_dry_diagnostic.json`.
`current_relative_rmse` on the dry stratum is not usable (near-zero
denominator inflates it to 3.5-4.4 regardless of model, a known artifact
already documented for this metric elsewhere in the repo) — read
`classical_rmse` (absolute, meters) and `negative_prediction_ratio` instead:

| | Vanilla | Mamba (naive) | Mamba vs vanilla |
|---|---:|---:|---:|
| classical RMSE, wet pixels | 0.005726 | **0.008227** | **+43.7% worse** |
| classical RMSE, dry pixels | 0.001707 | 0.001376 | -19.4% (Mamba slightly better) |
| negative_prediction_ratio, wet | 1.54% | 0.89% | Mamba slightly cleaner |
| negative_prediction_ratio, dry | 68.4% | 53.7% | Mamba slightly cleaner |

**This corrects, rather than confirms, this WP's original working
hypothesis.** §1.5 flagged Mamba's pooled `negative_prediction_ratio=0.196`
as a lead toward "Mamba destabilizes dry-region predictions specifically" —
but that comparison was incomplete: vanilla's own pooled ratio, measured
here for the first time, is 0.253 (25.3%), i.e. **higher** than Mamba's.
Mamba is not dirtier in dry regions than vanilla; if anything it's
marginally cleaner there. The real, now directly measured gap is in **wet
(flooded) pixels, in absolute terms**: Mamba's classical RMSE there is 44%
worse than vanilla's, and wet pixels dominate the pooled relRMSE (denominator
`sum(y^2)` is wet-pixel-dominated), which is why the pooled relRMSE gap
(+41%, §1.5) tracks the wet-pixel gap almost exactly, not a dry-region
artifact.

**Combined with the training-curve finding above** (2-3x rougher val_rrmse
curve throughout all 100 epochs, not just early on): the coherent picture
is a genuine optimization-stability problem that specifically corrupts
learning on the harder, information-dense wet regions -- not a dry-region
noise/magnitude issue. This is exactly what the LayerScale fix already
implemented above targets (a stable identity-preserving start lets the
network learn to use the Mamba branch gradually instead of absorbing an
untrained, full-strength perturbation from step 1), so no design change to
WPB4 is needed from this result -- it sharpens the diagnosis and predicts
the fix should show up mainly as a wet-pixel classical-RMSE improvement,
which the eventual WPB4 run should check explicitly (not just pooled
relRMSE).

### WPB4 — Mamba placement/hyperparameter sweep

Informed by WPB3. Single-seed (42) screening runs against the WPB2 best
vanilla baseline (not the original unmoved baseline, once/if WPB2 produces a
new reference):
- Placement: post-backbone (current/baseline), pre-backbone, interleaved
  (one Mamba block after each Fourier layer instead of one block at the end).
- Mamba sizing: `d_state` {8, 16 (baseline), 32}, `expand` {1, 2 (baseline)},
  `mamba_layers` {1 (baseline), 2}.
- Stability fixes per WPB3's finding: e.g. zero/near-zero residual-branch
  init, reduced Mamba-branch learning rate, added dropout.

Same decision criterion structure as WPB1 (beat the current best-known
baseline's relRMSE by more than 1 baseline-seed-std, pre-registered before
running).

**Stability fix implemented, 2026-07-15 (code, not yet trained/evaluated)**:
`TemporalMambaResidual` gained a LayerScale/ReZero-style gate
(`layer_scale_init` param, `models/fno_plus_official_mamba.py`) — a
learnable per-channel scale on the Mamba branch's output before the
residual add, initialized at 0 so the block starts as an exact identity
function and only gradually learns to trust the Mamba transform, instead of
an untrained transform perturbing the FNO latent at full strength from
step 1. Directly motivated by WPB3's training-curve finding (naive variant
2-3x rougher than vanilla). `layer_scale_init=None` (default) is an exact
regression match to the original ungated behavior — the already-trained
naive checkpoint's numbers in this ledger are unaffected. 4 new tests
(`tests/test_fno_plus_official_v1_mamba_smoke.py`): backward-compat
equality, exact identity at gate=0, gradient still flows despite zero
init, threading through the model. New config
`configs/floodcastbench_fno_plus_official_v1_mamba_layerscale_highfid_60m.yaml`
(`layer_scale_init: 0.0`, identical to the naive config otherwise) —
real-data dry-run + CPU/GPU smoke both pass; queued to actually train (seed
42 first, screening run per this WP's own protocol) once the GPU is free.

### WPB5 — Best Mamba config, 3-seed confirmation

Same structure as WPB2, for whichever Mamba variant clears WPB4's bar.

### WPB6 (conditional, only if WPB1-5 all fail to beat baseline outside noise)

Only reached if the paper's honest conclusion is "we tried N variants and
none clearly helped" (§2(c)). In that case, before writing that up as final,
consider ONE alternative architecture change as a last controlled test —
candidate: a lightweight non-Mamba temporal residual (e.g. plain GRU/Conv1d
over the same latent-sequence reshape used by `TemporalMambaResidual`) as a
sanity check on whether *any* added temporal-sequence mixer helps, or whether
the Fourier backbone is already capturing everything useful in the temporal
dimension (in which case the paper's conclusion sharpens to "temporal
sequence-mixing add-ons don't help this one-shot space-time-FFT
architecture," a real and citable negative result, not "Mamba specifically
fails" — a more defensible, general claim).

## 5. Evaluation protocol (identical across all WPs)

- Split: `test` only for final reported numbers (`val` used for checkpoint
  selection during training only, never reported as a result).
- Metric: pooled `current_relative_rmse`, `nse`, `pearson_r`, `csi_gamma_0_001`,
  `csi_gamma_0_01`, physical units after inverse-transform — same convention
  as `fno_plus_multiseed_results.md` and `fno_plus_dashboard_audit.md`.
- No wet/dry post-threshold applied to headline numbers (§1.3 point 2) —
  optional diagnostic only, clearly labelled if reported.
- Seeds: {42, 7, 123} for any confirmed/final number (same seeds as the
  existing vanilla baseline, for direct comparability).
- Every run's config, run dir, and checkpoint path recorded in §3's ledger
  the same day it completes — do not let results live only in memory/chat.

## 6. Paper skeleton (draft, revise once results exist)

1. Intro — FNO+ on FloodCastBench, official numbers vs known reproduction
   gap (cite/summarize, don't re-litigate — §1.3/§1.4 already close this).
2. Method — vanilla FNO+ recap (brief, it's a known architecture), the
   specific hyperparameter axes explored (WPB1), the Mamba integration
   design(s) explored (WPB4), each pre-registered before running per §4.
3. Results — table per WP (§3's ledger, cleaned up), headline: does any
   variant beat the repo's own vanilla baseline outside seed noise.
4. Analysis — for whichever branch (§2 a/b/c) actually happens; if Mamba
   underperforms, report and explain the naive-placement finding (§1.5) as a
   contribution in itself (saves other researchers the wasted compute).
5. Limitations — explicitly does not close the gap to the official paper's
   0.003941 (unpublished preprocessing details, §1.4); single dataset/event
   (Australia 60m only, unless WPB2/5 are cheap enough to extend to UK, TBD).
6. Conclusion.

## 7. Milestones

- M1: WPB1 screening runs complete, best-vanilla-candidate identified (or
  "none cleared the bar").
- M2: WPB2 3-seed confirmation of the vanilla result (or skip, recorded).
- M3: WPB3 diagnostic complete, WPB4 sweep design finalized from it.
- M4: WPB4 screening complete, best-Mamba-candidate identified (or none).
- M5: WPB5 3-seed confirmation.
- M6: (conditional) WPB6 alternative-architecture sanity check, only if M1-M5
  all negative.
- M7: Draft write-up.

## 8. Governance

Same rule as Paper 1: any new idea, scope change, or decision for this paper
gets written into this file before/as it happens, not just discussed in
chat. This file is the source of truth for Paper 2 going forward.

## 9. Changelog

- 2026-07-11: initial version. Code review of `models/fno_plus_official.py`,
  `models/fno_plus_official_mamba.py`, and all `reports/fno_plus_*` audit
  reports completed; established the vanilla 3-seed baseline (0.006550 mean
  relRMSE) as this paper's target-to-beat, and the existing naive Mamba
  single-seed result (0.009464, worse) as the key starting fact shaping WPB3
  before any Mamba sweep.
- 2026-07-12: added WPB0 (context-matched FNO+, K=24 to match Paper 1's V2),
  discovered while trying to build a fair V2-vs-FNO+ comparison for Paper 1 —
  confirmed by reading `datasets/floodcastbench_fno_plus_official_dataset.py`
  that FNO+'s current "20-frame" input is genuinely single-frame context (the
  one observed depth value is broadcast across all 20 T positions, not a real
  temporal signal), a 24x information-budget mismatch vs V2 that confounds
  any current V2-vs-FNO+ comparison independent of architecture. Set as
  priority 1 (serves both papers). Measured real epoch-time data from 3
  completed 100-epoch runs (53.6s/epoch clean P7 baseline, up to 181.3s/epoch
  when GPU-contended) to ground the ~3-3.5h/run P7 estimate and the
  recommendation to run WPB0's validation/confirmation runs on P7, not the
  Dell (violates the established small-task-only rule for Dell; A4000
  hardware gap alone could push a single run past half a day).
- 2026-07-13: WPB0 seed-42 result in (Dell, instruction 0006, exception
  approved by the user since P7 was handed to another student for 4 days
  starting 2026-07-12 midnight) — context24 relRMSE 0.007822, WORSE than
  vanilla (0.006694), refuting the initial "more context helps FNO+"
  hypothesis (effect size ~8.4x vanilla inter-seed std, not noise). Marked
  done; 3-seed confirmation deprioritized in favor of WPB1/WPB3 given the
  clear single-seed effect size. Flagged as a positive control point for
  Paper 1's V2-vs-FNO+ comparison (V2's advantage isn't explained by context
  budget alone, since giving FNO+ the same budget didn't help it).
