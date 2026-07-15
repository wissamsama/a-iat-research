#!/bin/bash
# Recovery for the 2026-07-13 Dell run of WPB0's remaining seeds (reports/
# fno_plus_beat_paper_plan.md), which partially failed: seed7's training
# succeeded but both its evals hit a relative-path bug (fixed in
# run_wpb0_context24_remaining_seeds.sh), and seed123's training then died
# silently at epoch 26/100 for an undetermined reason (no traceback, no
# further log output for 2+ days). This script does only what's left:
# seed7's 2 evals (checkpoint already exists, no retrain needed) + seed123's
# full retrain (from scratch -- the trainer has no resume support) + its 2
# evals. Meant to run on P7, now available again.
#
# Usage: nohup scripts/run_wpb0_context24_recovery.sh > /dev/null 2>&1 & disown
set -uo pipefail

PY="${PYTHON_BIN:-/home/wissam/miniforge3/envs/floodcast-mamba/bin/python}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT=/home/wissam/utem-workspace/logs/FloodCastBench/background_jobs
EXPERIMENT_ROOT=/home/wissam/utem-workspace/experiments/FloodCastBench
CHECKPOINT_ROOT=/home/wissam/utem-workspace/checkpoints/FloodCastBench

cd "$REPO_DIR"
mkdir -p "$LOG_ROOT"
ORCH_LOG="$LOG_ROOT/wpb0_recovery_$(date +%d-%m-%Y_%H-%M-%S).log"
exec > >(tee -a "$ORCH_LOG") 2>&1

echo "=== WPB0 recovery started at $(date -Is) ==="
echo "orchestrator log: $ORCH_LOG"

require_run_dir () {
    local run_dir="$1" label="$2"
    if [ -z "$run_dir" ] || [ ! -d "$run_dir" ]; then
        echo "FATAL: could not resolve $label run dir (got '$run_dir')" >&2
        exit 1
    fi
}

run_native_eval () {
    local run_dir="$1" ckpt="$2"
    echo "--- native eval $run_dir starting at $(date -Is) ---"
    "$PY" tools/evaluate_floodcastbench_fno_plus_official_v1.py \
        --run-dir "$run_dir" --checkpoint "$ckpt" --split test --device cuda
    echo "--- native eval $run_dir finished at $(date -Is) ---"
}

run_long_horizon () {
    local run_dir="$1" ckpt="$2"
    local out_dir="$run_dir/long_horizon_rollout_eval_dense_v2/checkpoint_best"
    echo "--- long-horizon eval $run_dir starting at $(date -Is) ---"
    "$PY" tools/evaluate_floodcastbench_fno_plus_official_v1_long_horizon_rollout.py \
        --run-dir "$run_dir" --checkpoint "$ckpt" --checkpoint-name checkpoint_best \
        --output-dir "$out_dir" --horizons 19 72 144 --device cuda
    echo "--- long-horizon eval $run_dir finished at $(date -Is) ---"
}

# --- Seed 7: already trained 2026-07-13, only the 2 evals are missing. ---
SEED7_RUN="$EXPERIMENT_ROOT/13-07-2026_04-04-20_fcb_fno_plus_official_v1_context24_seed7_highfid_60m"
SEED7_CKPT="$CHECKPOINT_ROOT/13-07-2026_04-04-20_fcb_fno_plus_official_v1_context24_seed7_highfid_60m/checkpoint_best.pth"
require_run_dir "$SEED7_RUN" seed7
run_native_eval "$SEED7_RUN" "$SEED7_CKPT"
run_long_horizon "$SEED7_RUN" "$SEED7_CKPT"

# --- Seed 123: died at epoch 26/100 on the Dell 2026-07-13 -- full retrain
# (no resume support), then both evals. ---
echo "--- training seed123 starting at $(date -Is) ---"
"$PY" tools/train_floodcastbench_fno_plus_official_v1.py \
    --config configs/floodcastbench_fno_plus_official_v1_context24_seed123_highfid_60m.yaml --device cuda
echo "--- training seed123 finished at $(date -Is) ---"
SEED123_RUN=$(ls -dt "$EXPERIMENT_ROOT"/*_fcb_fno_plus_official_v1_context24_seed123_highfid_60m 2>/dev/null | head -1)
require_run_dir "$SEED123_RUN" seed123
SEED123_CKPT="$CHECKPOINT_ROOT/$(basename "$SEED123_RUN")/checkpoint_best.pth"
run_native_eval "$SEED123_RUN" "$SEED123_CKPT"
run_long_horizon "$SEED123_RUN" "$SEED123_CKPT"

echo "=== ALL_WPB0_RECOVERY_DONE at $(date -Is) ==="
echo "seed7_run=$SEED7_RUN"
echo "seed123_run=$SEED123_RUN"
