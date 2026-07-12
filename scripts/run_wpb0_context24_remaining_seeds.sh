#!/bin/bash
# WPB0 (reports/fno_plus_beat_paper_plan.md): train the 2 remaining FNO+
# context24 seeds (7, 123), run the native-protocol test eval + long-horizon
# rollout eval on all 3 seeds (42 already trained, only needs long-horizon),
# so the dashboard's FNO+-context24 curve and 3-seed confirmation table can
# be built. All steps run sequentially in this one script/process -- no
# parallelism, no `wait` deadlock risk (see run_training_queue.sh's history
# with that bug), safe for an unattended multi-hour background run.
#
# Usage: nohup scripts/run_wpb0_context24_remaining_seeds.sh > /dev/null 2>&1 & disown
set -uo pipefail

PY="${PYTHON_BIN:-/home/wissam/miniforge3/envs/floodcast-mamba/bin/python}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT=/home/wissam/utem-workspace/logs/FloodCastBench/background_jobs

cd "$REPO_DIR"
mkdir -p "$LOG_ROOT"
ORCH_LOG="$LOG_ROOT/wpb0_remaining_seeds_$(date +%d-%m-%Y_%H-%M-%S).log"
exec > >(tee -a "$ORCH_LOG") 2>&1

echo "=== WPB0 remaining seeds + long-horizon evals started at $(date -Is) ==="
echo "orchestrator log: $ORCH_LOG"

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

# --- Seed 42: already trained (coordination instruction 0006), native eval
# already done. Only the long-horizon rollout is missing. ---
SEED42_RUN=/home/wissam/utem-workspace/experiments/FloodCastBench/12-07-2026_16-00-14_fcb_fno_plus_official_v1_context24_highfid_60m
SEED42_CKPT=/home/wissam/utem-workspace/checkpoints/FloodCastBench/12-07-2026_16-00-14_fcb_fno_plus_official_v1_context24_highfid_60m/checkpoint_best.pth
run_long_horizon "$SEED42_RUN" "$SEED42_CKPT"

# --- Seed 7: train, native eval, long-horizon eval. ---
echo "--- training seed7 starting at $(date -Is) ---"
"$PY" tools/train_floodcastbench_fno_plus_official_v1.py \
    --config configs/floodcastbench_fno_plus_official_v1_context24_seed7_highfid_60m.yaml --device cuda
echo "--- training seed7 finished at $(date -Is) ---"
SEED7_RUN=$(ls -dt experiments/FloodCastBench/*_fcb_fno_plus_official_v1_context24_seed7_highfid_60m 2>/dev/null | head -1)
SEED7_CKPT="$(echo "$SEED7_RUN" | sed 's#/experiments/#/checkpoints/#')/checkpoint_best.pth"
run_native_eval "$SEED7_RUN" "$SEED7_CKPT"
run_long_horizon "$SEED7_RUN" "$SEED7_CKPT"

# --- Seed 123: train, native eval, long-horizon eval. ---
echo "--- training seed123 starting at $(date -Is) ---"
"$PY" tools/train_floodcastbench_fno_plus_official_v1.py \
    --config configs/floodcastbench_fno_plus_official_v1_context24_seed123_highfid_60m.yaml --device cuda
echo "--- training seed123 finished at $(date -Is) ---"
SEED123_RUN=$(ls -dt experiments/FloodCastBench/*_fcb_fno_plus_official_v1_context24_seed123_highfid_60m 2>/dev/null | head -1)
SEED123_CKPT="$(echo "$SEED123_RUN" | sed 's#/experiments/#/checkpoints/#')/checkpoint_best.pth"
run_native_eval "$SEED123_RUN" "$SEED123_CKPT"
run_long_horizon "$SEED123_RUN" "$SEED123_CKPT"

echo "=== ALL_WPB0_REMAINING_SEEDS_DONE at $(date -Is) ==="
echo "seed42_run=$SEED42_RUN"
echo "seed7_run=$SEED7_RUN"
echo "seed123_run=$SEED123_RUN"
