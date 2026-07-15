#!/bin/bash
# WPB4 (reports/fno_plus_beat_paper_plan.md): 3-seed confirmation of the
# LayerScale-gated Mamba variant. seed42 already trained+evaluated
# (relRMSE 0.006442 pooled, wet-pixel classical RMSE 0.005554 vs vanilla's
# 0.005726 -- a genuine win on the physically-relevant metric, though pooled
# relRMSE doesn't clear the >1 baseline-std bar on this single seed). This
# script trains + evaluates seeds 7 and 123 sequentially so a proper 3-seed
# mean can be computed.
#
# Usage: nohup scripts/run_wpb4_mamba_layerscale_confirmation.sh > /dev/null 2>&1 & disown
set -uo pipefail

PY="${PYTHON_BIN:-/home/wissam/miniforge3/envs/floodcast-mamba/bin/python}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT=/home/wissam/utem-workspace/logs/FloodCastBench/background_jobs
EXPERIMENT_ROOT=/home/wissam/utem-workspace/experiments/FloodCastBench
CHECKPOINT_ROOT=/home/wissam/utem-workspace/checkpoints/FloodCastBench

cd "$REPO_DIR"
mkdir -p "$LOG_ROOT"
ORCH_LOG="$LOG_ROOT/wpb4_confirmation_$(date +%d-%m-%Y_%H-%M-%S).log"
exec > >(tee -a "$ORCH_LOG") 2>&1

echo "=== WPB4 confirmation started at $(date -Is) ==="

require_run_dir () {
    local run_dir="$1" label="$2"
    if [ -z "$run_dir" ] || [ ! -d "$run_dir" ]; then
        echo "FATAL: could not resolve $label run dir (got '$run_dir')" >&2
        exit 1
    fi
}

for seed in 7 123; do
    echo "--- training seed${seed} starting at $(date -Is) ---"
    "$PY" tools/train_floodcastbench_fno_plus_official_v1_mamba.py \
        --config "configs/floodcastbench_fno_plus_official_v1_mamba_layerscale_seed${seed}_highfid_60m.yaml" --device cuda
    echo "--- training seed${seed} finished at $(date -Is) ---"

    RUN=$(ls -dt "$EXPERIMENT_ROOT"/*_fcb_fno_plus_official_v1_mamba_layerscale_seed${seed}_highfid_60m 2>/dev/null | head -1)
    require_run_dir "$RUN" "seed${seed}"
    CKPT="$CHECKPOINT_ROOT/$(basename "$RUN")/checkpoint_best.pth"

    echo "--- eval seed${seed} starting at $(date -Is) ---"
    "$PY" tools/evaluate_floodcastbench_fno_plus_official_v1_mamba.py \
        --run-dir "$RUN" --checkpoint "$CKPT" --split test --device cuda
    echo "--- eval seed${seed} finished at $(date -Is) ---"
    echo "seed${seed}_run=$RUN"
done

echo "=== ALL_WPB4_CONFIRMATION_DONE at $(date -Is) ==="
