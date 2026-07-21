#!/bin/bash
# WP12 Phase 2 (paper master plan, CRITIQUE): dose-response test of the
# sampling-noise-floor hypothesis (Sec. "Why does the absolute field fail
# at fine cadence?"). One fixed skeleton (Delta-Diff's), single crossed
# factor {Delta t in 300/900/1800/7200s} x {target: absolute, delta},
# one-step training, dense regime, seed 42, screening-labeled -- decides
# whether the mechanism claim upgrades from "hypothesis" to "demonstrated"
# or gets retracted to an empirical observation (paper Sec 3.2 PENDING box).
#
# Meant to run unattended: trains + evaluates each of the 8 runs
# sequentially, then moves to the next automatically.
#
# Usage: nohup scripts/run_wp12_phase2_dose_response.sh > /dev/null 2>&1 & disown
set -uo pipefail

PY="${PYTHON_BIN:-$HOME/Desktop/Wissam/venvs/floodcast-mamba/bin/python}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "$REPO_DIR/../.." && pwd)"
LOG_ROOT="$WORKSPACE/logs/FloodCastBench/background_jobs"
EXPERIMENT_ROOT="$WORKSPACE/experiments/FloodCastBench"
CHECKPOINT_ROOT="$WORKSPACE/checkpoints/FloodCastBench"

cd "$REPO_DIR"
mkdir -p "$LOG_ROOT"
ORCH_LOG="$LOG_ROOT/wp12_phase2_$(date +%d-%m-%Y_%H-%M-%S).log"
exec > >(tee -a "$ORCH_LOG") 2>&1

echo "=== WP12 Phase 2b (extension) dose-response started at $(date -Is) ==="
echo "orchestrator log: $ORCH_LOG"

require_run_dir () {
    local run_dir="$1" label="$2"
    if [ -z "$run_dir" ] || [ ! -d "$run_dir" ]; then
        echo "FATAL: could not resolve $label run dir (got '$run_dir')" >&2
        exit 1
    fi
}

RUNS=(
    "dt600_absolute|configs/floodcastbench_wp12_dt600_absolute.yaml|fcb_wp12_dt600_absolute"
    "dt600_delta|configs/floodcastbench_wp12_dt600_delta.yaml|fcb_wp12_dt600_delta"
    "dt1200_absolute|configs/floodcastbench_wp12_dt1200_absolute.yaml|fcb_wp12_dt1200_absolute"
    "dt1200_delta|configs/floodcastbench_wp12_dt1200_delta.yaml|fcb_wp12_dt1200_delta"
    "dt2400_absolute|configs/floodcastbench_wp12_dt2400_absolute.yaml|fcb_wp12_dt2400_absolute"
    "dt2400_delta|configs/floodcastbench_wp12_dt2400_delta.yaml|fcb_wp12_dt2400_delta"
    "dt4800_absolute|configs/floodcastbench_wp12_dt4800_absolute.yaml|fcb_wp12_dt4800_absolute"
    "dt4800_delta|configs/floodcastbench_wp12_dt4800_delta.yaml|fcb_wp12_dt4800_delta"
)

for entry in "${RUNS[@]}"; do
    IFS='|' read -r name cfg dirsuffix <<< "$entry"
    LOCAL_CFG=$("$PY" scripts/make_spark_local_config.py "$cfg")
    echo "--- training $name starting at $(date -Is) (config: $LOCAL_CFG) ---"
    "$PY" tools/train_floodcastbench_diff_sparse_v2.py \
        --config "$LOCAL_CFG" --device cuda
    echo "--- training $name finished at $(date -Is) ---"

    RUN=$(ls -dt "$EXPERIMENT_ROOT"/*_${dirsuffix} 2>/dev/null | head -1)
    require_run_dir "$RUN" "$name"
    CKPT="$CHECKPOINT_ROOT/$(basename "$RUN")/checkpoint_best.pth"

    echo "--- eval $name starting at $(date -Is) ---"
    "$PY" tools/evaluate_floodcastbench_diff_sparse_v2.py \
        --config "$LOCAL_CFG" --checkpoint "$CKPT" \
        --split test --missing-rate 0.0 --num-scenarios 1 --device cuda \
        --persistence-mode oracle
    echo "--- eval $name finished at $(date -Is) ---"
    echo "${name}_run=$RUN"
done

echo "=== ALL_WP12_PHASE2B_DONE at $(date -Is) ==="
