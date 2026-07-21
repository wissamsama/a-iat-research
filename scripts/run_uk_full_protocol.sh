#!/bin/bash
# Item 1.5 (paper master plan, "LE saut de credibilite 1->2 evenements"):
# UK 2015 event, full protocol -- V2 and twin, 3 seeds (42/7/123), 3
# sparsity regimes (dense/m50/m95), 18 train+eval pairs. Uses the same
# base configs as Australia (only event/grid/delta_stats differ), with
# --missing-rate as a CLI override per regime (matches how the Australia
# m50/m95 runs were launched -- no separate committed config per
# sparsity). Sequenced after WP12 Phase 2b's per-window logging fix
# (2026-07-21) landed, so these evaluations directly support the
# pre-registered window x seed paired significance test (n=39) without
# needing to be re-run.
#
# Waits for the WP12 Phase 2b extension queue to finish before starting
# its first run, so it doesn't compound the current Pakistan+WP12
# contention -- only really parallel with Pakistan once WP12 clears.
#
# Usage: nohup scripts/run_uk_full_protocol.sh > /dev/null 2>&1 & disown
set -uo pipefail

PY="${PYTHON_BIN:-$HOME/Desktop/Wissam/venvs/floodcast-mamba/bin/python}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "$REPO_DIR/../.." && pwd)"
LOG_ROOT="$WORKSPACE/logs/FloodCastBench/background_jobs"
EXPERIMENT_ROOT="$WORKSPACE/experiments/FloodCastBench"
CHECKPOINT_ROOT="$WORKSPACE/checkpoints/FloodCastBench"

cd "$REPO_DIR"
mkdir -p "$LOG_ROOT"
ORCH_LOG="$LOG_ROOT/uk_full_protocol_$(date +%d-%m-%Y_%H-%M-%S).log"
exec > >(tee -a "$ORCH_LOG") 2>&1

echo "=== UK full protocol (item 1.5) started at $(date -Is) ==="
echo "orchestrator log: $ORCH_LOG"

WP12_2B_LOG=$(ls -t "$LOG_ROOT"/wp12_phase2_*.log 2>/dev/null | head -1)
if [ -n "$WP12_2B_LOG" ]; then
    echo "waiting for WP12 Phase 2b to finish (log: $WP12_2B_LOG) before starting, to avoid 3-way GPU contention..."
    while ! grep -q "ALL_WP12_PHASE2B_DONE" "$WP12_2B_LOG" 2>/dev/null; do
        sleep 60
    done
    echo "WP12 Phase 2b done at $(date -Is), proceeding."
fi

require_run_dir () {
    local run_dir="$1" label="$2"
    if [ -z "$run_dir" ] || [ ! -d "$run_dir" ]; then
        echo "FATAL: could not resolve $label run dir (got '$run_dir')" >&2
        exit 1
    fi
}

# model_label | config_dirsuffix | num_scenarios
MODELS=(
    "twin|configs/floodcastbench_det_twin_uk_highfid_60m.yaml|fcb_det_twin_uk_highfid_60m|1"
    "twin_seed7|configs/floodcastbench_det_twin_uk_seed7_highfid_60m.yaml|fcb_det_twin_uk_seed7_highfid_60m|1"
    "twin_seed123|configs/floodcastbench_det_twin_uk_seed123_highfid_60m.yaml|fcb_det_twin_uk_seed123_highfid_60m|1"
    "v2|configs/floodcastbench_diff_sparse_v2_uk_highfid_60m.yaml|fcb_diff_sparse_v2_uk_highfid_60m|8"
    "v2_seed7|configs/floodcastbench_diff_sparse_v2_uk_seed7_highfid_60m.yaml|fcb_diff_sparse_v2_uk_seed7_highfid_60m|8"
    "v2_seed123|configs/floodcastbench_diff_sparse_v2_uk_seed123_highfid_60m.yaml|fcb_diff_sparse_v2_uk_seed123_highfid_60m|8"
)
SPARSITIES=("dense|0.0" "m50|0.5" "m95|0.95")

for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r model_label cfg dirsuffix num_scenarios <<< "$model_entry"
    for sparsity_entry in "${SPARSITIES[@]}"; do
        IFS='|' read -r sparsity_label missing_rate <<< "$sparsity_entry"
        name="${model_label}_${sparsity_label}"
        LOCAL_CFG=$("$PY" scripts/make_spark_local_config.py "$cfg")

        echo "--- training $name starting at $(date -Is) (config: $LOCAL_CFG, missing_rate=$missing_rate) ---"
        "$PY" tools/train_floodcastbench_diff_sparse_v2.py \
            --config "$LOCAL_CFG" --device cuda --missing-rate "$missing_rate"
        echo "--- training $name finished at $(date -Is) ---"

        RUN=$(ls -dt "$EXPERIMENT_ROOT"/*_${dirsuffix} 2>/dev/null | head -1)
        require_run_dir "$RUN" "$name"
        CKPT="$CHECKPOINT_ROOT/$(basename "$RUN")/checkpoint_best.pth"

        echo "--- eval $name starting at $(date -Is) ---"
        "$PY" tools/evaluate_floodcastbench_diff_sparse_v2.py \
            --config "$LOCAL_CFG" --checkpoint "$CKPT" \
            --split test --missing-rate "$missing_rate" --num-scenarios "$num_scenarios" --device cuda \
            --persistence-mode oracle
        echo "--- eval $name finished at $(date -Is) ---"
        echo "${name}_run=$RUN"
    done
done

echo "=== ALL_UK_FULL_PROTOCOL_DONE at $(date -Is) ==="
