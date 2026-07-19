#!/bin/bash
# Tier-2 item 2.2 (reports/paper_master_plan.md, "5 seeds partout") applied
# to Paper 2's central result (reports/fno_plus_beat_paper_plan.md
# WPB4/WPB5): extend both the vanilla FNO+ baseline and the
# Mamba+LayerScale variant from seeds {42,7,123} to 5 seeds by adding
# seeds 1000 and 2000. Config-only -- the exact code paths already
# validated by the existing 3-seed runs, no new model/dataset code.
# Trains + evaluates each run sequentially, then moves to the next
# automatically. Meant to run unattended until ALL_WPB2_SEED_EXTENSION_DONE.
#
# Decision use (pre-registered in the WPB4/WPB5 section of the plan): the
# 5-seed pools re-test the existing verdict (mamba_layerscale
# 0.006353+/-0.000077 vs vanilla 0.006550+/-0.000135, 1.46 sigma at 3
# seeds). This script only trains+evaluates; reading the ledger and
# updating the plan is Claude-P7's step once results land.
#
# Usage: nohup scripts/run_wpb2_seed_extension.sh > /dev/null 2>&1 & disown
set -uo pipefail

PY="${PYTHON_BIN:-/home/wissam/miniforge3/envs/floodcast-mamba/bin/python}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT=/home/wissam/utem-workspace/logs/FloodCastBench/background_jobs
EXPERIMENT_ROOT=/home/wissam/utem-workspace/experiments/FloodCastBench
CHECKPOINT_ROOT=/home/wissam/utem-workspace/checkpoints/FloodCastBench

cd "$REPO_DIR"
mkdir -p "$LOG_ROOT"
ORCH_LOG="$LOG_ROOT/wpb2_seed_extension_$(date +%d-%m-%Y_%H-%M-%S).log"
exec > >(tee -a "$ORCH_LOG") 2>&1

echo "=== WPB2 seed extension started at $(date -Is) ==="
echo "orchestrator log: $ORCH_LOG"

require_run_dir () {
    local run_dir="$1" label="$2"
    if [ -z "$run_dir" ] || [ ! -d "$run_dir" ]; then
        echo "FATAL: could not resolve $label run dir (got '$run_dir')" >&2
        exit 1
    fi
}

# label | config_relpath | experiment_dir_suffix (for run-dir resolution)
RUNS=(
    "vanilla_seed1000|configs/floodcastbench_fno_plus_official_v1_seed1000_highfid_60m.yaml|fcb_fno_plus_official_v1_normalized_seed1000_highfid_60m"
    "vanilla_seed2000|configs/floodcastbench_fno_plus_official_v1_seed2000_highfid_60m.yaml|fcb_fno_plus_official_v1_normalized_seed2000_highfid_60m"
    "mamba_ls_seed1000|configs/floodcastbench_fno_plus_official_v1_mamba_layerscale_seed1000_highfid_60m.yaml|fcb_fno_plus_official_v1_mamba_layerscale_seed1000_highfid_60m"
    "mamba_ls_seed2000|configs/floodcastbench_fno_plus_official_v1_mamba_layerscale_seed2000_highfid_60m.yaml|fcb_fno_plus_official_v1_mamba_layerscale_seed2000_highfid_60m"
)

for entry in "${RUNS[@]}"; do
    IFS='|' read -r name cfg dirsuffix <<< "$entry"
    echo "--- training $name starting at $(date -Is) (config: $cfg) ---"
    "$PY" tools/train_floodcastbench_fno_plus_official_v1.py \
        --config "$cfg" --device cuda
    echo "--- training $name finished at $(date -Is) ---"

    RUN=$(ls -dt "$EXPERIMENT_ROOT"/*_${dirsuffix} 2>/dev/null | head -1)
    require_run_dir "$RUN" "$name"
    CKPT="$CHECKPOINT_ROOT/$(basename "$RUN")/checkpoint_best.pth"

    echo "--- eval $name starting at $(date -Is) ---"
    "$PY" tools/evaluate_floodcastbench_fno_plus_official_v1.py \
        --run-dir "$RUN" --checkpoint "$CKPT" --split test --device cuda
    echo "--- eval $name finished at $(date -Is) ---"
    echo "${name}_run=$RUN"
done

echo "=== ALL_WPB2_SEED_EXTENSION_DONE at $(date -Is) ==="
