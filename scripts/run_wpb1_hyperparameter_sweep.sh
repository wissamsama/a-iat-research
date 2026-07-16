#!/bin/bash
# WPB1 (reports/fno_plus_beat_paper_plan.md): vanilla FNO+ hyperparameter
# screening sweep, seed 42, single-axis-at-a-time vs the paper-conformant
# baseline (modes=12, width=20, fourier_layers=4, batch_size=1, lr=0.001
# cosine, 100 epochs). Config-only changes -- no untested model/dataset code
# paths, safe to run unattended on a machine without local supervision.
# Trains + evaluates each variant sequentially, then moves to the next
# automatically. Meant to run continuously (24/7) until ALL_WPB1_SWEEP_DONE.
#
# Decision criterion (pre-registered, reports/fno_plus_beat_paper_plan.md
# WPB1): a variant is a genuine improvement only if its seed-42 relRMSE
# beats the seed-42 baseline (0.006694) by more than one baseline 3-seed
# std (0.000135), i.e. relRMSE < 0.006559. This script does not decide --
# it only trains+evaluates; reading the ledger and updating the plan is a
# separate step (done by Claude-P7 once results land on the shared NFS).
#
# Usage: nohup scripts/run_wpb1_hyperparameter_sweep.sh > /dev/null 2>&1 & disown
set -uo pipefail

PY="${PYTHON_BIN:-/home/wissam/miniforge3/envs/floodcast-mamba/bin/python}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT=/home/wissam/utem-workspace/logs/FloodCastBench/background_jobs
EXPERIMENT_ROOT=/home/wissam/utem-workspace/experiments/FloodCastBench
CHECKPOINT_ROOT=/home/wissam/utem-workspace/checkpoints/FloodCastBench

cd "$REPO_DIR"
mkdir -p "$LOG_ROOT"
ORCH_LOG="$LOG_ROOT/wpb1_sweep_$(date +%d-%m-%Y_%H-%M-%S).log"
exec > >(tee -a "$ORCH_LOG") 2>&1

echo "=== WPB1 hyperparameter sweep started at $(date -Is) ==="
echo "orchestrator log: $ORCH_LOG"

require_run_dir () {
    local run_dir="$1" label="$2"
    if [ -z "$run_dir" ] || [ ! -d "$run_dir" ]; then
        echo "FATAL: could not resolve $label run dir (got '$run_dir')" >&2
        exit 1
    fi
}

# variant_name config_relpath extra_train_args
VARIANTS=(
    "bs2|configs/floodcastbench_fno_plus_official_v1_wpb1_bs2_highfid_60m.yaml|"
    "bs4|configs/floodcastbench_fno_plus_official_v1_wpb1_bs4_highfid_60m.yaml|"
    "bs8|configs/floodcastbench_fno_plus_official_v1_wpb1_bs8_highfid_60m.yaml|"
    "width32|configs/floodcastbench_fno_plus_official_v1_wpb1_width32_highfid_60m.yaml|"
    "modes16|configs/floodcastbench_fno_plus_official_v1_wpb1_modes16_highfid_60m.yaml|"
    "layers6|configs/floodcastbench_fno_plus_official_v1_wpb1_layers6_highfid_60m.yaml|"
    "lr3e4|configs/floodcastbench_fno_plus_official_v1_wpb1_lr3e4_highfid_60m.yaml|"
    "epochs200|configs/floodcastbench_fno_plus_official_v1_wpb1_bs2_highfid_60m.yaml_UNUSED|--epochs 200"
)

# epochs200 reuses the *baseline* hyperparameters (paper-conformant config),
# only --epochs is overridden -- it needs its own config copy rather than
# piggy-backing on bs2's file. Point it at a plain copy of the baseline.
BASELINE_CFG="configs/floodcastbench_fno_plus_official_v1_wpb1_epochs200_highfid_60m.yaml"
if [ ! -f "$BASELINE_CFG" ]; then
    echo "FATAL: expected $BASELINE_CFG to exist (baseline copy for the epochs200 variant)" >&2
    exit 1
fi
VARIANTS[7]="epochs200|${BASELINE_CFG}|--epochs 200"

for entry in "${VARIANTS[@]}"; do
    IFS='|' read -r name cfg extra_args <<< "$entry"
    echo "--- training $name starting at $(date -Is) (config: $cfg, extra: ${extra_args:-none}) ---"
    "$PY" tools/train_floodcastbench_fno_plus_official_v1.py \
        --config "$cfg" --device cuda $extra_args
    echo "--- training $name finished at $(date -Is) ---"

    RUN=$(ls -dt "$EXPERIMENT_ROOT"/*_fcb_fno_plus_official_v1_wpb1_${name}_highfid_60m 2>/dev/null | head -1)
    require_run_dir "$RUN" "$name"
    CKPT="$CHECKPOINT_ROOT/$(basename "$RUN")/checkpoint_best.pth"

    echo "--- eval $name starting at $(date -Is) ---"
    "$PY" tools/evaluate_floodcastbench_fno_plus_official_v1.py \
        --run-dir "$RUN" --checkpoint "$CKPT" --split test --device cuda
    echo "--- eval $name finished at $(date -Is) ---"
    echo "${name}_run=$RUN"
done

echo "=== ALL_WPB1_SWEEP_DONE at $(date -Is) ==="
