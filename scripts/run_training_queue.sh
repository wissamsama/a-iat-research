#!/bin/bash
# Generic training queue for the V2 family (diff_sparse_v2 / deterministic_twin
# / ablations). Runs waves of one config x all requested missing rates in
# PARALLEL (3 runs fit comfortably on both the P7 RTX 6000 Ada and the Dell
# A4000), waves sequentially. Logs under logs/FloodCastBench/background_jobs/.
#
# Usage:
#   scripts/run_training_queue.sh TAG "MISSING_RATES" CONFIG [CONFIG...] [-- EXTRA_ARGS...]
#
# Examples (paper master plan):
#   # WP1 deterministic twin, full protocol (3 waves x 3 sparsities):
#   scripts/run_training_queue.sh det_twin "0.0 0.5 0.95" \
#     configs/floodcastbench_det_twin_highfid_60m.yaml \
#     configs/floodcastbench_det_twin_seed7_highfid_60m.yaml \
#     configs/floodcastbench_det_twin_seed123_highfid_60m.yaml
#
#   # WP2 context ablation (one wave):
#   scripts/run_training_queue.sh v2_ctx12 "0.0 0.5 0.95" \
#     configs/floodcastbench_diff_sparse_v2_ctx12_highfid_60m.yaml
#
#   # WP4 ablation grid on m50 (waves of one rate each):
#   scripts/run_training_queue.sh v2_abl "0.5" \
#     configs/floodcastbench_diff_sparse_v2_abl_absolute_highfid_60m.yaml \
#     configs/floodcastbench_diff_sparse_v2_abl_notargetrain_highfid_60m.yaml \
#     configs/floodcastbench_diff_sparse_v2_abl_nospatial_highfid_60m.yaml
#
#   # WP6 extended-budget sparse reruns:
#   scripts/run_training_queue.sh v2_budget600 "0.5 0.95" \
#     configs/floodcastbench_diff_sparse_v2_highfid_60m.yaml \
#     configs/floodcastbench_diff_sparse_v2_seed7_highfid_60m.yaml \
#     -- --epochs 600 --early-stop-patience 120
#
# Recommended launch (survives the shell):
#   nohup scripts/run_training_queue.sh ... > /dev/null 2>&1 & disown
set -uo pipefail

PY="${PYTHON_BIN:-/home/wissam/miniforge3/envs/floodcast-mamba/bin/python}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT=/home/wissam/utem-workspace/logs/FloodCastBench/background_jobs

TAG="${1:?usage: run_training_queue.sh TAG \"MISSING_RATES\" CONFIG... [-- EXTRA_ARGS...]}"
RATES="${2:?missing rates, e.g. \"0.0 0.5 0.95\"}"
shift 2

CONFIGS=()
EXTRA_ARGS=()
seen_sep=0
for arg in "$@"; do
    if [ "$arg" = "--" ]; then seen_sep=1; continue; fi
    if [ "$seen_sep" = "1" ]; then EXTRA_ARGS+=("$arg"); else CONFIGS+=("$arg"); fi
done
[ "${#CONFIGS[@]}" -ge 1 ] || { echo "no configs given" >&2; exit 2; }

cd "$REPO_DIR"
mkdir -p "$LOG_ROOT"
ORCH_LOG="$LOG_ROOT/queue_${TAG}_$(date +%d-%m-%Y_%H-%M-%S).log"
exec > >(tee -a "$ORCH_LOG") 2>&1

echo "=== queue '$TAG' started at $(date -Is) ==="
echo "configs: ${CONFIGS[*]}"
echo "missing rates per wave: $RATES"
echo "extra args: ${EXTRA_ARGS[*]:-none}"
echo "orchestrator log: $ORCH_LOG"

for cfg in "${CONFIGS[@]}"; do
    [ -f "$cfg" ] || { echo "FATAL: config not found: $cfg"; exit 3; }
    wave_name="$(basename "$cfg" .yaml)"
    echo "--- wave $wave_name starting at $(date -Is) ---"
    for mr in $RATES; do
        run_log="$LOG_ROOT/${TAG}_${wave_name}_mr${mr}_$(date +%d-%m-%Y_%H-%M-%S).log"
        echo "launching: $cfg mr=$mr log=$run_log"
        "$PY" tools/train_floodcastbench_diff_sparse_v2.py \
            --config "$cfg" --missing-rate "$mr" --device cuda \
            "${EXTRA_ARGS[@]}" > "$run_log" 2>&1 &
        sleep 3  # run-dir names are second-granular; avoid collisions
    done
    wait
    echo "--- wave $wave_name finished at $(date -Is) ---"
done
echo "=== queue '$TAG' finished at $(date -Is) ==="
