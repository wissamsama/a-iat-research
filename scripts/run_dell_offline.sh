#!/bin/bash
# Runs a training/eval command with its output roots redirected to LOCAL
# staging instead of the NFS-shared paths, for use on the Dell when the P7
# is confirmed unreachable (scripts/check_p7_reachable.sh). Works because
# every trainer/evaluator tool in this repo already accepts
# --experiment-root / --checkpoint-root / --log-root overrides (used
# throughout the project for exactly this kind of redirection, e.g. the
# smoke-test runs). Local staging survives a P7 outage because it's on the
# Dell's own disk, not the mount -- sync back with
# scripts/dell_sync_to_p7.sh once P7 is reachable again.
#
# Usage:
#   scripts/run_dell_offline.sh tools/train_floodcastbench_fno_plus_official_v1.py \
#       --config configs/some_config.yaml --device cuda
#
# The three root flags are appended automatically; do not pass them
# yourself. Any tool that doesn't accept these three flags (rare -- check
# `--help` first) isn't safe to run through this wrapper.
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <python_script.py> [args...]" >&2
    exit 2
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING=/home/wissam/utem-workspace/local_staging
PY="${PYTHON_BIN:-/home/wissam/miniforge3/envs/floodcast-mamba/bin/python}"

if "$REPO_DIR/scripts/check_p7_reachable.sh"; then
    echo "P7 reachable -- this wrapper is for offline use only. Run the command directly (without this wrapper) so it uses the normal shared paths." >&2
    exit 3
fi

mkdir -p "$STAGING/experiments" "$STAGING/checkpoints" "$STAGING/logs"
echo "P7 unreachable -- running offline, output roots redirected to $STAGING" >&2

cd "$REPO_DIR"
exec "$PY" "$@" \
    --experiment-root "$STAGING/experiments" \
    --checkpoint-root "$STAGING/checkpoints" \
    --log-root "$STAGING/logs"
