#!/bin/bash
# Runs a training/eval command with its output roots redirected to THIS
# machine's own workspace copy, for use on the DGX Spark (Acer Veriton
# GN100 / NVIDIA GB10, ARM64) workstation set up 2026-07-21. This machine
# is a standalone restore of a P7 WSL backup, not networked to the P7/Dell
# NFS share -- every config in this repo hardcodes /home/wissam/utem-
# workspace/... paths (correct on P7/Dell), which do not resolve here
# (no /home/wissam symlink -- no root access in this sandboxed shell to
# create one). Same mechanism as scripts/run_dell_offline.sh: every
# trainer/evaluator tool in this repo accepts --dataset-root/
# --experiment-root/--checkpoint-root/--log-root overrides; this wrapper
# points all four at the real absolute path of the workspace we're
# actually running from, derived at call time (portable -- does not
# hardcode the Desktop/Wissam/... prefix).
#
# Python: uses the local venv built for this machine
# (~/Desktop/Wissam/venvs/floodcast-mamba, torch==2.12.1+cu130,
# mamba-ssm==2.3.2.post1 built from source against locally patched CUDA
# 13.0 headers -- see git commit adding this script for the glibc 2.42/
# CUDA rsqrt exception-spec conflict and its fix).
#
# Usage:
#   scripts/run_spark_local.sh tools/train_floodcastbench_diff_sparse_v2.py \
#       --config configs/some_config.yaml --device cuda
#
# The four root flags are appended automatically; do not pass them
# yourself. Any tool that doesn't accept these flags (rare -- check
# `--help` first) isn't safe to run through this wrapper.
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <python_script.py> [args...]" >&2
    exit 2
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "$REPO_DIR/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/Desktop/Wissam/venvs/floodcast-mamba/bin/python}"

if [ ! -x "$PY" ]; then
    echo "FATAL: python venv not found at $PY (set PYTHON_BIN to override)" >&2
    exit 3
fi

cd "$REPO_DIR"
exec "$PY" "$@" \
    --dataset-root "$WORKSPACE/data/FloodCastBench" \
    --experiment-root "$WORKSPACE/experiments/FloodCastBench" \
    --checkpoint-root "$WORKSPACE/checkpoints/FloodCastBench" \
    --log-root "$WORKSPACE/logs/FloodCastBench"
