#!/bin/bash
# Run on the Dell once scripts/check_p7_reachable.sh confirms P7 is back.
# Copies everything staged locally (while offline, via
# scripts/run_dell_offline.sh) into the real NFS-shared experiments/
# checkpoints/logs, then archives the local staging copy (renamed with a
# timestamp, never deleted) so nothing is lost even if a sync had a
# problem. Safe to run repeatedly / on an empty staging dir (no-op).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING=/home/wissam/utem-workspace/local_staging
REAL=/home/wissam/utem-workspace

if ! "$REPO_DIR/scripts/check_p7_reachable.sh"; then
    echo "FATAL: P7 still unreachable, nothing to sync yet." >&2
    exit 1
fi

if [ ! -d "$STAGING" ]; then
    echo "Nothing staged locally ($STAGING doesn't exist) -- nothing to do."
    exit 0
fi

synced_any=0
for sub in experiments checkpoints logs; do
    if [ -d "$STAGING/$sub" ] && [ -n "$(ls -A "$STAGING/$sub" 2>/dev/null)" ]; then
        echo "--- syncing $STAGING/$sub -> $REAL/$sub ---"
        # --ignore-existing: never overwrite something already on the
        # shared side under the same name (shouldn't happen given
        # timestamped run-dir naming, but a real safeguard costs nothing).
        rsync -av --ignore-existing "$STAGING/$sub"/ "$REAL/$sub"/
        synced_any=1
    fi
done

if [ "$synced_any" = "1" ]; then
    archived="${STAGING}_synced_$(date +%d-%m-%Y_%H-%M-%S)"
    mv "$STAGING" "$archived"
    echo "=== sync complete, local staging archived to $archived ==="
else
    echo "Local staging directories were all empty -- nothing to sync."
fi
