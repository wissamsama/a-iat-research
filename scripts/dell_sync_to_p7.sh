#!/bin/bash
# Run on the Dell once scripts/check_p7_reachable.sh confirms P7 is back.
# Copies everything staged locally (while offline, via
# scripts/run_dell_offline.sh) into the real NFS-shared experiments/
# checkpoints/logs, then archives the local staging copy (renamed with a
# timestamp, never deleted) so nothing is lost even if a sync had a
# problem. Safe to run repeatedly / concurrently / on an empty staging dir.
#
# Safety guards (each one exists because of a real failure mode):
# - flock: two callers (the /loop tick and the background watcher) can fire
#   at the same moment; without the lock, one could archive the staging dir
#   mid-rsync of the other.
# - running-job guard: if an offline job is STILL WRITING into local
#   staging when P7 comes back, syncing now would copy half-written
#   checkpoints and archiving would pull the directory out from under the
#   running process. Detected via pgrep on the command line (offline jobs
#   carry "local_staging" in their --experiment-root argument, put there by
#   run_dell_offline.sh). Sync is simply deferred -- callers are
#   condition-based and will retry on their next tick.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING=/home/wissam/utem-workspace/local_staging
REAL=/home/wissam/utem-workspace
LOCK=/tmp/dell_sync_to_p7.lock

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "Another sync is already running (lock held) -- skipping."
    exit 0
fi

if ! "$REPO_DIR/scripts/check_p7_reachable.sh"; then
    echo "FATAL: P7 still unreachable, nothing to sync yet." >&2
    exit 1
fi

if [ ! -d "$STAGING" ]; then
    echo "Nothing staged locally ($STAGING doesn't exist) -- nothing to do."
    exit 0
fi

# Match the exact signature run_dell_offline.sh puts on the command line
# ("--experiment-root ..../local_staging/experiments"), NOT the bare word
# "local_staging" -- a bare-word match false-positives on any interactive
# shell command that merely mentions the directory (observed in testing:
# the compound command's own bash process matched itself).
if pgrep -f -- "--experiment-root.*local_staging" >/dev/null 2>&1; then
    echo "An offline job is still running against $STAGING -- sync deferred (will retry on next tick)."
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
