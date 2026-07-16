#!/bin/bash
# Meant to run at the very start of EVERY Dell /loop tick, before checking
# for pending coordination instructions. Detects a P7 down -> up transition
# (using a LOCAL state file, not one on the NFS share -- it has to survive
# the outage it's tracking) and automatically runs dell_sync_to_p7.sh the
# moment P7 comes back, without needing a human or an instruction to
# trigger it. Prints its state so it shows up in the loop's own log.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_FILE=/home/wissam/utem-workspace/.p7_reachable_state

prev=$(cat "$STATE_FILE" 2>/dev/null || echo "unknown")
if "$REPO_DIR/scripts/check_p7_reachable.sh"; then
    now="up"
else
    now="down"
fi

if [ "$prev" != "$now" ]; then
    echo "P7 reachability: $prev -> $now at $(date -Is)"
fi

if [ "$prev" = "down" ] && [ "$now" = "up" ]; then
    echo "P7 just came back -- auto-syncing local staging"
    "$REPO_DIR/scripts/dell_sync_to_p7.sh" || echo "WARNING: auto-sync failed, check manually"
fi

echo "$now" > "$STATE_FILE"
