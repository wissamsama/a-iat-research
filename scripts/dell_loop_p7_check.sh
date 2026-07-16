#!/bin/bash
# Meant to run at the very start of EVERY Dell /loop tick, before touching
# anything on the NFS share. Prints the current P7 reachability, and -- the
# important part -- triggers the local->shared sync whenever the CONDITIONS
# are right (P7 up + non-empty local staging + no offline job still
# writing), rather than only on a down->up TRANSITION. Condition-based
# beats transition-based here because a failed or deferred sync simply gets
# retried on the next tick instead of being lost forever the moment the
# state file already says "up" (a real bug in the first version of this
# script).
#
# The state file is LOCAL (not on the NFS share -- it has to survive the
# very outage it's tracking) and is only used to print transitions for the
# loop's log, never to gate the sync.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_FILE=/home/wissam/utem-workspace/.p7_reachable_state
STAGING=/home/wissam/utem-workspace/local_staging

prev=$(cat "$STATE_FILE" 2>/dev/null || echo "unknown")
if "$REPO_DIR/scripts/check_p7_reachable.sh"; then
    now="up"
else
    now="down"
fi
[ "$prev" != "$now" ] && echo "P7 reachability: $prev -> $now at $(date -Is)"
echo "$now" > "$STATE_FILE"

if [ "$now" = "down" ]; then
    echo "P7 DOWN -- do not read status.md this tick; use scripts/run_dell_offline.sh for any new launch; journal locally (see PROTOCOL.md, Mode Dell hors-ligne)."
    exit 1
fi

# P7 is up. Sync anything still staged from an offline period -- the sync
# script itself handles every guard (lock, running offline job, empty
# staging) and is a fast no-op when there's nothing to do.
if [ -d "$STAGING" ]; then
    "$REPO_DIR/scripts/dell_sync_to_p7.sh" || echo "WARNING: sync attempt failed -- will retry next tick (this is expected if P7 just flapped)."
fi
exit 0
