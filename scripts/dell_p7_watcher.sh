#!/bin/bash
# Plain-bash background daemon for the Dell: polls P7 reachability every
# 60s and auto-syncs local staging the moment P7 comes back. This is the
# "instant detection" layer -- it costs zero LLM credits and reacts within
# ~1 minute, so the Claude /loop can stay on a lazy 20-30 min cadence and
# only do the judgment work (instructions, reports), never the plumbing.
#
# Design notes:
# - Single-instance lock (flock) so accidental double launches are no-ops.
# - Logs transitions + sync output to a LOCAL log file (not the NFS share,
#   which is exactly what may be down).
# - Sync itself is delegated to dell_sync_to_p7.sh, which carries all the
#   safety guards (its own lock, running-offline-job deferral,
#   --ignore-existing, archive-never-delete). Condition-based: a failed
#   sync is retried on the next poll automatically.
# - Survives session close (nohup) but NOT a reboot: relaunch it after any
#   Dell reboot (see PROTOCOL.md; a crontab @reboot entry can automate that
#   if desired).
#
# Usage: nohup scripts/dell_p7_watcher.sh > /dev/null 2>&1 & disown
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG=/home/wissam/utem-workspace/dell_p7_watcher.log
LOCK=/tmp/dell_p7_watcher.lock
STAGING=/home/wissam/utem-workspace/local_staging

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "watcher already running (lock held) -- exiting." >&2
    exit 0
fi

echo "=== watcher started at $(date -Is) (pid $$) ===" >> "$LOG"
prev="unknown"
while true; do
    if "$REPO_DIR/scripts/check_p7_reachable.sh"; then now="up"; else now="down"; fi
    if [ "$prev" != "$now" ]; then
        echo "P7 reachability: $prev -> $now at $(date -Is)" >> "$LOG"
    fi
    if [ "$now" = "up" ] && [ -d "$STAGING" ]; then
        "$REPO_DIR/scripts/dell_sync_to_p7.sh" >> "$LOG" 2>&1 || true
    fi
    prev="$now"
    sleep 60
done
