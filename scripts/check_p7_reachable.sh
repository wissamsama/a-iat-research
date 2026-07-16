#!/bin/bash
# Bounded-timeout check for whether the P7 (NFS server) is currently
# reachable. Never hangs, even if the NFS mount is in a bad state -- this
# is the whole point: a plain `stat`/`ls` on a dead NFS mount can block for
# a long time depending on mount options, so every caller must go through
# this wrapper instead of touching the mount directly to decide which mode
# to run in.
#
# Exit 0 = reachable (safe to use the real NFS-shared paths).
# Exit 1 = unreachable (use scripts/run_dell_offline.sh's local-staging
#          convention instead).
#
# Usage: scripts/check_p7_reachable.sh && echo up || echo down
timeout 10 stat /home/wissam/utem-workspace/experiments/FloodCastBench/coordination/status.md >/dev/null 2>&1
