#!/bin/bash
# Bounded-timeout check for whether the P7 (NFS server) is currently
# reachable. Never hangs, even if the NFS mount is in a bad state -- this
# is the whole point: a plain `stat`/`ls` on a dead NFS mount can block for
# a long time depending on mount options, so every caller must go through
# this wrapper instead of touching the mount directly to decide which mode
# to run in.
#
# Probes the coordination DIRECTORY (not a specific file, which could get
# renamed) inside the share. This is also what makes the unmounted case
# safe: if the NFS mount dropped, the Dell's local underlying mountpoint
# (`experiments/`) exists but is EMPTY -- the coordination dir inside it
# does not exist locally -- so the probe correctly reports "down" instead
# of being fooled by the empty local shadow directory.
#
# Known blind spot (documented, accepted): NFS attribute caching can make
# stat succeed from cache for up to ~60s after the server actually dies.
# Worst case: one launch decision made in that window targets NFS paths and
# that job then fails visibly with soft-mount I/O errors (the fstab options
# soft,timeo=30 make it an error, not a silent hang) -- annoying, not
# silent data loss.
#
# Exit 0 = reachable (safe to use the real NFS-shared paths).
# Exit 1 = unreachable (use scripts/run_dell_offline.sh's local-staging
#          convention instead).
#
# Usage: scripts/check_p7_reachable.sh && echo up || echo down
timeout 10 stat /home/wissam/utem-workspace/experiments/FloodCastBench/coordination >/dev/null 2>&1
