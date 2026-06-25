#!/usr/bin/env bash
# Backstop graceful stop for the v3 trainer.
#
# The trainer normally self-stops at --stop-at 19:55 (checked at the top of each
# optimizer step). This script is the safety net: SIGTERM the run so it flushes
# a resumable checkpoint and exits, then SIGKILL only if it overruns the grace
# period. Run from cron at 20:00 to *guarantee* the GPU is free for blackout.
#
# A single 1.5B optimizer step can take a few minutes, and the trainer only acts
# on the stop request at the next step boundary, so the grace period is generous.
set -uo pipefail

PATTERN='run.py --stage'        # matches the v3 training process
GRACE=420                       # seconds to wait for checkpoint+exit (covers one
                                # in-flight step + ~9GB checkpoint write)

pids=$(pgrep -f "$PATTERN" || true)
if [ -z "$pids" ]; then
    echo "stop_training: no v3 run found ($(date '+%F %T'))"
    exit 0
fi

echo "stop_training: SIGTERM $pids ($(date '+%F %T'))"
kill -TERM $pids 2>/dev/null || true

# Wait up to GRACE seconds for graceful exit (checkpoint-and-quit).
for ((i = 0; i < GRACE; i++)); do
    pgrep -f "$PATTERN" >/dev/null || { echo "stop_training: exited cleanly after ${i}s"; exit 0; }
    sleep 1
done

# Overran the grace period — force kill to free the GPU. Worst case loses
# progress since the last periodic checkpoint (ckpt_every=500 steps).
leftover=$(pgrep -f "$PATTERN" || true)
if [ -n "$leftover" ]; then
    echo "stop_training: grace expired — SIGKILL $leftover"
    kill -KILL $leftover 2>/dev/null || true
fi
exit 0
