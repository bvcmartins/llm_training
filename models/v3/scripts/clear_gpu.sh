#!/bin/bash
# Pre-relaunch GPU sweep: kill any process holding the GPU just before the
# 22:00 training resume, so a leftover blackout inference server can't OOM it.
# Safe window — training is stopped (19:55) during the 20:00-22:00 blackout.
# Defensive guard: never kill the v3 trainer itself, even if timing ever drifts.
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    cmd=$(ps -p "$pid" -o cmd= 2>/dev/null)
    [[ "$cmd" == *run.py* ]] && continue   # never kill the v3 trainer
    echo "$(date '+%Y-%m-%d %H:%M:%S') clear_gpu: killing $pid ($cmd)"
    kill -9 "$pid"
done
