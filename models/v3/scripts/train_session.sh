#!/usr/bin/env bash
# v3 scheduled-training launcher (idempotent, cron-friendly).
#
# Run every ~20 min from cron. Behaviour:
#   * single instance      — flock -n; a 2nd tick during a live run is a no-op.
#   * GPU blackout guard    — local time in [20:00, 22:00) -> exit 0 (no training
#                             during the user's daily 8-10pm GPU window).
#   * crash auto-restart    — if no run holds the lock, start/resume one.
#   * graceful pre-blackout — --stop-at 19:55 makes the trainer checkpoint and
#                             exit on its own ~5 min before blackout.
#   * --auto-resume         — continue the newest qwen3_v3_<stage>_step*.pt, or
#                             start fresh if none exist.
#
# Net state machine: train 22:00 -> 19:55 (~22h), graceful checkpoint, GPU free
# 20:00-22:00, auto-resume at 22:00, auto-restart within ~20 min of any crash.
#
# Switch stages by editing STAGE below (pretrain -> anneal) once pretrain's
# _final.pt is produced; the anneal seed is handed off manually (see crontab.txt).
set -uo pipefail

# --- config ---------------------------------------------------------------
ROOT=/home/bmartins/dev/llm_training
V3="$ROOT/models/v3"
PY="$ROOT/.venv/bin/python"
STAGE=pretrain                 # pretrain | anneal  (edit when handing off)
MODEL=1.5b
STOP_AT=19:55                  # graceful self-stop, ~5 min before blackout
BLACKOUT_START=$((20 * 60))    # 20:00 in minutes-since-midnight
BLACKOUT_END=$((22 * 60))      # 22:00
LOCK=/tmp/v3_train.lock
LOGDIR="$V3/logs"
PRUNE="$V3/scripts/prune_checkpoints.sh"

# --- blackout guard -------------------------------------------------------
# Done before taking the lock: during blackout this is a pure no-op so the
# cron tick costs nothing and never blocks a (non-existent) run.
now=$((10#$(date +%H) * 60 + 10#$(date +%M)))
if (( now >= BLACKOUT_START && now < BLACKOUT_END )); then
    exit 0
fi

# --- single instance ------------------------------------------------------
# Hold fd 9 for the lifetime of the script (== the training run). A concurrent
# tick fails flock -n and exits, leaving the running session untouched.
exec 9>"$LOCK" || exit 1
if ! flock -n 9; then
    exit 0
fi

mkdir -p "$LOGDIR"
LOG="$LOGDIR/session_$(date +%Y%m%d_%H%M%S).log"

# Prune once before launching (bounds disk at session start; an hourly cron
# prunes again *during* the long run, which this script can't because it is
# blocked holding the lock while training).
[ -x "$PRUNE" ] && "$PRUNE" >>"$LOG" 2>&1

{
    echo "=== train_session $(date '+%F %T') stage=$STAGE model=$MODEL stop_at=$STOP_AT ==="
} >>"$LOG" 2>&1

# Hard-exit note: on a clean finish the trainer calls os._exit(0) to dodge the
# cosmetic HF-streaming PyGILState_Release teardown crash, so a graceful stop
# still returns 0 here. We don't gate on the exit code — the next cron tick
# re-resumes regardless.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
    "$PY" "$V3/src/run.py" \
        --stage "$STAGE" --model "$MODEL" \
        --auto-resume --stop-at "$STOP_AT" \
        >>"$LOG" 2>&1

rc=$?
echo "=== run.py exited rc=$rc at $(date '+%F %T') ===" >>"$LOG" 2>&1

# Prune again after the run so a session that wrote several checkpoints doesn't
# leave them all on disk until the next hourly tick.
[ -x "$PRUNE" ] && "$PRUNE" >>"$LOG" 2>&1
exit 0
