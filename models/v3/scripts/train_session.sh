#!/usr/bin/env bash
# v3 scheduled-training launcher (idempotent, cron-friendly).
#
# Run every ~20 min from cron. Behaviour:
#   * single instance      — flock -n; a 2nd tick during a live run is a no-op.
#   * GPU-free guard        — local time in [09:00, 21:00) -> exit 0 (no training
#                             during the user's daytime GPU window, 9am-9pm).
#   * completion guard      — once this stage's _final.pt exists, stop relaunching.
#   * crash auto-restart    — if no run holds the lock, start/resume one.
#   * graceful pre-window   — --stop-at 08:55 makes the trainer checkpoint and
#                             exit on its own ~5 min before the daytime window.
#   * --auto-resume         — continue the newest qwen3_v3_<stage>_step*.pt, or
#                             start fresh if none exist.
#
# Net state machine: train 21:00 -> 08:55 (~12h overnight), graceful checkpoint,
# GPU free 09:00-21:00, auto-resume at 21:00, auto-restart within ~20 min of any crash.
#
# Stage-2 anneal has its OWN launcher (anneal_session.sh); this script only ever
# runs pretrain. When pretrain writes its _final.pt the completion guard below
# turns this into a no-op, and anneal_session.sh takes over (it waits for that
# same _final.pt), so there is no manual STAGE edit at the handoff.
set -uo pipefail

# --- config ---------------------------------------------------------------
ROOT=/home/bmartins/dev/llm_training
V3="$ROOT/models/v3"
PY="$ROOT/.venv/bin/python"
STAGE=pretrain                 # this launcher is pretrain-only (anneal_session.sh does anneal)
MODEL=1.5b
STOP_AT=08:55                  # graceful self-stop, ~5 min before the daytime window
EVAL_EVERY=25                  # matches training.py's ckpt_every=25 so every saved
                               # checkpoint has a matching val point (denser MLflow curve)
MLFLOW_EXPERIMENT=dense-model-1.5b-v3  # own branding, not the base arch's name
MLFLOW_RUN_ID_FILE="$V3/checkpoints/.mlflow_run_id_pretrain"  # persisted id file so
MLFLOW_RUN_NAME=pretrain       # every daily session continues THIS run. Prior qwen-named
                               # run (steps 0-19000) is left in the legacy llm-training-v3
                               # wandb project — see docs/superpowers/specs/2026-09-03-
                               # mlflow-local-tracking-design.md.
BLACKOUT_START=$((9 * 60))     # 09:00 in minutes-since-midnight (daytime GPU-free window start)
BLACKOUT_END=$((21 * 60))      # 21:00 (window end; training resumes)
LOCK=/tmp/v3_train.lock
LOGDIR="$V3/logs"
PRUNE="$V3/scripts/prune_checkpoints.sh"

# --- GPU-free-window guard ------------------------------------------------
# Done before taking the lock: during the daytime window this is a pure no-op
# so the cron tick costs nothing and never blocks a (non-existent) run. The
# free window [09:00, 21:00) does not wrap midnight, so the overnight training
# window 21:00 -> 09:00 is simply "not in [09:00, 21:00)".
now=$((10#$(date +%H) * 60 + 10#$(date +%M)))
if (( now >= BLACKOUT_START && now < BLACKOUT_END )); then
    exit 0
fi

# --- completion guard -----------------------------------------------------
# Once this stage has written its _final.pt, the stage is DONE. Stop relaunching
# so the cron doesn't reload + re-evaluate the finished model every 20 min, and
# so you never have to manually disable this line. The pretrain->anneal handoff
# is picked up by anneal_session.sh, which waits for this same _final.pt.
if [ -f "$V3/checkpoints/qwen3_v3_${STAGE}_final.pt" ]; then
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
        --eval-every "$EVAL_EVERY" \
        --mlflow-experiment "$MLFLOW_EXPERIMENT" \
        --mlflow-run-id-file "$MLFLOW_RUN_ID_FILE" --mlflow-run-name "$MLFLOW_RUN_NAME" \
        >>"$LOG" 2>&1

rc=$?
echo "=== run.py exited rc=$rc at $(date '+%F %T') ===" >>"$LOG" 2>&1

# Prune again after the run so a session that wrote several checkpoints doesn't
# leave them all on disk until the next hourly tick.
[ -x "$PRUNE" ] && "$PRUNE" >>"$LOG" 2>&1
exit 0
