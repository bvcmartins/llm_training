#!/usr/bin/env bash
# v3 ANNEAL launcher (idempotent, cron-friendly) — the stage-2 counterpart to
# train_session.sh. Same overnight 22:00->09:00 + daytime GPU-free model, but its OWN lock
# and its OWN self-sequencing guards, so the pretrain->anneal handoff needs no
# manual cron surgery:
#
#   * waits for pretrain : no-op until qwen3_v3_pretrain_final.pt exists.
#   * seeds then continues: --init-from the pretrain final on the FIRST launch
#     (fresh optimizer for the new stage), then --auto-resume the newest anneal
#     step checkpoint on every launch after.
#   * stops when done    : no-op once qwen3_v3_anneal_final.pt exists.
#
# Safe to schedule alongside train_session.sh: while pretrain runs, this is a
# sub-second no-op (the waits-for-pretrain guard); once pretrain writes _final,
# train_session.sh self-idles (its completion guard) and this takes over.
set -uo pipefail

# --- config ---------------------------------------------------------------
ROOT=/home/bmartins/dev/llm_training
V3="$ROOT/models/v3"
PY="$ROOT/.venv/bin/python"
STAGE=anneal
MODEL=1.5b
STOP_AT=08:55
BLACKOUT_START=$((9 * 60))
BLACKOUT_END=$((22 * 60))
LOCK=/tmp/v3_anneal.lock
LOGDIR="$V3/logs"
CKPT="$V3/checkpoints"
MLFLOW_EXPERIMENT=dense-model-1.5b-v3  # own branding, not the base arch's name (matches
MLFLOW_RUN_ID_FILE="$CKPT/.mlflow_run_id_anneal"  # train_session.sh). Anneal is its own
MLFLOW_RUN_NAME=anneal         # run in the same experiment; persisted id file so daily
                               # sessions continue THIS run.
PRETRAIN_FINAL="$CKPT/qwen3_v3_pretrain_final.pt"
ANNEAL_FINAL="$CKPT/qwen3_v3_anneal_final.pt"
PRUNE="$V3/scripts/prune_checkpoints.sh"

# --- GPU-free-window guard ------------------------------------------------
now=$((10#$(date +%H) * 60 + 10#$(date +%M)))
if (( now >= BLACKOUT_START && now < BLACKOUT_END )); then
    exit 0
fi

# --- sequencing guards ----------------------------------------------------
# Pretrain not finished yet -> nothing to anneal (this is the "wait" state that
# makes it safe to enable this launcher at any time, even during pretrain).
[ -f "$PRETRAIN_FINAL" ] || exit 0
# Anneal already finished -> done, stop relaunching.
[ -f "$ANNEAL_FINAL" ] && exit 0

# --- single instance ------------------------------------------------------
exec 9>"$LOCK" || exit 1
if ! flock -n 9; then
    exit 0
fi

mkdir -p "$LOGDIR"
LOG="$LOGDIR/anneal_session_$(date +%Y%m%d_%H%M%S).log"

[ -x "$PRUNE" ] && "$PRUNE" >>"$LOG" 2>&1

# First launch seeds a NEW anneal stage from the pretrain final (fresh
# optimizer, LR schedule reconstructed to continue from pretrain's end LR);
# later launches continue the anneal's own step checkpoints. --init-from and
# --auto-resume are mutually exclusive in run.py, so pick exactly one.
if ls "$CKPT"/qwen3_v3_anneal_step*.pt >/dev/null 2>&1; then
    SEED=(--auto-resume)
else
    SEED=(--init-from "$PRETRAIN_FINAL")
fi

{
    echo "=== anneal_session $(date '+%F %T') stage=$STAGE model=$MODEL stop_at=$STOP_AT seed=${SEED[*]} ==="
} >>"$LOG" 2>&1

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
    "$PY" "$V3/src/run.py" \
        --stage "$STAGE" --model "$MODEL" \
        "${SEED[@]}" --stop-at "$STOP_AT" \
        --mlflow-experiment "$MLFLOW_EXPERIMENT" \
        --mlflow-run-id-file "$MLFLOW_RUN_ID_FILE" --mlflow-run-name "$MLFLOW_RUN_NAME" \
        >>"$LOG" 2>&1

rc=$?
echo "=== run.py exited rc=$rc at $(date '+%F %T') ===" >>"$LOG" 2>&1
[ -x "$PRUNE" ] && "$PRUNE" >>"$LOG" 2>&1
exit 0
