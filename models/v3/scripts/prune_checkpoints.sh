#!/usr/bin/env bash
# Checkpoint retention for v3 (each ~9 GB, so this is mandatory, not cosmetic).
#
# Keeps, across all stages:
#   * every  *_final.pt                       (stage handoff artifacts)
#   * the 3 newest  *_step*.pt   (by mtime)   (latest resumable + a little slack)
#   * every milestone step where step % 5000 == 0   (long-term history)
# and deletes the rest. Safe to run anytime, including while a run is writing —
# it never touches the 3 newest or _final.
#
# Invoked by train_session.sh (session start/end) and hourly from cron (so the
# long mid-session run, during which the launcher is blocked on the lock, still
# gets pruned).
set -uo pipefail

CKPT_DIR=/home/bmartins/dev/llm_training/models/v3/checkpoints
KEEP_NEWEST=3
MILESTONE=5000

[ -d "$CKPT_DIR" ] || exit 0

# All step checkpoints, newest first by mtime.
mapfile -t step_ckpts < <(ls -1t "$CKPT_DIR"/qwen3_v3_*_step*.pt 2>/dev/null || true)
(( ${#step_ckpts[@]} == 0 )) && exit 0

deleted=0
idx=0
for f in "${step_ckpts[@]}"; do
    # Keep the KEEP_NEWEST most-recent unconditionally.
    if (( idx < KEEP_NEWEST )); then
        idx=$((idx + 1))
        continue
    fi
    idx=$((idx + 1))

    # Keep milestone steps (step number divisible by MILESTONE).
    base=${f##*/}                      # qwen3_v3_pretrain_step012345.pt
    num=${base##*_step}; num=${num%.pt}
    num=$((10#$num))                   # strip zero-padding
    if (( num % MILESTONE == 0 )); then
        continue
    fi

    rm -f -- "$f" && deleted=$((deleted + 1))
done

if (( deleted > 0 )); then
    echo "prune_checkpoints: removed $deleted old checkpoint(s) ($(date '+%F %T'))"
fi
exit 0
