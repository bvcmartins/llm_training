#!/usr/bin/env bash
# Run EleutherAI lm-evaluation-harness against an exported v2 checkpoint.
#
# First export a checkpoint to HF format:
#   uv run python notebooks/export_hf.py \
#       --ckpt checkpoints/qwen3_v2_anneal_final.pt --out exports/qwen3_v2_anneal_hf
#
# Then benchmark it:
#   notebooks/run_lm_eval.sh                         # defaults below
#   notebooks/run_lm_eval.sh exports/qwen3_v2_pretrain_hf
#   MODEL_DIR=... TASKS=lambada_openai DEVICE=cpu LIMIT=20 notebooks/run_lm_eval.sh
#
# Env knobs:
#   MODEL_DIR  exported HF folder            (default exports/qwen3_v2_anneal_hf, or $1)
#   TASKS      comma-separated lm-eval tasks (default: base-model suite below)
#   DEVICE     cuda:0 | cpu                  (default cuda:0)
#   BATCH      batch size or "auto"          (default auto)
#   LIMIT      cap examples/task for a smoke run (default: unset = full)
#   DTYPE      bfloat16 | float16 | float32  (default bfloat16)
set -euo pipefail

V2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V2_DIR"

MODEL_DIR="${1:-${MODEL_DIR:-exports/qwen3_v2_anneal_hf}}"
# A base (non-instruct) model suite: perplexity + zero-shot cloze/MCQ.
TASKS="${TASKS:-lambada_openai,wikitext,hellaswag,winogrande,piqa,arc_easy,arc_challenge,openbookqa,sciq,boolq}"
DEVICE="${DEVICE:-cuda:0}"
BATCH="${BATCH:-auto}"
DTYPE="${DTYPE:-bfloat16}"
OUT_DIR="${OUT_DIR:-notebooks/bench_results}"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: $MODEL_DIR not found. Export a checkpoint first (see header)." >&2
  exit 1
fi

LIMIT_ARGS=()
[[ -n "${LIMIT:-}" ]] && LIMIT_ARGS=(--limit "$LIMIT")

mkdir -p "$OUT_DIR"
echo "model=$MODEL_DIR  device=$DEVICE  batch=$BATCH  dtype=$DTYPE"
echo "tasks=$TASKS"
[[ -n "${LIMIT:-}" ]] && echo "limit=$LIMIT (smoke run)"

exec uv run lm_eval \
  --model hf \
  --model_args "pretrained=$MODEL_DIR,dtype=$DTYPE,trust_remote_code=False" \
  --tasks "$TASKS" \
  --device "$DEVICE" \
  --batch_size "$BATCH" \
  --output_path "$OUT_DIR" \
  "${LIMIT_ARGS[@]}"
