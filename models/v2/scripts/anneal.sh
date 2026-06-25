#!/usr/bin/env bash
# Stage 2 — ANNEAL. Runs src/anneal.py via uv. Forwards all args.
#
# Requires exactly one of --init-from / --resume:
#   ./anneal.sh --init-from ../checkpoints/qwen3_v2_pretrain_final.pt
#   ./anneal.sh --resume    ../checkpoints/qwen3_v2_anneal_step001000.pt
#
# Common overrides:
#   ./anneal.sh --init-from ../checkpoints/qwen3_v2_pretrain_final.pt --max-steps 2000 --no-wandb
#
# Pretraining is a separate script: ./train.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
V2_DIR="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "$V2_DIR/../.." &>/dev/null && pwd)"

LOG_DIR="$V2_DIR/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
CONSOLE_LOG="$LOG_DIR/anneal_${TS}.console.log"

export PYTHONUNBUFFERED=1

# Force line buffering so logs stream to the terminal/file as they happen
# instead of being held in libc's pipe block-buffer. `stdbuf -oL -eL` sets the
# whole child subtree (uv -> python -> torch/C extensions) to line-buffered;
# `python -u` unbuffers Python's own stdio; line-buffered `tee` flushes per line.
# stdbuf may be absent on some systems (it's coreutils) -- degrade gracefully.
if command -v stdbuf &>/dev/null; then
  STDBUF=(stdbuf -oL -eL)
else
  STDBUF=()
fi

echo "=== ANNEAL ==="
echo "project root : $PROJECT_ROOT"
echo "entrypoint   : $V2_DIR/src/anneal.py"
echo "args         : $*"
echo "console log  : $CONSOLE_LOG"
echo "(python writes a structured log under $LOG_DIR/anneal_*.log)"
echo "=============="

cd "$PROJECT_ROOT"
exec "${STDBUF[@]}" uv run python -u "$V2_DIR/src/anneal.py" "$@" 2>&1 \
  | "${STDBUF[@]}" tee "$CONSOLE_LOG"
