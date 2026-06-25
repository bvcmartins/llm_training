#!/usr/bin/env bash
# Stage 1 — PRETRAIN. Runs src/pretrain.py via uv. Forwards all args.
#
# Examples:
#   ./train.sh                                  # defaults (20k steps, wandb on)
#   ./train.sh --max-steps 5000 --no-wandb
#   ./train.sh --resume ../checkpoints/qwen3_v2_pretrain_step005000.pt
#
# Annealing is a separate script: ./anneal.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
V2_DIR="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "$V2_DIR/../.." &>/dev/null && pwd)"

LOG_DIR="$V2_DIR/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
CONSOLE_LOG="$LOG_DIR/pretrain_${TS}.console.log"

# Unbuffered so the live log + tee stay in lockstep.
export PYTHONUNBUFFERED=1

echo "=== PRETRAIN ==="
echo "project root : $PROJECT_ROOT"
echo "entrypoint   : $V2_DIR/src/pretrain.py"
echo "args         : $*"
echo "console log  : $CONSOLE_LOG"
echo "(python writes a structured log under $LOG_DIR/pretrain_*.log)"
echo "================"

cd "$PROJECT_ROOT"
# tee captures stdout+stderr (incl. tracebacks) alongside the structured log.
exec uv run python "$V2_DIR/src/pretrain.py" "$@" 2>&1 | tee "$CONSOLE_LOG"
