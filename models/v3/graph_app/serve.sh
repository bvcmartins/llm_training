#!/usr/bin/env bash
# Serve the v3 code-graph as a static site on the LAN.
#
#   ./serve.sh [port]      # default port 8503 (8501/8502 are quest/assistant)
#
# Reachable at  http://airig.local:<port>/  and  http://192.168.1.84:<port>/
# Regenerate the page first with:  python build_graph.py
set -euo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PORT="${1:-8503}"
PY=/home/bmartins/dev/llm_training/.venv/bin/python

[ -f "$DIR/index.html" ] || "$PY" "$DIR/build_graph.py"

echo "serving $DIR"
echo "  local : http://localhost:$PORT/"
echo "  LAN   : http://airig.local:$PORT/   http://192.168.1.84:$PORT/"
exec "$PY" -m http.server "$PORT" --bind 0.0.0.0 --directory "$DIR"
