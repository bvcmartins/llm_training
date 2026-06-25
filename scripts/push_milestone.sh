#!/usr/bin/env bash
# Push a milestone HF export to the Hugging Face Hub and record a git-tracked pointer.
#
# The weights themselves never enter git (exports/ is .gitignored). Instead this
# uploads a standard-HF export dir to a private Hub repo and appends a one-line
# entry to MANIFEST.json so the commit can always locate the exact weights again.
#
# Prereqs (one-time, run by you — login is interactive and needs your HF token):
#     uv tool install "huggingface_hub[cli]"      # provides the `hf` CLI
#     hf auth login                                # paste a write token from hf.co/settings/tokens
#
# Usage:
#     scripts/push_milestone.sh --export models/v3/exports/espopip_hf [options]
#
# Options:
#     --export DIR     Local HF export dir (config.json + model.safetensors + tokenizer). REQUIRED.
#     --repo  ID       Full Hub repo id. Default: <your-hf-username>/espopip (from `whoami`).
#     --tag   NAME     Immutable tag to pin this snapshot. Default: derived from --step or timestamp.
#     --step  N        Training step this export came from (for the manifest + default tag).
#     --note  TEXT     Free-text note recorded in the manifest (e.g. eval score).
#     --manifest FILE  Manifest to append to. Default: <export>/../../MANIFEST.json.
#
set -euo pipefail

MODEL_NAME="espopip"
HF_OWNER="bvcmartins"          # HF account; override the full id with --repo if this changes
EXPORT="" ; REPO="" ; TAG="" ; STEP="" ; NOTE="" ; MANIFEST=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --export)   EXPORT="$2"; shift 2 ;;
    --repo)     REPO="$2";   shift 2 ;;
    --tag)      TAG="$2";    shift 2 ;;
    --step)     STEP="$2";   shift 2 ;;
    --note)     NOTE="$2";   shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$EXPORT" ]] || { echo "error: --export DIR is required" >&2; exit 2; }
[[ -f "$EXPORT/model.safetensors" ]] || { echo "error: $EXPORT/model.safetensors not found — is this an HF export dir?" >&2; exit 2; }

# Auto-fill --step from the export provenance sidecar (written by export_hf.py) if not given.
if [[ -z "$STEP" && -f "$EXPORT/.export_meta.json" ]]; then
  STEP="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('step') or '')" "$EXPORT/.export_meta.json" 2>/dev/null || true)"
  [[ -n "$STEP" ]] && echo ">> step $STEP (auto-detected from .export_meta.json)"
fi

# Default repo id. (whoami parsing is avoided: `hf auth whoami` prints a pretty,
# multi-line form on a TTY that doesn't parse reliably.)
[[ -n "$REPO" ]] || REPO="$HF_OWNER/$MODEL_NAME"
hf auth whoami >/dev/null 2>&1 || {
  echo "error: not logged in. Run 'hf auth login' first." >&2; exit 1; }

# Default tag + manifest path.
if [[ -z "$TAG" ]]; then
  if [[ -n "$STEP" ]]; then TAG="step-$STEP"; else TAG="snap-$(date +%Y%m%d-%H%M%S)"; fi
fi
[[ -n "$MANIFEST" ]] || MANIFEST="$(cd "$EXPORT/../.." && pwd)/MANIFEST.json"

SHA="$(sha256sum "$EXPORT/model.safetensors" | cut -d' ' -f1)"
BYTES="$(stat -c%s "$EXPORT/model.safetensors")"
NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo ">> uploading $EXPORT  ->  $REPO  (private)"
hf upload "$REPO" "$EXPORT" . --type model --private \
  --exclude ".export_meta.json" \
  --commit-message "milestone: $MODEL_NAME ${STEP:+step $STEP} ($TAG)"

echo ">> tagging $REPO @ $TAG"
hf repos tag create "$REPO" "$TAG" --type model -m "$MODEL_NAME ${STEP:+step $STEP}" 2>/dev/null \
  || echo "   (tag '$TAG' may already exist — skipping)"

# Append a pointer record to MANIFEST.json (git-tracked).
echo ">> recording pointer in $MANIFEST"
python3 - "$MANIFEST" <<PY
import json, os, sys
path = sys.argv[1]
rec = {
    "model": "$MODEL_NAME",
    "repo": "$REPO",
    "revision": "$TAG",
    "step": ${STEP:-None} if "$STEP" else None,
    "export_dir": "$EXPORT",
    "safetensors_sha256": "$SHA",
    "safetensors_bytes": $BYTES,
    "uploaded_utc": "$NOW",
    "note": "$NOTE" or None,
}
data = []
if os.path.exists(path):
    with open(path) as f:
        data = json.load(f)
data.append(rec)
with open(path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
print(f"   appended -> {path}  ({len(data)} entries)")
PY

echo
echo "Done. Now commit the pointer (weights stay out of git):"
echo "    git add $MANIFEST && git commit -m 'milestone: $MODEL_NAME $TAG'"
echo "Load the weights anywhere with:"
echo "    AutoModelForCausalLM.from_pretrained('$REPO', revision='$TAG')"
