# Publishing an espopip milestone to the Hub

How to push a real milestone (e.g. **end of pretraining**) to `bvcmartins/espopip`,
record the git pointer, retire the throwaway `step-7686` pipeline-test upload, and
reclaim its disk space on the Hub.

See also: `scripts/push_milestone.sh`, `models/v3/MANIFEST.json`.

## Mental model

- Every push moves the repo's `main` branch forward — `main` always = latest weights.
- **Tags** are immutable pins (`step-7686`, `pretrain-final`, …). They don't move; you
  delete the ones you no longer want.
- Git keeps only `MANIFEST.json` (a ~400-byte pointer per milestone), never the weights.
- The `.export_meta.json` sidecar (written by `export_hf.py`) lets `push_milestone.sh`
  auto-detect `--step`; it is excluded from the upload.

## Prereqs (one-time)

```bash
hf auth whoami        # should print  user=bvcmartins  (else: hf auth login, paste a WRITE token)
```

## 1. Export the final pretrain checkpoint

Point `--ckpt` at the final pretrain checkpoint and `--src` at the v3 sources (1.5B config).
Runs on CPU only — safe to do mid-training.

```bash
uv run python models/v2/notebooks/export_hf.py \
    --ckpt models/v3/checkpoints/qwen3_v3_pretrain_step050000.pt \
    --out  models/v3/exports/espopip_hf \
    --src  models/v3/src \
    --name espopip
```

This writes the HF folder + model card + `.export_meta.json` (carrying the step).

## 2. Push the milestone

`--step` is auto-detected from the sidecar. Add a `--note` describing the milestone.

```bash
bash scripts/push_milestone.sh \
    --export models/v3/exports/espopip_hf \
    --note "end of pretraining"
```

This uploads to `bvcmartins/espopip` (private), tags `step-050000`, and appends an
entry to `models/v3/MANIFEST.json`.

## 3. Commit the git pointer

```bash
git add models/v3/MANIFEST.json
git commit -m "milestone: espopip end-of-pretraining (step-050000)"
```

## 4. Retire the throwaway step-7686

The first push (`step-7686`) was a pipeline test, not a real milestone. Drop its tag:

```bash
hf repos tag delete bvcmartins/espopip step-7686
```

Then remove its entry from `models/v3/MANIFEST.json` (delete the `step-7686` object from
the JSON array) and commit:

```bash
git add models/v3/MANIFEST.json
git commit -m "manifest: drop throwaway step-7686 test upload"
```

## 5. Reclaim the space on the Hub

Deleting the tag removes the *pin*, but the old ~3GB blob still lives in the repo's
git/LFS history. Collapse history to a single commit to drop all superseded blobs:

```bash
# Uses the stored write token automatically. IRREVERSIBLE: flattens all history to one commit.
uvx --from huggingface_hub python - <<'PY'
from huggingface_hub import HfApi
HfApi().super_squash_history(repo_id="bvcmartins/espopip", repo_type="model")
print("history squashed — superseded blobs reclaimed")
PY
```

> ⚠️ `super_squash_history` is irreversible: it replaces the entire commit history with a
> single commit holding the current files. Existing **tags still resolve** (they're
> re-pointed), but per-commit history before the squash is gone. Do this only after you're
> happy with the current `main` contents.

## 6. Verify

```bash
uvx --from huggingface_hub python - <<'PY'
from huggingface_hub import list_repo_files, list_repo_refs, HfApi
repo = "bvcmartins/espopip"
print("files:", sorted(list_repo_files(repo)))
print("tags :", [t.name for t in list_repo_refs(repo).tags])
info = HfApi().model_info(repo, files_metadata=True)
for s in info.siblings:
    if s.rfilename.endswith(".safetensors"):
        print("weights:", s.rfilename, s.size, "bytes")
PY
```

Expected: `main` holds the final pretrain weights, tags list `step-050000` (no
`step-7686`), and the safetensors size matches the new `MANIFEST.json` entry.

Load anywhere with:

```python
from transformers import AutoModelForCausalLM
AutoModelForCausalLM.from_pretrained("bvcmartins/espopip", revision="step-050000")
```
