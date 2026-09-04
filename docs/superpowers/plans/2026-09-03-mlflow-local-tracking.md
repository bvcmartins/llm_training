# MLflow Local Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace wandb.ai with a self-hosted MLflow tracking server (k3s) for `llm_training`'s v3 pretrain/anneal pipeline, with zero disruption to the currently-live overnight cron training.

**Architecture:** An MLflow tracking server (PVC + Deployment + LoadBalancer Service) on the existing `k3s` cluster, mirroring the `ragu`/`qdrant` pattern, reachable at `airig.local:5000`. `run.py`'s inline wandb block becomes a call into a new `mlflow_utils.py` helper module; `training.py`'s four wandb call sites swap to a `log_metrics_safe()` helper from the same module. Every tracking call is wrapped so a network failure degrades to a logged warning, never a crashed training run.

**Tech Stack:** MLflow 3.15.2 (self-hosted tracking server, `ghcr.io/mlflow/mlflow` image; pure-Python client, no native-extension version constraints), k3s (existing cluster, `k3s` kubectl context), SQLite (MLflow backend store), Python 3.13 (`llm_training`'s existing `.venv`).

**Spec:** `docs/superpowers/specs/2026-09-03-mlflow-local-tracking-design.md`

## Global Constraints

- k3s context is `k3s`, NOT the current default (`kind-nano-bank`) — every `kubectl`/`k3s`-targeting command in this plan MUST pass `--context k3s` explicitly.
- MLflow server port is `5000` (verified free, both on this host and across the `k3s` cluster's existing Services) — do not reuse or reassign it.
- `mlflow` client: pin `>=3.15,<4` in `pyproject.toml` (verified installs cleanly under Python 3.13; do not attempt `aim` — its native `aimrocks` dependency has no Python 3.13 wheels, confirmed in the spec).
- `mlflow.log_metrics()` requires all values to be numeric (`int`/`float`) — the existing metrics dicts include a `"stage"` string key inherited from the wandb code; every call site must filter it out (`log_metrics_safe()` does this once, centrally).
- `models/v2/`, `models/v3/src/pretrain.py`, `models/v3/src/anneal.py`, and `models/v3/src/bootstrap.py::init_wandb()` are OUT OF SCOPE — none are on the live cron path. Do not touch them.
- The live cron path is `run.py`, invoked only by `train_session.sh` and `anneal_session.sh` — verify this with `grep` before editing if anything here seems inconsistent with what you find on disk (training config gets hand-tuned between sessions; line numbers below are a snapshot, not gospel).
- Every `mlflow.*` call in `run.py`/`training.py`/`mlflow_utils.py` must be wrapped so a failure logs a warning via the module's existing `log` object and continues — never let a tracking failure propagate and kill a training run.

---

### Task 1: MLflow server on k3s

**Files:**
- Create: `deploy/k8s/namespace.yaml`
- Create: `deploy/k8s/mlflow.yaml`

**Interfaces:**
- Produces: a reachable MLflow tracking server at `http://airig.local:5000`, used by every later task's client code and by the user's browser for the UI.

- [ ] **Step 1: Write the namespace manifest**

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: mlflow
```

- [ ] **Step 2: Write the PVC + Deployment + Service manifest**

```yaml
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mlflow-storage
  namespace: mlflow
spec:
  accessModes: ["ReadWriteOnce"]
  storageClassName: local-path        # k3s ships local-path-provisioner as default
  resources:
    requests:
      storage: 20Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-server
  namespace: mlflow
  labels: { app: mlflow-server }
spec:
  replicas: 1
  strategy: { type: Recreate }        # single RWO volume + SQLite — never run two pods at once
  selector:
    matchLabels: { app: mlflow-server }
  template:
    metadata:
      labels: { app: mlflow-server }
    spec:
      containers:
        - name: mlflow-server
          image: ghcr.io/mlflow/mlflow:v3.15.2
          command:
            - mlflow
            - server
            - --backend-store-uri
            - sqlite:////mlflow-data/mlflow.db
            - --default-artifact-root
            - /mlflow-data/artifacts
            - --host
            - "0.0.0.0"
            - --port
            - "5000"
          ports:
            - { name: http, containerPort: 5000 }
          volumeMounts:
            - { name: storage, mountPath: /mlflow-data }
          readinessProbe:
            httpGet: { path: /health, port: 5000 }
            initialDelaySeconds: 10
            periodSeconds: 10
          livenessProbe:
            httpGet: { path: /health, port: 5000 }
            initialDelaySeconds: 20
            periodSeconds: 30
          resources:
            requests: { cpu: "100m", memory: "256Mi" }
            limits:   { memory: "1Gi" }
      volumes:
        - name: storage
          persistentVolumeClaim:
            claimName: mlflow-storage
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow-server
  namespace: mlflow
spec:
  # LoadBalancer => k3s ServiceLB (klipper) binds host port 5000, so the server is
  # reachable at http://airig.local:5000 — same pattern as ragu's :8600.
  type: LoadBalancer
  selector: { app: mlflow-server }
  ports:
    - { name: http, port: 5000, targetPort: 5000 }
```

- [ ] **Step 3: Verify the image tag exists before applying**

```bash
docker manifest inspect ghcr.io/mlflow/mlflow:v3.15.2 >/dev/null && echo "tag exists"
```

Expected: `tag exists`. If this fails, check https://github.com/mlflow/mlflow/pkgs/container/mlflow for the closest available tag to `3.15.2` and use that instead (update the manifest before continuing).

- [ ] **Step 4: Apply the manifests**

```bash
kubectl --context k3s apply -f deploy/k8s/namespace.yaml
kubectl --context k3s apply -f deploy/k8s/mlflow.yaml
```

- [ ] **Step 5: Wait for the pod and PVC to be ready**

```bash
kubectl --context k3s -n mlflow get pvc
kubectl --context k3s -n mlflow rollout status deployment/mlflow-server --timeout=120s
```

Expected: PVC `STATUS` is `Bound`; rollout reports `successfully rolled out`.

- [ ] **Step 6: Verify the server is reachable**

```bash
curl -sf http://airig.local:5000/health && echo " <- OK"
curl -sf http://airig.local:5000/ | head -c 200
```

Expected: first command prints `OK` (or similar) followed by ` <- OK`; second returns HTML (the MLflow UI shell), not a connection error.

- [ ] **Step 7: Commit**

```bash
cd ~/dev/llm_training
git add deploy/k8s/namespace.yaml deploy/k8s/mlflow.yaml
git commit -m "Add k3s manifests for a self-hosted MLflow tracking server"
```

---

### Task 2: Add the mlflow client dependency

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `mlflow` importable from `.venv/bin/python` — required by Task 3.

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, in the `dependencies` list, add a line after `"wandb>=0.18,"`:

```toml
    "wandb>=0.18",
    "mlflow>=3.15,<4",
```

(`wandb` stays — it's still needed by `import_wandb_history.py` in Task 7, and by the out-of-scope `bootstrap.py::init_wandb()` / `pretrain.py` / `anneal.py`.)

- [ ] **Step 2: Install and verify**

```bash
cd ~/dev/llm_training
uv sync   # or: .venv/bin/pip install "mlflow>=3.15,<4"
.venv/bin/python -c "import mlflow; print(mlflow.__version__)"
```

Expected: prints a version `>=3.15,<4` with no import error.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add mlflow client dependency"
```

---

### Task 3: mlflow_utils.py helper module

**Files:**
- Create: `models/v3/src/mlflow_utils.py`
- Test: manual verification script (this codebase has no existing pytest/test infra — see note below)

**Interfaces:**
- Consumes: the MLflow server from Task 1 (`http://airig.local:5000`), the `mlflow` package from Task 2.
- Produces (used by Tasks 4 and 5):
  - `TRACKING_URI: str` — `"http://airig.local:5000"`
  - `flatten_params(cfg: dict) -> dict` — one level of dict-flattening, `{"a": {"b": 1}} -> {"a.b": 1}`
  - `init_mlflow(enabled: bool, experiment: str, run_name: str, run_id_file: Path, stage: str, config: dict, tags: dict | None = None) -> bool` — starts/resumes a run, returns whether tracking is active
  - `log_metrics_safe(metrics: dict, step: int) -> None` — logs numeric values only, swallows failures
  - `end_run_safe(status: str = "FINISHED") -> None` — ends the active run, swallows failures

Note on testing: this repo has no `pytest`/test directory anywhere (`models/v2`, `models/v3` are both verified by running real training sessions and reading logs, per the project's existing convention — see e.g. the spec's own Testing section). This task follows that convention: verification is a real script run against the real server from Task 1, not a new test framework introduced for three functions.

- [ ] **Step 1: Write `mlflow_utils.py`**

```python
"""Local MLflow tracking helpers for the v3 trainer.

Every call here is best-effort: a network hiccup against the self-hosted
MLflow server (this laptop's k3s cluster) must never crash a training run
that's been going for months. See docs/superpowers/specs/2026-09-03-mlflow-
local-tracking-design.md for why, and for the wandb -> mlflow mapping this
module implements.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("v2.mlflow")

TRACKING_URI = "http://airig.local:5000"


def flatten_params(cfg: dict) -> dict:
    """One level of dict-flattening: {"a": {"b": 1}} -> {"a.b": 1}.

    mlflow.log_params wants flat key->value; the existing config dict nests
    model_config and mix one level deep and nothing deeper.
    """
    flat = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}.{kk}"] = vv
        else:
            flat[k] = v
    return flat


def _load_run_id(path: Path) -> str | None:
    return path.read_text().strip() if path.exists() else None


def _save_run_id(path: Path, run_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run_id)


def init_mlflow(
    enabled: bool,
    experiment: str,
    run_name: str,
    run_id_file: Path,
    stage: str,
    config: dict,
    tags: dict | None = None,
) -> bool:
    """Start (or resume) an MLflow run. Returns whether tracking is active.

    Continuous-run resume: mlflow generates its own run_id, so the id is
    persisted to run_id_file on first creation and read back on every
    later launch — the same trick wandb's fixed id + resume="allow" did.

    Never raises: any failure (server unreachable, etc.) logs a warning
    and returns False so the caller trains with tracking simply off.
    """
    if not enabled:
        log.info("mlflow disabled (--no-mlflow)")
        return False
    try:
        import mlflow

        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(experiment)
        existing_run_id = _load_run_id(run_id_file)
        run = mlflow.start_run(run_id=existing_run_id, run_name=run_name)
        if existing_run_id is None:
            _save_run_id(run_id_file, run.info.run_id)
        mlflow.set_tag("stage", stage)
        for k, v in (tags or {}).items():
            mlflow.set_tag(k, v)
        mlflow.log_params(flatten_params(config))
        log.info("mlflow run: experiment=%s name=%s id=%s", experiment, run_name, run.info.run_id)
        return True
    except Exception:
        log.warning("mlflow init failed — continuing without tracking", exc_info=True)
        return False


def log_metrics_safe(metrics: dict, step: int) -> None:
    """Log numeric metrics for the active run.

    Drops non-numeric values (e.g. a "stage" string carried over from the
    wandb-shaped metrics dicts) since mlflow.log_metrics requires floats.
    Never raises.
    """
    numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    if not numeric:
        return
    try:
        import mlflow

        mlflow.log_metrics(numeric, step=step)
    except Exception:
        log.warning("mlflow log_metrics failed", exc_info=True)


def end_run_safe(status: str = "FINISHED") -> None:
    """End the active run. Never raises."""
    try:
        import mlflow

        mlflow.end_run(status=status)
    except Exception:
        log.warning("mlflow end_run failed", exc_info=True)
```

- [ ] **Step 2: Verify `flatten_params` (pure logic, no server needed)**

```bash
cd ~/dev/llm_training
.venv/bin/python -c "
import sys; sys.path.insert(0, 'models/v3/src')
from mlflow_utils import flatten_params
result = flatten_params({'stage': 'pretrain', 'model_config': {'n_layers': 28, 'emb_dim': 1792}, 'mix': {'web': 0.7, 'math': 0.3}})
assert result == {'stage': 'pretrain', 'model_config.n_layers': 28, 'model_config.emb_dim': 1792, 'mix.web': 0.7, 'mix.math': 0.3}, result
print('flatten_params OK:', result)
"
```

Expected: prints `flatten_params OK: {...}` with no assertion error.

- [ ] **Step 3: Verify `init_mlflow` / `log_metrics_safe` / `end_run_safe` against the real server**

```bash
cd ~/dev/llm_training
rm -f /tmp/test_mlflow_run_id
.venv/bin/python -c "
import sys; sys.path.insert(0, 'models/v3/src')
from pathlib import Path
from mlflow_utils import init_mlflow, log_metrics_safe, end_run_safe

run_id_file = Path('/tmp/test_mlflow_run_id')
ok = init_mlflow(True, 'smoke-test', 'smoke-run', run_id_file, 'pretrain', {'stage': 'pretrain', 'model_config': {'n_layers': 1}})
assert ok, 'init_mlflow should succeed against a live server'
assert run_id_file.exists(), 'run id should be persisted on first launch'
log_metrics_safe({'loss': 1.23, 'stage': 'pretrain'}, step=1)  # 'stage' must be silently dropped
end_run_safe()
print('mlflow smoke test OK, run_id=', run_id_file.read_text())
"
```

Expected: prints `mlflow smoke test OK, run_id=<32-char hex>` with no exception. Confirm the run also shows up at `http://airig.local:5000` under experiment `smoke-test`, run `smoke-run`, with a single `loss` metric point (not a `stage` metric — that would mean the numeric filter didn't work).

- [ ] **Step 4: Verify graceful degradation when the server is unreachable**

MLflow's HTTP client defaults to a 120s timeout with retries, which would make
this hang far past a bite-sized verification step. Cap both via env vars so
it fails fast instead:

```bash
MLFLOW_HTTP_REQUEST_TIMEOUT=3 MLFLOW_HTTP_REQUEST_MAX_RETRIES=1 .venv/bin/python -c "
import sys; sys.path.insert(0, 'models/v3/src')
import mlflow_utils
mlflow_utils.TRACKING_URI = 'http://127.0.0.1:1'  # nothing listens here
from pathlib import Path
ok = mlflow_utils.init_mlflow(True, 'x', 'x', Path('/tmp/test_mlflow_run_id_2'), 'pretrain', {})
assert ok is False, 'init_mlflow must return False, not raise, when the server is unreachable'
mlflow_utils.log_metrics_safe({'loss': 1.0}, step=1)  # must also not raise
mlflow_utils.end_run_safe()  # must also not raise
print('graceful degradation OK')
"
```

Expected: prints `graceful degradation OK` — no traceback. Should complete within ~10-15s with the capped timeout/retries above.

- [ ] **Step 5: Commit**

```bash
git add models/v3/src/mlflow_utils.py
git commit -m "Add mlflow_utils: wandb->mlflow tracking helper for v3 training"
```

---

### Task 4: Wire mlflow into run.py

**Files:**
- Modify: `models/v3/src/run.py`

**Interfaces:**
- Consumes: `init_mlflow`, `end_run_safe` from `mlflow_utils` (Task 3).
- Produces: `train_stage(..., mlflow_enabled: bool)` call — Task 5 must accept this parameter name.

- [ ] **Step 1: Replace the wandb CLI flags**

Find (around line 63-70):

```python
    p.add_argument("--no-wandb",   action="store_true")
    p.add_argument("--wandb-project", default="llm-training-v3")
    p.add_argument("--wandb-name",    default=None)
    p.add_argument("--wandb-id",      default=None,
                   help="stable W&B run id so every resume continues ONE run "
                        "(default: qwen3_<model>_<stage>). Keeps the dashboard a "
                        "single continuous curve per stage instead of a new run "
                        "per session. Set --wandb-id fresh-<x> to start a new run.")
```

Replace with:

```python
    p.add_argument("--no-mlflow",   action="store_true")
    p.add_argument("--mlflow-experiment", default="dense-model-1.5b-v3")
    p.add_argument("--mlflow-run-name",   default=None)
    p.add_argument("--mlflow-run-id-file", type=Path, default=None,
                   help="path to a file holding the persisted MLflow run id, so "
                        "every resume continues ONE run (default: "
                        "<ckpt-dir>/.mlflow_run_id_<stage>). Keeps the dashboard a "
                        "single continuous curve per stage instead of a new run "
                        "per session. Delete the file to start a fresh run.")
```

- [ ] **Step 2: Import mlflow_utils**

Find (around line 107):

```python
    from training import train_stage, load_resume_state, stage_cfg_from_dict
    # Stage-config factories were moved out of training.py (the engine) and now
    # live with their stage entrypoints.
    from pretrain import default_pretrain_config
    from anneal import default_anneal_config
```

Add a line after it:

```python
    from training import train_stage, load_resume_state, stage_cfg_from_dict
    # Stage-config factories were moved out of training.py (the engine) and now
    # live with their stage entrypoints.
    from pretrain import default_pretrain_config
    from anneal import default_anneal_config
    from mlflow_utils import init_mlflow, end_run_safe
```

- [ ] **Step 3: Replace the wandb init block**

Find (around line 224-248):

```python
    # wandb
    wandb_run = None
    if not args.no_wandb:
        import wandb
        # ONE continuous run per (model, stage): a deterministic id + resume="allow"
        # means every daily session and every --resume appends to the SAME run, so
        # pretrain is a single unbroken dashboard curve (and anneal is its own,
        # separate run). Sanitise the id — W&B ids allow [A-Za-z0-9_-] (no dots).
        default_id = f"qwen3_{args.model}_{args.stage}".replace(".", "_")
        run_id = args.wandb_id or default_id
        wandb_run = wandb.init(
            project=args.wandb_project,
            id=run_id,
            name=args.wandb_name or default_id,
            resume="allow",         # resume run_id if it exists, else create it
            group=f"qwen3_{args.model}_{args.stage}",  # groups any legacy per-session runs too
            config={
                "model_config": model_cfg,
                "stage":        args.stage,
                "mix":          stage_cfg.mix.weights,
                "resume":       str(args.resume) if args.resume else None,
                "init_from":    str(args.init_from) if args.init_from else None,
                **{k: v for k, v in stage_cfg.__dict__.items() if k != "mix"},
            },
        )
```

Replace with:

```python
    # mlflow — ONE continuous run per (model, stage): a persisted run_id file
    # means every daily session and every --resume continues the SAME run, so
    # pretrain is a single unbroken dashboard curve (and anneal is its own,
    # separate run). See mlflow_utils.init_mlflow / docs/superpowers/specs/
    # 2026-09-03-mlflow-local-tracking-design.md.
    default_name = f"qwen3_{args.model}_{args.stage}".replace(".", "_")
    run_id_file = args.mlflow_run_id_file or (args.ckpt_dir / f".mlflow_run_id_{args.stage}")
    mlflow_enabled = init_mlflow(
        enabled=not args.no_mlflow,
        experiment=args.mlflow_experiment,
        run_name=args.mlflow_run_name or default_name,
        run_id_file=run_id_file,
        stage=args.stage,
        config={
            "model_config": model_cfg,
            "stage":        args.stage,
            "mix":          stage_cfg.mix.weights,
            "resume":       str(args.resume) if args.resume else None,
            "init_from":    str(args.init_from) if args.init_from else None,
            **{k: v for k, v in stage_cfg.__dict__.items() if k != "mix"},
        },
        tags={"group": f"qwen3_{args.model}_{args.stage}"},
    )
```

- [ ] **Step 4: Update the train_stage call**

Find (around line 253-261):

```python
        summary = train_stage(
            model=model,
            cfg=stage_cfg,
            device=device,
            ckpt_dir=args.ckpt_dir,
            wandb_run=wandb_run,
            resume_from=args.resume,
        )
```

Replace with:

```python
        summary = train_stage(
            model=model,
            cfg=stage_cfg,
            device=device,
            ckpt_dir=args.ckpt_dir,
            mlflow_enabled=mlflow_enabled,
            resume_from=args.resume,
        )
```

- [ ] **Step 5: Update the crash-path finish call**

Find (around line 269-279, inside the `except Exception:` block):

```python
        import traceback
        traceback.print_exc()
        if wandb_run is not None:
            try:
                wandb_run.finish(exit_code=1)
            except Exception:
                pass
        sys.stdout.flush()
```

Replace with:

```python
        import traceback
        traceback.print_exc()
        if mlflow_enabled:
            end_run_safe(status="FAILED")
        sys.stdout.flush()
```

- [ ] **Step 6: Update the success-path finish call**

Find (around line 281-282):

```python
    if wandb_run is not None:
        wandb_run.finish()
```

Replace with:

```python
    if mlflow_enabled:
        end_run_safe()
```

- [ ] **Step 7: Verify the file parses and --help works**

```bash
cd ~/dev/llm_training
.venv/bin/python -c "import ast; ast.parse(open('models/v3/src/run.py').read())" && echo "syntax OK"
.venv/bin/python models/v3/src/run.py --help
```

Expected: `syntax OK`, then `--help` output showing `--mlflow-experiment`, `--mlflow-run-name`, `--mlflow-run-id-file`, `--no-mlflow` and NOT any `--wandb-*` flags.

- [ ] **Step 8: Commit**

```bash
git add models/v3/src/run.py
git commit -m "run.py: switch wandb tracking to mlflow"
```

---

### Task 5: Wire mlflow into training.py

**Files:**
- Modify: `models/v3/src/training.py`

**Interfaces:**
- Consumes: `log_metrics_safe` from `mlflow_utils` (Task 3); `mlflow_enabled: bool` parameter from Task 4's `train_stage()` call.
- Produces: `train_stage(..., mlflow_enabled: bool = False, ...)` — matches what Task 4 already calls.

- [ ] **Step 1: Remove the wandb import**

Find (line 26):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
```

Replace with:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

- [ ] **Step 2: Add the mlflow_utils import**

Find (around line 33-35):

```python
from eval import evaluate_per_domain, PlateauDetector
from logging_utils import gpu_mem_str, log_config
from tokenizer import encode, decode, eot_id
```

Replace with:

```python
from eval import evaluate_per_domain, PlateauDetector
from logging_utils import gpu_mem_str, log_config
from mlflow_utils import log_metrics_safe
from tokenizer import encode, decode, eot_id
```

- [ ] **Step 3: Rename the train_stage parameter**

Find (around line 297-302, the `def train_stage(` signature):

```python
    wandb_run=None,
```

Replace with:

```python
    mlflow_enabled: bool = False,
```

- [ ] **Step 4: Drop the plateau alert (log.warning stays)**

Find (around line 364-372):

```python
    def _on_plateau(payload):
        log.warning("[plateau:%s] best=%.4f current=%.4f — val stopped improving",
                    cfg.name, payload["best"], payload["current"])
        if wandb_run is not None:
            wandb_run.alert(
                title=f"Plateau in {cfg.name}",
                text=f"best={payload['best']:.4f} current={payload['current']:.4f}",
            )
```

Replace with:

```python
    def _on_plateau(payload):
        log.warning("[plateau:%s] best=%.4f current=%.4f — val stopped improving",
                    cfg.name, payload["best"], payload["current"])
```

- [ ] **Step 5: Swap the per-step train-metrics log call**

Find (around line 489-490):

```python
        if wandb_run is not None:
            wandb_run.log(metrics, step=step)
```

Replace with:

```python
        if mlflow_enabled:
            log_metrics_safe(metrics, step=step)
```

- [ ] **Step 6: Swap the per-eval val-metrics log call**

Find (around line 507-511):

```python
            if wandb_run is not None:
                val_log = {f"val/{k}": v for k, v in val.items()}
                val_log |= {f"val_ppl/{k}": _safe_ppl(v) for k, v in val.items()}
                val_log["stage"] = cfg.name
                wandb_run.log(val_log, step=step)
```

Replace with:

```python
            if mlflow_enabled:
                val_log = {f"val/{k}": v for k, v in val.items()}
                val_log |= {f"val_ppl/{k}": _safe_ppl(v) for k, v in val.items()}
                log_metrics_safe(val_log, step=step)
```

- [ ] **Step 7: Drop the periodic text-sample tracking (log.info stays)**

Find (around line 519-534):

```python
        # Periodic inference sample (qualitative progress check)
        if cfg.sample_every and local % cfg.sample_every == 0:
            text = generate_sample_text(
                model, cfg.sample_prompt, device,
                context_length=cfg.context_length,
                max_new_tokens=cfg.sample_max_new,
                temperature=cfg.sample_temperature,
                top_k=cfg.sample_top_k,
            )
            pretty = text.replace("<|endoftext|>", " ⏎ ")
            log.info(f"[{cfg.name} sample gstep={step}] {pretty}")
            if wandb_run is not None:
                wandb_run.log(
                    {"train/sample": wandb.Html(f"<pre>{pretty}</pre>")},
                    step=step,
                )
```

Replace with:

```python
        # Periodic inference sample (qualitative progress check)
        if cfg.sample_every and local % cfg.sample_every == 0:
            text = generate_sample_text(
                model, cfg.sample_prompt, device,
                context_length=cfg.context_length,
                max_new_tokens=cfg.sample_max_new,
                temperature=cfg.sample_temperature,
                top_k=cfg.sample_top_k,
            )
            pretty = text.replace("<|endoftext|>", " ⏎ ")
            log.info(f"[{cfg.name} sample gstep={step}] {pretty}")
```

- [ ] **Step 8: Swap the final-eval metrics log call**

Find (around line 586-590):

```python
    if wandb_run is not None:
        final_log = {f"val/{k}": v for k, v in val.items()}
        final_log |= {f"val_ppl/{k}": _safe_ppl(v) for k, v in val.items()}
        final_log["stage"] = cfg.name
        wandb_run.log(final_log, step=step)
```

Replace with:

```python
    if mlflow_enabled:
        final_log = {f"val/{k}": v for k, v in val.items()}
        final_log |= {f"val_ppl/{k}": _safe_ppl(v) for k, v in val.items()}
        log_metrics_safe(final_log, step=step)
```

- [ ] **Step 9: Verify no wandb references remain and the file parses**

```bash
cd ~/dev/llm_training
grep -n "wandb" models/v3/src/training.py
```

Expected: no output (nothing left).

```bash
.venv/bin/python -c "import ast; ast.parse(open('models/v3/src/training.py').read())" && echo "syntax OK"
```

Expected: `syntax OK`.

- [ ] **Step 10: End-to-end smoke test — short real training run against the live server**

**SAFETY: this MUST use an isolated `--ckpt-dir`, never `models/v3/checkpoints`.**
That directory is the live, currently-running pretrain job's checkpoint store.
`run.py` always writes a `<model_tag>_<stage>_final.pt` at the end of *any*
run regardless of step count, and `train_session.sh`'s completion guard
(`if [ -f ".../qwen3_v3_pretrain_final.pt" ]; then exit 0`) would treat that
as "pretrain is done" and permanently stop the real training. A smoke-test
checkpoint could also get picked up by the live cron's `--auto-resume` (newest
step file) — an 0.6b smoke checkpoint is architecturally incompatible with the
real 1.5b run. `--ckpt-dir` isolation makes both impossible regardless of
naming. Also only run this when no cron tick is currently mid-training (check
`flock -n /tmp/v3_train.lock -c true` succeeds — if it fails, a live session
holds the GPU; wait or run during the 09:00-21:00 daytime window when
`train_session.sh` self-exits without training):

```bash
cd ~/dev/llm_training
flock -n /tmp/v3_train.lock -c true && echo "GPU free, safe to smoke-test" || echo "WAIT: a live session is training right now"
```

Only proceed once that prints `GPU free, safe to smoke-test`.

```bash
mkdir -p /tmp/mlflow-smoketest-ckpt
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python models/v3/src/run.py \
    --stage pretrain --model 0.6b --max-steps 10 --eval-every 5 --ckpt-every 5 \
    --ckpt-dir /tmp/mlflow-smoketest-ckpt \
    --mlflow-experiment smoke-test-e2e --mlflow-run-name smoke-e2e \
    --mlflow-run-id-file /tmp/mlflow-smoketest-ckpt/.mlflow_run_id_pretrain_smoketest
```

Expected: exits 0, prints `mlflow run: experiment=smoke-test-e2e name=smoke-e2e id=<hex>` and a `stage summary:` block. Then confirm at `http://airig.local:5000`: experiment `smoke-test-e2e` has one run named `smoke-e2e` with `loss`/`ppl`/`lr`/... metrics logged across ~10 steps and a `val/aggregate` point.

- [ ] **Step 11: Resume-continuity smoke test**

Same GPU-free check as Step 10 first. Then run the exact same command again, still pointed at the isolated `--ckpt-dir`:

```bash
flock -n /tmp/v3_train.lock -c true && echo "GPU free, safe to smoke-test" || echo "WAIT: a live session is training right now"
```

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python models/v3/src/run.py \
    --stage pretrain --model 0.6b --max-steps 10 --eval-every 5 --ckpt-every 5 --auto-resume \
    --ckpt-dir /tmp/mlflow-smoketest-ckpt \
    --mlflow-experiment smoke-test-e2e --mlflow-run-name smoke-e2e \
    --mlflow-run-id-file /tmp/mlflow-smoketest-ckpt/.mlflow_run_id_pretrain_smoketest
```

Expected: `mlflow run:` line prints the SAME run id as Step 10. In the UI, the run `smoke-e2e` shows metric points continuing past step 10 (not a second run created).

- [ ] **Step 12: Clean up smoke-test artifacts**

```bash
rm -rf /tmp/mlflow-smoketest-ckpt
```

(This is an isolated `/tmp` directory used only by Steps 10-11 — nothing under `models/v3/checkpoints` was ever touched, so there's no risk in this cleanup.)

- [ ] **Step 13: Commit**

```bash
git add models/v3/src/training.py
git commit -m "training.py: switch wandb metric logging to mlflow"
```

---

### Task 6: Cut over the live cron scripts

**Files:**
- Modify: `models/v3/scripts/train_session.sh`
- Modify: `models/v3/scripts/anneal_session.sh`

**Interfaces:**
- Consumes: the `--mlflow-*` flags from Task 4.
- Produces: the actual telemetry cutover point — after this task, the next cron tick sends training data to MLflow instead of wandb.

This is the point where the LIVE, currently-running overnight training switches destinations. Do this task only once Tasks 1-5 are verified working (the smoke tests in Task 5 must have passed).

- [ ] **Step 1: Update train_session.sh's config block**

Find:

```bash
WANDB_PROJECT=dense-model-1.5b-v3  # own branding, not the base arch's name
WANDB_ID=pretrain              # fresh restart 2026-07-17 (step-0 rerun). Stable id so
WANDB_NAME=pretrain            # every daily session continues THIS run. Prior qwen-named
                               # run (steps 0-19000) is left intact in its old project.
```

Replace with:

```bash
MLFLOW_EXPERIMENT=dense-model-1.5b-v3  # own branding, not the base arch's name
MLFLOW_RUN_ID_FILE="$V3/checkpoints/.mlflow_run_id_pretrain"  # persisted id file so
MLFLOW_RUN_NAME=pretrain       # every daily session continues THIS run. Prior qwen-named
                               # run (steps 0-19000) is left in the legacy llm-training-v3
                               # wandb project — see docs/superpowers/specs/2026-09-03-
                               # mlflow-local-tracking-design.md.
```

- [ ] **Step 2: Update train_session.sh's run.py invocation**

Find:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
    "$PY" "$V3/src/run.py" \
        --stage "$STAGE" --model "$MODEL" \
        --auto-resume --stop-at "$STOP_AT" \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-id "$WANDB_ID" --wandb-name "$WANDB_NAME" \
        >>"$LOG" 2>&1
```

Replace with:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
    "$PY" "$V3/src/run.py" \
        --stage "$STAGE" --model "$MODEL" \
        --auto-resume --stop-at "$STOP_AT" \
        --mlflow-experiment "$MLFLOW_EXPERIMENT" \
        --mlflow-run-id-file "$MLFLOW_RUN_ID_FILE" --mlflow-run-name "$MLFLOW_RUN_NAME" \
        >>"$LOG" 2>&1
```

- [ ] **Step 3: Apply the same two edits to anneal_session.sh**

Find:

```bash
WANDB_PROJECT=dense-model-1.5b-v3  # own branding, not the base arch's name (matches
WANDB_ID=anneal                # train_session.sh). Anneal is its own run in the same
WANDB_NAME=anneal              # project; stable id so daily sessions continue THIS run.
```

Replace with:

```bash
MLFLOW_EXPERIMENT=dense-model-1.5b-v3  # own branding, not the base arch's name (matches
MLFLOW_RUN_ID_FILE="$CKPT/.mlflow_run_id_anneal"  # train_session.sh). Anneal is its own
MLFLOW_RUN_NAME=anneal         # run in the same experiment; persisted id file so daily
                               # sessions continue THIS run.
```

Find:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
    "$PY" "$V3/src/run.py" \
        --stage "$STAGE" --model "$MODEL" \
        "${SEED[@]}" --stop-at "$STOP_AT" \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-id "$WANDB_ID" --wandb-name "$WANDB_NAME" \
        >>"$LOG" 2>&1
```

Replace with:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
    "$PY" "$V3/src/run.py" \
        --stage "$STAGE" --model "$MODEL" \
        "${SEED[@]}" --stop-at "$STOP_AT" \
        --mlflow-experiment "$MLFLOW_EXPERIMENT" \
        --mlflow-run-id-file "$MLFLOW_RUN_ID_FILE" --mlflow-run-name "$MLFLOW_RUN_NAME" \
        >>"$LOG" 2>&1
```

(`anneal_session.sh` already defines `CKPT="$V3/checkpoints"` near the top of the file — that's why this uses `$CKPT` instead of `$V3/checkpoints` inline like `train_session.sh` does.)

- [ ] **Step 4: Verify both scripts still pass a syntax check**

```bash
cd ~/dev/llm_training
bash -n models/v3/scripts/train_session.sh && echo "train_session.sh OK"
bash -n models/v3/scripts/anneal_session.sh && echo "anneal_session.sh OK"
```

Expected: both print OK.

- [ ] **Step 5: Commit**

```bash
git add models/v3/scripts/train_session.sh models/v3/scripts/anneal_session.sh
git commit -m "Cut over live training cron scripts from wandb to mlflow"
```

- [ ] **Step 6: Confirm the next real cron tick works**

Cron runs `train_session.sh` every 20 minutes. Within ~20 minutes of the commit, check:

```bash
tail -30 ~/dev/llm_training/models/v3/logs/session_*.log | tail -30
```

Expected: the newest session log shows `mlflow run: experiment=dense-model-1.5b-v3 name=pretrain id=<hex>` (not a wandb line), and training proceeds normally (loss/step lines). Also confirm at `http://airig.local:5000` that experiment `dense-model-1.5b-v3` now has a run named `pretrain` receiving live metrics at the current step (~020958+ as of this plan's writing).

---

### Task 7: One-time historical import from wandb

**Files:**
- Create: `models/v3/scripts/import_wandb_history.py`

**Interfaces:**
- Consumes: the wandb API (`wandb.Api()`, already authenticated via `~/.netrc` per this session's earlier work), the MLflow server from Task 1.
- Produces: nothing consumed by later tasks — this is the last task, standalone and disposable.

This only needs to run once, against `llm-training-v3` (38 runs), before that wandb project might get deleted (the user mentioned doing this manually in an earlier conversation — check whether it still exists before running this).

- [ ] **Step 1: Check the source project still exists**

```bash
cd ~/dev/llm_training
.venv/bin/python -c "
import wandb
api = wandb.Api()
runs = list(api.runs('bmartins/llm-training-v3'))
print(f'{len(runs)} runs found')
"
```

Expected: prints a run count > 0. If it prints 0 or errors because the project is gone, stop here — there's nothing left to import, this task is moot, tell the user.

- [ ] **Step 2: Write the import script**

```python
#!/usr/bin/env python3
"""One-time import of llm-training-v3's wandb history into MLflow.

Standalone and disposable — not part of the ongoing training pipeline.
Run once: .venv/bin/python models/v3/scripts/import_wandb_history.py
See docs/superpowers/specs/2026-09-03-mlflow-local-tracking-design.md.
"""

from __future__ import annotations

import mlflow
import wandb

TRACKING_URI = "http://airig.local:5000"
WANDB_ENTITY = "bmartins"
WANDB_PROJECT = "llm-training-v3"
MLFLOW_EXPERIMENT = "llm-training-v3"


def import_run(api_run) -> None:
    mlflow.start_run(run_name=api_run.name)
    try:
        mlflow.set_tag("source", "wandb-import")
        mlflow.set_tag("wandb_run_id", api_run.id)
        mlflow.set_tag("wandb_state", api_run.state)
        n_points = 0
        for row in api_run.scan_history():
            step = int(row.get("_step") or 0)
            numeric = {
                k: v for k, v in row.items()
                if isinstance(v, (int, float)) and not k.startswith("_")
            }
            if numeric:
                mlflow.log_metrics(numeric, step=step)
                n_points += 1
        mlflow.end_run()
        print(f"  imported {n_points} history rows")
    except Exception:
        mlflow.end_run(status="FAILED")
        raise


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    api = wandb.Api()
    runs = list(api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}"))
    print(f"importing {len(runs)} runs from {WANDB_ENTITY}/{WANDB_PROJECT} "
          f"-> mlflow experiment {MLFLOW_EXPERIMENT!r}")
    for i, api_run in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {api_run.id} ({api_run.name}, state={api_run.state})")
        import_run(api_run)
    print("done")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it**

```bash
cd ~/dev/llm_training
.venv/bin/python models/v3/scripts/import_wandb_history.py
```

Expected: prints one `[i/38] ...` line per run with an `imported N history rows` line after each, ending in `done`. No unhandled traceback.

- [ ] **Step 4: Spot-check against wandb**

Pick 2-3 imported runs at random. Compare their final `loss`/`val/aggregate` values between `http://airig.local:5000` (experiment `llm-training-v3`) and https://wandb.ai (project `bmartins/llm-training-v3`, if it's still there). They should match (same step, same value, modulo float printing).

- [ ] **Step 5: Commit**

```bash
git add models/v3/scripts/import_wandb_history.py
git commit -m "Add one-time wandb->mlflow history import script for llm-training-v3"
```
