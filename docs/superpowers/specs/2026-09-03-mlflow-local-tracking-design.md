# Local experiment tracking with MLflow (replacing wandb)

## Motivation

`llm_training` currently logs to Weights & Biases (wandb.ai), a third-party
cloud service. Aside from the free-tier quota (5GB — repeatedly exceeded by
other, unrelated projects sharing the account), the user wants training
telemetry on their own infra, consistent with how every other long-running
service in this environment is hosted (k3s: `ragu`, `janis-postgres`,
`daily_challenge`; podman: `quest_portfolio`, `personal_assistant`).

[MLflow](https://mlflow.org/) is the chosen replacement: a mature,
self-hostable, open-source ML experiment tracker with a REST-based tracking
server and a pure-Python client (no native-extension version constraints).

**Tool-choice history**: [Aim](https://github.com/aimhubio/aim) was the
original pick during brainstorming — self-hostable, real client-server
remote-tracking mode, good Metrics Explorer UI. It was dropped after
verification during plan-writing: `aim` 3.29.1's PyPI wheels (and its hard
native dependency `aimrocks`) top out at Python 3.12 — no 3.13 build exists,
and building `aimrocks` from source failed (needs a C++/Rust RocksDB
toolchain). `llm_training`'s `pyproject.toml` pins `requires-python =
">=3.13"` and the live `.venv` is 3.13.5 — the same venv `bootstrap.py`/
`training.py` run in — so `aim` cannot be installed there today. MLflow
3.15.2 was verified to install cleanly under Python 3.13 in a scratch venv
before committing to this design.

## Current state (what's being replaced)

- `models/v3/src/bootstrap.py::init_wandb()` — `wandb.init(project=, name=,
  config=...)`.
- `models/v3/src/training.py` — per-step `wandb_run.log(metrics, step=)`
  (train loss/ppl/lr/grad_norm/tok_per_sec/eta/per-source tokens),
  per-eval `wandb_run.log(val_log)` (val + val_ppl per domain),
  `wandb_run.alert()` on plateau detection, and periodic `wandb.Html` text
  samples.
- `models/v3/src/run.py` — `--wandb-project` (default varies by call site),
  `--wandb-id` / `--wandb-name` for continuous-run resume via
  `wandb.init(id=, resume="allow")`.
- `models/v3/scripts/train_session.sh` and `anneal_session.sh` — the live
  cron-driven launchers (every 20 min, overnight window). Both currently set
  `WANDB_PROJECT=dense-model-1.5b-v3`, `WANDB_ID=pretrain`/`anneal`,
  `WANDB_NAME=pretrain`/`anneal` — this is the **current, actively-training**
  project. It is a single continuously-resumed wandb run per stage (fixed id
  + `resume="allow"`), not a new run per day.
- `llm-training-v3` (the wandb project with 38 historical runs, one per
  cron-tick-era day, steps 0–19000) is the **legacy/prior** approach, before
  the switch to the single continuously-resumed run above. Nothing writes to
  it anymore. It exists today only as read-only history to optionally import.
- `llm-training-v2` and `dense-model-1.5b-v3`'s original wandb run are
  already gone (both projects emptied during this session's earlier wandb
  cleanup work; the latter's deletion was a mistake that lost its wandb
  metrics history but not any training state — see below).

## Architecture

Two independent pieces:

1. An **MLflow tracking server** running on k3s — PVC + Deployment +
   LoadBalancer Service, mirroring the existing `ragu`/`qdrant` pattern in
   this environment.
2. **Client-side changes** in `llm_training` (`bootstrap.py`, `training.py`,
   `run.py`, and the two cron launcher scripts) that swap
   `wandb.init`/`wandb.log`/`wandb.alert` for MLflow's `start_run`/
   `log_metrics`/`log_params`.

Training itself is unaffected — it keeps running as a laptop-side cron job
exactly as today, reading/writing local checkpoints exactly as today. Only
the telemetry destination changes.

## Components

### k3s side (new `deploy/k8s/` in `llm_training`)

- **`mlflow-storage`** — PersistentVolumeClaim, `local-path` storage class,
  `ReadWriteOnce`, 20Gi. (Metrics-only — no checkpoints or media go through
  MLflow — so even decades of per-step scalar history plus a SQLite backend
  file is a rounding error against this.)
- **`mlflow-server`** — Deployment, 1 replica, `Recreate` strategy (single
  RWO volume, same reasoning as `qdrant`'s manifest — and SQLite itself
  doesn't tolerate multiple concurrent writer processes, so single-pod is
  required regardless of the volume mode). Official
  `ghcr.io/mlflow/mlflow:v3.15.2` image (pin exact tag; verify it exists at
  deploy time — `ghcr.io/mlflow/mlflow` tags don't always mirror PyPI
  version numbers 1:1), one container:
  ```
  mlflow server \
    --backend-store-uri sqlite:////mlflow-data/mlflow.db \
    --default-artifact-root /mlflow-data/artifacts \
    --host 0.0.0.0 --port 5000
  ```
  Single port serves both the REST API (used by the training client) and
  the web UI (used by the user's browser) — simpler than a two-process,
  two-port setup.
- **`mlflow-server`** Service — `type: LoadBalancer` (k3s ServiceLB/klipper)
  exposing port `5000`. Reachable at `airig.local:5000` for both client and
  browser — same externally-reachable-by-hostname pattern as `ragu`'s
  `:8600`. Verified free (nothing else in this environment or on this host
  currently binds 5000).

### Client side (`models/v3/src/`)

- `bootstrap.py::init_wandb()` → `init_mlflow()`:
  ```python
  import mlflow
  mlflow.set_tracking_uri("http://airig.local:5000")
  mlflow.set_experiment(experiment_name)
  run = mlflow.start_run(run_id=existing_run_id, run_name=name)
  # existing_run_id=None on first-ever launch
  mlflow.log_params(flatten(config))  # config is currently a nested dict;
                                       # MLflow params are flat key->str
  ```
- **Continuous-run resume**: `mlflow.start_run(run_id=<existing id>)`
  continuing an existing run is the equivalent of wandb's fixed-id +
  `resume="allow"`. Since MLflow generates the run_id rather than letting
  the caller choose it (same situation as Aim's `run_hash` would have been),
  `init_mlflow()` persists it after first creation to a small local file
  next to the checkpoints — `checkpoints/.mlflow_run_id_<stage>` — and reads
  it back on every subsequent cron-triggered launch. This file is local
  laptop state, not tracked in git (add to `.gitignore`).
- **Metric logging**: `wandb_run.log(metrics, step=step)` → `mlflow.
  log_metrics(metrics, step=step, run_id=run.info.run_id)` directly — this
  is a genuine bulk call in the MLflow API, so call sites keep the exact
  same flat-dict shape they have today. No restructuring into per-metric
  calls needed (unlike the Aim design's `track()` helper).
- **Grouping**: MLflow has no per-metric context/grouping construct.
  Grouping instead comes for free from two things already true of the
  design: (1) pretrain vs. anneal are already separate runs (one per
  stage), so "group by stage" needs nothing extra; (2) the existing key
  *prefixing* the wandb code already does — `val/{domain}`,
  `val_ppl/{domain}`, `src_tokens/{source}` — carries over unchanged and is
  exactly what lets the MLflow UI's metric list/comparison view slice by
  domain/source. No renaming of metric keys is needed at all.
- **Plateau alert**: `wandb_run.alert(...)` call in `training.py`'s
  `_on_plateau` is deleted outright. The existing `log.warning(...)` in the
  same function is left as-is and is now the only signal — no replacement
  notification channel (tool-agnostic decision, unchanged from the original
  design).
- **Text samples**: the periodic `wandb.Html` sample block in `training.py`
  is deleted outright (not tracked). The existing `log.info(...)` right
  above it already prints the sample text — left as-is (tool-agnostic
  decision, unchanged).
- **Run end**: `mlflow.end_run()` at the end of the training script (success
  or failure path), mirroring wandb's implicit finish-on-exit — MLflow
  doesn't auto-close a run the way wandb's process-exit hook did.
- **CLI flags** (`run.py`): `--wandb-project` → `--mlflow-experiment`,
  `--wandb-id` → `--mlflow-run-id-file` (path to the persisted-id file,
  defaulting under `--ckpt-dir`), `--no-wandb` → `--no-mlflow`.
  `--wandb-name` maps directly to MLflow's `run_name`.
- **Cron launchers** (`train_session.sh`, `anneal_session.sh`): update the
  `WANDB_*` env vars to `MLFLOW_EXPERIMENT=dense-model-1.5b-v3` (branding
  carries over unchanged) and the `run.py` invocation flags accordingly.
  This is the cutover point for the *live* training — after this change,
  the next cron tick starts sending telemetry to the local MLflow server
  instead of wandb.ai.
- **Dependencies**: `pyproject.toml` — add `mlflow`. `wandb` stays as a dep
  only if the one-off import script (below) needs it; otherwise it can be
  dropped once that script has been run.

### One-time historical import (not part of the ongoing pipeline)

- `models/v3/scripts/import_wandb_history.py` — standalone, run once,
  disposable afterward.
- For each of `llm-training-v3`'s 38 runs: `run.scan_history()` via the
  wandb API, replayed into a new MLflow run (experiment
  `"llm-training-v3"`) via `mlflow.log_metrics(..., step=<original step>)`,
  preserving original step numbers and tagging the run
  (`mlflow.set_tag("source", "wandb-import")`) so imported data is visibly
  distinct from anything logged live.
- This only covers `llm-training-v3` (the legacy project). Nothing else
  remains in wandb worth importing — `dense-model-1.5b-v3`'s original run
  metrics were unrecoverably deleted earlier this session (see Current
  state); the live training's MLflow history simply starts fresh from the
  cutover point forward.

## Error handling

This matters more for a self-hosted server than it did for wandb.ai:
training runs continuously for months via cron, and this same environment
has a known failure mode (from prior nano-bank work) where a host reboot
resets `kube-proxy`/CoreDNS until manually restarted — meaning the MLflow
server could become briefly unreachable in ways the cloud service never was.

- Every `init_mlflow()` and `log_metrics()`/`log_params()` call site is
  wrapped in try/except; a network failure logs a warning (existing `log`
  object) and the function returns/no-ops rather than propagating. This
  mirrors the existing `if wandb_run is not None:` guard shape — `mlflow`
  calls can fail (tracking disabled or unreachable) and every call site
  already tolerates that via the same guard pattern, keyed off whether
  `init_mlflow()` itself succeeded.
- Local checkpoints (the thing that actually matters for the training run
  itself) are entirely independent of MLflow connectivity and never at
  risk.
- No retry/reconnect logic beyond this — if MLflow is down for a session,
  that session's telemetry is simply missing; training is unaffected and
  the next session tries again.

## Testing

1. Deploy `mlflow-server`; confirm the PVC binds and the UI/API is reachable
   (`curl http://airig.local:5000` and a trivial `mlflow.set_tracking_uri`
   + `start_run` + `log_metric` from a Python shell).
2. Point a short (few-step, `--stage pretrain --model 1.5b` truncated run)
   local invocation at it; confirm the run and its metrics appear in the UI
   under the right experiment, with `val/`/`src_tokens/` keys intact.
3. Kill/restart the `mlflow-server` pod mid-run; confirm the training
   process logs a warning and keeps training rather than crashing.
4. Two-launch resume test: run briefly, stop, relaunch with the same
   persisted-id file; confirm it's the *same* MLflow run continuing
   (matches today's wandb continuous-run behavior).
5. Cut over `train_session.sh`/`anneal_session.sh` for real; confirm the
   next cron tick logs to MLflow successfully.
6. Run `import_wandb_history.py` against `llm-training-v3`; spot-check a
   couple of runs' curves against what's still visible on wandb.ai before
   that project is deleted.

## Out of scope

- `models/v2` is not touched — it's finished/legacy work, not actively
  training.
- `models/v3/src/pretrain.py` is not touched either — it's a stray/older
  entrypoint (its own `--wandb-project` default is `llm-training-v2`) not
  referenced by `run.py` or either cron launcher script. Not part of the
  live pipeline.
- No auth/TLS on the MLflow server — this environment's other self-hosted
  services (`ragu`, `janis`) are similarly unauthenticated on the local
  network, and this matches that existing risk posture.
- No artifact/model-registry use of MLflow — checkpoints stay local-disk
  only, exactly as today. `--default-artifact-root` is configured only
  because MLflow's server requires one; it's expected to stay effectively
  empty.
