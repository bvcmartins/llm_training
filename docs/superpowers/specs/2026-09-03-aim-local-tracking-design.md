# Local experiment tracking with Aim (replacing wandb)

## Motivation

`llm_training` currently logs to Weights & Biases (wandb.ai), a third-party
cloud service. Aside from the free-tier quota (5GB — repeatedly exceeded by
other, unrelated projects sharing the account), the user wants training
telemetry on their own infra, consistent with how every other long-running
service in this environment is hosted (k3s: `ragu`, `janis-postgres`,
`daily_challenge`; podman: `quest_portfolio`, `personal_assistant`).

[Aim](https://github.com/aimhubio/aim) is the chosen replacement: a mature,
self-hostable, open-source ML experiment tracker with a real client-server
remote-tracking mode (not just a local-file tool), an official Docker image,
and a Metrics Explorer UI comparable to wandb's for this project's needs
(scalar metrics grouped/filtered by run and context).

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

1. An **Aim server** running on k3s — PVC + Deployment + LoadBalancer
   Service, mirroring the existing `ragu`/`qdrant` pattern in this
   environment.
2. **Client-side changes** in `llm_training` (`bootstrap.py`, `training.py`,
   `run.py`, and the two cron launcher scripts) that swap
   `wandb.init`/`wandb.log`/`wandb.alert` for Aim's `Run`/`track()`.

Training itself is unaffected — it keeps running as a laptop-side cron job
exactly as today, reading/writing local checkpoints exactly as today. Only
the telemetry destination changes.

## Components

### k3s side (new `deploy/k8s/` in `llm_training`)

- **`aim-storage`** — PersistentVolumeClaim, `local-path` storage class,
  `ReadWriteOnce`, 20Gi. (Metrics-only — no checkpoints or media go through
  Aim — so even decades of per-step scalar history is a rounding error
  against this.)
- **`aim-server`** — Deployment, 1 replica, `Recreate` strategy (single RWO
  volume, same reasoning as `qdrant`'s manifest). Official `aimstack/aim`
  image, two containers sharing the PVC:
  - `aim-server`: `aim server --repo /aim-repo` — tracking ingest, port
    `53800`.
  - `aim-ui`: `aim up --repo /aim-repo --host 0.0.0.0` — web UI, port
    `43800`.
- **`aim-server`** Service — `type: LoadBalancer` (k3s ServiceLB/klipper),
  exposing both ports. Reachable at `airig.local:53800` (tracking, used by
  the training client) and `airig.local:43800` (UI, used by the user's
  browser) — same externally-reachable-by-hostname pattern as `ragu`'s
  `:8600`.

### Client side (`models/v3/src/`)

- `bootstrap.py::init_wandb()` → `init_aim()`:
  ```python
  from aim import Run
  run = Run(repo="aim://airig.local:53800", experiment=experiment_name,
             run_hash=existing_hash)  # run_hash=None on first-ever launch
  ```
- **Continuous-run resume**: Aim's `Run(run_hash=...)` continuing an
  existing run is the equivalent of wandb's fixed-id + `resume="allow"`.
  Since Aim generates the hash rather than letting the caller choose it,
  `init_aim()` persists it after first creation to a small local file next
  to the checkpoints — `checkpoints/.aim_run_hash_<stage>` — and reads it
  back on every subsequent cron-triggered launch. This file is local
  laptop state, not tracked in git (add to `.gitignore`).
- **Metric logging**: a small helper wraps `track()` so call sites keep the
  same flat-dict shape they have today:
  ```python
  def track_all(run, metrics: dict, step: int, context: dict):
      for name, value in metrics.items():
          run.track(value, name=name, step=step, context=context)
  ```
  - Per-step train metrics: `context={"stage": cfg.name}`.
  - Per-source token counts: `context={"stage": cfg.name, "source": k}`
    (one `track()` call per source key, stripped of the `src_tokens/`
    prefix — the prefix becomes redundant once `source` is a context dim).
  - Per-eval val/val_ppl metrics: `context={"stage": cfg.name, "source":
    domain}` (same treatment, domain replaces the `val/`/`val_ppl/` prefix
    split into metric name `val`/`val_ppl` + `source` context).
  - This is what the Metrics Explorer groups/filters on — chosen so stage
    and per-source/domain comparisons are both easy without either being
    baked into the metric *name* (which would fragment the same logical
    series across differently-named lines).
- **Plateau alert**: `wandb_run.alert(...)` call in `training.py`'s
  `_on_plateau` is deleted outright. The existing `log.warning(...)` in the
  same function is left as-is and is now the only signal — no replacement
  notification channel.
- **Text samples**: the periodic `wandb.Html` sample block in `training.py`
  is deleted outright (not tracked in Aim). The existing `log.info(...)`
  right above it already prints the sample text — left as-is.
- **CLI flags** (`run.py`): `--wandb-project` → `--aim-experiment`,
  `--wandb-id` → `--aim-run-hash-file` (path to the persisted-hash file,
  defaulting under `--ckpt-dir`), `--no-wandb` → `--no-aim`. `--wandb-name`
  is dropped — Aim runs are identified by hash/experiment, not a separate
  display name field in the same way.
- **Cron launchers** (`train_session.sh`, `anneal_session.sh`): update the
  `WANDB_*` env vars to `AIM_EXPERIMENT=dense-model-1.5b-v3` (branding
  carries over unchanged) and the `run.py` invocation flags accordingly.
  This is the cutover point for the *live* training — after this change,
  the next cron tick starts sending telemetry to the local Aim server
  instead of wandb.ai.
- **Dependencies**: `pyproject.toml` — add `aim`. `wandb` stays as a dep
  only if the one-off import script (below) needs it; otherwise it can be
  dropped once that script has been run.

### One-time historical import (not part of the ongoing pipeline)

- `models/v3/scripts/import_wandb_history.py` — standalone, run once,
  disposable afterward.
- For each of `llm-training-v3`'s 38 runs: `run.scan_history()` via the
  wandb API, replayed into a new Aim run (`experiment="llm-training-v3"`)
  via the same `track()` helper, preserving original step numbers and
  tagging `context={"source": "wandb-import", "stage": ...}` so imported
  data is visibly distinct from anything logged live.
- This only covers `llm-training-v3` (the legacy project). Nothing else
  remains in wandb worth importing — `dense-model-1.5b-v3`'s original run
  metrics were unrecoverably deleted earlier this session (see Current
  state); the live training's Aim history simply starts fresh from the
  cutover point forward.

## Error handling

This matters more for Aim than it did for wandb.ai: training runs
continuously for months via cron, and this same environment has a known
failure mode (from prior nano-bank work) where a host reboot resets
`kube-proxy`/CoreDNS until manually restarted — meaning the Aim server could
become briefly unreachable in ways the cloud service never was.

- Every `init_aim()` and `track()` call site is wrapped in
  try/except; a network failure logs a warning (existing `log` object) and
  the function returns/no-ops rather than propagating. This mirrors the
  existing `if wandb_run is not None:` guard shape — `aim_run` can end up
  `None` (tracking disabled or failed to connect) and every call site
  already tolerates that.
- Local checkpoints (the thing that actually matters for the training run
  itself) are entirely independent of Aim connectivity and never at risk.
- No retry/reconnect logic beyond this — if Aim is down for a session, that
  session's telemetry is simply missing; training is unaffected and the
  next session tries again.

## Testing

1. Deploy `aim-server`; confirm the PVC binds and both ports are reachable
   (`curl` the UI on `:43800`; a trivial `aim.Run(repo="aim://...")` connect
   + `track()` from a Python shell against `:53800`).
2. Point a short (few-step, `--stage pretrain --model 1.5b` truncated run)
   local invocation at it; confirm the run and its metrics appear in the UI,
   grouped by `stage`/`source` as designed.
3. Kill/restart the `aim-server` pod mid-run; confirm the training process
   logs a warning and keeps training rather than crashing.
4. Two-launch resume test: run briefly, stop, relaunch with the same
   persisted-hash file; confirm it's the *same* Aim run continuing (matches
   today's wandb continuous-run behavior).
5. Cut over `train_session.sh`/`anneal_session.sh` for real; confirm the
   next cron tick logs to Aim successfully.
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
- No auth/TLS on the Aim server — this environment's other self-hosted
  services (`ragu`, `janis`) are similarly unauthenticated on the local
  network, and this matches that existing risk posture.
- No artifact/model-registry use of Aim — checkpoints stay local-disk only,
  exactly as today.
