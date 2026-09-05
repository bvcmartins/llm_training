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
        # log_system_metrics logs GPU/CPU/mem stats (system/gpu_0_utilization_percentage,
        # system/gpu_0_memory_usage_megabytes, ...) on a background sampling thread for
        # the run's lifetime — mlflow's own monitor swallows failures internally (e.g. no
        # pynvml, no GPU) and just logs a warning, so this is safe on any machine.
        run = mlflow.start_run(run_id=existing_run_id, run_name=run_name, log_system_metrics=True)
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
