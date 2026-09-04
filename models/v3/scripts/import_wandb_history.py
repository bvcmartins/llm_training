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
