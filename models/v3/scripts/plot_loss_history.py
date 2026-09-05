#!/usr/bin/env python3
"""Stitch the fragmented pretrain history (pre-mlflow wandb imports + the live
continuous mlflow run) into one long-running loss/ppl plot, like wandb used to
show as a single line.

The pretrain history lives in three places that don't connect automatically:
  1. experiment "llm-training-v3"      - 38 imported wandb runs, steps 0-~19k
     (each daily session was its own wandb run before the continuous-run
     pattern existed).
  2. a real gap, ~19k-~21k             - lived in the wandb "dense-model-1.5b-v3"
     project, deleted before this migration; not recoverable.
  3. experiment "dense-model-1.5b-v3"  - the live run, one continuous mlflow
     run from the cutover point onward.

This script pulls metric history from every run across both experiments,
concatenates by step, and plots loss/ppl as two stacked panels sharing a step
axis. A real gap in the data (missing steps) is left as a visible break in the
line rather than interpolated across, so it isn't mistaken for real data.

Usage: .venv/bin/python models/v3/scripts/plot_loss_history.py
Outputs: models/v3/logs/loss_history.png, models/v3/logs/loss_history.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient

TRACKING_URI = "http://airig.local:5000"
EXPERIMENTS = ["llm-training-v3", "dense-model-1.5b-v3"]
METRICS = ["loss", "ppl"]
OUT_DIR = Path(__file__).resolve().parent.parent / "logs"

# A gap wider than this (in steps) between consecutive points is drawn as a
# break in the line, not a straight interpolation across missing data.
GAP_THRESHOLD = 200


def fetch_history(client: MlflowClient) -> dict[str, list[tuple[int, float]]]:
    points: dict[str, list[tuple[int, float]]] = {m: [] for m in METRICS}
    for exp_name in EXPERIMENTS:
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            print(f"  (experiment {exp_name!r} not found, skipping)")
            continue
        runs = client.search_runs([exp.experiment_id], max_results=1000)
        print(f"  {exp_name}: {len(runs)} runs")
        for run in runs:
            for metric in METRICS:
                for m in client.get_metric_history(run.info.run_id, metric):
                    points[metric].append((m.step, m.value))
    for metric in METRICS:
        points[metric].sort(key=lambda p: p[0])
    return points


def with_gaps(points: list[tuple[int, float]]) -> tuple[list[float], list[float]]:
    """Insert a NaN between consecutive points whose step gap exceeds the
    threshold, so matplotlib draws a break instead of a straight line."""
    xs: list[float] = []
    ys: list[float] = []
    prev_step = None
    for step, value in points:
        if prev_step is not None and step - prev_step > GAP_THRESHOLD:
            xs.append(float("nan"))
            ys.append(float("nan"))
        xs.append(step)
        ys.append(value)
        prev_step = step
    return xs, ys


def main():
    client = MlflowClient(tracking_uri=TRACKING_URI)
    print(f"fetching metric history from {TRACKING_URI} ...")
    points = fetch_history(client)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "loss_history.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "step", "value"])
        for metric in METRICS:
            for step, value in points[metric]:
                w.writerow([metric, step, value])
    print(f"wrote {csv_path}")

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(11, 7), sharex=True)
    if len(METRICS) == 1:
        axes = [axes]
    hue = "#3B6FD4"
    for ax, metric in zip(axes, METRICS):
        xs, ys = with_gaps(points[metric])
        ax.plot(xs, ys, color=hue, linewidth=1.2)
        if metric == "ppl":
            ax.set_yscale("log")
        ax.set_ylabel(metric)
        ax.grid(True, color="#E5E5E5", linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[-1].set_xlabel("step")
    fig.suptitle(
        "qwen3_1.5b pretrain — stitched from llm-training-v3 (imported) +\n"
        "dense-model-1.5b-v3 (live). Break in line = real missing data "
        "(deleted wandb project, ~step 19k-21k).",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    png_path = OUT_DIR / "loss_history.png"
    fig.savefig(png_path, dpi=150)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
