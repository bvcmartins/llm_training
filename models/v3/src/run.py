"""CLI runner for v2 staged training.

Examples:

  # Start pretrain from scratch
  python run.py --stage pretrain

  # Resume pretrain (continues same stage from saved step/optimizer)
  python run.py --stage pretrain --resume ../checkpoints/qwen3_v2_pretrain_step005000.pt

  # Start anneal seeded from the final pretrain checkpoint (fresh optimizer)
  python run.py --stage anneal --init-from ../checkpoints/qwen3_v2_pretrain_final.pt

  # Override defaults
  python run.py --stage pretrain --max-steps 5000 --batch-size 2 --grad-accum 128

The handoff with the notebook is symmetric: any checkpoint produced by the
notebook is a valid `--resume` target here, and vice versa — both call into
the same `train_stage()` in training.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MODEL_CHOICES = ("0.6b", "1.5b", "1.7b")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v3 staged trainer")
    p.add_argument("--stage", choices=["pretrain", "anneal"], required=True)
    p.add_argument("--model", choices=MODEL_CHOICES, default="1.5b")
    p.add_argument("--ckpt-dir", type=Path, default=HERE.parent / "checkpoints")

    # Mutually exclusive resume modes.
    g = p.add_mutually_exclusive_group()
    g.add_argument("--resume",    type=Path, default=None,
                   help="continue SAME stage from this checkpoint (loads model+optimizer+counters)")
    g.add_argument("--init-from", type=Path, default=None,
                   help="start NEW stage seeded from this checkpoint's model weights (fresh optimizer)")
    g.add_argument("--auto-resume", action="store_true",
                   help="resume from the newest qwen3_v3_<stage>_step*.pt in --ckpt-dir; "
                        "start fresh if none exist (used by the scheduled-training launcher)")

    # Scheduled-training graceful stop: local 'HH:MM' wall-clock deadline at
    # which the trainer checkpoints and exits cleanly (yields GPU for blackout).
    p.add_argument("--stop-at", type=str, default=None, metavar="HH:MM")

    # Overrides (None = use stage defaults).
    p.add_argument("--max-steps",  type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--lr-peak",    type=float, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--ckpt-every", type=int, default=None)

    p.add_argument("--no-mlflow",   action="store_true")
    p.add_argument("--mlflow-experiment", default="dense-model-1.5b-v3")
    p.add_argument("--mlflow-run-name",   default=None)
    p.add_argument("--mlflow-run-id-file", type=Path, default=None,
                   help="path to a file holding the persisted MLflow run id, so "
                        "every resume continues ONE run (default: "
                        "<ckpt-dir>/.mlflow_run_id_<stage>). Keeps the dashboard a "
                        "single continuous curve per stage instead of a new run "
                        "per session. Delete the file to start a fresh run.")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                   help="torch.compile the training forward (~1.17x on the 5090; "
                        "one-time ~40s compile per process). --no-compile to disable.")
    p.add_argument("--seed",       type=int, default=123)
    p.add_argument("--allow-cpu", action="store_true",
                   help="Permit training on CPU. By default run.py ABORTS if CUDA is "
                        "unavailable — a driver/kernel-module mismatch (e.g. a kernel "
                        "update without a reboot) silently falls back to CPU and wastes "
                        "a whole scheduled session at ~1/100th throughput. Only pass this "
                        "for a deliberate CPU debug run.")
    return p.parse_args()


def resolve_auto_resume(ckpt_dir: Path, stage: str) -> Path | None:
    """Newest `qwen3_v3_<stage>_step*.pt` in ckpt_dir, or None for a fresh start.

    `_final.pt` is intentionally ignored — a completed stage should hand off to
    the next stage (manual `--init-from`), not silently re-resume here.
    """
    cands = sorted(
        ckpt_dir.glob(f"qwen3_v3_{stage}_step*.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    return cands[-1] if cands else None


def main():
    args = parse_args()

    # Heavy imports happen here so `--help` is fast and doesn't need
    # the full training env loaded.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from qwen3_model import (
        Qwen3Model, QWEN3_CONFIG_0_6B, QWEN3_CONFIG_1_5B, QWEN3_CONFIG_1_7B,
    )
    from training import train_stage, load_resume_state, stage_cfg_from_dict
    # Stage-config factories were moved out of training.py (the engine) and now
    # live with their stage entrypoints.
    from pretrain import default_pretrain_config
    from anneal import default_anneal_config
    from mlflow_utils import init_mlflow, end_run_safe

    model_configs = {
        "0.6b": QWEN3_CONFIG_0_6B, "1.5b": QWEN3_CONFIG_1_5B, "1.7b": QWEN3_CONFIG_1_7B,
    }

    # --auto-resume → resolve to a concrete --resume target (newest step ckpt for
    # this stage), or fall through to a fresh start if none exist.
    if args.auto_resume:
        newest = resolve_auto_resume(args.ckpt_dir, args.stage)
        if newest is not None:
            args.resume = newest
            print(f"[auto-resume] resuming from {newest.name}")
        else:
            print(f"[auto-resume] no qwen3_v3_{args.stage}_step*.pt in {args.ckpt_dir} — fresh start")

    # Hard GPU guard: a driver/kernel-module mismatch (Error 804 / "Can't initialize
    # NVML", typically a kernel update without a reboot) makes torch.cuda.is_available()
    # return False. The trainer would then silently run on CPU — the loop still "works"
    # and wandb keeps syncing, so a full scheduled session (~22h) burns with ~zero
    # progress before anyone notices (happened 2026-07-25). Fail LOUD instead unless a
    # CPU run was explicitly requested.
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit(
            "FATAL: CUDA is not available — refusing to train on CPU (would waste the "
            "session at ~1/100th throughput). This usually means an NVIDIA driver/"
            "kernel-module mismatch; reboot (or reload the nvidia module) and retry. "
            "Pass --allow-cpu only for a deliberate CPU debug run."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    model_cfg = model_configs[args.model]
    model = Qwen3Model(model_cfg).to(device=device, dtype=model_cfg["dtype"])

    def apply_overrides(cfg):
        for k in ("max_steps", "batch_size", "grad_accum", "lr_peak", "eval_every", "ckpt_every"):
            v = getattr(args, k)
            if v is not None:
                setattr(cfg, k, v)
        return cfg

    def build_stage_cfg():
        if args.stage == "pretrain":
            return default_pretrain_config(context_length=model_cfg["context_length"])
        # anneal — derive peak LR from the pretrain seed if available
        pretrain_end_lr = 3e-5
        if args.init_from is not None:
            payload = torch.load(args.init_from, map_location="cpu", weights_only=False)
            saved = payload.get("stage_cfg")
            if saved is not None:
                pretrain_end_lr = saved.get("lr_end", pretrain_end_lr)
        return default_anneal_config(
            context_length=model_cfg["context_length"],
            pretrain_end_lr=pretrain_end_lr,
        )

    # Decide stage cfg. If --resume, use the saved cfg verbatim (continuing the
    # same stage means the schedule must be the same). Otherwise build fresh.
    if args.resume is not None:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved_cfg = payload.get("stage_cfg")
        if saved_cfg is None:
            raise SystemExit(
                f"--resume target {args.resume} has no stage_cfg (likely an old checkpoint). "
                f"Use --init-from instead to seed a fresh stage from these weights."
            )
        if payload["stage"] != args.stage:
            raise SystemExit(
                f"--resume target is for stage '{payload['stage']}' but --stage={args.stage}. "
                f"Either match the stage, or use --init-from to start a new stage from these weights."
            )
        stage_cfg = stage_cfg_from_dict(saved_cfg)
        # Keep the LR-schedule fields (lr_peak/lr_end/warmup/max_steps) from the
        # saved cfg so the cosine curve stays continuous across resume. But the
        # DATA MIXTURE, val sampling, and CHECKPOINT CADENCE are NOT part of the
        # schedule: freezing them means edits to those knobs are silently ignored on
        # every --resume (the value stays whatever the very first checkpoint used —
        # e.g. the arxiv 0.12->0.04 trim never took effect and arxiv looped past
        # 1 epoch; likewise a ckpt_every 500->25 edit never applied, so a slow
        # session never crossed the 500-step boundary and no checkpoint was ever
        # written -> the resume-from-14052 groundhog loop, see the gpu-power-cap
        # memory). Re-read them from the live config so tuning actually applies to
        # the next session.
        live_cfg = build_stage_cfg()
        stage_cfg.mix = live_cfg.mix
        stage_cfg.val_docs_per_source = live_cfg.val_docs_per_source
        stage_cfg.ckpt_every = live_cfg.ckpt_every
    else:
        stage_cfg = build_stage_cfg()

    stage_cfg = apply_overrides(stage_cfg)

    # v3 identity + scheduled-stop deadline — always taken from the live args so a
    # resumed checkpoint's stale stop_at can't leak in, and the config_ref tracks
    # the chosen model size.
    stage_cfg.model_tag = "qwen3_v3"
    stage_cfg.model_config_ref = f"qwen3_model.QWEN3_CONFIG_{args.model.upper().replace('.', '_')}"
    stage_cfg.stop_at = args.stop_at
    # Live arg, like stop_at: resuming a checkpoint saved before this field existed
    # (e.g. step 17893) would otherwise default compile off.
    stage_cfg.compile = args.compile

    # If --init-from, load weights only (fresh optimizer). train_stage's
    # resume_from path is only for same-stage resume, so do this here.
    if args.init_from is not None:
        load_resume_state(args.init_from, model, optimizer=None)
        print(f"initialized model weights from {args.init_from.name}")

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
            **{k: v for k, v in stage_cfg.__dict__.items() if k != "mix"},
        },
        # resume/init_from change on every session (each night resumes from a
        # newer checkpoint) — they must be tags, not params: mlflow params are
        # immutable once logged for a run, and this is ONE continuous run
        # across all daily sessions, so re-logging a changed value as a param
        # throws and disables tracking for the whole session (see the mlflow-
        # local-tracking-design doc).
        tags={
            "group": f"qwen3_{args.model}_{args.stage}",
            **({"resume": str(args.resume)} if args.resume else {}),
            **({"init_from": str(args.init_from)} if args.init_from else {}),
        },
    )

    print(f"=== stage={args.stage} model={args.model} device={device} ===")
    print(stage_cfg)

    try:
        summary = train_stage(
            model=model,
            cfg=stage_cfg,
            device=device,
            ckpt_dir=args.ckpt_dir,
            mlflow_enabled=mlflow_enabled,
            resume_from=args.resume,
        )
    except Exception:
        # A crash (CUDA OOM, etc.) must NOT fall through to normal interpreter
        # finalization. The HF streaming dataloader's un-joinable background
        # threads (aiohttp + hf_xet Rust runtime) wedge the interpreter on exit,
        # so the process hangs *while still pinning GPU memory* — which then
        # starves the next --auto-resume session (it OOMs against the zombie).
        # Print the traceback, close wandb, and hard-exit non-zero so the GPU is
        # released immediately and the launcher can relaunch cleanly.
        import traceback
        traceback.print_exc()
        if mlflow_enabled:
            end_run_safe(status="FAILED")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    if mlflow_enabled:
        end_run_safe()

    print("\nstage summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # The HF streaming dataloader leaves background download threads alive
    # (aiohttp + hf_xet's Rust runtime). They aren't joined at shutdown, so
    # normal interpreter finalization tears their thread-state out and prints a
    # cosmetic `Fatal Python error: PyGILState_Release` *after* the run is fully
    # done — which also makes the process exit non-zero. Everything we care about
    # is already durable here (checkpoint on disk, wandb.finish() ran above), so
    # exit hard to skip the crash-prone finalization and return a clean 0. Only
    # reached on success; an exception still unwinds normally with its traceback.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
