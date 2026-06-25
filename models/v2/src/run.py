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

MODEL_CHOICES = ("0.6b", "1.7b")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v2 staged trainer")
    p.add_argument("--stage", choices=["pretrain", "anneal"], required=True)
    p.add_argument("--model", choices=MODEL_CHOICES, default="0.6b")
    p.add_argument("--ckpt-dir", type=Path, default=HERE.parent / "checkpoints")

    # Mutually exclusive resume modes.
    g = p.add_mutually_exclusive_group()
    g.add_argument("--resume",    type=Path, default=None,
                   help="continue SAME stage from this checkpoint (loads model+optimizer+counters)")
    g.add_argument("--init-from", type=Path, default=None,
                   help="start NEW stage seeded from this checkpoint's model weights (fresh optimizer)")

    # Overrides (None = use stage defaults).
    p.add_argument("--max-steps",  type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--lr-peak",    type=float, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--ckpt-every", type=int, default=None)

    p.add_argument("--no-wandb",   action="store_true")
    p.add_argument("--wandb-project", default="llm-training-v2")
    p.add_argument("--wandb-name",    default=None)
    p.add_argument("--seed",       type=int, default=123)
    return p.parse_args()


def main():
    args = parse_args()

    # Heavy imports happen here so `--help` is fast and doesn't need
    # the full training env loaded.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from qwen3_model import Qwen3Model, QWEN3_CONFIG_0_6B, QWEN3_CONFIG_1_7B
    from training import train_stage, load_resume_state, stage_cfg_from_dict
    # Stage-config factories were moved out of training.py (the engine) and now
    # live with their stage entrypoints.
    from pretrain import default_pretrain_config
    from anneal import default_anneal_config

    model_configs = {"0.6b": QWEN3_CONFIG_0_6B, "1.7b": QWEN3_CONFIG_1_7B}

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
    else:
        stage_cfg = build_stage_cfg()

    stage_cfg = apply_overrides(stage_cfg)

    # If --init-from, load weights only (fresh optimizer). train_stage's
    # resume_from path is only for same-stage resume, so do this here.
    if args.init_from is not None:
        load_resume_state(args.init_from, model, optimizer=None)
        print(f"initialized model weights from {args.init_from.name}")

    # wandb
    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"qwen3_{args.model}_{args.stage}",
            config={
                "model_config": model_cfg,
                "stage":        args.stage,
                "mix":          stage_cfg.mix.weights,
                "resume":       str(args.resume) if args.resume else None,
                "init_from":    str(args.init_from) if args.init_from else None,
                **{k: v for k, v in stage_cfg.__dict__.items() if k != "mix"},
            },
        )

    print(f"=== stage={args.stage} model={args.model} device={device} ===")
    print(stage_cfg)

    summary = train_stage(
        model=model,
        cfg=stage_cfg,
        device=device,
        ckpt_dir=args.ckpt_dir,
        wandb_run=wandb_run,
        resume_from=args.resume,
    )

    if wandb_run is not None:
        wandb_run.finish()

    print("\nstage summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
