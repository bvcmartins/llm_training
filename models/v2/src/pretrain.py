"""Stage 1 — PRETRAIN entrypoint.

Long run on the fixed PRETRAIN_MIX (Llama-3-style multi-source) with a cosine
LR schedule and warmup. Produces `checkpoints/qwen3_v2_pretrain_final.pt`,
which `anneal.py --init-from` then consumes.

Run it:

  python pretrain.py                       # defaults (20k steps, wandb on)
  python pretrain.py --max-steps 5000 --no-wandb
  python pretrain.py --resume ../checkpoints/qwen3_v2_pretrain_step005000.pt

The annealing stage lives in `anneal.py`; the shared training loop is in
`training.py`. This module owns only the pretrain-specific config.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import PRETRAIN_MIX            # noqa: E402
from training import StageConfig         # noqa: E402

V2_DIR = HERE.parent


def default_pretrain_config(context_length: int) -> StageConfig:
    """Cosine schedule, warmup, fixed pretrain mix. (~10B tokens at defaults.)"""
    return StageConfig(
        name="pretrain", mix=PRETRAIN_MIX,
        max_steps=20_000, lr_peak=3e-4, lr_end=3e-5,
        warmup_steps=1_000, schedule="cosine",
        context_length=context_length,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v2 stage-1 pretrain")
    p.add_argument("--model", choices=["0.6b", "1.7b"], default="0.6b")
    p.add_argument("--ckpt-dir", type=Path, default=V2_DIR / "checkpoints")
    p.add_argument("--log-dir",  type=Path, default=V2_DIR / "logs")
    p.add_argument("--resume", type=Path, default=None,
                   help="continue pretrain from this checkpoint (model+optimizer+counters)")
    # Overrides (None -> stage default).
    p.add_argument("--max-steps",  type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--lr-peak",    type=float, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--ckpt-every", type=int, default=None)
    p.add_argument("--no-wandb",   action="store_true")
    p.add_argument("--wandb-project", default="llm-training-v2")
    p.add_argument("--wandb-name",    default=None)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main():
    args = parse_args()

    from logging_utils import setup_logging, log_system_info
    logger, log_file = setup_logging("pretrain", args.log_dir)
    logger.info("=== PRETRAIN invocation: %s", " ".join(sys.argv))

    from bootstrap import setup_torch, build_model, init_wandb
    from training import train_stage

    device = setup_torch(args.seed)
    log_system_info(logger, device)
    model, model_cfg = build_model(args.model, device)

    cfg = default_pretrain_config(context_length=model_cfg["context_length"])
    for k in ("max_steps", "batch_size", "grad_accum", "lr_peak", "eval_every", "ckpt_every"):
        v = getattr(args, k)
        if v is not None:
            logger.info("override %s: %s -> %s", k, getattr(cfg, k), v)
            setattr(cfg, k, v)

    wandb_run = init_wandb(
        enabled=not args.no_wandb,
        project=args.wandb_project,
        name=args.wandb_name or f"qwen3_{args.model}_pretrain",
        model_cfg=model_cfg, stage="pretrain", stage_cfg=cfg,
        extra={"resume": str(args.resume) if args.resume else None},
    )

    try:
        summary = train_stage(
            model=model, cfg=cfg, device=device,
            ckpt_dir=args.ckpt_dir, wandb_run=wandb_run,
            resume_from=args.resume,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    logger.info("pretrain summary: %s", summary)
    logger.info("log written to %s", log_file)
    logger.info("next: python anneal.py --init-from %s",
                args.ckpt_dir / "qwen3_v2_pretrain_final.pt")

    # The HF streaming dataloader leaves background download threads alive
    # (aiohttp + hf_xet's Rust runtime — the cas-bridge.xethub.hf.co fetches).
    # They aren't joined at shutdown, so normal interpreter finalization tears
    # their thread-state out and prints a cosmetic
    # `Fatal Python error: PyGILState_Release` *after* the run is fully done.
    # Everything we care about is already durable by here (final checkpoint on
    # disk, wandb.finish() ran in the finally above, log handlers flushed next),
    # so exit hard to skip the crash-prone finalization. Only reached on success;
    # an exception still unwinds normally and prints its traceback.
    logging.shutdown()
    os._exit(0)


if __name__ == "__main__":
    main()
