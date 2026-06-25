"""Stage 2 — ANNEAL entrypoint (separated from training.py).

Short, reasoning-heavy run on ANNEAL_MIX with a LINEAR LR decay from the
pretrain end-LR down to 0. This is the stage that the original bug turned into
a no-op: it must run `max_steps` *additional* optimizer steps on top of the
pretrain handoff. The engine (`training.py`) now anchors the LR schedule to the
stage's starting step, so seeding from a step-20000 pretrain checkpoint works.

Typical handoff from pretrain:

  python anneal.py --init-from ../checkpoints/qwen3_v2_pretrain_final.pt

Continue an interrupted anneal (same schedule, model+optimizer restored):

  python anneal.py --resume ../checkpoints/qwen3_v2_anneal_step001000.pt

`--init-from` seeds a NEW anneal stage from prior weights (fresh optimizer) and
carries the global step/token counters + per-source doc cursors forward so
token accounting stays cumulative and data isn't re-read. `--resume` continues
the SAME anneal run.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import ANNEAL_MIX             # noqa: E402
from training import StageConfig        # noqa: E402

V2_DIR = HERE.parent

# Fallback if a seed checkpoint predates stage_cfg and no --lr-peak is given.
DEFAULT_PRETRAIN_END_LR = 3e-5


def resolve_ckpt(path: Path, ckpt_dir: Path) -> Path:
    """Resolve a checkpoint path independent of the current working directory.

    The docstrings suggest `--init-from ../checkpoints/...`, which only resolves
    when run from `src/`. If the path doesn't exist as given, fall back to its
    basename inside `--ckpt-dir` (the absolute default) before giving up.
    """
    if path.exists():
        return path
    fallback = ckpt_dir / path.name
    if fallback.exists():
        return fallback
    raise SystemExit(
        f"checkpoint not found: '{path}' (also tried '{fallback}'). "
        f"Pass an existing path via --init-from/--resume, or set --ckpt-dir."
    )


def default_anneal_config(context_length: int, pretrain_end_lr: float) -> StageConfig:
    """Linear decay from the pretrain end-LR to 0, up-weighted math/arxiv mix."""
    return StageConfig(
        name="anneal", mix=ANNEAL_MIX,
        max_steps=2_000, lr_peak=pretrain_end_lr, lr_end=0.0,
        warmup_steps=0, schedule="linear",
        context_length=context_length,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v3 stage-2 anneal")
    p.add_argument("--model", choices=["0.6b", "1.5b", "1.7b"], default="1.5b")
    p.add_argument("--ckpt-dir", type=Path, default=V2_DIR / "checkpoints")
    p.add_argument("--log-dir",  type=Path, default=V2_DIR / "logs")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--init-from", type=Path, default=None,
                   help="seed a NEW anneal stage from this pretrain checkpoint's weights")
    g.add_argument("--resume", type=Path, default=None,
                   help="continue an interrupted anneal (model+optimizer+counters)")

    # Overrides (None -> stage default).
    p.add_argument("--max-steps",  type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--lr-peak",    type=float, default=None,
                   help="anneal peak LR; defaults to the seed checkpoint's pretrain lr_end")
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
    logger, log_file = setup_logging("anneal", args.log_dir)
    logger.info("=== ANNEAL invocation: %s", " ".join(sys.argv))

    import torch
    from bootstrap import setup_torch, build_model, init_wandb
    from training import train_stage, load_resume_state, stage_cfg_from_dict

    device = setup_torch(args.seed)
    log_system_info(logger, device)
    model, model_cfg = build_model(args.model, device)
    ctx = model_cfg["context_length"]

    # train_stage handoff kwargs differ between the two entry paths.
    handoff: dict = {}

    if args.resume is not None:
        # Continue the SAME anneal — rebuild the exact saved schedule so LR
        # decay lines up, then let the engine restore model/optimizer/counters.
        args.resume = resolve_ckpt(args.resume, args.ckpt_dir)
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if payload.get("stage") != "anneal":
            raise SystemExit(f"--resume target is stage '{payload.get('stage')}', not 'anneal'.")
        saved = payload.get("stage_cfg")
        if saved is None:
            raise SystemExit(f"--resume target {args.resume} has no stage_cfg (old checkpoint).")
        cfg = stage_cfg_from_dict(saved)
        logger.info("resuming anneal from %s (saved lr_peak=%.3e)", args.resume.name, cfg.lr_peak)
        handoff["resume_from"] = args.resume
    else:
        # Seed a NEW anneal from a pretrain checkpoint: load weights only (fresh
        # optimizer) and carry global counters + doc cursors forward.
        args.init_from = resolve_ckpt(args.init_from, args.ckpt_dir)
        state = load_resume_state(args.init_from, model, optimizer=None)
        seed_stage = state.get("stage")
        seed_cfg = state.get("stage_cfg") or {}
        pretrain_end_lr = args.lr_peak or seed_cfg.get("lr_end", DEFAULT_PRETRAIN_END_LR)
        logger.info("seeding anneal from %s (stage=%s step=%d tokens=%s) -> peak LR=%.3e",
                    args.init_from.name, seed_stage, state["step"],
                    f"{state['tokens']:,}", pretrain_end_lr)
        cfg = default_anneal_config(context_length=ctx, pretrain_end_lr=pretrain_end_lr)
        handoff.update(
            starting_step=state["step"],
            starting_tokens=state["tokens"],
            skip_docs=state["docs_per_source"],
        )

    # Apply CLI overrides (after schedule reconstruction so they win).
    for k in ("max_steps", "batch_size", "grad_accum", "lr_peak", "eval_every", "ckpt_every"):
        v = getattr(args, k)
        if v is not None:
            logger.info("override %s: %s -> %s", k, getattr(cfg, k), v)
            setattr(cfg, k, v)

    wandb_run = init_wandb(
        enabled=not args.no_wandb,
        project=args.wandb_project,
        name=args.wandb_name or f"qwen3_{args.model}_anneal",
        model_cfg=model_cfg, stage="anneal", stage_cfg=cfg,
        extra={
            "init_from": str(args.init_from) if args.init_from else None,
            "resume":    str(args.resume) if args.resume else None,
        },
    )

    try:
        summary = train_stage(
            model=model, cfg=cfg, device=device,
            ckpt_dir=args.ckpt_dir, wandb_run=wandb_run,
            **handoff,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    logger.info("anneal summary: %s", summary)
    if sum(summary["tokens_per_source"].values()) == 0:
        logger.warning("anneal consumed 0 tokens — the stage did not actually train; "
                       "check max_steps and the handoff above")
    logger.info("log written to %s", log_file)

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
