"""Staged training loop for v2.

Two stages, Llama 3 style:
  1. PRETRAIN — long run on PRETRAIN_MIX, cosine LR with warmup.
  2. ANNEAL   — short run on ANNEAL_MIX (up-weighted math/arxiv),
                linear LR decay from the pretrain end-LR to 0.

The same `train_stage(...)` function runs both — the stage is just a config.
Every wandb log line and every checkpoint carries the stage name so the
training-data identity of any model state is unambiguous.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import (
    StageMix, PRETRAIN_MIX, ANNEAL_MIX,
    SOURCE_IDS, ID_TO_SOURCE,
    build_train_loader, build_val_loaders,
)
from eval import evaluate_per_domain, PlateauDetector


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class StageConfig:
    name:            str           # "pretrain" | "anneal"
    mix:             StageMix
    max_steps:       int
    batch_size:      int     = 4
    grad_accum:      int     = 64
    context_length:  int     = 2_048
    lr_peak:         float   = 3e-4
    lr_end:          float   = 3e-5
    warmup_steps:    int     = 1_000
    schedule:        str     = "cosine"   # "cosine" (pretrain) or "linear" (anneal)
    weight_decay:    float   = 0.1
    grad_clip:       float   = 1.0
    eval_every:      int     = 250
    eval_batches:    int     = 20
    ckpt_every:      int     = 1_000
    val_docs_per_source: int = 200
    grad_checkpoint: bool    = True


def default_pretrain_config(context_length: int) -> StageConfig:
    return StageConfig(
        name="pretrain", mix=PRETRAIN_MIX,
        max_steps=20_000, lr_peak=3e-4, lr_end=3e-5,
        warmup_steps=1_000, schedule="cosine",
        context_length=context_length,
    )


def default_anneal_config(context_length: int, pretrain_end_lr: float) -> StageConfig:
    return StageConfig(
        name="anneal", mix=ANNEAL_MIX,
        max_steps=2_000, lr_peak=pretrain_end_lr, lr_end=0.0,
        warmup_steps=0, schedule="linear",
        context_length=context_length,
    )


# ---------------------------------------------------------------------------
# Optimizer + schedule
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Apply weight decay only to matrices (Linear weights), not norms/biases/embeddings."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "norm" in n or "bias" in n or "tok_emb" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=True)


def lr_for_step(step: int, cfg: StageConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr_peak * (step + 1) / max(cfg.warmup_steps, 1)
    progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    if cfg.schedule == "cosine":
        return cfg.lr_end + 0.5 * (cfg.lr_peak - cfg.lr_end) * (1 + math.cos(math.pi * progress))
    if cfg.schedule == "linear":
        return cfg.lr_peak + (cfg.lr_end - cfg.lr_peak) * progress
    raise ValueError(f"unknown schedule: {cfg.schedule}")


# ---------------------------------------------------------------------------
# Checkpoints (stage-aware)
# ---------------------------------------------------------------------------

def save_checkpoint(
    path:       Path,
    model:      nn.Module,
    optimizer:  torch.optim.Optimizer,
    stage_cfg:  StageConfig,
    step:       int,
    tokens:     int,
    docs_per_source: dict,
    val_history:list,
):
    payload = {
        "model":            model.state_dict(),
        "optimizer":        optimizer.state_dict(),
        "stage":            stage_cfg.name,
        "stage_mix":        stage_cfg.mix.weights,
        "stage_cfg":        asdict(stage_cfg),
        "step":             step,
        "tokens":           tokens,
        "docs_per_source":  docs_per_source,
        "val_history":      val_history,
        "model_config_ref": "qwen3_model.QWEN3_CONFIG_0_6B",  # bump when scaling
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def stage_cfg_from_dict(d: dict) -> StageConfig:
    """Rebuild a StageConfig from the dict saved in a checkpoint."""
    d = dict(d)
    mix_d = d.pop("mix")
    mix = StageMix(name=mix_d["name"], weights=dict(mix_d["weights"]))
    return StageConfig(mix=mix, **d)


def load_resume_state(
    path:      Path,
    model:     nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """Load weights + (optionally) optimizer state + training counters.

    Returns a dict with keys: step, tokens, docs_per_source, val_history,
    stage, stage_cfg (raw dict). Pass `optimizer=None` to load *only* the model
    weights (use this for --init-from to start a new stage from prior weights).
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return {
        "step":            payload["step"],
        "tokens":          payload["tokens"],
        "docs_per_source": payload["docs_per_source"],
        "val_history":     payload.get("val_history", []),
        "stage":           payload["stage"],
        "stage_cfg":       payload.get("stage_cfg"),
    }


# ---------------------------------------------------------------------------
# One stage
# ---------------------------------------------------------------------------

def train_stage(
    model:           nn.Module,
    cfg:             StageConfig,
    device:          torch.device,
    ckpt_dir:        Path,
    wandb_run=None,
    starting_step:   int                  = 0,
    starting_tokens: int                  = 0,
    skip_docs:       dict[str, int] | None = None,
    resume_from:     Path | None           = None,
) -> dict:
    """Run one stage. Returns final state dict-like summary for handoff.

    `resume_from` continues the SAME stage from a checkpoint: model + optimizer
    state are loaded, and step/tokens/docs counters are taken from the
    checkpoint (overriding any starting_* / skip_docs args). The stage must
    match the checkpoint's stage — pass a fresh `cfg` only when starting a
    *new* stage seeded from prior weights (use `load_resume_state(...,
    optimizer=None)` outside this function for that).
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    if hasattr(model, "enable_grad_checkpointing"):
        model.enable_grad_checkpointing(cfg.grad_checkpoint)

    optimizer = build_optimizer(model, lr=cfg.lr_peak, weight_decay=cfg.weight_decay)

    val_history: list = []
    if resume_from is not None:
        state = load_resume_state(resume_from, model, optimizer)
        if state["stage"] != cfg.name:
            raise ValueError(
                f"resume_from is from stage '{state['stage']}' but cfg is for stage '{cfg.name}'. "
                f"To start a new stage from prior weights, load weights outside train_stage "
                f"(with optimizer=None) and pass a fresh cfg without resume_from."
            )
        starting_step   = state["step"]
        starting_tokens = state["tokens"]
        skip_docs       = state["docs_per_source"]
        val_history     = state["val_history"]
        print(f"resumed {cfg.name} from {resume_from.name} at step={starting_step} tokens={starting_tokens:,}")

    train_loader = build_train_loader(
        mix=cfg.mix, context_length=cfg.context_length,
        batch_size=cfg.batch_size, skip_docs=skip_docs,
    )
    val_loaders  = build_val_loaders(
        context_length=cfg.context_length, batch_size=cfg.batch_size,
        val_docs_per_source=cfg.val_docs_per_source,
    )

    def _on_plateau(payload):
        msg = f"[plateau:{cfg.name}] best={payload['best']:.4f} current={payload['current']:.4f}"
        print(msg)
        if wandb_run is not None:
            wandb_run.alert(title=f"Plateau in {cfg.name}", text=msg)

    detector = PlateauDetector(patience=5, min_delta=1e-3, cooldown=5, on_fire=_on_plateau)

    # Per-source token counter (for stage-level data attribution).
    tokens_per_source = {name: 0 for name in SOURCE_IDS}
    tokens = starting_tokens
    step   = starting_step
    train_iter = iter(train_loader)

    t_last = time.time()
    while step < cfg.max_steps:
        lr = lr_for_step(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_per_source = {name: 0 for name in SOURCE_IDS}

        for _ in range(cfg.grad_accum):
            x, y, src = next(train_iter)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten()) / cfg.grad_accum
            loss.backward()
            running_loss += loss.item()

            # Per-source accounting (token count is batch_size * context_length per micro-batch,
            # all from a single source because batches aren't mixed within a window).
            for s in src.tolist():
                running_per_source[ID_TO_SOURCE[s]] += x.shape[1]

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # Accumulate stage-level counters.
        for name, n in running_per_source.items():
            tokens_per_source[name] += n
        tokens += cfg.batch_size * cfg.context_length * cfg.grad_accum

        # Throughput
        now = time.time()
        dt = max(now - t_last, 1e-6)
        tok_per_sec = (cfg.batch_size * cfg.context_length * cfg.grad_accum) / dt
        t_last = now

        log = {
            "stage":     cfg.name,
            "step":      step,
            "tokens":    tokens,
            "loss":      running_loss,
            "lr":        lr,
            "grad_norm": float(grad_norm),
            "tok_per_sec": tok_per_sec,
            **{f"src_tokens/{k}": v for k, v in tokens_per_source.items()},
        }
        if wandb_run is not None:
            wandb_run.log(log, step=step)
        if step % 50 == 0:
            print(f"[{cfg.name} step {step:>6}] loss={running_loss:.4f} lr={lr:.2e} tok/s={tok_per_sec:,.0f}")

        # Evaluation
        if step > 0 and step % cfg.eval_every == 0:
            val = evaluate_per_domain(model, val_loaders, device, max_batches=cfg.eval_batches)
            val_history.append({"step": step, "stage": cfg.name, **val})
            if wandb_run is not None:
                wandb_run.log({f"val/{k}": v for k, v in val.items()} | {"stage": cfg.name}, step=step)
            print(
                f"[{cfg.name} eval step {step}] aggregate={val['aggregate']:.4f}  "
                + "  ".join(f"{k}={v:.3f}" for k, v in val.items() if k != "aggregate")
            )
            detector.update(val)

        # Checkpoint
        if step > 0 and step % cfg.ckpt_every == 0:
            path = ckpt_dir / f"qwen3_v2_{cfg.name}_step{step:06d}.pt"
            save_checkpoint(
                path, model, optimizer, cfg, step, tokens,
                docs_per_source=train_loader.dataset.docs_consumed,
                val_history=val_history,
            )
            print(f"saved {path.name}")

        step += 1

    # Final eval + checkpoint
    val = evaluate_per_domain(model, val_loaders, device, max_batches=cfg.eval_batches)
    val_history.append({"step": step, "stage": cfg.name, **val})
    final_path = ckpt_dir / f"qwen3_v2_{cfg.name}_final.pt"
    save_checkpoint(
        final_path, model, optimizer, cfg, step, tokens,
        docs_per_source=train_loader.dataset.docs_consumed,
        val_history=val_history,
    )
    print(f"stage {cfg.name} complete — saved {final_path.name}")

    return {
        "stage":             cfg.name,
        "final_step":        step,
        "final_tokens":      tokens,
        "tokens_per_source": tokens_per_source,
        "final_val":         val,
        "final_lr":          lr_for_step(step - 1, cfg),
        "docs_per_source":   train_loader.dataset.docs_consumed,
    }
