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

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from data import (
    StageMix, PRETRAIN_MIX, ANNEAL_MIX,
    SOURCE_IDS, ID_TO_SOURCE,
    build_train_loader, build_val_loaders,
)
from eval import evaluate_per_domain, PlateauDetector
from logging_utils import gpu_mem_str, log_config
from tokenizer import encode, decode, eot_id

# Shared engine logger. Entrypoints (pretrain.py / anneal.py) attach the
# handlers via logging_utils.setup_logging(); if nobody configured logging we
# still emit to stdout rather than silently dropping records.
log = logging.getLogger("v2.train")
if not logging.getLogger("v2").handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)-12s | %(message)s")


def _safe_ppl(loss: float) -> float:
    return math.exp(min(loss, 30.0))


def _fmt_dur(seconds: float) -> str:
    """Human-readable h/m/s, e.g. 3723s -> '1h02m03s'."""
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


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
    sample_every:    int     = 500
    sample_prompt:   str     = "The quick brown fox"
    sample_max_new:  int     = 60
    sample_top_k:    int     = 50
    sample_temperature: float = 0.8


# Stage-config factories now live with their stage entrypoints:
#   default_pretrain_config -> pretrain.py
#   default_anneal_config   -> anneal.py
# training.py is the stage-agnostic engine; it only needs StageConfig.


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
    stage_start_step: int = 0,
):
    payload = {
        "model":            model.state_dict(),
        "optimizer":        optimizer.state_dict(),
        "stage":            stage_cfg.name,
        "stage_mix":        stage_cfg.mix.weights,
        "stage_cfg":        asdict(stage_cfg),
        "step":             step,
        "stage_start_step": stage_start_step,
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
        "step":             payload["step"],
        "stage_start_step": payload.get("stage_start_step"),
        "tokens":           payload["tokens"],
        "docs_per_source":  payload["docs_per_source"],
        "val_history":      payload.get("val_history", []),
        "stage":            payload["stage"],
        "stage_cfg":        payload.get("stage_cfg"),
    }


# ---------------------------------------------------------------------------
# Inference sampling (qualitative progress check)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model:          nn.Module,
    idx:            torch.Tensor,        # [B, T]
    max_new_tokens: int,
    context_size:   int,
    device:         torch.device,
    temperature:    float = 0.0,
    top_k:          int | None = None,
    eos_id:         int | None = None,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    use_amp = device.type == "cuda"
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(idx_cond)
        else:
            logits = model(idx_cond)
        logits = logits[:, -1, :].float()

        if top_k is not None:
            top_vals, _ = torch.topk(logits, top_k, dim=-1)
            cutoff = top_vals[:, -1:].expand_as(logits)
            logits = torch.where(logits < cutoff, torch.full_like(logits, -float("inf")), logits)

        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and (idx_next == eos_id).all():
            break

        idx = torch.cat([idx, idx_next], dim=1)

    if was_training:
        model.train()
    return idx


def generate_sample_text(
    model:          nn.Module,
    prompt:         str,
    device:         torch.device,
    context_length: int,
    max_new_tokens: int = 60,
    temperature:    float = 0.8,
    top_k:          int   = 50,
) -> str:
    ids = encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate(
        model, idx,
        max_new_tokens=max_new_tokens,
        context_size=context_length,
        device=device,
        temperature=temperature,
        top_k=top_k,
        eos_id=eot_id(),
    )
    return decode(out[0].cpu().tolist())


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
    stage_start_step: int | None = None  # global step at which THIS stage began
    if resume_from is not None:
        state = load_resume_state(resume_from, model, optimizer)
        if state["stage"] != cfg.name:
            raise ValueError(
                f"resume_from is from stage '{state['stage']}' but cfg is for stage '{cfg.name}'. "
                f"To start a new stage from prior weights, load weights outside train_stage "
                f"(with optimizer=None) and pass a fresh cfg without resume_from."
            )
        starting_step    = state["step"]
        starting_tokens  = state["tokens"]
        skip_docs        = state["docs_per_source"]
        val_history      = state["val_history"]
        # Recover the stage anchor so the LR schedule continues from the right
        # point. Older checkpoints predate this field — fall back to assuming the
        # stage began here (restarts the schedule, but never crashes).
        stage_start_step = state.get("stage_start_step")
        log.info("resumed %s from %s at step=%d tokens=%s",
                 cfg.name, resume_from.name, starting_step, f"{starting_tokens:,}")

    log.info("building data loaders (mix=%s)", cfg.mix.name)
    train_loader = build_train_loader(
        mix=cfg.mix, context_length=cfg.context_length,
        batch_size=cfg.batch_size, skip_docs=skip_docs,
    )
    val_loaders  = build_val_loaders(
        context_length=cfg.context_length, batch_size=cfg.batch_size,
        val_docs_per_source=cfg.val_docs_per_source,
    )

    def _on_plateau(payload):
        log.warning("[plateau:%s] best=%.4f current=%.4f — val stopped improving",
                    cfg.name, payload["best"], payload["current"])
        if wandb_run is not None:
            wandb_run.alert(
                title=f"Plateau in {cfg.name}",
                text=f"best={payload['best']:.4f} current={payload['current']:.4f}",
            )

    detector = PlateauDetector(patience=5, min_delta=1e-3, cooldown=5, on_fire=_on_plateau)

    # --- stage banner: everything you'd want to reconstruct this run ---------
    tokens_per_step = cfg.batch_size * cfg.context_length * cfg.grad_accum
    n_params = model.num_parameters() if hasattr(model, "num_parameters") else sum(p.numel() for p in model.parameters())
    log.info("================ STAGE START: %s ================", cfg.name.upper())
    log_config(log, f"stage config [{cfg.name}]", cfg)
    log.info("mix weights        : %s", cfg.mix.weights)
    log.info("model params       : %s", f"{n_params:,}")
    log.info("grad checkpointing : %s", cfg.grad_checkpoint)
    log.info("tokens/opt-step    : %s  (bs=%d x ctx=%d x accum=%d)",
             f"{tokens_per_step:,}", cfg.batch_size, cfg.context_length, cfg.grad_accum)
    log.info("planned steps      : %d  (~%s tokens this stage)",
             cfg.max_steps, f"{tokens_per_step * cfg.max_steps:,}")
    log.info("starting global    : step=%d tokens=%s", starting_step, f"{starting_tokens:,}")
    log.info("optimizer          : AdamW(betas=(0.9,0.95), eps=1e-8, fused) wd=%.3g", cfg.weight_decay)
    log.info("gpu mem (pre-loop) : %s", gpu_mem_str(device))

    # Per-source token counter (for stage-level data attribution).
    tokens_per_source = {name: 0 for name in SOURCE_IDS}
    tokens = starting_tokens
    step   = starting_step    # global step — logging x-axis + checkpoint metadata
    # Stage-relative step drives the stop condition + LR schedule. On a fresh
    # stage it anchors to starting_step (so local starts at 0); on resume it is
    # recovered from the checkpoint so the schedule continues uninterrupted.
    stage_start_step = stage_start_step if stage_start_step is not None else starting_step
    local  = step - stage_start_step
    train_iter = iter(train_loader)

    t_last  = time.time()
    t_stage = t_last                  # wall-clock anchor for elapsed/ETA
    ema_tok_per_sec = None            # smoothed throughput for stable ETA
    log.info("entering training loop at local-step %d / %d", local, cfg.max_steps)
    while local < cfg.max_steps:
        lr = lr_for_step(local, cfg)
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

        # Throughput + ETA
        now = time.time()
        dt = max(now - t_last, 1e-6)
        tok_per_sec = tokens_per_step / dt
        ema_tok_per_sec = tok_per_sec if ema_tok_per_sec is None else 0.9 * ema_tok_per_sec + 0.1 * tok_per_sec
        t_last = now
        steps_left = cfg.max_steps - (local + 1)
        eta_s = steps_left * tokens_per_step / max(ema_tok_per_sec, 1e-6)
        elapsed_s = now - t_stage

        train_ppl = _safe_ppl(running_loss)
        metrics = {
            "stage":      cfg.name,
            "step":       step,
            "local_step": local,
            "tokens":     tokens,
            "loss":       running_loss,
            "ppl":        train_ppl,
            "lr":         lr,
            "grad_norm":  float(grad_norm),
            "tok_per_sec": tok_per_sec,
            "eta_hours":  eta_s / 3600.0,
            **{f"src_tokens/{k}": v for k, v in tokens_per_source.items()},
        }
        if wandb_run is not None:
            wandb_run.log(metrics, step=step)
        log.info(
            f"[{cfg.name} {local}/{cfg.max_steps} gstep={step}] "
            f"loss={running_loss:.4f} ppl={train_ppl:,.1f} lr={lr:.2e} "
            f"gnorm={float(grad_norm):.2f} tok/s={tok_per_sec:,.0f} "
            f"elapsed={_fmt_dur(elapsed_s)} eta={_fmt_dur(eta_s)}"
        )
        # Heavier per-step detail goes to the file handler only (DEBUG).
        log.debug(f"  tokens={tokens:,} gpu[{gpu_mem_str(device)}] src_tokens={tokens_per_source}")

        # Evaluation
        if local > 0 and local % cfg.eval_every == 0:
            log.info("[%s] running eval at gstep=%d (%d batches/source)...",
                     cfg.name, step, cfg.eval_batches)
            t_eval = time.time()
            val = evaluate_per_domain(model, val_loaders, device, max_batches=cfg.eval_batches)
            val_history.append({"step": step, "stage": cfg.name, **val})
            if wandb_run is not None:
                val_log = {f"val/{k}": v for k, v in val.items()}
                val_log |= {f"val_ppl/{k}": _safe_ppl(v) for k, v in val.items()}
                val_log["stage"] = cfg.name
                wandb_run.log(val_log, step=step)
            per_domain = "  ".join(f"{k}={v:.3f}" for k, v in val.items() if k != "aggregate")
            log.info(
                f"[{cfg.name} eval gstep={step}] aggregate={val['aggregate']:.4f} "
                f"(ppl={_safe_ppl(val['aggregate']):,.1f}) took={_fmt_dur(time.time()-t_eval)}  {per_domain}"
            )
            detector.update(val)

        # Periodic inference sample (qualitative progress check)
        if cfg.sample_every and local % cfg.sample_every == 0:
            text = generate_sample_text(
                model, cfg.sample_prompt, device,
                context_length=cfg.context_length,
                max_new_tokens=cfg.sample_max_new,
                temperature=cfg.sample_temperature,
                top_k=cfg.sample_top_k,
            )
            pretty = text.replace("<|endoftext|>", " ⏎ ")
            log.info(f"[{cfg.name} sample gstep={step}] {pretty}")
            if wandb_run is not None:
                wandb_run.log(
                    {"train/sample": wandb.Html(f"<pre>{pretty}</pre>")},
                    step=step,
                )

        # Checkpoint
        if local > 0 and local % cfg.ckpt_every == 0:
            path = ckpt_dir / f"qwen3_v2_{cfg.name}_step{step:06d}.pt"
            save_checkpoint(
                path, model, optimizer, cfg, step, tokens,
                docs_per_source=train_loader.dataset.docs_consumed,
                val_history=val_history,
                stage_start_step=stage_start_step,
            )
            log.info("saved checkpoint %s (gstep=%d tokens=%s) gpu[%s]",
                     path.name, step, f"{tokens:,}", gpu_mem_str(device))

        step  += 1
        local += 1

    # Final eval + checkpoint
    log.info("loop done (ran %d local-steps in %s) — final eval + checkpoint",
             local, _fmt_dur(time.time() - t_stage))
    val = evaluate_per_domain(model, val_loaders, device, max_batches=cfg.eval_batches)
    val_history.append({"step": step, "stage": cfg.name, **val})
    if wandb_run is not None:
        final_log = {f"val/{k}": v for k, v in val.items()}
        final_log |= {f"val_ppl/{k}": _safe_ppl(v) for k, v in val.items()}
        final_log["stage"] = cfg.name
        wandb_run.log(final_log, step=step)
    final_path = ckpt_dir / f"qwen3_v2_{cfg.name}_final.pt"
    save_checkpoint(
        final_path, model, optimizer, cfg, step, tokens,
        docs_per_source=train_loader.dataset.docs_consumed,
        val_history=val_history,
        stage_start_step=stage_start_step,
    )

    summary = {
        "stage":             cfg.name,
        "final_step":        step,
        "final_tokens":      tokens,
        "tokens_per_source": tokens_per_source,
        "final_val":         val,
        "final_lr":          lr_for_step(max(local - 1, 0), cfg),
        "docs_per_source":   train_loader.dataset.docs_consumed,
    }
    log.info("================ STAGE COMPLETE: %s ================", cfg.name.upper())
    log.info("saved final checkpoint : %s", final_path.name)
    log.info("local steps run        : %d / %d", local, cfg.max_steps)
    log.info("final global step      : %d", step)
    log.info("final tokens (cumul.)  : %s", f"{tokens:,}")
    log.info("tokens this stage      : %s", f"{sum(tokens_per_source.values()):,}")
    log.info("per-source tokens      : %s", tokens_per_source)
    log.info(f"final aggregate val    : {val['aggregate']:.4f} (ppl={_safe_ppl(val['aggregate']):,.1f})")
    log.info("final lr               : %.3e", summary["final_lr"])
    log.info("gpu mem (post-loop)    : %s", gpu_mem_str(device))
    if local == 0:
        log.warning("ran 0 steps this stage — check max_steps vs. starting_step! "
                    "(this is the bug class that left anneal a no-op)")
    return summary
