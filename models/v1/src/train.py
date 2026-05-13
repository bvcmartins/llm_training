#!/usr/bin/env python
"""
Pretrain GPT-2 Large (774M) on FineWeb-Edu.

Defaults match explore/train.ipynb (Plan C):
  - GPT_CONFIG_774M (36L × 1280d × 20h, dropout 0.0)
  - HuggingFaceFW/fineweb-edu sample-100BT, streamed
  - Effective batch 524 288 tokens (2 × 256 × 1024)
  - 28 000 steps, 700-step warmup, cosine to 2.5e-5
  - AdamW (β=0.9/0.95, wd=0.1 on 2D params), grad_clip=1.0
  - bf16 autocast forward, fp32 master weights + AdamW states

Usage:
    nohup python src/train.py > train.log 2>&1 &
    python src/train.py --resume
    python src/train.py --max-steps 200       # short smoke run
    python src/train.py --batch-size 1 --grad-accum-steps 512   # OOM fallback
"""
import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset

from gpt_model import GPTModel, GPT_CONFIG_774M as CONFIG


# ---------- Defaults ----------
SEED             = 123
BATCH_SIZE       = 2
GRAD_ACCUM_STEPS = 256
VAL_DOCS         = 500
SHUFFLE_BUFFER   = 10_000
DATASET_NAME     = "HuggingFaceFW/fineweb-edu"
DATASET_SUBSET   = "sample-100BT"

PEAK_LR      = 2.5e-4
MIN_LR       = 2.5e-5
BETAS        = (0.9, 0.95)
EPS          = 1e-8
WEIGHT_DECAY = 0.1

MAX_STEPS    = 28_000
WARMUP_STEPS = 700

LOG_EVERY     = 50
EVAL_EVERY    = 500
EVAL_BATCHES  = 20
SAVE_EVERY    = 2_000
KEEP_LAST_N   = 3
GRAD_CLIP     = 1.0
SAMPLE_PROMPT = "The quick brown fox"

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

EOT_ID = 50256   # tiktoken GPT-2 <|endoftext|>


def log(msg: str = ""):
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t}] {msg}", flush=True)


# ---------- Dataset ----------
class PackedTextDataset(IterableDataset):
    def __init__(self, hf_dataset, context_length: int, text_field: str = "text"):
        super().__init__()
        self.hf_dataset     = hf_dataset
        self.context_length = context_length
        self.text_field     = text_field
        self.tokenizer      = tiktoken.get_encoding("gpt2")

    def __iter__(self):
        window = self.context_length + 1
        buf: list[int] = []
        for row in self.hf_dataset:
            ids = self.tokenizer.encode_ordinary(row[self.text_field])
            buf.extend(ids)
            buf.append(EOT_ID)
            while len(buf) >= window:
                chunk = buf[:window]
                buf = buf[window:]
                t = torch.tensor(chunk, dtype=torch.long)
                yield t[:-1], t[1:]


def build_dataloaders(batch_size, context_length, val_docs, shuffle_buffer, seed,
                      dataset_subset=DATASET_SUBSET):
    raw = load_dataset(DATASET_NAME, name=dataset_subset, split="train", streaming=True)
    val_docs_stream   = raw.take(val_docs)
    train_docs_stream = raw.skip(val_docs).shuffle(buffer_size=shuffle_buffer, seed=seed)

    val_ds   = PackedTextDataset(val_docs_stream,   context_length)
    train_ds = PackedTextDataset(train_docs_stream, context_length)

    val_loader   = DataLoader(val_ds,   batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=0, pin_memory=True)
    return train_loader, val_loader


# ---------- Loss / generation ----------
def calc_loss_batch(x, y, model, device, dtype):
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    use_amp = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
    return loss


@torch.no_grad()
def calc_loss_loader(data_loader, model, num_batches, device, dtype):
    was_training = model.training
    model.eval()
    losses = []
    for i, (x, y) in enumerate(data_loader):
        if i >= num_batches:
            break
        losses.append(calc_loss_batch(x, y, model, device, dtype).item())
    if was_training:
        model.train()
    return sum(losses) / max(len(losses), 1)


def perplexity(loss: float) -> float:
    return math.exp(loss)


@torch.no_grad()
def generate(model, idx, max_new_tokens, context_size, device, dtype,
             temperature=0.0, top_k=None, eos_id=None):
    was_training = model.training
    model.eval()
    use_amp = device.type == "cuda"
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
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


def log_sample(model, prompt, tokenizer, device, dtype, context_size,
               max_new_tokens=40, temperature=0.8, top_k=50):
    ids = torch.tensor(
        tokenizer.encode(prompt, allowed_special={"<|endoftext|>"}),
        dtype=torch.long, device=device,
    ).unsqueeze(0)
    out = generate(model, ids, max_new_tokens, context_size, device, dtype,
                   temperature=temperature, top_k=top_k, eos_id=EOT_ID)
    text = tokenizer.decode(out.squeeze(0).cpu().tolist()).replace("<|endoftext|>", " ⏎ ")
    log("  sample: " + text)


# ---------- Optimizer / schedule ----------
def build_optimizer(model, peak_lr, weight_decay, betas, eps, device):
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() == 2 else no_decay).append(p)

    n_dec = sum(p.numel() for p in decay)
    n_nd  = sum(p.numel() for p in no_decay)
    log(f"decay group:    {len(decay):>3} tensors  ({n_dec:>13,} params)")
    log(f"no-decay group: {len(no_decay):>3} tensors  ({n_nd:>13,} params)")

    return torch.optim.AdamW(
        [{"params": decay,    "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=peak_lr, betas=betas, eps=eps,
        fused=(device.type == "cuda"),
    )


def get_lr(step, peak_lr, min_lr, warmup_steps, max_steps):
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + (peak_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g["lr"] = lr


# ---------- Checkpoints ----------
def checkpoint_path(ckpt_dir, step):
    return ckpt_dir / f"checkpoint_step{step:06d}.pt"


def save_checkpoint(path, model, optimizer, step, train_losses, val_losses, lr_history):
    torch.save({
        "step":            step,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_losses":    train_losses,
        "val_losses":      val_losses,
        "lr_history":      lr_history,
        "config":          dict(CONFIG),
    }, path)


def save_checkpoint_rotated(ckpt_dir, model, optimizer, step, train_losses, val_losses,
                            lr_history, keep_last_n):
    p = checkpoint_path(ckpt_dir, step)
    save_checkpoint(p, model, optimizer, step, train_losses, val_losses, lr_history)
    save_checkpoint(ckpt_dir / "checkpoint_latest.pt",
                    model, optimizer, step, train_losses, val_losses, lr_history)
    step_files = sorted(ckpt_dir.glob("checkpoint_step*.pt"))
    for old in step_files[:-keep_last_n]:
        old.unlink()
    return p


def load_checkpoint(path, model, optimizer, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


# ---------- Training loop ----------
def train(
    model, optimizer, train_loader, val_loader, tokenizer, device, dtype, *,
    max_steps, grad_accum_steps, batch_size, context_length, grad_clip,
    eval_every, eval_batches, save_every, log_every, sample_prompt,
    peak_lr, min_lr, warmup_steps,
    ckpt_dir, keep_last_n,
    start_step=0, train_losses=None, val_losses=None, lr_history=None,
):
    train_losses = train_losses if train_losses is not None else []
    val_losses   = val_losses   if val_losses   is not None else []
    lr_history   = lr_history   if lr_history   is not None else []

    model.train()
    optimizer.zero_grad(set_to_none=True)

    step       = start_step
    micro_step = 0
    accum_loss = 0.0
    train_iter = iter(train_loader)
    t0         = time.time()

    while step < max_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        loss = calc_loss_batch(x, y, model, device, dtype) / grad_accum_steps
        loss.backward()
        accum_loss += loss.item() * grad_accum_steps
        micro_step += 1

        if micro_step % grad_accum_steps != 0:
            continue

        lr = get_lr(step, peak_lr, min_lr, warmup_steps, max_steps)
        set_lr(optimizer, lr)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        mean_train_loss = accum_loss / grad_accum_steps
        train_losses.append(mean_train_loss)
        lr_history.append(lr)
        accum_loss = 0.0
        step += 1

        if step % log_every == 0:
            elapsed     = time.time() - t0
            local_step  = step - start_step
            tok_per_sec = local_step * grad_accum_steps * batch_size * context_length / max(elapsed, 1e-9)
            log(f"step {step:6d}/{max_steps} | lr {lr:.2e} | train_loss {mean_train_loss:.3f} | {tok_per_sec/1e3:.1f}K tok/s")

        if step % eval_every == 0 or step == max_steps:
            val_loss = calc_loss_loader(val_loader, model, eval_batches, device, dtype)
            val_losses.append((step, val_loss))
            log(f"  eval @ {step}: val_loss {val_loss:.3f} | PPL {perplexity(val_loss):,.1f}")
            log_sample(model, sample_prompt, tokenizer, device, dtype, context_length)
            model.train()

        if step % save_every == 0 or step == max_steps:
            p = save_checkpoint_rotated(ckpt_dir, model, optimizer, step,
                                        train_losses, val_losses, lr_history, keep_last_n)
            log(f"  saved {p.name}")

    return train_losses, val_losses, lr_history


# ---------- Main ----------
def parse_args():
    p = argparse.ArgumentParser(description="Pretrain GPT-2 Large on FineWeb-Edu")
    p.add_argument("--resume",           action="store_true",   help="resume from checkpoint_latest.pt")
    p.add_argument("--seed",             type=int,   default=SEED)
    p.add_argument("--max-steps",        type=int,   default=MAX_STEPS)
    p.add_argument("--warmup-steps",     type=int,   default=WARMUP_STEPS)
    p.add_argument("--batch-size",       type=int,   default=BATCH_SIZE)
    p.add_argument("--grad-accum-steps", type=int,   default=GRAD_ACCUM_STEPS)
    p.add_argument("--peak-lr",          type=float, default=PEAK_LR)
    p.add_argument("--min-lr",           type=float, default=MIN_LR)
    p.add_argument("--weight-decay",     type=float, default=WEIGHT_DECAY)
    p.add_argument("--grad-clip",        type=float, default=GRAD_CLIP)
    p.add_argument("--dataset-subset",   type=str,   default=DATASET_SUBSET)
    p.add_argument("--eval-every",       type=int,   default=EVAL_EVERY)
    p.add_argument("--eval-batches",     type=int,   default=EVAL_BATCHES)
    p.add_argument("--save-every",       type=int,   default=SAVE_EVERY)
    p.add_argument("--log-every",        type=int,   default=LOG_EVERY)
    p.add_argument("--keep-last-n",      type=int,   default=KEEP_LAST_N)
    p.add_argument("--sample-prompt",    type=str,   default=SAMPLE_PROMPT)
    p.add_argument("--checkpoint-dir",   type=Path,  default=CHECKPOINT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    args.checkpoint_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    log(f"device:        {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'})")
    log(f"dtype:         {dtype}")
    log(f"checkpoint:    {args.checkpoint_dir}")
    log(f"dataset:       {DATASET_NAME} / {args.dataset_subset}")
    log(f"max_steps:     {args.max_steps:,}")
    log(f"warmup_steps:  {args.warmup_steps:,}")
    log(f"batch_size:    {args.batch_size}  (× grad_accum {args.grad_accum_steps} = {args.batch_size * args.grad_accum_steps} seqs / step)")
    log(f"effective tok/step: {args.batch_size * args.grad_accum_steps * CONFIG['context_length']:,}")
    log(f"peak_lr / min_lr:   {args.peak_lr} / {args.min_lr}")

    log("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(
        args.batch_size, CONFIG["context_length"], VAL_DOCS, SHUFFLE_BUFFER, args.seed,
        dataset_subset=args.dataset_subset,
    )
    tokenizer = tiktoken.get_encoding("gpt2")

    log("Building model...")
    model = GPTModel(CONFIG).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model params:  {n_params:,}  ({n_params * 4 / 1024**2:.0f} MB fp32)")

    optimizer = build_optimizer(model, args.peak_lr, args.weight_decay, BETAS, EPS, device)

    start_step   = 0
    train_losses, val_losses, lr_history = [], [], []
    latest = args.checkpoint_dir / "checkpoint_latest.pt"
    if args.resume and latest.exists():
        log(f"Resuming from {latest.name}...")
        payload = load_checkpoint(latest, model, optimizer, device)
        start_step   = payload["step"]
        train_losses = payload.get("train_losses", [])
        val_losses   = payload.get("val_losses", [])
        lr_history   = payload.get("lr_history", [])
        log(f"  resumed at step {start_step:,}")
    elif args.resume:
        log(f"--resume requested but {latest} not found — starting fresh.")

    train(
        model, optimizer, train_loader, val_loader, tokenizer, device, dtype,
        max_steps=args.max_steps,
        grad_accum_steps=args.grad_accum_steps,
        batch_size=args.batch_size,
        context_length=CONFIG["context_length"],
        grad_clip=args.grad_clip,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        save_every=args.save_every,
        log_every=args.log_every,
        sample_prompt=args.sample_prompt,
        peak_lr=args.peak_lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        ckpt_dir=args.checkpoint_dir,
        keep_last_n=args.keep_last_n,
        start_step=start_step,
        train_losses=train_losses,
        val_losses=val_losses,
        lr_history=lr_history,
    )
    log("Training complete.")


if __name__ == "__main__":
    main()
