# LLM Training Plan — v2

Successor to v1 (`models/v1/`, GPT-2 Large 774M on FineWeb-Edu). v2 swaps in a
modern architecture and a multi-source mix, and adds an annealing stage.

## Goals

1. Train a Qwen3-style model from scratch on a single RTX 5090 Laptop (24 GB).
2. Use a diverse pre-training mix (wiki, books, stackoverflow, arxiv, math)
   with **per-stage attribution** — every checkpoint and every wandb log line
   records the data mix that produced it.
3. Detect val-loss plateau **per domain** so we can see which sources are still
   improving when aggregate loss has flattened.

## Architecture — Qwen3-0.6B

| Knob | Value |
|---|---|
| vocab_size | 151 936 (Qwen3 tokenizer) |
| emb_dim | 1024 |
| n_layers | 28 |
| n_heads / n_kv_groups | 16 / 8 (GQA, group_size=2) |
| head_dim | 128 |
| hidden_dim (FFN) | 3072 |
| context_length (training) | 2048 |
| RoPE θ | 1 000 000 |
| Norm | RMSNorm (pre-norm) |
| FFN | SwiGLU |
| QK-norm | yes |
| Weight tying | yes |

~596M params. Scale-up config `QWEN3_CONFIG_1_7B` is in `src/qwen3_model.py`
ready for re-use once the pipeline is validated at 0.6B.

## Stages (Llama 3 recipe)

### Stage 1 — `pretrain`
Long run at fixed mixture weights. Cosine LR with warmup.

| Source | Weight |
|---|---|
| wiki | 0.15 |
| books | 0.15 |
| stackoverflow | 0.20 |
| arxiv | 0.25 |
| math | 0.25 |

Defaults: peak LR 3e-4 → 3e-5 end, 1000-step warmup, 20 000 optimizer steps,
batch 4 × accum 64 × ctx 2048 ≈ 524 288 tokens/step ≈ 10.5B tokens total.

### Stage 2 — `anneal`
Short run with up-weighted reasoning sources. Linear LR decay from the pretrain
end-LR to 0.

| Source | Weight |
|---|---|
| wiki | 0.05 |
| books | 0.05 |
| stackoverflow | 0.10 |
| arxiv | 0.40 |
| math | 0.40 |

Defaults: 2 000 steps, no warmup. ~1B tokens at the same effective batch.

## Data attribution

- Every training batch carries a `source_id` integer tensor.
- Training loop accumulates `tokens_per_source` over the stage and logs them to
  wandb on every step (keys `src_tokens/wiki`, `src_tokens/math`, ...).
- Checkpoints embed: the stage name, the stage mix weights, total tokens,
  per-source token counts, per-source doc counts, and the full val-loss history.
- A loaded checkpoint can answer "exactly what data trained this model state?"
  by reading `payload["stage"]`, `payload["stage_mix"]`, `payload["docs_per_source"]`.

## Validation + plateau detection

- One val loader per source (held-out doc prefix, no shuffling — reproducible
  across runs).
- Every `eval_every` steps: run all six val loaders, record per-domain losses
  and a simple aggregate.
- `PlateauDetector` (patience=5, min_delta=1e-3, cooldown=5) watches the
  aggregate. Fires a `print` + `wandb.alert` when no improvement is seen for
  `patience` evals. The fire payload includes per-domain losses so you can see
  which sources have flattened and which are still moving.

## Memory plan (24 GB)

- Qwen3-0.6B weights ~1.2 GB (bf16) + grads ~1.2 GB + AdamW state (fp32 m, v)
  ~4.8 GB ≈ ~7.2 GB optimizer overhead.
- Activations at batch 4 × ctx 2048: ~6–10 GB without grad checkpointing,
  ~2–3 GB with. Default is checkpointing **on**.
- If OOM: drop `batch_size` to 2 and double `grad_accum` to keep the effective
  batch constant.

## Roadmap after v2-0.6B

1. Validate the v2 pipeline end-to-end at 0.6B (datasets load, attribution
   logged, plateau detector fires correctly).
2. Re-run with `QWEN3_CONFIG_1_7B`. Same data, same stages — only the model
   config changes. Expect ~2× wall clock per step.
3. Long-context extension: increase `context_length` in a short third stage,
   reusing the same anneal-style decayed LR.
4. Post-training (separate from pre-training): SFT on instruction data, then
   DPO on preference pairs. Out of scope for v2 pre-training.
