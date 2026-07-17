# W&B Metrics Reference — v3 pretrain/anneal

Every metric this project sends to Weights & Biases, what it means, how it is
computed, and how it should behave. Source of truth: `src/training.py`
(`train_stage`), `src/eval.py`, and the `wandb.init(...)` call in `src/run.py`.

All time-series are logged with `wandb_run.log(metrics, step=step)` where `step`
is the **global** optimizer step — that is the x-axis for every chart below.

Two loss conventions used throughout:
- **loss** is mean token-level **cross-entropy in nats** (natural log).
- **ppl** (perplexity) = `exp(loss)`, via `_safe_ppl` which clamps the exponent
  at 30 to avoid overflow (`math.exp(min(loss, 30.0))`). So any `ppl` ≥ ~1.07e13
  is saturated and only means "very large".

---

## 1. Per-step training metrics

Logged **every optimizer step** (`training.py:466-480`). One optimizer step =
`grad_accum` micro-batches of `batch_size × context_length` tokens.

| W&B key | Type | Meaning | How it's computed |
|---|---|---|---|
| `stage` | str | Current stage (`pretrain` / `anneal`). | `cfg.name`. |
| `step` | int | **Global** step — the logging x-axis and checkpoint id. | Continues across resume (recovered from checkpoint). |
| `local_step` | int | Stage-relative step (0 at the start of *this* stage). | `step − stage_start_step`. Drives the LR schedule and the `max_steps` stop. |
| `tokens` | int | Cumulative tokens processed by this run. | `+= batch_size × context_length × grad_accum` each step (resumes from the checkpoint's token count). |
| `loss` | float | Training cross-entropy (nats) for this step's effective batch. | Sum over `grad_accum` micro-batches of `F.cross_entropy(logits, y) / grad_accum` → i.e. the **mean per-token CE across the whole effective batch**. |
| `ppl` | float | Training perplexity for the step. | `exp(loss)` (clamped). |
| `lr` | float | Learning rate applied this step. | `lr_for_step(local_step)` — see §5. |
| `grad_norm` | float | Global L2 norm of the gradient **before** clipping. | Return value of `clip_grad_norm_(params, grad_clip)`. Clipping itself caps the *applied* norm at `cfg.grad_clip`, but this logs the pre-clip value. |
| `tok_per_sec` | float | Instantaneous throughput. | `tokens_per_step / dt`, where `dt` is wall-clock since the previous step. |
| `eta_hours` | float | Estimated hours remaining to `max_steps`. | `steps_left × tokens_per_step / ema_tok_per_sec / 3600`, using an EMA (0.9/0.1) of throughput. |
| `src_tokens/<source>` | int | Cumulative tokens drawn from each source. | 8 keys: `wiki`, `books`, `stackoverflow`, `arxiv`, `math`, `fineweb_edu`, `finemath`, `pes2o`. Each window is tagged with the source of its first token; that window's tokens are added to that source. |

### Expected behaviour
- **`loss` / `ppl`**: noisy step-to-step **sawtooth**, overall **downward** trend.
  The sawtooth is *not* instability — each window is dominated by one source, and
  consecutive steps sample different sources (highly-predictable arxiv/code/math
  read low; diverse `wiki`/`fineweb_edu` read high). Judge progress by the trend
  / a moving average, never a single step.
- **`grad_norm`**: should stay **bounded and roughly flat** (healthy runs here sit
  ~0.2–0.7). Sustained growth or spikes to many× the norm signal instability
  (LR too high, bad batch, numerical issue).
- **`tok_per_sec`**: roughly constant (~3,700 on the laptop 5090). It **dips on
  eval/checkpoint steps** because the *next* step's `dt` absorbs the eval +
  ~9 GB checkpoint write time; `eta_hours` bounces up on those same steps. The
  EMA smooths, but expect periodic teeth aligned with `eval_every`/`ckpt_every`.
- **`eta_hours`**: trends down over the run; jumps around eval steps (see above).
- **`src_tokens/<source>`**: **monotonically increasing**, each slope proportional
  to its mix weight. Useful to confirm the mixture is actually being sampled as
  configured, and to spot a source that has been exhausted and is looping.

---

## 2. Per-eval (validation) metrics

Logged every `cfg.eval_every` steps and once at stage end
(`training.py:497-501`, `574-580`). Computed by `evaluate_per_domain`
(`eval.py`) over `cfg.eval_batches` batches per source from the **held-out**
val loaders (see §6).

| W&B key | Type | Meaning |
|---|---|---|
| `val/<source>` | float | Held-out cross-entropy (nats) for each domain. 8 source keys. |
| `val/aggregate` | float | **Unweighted** mean of the per-source val losses. |
| `val_ppl/<source>` | float | `exp(val/<source>)` — per-domain held-out perplexity. |
| `val_ppl/aggregate` | float | `exp(val/aggregate)`. |
| `stage` | str | Stage name. |

Note `val/aggregate` is a **plain average across the 8 domains** — it deliberately
does *not* weight by the training mix, so a single regressing domain is visible
rather than being drowned out.

### Expected behaviour
- **`val/aggregate` should trend DOWN** and sit **modestly above** the training
  `loss` (val is held-out and unweighted; a small train↔val gap is normal).
- ⚠️ **History caveat (pre-2026-07-14):** before the held-out fix, the val set was
  the *first* N docs of each source — which were **inside** the training stream.
  So the old `val_*` curves *rose* (18→87 ppl) as the model drifted away from that
  stale in-sample slice; that was a **broken metric, not overfitting**. Runs after
  the fix use a reserved holdout (train never sees it), so a rising `val/aggregate`
  now genuinely means over-fitting / instability and should be investigated.
- A per-domain `val/<source>` climbing while others fall points to that domain
  being under-served or looping — cross-check `src_tokens/<source>`.

---

## 3. Qualitative sample

Logged every `cfg.sample_every` steps (`training.py:520-524`).

| W&B key | Type | Meaning |
|---|---|---|
| `train/sample` | HTML | A short greedy/sampled generation from `cfg.sample_prompt`, `<|endoftext|>` rendered as `⏎`. Eyeball check for coherence — not a number. |

**Expected:** early on, repetitive/looping text; with training, progressively more
fluent and on-topic. (At ~9 B tokens this 1.5 B model is fluent but repetitive —
undertrained, not broken.)

---

## 4. Run config (not a time-series)

Set once at `wandb.init(config=...)` (`run.py:183-194`); appears in the run's
**Overview → Config**, and is filterable/groupable across runs:

- `model_config` — the full `QWEN3_CONFIG_1_5B` dict (dims, layers, heads,
  `rope_base`, `dtype`, `context_length`, …).
- `stage`, `mix` (the source→weight dict), `resume`, `init_from`.
- All `StageConfig` fields except `mix`: `max_steps`, `batch_size`, `grad_accum`,
  `lr_peak`, `lr_end`, `warmup_steps`, `schedule`, `weight_decay`, `grad_clip`,
  `eval_every`, `eval_batches`, `ckpt_every`, `sample_every`, `sample_*`,
  `stop_at`, `model_tag`, `model_config_ref`, …

---

## 5. Learning-rate schedule (`lr_for_step`, `training.py:142`)

Driven by `local_step` (stage-relative), so it **continues uninterrupted across a
resume** — no warmup restart when the daily cron resumes.

1. **Warmup** (`local_step < warmup_steps`, default 1000): linear ramp
   `lr_peak × (local_step+1) / warmup_steps` from ~0 → `lr_peak` (3e-4).
2. **Decay** (after warmup): over `progress = (local_step − warmup) / (max_steps − warmup)`:
   - `cosine` (pretrain): `lr_end + 0.5·(lr_peak − lr_end)·(1 + cos(π·progress))`
     — smooth peak→`lr_end` (3e-5).
   - `linear` (anneal): straight line `lr_peak → lr_end`.

**Expected `lr` chart:** a short linear ramp, then a smooth cosine (or linear)
glide down — **no sawtooth** at resume boundaries. A sawtooth would mean
`stage_start_step` wasn't restored.

Optimizer: fused **AdamW**, β=(0.9, 0.95), eps=1e-8. Weight decay applies **only**
to ≥2-D matrices (Linear weights); norms, biases and `tok_emb` are excluded
(`build_optimizer`).

---

## 6. Held-out validation set (post-2026-07-14)

`VAL_HOLDOUT_DOCS = 1000` (`data.py`): the **first 1000 documents of every
source are reserved for validation only**. The training stream always
`.skip(VAL_HOLDOUT_DOCS + cursor)`, and `build_val_loaders` reads only inside
that prefix — so **train and val are provably disjoint**. This is what makes the
`val/*` curves a real generalization signal. (See `test_holdout` verification.)

---

## Quick reading guide

| If you want to know… | Look at |
|---|---|
| Is it learning? | `val/aggregate` (down) + `train/sample` (more coherent). `loss` alone is too noisy. |
| Is it stable? | `grad_norm` (flat/bounded) and `lr` (smooth, no sawtooth). |
| Is a domain regressing? | per-domain `val/<source>` vs the others. |
| Is the mixture right / is a source looping? | slopes of `src_tokens/<source>`. |
| How much longer? | `eta_hours` (bounces on eval steps; trust the trend). |
| Over/under-fitting? | gap between training `loss` and `val/aggregate` (post-fix only). |
