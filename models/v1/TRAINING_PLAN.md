# LLM Training Plan

## Goal
Understand the full LLM training process and produce a competent model.
Target architecture: **GPT-2 Small** (124M parameters).

---

## Phase 1 — Complete the Architecture

The current `src/base_model.py` is a skeleton. A competent model requires:

- **GELU activation**
- **Layer Normalization** (pre-norm style, as in GPT-2)
- **Feed-Forward Network** (2-layer MLP, 4× embedding dimension)
- **Transformer Block** (MHA + FFN + residual connections + LayerNorm)
- **Full GPT Model** (stacked blocks + final LN + LM head)

### GPT-2 Small configuration

| Hyperparameter   | Value              |
|------------------|--------------------|
| `vocab_size`     | 50257              |
| `emb_dim`        | 768                |
| `num_heads`      | 12                 |
| `num_layers`     | 12                 |
| `context_length` | 1024               |
| `ffn_dim`        | 3072 (4 × emb_dim) |

---

## Phase 2 — Training Infrastructure

Beyond the basic loop in `base_model.py`, the pipeline needs:

- **Evaluation loop** — validation loss + perplexity to track generalization
- **LR scheduler** — linear warmup followed by cosine decay (critical for convergence)
- **Gradient clipping** — `max_norm=1.0` to prevent exploding gradients
- **Checkpointing** — save best model; support resuming from a checkpoint
- **Logging** — loss curves at minimum; TensorBoard or W&B for plots

---

## Phase 3 — Dataset Strategy

Training a 124M model from scratch requires ~10B tokens for meaningful results
(Chinchilla scaling law). Two realistic paths:

### Path A — Train from scratch
*Learn everything; slower.*

Use **FineWeb-Edu** (`sample-10BT` split) — high-quality filtered web text,
exactly 10B tokens.

```python
from datasets import load_dataset
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", streaming=True)
```

Requirements: multi-GPU or weeks of single-GPU training.

### Path B — Fine-tune a pretrained checkpoint
*Faster; still covers the full architecture and training loop.*

Load OpenAI's GPT-2 weights, then continue training on a target domain.
Pre-training is skipped, but every other phase is identical.

```python
from transformers import GPT2LMHeadModel
pretrained = GPT2LMHeadModel.from_pretrained("gpt2")
# copy weights into your own GPTModel implementation
```

**Recommendation:** start with Path B, then run Path A on a smaller model
(e.g., 6 layers / 384 dim) once the pipeline is proven.

---

## Phase 4 — Evaluation

- **Perplexity** on a held-out validation split — primary metric
- **Text generation samples** — qualitative check every N steps
- Baseline reference: GPT-2 Small achieves ~29 PPL on WikiText-103

---

## Suggested Order of Work

```
1. Implement full GPT architecture          → src/gpt_model.py
2. Add training infrastructure              → src/train.py
3. Verify: overfit on the-verdict.txt in <5 min
4. Path B: load GPT-2 weights, fine-tune   → hours
5. Path A: train from scratch on FineWeb   → days / weeks
```

---

## File Map

| File                  | Purpose                                      |
|-----------------------|----------------------------------------------|
| `src/base_model.py`   | Dataset, DataLoader, attention components    |
| `src/gpt_model.py`    | Full GPT-2 architecture (to be implemented) |
| `src/train.py`        | Training + eval loop (to be implemented)    |
| `explore/base_model.ipynb` | Original exploration notebook           |
