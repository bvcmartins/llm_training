# v3 `src/` — code graph

Three views of `models/v3/src/`, zooming in from module wiring → runtime flow →
model internals. All diagrams are Mermaid (renders natively on GitHub/GitLab,
VS Code with the Mermaid extension, Obsidian, etc.).

---

## 1. Module dependency graph

Who imports whom. Three layers: thin CLI entrypoints on top, the
stage-agnostic engine in the middle, leaf components at the bottom. Note
`training.py` (the engine) never imports an entrypoint — the dependency arrows
only point *down*, which is what keeps `train_stage()` reusable by all three
launchers.

```mermaid
flowchart TD
    subgraph CLI["entrypoints (CLI)"]
        run["run.py<br/><i>unified staged runner</i>"]
        pre["pretrain.py<br/><i>stage 1</i>"]
        ann["anneal.py<br/><i>stage 2</i>"]
    end
    subgraph ENG["engine / shared setup"]
        train["training.py<br/><i>train_stage()</i>"]
        boot["bootstrap.py"]
    end
    subgraph COMP["components"]
        data["data.py"]
        evalm["eval.py"]
        model["qwen3_model.py"]
        tok["tokenizer.py"]
        logu["logging_utils.py"]
    end

    run --> model
    run --> train
    run -. "imports config<br/>factories" .-> pre
    run -.-> ann

    pre --> boot
    pre --> train
    pre --> data
    pre --> logu

    ann --> boot
    ann --> train
    ann --> data
    ann --> logu

    boot --> model
    train --> data
    train --> evalm
    train --> tok
    train --> logu
    data --> tok
```

Leaves with no first-party imports (`eval.py`, `qwen3_model.py`,
`logging_utils.py`, `tokenizer.py`) depend only on torch / HF — they're the
safe-to-test-in-isolation parts.

---

## 2. Runtime call graph (a training stage)

What fires when you run `python run.py --stage pretrain` (or `pretrain.py` /
`anneal.py` — all three converge on `train_stage()`). Solid = direct call,
dashed = "used inside the loop each step/eval/ckpt".

```mermaid
flowchart TD
    main["main()<br/>(run / pretrain / anneal)"]
    main --> bm["Qwen3Model(cfg)<br/>(via build_model)"]
    main --> bsc["default_pretrain_config /<br/>default_anneal_config"]
    main --> lrs0["load_resume_state(optimizer=None)<br/><i>--init-from: weights only</i>"]
    main --> ts["train_stage()"]

    ts --> bo["build_optimizer()"]
    ts --> lrs["load_resume_state()<br/><i>resume: model+opt+counters</i>"]
    ts --> btl["build_train_loader()"]
    ts --> bvl["build_val_loaders()"]
    ts --> pd["PlateauDetector"]

    ts -. "each step" .-> lfs["lr_for_step()"]
    ts -. "each step" .-> fwd["model.forward()<br/>+ F.cross_entropy"]
    ts -. "each eval" .-> epd["evaluate_per_domain()"]
    ts -. "each eval" .-> upd["detector.update()"]
    ts -. "each sample" .-> gst["generate_sample_text()"]
    ts -. "each ckpt" .-> sc["save_checkpoint()"]

    gst --> gen["generate()"]
    gst --> enc["encode / decode / eot_id"]
    gen --> fwd
    epd --> fwd

    btl --> msd["MultiSourcePackedDataset"]
    bvl --> ssd["SingleSourcePackedDataset"]
    msd --> gt["get_tokenizer / eot_id"]
    ssd --> gt
```

The four dashed-loop nodes (`lr_for_step`, eval, sample, checkpoint) are gated
by `cfg.eval_every` / `sample_every` / `ckpt_every`. `save_checkpoint` is also
called twice outside the loop: once on graceful stop (resumable step ckpt) and
once on completion (`_final.pt`).

---

## 3. Model class composition (`qwen3_model.py`)

The only genuinely hierarchical part — `nn.Module` nesting. `▢` = submodule
attribute, dashed = stateless helper function used in `forward`.

```mermaid
flowchart TD
    Q["Qwen3Model"]
    Q --> emb["tok_emb : nn.Embedding"]
    Q --> blk["blocks : TransformerBlock × n_layers"]
    Q --> nf["norm_f : RMSNorm"]
    Q --> lm["lm_head : nn.Linear<br/><i>weight tied to tok_emb</i>"]
    Q -. "buffers (init)" .-> prc["precompute_rope_cache()"]

    blk --> an["attn_norm : RMSNorm"]
    blk --> at["attn : GroupedQueryAttention"]
    blk --> fn["ffn_norm : RMSNorm"]
    blk --> ff["ffn : SwiGLU"]

    at --> qkvo["q_proj / k_proj / v_proj / o_proj : nn.Linear"]
    at --> qkn["q_norm / k_norm : RMSNorm (QK-norm)"]
    at -. "in forward" .-> ar["apply_rope()"]

    ff --> gud["gate_proj / up_proj / down_proj : nn.Linear"]
```

Pre-norm residual flow per block: `x = x + attn(attn_norm(x)); x = x +
ffn(ffn_norm(x))`. RoPE (`cos/sin`) is precomputed once and threaded through
every block's attention.

---

### Regenerating these automatically

If you'd rather have these derived from the source (so they can't drift), the
usual tools are:

- **`pydeps src/`** → module-import graph (view 1) as SVG/dot.
- **`pyan3 src/*.py --uses --no-defaults --dot`** → call graph (view 2).
- **`code2flow src/`** → lightweight call graph, quick to run.

These emit dot/SVG rather than Markdown, so the Mermaid above is the
repo-friendly hand-curated equivalent.
