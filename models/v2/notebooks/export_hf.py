#!/usr/bin/env python
"""Export a from-scratch v2 checkpoint to a HuggingFace `Qwen3ForCausalLM` folder.

Our `Qwen3Model` (src/qwen3_model.py) is architecturally identical to the HF
Qwen3 base: GQA + per-head QK-norm, SwiGLU MLP, RMSNorm pre-norm, RoPE theta=1e6,
tied input/output embeddings. So the export is a pure weight-key remap plus a
`config.json` derived from QWEN3_CONFIG_0_6B and the official tokenizer files.

The resulting directory loads with `AutoModelForCausalLM.from_pretrained(...)` and
is consumed directly by EleutherAI lm-evaluation-harness:

    lm_eval --model hf \
            --model_args pretrained=<out_dir>,dtype=bfloat16 \
            --tasks lambada_openai,hellaswag,wikitext \
            --device cuda:0 --batch_size auto

Usage:
    uv run python notebooks/export_hf.py \
        --ckpt checkpoints/qwen3_v2_anneal_final.pt \
        --out  exports/qwen3_v2_anneal_hf

By default it numerically verifies the HF model reproduces the native model's
logits (max-abs-diff over a random batch) before writing — pass --no-verify to skip.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# notebooks/ lives under models/v2; src/ is its sibling.
V2_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = V2_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qwen3_model import QWEN3_CONFIG_0_6B, QWEN3_CONFIG_1_7B, Qwen3Model  # noqa: E402

CONFIGS = {
    "qwen3_model.QWEN3_CONFIG_0_6B": QWEN3_CONFIG_0_6B,
    "qwen3_model.QWEN3_CONFIG_1_7B": QWEN3_CONFIG_1_7B,
}
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def remap_state_dict(sd: dict, n_layers: int) -> dict:
    """Map our parameter names onto HF Qwen3 names.

    Top level:
      tok_emb.weight                -> model.embed_tokens.weight
      norm_f.scale                  -> model.norm.weight
      lm_head.weight                -> dropped (HF ties it to embed_tokens)
    Per block i:
      blocks.i.attn_norm.scale      -> model.layers.i.input_layernorm.weight
      blocks.i.attn.{q,k,v,o}_proj  -> model.layers.i.self_attn.{q,k,v,o}_proj.weight
      blocks.i.attn.{q,k}_norm.scale-> model.layers.i.self_attn.{q,k}_norm.weight
      blocks.i.ffn_norm.scale       -> model.layers.i.post_attention_layernorm.weight
      blocks.i.ffn.{gate,up,down}   -> model.layers.i.mlp.{gate,up,down}_proj.weight
    """
    out: dict = {}
    out["model.embed_tokens.weight"] = sd["tok_emb.weight"]
    out["model.norm.weight"] = sd["norm_f.scale"]
    # lm_head intentionally dropped: tie_word_embeddings=True.

    for i in range(n_layers):
        b = f"blocks.{i}."
        L = f"model.layers.{i}."
        out[L + "input_layernorm.weight"] = sd[b + "attn_norm.scale"]
        out[L + "post_attention_layernorm.weight"] = sd[b + "ffn_norm.scale"]
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            out[L + f"self_attn.{proj}.weight"] = sd[b + f"attn.{proj}.weight"]
        out[L + "self_attn.q_norm.weight"] = sd[b + "attn.q_norm.scale"]
        out[L + "self_attn.k_norm.weight"] = sd[b + "attn.k_norm.scale"]
        for proj in ("gate_proj", "up_proj", "down_proj"):
            out[L + f"mlp.{proj}.weight"] = sd[b + f"ffn.{proj}.weight"]
    return out


def build_hf_config(cfg: dict, max_pos: int):
    from transformers import Qwen3Config

    return Qwen3Config(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["emb_dim"],
        intermediate_size=cfg["hidden_dim"],
        num_hidden_layers=cfg["n_layers"],
        num_attention_heads=cfg["n_heads"],
        num_key_value_heads=cfg["n_kv_groups"],
        head_dim=cfg["head_dim"],
        hidden_act="silu",
        max_position_embeddings=max_pos,
        rms_norm_eps=cfg["rms_eps"],
        rope_theta=cfg["rope_base"],
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=True,
        use_cache=True,
        # Qwen3 has no sliding window in the base models.
        sliding_window=None,
        use_sliding_window=False,
        bos_token_id=None,
        eos_token_id=151643,  # <|endoftext|>
        torch_dtype="bfloat16",
    )


@torch.no_grad()
def verify(native_sd: dict, cfg: dict, hf_model, dtype, seq_len: int = 32, vocab_cap: int = 1000):
    """Build the native model, run both on the same batch, report max-abs logit diff."""
    native = Qwen3Model(cfg).to(dtype)
    native.load_state_dict(native_sd, strict=True)
    native.eval()

    torch.manual_seed(0)
    ids = torch.randint(0, vocab_cap, (1, seq_len))
    native_logits = native(ids).float()
    hf_logits = hf_model(ids).logits.float()
    diff = (native_logits - hf_logits).abs().max().item()
    # Also check top-1 next-token agreement across positions.
    agree = (native_logits.argmax(-1) == hf_logits.argmax(-1)).float().mean().item()
    return diff, agree


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tokenizer-repo", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    ap.add_argument("--max-position-embeddings", type=int, default=40_960,
                    help="Native Qwen3 is 40960. The v2 model was *trained* at context 2048; "
                         "RoPE extends but quality past 2048 is untested. Set 2048 to be strict.")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]

    print(f"Loading checkpoint {args.ckpt} ...")
    ck = torch.load(args.ckpt, map_location="cpu", mmap=True, weights_only=False)
    ref = ck.get("model_config_ref", "qwen3_model.QWEN3_CONFIG_0_6B")
    cfg = CONFIGS[ref]
    sd = {k: v.to(dtype) for k, v in ck["model"].items()}
    print(f"  stage={ck.get('stage')} step={ck.get('step')} config={ref} "
          f"tensors={len(sd)} dtype->{args.dtype}")

    print("Remapping weight keys -> HF Qwen3 layout ...")
    hf_sd = remap_state_dict(sd, cfg["n_layers"])

    print("Instantiating Qwen3ForCausalLM ...")
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    config = build_hf_config(cfg, args.max_position_embeddings)
    with torch.device("cpu"):
        model = Qwen3ForCausalLM(config).to(dtype)
    missing, unexpected = model.load_state_dict(hf_sd, strict=False)
    # Only the tied lm_head.weight may be "missing" (it aliases embed_tokens).
    missing = [m for m in missing if m != "lm_head.weight"]
    if missing or unexpected:
        raise SystemExit(f"State-dict mismatch!\n  missing={missing}\n  unexpected={unexpected}")
    model.tie_weights()
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"  loaded cleanly. params={n/1e6:.1f}M (counting tied embedding once)")

    if not args.no_verify:
        print("Verifying against native model (logit parity) ...")
        diff, agree = verify(sd, cfg, model, dtype)
        print(f"  max|Δlogit|={diff:.3e}  top1-agreement={agree*100:.1f}%")
        if agree < 0.999:
            raise SystemExit("Verification FAILED: HF logits diverge from native model.")
        print("  ✓ conversion is faithful")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Saving model -> {args.out}")
    model.save_pretrained(args.out, safe_serialization=True)
    print(f"Saving tokenizer ({args.tokenizer_repo}) -> {args.out}")
    AutoTokenizer.from_pretrained(args.tokenizer_repo).save_pretrained(args.out)

    print("\nDone. Run benchmarks with:")
    print(f"  uv run lm_eval --model hf \\")
    print(f"    --model_args pretrained={args.out},dtype={args.dtype} \\")
    print(f"    --tasks lambada_openai,hellaswag,wikitext,arc_easy,piqa,winogrande \\")
    print(f"    --device cuda:0 --batch_size auto")


if __name__ == "__main__":
    main()
