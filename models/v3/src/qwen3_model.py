"""Qwen3-style decoder-only transformer.

Architecture:
- RMSNorm (pre-norm)
- Rotary positional embeddings (RoPE) with theta=1e6
- Grouped Query Attention (GQA) with QK-norm
- SwiGLU FFN
- Tied input/output embeddings

Two configs are exposed: QWEN3_CONFIG_0_6B (~596M params) is the default for
from-scratch training on a 24 GB GPU. QWEN3_CONFIG_1_7B is provided for the
scale-up after the v2 pipeline is validated.

The context_length here is the *training* context, not Qwen3's native 40 960.
We train shorter to keep step cost down; RoPE supports extension later.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


QWEN3_CONFIG_0_6B = {
    "vocab_size":      151_936,
    "context_length":   2_048,   # training context; native is 40 960
    "emb_dim":          1_024,
    "n_layers":            28,
    "n_heads":             16,
    "n_kv_groups":          8,   # GQA: 2 Q heads share each KV head
    "head_dim":           128,
    "hidden_dim":       3_072,
    "rope_base":    1_000_000.0,
    "qk_norm":           True,
    "rms_eps":           1e-6,
    "dtype":      torch.bfloat16,
}

QWEN3_CONFIG_1_5B = {
    "vocab_size":      151_936,
    "context_length":   2_048,   # training context; native is 40 960
    "emb_dim":          1_792,
    "n_layers":            28,
    "n_heads":             16,
    "n_kv_groups":          8,   # GQA: 2 Q heads share each KV head
    "head_dim":           128,
    "hidden_dim":       6_144,
    "rope_base":    1_000_000.0,
    "qk_norm":           True,
    "rms_eps":           1e-6,
    "dtype":      torch.bfloat16,
}

QWEN3_CONFIG_1_7B = {
    "vocab_size":      151_936,
    "context_length":   2_048,
    "emb_dim":          2_048,
    "n_layers":            28,
    "n_heads":             16,
    "n_kv_groups":          8,
    "head_dim":           128,
    "hidden_dim":       6_144,
    "rope_base":    1_000_000.0,
    "qk_norm":           True,
    "rms_eps":           1e-6,
    "dtype":      torch.bfloat16,
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_f32 = x.float()
        var = x_f32.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f32 * torch.rsqrt(var + self.eps)
        return (x_norm * self.scale).to(in_dtype)


def precompute_rope_cache(head_dim: int, context_length: int, base: float, device=None):
    """Return (cos, sin) tables of shape [context_length, head_dim]."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    positions = torch.arange(context_length, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)            # [T, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)             # [T, head_dim]
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, n_heads, T, head_dim]. cos/sin: [T, head_dim]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat((-x2, x1), dim=-1)
    cos = cos[: x.shape[-2]].unsqueeze(0).unsqueeze(0)
    sin = sin[: x.shape[-2]].unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rot * sin)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.n_heads      = cfg["n_heads"]
        self.n_kv_groups  = cfg["n_kv_groups"]
        self.head_dim     = cfg["head_dim"]
        self.group_size   = self.n_heads // self.n_kv_groups
        assert self.n_heads % self.n_kv_groups == 0

        emb_dim = cfg["emb_dim"]
        q_dim   = self.n_heads     * self.head_dim
        kv_dim  = self.n_kv_groups * self.head_dim

        self.q_proj = nn.Linear(emb_dim, q_dim,  bias=False)
        self.k_proj = nn.Linear(emb_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(emb_dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim,   emb_dim, bias=False)

        if cfg["qk_norm"]:
            self.q_norm = RMSNorm(self.head_dim, eps=cfg["rms_eps"])
            self.k_norm = RMSNorm(self.head_dim, eps=cfg["rms_eps"])
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads,     self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_groups, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Repeat KV heads to match Q heads (GQA expansion).
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # Flash-friendly path: PyTorch picks the best backend.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        emb_dim    = cfg["emb_dim"]
        hidden_dim = cfg["hidden_dim"]
        self.gate_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.up_proj   = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, emb_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.attn_norm = RMSNorm(cfg["emb_dim"], eps=cfg["rms_eps"])
        self.attn      = GroupedQueryAttention(cfg)
        self.ffn_norm  = RMSNorm(cfg["emb_dim"], eps=cfg["rms_eps"])
        self.ffn       = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Qwen3Model(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.blocks  = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.norm_f  = RMSNorm(cfg["emb_dim"], eps=cfg["rms_eps"])
        self.lm_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # Weight tying.
        self.lm_head.weight = self.tok_emb.weight

        cos, sin = precompute_rope_cache(cfg["head_dim"], cfg["context_length"], cfg["rope_base"])
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.use_grad_checkpoint = False
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def enable_grad_checkpointing(self, enabled: bool = True):
        self.use_grad_checkpoint = enabled

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(input_ids)
        cos, sin = self.rope_cos, self.rope_sin
        for block in self.blocks:
            if self.use_grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)
        x = self.norm_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n
