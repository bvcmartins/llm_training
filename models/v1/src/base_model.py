"""
Base model components for a decoder-only Transformer (GPT-style).

Components:
    - GPTDatasetV1      : sliding-window token dataset
    - create_dataloader : factory for the DataLoader
    - CausalAttention   : single-head masked self-attention
    - MultiHeadAttentionWrapper : naive multi-head wrapper
    - train             : basic training loop
"""

import argparse

import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Dataset & DataLoader
# ---------------------------------------------------------------------------

class GPTDatasetV1(Dataset):
    def __init__(self, txt: str, tokenizer, max_length: int, stride: int):
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i : i + max_length]))
            self.target_ids.append(torch.tensor(token_ids[i + 1 : i + max_length + 1]))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


# ---------------------------------------------------------------------------
# Attention layers
# ---------------------------------------------------------------------------

class CausalAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, d_in = x.shape
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return attn_weights @ values


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(x) for head in self.heads], dim=-1)


# ---------------------------------------------------------------------------
# Minimal language-model head (embeddings + MHA wrapper + linear projection)
# ---------------------------------------------------------------------------

class BaseLanguageModel(nn.Module):
    """
    Minimal trainable model: token + positional embeddings ->
    MultiHeadAttentionWrapper -> linear head over vocabulary.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        context_length: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb   = nn.Embedding(context_length, emb_dim)
        self.mha = MultiHeadAttentionWrapper(
            d_in=emb_dim,
            d_out=head_dim,
            context_length=context_length,
            dropout=dropout,
            num_heads=num_heads,
        )
        # MHA concatenates num_heads * head_dim
        self.lm_head = nn.Linear(num_heads * head_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device)
        x = self.token_emb(x) + self.pos_emb(positions)  # [b, seq_len, emb_dim]
        x = self.mha(x)                                   # [b, seq_len, num_heads * head_dim]
        return self.lm_head(x)                            # [b, seq_len, vocab_size]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
) -> list[float]:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    epoch_losses: list[float] = []

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)                         # [b, seq_len, vocab_size]
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),          # [b*seq_len, vocab_size]
                targets.view(-1),                          # [b*seq_len]
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(dataloader)
        epoch_losses.append(avg)
        print(f"Epoch {epoch}/{num_epochs}  loss={avg:.4f}")

    return epoch_losses


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the base language model.")
    p.add_argument("--data", required=True, help="Path to raw text file")
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--batch-size",   type=int,   default=8)
    p.add_argument("--max-length",   type=int,   default=64,  help="Context window / sequence length")
    p.add_argument("--stride",       type=int,   default=32)
    p.add_argument("--emb-dim",      type=int,   default=128, help="Token embedding dimension")
    p.add_argument("--num-heads",    type=int,   default=4)
    p.add_argument("--head-dim",     type=int,   default=32,  help="Output dim per attention head")
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--save",         default=None, help="Path to save model weights (.pt)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.data) as f:
        raw_text = f.read()
    print(f"Loaded {len(raw_text):,} characters from {args.data}")

    dataloader = create_dataloader(
        raw_text,
        batch_size=args.batch_size,
        max_length=args.max_length,
        stride=args.stride,
        shuffle=True,
    )
    print(f"Batches per epoch: {len(dataloader)}")

    vocab_size = tiktoken.get_encoding("gpt2").n_vocab  # 50257
    model = BaseLanguageModel(
        vocab_size=vocab_size,
        emb_dim=args.emb_dim,
        context_length=args.max_length,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        dropout=args.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train(model, dataloader, optimizer, device, args.epochs)

    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"Saved weights -> {args.save}")


if __name__ == "__main__":
    main()
