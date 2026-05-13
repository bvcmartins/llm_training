"""Qwen3 tokenizer wrapper.

Downloads the official Qwen3 tokenizer.json via huggingface_hub and exposes a
thin encode/decode interface. Vocab size 151 936, includes <|endoftext|>
(id 151 643) which we use as the inter-document separator during packing.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


DEFAULT_REPO = "Qwen/Qwen3-0.6B-Base"
EOT_TOKEN    = "<|endoftext|>"


@lru_cache(maxsize=4)
def get_tokenizer(repo_id: str = DEFAULT_REPO) -> Tokenizer:
    path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
    return Tokenizer.from_file(str(path))


def eot_id(repo_id: str = DEFAULT_REPO) -> int:
    tok = get_tokenizer(repo_id)
    return tok.token_to_id(EOT_TOKEN)


def encode(text: str, repo_id: str = DEFAULT_REPO) -> list[int]:
    return get_tokenizer(repo_id).encode(text).ids


def decode(ids: list[int], repo_id: str = DEFAULT_REPO) -> str:
    return get_tokenizer(repo_id).decode(ids)
