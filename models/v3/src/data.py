"""Multi-source streaming dataloader for v2 training.

Each *source* is a HuggingFace streaming dataset with a name, text field, and a
human label. A *mixture* assigns sampling weights to sources for a given
*stage* (pre-train mix vs anneal mix).

Every batch carries a `source_id` tensor so the training loop can log per-source
token counts and the eval loop can hold out per-source validation sets.

Sources are config-driven — swap HF dataset ids in SOURCES if a default is wrong
for your account or has been renamed. Defaults below are reasonable picks at
the time of writing; if a `load_dataset` call fails, change the id here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterator

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader

from tokenizer import get_tokenizer, eot_id


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceSpec:
    name:        str         # short label used in logs/checkpoints
    hf_path:     str         # HuggingFace dataset id
    hf_config:   str | None  # subset/config name (or None)
    text_field:  str         # which column holds the document text
    split:       str = "train"


# Canonical sources — all parquet-native mirrors (no remote loading scripts,
# which `datasets` ≥3.0 refuses to run). If a default ever breaks, swap the
# hf_path/hf_config to another parquet mirror.
SOURCES: dict[str, SourceSpec] = {
    "wiki":          SourceSpec("wiki",          "wikimedia/wikipedia",         "20231101.en",    "text"),
    "books":         SourceSpec("books",         "emozilla/pg19",               None,             "text"),
    "stackoverflow": SourceSpec("stackoverflow", "mikex86/stackoverflow-posts", None,             "Body"),
    "arxiv":         SourceSpec("arxiv",         "neuralwork/arxiver",          None,             "markdown"),
    "math":          SourceSpec("math",          "open-web-math/open-web-math", None,             "text"),
    # v3 additions — curated Common-Crawl derivative + dedicated math/science.
    # All parquet-native streaming sets (verified to stream a row). peS2o uses
    # the common-pile parquet mirror because the canonical allenai/peS2o is a
    # (now-unsupported by datasets>=3.0) dataset script.
    "fineweb_edu":   SourceSpec("fineweb_edu",   "HuggingFaceFW/fineweb-edu",   "sample-10BT",    "text"),
    "finemath":      SourceSpec("finemath",      "HuggingFaceTB/finemath",      "finemath-4plus", "text"),
    "pes2o":         SourceSpec("pes2o",         "common-pile/peS2o",           None,             "text"),
}

# Stable integer ids for fast tensor tagging.
SOURCE_IDS: dict[str, int] = {name: i for i, name in enumerate(SOURCES)}
ID_TO_SOURCE: dict[int, str] = {i: name for name, i in SOURCE_IDS.items()}


# ---------------------------------------------------------------------------
# Stage mixtures (Llama 3 style)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StageMix:
    name:    str
    weights: dict[str, float]  # source name -> sampling weight (need not sum to 1)


# v3 mixtures: science + math heavy. fineweb_edu is the curated-CC web backbone;
# math = finemath + open-web-math; science papers = arxiv + peS2o.
PRETRAIN_MIX = StageMix(
    name="pretrain",
    weights={
        "fineweb_edu":   0.30,   # curated Common-Crawl (edu/science) backbone
        "finemath":      0.15,   # math  ┐
        "math":          0.15,   #       ┘ = 0.30 math
        "pes2o":         0.13,   # science papers ┐
        "arxiv":         0.12,   #                ┘ = 0.25 science
        "stackoverflow": 0.08,   # code/reasoning
        "wiki":          0.05,
        "books":         0.02,
    },
)

ANNEAL_MIX = StageMix(
    name="anneal",
    weights={
        "finemath":      0.22,   # math  ┐
        "math":          0.18,   #       ┘ = 0.40 math
        "pes2o":         0.18,   # science papers ┐
        "arxiv":         0.12,   #                ┘ = 0.30 science
        "fineweb_edu":   0.15,
        "stackoverflow": 0.10,
        "wiki":          0.03,
        "books":         0.02,
    },
)


# ---------------------------------------------------------------------------
# Packed multi-source dataset
# ---------------------------------------------------------------------------

class MultiSourcePackedDataset(IterableDataset):
    """Pulls from several streaming sources weighted by a StageMix.

    Each iteration yields (input, target, source_id) where:
      input  : [context_length] token tensor
      target : [context_length] tensor, == input shifted left by one
      source_id : scalar tensor, the source that produced the *first* token
                  of this window. (Most windows are dominated by one source
                  because we pack within a per-source buffer.)
    """

    def __init__(
        self,
        sources:        dict[str, SourceSpec],
        mix:            StageMix,
        context_length: int,
        seed:           int = 123,
        skip_docs:      dict[str, int] | None = None,
        shuffle_buffer: int = 5_000,
        tokenizer_repo: str = "Qwen/Qwen3-0.6B-Base",
    ):
        super().__init__()
        self.sources        = sources
        self.mix            = mix
        self.context_length = context_length
        self.seed           = seed
        self.skip_docs      = skip_docs or {}
        self.shuffle_buffer = shuffle_buffer
        self.tokenizer_repo = tokenizer_repo
        self._docs_consumed = {name: self.skip_docs.get(name, 0) for name in sources}

        # Resolve weights to a list aligned with self._source_names.
        self._source_names = [n for n in sources if mix.weights.get(n, 0.0) > 0.0]
        w = [mix.weights[n] for n in self._source_names]
        total = sum(w)
        self._weights = [x / total for x in w]

    @property
    def docs_consumed(self) -> dict[str, int]:
        return dict(self._docs_consumed)

    def _build_stream(self, spec: SourceSpec, skip: int):
        ds = load_dataset(
            spec.hf_path,
            name=spec.hf_config,
            split=spec.split,
            streaming=True,
        )
        if skip:
            ds = ds.skip(skip)
        return ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        tok = get_tokenizer(self.tokenizer_repo)
        EOT = eot_id(self.tokenizer_repo)

        # One iterator + buffer per active source.
        streams = {
            name: iter(self._build_stream(self.sources[name], self._docs_consumed[name]))
            for name in self._source_names
        }
        buffers: dict[str, list[int]] = {n: [] for n in self._source_names}
        window  = self.context_length + 1
        rng     = random.Random(self.seed)

        while True:
            name = rng.choices(self._source_names, weights=self._weights, k=1)[0]
            spec = self.sources[name]
            buf  = buffers[name]

            while len(buf) < window:
                try:
                    row = next(streams[name])
                except StopIteration:
                    streams[name] = iter(self._build_stream(spec, 0))
                    self._docs_consumed[name] = 0
                    continue
                text = row[spec.text_field]
                if not isinstance(text, str) or not text:
                    continue
                buf.extend(tok.encode(text).ids)
                buf.append(EOT)
                self._docs_consumed[name] += 1

            chunk = buf[:window]
            del buf[:window]
            t = torch.tensor(chunk, dtype=torch.long)
            yield t[:-1], t[1:], torch.tensor(SOURCE_IDS[name], dtype=torch.long)


# ---------------------------------------------------------------------------
# Per-source validation
# ---------------------------------------------------------------------------

class SingleSourcePackedDataset(IterableDataset):
    """One source only, no shuffling, fixed take — for reproducible val loss."""

    def __init__(
        self,
        spec:           SourceSpec,
        context_length: int,
        val_docs:       int,
        tokenizer_repo: str = "Qwen/Qwen3-0.6B-Base",
    ):
        super().__init__()
        self.spec           = spec
        self.context_length = context_length
        self.val_docs       = val_docs
        self.tokenizer_repo = tokenizer_repo

    def __iter__(self):
        tok = get_tokenizer(self.tokenizer_repo)
        EOT = eot_id(self.tokenizer_repo)

        ds = load_dataset(
            self.spec.hf_path,
            name=self.spec.hf_config,
            split=self.spec.split,
            streaming=True,
        ).take(self.val_docs)

        window = self.context_length + 1
        buf: list[int] = []
        for row in ds:
            text = row[self.spec.text_field]
            if not isinstance(text, str) or not text:
                continue
            buf.extend(tok.encode(text).ids)
            buf.append(EOT)
            while len(buf) >= window:
                chunk = buf[:window]
                del buf[:window]
                t = torch.tensor(chunk, dtype=torch.long)
                yield t[:-1], t[1:]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_train_loader(
    mix:            StageMix,
    context_length: int,
    batch_size:     int,
    seed:           int = 123,
    skip_docs:      dict[str, int] | None = None,
) -> DataLoader:
    ds = MultiSourcePackedDataset(
        sources=SOURCES, mix=mix, context_length=context_length,
        seed=seed, skip_docs=skip_docs,
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=True)


def build_val_loaders(
    context_length: int,
    batch_size:     int,
    val_docs_per_source: int = 200,
) -> dict[str, DataLoader]:
    loaders = {}
    for name, spec in SOURCES.items():
        ds = SingleSourcePackedDataset(spec, context_length, val_docs_per_source)
        loaders[name] = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=True)
    return loaders
