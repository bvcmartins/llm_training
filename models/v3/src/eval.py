"""Per-domain validation + plateau detection.

`evaluate_per_domain` runs each val loader once and returns a dict of
{source_name: loss}. Aggregate loss is the simple mean across sources (we don't
weight by stage mix — we want to know if any single domain is regressing).

`PlateauDetector` watches the aggregate loss and fires when no improvement has
been seen for `patience` consecutive evaluations. Hysteresis via `min_delta`
prevents firing on noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def evaluate_per_domain(
    model:        nn.Module,
    val_loaders:  dict,
    device:       torch.device,
    max_batches:  int = 20,
) -> dict[str, float]:
    """Return {source_name: mean_loss}. Also returns 'aggregate' key."""
    model.eval()
    losses: dict[str, float] = {}
    for name, loader in val_loaders.items():
        total, count = 0.0, 0
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
            total += loss.item()
            count += 1
        losses[name] = total / max(count, 1)
    model.train()
    if losses:
        losses["aggregate"] = sum(v for k, v in losses.items() if k != "aggregate") / len(losses)
    return losses


@dataclass
class PlateauDetector:
    """Fire when aggregate val loss hasn't improved for `patience` evals.

    `min_delta` is the smallest improvement we consider real. `cooldown` keeps
    us from firing repeatedly on the same plateau — after a fire, we wait that
    many evals before becoming armed again.
    """
    patience:  int                       = 5
    min_delta: float                     = 1e-3
    cooldown:  int                       = 5
    on_fire:   Callable[[dict], None] | None = None

    _best:      float = field(default=float("inf"), init=False)
    _bad_evals: int   = field(default=0, init=False)
    _cooldown:  int   = field(default=0, init=False)
    _history:   list  = field(default_factory=list, init=False)

    def update(self, eval_metrics: dict) -> bool:
        """Call after each evaluation. Returns True iff a plateau just fired."""
        agg = eval_metrics.get("aggregate")
        if agg is None:
            return False

        self._history.append(eval_metrics)

        if self._cooldown > 0:
            self._cooldown -= 1
            if agg + self.min_delta < self._best:
                self._best = agg
            return False

        if agg + self.min_delta < self._best:
            self._best = agg
            self._bad_evals = 0
            return False

        self._bad_evals += 1
        if self._bad_evals >= self.patience:
            self._bad_evals = 0
            self._cooldown = self.cooldown
            payload = {
                "best":        self._best,
                "current":     agg,
                "patience":    self.patience,
                "per_domain":  {k: v for k, v in eval_metrics.items() if k != "aggregate"},
            }
            if self.on_fire is not None:
                self.on_fire(payload)
            return True
        return False

    @property
    def best(self) -> float:
        return self._best
