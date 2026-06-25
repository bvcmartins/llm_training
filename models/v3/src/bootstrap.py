"""Shared setup for the stage entrypoints (pretrain.py / anneal.py).

Keeps device/seed/model/wandb boilerplate in one place so the two stage
scripts stay short and differ only in their stage-specific config and handoff
logic.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("v2.boot")

MODEL_CHOICES = ("0.6b", "1.5b", "1.7b")


def setup_torch(seed: int):
    """Set CUDA alloc conf, seeds, matmul precision; return the device."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    log.debug("torch set up: device=%s seed=%d", device, seed)
    return device


def build_model(model_name: str, device):
    """Instantiate the requested Qwen3 config on `device`; return (model, cfg)."""
    from qwen3_model import (
        Qwen3Model, QWEN3_CONFIG_0_6B, QWEN3_CONFIG_1_5B, QWEN3_CONFIG_1_7B,
    )

    cfgs = {"0.6b": QWEN3_CONFIG_0_6B, "1.5b": QWEN3_CONFIG_1_5B, "1.7b": QWEN3_CONFIG_1_7B}
    model_cfg = cfgs[model_name]
    log.info("building model %s on %s (dtype=%s)", model_name, device, model_cfg["dtype"])
    model = Qwen3Model(model_cfg).to(device=device, dtype=model_cfg["dtype"])
    log.info("model built: %s params (%s non-embedding)",
             f"{model.num_parameters():,}", f"{model.num_parameters(non_embedding=True):,}")
    return model, model_cfg


def init_wandb(enabled: bool, project: str, name: str, model_cfg: dict, stage: str, stage_cfg, extra: dict | None = None):
    """Start a wandb run (or return None when disabled)."""
    if not enabled:
        log.info("wandb disabled (--no-wandb)")
        return None
    import wandb

    run = wandb.init(
        project=project,
        name=name,
        config={
            "model_config": model_cfg,
            "stage":        stage,
            "mix":          stage_cfg.mix.weights,
            **(extra or {}),
            **{k: v for k, v in stage_cfg.__dict__.items() if k != "mix"},
        },
    )
    log.info("wandb run: project=%s name=%s id=%s", project, name, run.id)
    return run
