"""Verbose, dual-sink logging for v2 training/annealing scripts.

Everything routes through the stdlib `logging` module so the engine
(`training.py`) and the stage entrypoints (`pretrain.py`, `anneal.py`) all
emit to the same place. `setup_logging(...)` wires two handlers:

  * console (stdout) — what you watch live
  * a timestamped file under `<v2>/logs/` — the durable record

Both get full timestamps + level + logger name. Call it once, early, from each
entrypoint; the engine just does `logging.getLogger("v2.train")` and inherits
the handlers.
"""

from __future__ import annotations

import logging
import platform
import sys
from datetime import datetime
from pathlib import Path

# Single namespace root; submodules use "v2.<x>" so one config covers all.
ROOT_LOGGER = "v2"

_FMT = "%(asctime)s | %(levelname)-7s | %(name)-12s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    stage: str,
    log_dir: Path,
    level: int = logging.DEBUG,
    console_level: int = logging.INFO,
) -> tuple[logging.Logger, Path]:
    """Configure console + file logging and return (logger, log_file_path).

    `level` is the floor for the file (keep it DEBUG to capture everything);
    `console_level` is the floor for stdout (INFO keeps the live view readable
    while the file still gets DEBUG).
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{stage}_{ts}.log"

    logger = logging.getLogger(ROOT_LOGGER)
    logger.setLevel(min(level, console_level))
    logger.handlers.clear()  # idempotent across re-runs in the same process
    logger.propagate = False

    fmt = logging.Formatter(_FMT, datefmt=_DATEFMT)

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info("logging initialized — stage=%s file=%s", stage, log_file)
    return logger, log_file


def log_system_info(logger: logging.Logger, device) -> None:
    """Dump the host/torch/GPU environment — invaluable when reading old logs."""
    import torch

    logger.info("=== system ===")
    logger.info("python      : %s", sys.version.split()[0])
    logger.info("platform    : %s", platform.platform())
    logger.info("torch       : %s", torch.__version__)
    logger.info("cuda avail  : %s", torch.cuda.is_available())
    logger.info("device      : %s", device)
    if getattr(device, "type", None) == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        logger.info("gpu         : %s", props.name)
        logger.info("gpu memory  : %.1f GB total", props.total_memory / 1e9)
        logger.info("gpu cc      : %d.%d", props.major, props.minor)
        logger.info("cudnn       : %s", torch.backends.cudnn.version())


def gpu_mem_str(device) -> str:
    """Compact 'alloc/reserved/peak GB' string for periodic memory logging."""
    import torch

    if getattr(device, "type", None) != "cuda":
        return "cpu (no gpu mem)"
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    p = torch.cuda.max_memory_allocated() / 1e9
    return f"alloc={a:.2f}GB reserved={r:.2f}GB peak={p:.2f}GB"


def log_config(logger: logging.Logger, title: str, cfg) -> None:
    """Pretty-print a dataclass/dict config field-by-field at INFO."""
    logger.info("=== %s ===", title)
    items = cfg.items() if isinstance(cfg, dict) else vars(cfg).items()
    for k, v in items:
        logger.info("  %-20s %s", k, v)
