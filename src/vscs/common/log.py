"""Logging setup.

Every run folder gets a ``run.log`` (CLAUDE.md section 4.3), so that a result can be
read back later together with the warnings that were printed while producing it.
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False


def setup_logging(
    *,
    run_dir: Path | None = None,
    level: int = logging.INFO,
    force: bool = False,
) -> logging.Logger:
    """Configure the ``vscs`` logger to write to stderr and, optionally, ``run.log``.

    Idempotent: calling it twice does not duplicate handlers, which would otherwise
    print every line twice once a script imports two modules that both set up logging.
    """
    global _CONFIGURED
    logger = logging.getLogger("vscs")
    if _CONFIGURED and not force:
        if run_dir is not None and not _has_file_handler(logger, run_dir / "run.log"):
            logger.addHandler(_file_handler(run_dir / "run.log", level))
        return logger

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    logger.setLevel(level)
    logger.propagate = False

    stream = logging.StreamHandler(sys.stderr)
    stream.setLevel(level)
    stream.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(stream)

    if run_dir is not None:
        logger.addHandler(_file_handler(run_dir / "run.log", level))

    _CONFIGURED = True
    return logger


def _file_handler(path: Path, level: int) -> logging.FileHandler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    return handler


def _has_file_handler(logger: logging.Logger, path: Path) -> bool:
    target = str(Path(path).resolve())
    return any(
        isinstance(h, logging.FileHandler) and str(Path(h.baseFilename).resolve()) == target
        for h in logger.handlers
    )


def get_logger(name: str) -> logging.Logger:
    """Return a child of the ``vscs`` logger, e.g. ``get_logger("recon.scale")``."""
    return logging.getLogger(f"vscs.{name}")


def seed_everything(seed: int, *, logger: logging.Logger | None = None) -> int:
    """Seed Python and numpy RNGs and log the seed.

    CLAUDE.md section 4.3 requires every random process to be deterministic and the
    seed to be logged - otherwise a RANSAC result cannot be reproduced, and a metric
    that moved cannot be attributed to a code change rather than to luck.

    Torch is seeded only if it is already imported, so the laptop environment (which
    has no torch) does not import it as a side effect.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass

    torch = sys.modules.get("torch")
    if torch is not None:  # pragma: no cover - torch only exists on Colab
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    (logger or get_logger("seed")).info("seed=%d", seed)
    return seed
