"""Logging and seeding tests.

Section 4.3 requires a ``run.log`` per run folder, and every random process seeded
with the seed logged. Both are easy to get subtly wrong: duplicated handlers print
everything twice, and a seed that is set but not logged cannot be used to reproduce a
result later.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from vscs.common import log as L


@pytest.fixture(autouse=True)
def _reset_logging():
    """Each test gets a clean ``vscs`` logger, and leaves one behind."""
    L._CONFIGURED = False
    logger = logging.getLogger("vscs")
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    yield
    L._CONFIGURED = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()


def test_setup_logging_adds_a_stream_handler():
    logger = L.setup_logging()
    assert logger.name == "vscs"
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    assert logger.level == logging.INFO


def test_setup_logging_writes_a_run_log(tmp_path):
    logger = L.setup_logging(run_dir=tmp_path)
    logger.info("hello from the test")
    for h in logger.handlers:
        h.flush()
    run_log = tmp_path / "run.log"
    assert run_log.is_file()
    assert "hello from the test" in run_log.read_text(encoding="utf-8")


def test_setup_logging_is_idempotent():
    """Calling it twice must not duplicate handlers, or every line prints twice."""
    first = L.setup_logging()
    n = len(first.handlers)
    second = L.setup_logging()
    assert second is first
    assert len(second.handlers) == n


def test_second_call_with_a_run_dir_adds_the_file_handler_once(tmp_path):
    L.setup_logging()
    logger = L.setup_logging(run_dir=tmp_path)
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
    # And a third call must not add another.
    logger = L.setup_logging(run_dir=tmp_path)
    assert len([h for h in logger.handlers if isinstance(h, logging.FileHandler)]) == 1


def test_force_rebuilds_handlers(tmp_path):
    L.setup_logging(run_dir=tmp_path)
    logger = L.setup_logging(force=True)
    assert not [h for h in logger.handlers if isinstance(h, logging.FileHandler)]


def test_logger_does_not_propagate_to_root():
    """Prevents double output when something else configures the root logger."""
    assert L.setup_logging().propagate is False


def test_get_logger_is_a_child_of_vscs():
    child = L.get_logger("recon.scale")
    assert child.name == "vscs.recon.scale"
    assert child.parent is not None


# --------------------------------------------------------------------------- #
# Seeding                                                                      #
# --------------------------------------------------------------------------- #
def test_seed_everything_makes_numpy_deterministic():
    L.seed_everything(20260924)
    a = np.random.rand(5)
    L.seed_everything(20260924)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_seed_everything_seeds_python_random():
    import random

    L.seed_everything(7)
    a = [random.random() for _ in range(5)]
    L.seed_everything(7)
    assert [random.random() for _ in range(5)] == a


def test_seed_everything_returns_and_logs_the_seed(tmp_path):
    """Section 4.3: the seed has to be *logged*, not just set."""
    L.setup_logging(run_dir=tmp_path)
    assert L.seed_everything(12345) == 12345
    for h in logging.getLogger("vscs").handlers:
        h.flush()
    assert "seed=12345" in (tmp_path / "run.log").read_text(encoding="utf-8")


def test_different_seeds_give_different_draws():
    L.seed_everything(1)
    a = np.random.rand(5)
    L.seed_everything(2)
    assert not np.array_equal(a, np.random.rand(5))
