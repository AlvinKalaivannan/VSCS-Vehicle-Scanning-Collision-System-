"""Loading and resolving the YAML configs in ``configs/``.

CLAUDE.md section 4.3: every tunable number lives in ``configs/*.yaml``, never as a
constant in code. This module is how code reaches them, and how the *resolved* config
gets written into a run folder so a result can always be traced back to the exact
numbers that produced it.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class _Missing:
    """Sentinel so that ``None`` remains a usable default value."""

    def __repr__(self) -> str:
        return "<missing>"


_MISSING = _Missing()

#: Config file stems that exist under ``configs/`` (CLAUDE.md section 3).
KNOWN_CONFIGS = (
    "capture",
    "recon",
    "seg",
    "model",
    "severity",
    "perception",
    "risk",
    "eval",
)


def repo_root() -> Path:
    """Absolute path of the repository root.

    Resolved from this file's location (``src/vscs/common/config.py``) rather than
    from the current working directory, so scripts behave the same whether they are
    run from the repo root, from ``scripts/``, or from a Colab notebook.
    """
    return Path(__file__).resolve().parents[3]


def config_dir() -> Path:
    """Absolute path of the ``configs/`` directory."""
    return repo_root() / "configs"


def load_config(name: str, *, config_dir_override: Path | None = None) -> dict[str, Any]:
    """Load one config by stem, e.g. ``load_config("risk")``.

    Accepts a bare stem, a filename, or a full path. Raises rather than returning a
    default: a missing config means the caller would silently run on hardcoded
    numbers, which is exactly what section 4.3 forbids.
    """
    base = config_dir_override or config_dir()
    candidate = Path(name)
    if candidate.suffix in (".yaml", ".yml"):
        path = candidate if candidate.is_absolute() else base / candidate
    else:
        path = base / f"{name}.yaml"

    if not path.is_file():
        available = sorted(p.stem for p in base.glob("*.yaml")) if base.is_dir() else []
        raise FileNotFoundError(f"no config at {path}. Available in {base}: {available}")

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        raise ValueError(f"config {path} is empty")
    if not isinstance(data, dict):
        raise TypeError(f"config {path} must be a mapping at the top level, got {type(data)}")
    return data


def load_all_configs(*, config_dir_override: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load every config in :data:`KNOWN_CONFIGS`, keyed by stem."""
    return {n: load_config(n, config_dir_override=config_dir_override) for n in KNOWN_CONFIGS}


def get(cfg: dict[str, Any], dotted_key: str, default: Any = _MISSING) -> Any:
    """Read a nested value by dotted path: ``get(cfg, "alerts.hysteresis.min_dwell_s")``.

    Without a ``default``, a missing key raises. That is deliberate: a typo in a
    config key should stop the run, not quietly substitute zero.
    """
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is _MISSING:
                raise KeyError(f"config key {dotted_key!r} not found (failed at {part!r})")
            return default
        node = node[part]
    return node


def merge_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of ``cfg`` with dotted-key ``overrides`` applied.

    Used for CLI ``--set key=value`` style flags. Only overrides keys that already
    exist, so a typo cannot introduce a silently unused setting.
    """
    out = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        node: Any = out
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"cannot override {dotted!r}: {part!r} does not exist")
            node = node[part]
        if not isinstance(node, dict) or parts[-1] not in node:
            raise KeyError(f"cannot override {dotted!r}: key does not exist")
        node[parts[-1]] = value
    return out


def dump_config(cfg: dict[str, Any], path: Path) -> Path:
    """Write a resolved config to ``path``. Every run folder gets one (section 4.3)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False, allow_unicode=True)
    return path
