"""Run folders, JSON Lines streams, checksums, and metric appends.

CLAUDE.md section 4.3: every script writes its outputs to a *new* run folder

    data/processed/<stage>/<YYYYMMDD-HHMM>_<shortsha>/

containing the resolved config, the git commit hash, and a ``run.log``. Nothing is
ever written back over a previous run, so any number in ``metrics/results.jsonl`` can
be traced to the code and config that produced it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vscs.common.config import dump_config, repo_root

#: Stages that get their own subtree under ``data/processed/``.
STAGES = ("capture", "recon", "seg", "model", "perception", "risk", "eval", "stream")

_RUN_STAMP_FMT = "%Y%m%d-%H%M"


# --------------------------------------------------------------------------- #
# Git provenance                                                               #
# --------------------------------------------------------------------------- #
def git_short_sha(*, default: str = "nogit") -> str:
    """Short commit SHA of the working tree, or ``default`` outside a git repo.

    A ``-dirty`` suffix is appended when there are uncommitted changes, because a run
    from a dirty tree is not reproducible from its SHA and the run folder name should
    say so.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return default
    if not sha:
        return default
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        dirty = ""
    return f"{sha}-dirty" if dirty else sha


def git_commit_info() -> dict[str, str]:
    """Commit SHA, branch and dirty flag, for the run folder's provenance file."""

    def _run(args: list[str]) -> str:
        try:
            return subprocess.run(
                args, cwd=repo_root(), capture_output=True, text=True, check=True, timeout=10
            ).stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return ""

    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "short": _run(["git", "rev-parse", "--short", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": "true" if _run(["git", "status", "--porcelain"]) else "false",
    }


# --------------------------------------------------------------------------- #
# Run folders                                                                  #
# --------------------------------------------------------------------------- #
def make_run_dir(
    stage: str,
    *,
    config: dict[str, Any] | None = None,
    root: Path | None = None,
    now: datetime | None = None,
) -> Path:
    """Create and return a new run folder for ``stage``.

    Writes ``resolved_config.yaml`` (when a config is given) and ``provenance.json``.
    If a folder for this minute and SHA already exists, a ``-2``, ``-3`` ... suffix is
    added rather than reusing it: two runs in the same minute must not overwrite each
    other's outputs.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    base = (root or repo_root() / "data" / "processed") / stage
    stamp = (now or datetime.now()).strftime(_RUN_STAMP_FMT)
    name = f"{stamp}_{git_short_sha()}"

    run_dir = base / name
    suffix = 2
    while run_dir.exists():
        run_dir = base / f"{name}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)

    provenance = {
        "stage": stage,
        "created_utc": datetime.now(UTC).isoformat(),
        "git": git_commit_info(),
        "run_dir": str(run_dir.relative_to(repo_root())) if _under_repo(run_dir) else str(run_dir),
    }
    write_json(run_dir / "provenance.json", provenance)
    if config is not None:
        dump_config(config, run_dir / "resolved_config.yaml")
    return run_dir


def _under_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(repo_root())
    except ValueError:
        return False
    return True


# --------------------------------------------------------------------------- #
# JSON / JSON Lines                                                            #
# --------------------------------------------------------------------------- #
def write_json(path: Path, obj: Any) -> Path:
    """Write pretty-printed JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    return path


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_jsonl(path: Path, records: Iterable[Any]) -> Path:
    """Write an iterable of records as JSON Lines.

    Pydantic models are serialized via ``model_dump``; anything else must be
    JSON-serializable. ``RiskFrame`` streams use this (CLAUDE.md section 4.2).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(_to_jsonable(rec), default=str) + "\n")
    return path


def read_jsonl(path: Path, *, skip_blank: bool = True) -> Iterator[dict[str, Any]]:
    """Yield records from a JSON Lines file, reporting the line number on a bad line."""
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                if skip_blank:
                    continue
                raise ValueError(f"{path}:{lineno}: blank line")
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc


def append_jsonl(path: Path, record: Any) -> Path:
    """Append one record to a JSON Lines file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_to_jsonable(record), default=str) + "\n")
    return path


def _to_jsonable(rec: Any) -> Any:
    """Convert pydantic models (and plain objects) to JSON-ready dicts."""
    dump = getattr(rec, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return rec


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def metrics_path() -> Path:
    """Path of the append-only metrics file."""
    return repo_root() / "metrics" / "results.jsonl"


def append_metric(
    *,
    task: str,
    metric: str,
    value: float,
    split: str,
    run_dir: str | Path,
    notes: str = "",
    path: Path | None = None,
) -> dict[str, Any]:
    """Append one metric line to ``metrics/results.jsonl`` (CLAUDE.md section 7.4).

    Append-only. Never hand-edit the file and never delete a line: to correct a
    number, append a new line with the correction explained in ``notes``.

    Only evaluation code should call this, and only with a value computed in the same
    run - a number that was not measured must never end up here (section 0.4).
    """
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "git": git_commit_info()["short"] or "nogit",
        "task": task,
        "metric": metric,
        "value": float(value),
        "split": split,
        "run_dir": str(run_dir),
        "notes": notes,
    }
    append_jsonl(path or metrics_path(), record)
    return record


# --------------------------------------------------------------------------- #
# Checksums                                                                    #
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed so multi-gigabyte recordings do not load into RAM.

    Used by ingest for ``data/MANIFEST.md`` and to verify each of the three backup
    copies *after* copying (risk R-13).
    """
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()
