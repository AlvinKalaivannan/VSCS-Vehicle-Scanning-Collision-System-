"""Ingest raw capture files: checksum, manifest, dev/test split (P1-T4).

This is the one place raw data enters the project, and CLAUDE.md §0 makes raw data
irreplaceable: nothing here ever modifies or deletes a source file. Ingest **copies**, and
verifies the copy by checksum afterwards (R-13) - a copy that was not verified after
copying is not a backup.

It also assigns the **dev/test split**, once, permanently (R-09). The held-out test split
is opened exactly once, at P5-T2. Assigning it here, mechanically, at the moment data
arrives, is what stops the split from being quietly reshuffled later once results start
coming in.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from vscs.common.config import config_dir, load_config, replace_top_level_block, repo_root
from vscs.common.io import sha256_file
from vscs.common.log import get_logger

logger = get_logger("capture.ingest")

RawKind = Literal["scan", "lot", "calib"]
Split = Literal["dev", "test"]

VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v", ".avi", ".mkv")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".dng", ".heic")
IMU_SUFFIXES = (".csv", ".json", ".txt")

#: Categories the split is stratified over, so both splits contain each difficult case
#: (CLAUDE.md §11). Inferred from the filename, overridable on the CLI.
STRATA = ("pole", "curb", "box", "mixed")

_MANIFEST_PLACEHOLDER = "_(no raw data ingested yet)_"


@dataclass
class RawRecord:
    """One row of ``data/MANIFEST.md``."""

    id: str
    date: str
    kind: str
    sha256: str
    size_bytes: int
    description: str
    split: str | None
    stratum: str | None
    source_name: str
    dest_path: str
    backups: list[str] = field(default_factory=list)
    verified: bool = False
    duration_s: float | None = None
    n_frames: int | None = None

    def manifest_row(self) -> str:
        frames = "-" if self.n_frames is None else str(self.n_frames)
        dur = "-" if self.duration_s is None else f"{self.duration_s:.1f} s"
        extent = f"{dur} / {frames}"
        backups = ", ".join(self.backups) if self.backups else "laptop only"
        return (
            f"| `{self.id}` | {self.date} | {self.kind} | `{self.sha256[:16]}...` | {extent} "
            f"| {self.description} | {backups} | {self.split or '-'} "
            f"| {'yes' if self.verified else '**NO**'} |"
        )


# --------------------------------------------------------------------------- #
# Split assignment - R-09                                                      #
# --------------------------------------------------------------------------- #
def infer_stratum(name: str) -> str:
    """Guess the obstacle category from a filename, for stratification.

    Deliberately crude and overridable. It exists so that a batch of sensibly named files
    stratifies correctly without hand-annotation, not to be clever.
    """
    low = name.lower()
    if "pole" in low:
        return "pole"
    if "curb" in low or "kerb" in low:
        return "curb"
    if "box" in low or "cone" in low:
        return "box"
    return "mixed"


def assign_split(raw_id: str, *, stratum: str, ratio_dev: float, seed: int) -> Split:
    """Deterministically assign ``raw_id`` to ``dev`` or ``test``.

    Deterministic by construction: the assignment is a hash of
    ``(seed, stratum, raw_id)``, so re-running ingest on the same file always produces the
    same answer, and no state has to be carried between runs.

    Stratification (CLAUDE.md §11) works because the stratum is mixed into the hash:
    each category is split at roughly ``ratio_dev`` independently, so both splits end up
    containing pole, kerb and box cases rather than the test split happening to be all
    kerbs.

    Note this is per-file and approximate: with only a dozen passes the realised ratio
    will not be exactly 70/30. That is fine. What matters is that it is fixed before
    anyone has seen a result, and cannot be nudged afterwards.
    """
    if not 0.0 < ratio_dev < 1.0:
        raise ValueError(f"ratio_dev must be in (0, 1), got {ratio_dev}")
    digest = hashlib.sha256(f"{seed}|{stratum}|{raw_id}".encode()).digest()
    # First 8 bytes as a big-endian integer, mapped to [0, 1).
    draw = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return "dev" if draw < ratio_dev else "test"


def record_split_in_eval_config(
    raw_id: str,
    split: Split,
    *,
    config_path: Path | None = None,
) -> Path:
    """Add ``raw_id`` to ``splits.dev`` or ``splits.test`` in ``configs/eval.yaml``.

    Refuses to move an id that is already recorded under the other split. Re-splitting
    after the fact is precisely the failure R-09 describes, so it has to be an explicit,
    deliberate edit rather than something ingest can do by accident.
    """
    path = Path(config_path) if config_path is not None else config_dir() / "eval.yaml"
    cfg = load_config(str(path))
    splits = dict(cfg["splits"])
    dev = list(splits.get("dev") or [])
    test = list(splits.get("test") or [])

    other = test if split == "dev" else dev
    if raw_id in other:
        raise ValueError(
            f"{raw_id} is already recorded in the "
            f"{'test' if split == 'dev' else 'dev'} split. Moving a file between splits "
            "invalidates the held-out evaluation (R-09); if this is genuinely required it "
            "needs an ADR and the developer's decision."
        )

    target = dev if split == "dev" else test
    if raw_id not in target:
        target.append(raw_id)
        target.sort()
    splits["dev"] = dev
    splits["test"] = test
    replace_top_level_block(path, "splits", splits)
    return path


# --------------------------------------------------------------------------- #
# Manifest - R-13                                                              #
# --------------------------------------------------------------------------- #
def append_manifest_row(record: RawRecord, *, manifest_path: Path | None = None) -> Path:
    """Append (or update) this file's row in ``data/MANIFEST.md``.

    The manifest is the only committed file under ``data/`` (§4.4) and is the index of
    everything irreplaceable. Re-ingesting the same id replaces its row rather than
    duplicating it.
    """
    path = (
        Path(manifest_path) if manifest_path is not None else repo_root() / "data" / "MANIFEST.md"
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    row = record.manifest_row()
    marker = f"| `{record.id}` |"

    existing = next((i for i, ln in enumerate(lines) if ln.startswith(marker)), None)
    if existing is not None:
        lines[existing] = row
    else:
        placeholder = next((i for i, ln in enumerate(lines) if _MANIFEST_PLACEHOLDER in ln), None)
        if placeholder is not None:
            lines[placeholder] = row
        else:
            # Insert after the last existing table row.
            last = max(
                (i for i, ln in enumerate(lines) if ln.startswith("| ")),
                default=None,
            )
            if last is None:
                raise ValueError(f"{path} has no table to append to")
            lines.insert(last + 1, row)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def verify_copy(expected_sha256: str, dest: Path) -> bool:
    """Re-hash ``dest`` and compare. This is the *after* in "verify after copy" (R-13)."""
    actual = sha256_file(dest)
    ok = actual == expected_sha256
    if not ok:
        logger.error("checksum mismatch for %s: expected %s got %s", dest, expected_sha256, actual)
    return ok


# --------------------------------------------------------------------------- #
# Ingest                                                                       #
# --------------------------------------------------------------------------- #
def _probe_video(path: Path) -> tuple[float | None, int | None]:
    """Duration and frame count, or ``(None, None)`` if it is not a readable video."""
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        return None, None
    try:
        from vscs.capture.frames import analyse_timing, probe_timestamps

        stamps = probe_timestamps(path)
        report = analyse_timing(stamps.t_ns)
        return report.duration_s, report.n_frames
    except Exception as exc:
        logger.warning("could not probe %s: %s", path.name, exc)
        return None, None


def make_raw_id(kind: str, src: Path, *, on: date | None = None, index: int | None = None) -> str:
    """Stable identifier, e.g. ``lot_20261011_pass07`` or ``scan_20261003_loop1``.

    Built from the kind, the date and the source stem, so the id says what the file is
    without opening the manifest.
    """
    stamp = (on or date.today()).strftime("%Y%m%d")
    stem = "".join(c if c.isalnum() else "_" for c in src.stem).strip("_").lower()
    suffix = f"_{index:02d}" if index is not None else ""
    return f"{kind}_{stamp}_{stem}{suffix}"


def ingest_file(
    src: Path,
    *,
    kind: RawKind,
    description: str = "",
    stratum: str | None = None,
    dest_root: Path | None = None,
    eval_config_path: Path | None = None,
    manifest_path: Path | None = None,
    assign_splits: bool = True,
    on: date | None = None,
) -> RawRecord:
    """Copy one raw file into ``data/raw/``, checksum it, split it, and record it.

    The source is never modified or removed. The copy is verified by re-hashing it, and
    ``verified`` in the manifest stays ``NO`` unless that check passed.

    Only ``lot`` files get a dev/test split: scans and calibration images are not
    evaluation passes.
    """
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(src)
    root = Path(dest_root) if dest_root is not None else repo_root() / "data" / "raw"

    raw_id = make_raw_id(kind, src, on=on)
    dest_dir = root / raw_id
    dest = dest_dir / src.name
    if dest.exists():
        raise FileExistsError(
            f"{dest} already exists. Raw data is never overwritten (CLAUDE.md §0); "
            "remove it deliberately or ingest under a different id."
        )

    logger.info("hashing %s (%.1f MB)", src.name, src.stat().st_size / 1e6)
    source_sha = sha256_file(src)

    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    verified = verify_copy(source_sha, dest)
    if not verified:
        raise OSError(
            f"the copy of {src.name} does not match the source checksum. The destination "
            "may be faulty - do not delete the source."
        )

    duration_s, n_frames = _probe_video(dest)

    strat = stratum or infer_stratum(src.name)
    if strat not in STRATA:
        raise ValueError(f"unknown stratum {strat!r}; expected one of {STRATA}")

    split: Split | None = None
    if kind == "lot" and assign_splits:
        eval_cfg = load_config(str(eval_config_path) if eval_config_path else "eval")
        split = assign_split(
            raw_id,
            stratum=strat,
            ratio_dev=float(eval_cfg["splits"]["ratio_dev"]),
            seed=int(eval_cfg["splits"]["seed"]),
        )
        record_split_in_eval_config(raw_id, split, config_path=eval_config_path)

    record = RawRecord(
        id=raw_id,
        date=(on or date.today()).isoformat(),
        kind=kind,
        sha256=source_sha,
        size_bytes=src.stat().st_size,
        description=description or src.name,
        split=split,
        stratum=strat,
        source_name=src.name,
        dest_path=str(dest.relative_to(repo_root())) if _under_repo(dest) else str(dest),
        backups=["laptop"],
        verified=verified,
        duration_s=duration_s,
        n_frames=n_frames,
    )

    append_manifest_row(record, manifest_path=manifest_path)
    _write_sidecar(dest_dir, record)
    logger.info("ingested %s (%s%s)", raw_id, strat, f", split={split}" if split else ", no split")
    return record


def _under_repo(path: Path) -> bool:
    try:
        Path(path).resolve().relative_to(repo_root())
    except ValueError:
        return False
    return True


def _write_sidecar(dest_dir: Path, record: RawRecord) -> Path:
    """Write ``ingest.json`` beside the raw file, so the folder is self-describing."""
    from vscs.common.io import write_json

    payload: dict[str, Any] = asdict(record)
    payload["ingested_utc"] = datetime.now().astimezone().isoformat()
    return write_json(dest_dir / "ingest.json", payload)


def ingest_folder(
    src_dir: Path,
    *,
    kind: RawKind,
    description: str = "",
    **kwargs: Any,
) -> list[RawRecord]:
    """Ingest every video, image and IMU log directly under ``src_dir``."""
    src_dir = Path(src_dir)
    if not src_dir.is_dir():
        raise NotADirectoryError(src_dir)
    wanted = VIDEO_SUFFIXES + IMAGE_SUFFIXES + IMU_SUFFIXES
    files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in wanted)
    if not files:
        raise ValueError(f"no ingestable files in {src_dir} (looked for {wanted})")
    return [ingest_file(p, kind=kind, description=description, **kwargs) for p in files]
