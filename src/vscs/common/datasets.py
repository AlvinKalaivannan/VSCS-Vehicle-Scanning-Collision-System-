"""External (third-party) datasets: registry and use guard. ADR 0006.

``configs/datasets.yaml`` lists every external dataset considered, with its licence and
whether that licence was stated consistently. What a dataset may be *used for* is derived
here from its licence class, never written by hand, so an entry cannot grant itself
publication rights by mistake.

Anything that reads external data calls :func:`require_use` first, e.g.::

    root = require_use("roboflow_car_parts_19", "private_eval")

which returns ``data/external/<id>/`` or raises :class:`DatasetUseError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

from vscs.common.config import load_config, repo_root

Use = Literal["rehearsal", "private_eval", "publish_results", "train", "publish_media"]
USES: tuple[str, ...] = ("rehearsal", "private_eval", "publish_results", "train", "publish_media")
LICENSE_STATUSES: tuple[str, ...] = ("clear", "conflicting", "unverified", "none")
PROVENANCES: tuple[str, ...] = ("documented", "undocumented")

EXTERNAL_DATA_DIR = Path("data") / "external"


class DatasetUseError(PermissionError):
    """A dataset was about to be used in a way its licence does not allow."""


@dataclass(frozen=True)
class DatasetEntry:
    id: str
    name: str
    url: str
    license: str | None
    license_url: str | None
    license_status: str
    provenance: str
    verified: date | None
    adopted: bool
    covers: tuple[str, ...]
    notes: str
    source: dict[str, Any]


def _parse_entry(dataset_id: str, raw: dict[str, Any]) -> DatasetEntry:
    status = raw.get("license_status")
    if status not in LICENSE_STATUSES:
        raise ValueError(f"{dataset_id}: license_status must be one of {LICENSE_STATUSES}")
    provenance = raw.get("provenance", "documented")
    if provenance not in PROVENANCES:
        raise ValueError(f"{dataset_id}: provenance must be one of {PROVENANCES}")
    verified = raw.get("verified")
    if verified is not None and not isinstance(verified, date):
        raise ValueError(f"{dataset_id}: verified must be a YYYY-MM-DD date or null")
    return DatasetEntry(
        id=dataset_id,
        name=str(raw["name"]),
        url=str(raw["url"]),
        license=raw.get("license"),
        license_url=raw.get("license_url"),
        license_status=status,
        provenance=provenance,
        verified=verified,
        adopted=bool(raw.get("adopted", False)),
        covers=tuple(raw.get("covers") or ()),
        notes=str(raw.get("notes") or ""),
        source=dict(raw.get("source") or {}),
    )


def load_registry(cfg: dict[str, Any] | None = None) -> dict[str, DatasetEntry]:
    cfg = cfg if cfg is not None else load_config("datasets")
    return {k: _parse_entry(k, v) for k, v in (cfg.get("datasets") or {}).items()}


def license_class(entry: DatasetEntry, policy: dict[str, Any]) -> str:
    """``permissive`` / ``noncommercial`` / ``conflicting`` / ``unverified`` / ``none``.

    Only a *clear* licence is classified by its text. A licence that is contradicted
    elsewhere, or not yet checked, is never promoted by what it claims to be.
    """
    if entry.license_status != "clear":
        return entry.license_status
    if entry.license in policy["permissive_licenses"]:
        return "permissive"
    if entry.license in policy["noncommercial_licenses"]:
        return "noncommercial"
    # A clear but unrecognised licence needs a human to classify it (§0, §10).
    return "unverified"


def allowed_uses(entry: DatasetEntry, policy: dict[str, Any]) -> frozenset[str]:
    """Uses granted by the licence class, minus what undocumented image provenance removes."""
    uses = set(policy["uses_by_class"].get(license_class(entry, policy), []))
    if entry.provenance == "undocumented":
        uses -= set(policy.get("undocumented_provenance_removes", []))
    return frozenset(uses)


def require_use(dataset_id: str, use: str, *, cfg: dict[str, Any] | None = None) -> Path:
    """Return ``data/external/<id>/`` if ``use`` is allowed, else raise ``DatasetUseError``."""
    if use not in USES:
        raise ValueError(f"unknown use {use!r}; expected one of {USES}")
    cfg = cfg if cfg is not None else load_config("datasets")
    registry = load_registry(cfg)
    if dataset_id not in registry:
        raise DatasetUseError(
            f"{dataset_id!r} is not in configs/datasets.yaml; register it (with its licence) "
            "before using it"
        )
    entry = registry[dataset_id]
    allowed = allowed_uses(entry, cfg["policy"])
    if use not in allowed:
        raise DatasetUseError(
            f"{dataset_id}: '{use}' is not allowed (licence {entry.license or 'none'}, "
            f"status {entry.license_status}; allowed: {sorted(allowed) or 'nothing'}). "
            "Resolve the licence first - ADR 0006."
        )
    return repo_root() / EXTERNAL_DATA_DIR / dataset_id
