"""Download registered external datasets into ``data/external/<id>/`` (ADR 0006).

Only datasets in ``configs/datasets.yaml`` can be fetched, and only if their licence
allows at least private rehearsal: the guard in ``datasets.require_use`` runs first.
Each download is checksummed and gets a ``PROVENANCE.json`` beside it, recording
the licence, the source, the date, and what the data may be used for.

Secrets: the Roboflow key is read from ``.env`` (gitignored), passed only in the
request URL, and scrubbed from every error message and log line.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vscs.common.config import load_config, repo_root
from vscs.common.datasets import allowed_uses, load_registry, require_use
from vscs.common.io import sha256_file
from vscs.common.log import get_logger

logger = get_logger("common.fetch")

ROBOFLOW_API = "https://api.roboflow.com"

GetJson = Callable[[str], dict[str, Any]]
Download = Callable[[str, Path], None]


def read_env_value(name: str, env_path: Path | None = None) -> str:
    """Read ``NAME=value`` from ``.env``. Raises if missing or empty; never logs the value."""
    path = env_path or repo_root() / ".env"
    if not path.is_file():
        raise RuntimeError(f"{path} not found; add {name}=... to it (it is gitignored)")
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(f"{name}="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            if value:
                return value
    raise RuntimeError(f"{name} is missing or empty in {path}")


def _scrub(text: str, secret: str) -> str:
    return text.replace(secret, "***") if secret else text


def _http_get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def _http_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=600) as r, tmp.open("wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    tmp.replace(dest)


def fetch_roboflow(
    dataset_id: str,
    export_format: str,
    *,
    api_key: str | None = None,
    get_json: GetJson = _http_get_json,
    download: Download = _http_download,
    poll_s: float = 5.0,
    max_polls: int = 60,
    cfg: dict[str, Any] | None = None,
    dest_root: Path | None = None,
) -> Path:
    """Export and download a Roboflow Universe dataset version. Returns the extracted dir.

    ``dest_root`` replaces ``data/external`` (tests only).
    """
    cfg = cfg if cfg is not None else load_config("datasets")
    root = require_use(dataset_id, "rehearsal", cfg=cfg)
    if dest_root is not None:
        root = Path(dest_root) / dataset_id
    entry = load_registry(cfg)[dataset_id]
    src = entry.source
    for k in ("workspace", "project", "version"):
        if k not in src:
            raise ValueError(f"{dataset_id}: registry 'source' needs {k!r} for a Roboflow fetch")
    key = api_key or read_env_value("ROBOFLOW_API_KEY")

    endpoint = (
        f"{ROBOFLOW_API}/{src['workspace']}/{src['project']}/{src['version']}/{export_format}"
    )
    link = None
    for _ in range(max_polls):
        try:
            resp = get_json(f"{endpoint}?api_key={key}")
        except urllib.error.HTTPError as exc:
            body = exc.read()[:300].decode("utf-8", "replace")
            raise RuntimeError(_scrub(f"Roboflow {exc.code}: {body}", key)) from None
        link = (resp.get("export") or {}).get("link")
        if link:
            break
        logger.info("%s: export still generating, waiting", dataset_id)
        time.sleep(poll_s)
    if not link:
        raise RuntimeError(f"{dataset_id}: export never became ready")

    version_tag = f"v{src['version']}_{export_format}"
    zip_path = root / f"{version_tag}.zip"
    logger.info("%s: downloading %s", dataset_id, version_tag)
    download(link, zip_path)
    digest = sha256_file(zip_path)

    out_dir = root / version_tag
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            target = (out_dir / name).resolve()
            if not target.is_relative_to(out_dir.resolve()):
                raise RuntimeError(f"{dataset_id}: unsafe path in archive: {name}")
        zf.extractall(out_dir)

    provenance = {
        "dataset": dataset_id,
        "name": entry.name,
        "url": entry.url,
        "license": entry.license,
        "license_status": entry.license_status,
        "provenance": entry.provenance,
        "allowed_uses": sorted(allowed_uses(entry, cfg["policy"])),
        "source": src,
        "export_format": export_format,
        "archive": zip_path.name,
        "sha256": digest,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (root / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    logger.info("%s: %s sha256 %s -> %s", dataset_id, zip_path.name, digest[:16], out_dir)
    return out_dir
