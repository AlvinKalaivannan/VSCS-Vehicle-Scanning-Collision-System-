"""Dataset fetcher (ADR 0006). No network: the HTTP calls are injected fakes."""

from __future__ import annotations

import copy
import io
import json
import urllib.error
import zipfile
from datetime import date

import pytest

from vscs.common import fetch as FT
from vscs.common.config import load_config
from vscs.common.datasets import DatasetUseError

SECRET = "sk_test_do_not_leak_123"


def _cfg(**over):
    cfg = copy.deepcopy(load_config("datasets"))
    entry = {
        "name": "x",
        "url": "https://example.org",
        "license": "CC-BY-4.0",
        "license_url": "https://example.org/l",
        "license_status": "clear",
        "provenance": "undocumented",
        "verified": date(2026, 9, 29),
        "adopted": True,
        "covers": ["part_vocabulary"],
        "source": {"workspace": "ws", "project": "proj", "version": 2},
    }
    entry.update(over)
    cfg["datasets"] = {"x": entry}
    return cfg


def _zip_bytes(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text)
    return buf.getvalue()


def _fakes(archive: bytes, pending: int = 0):
    seen_urls: list[str] = []
    state = {"n": 0}

    def get_json(url):
        seen_urls.append(url)
        state["n"] += 1
        if state["n"] <= pending:
            return {"progress": 0.5}
        return {"export": {"link": "https://files.example/export.zip"}}

    def download(url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(archive)

    return get_json, download, seen_urls


def test_fetch_extracts_checksums_and_records_provenance(tmp_path):
    archive = _zip_bytes({"train/_annotations.coco.json": "{}", "README.txt": "hi"})
    get_json, download, urls = _fakes(archive, pending=2)
    out = FT.fetch_roboflow(
        "x",
        "coco-segmentation",
        api_key=SECRET,
        get_json=get_json,
        download=download,
        poll_s=0,
        cfg=_cfg(),
        dest_root=tmp_path,
    )
    assert (out / "train" / "_annotations.coco.json").is_file()
    assert urls[-1].startswith("https://api.roboflow.com/ws/proj/2/coco-segmentation?")
    prov = json.loads((tmp_path / "x" / "PROVENANCE.json").read_text(encoding="utf-8"))
    assert prov["license"] == "CC-BY-4.0" and len(prov["sha256"]) == 64
    assert "publish_media" not in prov["allowed_uses"]  # undocumented provenance
    assert SECRET not in json.dumps(prov)


def test_fetch_refuses_a_dataset_its_licence_does_not_allow(tmp_path):
    get_json, download, urls = _fakes(_zip_bytes({"a": "b"}))
    with pytest.raises(DatasetUseError):
        FT.fetch_roboflow(
            "x",
            "coco",
            api_key=SECRET,
            get_json=get_json,
            download=download,
            cfg=_cfg(license_status="none", license=None),
            dest_root=tmp_path,
        )
    assert urls == []  # refused before any request


def test_api_errors_never_echo_the_key(tmp_path):
    def get_json(url):
        raise urllib.error.HTTPError(url, 401, "no", {}, io.BytesIO(f"bad key {SECRET}".encode()))

    with pytest.raises(RuntimeError) as exc:
        FT.fetch_roboflow(
            "x",
            "coco",
            api_key=SECRET,
            get_json=get_json,
            download=lambda u, d: None,
            cfg=_cfg(),
            dest_root=tmp_path,
        )
    assert SECRET not in str(exc.value) and "401" in str(exc.value)


def test_archive_path_traversal_is_rejected(tmp_path):
    get_json, download, _ = _fakes(_zip_bytes({"../../evil.txt": "x"}))
    with pytest.raises(RuntimeError, match="unsafe path"):
        FT.fetch_roboflow(
            "x",
            "coco",
            api_key=SECRET,
            get_json=get_json,
            download=download,
            poll_s=0,
            cfg=_cfg(),
            dest_root=tmp_path,
        )
    assert not (tmp_path.parent / "evil.txt").exists()


def test_read_env_value(tmp_path):
    env = tmp_path / ".env"
    env.write_text('# c\nROBOFLOW_API_KEY="abc"\nOTHER=1\n', encoding="utf-8")
    assert FT.read_env_value("ROBOFLOW_API_KEY", env) == "abc"
    env.write_text("ROBOFLOW_API_KEY=\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing or empty"):
        FT.read_env_value("ROBOFLOW_API_KEY", env)
