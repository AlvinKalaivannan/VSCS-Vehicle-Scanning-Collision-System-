"""Ingest tests (P1-T4): checksums, manifest, and the dev/test split.

The split assignment carries the most weight here. R-09 is about not being able to
reshuffle dev/test once results exist, so these tests pin determinism, stratification,
and the refusal to move a file between splits.
"""

from __future__ import annotations

import collections
from datetime import date
from pathlib import Path

import pytest

from vscs.capture import ingest as I
from vscs.common.config import load_config


@pytest.fixture
def manifest(tmp_path):
    """A copy of the real MANIFEST.md, including its placeholder row."""
    from vscs.common.config import repo_root

    src = repo_root() / "data" / "MANIFEST.md"
    dst = tmp_path / "MANIFEST.md"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dst


@pytest.fixture
def eval_cfg(tmp_path):
    """A copy of the real eval.yaml, so split writes do not touch the repo."""
    from vscs.common.config import config_dir

    src = config_dir() / "eval.yaml"
    dst = tmp_path / "eval.yaml"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dst


def _src_file(tmp_path, name="lot_pass01_pole.mp4", content=b"fake video bytes") -> Path:
    p = tmp_path / "src" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


# --------------------------------------------------------------------------- #
# Split assignment - R-09                                                      #
# --------------------------------------------------------------------------- #
def test_split_is_deterministic():
    """Re-running ingest on the same file must give the same split, always."""
    args = {"stratum": "pole", "ratio_dev": 0.7, "seed": 20260924}
    first = I.assign_split("lot_20261011_pass07", **args)
    for _ in range(5):
        assert I.assign_split("lot_20261011_pass07", **args) == first


def test_split_depends_on_the_id():
    args = {"stratum": "pole", "ratio_dev": 0.7, "seed": 20260924}
    splits = {I.assign_split(f"lot_{i:03d}", **args) for i in range(40)}
    assert splits == {"dev", "test"}, "both splits must be reachable"


def test_split_ratio_is_roughly_honoured():
    args = {"stratum": "mixed", "ratio_dev": 0.7, "seed": 20260924}
    labels = [I.assign_split(f"lot_{i:04d}", **args) for i in range(4000)]
    frac_dev = labels.count("dev") / len(labels)
    assert frac_dev == pytest.approx(0.7, abs=0.03)


def test_split_is_stratified_so_each_category_is_represented():
    """CLAUDE.md §11: both splits must contain pole, kerb and box cases."""
    per_stratum: dict[str, collections.Counter] = {}
    for stratum in ("pole", "curb", "box"):
        counter: collections.Counter = collections.Counter()
        for i in range(200):
            counter[
                I.assign_split(f"lot_{stratum}_{i:03d}", stratum=stratum, ratio_dev=0.7, seed=1)
            ] += 1
        per_stratum[stratum] = counter

    for stratum, counter in per_stratum.items():
        assert counter["dev"] > 0, f"{stratum} never lands in dev"
        assert counter["test"] > 0, f"{stratum} never lands in test"
        assert counter["dev"] / 200 == pytest.approx(0.7, abs=0.08), stratum


def test_split_changes_with_the_seed():
    a = [I.assign_split(f"x{i}", stratum="pole", ratio_dev=0.7, seed=1) for i in range(60)]
    b = [I.assign_split(f"x{i}", stratum="pole", ratio_dev=0.7, seed=2) for i in range(60)]
    assert a != b


def test_split_rejects_an_impossible_ratio():
    for bad in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError, match="ratio_dev"):
            I.assign_split("x", stratum="pole", ratio_dev=bad, seed=1)


def test_the_real_config_ratio_and_seed_are_usable():
    cfg = load_config("eval")["splits"]
    label = I.assign_split(
        "lot_20261011_pass01", stratum="pole", ratio_dev=cfg["ratio_dev"], seed=cfg["seed"]
    )
    assert label in ("dev", "test")


# --------------------------------------------------------------------------- #
# Stratum inference                                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name,expected",
    [
        ("lot_pass01_pole.mp4", "pole"),
        ("PASS_02_POLE.MOV", "pole"),
        ("pass03_curb.mp4", "curb"),
        ("pass04_kerb.mp4", "curb"),
        ("pass05_boxes.mp4", "box"),
        ("pass06_cones.mp4", "box"),
        ("pass07.mp4", "mixed"),
    ],
)
def test_infer_stratum(name, expected):
    assert I.infer_stratum(name) == expected


def test_all_inferred_strata_are_declared():
    for name in ("a_pole.mp4", "a_curb.mp4", "a_box.mp4", "whatever.mp4"):
        assert I.infer_stratum(name) in I.STRATA


# --------------------------------------------------------------------------- #
# Raw ids                                                                      #
# --------------------------------------------------------------------------- #
def test_raw_id_shape():
    rid = I.make_raw_id("lot", Path("Pass 07.MP4"), on=date(2026, 10, 11))
    assert rid == "lot_20261011_pass_07"


def test_raw_id_with_index():
    rid = I.make_raw_id("scan", Path("loop1.mp4"), on=date(2026, 10, 3), index=2)
    assert rid == "scan_20261003_loop1_02"


# --------------------------------------------------------------------------- #
# Manifest - R-13                                                              #
# --------------------------------------------------------------------------- #
def _record(**kw) -> I.RawRecord:
    base = dict(
        id="lot_20261011_pass07",
        date="2026-10-11",
        kind="lot",
        sha256="a" * 64,
        size_bytes=1234,
        description="reverse toward pole",
        split="dev",
        stratum="pole",
        source_name="pass07.mp4",
        dest_path="data/raw/lot_20261011_pass07/pass07.mp4",
        backups=["laptop"],
        verified=True,
        duration_s=12.5,
        n_frames=375,
    )
    base.update(kw)
    return I.RawRecord(**base)


def test_manifest_row_replaces_the_placeholder(manifest):
    I.append_manifest_row(_record(), manifest_path=manifest)
    text = manifest.read_text(encoding="utf-8")
    assert I._MANIFEST_PLACEHOLDER not in text
    assert "lot_20261011_pass07" in text
    assert "reverse toward pole" in text


def test_manifest_rows_accumulate(manifest):
    I.append_manifest_row(_record(id="a"), manifest_path=manifest)
    I.append_manifest_row(_record(id="b"), manifest_path=manifest)
    text = manifest.read_text(encoding="utf-8")
    assert "| `a` |" in text and "| `b` |" in text


def test_re_ingesting_updates_the_row_rather_than_duplicating(manifest):
    I.append_manifest_row(_record(description="first"), manifest_path=manifest)
    I.append_manifest_row(_record(description="second"), manifest_path=manifest)
    text = manifest.read_text(encoding="utf-8")
    assert text.count("| `lot_20261011_pass07` |") == 1
    assert "second" in text and "first" not in text


def test_unverified_row_is_shouted_about(manifest):
    I.append_manifest_row(_record(verified=False), manifest_path=manifest)
    assert "**NO**" in manifest.read_text(encoding="utf-8")


def test_manifest_row_shows_backups(manifest):
    I.append_manifest_row(_record(backups=["laptop", "external", "cloud"]), manifest_path=manifest)
    assert "laptop, external, cloud" in manifest.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# eval.yaml split recording                                                    #
# --------------------------------------------------------------------------- #
def test_recording_a_split_preserves_comments_and_other_keys(eval_cfg):
    I.record_split_in_eval_config("lot_20261011_pass07", "dev", config_path=eval_cfg)
    text = eval_cfg.read_text(encoding="utf-8")
    assert "R-09" in text, "the comments explaining the split rule must survive"

    cfg = load_config(str(eval_cfg))
    assert cfg["splits"]["dev"] == ["lot_20261011_pass07"]
    assert cfg["splits"]["test"] == []
    assert cfg["splits"]["test_split_used"] is False
    # Untouched sections must remain.
    assert cfg["thresholds"]["scale_error_m"]["limit"] == 0.02
    assert cfg["baseline"]["name"] == "single_obb"


def test_recording_is_idempotent(eval_cfg):
    I.record_split_in_eval_config("x", "test", config_path=eval_cfg)
    I.record_split_in_eval_config("x", "test", config_path=eval_cfg)
    assert load_config(str(eval_cfg))["splits"]["test"] == ["x"]


def test_split_lists_stay_sorted(eval_cfg):
    for rid in ("c", "a", "b"):
        I.record_split_in_eval_config(rid, "dev", config_path=eval_cfg)
    assert load_config(str(eval_cfg))["splits"]["dev"] == ["a", "b", "c"]


def test_moving_a_file_between_splits_is_refused(eval_cfg):
    """R-09: re-splitting after the fact invalidates the held-out evaluation."""
    I.record_split_in_eval_config("lot_x", "dev", config_path=eval_cfg)
    with pytest.raises(ValueError, match="already recorded in the dev split"):
        I.record_split_in_eval_config("lot_x", "test", config_path=eval_cfg)


# --------------------------------------------------------------------------- #
# Ingest end to end                                                            #
# --------------------------------------------------------------------------- #
def test_ingest_copies_verifies_and_records(tmp_path, manifest, eval_cfg):
    src = _src_file(tmp_path)
    record = I.ingest_file(
        src,
        kind="lot",
        description="reverse toward pole",
        dest_root=tmp_path / "raw",
        eval_config_path=eval_cfg,
        manifest_path=manifest,
        on=date(2026, 10, 11),
    )

    assert record.id == "lot_20261011_lot_pass01_pole"
    assert record.verified is True
    assert record.stratum == "pole"
    assert record.split in ("dev", "test")

    # The source must survive untouched.
    assert src.is_file() and src.read_bytes() == b"fake video bytes"

    dest = tmp_path / "raw" / record.id / src.name
    assert dest.is_file() and dest.read_bytes() == src.read_bytes()
    assert (dest.parent / "ingest.json").is_file()
    assert record.id in manifest.read_text(encoding="utf-8")
    assert record.id in str(load_config(str(eval_cfg))["splits"])


def test_ingest_never_overwrites_existing_raw_data(tmp_path, manifest, eval_cfg):
    """Raw data is irreplaceable (CLAUDE.md §0)."""
    src = _src_file(tmp_path)
    kwargs = dict(
        kind="lot",
        dest_root=tmp_path / "raw",
        eval_config_path=eval_cfg,
        manifest_path=manifest,
        on=date(2026, 10, 11),
    )
    I.ingest_file(src, **kwargs)
    with pytest.raises(FileExistsError, match="never overwritten"):
        I.ingest_file(src, **kwargs)


def test_scan_and_calib_get_no_split(tmp_path, manifest, eval_cfg):
    """Only lot passes are evaluation data."""
    for kind in ("scan", "calib"):
        src = _src_file(tmp_path, name=f"{kind}_file.mp4")
        record = I.ingest_file(
            src,
            kind=kind,
            dest_root=tmp_path / f"raw_{kind}",
            eval_config_path=eval_cfg,
            manifest_path=manifest,
            on=date(2026, 10, 3),
        )
        assert record.split is None
    assert load_config(str(eval_cfg))["splits"]["dev"] == []


def test_ingest_records_the_source_checksum(tmp_path, manifest, eval_cfg):
    from vscs.common.io import sha256_file

    src = _src_file(tmp_path, content=b"some distinctive bytes")
    record = I.ingest_file(
        src,
        kind="scan",
        dest_root=tmp_path / "raw",
        eval_config_path=eval_cfg,
        manifest_path=manifest,
    )
    assert record.sha256 == sha256_file(src)
    assert len(record.sha256) == 64


def test_verify_copy_detects_corruption(tmp_path):
    from vscs.common.io import sha256_file

    good = tmp_path / "good.bin"
    good.write_bytes(b"\x00" * 512)
    sha = sha256_file(good)
    assert I.verify_copy(sha, good)

    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"\x00" * 511 + b"\x01")
    assert not I.verify_copy(sha, bad)


def test_ingest_rejects_an_unknown_stratum(tmp_path, manifest, eval_cfg):
    src = _src_file(tmp_path)
    with pytest.raises(ValueError, match="unknown stratum"):
        I.ingest_file(
            src,
            kind="lot",
            stratum="banana",
            dest_root=tmp_path / "raw",
            eval_config_path=eval_cfg,
            manifest_path=manifest,
        )


def test_ingest_rejects_a_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        I.ingest_file(tmp_path / "nope.mp4", kind="scan")


def test_ingest_folder_handles_several_files(tmp_path, manifest, eval_cfg):
    for i in range(3):
        _src_file(tmp_path, name=f"pass{i:02d}_pole.mp4", content=f"clip {i}".encode())
    (tmp_path / "src" / "notes.md").write_text("ignored", encoding="utf-8")

    records = I.ingest_folder(
        tmp_path / "src",
        kind="lot",
        dest_root=tmp_path / "raw",
        eval_config_path=eval_cfg,
        manifest_path=manifest,
        on=date(2026, 10, 11),
    )
    assert len(records) == 3
    assert len({r.id for r in records}) == 3
    # The .md is not an ingestable capture file.
    assert all("notes" not in r.id for r in records)


def test_ingest_folder_rejects_an_empty_folder(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no ingestable files"):
        I.ingest_folder(empty, kind="lot")


def test_ingest_folder_rejects_a_file(tmp_path):
    src = _src_file(tmp_path)
    with pytest.raises(NotADirectoryError):
        I.ingest_folder(src, kind="lot")
