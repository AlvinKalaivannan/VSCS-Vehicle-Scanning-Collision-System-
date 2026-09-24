"""Run-folder, JSON Lines, metric-append and checksum tests (CLAUDE.md section 4.3)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

import pytest

from vscs.common import io as IO
from vscs.common.types import ComponentRisk, RiskFrame


# --------------------------------------------------------------------------- #
# Run folders                                                                  #
# --------------------------------------------------------------------------- #
def test_make_run_dir_layout_and_provenance(tmp_path):
    cfg = {"a": 1, "nested": {"b": 2.5}}
    run_dir = IO.make_run_dir("recon", config=cfg, root=tmp_path, now=datetime(2026, 10, 19, 14, 2))

    assert run_dir.parent.name == "recon"
    assert run_dir.name.startswith("20261019-1402_")
    prov = IO.read_json(run_dir / "provenance.json")
    assert prov["stage"] == "recon"
    assert "git" in prov and "created_utc" in prov
    # The resolved config must be recoverable from the run folder alone.
    assert (run_dir / "resolved_config.yaml").is_file()


def test_make_run_dir_never_overwrites_a_previous_run(tmp_path):
    """Two runs in the same minute on the same commit must not collide."""
    now = datetime(2026, 10, 19, 14, 2)
    a = IO.make_run_dir("risk", root=tmp_path, now=now)
    b = IO.make_run_dir("risk", root=tmp_path, now=now)
    c = IO.make_run_dir("risk", root=tmp_path, now=now)
    assert len({a, b, c}) == 3
    assert b.name.endswith("-2") and c.name.endswith("-3")


def test_make_run_dir_rejects_unknown_stage(tmp_path):
    with pytest.raises(ValueError, match="unknown stage"):
        IO.make_run_dir("not_a_stage", root=tmp_path)


def test_run_dir_without_config_still_records_provenance(tmp_path):
    run_dir = IO.make_run_dir("eval", root=tmp_path)
    assert (run_dir / "provenance.json").is_file()
    assert not (run_dir / "resolved_config.yaml").exists()


def test_git_short_sha_is_a_string():
    sha = IO.git_short_sha()
    assert isinstance(sha, str) and sha


def test_git_commit_info_keys():
    info = IO.git_commit_info()
    assert set(info) == {"commit", "short", "branch", "dirty"}


# --------------------------------------------------------------------------- #
# JSON Lines                                                                   #
# --------------------------------------------------------------------------- #
def _frame(t_ns: int, p: float) -> RiskFrame:
    return RiskFrame.from_components(
        t_ns=t_ns,
        ego_speed_mps=1.1,
        per_component=[
            ComponentRisk(
                component="rear_right_bumper_corner",
                min_distance_m=0.5,
                ttc_s=1.2,
                p_contact=p,
                severity=1.5,
                risk=p * 1.5,
            )
        ],
        alert_level="caution",
    )


def test_riskframe_jsonl_round_trip(tmp_path):
    """RiskFrame streams serialize as JSON Lines (CLAUDE.md section 4.2)."""
    frames = [_frame(1_000_000_000, 0.2), _frame(1_033_000_000, 0.4)]
    path = IO.write_jsonl(tmp_path / "stream.jsonl", frames)

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2

    back = [RiskFrame(**rec) for rec in IO.read_jsonl(path)]
    assert back == frames
    assert back[0].worst_component == "rear_right_bumper_corner"
    assert back[0].schema_version == frames[0].schema_version


def test_write_jsonl_handles_plain_dicts(tmp_path):
    path = IO.write_jsonl(tmp_path / "d.jsonl", [{"a": 1}, {"a": 2}])
    assert [r["a"] for r in IO.read_jsonl(path)] == [1, 2]


def test_read_jsonl_skips_blank_lines_by_default(tmp_path):
    p = tmp_path / "blank.jsonl"
    p.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert len(list(IO.read_jsonl(p))) == 2
    with pytest.raises(ValueError, match="blank line"):
        list(IO.read_jsonl(p, skip_blank=False))


def test_read_jsonl_reports_the_offending_line_number(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r":2: invalid JSON"):
        list(IO.read_jsonl(p))


def test_append_jsonl_accumulates(tmp_path):
    p = tmp_path / "log.jsonl"
    IO.append_jsonl(p, {"i": 1})
    IO.append_jsonl(p, {"i": 2})
    assert [r["i"] for r in IO.read_jsonl(p)] == [1, 2]


def test_write_json_creates_parents(tmp_path):
    p = IO.write_json(tmp_path / "deep" / "x.json", {"k": "v"})
    assert json.loads(p.read_text(encoding="utf-8")) == {"k": "v"}


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def test_append_metric_writes_the_section_7_4_schema(tmp_path):
    p = tmp_path / "results.jsonl"
    rec = IO.append_metric(
        task="P1-T6",
        metric="scale_error_m",
        value=0.013,
        split="dev",
        run_dir="data/processed/recon/20261019-1402_abc1234",
        notes="four dimensions checked",
        path=p,
    )
    assert set(rec) == {"ts", "git", "task", "metric", "value", "split", "run_dir", "notes"}
    assert rec["value"] == 0.013
    on_disk = list(IO.read_jsonl(p))
    assert len(on_disk) == 1 and on_disk[0]["metric"] == "scale_error_m"


def test_append_metric_is_append_only(tmp_path):
    """Corrections are appended, never edited in place (section 7.4)."""
    p = tmp_path / "results.jsonl"
    IO.append_metric(
        task="P1-T6", metric="scale_error_m", value=0.05, split="dev", run_dir="r1", path=p
    )
    IO.append_metric(
        task="P1-T6",
        metric="scale_error_m",
        value=0.013,
        split="dev",
        run_dir="r2",
        notes="corrects the previous line: wrong marker size used",
        path=p,
    )
    rows = list(IO.read_jsonl(p))
    assert len(rows) == 2
    assert rows[0]["value"] == 0.05, "the earlier line must survive"


def test_metrics_path_points_into_the_repo():
    p = IO.metrics_path()
    assert p.name == "results.jsonl" and p.parent.name == "metrics"


# --------------------------------------------------------------------------- #
# Checksums - R-13                                                             #
# --------------------------------------------------------------------------- #
def test_sha256_matches_hashlib_and_streams_correctly(tmp_path):
    data = b"vscs raw capture bytes" * 1000
    p = tmp_path / "clip.bin"
    p.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    assert IO.sha256_file(p) == expected
    # A tiny chunk size forces many iterations; the result must be identical.
    assert IO.sha256_file(p, chunk_size=7) == expected


def test_sha256_of_empty_file(tmp_path):
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")
    assert IO.sha256_file(p) == hashlib.sha256(b"").hexdigest()


def test_sha256_detects_a_single_flipped_byte(tmp_path):
    """The point of the manifest checksum: catch silent corruption after a copy."""
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"\x00" * 4096)
    b.write_bytes(b"\x00" * 4095 + b"\x01")
    assert IO.sha256_file(a) != IO.sha256_file(b)
