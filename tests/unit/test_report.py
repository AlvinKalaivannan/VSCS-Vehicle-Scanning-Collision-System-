"""Report generation tests - the P0-T2 acceptance criterion.

"`python scripts/report.py` generates `docs/REPORT.md` without error"
(CLAUDE.md section 6), including when there are no metrics at all.
"""

from __future__ import annotations

import pytest

from vscs.common.io import append_metric
from vscs.eval import report as R

THRESHOLDS = {
    "scale_error_m": {"task": "P1-T6", "limit": 0.02, "direction": "below"},
    "component_iou_mean": {"task": "P2-T3", "limit": 0.70, "direction": "above"},
}


def _rec(metric: str, value: float, ts: str, task: str = "P1-T6", **kw):
    base = {
        "ts": ts,
        "git": "abc1234",
        "task": task,
        "metric": metric,
        "value": value,
        "split": "dev",
        "run_dir": "data/processed/recon/20261019-1402_abc1234",
        "notes": "",
    }
    base.update(kw)
    return base


# --------------------------------------------------------------------------- #
# The empty case - P0-T2                                                       #
# --------------------------------------------------------------------------- #
def test_report_generates_from_an_empty_metrics_file(tmp_path):
    metrics = tmp_path / "results.jsonl"
    metrics.write_text("", encoding="utf-8")
    out = R.generate_report(
        metrics_file=metrics, out_path=tmp_path / "REPORT.md", eval_config={"thresholds": {}}
    )
    assert out.is_file()
    assert "VSCS metrics report" in out.read_text(encoding="utf-8")


def test_report_generates_when_the_metrics_file_is_missing(tmp_path):
    out = R.generate_report(
        metrics_file=tmp_path / "nope.jsonl",
        out_path=tmp_path / "REPORT.md",
        eval_config={"thresholds": {}},
    )
    assert "No metrics measured yet" in out.read_text(encoding="utf-8")


def test_empty_report_still_lists_unproven_gates(tmp_path):
    """An unmeasured acceptance gate must appear, so it cannot be quietly forgotten."""
    out = R.generate_report(
        metrics_file=tmp_path / "none.jsonl",
        out_path=tmp_path / "REPORT.md",
        eval_config={"thresholds": THRESHOLDS},
    )
    text = out.read_text(encoding="utf-8")
    assert "no measurement yet" in text
    assert "scale_error_m" in text and "component_iou_mean" in text
    # Crucially, an unmeasured gate is not reported as passing.
    assert "PASS" not in text


def test_the_real_repo_config_generates_a_report(tmp_path):
    """Runs against the actual configs/eval.yaml, as the script does."""
    out = R.generate_report(metrics_file=tmp_path / "empty.jsonl", out_path=tmp_path / "R.md")
    assert out.is_file() and out.stat().st_size > 0


# --------------------------------------------------------------------------- #
# Loading and validation                                                      #
# --------------------------------------------------------------------------- #
def test_load_metrics_rejects_a_line_missing_required_fields(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"ts": "2026-01-01", "metric": "x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing required field"):
        R.load_metrics(p)


def test_load_metrics_round_trips_append_metric(tmp_path):
    p = tmp_path / "results.jsonl"
    append_metric(
        task="P1-T6", metric="scale_error_m", value=0.013, split="dev", run_dir="r", path=p
    )
    recs = R.load_metrics(p)
    assert len(recs) == 1 and recs[0]["metric"] == "scale_error_m"


# --------------------------------------------------------------------------- #
# Summarising                                                                  #
# --------------------------------------------------------------------------- #
def test_latest_is_by_timestamp_not_file_order():
    """The log is append-only but a correction may be appended out of order."""
    records = [
        _rec("scale_error_m", 0.05, "2026-10-19T10:00:00+00:00"),
        _rec("scale_error_m", 0.013, "2026-10-20T10:00:00+00:00"),
        _rec("scale_error_m", 0.030, "2026-10-19T12:00:00+00:00"),
    ]
    s = R.summarize(records, THRESHOLDS)[1]
    assert s.metric == "scale_error_m"
    assert s.latest_value == 0.013
    assert s.previous_value == 0.030
    assert s.n_samples == 3


def test_pass_and_fail_for_below_direction():
    passing = R.summarize([_rec("scale_error_m", 0.013, "t1")], THRESHOLDS)
    assert next(s for s in passing if s.metric == "scale_error_m").status == "pass"
    failing = R.summarize([_rec("scale_error_m", 0.05, "t1")], THRESHOLDS)
    assert next(s for s in failing if s.metric == "scale_error_m").status == "fail"


def test_pass_and_fail_for_above_direction():
    good = R.summarize([_rec("component_iou_mean", 0.82, "t1", task="P2-T3")], THRESHOLDS)
    assert next(s for s in good if s.metric == "component_iou_mean").status == "pass"
    bad = R.summarize([_rec("component_iou_mean", 0.61, "t1", task="P2-T3")], THRESHOLDS)
    assert next(s for s in bad if s.metric == "component_iou_mean").status == "fail"


def test_exactly_on_the_limit_counts_as_failing():
    """Section 6 says "within +/- 2 cm" and "IoU >= 0.70" as targets to beat; a value
    exactly on the boundary is not evidence of meeting it, so it is not a pass."""
    on_limit = R.summarize([_rec("scale_error_m", 0.02, "t1")], THRESHOLDS)
    assert next(s for s in on_limit if s.metric == "scale_error_m").status == "fail"


def test_metric_without_a_gate_is_no_gate():
    s = R.summarize([_rec("mystery_metric", 1.0, "t1")], THRESHOLDS)
    assert next(x for x in s if x.metric == "mystery_metric").status == "no gate"


def test_unmeasured_gate_is_not_measured():
    s = R.summarize([], THRESHOLDS)
    assert {x.status for x in s} == {"not measured"}
    assert all(x.latest_value is None for x in s)
    # The task is still known, from the threshold spec.
    assert {x.task for x in s} == {"P1-T6", "P2-T3"}


def test_trend_reports_improvement_in_the_good_direction():
    lower_is_better = R.summarize(
        [
            _rec("scale_error_m", 0.05, "2026-10-19T10:00:00+00:00"),
            _rec("scale_error_m", 0.013, "2026-10-20T10:00:00+00:00"),
        ],
        THRESHOLDS,
    )
    assert "improved" in next(s for s in lower_is_better if s.metric == "scale_error_m").trend

    higher_is_better = R.summarize(
        [
            _rec("component_iou_mean", 0.80, "2026-10-19T10:00:00+00:00", task="P2-T3"),
            _rec("component_iou_mean", 0.72, "2026-10-20T10:00:00+00:00", task="P2-T3"),
        ],
        THRESHOLDS,
    )
    assert "worse" in next(s for s in higher_is_better if s.metric == "component_iou_mean").trend


def test_trend_is_flat_and_dash_cases():
    flat = R.summarize(
        [
            _rec("scale_error_m", 0.013, "2026-10-19T10:00:00+00:00"),
            _rec("scale_error_m", 0.013, "2026-10-20T10:00:00+00:00"),
        ],
        THRESHOLDS,
    )
    assert next(s for s in flat if s.metric == "scale_error_m").trend == "flat"
    single = R.summarize([_rec("scale_error_m", 0.013, "t1")], THRESHOLDS)
    assert next(s for s in single if s.metric == "scale_error_m").trend == "-"


# --------------------------------------------------------------------------- #
# Rendering                                                                    #
# --------------------------------------------------------------------------- #
def test_rendered_report_shows_values_gates_and_evidence(tmp_path):
    metrics = tmp_path / "results.jsonl"
    import json

    with metrics.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_rec("scale_error_m", 0.013, "2026-10-19T14:02:00+00:00")) + "\n")
        fh.write(
            json.dumps(_rec("component_iou_mean", 0.61, "2026-10-20T09:00:00+00:00", task="P2-T3"))
            + "\n"
        )

    out = R.generate_report(
        metrics_file=metrics,
        out_path=tmp_path / "REPORT.md",
        eval_config={"thresholds": THRESHOLDS},
    )
    text = out.read_text(encoding="utf-8")

    assert "0.013" in text
    assert "PASS" in text and "FAIL" in text
    assert "Failing gates" in text
    assert "component_iou_mean" in text
    # Evidence: the run dir must be carried through to the report.
    assert "20261019-1402_abc1234" in text
    assert "1 failing its acceptance gate" in text


def test_report_warns_that_it_is_auto_generated(tmp_path):
    out = R.generate_report(
        metrics_file=tmp_path / "x.jsonl",
        out_path=tmp_path / "REPORT.md",
        eval_config={"thresholds": {}},
    )
    assert "AUTO-GENERATED" in out.read_text(encoding="utf-8")


def test_report_creates_missing_parent_directories(tmp_path):
    out = R.generate_report(
        metrics_file=tmp_path / "x.jsonl",
        out_path=tmp_path / "a" / "b" / "REPORT.md",
        eval_config={"thresholds": {}},
    )
    assert out.is_file()


# --------------------------------------------------------------------------- #
# The CLI                                                                      #
# --------------------------------------------------------------------------- #
def test_script_cli_runs_and_check_flag_detects_failure(tmp_path):
    """scripts/report.py must run as a plain script, which is how it is documented."""
    import json
    import subprocess
    import sys

    from vscs.common.config import repo_root

    metrics = tmp_path / "results.jsonl"
    with metrics.open("w", encoding="utf-8") as fh:
        # scale_error_m of 0.05 exceeds the 0.02 gate in the real configs/eval.yaml.
        fh.write(json.dumps(_rec("scale_error_m", 0.05, "2026-10-19T14:02:00+00:00")) + "\n")

    script = repo_root() / "scripts" / "report.py"
    ok = subprocess.run(
        [sys.executable, str(script), "--metrics", str(metrics), "--out", str(tmp_path / "R.md")],
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0, ok.stderr
    assert "wrote" in ok.stdout

    checked = subprocess.run(
        [
            sys.executable,
            str(script),
            "--metrics",
            str(metrics),
            "--out",
            str(tmp_path / "R2.md"),
            "--check",
        ],
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 1
    assert "scale_error_m" in checked.stdout
