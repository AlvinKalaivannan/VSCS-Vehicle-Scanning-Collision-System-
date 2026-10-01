"""P6-T3: latency accounting is exact, statistics are right, budgets are checked."""

from __future__ import annotations

import pytest

from vscs.stream.metrics import LatencyRecorder, StreamFrame, breakdown, over_budget

MS = 1_000_000


def _frame(seq, emit_ms, stages):
    """``stages``: list of (name, wait_ms, service_ms) in order."""
    f = StreamFrame(seq=seq, t_capture_ns=seq * 66 * MS, t_emit_ns=emit_ms * MS)
    t = emit_ms * MS
    for name, wait, service in stages:
        t_in = t + wait * MS
        t_out = t_in + service * MS
        f.stamp(name, t_in, t_out)
        t = t_out
    return f


def test_stage_latencies_sum_exactly_to_end_to_end():
    """CLAUDE.md §5 required test: no unaccounted time."""
    f = _frame(0, 100, [("decode", 2, 5), ("detect", 7, 40), ("risk", 1, 3)])
    b = breakdown(f)
    assert b["service"] == {"decode": 5 * MS, "detect": 40 * MS, "risk": 3 * MS}
    assert b["wait"] == {"decode": 2 * MS, "detect": 7 * MS, "risk": 1 * MS}
    assert b["end_to_end"] == 58 * MS
    assert sum(b["service"].values()) + sum(b["wait"].values()) == b["end_to_end"]


def test_a_frame_without_stamps_is_an_error():
    with pytest.raises(ValueError, match="no stage stamps"):
        breakdown(StreamFrame(seq=1, t_capture_ns=0, t_emit_ns=0))


def test_percentiles_fps_and_drops():
    rec = LatencyRecorder(window=100)
    # 100 frames emitted every 50 ms; detect takes 1..100 ms (so p50 ~ 50.5, p99 ~ 99).
    for i in range(100):
        rec.finish(_frame(i, i * 50, [("detect", 0, i + 1)]))
    rec.drop("detect_in", 7)
    s = rec.summary()
    assert s["stages"]["detect"]["p50_ms"] == pytest.approx(50.5)
    assert s["stages"]["detect"]["p99_ms"] == pytest.approx(99.01)
    assert s["stages"]["detect"]["n"] == 100
    # Finish times run from 1 ms to 4950 + 100 = 5050 ms: 99 intervals over 5.049 s.
    assert s["fps"] == pytest.approx(99 / 5.049)
    assert s["dropped"] == {"detect_in": 7}


def test_the_window_rolls():
    rec = LatencyRecorder(window=10)
    for i in range(50):
        rec.finish(_frame(i, i * 10, [("s", 0, 100 if i < 40 else 1)]))
    assert rec.summary()["stages"]["s"]["p99_ms"] == pytest.approx(1.0)  # old frames gone
    assert rec.finished == 50


def test_over_budget_reports_only_the_offenders():
    rec = LatencyRecorder(window=50)
    for i in range(50):
        rec.finish(_frame(i, i * 70, [("decode", 0, 5), ("detect", 0, 60)]))
    s = rec.summary()
    assert over_budget(s, {"decode": 10, "detect": 50, "end_to_end": 150}) == {"detect": 60.0}


def test_empty_summary_is_well_formed():
    s = LatencyRecorder(window=5).summary()
    assert s["end_to_end"] is None and s["fps"] == 0.0
