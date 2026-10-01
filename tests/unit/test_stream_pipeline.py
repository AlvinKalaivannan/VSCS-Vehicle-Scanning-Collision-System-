"""Staged pipeline: correctness, conservation, error isolation, and a real 2x overload."""

from __future__ import annotations

import time

import numpy as np

from vscs.common.config import load_config
from vscs.stream.metrics import LatencyRecorder, breakdown
from vscs.stream.pipeline import Pipeline, Stage
from vscs.stream.replay import FileReplay

MS = 1_000_000
CFG = load_config("stream")


def _instant(n):
    """A replay with no waiting (fake clock jumps), for logic-only tests."""
    clock = {"t": 0}

    def now():
        clock["t"] += 1
        return clock["t"]

    return FileReplay([(i * 33 * MS, i) for i in range(n)], clock=now, sleep=lambda s: None)


def test_config_budget_is_consistent():
    b = CFG["budget_ms"]
    assert set(CFG["stages"]) <= set(b)
    assert sum(b[s] for s in CFG["stages"]) <= b["end_to_end"]
    assert CFG["queue_size"] >= 1


def test_stages_run_in_order_and_every_frame_finishes_when_not_overloaded():
    out = []
    p = Pipeline(
        [Stage("a", lambda x: x + 1), Stage("b", lambda x: x * 2)],
        queue_size=1000,
        recorder=LatencyRecorder(500),
        sink=lambda f: out.append(f),
    )
    s = p.run(_instant(200))
    assert [f.payload for f in out] == [(i + 1) * 2 for i in range(200)]
    assert s["dropped"] == {} and s["errors"] == {"a": 0, "b": 0}
    for f in out:  # every finished frame's latency is fully accounted for
        b = breakdown(f)
        assert sum(b["service"].values()) + sum(b["wait"].values()) == b["end_to_end"]


def test_every_frame_is_finished_dropped_or_failed_never_lost():
    finished = []

    def flaky(x):
        if x % 10 == 3:
            raise RuntimeError("bad frame")
        time.sleep(0.001)
        return x

    rec = LatencyRecorder(1000)
    p = Pipeline(
        [Stage("flaky", flaky), Stage("slow", lambda x: (time.sleep(0.002), x)[1])],
        queue_size=2,
        recorder=rec,
        sink=finished.append,
    )
    # Paced in real time (3 ms) so frames reach the failing stage: an instant source floods
    # the 2-slot first queue, and the bad frames are evicted before they can fail.
    s = p.run(FileReplay([(i * 3 * MS, i) for i in range(300)]))
    dropped = sum(s["dropped"].values())
    assert len(finished) + dropped + s["errors"]["flaky"] == 300
    assert s["errors"]["flaky"] >= 1


def test_latency_stays_bounded_under_2x_overload_on_real_threads():
    """200 fps in, a 10 ms stage (100 fps): half the frames must go, and latency must not grow."""
    finished = []
    rec = LatencyRecorder(1000)
    p = Pipeline(
        [Stage("decode", lambda x: x), Stage("detect", lambda x: (time.sleep(0.010), x)[1])],
        queue_size=2,
        recorder=rec,
        sink=finished.append,
    )
    source = FileReplay([(i * 5 * MS, i) for i in range(240)])
    s = p.run(source)
    e2e_ms = np.array([breakdown(f)["end_to_end"] for f in finished]) / MS
    assert sum(s["dropped"].values()) >= 60  # overload shows up as counted drops
    # Bounded: about queue_size + 1 services; generous for Windows timer granularity.
    assert np.percentile(e2e_ms, 95) < 120
    # Not growing: the last third is no slower than the first third (plus timing slack).
    third = len(e2e_ms) // 3
    assert e2e_ms[-third:].mean() <= e2e_ms[:third].mean() + 20
    assert s["fps"] > 0
