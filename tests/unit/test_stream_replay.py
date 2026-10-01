"""P6-T1 required test: a recorded clip replays at its native timestamps."""

from __future__ import annotations

import time

import numpy as np
import pytest

from fixtures.synthetic import load_scene
from vscs.common.io import write_jsonl
from vscs.stream.replay import FileReplay, frames_from_capture_run

MS = 1_000_000


class FakeClock:
    def __init__(self, start=10_000 * MS):
        self.t = start

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += round(s * 1e9)


def _vfr_times():
    """The fixture trajectory's jittered timestamps: a variable-frame-rate recording."""
    return [p.t_ns for p in load_scene().camera_trajectory()]


def test_emits_on_the_recording_schedule_exactly_including_vfr():
    t = _vfr_times()
    clk = FakeClock()
    out = list(FileReplay([(ti, i) for i, ti in enumerate(t)], clock=clk, sleep=clk.sleep))
    emitted = np.array([f.t_emit_ns for f in out]) - out[0].t_emit_ns
    np.testing.assert_array_equal(emitted, np.array(t) - t[0])  # not i / fps
    assert [f.t_capture_ns for f in out] == t and [f.seq for f in out] == list(range(len(t)))
    assert np.ptp(np.diff(t)) > 0  # the fixture really is variable-rate


def test_speed_scales_the_schedule():
    clk = FakeClock()
    out = list(FileReplay([(0, "a"), (100 * MS, "b")], speed=2.0, clock=clk, sleep=clk.sleep))
    assert out[1].t_emit_ns - out[0].t_emit_ns == 50 * MS


def test_a_slow_consumer_gets_late_frames_never_skipped_ones():
    clk = FakeClock()
    rep = FileReplay(
        [(i * 33 * MS, i) for i in range(10)], late_tolerance_ns=MS, clock=clk, sleep=clk.sleep
    )
    got = []
    for f in rep:
        got.append(f.payload)
        clk.t += 50 * MS  # the consumer takes 50 ms per frame on a 33 ms schedule
    assert got == list(range(10))  # nothing skipped
    assert rep.late == 9 and rep.max_lateness_ns == 9 * (50 - 33) * MS


@pytest.mark.parametrize("frames", [[], [(5, "a"), (5, "b")], [(5, "a"), (4, "b")]])
def test_bad_input_is_rejected(frames):
    with pytest.raises(ValueError):
        FileReplay(frames)


def test_real_clock_stays_within_one_frame():
    """P6-T1 acceptance, on real time: emitted gaps match the recording within one frame."""
    period = 20 * MS
    t = [i * period for i in range(15)]
    out = list(FileReplay([(ti, None) for ti in t]))
    emitted = np.array([f.t_emit_ns for f in out]) - out[0].t_emit_ns
    assert np.abs(emitted - np.array(t)).max() < period


def test_reads_an_extract_frames_run(tmp_path):
    rows = [
        {"frame_index": i * 10, "t_ns": 1000 + i * 33 * MS, "file": f"frames/{i * 10:06d}.png"}
        for i in (2, 0, 1)
    ]
    write_jsonl(tmp_path / "frames.jsonl", rows)
    frames = frames_from_capture_run(tmp_path)
    assert [f[0] for f in frames] == sorted(r["t_ns"] for r in rows)
    assert frames[0][1] == tmp_path / "frames/000000.png"


def test_the_real_clock_is_monotonic_ns():
    a = time.perf_counter_ns()
    assert time.perf_counter_ns() >= a
