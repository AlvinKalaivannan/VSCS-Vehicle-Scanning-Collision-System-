"""Frame extraction and timing tests (P1-T4).

The VFR behaviour of *real phone footage* cannot be verified here - there is no phone
clip yet, and ffprobe is not installed (ADR 0003). What is verified: the ffprobe output
parser, the OpenCV timestamp path against a generated clip, the timing analysis
(including VFR and gap detection on synthetic timestamp arrays), and that extraction
carries real timestamps rather than frame indices.
"""

from __future__ import annotations

import itertools
import json

import cv2
import numpy as np
import pytest

from vscs.capture import frames as FR
from vscs.common.types import NS_PER_S

FPS = 30.0
SIZE = (160, 120)


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """A short generated clip with a moving bar, so frames actually differ."""
    path = tmp_path_factory.mktemp("vid") / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    assert writer.isOpened(), "OpenCV could not open an mp4v writer"
    for i in range(30):
        frame = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
        x = (i * 5) % (SIZE[0] - 10)
        frame[:, x : x + 10] = 255
        writer.write(frame)
    writer.release()
    assert path.stat().st_size > 0
    return path


# --------------------------------------------------------------------------- #
# ffprobe parsing (pure, so testable without ffmpeg)                           #
# --------------------------------------------------------------------------- #
def test_parse_ffprobe_prefers_best_effort_timestamp():
    payload = json.dumps(
        {
            "frames": [
                {"best_effort_timestamp_time": "0.000000", "pts_time": "9.999"},
                {"best_effort_timestamp_time": "0.033367", "pts_time": "9.999"},
                {"best_effort_timestamp_time": "0.070000", "pts_time": "9.999"},
            ]
        }
    )
    assert FR.parse_ffprobe_frames(payload) == [0, 33_367_000, 70_000_000]


def test_parse_ffprobe_falls_back_to_pts_time():
    payload = json.dumps({"frames": [{"pts_time": "0.5"}, {"pts_time": "1.0"}]})
    assert FR.parse_ffprobe_frames(payload) == [500_000_000, 1_000_000_000]


def test_parse_ffprobe_skips_unusable_frames_but_keeps_the_rest():
    payload = json.dumps(
        {
            "frames": [
                {"pts_time": "0.0"},
                {"pts_time": "N/A"},
                {"best_effort_timestamp_time": None},
                {"pts_time": "0.1"},
            ]
        }
    )
    assert FR.parse_ffprobe_frames(payload) == [0, 100_000_000]


def test_parse_ffprobe_rejects_garbage():
    with pytest.raises(ValueError, match="not valid JSON"):
        FR.parse_ffprobe_frames("<not json>")
    with pytest.raises(ValueError, match="no frames"):
        FR.parse_ffprobe_frames(json.dumps({"frames": []}))
    with pytest.raises(ValueError, match="usable timestamp"):
        FR.parse_ffprobe_frames(json.dumps({"frames": [{"pts_time": "N/A"}]}))


def test_ffprobe_available_returns_a_bool():
    assert isinstance(FR.ffprobe_available(), bool)


# --------------------------------------------------------------------------- #
# OpenCV backend                                                               #
# --------------------------------------------------------------------------- #
def test_opencv_timestamps_are_monotonic_and_correctly_spaced(video):
    """Pins the read-after-read() order.

    Querying POS_MSEC *before* read() duplicates 0.0 on the first two frames and the
    stream stops being monotonic. This test fails if that regresses.
    """
    stamps = FR.probe_timestamps(video, backend="opencv")
    assert stamps.backend == "opencv"
    assert stamps.n_frames == 30
    assert stamps.image_size == SIZE
    assert all(b > a for a, b in itertools.pairwise(stamps.t_ns))
    assert stamps.t_ns[0] == 0
    expected_dt = NS_PER_S / FPS
    assert stamps.t_ns[1] == pytest.approx(expected_dt, rel=1e-3)


def test_probe_auto_picks_a_backend(video):
    stamps = FR.probe_timestamps(video, backend="auto")
    assert stamps.backend in ("ffprobe", "opencv")
    assert stamps.n_frames > 0


def test_probe_rejects_unknown_backend(video):
    with pytest.raises(ValueError, match="unknown backend"):
        FR.probe_timestamps(video, backend="magic")


def test_probe_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        FR.probe_timestamps(tmp_path / "nope.mp4")


# --------------------------------------------------------------------------- #
# Timing analysis                                                              #
# --------------------------------------------------------------------------- #
def _cfr(n: int = 60, fps: float = 30.0) -> list[int]:
    dt = int(NS_PER_S / fps)
    return [i * dt for i in range(n)]


def test_constant_frame_rate_is_not_flagged_as_vfr():
    report = FR.analyse_timing(_cfr())
    assert not report.is_vfr
    assert report.jitter_ratio == pytest.approx(0.0, abs=1e-9)
    assert report.mean_fps == pytest.approx(30.0, rel=1e-6)
    assert report.n_frames == 60
    assert report.gap_indices == []


def test_variable_frame_rate_is_detected():
    rng = np.random.default_rng(0)
    dt = int(NS_PER_S / 30)
    jitter = (rng.random(40) - 0.5) * 0.4 * dt
    t = np.cumsum(np.maximum(dt + jitter, dt * 0.5)).astype(np.int64)
    report = FR.analyse_timing([int(x) for x in t])
    assert report.is_vfr
    assert report.jitter_ratio > FR.VFR_JITTER_THRESHOLD


def test_dropped_frame_gap_is_reported():
    t = _cfr(20)
    dt = t[1] - t[0]
    # Insert a gap of four frame intervals after index 9.
    t = t[:10] + [x + 4 * dt for x in t[10:]]
    report = FR.analyse_timing(t)
    assert report.gap_indices == [10]


def test_analyse_timing_rejects_non_monotonic():
    """The P1-T4 monotonic check."""
    t = _cfr(10)
    t[5], t[6] = t[6], t[5]
    with pytest.raises(ValueError, match="not strictly increasing"):
        FR.analyse_timing(t)


def test_analyse_timing_rejects_frame_indices():
    """R-03: frame index used as time."""
    with pytest.raises(ValueError, match="frame indices"):
        FR.analyse_timing(list(range(30)))


def test_analyse_timing_needs_two_timestamps():
    with pytest.raises(ValueError, match="at least 2"):
        FR.analyse_timing([0])


def test_timing_summary_mentions_cfr_or_vfr():
    assert "CFR" in FR.analyse_timing(_cfr()).summary()


def test_duration_uses_timestamps_not_frame_count():
    """Duration must come from the clock, not from n_frames / nominal_fps."""
    t = _cfr(31)  # 30 intervals at 1/30 s = exactly 1.0 s
    assert FR.analyse_timing(t).duration_s == pytest.approx(1.0, rel=1e-6)


# --------------------------------------------------------------------------- #
# Extraction                                                                   #
# --------------------------------------------------------------------------- #
def test_extract_frames_writes_images_and_a_timestamped_index(video, tmp_path):
    meta = FR.extract_frames(video, tmp_path, backend="opencv")
    assert meta["n_frames_written"] == 30
    assert (tmp_path / "frames.jsonl").is_file()
    assert (tmp_path / "frames_meta.json").is_file()
    assert len(list((tmp_path / "frames").glob("*.png"))) == 30

    from vscs.common.io import read_jsonl

    rows = list(read_jsonl(tmp_path / "frames.jsonl"))
    assert len(rows) == 30
    assert {"frame_index", "t_ns", "file"} == set(rows[0])
    # t_ns must be real time, not the frame index.
    assert rows[1]["t_ns"] != rows[1]["frame_index"]
    assert all(isinstance(r["t_ns"], int) for r in rows)
    assert [r["t_ns"] for r in rows] == sorted(r["t_ns"] for r in rows)
    assert (tmp_path / rows[0]["file"]).is_file()


def test_extract_frames_stride_keeps_real_timestamps(video, tmp_path):
    meta = FR.extract_frames(video, tmp_path, backend="opencv", stride=5)
    assert meta["n_frames_written"] == 6

    from vscs.common.io import read_jsonl

    rows = list(read_jsonl(tmp_path / "frames.jsonl"))
    assert [r["frame_index"] for r in rows] == [0, 5, 10, 15, 20, 25]
    # Subsampling must widen the real interval, not renumber time.
    gaps = np.diff([r["t_ns"] for r in rows])
    assert np.all(gaps > 4 * NS_PER_S / FPS)


def test_extract_frames_max_frames(video, tmp_path):
    meta = FR.extract_frames(video, tmp_path, backend="opencv", max_frames=4)
    assert meta["n_frames_written"] == 4


def test_extract_frames_jpg(video, tmp_path):
    FR.extract_frames(video, tmp_path, backend="opencv", image_format="jpg", max_frames=3)
    assert len(list((tmp_path / "frames").glob("*.jpg"))) == 3


def test_extract_frames_records_timing_and_backend(video, tmp_path):
    meta = FR.extract_frames(video, tmp_path, backend="opencv")
    assert meta["backend"] == "opencv"
    assert meta["timing"]["is_vfr"] is False
    assert meta["timing"]["mean_fps"] == pytest.approx(FPS, rel=1e-3)
    assert meta["image_size"] == list(SIZE)


def test_extract_frames_rejects_bad_arguments(video, tmp_path):
    with pytest.raises(ValueError, match="stride must be"):
        FR.extract_frames(video, tmp_path, stride=0)
    with pytest.raises(ValueError, match="unsupported image_format"):
        FR.extract_frames(video, tmp_path, image_format="tiff")
