"""Pre-flight validator tests.

The validator exists to answer P1-T1's two open questions mechanically, so the tests are
about whether it actually distinguishes a good capture from a bad one — not merely whether
it runs.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from fixtures import boards
from vscs.capture import preflight as P
from vscs.common.types import NS_PER_S

FPS = 30.0
SIZE = (160, 120)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """A short clip whose image motion has a clear spike part-way through."""
    path = tmp_path_factory.mktemp("pf") / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    assert writer.isOpened()
    for i in range(60):
        frame = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
        # Still, then a burst of large motion, then still again.
        x = 5 if i < 25 or i > 35 else (5 + (i - 25) * 14) % (SIZE[0] - 12)
        frame[:, x : x + 12] = 255
        writer.write(frame)
    writer.release()
    return path


def _write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join(str(c) for c in r) + "\n")
    return path


def _gyro_rows(n=1200, rate=200.0, spike_at_s=1.0, t0=0.0, scale=1.0):
    """Gyro samples that are quiet except for a spike, in whatever time unit `scale` is."""
    rows = []
    for i in range(n):
        t_s = i / rate
        mag = 4.0 if abs(t_s - spike_at_s) < 0.05 else 0.02
        rows.append([(t0 + t_s) * scale, mag, mag * 0.3, mag * 0.1])
    return rows


# --------------------------------------------------------------------------- #
# IMU loading: formats and units                                               #
# --------------------------------------------------------------------------- #
def test_loads_wide_format_in_seconds(tmp_path):
    p = _write_csv(tmp_path / "g.csv", ["time", "x", "y", "z"], _gyro_rows())
    t_ns, xyz, meta = P.load_imu_csv(p)
    assert t_ns.size == 1200 and xyz.shape == (1200, 3)
    assert meta["time_unit"] == "s"
    assert meta["rate_hz"] == pytest.approx(200.0, rel=0.01)


def test_loads_nanosecond_epoch_clock(tmp_path):
    epoch = 1_700_000_000.0
    rows = _gyro_rows(t0=epoch, scale=1e9)
    p = _write_csv(tmp_path / "g.csv", ["time", "x", "y", "z"], [[int(r[0]), *r[1:]] for r in rows])
    t_ns, _, meta = P.load_imu_csv(p)
    assert meta["time_unit"] == "ns"
    assert meta["absolute_clock"] is True
    assert meta["rate_hz"] == pytest.approx(200.0, rel=0.01)
    assert t_ns[0] == int(epoch * 1e9)


def test_loads_millisecond_clock(tmp_path):
    p = _write_csv(tmp_path / "g.csv", ["time", "x", "y", "z"], _gyro_rows(scale=1e3))
    _, _, meta = P.load_imu_csv(p)
    assert meta["time_unit"] == "ms"
    assert meta["rate_hz"] == pytest.approx(200.0, rel=0.01)


@pytest.mark.parametrize(
    "header",
    [
        ["seconds_elapsed", "x", "y", "z"],
        ["timestamp", "wx", "wy", "wz"],
        ["time", "gyroX", "gyroY", "gyroZ"],
        ["time", "rotationRateX", "rotationRateY", "rotationRateZ"],
    ],
)
def test_tolerates_column_naming_variants(tmp_path, header):
    p = _write_csv(tmp_path / "g.csv", header, _gyro_rows(n=400))
    t_ns, xyz, _ = P.load_imu_csv(p)
    assert t_ns.size == 400 and xyz.shape == (400, 3)


def test_long_format_with_sensor_column_FIRST(tmp_path):
    """Regression: a sensor column at index 0 is falsy.

    The original implementation used `cols.get("sensor") or ...`, so a sensor column at
    index 0 - exactly where loggers tend to put it - silently skipped the filter and mixed
    accelerometer rows in with the gyro.
    """
    rows = []
    for t, x, y, z in _gyro_rows(n=400):
        rows.append(["Gyroscope", t, x, y, z])
        rows.append(["Accelerometer", t, 9.81, 0.0, 0.0])
    p = _write_csv(tmp_path / "g.csv", ["sensor", "time", "x", "y", "z"], rows)

    _, xyz, meta = P.load_imu_csv(p, sensor="gyro")
    assert meta["n_samples"] == 400, "accelerometer rows leaked into the gyro stream"
    assert float(np.abs(xyz[:, 0]).max()) < 9.0, "9.81 present => accelerometer rows leaked"


def test_long_format_unknown_sensor_lists_what_exists(tmp_path):
    rows = [["Gyroscope", t, x, y, z] for t, x, y, z in _gyro_rows(n=50)]
    p = _write_csv(tmp_path / "g.csv", ["sensor", "time", "x", "y", "z"], rows)
    with pytest.raises(ValueError, match=r"Magnetometer|no rows for sensor"):
        P.load_imu_csv(p, sensor="magnetometer")


def test_rows_are_sorted_by_time(tmp_path):
    rows = _gyro_rows(n=200)
    shuffled = [rows[i] for i in np.random.default_rng(0).permutation(len(rows))]
    p = _write_csv(tmp_path / "g.csv", ["time", "x", "y", "z"], shuffled)
    t_ns, _, _ = P.load_imu_csv(p)
    assert list(t_ns) == sorted(t_ns)


def test_missing_time_column_says_what_it_looked_for(tmp_path):
    p = _write_csv(tmp_path / "g.csv", ["a", "x", "y", "z"], [[1, 2, 3, 4], [2, 3, 4, 5]])
    with pytest.raises(ValueError, match="no time column"):
        P.load_imu_csv(p)


def test_missing_axes_says_what_it_looked_for(tmp_path):
    p = _write_csv(tmp_path / "g.csv", ["time", "a", "b"], [[1, 2, 3], [2, 3, 4]])
    with pytest.raises(ValueError, match="no x/y/z gyro columns"):
        P.load_imu_csv(p)


def test_empty_and_headerless_files(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="is empty"):
        P.load_imu_csv(empty)

    header_only = _write_csv(tmp_path / "h.csv", ["time", "x", "y", "z"], [])
    with pytest.raises(ValueError, match="no data rows"):
        P.load_imu_csv(header_only)


def test_implausible_rate_is_reported_not_guessed(tmp_path):
    """A span that is implausible under every unit must be refused, not guessed at.

    1e12 raw units between two samples reads as 1e12 s, 1e9 s, 1e6 s or 1000 s depending
    on the unit, and every one of those implies a sample rate far below 1 Hz.
    """
    p = _write_csv(tmp_path / "g.csv", ["time", "x", "y", "z"], [[0, 1, 1, 1], [1e12, 1, 1, 1]])
    with pytest.raises(ValueError, match=r"time units|plausible sample rate"):
        P.load_imu_csv(p)


# --------------------------------------------------------------------------- #
# Individual checks                                                            #
# --------------------------------------------------------------------------- #
def test_video_timing_check_passes_on_a_real_clip(clip):
    check, payload = P.check_video_timing(clip, backend="opencv")
    assert check.status in ("OK", "WARN")
    assert payload is not None
    assert "frames" in check.detail


def test_video_timing_check_fails_gracefully_on_garbage(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    check, payload = P.check_video_timing(bad)
    assert check.status == "FAIL"
    assert payload is None
    assert check.advice, "a failure must come with advice"


def test_imu_check_warns_on_a_low_rate(tmp_path):
    p = _write_csv(tmp_path / "g.csv", ["time", "x", "y", "z"], _gyro_rows(n=100, rate=10.0))
    check, _ = P.check_imu(p)
    assert check.status == "WARN"
    assert "Hz" in check.advice


def test_imu_check_fails_when_the_sensor_recorded_nothing(tmp_path):
    rows = [[i / 200.0, 0.0, 0.0, 0.0] for i in range(400)]
    p = _write_csv(tmp_path / "g.csv", ["time", "x", "y", "z"], rows)
    check, _ = P.check_imu(p)
    assert check.status == "FAIL"
    assert "not actually recording" in check.advice


def test_imu_check_fails_on_an_unreadable_log(tmp_path):
    p = tmp_path / "g.csv"
    p.write_text("nonsense\n", encoding="utf-8")
    check, payload = P.check_imu(p)
    assert check.status == "FAIL" and payload is None


# --------------------------------------------------------------------------- #
# Stabilisation / frame warp                                                   #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def calibration(tmp_path_factory):
    """Calibrate from still synthetic boards, as P0-T7 does."""
    from vscs.capture.calib import calibrate_intrinsics

    d = tmp_path_factory.mktemp("stills")
    result = calibrate_intrinsics(
        boards.synth_still_images(d, n=18),
        pattern_size=boards.PATTERN,
        square_size_m=boards.SQUARE_SIZE_M,
        min_images=10,
    )
    assert result.mean_reprojection_error_px < 0.5
    return result


def test_rigid_camera_motion_does_not_trip_the_warp_check(tmp_path, calibration):
    """A moving but unwarped camera must pass: pose absorbs the motion, error stays low."""
    video = boards.synth_board_video(tmp_path / "clean.mp4", n_frames=30, rolling_shutter_px=0.0)
    check = P.check_frame_warp(
        video,
        calibration.K,
        calibration.dist,
        pattern_size=boards.PATTERN,
        square_size_m=boards.SQUARE_SIZE_M,
        static_error_px=calibration.mean_reprojection_error_px,
    )
    assert check.status in ("OK", "SKIP")
    if check.status == "SKIP":
        pytest.skip(f"board not detected reliably in the synthetic clip: {check.detail}")


def test_rolling_shutter_warping_is_detected(tmp_path, calibration):
    """Non-rigid per-frame warping cannot be absorbed by a pose, so the error must rise."""
    video = boards.synth_board_video(tmp_path / "warped.mp4", n_frames=30, rolling_shutter_px=45.0)
    check = P.check_frame_warp(
        video,
        calibration.K,
        calibration.dist,
        pattern_size=boards.PATTERN,
        square_size_m=boards.SQUARE_SIZE_M,
        static_error_px=calibration.mean_reprojection_error_px,
    )
    if check.status == "SKIP":
        pytest.skip(f"board not detected in the warped clip: {check.detail}")
    assert check.status == "WARN"
    assert "stabilisation" in check.advice.lower() or "rolling shutter" in check.advice.lower()


def test_warp_check_skips_when_the_board_is_never_visible(tmp_path, calibration, clip):
    check = P.check_frame_warp(
        clip,
        calibration.K,
        calibration.dist,
        pattern_size=boards.PATTERN,
        square_size_m=boards.SQUARE_SIZE_M,
        static_error_px=calibration.mean_reprojection_error_px,
    )
    assert check.status == "SKIP"
    assert check.advice


# --------------------------------------------------------------------------- #
# Report and driver                                                            #
# --------------------------------------------------------------------------- #
def test_report_verdicts():
    r = P.PreflightReport()
    r.add(P.Check("a", "OK", "fine"))
    assert r.ok and "GO" in r.render()

    r.add(P.Check("b", "WARN", "hmm", "look at this"))
    assert r.ok and "CHECK" in r.render() and "look at this" in r.render()

    r.add(P.Check("c", "FAIL", "broken", "fix it"))
    assert not r.ok and "NO-GO" in r.render() and "fix it" in r.render()


def test_skipped_checks_do_not_count_toward_the_score():
    r = P.PreflightReport()
    r.add(P.Check("a", "OK", "fine"))
    r.add(P.Check("b", "SKIP", "not given"))
    assert "1/1" in r.render()


def test_run_preflight_without_an_imu_log_skips_rather_than_fails(clip):
    report = P.run_preflight(clip, None, backend="opencv")
    by_name = {c.name: c for c in report.checks}
    assert by_name["imu log"].status == "SKIP"
    assert by_name["video/imu sync"].status == "SKIP"
    assert by_name["stabilisation"].status == "SKIP"
    assert report.ok, "a missing optional input is not a failure"


def test_run_preflight_end_to_end_recovers_the_sync(clip, tmp_path):
    """The whole point: video + IMU in, a usable offset out."""
    from vscs.capture.frames import probe_timestamps

    video_t = probe_timestamps(clip, backend="opencv").t_ns
    duration_s = (video_t[-1] - video_t[0]) / NS_PER_S

    # IMU on the same clock, spiking when the clip's motion burst happens (~frame 25-35).
    rate = 200.0
    n = int(duration_s * rate)
    spike_s = 25 / FPS
    rows = []
    for i in range(n):
        t_s = i / rate
        mag = 4.0 if abs(t_s - spike_s) < 0.12 else 0.02
        rows.append([t_s, mag, mag * 0.3, mag * 0.1])
    imu = _write_csv(tmp_path / "gyro.csv", ["time", "x", "y", "z"], rows)

    report = P.run_preflight(clip, imu, backend="opencv")
    by_name = {c.name: c for c in report.checks}
    assert by_name["imu log"].status == "OK"
    assert by_name["video/imu sync"].status in ("OK", "WARN")
    assert "offset" in by_name["video/imu sync"].detail
    assert report.ok


def test_run_preflight_reports_a_bad_imu_log_without_crashing(clip, tmp_path):
    bad = tmp_path / "g.csv"
    bad.write_text("garbage\n1,2\n", encoding="utf-8")
    report = P.run_preflight(clip, bad, backend="opencv")
    by_name = {c.name: c for c in report.checks}
    assert by_name["imu log"].status == "FAIL"
    assert by_name["video/imu sync"].status == "SKIP"
    assert not report.ok
    assert "NO-GO" in report.render()
