"""Pre-flight validation of a test capture (supports P1-T1's open questions).

Two questions block booking a capture day, and both are invisible on the day itself:

1. **Does the recording setup actually produce usable video + IMU?** A phone's stock
   camera does not log IMU at all, and a two-app setup produces two clocks. Either way
   the failure surfaces weeks later, in the data, when it is far too late.
2. **Is stabilisation warping the frames?** Electronic stabilisation warps each frame
   independently, which means intrinsics calibrated on a still board stop describing the
   images as soon as the camera moves (R-04).

This module turns both into a measurement you can run at home in half an hour, and rerun
on the morning of capture day as a go/no-go.

Nothing here trusts a claim made by an app's marketing. It reads the files that came out
of the phone and reports what is actually in them.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import numpy.typing as npt

from vscs.capture.calib import detect_corners_in_image, object_points
from vscs.capture.frames import analyse_timing, probe_timestamps
from vscs.capture.sync import estimate_video_imu_offset
from vscs.common.log import get_logger
from vscs.common.types import NS_PER_S

logger = get_logger("capture.preflight")

FloatArray = npt.NDArray[np.float64]

Status = Literal["OK", "WARN", "FAIL", "SKIP"]

#: Column names that might hold a timestamp, in preference order. Absolute clocks are
#: preferred over elapsed ones, because two files can only be aligned on a shared clock.
_TIME_COLUMNS = (
    "time",
    "timestamp",
    "time_ns",
    "t_ns",
    "host_timestamp",
    "seconds_elapsed",
    "elapsed",
    "time_s",
    "t",
)

#: Axis column groups, tried in order. Different loggers name these very differently.
_AXIS_GROUPS = (
    ("x", "y", "z"),
    ("gyrox", "gyroy", "gyroz"),
    ("gyroscopex", "gyroscopey", "gyroscopez"),
    ("wx", "wy", "wz"),
    ("rotationratex", "rotationratey", "rotationratez"),
    ("gyro_x", "gyro_y", "gyro_z"),
)

#: Plausible IMU sample rates, used to work out what units the time column is in.
_MIN_RATE_HZ, _MAX_RATE_HZ = 1.0, 2000.0
_PREFERRED_RATE_HZ = (20.0, 1000.0)


@dataclass
class Check:
    """One pre-flight check."""

    name: str
    status: Status
    detail: str
    advice: str = ""

    def line(self) -> str:
        return f"{self.name:<16}{self.status:<6}{self.detail}"


@dataclass
class PreflightReport:
    """The verdict on a test capture."""

    checks: list[Check] = field(default_factory=list)

    def add(self, check: Check) -> None:
        self.checks.append(check)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == "FAIL"]

    @property
    def warned(self) -> list[Check]:
        return [c for c in self.checks if c.status == "WARN"]

    @property
    def ok(self) -> bool:
        return not self.failed

    def render(self) -> str:
        lines = [c.line() for c in self.checks]
        considered = [c for c in self.checks if c.status != "SKIP"]
        passed = sum(1 for c in considered if c.status == "OK")
        verdict = "GO" if self.ok and not self.warned else ("CHECK" if self.ok else "NO-GO")
        lines.append("")
        lines.append(f"VERDICT: {verdict} - {passed}/{len(considered)} checks clean")
        for c in self.failed + self.warned:
            if c.advice:
                lines.append(f"  - {c.name}: {c.advice}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# IMU log loading                                                              #
# --------------------------------------------------------------------------- #
def _normalise(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum() or ch == "_")


def _pick_time_scale(raw: FloatArray) -> tuple[float, str]:
    """Work out whether a time column is in s, ms, us or ns.

    Guessing from magnitude alone is unreliable: an *elapsed* column starting at zero has
    small values whatever its unit. So instead each candidate unit is tried and the one
    that implies a plausible IMU sample rate is chosen. A 30 s log at 200 Hz is only
    plausible under one interpretation.
    """
    span = float(raw[-1] - raw[0])
    n = raw.size
    if span <= 0:
        raise ValueError("IMU timestamps do not advance")

    candidates = [
        (1e9, "s"),
        (1e6, "ms"),
        (1e3, "us"),
        (1.0, "ns"),
    ]
    plausible: list[tuple[float, float, str]] = []
    for to_ns, unit in candidates:
        span_s = span * to_ns / NS_PER_S
        if span_s <= 0:
            continue
        rate = (n - 1) / span_s
        if _MIN_RATE_HZ <= rate <= _MAX_RATE_HZ:
            preferred = _PREFERRED_RATE_HZ[0] <= rate <= _PREFERRED_RATE_HZ[1]
            plausible.append((0 if preferred else 1, to_ns, unit))
    if not plausible:
        raise ValueError(
            f"could not work out the time units of the IMU log: {n} samples spanning "
            f"{span:g} raw units implies no plausible sample rate. Check the time column."
        )
    plausible.sort()
    _, to_ns, unit = plausible[0]
    return to_ns, unit


def load_imu_csv(
    path: Path,
    *,
    sensor: str = "gyro",
) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
    """Load a gyro log from CSV, tolerating the formats common loggers emit.

    Returns ``(t_ns, xyz, meta)``. Handles wide format (one row per sample with x/y/z
    columns) and long format (a ``sensor`` column naming which sensor each row is).

    Deliberately permissive about column naming and units, because the app has not been
    chosen yet and every logger names these differently. What it will *not* do is guess
    silently: if it cannot find a time column or three axes, it says so and lists what it
    did find.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path.name} is empty") from exc
        rows = [r for r in reader if r and any(c.strip() for c in r)]

    if not rows:
        raise ValueError(f"{path.name} has a header but no data rows")

    cols = {_normalise(h): i for i, h in enumerate(header)}

    # Long format: filter to the sensor of interest before anything else.
    # Explicit None checks, not `or` chaining: a sensor column at index 0 is falsy, and
    # index 0 is exactly where loggers tend to put it. `or` would silently skip the
    # filter and then average gyro rows together with accelerometer rows.
    sensor_idx: int | None = None
    for key in ("sensor", "sensortype", "name"):
        if key in cols:
            sensor_idx = cols[key]
            break
    if sensor_idx is not None:
        wanted = sensor.lower()
        filtered = [r for r in rows if len(r) > sensor_idx and wanted in r[sensor_idx].lower()]
        if not filtered:
            seen = sorted({r[sensor_idx] for r in rows if len(r) > sensor_idx})
            raise ValueError(f"{path.name}: no rows for sensor {sensor!r}. Found: {seen}")
        rows = filtered

    time_key = next((k for k in _TIME_COLUMNS if k in cols), None)
    if time_key is None:
        raise ValueError(
            f"{path.name}: no time column found. Looked for {list(_TIME_COLUMNS)}, "
            f"file has {header}"
        )
    axes = next((g for g in _AXIS_GROUPS if all(a in cols for a in g)), None)
    if axes is None:
        raise ValueError(
            f"{path.name}: no x/y/z gyro columns found. Looked for {[g[0] for g in _AXIS_GROUPS]}, "
            f"file has {header}"
        )

    t_idx = cols[time_key]
    a_idx = [cols[a] for a in axes]
    t_raw: list[float] = []
    vals: list[list[float]] = []
    skipped = 0
    for r in rows:
        try:
            t_raw.append(float(r[t_idx]))
            vals.append([float(r[i]) for i in a_idx])
        except (ValueError, IndexError):
            skipped += 1
    if len(t_raw) < 2:
        raise ValueError(f"{path.name}: fewer than 2 usable rows")

    t_arr = np.asarray(t_raw, dtype=np.float64)
    order = np.argsort(t_arr, kind="stable")
    t_arr = t_arr[order]
    xyz = np.asarray(vals, dtype=np.float64)[order]

    to_ns, unit = _pick_time_scale(t_arr)
    t_ns = (t_arr * to_ns).astype(np.int64)

    span_s = float(t_ns[-1] - t_ns[0]) / NS_PER_S
    meta = {
        "n_samples": int(t_ns.size),
        "duration_s": span_s,
        "rate_hz": (t_ns.size - 1) / span_s if span_s > 0 else 0.0,
        "time_column": time_key,
        "time_unit": unit,
        "axis_columns": list(axes),
        "absolute_clock": bool(t_ns[0] > NS_PER_S * 3600),
        "skipped_rows": skipped,
    }
    return t_ns, xyz, meta


# --------------------------------------------------------------------------- #
# Individual checks                                                            #
# --------------------------------------------------------------------------- #
def check_video_timing(video: Path, *, backend: str = "auto") -> tuple[Check, Any]:
    """Container timestamps present, monotonic, and genuinely per-frame."""
    try:
        stamps = probe_timestamps(video, backend=backend)  # type: ignore[arg-type]
        report = analyse_timing(stamps.t_ns)
    except Exception as exc:  # report the failure; never crash the preflight
        return (
            Check(
                "timestamps",
                "FAIL",
                f"{type(exc).__name__}: {exc}",
                "The video's timestamps are unusable. Frame index is never a substitute "
                "(R-03). Try a different recording app, or install ffmpeg so ffprobe can "
                "read the container directly.",
            ),
            None,
        )

    detail = (
        f"{report.n_frames} frames, {report.duration_s:.2f} s, "
        f"{report.mean_fps:.1f} fps, {'VFR' if report.is_vfr else 'CFR'} "
        f"(jitter {report.jitter_ratio:.0%}), {len(report.gap_indices)} gap(s)"
    )
    status: Status = "OK"
    advice = ""
    if stamps.backend == "opencv":
        status = "WARN"
        detail += "  [via OpenCV, not ffprobe]"
        advice = (
            "ffprobe is not installed, so these timestamps come from OpenCV, which is not "
            "authoritative for variable frame rate footage. Install ffmpeg (ADR 0003) "
            "before trusting this."
        )
    if len(report.gap_indices) > report.n_frames * 0.02:
        status = "WARN"
        advice = (
            f"{len(report.gap_indices)} dropped-frame gaps in a short clip. The phone may "
            "be struggling at this resolution - try a lower resolution or frame rate."
        )
    return Check("timestamps", status, detail, advice), (stamps, report)


def check_imu(imu_csv: Path, *, sensor: str = "gyro") -> tuple[Check, Any]:
    """The IMU log parses, has three axes, and a plausible rate."""
    try:
        t_ns, xyz, meta = load_imu_csv(imu_csv, sensor=sensor)
    except Exception as exc:
        return (
            Check(
                "imu log",
                "FAIL",
                f"{type(exc).__name__}: {exc}",
                "P1-T3 requires video + IMU. If no app on the phone can log IMU with real "
                "timestamps, that is a decision to take now: §8.2 falls back to "
                "visual-only odometry, which costs accuracy at P4-T2.",
            ),
            None,
        )

    detail = (
        f"{meta['n_samples']} samples, {meta['duration_s']:.2f} s, "
        f"{meta['rate_hz']:.1f} Hz, cols {meta['time_column']}/{'/'.join(meta['axis_columns'])} "
        f"({meta['time_unit']})"
    )
    status: Status = "OK"
    advice = ""
    if meta["rate_hz"] < 50:
        status = "WARN"
        advice = (
            f"{meta['rate_hz']:.0f} Hz is low for visual-inertial work; 100-200 Hz is "
            "typical. Check whether the app can log faster."
        )
    if float(np.abs(xyz).max()) == 0.0:
        status = "FAIL"
        advice = "All gyro samples are zero - the sensor was not actually recording."
    return Check("imu log", status, detail, advice), (t_ns, xyz, meta)


def check_sync(
    video: Path,
    video_t_ns: npt.ArrayLike,
    imu_t_ns: npt.ArrayLike,
    gyro_xyz: npt.ArrayLike,
    *,
    max_offset_s: float = 5.0,
) -> tuple[Check, Any]:
    """The video/IMU offset can actually be recovered from this recording."""
    try:
        result = estimate_video_imu_offset(
            video, video_t_ns, imu_t_ns, gyro_xyz, max_offset_s=max_offset_s
        )
    except Exception as exc:
        return (
            Check(
                "video/imu sync",
                "FAIL",
                f"{type(exc).__name__}: {exc}",
                "Could not align the two clocks. Most often this means the streams do not "
                "overlap (one clock is elapsed-from-zero and the other is wall clock), or "
                "there was no sharp motion to lock onto. Shake the phone hard at the start "
                "and end of the recording.",
            ),
            None,
        )

    detail = f"offset {result.offset_s * 1000:+.0f} ms (correlation {result.correlation:.2f})"
    status: Status = "OK" if result.trustworthy else "WARN"
    advice = (
        ""
        if result.trustworthy
        else (
            "Low correlation, so this offset is not reliable. Re-record with a deliberate "
            "sharp shake or clap at the start and end - that is what gives the correlation "
            "something unambiguous to lock onto."
        )
    )
    return Check("video/imu sync", status, detail, advice), result


def check_frame_warp(
    moving_video: Path,
    K: npt.ArrayLike,
    dist: npt.ArrayLike,
    *,
    pattern_size: tuple[int, int],
    square_size_m: float,
    static_error_px: float,
    max_frames: int = 60,
    warn_ratio: float = 2.0,
) -> Check:
    """Detect per-frame warping (electronic stabilisation) or heavy rolling shutter.

    The idea: intrinsics calibrated from *still* checkerboard images describe the lens. If
    the camera then moves and the images are being warped per frame, those intrinsics stop
    fitting - the reprojection error rises sharply. A rigid lens with no warping keeps
    roughly the same error whether the camera is still or moving.

    **This cannot separate stabilisation from rolling shutter**, and does not try to. Both
    warp a moving frame, both invalidate the pinhole model the same way, and the response
    is the same: turn off what you can, and record the rest as a known limitation.
    """
    K = np.asarray(K, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64)
    objp = object_points(pattern_size, square_size_m)

    cap = cv2.VideoCapture(str(moving_video))
    if not cap.isOpened():
        return Check("stabilisation", "FAIL", f"could not open {Path(moving_video).name}")

    errors: list[float] = []
    frames_seen = 0
    try:
        while frames_seen < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames_seen += 1
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = detect_corners_in_image(grey, pattern_size)
            if corners is None:
                continue
            found, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
            if not found:
                continue
            projected, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
            residual = projected.reshape(-1, 2) - corners.reshape(-1, 2)
            errors.append(float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))))
    finally:
        cap.release()

    if len(errors) < 5:
        return Check(
            "stabilisation",
            "SKIP",
            f"board detected in only {len(errors)} of {frames_seen} frames",
            "Re-shoot with the board filling more of the frame and staying in view "
            "throughout the walk.",
        )

    moving_error = float(np.mean(errors))
    ratio = moving_error / static_error_px if static_error_px > 0 else float("inf")
    detail = (
        f"reprojection {static_error_px:.2f} px still -> {moving_error:.2f} px moving "
        f"({ratio:.1f}x, {len(errors)} frames)"
    )
    if ratio >= warn_ratio:
        return Check(
            "stabilisation",
            "WARN",
            detail,
            "Error rises sharply once the camera moves, so frames are being warped - "
            "electronic stabilisation, rolling shutter, or both. Turn off stabilisation "
            "and 'action mode' if the app allows it and re-test. If it cannot be turned "
            "off, record it as a known limitation (R-04) and expect it to be the first "
            "suspect if reconstruction quality is poor.",
        )
    return Check("stabilisation", "OK", detail)


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def run_preflight(
    video: Path,
    imu_csv: Path | None = None,
    *,
    sensor: str = "gyro",
    backend: str = "auto",
    moving_board_video: Path | None = None,
    calibration: dict[str, Any] | None = None,
    pattern_size: tuple[int, int] | None = None,
    square_size_m: float | None = None,
) -> PreflightReport:
    """Run every applicable check over a test capture and return the report."""
    report = PreflightReport()

    timing_check, timing = check_video_timing(video, backend=backend)
    report.add(timing_check)

    imu_data = None
    if imu_csv is None:
        report.add(
            Check(
                "imu log",
                "SKIP",
                "no IMU log given",
                "P1-T3 requires video + IMU; run this again with --imu once an app is chosen.",
            )
        )
    else:
        imu_check, imu_data = check_imu(imu_csv, sensor=sensor)
        report.add(imu_check)

    if timing is not None and imu_data is not None:
        stamps, _ = timing
        imu_t_ns, gyro_xyz, _ = imu_data
        sync_check, _ = check_sync(video, stamps.t_ns, imu_t_ns, gyro_xyz)
        report.add(sync_check)
    else:
        report.add(Check("video/imu sync", "SKIP", "needs both video timestamps and an IMU log"))

    if moving_board_video is None or calibration is None:
        report.add(
            Check(
                "stabilisation",
                "SKIP",
                "no moving-checkerboard clip given",
                "Shoot 20 s of the checkerboard while walking slowly, then rerun with "
                "--moving-board. It reuses the P0-T7 calibration.",
            )
        )
    else:
        report.add(
            check_frame_warp(
                moving_board_video,
                calibration["K"],
                calibration["dist"],
                pattern_size=pattern_size or (9, 6),
                square_size_m=square_size_m or 0.025,
                static_error_px=float(calibration["mean_reprojection_error_px"]),
            )
        )
    return report
