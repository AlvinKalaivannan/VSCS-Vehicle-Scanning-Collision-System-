"""Video/IMU time-offset estimation (P1-T4).

The phone's video and its IMU log are two clocks. If they are recorded by separate apps
they are definitely two clocks, and even within one app the offset is rarely zero. Every
visual-inertial estimate at P4-T2 depends on knowing that offset, so it is estimated here
and **logged per raw file**.

Method
------
Both streams are reduced to a single 1-D "how much is it moving right now" signal:

* IMU: the magnitude of the angular rate vector, ``|omega|``.
* Video: the mean absolute difference between consecutive downscaled greyscale frames.

Neither is calibrated, and they are not in the same units - which is fine, because both
are normalised to zero mean and unit variance before correlating. What matters is that a
sharp shake produces a spike in *both* at the same real-world instant. That is why the
capture checklist asks for a deliberate shake or clap at the start and end of every
recording: it manufactures a sharp, unambiguous feature for this to lock on to.

The two signals are resampled onto a common uniform grid, cross-correlated, and the peak
located to sub-sample precision by fitting a parabola to the three samples around it.

Sign convention
---------------
``offset_ns`` is defined so that

    t_video ≈ t_imu + offset_ns

i.e. **add** the offset to IMU timestamps to bring them into the video clock. A positive
offset means the IMU clock is running behind the video clock. This is asserted in the
tests, because getting it backwards produces a plausible result that is wrong by twice
the offset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from vscs.common.log import get_logger
from vscs.common.types import NS_PER_S

logger = get_logger("capture.sync")

FloatArray = npt.NDArray[np.float64]

#: Below this peak correlation the estimate should not be trusted without a look.
MIN_TRUSTWORTHY_CORRELATION = 0.30


@dataclass
class SyncResult:
    """Estimated offset between two clocks."""

    offset_ns: int
    correlation: float
    resample_hz: float
    n_samples: int
    max_offset_s: float

    @property
    def offset_s(self) -> float:
        return self.offset_ns / NS_PER_S

    @property
    def trustworthy(self) -> bool:
        return self.correlation >= MIN_TRUSTWORTHY_CORRELATION

    def summary(self) -> str:
        flag = "" if self.trustworthy else "  <-- LOW CORRELATION, check by hand"
        return (
            f"offset {self.offset_s * 1000:+.1f} ms (peak correlation {self.correlation:.3f}, "
            f"{self.n_samples} samples at {self.resample_hz:.0f} Hz){flag}"
        )


# --------------------------------------------------------------------------- #
# Signal builders                                                              #
# --------------------------------------------------------------------------- #
def angular_rate_magnitude(gyro_xyz: npt.ArrayLike) -> FloatArray:
    """``|omega|`` per sample from an ``(N, 3)`` gyro array, in whatever units it came in.

    Magnitude is used rather than a single axis so the result does not depend on how the
    phone happened to be oriented in the mount.
    """
    g = np.asarray(gyro_xyz, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != 3:
        raise ValueError(f"gyro must be (N, 3), got {g.shape}")
    return np.linalg.norm(g, axis=1)


def frame_motion_signal(
    video: Path,
    *,
    downscale_to: int = 160,
    max_frames: int | None = None,
) -> tuple[FloatArray, int]:
    """Mean absolute inter-frame difference per frame, as a proxy for image motion.

    Returns ``(signal, n_frames_read)``. The signal has one entry per frame, with the
    first entry 0 since there is no preceding frame to difference against.

    Frames are downscaled hard: this is a motion *magnitude* signal, and the detail costs
    time without improving the correlation.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video}")
    signal: list[float] = []
    prev: np.ndarray | None = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = grey.shape
            if max(h, w) > downscale_to:
                scale = downscale_to / max(h, w)
                grey = cv2.resize(grey, (max(1, int(w * scale)), max(1, int(h * scale))))
            small = grey.astype(np.float32)
            signal.append(0.0 if prev is None else float(np.mean(np.abs(small - prev))))
            prev = small
            if max_frames is not None and len(signal) >= max_frames:
                break
    finally:
        cap.release()
    if len(signal) < 2:
        raise RuntimeError(f"{Path(video).name}: need at least 2 frames for a motion signal")
    return np.asarray(signal, dtype=np.float64), len(signal)


# --------------------------------------------------------------------------- #
# Cross-correlation                                                            #
# --------------------------------------------------------------------------- #
def _normalise(x: FloatArray) -> FloatArray:
    """Zero mean, unit standard deviation. Flat signals become zeros."""
    x = np.asarray(x, dtype=np.float64)
    std = float(x.std())
    if std < 1e-12:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def _resample(t_ns: FloatArray, values: FloatArray, grid_ns: FloatArray) -> FloatArray:
    """Linear interpolation onto ``grid_ns``, holding the end values outside the range."""
    return np.interp(grid_ns, t_ns, values)


def estimate_time_offset(
    t_a_ns: npt.ArrayLike,
    signal_a: npt.ArrayLike,
    t_b_ns: npt.ArrayLike,
    signal_b: npt.ArrayLike,
    *,
    max_offset_s: float = 2.0,
    resample_hz: float = 200.0,
) -> SyncResult:
    """Estimate the offset that brings stream ``b`` onto stream ``a``'s clock.

    With ``a`` = video and ``b`` = IMU, the result satisfies
    ``t_video ≈ t_imu + offset_ns``.

    Both streams are resampled onto a common grid over their overlapping interval, so the
    two may have completely different sample rates.
    """
    ta = np.asarray(t_a_ns, dtype=np.float64)
    tb = np.asarray(t_b_ns, dtype=np.float64)
    sa = np.asarray(signal_a, dtype=np.float64)
    sb = np.asarray(signal_b, dtype=np.float64)
    for name, t, s in (("a", ta, sa), ("b", tb, sb)):
        if t.shape != s.shape:
            raise ValueError(f"stream {name}: {t.shape} timestamps but {s.shape} samples")
        if t.size < 2:
            raise ValueError(f"stream {name}: need at least 2 samples, got {t.size}")
        if np.any(np.diff(t) <= 0):
            raise ValueError(f"stream {name}: timestamps must be strictly increasing")

    start = max(ta[0], tb[0])
    stop = min(ta[-1], tb[-1])
    if stop - start <= 0:
        raise ValueError(
            "the two streams do not overlap in time; check that both clocks are in "
            "nanoseconds and refer to the same recording"
        )

    step_ns = NS_PER_S / float(resample_hz)
    grid = np.arange(start, stop, step_ns)
    if grid.size < 4:
        raise ValueError(
            f"overlap of {(stop - start) / NS_PER_S:.3f} s is too short at {resample_hz} Hz"
        )

    ra = _normalise(_resample(ta, sa, grid))
    rb = _normalise(_resample(tb, sb, grid))
    if not ra.any() or not rb.any():
        raise ValueError(
            "one of the signals is flat, so there is nothing to correlate. A deliberate "
            "shake or clap at the start and end of the recording is what gives this "
            "something to lock on to (see docs/capture_checklists.md)."
        )

    # Full cross-correlation, normalised so the peak is a correlation coefficient.
    corr = np.correlate(ra, rb, mode="full") / grid.size
    lags = np.arange(-(grid.size - 1), grid.size)

    max_lag = round(max_offset_s * resample_hz)
    keep = np.abs(lags) <= max_lag
    corr_w = corr[keep]
    lags_w = lags[keep]
    if corr_w.size == 0:
        raise ValueError("max_offset_s is too small for the resample rate")

    peak = int(np.argmax(corr_w))
    lag = float(lags_w[peak])

    # Parabolic refinement through the three samples around the peak, for sub-sample
    # precision. At 200 Hz a whole sample is 5 ms, which is worth refining.
    if 0 < peak < corr_w.size - 1:
        y0, y1, y2 = corr_w[peak - 1], corr_w[peak], corr_w[peak + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            lag += float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))

    # A positive lag means signal b must be shifted later to line up with a.
    offset_ns = round(lag * step_ns)
    result = SyncResult(
        offset_ns=offset_ns,
        correlation=float(corr_w[peak]),
        resample_hz=float(resample_hz),
        n_samples=int(grid.size),
        max_offset_s=float(max_offset_s),
    )
    if not result.trustworthy:
        logger.warning("video/IMU sync is weak: %s", result.summary())
    else:
        logger.info("video/IMU sync: %s", result.summary())
    return result


def estimate_video_imu_offset(
    video: Path,
    video_t_ns: npt.ArrayLike,
    imu_t_ns: npt.ArrayLike,
    gyro_xyz: npt.ArrayLike,
    *,
    max_offset_s: float = 2.0,
    resample_hz: float = 200.0,
) -> SyncResult:
    """Convenience wrapper: build both signals from a video and a gyro log, then correlate.

    ``video_t_ns`` must be the container timestamps from
    :func:`vscs.capture.frames.probe_timestamps`, not frame indices.
    """
    motion, n_read = frame_motion_signal(video)
    vt = np.asarray(video_t_ns, dtype=np.float64)
    if n_read != vt.size:
        # Decoders occasionally yield a different count than the container advertises.
        n = min(n_read, vt.size)
        logger.warning(
            "%s: %d decoded frames vs %d timestamps; using the first %d",
            Path(video).name,
            n_read,
            vt.size,
            n,
        )
        motion, vt = motion[:n], vt[:n]
    return estimate_time_offset(
        vt,
        motion,
        imu_t_ns,
        angular_rate_magnitude(gyro_xyz),
        max_offset_s=max_offset_s,
        resample_hz=resample_hz,
    )
