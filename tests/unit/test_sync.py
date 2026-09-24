"""Video/IMU time-offset estimation tests (P1-T4).

A known lag is injected into a synthetic signal pair and the estimator has to recover it,
including its sign. The sign is the part worth pinning: getting it backwards gives a
plausible-looking answer that is wrong by twice the offset, and nothing downstream would
notice until ego-motion drifted (P4-T2).
"""

from __future__ import annotations

import numpy as np
import pytest

from vscs.capture import sync as S
from vscs.common.types import NS_PER_S


def _shake_signal(t_s: np.ndarray, *, spikes=(0.5, 4.5), width=0.05) -> np.ndarray:
    """A quiet trace with sharp spikes, like a deliberate shake at start and end."""
    out = 0.02 * np.sin(2 * np.pi * 1.3 * t_s)
    for centre in spikes:
        out += np.exp(-0.5 * ((t_s - centre) / width) ** 2)
    return out


def _streams(true_offset_s: float, *, video_hz=30.0, imu_hz=200.0, duration=6.0, seed=0):
    """Build a video-rate and an IMU-rate view of the same motion.

    The IMU clock is *behind* the video clock by ``true_offset_s``, so that
    ``t_video = t_imu + true_offset_s`` is the relation to recover.
    """
    rng = np.random.default_rng(seed)

    t_vid_s = np.arange(0.0, duration, 1.0 / video_hz)
    vid = _shake_signal(t_vid_s) + 0.01 * rng.normal(size=t_vid_s.size)

    t_imu_s = np.arange(0.0, duration, 1.0 / imu_hz)
    # An IMU sample stamped t sees the motion that happened at t + offset in video time.
    imu = _shake_signal(t_imu_s + true_offset_s) + 0.01 * rng.normal(size=t_imu_s.size)

    return (
        (t_vid_s * NS_PER_S).astype(np.int64),
        vid,
        (t_imu_s * NS_PER_S).astype(np.int64),
        imu,
    )


# --------------------------------------------------------------------------- #
# Offset recovery                                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("true_offset_s", [0.0, 0.25, -0.25, 0.4, -0.13])
def test_recovers_a_known_offset_with_the_right_sign(true_offset_s):
    t_v, v, t_i, i = _streams(true_offset_s)
    res = S.estimate_time_offset(t_v, v, t_i, i)
    # 10 ms tolerance: two resample steps at 200 Hz.
    assert res.offset_s == pytest.approx(true_offset_s, abs=0.01)
    assert res.trustworthy


def test_sign_convention_is_add_offset_to_imu():
    """t_video ~= t_imu + offset_ns. Documented, and asserted here.

    Uses a *single* spike so that argmax identifies the same physical event in both
    streams; the two-spike trace used elsewhere makes argmax ambiguous.
    """
    true_offset_s = 0.3
    t_vid_s = np.arange(0.0, 6.0, 1.0 / 30.0)
    t_imu_s = np.arange(0.0, 6.0, 1.0 / 200.0)
    vid = _shake_signal(t_vid_s, spikes=(2.0,))
    imu = _shake_signal(t_imu_s + true_offset_s, spikes=(2.0,))
    t_v = (t_vid_s * NS_PER_S).astype(np.int64)
    t_i = (t_imu_s * NS_PER_S).astype(np.int64)

    res = S.estimate_time_offset(t_v, vid, t_i, imu)
    assert res.offset_s == pytest.approx(true_offset_s, abs=0.01)

    # Adding the offset to the IMU clock must align the spikes in absolute time.
    corrected_t_i = t_i + res.offset_ns
    peak_video_t = t_v[int(np.argmax(vid))]
    peak_imu_t = corrected_t_i[int(np.argmax(imu))]
    assert abs(peak_video_t - peak_imu_t) < 0.02 * NS_PER_S


def test_zero_offset_gives_near_zero():
    t_v, v, t_i, i = _streams(0.0)
    assert abs(S.estimate_time_offset(t_v, v, t_i, i).offset_s) < 0.01


def test_correlation_is_high_for_matching_signals():
    t_v, v, t_i, i = _streams(0.1)
    assert S.estimate_time_offset(t_v, v, t_i, i).correlation > 0.5


def test_uncorrelated_signals_are_flagged_untrustworthy():
    """Random noise against a real shake must not be reported as a confident sync."""
    rng = np.random.default_rng(3)
    t_v, v, t_i, _ = _streams(0.0)
    noise = rng.normal(size=t_i.size)
    res = S.estimate_time_offset(t_v, v, t_i, noise)
    assert not res.trustworthy


def test_different_sample_rates_are_handled():
    t_v, v, t_i, i = _streams(0.2, video_hz=24.0, imu_hz=104.0)
    assert S.estimate_time_offset(t_v, v, t_i, i).offset_s == pytest.approx(0.2, abs=0.015)


def test_subsample_refinement_beats_the_grid_step():
    """An offset deliberately between grid samples should still land close.

    At 50 Hz the grid step is 20 ms, so a 0.131 s offset is not on a sample boundary;
    parabolic refinement is what keeps the error well under a whole step.
    """
    t_v, v, t_i, i = _streams(0.131)
    res = S.estimate_time_offset(t_v, v, t_i, i, resample_hz=50.0)
    assert res.offset_s == pytest.approx(0.131, abs=0.015)


def test_max_offset_bounds_the_search():
    t_v, v, t_i, i = _streams(0.5)
    res = S.estimate_time_offset(t_v, v, t_i, i, max_offset_s=0.1)
    assert abs(res.offset_s) <= 0.1 + 1e-9


def test_result_summary_and_properties():
    t_v, v, t_i, i = _streams(0.25)
    res = S.estimate_time_offset(t_v, v, t_i, i)
    assert "offset" in res.summary()
    assert res.offset_ns == round(res.offset_s * NS_PER_S)
    assert res.n_samples > 0


# --------------------------------------------------------------------------- #
# Input validation                                                             #
# --------------------------------------------------------------------------- #
def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="timestamps but"):
        S.estimate_time_offset([0, 1_000_000], [1.0], [0, 1_000_000], [1.0, 2.0])


def test_rejects_too_few_samples():
    with pytest.raises(ValueError, match="at least 2"):
        S.estimate_time_offset([0], [1.0], [0, 1000], [1.0, 2.0])


def test_rejects_non_monotonic_timestamps():
    with pytest.raises(ValueError, match="strictly increasing"):
        S.estimate_time_offset([0, 2000, 1000], [1.0, 2.0, 3.0], [0, 1000, 2000], [1.0, 2.0, 3.0])


def test_rejects_non_overlapping_streams():
    a_t = np.arange(0, 100) * 10_000_000
    b_t = a_t + 10 * NS_PER_S
    with pytest.raises(ValueError, match="do not overlap"):
        S.estimate_time_offset(a_t, np.sin(a_t / 1e9), b_t, np.sin(b_t / 1e9))


def test_rejects_a_flat_signal_with_a_useful_message():
    """A flat trace means no shake was recorded; the message should say so."""
    t_v, v, t_i, _ = _streams(0.0)
    with pytest.raises(ValueError, match="shake or clap"):
        S.estimate_time_offset(t_v, v, t_i, np.zeros(t_i.size))


def test_rejects_too_short_an_overlap():
    a_t = np.array([0, 1_000_000, 2_000_000])
    with pytest.raises(ValueError, match="too short"):
        S.estimate_time_offset(a_t, [0.0, 1.0, 0.0], a_t, [0.0, 1.0, 0.0], resample_hz=10.0)


# --------------------------------------------------------------------------- #
# Signal builders                                                              #
# --------------------------------------------------------------------------- #
def test_angular_rate_magnitude_is_orientation_independent():
    """Magnitude must not depend on which axis the phone was rotated about."""
    about_x = S.angular_rate_magnitude([[2.0, 0.0, 0.0]])
    about_z = S.angular_rate_magnitude([[0.0, 0.0, 2.0]])
    assert about_x[0] == pytest.approx(about_z[0]) == pytest.approx(2.0)
    assert S.angular_rate_magnitude([[3.0, 4.0, 0.0]])[0] == pytest.approx(5.0)


def test_angular_rate_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        S.angular_rate_magnitude([[1.0, 2.0]])


def test_frame_motion_signal_responds_to_motion(tmp_path):
    """A clip that is still, then moves, must show that in the signal."""
    import cv2

    path = tmp_path / "motion.mp4"
    size = (160, 120)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    assert writer.isOpened()
    for i in range(30):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        # Still for the first half, then a bar sweeps across.
        x = 0 if i < 15 else ((i - 15) * 20) % (size[0] - 10)
        frame[:, x : x + 10] = 255
        writer.write(frame)
    writer.release()

    signal, n = S.frame_motion_signal(path)
    assert n == 30 and signal.size == 30
    assert signal[0] == 0.0  # no preceding frame to difference
    assert signal[2:14].mean() < signal[17:29].mean()


def test_frame_motion_signal_rejects_unopenable(tmp_path):
    bad = tmp_path / "x.mp4"
    bad.write_bytes(b"nope")
    with pytest.raises(RuntimeError):
        S.frame_motion_signal(bad)
