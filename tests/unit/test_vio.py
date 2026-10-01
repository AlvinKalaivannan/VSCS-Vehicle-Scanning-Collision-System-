"""Gyro fusion: axis self-calibration, yaw integration, and a VO dropout through a turn."""

from __future__ import annotations

import numpy as np
import pytest

from vscs.common.config import load_config
from vscs.common.frames import T_from_Rt, axis_angle_to_R, rot_z
from vscs.perception.vio import GyroFusion, estimate_up_axis, gyro_yaw_increment

VIO = load_config("perception")["egomotion"]["vio"]
S = 1_000_000_000
#: An awkward, unknown phone mounting: the van's up axis seen in phone coordinates.
R_PHONE_VEH = axis_angle_to_R([0.3, -0.8, 0.5], np.deg2rad(70.0))
UP_IN_PHONE = R_PHONE_VEH @ np.array([0.0, 0.0, 1.0])


def _gyro(yaw_rate_fn, t_s, rng, noise=0.01, bias=(0.004, -0.002, 0.003)):
    """Phone gyro samples at 200 Hz for a van yawing at yaw_rate_fn(t)."""
    return np.array([yaw_rate_fn(t) * UP_IN_PHONE + rng.normal(0, noise, 3) + bias for t in t_s])


def test_the_up_axis_is_recovered_from_turns():
    rng = np.random.default_rng(3)
    r = rng.uniform(-0.6, 0.6, 400)
    w = r[:, None] * UP_IN_PHONE + rng.normal(0, 0.01, (400, 3))
    a = estimate_up_axis(w, r)
    assert np.degrees(np.arccos(np.clip(a @ UP_IN_PHONE, -1, 1))) < 1.0
    with pytest.raises(ValueError, match="no turning"):
        estimate_up_axis(w, np.zeros(400))


def test_gyro_yaw_integrates_exactly_for_a_constant_rate():
    t = np.arange(0, 1.0001, 0.005)
    g = np.array([0.4 * UP_IN_PHONE for _ in t])
    assert gyro_yaw_increment(
        (t * S).astype(np.int64), g, UP_IN_PHONE, int(0.2 * S), int(0.7 * S)
    ) == pytest.approx(0.2)


def _rate(t):
    """Turn left for 3 s, then swing through straight to a right turn over 1 s, then hold."""
    if t < 3.0:
        return 0.3
    if t < 4.0:
        return 0.3 - 0.6 * (t - 3.0)
    return -0.3


def _drive(dropout):
    """Reverse at 1 m/s at 10 fps for 6 s with yaw rate _rate(t). VO is perfect except for
    frames in ``dropout``. Returns (heading error deg, fused position error m, coasting
    position error m, estimated axis)."""
    rng = np.random.default_rng(5)
    speed, dt = -1.0, 0.1
    t_s = np.arange(0, 6.0 + 1e-9, 0.005)
    fusion = GyroFusion(VIO, (t_s * S).astype(np.int64), _gyro(_rate, t_s, rng))
    truth, coast, last = np.eye(4), np.eye(4), None
    for k in range(60):
        # Exact frame motion: integrate the yaw rate over the interval, move along the mean heading.
        sub = np.linspace(k * dt, (k + 1) * dt, 41)
        dyaw = float(np.trapezoid([_rate(u) for u in sub], sub))
        step_T = T_from_Rt(
            rot_z(dyaw), [speed * dt * np.cos(dyaw / 2), speed * dt * np.sin(dyaw / 2), 0]
        )
        truth = truth @ step_T
        vo = None if k in dropout else step_T
        fusion.step(vo, round(k * dt * S), round((k + 1) * dt * S))
        last = vo if vo is not None else last
        coast = coast @ (vo if vo is not None else (last if last is not None else np.eye(4)))

    def head(T):
        return np.degrees(np.arctan2(T[1, 0], T[0, 0]))

    return (
        abs(head(fusion.T_world_veh) - head(truth)),
        np.linalg.norm(fusion.T_world_veh[:2, 3] - truth[:2, 3]),
        np.linalg.norm(coast[:2, 3] - truth[:2, 3]),
        fusion.axis,
    )


DROPOUT = set(range(30, 40))  # 1 s with no VO, exactly while the turn reverses


def test_calibrates_then_carries_heading_through_a_changing_turn():
    head_err, pos_err, coast_err, axis = _drive(DROPOUT)
    assert axis is not None and np.degrees(np.arccos(np.clip(axis @ UP_IN_PHONE, -1, 1))) < 3.0
    assert head_err < 1.0 and pos_err < 0.10
    # Why the gyro is worth having: coasting repeats the old left turn through the reversal.
    assert coast_err > 3 * pos_err


def test_with_no_dropout_fusion_does_not_hurt():
    head_err, pos_err, _, _ = _drive(set())
    assert head_err < 0.5 and pos_err < 0.05
