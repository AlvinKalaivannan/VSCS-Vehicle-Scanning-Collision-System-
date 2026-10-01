"""Gyro + ground visual odometry: the "-inertial" half of P4-T2's visual-inertial primary.

The phone's gyroscope measures rotation directly, never runs out of texture, and keeps
working when visual odometry loses the road. But it reports rotation about the *phone's*
axes, and how the phone is mounted is not measured precisely. So the van's vertical axis,
expressed in gyro coordinates, is **estimated from the drive itself**:

While the van turns at yaw rate ``r``, the gyro reads ``w ~ r a`` (plus noise and bias),
where ``a`` is the van's up axis in phone coordinates. Over frames where VO is confident
and the van is genuinely turning (``|r| >= min_turn_rate_radps``), the least-squares
``a = sum(r_i w_i) / sum(r_i^2)`` is normalised to a unit vector (a 1-parameter linear
regression per axis, MAT188 normal equations). The sign is fixed by the regression itself.
Before ``min_calibration_frames`` such frames exist, the gyro is not used.

Fusion per frame, once calibrated:

* **VO good:** yaw increment ``w * yaw_vo + (1 - w) * yaw_gyro`` (``vo_yaw_weight``), and VO's
  translation.
* **VO lost:** the gyro's yaw increment, and the last good *speed* carried along the new
  heading. That beats plain coasting (repeat the last motion) through any turn.

Inputs are time-aligned already: gyro samples on the video clock
(``capture.sync.estimate_video_imu_offset``). Yaw increments come from integrating
``a . w`` over each frame interval (trapezoid rule).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from vscs.common.frames import T_from_Rt, rot_z

FloatArray = npt.NDArray[np.float64]
NS_PER_S = 1_000_000_000


def estimate_up_axis(gyro: npt.ArrayLike, yaw_rates: npt.ArrayLike) -> FloatArray:
    """Unit vector ``a`` with ``gyro_i ~ yaw_rate_i * a`` (least squares)."""
    w = np.asarray(gyro, dtype=np.float64).reshape(-1, 3)
    r = np.asarray(yaw_rates, dtype=np.float64).reshape(-1)
    denom = float(r @ r)
    if denom <= 0:
        raise ValueError("no turning: the axis cannot be estimated")
    a = (r @ w) / denom
    return a / np.linalg.norm(a)


def gyro_yaw_increment(
    t_ns: npt.ArrayLike, gyro: npt.ArrayLike, axis: npt.ArrayLike, t0_ns: int, t1_ns: int
) -> float:
    """Integral of ``axis . w`` over ``[t0, t1]`` (trapezoid on the samples inside, with the
    end points linearly interpolated)."""
    t = np.asarray(t_ns, dtype=np.float64)
    rate = np.asarray(gyro, dtype=np.float64).reshape(-1, 3) @ np.asarray(axis, dtype=np.float64)
    inside = (t > t0_ns) & (t < t1_ns)
    ts = np.concatenate([[t0_ns], t[inside], [t1_ns]])
    rs = np.concatenate([[np.interp(t0_ns, t, rate)], rate[inside], [np.interp(t1_ns, t, rate)]])
    return float(np.trapezoid(rs, ts / NS_PER_S))


def _yaw(T: FloatArray) -> float:
    return float(np.arctan2(T[1, 0], T[0, 0]))


class GyroFusion:
    """Fuses per-frame VO steps (``T_prev_curr`` or ``None``) with gyro samples."""

    def __init__(self, vio_cfg: dict[str, Any], t_ns: npt.ArrayLike, gyro: npt.ArrayLike) -> None:
        self.cfg = vio_cfg
        self.t = np.asarray(t_ns, dtype=np.float64)
        self.gyro = np.asarray(gyro, dtype=np.float64).reshape(-1, 3)
        self.axis: FloatArray | None = None
        self._cal_w: list[FloatArray] = []
        self._cal_r: list[float] = []
        self._last_speed = np.zeros(2)  # last good translation per second, in the previous frame
        self.T_world_veh = np.eye(4)

    def _mean_gyro(self, t0: int, t1: int) -> FloatArray:
        inside = (self.t >= t0) & (self.t <= t1)
        return self.gyro[inside].mean(axis=0) if inside.any() else np.zeros(3)

    def step(self, T_vo: FloatArray | None, t0_ns: int, t1_ns: int) -> FloatArray:
        """Fold in one frame interval; returns the fused ``T_prev_curr``."""
        dt = (t1_ns - t0_ns) / NS_PER_S
        if T_vo is not None and dt > 0:
            r = _yaw(T_vo) / dt
            if abs(r) >= float(self.cfg["min_turn_rate_radps"]):
                self._cal_w.append(self._mean_gyro(t0_ns, t1_ns))
                self._cal_r.append(r)
                if len(self._cal_r) >= int(self.cfg["min_calibration_frames"]):
                    self.axis = estimate_up_axis(self._cal_w, self._cal_r)
        gyro_yaw = (
            gyro_yaw_increment(self.t, self.gyro, self.axis, t0_ns, t1_ns)
            if self.axis is not None
            else None
        )
        if T_vo is not None:
            yaw = _yaw(T_vo)
            if gyro_yaw is not None:
                w = float(self.cfg["vo_yaw_weight"])
                yaw = w * yaw + (1 - w) * gyro_yaw
            self._last_speed = T_vo[:2, 3] / dt if dt > 0 else self._last_speed
            T = T_from_Rt(rot_z(yaw), [T_vo[0, 3], T_vo[1, 3], 0.0])
        else:
            yaw = gyro_yaw if gyro_yaw is not None else 0.0
            T = T_from_Rt(rot_z(yaw), [*(self._last_speed * dt), 0.0])
        self.T_world_veh = self.T_world_veh @ T
        return T
