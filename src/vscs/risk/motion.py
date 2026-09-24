"""Constant-curvature motion model and path fan (P3-T3).

Predicts where the vehicle will be over a short horizon, for a *fan* of candidate
steering curvatures rather than a single guess. Everything is expressed in the vehicle
frame at the current instant, ``veh0`` (x forward, y left, z up; origin on the ground
below the rear-axle centre, CLAUDE.md §4.1).

The model
---------
At manoeuvring speeds a car follows the kinematic bicycle model: with the steering held
fixed, the rear-axle centre travels along a circle of radius ``R = 1 / kappa``. The
rear axle is the natural reference point because the rear wheels do not steer, so that
point never slides sideways — which is also exactly why §4.1 puts the ``veh`` origin
there.

The circle's centre sits on the rear-axle line, at ``(0, R)`` in ``veh0``. After
travelling a signed arc length ``s`` the heading has turned by ``theta = kappa * s``,
and the position is

    x(s) = R sin(theta)          = sin(kappa s) / kappa
    y(s) = R (1 - cos(theta))    = (1 - cos(kappa s)) / kappa
    heading(s) = theta           = kappa s

Two details matter in practice:

* **Signs.** ``kappa > 0`` turns left (counter-clockwise seen from above, because ``y``
  points left). ``s = v t`` is *signed*, so reversing (``v < 0``) is not a special case:
  the same formulas run the arc backwards.
* **Straight ahead is a limit, not a special case.** As ``kappa -> 0`` the formulas are
  ``0/0``. Written as ``x = s * sinc(u)`` and ``y = s * cosc(u)`` with ``u = kappa s``,
  ``sinc(u) = sin(u)/u -> 1`` and ``cosc(u) = (1 - cos u)/u -> 0``, so the straight
  line falls out continuously instead of needing an ``if``. Near zero a short Taylor
  series is used, because evaluating ``1 - cos(u)`` for tiny ``u`` cancels
  catastrophically in floating point.

Each predicted pose is a planar rigid transform, lifted to the §4.1 4x4 form

    T_veh0_veh(t) = [ Rz(theta)  [x, y, 0]^T ]
                    [ 0          1          ]

which maps points on the vehicle *at time t* into the ``veh0`` frame. Sweeping a
component footprint along a path is then just applying these transforms to it.

Why a fan
---------
The constant-curvature assumption is good for a second or two and wrong as soon as the
driver moves the wheel. Rather than pretend to know the future steering, the risk engine
sweeps several curvatures that bracket plausible steering over the horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from vscs.common.frames import T_from_Rt, rot_z

FloatArray = npt.NDArray[np.float64]

#: Below this |kappa * s| the Taylor series is used instead of the closed form.
_SERIES_THRESHOLD = 1e-4


def _sinc(u: FloatArray) -> FloatArray:
    """``sin(u) / u``, with its limit 1 at ``u = 0``."""
    u = np.asarray(u, dtype=np.float64)
    small = np.abs(u) < _SERIES_THRESHOLD
    safe = np.where(small, 1.0, u)
    return np.where(small, 1.0 - u * u / 6.0, np.sin(safe) / safe)


def _cosc(u: FloatArray) -> FloatArray:
    """``(1 - cos(u)) / u``, with its limit 0 at ``u = 0``.

    The closed form cancels catastrophically for small ``u``, since ``cos(u)`` is then
    within rounding error of 1; the series ``u/2 - u^3/24`` does not.
    """
    u = np.asarray(u, dtype=np.float64)
    small = np.abs(u) < _SERIES_THRESHOLD
    safe = np.where(small, 1.0, u)
    return np.where(small, u / 2.0 - u**3 / 24.0, (1.0 - np.cos(safe)) / safe)


def constant_curvature_pose(
    kappa: float, s: npt.ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Pose ``(x, y, heading)`` in ``veh0`` after signed arc length ``s`` at curvature ``kappa``.

    ``s`` may be a scalar or an array; negative ``s`` means reversing.
    """
    s = np.asarray(s, dtype=np.float64)
    u = float(kappa) * s
    return s * _sinc(u), s * _cosc(u), u


@dataclass(frozen=True)
class PathFan:
    """Predicted poses for every candidate curvature over the horizon.

    ``x``, ``y`` and ``heading`` have shape ``(n_curvatures, n_steps)`` and are expressed
    in ``veh0``. Row ``k`` is the path for ``curvatures[k]``; column ``i`` is time
    ``t_s[i]``. Column 0 is always the identity pose, because at ``t = 0`` the vehicle is
    where it is.
    """

    t_s: FloatArray
    curvatures: FloatArray
    speed_mps: float
    x: FloatArray
    y: FloatArray
    heading: FloatArray

    @property
    def n_curvatures(self) -> int:
        return int(self.curvatures.size)

    @property
    def n_steps(self) -> int:
        return int(self.t_s.size)

    @property
    def t_ns(self) -> npt.NDArray[np.int64]:
        """Step times as int64 nanoseconds from now, per §4.1."""
        return np.round(self.t_s * 1e9).astype(np.int64)

    def T_veh0_veh(self, k: int, i: int) -> FloatArray:
        """4x4 transform mapping points on the vehicle at step ``i`` of path ``k`` into ``veh0``."""
        return T_from_Rt(rot_z(float(self.heading[k, i])), [self.x[k, i], self.y[k, i], 0.0])

    def transforms(self, k: int) -> FloatArray:
        """All ``T_veh0_veh`` along path ``k``, shape ``(n_steps, 4, 4)``."""
        return np.stack([self.T_veh0_veh(k, i) for i in range(self.n_steps)])

    def index_of_curvature(self, kappa: float) -> int:
        """Row index of the fan curvature closest to ``kappa``."""
        return int(np.argmin(np.abs(self.curvatures - float(kappa))))


def path_fan(
    speed_mps: float,
    *,
    horizon_s: float,
    dt_s: float,
    curvatures: npt.ArrayLike,
) -> PathFan:
    """Constant-speed, constant-curvature predictions for each curvature in the fan.

    ``speed_mps`` is signed: negative means reversing, which is the case VSCS exists for.
    Speed is held constant over the horizon; at manoeuvring speeds over two or three
    seconds that is a smaller error than the steering uncertainty the fan already covers.
    """
    if dt_s <= 0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")
    if horizon_s <= 0:
        raise ValueError(f"horizon_s must be positive, got {horizon_s}")
    kappas = np.asarray(curvatures, dtype=np.float64).reshape(-1)
    if kappas.size == 0:
        raise ValueError("the curvature fan is empty")

    n_steps = round(horizon_s / dt_s) + 1
    t = np.arange(n_steps, dtype=np.float64) * dt_s
    s = float(speed_mps) * t

    x = np.empty((kappas.size, n_steps))
    y = np.empty_like(x)
    heading = np.empty_like(x)
    for k, kappa in enumerate(kappas):
        x[k], y[k], heading[k] = constant_curvature_pose(float(kappa), s)

    return PathFan(t_s=t, curvatures=kappas, speed_mps=float(speed_mps), x=x, y=y, heading=heading)


def path_fan_from_config(speed_mps: float, risk_cfg: dict[str, Any]) -> PathFan:
    """Build the fan from ``configs/risk.yaml`` (the ``motion`` block)."""
    m = risk_cfg["motion"]
    return path_fan(
        speed_mps,
        horizon_s=float(m["horizon_s"]),
        dt_s=float(m["dt_s"]),
        curvatures=m["curvatures_inv_m"],
    )


def is_moving(speed_mps: float, risk_cfg: dict[str, Any]) -> bool:
    """False below ``motion.min_speed_mps``, where nothing is reachable within the horizon."""
    return abs(float(speed_mps)) >= float(risk_cfg["motion"]["min_speed_mps"])


def curvature_from_yaw_rate(yaw_rate_radps: float, speed_mps: float) -> float:
    """Current curvature estimate ``kappa = omega / v`` from ego-motion.

    Undefined when stationary, so 0 (straight) is returned there; the caller should
    already have skipped the sweep via :func:`is_moving`.
    """
    if abs(speed_mps) < 1e-6:
        return 0.0
    return float(yaw_rate_radps) / float(speed_mps)
