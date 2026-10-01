"""A whole recorded drive through the risk engine: obstacles + ego-motion -> RiskFrames.

The per-frame work is ``risk.engine.assess_frame``. This module supplies the inputs a
recorded drive provides:

* **Ego state per frame** from the pose stream (``ego_states``): signed speed along the
  vehicle's own x axis (negative when reversing) and yaw rate, from consecutive poses
  ``T_world_veh``. Curvature is then ``yaw_rate / speed`` (``motion.curvature_from_yaw_rate``).
* **Obstacle footprints** from §4.2 ``Obstacle`` boxes, already in the current ``veh``
  frame (``perception``), duck-typed as the footprints the sweep takes.

One ``AlertStateMachine`` runs across the whole drive, so hysteresis spans frames as it
would live. The sweep is the developer's ``risk/sweep.py`` (P3-T4) unless one is injected.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import numpy.typing as npt
from shapely.geometry import Polygon, box

from vscs.common.types import Obstacle, RiskFrame
from vscs.risk.alerts import AlertStateMachine
from vscs.risk.engine import SweepFn, assess_frame
from vscs.risk.motion import curvature_from_yaw_rate

NS_PER_S = 1_000_000_000


@dataclass(frozen=True)
class EgoState:
    t_ns: int
    speed_mps: float  # signed along veh x: negative = reversing
    yaw_rate_radps: float


@dataclass(frozen=True)
class Shape:
    """A footprint the sweep accepts: name, plan polygon, vertical extent, obstacle id."""

    name: str
    polygon: Polygon
    z_min: float
    z_max: float
    obstacle_id: int | None = None


def ego_states(poses: Sequence[tuple[int, npt.ArrayLike]]) -> list[EgoState]:
    """Speed and yaw rate from consecutive ``(t_ns, T_world_veh)``; the first frame copies
    the second's motion (there is nothing earlier to difference against)."""
    if len(poses) < 2:
        return [EgoState(int(poses[0][0]), 0.0, 0.0)] if poses else []
    out = []
    for (t0, A), (t1, B) in pairwise(poses):
        dt = (t1 - t0) / NS_PER_S
        rel = np.linalg.inv(np.asarray(A)) @ np.asarray(B)  # motion in the earlier frame
        out.append(
            EgoState(int(t1), float(rel[0, 3] / dt), float(np.arctan2(rel[1, 0], rel[0, 0]) / dt))
        )
    return [EgoState(int(poses[0][0]), out[0].speed_mps, out[0].yaw_rate_radps), *out]


def obstacle_shapes(obstacles: Sequence[Obstacle]) -> list[Shape]:
    shapes = []
    for o in obstacles:
        cx, cy, cz = o.center_veh
        hx, hy, hz = (e / 2.0 for e in o.extent)
        shapes.append(
            Shape(f"ob{o.id}", box(cx - hx, cy - hy, cx + hx, cy + hy), cz - hz, cz + hz, o.id)
        )
    return shapes


def component_shapes(footprints: Mapping[str, tuple[Polygon, float, float]]) -> list[Shape]:
    """From ``model.urdf.plan_footprints``."""
    return [Shape(n, p, zmin, zmax) for n, (p, zmin, zmax) in footprints.items()]


def assess_drive(
    components: Sequence[Shape],
    obstacles_by_t: Mapping[int, Sequence[Obstacle]],
    ego: Sequence[EgoState],
    risk_cfg: dict[str, Any],
    severity_cfg: dict[str, Any],
    *,
    sweep_fn: SweepFn | None = None,
) -> list[RiskFrame]:
    sm = AlertStateMachine(risk_cfg)
    frames = []
    for e in ego:
        obs = list(obstacles_by_t.get(e.t_ns, ()))
        frames.append(
            assess_frame(
                t_ns=e.t_ns,
                ego_speed_mps=e.speed_mps,
                curvature=curvature_from_yaw_rate(e.yaw_rate_radps, e.speed_mps),
                components=list(components),
                obstacles=obstacle_shapes(obs),
                obstacle_kinds={o.id: o.kind for o in obs},
                state_machine=sm,
                risk_cfg=risk_cfg,
                severity_cfg=severity_cfg,
                sweep_fn=sweep_fn,
            )
        )
    return frames
