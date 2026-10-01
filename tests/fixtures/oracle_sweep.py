"""A brute-force reference implementation of the ``sweep_components`` contract.

**Test infrastructure, not a template.** It samples the continuous motion model every
couple of milliseconds and measures every pose - obviously correct, far too slow for real
use, and structurally nothing like the swept-hull approach ``risk/sweep.py`` is meant to
take. It exists so ``risk/engine.py`` can be tested before the developer's draft of
``sweep.py`` exists (pair mode, P3-T4), and so the two can later be cross-checked.

It honours the same contract as ``sweep.sweep_components``: inflated clearance, a height
gate, ``min_distance_m == 0`` whenever contact occurs, and per component the most
threatening obstacle (earliest contact, else nearest) within ``report_distance_m``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, box

from vscs.risk.motion import PathFan, constant_curvature_pose


@dataclass(frozen=True)
class RefFootprint:
    name: str
    polygon: Polygon
    z_min: float
    z_max: float
    obstacle_id: int | None = None


@dataclass(frozen=True)
class RefResult:
    component: str
    obstacle_id: int | None
    min_distance_m: float
    ttc_s: float | None
    curvature: float
    t_closest_s: float | None = None  # ADR 0010: time of the minimum clearance


def ref_box(name: str, lo, hi, obstacle_id: int | None = None) -> RefFootprint:
    return RefFootprint(
        name, box(lo[0], lo[1], hi[0], hi[1]), float(lo[2]), float(hi[2]), obstacle_id
    )


def _pair(comp, obst, fan: PathFan, k: int, m_c: float, m_o: float, dt: float) -> RefResult | None:
    if comp.z_max < obst.z_min or obst.z_max < comp.z_min:
        return None
    P, obs = comp.polygon.buffer(m_c), obst.polygon.buffer(m_o)
    kappa = float(fan.curvatures[k])
    horizon = float(fan.t_s[-1])
    ts = np.append(np.arange(0.0, horizon, dt), horizon)
    ttc, d_min, t_min = None, math.inf, 0.0
    for t in ts:
        x, y, h = (float(v) for v in constant_curvature_pose(kappa, fan.speed_mps * t))
        c, s = math.cos(h), math.sin(h)
        g = affinity.affine_transform(P, [c, -s, s, c, x, y]).distance(obs)
        if g < d_min:  # strict: the earliest time of the minimum
            d_min, t_min = g, float(t)
        if g <= 0.0:
            ttc = float(t)
            break
    if ttc is not None:
        d_min, t_min = 0.0, ttc
    return RefResult(comp.name, obst.obstacle_id, d_min, ttc, kappa, t_min)


def ref_sweep_components(
    components: list[RefFootprint],
    obstacles: list[RefFootprint],
    fan: PathFan,
    k: int,
    risk_cfg: dict[str, Any],
    dt: float = 2e-3,
) -> dict[str, RefResult]:
    sw = risk_cfg["sweep"]
    m_c, m_o = float(sw["component_margin_m"]), float(sw["obstacle_margin_m"])
    report = float(sw["report_distance_m"])
    out: dict[str, RefResult] = {}
    for comp in components:
        results = [r for o in obstacles if (r := _pair(comp, o, fan, k, m_c, m_o, dt))]
        if not results:
            continue
        best = min(
            results,
            key=lambda r: (
                r.ttc_s is None,
                r.ttc_s if r.ttc_s is not None else 0.0,
                r.min_distance_m,
            ),
        )
        if best.min_distance_m <= report:
            out[comp.name] = best
    return out
