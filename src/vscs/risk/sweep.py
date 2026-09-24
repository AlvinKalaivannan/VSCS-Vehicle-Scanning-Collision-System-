"""2D per-component swept clearance and time-to-contact (P3-T4).

CORE CONTRIBUTION MODULE — pair mode (CLAUDE.md §0). The developer writes the first
draft; this file currently holds only the contract: data containers, signatures and
docstrings. Every function raises ``NotImplementedError`` until implemented.

The target is ``tests/unit/test_sweep.py``. It fails until this module is complete.

Approach (agreed 2026-09-24)
----------------------------
For each component footprint ``P`` (veh frame) and obstacle footprint ``O`` (veh0 frame):

1. Inflate both: ``P`` by ``sweep.component_margin_m``, ``O`` by ``sweep.obstacle_margin_m``
   (a Minkowski sum with a disc — shapely ``buffer``). Contact means the inflated gap
   reaches zero.
2. Height gate ("2.5D"): skip the pair unless their z-ranges overlap. A kerb under a mirror
   is not a mirror strike.
3. Move ``P`` along one path of the fan: at step ``i`` its footprint in veh0 is
   ``A_i = T_i · P`` with ``T_i = fan.T_veh0_veh(k, i)``. Track the minimum
   ``A_i.distance(O)``.
4. Tunnelling guard: between steps, test the swept region against ``O`` - the convex
   hull of the union of ``A_i`` and ``A_{i+1}``. Sampled gaps alone can miss a thin
   obstacle that falls between two samples.
5. TTC refinement: inside the first segment whose swept region hits ``O``, re-evaluate
   with the continuous pose ``motion.constant_curvature_pose(kappa, v * t)`` until the
   first-contact time is bracketed to within ``sweep.ttc_tolerance_s``.

Path choice (agreed): the risk reported in ``ComponentRisk`` comes from the path at the
current estimated curvature. :func:`sweep_fan` still sweeps every curvature and keeps the
results, for P4-T7 to turn into a contact probability.

Known answer the tests pin
--------------------------
Fixture, reversing straight at 1 m/s: rear-right bumper corner to the 0.06 m pole is
0.47 m surface-to-surface; with 0.05 m + 0.05 m margins the inflated gap is 0.37 m, so
**TTC = 0.37 s**. Naive per-sample checking would report 0.40 s.

Useful shapely calls
--------------------
``shapely.geometry.box(xmin, ymin, xmax, ymax)``, ``geom.buffer(r)``,
``shapely.affinity.affine_transform(geom, [a, b, d, e, xoff, yoff])`` (for a planar rigid
transform: ``a = e = cos θ``, ``b = -sin θ``, ``d = sin θ``), ``a.union(b).convex_hull``,
``a.intersects(b)``, ``a.distance(b)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy.typing as npt
from shapely.geometry import Polygon

from vscs.common.types import Obstacle
from vscs.risk.motion import PathFan


@dataclass(frozen=True)
class Footprint:
    """A plan-view footprint with a vertical extent.

    ``polygon`` is in the ``veh`` frame for components (it moves with the vehicle) and in
    ``veh0`` for obstacles (static in v0.1). ``obstacle_id`` is set for obstacles only.
    """

    name: str
    polygon: Polygon
    z_min: float
    z_max: float
    obstacle_id: int | None = None


@dataclass(frozen=True)
class SweepResult:
    """Clearance and time-to-contact for one component against one obstacle, on one path.

    ``min_distance_m`` is the minimum *inflated* clearance over ``[0, horizon]``, never
    negative, and exactly ``0.0`` whenever contact occurs. ``ttc_s`` is the first time the
    inflated shapes touch, ``0.0`` if they already overlap at ``t = 0``, and ``None`` if
    they never touch within the horizon. ``curvature`` records which fan path was swept.
    """

    component: str
    obstacle_id: int | None
    min_distance_m: float
    ttc_s: float | None
    curvature: float


def box_footprint(name: str, lo: npt.ArrayLike, hi: npt.ArrayLike) -> Footprint:
    """Footprint of an axis-aligned box given its ``[x, y, z]`` min and max corners."""
    raise NotImplementedError("P3-T4: developer's first draft")


def obstacle_footprint(obstacle: Obstacle) -> Footprint:
    """Footprint of a §4.2 ``Obstacle``: an axis-aligned box from ``center_veh`` and ``extent``.

    ``extent`` is the full size, not the half-size. The result carries ``obstacle.id``.
    """
    raise NotImplementedError("P3-T4: developer's first draft")


def z_overlap(a: Footprint, b: Footprint) -> bool:
    """True if the two vertical ranges overlap. Touching ranges count as overlapping."""
    raise NotImplementedError("P3-T4: developer's first draft")


def sweep_pair(
    component: Footprint,
    obstacle: Footprint,
    fan: PathFan,
    k: int,
    *,
    component_margin_m: float,
    obstacle_margin_m: float,
    ttc_tolerance_s: float,
) -> SweepResult | None:
    """Sweep one component along path ``k`` of ``fan`` against one obstacle.

    Returns ``None`` when the height gate excludes the pair.
    """
    raise NotImplementedError("P3-T4: developer's first draft")


def sweep_components(
    components: list[Footprint],
    obstacles: list[Footprint],
    fan: PathFan,
    k: int,
    risk_cfg: dict[str, Any],
) -> dict[str, SweepResult]:
    """Per component, the single most threatening obstacle on path ``k``.

    Margins, ``ttc_tolerance_s`` and ``report_distance_m`` come from ``risk_cfg["sweep"]``.

    Most threatening = earliest ``ttc_s``; if no obstacle gives contact, smallest
    ``min_distance_m``. A component appears in the result only if its most threatening
    obstacle comes within ``report_distance_m``.
    """
    raise NotImplementedError("P3-T4: developer's first draft")


def sweep_fan(
    components: list[Footprint],
    obstacles: list[Footprint],
    fan: PathFan,
    risk_cfg: dict[str, Any],
) -> list[dict[str, SweepResult]]:
    """:func:`sweep_components` for every curvature; entry ``k`` is fan row ``k``."""
    raise NotImplementedError("P3-T4: developer's first draft")
