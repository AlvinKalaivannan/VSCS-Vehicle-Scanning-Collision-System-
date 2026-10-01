"""Vertical clearance: low components against the persistent height map (P4-T6).

Acceptance (CLAUDE.md §6): the kerb / speed-bump fixture is flagged for the correct
component.

The plan-view sweep (P3-T4) asks "will a component's outline touch an obstacle's
outline?". For things the van drives *over* (a kerb, a speed bump, a rock), that is the
wrong question: the bumper passes over a 12 cm kerb with room to spare, and a wheel rolls
into it. What matters is height. For each low component (``min_z`` at most
``underbody.max_component_min_z_m``) and each step along the predicted path:

1. move the component's plan footprint to that pose (``fan.T_veh0_veh``);
2. collect the mapped ground cells (``perception.occupancy``) whose centres fall inside
   it;
3. clearance = component ``min_z`` - cell height. The first step where it drops below
   ``underbody.margin_m`` is a predicted underside contact, at that step's time.

Cells are tested at their centres, so footprints are resolved to about one cell (5 cm).
Heights are in the ``veh0`` frame of the current frame, the same frame the path fan is
in; the caller maps the world height map into it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import shapely
from shapely import affinity
from shapely.geometry import Polygon

from vscs.common.frames import transform_points
from vscs.perception.occupancy import HeightMap
from vscs.risk.motion import PathFan

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class UnderbodyResult:
    component: str
    min_clearance_m: float  # inf if nothing mapped ever passed under it
    ttc_s: float | None  # first predicted underside contact, or None


def ground_cells(hmap: HeightMap, T_veh0_world: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Occupied cells' centres in ``veh0`` (``(M, 2)``) and their heights (``(M,)``)."""
    occupied = np.isfinite(hmap.height) & (hmap.height > hmap.threshold)
    centres = hmap.cell_centres()[occupied]
    pts = np.column_stack([centres, np.zeros(len(centres))])
    xy = transform_points(np.asarray(T_veh0_world), pts)[:, :2] if len(pts) else np.zeros((0, 2))
    return xy, hmap.height[occupied]


def _moved(polygon: Polygon, T: FloatArray) -> Polygon:
    # 2D part of a 4x4 rigid transform: [a b xoff; d e yoff].
    return affinity.affine_transform(
        polygon, [T[0, 0], T[0, 1], T[1, 0], T[1, 1], T[0, 3], T[1, 3]]
    )


def underbody_check(
    footprints: dict[str, tuple[Polygon, float, float]],
    cells_xy: npt.ArrayLike,
    cells_h: npt.ArrayLike,
    fan: PathFan,
    k: int,
    underbody_cfg: dict[str, Any],
) -> dict[str, UnderbodyResult]:
    """Per low component: minimum clearance over the path, and first contact time."""
    xy = np.asarray(cells_xy, dtype=np.float64).reshape(-1, 2)
    h = np.asarray(cells_h, dtype=np.float64)
    margin = float(underbody_cfg["margin_m"])
    max_low = float(underbody_cfg["max_component_min_z_m"])
    out: dict[str, UnderbodyResult] = {}
    for name, (poly, z_min, _z_max) in footprints.items():
        if z_min > max_low:
            continue
        best, ttc = math.inf, None
        if len(xy):
            for i in range(fan.n_steps):
                inside = shapely.intersects_xy(
                    _moved(poly, fan.T_veh0_veh(k, i)), xy[:, 0], xy[:, 1]
                )
                if not inside.any():
                    continue
                clearance = float(z_min - h[inside].max())
                best = min(best, clearance)
                if ttc is None and clearance < margin:
                    ttc = float(fan.t_s[i])
        out[name] = UnderbodyResult(name, best, ttc)
    return out
