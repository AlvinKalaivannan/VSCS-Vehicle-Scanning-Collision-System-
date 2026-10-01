"""Persistent height map: obstacles stay known after leaving the camera's view (P4-T3).

Acceptance (CLAUDE.md §6): obstacles remain tracked beside the wheels after exiting view.

A reversing camera loses sight of a pole as soon as the van draws level with it, which is
exactly when the pole is most dangerous to the side of the van. So obstacles are kept in a
map in the fixed ``world`` frame (the vehicle's pose at the start of the drive, §4.1),
not in the moving camera's view.

The map is a 2D grid over the ground. Each cell holds the tallest point seen in it and
when it was last confirmed:

* **Insert.** Every observed 3D point (already in ``world``) raises its cell's height.
  Points below ``height_min_m`` or above ``height_max_m`` are rejected as noise or sky.
* **Carve free space.** A cell the camera can see *this frame* that received no obstacle
  point is cleared. Without this, anything that once crossed the view (a person walking
  past) would stay in the map until it expired.
* **Persist.** A cell the camera cannot see keeps its last height, for at most
  ``max_persist_s`` (R-08: ego-motion drift makes old cells untrustworthy).
* **Reset** per manoeuvre (``reset_per_manoeuvre``), so drift cannot accumulate forever.

Obstacles are connected groups of cells taller than ``obstacle_height_threshold_m``,
returned as §4.2 ``Obstacle`` boxes in the vehicle's *current* frame (``source:
"occupancy"``). That is independent of any detector, which R-07 needs for thin poles and
kerbs a detector misses.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from vscs.common.frames import invert, transform_points
from vscs.common.types import Obstacle

FloatArray = npt.NDArray[np.float64]

NS_PER_S = 1_000_000_000


class HeightMap:
    """A ``world``-frame height grid centred on the world origin."""

    def __init__(self, occ_cfg: dict[str, Any]) -> None:
        self.cell = float(occ_cfg["cell_size_m"])
        ex, ey = (float(v) for v in occ_cfg["grid_extent_m"])
        self.nx, self.ny = round(ex / self.cell), round(ey / self.cell)
        self.origin = np.array([-ex / 2.0, -ey / 2.0])  # world xy of cell (0, 0)'s corner
        self.h_min = float(occ_cfg["height_min_m"])
        self.h_max = float(occ_cfg["height_max_m"])
        self.threshold = float(occ_cfg["obstacle_height_threshold_m"])
        self.persist = bool(occ_cfg["persist_after_leaving_fov"])
        self.max_persist_ns = round(float(occ_cfg["max_persist_s"]) * NS_PER_S)
        self.reset()

    def reset(self) -> None:
        self.height = np.full((self.nx, self.ny), -np.inf)
        self.last_seen_ns = np.zeros((self.nx, self.ny), dtype=np.int64)
        self.dropped_outside = 0

    # ------------------------------------------------------------------ cells
    def cell_of(self, xy: npt.ArrayLike) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
        """Grid indices of world ``xy`` points, and whether each lies inside the grid."""
        ij = np.floor((np.atleast_2d(np.asarray(xy, dtype=np.float64)) - self.origin) / self.cell)
        ij = ij.astype(np.int64)
        inside = (ij[:, 0] >= 0) & (ij[:, 0] < self.nx) & (ij[:, 1] >= 0) & (ij[:, 1] < self.ny)
        return ij, inside

    def cell_centres(self) -> FloatArray:
        """World xy of every cell centre, shape ``(nx, ny, 2)``."""
        i, j = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="ij")
        return np.stack([i, j], axis=-1) * self.cell + self.origin + self.cell / 2.0

    # ----------------------------------------------------------------- update
    def update(
        self,
        points_world: npt.ArrayLike,
        t_ns: int,
        visible: npt.NDArray[np.bool_] | None = None,
    ) -> None:
        """Fold one frame in. ``visible`` (``(nx, ny)``) marks cells the camera can see now."""
        pts = np.atleast_2d(np.asarray(points_world, dtype=np.float64)).reshape(-1, 3)
        pts = pts[(pts[:, 2] >= self.h_min) & (pts[:, 2] <= self.h_max)]
        ij, inside = self.cell_of(pts[:, :2])
        self.dropped_outside += int((~inside).sum())
        ij, z = ij[inside], pts[inside, 2]

        hit = np.zeros((self.nx, self.ny), dtype=bool)
        if len(z):
            frame_h = np.full((self.nx, self.ny), -np.inf)
            np.maximum.at(frame_h, (ij[:, 0], ij[:, 1]), z)
            hit = np.isfinite(frame_h)
            if visible is None:
                self.height[hit] = np.maximum(self.height[hit], frame_h[hit])
            else:
                # In view, this frame is the truth for that cell: replace, don't accumulate.
                self.height[hit] = frame_h[hit]
            self.last_seen_ns[hit] = int(t_ns)
        if visible is not None:
            carve = visible & ~hit
            self.height[carve] = -np.inf
            if not self.persist:
                self.height[~visible] = -np.inf
        stale = np.isfinite(self.height) & (int(t_ns) - self.last_seen_ns > self.max_persist_ns)
        self.height[stale] = -np.inf

    # -------------------------------------------------------------- obstacles
    def obstacles(
        self, T_veh_world: npt.ArrayLike, t_ns: int, *, first_id: int = 0, pos_sigma_m: float
    ) -> list[Obstacle]:
        """Connected groups of tall cells, as ``Obstacle`` boxes in the current ``veh`` frame.

        The box is axis-aligned in ``world`` and its centre is mapped into ``veh``. Extent
        is measured in ``world`` (cells are square, so a rotation of up to 45 deg can make
        an axis-aligned box conservative, never too small).
        """
        occupied = np.isfinite(self.height) & (self.height > self.threshold)
        labels, n = ndimage.label(occupied, structure=np.ones((3, 3), dtype=bool))
        centres = self.cell_centres()
        out: list[Obstacle] = []
        for k in range(1, n + 1):
            m = labels == k
            xy = centres[m]
            lo = xy.min(axis=0) - self.cell / 2.0
            hi = xy.max(axis=0) + self.cell / 2.0
            top = float(self.height[m].max())
            centre_world = np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, top / 2.0])
            centre_veh = transform_points(np.asarray(T_veh_world), centre_world[None])[0]
            # Size in veh: rotate the world-aligned box's half extents and take the hull.
            R = np.asarray(T_veh_world)[:3, :3]
            half = np.array([(hi[0] - lo[0]) / 2.0, (hi[1] - lo[1]) / 2.0])
            ext_xy = 2.0 * (np.abs(R[:2, :2]) @ half)
            out.append(
                Obstacle(
                    id=first_id + k - 1,
                    t_ns=int(t_ns),
                    kind="static_geom",
                    center_veh=tuple(float(v) for v in centre_veh),
                    extent=(float(ext_xy[0]), float(ext_xy[1]), top),
                    velocity_veh=(0.0, 0.0, 0.0),
                    pos_sigma_m=float(pos_sigma_m),
                    source="occupancy",
                )
            )
        return out


def visible_cells(
    hmap: HeightMap,
    K: npt.ArrayLike,
    T_world_cam: npt.ArrayLike,
    image_size: tuple[int, int],
    max_range_m: float,
) -> npt.NDArray[np.bool_]:
    """Cells whose ground centre projects inside the image, within range, in front."""
    from vscs.common.frames import project

    c = hmap.cell_centres().reshape(-1, 2)
    pts = np.column_stack([c, np.zeros(len(c))])
    cam = transform_points(invert(np.asarray(T_world_cam)), pts)
    _, ok = project(np.asarray(K), cam, image_size)
    near = np.linalg.norm(c - np.asarray(T_world_cam)[:2, 3], axis=1) <= max_range_m
    return (ok & near).reshape(hmap.nx, hmap.ny)
