"""Per-frame perception: detector boxes -> one obstacle list in the current ``veh`` frame.

Glue between the pieces built separately (P3-T2, P4-T3, P4-T4), so a recorded drive can
run end to end (P3-T6) and the streaming pipeline has a real "detect -> track -> depth"
stage (Phase 6):

1. **Place** each detected box on the ground with the v0 geometry (``depth.obstacle_from_box``).
2. **Split by kind** (``perception.yaml pipeline``):
   * *static* (cones, poles, boxes, anything unlisted): written into the persistent height
     map (``occupancy.HeightMap``) as a filled footprint, so it stays known after it leaves
     the camera's view - the case that matters beside the wheels;
   * *dynamic* (person, vehicle, cyclist): sent to the tracker (``track.Tracker``), which
     supplies a velocity and a stable id. Dynamic objects are deliberately *not* mapped: a
     person who walks away must not leave a ghost obstacle behind.
3. **Report** height-map obstacles plus confirmed tracks, in the vehicle's current frame,
   with ids that never collide (tracks first, map obstacles numbered after them).

The detector itself is injected as plain boxes, because it runs on the GPU (Colab) and
licensing constrains the choice (§10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from vscs.common.frames import invert, transform_points
from vscs.common.types import Obstacle
from vscs.perception.depth import obstacle_from_box
from vscs.perception.occupancy import HeightMap, visible_cells
from vscs.perception.track import Detection, Tracker


@dataclass(frozen=True)
class Box2D:
    """One detector output: pixel box, class name, score."""

    xyxy: tuple[float, float, float, float]
    cls: str
    score: float


class Perception:
    def __init__(
        self,
        perception_cfg: dict[str, Any],
        K: npt.ArrayLike,
        T_veh_cam: npt.ArrayLike,
        image_size: tuple[int, int],
    ) -> None:
        self.cfg = perception_cfg
        self.K = np.asarray(K, dtype=np.float64)
        self.T_veh_cam = np.asarray(T_veh_cam, dtype=np.float64)
        self.image_size = image_size
        self.map = HeightMap(perception_cfg["occupancy"])
        self.tracker = Tracker(perception_cfg["track"])
        pc = perception_cfg["pipeline"]
        self.class_to_kind = dict(pc["class_to_kind"])
        self.dynamic = set(pc["dynamic_kinds"])
        self.fill = float(pc["fill_step_m"])

    def kind_of(self, cls: str) -> str:
        return self.class_to_kind.get(cls, "static_geom")

    def _footprint_points(self, ob: Obstacle, T_world_veh: npt.ArrayLike) -> npt.NDArray:
        """A filled grid of points over the obstacle's footprint, at its top, in ``world``."""
        cx, cy, _ = ob.center_veh
        hx, hy = ob.extent[0] / 2.0, ob.extent[1] / 2.0
        xs = np.arange(cx - hx, cx + hx + 1e-9, self.fill)
        ys = np.arange(cy - hy, cy + hy + 1e-9, self.fill)
        X, Y = np.meshgrid(xs, ys)
        pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, ob.extent[2])])
        return transform_points(np.asarray(T_world_veh), pts)

    def step(self, t_ns: int, boxes: list[Box2D], T_world_veh: npt.ArrayLike) -> list[Obstacle]:
        """Fold in one frame; return every known obstacle in the current ``veh`` frame."""
        T_world_veh = np.asarray(T_world_veh, dtype=np.float64)
        depth_cfg = self.cfg["depth"]
        static_pts, dynamic = [], []
        for i, b in enumerate(boxes):
            kind = self.kind_of(b.cls)
            ob = obstacle_from_box(
                b.xyxy, self.K, self.T_veh_cam, depth_cfg, obstacle_id=i, t_ns=t_ns, kind=kind
            )
            if ob is None:
                continue
            if kind in self.dynamic:
                c_world = transform_points(T_world_veh, np.array([ob.center_veh]))[0]
                dynamic.append(
                    Detection(
                        xy=(float(c_world[0]), float(c_world[1])),
                        extent=ob.extent,
                        kind=kind,
                        score=b.score,
                    )
                )
            else:
                static_pts.append(self._footprint_points(ob, T_world_veh))
        visible = visible_cells(
            self.map,
            self.K,
            T_world_veh @ self.T_veh_cam,
            self.image_size,
            float(depth_cfg["max_range_m"]),
        )
        pts = np.vstack(static_pts) if static_pts else np.zeros((0, 3))
        self.map.update(pts, t_ns, visible)
        self.tracker.update(dynamic, t_ns)

        T_veh_world = invert(T_world_veh)
        sigma = float(self.cfg["track"]["kalman"]["measurement_std_m"])
        tracked = self.tracker.obstacles(T_veh_world, t_ns, pos_sigma_m=sigma)
        first = 1 + max((o.id for o in tracked), default=-1)
        mapped = self.map.obstacles(
            T_veh_world,
            t_ns,
            first_id=max(first, self.tracker.next_id),
            pos_sigma_m=float(self.cfg["occupancy"]["cell_size_m"]),
        )
        return tracked + mapped
