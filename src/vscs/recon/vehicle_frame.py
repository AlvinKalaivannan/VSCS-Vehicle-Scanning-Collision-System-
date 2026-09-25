"""Construct the vehicle frame ``veh`` from a metric reconstruction (P1-T6 plumbing).

§4.1: ``veh`` is x forward, y left, z up, with its origin on the ground plane directly below
the centre of the rear axle. This builds ``T_veh_world`` - the transform taking points in
the (already metric) reconstruction frame into ``veh`` - from three things:

1. **z** is the ground-plane normal (``ground.py``), pointing up.
2. **x** is the long axis of the body. The body points are projected onto the ground plane
   and the principal component of that 2D spread is taken. A van is much longer than it is
   wide, so the long axis is well defined. Principal components have no sign, so a point
   known to be toward the front picks which way is forward.
3. **The origin.** Along x, the rearmost body point is the rear bumper; the rear axle is
   ``rear_overhang_m`` ahead of it, a figure measured by hand on scan day. Across y, the
   origin is the midpoint of the body. It is placed on the ground plane.

``y = cross(z, x)`` completes a right-handed frame (since ``cross(x, y) = z``). The rotation
``R_world_veh`` has columns ``[x y z]`` - each ``veh`` axis written in world coordinates -
and its inverse gives ``T_veh_world``.

Dimensions are measured in the new frame for validation against the tape measurements
(P1-T6's ±2 cm gate). **Width is the median over slices along x**, so the mirrors, which
widen only a handful of slices, do not count. That matches the checklist, which measures
width without the mirrors.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from vscs.common.frames import T_from_Rt, invert, transform_points
from vscs.recon.ground import Plane

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class VehicleFrame:
    T_veh_world: FloatArray
    length_m: float
    width_m: float
    width_with_mirrors_m: float
    height_m: float

    @property
    def T_world_veh(self) -> FloatArray:
        return invert(self.T_veh_world)

    def dimensions(self) -> dict[str, float]:
        """The measurable dimensions, keyed as in ``recon.yaml`` ``scale.validate_against``."""
        return {"length": self.length_m, "width": self.width_m, "height": self.height_m}


def _trimmed_range(v: FloatArray, pct: float) -> tuple[float, float]:
    return float(np.percentile(v, pct)), float(np.percentile(v, 100.0 - pct))


def build_vehicle_frame(
    body_points_world: npt.ArrayLike,
    ground: Plane,
    *,
    rear_overhang_m: float,
    front_hint_world: npt.ArrayLike,
    width_slice_m: float = 0.10,
    trim_percentile: float = 0.5,
) -> VehicleFrame:
    """Build ``veh`` from the body points of a metric reconstruction.

    ``body_points_world`` should be the vehicle, not the ground: pass the points RANSAC did
    not classify as ground (and, once segmentation exists, only the van's points).
    """
    P = np.asarray(body_points_world, dtype=np.float64)
    if P.shape[0] < 10:
        raise ValueError(f"too few body points to build a frame: {P.shape[0]}")
    if rear_overhang_m is None or rear_overhang_m < 0:
        raise ValueError(
            "rear_overhang_m must be the hand-measured distance from the rear bumper to the "
            "rear axle centre (docs/capture_checklists.md). It places the frame origin."
        )

    z = ground.normal / np.linalg.norm(ground.normal)

    # Long axis of the footprint: PCA of the body projected into the ground plane.
    foot = ground.project(P)
    c = foot.mean(axis=0)
    flat = foot - c
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    x = vt[0] - (vt[0] @ z) * z  # remove any numerical out-of-plane component
    x = x / np.linalg.norm(x)

    hint = np.asarray(front_hint_world, dtype=np.float64).reshape(3)
    if (hint - c) @ x < 0:
        x = -x
    y = np.cross(z, x)

    # Extents in the ground-aligned axes.
    s = flat @ x
    t = flat @ y
    h = ground.signed_distance(P)
    rear_s, front_s = _trimmed_range(s, trim_percentile)

    # Width: per-slice lateral extent, median across slices, so mirrors do not count.
    edges = np.arange(rear_s, front_s + width_slice_m, width_slice_m)
    widths = []
    for lo, hi in itertools.pairwise(edges):
        sel = (s >= lo) & (s < hi)
        if sel.sum() >= 2:
            widths.append(float(t[sel].max() - t[sel].min()))
    if not widths:
        raise ValueError("no populated slices along the vehicle; check width_slice_m")
    mid_t = 0.5 * (float(t.min()) + float(t.max()))

    origin = c + (rear_s + rear_overhang_m) * x + mid_t * y
    R_world_veh = np.column_stack([x, y, z])
    T_veh_world = invert(T_from_Rt(R_world_veh, origin))

    return VehicleFrame(
        T_veh_world=T_veh_world,
        length_m=front_s - rear_s,
        width_m=float(np.median(widths)),
        width_with_mirrors_m=float(t.max() - t.min()),
        height_m=float(np.percentile(h, 100.0 - trim_percentile)),
    )


def vehicle_frame_from_config(
    body_points_world: npt.ArrayLike,
    ground: Plane,
    recon_cfg: dict[str, Any],
    *,
    front_hint_world: npt.ArrayLike,
) -> VehicleFrame:
    vf = recon_cfg["vehicle_frame"]
    if vf.get("rear_overhang_m") is None:
        raise ValueError(
            "configs/recon.yaml vehicle_frame.rear_overhang_m is not set. Measure the rear "
            "bumper to rear-axle-centre distance on scan day and record it there."
        )
    return build_vehicle_frame(
        body_points_world,
        ground,
        rear_overhang_m=float(vf["rear_overhang_m"]),
        front_hint_world=front_hint_world,
        width_slice_m=float(vf["width_slice_m"]),
        trim_percentile=float(vf["trim_percentile"]),
    )


def to_vehicle_frame(frame: VehicleFrame, points_world: npt.ArrayLike) -> FloatArray:
    """Express world points in ``veh``."""
    return transform_points(frame.T_veh_world, points_world)
