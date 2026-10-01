"""From a sparse reconstruction to the metric vehicle frame (P1-T6 plumbing).

The steps, in the order they have to happen:

1. **Marker corners in 3D.** Each marker corner is seen in several registered images
   (``marker_tracks.py``). Triangulating it from those views places it in the
   reconstruction, in SfM units. Four corners give the marker's side length there.
2. **Scale.** The printed side is known in metres, so each marker implies a scale
   ``s = side_m / side_sfm``; the markers' median is the estimate (R-02).
3. **Ground.** Scale first, because the RANSAC inlier threshold is in metres. "Up" for the
   tilt check comes from the cameras (a phone is held roughly upright).
4. **Body points.** The sparse cloud also holds the lot, other cars and the marker boards.
   Kept: points above the ground, inside the camera loop's footprint (the walk-round
   circles the van), away from every marker board (ID 3 stands *at* the front bumper, so
   its board would otherwise join the van and lengthen it), in clusters big enough to be
   the van rather than anything else small.
5. **Vehicle frame** (``vehicle_frame.py``), from the body points and the front marker,
   which says which end is the front.
6. **Validation** against the tape (``scale_error_m``, the ±2 cm gate). Wheelbase needs the
   wheel centres, which this step does not have: it is reported as not validated here,
   never silently dropped.

Steps 1, 2 and 6 call the developer's ``recon/scale.py`` (pair mode), passed in as
``scale_mod``, so this module never re-implements it.

Output: ``s`` and ``T_veh_world`` (scaled reconstruction frame -> ``veh``), the pair that
``seg.fuse_inputs.SfmToVeh`` consumes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely import contains_xy
from shapely.geometry import MultiPoint

from vscs.recon.colmap_io import Model
from vscs.recon.ground import ground_from_config, up_hint_from_cameras
from vscs.recon.marker_tracks import TrackReport, detection_fallback_triggered
from vscs.recon.vehicle_frame import to_vehicle_frame, vehicle_frame_from_config

FloatArray = npt.NDArray[np.float64]


@dataclass
class MetricResult:
    scale: float
    per_marker_scale: dict[int, float]
    spread_frac: float
    markers_consistent: bool
    detection_fraction: float
    detection_fallback_triggered: bool
    T_veh_world: list[list[float]]
    dimensions_m: dict[str, float]
    measured_m: dict[str, float]
    errors_m: dict[str, float]
    not_validated: list[str]
    scale_error_m: float
    max_dimension_error_m: float
    n_body_points: int
    body_cluster_sizes: list[int] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.scale_error_m <= self.max_dimension_error_m and self.markers_consistent

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "passed": self.passed}


def marker_corners(
    report: TrackReport, model: Model, scale_mod: ModuleType, min_views: int = 2
) -> dict[int, FloatArray]:
    """Triangulated ``(4, 3)`` corners, SfM units, for every marker seen in enough views."""
    out: dict[int, FloatArray] = {}
    for mid, track in sorted(report.usable(min_views).items()):
        corners = []
        for obs in track.corners:
            Ps = [
                scale_mod.projection_matrix(
                    model.cameras[model.images[i].camera_id].K, model.images[i].T_cam_world
                )
                for i, _ in obs
            ]
            corners.append(scale_mod.triangulate_point(Ps, [uv for _, uv in obs]))
        out[mid] = np.asarray(corners, dtype=np.float64)
    return out


def _clusters(points: FloatArray, eps: float) -> npt.NDArray[np.int64]:
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    pairs = cKDTree(points).query_pairs(eps, output_type="ndarray")
    graph = coo_matrix(
        (np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
    )
    return connected_components(graph, directed=False)[1].astype(np.int64)


def select_body(
    points: FloatArray,
    heights: FloatArray,
    foot_xy: FloatArray,
    camera_foot_xy: FloatArray,
    marker_centres: FloatArray,
    body_cfg: dict[str, Any],
) -> tuple[npt.NDArray[np.bool_], list[int]]:
    """Step 4. ``heights`` above the ground; ``foot_xy`` in any 2D basis of the ground plane.

    Returns the body mask and the sizes of the clusters kept (largest first).
    """
    loop = MultiPoint(camera_foot_xy).convex_hull
    cand = (
        (heights > float(body_cfg["min_height_m"]))
        & (heights < float(body_cfg["max_height_m"]))
        & contains_xy(loop, foot_xy[:, 0], foot_xy[:, 1])
    )
    if len(marker_centres):
        near, _ = cKDTree(np.atleast_2d(marker_centres)).query(points, k=1)
        cand &= near > float(body_cfg["marker_exclusion_m"])
    idx = np.flatnonzero(cand)
    mask = np.zeros(len(points), dtype=bool)
    if not len(idx):
        return mask, []
    ids = _clusters(points[idx], float(body_cfg["cluster_eps_m"]))
    sizes = np.bincount(ids)
    keep = np.flatnonzero(sizes >= int(body_cfg["min_cluster_points"]))
    mask[idx[np.isin(ids, keep)]] = True
    return mask, sorted((int(sizes[k]) for k in keep), reverse=True)


def metric_frame(
    model: Model,
    report: TrackReport,
    recon_cfg: dict[str, Any],
    measured_m: dict[str, float],
    scale_mod: ModuleType,
) -> tuple[MetricResult, FloatArray]:
    """Steps 1-6. Returns the result and the body points in ``veh`` (for a visual check)."""
    corners = marker_corners(report, model, scale_mod)
    sides = {m: scale_mod.marker_side_length(c) for m, c in corners.items()}
    est = scale_mod.estimate_scale_from_config(sides, recon_cfg)
    s = float(est.scale)

    front_id = int(recon_cfg["vehicle_frame"]["front_marker_id"])
    if front_id not in corners:
        raise ValueError(
            f"front marker {front_id} was not triangulated (seen in fewer than two good "
            "views), so the front of the van is unknown"
        )
    front = scale_mod.scale_points(corners[front_id], s).mean(axis=0)

    pts = scale_mod.scale_points(np.array([p.xyz for p in model.points.values()]), s)
    # Camera poses in the scaled frame: rotations unchanged, centres scaled like points.
    T_world_cams = np.array([im.T_world_cam for im in model.images.values()])
    T_world_cams[:, :3, 3] = scale_mod.scale_points(T_world_cams[:, :3, 3], s)
    ground, _ = ground_from_config(pts, recon_cfg, up_hint=up_hint_from_cameras(T_world_cams))

    # A 2D basis of the ground plane, for the camera-loop test.
    n = ground.normal
    e1 = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    basis = np.column_stack([e1, e2])
    body_mask, sizes = select_body(
        pts,
        ground.signed_distance(pts),
        ground.project(pts) @ basis,
        ground.project(T_world_cams[:, :3, 3]) @ basis,
        np.array([scale_mod.scale_points(c, s).mean(axis=0) for c in corners.values()]),
        recon_cfg["vehicle_frame"]["body_select"],
    )
    frame = vehicle_frame_from_config(pts[body_mask], ground, recon_cfg, front_hint_world=front)

    dims = frame.dimensions()
    wanted = list(recon_cfg["scale"]["validate_against"])
    errors = scale_mod.dimension_errors(dims, {k: measured_m[k] for k in wanted if k in measured_m})
    result = MetricResult(
        scale=s,
        per_marker_scale={int(k): float(v) for k, v in est.per_marker.items()},
        spread_frac=float(est.spread_frac),
        markers_consistent=bool(est.consistent),
        detection_fraction=report.detection_fraction,
        detection_fallback_triggered=detection_fallback_triggered(report, recon_cfg),
        T_veh_world=frame.T_veh_world.tolist(),
        dimensions_m=dims,
        measured_m=dict(measured_m),
        errors_m={k: float(v) for k, v in errors.items()},
        not_validated=[k for k in wanted if k not in errors],
        scale_error_m=float(scale_mod.scale_error_m(dims, measured_m)),
        max_dimension_error_m=float(recon_cfg["scale"]["max_dimension_error_m"]),
        n_body_points=int(body_mask.sum()),
        body_cluster_sizes=sizes,
    )
    return result, to_vehicle_frame(frame, pts[body_mask])
