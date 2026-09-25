"""Ground-plane estimation by RANSAC (P1-T6 plumbing).

The ground plane defines the vehicle frame's ``z = 0`` and its up axis (§4.1: origin on the
ground below the rear axle, z up), so every clearance and every underbody check later
depends on it.

A plane is ``{p : n . p + d = 0}`` with ``n`` a unit normal. The signed distance of a point
is ``n . p + d``; with ``n`` oriented up, the van sits at positive distances.

Why RANSAC, and why the tilt limit
----------------------------------
A least-squares fit to every point is dragged toward the van, since the van is not on the
plane. RANSAC instead repeatedly fits a plane to three random points and keeps whichever
has the most points within ``distance_threshold_m``, which ignores everything off the
plane. It then refines that winner by least squares on its inliers alone.

But "most points" is the weakness here. A van's side is a large flat vertical panel, and on
a scan that circles it closely the panel can carry more points than the visible ground.
So candidates whose normal is more than ``max_tilt_deg`` from "up" are rejected outright.
"Up" comes from :func:`up_hint_from_cameras`: in the OpenCV camera frame y points *down* the
image, and a phone is held roughly upright, so the average of the cameras' -y axes is a
serviceable estimate of world up without any extra sensor.

Implemented with numpy rather than Open3D, which is deliberately not installed yet
(ADR 0001), and seeded per §4.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Plane:
    """``n . p + d = 0``, with ``normal`` a unit vector oriented up."""

    normal: FloatArray
    d: float

    def signed_distance(self, points: npt.ArrayLike) -> FloatArray:
        return np.atleast_2d(np.asarray(points, dtype=np.float64)) @ self.normal + self.d

    def project(self, points: npt.ArrayLike) -> FloatArray:
        """Foot of the perpendicular from each point onto the plane."""
        P = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return P - np.outer(self.signed_distance(P), self.normal)

    def flipped(self) -> Plane:
        return Plane(-self.normal, -self.d)

    def tilt_deg(self, up: npt.ArrayLike) -> float:
        """Angle between this plane's normal and ``up``, in degrees."""
        u = np.asarray(up, dtype=np.float64)
        u = u / np.linalg.norm(u)
        return float(np.degrees(np.arccos(np.clip(abs(self.normal @ u), -1.0, 1.0))))


def fit_plane_lstsq(points: npt.ArrayLike) -> Plane:
    """Total least-squares plane: the normal is the direction of least spread.

    Centre the points; the right singular vector with the smallest singular value is the
    direction the points vary least along - the plane normal. This minimises
    *perpendicular* distance, unlike fitting ``z = ax + by + c``, which minimises vertical
    distance and depends on the arbitrary orientation of the SfM world.
    """
    P = np.asarray(points, dtype=np.float64)
    if P.shape[0] < 3:
        raise ValueError(f"need at least 3 points to fit a plane, got {P.shape[0]}")
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[-1] / np.linalg.norm(vt[-1])
    return Plane(n, float(-n @ c))


def up_hint_from_cameras(T_world_cams: npt.ArrayLike) -> FloatArray:
    """Approximate world "up" from camera poses: the mean of each camera's -y axis.

    Column 1 of ``R_world_cam`` is the camera's y axis in the world, and camera y points
    down the image.
    """
    Ts = np.asarray(T_world_cams, dtype=np.float64)
    if Ts.ndim != 3 or Ts.shape[1:] != (4, 4):
        raise ValueError(f"expected (N, 4, 4) transforms, got {Ts.shape}")
    up = -Ts[:, :3, 1].mean(axis=0)
    norm = float(np.linalg.norm(up))
    if norm < 1e-6:
        raise ValueError("camera orientations cancel out; cannot estimate an up direction")
    return up / norm


def ransac_plane(
    points: npt.ArrayLike,
    *,
    distance_threshold_m: float,
    num_iterations: int,
    seed: int,
    up_hint: npt.ArrayLike | None = None,
    max_tilt_deg: float | None = None,
) -> tuple[Plane, npt.NDArray[np.bool_]]:
    """Fit the dominant plane, optionally constrained to be near-horizontal.

    Returns ``(plane, inlier_mask)``. The plane's normal points toward ``up_hint`` when one
    is given, otherwise toward the side holding most of the non-plane points (the vehicle
    sits above the ground, so the up side is where the rest of the scene is).
    """
    P = np.asarray(points, dtype=np.float64)
    n_pts = P.shape[0]
    if n_pts < 3:
        raise ValueError(f"need at least 3 points, got {n_pts}")
    up = None
    if up_hint is not None:
        up = np.asarray(up_hint, dtype=np.float64)
        up = up / np.linalg.norm(up)
    cos_limit = np.cos(np.radians(max_tilt_deg)) if max_tilt_deg is not None else None
    if cos_limit is not None and up is None:
        raise ValueError("max_tilt_deg needs an up_hint")

    rng = np.random.default_rng(seed)
    best_count, best_res, best_mask = -1, np.inf, None
    for _ in range(int(num_iterations)):
        a, b, c = P[rng.choice(n_pts, size=3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(n))
        if norm < 1e-12:
            continue  # collinear sample
        n = n / norm
        if cos_limit is not None and abs(float(n @ up)) < cos_limit:
            continue  # too steep to be ground - e.g. a van's side panel
        dist = np.abs(P @ n - float(n @ a))
        mask = dist < distance_threshold_m
        count = int(mask.sum())
        if count > best_count or (count == best_count and dist[mask].mean() < best_res):
            best_count, best_res, best_mask = count, float(dist[mask].mean()), mask

    if best_mask is None or best_count < 3:
        raise RuntimeError(
            "RANSAC found no acceptable plane. If max_tilt_deg is set, the up estimate may "
            "be wrong, or the ground may not be visible in the scan."
        )

    plane = fit_plane_lstsq(P[best_mask])
    # Recompute inliers against the refined plane, so the mask matches what is returned.
    mask = np.abs(plane.signed_distance(P)) < distance_threshold_m

    if up is not None:
        if plane.normal @ up < 0:
            plane = plane.flipped()
    else:
        others = plane.signed_distance(P[~mask])
        if others.size and np.median(others) < 0:
            plane = plane.flipped()
    return plane, mask


def ground_from_config(
    points: npt.ArrayLike,
    recon_cfg: dict[str, Any],
    *,
    up_hint: npt.ArrayLike | None = None,
) -> tuple[Plane, npt.NDArray[np.bool_]]:
    """:func:`ransac_plane` with parameters from ``configs/recon.yaml``.

    The tilt limit is only applied when an up hint is available.
    """
    g = recon_cfg["ground_plane"]
    if g["method"] != "ransac":
        raise ValueError(f"unsupported ground_plane.method {g['method']!r}")
    return ransac_plane(
        points,
        distance_threshold_m=float(g["distance_threshold_m"]),
        num_iterations=int(g["num_iterations"]),
        seed=int(g["seed"]),
        up_hint=up_hint,
        max_tilt_deg=float(g["max_tilt_deg"]) if up_hint is not None else None,
    )
