"""Perception v0: ground-plane-anchored geometry for static obstacles (P3-T2).

Acceptance (CLAUDE.md §6): cone positions within ±25 cm at ≤ 3 m on the dev split.

The idea
--------
One camera cannot measure depth, but anything touching the ground has a known height:
zero. So a pixel *on the ground* fixes a 3D point. In MAT188 terms:

1. The pixel ``(u, v)`` defines a ray. In the camera frame its direction is
   ``d_cam = K^-1 [u, v, 1]^T`` (scaled so its z-component is 1).
2. The camera's mounting ``T_veh_cam = [R | t]`` maps it into the vehicle frame: the ray
   starts at the camera centre ``o = t`` and runs along ``d = R d_cam``.
3. Points on the ray are ``o + λ d`` for ``λ > 0``. The ground is the plane ``z = 0``
   (the ``veh`` origin is on the ground, §4.1), so ``o_z + λ d_z = 0`` gives
   ``λ = -o_z / d_z``. That only works if the ray points *down* (``d_z < 0``); a ray at
   or above the horizon never meets the ground.

Because ``d_cam`` has z-component 1, ``λ`` is also the point's depth along the optical
axis.

Obstacles
---------
A detected box's **bottom-centre** pixel is where the object meets the ground, so it fixes
the object's position: precisely, the base's *near* edge, so the centre is placed half a
footprint beyond it, along the viewing direction. The box's bottom-left and bottom-right
corners, put on the ground the same way, give its width. Its height comes from the
top-centre pixel: the point on that ray directly above the ground contact.

How accurate
------------
Near the horizon, a one-pixel error moves the ground point a long way: range error grows
with roughly the *square* of range. ``range_sigma`` measures this directly by moving the
pixel ``±pixel_sigma_px`` and seeing how far the ground point moves. That sigma becomes
the obstacle's ``pos_sigma_m``, so the risk engine sees honest uncertainty (R-06).

Measured on the fixture (0.3 x 0.3 x 0.5 m cones, 2.5-5 m): width and height come out
high (perspective widens the box), which errs toward larger obstacles - conservative.

Assumptions to remember: a flat ground plane at ``z = 0``, a known fixed camera mount, and
a box bottom edge that really is the ground contact (an overhanging object breaks it).
A box clipped at the image's *top* is harmless (the base is what is used), but one clipped
at a *side* moves its bottom-centre inward and biases the position: found on the
perception-pipeline test, where a walker cut off at the right edge read -0.69 m/s
instead of -1.0 m/s. Treat side-clipped boxes with suspicion.

Tall thin objects off to the side are the worst case (R-07). With the camera pitched down,
a vertical pole projects as a *slanted* line, so its box is wide and its bottom-centre is
not where the pole meets the ground. Over the end-to-end fixture drive, boxes placed a 6 cm
pole a median 0.35 m off and 0.65 m wide. A box cannot say which side the base is on; a
mask's lowest pixels can: ``obstacle_from_mask`` places the same pole a median 0.05 m off
and 0.10 m wide. Use masks whenever the detector provides them (SAM 2, P2-T2).
Learned and motion-stereo depth replace this at P4-T1.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from vscs.common.types import Obstacle, ObstacleKind

FloatArray = npt.NDArray[np.float64]


def rays_veh(
    K: npt.ArrayLike, T_veh_cam: npt.ArrayLike, uv: npt.ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """``(origin (3,), directions (N, 3))`` in ``veh``; direction = ``R K^-1 [u, v, 1]``."""
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T_veh_cam, dtype=np.float64)
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    pix = np.column_stack([uv, np.ones(len(uv))])
    d_cam = pix @ np.linalg.inv(K).T
    return T[:3, 3].copy(), d_cam @ T[:3, :3].T


def ground_points(
    K: npt.ArrayLike,
    T_veh_cam: npt.ArrayLike,
    uv: npt.ArrayLike,
    max_range_m: float = np.inf,
) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """Where each pixel's ray meets the ground ``z = 0``.

    Returns ``(points (N, 3), valid (N,))``. A pixel is invalid if its ray points at or
    above the horizon, or lands farther than ``max_range_m`` (horizontal distance from the
    camera). Invalid rows are NaN.
    """
    o, d = rays_veh(K, T_veh_cam, uv)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = -o[2] / d[:, 2]
    pts = o + lam[:, None] * d
    horiz = np.linalg.norm(pts[:, :2] - o[:2], axis=1)
    valid = (d[:, 2] < 0) & np.isfinite(lam) & (lam > 0) & (horiz <= max_range_m)
    pts[~valid] = np.nan
    pts[valid, 2] = 0.0  # exactly on the plane, not 1e-16 off it
    return pts, valid


def range_sigma(
    K: npt.ArrayLike, T_veh_cam: npt.ArrayLike, uv: npt.ArrayLike, pixel_sigma_px: float
) -> FloatArray:
    """1-sigma ground-position error from ``pixel_sigma_px`` of vertical pixel noise.

    Half the distance between the ground points at ``v - sigma`` and ``v + sigma``. Infinite where
    either perturbed ray misses the ground (the pixel is too close to the horizon to trust).
    """
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    shift = np.array([0.0, pixel_sigma_px])
    up, ok_up = ground_points(K, T_veh_cam, uv - shift)
    dn, ok_dn = ground_points(K, T_veh_cam, uv + shift)
    sig = np.linalg.norm(up - dn, axis=1) / 2.0
    return np.where(ok_up & ok_dn, sig, np.inf)


def height_above(
    K: npt.ArrayLike, T_veh_cam: npt.ArrayLike, uv_top: npt.ArrayLike, ground_xy: npt.ArrayLike
) -> float:
    """Height of the point on ``uv_top``'s ray horizontally closest to ``ground_xy``."""
    o, d = rays_veh(K, T_veh_cam, uv_top)
    d = d[0]
    g = np.asarray(ground_xy, dtype=np.float64)
    dxy = d[:2]
    # Minimise |o_xy + s d_xy - g|^2 over s: s = (g - o_xy) . d_xy / |d_xy|^2.
    s = float(np.dot(g - o[:2], dxy) / np.dot(dxy, dxy))
    return float(o[2] + s * d[2])


def obstacle_from_box(
    box_xyxy: npt.ArrayLike,
    K: npt.ArrayLike,
    T_veh_cam: npt.ArrayLike,
    depth_cfg: dict[str, Any],
    *,
    obstacle_id: int,
    t_ns: int,
    kind: ObstacleKind = "static_geom",
) -> Obstacle | None:
    """A §4.2 ``Obstacle`` from one detected box, or ``None`` if its base is unusable.

    The footprint is taken as square (width x width): one view cannot see an object's
    depth. ``pos_sigma_m`` is the pixel-noise range error at the ground contact.
    """
    x0, y0, x1, y1 = (float(v) for v in np.asarray(box_xyxy).reshape(4))
    max_range = float(depth_cfg["max_range_m"])
    base, ok = ground_points(K, T_veh_cam, [[(x0 + x1) / 2, y1]], max_range)
    if not ok[0]:
        return None
    corners, ok_c = ground_points(K, T_veh_cam, [[x0, y1], [x1, y1]], max_range)
    width = float(np.linalg.norm(corners[0] - corners[1])) if ok_c.all() else 0.0
    height = max(0.0, height_above(K, T_veh_cam, [[(x0 + x1) / 2, y0]], base[0, :2]))
    sigma = float(
        range_sigma(K, T_veh_cam, [[(x0 + x1) / 2, y1]], float(depth_cfg["pixel_sigma_px"]))[0]
    )
    # The box's bottom edge is the base's NEAR edge, not its centre: measured on the
    # fixture, centring there puts every cone ~half its depth too close (0.15 m for a
    # 0.3 m cone). Anchor the near face instead: move the centre half a footprint
    # further along the horizontal viewing direction.
    o, _ = rays_veh(K, T_veh_cam, [[0.0, 0.0]])
    away = base[0, :2] - o[:2]
    away = away / np.linalg.norm(away)
    centre_xy = base[0, :2] + away * width / 2.0
    return Obstacle(
        id=obstacle_id,
        t_ns=int(t_ns),
        kind=kind,
        center_veh=(float(centre_xy[0]), float(centre_xy[1]), height / 2.0),
        extent=(width, width, height),
        velocity_veh=(0.0, 0.0, 0.0),
        pos_sigma_m=sigma if np.isfinite(sigma) else max_range,
        source="detector",
    )


def obstacle_from_mask(
    mask: npt.ArrayLike,
    K: npt.ArrayLike,
    T_veh_cam: npt.ArrayLike,
    depth_cfg: dict[str, Any],
    *,
    obstacle_id: int,
    t_ns: int,
    kind: ObstacleKind = "static_geom",
) -> Obstacle | None:
    """A §4.2 ``Obstacle`` from a segmentation mask, using where it actually meets the ground.

    The mask's lowest pixels (within ``mask_base_rows_px`` rows of its bottom-most pixel)
    are the object's ground contact. Put on the ground, their spread is the contact width
    and their mean is the base, with no guess about which side of a box the base is on (the
    box-only failure for slanted thin objects, R-07). Height comes from the mask's top-most
    pixel above the base. If the mask touches the image's top row, that height is only a
    lower bound, which is acceptable: a taller object is never less dangerous.

    Returns ``None`` if the mask is empty, or its base touches the image bottom (the true
    contact is out of view) or lies beyond ``max_range_m``.
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return None
    h_img = m.shape[0]
    rows, cols = np.nonzero(m)
    v_bottom = rows.max()
    if v_bottom >= h_img - 1:
        return None
    base = rows >= v_bottom - int(depth_cfg["mask_base_rows_px"])
    uv = np.column_stack([cols[base], rows[base]]).astype(np.float64)
    pts, ok = ground_points(K, T_veh_cam, uv, float(depth_cfg["max_range_m"]))
    if not ok.any():
        return None
    pts = pts[ok]
    lo, hi = pts[:, :2].min(axis=0), pts[:, :2].max(axis=0)
    width = float(max(hi[0] - lo[0], hi[1] - lo[1]))
    # The lowest rows trace the base's NEAR edge (as a box's bottom edge does): anchor the
    # near face and put the centre half a footprint further along the viewing direction.
    near = (lo + hi) / 2.0
    o, _ = rays_veh(K, T_veh_cam, [[0.0, 0.0]])
    away = (near - o[:2]) / np.linalg.norm(near - o[:2])
    centre = near + away * width / 2.0
    top_row = rows.min()
    u_top = float(cols[rows == top_row].mean())
    height = max(0.0, height_above(K, T_veh_cam, [[u_top, float(top_row)]], centre))
    u_base = float(cols[rows == v_bottom].mean())
    sigma = float(
        range_sigma(K, T_veh_cam, [[u_base, float(v_bottom)]], float(depth_cfg["pixel_sigma_px"]))[
            0
        ]
    )
    max_range = float(depth_cfg["max_range_m"])
    return Obstacle(
        id=obstacle_id,
        t_ns=int(t_ns),
        kind=kind,
        center_veh=(float(centre[0]), float(centre[1]), height / 2.0),
        extent=(width, width, height),
        velocity_veh=(0.0, 0.0, 0.0),
        pos_sigma_m=sigma if np.isfinite(sigma) else max_range,
        source="detector",
    )
