"""Exact per-pixel label and depth images of the fixture scene (P2-T3 test input).

A ray is cast through every pixel centre and intersected analytically with every box in
the scene: the body shell, the labelled components, and the obstacles. The nearest hit
gives the pixel's depth and its label. That is what a *perfect* 2D segmenter plus a
*perfect* dense depth map would produce, so label fusion can be tested against a known
answer before SAM 2 or COLMAP dense has run on anything real.

Conventions match the pipeline:

* pixel ``(u, v)`` has its centre at integer coordinates (OpenCV);
* depth is z along the optical axis, not distance along the ray (see ``frames.unproject``);
* label is the component's index in ``component_names(scene)`` (sorted names), or
  :data:`NONE_LABEL` for the body shell, an obstacle, or empty background;
* empty background has depth ``inf``.

Component boxes share faces with the body shell (the rear bumper's back face *is* the
shell's back face). On such ties the component wins, so the shell never hides a
component's outer surface.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from fixtures.synthetic import Box, Scene
from vscs.common.frames import invert

NONE_LABEL = -1

#: Added to a non-component box's hit distance so a component wins a shared-face tie.
_TIE_BREAK_M = 1e-6


def component_names(scene: Scene) -> list[str]:
    """Class index -> component name. Sorted, matching ``Scene.labelled_point_cloud``."""
    return sorted(scene.components)


def scaled_intrinsics(scene: Scene, factor: float) -> tuple[npt.NDArray, tuple[int, int]]:
    """The fixture camera at ``factor`` x resolution, for fast tests.

    Pixel centres sit at integers, so the principal point scales as
    ``c' = (c + 0.5) * factor - 0.5``; plain ``c * factor`` would shift the image by a
    fraction of a pixel.
    """
    K = scene.K.copy()
    K[0, 0] *= factor
    K[1, 1] *= factor
    K[0, 2] = (K[0, 2] + 0.5) * factor - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * factor - 0.5
    w, h = scene.image_size
    return K, (round(w * factor), round(h * factor))


def _occluders(scene: Scene) -> list[Box]:
    """Non-component geometry: the shell and the obstacles (the pole as a square prism)."""
    p = scene.pole
    r = p.radius_m
    pole_box = Box.from_bounds(
        "pole",
        [p.axis_xy[0] - r, p.axis_xy[1] - r, 0.0, p.axis_xy[0] + r, p.axis_xy[1] + r, p.height_m],
    )
    return [scene.vehicle, scene.curb, scene.box_obstacle, pole_box]


def _ray_box_t(origin: npt.NDArray, dirs: npt.NDArray, box: Box) -> npt.NDArray:
    """Entry parameter ``t`` of each ray ``origin + t * dir`` into ``box``; inf on a miss.

    Slab method. ``dirs`` are *not* normalised: they are the camera ray scaled so that
    its optical-axis component is 1, which makes ``t`` equal to z-depth directly.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs
        t1 = (box.lo - origin) * inv
        t2 = (box.hi - origin) * inv
    t_near = np.nanmax(np.minimum(t1, t2), axis=1)
    t_far = np.nanmin(np.maximum(t1, t2), axis=1)
    hit = (t_near <= t_far) & (t_far > 0) & (t_near > 0)
    return np.where(hit, t_near, np.inf)


def render(
    scene: Scene,
    T_world_cam: npt.ArrayLike,
    K: npt.ArrayLike,
    image_size: tuple[int, int],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """``(labels (H, W) int, depth (H, W) float)`` seen from ``T_world_cam``."""
    T_world_cam = np.asarray(T_world_cam, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    w, h = image_size
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    pix = np.stack([u.ravel(), v.ravel(), np.ones(u.size)], axis=1)
    rays_cam = pix @ np.linalg.inv(K).T  # z component is exactly 1
    R = T_world_cam[:3, :3]
    origin = T_world_cam[:3, 3]
    dirs = rays_cam @ R.T

    # ``key`` decides who wins (occluders penalised by the tie-break); ``depth`` stays exact.
    best_key = np.full(u.size, np.inf)
    best_depth = np.full(u.size, np.inf)
    best_label = np.full(u.size, NONE_LABEL, dtype=np.int64)
    for box in _occluders(scene):
        t = _ray_box_t(origin, dirs, box)
        closer = t + _TIE_BREAK_M < best_key
        best_key[closer] = t[closer] + _TIE_BREAK_M
        best_depth[closer] = t[closer]
        best_label[closer] = NONE_LABEL
    for idx, name in enumerate(component_names(scene)):
        t = _ray_box_t(origin, dirs, scene.components[name])
        closer = np.isfinite(t) & (t <= best_key)  # inf <= inf must not label a miss
        best_key[closer] = t[closer]
        best_depth[closer] = t[closer]
        best_label[closer] = idx

    return best_label.reshape(h, w), best_depth.reshape(h, w)


def render_views(scene: Scene, factor: float = 0.5):
    """Render every trajectory pose. Returns ``[(T_cam_world, K, labels, depth), ...]``."""
    K, size = scaled_intrinsics(scene, factor)
    out = []
    for pose in scene.camera_trajectory():
        labels, depth = render(scene, pose.T_world_cam, K, size)
        out.append((invert(pose.T_world_cam), K, labels, depth))
    return out
