"""Ego-motion from the road surface: ground-plane visual odometry (P4-T2, visual half).

Acceptance (CLAUDE.md §6): drift < 5% of distance travelled on dev passes.

Why the ground: one camera alone recovers motion only up to scale. But the camera's height
above the road is known (``capture.yaml mount``), so every pixel *on the road* maps to a
metric ground point (``depth.ground_points``). Two frames, the same road texture seen in
both, two sets of metric points: the van's motion is whatever rotation and translation in
the plane lines them up. No scale ambiguity, no learned model.

Steps, per frame pair
---------------------
1. **Features:** ORB keypoints (OpenCV, Apache-2.0) restricted to pixels that see the
   ground within range, matched by Hamming distance with Lowe's ratio test.
2. **To the ground:** both frames' matched pixels -> ``veh`` ground points ``p_prev``,
   ``p_curr``.
3. **Rigid fit (2D Kabsch / Procrustes, MAT188 least squares):** for a static ground point,
   ``p_prev = R p_curr + t`` where ``(R, t)`` is the van's motion. Centre both sets, form
   ``M = sum (p_curr - c_curr)(p_prev - c_prev)^T``, take its SVD ``M = U S V^T``; then
   ``R = V diag(1, det(V U^T)) U^T`` (the ``det`` term forbids a reflection) and
   ``t = c_prev - R c_curr``.
4. **RANSAC:** fit on random pairs, keep the fit most points agree with (within
   ``ransac_threshold_m``), and refit on those inliers. That rejects moving objects and
   bad matches.

The result is ``T_prev_curr``, the current vehicle pose in the previous one, and
``T_world_curr = T_world_prev @ T_prev_curr``. Assumes a flat road, as all v0 geometry
does. The IMU (gyro yaw rate) is fused at P4-T2 proper, as the "visual-inertial" primary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from vscs.common.frames import T_from_Rt, rot_z
from vscs.perception.depth import ground_points

FloatArray = npt.NDArray[np.float64]


def kabsch_2d(src: npt.ArrayLike, dst: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Least-squares ``(R 2x2, t 2,)`` with ``dst ~ R src + t``. Needs >= 2 points."""
    a = np.asarray(src, dtype=np.float64)
    b = np.asarray(dst, dtype=np.float64)
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    M = (a - ca).T @ (b - cb)
    U, _, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    return R, cb - R @ ca


def ransac_rigid_2d(
    src: npt.ArrayLike, dst: npt.ArrayLike, vo_cfg: dict[str, Any], rng: np.random.Generator
) -> tuple[FloatArray, FloatArray, npt.NDArray[np.bool_]] | None:
    """Robust ``(R, t, inliers)`` with ``dst ~ R src + t``, or ``None`` if too few agree."""
    a = np.asarray(src, dtype=np.float64)
    b = np.asarray(dst, dtype=np.float64)
    n = len(a)
    if n < 2:
        return None
    thr = float(vo_cfg["ransac_threshold_m"])
    best = np.zeros(n, dtype=bool)
    for _ in range(int(vo_cfg["ransac_iterations"])):
        i, j = rng.choice(n, size=2, replace=False)
        if np.linalg.norm(a[i] - a[j]) < 1e-6:
            continue
        R, t = kabsch_2d(a[[i, j]], b[[i, j]])
        inl = np.linalg.norm(a @ R.T + t - b, axis=1) < thr
        if inl.sum() > best.sum():
            best = inl
    if best.sum() < int(vo_cfg["min_inliers"]):
        return None
    R, t = kabsch_2d(a[best], b[best])
    inl = np.linalg.norm(a @ R.T + t - b, axis=1) < thr  # final inliers under the refit
    return R, t, inl


def planar_T(R: FloatArray, t: FloatArray) -> FloatArray:
    """A 2D rigid motion as a 4x4 ``T`` (rotation about z, translation in x-y)."""
    return T_from_Rt(rot_z(float(np.arctan2(R[1, 0], R[0, 0]))), [t[0], t[1], 0.0])


@dataclass
class VOStep:
    T_prev_curr: FloatArray | None  # None: lost (too few inliers)
    n_matches: int
    n_inliers: int


class GroundVO:
    """Ground-plane visual odometry over a sequence of undistorted frames."""

    def __init__(
        self,
        K: npt.ArrayLike,
        T_veh_cam: npt.ArrayLike,
        max_range_m: float,
        vo_cfg: dict[str, Any],
        seed: int,
    ) -> None:
        import cv2

        self.cv2 = cv2
        self.K = np.asarray(K, dtype=np.float64)
        self.T_veh_cam = np.asarray(T_veh_cam, dtype=np.float64)
        self.max_range = float(max_range_m)
        self.cfg = vo_cfg
        self.orb = cv2.ORB_create(nfeatures=int(vo_cfg["orb_features"]))
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.rng = np.random.default_rng(seed)
        self.T_world_veh = np.eye(4)
        self._prev: tuple[Any, Any] | None = None
        self._mask: npt.NDArray | None = None

    def _features(self, gray: npt.NDArray):
        h, w = gray.shape
        if self._mask is None or self._mask.shape != (h, w):
            # Only pixels that see the ground within range carry metric information. The
            # mask depends only on the fixed camera mount, so it is built once.
            step = 8
            vv, uu = np.mgrid[0:h:step, 0:w:step]
            _, ok = ground_points(
                self.K, self.T_veh_cam, np.column_stack([uu.ravel(), vv.ravel()]), self.max_range
            )
            cells = ok.reshape(vv.shape).astype(np.uint8) * 255
            self._mask = np.kron(cells, np.ones((step, step), np.uint8))[:h, :w]
        return self.orb.detectAndCompute(gray, self._mask)

    def step(self, image_rgb: npt.NDArray) -> VOStep:
        gray = self.cv2.cvtColor(image_rgb, self.cv2.COLOR_RGB2GRAY)
        kp, des = self._features(gray)
        prev, self._prev = self._prev, (kp, des)
        if prev is None or des is None or prev[1] is None:
            return VOStep(None, 0, 0)
        pairs = self.matcher.knnMatch(prev[1], des, k=2)
        ratio = float(self.cfg["ratio_test"])
        good = [m for m, *rest in pairs if rest and m.distance < ratio * rest[0].distance]
        if len(good) < 2:
            return VOStep(None, len(good), 0)
        uv_prev = np.array([prev[0][m.queryIdx].pt for m in good])
        uv_curr = np.array([kp[m.trainIdx].pt for m in good])
        g_prev, ok_p = ground_points(self.K, self.T_veh_cam, uv_prev, self.max_range)
        g_curr, ok_c = ground_points(self.K, self.T_veh_cam, uv_curr, self.max_range)
        ok = ok_p & ok_c
        fit = ransac_rigid_2d(g_curr[ok, :2], g_prev[ok, :2], self.cfg, self.rng)
        if fit is None:
            return VOStep(None, int(ok.sum()), 0)
        R, t, inl = fit
        T = planar_T(R, t)
        self.T_world_veh = self.T_world_veh @ T
        return VOStep(T, int(ok.sum()), int(inl.sum()))
