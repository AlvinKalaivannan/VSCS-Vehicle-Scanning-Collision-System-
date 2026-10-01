"""P4-T2 ground-plane visual odometry: exact fits, outliers, and drift on a rendered road."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from fixtures.synthetic import load_scene
from vscs.common.config import load_config
from vscs.common.frames import T_from_Rt, invert, look_at_T_world_cam, rot_z
from vscs.perception.egomotion import GroundVO, kabsch_2d, ransac_rigid_2d

SCENE = load_scene()
K = SCENE.K
W, H = SCENE.image_size
PCFG = load_config("perception")
VO = PCFG["egomotion"]["vo"]
SEED = int(PCFG["egomotion"]["seed"])
MAX_RANGE = float(PCFG["depth"]["max_range_m"])
T_VEH_CAM = look_at_T_world_cam(np.array([-1.0, 0.0, 1.2]), np.array([-3.0, 0.0, 0.0]))

#: The road: a textured patch of ground, x in [-14, 4], y in [-8, 8] m, 1 cm per pixel.
RES, X0, Y0 = 0.01, -14.0, -8.0
_rng = np.random.default_rng(7)
TEXTURE = np.full((1600, 1800), 90, np.uint8)
for _ in range(9000):
    c = (int(_rng.integers(0, 1800)), int(_rng.integers(0, 1600)))
    cv2.circle(TEXTURE, c, int(_rng.integers(2, 9)), int(_rng.integers(0, 256)), -1)


def _render(T_world_veh):
    """What the rear camera sees of the textured road from this vehicle pose."""
    T_cam_world = invert(T_world_veh @ T_VEH_CAM)
    R, t = T_cam_world[:3, :3], T_cam_world[:3, 3]
    H_ground = K @ np.column_stack([R[:, 0], R[:, 1], t])  # world (X, Y, 1) on z = 0 -> pixel
    A = np.array([[RES, 0, X0], [0, RES, Y0], [0, 0, 1]])  # texture (col, row, 1) -> world
    gray = cv2.warpPerspective(TEXTURE, H_ground @ A, (W, H), flags=cv2.INTER_LINEAR)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def _pose(x, y, yaw_deg):
    return T_from_Rt(rot_z(np.deg2rad(yaw_deg)), [x, y, 0.0])


def test_kabsch_recovers_an_exact_motion_and_never_reflects():
    rng = np.random.default_rng(1)
    a = rng.uniform(-3, 3, (50, 2))
    th = np.deg2rad(17.0)
    R_true = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    R, t = kabsch_2d(a, a @ R_true.T + [0.4, -1.2])
    np.testing.assert_allclose(R, R_true, atol=1e-12)
    np.testing.assert_allclose(t, [0.4, -1.2], atol=1e-12)
    R_ref, _ = kabsch_2d(a, a * [1, -1])  # a mirrored cloud: best ROTATION, not a reflection
    assert np.linalg.det(R_ref) == pytest.approx(1.0)


def test_ransac_ignores_30_percent_bad_matches():
    rng = np.random.default_rng(2)
    a = rng.uniform(-4, 0, (200, 2))
    b = a + np.array([0.1, 0.0])
    bad = rng.random(200) < 0.3
    b[bad] = rng.uniform(-4, 0, (bad.sum(), 2))
    _, t, inl = ransac_rigid_2d(a, b, VO, np.random.default_rng(SEED))
    np.testing.assert_allclose(t, [0.1, 0.0], atol=1e-9)
    assert not inl[bad].any() and inl[~bad].all()


def test_one_step_motion_is_recovered_from_the_road_texture():
    vo = GroundVO(K, T_VEH_CAM, MAX_RANGE, VO, SEED)
    vo.step(_render(_pose(0.0, 0.0, 0.0)))
    step = vo.step(_render(_pose(-0.10, 0.0, 2.0)))  # reverse 10 cm while turning 2 deg
    assert step.T_prev_curr is not None and step.n_inliers >= VO["min_inliers"]
    np.testing.assert_allclose(step.T_prev_curr[:2, 3], [-0.10, 0.0], atol=0.01)
    yaw = np.rad2deg(np.arctan2(step.T_prev_curr[1, 0], step.T_prev_curr[0, 0]))
    assert yaw == pytest.approx(2.0, abs=0.3)


def test_drift_over_a_two_metre_reversing_arc_is_under_five_percent():
    """P4-T2 acceptance shape: drift < 5% of distance travelled."""
    vo = GroundVO(K, T_VEH_CAM, MAX_RANGE, VO, SEED)
    x, y, yaw, travelled = 0.0, 0.0, 0.0, 0.0
    for k in range(21):
        vo.step(_render(_pose(x, y, yaw)))
        if k < 20:  # reverse 10 cm per frame with a gentle left turn
            step = 0.10
            x -= step * np.cos(np.deg2rad(yaw))
            y -= step * np.sin(np.deg2rad(yaw))
            yaw += 1.0
            travelled += step
    true = _pose(x, y, yaw)  # the last pose rendered (no update after the final frame)
    err = np.linalg.norm(vo.T_world_veh[:2, 3] - true[:2, 3])
    assert err < 0.05 * travelled  # < 5% of the 2.0 m travelled


def test_a_lost_frame_coasts_on_the_last_motion():
    vo = GroundVO(K, T_VEH_CAM, MAX_RANGE, VO, SEED)
    vo.step(_render(_pose(0.0, 0.0, 0.0)))
    vo.step(_render(_pose(-0.10, 0.0, 0.0)))  # learn a 10 cm reverse
    before = vo.T_world_veh.copy()
    blank = np.zeros((H, W, 3), np.uint8)
    step = vo.step(blank)  # no texture: lost
    assert step.T_prev_curr is None and vo.lost == 1
    np.testing.assert_allclose(vo.T_world_veh[:2, 3] - before[:2, 3], [-0.10, 0.0], atol=0.01)
