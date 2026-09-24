"""Transform and projection tests - the P0-T3 acceptance criterion.

These are the guard rails for CLAUDE.md section 4.1. If any of them fails, every
downstream distance and risk number is suspect (risk R-16).
"""

from __future__ import annotations

import numpy as np
import pytest

from vscs.common import frames as F


# --------------------------------------------------------------------------- #
# Rotations                                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fn", [F.rot_x, F.rot_y, F.rot_z])
@pytest.mark.parametrize("angle", [0.0, 0.3, np.pi / 2, np.pi, -1.1, 2 * np.pi])
def test_elementary_rotations_are_rotations(fn, angle):
    assert F.is_rotation(fn(angle))


def test_rot_z_turns_x_toward_y():
    """A positive rotation about +z takes +x toward +y (right-handed)."""
    p = F.rot_z(np.pi / 2) @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(p, [0.0, 1.0, 0.0], atol=1e-12)


def test_is_rotation_rejects_reflection_and_scaling():
    reflection = np.diag([1.0, 1.0, -1.0])  # det = -1
    assert not F.is_rotation(reflection)
    assert not F.is_rotation(2.0 * np.eye(3))
    assert not F.is_rotation(np.eye(2))


@pytest.mark.parametrize(
    "axis,angle",
    [
        ([0, 0, 1], 0.7),
        ([1, 0, 0], -0.4),
        ([1, 1, 1], 2.0),
        ([0, 1, 0], np.pi),  # the degenerate 180 degree case
        ([2, -3, 1], 1.234),
    ],
)
def test_axis_angle_round_trip(axis, angle):
    R = F.axis_angle_to_R(axis, angle)
    assert F.is_rotation(R)
    axis_back, angle_back = F.R_to_axis_angle(R)
    # The same rotation can be reported as (axis, angle) or (-axis, -angle); compare
    # by rebuilding the matrix rather than by comparing the parameters.
    np.testing.assert_allclose(F.axis_angle_to_R(axis_back, angle_back), R, atol=1e-9)


def test_axis_angle_identity_is_zero_angle():
    axis, angle = F.R_to_axis_angle(np.eye(3))
    assert angle == 0.0
    np.testing.assert_allclose(np.linalg.norm(axis), 1.0)


def test_axis_angle_rejects_zero_axis():
    with pytest.raises(ValueError, match="non-zero"):
        F.axis_angle_to_R([0, 0, 0], 1.0)


def test_quaternion_order_is_w_first():
    """A (w, x, y, z) quaternion about +z must equal rot_z.

    If the convention were (x, y, z, w) this test fails, which is the whole point:
    phone IMUs use the other order.
    """
    angle = 0.9
    q = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
    np.testing.assert_allclose(F.quat_to_R(q), F.rot_z(angle), atol=1e-12)


def test_quat_round_trip_and_sign_convention():
    R = F.axis_angle_to_R([1, -2, 0.5], 1.7)
    q = F.R_to_quat(R)
    assert q[0] >= 0.0
    np.testing.assert_allclose(F.quat_to_R(q), R, atol=1e-9)


def test_quat_rejects_zero():
    with pytest.raises(ValueError, match="non-zero"):
        F.quat_to_R([0, 0, 0, 0])


# --------------------------------------------------------------------------- #
# Transforms: inverse and composition (required by CLAUDE.md section 5)        #
# --------------------------------------------------------------------------- #
def _sample_T(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    angle = float(rng.uniform(-np.pi, np.pi))
    t = rng.uniform(-5.0, 5.0, size=3)
    return F.T_from_Rt(F.axis_angle_to_R(axis, angle), t)


@pytest.mark.parametrize("seed", range(6))
def test_invert_round_trip(seed):
    T_a_b = _sample_T(seed)
    T_b_a = F.invert(T_a_b)
    np.testing.assert_allclose(T_a_b @ T_b_a, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(T_b_a @ T_a_b, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(F.invert(T_b_a), T_a_b, atol=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_invert_matches_generic_inverse_but_is_exact(seed):
    """The closed form agrees with a general inverse, and is at least as accurate."""
    T = _sample_T(seed)
    np.testing.assert_allclose(F.invert(T), np.linalg.inv(T), atol=1e-10)
    # Closed form should hit identity essentially exactly.
    assert np.max(np.abs(F.invert(T) @ T - np.eye(4))) <= 1e-14


def test_compose_cancels_inner_subscripts():
    """T_a_b @ T_b_c = T_a_c, and T_a_b @ T_b_a = I."""
    T_a_b = _sample_T(1)
    T_b_c = _sample_T(2)
    T_a_c = F.compose(T_a_b, T_b_c)
    np.testing.assert_allclose(T_a_c, T_a_b @ T_b_c, atol=1e-12)
    np.testing.assert_allclose(F.compose(T_a_b, F.invert(T_a_b)), np.eye(4), atol=1e-12)


def test_compose_is_associative_but_not_commutative():
    A, B, C = _sample_T(3), _sample_T(4), _sample_T(5)
    np.testing.assert_allclose(
        F.compose(F.compose(A, B), C), F.compose(A, F.compose(B, C)), atol=1e-12
    )
    # SO(3) is not commutative; these specific samples must differ.
    assert not np.allclose(F.compose(A, B), F.compose(B, A), atol=1e-6)


def test_compose_of_nothing_is_identity():
    np.testing.assert_allclose(F.compose(), np.eye(4))


def test_compose_rejects_wrong_shape():
    with pytest.raises(ValueError, match="4x4"):
        F.compose(np.eye(3))


def test_T_from_Rt_rejects_non_rotation():
    with pytest.raises(ValueError, match="not a valid rotation"):
        F.T_from_Rt(np.diag([1.0, 1.0, -1.0]), [0, 0, 0])


def test_Rt_from_T_round_trip():
    T = _sample_T(7)
    R, t = F.Rt_from_T(T)
    np.testing.assert_allclose(F.T_from_Rt(R, t), T, atol=1e-15)


def test_Rt_from_T_returns_copies():
    """Mutating the returned R must not corrupt the original transform."""
    T = _sample_T(8)
    R, t = F.Rt_from_T(T)
    R[0, 0] = 99.0
    t[0] = 99.0
    assert T[0, 0] != 99.0
    assert T[0, 3] != 99.0


# --------------------------------------------------------------------------- #
# Point and vector transforms                                                  #
# --------------------------------------------------------------------------- #
def test_transform_points_matches_explicit_homogeneous_product():
    T_a_b = _sample_T(9)
    rng = np.random.default_rng(0)
    P_b = rng.uniform(-3, 3, size=(25, 3))
    expected = (T_a_b @ np.hstack([P_b, np.ones((25, 1))]).T).T[:, :3]
    np.testing.assert_allclose(F.transform_points(T_a_b, P_b), expected, atol=1e-12)


def test_transform_points_accepts_single_point():
    T_a_b = _sample_T(10)
    out = F.transform_points(T_a_b, [1.0, 2.0, 3.0])
    assert out.shape == (3,)
    np.testing.assert_allclose(out, F.transform_points(T_a_b, [[1.0, 2.0, 3.0]])[0])


def test_transform_points_round_trip_through_inverse():
    T_a_b = _sample_T(11)
    rng = np.random.default_rng(1)
    P_b = rng.uniform(-3, 3, size=(10, 3))
    P_a = F.transform_points(T_a_b, P_b)
    np.testing.assert_allclose(F.transform_points(F.invert(T_a_b), P_a), P_b, atol=1e-12)


def test_rotate_vectors_ignores_translation():
    """A direction must be rotated, never translated."""
    R = F.axis_angle_to_R([0, 0, 1], 0.6)
    T_no_t = F.T_from_Rt(R, [0.0, 0.0, 0.0])
    T_with_t = F.T_from_Rt(R, [10.0, -4.0, 3.0])
    v = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    np.testing.assert_allclose(
        F.rotate_vectors(T_no_t, v), F.rotate_vectors(T_with_t, v), atol=1e-15
    )
    np.testing.assert_allclose(F.rotate_vectors(T_with_t, v), v @ R.T, atol=1e-15)


def test_transform_points_rejects_bad_shape():
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        F.transform_points(np.eye(4), np.zeros((4, 2)))


# --------------------------------------------------------------------------- #
# The veh <-> cam convention                                                   #
# --------------------------------------------------------------------------- #
def test_R_CAM_VEH_is_a_rotation():
    assert F.is_rotation(F.R_CAM_VEH)
    assert np.isclose(np.linalg.det(F.R_CAM_VEH), 1.0)


@pytest.mark.parametrize(
    "p_veh,p_cam,what",
    [
        ([1, 0, 0], [0, 0, 1], "veh forward -> cam forward (+z)"),
        ([0, 1, 0], [-1, 0, 0], "veh left -> cam -x, because cam x is right"),
        ([0, 0, 1], [0, -1, 0], "veh up -> cam -y, because cam y is down"),
        ([0, -1, 0], [1, 0, 0], "veh right -> cam +x"),
    ],
)
def test_veh_to_cam_axis_mapping(p_veh, p_cam, what):
    np.testing.assert_allclose(F.R_CAM_VEH @ np.array(p_veh, float), p_cam, atol=1e-15), what


def test_T_CAM_VEH_and_T_VEH_CAM_are_inverses():
    np.testing.assert_allclose(F.T_CAM_VEH @ F.T_VEH_CAM, np.eye(4), atol=1e-15)


def test_a_point_ahead_of_the_van_is_in_front_of_the_camera():
    """Sanity check with real meaning: something 3 m ahead must have cam z = +3."""
    p_cam = F.transform_points(F.T_CAM_VEH, [3.0, 0.0, 0.5])
    assert p_cam[2] == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# Pinhole projection                                                           #
# --------------------------------------------------------------------------- #
K_TEST = np.array([[900.0, 0.0, 639.5], [0.0, 900.0, 359.5], [0.0, 0.0, 1.0]])


def test_projection_round_trip():
    """P0-T3: project then unproject must return the original point."""
    rng = np.random.default_rng(2)
    P_cam = np.column_stack(
        [rng.uniform(-2, 2, 40), rng.uniform(-2, 2, 40), rng.uniform(0.5, 15.0, 40)]
    )
    uv, valid = F.project(K_TEST, P_cam)
    assert valid.all()
    back = F.unproject(K_TEST, uv, P_cam[:, 2])
    np.testing.assert_allclose(back, P_cam, atol=1e-9)


def test_principal_point_projects_to_center():
    uv, valid = F.project(K_TEST, [0.0, 0.0, 5.0])
    assert valid
    np.testing.assert_allclose(uv, [639.5, 359.5], atol=1e-12)


def test_points_behind_camera_are_invalid_and_nan():
    """A bare divide would map z<0 to a finite pixel - the classic occlusion bug."""
    P = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    uv, valid = F.project(K_TEST, P)
    assert list(valid) == [False, False, True]
    assert np.isnan(uv[0]).all()
    assert np.isnan(uv[1]).all()
    assert np.isfinite(uv[2]).all()


def test_image_bounds_filter():
    # Far off-axis at close range lands outside a 1280x720 image.
    P = np.array([[0.0, 0.0, 5.0], [50.0, 0.0, 1.0]])
    _, valid_unbounded = F.project(K_TEST, P)
    _, valid_bounded = F.project(K_TEST, P, image_size=(1280, 720))
    assert valid_unbounded.all()
    assert list(valid_bounded) == [True, False]


def test_unproject_scalar_and_per_pixel_depth_agree():
    uv = np.array([[100.0, 200.0], [900.0, 400.0]])
    a = F.unproject(K_TEST, uv, 4.0)
    b = F.unproject(K_TEST, uv, [4.0, 4.0])
    np.testing.assert_allclose(a, b, atol=1e-15)


def test_unproject_rejects_mismatched_depth_count():
    with pytest.raises(ValueError, match="one value per pixel"):
        F.unproject(K_TEST, np.zeros((3, 2)), [1.0, 2.0])


def test_unproject_depth_is_along_optical_axis():
    """depth is the z coordinate, not the range from the pinhole."""
    p = F.unproject(K_TEST, [0.0, 359.5], 10.0)
    assert p[2] == pytest.approx(10.0)
    assert np.linalg.norm(p) > 10.0  # off-axis: range exceeds depth


def test_project_rejects_bad_K():
    with pytest.raises(ValueError, match="3x3"):
        F.project(np.eye(4), [0, 0, 1])


# --------------------------------------------------------------------------- #
# look_at, used to build the fixture trajectory                                #
# --------------------------------------------------------------------------- #
def test_look_at_builds_a_valid_pose_aimed_at_the_target():
    eye = np.array([6.0, 0.0, 1.5])
    target = np.array([1.5, 0.0, 1.1])
    T_world_cam = F.look_at_T_world_cam(eye, target)
    R, t = F.Rt_from_T(T_world_cam)
    assert F.is_rotation(R)
    np.testing.assert_allclose(t, eye, atol=1e-12)

    # The target must sit on the optical axis: x = y = 0 in cam coordinates, z > 0.
    p_cam = F.transform_points(F.invert(T_world_cam), target)
    assert p_cam[0] == pytest.approx(0.0, abs=1e-9)
    assert p_cam[1] == pytest.approx(0.0, abs=1e-9)
    assert p_cam[2] == pytest.approx(float(np.linalg.norm(target - eye)))


def test_look_at_y_axis_points_down():
    """Camera y must point downward in world terms (OpenCV convention)."""
    T = F.look_at_T_world_cam([5.0, 0.0, 2.0], [0.0, 0.0, 2.0])
    R, _ = F.Rt_from_T(T)
    y_cam_in_world = R[:, 1]
    assert y_cam_in_world[2] < 0.0


def test_look_at_rejects_degenerate_inputs():
    with pytest.raises(ValueError, match="coincide"):
        F.look_at_T_world_cam([1, 1, 1], [1, 1, 1])
    with pytest.raises(ValueError, match="parallel"):
        F.look_at_T_world_cam([0, 0, 5], [0, 0, 0], up_world=[0, 0, 1])
