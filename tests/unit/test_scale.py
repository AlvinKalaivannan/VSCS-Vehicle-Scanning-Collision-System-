"""Scale recovery tests - CLAUDE.md §5's required "scale recovery from a known marker" test.

Written BEFORE the implementation, as the target for the developer's first draft of
``recon/scale.py`` (pair mode). These fail with NotImplementedError until then.

The synthetic setup: three 0.150 m markers lie flat on the ground of the fixture world, seen
by the fixture's 20-camera loop. The "SfM world" is the true world shrunk by an arbitrary
``LAMBDA = 0.37`` - exactly the ambiguity monocular reconstruction leaves - so the correct
scale is ``1 / 0.37``. Before these tests were written, each marker was checked to be fully
visible in 10-20 cameras with baselines of 94-150 degrees, so triangulation is well-posed.
"""

from __future__ import annotations

import numpy as np
import pytest

from fixtures.synthetic import load_scene
from vscs.common.config import load_config
from vscs.common.frames import Rt_from_T, T_from_Rt, invert, project, transform_points
from vscs.recon import scale as S

RECON = load_config("recon")
SIDE = RECON["scale"]["marker"]["side_length_m"]  # 0.150
LAMBDA = 0.37  # the unknown scale SfM leaves behind
MARKERS = {0: (-2.5, -1.8), 1: (1.5, 2.2), 2: (5.0, -1.5)}


def _corners(cx, cy, side=SIDE):
    """Corners in order around the square, flat on the ground."""
    h = side / 2
    return np.array(
        [[cx - h, cy + h, 0.0], [cx + h, cy + h, 0.0], [cx + h, cy - h, 0.0], [cx - h, cy - h, 0.0]]
    )


@pytest.fixture(scope="module")
def setup():
    scene = load_scene()
    K, size = scene.K, scene.image_size
    true_T = [invert(p.T_world_cam) for p in scene.camera_trajectory()]  # T_cam_world
    sfm_T = []
    for T in true_T:
        R, t = Rt_from_T(T)
        sfm_T.append(T_from_Rt(R, LAMBDA * t))  # same images, world shrunk by LAMBDA
    return {"K": K, "size": size, "true_T": true_T, "sfm_T": sfm_T}


def _observations(setup, X_true, poses_key="sfm_T", noise_px=0.0, seed=0):
    """Projection matrices (from the chosen poses) and pixels (from the true scene)."""
    rng = np.random.default_rng(seed)
    Ps, uvs = [], []
    for T_true, T_use in zip(setup["true_T"], setup[poses_key], strict=True):
        uv, ok = project(setup["K"], transform_points(T_true, X_true), image_size=setup["size"])
        if ok:
            Ps.append(S.projection_matrix(setup["K"], T_use))
            uvs.append(uv + rng.normal(0.0, noise_px, 2))
    return Ps, uvs


# --------------------------------------------------------------------------- #
# Projection matrix                                                            #
# --------------------------------------------------------------------------- #
def test_projection_matrix_reproduces_the_pinhole(setup):
    X = np.array([1.0, 0.5, 0.8])
    T = setup["true_T"][3]
    P = S.projection_matrix(setup["K"], T)
    assert P.shape == (3, 4)
    h = P @ np.append(X, 1.0)
    uv_expected, _ = project(setup["K"], transform_points(T, X))
    np.testing.assert_allclose(h[:2] / h[2], uv_expected, atol=1e-9)


# --------------------------------------------------------------------------- #
# Triangulation                                                                #
# --------------------------------------------------------------------------- #
def test_triangulation_is_exact_on_noise_free_data(setup):
    X = np.array([-2.5, -1.8, 0.0])
    Ps, uvs = _observations(setup, X, poses_key="true_T")
    np.testing.assert_allclose(S.triangulate_point(Ps, uvs), X, atol=1e-6)


def test_triangulation_in_the_sfm_world_is_shrunk_by_lambda(setup):
    X = np.array([1.5, 2.2, 0.0])
    Ps, uvs = _observations(setup, X)
    np.testing.assert_allclose(S.triangulate_point(Ps, uvs), LAMBDA * X, atol=1e-6)


def test_two_views_are_enough(setup):
    X = np.array([1.5, 2.2, 0.0])
    Ps, uvs = _observations(setup, X, poses_key="true_T")
    np.testing.assert_allclose(S.triangulate_point(Ps[:2], uvs[:2]), X, atol=1e-6)


def test_one_view_is_a_ray_not_a_point(setup):
    Ps, uvs = _observations(setup, np.array([1.5, 2.2, 0.0]))
    with pytest.raises(ValueError, match=r"two views|2 views|at least"):
        S.triangulate_point(Ps[:1], uvs[:1])


def test_mismatched_inputs(setup):
    Ps, uvs = _observations(setup, np.array([1.5, 2.2, 0.0]))
    with pytest.raises(ValueError):
        S.triangulate_point(Ps[:3], uvs[:2])


def test_triangulation_tolerates_pixel_noise(setup):
    """0.3 px of noise over many views must stay well inside the 2 cm gate."""
    X = np.array([5.0, -1.5, 0.0])
    Ps, uvs = _observations(setup, X, poses_key="true_T", noise_px=0.3, seed=4)
    assert np.linalg.norm(S.triangulate_point(Ps, uvs) - X) < 0.01


# --------------------------------------------------------------------------- #
# Marker size                                                                  #
# --------------------------------------------------------------------------- #
def test_side_length_of_a_perfect_square():
    assert S.marker_side_length(_corners(0.0, 0.0)) == pytest.approx(SIDE, abs=1e-12)


def test_side_length_is_rotation_invariant():
    from vscs.common.frames import axis_angle_to_R

    R = axis_angle_to_R([0.3, 1.0, 0.2], 0.9)
    rotated = _corners(2.0, 1.0) @ R.T + np.array([4.0, -1.0, 2.0])
    assert S.marker_side_length(rotated) == pytest.approx(SIDE, abs=1e-12)


# --------------------------------------------------------------------------- #
# The known-marker test (§5)                                                   #
# --------------------------------------------------------------------------- #
def _reconstructed_sides(setup, sizes=None):
    """Triangulate every corner of every marker in the SfM world; return side lengths."""
    sizes = sizes or {}
    out = {}
    for mid, (cx, cy) in MARKERS.items():
        corners = _corners(cx, cy, sizes.get(mid, SIDE))
        tri = [S.triangulate_point(*_observations(setup, c)) for c in corners]
        out[mid] = S.marker_side_length(np.array(tri))
    return out


def test_recovers_the_known_scale(setup):
    """Headline: three markers of known size recover scale = 1 / 0.37."""
    est = S.estimate_scale(_reconstructed_sides(setup), SIDE, min_markers=3, max_spread_frac=0.02)
    assert est.scale == pytest.approx(1.0 / LAMBDA, rel=1e-6)
    assert est.n_markers == 3 and set(est.per_marker) == set(MARKERS)
    assert est.consistent and est.spread_frac == pytest.approx(0.0, abs=1e-6)


def test_config_wiring(setup):
    est = S.estimate_scale_from_config(_reconstructed_sides(setup), RECON)
    assert est.scale == pytest.approx(1.0 / LAMBDA, rel=1e-6)


def test_a_misprinted_marker_is_flagged_but_does_not_move_the_answer():
    """Marker 2 printed 10% too large. The median ignores it; the range catches it."""
    sides = {0: SIDE * LAMBDA, 1: SIDE * LAMBDA, 2: 1.10 * SIDE * LAMBDA}
    est = S.estimate_scale(sides, SIDE, min_markers=3, max_spread_frac=0.02)
    assert est.scale == pytest.approx(1.0 / LAMBDA, rel=1e-9)
    assert not est.consistent
    assert est.spread_frac == pytest.approx((1 / LAMBDA - 1 / (1.1 * LAMBDA)) * LAMBDA, rel=1e-9)


def test_too_few_markers_is_a_hard_failure():
    """Fewer than min_markers means §8.2's fallback - the developer's call, not a guess."""
    with pytest.raises(ValueError, match="marker"):
        S.estimate_scale({0: 0.05, 1: 0.05}, SIDE, min_markers=3, max_spread_frac=0.02)


def test_non_positive_side_is_rejected():
    with pytest.raises(ValueError):
        S.estimate_scale({0: 0.05, 1: 0.0, 2: 0.05}, SIDE, min_markers=3, max_spread_frac=0.02)


# --------------------------------------------------------------------------- #
# Applying the scale                                                           #
# --------------------------------------------------------------------------- #
def test_scale_points():
    np.testing.assert_allclose(S.scale_points([[1.0, 2.0, 3.0]], 2.5), [[2.5, 5.0, 7.5]])


def test_scaled_pose_keeps_rotation_and_scales_translation(setup):
    T = setup["sfm_T"][6]
    Ts = S.scale_T_cam_world(T, 1.0 / LAMBDA)
    np.testing.assert_allclose(Ts[:3, :3], T[:3, :3], atol=1e-15)
    np.testing.assert_allclose(Ts[:3, 3], T[:3, 3] / LAMBDA, atol=1e-12)


def test_rescaling_the_sfm_world_restores_the_true_one(setup):
    """Scaling poses by s recovers the true cameras exactly, centres included."""
    for T_sfm, T_true in zip(setup["sfm_T"], setup["true_T"], strict=True):
        np.testing.assert_allclose(S.scale_T_cam_world(T_sfm, 1.0 / LAMBDA), T_true, atol=1e-9)


def test_scale_does_not_change_any_image(setup):
    """The whole reason scale is unobservable: scaled points through scaled poses land on
    the same pixels."""
    X = np.array([[1.0, -0.5, 0.7], [3.0, 1.0, 1.9]])
    T = setup["true_T"][2]
    uv1, _ = project(setup["K"], transform_points(T, X))
    uv2, _ = project(
        setup["K"], transform_points(S.scale_T_cam_world(T, 3.3), S.scale_points(X, 3.3))
    )
    np.testing.assert_allclose(uv1, uv2, atol=1e-9)


# --------------------------------------------------------------------------- #
# Validation against the tape (the P1-T6 acceptance figure)                    #
# --------------------------------------------------------------------------- #
def test_dimension_errors_are_signed_and_ignore_unshared_keys():
    rec = {"length": 5.012, "width": 1.991, "height": 2.2}
    tape = {"length": 5.0, "width": 2.0, "height": 2.2, "wheelbase": 3.0}
    errs = S.dimension_errors(rec, tape)
    assert set(errs) == {"length", "width", "height"}
    assert errs["length"] == pytest.approx(0.012)
    assert errs["width"] == pytest.approx(-0.009)


def test_scale_error_is_the_worst_absolute_error():
    rec = {"length": 5.012, "width": 1.975, "height": 2.2}
    tape = {"length": 5.0, "width": 2.0, "height": 2.2}
    assert S.scale_error_m(rec, tape) == pytest.approx(0.025)
    assert S.scale_error_m(rec, tape) > RECON["scale"]["max_dimension_error_m"]


def test_scale_error_needs_something_to_compare():
    with pytest.raises(ValueError):
        S.scale_error_m({"length": 5.0}, {"wheelbase": 3.0})
