"""P3-T2 perception v0: ground-plane geometry, known answers on the fixture camera."""

from __future__ import annotations

import numpy as np
import pytest

from fixtures.synthetic import Box, load_scene
from vscs.common.config import load_config
from vscs.common.frames import invert, look_at_T_world_cam, project, transform_points
from vscs.perception.depth import ground_points, obstacle_from_box, range_sigma

SCENE = load_scene()
K = SCENE.K
W, H = SCENE.image_size
CFG = load_config("perception")["depth"]
#: A rear-facing camera on the fixture van: 1.2 m up at the rear face, pitched down.
T_VEH_CAM = look_at_T_world_cam(np.array([-1.0, 0.0, 1.2]), np.array([-3.0, 0.0, 0.0]))


def _pixels(points_veh):
    uv, ok = project(K, transform_points(invert(T_VEH_CAM), np.asarray(points_veh)), (W, H))
    assert ok.all()
    return uv


def _bbox(box):
    uv = _pixels(box.corners())
    return [
        np.floor(uv[:, 0].min()),
        np.floor(uv[:, 1].min()),
        np.ceil(uv[:, 0].max()),
        np.ceil(uv[:, 1].max()),
    ]


def test_ground_points_round_trip_exactly():
    """Project ground points to pixels and back: the ray-plane intersection is exact."""
    xs, ys = np.meshgrid(np.linspace(-2.5, -6.0, 6), np.linspace(-1.0, 1.0, 5))
    truth = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    pts, ok = ground_points(K, T_VEH_CAM, _pixels(truth))
    assert ok.all()
    np.testing.assert_allclose(pts, truth, atol=1e-9)


def test_rays_at_or_above_the_horizon_have_no_ground_point():
    """This camera is pitched down 31 deg and its top row is only 22 deg above the axis, so
    every in-image pixel sees the ground. A ray above the frame (v = -500) points up."""
    pts, ok = ground_points(K, T_VEH_CAM, [[640.0, -500.0]])
    assert not ok[0] and np.isnan(pts[0]).all()


def test_max_range_is_enforced():
    far = _pixels([[-1.0 - 6.0, 0.0, 0.0]])  # 6 m from the camera horizontally
    assert ground_points(K, T_VEH_CAM, far, max_range_m=10.0)[1][0]
    assert not ground_points(K, T_VEH_CAM, far, max_range_m=5.0)[1][0]


def test_range_error_grows_about_with_range_squared():
    """1 px of noise: ~0.9 cm at 3 m, ~3.1 cm at 6 m - a ratio near 4 for 2x the range."""
    s3 = range_sigma(K, T_VEH_CAM, _pixels([[-4.0, 0.0, 0.0]]), 1.0)[0]
    s6 = range_sigma(K, T_VEH_CAM, _pixels([[-7.0, 0.0, 0.0]]), 1.0)[0]
    assert 0.005 < s3 < 0.015
    assert 3.0 < s6 / s3 < 4.5


@pytest.mark.parametrize("x,y", [(-2.5, 0.5), (-3.0, 0.5), (-4.0, -0.8), (-5.0, 0.5)])
def test_a_cone_is_placed_within_the_p3_t2_gate_and_its_near_face_is_accurate(x, y):
    """0.3 x 0.3 x 0.5 m cone. Gate: centre within 0.25 m. The near face - what the van
    would hit - is placed within 3 cm (the box's bottom edge is that face)."""
    cone = Box.from_bounds("cone", [x - 0.15, y - 0.15, 0.0, x + 0.15, y + 0.15, 0.5])
    ob = obstacle_from_box(_bbox(cone), K, T_VEH_CAM, CFG, obstacle_id=7, t_ns=123)
    assert ob is not None and ob.id == 7 and ob.t_ns == 123 and ob.source == "detector"
    assert np.hypot(ob.center_veh[0] - x, ob.center_veh[1] - y) < 0.25
    near_face_x = ob.center_veh[0] + ob.extent[0] / 2.0
    assert near_face_x == pytest.approx(x + 0.15, abs=0.03)
    # Perspective makes width and height come out high: conservative, never too small.
    assert ob.extent[0] >= 0.3 and ob.extent[2] >= 0.5
    assert 0.0 < ob.pos_sigma_m < 0.05


def test_a_box_whose_base_is_above_the_horizon_is_rejected():
    assert (
        obstacle_from_box(
            [600, 10, 640, 20], K, T_VEH_CAM, {**CFG, "max_range_m": 3.0}, obstacle_id=1, t_ns=0
        )
        is None
    )


# --------------------------------------------------------------------------- #
# Mask-based ground contact: the fix for slanted thin objects (R-07)           #
# --------------------------------------------------------------------------- #
from PIL import Image, ImageDraw  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

from vscs.perception.depth import obstacle_from_mask  # noqa: E402


def _mask(box):
    """Silhouette of a convex box as a real mask would be: the filled hull of its projected
    corners, clipped by the image edges (PIL clips the drawing)."""
    uv, ok = project(K, transform_points(invert(T_VEH_CAM), box.corners()))
    assert ok.all()  # in front of the camera; may extend past the frame
    hull = uv[ConvexHull(uv).vertices]
    img = Image.new("1", (W, H), 0)
    ImageDraw.Draw(img).polygon([tuple(p) for p in hull], fill=1)
    return np.asarray(img, dtype=bool)


def test_mask_places_a_thin_off_axis_pole_that_a_box_cannot():
    """6 cm pole off to the side, short enough (0.6 m) to be fully in frame."""
    pole = Box.from_bounds("pole", [-3.53, -1.03, 0.0, -3.47, -0.97, 0.6])
    by_mask = obstacle_from_mask(_mask(pole), K, T_VEH_CAM, CFG, obstacle_id=1, t_ns=0)
    by_box = obstacle_from_box(_bbox(pole), K, T_VEH_CAM, CFG, obstacle_id=1, t_ns=0)
    err_mask = np.hypot(by_mask.center_veh[0] + 3.5, by_mask.center_veh[1] + 1.0)
    err_box = np.hypot(by_box.center_veh[0] + 3.5, by_box.center_veh[1] + 1.0)
    assert err_mask < 0.03 and by_mask.extent[1] < 0.12  # true: centre exact, width 0.06
    assert err_box > 3 * err_mask and by_box.extent[1] > 2 * by_mask.extent[1]
    assert by_mask.extent[2] == pytest.approx(0.6, abs=0.1)


def test_mask_base_is_exact_even_when_the_top_is_cut_off():
    """The end-to-end drive's 1.2 m pole: its top is above the frame, its base is not."""
    pole = Box.from_bounds("pole", [-3.53, -1.03, 0.0, -3.47, -0.97, 1.2])
    m = _mask(pole)
    assert m[0].any(), "setup: the mask touches the image's top row"
    ob = obstacle_from_mask(m, K, T_VEH_CAM, CFG, obstacle_id=1, t_ns=0)
    assert np.hypot(ob.center_veh[0] + 3.5, ob.center_veh[1] + 1.0) < 0.03
    assert ob.extent[1] < 0.12
    assert ob.extent[2] < 1.2  # a lower bound only: the top is out of view


def test_mask_places_a_cone_at_least_as_well_as_a_box():
    cone = Box.from_bounds("cone", [-3.15, 0.35, 0.0, -2.85, 0.65, 0.5])
    ob = obstacle_from_mask(_mask(cone), K, T_VEH_CAM, CFG, obstacle_id=2, t_ns=0)
    assert np.hypot(ob.center_veh[0] + 3.0, ob.center_veh[1] - 0.5) < 0.12
    assert ob.center_veh[0] + ob.extent[0] / 2.0 == pytest.approx(-2.85, abs=0.03)  # near face


def test_masks_with_no_visible_base_are_rejected():
    empty = np.zeros((H, W), dtype=bool)
    assert obstacle_from_mask(empty, K, T_VEH_CAM, CFG, obstacle_id=1, t_ns=0) is None
    touching_bottom = empty.copy()
    touching_bottom[H - 5 :, 600:620] = True
    assert obstacle_from_mask(touching_bottom, K, T_VEH_CAM, CFG, obstacle_id=1, t_ns=0) is None
