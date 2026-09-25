"""COLMAP text model I/O tests - above all, the pose and quaternion conventions.

The synthetic model is built from the fixture world, so every camera pose is known exactly.
If the world-to-camera direction or the scalar-first quaternion order were wrong anywhere,
the reprojection test below would fail by many pixels.
"""

from __future__ import annotations

import numpy as np
import pytest

from fixtures.colmap_synth import build_fixture_model
from fixtures.synthetic import load_scene
from vscs.common.frames import project, transform_points
from vscs.recon import colmap_io as C


@pytest.fixture(scope="module")
def model():
    return build_fixture_model()


@pytest.fixture(scope="module")
def scene():
    return load_scene()


# --------------------------------------------------------------------------- #
# Conventions                                                                  #
# --------------------------------------------------------------------------- #
def test_camera_centres_match_the_fixture_trajectory(model, scene):
    """COLMAP stores world->camera; the camera centre must come out as -R^T t."""
    for idx, pose in enumerate(scene.camera_trajectory(), start=1):
        np.testing.assert_allclose(model.images[idx].centre, pose.T_world_cam[:3, 3], atol=1e-9)


def test_stored_observations_reproject_exactly(model):
    """The decisive convention test: pose direction and quaternion order together."""
    cam = model.cameras[1]
    for im in model.images.values():
        keep = im.point3D_ids >= 0
        xyz = np.array([model.points[int(p)].xyz for p in im.point3D_ids[keep]])
        uv, _ = project(cam.K, transform_points(im.T_cam_world, xyz))
        np.testing.assert_allclose(uv, im.xys[keep], atol=1e-6)


def test_scalar_last_quaternion_would_be_wrong(model):
    """Guards the (w, x, y, z) order: reading it as (x, y, z, w) must break reprojection."""
    from vscs.common.frames import T_from_Rt, quat_to_R

    cam, im = model.cameras[1], model.images[5]
    keep = im.point3D_ids >= 0
    xyz = np.array([model.points[int(p)].xyz for p in im.point3D_ids[keep]])
    wrong = T_from_Rt(quat_to_R(np.roll(im.qvec, -1)), im.tvec)
    uv, _ = project(cam.K, transform_points(wrong, xyz))
    assert np.nanmax(np.abs(uv - im.xys[keep])) > 10.0


def test_look_at_target_is_on_every_optical_axis(model, scene):
    target = np.array(scene.raw["trajectory"]["look_at"], dtype=float)
    for im in model.images.values():
        p = transform_points(im.T_cam_world, target)
        assert abs(p[0]) < 1e-9 and abs(p[1]) < 1e-9 and p[2] > 0


# --------------------------------------------------------------------------- #
# Round trip                                                                   #
# --------------------------------------------------------------------------- #
def test_write_then_read_round_trips(model, tmp_path):
    back = C.read_model(C.write_model(model, tmp_path / "m"))
    assert set(back.images) == set(model.images)
    assert set(back.points) == set(model.points)
    for i, im in model.images.items():
        np.testing.assert_allclose(back.images[i].T_cam_world, im.T_cam_world, atol=1e-12)
        np.testing.assert_array_equal(back.images[i].point3D_ids, im.point3D_ids)
        np.testing.assert_allclose(back.images[i].xys, im.xys, atol=1e-12)
        assert back.images[i].name == im.name
    for pid, p in model.points.items():
        np.testing.assert_allclose(back.points[pid].xyz, p.xyz, atol=1e-12)
        np.testing.assert_array_equal(back.points[pid].image_ids, p.image_ids)
    np.testing.assert_allclose(back.cameras[1].K, model.cameras[1].K)


def test_untriangulated_observations_keep_id_minus_one(tmp_path):
    """COLMAP marks a 2D feature with no 3D point as -1; that must survive a round trip."""
    cam = C.Camera(1, "SIMPLE_PINHOLE", 640, 480, [500.0, 320.0, 240.0])
    im = C.Image(
        1,
        np.array([1.0, 0, 0, 0]),
        np.zeros(3),
        1,
        "a.png",
        xys=np.array([[10.0, 20.0], [30.0, 40.0]]),
        point3D_ids=np.array([7, -1], dtype=np.int64),
    )
    back = C.read_model(C.write_model(C.Model({1: cam}, {1: im}, {}), tmp_path / "m"))
    np.testing.assert_array_equal(back.images[1].point3D_ids, [7, -1])
    np.testing.assert_allclose(back.images[1].xys, [[10.0, 20.0], [30.0, 40.0]])


def test_image_with_no_observations_parses(tmp_path):
    cam = C.Camera(1, "SIMPLE_PINHOLE", 640, 480, [500.0, 320.0, 240.0])
    im = C.Image(1, np.array([1.0, 0, 0, 0]), np.zeros(3), 1, "a.png")
    im2 = C.Image(2, np.array([1.0, 0, 0, 0]), np.array([1.0, 0, 0]), 1, "b.png")
    d = C.write_model(C.Model({1: cam}, {1: im, 2: im2}, {}), tmp_path / "m")
    back = C.read_model(d)
    assert len(back.images) == 2
    assert back.images[2].xys.shape == (0, 2)


def test_names_with_spaces_survive(tmp_path):
    cam = C.Camera(1, "SIMPLE_PINHOLE", 640, 480, [500.0, 320.0, 240.0])
    im = C.Image(1, np.array([1.0, 0, 0, 0]), np.zeros(3), 1, "frame 0001.png")
    back = C.read_model(C.write_model(C.Model({1: cam}, {1: im}, {}), tmp_path / "m"))
    assert back.images[1].name == "frame 0001.png"


# --------------------------------------------------------------------------- #
# Cameras                                                                      #
# --------------------------------------------------------------------------- #
def test_intrinsics_per_model():
    sp = C.Camera(1, "SIMPLE_PINHOLE", 10, 10, [500.0, 5.0, 6.0])
    assert sp.K[0, 0] == sp.K[1, 1] == 500.0 and sp.K[0, 2] == 5.0
    ph = C.Camera(1, "PINHOLE", 10, 10, [500.0, 510.0, 5.0, 6.0])
    assert ph.K[1, 1] == 510.0
    cv = C.Camera(1, "OPENCV", 10, 10, [500.0, 510.0, 5.0, 6.0, 0.1, -0.2, 0.01, 0.02])
    np.testing.assert_allclose(cv.dist, [0.1, -0.2, 0.01, 0.02])
    sr = C.Camera(1, "SIMPLE_RADIAL", 10, 10, [500.0, 5.0, 6.0, 0.05])
    np.testing.assert_allclose(sr.dist, [0.05, 0, 0, 0])


def test_rejects_unsupported_model_and_wrong_param_count():
    with pytest.raises(ValueError, match="unsupported"):
        C.Camera(1, "FISHEYE_THING", 10, 10, [1.0])
    with pytest.raises(ValueError, match="takes 4 params"):
        C.Camera(1, "PINHOLE", 10, 10, [1.0, 2.0])


def test_binary_model_gives_conversion_advice(tmp_path):
    (tmp_path / "cameras.bin").write_bytes(b"\x00")
    with pytest.raises(FileNotFoundError, match="model_converter"):
        C.read_model(tmp_path)


def test_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="no COLMAP model"):
        C.read_model(tmp_path)


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def test_mean_error_is_weighted_by_track_length():
    """Mean over observations, not over points: (1*2 + 3*6) / 8 = 2.5, not 2.0."""
    pts = {
        1: C.Point3D(
            1, np.zeros(3), np.zeros(3, np.int64), 1.0, np.arange(2), np.zeros(2, np.int64)
        ),
        2: C.Point3D(
            2, np.zeros(3), np.zeros(3, np.int64), 3.0, np.arange(6), np.zeros(6, np.int64)
        ),
    }
    m = C.Model({}, {}, pts)
    assert m.mean_reprojection_error_px() == pytest.approx(2.5)


def test_fixture_model_metrics(model):
    assert model.n_registered == 20
    assert model.mean_reprojection_error_px() == pytest.approx(0.4)
    assert model.registered_fraction(25) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        model.registered_fraction(0)


def test_empty_model_error_is_nan():
    assert np.isnan(C.Model({}, {}, {}).mean_reprojection_error_px())
