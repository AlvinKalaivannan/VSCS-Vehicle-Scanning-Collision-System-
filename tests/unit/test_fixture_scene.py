"""Fixture world tests - the P0-T4 acceptance criterion.

"Fixture loads; known-distance test passes" (CLAUDE.md section 6). The headline
assertion is :func:`test_known_distance_rear_right_corner_to_pole`, which is the
0.500 m case named in section 5.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from fixtures.synthetic import Box, load_scene
from vscs.common import frames as F
from vscs.common.types import validate_timestamp_stream

EXACT = 1e-6  # the tolerance CLAUDE.md section 5 asks for


@pytest.fixture(scope="module")
def scene():
    return load_scene()


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #
def test_scene_loads(scene):
    assert scene.raw["schema_version"] == 1
    assert len(scene.components) >= 8, "P2-T1 expects at least 8 components"


def test_vehicle_dimensions_match_declared_bounds(scene):
    v = scene.vehicle
    assert v.extent[0] == pytest.approx(scene.raw["vehicle"]["length_m"], abs=EXACT)
    assert v.extent[1] == pytest.approx(scene.raw["vehicle"]["width_m"], abs=EXACT)
    assert v.extent[2] == pytest.approx(scene.raw["vehicle"]["height_m"], abs=EXACT)


def test_frame_origin_convention(scene):
    """Origin on the ground plane below the rear axle centre (section 4.1).

    So: z starts at 0, the body is centred in y, the rear axle is at x = 0, and the
    front axle is one wheelbase ahead of it.
    """
    v = scene.vehicle
    assert v.lo[2] == pytest.approx(0.0, abs=EXACT)
    assert v.lo[1] == pytest.approx(-v.hi[1], abs=EXACT)
    rear_axle_x = scene.component("wheel_rear_right").center[0]
    front_axle_x = scene.component("wheel_front_right").center[0]
    assert rear_axle_x == pytest.approx(0.0, abs=EXACT)
    assert front_axle_x - rear_axle_x == pytest.approx(scene.wheelbase_m, abs=EXACT)


def test_components_lie_within_vehicle_bounds_except_mirrors(scene):
    """Mirrors protrude past the body shell on purpose; nothing else may."""
    for name, box in scene.components.items():
        if "mirror" in name:
            continue
        assert np.all(box.lo >= scene.vehicle.lo - EXACT), name
        assert np.all(box.hi <= scene.vehicle.hi + EXACT), name


def test_mirrors_protrude_beyond_the_body(scene):
    """This is precisely what a single bounding box cannot represent."""
    assert scene.component("right_mirror").lo[1] < scene.vehicle.lo[1]
    assert scene.component("left_mirror").hi[1] > scene.vehicle.hi[1]


def test_components_do_not_overlap(scene):
    """Touching is allowed (distance 0), sharing volume is not."""
    for (n1, b1), (n2, b2) in itertools.combinations(scene.components.items(), 2):
        overlap = np.minimum(b1.hi, b2.hi) - np.maximum(b1.lo, b2.lo)
        assert np.any(overlap <= EXACT), f"{n1} and {n2} share volume: {overlap}"


def test_unknown_component_gives_helpful_error(scene):
    with pytest.raises(KeyError, match="no fixture component"):
        scene.component("spoiler")


# --------------------------------------------------------------------------- #
# The known-distance test (P0-T4 acceptance)                                   #
# --------------------------------------------------------------------------- #
def test_known_distance_rear_right_corner_to_pole(scene):
    """Rear right bumper corner to pole axis is exactly 0.500 m.

    Corner box nearest point is (x=-1.00, y=-1.00); the pole axis is at
    (x=-1.50, y=-1.00). The separation is purely along x, so the answer is exact
    with no floating-point slack.
    """
    corner = scene.component("rear_right_bumper_corner")
    assert scene.pole.distance_to_box_axis(corner) == pytest.approx(0.500, abs=EXACT)


def test_known_distance_to_pole_surface(scene):
    """Axis distance minus the 0.03 m pole radius."""
    corner = scene.component("rear_right_bumper_corner")
    assert scene.pole.distance_to_box_surface(corner) == pytest.approx(0.470, abs=EXACT)


def test_pole_threatens_the_rear_right_corner_and_nothing_closer(scene):
    """Ground truth for component attribution accuracy (section 11)."""
    name, dist = scene.nearest_component_to_pole()
    assert name == "rear_right_bumper_corner"
    assert dist == pytest.approx(0.500, abs=EXACT)


def test_known_distance_wheel_to_curb(scene):
    """Rear right wheel to kerb: 1.650 m, exact.

    The wheel spans x in [-0.35, 0.35] and the kerb's near face is at x = -2.00, so
    dx = 1.65. They overlap in y, and in z (both start at the ground), so the 3D
    distance reduces to that single axis.
    """
    wheel = scene.component("wheel_rear_right")
    assert wheel.distance_to_box(scene.curb) == pytest.approx(1.650, abs=EXACT)


def test_known_horizontal_distance_bumper_to_curb(scene):
    """Rear bumper footprint to kerb footprint: exactly 1.000 m.

    The full 3D distance is larger (about 1.038 m) because the bumper sits 0.28 m
    above the top of the kerb. For a kerb the horizontal gap is the meaningful
    number, which is why both queries exist.
    """
    bumper = scene.component("rear_bumper")
    assert bumper.distance_to_box_horizontal(scene.curb) == pytest.approx(1.000, abs=EXACT)
    assert bumper.distance_to_box(scene.curb) == pytest.approx(
        np.hypot(1.0, 0.40 - 0.12), abs=EXACT
    )


def test_underbody_clears_the_curb(scene):
    """Underbody at 0.25 m clears a 0.12 m kerb, by 0.13 m (P4-T6 fixture case)."""
    assert scene.component("underbody").min_z - float(scene.curb.hi[2]) == pytest.approx(
        0.13, abs=EXACT
    )


def test_distance_is_zero_inside_and_symmetric(scene):
    box = scene.component("rear_bumper")
    assert box.distance_to_point(box.center) == 0.0
    assert box.distance_to_box(box) == 0.0
    other = scene.curb
    assert box.distance_to_box(other) == pytest.approx(other.distance_to_box(box), abs=1e-15)


def test_nearest_point_is_clamped_onto_the_box(scene):
    box = scene.component("right_mirror")
    far = np.array([100.0, -100.0, 100.0])
    near = box.nearest_point_to(far)
    np.testing.assert_allclose(near, box.hi * [1, 0, 1] + box.lo * [0, 1, 0], atol=EXACT)
    assert box.distance_to_point(far) == pytest.approx(float(np.linalg.norm(far - near)))


# --------------------------------------------------------------------------- #
# Box geometry helpers                                                         #
# --------------------------------------------------------------------------- #
def test_box_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="inverted"):
        Box.from_bounds("bad", [1, 1, 1, 0, 0, 0])


def test_box_rejects_wrong_length():
    with pytest.raises(ValueError, match="6 numbers"):
        Box.from_bounds("bad", [0, 0, 0])


def test_box_volume_and_corners(scene):
    box = Box.from_bounds("unit", [0, 0, 0, 2, 3, 4])
    assert box.volume == pytest.approx(24.0)
    assert box.corners().shape == (8, 3)
    # Every corner must be at the boundary in all three axes.
    for c in box.corners():
        assert np.all((np.isclose(c, box.lo)) | (np.isclose(c, box.hi)))


def test_surface_sampling_is_on_the_surface_and_deterministic():
    box = Box.from_bounds("b", [0, 0, 0, 1, 2, 3])
    a = box.sample_surface_points(300, seed=7)
    b = box.sample_surface_points(300, seed=7)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, box.sample_surface_points(300, seed=8))
    # Inside the box, and touching a face in at least one axis.
    assert np.all(a >= box.lo - EXACT) and np.all(a <= box.hi + EXACT)
    on_face = np.isclose(a, box.lo).any(axis=1) | np.isclose(a, box.hi).any(axis=1)
    assert on_face.all()


# --------------------------------------------------------------------------- #
# Trajectory                                                                   #
# --------------------------------------------------------------------------- #
def test_trajectory_has_the_declared_number_of_poses(scene):
    assert len(scene.camera_trajectory()) == scene.raw["trajectory"]["n_frames"] == 20


def test_trajectory_timestamps_are_valid_and_jittered(scene):
    ts = [p.t_ns for p in scene.camera_trajectory()]
    assert all(isinstance(t, int) for t in ts)
    validate_timestamp_stream(ts)
    gaps = np.diff(ts)
    assert gaps.min() > 0
    # Deliberately variable frame rate (R-03): a constant-interval assumption fails.
    assert len(set(gaps.tolist())) > 1
    assert gaps.max() > gaps.min() * 1.1


def test_trajectory_is_deterministic(scene):
    a = [p.t_ns for p in scene.camera_trajectory()]
    b = [p.t_ns for p in scene.camera_trajectory()]
    assert a == b


def test_trajectory_poses_are_valid_and_aimed_at_the_vehicle(scene):
    target = np.array(scene.raw["trajectory"]["look_at"], dtype=float)
    radius = scene.raw["trajectory"]["radius_m"]
    center = np.array(scene.raw["trajectory"]["center_xy"], dtype=float)

    for pose in scene.camera_trajectory():
        R, t = F.Rt_from_T(pose.T_world_cam)
        assert F.is_rotation(R)
        # On the declared circle at the declared height.
        assert float(np.linalg.norm(t[:2] - center)) == pytest.approx(radius, abs=1e-9)
        assert t[2] == pytest.approx(scene.raw["trajectory"]["height_m"], abs=1e-12)
        # Target on the optical axis, in front of the camera.
        p_cam = F.transform_points(F.invert(pose.T_world_cam), target)
        assert p_cam[0] == pytest.approx(0.0, abs=1e-9)
        assert p_cam[1] == pytest.approx(0.0, abs=1e-9)
        assert p_cam[2] > 0


def test_vehicle_is_visible_from_every_pose(scene):
    """At least some vehicle corners must land inside the image in every frame.

    A fixture whose subject falls outside the image would make later fusion tests
    pass for the wrong reason.
    """
    corners = scene.vehicle.corners()
    for i, pose in enumerate(scene.camera_trajectory()):
        p_cam = F.transform_points(F.invert(pose.T_world_cam), corners)
        _, valid = F.project(scene.K, p_cam, image_size=scene.image_size)
        assert valid.sum() >= 2, f"frame {i}: only {valid.sum()} vehicle corners visible"


# --------------------------------------------------------------------------- #
# Labelled point cloud (stand-in for the P2-T4 hand-labelled subset)           #
# --------------------------------------------------------------------------- #
def test_labelled_point_cloud_covers_every_component(scene):
    pts, labels = scene.labelled_point_cloud(points_per_component=25)
    assert pts.shape == (25 * len(scene.components), 3)
    assert set(labels) == set(scene.components)


def test_labelled_points_lie_on_their_own_component(scene):
    pts, labels = scene.labelled_point_cloud(points_per_component=25)
    for name in scene.components:
        box = scene.component(name)
        mine = pts[labels == name]
        dists = [box.distance_to_point(p) for p in mine]
        assert max(dists) < EXACT, f"{name}: labelled point off its own box"


def test_labelled_point_cloud_is_deterministic(scene):
    a, la = scene.labelled_point_cloud(points_per_component=10, seed=3)
    b, lb = scene.labelled_point_cloud(points_per_component=10, seed=3)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(la, lb)
