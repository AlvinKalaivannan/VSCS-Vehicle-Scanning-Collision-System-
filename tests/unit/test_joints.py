"""P2-T7 joint kinematics: known-answer swings."""

from __future__ import annotations

import numpy as np
import pytest

from vscs.common.config import load_config
from vscs.common.types import JointSpec
from vscs.model.joints import (
    joint_specs_from_config,
    posed,
    rpy_to_R,
    sweep_values,
    to_child_frame,
    validate_joints,
)

HINGE = JointSpec(
    type="revolute",
    parent_link="base_link",
    child_link="door",
    axis=(0.0, 0.0, 1.0),
    origin_xyz=(-1.0, -1.0, 0.0),
    limit_lower=0.0,
    limit_upper=np.pi / 2,
)


def test_closed_points_round_trip_through_the_child_frame():
    p = np.array([[-1.0, -0.5, 1.0], [-1.0, 0.0, 0.5]])
    assert posed(HINGE, to_child_frame(HINGE, p), 0.0) == pytest.approx(p)


def test_a_door_swings_ninety_degrees_about_its_hinge():
    """A point 0.5 m along the door from the hinge ends up 0.5 m straight out behind."""
    closed = np.array([[-1.0, -0.5, 1.0]])  # on the rear face, 0.5 m left of the hinge
    child = to_child_frame(HINGE, closed)
    # +90 deg about +z (counter-clockwise from above): (0, +0.5) -> (-0.5, 0) from the hinge.
    np.testing.assert_allclose(posed(HINGE, child, np.pi / 2), [[-1.5, -1.0, 1.0]], atol=1e-12)
    # The hinge line itself does not move.
    on_axis = to_child_frame(HINGE, np.array([[-1.0, -1.0, 2.0]]))
    np.testing.assert_allclose(posed(HINGE, on_axis, 1.0), [[-1.0, -1.0, 2.0]], atol=1e-12)


def test_limits_are_enforced():
    child = to_child_frame(HINGE, np.array([[-1.0, -0.5, 1.0]]))
    with pytest.raises(ValueError, match="outside"):
        posed(HINGE, child, np.pi)  # past the stop


def test_prismatic_slides_along_its_axis():
    slide = JointSpec(
        type="prismatic",
        parent_link="base_link",
        child_link="step",
        axis=(0.0, -2.0, 0.0),  # not unit length: normalised, so q stays in metres
        origin_xyz=(1.0, -1.0, 0.3),
        limit_lower=0.0,
        limit_upper=0.3,
    )
    p = np.array([[1.2, -1.0, 0.3]])
    np.testing.assert_allclose(
        posed(slide, to_child_frame(slide, p), 0.25), [[1.2, -1.25, 0.3]], atol=1e-12
    )


def test_rpy_is_urdf_fixed_axis_xyz():
    """URDF rpy = Rz(yaw) Ry(pitch) Rx(roll): a pure yaw of 90 deg sends x to y."""
    assert rpy_to_R((0.0, 0.0, np.pi / 2)) @ [1.0, 0.0, 0.0] == pytest.approx([0.0, 1.0, 0.0])
    R = rpy_to_R((0.3, -0.2, 1.1))
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_sweep_covers_the_range():
    q = sweep_values(HINGE, 5)
    assert q[0] == 0.0 and q[-1] == pytest.approx(np.pi / 2) and len(q) == 5


def test_unmeasured_hinges_are_listed_not_invented():
    """The shipped model.yaml has no measured hinge origins yet (P2-T7 measures them)."""
    cfg = load_config("model")
    specs, unmeasured = joint_specs_from_config(cfg["joints"], cfg["urdf"]["base_link"])
    assert specs == {}
    assert set(unmeasured) == {"rear_door_left", "rear_door_right", "left_mirror", "right_mirror"}


def test_config_joints_are_built_once_measured():
    cfg = {
        "m": {
            "type": "revolute",
            "axis": [0, 0, 1],
            "origin_xyz": [2.5, -1.0, 1.5],
            "limit_lower": 0.0,
            "limit_upper": 1.57,
            "child_link": "right_mirror",
        }
    }
    specs, unmeasured = joint_specs_from_config(cfg, "base_link")
    assert unmeasured == [] and specs["m"].child_link == "right_mirror"


def test_validation_catches_joints_without_matching_components():
    """E.g. the config's rear_door_left/right against the vocabulary's single rear_doors."""
    doors = {
        n: JointSpec(type="revolute", parent_link="base_link", child_link=n, limit_upper=1.0)
        for n in ("rear_door_left", "rear_door_right")
    }
    with pytest.raises(ValueError, match=r"rear_door_left.*not a component"):
        validate_joints(doors, {"rear_doors", "rear_bumper"}, "base_link")


def test_validation_catches_two_joints_moving_one_link():
    j = JointSpec(type="revolute", parent_link="base_link", child_link="door", limit_upper=1.0)
    with pytest.raises(ValueError, match="already moved"):
        validate_joints({"a": j, "b": j}, {"door"}, "base_link")
