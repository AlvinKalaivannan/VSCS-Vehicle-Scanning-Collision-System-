"""The fixture renderer is itself test infrastructure, so it gets known-answer tests."""

from __future__ import annotations

import numpy as np
import pytest

from fixtures.render import NONE_LABEL, component_names, render, scaled_intrinsics
from fixtures.synthetic import load_scene
from vscs.common.frames import invert, look_at_T_world_cam, project, transform_points

SCENE = load_scene()
NAMES = component_names(SCENE)


def _pixel_of(T_world_cam, K, p_world):
    uv, ok = project(K, transform_points(invert(T_world_cam), np.asarray(p_world)[None])[0])
    assert ok
    return round(uv[1]), round(uv[0])


def test_scaled_intrinsics_keeps_pixel_centres_aligned():
    K, size = scaled_intrinsics(SCENE, 0.5)
    assert size == (640, 360)
    # The last full-res pixel centre (1279.5 edge) maps to the last half-res edge.
    assert K[0, 2] == pytest.approx((639.5 + 0.5) * 0.5 - 0.5)


def test_camera_behind_the_van_sees_the_rear_right_corner_at_the_right_depth():
    """Looking straight forward (+x) from 3 m behind the rear face at x = -1.0."""
    eye = np.array([-4.0, -0.85, 0.575])
    T = look_at_T_world_cam(eye, eye + np.array([1.0, 0.0, 0.0]))
    K, size = scaled_intrinsics(SCENE, 0.5)
    labels, depth = render(SCENE, T, K, size)
    # Straight ahead lands on the rear right corner box's back face, 3.0 m away.
    r, c = _pixel_of(T, K, [-1.0, -0.85, 0.575])
    assert NAMES[labels[r, c]] == "rear_right_bumper_corner"
    assert depth[r, c] == pytest.approx(3.0, abs=1e-6)


def test_body_shell_is_none_and_sky_is_inf():
    eye = np.array([-4.0, 0.0, 1.5])
    T = look_at_T_world_cam(eye, eye + np.array([1.0, 0.0, 0.0]))
    K, size = scaled_intrinsics(SCENE, 0.5)
    labels, depth = render(SCENE, T, K, size)
    r, c = _pixel_of(T, K, [-1.0, 0.0, 1.5])  # rear face, above the bumper: bare shell
    assert labels[r, c] == NONE_LABEL and depth[r, c] == pytest.approx(3.0, abs=1e-6)
    assert np.isinf(depth[0, 0]) and labels[0, 0] == NONE_LABEL  # top-left: sky


def test_a_component_inside_the_shell_is_hidden():
    """The underbody (z 0.25-0.30) is never visible from the 1.5 m scan height."""
    K, size = scaled_intrinsics(SCENE, 0.25)
    under = NAMES.index("underbody")
    for pose in SCENE.camera_trajectory():
        labels, _ = render(SCENE, pose.T_world_cam, K, size)
        assert not np.any(labels == under)
