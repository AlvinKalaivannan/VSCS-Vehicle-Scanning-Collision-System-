"""Ground plane and vehicle frame tests (P1-T6 plumbing).

The synthetic reconstruction is the fixture van (body shell plus protruding mirrors) sitting
on a ground plane, placed into an arbitrary "SfM world" by a known rigid transform. The
tests then check the known transform and the known dimensions come back out.
"""

from __future__ import annotations

import numpy as np
import pytest

from fixtures.synthetic import Box, load_scene
from vscs.common.config import load_config
from vscs.common.frames import (
    Rt_from_T,
    T_from_Rt,
    axis_angle_to_R,
    rotate_vectors,
    transform_points,
)
from vscs.recon import ground as G
from vscs.recon import vehicle_frame as V

RECON = load_config("recon")
SEED = 20260924


# --------------------------------------------------------------------------- #
# Synthetic reconstruction                                                     #
# --------------------------------------------------------------------------- #
def _veh_scene(n_body=6000, n_ground=4000, noise=0.002, seed=SEED):
    """Body (shell minus its underside, plus mirrors) and ground, in veh coordinates."""
    rng = np.random.default_rng(seed)
    scene = load_scene()
    shell = scene.vehicle.sample_surface_points(n_body, seed=seed)
    shell = shell[shell[:, 2] > 1e-6]  # the underside is never seen by the camera
    mirrors = np.vstack(
        [
            scene.component(m).sample_surface_points(400, seed=seed + i)
            for i, m in enumerate(("left_mirror", "right_mirror"))
        ]
    )
    body = np.vstack([shell, mirrors]) + rng.normal(0, noise, (len(shell) + len(mirrors), 3))
    gx = rng.uniform(-5.0, 8.0, n_ground)
    gy = rng.uniform(-5.0, 5.0, n_ground)
    ground = np.column_stack([gx, gy, rng.normal(0, noise, n_ground)])
    # Keep ground from under the van, where a real scan cannot see it.
    under = (gx > -1.0) & (gx < 4.0) & (np.abs(gy) < 1.0)
    return body, ground[~under]


def _T_world_veh(seed=3):
    rng = np.random.default_rng(seed)
    R = axis_angle_to_R(rng.normal(size=3), float(rng.uniform(0.3, 2.5)))
    return T_from_Rt(R, rng.uniform(-10, 10, 3))


@pytest.fixture(scope="module")
def world():
    body_v, ground_v = _veh_scene()
    T = _T_world_veh()
    up = rotate_vectors(T, [0.0, 0.0, 1.0])
    return {
        "T_world_veh": T,
        "body": transform_points(T, body_v),
        "ground": transform_points(T, ground_v),
        "up_true": up,
        "front": transform_points(T, [4.0, 0.0, 1.0]),
    }


def _angle_deg(a, b):
    a, b = np.asarray(a) / np.linalg.norm(a), np.asarray(b) / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))


def _tilted(v, deg, seed=9):
    """A unit vector `deg` away from v, as a realistic imperfect up hint."""
    rng = np.random.default_rng(seed)
    perp = np.cross(v, rng.normal(size=3))
    return rotate_vectors(T_from_Rt(axis_angle_to_R(perp, np.radians(deg)), np.zeros(3)), v)


# --------------------------------------------------------------------------- #
# Plane fitting                                                                #
# --------------------------------------------------------------------------- #
def test_lstsq_recovers_an_exact_plane():
    rng = np.random.default_rng(0)
    n = np.array([1.0, 2.0, 2.0]) / 3.0
    basis = np.linalg.svd(n[None, :])[2][1:]
    pts = rng.uniform(-3, 3, (200, 2)) @ basis + 0.7 * n  # plane n.p = 0.7
    plane = G.fit_plane_lstsq(pts)
    assert _angle_deg(plane.normal, n) < 1e-6 or _angle_deg(plane.normal, -n) < 1e-6
    np.testing.assert_allclose(plane.signed_distance(pts), 0.0, atol=1e-9)


def test_lstsq_needs_three_points():
    with pytest.raises(ValueError, match="at least 3"):
        G.fit_plane_lstsq(np.zeros((2, 3)))


def test_ransac_finds_the_ground(world):
    pts = np.vstack([world["ground"], world["body"]])
    up_hint = _tilted(world["up_true"], 10.0)
    plane, mask = G.ground_from_config(pts, RECON, up_hint=up_hint)
    assert _angle_deg(plane.normal, world["up_true"]) < 0.2
    # Every true ground point lies within threshold; the van does not.
    n_ground = len(world["ground"])
    assert mask[:n_ground].mean() > 0.99
    assert mask[n_ground:].mean() < 0.05


def test_normal_is_oriented_up_even_from_a_rough_hint(world):
    pts = np.vstack([world["ground"], world["body"]])
    plane, _ = G.ground_from_config(pts, RECON, up_hint=_tilted(world["up_true"], 25.0))
    assert plane.normal @ world["up_true"] > 0
    assert np.median(plane.signed_distance(world["body"])) > 0, "the van must sit above"


def test_without_a_hint_the_van_side_is_up(world):
    pts = np.vstack([world["ground"], world["body"]])
    plane, _ = G.ransac_plane(pts, distance_threshold_m=0.02, num_iterations=500, seed=SEED)
    assert plane.normal @ world["up_true"] > 0.99


def test_side_panel_trap_needs_the_tilt_limit():
    """A van's flat side can carry more points than the ground. Without the tilt limit
    RANSAC picks the panel; with it, the ground."""
    rng = np.random.default_rng(1)
    ground = np.column_stack([rng.uniform(-5, 5, 1500), rng.uniform(-5, 5, 1500), np.zeros(1500)])
    panel = np.column_stack(
        [rng.uniform(0, 5, 4500), np.full(4500, -1.0), rng.uniform(0.3, 2.2, 4500)]
    )
    pts = np.vstack([ground, panel])
    naive, _ = G.ransac_plane(pts, distance_threshold_m=0.02, num_iterations=400, seed=SEED)
    assert abs(naive.normal[2]) < 0.1, "expected the naive fit to fall into the trap"

    guarded, _ = G.ransac_plane(
        pts,
        distance_threshold_m=0.02,
        num_iterations=400,
        seed=SEED,
        up_hint=_tilted(np.array([0.0, 0.0, 1.0]), 12.0),
        max_tilt_deg=30.0,
    )
    assert guarded.normal[2] > 0.999


def test_ransac_is_deterministic(world):
    pts = np.vstack([world["ground"], world["body"]])
    a, _ = G.ground_from_config(pts, RECON, up_hint=world["up_true"])
    b, _ = G.ground_from_config(pts, RECON, up_hint=world["up_true"])
    np.testing.assert_array_equal(a.normal, b.normal)
    assert a.d == b.d


def test_up_hint_from_the_fixture_cameras():
    """The fixture loop looks slightly down at the van from 1.5 m; up must come out ~+z."""
    Ts = np.stack([p.T_world_cam for p in load_scene().camera_trajectory()])
    assert _angle_deg(G.up_hint_from_cameras(Ts), [0, 0, 1]) < 5.0


def test_plane_helpers():
    plane = G.Plane(np.array([0.0, 0.0, 1.0]), -1.0)  # z = 1
    np.testing.assert_allclose(plane.project([[3.0, 4.0, 7.0]]), [[3.0, 4.0, 1.0]])
    assert plane.flipped().signed_distance([0, 0, 3.0])[0] == pytest.approx(-2.0)
    assert plane.tilt_deg([0, 0, -1]) == pytest.approx(0.0)
    assert plane.tilt_deg([1, 0, 0]) == pytest.approx(90.0)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"max_tilt_deg": 30.0}, "needs an up_hint"),
    ],
)
def test_ransac_argument_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        G.ransac_plane(
            np.random.default_rng(0).normal(size=(20, 3)),
            distance_threshold_m=0.02,
            num_iterations=10,
            seed=0,
            **kwargs,
        )


def test_ransac_fails_loudly_when_no_plane_is_acceptable():
    """All points on a vertical wall, with a strict tilt limit: nothing qualifies."""
    rng = np.random.default_rng(2)
    wall = np.column_stack([rng.uniform(0, 5, 300), np.zeros(300), rng.uniform(0, 2, 300)])
    with pytest.raises(RuntimeError, match="no acceptable plane"):
        G.ransac_plane(
            wall,
            distance_threshold_m=0.02,
            num_iterations=200,
            seed=0,
            up_hint=[0, 0, 1],
            max_tilt_deg=10.0,
        )


def test_unsupported_method():
    cfg = {"ground_plane": {**RECON["ground_plane"], "method": "hough"}}
    with pytest.raises(ValueError, match="unsupported"):
        G.ground_from_config(np.zeros((10, 3)), cfg)


def test_up_hint_rejects_bad_shapes():
    with pytest.raises(ValueError, match=r"\(N, 4, 4\)"):
        G.up_hint_from_cameras(np.eye(4))


# --------------------------------------------------------------------------- #
# Vehicle frame                                                                #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def frame(world):
    pts = np.vstack([world["ground"], world["body"]])
    plane, mask = G.ground_from_config(pts, RECON, up_hint=world["up_true"])
    body = pts[~mask]
    return (
        V.build_vehicle_frame(
            body,
            plane,
            rear_overhang_m=1.0,
            front_hint_world=world["front"],
            width_slice_m=0.10,
            trim_percentile=0.5,
        ),
        plane,
        body,
    )


def test_recovers_the_known_vehicle_frame(world, frame):
    """veh -> world (known) -> veh (estimated) must return the original coordinates."""
    vf, _, _ = frame
    T_est_true = vf.T_veh_world @ world["T_world_veh"]  # should be ~identity
    R, t = Rt_from_T(T_est_true)
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    assert angle < 0.5, f"frame rotated by {angle:.3f} deg"
    assert np.linalg.norm(t) < 0.02, f"origin off by {np.linalg.norm(t) * 100:.1f} cm"


def test_origin_is_on_the_ground_below_the_rear_axle(world, frame):
    """The fixture rear axle is at veh (0, 0, 0): 1.0 m ahead of the rear bumper."""
    vf, _, _ = frame
    true_origin_world = world["T_world_veh"][:3, 3]
    np.testing.assert_allclose(V.to_vehicle_frame(vf, true_origin_world), 0.0, atol=0.02)


def test_dimensions_match_the_fixture_and_exclude_the_mirrors(frame):
    vf, _, _ = frame
    assert vf.length_m == pytest.approx(5.0, abs=0.02)
    assert vf.width_m == pytest.approx(2.0, abs=0.02), "mirrors leaked into the body width"
    assert vf.width_with_mirrors_m == pytest.approx(2.5, abs=0.02)
    assert vf.height_m == pytest.approx(2.2, abs=0.02)
    assert set(vf.dimensions()) == {"length", "width", "height"}


def test_axes_follow_rep_103(world, frame):
    """x toward the front, z up, y = z cross x (left)."""
    vf, _, _ = frame
    R = vf.T_world_veh[:3, :3]
    assert np.linalg.det(R) == pytest.approx(1.0)
    assert R[:, 2] @ world["up_true"] > 0.999
    front_in_veh = V.to_vehicle_frame(vf, world["front"])
    assert front_in_veh[0] > 2.0


def test_front_hint_decides_forward(world, frame):
    """A hint at the rear flips x; the frame must then call the rear the front."""
    _, plane, body = frame
    rear_hint = transform_points(world["T_world_veh"], [-1.0, 0.0, 1.0])
    flipped = V.build_vehicle_frame(body, plane, rear_overhang_m=1.0, front_hint_world=rear_hint)
    assert V.to_vehicle_frame(flipped, rear_hint)[0] > 2.0
    assert flipped.length_m == pytest.approx(5.0, abs=0.02)


def test_rear_overhang_places_the_origin(world, frame):
    _, plane, body = frame
    vf = V.build_vehicle_frame(body, plane, rear_overhang_m=0.5, front_hint_world=world["front"])
    true_axle_world = world["T_world_veh"][:3, 3]
    # The true axle is 0.5 m ahead of where this (wrong) overhang puts the origin.
    assert V.to_vehicle_frame(vf, true_axle_world)[0] == pytest.approx(0.5, abs=0.02)


def test_config_without_measured_overhang_explains_itself(world, frame):
    _, plane, body = frame
    assert RECON["vehicle_frame"]["rear_overhang_m"] is None  # not measured yet
    with pytest.raises(ValueError, match="rear_overhang_m is not set"):
        V.vehicle_frame_from_config(body, plane, RECON, front_hint_world=world["front"])


def test_config_path_works_once_measured(world, frame):
    _, plane, body = frame
    cfg = {**RECON, "vehicle_frame": {**RECON["vehicle_frame"], "rear_overhang_m": 1.0}}
    vf = V.vehicle_frame_from_config(body, plane, cfg, front_hint_world=world["front"])
    assert vf.length_m == pytest.approx(5.0, abs=0.02)


@pytest.mark.parametrize(
    "kwargs,match",
    [({"rear_overhang_m": -1.0}, "hand-measured"), ({"rear_overhang_m": None}, "hand-measured")],
)
def test_bad_overhang_is_rejected(frame, world, kwargs, match):
    _, plane, body = frame
    with pytest.raises(ValueError, match=match):
        V.build_vehicle_frame(body, plane, front_hint_world=world["front"], **kwargs)


def test_too_few_points(frame, world):
    _, plane, _ = frame
    with pytest.raises(ValueError, match="too few"):
        V.build_vehicle_frame(
            np.zeros((3, 3)), plane, rear_overhang_m=1.0, front_hint_world=world["front"]
        )


def test_box_helper_used_by_the_scene_is_the_fixture_box():
    """Sanity: the shell really is 5.0 x 2.0 x 2.2 m, so the dimension answers are right."""
    shell: Box = load_scene().vehicle
    np.testing.assert_allclose(shell.extent, [5.0, 2.0, 2.2])
