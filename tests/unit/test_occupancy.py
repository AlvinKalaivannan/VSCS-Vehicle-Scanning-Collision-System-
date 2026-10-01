"""P4-T3 persistent height map: a pole seen while reversing stays known beside the wheels."""

from __future__ import annotations

import numpy as np
import pytest

from fixtures.synthetic import load_scene
from vscs.common.config import load_config
from vscs.common.frames import T_from_Rt, invert, look_at_T_world_cam
from vscs.perception.occupancy import HeightMap, visible_cells

SCENE = load_scene()
CFG = load_config("perception")["occupancy"]
MAX_RANGE = float(load_config("perception")["depth"]["max_range_m"])
K, SIZE = SCENE.K, SCENE.image_size
#: Rear-facing camera on the van, as in the perception v0 tests.
T_VEH_CAM = look_at_T_world_cam(np.array([-1.0, 0.0, 1.2]), np.array([-3.0, 0.0, 0.0]))
POLE_WORLD = np.array([-3.5, -1.0])  # behind the van, in line with its right side
S = 1_000_000_000


def _T_world_veh(reversed_m: float):
    return T_from_Rt(np.eye(3), [-reversed_m, 0.0, 0.0])


def _pole_points():
    zs = np.arange(0.0, 1.2001, 0.05)
    r = 0.03
    return np.array(
        [
            [POLE_WORLD[0] + dx, POLE_WORLD[1] + dy, z]
            for dx in (-r, r)
            for dy in (-r, r)
            for z in zs
        ]
    )


def _frame(hmap, reversed_m, t_ns, points):
    T_world_veh = _T_world_veh(reversed_m)
    vis = visible_cells(hmap, K, T_world_veh @ T_VEH_CAM, SIZE, MAX_RANGE)
    ij, inside = (
        hmap.cell_of(points[:, :2]) if len(points) else (np.zeros((0, 2), int), np.zeros(0, bool))
    )
    seen = points[inside][vis[ij[inside, 0], ij[inside, 1]]] if len(points) else points
    hmap.update(seen, t_ns, vis)
    return T_world_veh, vis


def test_the_pole_stays_known_beside_the_wheels_after_leaving_view():
    hmap = HeightMap(CFG)
    _, vis0 = _frame(hmap, 0.0, 0, _pole_points())
    i, j = hmap.cell_of(POLE_WORLD)[0][0]
    assert vis0[i, j], "test setup: the rear camera must see the pole at the start"
    # Reverse 3 m: the pole is now level with the rear wheels, out of the rear camera's view.
    T_world_veh, vis1 = _frame(hmap, 3.0, 2 * S, _pole_points())
    assert not vis1[i, j]
    obs = hmap.obstacles(invert(T_world_veh), 2 * S, pos_sigma_m=0.05)
    assert len(obs) == 1
    ob = obs[0]
    # World (-3.5, -1.0) is veh (-0.5, -1.0) after reversing 3 m: beside the rear wheel.
    assert ob.center_veh[0] == pytest.approx(-0.5, abs=CFG["cell_size_m"])
    assert ob.center_veh[1] == pytest.approx(-1.0, abs=CFG["cell_size_m"])
    assert ob.extent[2] == pytest.approx(1.2, abs=1e-9)
    assert ob.source == "occupancy" and ob.kind == "static_geom"


def test_a_visible_empty_cell_is_carved_free():
    """Someone walks through view, then leaves: the map must not keep them."""
    hmap = HeightMap(CFG)
    _frame(hmap, 0.0, 0, _pole_points())
    _frame(hmap, 0.0, S, np.zeros((0, 3)))  # same view, nothing there now
    assert hmap.obstacles(np.eye(4), S, pos_sigma_m=0.05) == []


def test_persistence_expires_and_reset_clears():
    hmap = HeightMap(CFG)
    _frame(hmap, 0.0, 0, _pole_points())
    _frame(hmap, 3.0, S, np.zeros((0, 3)))
    assert len(hmap.obstacles(np.eye(4), S, pos_sigma_m=0.05)) == 1
    late = int((CFG["max_persist_s"] + 1) * S)
    _frame(hmap, 3.0, late, np.zeros((0, 3)))
    assert hmap.obstacles(np.eye(4), late, pos_sigma_m=0.05) == []
    _frame(hmap, 0.0, 0, _pole_points())
    hmap.reset()
    assert hmap.obstacles(np.eye(4), 0, pos_sigma_m=0.05) == []


def test_ground_clutter_and_sky_are_not_obstacles():
    hmap = HeightMap(CFG)
    hmap.update(np.array([[-2.0, 0.0, 0.02], [-2.5, 0.5, 9.0]]), 0)  # paint-thin; above h_max
    assert hmap.obstacles(np.eye(4), 0, pos_sigma_m=0.05) == []


def test_separate_objects_are_separate_obstacles():
    hmap = HeightMap(CFG)
    hmap.update(np.array([[-2.0, 0.0, 0.4], [-2.0, 2.0, 0.3]]), 0)
    obs = sorted(hmap.obstacles(np.eye(4), 0, pos_sigma_m=0.05), key=lambda o: o.center_veh[1])
    assert len(obs) == 2 and [o.id for o in obs] == sorted(o.id for o in obs)
    assert obs[0].extent[2] == pytest.approx(0.4) and obs[1].extent[2] == pytest.approx(0.3)


def test_points_outside_the_grid_are_counted_not_lost_silently():
    hmap = HeightMap(CFG)
    hmap.update(np.array([[100.0, 0.0, 0.5]]), 0)
    assert hmap.dropped_outside == 1


def test_a_rotated_report_is_conservative_never_too_small():
    """The docstring's promise: seen from a vehicle turned 30 deg, the axis-aligned box
    reported in veh still contains every occupied cell of the object."""
    from vscs.common.frames import rot_z

    hmap = HeightMap(CFG)
    xs, ys = np.meshgrid(np.arange(-2.0, -1.0, 0.02), np.arange(0.0, 0.4, 0.02))
    hmap.update(np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 0.5)]), 0)
    T_world_veh = T_from_Rt(rot_z(np.deg2rad(30.0)), [0.5, -0.3, 0.0])
    T_veh_world = invert(T_world_veh)
    (ob,) = hmap.obstacles(T_veh_world, 0, pos_sigma_m=0.05)
    occupied = hmap.cell_centres()[np.isfinite(hmap.height) & (hmap.height > hmap.threshold)]
    pts = (
        np.column_stack([occupied, np.zeros(len(occupied)), np.ones(len(occupied))]) @ T_veh_world.T
    )
    lo = np.array(ob.center_veh[:2]) - np.array(ob.extent[:2]) / 2
    hi = np.array(ob.center_veh[:2]) + np.array(ob.extent[:2]) / 2
    assert (pts[:, :2] >= lo - 1e-9).all() and (pts[:, :2] <= hi + 1e-9).all()
