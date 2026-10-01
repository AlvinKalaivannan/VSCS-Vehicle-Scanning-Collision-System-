"""P4-T6: the kerb / speed-bump fixture is flagged for the correct component."""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import box

from fixtures.synthetic import load_scene
from vscs.common.config import load_config
from vscs.perception.occupancy import HeightMap
from vscs.risk.motion import path_fan_from_config
from vscs.risk.underbody import ground_cells, underbody_check

SCENE = load_scene()
RISK = load_config("risk")
UB = RISK["underbody"]
OCC = load_config("perception")["occupancy"]
CELL = float(OCC["cell_size_m"])


def _footprints():
    return {
        n: (box(b.lo[0], b.lo[1], b.hi[0], b.hi[1]), float(b.lo[2]), float(b.hi[2]))
        for n, b in SCENE.components.items()
    }


def _map_box(lo, hi):
    """A height map holding one box obstacle, sampled densely on its top."""
    hmap = HeightMap(OCC)
    xs = np.arange(lo[0] + CELL / 2, hi[0], CELL / 2)
    ys = np.arange(lo[1] + CELL / 2, hi[1], CELL / 2)
    X, Y = np.meshgrid(xs, ys)
    hmap.update(np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, hi[2])]), 0)
    return ground_cells(hmap, np.eye(4))


def _reverse(cells, speed=-1.0):
    fan = path_fan_from_config(speed, RISK)
    return underbody_check(_footprints(), *cells, fan, fan.index_of_curvature(0.0), UB), fan


def test_reversing_onto_the_kerb_flags_the_rear_wheels_only():
    """Kerb 0.12 m high, its near face 1.0 m behind the bumper (x = -2.0)."""
    res, fan = _reverse(_map_box(SCENE.curb.lo, SCENE.curb.hi))
    flagged = {n for n, r in res.items() if r.ttc_s is not None}
    reach = fan.t_s[-1] * abs(fan.speed_mps)
    if reach < 1.65 - CELL:
        pytest.skip(f"configured horizon reaches only {reach:.2f} m")
    assert flagged == {"wheel_rear_left", "wheel_rear_right"}
    # Rear wheels' back edge (x = -0.35) reaches the kerb face (x = -2.0) after 1.65 m.
    dt = float(fan.t_s[1] - fan.t_s[0])
    assert res["wheel_rear_right"].ttc_s == pytest.approx(1.65, abs=dt + CELL)


def test_the_underbody_and_bumper_clear_the_kerb_by_the_known_margins():
    res, fan = _reverse(_map_box(SCENE.curb.lo, SCENE.curb.hi))
    if fan.t_s[-1] * abs(fan.speed_mps) < 2.0 + CELL:
        pytest.skip("horizon too short for the underbody to reach the kerb")
    assert res["underbody"].ttc_s is None
    assert res["underbody"].min_clearance_m == pytest.approx(0.25 - 0.12)
    assert res["rear_bumper"].min_clearance_m == pytest.approx(0.40 - 0.12)


def test_a_taller_bump_reaches_the_underbody():
    bump = ([-3.0, -0.5, 0.0], [-2.0, 0.5, 0.30])  # 0.30 m: above the underbody's 0.25 m
    res, fan = _reverse(_map_box(*bump))
    if fan.t_s[-1] * abs(fan.speed_mps) < 2.0 + CELL:
        pytest.skip("horizon too short")
    assert res["underbody"].ttc_s is not None
    assert res["underbody"].min_clearance_m == pytest.approx(0.25 - 0.30)
    assert res["rear_bumper"].ttc_s is None  # 0.40 m: clears 0.30 m


def test_high_components_are_not_checked_and_an_empty_map_is_clear():
    res, _ = _reverse((np.zeros((0, 2)), np.zeros(0)))
    assert "right_mirror" not in res and "left_mirror" not in res  # min_z 1.40 m > 0.60 m
    assert "sliding_door_right" in res  # its lower edge is at 0.50 m: a bollard could reach it
    assert all(r.ttc_s is None and math.isinf(r.min_clearance_m) for r in res.values())
