"""Swept-clearance tests - the P3-T4 acceptance criterion ("fixture known-answer tests pass").

Written BEFORE the implementation, as the target for the developer's first draft of
``risk/sweep.py`` (pair mode). These fail with NotImplementedError until then.

Known answers come from the fixture world (``tests/fixtures/scene.yaml``) and the margins
in ``configs/risk.yaml``. Where no closed form exists (arcs), results are checked against
``_oracle``: a brute-force reference that samples the continuous motion model every
millisecond. The oracle is deliberately naive - it exists to be obviously correct, not
fast, and is not a template for the implementation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely import affinity

from fixtures.synthetic import load_scene
from vscs.common.config import load_config
from vscs.common.types import Obstacle
from vscs.risk import sweep as S
from vscs.risk.motion import constant_curvature_pose, path_fan

CFG = load_config("risk")
M_C = CFG["sweep"]["component_margin_m"]  # 0.05
M_O = CFG["sweep"]["obstacle_margin_m"]  # 0.05
TOL = CFG["sweep"]["ttc_tolerance_s"]  # 0.005


@pytest.fixture(scope="module")
def scene():
    return load_scene()


def _component(scene, name) -> S.Footprint:
    box = scene.component(name)
    return S.box_footprint(name, box.lo, box.hi)


def _all_components(scene) -> list[S.Footprint]:
    return [_component(scene, n) for n in scene.components]


def _obstacle(oid, lo, hi, kind="static_geom") -> Obstacle:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c, e = (lo + hi) / 2, hi - lo
    return Obstacle(
        id=oid,
        t_ns=0,
        kind=kind,
        center_veh=tuple(c),
        extent=tuple(e),
        pos_sigma_m=0.05,
        source="occupancy",
    )


def _pole(scene) -> S.Footprint:
    """The fixture pole as a 0.06 m square post, axis at (-1.5, -1.0), 1.2 m tall."""
    (ax, ay), r, h = scene.pole.axis_xy, scene.pole.radius_m, scene.pole.height_m
    return S.obstacle_footprint(_obstacle(1, [ax - r, ay - r, 0.0], [ax + r, ay + r, h]))


def _kerb(scene) -> S.Footprint:
    return S.obstacle_footprint(_obstacle(2, scene.curb.lo, scene.curb.hi))


def _fan(speed, kappas=(0.0,), horizon=3.0, dt=0.1):
    return path_fan(speed, horizon_s=horizon, dt_s=dt, curvatures=list(kappas))


def _oracle(comp, obst, speed, kappa, horizon, m_c, m_o, dt=1e-3):
    """Brute-force reference: sample the continuous path every `dt` seconds."""
    P, obs = comp.polygon.buffer(m_c), obst.polygon.buffer(m_o)
    ttc, d_min = None, math.inf
    for t in np.arange(0.0, horizon + dt / 2, dt):
        x, y, h = (float(v) for v in constant_curvature_pose(kappa, speed * t))
        c, s = math.cos(h), math.sin(h)
        A = affinity.affine_transform(P, [c, -s, s, c, x, y])
        g = A.distance(obs)
        d_min = min(d_min, g)
        if ttc is None and g <= 0.0:
            ttc = float(t)
    return d_min, ttc


# --------------------------------------------------------------------------- #
# Footprints and the height gate                                               #
# --------------------------------------------------------------------------- #
def test_box_footprint_geometry(scene):
    fp = _component(scene, "rear_right_bumper_corner")
    assert fp.polygon.bounds == pytest.approx((-1.00, -1.00, -0.85, -0.70))
    assert (fp.z_min, fp.z_max) == pytest.approx((0.40, 0.75))
    assert fp.obstacle_id is None


def test_obstacle_footprint_uses_full_extent_and_keeps_the_id():
    fp = S.obstacle_footprint(_obstacle(7, [1.0, 2.0, 0.0], [3.0, 3.0, 0.5]))
    assert fp.polygon.bounds == pytest.approx((1.0, 2.0, 3.0, 3.0))
    assert (fp.z_min, fp.z_max) == pytest.approx((0.0, 0.5))
    assert fp.obstacle_id == 7


def test_z_overlap(scene):
    kerb, pole = _kerb(scene), _pole(scene)
    assert S.z_overlap(_component(scene, "wheel_rear_right"), kerb)  # 0-0.7 vs 0-0.12
    assert not S.z_overlap(_component(scene, "right_mirror"), kerb)  # 1.4-1.6 vs 0-0.12
    assert not S.z_overlap(_component(scene, "rear_bumper"), kerb)  # 0.40 above 0.12
    # A 1.2 m post cannot clip a mirror mounted at 1.40-1.60 m.
    assert not S.z_overlap(_component(scene, "right_mirror"), pole)
    touching = S.box_footprint("t", [0, 0, 0.12], [1, 1, 0.5])
    assert S.z_overlap(touching, kerb), "touching ranges count as overlapping"


def test_height_gate_returns_none(scene):
    fan = _fan(-1.0)
    res = S.sweep_pair(
        _component(scene, "rear_bumper"),
        _kerb(scene),
        fan,
        0,
        component_margin_m=M_C,
        obstacle_margin_m=M_O,
        ttc_tolerance_s=TOL,
    )
    assert res is None, "a bumper 0.40 m up cannot strike a 0.12 m kerb"


# --------------------------------------------------------------------------- #
# The headline known answer                                                    #
# --------------------------------------------------------------------------- #
def _pair(scene, comp, obst, speed, kappas=(0.0,), m_c=M_C, m_o=M_O, **fan_kw):
    fan = _fan(speed, kappas, **fan_kw)
    return S.sweep_pair(
        _component(scene, comp),
        obst,
        fan,
        0,
        component_margin_m=m_c,
        obstacle_margin_m=m_o,
        ttc_tolerance_s=TOL,
    )


def test_corner_to_pole_ttc_is_0_37_s(scene):
    """0.5 m axis - 0.03 radius - 0.05 - 0.05 margins = 0.37 m at 1 m/s reversing."""
    res = _pair(scene, "rear_right_bumper_corner", _pole(scene), -1.0)
    assert res is not None
    assert res.ttc_s == pytest.approx(0.37, abs=TOL)
    assert res.min_distance_m == 0.0
    assert res.obstacle_id == 1
    assert res.curvature == 0.0


def test_ttc_is_not_quantised_to_dt(scene):
    """Per-sample checking would say 0.40 s. Refinement must beat that."""
    res = _pair(scene, "rear_right_bumper_corner", _pole(scene), -1.0)
    assert abs(res.ttc_s - 0.40) > 0.02


def test_without_margins_ttc_is_0_47_s(scene):
    res = _pair(scene, "rear_right_bumper_corner", _pole(scene), -1.0, m_c=0.0, m_o=0.0)
    assert res.ttc_s == pytest.approx(0.47, abs=TOL)


def test_ttc_scales_with_speed(scene):
    res = _pair(scene, "rear_right_bumper_corner", _pole(scene), -2.0)
    assert res.ttc_s == pytest.approx(0.185, abs=TOL)


def test_driving_away_never_contacts(scene):
    """Forward at 1 m/s moves the corner away from the pole behind it."""
    res = _pair(scene, "rear_right_bumper_corner", _pole(scene), +1.0)
    assert res.ttc_s is None
    assert res.min_distance_m == pytest.approx(0.37, abs=1e-6)  # closest at t = 0


def test_already_in_contact_gives_ttc_zero(scene):
    touching = S.obstacle_footprint(_obstacle(9, [-1.10, -0.95, 0.0], [-1.02, -0.90, 1.0]))
    res = _pair(scene, "rear_right_bumper_corner", touching, 0.0)
    assert res.ttc_s == 0.0
    assert res.min_distance_m == 0.0


def test_stationary_vehicle_reports_the_static_gap(scene):
    res = _pair(scene, "rear_right_bumper_corner", _pole(scene), 0.0)
    assert res.ttc_s is None
    assert res.min_distance_m == pytest.approx(0.37, abs=1e-6)


def test_wheel_reaches_the_kerb_at_1_55_s(scene):
    """Wheel back face x = -0.35, kerb face x = -2.0: 1.65 - 0.10 margins = 1.55 m."""
    res = _pair(scene, "wheel_rear_right", _kerb(scene), -1.0)
    assert res.ttc_s == pytest.approx(1.55, abs=TOL)
    assert res.obstacle_id == 2


def test_lateral_clearance_is_constant_on_a_straight_path(scene):
    """rear_bumper spans |y| <= 0.70; the pole's inflated edge is 0.17 m clear of it."""
    res = _pair(scene, "rear_bumper", _pole(scene), -1.0)
    assert res.ttc_s is None
    assert res.min_distance_m == pytest.approx(0.17, abs=1e-6)


# --------------------------------------------------------------------------- #
# Tunnelling                                                                   #
# --------------------------------------------------------------------------- #
def test_thin_obstacle_between_samples_is_not_missed(scene):
    """At 10 m/s and dt = 0.1 s the corner moves 1 m per step.

    A 4 cm wall 0.48 m behind the corner is clear at t = 0 (gap 0.48) and already passed
    at t = 0.1 (gap 0.33). Both samples show clearance; only the swept region shows the
    strike. True contact is at 0.048 s.
    """
    wall = S.obstacle_footprint(_obstacle(5, [-1.52, -0.90, 0.0], [-1.48, -0.80, 1.0]))
    res = _pair(scene, "rear_right_bumper_corner", wall, -10.0, m_c=0.0, m_o=0.0, horizon=1.0)
    assert res.ttc_s is not None, "tunnelled straight through the wall"
    assert res.ttc_s == pytest.approx(0.048, abs=TOL)
    assert res.min_distance_m == 0.0


# --------------------------------------------------------------------------- #
# Arcs: agreement with the brute-force oracle                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kappa", CFG["motion"]["curvatures_inv_m"])
def test_matches_oracle_on_every_fan_curvature(scene, kappa):
    comp, obst = _component(scene, "rear_right_bumper_corner"), _pole(scene)
    res = _pair(scene, "rear_right_bumper_corner", obst, -1.0, kappas=(kappa,))
    d_ref, ttc_ref = _oracle(comp, obst, -1.0, kappa, 3.0, M_C, M_O)

    assert res.curvature == kappa
    if ttc_ref is None:
        assert res.ttc_s is None
        assert res.min_distance_m == pytest.approx(d_ref, abs=2e-3)
    else:
        assert res.ttc_s is not None
        assert res.ttc_s == pytest.approx(ttc_ref, abs=TOL + 1e-3)
        assert res.min_distance_m == 0.0


def test_matches_oracle_for_the_protruding_mirror_on_a_turn(scene):
    """The mirror is the part a bounding box cannot represent - check it on an arc."""
    post = S.obstacle_footprint(_obstacle(3, [0.5, -2.3, 0.0], [0.6, -2.2, 2.0]))
    comp = _component(scene, "right_mirror")
    res = _pair(scene, "right_mirror", post, -1.0, kappas=(-0.2,))
    d_ref, ttc_ref = _oracle(comp, post, -1.0, -0.2, 3.0, M_C, M_O)
    if ttc_ref is None:
        assert res.ttc_s is None
        assert res.min_distance_m == pytest.approx(d_ref, abs=2e-3)
    else:
        assert res.ttc_s == pytest.approx(ttc_ref, abs=TOL + 1e-3)


# --------------------------------------------------------------------------- #
# Per-component selection                                                      #
# --------------------------------------------------------------------------- #
def test_kerb_flags_the_wheels_not_the_bumpers(scene):
    """The per-component result the project exists for.

    In pure 2D the rear bumper would 'hit' the kerb first (0.9 s). With the height gate
    it correctly does not, and the rear wheels are named instead.
    """
    out = S.sweep_components(_all_components(scene), [_kerb(scene)], _fan(-1.0), 0, CFG)
    for bumper in ("rear_bumper", "rear_left_bumper_corner", "rear_right_bumper_corner"):
        assert bumper not in out, f"{bumper} rides over a 0.12 m kerb"
    for wheel in ("wheel_rear_left", "wheel_rear_right"):
        assert out[wheel].ttc_s == pytest.approx(1.55, abs=TOL)
    assert "right_mirror" not in out and "underbody" not in out


def test_pole_names_the_rear_right_corner_first(scene):
    out = S.sweep_components(_all_components(scene), [_pole(scene)], _fan(-1.0), 0, CFG)
    first = min((r for r in out.values() if r.ttc_s is not None), key=lambda r: r.ttc_s)
    assert first.component == "rear_right_bumper_corner"
    assert first.ttc_s == pytest.approx(0.37, abs=TOL)


def test_most_threatening_obstacle_is_the_earliest_contact(scene):
    """wheel_rear_right meets the pole at 1.02 s and the kerb at 1.55 s."""
    out = S.sweep_components(
        [_component(scene, "wheel_rear_right")],
        [_kerb(scene), _pole(scene)],
        _fan(-1.0),
        0,
        CFG,
    )
    assert out["wheel_rear_right"].obstacle_id == 1
    assert out["wheel_rear_right"].ttc_s == pytest.approx(1.02, abs=TOL)


def test_without_contact_the_nearest_obstacle_is_chosen(scene):
    # Stationary. Corner spans x [-1.00, -0.85], y [0.70, 1.00].
    # near: directly behind, 0.90 m raw -> 0.80 m inflated.
    # far:  behind and outboard, sqrt(1.7^2 + 0.4^2) = 1.746 m raw -> 1.646 m inflated.
    near = S.obstacle_footprint(_obstacle(4, [-2.0, 0.8, 0.0], [-1.9, 0.9, 1.0]))
    far = S.obstacle_footprint(_obstacle(6, [-2.8, 1.4, 0.0], [-2.7, 1.5, 1.0]))
    out = S.sweep_components(
        [_component(scene, "rear_left_bumper_corner")], [far, near], _fan(0.0), 0, CFG
    )
    assert out["rear_left_bumper_corner"].obstacle_id == 4


def test_components_beyond_report_distance_are_omitted(scene):
    """Front wheels are > 2 m from the pole for the whole horizon when stationary."""
    out = S.sweep_components(_all_components(scene), [_pole(scene)], _fan(0.0), 0, CFG)
    assert "wheel_front_left" not in out and "wheel_front_right" not in out
    assert all(r.min_distance_m <= CFG["sweep"]["report_distance_m"] for r in out.values())


def test_no_obstacles_gives_no_results(scene):
    assert S.sweep_components(_all_components(scene), [], _fan(-1.0), 0, CFG) == {}


# --------------------------------------------------------------------------- #
# The whole fan (kept for P4-T7)                                               #
# --------------------------------------------------------------------------- #
def test_sweep_fan_returns_one_result_set_per_curvature(scene):
    fan = path_fan(-1.0, horizon_s=3.0, dt_s=0.1, curvatures=CFG["motion"]["curvatures_inv_m"])
    per_path = S.sweep_fan(_all_components(scene), [_pole(scene)], fan, CFG)
    assert len(per_path) == fan.n_curvatures
    for k, results in enumerate(per_path):
        for r in results.values():
            assert r.curvature == fan.curvatures[k]
    # The straight path is row 3 and must agree with sweep_components on k = 3.
    straight = S.sweep_components(_all_components(scene), [_pole(scene)], fan, 3, CFG)
    assert per_path[3] == straight


def test_results_fit_the_component_risk_schema(scene):
    """SweepResult must drop straight into §4.2 ComponentRisk."""
    from vscs.common.types import ComponentRisk

    out = S.sweep_components(_all_components(scene), [_pole(scene)], _fan(-1.0), 0, CFG)
    for r in out.values():
        ComponentRisk(
            component=r.component,
            min_distance_m=r.min_distance_m,
            ttc_s=r.ttc_s,
            p_contact=0.0,
            severity=1.0,
            risk=0.0,
            obstacle_id=r.obstacle_id,
        )
