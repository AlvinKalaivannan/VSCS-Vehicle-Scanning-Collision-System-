"""Per-frame engine tests, including the reachable-horizon rule (ADR 0005).

The sweep is injected: these use the brute-force reference in
``tests/fixtures/oracle_sweep.py`` so the engine can be tested before the developer's
``risk/sweep.py`` exists.
"""

from __future__ import annotations

import pytest

from fixtures.oracle_sweep import ref_box, ref_sweep_components
from fixtures.synthetic import load_scene
from vscs.common.config import load_config
from vscs.common.io import read_jsonl, write_jsonl
from vscs.common.types import RiskFrame
from vscs.risk.aggregate import component_risks
from vscs.risk.alerts import AlertStateMachine
from vscs.risk.engine import assess_frame, first_contact_s, reachable_clearances
from vscs.risk.motion import path_fan_from_config

RISK = load_config("risk")
SEV = load_config("severity")
TOL = RISK["sweep"]["ttc_tolerance_s"]
POLE_ID, KERB_ID = 1, 2
KINDS = {POLE_ID: "static_geom", KERB_ID: "static_geom"}


@pytest.fixture(scope="module")
def scene():
    return load_scene()


@pytest.fixture(scope="module")
def components(scene):
    return [ref_box(n, b.lo, b.hi) for n, b in scene.components.items()]


@pytest.fixture(scope="module")
def pole(scene):
    (ax, ay), r, h = scene.pole.axis_xy, scene.pole.radius_m, scene.pole.height_m
    return ref_box("pole", [ax - r, ay - r, 0.0], [ax + r, ay + r, h], obstacle_id=POLE_ID)


@pytest.fixture(scope="module")
def kerb(scene):
    return ref_box("kerb", scene.curb.lo, scene.curb.hi, obstacle_id=KERB_ID)


def _assess(components, obstacles, speed=-1.0, curvature=0.0, t_ns=1, sm=None):
    return assess_frame(
        t_ns=t_ns,
        ego_speed_mps=speed,
        curvature=curvature,
        components=components,
        obstacles=obstacles,
        obstacle_kinds=KINDS,
        state_machine=sm or AlertStateMachine(RISK),
        risk_cfg=RISK,
        severity_cfg=SEV,
        sweep_fn=ref_sweep_components,
    )


# --------------------------------------------------------------------------- #
# The reachable horizon                                                        #
# --------------------------------------------------------------------------- #
def test_first_contact_is_the_earliest_ttc(components, pole):
    fan = path_fan_from_config(-1.0, RISK)
    full = ref_sweep_components(components, [pole], fan, fan.index_of_curvature(0.0), RISK)
    assert first_contact_s(full.values()) == pytest.approx(0.37, abs=TOL)


def test_without_truncation_the_wrong_component_is_named(components, pole):
    """Why ADR 0005 exists. The full-horizon sweep reports a door strike at 2.37 s that
    cannot happen (the corner hits the pole at 0.37 s first), and the door's higher
    severity then makes it the 'worst' component."""
    fan = path_fan_from_config(-1.0, RISK)
    full = ref_sweep_components(components, [pole], fan, fan.index_of_curvature(0.0), RISK)
    assert full["sliding_door_right"].ttc_s == pytest.approx(2.37, abs=TOL)
    risks = component_risks(full.values(), KINDS, -1.0, SEV, RISK)
    assert risks[0].component == "sliding_door_right"


def test_truncation_removes_unreachable_contacts(components, pole):
    fan = path_fan_from_config(-1.0, RISK)
    reach, t_star = reachable_clearances(
        components, [pole], fan, fan.index_of_curvature(0.0), RISK, ref_sweep_components
    )
    assert t_star == pytest.approx(0.37, abs=TOL)
    assert reach["rear_right_bumper_corner"].ttc_s == pytest.approx(0.37, abs=TOL)
    for name, r in reach.items():
        if name != "rear_right_bumper_corner":
            assert r.ttc_s is None, f"{name} contact is beyond the first impact"


def test_no_contact_means_a_single_sweep(components):
    calls = []

    def counting_sweep(*args):
        calls.append(1)
        return ref_sweep_components(*args)

    far = ref_box("far", [-9.0, 5.0, 0.0], [-8.9, 5.1, 1.0], obstacle_id=POLE_ID)
    fan = path_fan_from_config(-1.0, RISK)
    _, t_star = reachable_clearances(components, [far], fan, 3, RISK, counting_sweep)
    assert t_star is None and len(calls) == 1


# --------------------------------------------------------------------------- #
# Whole frames                                                                 #
# --------------------------------------------------------------------------- #
def test_frame_names_the_corner_for_the_pole(components, pole):
    """P3-T6's metric in miniature: the right component is named."""
    frame = _assess(components, [pole])
    assert frame.worst_component == "rear_right_bumper_corner"
    corner = next(r for r in frame.per_component if r.component == "rear_right_bumper_corner")
    assert corner.ttc_s == pytest.approx(0.37, abs=TOL)


def test_frame_names_a_rear_wheel_for_the_kerb(components, kerb):
    """The bumpers overhang a 0.12 m kerb; the wheels do not."""
    frame = _assess(components, [kerb])
    assert frame.worst_component in ("wheel_rear_left", "wheel_rear_right")
    for r in frame.per_component:
        assert "bumper" not in r.component


def test_alert_goes_critical_for_an_imminent_contact(components, pole):
    sm = AlertStateMachine(RISK)
    levels = [
        _assess(components, [pole], t_ns=1 + i * 33_333_333, sm=sm).alert_level for i in range(3)
    ]
    assert levels[-1] == "critical"


def test_stationary_vehicle_reports_static_clearance_only(components, pole):
    frame = _assess(components, [pole], speed=0.0)
    corner = next(r for r in frame.per_component if r.component == "rear_right_bumper_corner")
    assert corner.ttc_s is None
    assert corner.min_distance_m == pytest.approx(0.37, abs=1e-3)
    assert frame.p_any_contact == 0.0


def test_no_obstacles_gives_an_empty_frame(components):
    frame = _assess(components, [])
    assert frame.per_component == [] and frame.worst_component is None
    assert frame.alert_level == "none"


def test_injected_probability_model_reaches_the_frame(components, pole):
    frame = assess_frame(
        t_ns=1,
        ego_speed_mps=-1.0,
        curvature=0.0,
        components=components,
        obstacles=[pole],
        obstacle_kinds=KINDS,
        state_machine=AlertStateMachine(RISK),
        risk_cfg=RISK,
        severity_cfg=SEV,
        sweep_fn=ref_sweep_components,
        p_contact_fn=lambda ttc: 0.5 if ttc is not None else 0.0,
    )
    assert frame.p_any_contact == pytest.approx(0.5)


def test_frames_serialise_as_json_lines(components, pole, tmp_path):
    sm = AlertStateMachine(RISK)
    frames = [_assess(components, [pole], t_ns=1 + i * 33_333_333, sm=sm) for i in range(3)]
    path = write_jsonl(tmp_path / "risk.jsonl", frames)
    assert [RiskFrame(**r) for r in read_jsonl(path)] == frames
