"""P5-T1 single-box baseline: same engine, one oriented box; known answers on the fixture."""

from __future__ import annotations

import pytest
from shapely.geometry import Point, box

from fixtures.oracle_sweep import ref_box, ref_sweep_components
from fixtures.synthetic import load_scene
from vscs.common.config import load_config
from vscs.eval.baseline_bbox import baseline_severity_cfg, single_box_footprint
from vscs.risk.alerts import RANK, AlertStateMachine
from vscs.risk.engine import assess_frame

SCENE = load_scene()
RISK = load_config("risk")
SEV = load_config("severity")
BASE = load_config("eval")["baseline"]
POLE_ID = 1


def _footprints():
    """Fixture components in the ``urdf.plan_footprints`` format."""
    return {
        n: (box(b.lo[0], b.lo[1], b.hi[0], b.hi[1]), float(b.lo[2]), float(b.hi[2]))
        for n, b in SCENE.components.items()
    }


def test_the_box_encloses_every_component_mirrors_included():
    fp = single_box_footprint(_footprints(), BASE["name"])
    minx, miny, maxx, maxy = fp.polygon.bounds
    # Body x -1.0..4.0; mirrors stick out to |y| = 1.25.
    assert (minx, miny, maxx, maxy) == pytest.approx((-1.0, -1.25, 4.0, 1.25))
    assert fp.polygon.area == pytest.approx(5.0 * 2.5)
    assert (fp.z_min, fp.z_max) == pytest.approx((0.0, 1.9))  # wheels on the ground; door top
    assert fp.name == BASE["name"]


def test_behind_the_corner_both_models_agree_on_the_known_half_metre():
    fp = single_box_footprint(_footprints(), BASE["name"])
    assert fp.polygon.distance(Point(SCENE.pole.axis_xy)) == pytest.approx(0.5)


def test_beside_the_body_the_box_invents_clearance_loss():
    """A pole 0.4 m off the sliding door: the box's side runs along the mirror tips."""
    pole = Point(1.8, -1.4)
    fp = single_box_footprint(_footprints(), BASE["name"])
    door = _footprints()["sliding_door_right"][0]
    assert door.distance(pole) == pytest.approx(0.40)
    assert fp.polygon.distance(pole) == pytest.approx(0.15)


def test_baseline_severity_has_one_component_and_keeps_multipliers():
    sev = baseline_severity_cfg(SEV, BASE)
    assert sev["components"] == {BASE["name"]: float(BASE["severity"])}
    assert sev["overrides"] == {}
    assert sev["obstacle_multiplier"] == SEV["obstacle_multiplier"]


def _assess(components, severity_cfg):
    pole = ref_box("pole", [1.77, -1.43, 0.0], [1.83, -1.37, 1.2], obstacle_id=POLE_ID)
    return assess_frame(
        t_ns=1,
        ego_speed_mps=0.0,
        curvature=0.0,
        components=components,
        obstacles=[pole],
        obstacle_kinds={POLE_ID: "static_geom"},
        state_machine=AlertStateMachine(RISK),
        risk_cfg=RISK,
        severity_cfg=severity_cfg,
        sweep_fn=ref_sweep_components,
    )


def test_same_engine_baseline_reports_less_clearance_and_a_harsher_alert():
    """Stationary van, pole beside the sliding door, the SAME engine and sweep.

    The only difference is geometry, so the clearance gap is exactly the mirror's 0.25 m
    protrusion, and the baseline names 'the vehicle' rather than a component.
    """
    parts = [ref_box(n, b.lo, b.hi) for n, b in SCENE.components.items()]
    fp = single_box_footprint(_footprints(), BASE["name"])
    vscs = _assess(parts, SEV)
    base = _assess([fp], baseline_severity_cfg(SEV, BASE))
    d_vscs = min(c.min_distance_m for c in vscs.per_component)
    d_base = min(c.min_distance_m for c in base.per_component)
    assert d_vscs - d_base == pytest.approx(0.25, abs=1e-6)
    near = min(vscs.per_component, key=lambda c: c.min_distance_m)
    assert near.component == "sliding_door_right"
    assert [c.component for c in base.per_component] == [BASE["name"]]
    assert RANK[base.alert_level] >= RANK[vscs.alert_level]
