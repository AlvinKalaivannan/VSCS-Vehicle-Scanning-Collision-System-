"""Severity lookup and per-component risk aggregation tests (P3-T5)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from vscs.common.config import load_config
from vscs.risk import aggregate as A

RISK = load_config("risk")
SEV = load_config("severity")


@dataclass
class Clearance:
    """Stands in for risk/sweep.py's SweepResult, which aggregate must not import."""

    component: str
    obstacle_id: int | None
    min_distance_m: float
    ttc_s: float | None


# --------------------------------------------------------------------------- #
# Severity                                                                     #
# --------------------------------------------------------------------------- #
def test_base_severity_without_an_obstacle_kind():
    assert A.severity_for("right_mirror", None, SEV) == SEV["components"]["right_mirror"]


def test_obstacle_multiplier_applies():
    base = SEV["components"]["right_mirror"]
    assert A.severity_for("right_mirror", "vehicle", SEV) == pytest.approx(
        base * SEV["obstacle_multiplier"]["vehicle"]
    )


def test_override_beats_the_product():
    """A kerb against the underbody is set explicitly, not by multiplication."""
    assert (
        A.severity_for("underbody", "static_geom", SEV)
        == SEV["overrides"]["underbody__static_geom"]
    )
    assert A.severity_for("wheel_rear_left", "static_geom", SEV) == pytest.approx(0.4)


def test_unknown_component_falls_back_to_default_not_zero():
    """A vocabulary gap must under-rank a component, never make it free to hit."""
    assert A.severity_for("spoiler", "static_geom", SEV) == pytest.approx(SEV["default"])


def test_a_person_outranks_any_panel():
    person_wheel = A.severity_for("wheel_rear_left", "person", SEV)
    worst_panel = max(A.severity_for(c, "static_geom", SEV) for c in SEV["components"])
    assert person_wheel > worst_panel


# --------------------------------------------------------------------------- #
# Speed factor                                                                 #
# --------------------------------------------------------------------------- #
def test_speed_factor_is_linear_between_the_clamps():
    ref = RISK["aggregate"]["speed_factor"]["reference_speed_mps"]
    assert A.speed_factor(ref, RISK) == pytest.approx(1.0)
    assert A.speed_factor(ref * 1.5, RISK) == pytest.approx(1.5)


def test_speed_factor_ignores_direction():
    assert A.speed_factor(-1.3, RISK) == A.speed_factor(1.3, RISK)


def test_speed_factor_clamps_both_ends():
    sf = RISK["aggregate"]["speed_factor"]
    assert A.speed_factor(0.0, RISK) == sf["min_factor"]
    assert A.speed_factor(100.0, RISK) == sf["max_factor"]


def test_unknown_speed_factor_method_is_rejected():
    cfg = {"aggregate": {"speed_factor": {**RISK["aggregate"]["speed_factor"], "method": "x"}}}
    with pytest.raises(ValueError, match="unsupported"):
        A.speed_factor(1.0, cfg)


# --------------------------------------------------------------------------- #
# Component risks and the frame                                                #
# --------------------------------------------------------------------------- #
def test_deterministic_p_contact():
    assert A.deterministic_p_contact(None) == 0.0
    assert A.deterministic_p_contact(0.37) == 1.0


def test_risk_is_p_times_severity_times_speed_factor():
    risks = A.component_risks(
        [Clearance("rear_right_bumper_corner", 1, 0.0, 0.37)],
        {1: "static_geom"},
        -1.0,
        SEV,
        RISK,
    )
    r = risks[0]
    expected = 1.0 * A.severity_for("rear_right_bumper_corner", "static_geom", SEV)
    expected *= A.speed_factor(-1.0, RISK)
    assert r.risk == pytest.approx(expected)
    assert r.p_contact == 1.0 and r.ttc_s == 0.37 and r.obstacle_id == 1


def test_no_predicted_contact_means_zero_risk_in_v01():
    risks = A.component_risks(
        [Clearance("right_mirror", 1, 0.4, None)], {1: "static_geom"}, -1.0, SEV, RISK
    )
    assert risks[0].risk == 0.0
    assert risks[0].min_distance_m == 0.4  # clearance is still reported


def test_severity_can_outrank_imminence():
    """The project's premise: a costly mirror strike outranks a cheaper, sooner scuff."""
    risks = A.component_risks(
        [
            Clearance("wheel_rear_right", 2, 0.0, 0.5),  # sooner, but a tyre on a kerb
            Clearance("right_mirror", 1, 0.0, 1.5),  # later, but a mirror
        ],
        {1: "static_geom", 2: "static_geom"},
        -1.0,
        SEV,
        RISK,
    )
    assert risks[0].component == "right_mirror"


def test_injected_probability_model_is_used():
    """P4-T7 swaps in a real model without touching this module."""
    risks = A.component_risks(
        [Clearance("rear_bumper", 1, 0.2, 1.0)],
        {1: "static_geom"},
        -1.0,
        SEV,
        RISK,
        p_contact_fn=lambda ttc: 0.25,
    )
    assert risks[0].p_contact == 0.25


def test_missing_obstacle_kind_uses_base_severity():
    risks = A.component_risks([Clearance("rear_bumper", 99, 0.0, 1.0)], {}, -1.0, SEV, RISK)
    assert risks[0].severity == SEV["components"]["rear_bumper"]


def test_aggregate_frame_derives_the_schema_fields():
    risks = A.component_risks(
        [Clearance("right_mirror", 1, 0.0, 1.5), Clearance("rear_bumper", 1, 0.8, None)],
        {1: "static_geom"},
        -1.0,
        SEV,
        RISK,
    )
    frame = A.aggregate_frame(1_000_000_000, -1.0, risks, "warning")
    assert frame.worst_component == "right_mirror"
    assert frame.p_any_contact == pytest.approx(1.0)
    assert frame.expected_damage == pytest.approx(sum(r.risk for r in risks))
    assert frame.alert_level == "warning"
