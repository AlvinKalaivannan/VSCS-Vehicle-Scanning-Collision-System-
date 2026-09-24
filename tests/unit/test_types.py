"""Schema contract tests (CLAUDE.md section 4.2) and the p_any_contact math."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from vscs.common.types import (
    NS_PER_S,
    SCHEMA_VERSION,
    Component,
    ComponentModel,
    ComponentRisk,
    JointSpec,
    Obstacle,
    RiskFrame,
    p_any_contact,
    validate_timestamp_stream,
)


def _risk(component: str, p: float, severity: float = 1.0, risk: float | None = None):
    return ComponentRisk(
        component=component,
        min_distance_m=0.4,
        ttc_s=1.5,
        p_contact=p,
        severity=severity,
        risk=p * severity if risk is None else risk,
    )


# --------------------------------------------------------------------------- #
# Versioning                                                                   #
# --------------------------------------------------------------------------- #
def test_schema_version_is_present_and_stamped_on_streams():
    assert isinstance(SCHEMA_VERSION, int)
    frame = RiskFrame(t_ns=1_000_000_000, ego_speed_mps=0.0, p_any_contact=0.0, expected_damage=0.0)
    assert frame.schema_version == SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# Timestamps - R-03                                                            #
# --------------------------------------------------------------------------- #
def test_t_ns_rejects_float():
    """A float timestamp means seconds crept in somewhere. StrictInt must reject it."""
    with pytest.raises(ValidationError):
        Obstacle(
            id=1,
            t_ns=1.5,
            kind="static_geom",
            center_veh=(0, 0, 0),
            extent=(1, 1, 1),
            pos_sigma_m=0.1,
            source="occupancy",
        )


def test_t_ns_rejects_negative():
    with pytest.raises(ValidationError):
        RiskFrame(t_ns=-1, ego_speed_mps=0.0, p_any_contact=0.0, expected_damage=0.0)


def test_timestamp_stream_accepts_jittered_but_increasing():
    base = 1_700_000_000_000_000_000
    stream = [base, base + 33_000_000, base + 70_000_000, base + 99_000_000]
    validate_timestamp_stream(stream)  # must not raise


def test_timestamp_stream_rejects_non_monotonic():
    base = 1_700_000_000_000_000_000
    with pytest.raises(ValueError, match="not strictly increasing"):
        validate_timestamp_stream([base, base + 1000, base + 500])


def test_timestamp_stream_rejects_duplicates():
    base = 1_700_000_000_000_000_000
    with pytest.raises(ValueError, match="not strictly increasing"):
        validate_timestamp_stream([base, base, base + 1000])


def test_timestamp_stream_rejects_frame_indices():
    """0, 1, 2, ... is the R-03 failure mode: frame index used as time."""
    with pytest.raises(ValueError, match="frame indices"):
        validate_timestamp_stream(list(range(20)))


def test_timestamp_stream_rejects_seconds_scale():
    """Seconds as floats-turned-ints span far too little for a real clip."""
    with pytest.raises(ValueError, match=r"frame indices|nanoseconds"):
        validate_timestamp_stream([0, 1, 2, 3])


def test_timestamp_stream_short_input_is_allowed():
    validate_timestamp_stream([])
    validate_timestamp_stream([5])


def test_ns_per_s_constant():
    assert NS_PER_S == 1_000_000_000


# --------------------------------------------------------------------------- #
# Schema drift guard - R-16                                                     #
# --------------------------------------------------------------------------- #
def test_unknown_field_is_rejected():
    """extra='forbid' turns schema drift into an error at construction time."""
    with pytest.raises(ValidationError):
        Component(
            name="a",
            display_name="a",
            link="a",
            severity=1.0,
            min_z=0.0,
            typo_field=3,
        )


def test_extent_must_be_positive():
    with pytest.raises(ValidationError):
        Obstacle(
            id=1,
            t_ns=1_000_000_000,
            kind="person",
            center_veh=(0, 0, 0),
            extent=(0.0, 1.0, 1.0),
            pos_sigma_m=0.1,
            source="detector",
        )


def test_obstacle_kind_is_constrained():
    with pytest.raises(ValidationError):
        Obstacle(
            id=1,
            t_ns=1_000_000_000,
            kind="spaceship",
            center_veh=(0, 0, 0),
            extent=(1, 1, 1),
            pos_sigma_m=0.1,
            source="detector",
        )


def test_p_contact_must_be_a_probability():
    with pytest.raises(ValidationError):
        _risk("x", 1.4)
    with pytest.raises(ValidationError):
        _risk("x", -0.1)


def test_ttc_none_is_distinct_from_large():
    """None means 'no predicted contact', which is not the same as a big TTC."""
    r = ComponentRisk(
        component="x", min_distance_m=5.0, ttc_s=None, p_contact=0.0, severity=1.0, risk=0.0
    )
    assert r.ttc_s is None


def test_negative_ttc_rejected():
    with pytest.raises(ValidationError):
        ComponentRisk(
            component="x", min_distance_m=1.0, ttc_s=-0.5, p_contact=0.1, severity=1.0, risk=0.1
        )


# --------------------------------------------------------------------------- #
# p_any_contact - required by CLAUDE.md section 5                               #
# --------------------------------------------------------------------------- #
def test_p_any_contact_two_coin_flips():
    """Two independent 0.5 events: 1 - 0.5*0.5 = 0.75."""
    assert p_any_contact([0.5, 0.5]) == pytest.approx(0.75)


def test_p_any_contact_empty_is_zero():
    assert p_any_contact([]) == 0.0


def test_p_any_contact_certainty_dominates():
    assert p_any_contact([1.0, 0.0, 0.3]) == pytest.approx(1.0)


def test_p_any_contact_all_zero():
    assert p_any_contact([0.0] * 12) == 0.0


def test_p_any_contact_matches_explicit_product():
    probs = [0.01, 0.2, 0.35, 0.02, 0.9]
    expected = 1.0 - math.prod(1.0 - p for p in probs)
    assert p_any_contact(probs) == pytest.approx(expected, rel=1e-12)


def test_p_any_contact_is_monotonic_in_each_term():
    assert p_any_contact([0.1, 0.1]) < p_any_contact([0.1, 0.2])


def test_p_any_contact_never_exceeds_one():
    assert p_any_contact([0.99] * 50) <= 1.0


def test_p_any_contact_rejects_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        p_any_contact([0.5, 1.2])


# --------------------------------------------------------------------------- #
# RiskFrame aggregation                                                        #
# --------------------------------------------------------------------------- #
def test_from_components_derives_aggregates():
    comps = [
        _risk("rear_right_bumper_corner", 0.6, severity=1.5),
        _risk("wheel_rear_right", 0.2, severity=0.6),
    ]
    frame = RiskFrame.from_components(t_ns=1_000_000_000, ego_speed_mps=1.2, per_component=comps)
    assert frame.p_any_contact == pytest.approx(1 - 0.4 * 0.8)
    assert frame.expected_damage == pytest.approx(0.6 * 1.5 + 0.2 * 0.6)
    # Highest risk is the bumper corner (0.9 vs 0.12), and severity is why.
    assert frame.worst_component == "rear_right_bumper_corner"
    assert frame.alert_level == "none"


def test_from_components_empty_frame():
    frame = RiskFrame.from_components(t_ns=1_000_000_000, ego_speed_mps=0.0, per_component=[])
    assert frame.worst_component is None
    assert frame.p_any_contact == 0.0
    assert frame.expected_damage == 0.0


def test_worst_component_must_exist_in_list():
    with pytest.raises(ValidationError):
        RiskFrame(
            t_ns=1_000_000_000,
            ego_speed_mps=0.0,
            per_component=[_risk("a", 0.5)],
            p_any_contact=0.5,
            expected_damage=0.5,
            worst_component="not_a_component",
        )


def test_alert_level_is_constrained():
    with pytest.raises(ValidationError):
        RiskFrame(
            t_ns=1_000_000_000,
            ego_speed_mps=0.0,
            p_any_contact=0.0,
            expected_damage=0.0,
            alert_level="panic",
        )


def test_severity_weighting_can_outrank_proximity():
    """The reason VSCS exists: a likely cheap scuff must not outrank a costly hit."""
    wheel = _risk("wheel_rear_right", 0.9, severity=0.6)  # very likely, cheap
    mirror = _risk("right_mirror", 0.4, severity=2.0)  # less likely, costly
    frame = RiskFrame.from_components(1_000_000_000, 1.0, [wheel, mirror])
    assert frame.worst_component == "right_mirror"


# --------------------------------------------------------------------------- #
# JointSpec - defined in ADR 0002                                              #
# --------------------------------------------------------------------------- #
def test_jointspec_defaults_and_validation():
    j = JointSpec(type="revolute", parent_link="body", child_link="door", limit_upper=1.5)
    assert j.axis == (0.0, 0.0, 1.0)
    assert j.limit_lower == 0.0


def test_jointspec_rejects_zero_axis():
    with pytest.raises(ValidationError):
        JointSpec(type="revolute", parent_link="a", child_link="b", axis=(0.0, 0.0, 0.0))


def test_jointspec_rejects_inverted_limits():
    with pytest.raises(ValidationError):
        JointSpec(
            type="prismatic",
            parent_link="a",
            child_link="b",
            limit_lower=1.0,
            limit_upper=0.0,
        )


def test_fixed_joint_ignores_limit_order():
    JointSpec(type="fixed", parent_link="a", child_link="b", limit_lower=0.0, limit_upper=0.0)


# --------------------------------------------------------------------------- #
# ComponentModel                                                               #
# --------------------------------------------------------------------------- #
def _component(name: str) -> Component:
    return Component(name=name, display_name=name, link=f"{name}_link", severity=1.0, min_z=0.3)


def test_component_model_lookup_and_duplicate_rejection():
    model = ComponentModel(
        urdf_path=Path("van.urdf"),
        components=[_component("rear_bumper"), _component("right_mirror")],
        scale_error_m=0.013,
    )
    assert model.by_name("right_mirror").link == "right_mirror_link"
    with pytest.raises(KeyError, match="no component"):
        model.by_name("nope")

    with pytest.raises(ValidationError):
        ComponentModel(
            urdf_path=Path("van.urdf"),
            components=[_component("a"), _component("a")],
            scale_error_m=0.0,
        )


def test_scale_error_cannot_be_negative():
    with pytest.raises(ValidationError):
        ComponentModel(urdf_path=Path("v.urdf"), components=[], scale_error_m=-0.01)
