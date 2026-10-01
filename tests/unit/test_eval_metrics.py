"""Metric definitions in eval/metrics.py."""

from __future__ import annotations

import numpy as np
import pytest

from vscs.common.config import load_config
from vscs.common.types import ComponentRisk, RiskFrame
from vscs.eval import metrics as Mx
from vscs.eval.metrics import mean_iou, per_component_iou


def test_iou_known_answer():
    truth = np.array([0, 0, 0, 1, 1, -1])
    pred = np.array([0, 0, 1, 1, 1, 0])
    ious = per_component_iou(pred, truth, range(2))
    # class 0: inter {0,1} = 2, union {0,1,2,5} = 4 -> 0.5
    # class 1: inter {3,4} = 2, union {2,3,4} = 3
    assert ious == {0: pytest.approx(0.5), 1: pytest.approx(2 / 3)}
    assert mean_iou(ious) == pytest.approx((0.5 + 2 / 3) / 2)


def test_missing_a_point_as_none_counts_against_the_class():
    assert per_component_iou(np.array([-1, 0]), np.array([0, 0]), [0]) == {0: 0.5}


def test_absent_class_is_skipped_not_perfect():
    assert per_component_iou(np.array([0]), np.array([0]), range(3)) == {0: 1.0}


def test_shape_mismatch_and_empty_mean_are_errors():
    with pytest.raises(ValueError, match="differ"):
        per_component_iou([0, 1], [0], [0])
    with pytest.raises(ValueError, match="no classes"):
        mean_iou({})


# --------------------------------------------------------------------------- #
# Risk metrics, on hand-built RiskFrame streams with known answers             #
# --------------------------------------------------------------------------- #

S = 1_000_000_000
FLAG = load_config("eval")["risk_metrics"]["flag_level"]


def _f(t_s, level, worst=None, ttc=None, comp=None):
    comp = comp or worst or "rear_right_bumper_corner"
    per = [
        ComponentRisk(
            component=comp,
            min_distance_m=0.0 if ttc else 1.0,
            ttc_s=ttc,
            p_contact=1.0 if ttc else 0.0,
            severity=1.5,
            risk=1.5 if ttc else 0.0,
            obstacle_id=1,
        )
    ]
    return RiskFrame(
        t_ns=round(t_s * S),
        ego_speed_mps=1.0,
        per_component=per,
        p_any_contact=1.0 if ttc else 0.0,
        expected_damage=0.0,
        worst_component=worst,
        alert_level=level,
    )


TRUTH = Mx.PassTruth(component="rear_right_bumper_corner", t_event_ns=5 * S)
GOOD = [
    _f(0, "none"),
    _f(1, "caution", "rear_right_bumper_corner", 4.2),
    _f(2, "warning", "rear_right_bumper_corner", 3.1),
    _f(3, "critical", "rear_right_bumper_corner", 1.9),
    _f(6, "none"),
]


def test_flag_level_comes_from_config():
    assert FLAG == "warning"


def test_attribution_needs_the_right_component_before_the_event():
    assert Mx.attribution_correct(GOOD, TRUTH, FLAG)
    wrong = [_f(0, "none"), _f(2, "warning", "right_mirror", 3.0)]
    late = [_f(0, "none"), _f(5.5, "warning", "rear_right_bumper_corner", 0.1)]
    never = [_f(0, "caution", "rear_right_bumper_corner", 4.0)]  # caution only
    assert not Mx.attribution_correct(wrong, TRUTH, FLAG)
    assert not Mx.attribution_correct(late, TRUTH, FLAG)
    assert not Mx.attribution_correct(never, TRUTH, FLAG)
    assert (
        Mx.attribution_accuracy(
            [(GOOD, TRUTH), (wrong, TRUTH), (late, TRUTH), (never, TRUTH)], FLAG
        )
        == 0.25
    )


def test_lead_time_is_signed_and_none_when_never_warned():
    assert Mx.lead_time_s(GOOD, TRUTH, FLAG) == pytest.approx(3.0)  # warned at 2 s, event at 5 s
    assert Mx.lead_time_s([_f(5.5, "warning", "x")], TRUTH, FLAG) == pytest.approx(-0.5)
    assert Mx.lead_time_s([_f(0, "caution", "x")], TRUTH, FLAG) is None


def test_ttc_error_against_true_time_remaining():
    # Predicted 4.2 / 3.1 / 1.9 at t = 1 / 2 / 3; truth 4 / 3 / 2 -> errors 0.2, 0.1, 0.1.
    np.testing.assert_allclose(sorted(Mx.ttc_errors_s(GOOD, TRUTH)), [0.1, 0.1, 0.2], atol=1e-9)


def test_episodes_and_false_alarms():
    frames = [
        _f(t, lvl)
        for t, lvl in [
            (0, "none"),
            (10, "warning"),
            (11, "warning"),
            (12, "none"),
            (30, "critical"),
            (31, "none"),
            (60, "none"),
        ]
    ]
    assert Mx.alert_episodes(frames, FLAG) == [(10 * S, 11 * S), (30 * S, 30 * S)]
    # A true event at 12.5 s explains the first episode (within 3 s); the second is false.
    rate = Mx.false_alarms_per_min(frames, [round(12.5 * S)], FLAG, horizon_s=3.0)
    assert rate == pytest.approx(1.0)  # 1 false alarm in 1 minute
    assert Mx.false_alarms_per_min(frames, [], FLAG, horizon_s=3.0) == pytest.approx(2.0)


def test_false_alarms_need_a_duration():
    with pytest.raises(ValueError):
        Mx.false_alarms_per_min([_f(0, "none")], [], FLAG, 3.0)
