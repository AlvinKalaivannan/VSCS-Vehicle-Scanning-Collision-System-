"""Metric definitions in eval/metrics.py."""

from __future__ import annotations

import numpy as np
import pytest

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
