"""Evaluation metric definitions (CLAUDE.md §11). One place, so every report agrees.

Scan metrics are here first; perception, risk and system metrics are added as their
phases begin.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def per_component_iou(
    pred: npt.ArrayLike, truth: npt.ArrayLike, classes: list[int] | range
) -> dict[int, float]:
    """IoU of each class over a set of labelled 3D points (P2-T3, P2-T5 acceptance).

    For class ``c``: ``|pred == c  AND  truth == c|  /  |pred == c  OR  truth == c|``.
    A point predicted as ``c`` but truly something else, or truly ``c`` but predicted as
    something else (including "none"), counts against ``c``. A class absent from both is
    left out rather than scored 1.0, so an empty class cannot inflate the mean.
    """
    p = np.asarray(pred)
    t = np.asarray(truth)
    if p.shape != t.shape:
        raise ValueError(f"pred {p.shape} and truth {t.shape} differ in shape")
    out: dict[int, float] = {}
    for c in classes:
        union = np.count_nonzero((p == c) | (t == c))
        if union:
            out[int(c)] = np.count_nonzero((p == c) & (t == c)) / union
    return out


def mean_iou(ious: dict[int, float]) -> float:
    """Unweighted mean over the classes present: each component counts equally."""
    if not ious:
        raise ValueError("no classes present to average")
    return float(np.mean(list(ious.values())))
