"""Evaluation metric definitions (CLAUDE.md §11). One place, so every report agrees.

Scan metrics are here first; perception, risk and system metrics are added as their
phases begin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from vscs.common.types import RiskFrame
from vscs.risk.alerts import RANK


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


# --------------------------------------------------------------------------- #
# Risk metrics (CLAUDE.md §11; P3-T6, P5-T2)                                  #
# --------------------------------------------------------------------------- #
#
# Ground truth for one recorded pass comes from the tape-measured lot layout (P1-T3) or a
# dataset's 3D tracks (P4-T14): which component truly passes closest to an obstacle, and
# when that closest approach / contact happens. "Flagged" means the alert reached
# ``flag_level`` (eval.yaml ``risk_metrics``), using the same level order as the alerts.

NS_PER_S = 1_000_000_000


@dataclass(frozen=True)
class PassTruth:
    """Ground truth for one pass. ``t_event_ns`` is the true closest approach (or contact)."""

    component: str
    t_event_ns: int


def first_flag(frames: list[RiskFrame], flag_level: str) -> RiskFrame | None:
    """The first frame whose alert is at or above ``flag_level``."""
    need = RANK[flag_level]
    return next((f for f in frames if RANK[f.alert_level] >= need), None)


def attribution_correct(frames: list[RiskFrame], truth: PassTruth, flag_level: str) -> bool:
    """Was the right component flagged *before* the event? (P3-T6's criterion.)

    No flag, a late flag, or a flag naming another component all count as wrong: the
    driver would not have been told the right thing in time.
    """
    f = first_flag(frames, flag_level)
    return f is not None and f.t_ns < truth.t_event_ns and f.worst_component == truth.component


def attribution_accuracy(passes: list[tuple[list[RiskFrame], PassTruth]], flag_level: str) -> float:
    if not passes:
        raise ValueError("no passes to score")
    return float(np.mean([attribution_correct(fr, tr, flag_level) for fr, tr in passes]))


def lead_time_s(frames: list[RiskFrame], truth: PassTruth, flag_level: str) -> float | None:
    """Seconds from the first flag to the event: positive = early, negative = late.

    ``None`` if the alert never reached ``flag_level`` (report those passes separately; a
    missing warning must not be averaged in as zero).
    """
    f = first_flag(frames, flag_level)
    return None if f is None else (truth.t_event_ns - f.t_ns) / NS_PER_S


def ttc_errors_s(frames: list[RiskFrame], truth: PassTruth) -> npt.NDArray[np.float64]:
    """|predicted TTC - true time remaining| for the true component, frames before the event.

    Only frames where the true component had a predicted contact are scored: a frame with
    no prediction is a miss, which attribution and lead time already count.
    """
    errs = []
    for f in frames:
        if f.t_ns >= truth.t_event_ns:
            continue
        for cr in f.per_component:
            if cr.component == truth.component and cr.ttc_s is not None:
                errs.append(abs(cr.ttc_s - (truth.t_event_ns - f.t_ns) / NS_PER_S))
    return np.asarray(errs, dtype=np.float64)


def alert_episodes(frames: list[RiskFrame], flag_level: str) -> list[tuple[int, int]]:
    """``(start_ns, end_ns)`` of each run of frames at or above ``flag_level``."""
    need = RANK[flag_level]
    episodes: list[tuple[int, int]] = []
    start: int | None = None
    last = None
    for f in frames:
        on = RANK[f.alert_level] >= need
        if on and start is None:
            start = f.t_ns
        if not on and start is not None:
            episodes.append((start, last))
            start = None
        last = f.t_ns
    if start is not None:
        episodes.append((start, last))
    return episodes


def false_alarms_per_min(
    frames: list[RiskFrame], event_onsets_ns: list[int], flag_level: str, horizon_s: float
) -> float:
    """Alert episodes with no true event starting within ``horizon_s`` after they began,
    per minute of recording.

    An episode is "true" if an event begins inside ``[start, start + horizon]``, or had
    already begun when the episode started and is still the reason for it (an event
    between the episode's start and end). Everything else is a false alarm.
    """
    if len(frames) < 2:
        raise ValueError("need at least two frames to know the duration")
    minutes = (frames[-1].t_ns - frames[0].t_ns) / NS_PER_S / 60.0
    if minutes <= 0:
        raise ValueError("frames must span a positive duration")
    horizon = round(horizon_s * NS_PER_S)
    false = 0
    for start, end in alert_episodes(frames, flag_level):
        if not any(start <= t <= max(end, start + horizon) for t in event_onsets_ns):
            false += 1
    return false / minutes
