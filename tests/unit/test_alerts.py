"""Alert level and hysteresis tests - the P3-T5 acceptance criterion.

"Hysteresis test: no flicker on noisy fixture input" (CLAUDE.md §6). R-10 names the
failure concretely: alerts toggling more than twice a second.

The noisy input is derived from the fixture world rather than invented: the rear-right
bumper corner against the fixture pole has an inflated clearance of 0.37 m at the start
pose (0.5 m axis - 0.03 m radius - 0.05 m - 0.05 m margins). Parked 0.13 m further
forward, that clearance is exactly 0.50 m - on the WARNING threshold - which is the worst
case for flicker. Measurement noise is then added on top.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from vscs.common.config import load_config
from vscs.common.types import NS_PER_S, ComponentRisk
from vscs.risk.alerts import LEVELS, RANK, AlertStateMachine, raw_level

CFG = load_config("risk")
FIXTURE_START_GAP_M = 0.37  # corner -> pole, inflated, at the fixture start pose
FPS = 30.0


def _risk(d: float, ttc: float | None = None, p: float = 0.0, name="c") -> ComponentRisk:
    return ComponentRisk(
        component=name,
        min_distance_m=max(d, 0.0),
        ttc_s=ttc,
        p_contact=p,
        severity=1.0,
        risk=p,
    )


def _stamps(n: int, fps: float = FPS) -> list[int]:
    return [round(i * NS_PER_S / fps) + 1 for i in range(n)]


def _run(sm: AlertStateMachine, frames: list[list[ComponentRisk]], fps: float = FPS):
    return [sm.update(t, r) for t, r in zip(_stamps(len(frames), fps), frames, strict=True)]


def _changes(levels: list[str]) -> list[int]:
    return [i for i in range(1, len(levels)) if levels[i] != levels[i - 1]]


# --------------------------------------------------------------------------- #
# Raw level                                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "d,expected",
    [(1.5, "none"), (0.99, "caution"), (0.49, "warning"), (0.24, "critical"), (0.0, "critical")],
)
def test_raw_level_by_clearance(d, expected):
    assert raw_level([_risk(d)], CFG) == expected


@pytest.mark.parametrize(
    "ttc,expected", [(None, "none"), (2.9, "caution"), (1.9, "warning"), (0.9, "critical")]
)
def test_raw_level_by_time_to_contact(ttc, expected):
    """Far away but closing fast still escalates - TTC alone can set the level."""
    assert raw_level([_risk(5.0, ttc)], CFG) == expected


def test_worst_single_component_sets_the_level():
    """One critical component is not averaged away by many safe ones."""
    risks = [_risk(3.0, name=f"safe{i}") for i in range(10)] + [_risk(0.1, name="mirror")]
    assert raw_level(risks, CFG) == "critical"


def test_no_components_is_none():
    assert raw_level([], CFG) == "none"


def test_p_contact_is_ignored_in_v01():
    """The v0.1 0/1 indicator must not turn every predicted contact into 'critical'."""
    assert CFG["alerts"]["use_p_contact"] is False
    assert raw_level([_risk(5.0, ttc=None, p=1.0)], CFG) == "none"


def test_p_contact_counts_once_enabled():
    cfg = {**CFG, "alerts": {**CFG["alerts"], "use_p_contact": True}}
    assert raw_level([_risk(5.0, p=0.9)], cfg) == "critical"
    assert raw_level([_risk(5.0, p=0.3)], cfg) == "caution"


def test_distance_offset_widens_thresholds():
    """The release level: 0.55 m is 'caution' raw but still 'warning' inside the deadband."""
    assert raw_level([_risk(0.55)], CFG) == "caution"
    assert raw_level([_risk(0.55)], CFG, distance_offset_m=0.10) == "warning"


# --------------------------------------------------------------------------- #
# Escalation: fast, but not on a single glitch                                 #
# --------------------------------------------------------------------------- #
def test_starts_at_none():
    assert AlertStateMachine(CFG).level == "none"


def test_single_frame_spike_is_ignored():
    sm = AlertStateMachine(CFG)
    frames = [[_risk(2.0)]] * 5 + [[_risk(0.1)]] + [[_risk(2.0)]] * 5
    assert set(_run(sm, frames)) == {"none"}


def test_escalates_after_frames_to_escalate():
    n = CFG["alerts"]["hysteresis"]["frames_to_escalate"]
    sm = AlertStateMachine(CFG)
    levels = _run(sm, [[_risk(0.1)]] * (n + 2))
    assert levels[n - 2] == "none"
    assert levels[n - 1] == "critical"


def test_escalation_targets_the_level_sustained_over_the_streak():
    """warning then critical escalates to warning - critical was only seen once."""
    sm = AlertStateMachine(CFG)
    levels = _run(sm, [[_risk(0.4)], [_risk(0.1)]])
    assert levels[-1] == "warning"


def test_escalation_ignores_dwell():
    """Getting worse must never wait on the dwell timer."""
    sm = AlertStateMachine(CFG)
    levels = _run(sm, [[_risk(0.9)]] * 2 + [[_risk(0.1)]] * 2)
    assert levels[1] == "caution"
    assert levels[3] == "critical"  # 1/30 s later, far inside the 0.6 s dwell


# --------------------------------------------------------------------------- #
# De-escalation: slow, and only past the deadband                              #
# --------------------------------------------------------------------------- #
def _escalated_to(level_d: float) -> AlertStateMachine:
    sm = AlertStateMachine(CFG)
    _run(sm, [[_risk(level_d)]] * 3)
    return sm


def test_deadband_holds_the_level():
    """0.55 m is past the 0.50 m warning line but inside its 0.10 m deadband: stay."""
    sm = _escalated_to(0.4)
    t0 = _stamps(3)[-1]
    for i in range(1, 200):
        level = sm.update(t0 + round(i * NS_PER_S / FPS), [_risk(0.55)])
    assert level == "warning"


def test_deescalates_once_clear_of_the_deadband():
    sm = _escalated_to(0.4)
    t0 = _stamps(3)[-1]
    levels = [sm.update(t0 + round(i * NS_PER_S / FPS), [_risk(2.0)]) for i in range(1, 60)]
    assert levels[-1] == "none"


def test_deescalation_waits_for_dwell_even_at_high_frame_rate():
    """Six frames at 120 fps is 50 ms. Dwell is 0.6 s of clock time, not six frames."""
    sm = AlertStateMachine(CFG)
    frames = [[_risk(0.1)]] * 2 + [[_risk(2.0)]] * 200
    levels = _run(sm, frames, fps=120.0)
    first_drop = next(i for i in range(2, len(levels)) if levels[i] != "critical")
    held_s = (first_drop - 1) / 120.0
    assert held_s >= CFG["alerts"]["hysteresis"]["min_dwell_s"]


def test_deescalation_needs_frames_to_deescalate_consecutive_frames():
    n = CFG["alerts"]["hysteresis"]["frames_to_deescalate"]
    sm = _escalated_to(0.1)
    t0 = _stamps(3)[-1] + NS_PER_S  # well past dwell
    # n - 1 clear frames, then one critical frame breaks the streak.
    frames = [[_risk(2.0)]] * (n - 1) + [[_risk(0.1)]] + [[_risk(2.0)]] * (n - 1)
    for i, f in enumerate(frames):
        level = sm.update(t0 + i * 1_000_000, f)
    assert level == "critical"


def test_rejects_non_increasing_timestamps():
    sm = AlertStateMachine(CFG)
    sm.update(100, [_risk(2.0)])
    with pytest.raises(ValueError, match="strictly increase"):
        sm.update(100, [_risk(2.0)])


def test_reset_returns_to_none():
    sm = _escalated_to(0.1)
    sm.reset()
    assert sm.level == "none"


def test_levels_are_ordered():
    assert list(LEVELS) == ["none", "caution", "warning", "critical"]
    assert RANK["critical"] > RANK["warning"] > RANK["caution"] > RANK["none"]


# --------------------------------------------------------------------------- #
# ACCEPTANCE: no flicker on noisy fixture input                                #
# --------------------------------------------------------------------------- #
def test_no_flicker_on_noisy_fixture_input():
    """Parked on the WARNING threshold with measurement noise, for ten seconds.

    The raw per-frame level must flicker heavily (otherwise this test would prove
    nothing), and the filtered level must not: no more than two transitions in total, and
    never more than two in any one-second window (the R-10 detection signal).
    """
    rng = np.random.default_rng(20260924)
    parked_forward_m = 0.13
    true_gap = FIXTURE_START_GAP_M + parked_forward_m
    assert true_gap == pytest.approx(CFG["alerts"]["levels"]["warning"]["min_distance_m_below"])

    n = int(10 * FPS)
    noisy = true_gap + rng.normal(0.0, 0.04, size=n)
    frames = [[_risk(float(d))] for d in noisy]

    raw = [raw_level(f, CFG) for f in frames]
    filtered = _run(AlertStateMachine(CFG), frames)

    raw_changes, filt_changes = _changes(raw), _changes(filtered)
    assert len(raw_changes) >= 50, "input is not noisy enough to test anything"
    assert len(filt_changes) <= 2, f"filtered level flickered: {filtered}"

    window = int(FPS)
    for start in range(0, n - window):
        in_window = [c for c in filt_changes if start <= c < start + window]
        assert len(in_window) <= 2, "more than 2 transitions in one second (R-10)"


def test_noisy_approach_only_ever_escalates():
    """Reversing toward the fixture pole with noise: the alert may only get worse.

    A downward flicker mid-approach is exactly the false reassurance the hysteresis
    exists to prevent.
    """
    rng = np.random.default_rng(7)
    speed = 0.25
    t = np.arange(0.0, 5.2, 1.0 / FPS)
    gap = np.clip(1.3 - speed * t + rng.normal(0.0, 0.03, t.size), 0.0, None)
    frames = [[_risk(float(g), ttc=float(g) / speed)] for g in gap]

    levels = _run(AlertStateMachine(CFG), frames)
    ranks = [RANK[lv] for lv in levels]
    assert all(b >= a for a, b in itertools.pairwise(ranks))
    assert levels[-1] == "critical"


def test_a_real_retreat_is_still_reported():
    """Hysteresis must suppress noise, not hide genuine improvement."""
    frames = [[_risk(0.1)]] * 10 + [[_risk(3.0)]] * int(2 * FPS)
    levels = _run(AlertStateMachine(CFG), frames)
    assert levels[9] == "critical"
    assert levels[-1] == "none"
