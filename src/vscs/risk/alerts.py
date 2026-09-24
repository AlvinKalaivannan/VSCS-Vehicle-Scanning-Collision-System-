"""Alert levels and the anti-flicker hysteresis state machine (P3-T5).

Acceptance criterion: "Hysteresis test: no flicker on noisy fixture input". R-10 names
the failure: an alert that toggles more than twice a second is noise a driver learns to
ignore, which is worse than no alert at all.

Two stages
----------
1. :func:`raw_level` classifies one frame. The level is set by the **worst single
   component**, not an aggregate, so a mirror about to be struck is not averaged away by
   ten safe components. A component meets a level if its clearance *or* its time to
   contact is inside that level's threshold (and its contact probability is above it, once
   ``alerts.use_p_contact`` is enabled at P4-T7).

2. :class:`AlertStateMachine` filters the raw sequence, deliberately asymmetrically:

   * **Escalate fast.** A higher level must persist for ``frames_to_escalate`` consecutive
     frames (2 by default) - enough to reject a single-frame glitch, short enough not to
     delay a real warning. No dwell time applies to escalation.
   * **De-escalate slowly.** Three conditions must *all* hold before stepping down:
     the lower level has persisted for ``frames_to_deescalate`` frames (6), the current
     level has been held for at least ``min_dwell_s`` (0.6 s), and the clearance has
     cleared the threshold by ``deescalate_distance_hysteresis_m`` (0.10 m).

The deadband is what actually kills flicker. Without it, a clearance hovering at 0.50 m
with a few centimetres of measurement noise crosses the 0.50 m warning threshold every
other frame. With it, entering WARNING needs ``d < 0.50`` but leaving needs ``d > 0.60``,
so noise smaller than the deadband cannot produce a transition in either direction.

Dwell is measured in **nanoseconds from the frame timestamps**, never in frame counts,
because phone footage is variable frame rate (§4.1, R-03). Six frames at 120 fps is 50 ms;
the 0.6 s dwell has to hold regardless.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from vscs.common.types import NS_PER_S, AlertLevel, ComponentRisk

#: Ordered from least to most severe.
LEVELS: tuple[AlertLevel, ...] = ("none", "caution", "warning", "critical")
RANK: dict[str, int] = {name: i for i, name in enumerate(LEVELS)}


def _meets(
    r: ComponentRisk,
    thresholds: dict[str, Any],
    *,
    use_p: bool,
    distance_offset_m: float,
) -> bool:
    if r.min_distance_m < float(thresholds["min_distance_m_below"]) + distance_offset_m:
        return True
    if r.ttc_s is not None and r.ttc_s < float(thresholds["ttc_s_below"]):
        return True
    return use_p and r.p_contact > float(thresholds["p_contact_above"])


def raw_level(
    risks: Iterable[ComponentRisk],
    risk_cfg: dict[str, Any],
    *,
    distance_offset_m: float = 0.0,
) -> AlertLevel:
    """Unfiltered alert level for one frame: the highest level any component meets.

    ``distance_offset_m`` widens every clearance threshold. The state machine uses it to
    compute the *release* level - the level a frame must fall below, with the deadband
    applied, before de-escalation can begin.
    """
    alerts = risk_cfg["alerts"]
    use_p = bool(alerts.get("use_p_contact", False))
    risks = list(risks)
    for level in reversed(LEVELS[1:]):
        thresholds = alerts["levels"][level]
        if any(
            _meets(r, thresholds, use_p=use_p, distance_offset_m=distance_offset_m) for r in risks
        ):
            return level
    return "none"


class AlertStateMachine:
    """Filters raw per-frame levels into a stable alert. One instance per drive."""

    def __init__(self, risk_cfg: dict[str, Any]) -> None:
        self._cfg = risk_cfg
        h = risk_cfg["alerts"]["hysteresis"]
        self._frames_up = int(h["frames_to_escalate"])
        self._frames_down = int(h["frames_to_deescalate"])
        self._dwell_ns = round(float(h["min_dwell_s"]) * NS_PER_S)
        self._deadband_m = float(h["deescalate_distance_hysteresis_m"])
        if self._frames_up < 1 or self._frames_down < 1:
            raise ValueError("frames_to_escalate and frames_to_deescalate must be >= 1")
        self.reset()

    def reset(self) -> None:
        """Back to ``none`` with no history, e.g. at the start of a new manoeuvre."""
        self.level: AlertLevel = "none"
        self._entered_ns: int | None = None
        self._last_t_ns: int | None = None
        self._up_count = 0
        self._up_target: AlertLevel | None = None
        self._down_count = 0
        self._down_target: AlertLevel | None = None

    def update(self, t_ns: int, risks: Iterable[ComponentRisk]) -> AlertLevel:
        """Feed one frame and return the filtered level.

        ``t_ns`` must be strictly increasing; a repeated or backwards timestamp is a
        pipeline bug (R-03) and is rejected rather than silently absorbed.
        """
        if self._last_t_ns is not None and t_ns <= self._last_t_ns:
            raise ValueError(
                f"alert timestamps must strictly increase: {self._last_t_ns} -> {t_ns}"
            )
        self._last_t_ns = t_ns
        if self._entered_ns is None:
            self._entered_ns = t_ns

        risks = list(risks)
        raw = raw_level(risks, self._cfg)
        release = raw_level(risks, self._cfg, distance_offset_m=self._deadband_m)
        current = RANK[self.level]

        if RANK[raw] > current:
            # Escalation streak. Target the level sustained across the WHOLE streak, so a
            # warning,critical pair escalates to warning, not critical.
            self._down_count, self._down_target = 0, None
            self._up_count += 1
            self._up_target = (
                raw if self._up_target is None else min(self._up_target, raw, key=RANK.__getitem__)
            )
            if self._up_count >= self._frames_up:
                self._change(self._up_target, t_ns)
        elif RANK[release] < current:
            # De-escalation streak. Target the HIGHEST release level seen in the streak,
            # so stepping down is always to the most cautious level the streak supports.
            self._up_count, self._up_target = 0, None
            self._down_count += 1
            self._down_target = (
                release
                if self._down_target is None
                else max(self._down_target, release, key=RANK.__getitem__)
            )
            dwell_ok = t_ns - self._entered_ns >= self._dwell_ns
            if self._down_count >= self._frames_down and dwell_ok:
                self._change(self._down_target, t_ns)
        else:
            # Inside the deadband, or steady: neither streak continues.
            self._up_count, self._up_target = 0, None
            self._down_count, self._down_target = 0, None
        return self.level

    def _change(self, level: AlertLevel, t_ns: int) -> None:
        self.level = level
        self._entered_ns = t_ns
        self._up_count, self._up_target = 0, None
        self._down_count, self._down_target = 0, None
