"""Severity lookup, per-component risk, and frame aggregation (P3-T5).

Turns per-component clearance results into §4.2 ``ComponentRisk`` records and a
``RiskFrame``::

    risk_i = p_contact_i * severity_i * speed_factor

Severity is the component's cost from ``configs/severity.yaml``, scaled by what it is
about to hit (``obstacle_multiplier``), with specific component/obstacle pairs
overridden where the simple product is wrong (a kerb against the underbody is worse than
the product suggests; a kerb against a tyre is milder).

v0.1 has no real contact probability - that is P4-T7, ``risk/probability.py``, a core
module. Until then :func:`deterministic_p_contact` stands in: 1 if contact is predicted
within the horizon, else 0. This is CLAUDE.md §6 cut-order fallback #3 ("use
clearance/TTC thresholds"), and it is named and injected as a parameter precisely so the
real model replaces it without touching anything else.

This module consumes anything shaped like a clearance result (``component``,
``obstacle_id``, ``min_distance_m``, ``ttc_s``) rather than importing ``risk/sweep.py``
directly, so the two can be built and tested independently (§2: modules communicate
through contracts, not internals).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Protocol

from vscs.common.types import AlertLevel, ComponentRisk, RiskFrame


class ClearanceLike(Protocol):
    """The fields :func:`component_risks` needs from a sweep result."""

    component: str
    obstacle_id: int | None
    min_distance_m: float
    ttc_s: float | None


def severity_for(component: str, obstacle_kind: str | None, severity_cfg: dict[str, Any]) -> float:
    """Unitless severity of ``component`` striking an obstacle of ``obstacle_kind``.

    Resolution order: an explicit ``<component>__<kind>`` override wins; otherwise the
    component's base cost times the obstacle multiplier. An unknown component falls back
    to ``default`` rather than zero, so a vocabulary gap under-ranks a component instead
    of silently making it free to hit.
    """
    base = float(severity_cfg["components"].get(component, severity_cfg["default"]))
    if obstacle_kind is None:
        return base
    override = severity_cfg.get("overrides", {}).get(f"{component}__{obstacle_kind}")
    if override is not None:
        return float(override)
    multiplier = float(severity_cfg["obstacle_multiplier"].get(obstacle_kind, 1.0))
    return base * multiplier


def speed_factor(speed_mps: float, risk_cfg: dict[str, Any]) -> float:
    """``clamp(|v| / reference_speed, min_factor, max_factor)``. Direction is irrelevant."""
    sf = risk_cfg["aggregate"]["speed_factor"]
    if sf["method"] != "linear_clamped":
        raise ValueError(f"unsupported speed_factor method {sf['method']!r}")
    raw = abs(float(speed_mps)) / float(sf["reference_speed_mps"])
    return min(max(raw, float(sf["min_factor"])), float(sf["max_factor"]))


def deterministic_p_contact(ttc_s: float | None) -> float:
    """v0.1 stand-in for a contact probability: 1 if contact is predicted, else 0.

    Replaced by ``risk/probability.py`` (``p = Phi(-d / sigma)``) at P4-T7.
    """
    return 0.0 if ttc_s is None else 1.0


def component_risks(
    clearances: Iterable[ClearanceLike],
    obstacle_kinds: Mapping[int, str],
    ego_speed_mps: float,
    severity_cfg: dict[str, Any],
    risk_cfg: dict[str, Any],
    p_contact_fn: Callable[[float | None], float] = deterministic_p_contact,
) -> list[ComponentRisk]:
    """Build §4.2 ``ComponentRisk`` records, sorted by risk (highest first).

    ``obstacle_kinds`` maps obstacle id to its §4.2 ``kind`` so severity can depend on
    what is being hit - a person always dominates.
    """
    sf = speed_factor(ego_speed_mps, risk_cfg)
    out: list[ComponentRisk] = []
    for c in clearances:
        kind = obstacle_kinds.get(c.obstacle_id) if c.obstacle_id is not None else None
        severity = severity_for(c.component, kind, severity_cfg)
        p = float(p_contact_fn(c.ttc_s))
        out.append(
            ComponentRisk(
                component=c.component,
                min_distance_m=float(c.min_distance_m),
                ttc_s=c.ttc_s,
                p_contact=p,
                severity=severity,
                risk=p * severity * sf,
                obstacle_id=c.obstacle_id,
            )
        )
    out.sort(key=lambda r: (-r.risk, r.min_distance_m))
    return out


def aggregate_frame(
    t_ns: int,
    ego_speed_mps: float,
    risks: list[ComponentRisk],
    alert_level: AlertLevel = "none",
) -> RiskFrame:
    """Assemble the frame. The aggregate fields themselves are defined by the schema."""
    return RiskFrame.from_components(
        t_ns=t_ns, ego_speed_mps=ego_speed_mps, per_component=risks, alert_level=alert_level
    )
