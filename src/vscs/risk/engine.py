"""Per-frame risk assessment: path fan -> sweep -> reachable horizon -> risks -> alert.

Orchestration only. The algorithms live in ``motion.py``, ``sweep.py`` (core, the
developer's), ``aggregate.py`` and ``alerts.py``; this module wires one frame through them
and applies the one rule none of them can apply alone.

The reachable horizon (ADR 0005)
--------------------------------
Each component is swept independently, so on its own the sweep will happily report that
the sliding door reaches a pole at 2.37 s - even though the rear corner strikes that same
pole at 0.37 s and the van cannot pass through it. Left in, that phantom contact both
inflates the risk and, because the door's severity (2.5) beats the corner's (1.5), makes the
frame name the **door** as the worst component. Component attribution is the metric P3-T6
and P5-T2 are judged on, so that is not a cosmetic error.

The fix: sweep the full horizon once, take the earliest predicted contact ``t*`` over every
component and obstacle, then sweep again with the horizon cut at ``t*``. Nothing after the
first impact is reachable without that impact happening first. The cut is padded by
``sweep.ttc_tolerance_s`` so a contact reported at the upper edge of its tolerance bracket
is still captured on the second pass.

The sweep function is injected rather than imported at module level. That keeps this
module testable before ``risk/sweep.py`` exists (tests pass a brute-force reference), and
keeps §2's rule that modules meet through contracts rather than internals.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from vscs.common.types import RiskFrame
from vscs.risk.aggregate import (
    ClearanceLike,
    aggregate_frame,
    component_risks,
    deterministic_p_contact,
)
from vscs.risk.alerts import AlertStateMachine
from vscs.risk.motion import PathFan, is_moving, path_fan_from_config

#: ``sweep_components(components, obstacles, fan, k, risk_cfg) -> {component: result}``.
SweepFn = Callable[
    [list[Any], list[Any], PathFan, int, dict[str, Any]], Mapping[str, ClearanceLike]
]


def first_contact_s(clearances: Iterable[ClearanceLike]) -> float | None:
    """Earliest predicted contact over all results, or ``None`` if nothing is struck."""
    times = [c.ttc_s for c in clearances if c.ttc_s is not None]
    return min(times) if times else None


def reachable_clearances(
    components: list[Any],
    obstacles: list[Any],
    fan: PathFan,
    k: int,
    risk_cfg: dict[str, Any],
    sweep_fn: SweepFn,
) -> tuple[dict[str, ClearanceLike], float | None]:
    """Sweep, then re-sweep with the horizon cut at the first predicted contact.

    Returns ``(clearances, t_first_contact)``. Without any predicted contact the first pass
    is already the answer and no second sweep is run.
    """
    full = dict(sweep_fn(components, obstacles, fan, k, risk_cfg))
    t_star = first_contact_s(full.values())
    if t_star is None:
        return full, None
    tol = float(risk_cfg["sweep"]["ttc_tolerance_s"])
    cut = fan.truncated(min(t_star + tol, float(fan.t_s[-1])))
    return dict(sweep_fn(components, obstacles, cut, k, risk_cfg)), t_star


def assess_frame(
    *,
    t_ns: int,
    ego_speed_mps: float,
    curvature: float,
    components: list[Any],
    obstacles: list[Any],
    obstacle_kinds: Mapping[int, str],
    state_machine: AlertStateMachine,
    risk_cfg: dict[str, Any],
    severity_cfg: dict[str, Any],
    sweep_fn: SweepFn | None = None,
    p_contact_fn: Callable[[float | None], float] = deterministic_p_contact,
) -> RiskFrame:
    """Assess one frame end to end and return its ``RiskFrame``.

    ``curvature`` is the current estimate (``motion.curvature_from_yaw_rate``); the
    reported risk comes from the fan path nearest to it, as agreed for v0.1. Below
    ``motion.min_speed_mps`` only the current pose is assessed: nothing is moving, but a
    close static obstacle still matters.
    """
    if sweep_fn is None:
        from vscs.risk.sweep import sweep_components

        sweep_fn = sweep_components

    moving = is_moving(ego_speed_mps, risk_cfg)
    fan = path_fan_from_config(ego_speed_mps if moving else 0.0, risk_cfg)
    if not moving:
        fan = fan.truncated(0.0)
    k = fan.index_of_curvature(curvature)

    clearances, _ = reachable_clearances(components, obstacles, fan, k, risk_cfg, sweep_fn)
    risks = component_risks(
        clearances.values(),
        obstacle_kinds,
        ego_speed_mps,
        severity_cfg,
        risk_cfg,
        p_contact_fn=p_contact_fn,
    )
    level = state_machine.update(t_ns, risks)
    return aggregate_frame(t_ns, ego_speed_mps, risks, level)
