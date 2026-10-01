"""Score recorded passes: VSCS vs the single-box baseline on the §11 risk metrics (P5-T2).

Inputs per pass: the VSCS ``risk_frames.jsonl``, the baseline's (``scripts/risk.py
--baseline``), and the pass's ground truth: which component truly comes closest, when, and
when any genuine close approach begins (``truth.yaml``, from the tape-measured lot layout,
P1-T3). Outputs, for each system: component attribution accuracy (VSCS only: the baseline
cannot attribute, so it is reported as n/a, never 0 or 1), median warning lead time and how
many passes were never warned, median TTC error, and false alarms per minute.

R-09 - the test split is touched ONCE. ``guard_split`` refuses ``test`` unless the caller
says ``final=True``, and refuses outright if ``eval.yaml`` already records that the test
split was used. ``mark_test_split_used`` writes that record, with the date, after a final
run. Tuning happens on ``dev`` only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from vscs.common.config import replace_top_level_block
from vscs.common.types import RiskFrame
from vscs.eval import metrics as Mx


@dataclass(frozen=True)
class PassTruthFull:
    pass_id: str
    component: str
    t_event_ns: int
    event_onsets_ns: list[int] = field(default_factory=list)

    @property
    def truth(self) -> Mx.PassTruth:
        return Mx.PassTruth(self.component, self.t_event_ns)


def guard_split(split: str, splits_cfg: dict[str, Any], *, final: bool) -> None:
    if split not in ("dev", "test"):
        raise ValueError(f"split must be dev or test, got {split!r}")
    if split == "test":
        if splits_cfg.get("test_split_used"):
            raise PermissionError(
                f"the test split was already used on {splits_cfg.get('test_split_used_date')}; "
                "R-09: it is touched once. Re-evaluating needs the developer's decision (ADR)."
            )
        if not final:
            raise PermissionError(
                "the test split is used once, at P5-T2: pass final=True (--final)"
            )


def mark_test_split_used(
    eval_yaml: Path, splits_cfg: dict[str, Any], on: date | None = None
) -> None:
    block = dict(splits_cfg)
    block["test_split_used"] = True
    block["test_split_used_date"] = (on or date.today()).isoformat()
    replace_top_level_block(eval_yaml, "splits", block)


def score_system(
    passes: list[tuple[list[RiskFrame], PassTruthFull]],
    flag_level: str,
    horizon_s: float,
    *,
    attributes: bool,
) -> dict[str, Any]:
    """The §11 risk metrics for one system over a set of passes."""
    leads = [Mx.lead_time_s(fr, t.truth, flag_level) for fr, t in passes]
    ttc = (
        np.concatenate([Mx.ttc_errors_s(fr, t.truth) for fr, t in passes])
        if passes
        else np.zeros(0)
    )
    total_min = sum((fr[-1].t_ns - fr[0].t_ns) / 1e9 / 60.0 for fr, _ in passes if len(fr) > 1)
    false = sum(
        Mx.false_alarms_per_min(fr, t.event_onsets_ns, flag_level, horizon_s)
        * ((fr[-1].t_ns - fr[0].t_ns) / 6e10)
        for fr, t in passes
        if len(fr) > 1
    )
    warned = [x for x in leads if x is not None]
    return {
        "n_passes": len(passes),
        "component_attribution_accuracy": Mx.attribution_accuracy(passes, flag_level)
        if attributes
        else None,
        "lead_time_s_median": float(np.median(warned)) if warned else None,
        "passes_never_warned": sum(x is None for x in leads),
        "ttc_error_s_median": float(np.median(ttc)) if len(ttc) else None,
        "false_alarms_per_min": false / total_min if total_min > 0 else None,
    }
