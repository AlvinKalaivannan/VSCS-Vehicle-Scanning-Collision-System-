"""End-to-end smoke test: fixture scene -> valid RiskFrame stream in < 60 s (CLAUDE.md §5).

Runs the real per-drive risk pipeline - motion fan, the developer's ``risk/sweep.py``,
the reachable-horizon cut, aggregation and the alert state machine - over a simulated
reversing manoeuvre in the fixture world.

**Red until ``risk/sweep.py`` is implemented** (P3-T4, the developer's first draft). Every
other stage already exists and is tested on its own; this is the test that proves they fit
together, and it is also P3-T6's acceptance criterion in miniature: the correct component is
flagged before the vehicle reaches the obstacle.

The manoeuvre
-------------
The van reverses straight at 1 m/s at 30 fps. Obstacles are static in the world frame
(fixed at the van's pose at t = 0, §4.1), so in each frame's vehicle frame they sit
``|v| * t`` closer. The run stops before the predicted impact: the rear-right corner meets
the pole 0.37 s in, so the last frame is at 0.33 s.
"""

from __future__ import annotations

import time

import pytest

from fixtures.synthetic import load_scene
from vscs.common.config import load_config
from vscs.common.io import read_jsonl, write_jsonl
from vscs.common.types import NS_PER_S, Obstacle, RiskFrame, validate_timestamp_stream
from vscs.risk import sweep as S
from vscs.risk.alerts import RANK, AlertStateMachine
from vscs.risk.engine import assess_frame

#: CLAUDE.md §5: the whole fixture pipeline must finish inside this on laptop CPU.
SMOKE_BUDGET_S = 60.0

RISK = load_config("risk")
SEV = load_config("severity")
SPEED = -1.0
FPS = 30.0
IMPACT_S = 0.37  # rear-right corner to pole, inflated gap, at 1 m/s
POLE_ID, KERB_ID = 1, 2


def _obstacle(oid: int, lo, hi, shift_x: float) -> Obstacle:
    c = [(a + b) / 2 for a, b in zip(lo, hi, strict=True)]
    e = [b - a for a, b in zip(lo, hi, strict=True)]
    return Obstacle(
        id=oid,
        t_ns=0,
        kind="static_geom",
        center_veh=(c[0] + shift_x, c[1], c[2]),
        extent=tuple(e),
        pos_sigma_m=0.05,
        source="occupancy",
    )


def _run_drive():
    scene = load_scene()
    components = [S.box_footprint(n, b.lo, b.hi) for n, b in scene.components.items()]
    (ax, ay), r, h = scene.pole.axis_xy, scene.pole.radius_m, scene.pole.height_m
    pole_lo, pole_hi = [ax - r, ay - r, 0.0], [ax + r, ay + r, h]
    kinds = {POLE_ID: "static_geom", KERB_ID: "static_geom"}

    sm = AlertStateMachine(RISK)
    frames: list[RiskFrame] = []
    n = int(IMPACT_S * FPS)  # stop before impact
    for i in range(n):
        t_s = i / FPS
        advance = abs(SPEED) * t_s  # how much closer static obstacles are by now
        obstacles = [
            S.obstacle_footprint(_obstacle(POLE_ID, pole_lo, pole_hi, advance)),
            S.obstacle_footprint(_obstacle(KERB_ID, scene.curb.lo, scene.curb.hi, advance)),
        ]
        frames.append(
            assess_frame(
                t_ns=round(t_s * NS_PER_S) + 1,
                ego_speed_mps=SPEED,
                curvature=0.0,
                components=components,
                obstacles=obstacles,
                obstacle_kinds=kinds,
                state_machine=sm,
                risk_cfg=RISK,
                severity_cfg=SEV,
            )
        )
    return frames


@pytest.fixture(scope="module")
def drive():
    started = time.perf_counter()
    frames = _run_drive()
    return frames, time.perf_counter() - started


def test_runs_inside_the_budget(drive):
    frames, elapsed = drive
    assert frames, "pipeline produced no frames"
    assert elapsed < SMOKE_BUDGET_S, f"pipeline took {elapsed:.1f}s, budget {SMOKE_BUDGET_S}s"


def test_stream_is_valid_json_lines(drive, tmp_path):
    frames, _ = drive
    path = write_jsonl(tmp_path / "risk.jsonl", frames)
    reloaded = [RiskFrame(**rec) for rec in read_jsonl(path)]
    assert reloaded == frames
    validate_timestamp_stream([f.t_ns for f in reloaded])


def test_the_correct_component_is_flagged_before_impact(drive):
    """P3-T6's criterion in miniature: every frame names the rear-right corner."""
    frames, _ = drive
    for f in frames:
        assert f.worst_component == "rear_right_bumper_corner", (
            f"t={f.t_ns / NS_PER_S:.3f}s named {f.worst_component}"
        )


def test_time_to_contact_counts_down(drive):
    frames, _ = drive
    for f in frames:
        corner = next(r for r in f.per_component if r.component == "rear_right_bumper_corner")
        expected = IMPACT_S - f.t_ns / NS_PER_S
        assert corner.ttc_s == pytest.approx(expected, abs=RISK["sweep"]["ttc_tolerance_s"] + 1e-3)


def test_alert_escalates_and_never_backs_off_during_the_approach(drive):
    frames, _ = drive
    ranks = [RANK[f.alert_level] for f in frames]
    assert all(b >= a for a, b in zip(ranks, ranks[1:], strict=False))
    assert frames[-1].alert_level == "critical"


def test_bumpers_are_not_blamed_for_the_kerb(drive):
    """The kerb is present throughout; a 0.40 m bumper rides over it."""
    frames, _ = drive
    for f in frames:
        for r in f.per_component:
            if "bumper" in r.component and r.obstacle_id == KERB_ID:
                pytest.fail(f"{r.component} attributed to the kerb")
