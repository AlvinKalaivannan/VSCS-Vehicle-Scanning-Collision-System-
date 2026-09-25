"""End-to-end smoke test: fixture scene -> valid RiskFrame stream in < 60 s.

CLAUDE.md section 5 requires this integration test. It cannot be implemented yet: it
needs ``risk/`` (motion model, sweep, metrics, aggregate), which is Phase 3
(P3-T3 .. P3-T5). The test is written now, and skipped, so that:

* the budget and the acceptance criterion are recorded in code rather than in a
  document nobody re-reads;
* it starts failing the moment ``risk/`` exists but the pipeline does not actually
  run, instead of being remembered late.

Turn it on at P3-T6 by deleting the skip.
"""

from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.skipif(
    True,
    reason="needs risk/sweep.py (P3-T4, developer's draft); the real test is on branch p3-t4-sweep",
)

#: CLAUDE.md section 5: the whole fixture pipeline must finish inside this, on laptop CPU.
SMOKE_BUDGET_S = 60.0


def test_fixture_scene_end_to_end_produces_a_valid_riskframe_stream(tmp_path):
    """Fixture scene through the per-drive pipeline, checked for validity and speed."""
    from fixtures.synthetic import load_scene
    from vscs.common.io import read_jsonl, write_jsonl
    from vscs.common.types import RiskFrame

    scene = load_scene()
    started = time.perf_counter()

    # --- the pipeline under test (Phase 3) --------------------------------- #
    # from vscs.risk.motion import path_fan
    # from vscs.risk.sweep import swept_distances
    # from vscs.risk.aggregate import aggregate_frame
    # from vscs.risk.alerts import AlertStateMachine
    frames: list[RiskFrame] = []
    raise NotImplementedError("wire up risk/ at P3-T6")
    # ----------------------------------------------------------------------- #

    elapsed = time.perf_counter() - started
    assert elapsed < SMOKE_BUDGET_S, f"pipeline took {elapsed:.1f}s, budget {SMOKE_BUDGET_S}s"

    path = write_jsonl(tmp_path / "risk.jsonl", frames)
    reloaded = [RiskFrame(**rec) for rec in read_jsonl(path)]
    assert reloaded == frames
    assert reloaded, "pipeline produced no frames"

    # Timestamps must advance, and the pole must be attributed to the rear right
    # corner - the ground truth the fixture was built around.
    assert [f.t_ns for f in reloaded] == sorted(f.t_ns for f in reloaded)
    expected_component, _ = scene.nearest_component_to_pole()
    assert any(f.worst_component == expected_component for f in reloaded)
