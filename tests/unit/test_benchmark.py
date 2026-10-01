"""P5-T3 benchmark timing logic, on a fake clock."""

from __future__ import annotations

import pytest

from vscs.eval.benchmark import StageTiming, hardware_tag, pipeline_fps, time_stage


class Clock:
    def __init__(self):
        self.t = 0

    def __call__(self):
        return self.t


def test_warmup_is_excluded_and_percentiles_are_right():
    clk = Clock()
    costs_ms = [500, 400, *range(1, 21)]  # two slow warm-up calls, then 1..20 ms

    def stage(cost):
        clk.t += round(cost * 1e6)

    t = time_stage("detect", stage, costs_ms, warmup=2, clock=clk)
    assert t.n == 20
    assert t.p50_ms == pytest.approx(10.5) and t.p95_ms == pytest.approx(19.05)
    assert t.fps == pytest.approx(1000 / 10.5)


def test_pipeline_rate_sums_stage_latencies():
    ts = [StageTiming("a", 10, 40.0, 50.0), StageTiming("b", 10, 26.67, 30.0)]
    assert pipeline_fps(ts) == pytest.approx(1000 / 66.67)


def test_too_few_inputs_is_an_error():
    with pytest.raises(ValueError, match="warm-up"):
        time_stage("x", lambda v: v, [1, 2], warmup=2)


def test_hardware_tag_names_the_machine():
    assert len(hardware_tag()) > 3


def test_cli_runs_synthetic_without_logging_metrics(tmp_path):
    import importlib.util
    import json

    from vscs.common.config import repo_root

    spec = importlib.util.spec_from_file_location(
        "bench_cli", repo_root() / "scripts" / "benchmark.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.main(["--synthetic", "8", "--warmup", "2", "--out-root", str(tmp_path)]) == 0
    (run,) = list((tmp_path / "eval").iterdir())
    out = json.loads((run / "benchmark.json").read_text(encoding="utf-8"))
    assert {s["stage"] for s in out["stages"]} == {"decode", "perception"} and out["hardware"]
