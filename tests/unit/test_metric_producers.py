"""Every gated metric has a producer: calibrate.py (P0-T7) and evaluate.py (P3-T6).

metrics/results.jsonl is machine-written only (CLAUDE.md §7.4), so a script must append
each figure its gate in eval.yaml is checked against. These tests catch the two gaps found
on 2026-10-01: calibrate.py told the developer to append its metric by hand, and nothing
wrote P3-T6's ``component_flagged_frac``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from vscs.capture.calib import CalibrationResult
from vscs.common.config import load_config, repo_root


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"cli_{name}", repo_root() / "scripts" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _result(err):
    K = np.array([[900.0, 0, 639.5], [0, 900.0, 359.5], [0, 0, 1]])
    return CalibrationResult(
        K=K,
        dist=np.zeros(5),
        image_size=(1280, 720),
        mean_reprojection_error_px=err,
        max_reprojection_error_px=err * 1.5,
        per_image_error_px={"a.png": err},
        n_images_used=18,
        n_images_total=18,
    )


@pytest.fixture()
def calib_cli(monkeypatch):
    cli = _load("calibrate")
    calls = {"metric": [], "config": []}
    monkeypatch.setattr(cli, "find_images", lambda d: [Path("a.png")] * 18)
    monkeypatch.setattr(cli, "append_metric", lambda **kw: calls["metric"].append(kw))
    monkeypatch.setattr(
        cli, "write_to_capture_config", lambda r, **kw: calls["config"].append(r) or Path("x")
    )
    return cli, calls


@pytest.mark.parametrize(("err", "code"), [(0.3, 0), (0.8, 1)])
def test_calibrate_logs_its_metric_pass_or_fail(calib_cli, monkeypatch, tmp_path, err, code):
    cli, calls = calib_cli
    monkeypatch.setattr(cli, "calibrate_intrinsics", lambda *a, **k: _result(err))
    args = ["--images", str(tmp_path), "--device", "phone_main", "--out-root", str(tmp_path)]
    assert cli.main(args) == code
    (m,) = calls["metric"]
    assert (m["task"], m["metric"], m["value"]) == ("P0-T7", "reprojection_error_px", err)
    (run,) = list((tmp_path / "capture").iterdir())
    saved = json.loads((run / "calibration_result.json").read_text(encoding="utf-8"))
    assert saved["passed"] is (code == 0) and saved["K"][0][0] == 900.0
    assert len(calls["config"]) == (1 if code == 0 else 0)  # only a pass touches the config


def test_calibrate_dry_run_writes_nothing(calib_cli, monkeypatch, tmp_path):
    cli, calls = calib_cli
    monkeypatch.setattr(cli, "calibrate_intrinsics", lambda *a, **k: _result(0.3))
    args = ["--images", str(tmp_path), "--dry-run", "--out-root", str(tmp_path)]
    assert cli.main(args) == 0
    assert not calls["metric"] and not calls["config"] and not (tmp_path / "capture").exists()


def test_every_gated_metric_has_a_producer():
    """Each eval.yaml threshold is appended by some script (or is a known gap below)."""
    produced = {
        p.stem: p.read_text(encoding="utf-8") for p in (repo_root() / "scripts").glob("*.py")
    }
    # Gates whose producer is planned with its phase, not yet written. Listed, not hidden.
    planned = {
        "egomotion_drift_frac": "P4-T2: needs lot runs with surveyed paths",
        "depth_error_m_at_3m": "P4-T1: needs lot ground truth by range bucket",
        "pipeline_smoke_seconds": "P0-T4: the smoke test is skipped until sweep.py exists",
    }
    missing = []
    for metric in load_config("eval")["thresholds"]:
        if metric in planned:
            continue
        if not any(f'"{metric}"' in src or f"'{metric}'" in src for src in produced.values()):
            missing.append(metric)
    assert not missing, f"gated metrics nothing appends: {missing}"


def test_evaluate_logs_the_p3_t6_gate_on_dev_only(tmp_path, monkeypatch):
    """VSCS flags the right corner 3 s before the event (as in test_evaluate)."""
    import yaml

    from vscs.common.io import write_jsonl
    from vscs.common.types import ComponentRisk, RiskFrame

    S = 1_000_000_000

    def _f(t, level, worst, ttc=None):
        cr = ComponentRisk(
            component=worst,
            min_distance_m=0.0 if ttc else 1.0,
            ttc_s=ttc,
            p_contact=1.0 if ttc else 0.0,
            severity=1.5,
            risk=1.5 if ttc else 0.0,
            obstacle_id=1,
        )
        return RiskFrame(
            t_ns=round(t * S),
            ego_speed_mps=-1.0,
            per_component=[cr],
            p_any_contact=0.0,
            expected_damage=0.0,
            worst_component=worst if level != "none" else None,
            alert_level=level,
        )

    eval_copy = tmp_path / "eval.yaml"
    eval_copy.write_text(
        (repo_root() / "configs" / "eval.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    from vscs.common.config import replace_top_level_block

    replace_top_level_block(
        eval_copy, "splits", {**load_config("eval")["splits"], "dev": ["p1"], "test": ["p2"]}
    )
    for name, frames in (
        (
            "v",
            [
                _f(0, "none", "rear_right_bumper_corner"),
                _f(2, "warning", "rear_right_bumper_corner", 3.0),
            ],
        ),
        ("b", [_f(0, "none", "vehicle"), _f(3, "warning", "vehicle", 2.0)]),
    ):
        (tmp_path / name).mkdir()
        write_jsonl(tmp_path / name / "risk_frames.jsonl", frames)
    truth = {
        "component": "rear_right_bumper_corner",
        "t_event_ns": 5 * S,
        "event_onsets_ns": [5 * S],
    }

    def _passes(pass_id):
        path = tmp_path / f"passes_{pass_id}.yaml"
        one = {"id": pass_id, "vscs": str(tmp_path / "v"), "baseline": str(tmp_path / "b")}
        path.write_text(yaml.safe_dump({"passes": [{**one, "truth": truth}]}), encoding="utf-8")
        return path

    cli = _load("evaluate")
    logged = []
    monkeypatch.setattr(cli, "append_metric", lambda **kw: logged.append(kw))
    base = [
        "--eval-config",
        str(eval_copy),
        "--out-root",
        str(tmp_path / "o"),
        "--log-metrics",
    ]
    assert cli.main([*base, "--passes", str(_passes("p1")), "--split", "dev"]) == 0
    gate = [m for m in logged if m["metric"] == "component_flagged_frac"]
    assert len(gate) == 1 and gate[0]["task"] == "P3-T6" and gate[0]["value"] == 1.0
    logged.clear()
    assert cli.main([*base, "--passes", str(_passes("p2")), "--split", "test", "--final"]) == 0
    assert not [m for m in logged if m["metric"] == "component_flagged_frac"]  # P5-T2 reports
