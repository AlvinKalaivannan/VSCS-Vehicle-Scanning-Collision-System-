"""P5-T2 scoring and the R-09 test-split guard."""

from __future__ import annotations

from datetime import date

import pytest

from vscs.common.config import load_config
from vscs.common.types import ComponentRisk, RiskFrame
from vscs.eval.evaluate import PassTruthFull, guard_split, mark_test_split_used, score_system

S = 1_000_000_000
EV = load_config("eval")


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


TRUTH = PassTruthFull("lot_001", "rear_right_bumper_corner", 5 * S, [5 * S])


def test_vscs_and_baseline_scores():
    vscs = [
        _f(0, "none", "rear_right_bumper_corner"),
        _f(2, "warning", "rear_right_bumper_corner", 3.0),
        _f(60, "none", "rear_right_bumper_corner"),
    ]
    # A true warning 2 s before the event, then a separate spurious one at 30 s.
    base = [
        _f(0, "none", "vehicle"),
        _f(3, "warning", "vehicle", 2.0),
        _f(4, "none", "vehicle"),
        _f(30, "warning", "vehicle"),
        _f(31, "none", "vehicle"),
        _f(60, "none", "vehicle"),
    ]
    rm = EV["risk_metrics"]
    v = score_system(
        [(vscs, TRUTH)], rm["flag_level"], rm["false_alarm_horizon_s"], attributes=True
    )
    b = score_system(
        [(base, TRUTH)], rm["flag_level"], rm["false_alarm_horizon_s"], attributes=False
    )
    assert v["component_attribution_accuracy"] == 1.0 and v["lead_time_s_median"] == pytest.approx(
        3.0
    )
    assert v["ttc_error_s_median"] == pytest.approx(0.0) and v[
        "false_alarms_per_min"
    ] == pytest.approx(0.0)
    assert b["component_attribution_accuracy"] is None  # n/a, never 0 or 1
    assert b["lead_time_s_median"] == pytest.approx(2.0)
    assert b["false_alarms_per_min"] == pytest.approx(1.0)  # the 30 s episode, over one minute


def test_the_test_split_is_touched_once(tmp_path):
    with pytest.raises(PermissionError, match="final"):
        guard_split("test", {"test_split_used": False}, final=False)
    guard_split("test", {"test_split_used": False}, final=True)  # allowed exactly once
    with pytest.raises(PermissionError, match="already used"):
        guard_split(
            "test", {"test_split_used": True, "test_split_used_date": "2027-03-20"}, final=True
        )
    guard_split("dev", {"test_split_used": True}, final=False)  # dev is always fine
    with pytest.raises(ValueError):
        guard_split("train", {}, final=False)


def test_marking_the_test_split_used_keeps_the_rest_of_the_file(tmp_path):
    src = (load_config.__globals__["config_dir"]() / "eval.yaml").read_text(encoding="utf-8")
    p = tmp_path / "eval.yaml"
    p.write_text(src, encoding="utf-8")
    mark_test_split_used(p, EV["splits"], on=date(2027, 3, 20))
    text = p.read_text(encoding="utf-8")
    assert "test_split_used: true" in text and "2027-03-20" in text
    assert "risk_metrics:" in text and "# Evaluation protocol" in text  # comments and blocks kept


def test_cli_dev_then_test_once(tmp_path):
    import importlib.util

    import yaml as _yaml

    from vscs.common.config import repo_root
    from vscs.common.io import write_jsonl

    eval_copy = tmp_path / "eval.yaml"
    eval_copy.write_text(
        (repo_root() / "configs" / "eval.yaml").read_text(encoding="utf-8"), encoding="utf-8"
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
    passes = tmp_path / "passes.yaml"
    passes.write_text(
        _yaml.safe_dump(
            {
                "passes": [
                    {
                        "id": "lot_001",
                        "vscs": str(tmp_path / "v"),
                        "baseline": str(tmp_path / "b"),
                        "truth": {
                            "component": "rear_right_bumper_corner",
                            "t_event_ns": 5 * S,
                            "event_onsets_ns": [5 * S],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location(
        "eval_cli", repo_root() / "scripts" / "evaluate.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    base_args = [
        "--passes",
        str(passes),
        "--eval-config",
        str(eval_copy),
        "--out-root",
        str(tmp_path / "o"),
    ]
    assert cli.main([*base_args, "--split", "dev"]) == 0
    assert cli.main([*base_args, "--split", "test"]) == 2  # no --final
    assert cli.main([*base_args, "--split", "test", "--final"]) == 0  # the one permitted run
    assert cli.main([*base_args, "--split", "test", "--final"]) == 2  # never twice
    assert "test_split_used: true" in eval_copy.read_text(encoding="utf-8")
