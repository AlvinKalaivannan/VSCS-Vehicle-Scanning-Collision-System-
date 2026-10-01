"""Wheelbase from the segmented wheels (P1-T6 wheelbase check, ADR 0013)."""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest
import yaml

from fixtures.synthetic import load_scene
from vscs.common.config import load_config, repo_root
from vscs.eval.dimensions import hub_x, wheelbase_from_wheels

SCENE = load_scene()
CFG = load_config("eval")["wheelbase"]


def _cloud(points_per_component=400, strays=0, seed=0):
    pts, lab_names = SCENE.labelled_point_cloud(points_per_component, seed=20260924)
    names = sorted(SCENE.components)
    labels = np.array([names.index(n) for n in lab_names])
    if strays:
        # Stray labels: a few far-away points mislabelled as the front-left wheel.
        rng = np.random.default_rng(seed)
        extra = rng.uniform([-1.0, -1.0, 0.0], [4.0, 1.0, 2.0], (strays, 3))
        pts = np.vstack([pts, extra])
        labels = np.r_[labels, np.full(strays, names.index("wheel_front_left"))]
    return pts, labels, names


def _wb(pts, labels, names, trim=CFG["trim_percentile"]):
    return wheelbase_from_wheels(pts, labels, names, CFG["front_wheels"], CFG["rear_wheels"], trim)


def test_fixture_wheelbase_is_three_metres():
    wb = _wb(*_cloud())
    assert wb.wheelbase_m == pytest.approx(SCENE.wheelbase_m, abs=0.01)
    assert wb.rear_axle_x_m == pytest.approx(0.0, abs=0.01)  # veh origin under the rear axle
    assert wb.front_left_right_gap_m < 0.01 and wb.rear_left_right_gap_m < 0.01


def test_trimming_resists_a_few_stray_labels():
    pts, labels, names = _cloud(strays=5)  # 5 strays among 400 wheel points (1.2%)
    assert _wb(pts, labels, names).wheelbase_m == pytest.approx(3.0, abs=0.02)
    # Without trimming the strays stretch the front-left wheel across the van.
    assert abs(_wb(pts, labels, names, trim=0.0).wheelbase_m - 3.0) > 0.05


def test_left_right_gap_flags_a_bad_wheel():
    pts, labels, names = _cloud()
    sel = labels == names.index("wheel_front_right")
    pts = pts.copy()
    pts[sel, 0] += 0.10  # front-right wheel segmented 10 cm too far forward
    wb = _wb(pts, labels, names)
    assert wb.front_left_right_gap_m == pytest.approx(0.10, abs=0.01)


def test_missing_wheels_and_tiny_wheels_are_refused():
    pts, labels, names = _cloud()
    with pytest.raises(ValueError, match="not in the labelled cloud"):
        wheelbase_from_wheels(pts, labels, names, ["wheel_middle"], CFG["rear_wheels"], 2.0)
    with pytest.raises(ValueError, match="too few"):
        hub_x([1.0, 2.0], 2.0)


def _cli():
    spec = importlib.util.spec_from_file_location(
        "wb_cli", repo_root() / "scripts" / "check_wheelbase.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(("tape", "code"), [(3.01, 0), (3.05, 1)])
def test_cli_logs_wheelbase_error(tmp_path, monkeypatch, tape, code):
    pts, labels, names = _cloud()
    npz = tmp_path / "cleaned.npz"
    np.savez(npz, points=pts, labels=labels, names=np.array(names))
    meas = tmp_path / "m.yaml"
    meas.write_text(yaml.safe_dump({"length": 5.0, "wheelbase": tape}), encoding="utf-8")
    cli = _cli()
    logged = []
    monkeypatch.setattr(cli, "append_metric", lambda **kw: logged.append(kw))
    args = ["--labelled", str(npz), "--measured", str(meas), "--out-root", str(tmp_path / "o")]
    assert cli.main(args) == code
    (m,) = logged
    assert (m["task"], m["metric"]) == ("P1-T6", "wheelbase_error_m")
    (run,) = list((tmp_path / "o" / "eval").iterdir())
    saved = json.loads((run / "wheelbase.json").read_text(encoding="utf-8"))
    assert saved["error_m"] == pytest.approx(saved["wheelbase_m"] - tape)
    assert m["value"] == pytest.approx(abs(saved["error_m"]))


def test_cli_refuses_centimetres(tmp_path):
    pts, labels, names = _cloud()
    npz = tmp_path / "c.npz"
    np.savez(npz, points=pts, labels=labels, names=np.array(names))
    meas = tmp_path / "m.yaml"
    meas.write_text(yaml.safe_dump({"wheelbase": 300.0}), encoding="utf-8")
    assert _cli().main(["--labelled", str(npz), "--measured", str(meas), "--no-metrics"]) == 2
