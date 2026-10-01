"""Scoring 3D labels against the hand-labelled gold subset (P2-T3/T4/T5 plumbing)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import trimesh

from vscs.common.config import repo_root
from vscs.eval.labels3d import read_gold_points, score_labels

NAMES = ["a", "b", "c"]


def test_reads_a_cloudcompare_ascii_export(tmp_path):
    p = tmp_path / "gold.txt"
    p.write_text(
        "//X,Y,Z,R,G,B,Scalar field\n1.0,2.0,3.0,255,0,0,2\n4.5,5.5,6.5,0,0,0,-1\n",
        encoding="utf-8",
    )
    pts, lab = read_gold_points(p)
    np.testing.assert_array_equal(pts, [[1, 2, 3], [4.5, 5.5, 6.5]])
    np.testing.assert_array_equal(lab, [2, -1])


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("// header only\n", "no points"),
        ("1 2 3 0\n1 2 3\n", "same columns"),
        ("1 2 3 0.5\n", "whole numbers"),
    ],
)
def test_bad_gold_files_are_refused(tmp_path, text, match):
    p = tmp_path / "g.txt"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        read_gold_points(p)


def _cloud():
    """Ten points on a line; truth a,a,a,b,b,b,c,c,none,none."""
    pts = np.column_stack([np.arange(10) * 0.05, np.zeros(10), np.zeros(10)])
    truth = np.array([0, 0, 0, 1, 1, 1, 2, 2, -1, -1])
    return pts, truth


def test_known_iou():
    pts, truth = _cloud()
    pred = truth.copy()
    pred[2] = 1  # an a point called b
    pred[8] = 2  # a none point called c
    # Gold: every point, as written to and read back from float32 PLY coordinates.
    gold = pts.astype(np.float32).astype(np.float64)
    s = score_labels(pts, pred, NAMES, gold, truth, 0.001)
    # a: 2 / 3.  b: 3 / 4 (the stolen a point).  c: 2 / 3 (the none point).
    assert s.iou == pytest.approx({"a": 2 / 3, "b": 3 / 4, "c": 2 / 3})
    assert s.mean_iou == pytest.approx((2 / 3 + 3 / 4 + 2 / 3) / 3)
    assert s.n_gold == s.n_matched == 10
    assert s.gold_per_component == {"a": 3, "b": 3, "c": 2}


def test_only_matched_gold_points_are_scored():
    pts, truth = _cloud()
    gold = pts.copy()
    gold[:3] += 0.02  # the three a points come from somewhere else
    s = score_labels(pts, truth, NAMES, gold, truth, 0.001)
    assert s.n_matched == 7 and "a" not in s.iou and s.mean_iou == 1.0


def test_a_gold_set_from_another_frame_is_refused():
    pts, truth = _cloud()
    with pytest.raises(ValueError, match=r"points_veh.ply"):
        score_labels(pts, truth, NAMES, pts + 5.0, truth, 0.001)


def test_gold_labels_must_name_components():
    pts, truth = _cloud()
    bad = truth.copy()
    bad[0] = 7
    with pytest.raises(ValueError, match="7"):
        score_labels(pts, truth, NAMES, pts, bad, 0.001)


# --------------------------------------------------------------------------- #
# scripts/score_labels.py                                                      #
# --------------------------------------------------------------------------- #
def _cli():
    spec = importlib.util.spec_from_file_location(
        "score_cli", repo_root() / "scripts" / "score_labels.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fuse_run(tmp_path, fused, cleaned, n_gold=2400):
    """A fuse run of 3000 points in 3 blocks, and a gold file on its points_veh.ply."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 3, (3000, 3))
    truth = (pts[:, 0] // 1).astype(np.int64)
    run = tmp_path / "fuse_run"
    run.mkdir()
    for name, lab in (("fused", fused(truth)), ("cleaned", cleaned(truth))):
        np.savez(run / f"{name}.npz", points=pts, labels=lab, names=np.array(NAMES))
    trimesh.PointCloud(pts).export(run / "points_veh.ply")
    # What a CloudCompare export of a labelled subset of points_veh.ply looks like.
    ply = np.asarray(trimesh.load(run / "points_veh.ply").vertices)
    rows = [
        f"{x:.6f},{y:.6f},{z:.6f},{t}"
        for (x, y, z), t in zip(ply[:n_gold], truth[:n_gold], strict=True)
    ]
    gold = tmp_path / "gold.txt"
    gold.write_text("//X,Y,Z,Scalar field\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return run, gold


def _noisy(truth, frac, seed):
    lab = truth.copy()
    flip = np.random.default_rng(seed).random(len(lab)) < frac
    lab[flip] = (lab[flip] + 1) % 3
    return lab


def test_cli_passes_and_logs_three_metrics(tmp_path, monkeypatch):
    run, gold = _fuse_run(tmp_path, fused=lambda t: _noisy(t, 0.05, 1), cleaned=lambda t: t)
    cli = _cli()
    logged = []
    monkeypatch.setattr(cli, "append_metric", lambda **kw: logged.append(kw))
    args = ["--fuse-run", str(run), "--gold", str(gold), "--out-root", str(tmp_path / "o")]
    assert cli.main(args) == 0
    got = {r["metric"]: r for r in logged}
    assert got["labeled_points_count"]["value"] == 2400
    assert got["labeled_points_count"]["task"] == "P2-T4"
    assert 0.85 < got["component_iou_mean"]["value"] < 0.95
    assert got["component_iou_mean_cleaned"]["value"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("fused_frac", "cleaned_frac", "n_gold"),
    [
        (0.30, 0.0, 2400),  # fusion below 0.70
        (0.02, 0.10, 2400),  # cleanup made it worse
        (0.02, 0.0, 1500),  # too few gold points
    ],
)
def test_cli_fails_each_gate(tmp_path, fused_frac, cleaned_frac, n_gold):
    run, gold = _fuse_run(
        tmp_path,
        fused=lambda t: _noisy(t, fused_frac, 1),
        cleaned=lambda t: _noisy(t, cleaned_frac, 2),
        n_gold=n_gold,
    )
    args = [
        "--fuse-run",
        str(run),
        "--gold",
        str(gold),
        "--out-root",
        str(tmp_path / "o"),
        "--no-metrics",
    ]
    assert _cli().main(args) == 1
