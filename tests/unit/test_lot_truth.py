"""Lot-day ground truth (format: ADR 0012) and the P3-T2 scorer."""

from __future__ import annotations

import copy
import importlib.util
import math

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from vscs.common.config import load_config, repo_root
from vscs.common.io import write_jsonl
from vscs.common.types import Obstacle
from vscs.eval.lot_truth import (
    HubMarks,
    cone_errors,
    load_lot_truth,
    lot_to_veh,
    veh_pose_in_lot,
)

TRACK = 1.72
S = 1_000_000_000


def _marks(o, yaw):
    """Hub marks of a van whose veh origin sits at lot ``o`` with heading ``yaw``."""
    y_axis = np.array([-math.sin(yaw), math.cos(yaw)])
    o = np.asarray(o, float)
    return HubMarks(
        rear_left_hub_m=tuple(o + TRACK / 2 * y_axis),
        rear_right_hub_m=tuple(o - TRACK / 2 * y_axis),
    )


def test_pose_from_hub_marks_maps_lot_points_into_veh():
    R, o = veh_pose_in_lot(_marks([5.0, 1.0], 0.3), TRACK, 0.03)
    np.testing.assert_allclose(o, [5.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(
        R, [[math.cos(0.3), -math.sin(0.3)], [math.sin(0.3), math.cos(0.3)]], atol=1e-12
    )
    # A point 2 m straight behind the rear axle, in the lot frame:
    behind = np.array([5.0, 1.0]) - 2.0 * np.array([math.cos(0.3), math.sin(0.3)])
    np.testing.assert_allclose(lot_to_veh(behind, R, o), [[-2.0, 0.0]], atol=1e-12)


def test_a_misread_tape_is_caught_by_the_track():
    """A misread *along* the hub-to-hub line changes the span and is caught. One *across*
    it barely changes the span (10 cm turns the heading ~3.3 deg instead), so the track
    check cannot see it: the lot-day note should read each tape twice (ADR 0012)."""
    m = _marks([5.0, 0.0], 0.0)  # hubs at (5, +-0.86)
    along = HubMarks(rear_left_hub_m=(5.0, 0.96), rear_right_hub_m=m.rear_right_hub_m)
    with pytest.raises(ValueError, match="rear track"):
        veh_pose_in_lot(along, TRACK, 0.03)
    across = HubMarks(rear_left_hub_m=(5.10, 0.86), rear_right_hub_m=m.rear_right_hub_m)
    R, _ = veh_pose_in_lot(across, TRACK, 0.03)
    assert math.degrees(math.atan2(R[1, 0], R[0, 0])) == pytest.approx(-3.33, abs=0.05)


def test_swapped_marks_turn_the_van_around():
    """The track check cannot see a left/right swap: it reverses the heading. Documented
    so the lot-day note says left = the van's own left."""
    m = _marks([0.0, 0.0], 0.0)
    swapped = HubMarks(rear_left_hub_m=m.rear_right_hub_m, rear_right_hub_m=m.rear_left_hub_m)
    R, _ = veh_pose_in_lot(swapped, TRACK, 0.03)
    np.testing.assert_allclose(R[:, 0], [-1.0, 0.0], atol=1e-12)


GT = {
    "schema_version": 1,
    "origin": "chalk X",
    "rear_track_m": TRACK,
    "obstacles": [
        {"id": "cone_1", "kind": "cone", "position_m": [3.0, 0.5]},
        {"id": "cone_far", "kind": "cone", "position_m": [-1.0, 0.0]},
        {"id": "kerb", "kind": "curb", "position_m": [4.0, -1.5]},
    ],
    "passes": [
        {
            "id": "lot_001",
            "start": {"rear_left_hub_m": [5.0, 0.86], "rear_right_hub_m": [5.0, -0.86]},
            "designed_nearest": {"component": "rear_bumper", "obstacle": "cone_1"},
        }
    ],
}


def test_the_adr_example_format_loads(tmp_path):
    p = tmp_path / "gt.yaml"
    p.write_text(yaml.safe_dump(GT), encoding="utf-8")
    t = load_lot_truth(p)
    assert t.pass_("lot_001").designed_nearest.obstacle == "cone_1"
    with pytest.raises(KeyError):
        t.pass_("lot_999")


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda g: g["obstacles"].append(dict(g["obstacles"][0])), "duplicate obstacle"),
        (lambda g: g["passes"][0]["designed_nearest"].update(obstacle="nope"), "unknown obstacle"),
        (lambda g: g.update(rear_track_m=172.0), "metres"),
        (lambda g: g.update(colour="red"), "colour"),
    ],
)
def test_bad_ground_truth_is_refused(tmp_path, change, match):
    g = copy.deepcopy(GT)
    change(g)
    p = tmp_path / "gt.yaml"
    p.write_text(yaml.safe_dump(g), encoding="utf-8")
    with pytest.raises(ValidationError, match=match):
        load_lot_truth(p)


def _ob(t, x, y, oid=1):
    return Obstacle(
        id=oid,
        t_ns=t,
        kind="static_geom",
        center_veh=(x, y, 0.2),
        extent=(0.3, 0.3, 0.4),
        velocity_veh=(0.0, 0.0, 0.0),
        pos_sigma_m=0.05,
        source="detector",
    )


def test_cone_errors_known_answer():
    truth = {"near": np.array([-2.0, 0.5]), "far": np.array([-6.0, 0.0])}
    frames = list(range(0, 10 * S, S // 10))[:10]  # ten frames, 0.1 s apart
    obs = [_ob(t, -2.0 + 0.10, 0.5) for t in frames[:8]]  # seen 8 of 10 frames, 10 cm off
    obs.append(_ob(frames[0], 0.5, 0.5, oid=2))  # clutter, beyond the match gate
    res = cone_errors(
        obs,
        truth,
        [-1.0, 0.0],
        frames,
        t_from_ns=0,
        t_to_ns=frames[-1],
        max_range_m=3.0,
        match_gate_m=1.0,
    )
    assert set(res) == {"near"}  # "far" is beyond 3 m
    assert res["near"]["n_frames"] == 10 and res["near"]["n_detected"] == 8
    assert res["near"]["median_error_m"] == pytest.approx(0.10)


# --------------------------------------------------------------------------- #
# scripts/score_perception.py                                                  #
# --------------------------------------------------------------------------- #
def _cli():
    spec = importlib.util.spec_from_file_location(
        "sp_cli", repo_root() / "scripts" / "score_perception.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _setup(tmp_path, monkeypatch, offset_m, seen_frames=30):
    """Van parked at lot (5, 0) facing +x; cone_1 is then 2 m behind and 0.5 m left."""
    ev = copy.deepcopy(load_config("eval"))
    ev["splits"]["dev"], ev["splits"]["test"] = ["lot_001"], ["lot_101"]
    cap = copy.deepcopy(load_config("capture"))
    cap["mount"].update(measured=True, position_veh_m=[-1.0, 0.0, 1.2])
    cli = _cli()
    monkeypatch.setattr(cli, "load_config", lambda n: {"eval": ev, "capture": cap}[n])
    tmp_path.mkdir(parents=True, exist_ok=True)
    gt = tmp_path / "gt.yaml"
    gt.write_text(yaml.safe_dump(GT), encoding="utf-8")
    run = tmp_path / "perc"
    run.mkdir()
    frames = [k * S // 30 for k in range(120)]  # 4 s at 30 fps
    write_jsonl(
        run / "ego.jsonl", [{"t_ns": t, "T_world_veh": np.eye(4).ravel().tolist()} for t in frames]
    )
    win = [t for t in frames if 1.5 * S <= t <= 2.5 * S]
    write_jsonl(run / "obstacles.jsonl", [_ob(t, -2.0 + offset_m, 0.5) for t in win[:seen_frames]])
    passes = tmp_path / "passes.yaml"

    def write(pid):
        passes.write_text(
            yaml.safe_dump({"passes": [{"id": pid, "perception": str(run)}]}), encoding="utf-8"
        )
        return ["--gt", str(gt), "--passes", str(passes), "--out-root", str(tmp_path / "o")]

    return cli, write


def test_cli_scores_and_logs_the_p3_t2_metric(tmp_path, monkeypatch):
    cli, write = _setup(tmp_path, monkeypatch, offset_m=0.12)
    logged = []
    monkeypatch.setattr(cli, "append_metric", lambda **kw: logged.append(kw))
    assert cli.main(write("lot_001")) == 0
    (m,) = [r for r in logged if r["metric"] == "cone_position_error_m"]
    assert (m["task"], m["metric"]) == ("P3-T2", "cone_position_error_m")
    assert m["value"] == pytest.approx(0.12)


def test_cli_fails_on_error_and_on_a_miss(tmp_path, monkeypatch):
    cli, write = _setup(tmp_path, monkeypatch, offset_m=0.40)
    assert cli.main([*write("lot_001"), "--no-metrics"]) == 1  # 40 cm > 25 cm
    cli, write = _setup(tmp_path / "b", monkeypatch, offset_m=0.05, seen_frames=5)
    assert cli.main([*write("lot_001"), "--no-metrics"]) == 1  # seen in 5 of ~31 frames


def test_cli_refuses_test_split_passes(tmp_path, monkeypatch):
    cli, write = _setup(tmp_path, monkeypatch, offset_m=0.05)
    assert cli.main([*write("lot_101"), "--no-metrics"]) == 2


def test_ego_drift_known_answer():
    from vscs.eval.lot_truth import ego_drift

    start = veh_pose_in_lot(_marks([5.0, 0.0], 0.0), TRACK, 0.03)
    # Reversed 3 m and turned 10 deg left (heading +10 deg in the lot frame).
    end = veh_pose_in_lot(_marks([2.0, 0.0], math.radians(10)), TRACK, 0.03)
    yaw = math.radians(8)  # the estimate turned 8 deg and stopped 10 cm short, 5 cm left
    T_last = np.eye(4)
    T_last[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    T_last[:2, 3] = [-2.9, 0.05]
    d = ego_drift(np.eye(4), T_last, start, end)
    assert d["displacement_m"] == pytest.approx(3.0)
    assert d["endpoint_error_m"] == pytest.approx(math.hypot(0.1, 0.05))
    assert d["drift_frac"] == pytest.approx(math.hypot(0.1, 0.05) / 3.0)
    assert d["heading_error_deg"] == pytest.approx(-2.0)


@pytest.mark.parametrize(("est_x", "code", "drift"), [(-2.9, 0, 0.1 / 3), (-2.7, 1, 0.3 / 3)])
def test_cli_scores_ego_drift_from_the_end_pose(tmp_path, monkeypatch, est_x, code, drift):
    cli, write = _setup(tmp_path, monkeypatch, offset_m=0.05)
    g = copy.deepcopy(GT)
    g["passes"][0]["end"] = {"rear_left_hub_m": [2.0, 0.86], "rear_right_hub_m": [2.0, -0.86]}
    g["passes"][0]["shape"] = "straight"
    (tmp_path / "gt.yaml").write_text(yaml.safe_dump(g), encoding="utf-8")
    ego = tmp_path / "perc" / "ego.jsonl"
    rows = [__import__("json").loads(ln) for ln in ego.read_text(encoding="utf-8").splitlines()]
    T = np.eye(4)
    T[0, 3] = est_x
    rows[-1]["T_world_veh"] = T.ravel().tolist()
    write_jsonl(ego, rows)
    logged = []
    monkeypatch.setattr(cli, "append_metric", lambda **kw: logged.append(kw))
    assert cli.main(write("lot_001")) == code
    (m,) = [r for r in logged if r["metric"] == "egomotion_drift_frac"]
    assert m["task"] == "P4-T2" and m["value"] == pytest.approx(drift)


def test_a_curved_pass_is_reported_but_not_gated(tmp_path, monkeypatch):
    """Displacement understates a curved path, so only straight passes gate P4-T2."""
    cli, write = _setup(tmp_path, monkeypatch, offset_m=0.05)
    g = copy.deepcopy(GT)
    g["passes"][0]["end"] = {"rear_left_hub_m": [2.0, 0.86], "rear_right_hub_m": [2.0, -0.86]}
    g["passes"][0]["shape"] = "curve"
    (tmp_path / "gt.yaml").write_text(yaml.safe_dump(g), encoding="utf-8")
    ego = tmp_path / "perc" / "ego.jsonl"
    rows = [__import__("json").loads(ln) for ln in ego.read_text(encoding="utf-8").splitlines()]
    T = np.eye(4)
    T[0, 3] = -2.5  # 17% drift: would fail the 5% gate if it were gated
    rows[-1]["T_world_veh"] = T.ravel().tolist()
    write_jsonl(ego, rows)
    logged = []
    monkeypatch.setattr(cli, "append_metric", lambda **kw: logged.append(kw))
    assert cli.main(write("lot_001")) == 0
    assert not [r for r in logged if r["metric"] == "egomotion_drift_frac"]
    (run,) = list((tmp_path / "o" / "eval").iterdir())
    saved = __import__("json").loads((run / "perception_scores.json").read_text(encoding="utf-8"))
    assert saved["drift"]["lot_001"]["shape"] == "curve" and saved["drift_passed"] is None


def test_range_errors_and_buckets_known_answer():
    from vscs.eval.lot_truth import depth_bucket, range_errors

    cam = [-1.0, 0.0]
    truth = {"c3": np.array([-4.0, 0.0])}  # 3.0 m behind the camera
    frames = [0, S // 10, 2 * S // 10]
    obs = [_ob(t, -4.2, 0.0) for t in frames]  # placed 0.2 m too far
    r = range_errors(obs, truth, cam, frames, t_from_ns=0, t_to_ns=frames[-1], match_gate_m=1.0)
    assert r["c3"]["range_m"] == pytest.approx(3.0)
    assert r["c3"]["median_signed_error_m"] == pytest.approx(0.2)
    buckets = [1.0, 3.0, 5.0, 10.0]
    assert depth_bucket(3.0, buckets, 0.25) == 3.0
    assert depth_bucket(3.7, buckets, 0.25) == 3.0  # within 25% of 3 m
    assert depth_bucket(4.0, buckets, 0.25) == 5.0  # within 25% of 5 m
    assert depth_bucket(1.6, buckets, 0.25) is None  # near no bucket


def test_cli_logs_depth_error_by_bucket(tmp_path, monkeypatch):
    """cone_1 is 1.118 m from the camera; perceived 12 cm closer along x."""
    cli, write = _setup(tmp_path, monkeypatch, offset_m=0.12)
    logged = []
    monkeypatch.setattr(cli, "append_metric", lambda **kw: logged.append(kw))
    assert cli.main(write("lot_001")) == 0
    (m,) = [r for r in logged if r["metric"].startswith("depth_error_m_at_")]
    true_rng = math.hypot(1.0, 0.5)
    seen_rng = math.hypot(0.88, 0.5)
    assert (m["task"], m["metric"]) == ("P4-T1", "depth_error_m_at_1m")
    assert m["value"] == pytest.approx(abs(seen_rng - true_rng))
