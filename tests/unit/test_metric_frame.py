"""SfM -> metric veh frame, and scripts/scale.py (P1-T6 plumbing).

A synthetic sparse reconstruction: the fixture van's body and mirrors, the ground, four
propped marker boards (ID 3 right at the front bumper, as the checklist places it), a
parked car outside the walk-round, all put into a made-up SfM frame - rotated, shifted and
4x the metric size. Marker corners are observed by projecting the true corners into the
fixture's 20-pose scan loop.

The scale maths is the developer's ``recon/scale.py`` (pair mode). The tests drive the
plumbing with a stand-in: OpenCV's two-view triangulation and a plain median - library
calls and one-liners, not the developer's DLT or robust estimate. So these tests check
the plumbing (frames, units, body selection, outputs), not the scale maths.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import types

import cv2
import numpy as np
import pytest
import yaml

from fixtures.synthetic import Box, load_scene
from vscs.common.config import load_config, repo_root
from vscs.common.frames import (
    T_from_Rt,
    axis_angle_to_R,
    invert,
    project,
    transform_points,
)
from vscs.recon.colmap_io import Camera, Image, Model, Point3D, write_model
from vscs.recon.marker_tracks import MarkerTrack, TrackReport
from vscs.recon.metric import metric_frame, select_body

SCENE = load_scene()
S = 0.25  # metres per SfM unit
T_W_VEH = T_from_Rt(axis_angle_to_R([0.3, -0.2, 1.0], 1.3), [4.0, -7.0, 2.0])
SIDE = 0.15
MEASURED = {"length": 5.0, "width": 2.0, "height": 2.2, "wheelbase": 3.0}
SEED = 20260924


def _cfg():
    cfg = copy.deepcopy(load_config("recon"))
    cfg["vehicle_frame"]["rear_overhang_m"] = 1.0  # the fixture's rear bumper to axle
    cfg["scale"]["marker"]["side_length_m"] = SIDE
    # On the developer's p1-t6-scale branch, not yet on main.
    cfg["scale"]["marker"].setdefault("max_spread_frac", 0.02)
    return cfg


def _marker(centre, normal):
    """Corners of a SIDE x SIDE square at ``centre`` facing ``normal`` (veh)."""
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    a = np.cross([0.0, 0.0, 1.0], n)
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    h = SIDE / 2
    return np.array(
        [centre + h * (-a + b), centre + h * (a + b), centre + h * (a - b), centre + h * (-a - b)]
    )


# Boards propped at 45 deg, facing out; ID 3 just in front of the bumper (x = 4.0).
OUT45 = {
    0: ([-3.0, 0.0], [-1, 0]),
    1: ([1.5, 3.0], [0, 1]),
    2: ([1.5, -3.0], [0, -1]),
    3: ([4.12, 0.0], [1, 0]),
}
MARKERS = {m: _marker(np.array([*xy, 0.35]), np.array([*d, 1.0])) for m, (xy, d) in OUT45.items()}


def _board_points(centre, rng, n=60):
    """A clipboard (~0.23 x 0.32 m) around the marker: the textured points SfM finds."""
    return centre + rng.uniform([-0.12, -0.12, -0.16], [0.12, 0.12, 0.16], (n, 3))


def _veh_cloud(rng):
    shell = SCENE.vehicle.sample_surface_points(6000, seed=SEED)
    shell = shell[shell[:, 2] > 1e-6]
    mirrors = np.vstack(
        [
            SCENE.component(m).sample_surface_points(400, seed=SEED + i)
            for i, m in enumerate(("left_mirror", "right_mirror"))
        ]
    )
    body = np.vstack([shell, mirrors])
    gx, gy = rng.uniform(-5.0, 8.0, 4000), rng.uniform(-5.0, 5.0, 4000)
    ground = np.column_stack([gx, gy, np.zeros(4000)])
    ground = ground[~((gx > -1.0) & (gx < 4.0) & (np.abs(gy) < 1.0))]
    boards = np.vstack([_board_points(c.mean(axis=0), rng) for c in MARKERS.values()])
    car = Box.from_bounds("car", [10.0, -1.0, 0.0, 14.5, 1.0, 1.5]).sample_surface_points(
        1500, seed=SEED
    )
    return body, np.vstack([body, ground, boards, car])


@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(SEED)
    body_veh, cloud_veh = _veh_cloud(rng)
    K = SCENE.K
    w, h = SCENE.image_size
    cam = Camera(1, "PINHOLE", w, h, np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]))
    images, tracks = {}, {m: MarkerTrack(m) for m in MARKERS}
    with_markers = 0
    for i, pose in enumerate(SCENE.camera_trajectory(), start=1):
        T_cam_veh = invert(pose.T_world_cam)
        T_cam_W = T_cam_veh @ invert(T_W_VEH)
        T_cam_sfm = T_from_Rt(T_cam_W[:3, :3], T_cam_W[:3, 3] / S)
        images[i] = Image.from_T_world_cam(i, invert(T_cam_sfm), 1, f"{i:06d}.png")
        seen = False
        for m, corners in MARKERS.items():
            uv, ok = project(K, transform_points(T_cam_veh, corners), image_size=(w, h))
            if ok.all():
                seen = True
                for k in range(4):
                    tracks[m].corners[k].append((i, uv[k]))
        with_markers += seen
    pts_sfm = transform_points(T_W_VEH, cloud_veh) / S
    points = {
        j + 1: Point3D(
            j + 1, p, np.zeros(3, np.int64), 0.5, np.zeros(0, np.int64), np.zeros(0, np.int64)
        )
        for j, p in enumerate(pts_sfm)
    }
    model = Model({1: cam}, images, points)
    report = TrackReport(tracks, len(images), with_markers)
    return model, report, body_veh


def _two_view(projections, uvs):
    X = cv2.triangulatePoints(
        np.asarray(projections[0]),
        np.asarray(projections[-1]),
        np.asarray(uvs[0], float).reshape(2, 1),
        np.asarray(uvs[-1], float).reshape(2, 1),
    )
    return (X[:3] / X[3]).ravel()


def _stand_in():
    def side(c):
        c = np.asarray(c)
        return float(np.mean([np.linalg.norm(c[k] - c[(k + 1) % 4]) for k in range(4)]))

    def estimate(sides, cfg):
        per = {m: cfg["scale"]["marker"]["side_length_m"] / v for m, v in sides.items()}
        vals = np.array(list(per.values()))
        med = float(np.median(vals))
        spread = float((vals.max() - vals.min()) / med)
        return types.SimpleNamespace(
            scale=med,
            per_marker=per,
            spread_frac=spread,
            consistent=spread <= cfg["scale"]["marker"]["max_spread_frac"],
        )

    def errors(rec, meas):
        return {k: rec[k] - meas[k] for k in rec if k in meas}

    return types.SimpleNamespace(
        projection_matrix=lambda K, T: np.asarray(K) @ np.asarray(T)[:3],
        triangulate_point=_two_view,
        marker_side_length=side,
        estimate_scale_from_config=estimate,
        scale_points=lambda p, s: s * np.asarray(p, float),
        dimension_errors=errors,
        scale_error_m=lambda rec, meas: max(abs(v) for v in errors(rec, meas).values()),
    )


def test_every_marker_is_seen_from_at_least_two_views(synth):
    _, report, _ = synth
    assert all(t.n_views >= 2 for t in report.tracks.values())


def test_metric_frame_recovers_scale_frame_and_dimensions(synth):
    model, report, body_veh = synth
    cfg = _cfg()
    res, body = metric_frame(model, report, cfg, MEASURED, _stand_in())
    assert res.scale == pytest.approx(S, rel=1e-6)
    assert res.markers_consistent and sorted(res.per_marker_scale) == [0, 1, 2, 3]
    # The frame maps the scaled SfM cloud back onto the true veh coordinates.
    T = np.asarray(res.T_veh_world)
    true = transform_points(T @ T_W_VEH, body_veh)  # veh -> world (metric) -> estimated veh
    assert np.abs(true - body_veh).max() < 0.03
    for k in ("length", "width", "height"):
        assert res.errors_m[k] == pytest.approx(0.0, abs=0.02), k
    assert res.not_validated == ["wheelbase"]  # needs wheel centres; said so, not dropped
    assert res.passed
    # The van and only the van: boards, ground and the parked car are all out.
    # (A few van points are lost on purpose: the lowest 5 cm of the panels, and bumper
    # points inside ID 3's exclusion sphere.)
    assert 0.97 * len(body_veh) <= res.n_body_points <= len(body_veh)
    lo, hi = np.array([-1.0, -1.25, 0.0]) - 0.03, np.array([4.0, 1.25, 2.2]) + 0.03
    assert ((body >= lo) & (body <= hi)).all()  # nothing kept lies outside the van
    assert body[:, 0].max() < 4.0 + 0.01  # ID 3's board did not lengthen the van


def test_without_the_marker_exclusion_the_front_board_joins_the_van(synth):
    """Why marker_exclusion_m exists: ID 3's board touches the bumper's cluster."""
    model, report, _ = synth
    cfg = _cfg()
    cfg["vehicle_frame"]["body_select"]["marker_exclusion_m"] = 0.0
    res, _ = metric_frame(model, report, cfg, MEASURED, _stand_in())
    assert res.dimensions_m["length"] > 5.1


def test_missing_front_marker_is_a_clear_failure(synth):
    model, report, _ = synth
    tracks = {m: t for m, t in report.tracks.items() if m != 3}
    with pytest.raises(ValueError, match="front marker 3"):
        metric_frame(
            model,
            TrackReport(tracks, report.n_images, report.n_images_with_markers),
            _cfg(),
            MEASURED,
            _stand_in(),
        )


def test_select_body_keeps_big_clusters_inside_the_loop():
    rng = np.random.default_rng(1)
    van = rng.uniform([0, 0, 0.5], [2, 1, 1.5], (500, 3))
    speck = rng.uniform([3, 3, 0.5], [3.1, 3.1, 0.6], (20, 3))
    outside = rng.uniform([20, 0, 0.5], [22, 1, 1.5], (500, 3))
    pts = np.vstack([van, speck, outside])
    loop = np.array([[-5, -5], [10, -5], [10, 10], [-5, 10]], float)
    cfg = {
        "min_height_m": 0.05,
        "max_height_m": 4.0,
        "cluster_eps_m": 0.3,
        "min_cluster_points": 200,
        "marker_exclusion_m": 0.25,
    }
    mask, sizes = select_body(pts, pts[:, 2], pts[:, :2], loop, np.zeros((0, 3)), cfg)
    assert mask[:500].all() and not mask[500:].any() and sizes == [500]


# --------------------------------------------------------------------------- #
# scripts/scale.py                                                             #
# --------------------------------------------------------------------------- #
def _cli():
    spec = importlib.util.spec_from_file_location("scale_cli", repo_root() / "scripts" / "scale.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def recon_run(synth, tmp_path):
    model, _, _ = synth
    run = tmp_path / "recon_run"
    write_model(model, run / "sparse_txt" / "0")
    (run / "sfm_result.json").write_text(
        json.dumps({"model_dir": str(run / "sparse_txt" / "0")}), encoding="utf-8"
    )
    (tmp_path / "frames").mkdir()
    meas = tmp_path / "measurements.yaml"
    meas.write_text(yaml.safe_dump(MEASURED), encoding="utf-8")
    return run, tmp_path / "frames", meas


def _args(recon_run, out):
    run, frames, meas = recon_run
    return [
        "--recon-run",
        str(run),
        "--images",
        str(frames),
        "--measured",
        str(meas),
        "--out-root",
        str(out),
        "--no-metrics",
    ]


def _patch_cfg(cli, monkeypatch):
    cfg = _cfg()
    monkeypatch.setattr(cli, "load_config", lambda n: cfg if n == "recon" else load_config(n))


def test_cli_stops_cleanly_without_the_scale_maths(recon_run, tmp_path, monkeypatch):
    def stub(*a, **k):
        raise NotImplementedError("P1-T6: developer's first draft")

    cli = _cli()
    _patch_cfg(cli, monkeypatch)
    monkeypatch.setattr(cli, "SCALE", types.SimpleNamespace(scale_points=stub))
    assert cli.main(_args(recon_run, tmp_path / "out")) == 3
    assert not (tmp_path / "out").exists()  # no empty run folder


def test_cli_refuses_without_the_rear_overhang(recon_run, tmp_path, monkeypatch):
    cli = _cli()
    cfg = _cfg()
    cfg["vehicle_frame"]["rear_overhang_m"] = None
    monkeypatch.setattr(cli, "load_config", lambda n: cfg)
    assert cli.main(_args(recon_run, tmp_path / "out")) == 2


def test_cli_refuses_centimetres(recon_run, tmp_path, monkeypatch):
    cli = _cli()
    _patch_cfg(cli, monkeypatch)
    _, _, meas = recon_run
    meas.write_text(yaml.safe_dump({"length": 500.0}), encoding="utf-8")
    assert cli.main(_args(recon_run, tmp_path / "out")) == 2


def test_cli_writes_the_handover_for_fuse(synth, recon_run, tmp_path, monkeypatch):
    _, report, body_veh = synth
    cli = _cli()
    _patch_cfg(cli, monkeypatch)
    monkeypatch.setattr(cli, "SCALE", _stand_in())
    # No real images: hand the CLI the synthetic marker observations.
    monkeypatch.setattr(cli, "collect_marker_tracks_from_config", lambda *a: report)
    assert cli.main(_args(recon_run, tmp_path / "out")) == 0
    (run,) = list((tmp_path / "out" / "recon").iterdir())

    from vscs.seg.fuse_inputs import read_sfm_to_veh

    sim = read_sfm_to_veh(run / "sfm_to_veh.json")  # exactly what scripts/fuse.py reads
    assert sim.scale == pytest.approx(S, rel=1e-6)
    p_sfm = transform_points(T_W_VEH, body_veh) / S
    assert np.abs(sim.points(p_sfm) - body_veh).max() < 0.03
    result = json.loads((run / "metric_result.json").read_text(encoding="utf-8"))
    assert result["passed"] and result["not_validated"] == ["wheelbase"]
    assert (run / "body_points.ply").is_file()
