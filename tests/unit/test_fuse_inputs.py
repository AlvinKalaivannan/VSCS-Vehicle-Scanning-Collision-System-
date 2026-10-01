"""Fusion inputs from a dense workspace, and the scripts/fuse.py plumbing (P2-T3 plumbing).

The workspace is synthetic: the fixture scene rendered exactly (``fixtures.render``), then
expressed in a made-up SfM frame - rotated, shifted and 4x the metric size - the way a real
reconstruction is. Reading it back through :mod:`vscs.seg.fuse_inputs` must recover the
true ``veh`` poses, depths and masks. The fusion itself is the developer's (P2-T3), so the
CLI is exercised with a stand-in that votes from the known truth: it checks the plumbing
around the fusion, not the fusion.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types

import numpy as np
import pytest
import trimesh
import yaml

from fixtures.render import component_names, render_views
from fixtures.synthetic import load_scene
from vscs.common.config import load_config, repo_root
from vscs.common.frames import T_from_Rt, invert, rot_x, rot_z, transform_points
from vscs.recon.colmap_io import Camera, Image, Model, write_model
from vscs.seg.cleanup import cleanup_labels
from vscs.seg.fuse_inputs import (
    SfmToVeh,
    depth_map_path,
    iter_views,
    read_colmap_array,
    read_sfm_to_veh,
    write_colmap_array,
)
from vscs.seg.labels import NONE_LABEL
from vscs.seg.masks2d import write_label_png

SCENE = load_scene()
NAMES = component_names(SCENE)
S = 0.25  # metres per SfM unit: the SfM frame is 4x the metric size
T_VEH_WORLD = T_from_Rt(rot_z(0.7) @ rot_x(0.3), [1.0, -2.0, 0.5])


# --------------------------------------------------------------------------- #
# COLMAP arrays                                                                #
# --------------------------------------------------------------------------- #
def test_array_payload_is_the_image_in_row_order(tmp_path):
    """COLMAP stores (W, H, C) column-major, which is the image row after row."""
    img = np.arange(6, dtype=np.float32).reshape(2, 3)  # H=2, W=3
    p = write_colmap_array(tmp_path / "d.bin", img)
    data = p.read_bytes()
    assert data.startswith(b"3&2&1&")
    assert data[len(b"3&2&1&") :] == np.arange(6, dtype="<f4").tobytes()
    np.testing.assert_array_equal(read_colmap_array(p), img)


def test_array_round_trip_multichannel(tmp_path):
    a = np.random.default_rng(0).random((5, 7, 3)).astype(np.float32)
    np.testing.assert_array_equal(read_colmap_array(write_colmap_array(tmp_path / "n.bin", a)), a)


def test_truncated_array_is_refused(tmp_path):
    p = tmp_path / "bad.bin"
    p.write_bytes(b"3&2&1&" + np.zeros(5, "<f4").tobytes())
    with pytest.raises(ValueError, match="holds 5 values"):
        read_colmap_array(p)


# --------------------------------------------------------------------------- #
# SfM -> veh                                                                   #
# --------------------------------------------------------------------------- #
def _to_sfm_points(p_veh):
    return transform_points(invert(T_VEH_WORLD), p_veh) / S


def _to_sfm_pose(T_cam_veh):
    T = T_cam_veh @ T_VEH_WORLD  # camera <- metric reconstruction frame
    return T_from_Rt(T[:3, :3], T[:3, 3] / S)


def test_points_and_poses_map_back_to_veh():
    sim = SfmToVeh(S, T_VEH_WORLD)
    p_veh = np.random.default_rng(1).normal(size=(50, 3))
    np.testing.assert_allclose(sim.points(_to_sfm_points(p_veh)), p_veh, atol=1e-12)
    T_cam_veh = invert(T_from_Rt(rot_z(-1.1), [3.0, 1.0, 1.2]))
    np.testing.assert_allclose(sim.T_cam_veh(_to_sfm_pose(T_cam_veh)), T_cam_veh, atol=1e-12)
    # A camera-frame depth is metric after scaling: z of a point seen from the camera.
    z_sfm = transform_points(_to_sfm_pose(T_cam_veh), _to_sfm_points(p_veh))[:, 2]
    z_veh = transform_points(T_cam_veh, p_veh)[:, 2]
    np.testing.assert_allclose(sim.depth(z_sfm), z_veh, atol=1e-12)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan")])
def test_bad_scale_is_refused(scale):
    with pytest.raises(ValueError, match="positive"):
        SfmToVeh(scale, np.eye(4))


def test_sfm_to_veh_file(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"scale": S, "T_veh_world": T_VEH_WORLD.tolist()}), encoding="utf-8")
    assert read_sfm_to_veh(p).scale == S
    p.write_text(json.dumps({"scale": S}), encoding="utf-8")
    with pytest.raises(ValueError, match="T_veh_world"):
        read_sfm_to_veh(p)


# --------------------------------------------------------------------------- #
# A synthetic dense workspace                                                  #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def truth():
    return render_views(SCENE, factor=0.25)


@pytest.fixture(scope="module")
def runs(tmp_path_factory, truth):
    """``(dense_run, seg_run, sfm_to_veh.json, points_veh, point_labels)``."""
    root = tmp_path_factory.mktemp("fuse")
    dense, seg = root / "dense_run", root / "seg_run"
    ws = dense / "dense"
    K = truth[0][1]
    h, w = truth[0][2].shape
    cam = Camera(1, "PINHOLE", w, h, np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]))
    images = {}
    for i, (T_cam_veh, _, labels, depth) in enumerate(truth, start=1):
        name = f"{i:06d}.png"
        images[i] = Image.from_T_world_cam(i, invert(_to_sfm_pose(T_cam_veh)), 1, name)
        write_colmap_array(
            depth_map_path(ws, name, "geometric"), np.where(np.isfinite(depth), depth / S, 0.0)
        )
        write_label_png(seg / "labels" / name, labels)
    write_model(Model({1: cam}, images, {}), ws / "sparse_txt")

    pts, lab_names = SCENE.labelled_point_cloud(40)
    trimesh.PointCloud(_to_sfm_points(pts)).export(ws / "fused.ply")
    (dense / "resolved_config.yaml").write_text(
        yaml.safe_dump({"dense": {"geom_consistency": True}}), encoding="utf-8"
    )
    (seg / "resolved_config.yaml").write_text(
        yaml.safe_dump({"seg": {"components": {n: n for n in NAMES}}}, sort_keys=False),
        encoding="utf-8",
    )
    sim = root / "sfm_to_veh.json"
    sim.write_text(json.dumps({"scale": S, "T_veh_world": T_VEH_WORLD.tolist()}), encoding="utf-8")
    labels = np.array([NAMES.index(n) for n in lab_names])
    return dense, seg, sim, pts, labels


def _model(dense):
    from vscs.recon.colmap_io import read_model

    return read_model(dense / "dense" / "sparse_txt")


def test_views_come_back_in_veh_and_metres(runs, truth):
    dense, seg, sim, _, _ = runs
    views = list(
        iter_views(
            _model(dense), dense / "dense", seg / "labels", read_sfm_to_veh(sim), "geometric"
        )
    )
    assert len(views) == len(truth)
    for v, (T_cam_veh, K, labels, depth) in zip(views, truth, strict=True):
        np.testing.assert_allclose(v.T_cam_world, T_cam_veh, atol=1e-9)
        np.testing.assert_allclose(v.K, K, atol=1e-9)
        np.testing.assert_array_equal(v.labels, labels)
        finite = np.isfinite(depth)
        assert np.array_equal(np.isfinite(v.depth), finite)  # COLMAP's 0 -> inf
        np.testing.assert_allclose(v.depth[finite], depth[finite], rtol=1e-6)  # float32 file


def test_views_without_a_mask_are_listed_not_fused(runs, tmp_path):
    dense, seg, sim, _, _ = runs
    labels = tmp_path / "labels"
    labels.mkdir()
    for p in sorted((seg / "labels").glob("*.png"))[:-3]:
        (labels / p.name).write_bytes(p.read_bytes())
    missing: list[str] = []
    views = list(
        iter_views(
            _model(dense), dense / "dense", labels, read_sfm_to_veh(sim), "geometric", missing
        )
    )
    assert len(views) == 17 and missing == ["000018.png", "000019.png", "000020.png"]


def test_masks_from_the_original_frames_are_refused(runs, tmp_path):
    """Original frames are larger than the undistorted images the depth maps belong to."""
    dense, seg, sim, _, _ = runs
    labels = tmp_path / "labels"
    big = np.full((300, 400), NONE_LABEL)
    for p in (seg / "labels").glob("*.png"):
        write_label_png(labels / p.name, big)
    with pytest.raises(ValueError, match="undistorted"):
        next(iter_views(_model(dense), dense / "dense", labels, read_sfm_to_veh(sim), "geometric"))


def test_a_mask_label_beyond_the_named_components_is_refused(runs):
    dense, seg, sim, _, _ = runs
    with pytest.raises(ValueError, match="names only"):
        list(
            iter_views(
                _model(dense),
                dense / "dense",
                seg / "labels",
                read_sfm_to_veh(sim),
                "geometric",
                n_classes=2,
            )
        )


def test_the_sfm_model_with_distortion_is_refused(runs, tmp_path):
    from vscs.seg.fuse_inputs import load_view

    dense, seg, sim, _, _ = runs
    model = _model(dense)
    cam = model.cameras[1]
    distorted = Camera(1, "OPENCV", cam.width, cam.height, np.r_[cam.params, 0.1, 0, 0, 0])
    img = model.images[1]
    with pytest.raises(ValueError, match="not a pinhole"):
        load_view(
            img,
            distorted,
            seg / "labels" / img.name,
            depth_map_path(dense / "dense", img.name, "geometric"),
            read_sfm_to_veh(sim),
        )


# --------------------------------------------------------------------------- #
# scripts/fuse.py                                                              #
# --------------------------------------------------------------------------- #
def _cli():
    spec = importlib.util.spec_from_file_location("fuse_cli", repo_root() / "scripts" / "fuse.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _args(runs, out):
    dense, seg, sim, _, _ = runs
    return [
        "--dense-run",
        str(dense),
        "--seg-run",
        str(seg),
        "--sfm-to-veh",
        str(sim),
        "--out-root",
        str(out),
    ]


def test_cli_stops_cleanly_while_the_fusion_is_a_stub(runs, tmp_path, monkeypatch):
    def stub(*a, **k):
        raise NotImplementedError("P2-T3: developer's first draft")

    fake = types.SimpleNamespace(decide=stub, accumulate_votes=stub, PosedView=tuple)
    monkeypatch.setitem(sys.modules, "vscs.seg.fusion3d", fake)
    assert _cli().main(_args(runs, tmp_path)) == 3
    assert not (tmp_path / "seg").exists()  # no empty run folder left behind


def test_cli_refuses_a_dense_run_without_the_text_model(runs, tmp_path):
    dense, *_ = runs
    bare = tmp_path / "dense_run"
    (bare / "dense").mkdir(parents=True)
    (bare / "dense" / "fused.ply").write_bytes((dense / "dense" / "fused.ply").read_bytes())
    args = _args(runs, tmp_path)
    args[1] = str(bare)
    assert _cli().main(args) == 2


class _TruthFusion:
    """Stands in for fusion3d: one vote per view for each point's true label.

    Not a fusion algorithm - it ignores the images. It records what the CLI hands over,
    so the test can check the batching and the frames.
    """

    def __init__(self, labels):
        self.labels = labels
        self.batches: list[int] = []
        self.points = None

    @staticmethod
    def PosedView(K, T_cam_world, labels, depth):
        return (K, T_cam_world, labels, depth)

    def accumulate_votes(self, points, views, n_classes, tol):
        self.batches.append(len(views))
        self.points = np.asarray(points)
        votes = np.zeros((len(points), n_classes + 1), np.int64)
        votes[np.arange(len(points)), self.labels] = len(views)
        return votes

    @staticmethod
    def decide(votes, *, min_observations, min_vote_fraction):
        votes = np.asarray(votes)
        n_obs = votes.sum(axis=1)
        win = votes.argmax(axis=1)
        conf = np.where(n_obs > 0, votes.max(axis=1) / np.maximum(n_obs, 1), 0.0)
        ok = (n_obs >= min_observations) & (win < votes.shape[1] - 1)
        return np.where(ok, win, NONE_LABEL), conf, n_obs


def test_cli_end_to_end_with_a_stand_in_fusion(runs, tmp_path, monkeypatch):
    _, _, _, pts, labels = runs
    cli = _cli()
    fusion = _TruthFusion(labels)
    monkeypatch.setattr(cli, "FUSION", fusion)
    assert cli.main(_args(runs, tmp_path)) == 0

    per = int(load_config("seg")["fusion3d"]["views_per_batch"])
    assert sum(fusion.batches) == 20 and max(fusion.batches) == per  # every view, batched
    # In veh, metres. PLY holds float32 (as COLMAP's fused.ply does), hence the tolerance.
    np.testing.assert_allclose(fusion.points, pts, atol=1e-5)

    (run,) = list((tmp_path / "seg").iterdir())
    with np.load(run / "fused.npz") as z:
        assert (z["n_obs"] == 20).all()  # the batches' votes were summed
        np.testing.assert_array_equal(z["labels"], labels)
    from vscs.model.build import load_labelled_points

    p, lab, names = load_labelled_points(run / "cleaned.npz")  # export_model.py's reader
    assert names == NAMES
    np.testing.assert_allclose(p, pts, atol=1e-5)
    # The configured cleanup was applied. (This 40-points-per-part cloud is far sparser
    # than a dense scan, so cleanup's 5 cm clusters legitimately clear much of it.)
    expected = cleanup_labels(p, labels, load_config("seg")["cleanup"]).labels
    np.testing.assert_array_equal(lab, expected)
    summary = json.loads((run / "fuse_summary.json").read_text(encoding="utf-8"))
    assert summary["n_views"] == 20 and summary["unlabelled_fraction_fused"] == 0.0
