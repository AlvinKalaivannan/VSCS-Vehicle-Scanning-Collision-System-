"""Labelled fixture points -> decomposed, exported, reloadable collision model."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from shapely.geometry import Point

from fixtures.render import component_names
from fixtures.synthetic import load_scene
from vscs.common.config import load_config, repo_root
from vscs.model.build import build_collision_model, load_labelled_points
from vscs.model.urdf import load_component_model, load_parts, plan_footprints

SCENE = load_scene()
NAMES = component_names(SCENE)
#: The OBB engine keeps these tests fast; CoACD itself is covered in test_decompose.py.
MODEL_CFG = {**load_config("model")}
MODEL_CFG["decompose"] = {**MODEL_CFG["decompose"], "engine": "obb"}
H = float(MODEL_CFG["decompose"]["voxel_m"])
DENSITY_PER_M2 = 10000


@pytest.fixture(scope="module")
def labelled(tmp_path_factory):
    pts, labels = [], []
    for i, n in enumerate(NAMES):
        box = SCENE.components[n]
        dx, dy, dz = box.extent
        # ~0.5 cm spacing, as a dense reconstruction gives: sampled by area, not per part.
        k = int(2 * (dx * dy + dy * dz + dx * dz) * DENSITY_PER_M2)
        pts.append(box.sample_surface_points(k, seed=50 + i))
        labels.append(np.full(k, i))
    path = tmp_path_factory.mktemp("lab") / "cleaned.npz"
    np.savez(path, points=np.vstack(pts), labels=np.concatenate(labels), names=np.array(NAMES))
    return path


def test_build_exports_a_model_that_reloads_and_gives_the_pole_distance(labelled, tmp_path):
    pts, labels, names = load_labelled_points(labelled)
    res = build_collision_model(
        pts,
        labels,
        names,
        tmp_path,
        model_cfg=MODEL_CFG,
        severity_cfg=load_config("severity"),
        scale_error_m=0.013,
    )
    assert res.passed and len(res.decomposed) == len(NAMES)
    # Shipped config: no hinge measured yet, so every joint is reported, none invented.
    assert set(res.unmeasured_joints) == {
        "rear_door_left",
        "rear_door_right",
        "left_mirror",
        "right_mirror",
    }
    model = load_component_model(tmp_path)
    fp = plan_footprints(load_parts(tmp_path, model))
    d = fp["rear_right_bumper_corner"][0].distance(Point(SCENE.pole.axis_xy))
    # The known 0.500 m, less up to one voxel of the documented outward voxel bias.
    assert 0.5 - H <= d <= 0.5


def test_sparse_points_are_warned_about(caplog):
    """The pinhole failure found while writing these tests: a sparse closed shell."""
    from vscs.model.decompose import decompose_component

    box = SCENE.component("wheel_front_left")
    sparse = box.sample_surface_points(1500, seed=9)  # ~1.5 cm apart: > 0.3 voxel
    with caplog.at_level("WARNING"):
        res = decompose_component("wheel_front_left", sparse, MODEL_CFG["decompose"])
    assert "coarse" in caplog.text
    assert res.point_spacing_m > MODEL_CFG["decompose"]["max_spacing_fraction_of_voxel"] * H


def test_labels_without_a_name_are_rejected(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, points=np.zeros((3, 3)), labels=np.array([0, 5, -1]), names=np.array(["a"]))
    with pytest.raises(ValueError, match=r"labels \[5\] have no name"):
        load_labelled_points(p)


def test_cli_writes_a_run_folder(labelled, tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "export_cli", repo_root() / "scripts" / "export_model.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "load_config", lambda n: MODEL_CFG if n == "model" else load_config(n))
    rc = cli.main(
        [
            "--labelled",
            str(labelled),
            "--scale-error-m",
            "0.013",
            "--out-root",
            str(tmp_path),
            "--no-metrics",
        ]
    )
    assert rc == 0
    (run,) = list((tmp_path / "model").iterdir())
    assert (run / "components.yaml").is_file() and (run / "run.log").is_file()
    assert (run / "vscs_van.urdf").is_file() and any((run / "meshes").iterdir())
