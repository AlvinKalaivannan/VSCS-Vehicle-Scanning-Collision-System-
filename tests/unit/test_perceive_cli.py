"""scripts/perceive.py: refuses without calibration/mount; frames -> obstacles.jsonl."""

from __future__ import annotations

import importlib.util

import cv2
import numpy as np
import pytest

from fixtures.synthetic import Box, load_scene
from vscs.common.config import load_config, repo_root
from vscs.common.frames import invert, look_at_T_world_cam, project, transform_points
from vscs.common.io import read_jsonl, write_jsonl
from vscs.perception.pipeline import Box2D

SCENE = load_scene()
W, H = SCENE.image_size
POS, LOOK = [-1.0, 0.0, 1.2], [-3.0, 0.0, 0.0]
POLE = Box.from_bounds("pole", [-3.53, -1.03, 0.0, -3.47, -0.97, 0.6])


@pytest.fixture()
def cli(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "perceive_cli", repo_root() / "scripts" / "perceive.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _capture_cfg(calibrated=True, measured=True):
    cap = load_config("capture")
    cap["intrinsics"] = {
        **cap["intrinsics"],
        "calibrated": calibrated,
        "K": SCENE.K.tolist(),
        "dist": [],
        "image_size": [W, H],
    }
    cap["mount"] = {
        **cap["mount"],
        "measured": measured,
        "position_veh_m": POS,
        "look_at_veh_m": LOOK,
    }
    return cap


def _frames_run(tmp_path, n=5):
    (tmp_path / "frames").mkdir()
    rows = []
    for i in range(n):
        name = f"frames/{i * 10:06d}.png"
        cv2.imwrite(str(tmp_path / name), np.zeros((H, W, 3), np.uint8))
        rows.append({"frame_index": i * 10, "t_ns": i * 100_000_000, "file": name})
    write_jsonl(tmp_path / "frames.jsonl", rows)
    return tmp_path


class FakeDetector:
    """Reports the pole's true box in every frame (the van is stationary)."""

    def __init__(self):
        T_cam_veh = invert(look_at_T_world_cam(np.array(POS), np.array(LOOK)))
        uv, _ = project(SCENE.K, transform_points(T_cam_veh, POLE.corners()))
        self.box = (
            np.floor(uv[:, 0].min()),
            np.floor(uv[:, 1].min()),
            np.ceil(uv[:, 0].max()),
            np.ceil(uv[:, 1].max()),
        )

    def detect(self, image):
        assert image.shape == (H, W, 3)
        return [Box2D(self.box, "pole", 0.9)]


@pytest.mark.parametrize("calibrated,measured", [(False, True), (True, False)])
def test_refuses_without_calibration_or_mount(cli, monkeypatch, tmp_path, calibrated, measured):
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda n: _capture_cfg(calibrated, measured) if n == "capture" else load_config(n),
    )
    assert (
        cli.main(["--frames-run", str(_frames_run(tmp_path)), "--out-root", str(tmp_path / "o")])
        == 2
    )


def test_frames_become_an_obstacle_stream(cli, monkeypatch, tmp_path):
    monkeypatch.setattr(
        cli, "load_config", lambda n: _capture_cfg() if n == "capture" else load_config(n)
    )
    monkeypatch.setattr(cli, "make_detector", lambda cfg, device: FakeDetector())
    rc = cli.main(
        [
            "--frames-run",
            str(_frames_run(tmp_path)),
            "--out-root",
            str(tmp_path / "o"),
        ]
    )
    assert rc == 0
    (run,) = list((tmp_path / "o" / "perception").iterdir())
    obs = list(read_jsonl(run / "obstacles.jsonl"))
    dets = list(read_jsonl(run / "detections.jsonl"))
    assert len(dets) == 5 and len(obs) == 5  # one mapped pole per frame
    assert all(o["source"] == "occupancy" for o in obs)
    last = obs[-1]
    assert abs(last["center_veh"][0] - (-3.5)) < 0.35 and abs(last["center_veh"][1] - (-1.0)) < 0.35
