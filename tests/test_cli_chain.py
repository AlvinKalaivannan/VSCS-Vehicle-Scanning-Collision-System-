"""The whole per-drive command-line chain on a synthetic drive (integration test).

    extracted frames -> perceive.py -> risk.py (VSCS and --baseline) -> evaluate.py
                                    -> view.py / replay.py

Each script is unit-tested on its own; this checks that their *files* chain: the frames
index, obstacles.jsonl + ego.jsonl, risk_frames.jsonl, the model export and the passes
file. The drive: the fixture van reverses at 1 m/s over a textured road toward a pole
2.5 m behind its rear-right corner. Ego-motion comes from real ground visual odometry on
the rendered road; detections from a stub that projects the pole's true box (the real
detector needs torch). The risk sweep is the reference implementation (the developer's
``sweep.py`` does not exist yet).
"""

from __future__ import annotations

import importlib.util
import json

import cv2
import numpy as np
import pytest
import trimesh
import yaml

from fixtures.oracle_sweep import ref_sweep_components
from fixtures.synthetic import Box, load_scene
from vscs.common.config import load_config, replace_top_level_block, repo_root
from vscs.common.frames import T_from_Rt, invert, look_at_T_world_cam, project, transform_points
from vscs.common.io import read_jsonl, write_jsonl
from vscs.model.urdf import export_model
from vscs.perception.pipeline import Box2D

SCENE = load_scene()
K, (W, H) = SCENE.K, SCENE.image_size
POS, LOOK = [-1.0, 0.0, 1.2], [-3.0, 0.0, 0.0]
T_VEH_CAM = look_at_T_world_cam(np.array(POS), np.array(LOOK))
POLE = Box.from_bounds("pole", [-3.53, -1.03, 0.0, -3.47, -0.97, 0.6])
S, DT, SPEED, N = 1_000_000_000, 0.2, -1.0, 13
T_EVENT_NS = round(2.47 * S)

_rng = np.random.default_rng(7)
TEXTURE = np.full((1600, 1800), 90, np.uint8)
for _ in range(9000):
    cv2.circle(
        TEXTURE,
        (int(_rng.integers(0, 1800)), int(_rng.integers(0, 1600))),
        int(_rng.integers(2, 9)),
        int(_rng.integers(0, 256)),
        -1,
    )


def _T_world_veh(k):
    return T_from_Rt(np.eye(3), [SPEED * DT * k, 0.0, 0.0])


def _render(T_world_veh):
    T_cam_world = invert(T_world_veh @ T_VEH_CAM)
    R, t = T_cam_world[:3, :3], T_cam_world[:3, 3]
    A = np.array([[0.01, 0, -14.0], [0, 0.01, -8.0], [0, 0, 1]])
    gray = cv2.warpPerspective(TEXTURE, K @ np.column_stack([R[:, 0], R[:, 1], t]) @ A, (W, H))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"cli_{name}", repo_root() / "scripts" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StubDetector:
    """The pole's true box, keyed by frame (images are matched by order of calls)."""

    def __init__(self):
        self.k = 0

    def detect(self, image):
        T_cam_world = invert(_T_world_veh(self.k) @ T_VEH_CAM)
        self.k += 1
        uv, _ = project(K, transform_points(T_cam_world, POLE.corners()))
        y1 = np.ceil(uv[:, 1].max())
        if y1 > H - 1:
            return []
        return [
            Box2D(
                (
                    max(0.0, np.floor(uv[:, 0].min())),
                    max(0.0, np.floor(uv[:, 1].min())),
                    min(W - 1.0, np.ceil(uv[:, 0].max())),
                    y1,
                ),
                "pole",
                0.9,
            )
        ]


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("chain")
    # 1. An extract_frames-style run: frames/*.png + frames.jsonl with real timestamps.
    (tmp / "cap" / "frames").mkdir(parents=True)
    rows = []
    for k in range(N):
        name = f"frames/{k * 6:06d}.png"
        cv2.imwrite(str(tmp / "cap" / name), _render(_T_world_veh(k)))
        rows.append({"frame_index": k * 6, "t_ns": round(k * DT * S), "file": name})
    write_jsonl(tmp / "cap" / "frames.jsonl", rows)
    # 2. A P2-T8 model export of the fixture van.
    parts = {}
    for n, b in SCENE.components.items():
        m = trimesh.creation.box(extents=b.extent)
        m.apply_translation(b.center)
        parts[n] = [m]
    export_model(
        tmp / "model",
        parts,
        {},
        severity_cfg=load_config("severity"),
        urdf_cfg=load_config("model")["urdf"],
        scale_error_m=0.01,
    )
    return tmp


def _cap_cfg():
    cap = load_config("capture")
    cap["intrinsics"] = {
        **cap["intrinsics"],
        "calibrated": True,
        "K": K.tolist(),
        "dist": [],
        "image_size": [W, H],
    }
    cap["mount"] = {**cap["mount"], "measured": True, "position_veh_m": POS, "look_at_veh_m": LOOK}
    return cap


def test_the_whole_chain(chain, monkeypatch):
    out = chain / "out"
    # perceive: real ground VO from the road texture, stub detector.
    perceive = _load("perceive")
    monkeypatch.setattr(
        perceive, "load_config", lambda n: _cap_cfg() if n == "capture" else load_config(n)
    )
    monkeypatch.setattr(perceive, "make_detector", lambda cfg, device: StubDetector())
    assert (
        perceive.main(["--frames-run", str(chain / "cap"), "--ego", "vo", "--out-root", str(out)])
        == 0
    )
    (prun,) = list((out / "perception").iterdir())
    ego = list(read_jsonl(prun / "ego.jsonl"))
    # VO from the texture recovers the 2.4 m reverse.
    assert ego[-1]["T_world_veh"][3] == pytest.approx(SPEED * DT * (N - 1), abs=0.05)

    # risk: VSCS and the baseline on the same drive.
    risk = _load("risk")
    monkeypatch.setattr(risk, "SWEEP_FN", ref_sweep_components)
    assert (
        risk.main(
            [
                "--model",
                str(chain / "model"),
                "--perception",
                str(prun),
                "--out-root",
                str(out / "v"),
            ]
        )
        == 0
    )
    assert (
        risk.main(
            [
                "--model",
                str(chain / "model"),
                "--perception",
                str(prun),
                "--baseline",
                "--out-root",
                str(out / "b"),
            ]
        )
        == 0
    )
    (vrun,) = list((out / "v" / "risk").iterdir())
    (brun,) = list((out / "b" / "risk").iterdir())

    # evaluate on dev: VSCS names the right corner in time.
    passes = chain / "passes.yaml"
    passes.write_text(
        yaml.safe_dump(
            {
                "passes": [
                    {
                        "id": "synthetic_01",
                        "vscs": str(vrun),
                        "baseline": str(brun),
                        "truth": {
                            "component": "rear_right_bumper_corner",
                            "t_event_ns": T_EVENT_NS,
                            "event_onsets_ns": [T_EVENT_NS],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    evaluate = _load("evaluate")
    eval_copy = chain / "eval.yaml"
    eval_copy.write_text(
        (repo_root() / "configs" / "eval.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    # As ingest would have: the synthetic pass is a dev pass (R-09).
    replace_top_level_block(
        eval_copy, "splits", {**load_config("eval")["splits"], "dev": ["synthetic_01"]}
    )
    assert (
        evaluate.main(
            [
                "--passes",
                str(passes),
                "--split",
                "dev",
                "--eval-config",
                str(eval_copy),
                "--out-root",
                str(out / "e"),
            ]
        )
        == 0
    )
    (erun,) = list((out / "e" / "eval").iterdir())
    result = json.loads((erun / "evaluation.json").read_text(encoding="utf-8"))
    assert result["vscs"]["component_attribution_accuracy"] == 1.0
    assert result["vscs"]["lead_time_s_median"] > 1.0
    assert result["baseline"]["component_attribution_accuracy"] is None

    # view and replay read the same files.
    assert (
        _load("view").main(
            [
                "--model",
                str(chain / "model"),
                "--risk",
                str(vrun / "risk_frames.jsonl"),
                "--obstacles",
                str(prun / "obstacles.jsonl"),
                "--out-root",
                str(out / "w"),
            ]
        )
        == 0
    )
    assert (
        _load("replay").main(
            [
                "--model",
                str(chain / "model"),
                "--risk",
                str(vrun / "risk_frames.jsonl"),
                "--out-root",
                str(out / "r"),
            ]
        )
        == 0
    )
    (rrun,) = list((out / "r" / "eval").iterdir())
    assert len(list((rrun / "frames").glob("*.png"))) == N and (rrun / "replay.wav").is_file()
