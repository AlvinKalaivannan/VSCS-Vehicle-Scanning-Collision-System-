"""Whole-drive risk: ego state from poses, and the engine over a scripted drive."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box

from fixtures.oracle_sweep import ref_sweep_components
from fixtures.synthetic import load_scene
from vscs.common.config import load_config
from vscs.common.frames import T_from_Rt, rot_z
from vscs.common.types import Obstacle
from vscs.risk.drive import assess_drive, component_shapes, ego_states

SCENE = load_scene()
RISK, SEV = load_config("risk"), load_config("severity")
S = 1_000_000_000


def test_ego_state_is_signed_and_has_yaw_rate():
    poses = [
        (round(k * 0.1 * S), T_from_Rt(rot_z(0.02 * k), [-0.1 * k, 0.0, 0.0])) for k in range(5)
    ]
    # Reversing 0.1 m per 0.1 s while yawing 0.02 rad per 0.1 s. The world translation is
    # along world x, not the turned vehicle's x, so speed reads cos(yaw) x 1.0 m/s: ~-1.0.
    st = ego_states(poses)
    assert len(st) == 5 and st[0].t_ns == 0
    assert all(s.speed_mps < 0 for s in st)
    assert st[2].yaw_rate_radps == pytest.approx(0.2, abs=1e-9)
    assert st[2].speed_mps == pytest.approx(-1.0, abs=0.02)


def test_single_pose_is_stationary():
    (s,) = ego_states([(0, np.eye(4))])
    assert s.speed_mps == 0.0 and s.yaw_rate_radps == 0.0


def test_a_drive_toward_the_pole_names_the_rear_right_corner():
    fps = {
        n: (box(b.lo[0], b.lo[1], b.hi[0], b.hi[1]), float(b.lo[2]), float(b.hi[2]))
        for n, b in SCENE.components.items()
    }
    poses = [(round(k * 0.2 * S), T_from_Rt(np.eye(3), [-0.2 * k, 0.0, 0.0])) for k in range(4)]
    obstacles = {}
    for t, T in poses:  # the pole, static in world at (-1.5, -1.0), seen from each pose
        x = -1.5 - T[0, 3]
        obstacles[t] = [
            Obstacle(
                id=1,
                t_ns=t,
                kind="static_geom",
                center_veh=(x, -1.0, 0.6),
                extent=(0.06, 0.06, 1.2),
                velocity_veh=(0, 0, 0),
                pos_sigma_m=0.02,
                source="occupancy",
            )
        ]
    frames = assess_drive(
        component_shapes(fps),
        obstacles,
        ego_states(poses),
        RISK,
        SEV,
        sweep_fn=ref_sweep_components,
    )
    assert len(frames) == 4
    assert frames[-1].worst_component == "rear_right_bumper_corner"
    assert frames[-1].alert_level == "critical"  # contact well under 1 s away by then


def _model_and_perception(tmp_path):
    import trimesh

    from vscs.common.io import write_jsonl
    from vscs.model.urdf import export_model

    parts = {}
    for n, b in SCENE.components.items():
        m = trimesh.creation.box(extents=b.extent)
        m.apply_translation(b.center)
        parts[n] = [m]
    export_model(
        tmp_path / "model",
        parts,
        {},
        severity_cfg=SEV,
        urdf_cfg=load_config("model")["urdf"],
        scale_error_m=0.01,
    )
    perc = tmp_path / "perc"
    perc.mkdir()
    poses = [(round(k * 0.2 * S), T_from_Rt(np.eye(3), [-0.2 * k, 0.0, 0.0])) for k in range(4)]
    write_jsonl(
        perc / "ego.jsonl", [{"t_ns": t, "T_world_veh": T.ravel().tolist()} for t, T in poses]
    )
    write_jsonl(
        perc / "obstacles.jsonl",
        [
            Obstacle(
                id=1,
                t_ns=t,
                kind="static_geom",
                center_veh=(-1.5 - T[0, 3], -1.0, 0.6),
                extent=(0.06, 0.06, 1.2),
                velocity_veh=(0, 0, 0),
                pos_sigma_m=0.02,
                source="occupancy",
            )
            for t, T in poses
        ],
    )
    return tmp_path / "model", perc


def _cli():
    import importlib.util

    from vscs.common.config import repo_root

    spec = importlib.util.spec_from_file_location("risk_cli", repo_root() / "scripts" / "risk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_runs_vscs_and_baseline_on_the_same_drive(tmp_path, monkeypatch):
    from vscs.common.io import read_jsonl

    cli = _cli()
    monkeypatch.setattr(cli, "SWEEP_FN", ref_sweep_components)
    model, perc = _model_and_perception(tmp_path)
    assert (
        cli.main(
            ["--model", str(model), "--perception", str(perc), "--out-root", str(tmp_path / "a")]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "--model",
                str(model),
                "--perception",
                str(perc),
                "--baseline",
                "--out-root",
                str(tmp_path / "b"),
            ]
        )
        == 0
    )
    (ra,) = list((tmp_path / "a" / "risk").iterdir())
    (rb,) = list((tmp_path / "b" / "risk").iterdir())
    vscs = list(read_jsonl(ra / "risk_frames.jsonl"))
    base = list(read_jsonl(rb / "risk_frames.jsonl"))
    assert len(vscs) == len(base) == 4
    assert vscs[-1]["worst_component"] == "rear_right_bumper_corner"
    assert base[-1]["worst_component"] == load_config("eval")["baseline"]["name"]


def test_cli_says_the_sweep_is_not_written_yet(tmp_path):
    cli = _cli()  # SWEEP_FN None: the engine imports risk/sweep.py, which does not exist on main
    model, perc = _model_and_perception(tmp_path)
    rc = cli.main(
        ["--model", str(model), "--perception", str(perc), "--out-root", str(tmp_path / "o")]
    )
    assert rc == 3
