"""P5-T4 driver replay: pixels where they should be, beeps at the configured rates."""

from __future__ import annotations

import json
import wave

import numpy as np
import pytest
from shapely.geometry import box

from fixtures.synthetic import load_scene
from vscs.common.config import load_config
from vscs.common.types import ComponentRisk, Obstacle, RiskFrame
from vscs.ui import driver_replay as D

SCENE = load_scene()
UI = load_config("ui")
CFG = UI["driver_replay"]
AUDIO = CFG["audio"]
COLOURS = UI["view"]["alert_colours"]
RISK = load_config("risk")
S = 1_000_000_000


def _footprints():
    return {
        n: (box(b.lo[0], b.lo[1], b.hi[0], b.hi[1]), float(b.lo[2]), float(b.hi[2]))
        for n, b in SCENE.components.items()
    }


def _frame(t_s, level, corner_ttc=None):
    cr = ComponentRisk(
        component="rear_right_bumper_corner",
        min_distance_m=0.0 if corner_ttc else 2.0,
        ttc_s=corner_ttc,
        p_contact=1.0 if corner_ttc else 0.0,
        severity=1.5,
        risk=1.5 if corner_ttc else 0.0,
        obstacle_id=1,
    )
    return RiskFrame(
        t_ns=round(t_s * S),
        ego_speed_mps=-1.0,
        per_component=[cr],
        p_any_contact=1.0 if corner_ttc else 0.0,
        expected_damage=0.0,
        worst_component="rear_right_bumper_corner" if corner_ttc else None,
        alert_level=level,
    )


def _pixel(img, x_m, y_m):
    u, v = D.TopDown(CFG).px(x_m, y_m)
    return img.getpixel((round(u), round(v)))


def test_components_are_coloured_by_their_own_alert_at_the_right_pixels():
    img = D.render_frame(_footprints(), _frame(0, "critical", 0.5), CFG, COLOURS, RISK)
    corner = SCENE.component("rear_right_bumper_corner").center
    assert _pixel(img, corner[0], corner[1]) == tuple(COLOURS["critical"][:3])
    door = SCENE.component("sliding_door_right").center  # not in the frame's risks: "none"
    assert _pixel(img, door[0], door[1]) == tuple(COLOURS["none"][:3])
    # Left side at y = 0.9: beside the underbody (|y| <= 0.78), no door there: background.
    assert _pixel(img, 1.5, 0.9) == tuple(CFG["background"])
    assert img.size == tuple(CFG["image_px"])


def test_obstacles_and_path_are_drawn_where_they_are():
    pole = Obstacle(
        id=1,
        t_ns=0,
        kind="static_geom",
        center_veh=(-1.5, -1.0, 0.6),
        extent=(0.4, 0.4, 1.2),
        velocity_veh=(0, 0, 0),
        pos_sigma_m=0.05,
        source="occupancy",
    )
    img = D.render_frame(
        _footprints(),
        _frame(0, "none"),
        CFG,
        COLOURS,
        RISK,
        obstacles=[pole],
        path_xy=[[0.0, 0.0], [-3.0, 0.0]],
    )
    assert _pixel(img, -1.5 + 0.2, -1.0) == tuple(UI["view"]["obstacle_colour"][:3])  # box edge
    assert _pixel(img, -2.5, 0.0) == tuple(UI["view"]["path_colour"][:3])


def test_the_banner_is_on_every_frame():
    img = D.render_frame(_footprints(), _frame(0, "none"), CFG, COLOURS, RISK)
    top = np.asarray(img)[4:22, 4:400].astype(int)
    # Pillow's default font is anti-aliased, so look for bright text pixels, not exact white.
    assert (top.min(axis=-1) > 200).sum() > 50


@pytest.mark.parametrize("level,beeps", [("caution", 2), ("warning", 4)])
def test_beep_rates(level, beeps):
    x = D.tone(level, 1.0, AUDIO)
    on = np.abs(x) > 0
    # Count rising edges of the gate (each beep starts with a run of non-silent samples).
    sr = AUDIO["sample_rate_hz"]
    runs = np.diff(
        np.concatenate([[0], (np.convolve(on, np.ones(sr // 200), "same") > 0).astype(int)])
    )
    assert (runs == 1).sum() == beeps


def test_none_is_silent_and_critical_is_continuous():
    assert not D.tone("none", 0.5, AUDIO).any()
    crit = D.tone("critical", 0.5, AUDIO)
    sr, f = AUDIO["sample_rate_hz"], AUDIO["tone_hz"]
    # A sine has zeros twice per cycle; outside those, a continuous tone is never silent
    # for longer than a sample or two.
    silent_runs = np.diff(np.flatnonzero(np.abs(crit) > 0))
    assert silent_runs.max() <= 3 and len(crit) == sr // 2 and f > 0


def test_write_replay_outputs_frames_audio_and_index(tmp_path):
    frames = [_frame(0.0, "none"), _frame(0.5, "caution", 2.5), _frame(1.0, "critical", 0.4)]
    D.write_replay(tmp_path, _footprints(), frames, UI, RISK)
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert [e["alert_level"] for e in index] == ["none", "caution", "critical"]
    assert all((tmp_path / e["file"]).is_file() for e in index)
    with wave.open(str(tmp_path / "replay.wav")) as w:
        assert w.getframerate() == AUDIO["sample_rate_hz"] and w.getnchannels() == 1
        # 1.0 s of frames plus the last frame held for one gap (0.5 s).
        assert w.getnframes() == pytest.approx(1.5 * AUDIO["sample_rate_hz"], abs=2)


def test_cli_writes_a_replay_from_an_exported_model(tmp_path):
    import importlib.util

    import trimesh

    from vscs.common.config import repo_root
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
        severity_cfg=load_config("severity"),
        urdf_cfg=load_config("model")["urdf"],
        scale_error_m=0.01,
    )
    risk = write_jsonl(tmp_path / "risk.jsonl", [_frame(0.0, "none"), _frame(0.1, "warning", 1.5)])
    spec = importlib.util.spec_from_file_location(
        "replay_cli", repo_root() / "scripts" / "replay.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert (
        cli.main(
            [
                "--model",
                str(tmp_path / "model"),
                "--risk",
                str(risk),
                "--out-root",
                str(tmp_path / "o"),
            ]
        )
        == 0
    )
    (run,) = list((tmp_path / "o" / "eval").iterdir())
    assert (run / "replay.wav").is_file() and len(list((run / "frames").glob("*.png"))) == 2
