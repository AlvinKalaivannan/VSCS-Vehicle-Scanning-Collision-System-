"""P3-T1 Rerun dev view: what gets logged where, and a real .rrd that Rerun verifies."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

from fixtures.synthetic import load_scene
from vscs.common.config import load_config
from vscs.common.types import ComponentRisk, Obstacle, RiskFrame
from vscs.risk.motion import path_fan_from_config
from vscs.ui import rerun_view as V

SCENE = load_scene()
VIEW = load_config("ui")["view"]
RISK = load_config("risk")


class FakeRec:
    def __init__(self):
        self.logged: list[tuple[float | None, str, object, bool]] = []
        self.t: float | None = None

    def set_time(self, timeline, *, duration):
        assert timeline == V.TIMELINE
        self.t = duration

    def log(self, path, entity, *, static=False):
        self.logged.append((self.t, path, entity, static))

    def paths(self):
        return {p for _, p, _, _ in self.logged}


def _parts():
    out = {}
    for name in ("rear_right_bumper_corner", "right_mirror"):
        b = SCENE.components[name]
        m = trimesh.creation.box(extents=b.extent)
        m.apply_translation(b.center)
        out[name] = [m]
    return out


def _frame(t_ns, corner_ttc, alert):
    return RiskFrame(
        t_ns=t_ns,
        ego_speed_mps=1.0,
        per_component=[
            ComponentRisk(
                component="rear_right_bumper_corner",
                min_distance_m=0.0 if corner_ttc else 0.8,
                ttc_s=corner_ttc,
                p_contact=1.0 if corner_ttc else 0.0,
                severity=1.5,
                risk=1.5 if corner_ttc else 0.0,
                obstacle_id=1,
            ),
            ComponentRisk(
                component="right_mirror",
                min_distance_m=2.0,
                ttc_s=None,
                p_contact=0.0,
                severity=2.0,
                risk=0.0,
                obstacle_id=None,
            ),
        ],
        p_any_contact=1.0 if corner_ttc else 0.0,
        expected_damage=1.5 if corner_ttc else 0.0,
        worst_component="rear_right_bumper_corner" if corner_ttc else None,
        alert_level=alert,
    )


FRAMES = [
    _frame(1_000_000_000, None, "none"),
    _frame(1_100_000_000, 2.5, "caution"),
    _frame(1_200_000_000, 0.8, "critical"),
]
POLE = Obstacle(
    id=1,
    t_ns=1_000_000_000,
    kind="static_geom",
    center_veh=(-1.5, -1.0, 0.6),
    extent=(0.06, 0.06, 1.2),
    velocity_veh=(0.0, 0.0, 0.0),
    pos_sigma_m=0.05,
    source="occupancy",
)


def test_colours_follow_the_alert_grading():
    """Contact in 0.8 s is critical (TTC ladder, ADR 0005); a 2 m pass is none."""
    c = V.component_colours(FRAMES[2], VIEW["alert_colours"], RISK)
    assert c["rear_right_bumper_corner"] == VIEW["alert_colours"]["critical"]
    assert c["right_mirror"] == VIEW["alert_colours"]["none"]
    c = V.component_colours(FRAMES[0], VIEW["alert_colours"], RISK)
    assert c["rear_right_bumper_corner"] == VIEW["alert_colours"]["caution"]  # 0.8 m < 1.0 m


def test_record_drive_logs_the_expected_entities_on_the_drive_timeline():
    rec = FakeRec()
    fan = path_fan_from_config(-1.0, RISK)
    n = V.record_drive(
        rec,
        _parts(),
        FRAMES,
        VIEW,
        RISK,
        obstacles_by_t={FRAMES[0].t_ns: [POLE]},
        fans_by_t={FRAMES[0].t_ns: fan},
    )
    assert n == 3
    assert {
        "world/ego/rear_right_bumper_corner",
        "world/ego/right_mirror",
        "world/obstacles/1",
        "world/ego/path_fan",
        "risk/rear_right_bumper_corner/distance_m",
        "risk/rear_right_bumper_corner/ttc_s",
        "risk/right_mirror/distance_m",
        "risk/alert",
    } <= rec.paths()
    # Times are seconds since the first frame: 0.0, 0.1, 0.2.
    times = sorted({t for t, *_ in rec.logged if t is not None})
    assert times == pytest.approx([0.0, 0.1, 0.2])
    # The mirror never had a predicted contact, so it has no TTC series.
    assert "risk/right_mirror/ttc_s" not in rec.paths()


def test_alert_text_is_logged_only_on_changes():
    rec = FakeRec()
    V.record_drive(rec, _parts(), [FRAMES[0], FRAMES[0], FRAMES[2]], VIEW, RISK)
    alerts = [p for _, p, _, _ in rec.logged if p == "risk/alert"]
    assert len(alerts) == 2  # none, then critical


def test_static_vehicle_is_logged_once_before_the_drive():
    rec = FakeRec()
    V.record_drive(rec, _parts(), [], VIEW, RISK)
    assert {p for _, p, _, s in rec.logged if s} == {
        "world/ego/rear_right_bumper_corner",
        "world/ego/right_mirror",
    }


def _rerun_cli() -> str | None:
    exe = Path(sys.executable).parent / ("rerun.exe" if sys.platform == "win32" else "rerun")
    return str(exe) if exe.exists() else shutil.which("rerun")


@pytest.mark.skipif(_rerun_cli() is None, reason="rerun CLI not installed")
def test_a_real_recording_verifies_and_contains_the_entities(tmp_path):
    path = tmp_path / "drive.rrd"
    rec = V.open_recording(path)
    V.record_drive(rec, _parts(), FRAMES, VIEW, RISK, obstacles_by_t={FRAMES[0].t_ns: [POLE]})
    rec.flush()
    rec.disconnect()
    cli = _rerun_cli()
    assert subprocess.run([cli, "rrd", "verify", str(path)], capture_output=True).returncode == 0
    printed = subprocess.run(
        [cli, "rrd", "print", str(path)], capture_output=True, text=True
    ).stdout
    for entity in ("/world/ego/rear_right_bumper_corner", "/world/obstacles/1", "/risk/alert"):
        assert entity in printed, entity
    assert np.isfinite(path.stat().st_size) and path.stat().st_size > 0


def test_cli_writes_a_recording_from_an_exported_model(tmp_path):
    """scripts/view.py end to end: a P2-T8 export + a RiskFrame stream -> drive.rrd."""
    import importlib.util

    from vscs.common.config import repo_root
    from vscs.common.io import write_jsonl
    from vscs.model.urdf import export_model

    model_dir = tmp_path / "model"
    export_model(
        model_dir,
        _parts(),
        {},
        severity_cfg=load_config("severity"),
        urdf_cfg=load_config("model")["urdf"],
        scale_error_m=0.01,
    )
    risk = write_jsonl(tmp_path / "risk.jsonl", FRAMES)
    obs = write_jsonl(tmp_path / "obs.jsonl", [POLE])
    spec = importlib.util.spec_from_file_location("view_cli", repo_root() / "scripts" / "view.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    rc = cli.main(
        [
            "--model",
            str(model_dir),
            "--risk",
            str(risk),
            "--obstacles",
            str(obs),
            "--out-root",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    (run,) = list((tmp_path / "out" / "eval").iterdir())
    assert (run / "drive.rrd").stat().st_size > 0 and (run / "run.log").is_file()
