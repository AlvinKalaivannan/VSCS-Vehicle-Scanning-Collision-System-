"""Rerun developer view: vehicle parts, obstacles, path fan and risk over time (P3-T1).

Acceptance (CLAUDE.md §6): any lot run replays with synced video. This module provides
everything except the video: the frame images are added by ``log_frame_image`` once real
footage exists, on the same timeline.

Entity layout (all in the ``veh0`` frame of the drive, metres)::

    world/ego/<component>        Mesh3D per convex part; colour = that component's alert
    world/obstacles/<id>         Boxes3D per obstacle
    world/ego/path_fan           LineStrips3D, one strip per candidate curvature
    world/camera/image           camera frames (later, with real footage)
    risk/<component>/distance_m  Scalars over time
    risk/<component>/ttc_s       Scalars over time (only while a contact is predicted)
    risk/alert                   TextLog line whenever the alert level changes

Time: every logged item sits on the ``drive`` timeline at the frame's ``t_ns``, recorded
as a duration since the start of the drive, so the view scrubs in real seconds.

The recording stream is injectable. Tests pass a fake that records the calls, and
``scripts/view.py`` writes a real ``.rrd`` that ``rerun file.rrd`` opens.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import trimesh

from vscs.common.types import Obstacle, RiskFrame
from vscs.risk.alerts import raw_level

TIMELINE = "drive"


class Recorder(Protocol):
    def log(self, entity_path: str, entity: Any, *, static: bool = False) -> None: ...
    def set_time(self, timeline: str, *, duration: float) -> None: ...


def _rgba(colour: Iterable[int]) -> list[int]:
    c = [int(v) for v in colour]
    return c if len(c) == 4 else [*c, 255]


def _rr():
    import rerun as rr

    return rr


def open_recording(path: Path, application_id: str = "vscs") -> Any:
    """A Rerun recording that streams to ``path`` (an ``.rrd`` file)."""
    rr = _rr()
    rec = rr.RecordingStream(application_id)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rec.save(str(path))
    return rec


def set_frame_time(rec: Recorder, t_ns: int, t0_ns: int) -> None:
    rec.set_time(TIMELINE, duration=(int(t_ns) - int(t0_ns)) / 1e9)


def log_vehicle(
    rec: Recorder,
    parts_veh: Mapping[str, list[trimesh.Trimesh]],
    colours: Mapping[str, Iterable[int]],
    default_colour: Iterable[int],
    *,
    static: bool = True,
) -> None:
    """Each component's convex parts as one mesh, coloured per component."""
    rr = _rr()
    for name, parts in parts_veh.items():
        merged = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
        colour = _rgba(colours.get(name, default_colour))
        rec.log(
            f"world/ego/{name}",
            rr.Mesh3D(
                vertex_positions=np.asarray(merged.vertices, dtype=np.float32),
                triangle_indices=np.asarray(merged.faces, dtype=np.uint32),
                vertex_colors=np.tile(colour, (len(merged.vertices), 1)).astype(np.uint8),
            ),
            static=static,
        )


def log_obstacles(rec: Recorder, obstacles: Iterable[Obstacle], colour: Iterable[int]) -> None:
    rr = _rr()
    for ob in obstacles:
        rec.log(
            f"world/obstacles/{ob.id}",
            rr.Boxes3D(
                centers=[list(ob.center_veh)],
                half_sizes=[[e / 2.0 for e in ob.extent]],
                colors=[_rgba(colour)],
                labels=[f"{ob.kind} #{ob.id}"],
            ),
        )


def log_path_fan(rec: Recorder, fan: Any, colour: Iterable[int]) -> None:
    """One line strip per candidate curvature, on the ground plane."""
    rr = _rr()
    strips = [
        np.column_stack([fan.x[k], fan.y[k], np.zeros(fan.n_steps)])
        for k in range(fan.n_curvatures)
    ]
    rec.log("world/ego/path_fan", rr.LineStrips3D(strips, colors=[_rgba(colour)] * len(strips)))


def component_colours(
    frame: RiskFrame, alert_colours: Mapping[str, Iterable[int]], risk_cfg: dict[str, Any]
) -> dict[str, list[int]]:
    """Colour each component by the alert level it reaches on its own.

    Uses ``risk.alerts.raw_level`` on that one component, so the colours follow exactly
    the grading the alerts use (TTC for predicted contacts, closest approach otherwise;
    ADR 0005), never a second scale.
    """
    return {
        cr.component: _rgba(alert_colours[raw_level([cr], risk_cfg)]) for cr in frame.per_component
    }


def log_risk_frame(rec: Recorder, frame: RiskFrame, previous_alert: str | None) -> str:
    """Scalars per component, plus a text line when the alert level changes. Returns the level."""
    rr = _rr()
    for cr in frame.per_component:
        rec.log(f"risk/{cr.component}/distance_m", rr.Scalars([cr.min_distance_m]))
        if cr.ttc_s is not None:
            rec.log(f"risk/{cr.component}/ttc_s", rr.Scalars([cr.ttc_s]))
    if frame.alert_level != previous_alert:
        worst = f" - {frame.worst_component}" if frame.worst_component else ""
        rec.log(
            "risk/alert",
            rr.TextLog(f"{frame.alert_level.upper()}{worst}", level=_text_level(frame.alert_level)),
        )
    return frame.alert_level


def _text_level(alert: str) -> str:
    return {"none": "INFO", "caution": "WARN", "warning": "WARN", "critical": "CRITICAL"}[alert]


def record_drive(
    rec: Recorder,
    parts_veh: Mapping[str, list[trimesh.Trimesh]],
    frames: list[RiskFrame],
    view_cfg: dict[str, Any],
    risk_cfg: dict[str, Any],
    *,
    obstacles_by_t: Mapping[int, list[Obstacle]] | None = None,
    fans_by_t: Mapping[int, Any] | None = None,
) -> int:
    """Log a whole drive. Returns the number of frames logged."""
    colours = view_cfg["alert_colours"]
    log_vehicle(rec, parts_veh, {}, view_cfg["vehicle_colour"])
    if not frames:
        return 0
    t0 = frames[0].t_ns
    previous: str | None = None
    for frame in frames:
        set_frame_time(rec, frame.t_ns, t0)
        log_vehicle(
            rec,
            parts_veh,
            component_colours(frame, colours, risk_cfg),
            view_cfg["vehicle_colour"],
            static=False,
        )
        if obstacles_by_t and frame.t_ns in obstacles_by_t:
            log_obstacles(rec, obstacles_by_t[frame.t_ns], view_cfg["obstacle_colour"])
        if fans_by_t and frame.t_ns in fans_by_t:
            log_path_fan(rec, fans_by_t[frame.t_ns], view_cfg["path_colour"])
        previous = log_risk_frame(rec, frame, previous)
    return len(frames)
