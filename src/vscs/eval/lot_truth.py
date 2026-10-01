"""Lot-day ground truth and the P3-T2 cone-position score (format PROPOSED, ADR 0012).

``gt.yaml`` holds the obstacle layout, measured from a chalk origin, and the van's pose at
the start and end of each pass, measured as two chalk marks plumbed from the rear wheel
hubs. The marks give the pose directly (MAT188 notation, 2D):

* ``o = (l + r) / 2``: the midpoint of the left and right hub marks is the ``veh`` origin
  (§4.1: on the ground below the rear-axle centre);
* ``y = (l - r) / |l - r|``: the ``veh`` y axis points from the right hub to the left;
* ``x = (y_2, -y_1)``: y rotated by -90°, so x is forward and (x, y) is right-handed.

``R_lot_veh = [x y]`` (columns) and a lot point maps into ``veh`` as
``p_veh = R_lot_veh^T (p_lot - o)``. ``|l - r|`` must equal the rear track measured on scan
day, which checks every pose for a misread tape.

P3-T2 is scored on the stationary start of each pass, so ego-motion plays no part: every
perceived obstacle in a short window after the start sync clap is compared with the
measured layout carried into ``veh`` by the start pose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from vscs.common.types import Obstacle

FloatArray = npt.NDArray[np.float64]
XY = tuple[float, float]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HubMarks(_Strict):
    rear_left_hub_m: XY
    rear_right_hub_m: XY


class LotObstacle(_Strict):
    id: str
    kind: Literal["pole", "cone", "box", "curb", "other"]
    position_m: XY
    size_m: tuple[float, float, float] | None = None


class DesignedNearest(_Strict):
    component: str
    obstacle: str


class LotPass(_Strict):
    id: str
    start: HubMarks
    end: HubMarks | None = None
    designed_nearest: DesignedNearest | None = None
    notes: str = ""


class LotTruth(_Strict):
    schema_version: Literal[1]
    origin: str
    rear_track_m: float
    obstacles: list[LotObstacle]
    passes: list[LotPass]

    @model_validator(mode="after")
    def _consistent(self) -> LotTruth:
        ob_ids = [o.id for o in self.obstacles]
        for what, ids in (("obstacle", ob_ids), ("pass", [p.id for p in self.passes])):
            dup = sorted({i for i in ids if ids.count(i) > 1})
            if dup:
                raise ValueError(f"duplicate {what} ids: {dup}")
        for p in self.passes:
            if p.designed_nearest and p.designed_nearest.obstacle not in ob_ids:
                raise ValueError(f"pass {p.id}: unknown obstacle {p.designed_nearest.obstacle}")
        if not 0.5 < self.rear_track_m < 3.0:
            raise ValueError(f"rear_track_m {self.rear_track_m} - is this metres?")
        return self

    def pass_(self, pass_id: str) -> LotPass:
        for p in self.passes:
            if p.id == pass_id:
                return p
        raise KeyError(f"no pass {pass_id!r} in the ground truth")


def load_lot_truth(path: Path) -> LotTruth:
    return LotTruth.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def veh_pose_in_lot(
    marks: HubMarks, rear_track_m: float, tolerance_m: float
) -> tuple[FloatArray, FloatArray]:
    """``(R_lot_veh (2, 2), o (2,))`` from the two hub marks; checked against the track."""
    left = np.asarray(marks.rear_left_hub_m, dtype=np.float64)
    right = np.asarray(marks.rear_right_hub_m, dtype=np.float64)
    span = float(np.linalg.norm(left - right))
    if abs(span - rear_track_m) > tolerance_m:
        raise ValueError(
            f"hub marks are {span:.3f} m apart but the rear track is {rear_track_m:.3f} m: "
            "a misread or swapped tape measurement"
        )
    y = (left - right) / span
    x = np.array([y[1], -y[0]])
    return np.column_stack([x, y]), (left + right) / 2.0


def lot_to_veh(points_lot: npt.ArrayLike, R_lot_veh: FloatArray, o: FloatArray) -> FloatArray:
    return (np.atleast_2d(np.asarray(points_lot, dtype=np.float64)) - o) @ R_lot_veh


def cone_errors(
    obstacles: list[Obstacle],
    truth_veh: dict[str, FloatArray],
    camera_xy: npt.ArrayLike,
    frame_times_ns: list[int],
    *,
    t_from_ns: int,
    t_to_ns: int,
    max_range_m: float,
    match_gate_m: float,
) -> dict[str, dict[str, Any]]:
    """Per measured obstacle within ``max_range_m`` of the camera: errors in the window.

    ``frame_times_ns`` are the perception run's frames (``ego.jsonl``), so a frame in which
    nothing was perceived still counts as a frame the obstacle was missed in.
    For each frame in ``[t_from_ns, t_to_ns]``, the perceived obstacle nearest to the true
    footprint centre (within ``match_gate_m``) is its detection; the error is the
    horizontal distance between centres. Returns, per obstacle: the range, the frames in
    the window, the frames it was detected in, and the median error (``None`` if never).
    """
    cam = np.asarray(camera_xy, dtype=np.float64)
    by_t: dict[int, list[Obstacle]] = {
        int(t): [] for t in frame_times_ns if t_from_ns <= t <= t_to_ns
    }
    for ob in obstacles:
        if ob.t_ns in by_t:
            by_t[ob.t_ns].append(ob)
    out: dict[str, dict[str, Any]] = {}
    for oid, p in truth_veh.items():
        rng = float(np.linalg.norm(np.asarray(p) - cam))
        if rng > max_range_m:
            continue
        errs = []
        for obs in by_t.values():
            d = [float(np.hypot(o.center_veh[0] - p[0], o.center_veh[1] - p[1])) for o in obs]
            if d and min(d) <= match_gate_m:
                errs.append(min(d))
        out[oid] = {
            "range_m": rng,
            "n_frames": len(by_t),
            "n_detected": len(errs),
            "median_error_m": float(np.median(errs)) if errs else None,
        }
    return out


def ego_drift(
    T_world_veh_first: npt.ArrayLike,
    T_world_veh_last: npt.ArrayLike,
    start: tuple[FloatArray, FloatArray],
    end: tuple[FloatArray, FloatArray],
) -> dict[str, float]:
    """Ego-motion error over one pass, from its start and end hub-mark poses (P4-T2).

    ``start`` and ``end`` are ``(R_lot_veh, o)`` from :func:`veh_pose_in_lot`. The true
    motion is the end origin seen from the start pose; the estimate is the same quantity
    from the ego-motion's first and last poses, ``T_first^-1 @ T_last``. Drift is the
    endpoint error over the true displacement. The displacement is never longer than the
    path driven, so this overstates drift on curved passes - the cautious direction.
    """
    T0 = np.asarray(T_world_veh_first, dtype=np.float64).reshape(4, 4)
    T1 = np.asarray(T_world_veh_last, dtype=np.float64).reshape(4, 4)
    rel = np.linalg.inv(T0) @ T1
    est_xy, est_yaw = rel[:2, 3], float(np.arctan2(rel[1, 0], rel[0, 0]))
    (R_s, o_s), (R_e, o_e) = start, end
    true_xy = lot_to_veh(o_e, R_s, o_s)[0]
    R_rel = R_s.T @ R_e
    true_yaw = float(np.arctan2(R_rel[1, 0], R_rel[0, 0]))
    disp = float(np.linalg.norm(true_xy))
    err = float(np.linalg.norm(est_xy - true_xy))
    return {
        "displacement_m": disp,
        "endpoint_error_m": err,
        "drift_frac": err / disp if disp > 0 else float("nan"),
        "heading_error_deg": float(np.degrees(np.angle(np.exp(1j * (est_yaw - true_yaw))))),
    }
