"""The whole per-drive chain on a scripted fixture drive (the shape of P3-T6).

stub detector -> perception (ground plane, height map, tracker) -> risk engine (reference
sweep, standing in for the developer's sweep.py) -> RiskFrames -> P5-T2 metrics.

The van reverses at 1 m/s toward a pole 2.5 m behind its rear-right corner. Ground truth:
the rear-right bumper corner touches the pole at t = 2.47 s (2.5 m less the pole's 3 cm
radius). The chain must warn, naming that component, before then.

Known limitation shown here (documented in perception/depth.py, R-07): the box-only v0
geometry fattens the thin pole (6 cm -> ~45 cm) and shifts it ~22 cm sideways, so the
rear bumper is graded critical on a near miss that is really ~0.3 m. The test pins that
down so it shows up the day masks fix it, instead of being forgotten.
"""

from __future__ import annotations

import numpy as np

from fixtures.oracle_sweep import ref_box, ref_sweep_components
from fixtures.synthetic import Box, load_scene
from vscs.common.config import load_config
from vscs.common.frames import T_from_Rt, invert, look_at_T_world_cam, project, transform_points
from vscs.eval import metrics as Mx
from vscs.perception.pipeline import Box2D, Perception
from vscs.risk.alerts import AlertStateMachine
from vscs.risk.engine import assess_frame

SCENE = load_scene()
PCFG, RISK, SEV = load_config("perception"), load_config("risk"), load_config("severity")
FLAG = load_config("eval")["risk_metrics"]["flag_level"]
K, SIZE = SCENE.K, SCENE.image_size
T_VEH_CAM = look_at_T_world_cam(np.array([-1.0, 0.0, 1.2]), np.array([-3.0, 0.0, 0.0]))
S, DT, SPEED = 1_000_000_000, 0.2, -1.0
POLE = Box.from_bounds("pole", [-3.53, -1.03, 0.0, -3.47, -0.97, 1.2])
T_EVENT_S = 2.47


def _T_world_veh(t):
    return T_from_Rt(np.eye(3), [SPEED * t, 0.0, 0.0])


def _detect(t):
    cam = transform_points(invert(_T_world_veh(t) @ T_VEH_CAM), POLE.corners())
    if (cam[:, 2] <= 0.1).any():
        return []
    uv, _ = project(K, cam)
    w, h = SIZE
    x0, y0 = max(0.0, np.floor(uv[:, 0].min())), max(0.0, np.floor(uv[:, 1].min()))
    x1, y1 = min(w - 1.0, np.ceil(uv[:, 0].max())), np.ceil(uv[:, 1].max())
    return [] if y1 > h - 1 else [Box2D((x0, y0, x1, y1), "pole", 0.9)]


def _drive():
    perception = Perception(PCFG, K, T_VEH_CAM, SIZE)
    sm = AlertStateMachine(RISK)
    components = [ref_box(n, b.lo, b.hi) for n, b in SCENE.components.items()]
    frames = []
    for k in range(int(T_EVENT_S / DT) + 1):
        t = k * DT
        obs = perception.step(round(t * S), _detect(t), _T_world_veh(t))
        boxes = [
            ref_box(
                f"ob{o.id}",
                np.subtract(o.center_veh, np.divide(o.extent, 2)),
                np.add(o.center_veh, np.divide(o.extent, 2)),
                obstacle_id=o.id,
            )
            for o in obs
        ]
        frames.append(
            assess_frame(
                t_ns=round(t * S),
                ego_speed_mps=SPEED,
                curvature=0.0,
                components=components,
                obstacles=boxes,
                obstacle_kinds={o.id: o.kind for o in obs},
                state_machine=sm,
                risk_cfg=RISK,
                severity_cfg=SEV,
                sweep_fn=ref_sweep_components,
            )
        )
    return frames


def test_the_chain_warns_about_the_right_corner_before_contact():
    frames = _drive()
    truth = Mx.PassTruth(component="rear_right_bumper_corner", t_event_ns=round(T_EVENT_S * S))
    assert Mx.attribution_correct(frames, truth, FLAG), [
        (f.t_ns / S, f.alert_level, f.worst_component) for f in frames
    ]
    lead = Mx.lead_time_s(frames, truth, FLAG)
    assert lead is not None and lead > 1.0  # warned more than a second ahead
    # Predicted TTC for the corner tracks the truth (pole placed by v0 from its near face).
    errs = Mx.ttc_errors_s(frames, truth)
    assert len(errs) > 3 and float(np.median(errs)) < 0.25


def test_box_only_v0_fattens_the_thin_pole_known_limitation():
    """R-07: the perceived pole is far wider than its true 0.06 m. When masks replace boxes
    this test should start failing - update it then, with the new, smaller number."""
    perception = Perception(PCFG, K, T_VEH_CAM, SIZE)
    (ob,) = perception.step(0, _detect(0.0), _T_world_veh(0.0))
    assert ob.extent[1] > 0.3  # true width 0.06 m
    assert abs(ob.center_veh[1] - (-1.0)) > 0.15  # true y = -1.0 m
