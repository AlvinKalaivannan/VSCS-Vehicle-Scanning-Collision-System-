"""The whole per-drive chain on a scripted fixture drive (the shape of P3-T6).

stub detector -> perception (ground plane, height map, tracker) -> risk engine (reference
sweep, standing in for the developer's sweep.py) -> RiskFrames -> P5-T2 metrics.

The van reverses at 1 m/s toward a pole 2.5 m behind its rear-right corner. Ground truth:
the rear-right bumper corner touches the pole at t = 2.47 s (2.5 m less the pole's 3 cm
radius). The chain must warn, naming that component, before then.

Two findings pinned here:

* Perception (R-07): box-only v0 places the thin pole a median ~0.35 m off and ~0.65 m wide;
  mask-based placement brings that to ~0.05 m / ~0.10 m.
* Alert grading (proposed ADR 0010): the alert is 'critical' from t = 0.2 s with BOTH
  boxes and masks. The cause is not perception: the rear bumper's true 0.30 m near miss,
  after sweep margins, reads 0.215 m (< 0.25 m), and a near miss is graded by its closest
  approach over the whole horizon (ADR 0005), regardless of being 2.2 s away. Changing
  that needs a schema field (time of closest approach) and the developer's decision.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull

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


def _detect(t, with_mask=False):
    cam = transform_points(invert(_T_world_veh(t) @ T_VEH_CAM), POLE.corners())
    if (cam[:, 2] <= 0.1).any():
        return []
    uv, _ = project(K, cam)
    w, h = SIZE
    x0, y0 = max(0.0, np.floor(uv[:, 0].min())), max(0.0, np.floor(uv[:, 1].min()))
    x1, y1 = min(w - 1.0, np.ceil(uv[:, 0].max())), np.ceil(uv[:, 1].max())
    if y1 > h - 1:
        return []
    mask = None
    if with_mask:  # what SAM 2 would give: the silhouette, clipped by the frame
        hull = uv[ConvexHull(uv).vertices]
        img = Image.new("1", (w, h), 0)
        ImageDraw.Draw(img).polygon([tuple(p) for p in hull], fill=1)
        mask = np.asarray(img, dtype=bool)
    return [Box2D((x0, y0, x1, y1), "pole", 0.9, mask)]


def _drive(with_mask=False):
    perception = Perception(PCFG, K, T_VEH_CAM, SIZE)
    sm = AlertStateMachine(RISK)
    components = [ref_box(n, b.lo, b.hi) for n, b in SCENE.components.items()]
    frames = []
    for k in range(int(T_EVENT_S / DT) + 1):
        t = k * DT
        obs = perception.step(round(t * S), _detect(t, with_mask), _T_world_veh(t))
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
    """R-07: the perceived pole is far wider than its true 0.06 m when placed from a box."""
    perception = Perception(PCFG, K, T_VEH_CAM, SIZE)
    (ob,) = perception.step(0, _detect(0.0), _T_world_veh(0.0))
    assert ob.extent[1] > 0.3  # true width 0.06 m
    assert abs(ob.center_veh[1] - (-1.0)) > 0.15  # true y = -1.0 m


def test_masks_place_the_pole_accurately_throughout_the_drive():
    """The perception fix: from its mask, the pole is within ~7 cm in every frame."""
    perception = Perception(PCFG, K, T_VEH_CAM, SIZE)
    for k in range(int(T_EVENT_S / DT) + 1):
        t = k * DT
        (ob,) = perception.step(round(t * S), _detect(t, with_mask=True), _T_world_veh(t))
        true_xy = np.array([-3.5 - SPEED * t, -1.0])  # pole in veh after reversing |SPEED| t
        assert np.hypot(*(np.array(ob.center_veh[:2]) - true_xy)) < 0.1
        assert ob.extent[1] < 0.15


def test_near_miss_grading_makes_it_critical_early_with_masks_too_adr_0010():
    """Pinned for ADR 0010 (proposed): with correct perception, the rear bumper's 0.30 m
    near miss 2.2 s ahead still grades 'critical' now. Update this test if the developer
    approves grading near misses by time of closest approach."""
    frames = _drive(with_mask=True)
    f = frames[1]
    bumper = next(c for c in f.per_component if c.component == "rear_bumper")
    assert (
        bumper.ttc_s is None
        and bumper.min_distance_m < RISK["alerts"]["levels"]["critical"]["min_distance_m_below"]
    )
    assert f.alert_level == "critical"
    truth = Mx.PassTruth(component="rear_right_bumper_corner", t_event_ns=round(T_EVENT_S * S))
    assert Mx.attribution_correct(frames, truth, FLAG)  # still names the right corner
