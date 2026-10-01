"""Perception glue on a scripted reversing drive: a cone (static) and a walker (dynamic)."""

from __future__ import annotations

import numpy as np
import pytest

from fixtures.synthetic import Box, load_scene
from vscs.common.config import load_config
from vscs.common.frames import T_from_Rt, invert, look_at_T_world_cam, project, transform_points
from vscs.perception.pipeline import Box2D, Perception

SCENE = load_scene()
CFG = load_config("perception")
K, SIZE = SCENE.K, SCENE.image_size
T_VEH_CAM = look_at_T_world_cam(np.array([-1.0, 0.0, 1.2]), np.array([-3.0, 0.0, 0.0]))
S, DT = 1_000_000_000, 0.1
CONE_WORLD = (-3.0, -1.0)  # behind the van, in line with its right side


def _T_world_veh(t_s):
    return T_from_Rt(np.eye(3), [-1.0 * t_s, 0.0, 0.0])  # reversing at 1 m/s


def _walker_xy(t_s):
    # Crosses behind the van at 1 m/s, 6 m back, inside the camera's horizontal view (a box
    # clipped at the image SIDE shifts its bottom-centre and biases the position).
    return (-6.0, 2.0 - 1.0 * t_s)


def _box_of(world_box: Box, T_world_veh):
    """What a real detector reports: the box clipped to the image. Skipped only if the
    object is behind the camera or its base (the ground contact v0 needs) is out of view."""
    T_cam_world = invert(T_world_veh @ T_VEH_CAM)
    cam = transform_points(T_cam_world, world_box.corners())
    if (cam[:, 2] <= 0.1).any():
        return None
    uv, _ = project(K, cam)
    w, h = SIZE
    x0, y0 = max(0.0, np.floor(uv[:, 0].min())), max(0.0, np.floor(uv[:, 1].min()))
    x1, y1 = min(w - 1.0, np.ceil(uv[:, 0].max())), np.ceil(uv[:, 1].max())
    if y1 > h - 1 or x1 <= x0 or x0 >= w - 1 or x1 <= 0:
        return None  # base below the frame, or entirely off to a side
    return (x0, y0, x1, y1)


def _detect(t_s):
    """A perfect stub detector: true boxes of whatever is fully in view."""
    T = _T_world_veh(t_s)
    cone = Box.from_bounds(
        "cone",
        [
            CONE_WORLD[0] - 0.15,
            CONE_WORLD[1] - 0.15,
            0.0,
            CONE_WORLD[0] + 0.15,
            CONE_WORLD[1] + 0.15,
            0.5,
        ],
    )
    wx, wy = _walker_xy(t_s)
    person = Box.from_bounds("person", [wx - 0.25, wy - 0.25, 0.0, wx + 0.25, wy + 0.25, 1.7])
    out = []
    for b, cls in ((cone, "traffic_cone"), (person, "person")):
        xyxy = _box_of(b, T)
        if xyxy is not None:
            out.append(Box2D(xyxy, cls, 0.9))
    return out


def _run(n):
    p = Perception(CFG, K, T_VEH_CAM, SIZE)
    history = []
    for k in range(n):
        t = k * DT
        history.append((t, p.step(round(t * S), _detect(t), _T_world_veh(t))))
    return p, history


def test_the_cone_is_mapped_and_stays_known_after_leaving_view():
    _, history = _run(26)  # 2.5 s of reversing
    t, obs = history[-1]
    assert not any(b.cls == "traffic_cone" for b in _detect(t)), "setup: cone out of view by now"
    static = [o for o in obs if o.source == "occupancy"]
    assert len(static) == 1
    # World (-3.0, -1.0) seen from the van after reversing 2.5 m: veh (-0.5, -1.0).
    assert static[0].center_veh[0] == pytest.approx(-0.5 + 0.0, abs=0.25)
    assert static[0].center_veh[1] == pytest.approx(-1.0, abs=0.15)


def test_the_walker_is_tracked_with_velocity_and_never_mapped():
    _, history = _run(20)
    tracked = [o for _, obs in history[-5:] for o in obs if o.kind == "person"]
    assert tracked, "the walker should be a confirmed track"
    assert {o.id for o in tracked} == {tracked[0].id}  # one stable id
    assert tracked[-1].velocity_veh[1] == pytest.approx(-1.0, abs=0.3)
    assert all(o.source == "detector" for o in tracked)
    assert all(o.kind != "person" for _, obs in history for o in obs if o.source == "occupancy")


def test_ids_never_collide_and_unknown_classes_are_static():
    p, history = _run(20)
    for _, obs in history:
        assert len({o.id for o in obs}) == len(obs)
    assert p.kind_of("traffic_cone") == "static_geom" and p.kind_of("bus") == "vehicle"
