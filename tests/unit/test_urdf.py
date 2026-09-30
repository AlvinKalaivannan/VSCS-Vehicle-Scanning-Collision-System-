"""P2-T8 export/load round trip on the fixture, with exact box geometry as the parts."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh
from shapely.geometry import Point

from fixtures.synthetic import load_scene
from vscs.common.config import load_config
from vscs.common.types import JointSpec
from vscs.model.urdf import (
    display_name,
    export_model,
    load_component_model,
    load_parts,
    plan_footprints,
)

SCENE = load_scene()
MODEL_CFG = load_config("model")
SEVERITY = load_config("severity")

#: Fold hinge at the right mirror's inner rear-facing edge line: x 2.5 (mid), y -1.0
#: (body side), vertical axis. The mirror box spans x 2.40-2.60, y -1.25..-1.00.
MIRROR_HINGE = JointSpec(
    type="revolute",
    parent_link="base_link",
    child_link="right_mirror",
    axis=(0.0, 0.0, 1.0),
    origin_xyz=(2.5, -1.0, 1.5),
    limit_lower=0.0,
    limit_upper=np.pi / 2,
)


def _box_parts():
    """Each fixture component as one exact convex box (what P2-T6 yields for a box)."""
    out = {}
    for name, b in SCENE.components.items():
        m = trimesh.creation.box(extents=b.extent)
        m.apply_translation(b.center)
        out[name] = [m]
    return out


def _export(tmp_path, joints=None):
    return export_model(
        tmp_path,
        _box_parts(),
        joints or {},
        severity_cfg=SEVERITY,
        urdf_cfg=MODEL_CFG["urdf"],
        scale_error_m=0.013,
    )


def _bounds(parts):
    return np.min([p.bounds[0] for p in parts], 0), np.max([p.bounds[1] for p in parts], 0)


def test_export_writes_a_well_formed_urdf(tmp_path):
    model = _export(tmp_path)
    root = ET.parse(tmp_path / model.urdf_path).getroot()
    assert root.tag == "robot" and root.get("name") == MODEL_CFG["urdf"]["robot_name"]
    links = [lk.get("name") for lk in root.findall("link")]
    assert links[0] == "base_link" and len(links) == 1 + len(SCENE.components)
    joints = root.findall("joint")
    assert len(joints) == len(SCENE.components)
    assert all(
        j.get("type") == "fixed" and j.find("parent").get("link") == "base_link" for j in joints
    )
    # Every mesh the URDF names exists, is relative, and loads as a convex mesh.
    for m in root.iter("mesh"):
        rel = m.get("filename")
        assert not rel.startswith("/") and ":" not in rel
        assert trimesh.load(tmp_path / rel, force="mesh").is_convex


def test_components_yaml_round_trips_to_the_schema(tmp_path):
    model = _export(tmp_path)
    back = load_component_model(tmp_path)
    assert back == model
    corner = back.by_name("rear_right_bumper_corner")
    assert corner.severity == SEVERITY["components"]["rear_right_bumper_corner"]
    assert corner.display_name == "rear right bumper corner"
    assert back.by_name("underbody").min_z == pytest.approx(0.25)
    assert back.by_name("wheel_rear_left").min_z == pytest.approx(0.0)


def test_loaded_parts_are_back_in_the_veh_frame_exactly(tmp_path):
    model = _export(tmp_path)
    parts = load_parts(tmp_path, model)
    for name, b in SCENE.components.items():
        lo, hi = _bounds(parts[name])
        np.testing.assert_allclose(lo, b.lo, atol=1e-9)
        np.testing.assert_allclose(hi, b.hi, atol=1e-9)


def test_risk_footprint_gives_the_known_half_metre_to_the_pole(tmp_path):
    """CLAUDE.md §5's known answer, through export -> load -> footprint."""
    model = _export(tmp_path)
    fp = plan_footprints(load_parts(tmp_path, model))
    poly, z_min, z_max = fp["rear_right_bumper_corner"]
    assert poly.distance(Point(SCENE.pole.axis_xy)) == pytest.approx(0.5, abs=1e-6)
    assert (z_min, z_max) == pytest.approx((0.40, 0.75))
    name, d = SCENE.nearest_component_to_pole()
    assert min(fp, key=lambda n: fp[n][0].distance(Point(SCENE.pole.axis_xy))) == name
    assert d == pytest.approx(0.5)


def test_a_hinged_mirror_exports_and_folds(tmp_path):
    model = _export(tmp_path, {"right_mirror_fold": MIRROR_HINGE})
    root = ET.parse(tmp_path / model.urdf_path).getroot()
    j = next(j for j in root.findall("joint") if j.get("name") == "right_mirror_fold")
    assert j.get("type") == "revolute"
    assert float(j.find("limit").get("upper")) == pytest.approx(np.pi / 2)
    b = SCENE.component("right_mirror")
    # Closed: exactly where it was scanned.
    lo, hi = _bounds(load_parts(tmp_path, model)["right_mirror"])
    np.testing.assert_allclose(lo, b.lo, atol=1e-9)
    # Folded 90 deg about the vertical hinge at (2.5, -1.0): the 0.25 m outward reach turns
    # into x, and only half the 0.20 m x-width (0.10 m) is left sticking out in y.
    lo, hi = _bounds(load_parts(tmp_path, model, {"right_mirror_fold": np.pi / 2})["right_mirror"])
    assert lo[1] == pytest.approx(-1.10, abs=1e-9)
    assert (lo[0], hi[0]) == pytest.approx((2.50, 2.75), abs=1e-9)


def test_unknown_joint_state_is_rejected(tmp_path):
    model = _export(tmp_path)
    with pytest.raises(KeyError, match="no joints"):
        load_parts(tmp_path, model, {"left_mirror_fold": 0.5})


def test_every_component_needs_a_severity(tmp_path):
    parts = _box_parts()
    parts["tow_bar"] = parts["rear_bumper"]
    with pytest.raises(ValueError, match="no severity"):
        export_model(
            tmp_path,
            parts,
            {},
            severity_cfg=SEVERITY,
            urdf_cfg=MODEL_CFG["urdf"],
            scale_error_m=0.0,
        )


def test_display_name_override():
    assert (
        display_name("rear_right_bumper_corner", {"rear_right_bumper_corner": "rear right corner"})
        == "rear right corner"
    )
