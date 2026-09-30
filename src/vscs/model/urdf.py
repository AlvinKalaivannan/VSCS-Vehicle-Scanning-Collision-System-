"""Collision-model export and loading: URDF + meshes + components.yaml (P2-T8).

Acceptance (CLAUDE.md §6): the model loads in a URDF viewer and in ``risk/``; a fixture
test for loading passes.

Layout written to ``out_dir``::

    vscs_van.urdf         links + joints; mesh paths relative to this file
    meshes/<link>_<i>.obj one convex part per file (P2-T6 output)
    components.yaml       the §4.2 ComponentModel (names, severities, min_z, joints)

Every component is one URDF link. A link moved by a joint (P2-T7) is attached to its
parent through that joint, and its meshes are stored in the joint's frame; every other
link is attached to ``base_link`` by a fixed joint, with its meshes in ``veh``. So the
model loads back into the ``veh`` frame at any joint state (``load_parts``).

``plan_footprints`` gives ``risk/`` what the 2D engine needs (P3-T4): each component's
plan-view outline and vertical extent, which ``sweep.py`` wraps as its ``Footprint``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import yaml
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

from vscs.common.types import Component, ComponentModel, JointSpec
from vscs.model.joints import joint_motion, joint_origin, to_child_frame, validate_joints

COMPONENTS_FILE = "components.yaml"


def display_name(name: str, overrides: dict[str, str] | None = None) -> str:
    """Driver-facing name: an override if configured, else the machine name in words."""
    return (overrides or {}).get(name, name.replace("_", " "))


def _fmt(v: Any) -> str:
    return " ".join(f"{float(x):.9g}" for x in v)


def _joint_element(robot: ET.Element, name: str, spec: JointSpec) -> None:
    j = ET.SubElement(robot, "joint", name=name, type=spec.type)
    ET.SubElement(j, "parent", link=spec.parent_link)
    ET.SubElement(j, "child", link=spec.child_link)
    ET.SubElement(j, "origin", xyz=_fmt(spec.origin_xyz), rpy=_fmt(spec.origin_rpy))
    if spec.type != "fixed":
        ET.SubElement(j, "axis", xyz=_fmt(spec.axis))
    if spec.type in ("revolute", "prismatic"):
        # URDF requires effort and velocity on <limit>; VSCS never drives joints (advisory).
        ET.SubElement(
            j,
            "limit",
            lower=f"{spec.limit_lower:.9g}",
            upper=f"{spec.limit_upper:.9g}",
            effort="0",
            velocity="0",
        )


def export_model(
    out_dir: Path,
    parts_veh: dict[str, list[trimesh.Trimesh]],
    joints: dict[str, JointSpec],
    *,
    severity_cfg: dict[str, Any],
    urdf_cfg: dict[str, Any],
    scale_error_m: float,
    display_names: dict[str, str] | None = None,
) -> ComponentModel:
    """Write the URDF, meshes and ``components.yaml``. ``parts_veh`` is P2-T6's output."""
    out_dir = Path(out_dir)
    base = str(urdf_cfg["base_link"])
    ext = str(urdf_cfg["mesh_format"])
    mesh_dir = out_dir / str(urdf_cfg["mesh_dir"])
    mesh_dir.mkdir(parents=True, exist_ok=True)

    names = sorted(parts_veh)
    missing = [n for n in names if n not in severity_cfg["components"]]
    if missing:
        raise ValueError(f"no severity for {missing}; severity.yaml must cover every component")
    validate_joints(joints, set(names), base)
    joint_of = {j.child_link: (jn, j) for jn, j in joints.items()}

    robot = ET.Element("robot", name=str(urdf_cfg["robot_name"]))
    ET.SubElement(robot, "link", name=base)
    components: list[Component] = []
    for name in names:
        parts = parts_veh[name]
        if not parts:
            raise ValueError(f"{name}: no collision parts")
        link = ET.SubElement(robot, "link", name=name)
        for i, part in enumerate(parts):
            verts = part.vertices
            if name in joint_of:
                verts = to_child_frame(joint_of[name][1], verts)
            rel = f"{urdf_cfg['mesh_dir']}/{name}_{i}.{ext}"
            trimesh.Trimesh(verts, part.faces, process=False).export(out_dir / rel)
            for tag in ("visual", "collision"):
                el = ET.SubElement(link, tag)
                geom = ET.SubElement(el, "geometry")
                ET.SubElement(geom, "mesh", filename=rel)
        if name in joint_of:
            jn, spec = joint_of[name]
            _joint_element(robot, jn, spec)
        else:
            _joint_element(
                robot, f"{name}_fixed", JointSpec(type="fixed", parent_link=base, child_link=name)
            )
        components.append(
            Component(
                name=name,
                display_name=display_name(name, display_names),
                link=name,
                severity=float(severity_cfg["components"][name]),
                min_z=float(min(p.bounds[0][2] for p in parts)),
            )
        )

    urdf_path = out_dir / f"{urdf_cfg['robot_name']}.urdf"
    ET.indent(robot)
    ET.ElementTree(robot).write(urdf_path, encoding="utf-8", xml_declaration=True)

    model = ComponentModel(
        urdf_path=Path(urdf_path.name),
        components=components,
        joints=joints,
        scale_error_m=scale_error_m,
    )
    (out_dir / COMPONENTS_FILE).write_text(
        yaml.safe_dump(model.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    return model


def load_component_model(out_dir: Path) -> ComponentModel:
    """Read ``components.yaml`` back into the §4.2 schema (validated)."""
    raw = yaml.safe_load((Path(out_dir) / COMPONENTS_FILE).read_text(encoding="utf-8"))
    return ComponentModel.model_validate(raw)


def load_parts(
    out_dir: Path, model: ComponentModel, joint_values: dict[str, float] | None = None
) -> dict[str, list[trimesh.Trimesh]]:
    """Every component's convex parts in the ``veh`` frame at the given joint values.

    Joints not named in ``joint_values`` are at ``q = 0`` (closed, as scanned).
    """
    out_dir = Path(out_dir)
    q = joint_values or {}
    unknown = set(q) - set(model.joints)
    if unknown:
        raise KeyError(f"no joints {sorted(unknown)}; have {sorted(model.joints)}")
    joint_of = {j.child_link: (jn, j) for jn, j in model.joints.items()}
    root = ET.parse(out_dir / model.urdf_path).getroot()
    parts: dict[str, list[trimesh.Trimesh]] = {}
    for link in root.findall("link"):
        meshes = [m.get("filename") for m in link.findall("collision/geometry/mesh")]
        if not meshes:
            continue  # base_link
        name = link.get("name")
        T = np.eye(4)
        if name in joint_of:
            jn, spec = joint_of[name]
            T = joint_origin(spec) @ joint_motion(spec, q.get(jn, 0.0))
        loaded = []
        for rel in meshes:
            m = trimesh.load(out_dir / rel, force="mesh", process=False)
            m.apply_transform(T)
            loaded.append(m)
        parts[name] = loaded
    return parts


def plan_footprints(
    parts_veh: dict[str, list[trimesh.Trimesh]],
) -> dict[str, tuple[Polygon, float, float]]:
    """``{name: (plan-view polygon, z_min, z_max)}`` in ``veh`` for the 2D risk engine.

    Each convex part projects to a convex polygon; a component's outline is their union,
    so a concave part keeps its concavity in plan.
    """
    out: dict[str, tuple[Polygon, float, float]] = {}
    for name, parts in parts_veh.items():
        polys = [MultiPoint([tuple(v) for v in p.vertices[:, :2]]).convex_hull for p in parts]
        z_min = float(min(p.bounds[0][2] for p in parts))
        z_max = float(max(p.bounds[1][2] for p in parts))
        out[name] = (unary_union(polys), z_min, z_max)
    return out
