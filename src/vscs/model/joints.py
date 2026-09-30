"""Joints for articulated parts: hinged doors and folding mirrors (P2-T7).

Acceptance (CLAUDE.md §6): Rerun shows the door swinging correctly through its range.
The swing itself is this module's forward kinematics; the Rerun view comes with P3-T1.

Frames (MAT188 notation, §4.1 naming)
-------------------------------------
A part is scanned **closed**, so its points arrive in the ``veh`` frame at joint value
``q = 0``. A URDF joint places the child link's frame at the hinge::

    T_parent_joint = [ R_rpy  t_xyz ]        (the joint's <origin>)
                     [ 0      1     ]

and the joint then moves the child relative to that frame by ``M(q)``: a rotation of
``q`` radians about ``axis`` (revolute) or a slide of ``q`` metres along it (prismatic).
So a point ``p_child`` fixed to the part sits in the parent (``veh``) frame at::

    p_veh(q) = T_parent_joint @ M(q) @ p_child

At ``q = 0``, ``M = I``, which gives ``p_child = inv(T_parent_joint) @ p_veh(0)``: the
scanned points re-expressed relative to the hinge. That is what the URDF stores.

The sliding door is not here: it runs on a curved track and is two scanned variants
instead (ADR 0004).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from vscs.common.frames import (
    T_from_Rt,
    axis_angle_to_R,
    invert,
    rot_x,
    rot_y,
    rot_z,
    transform_points,
)
from vscs.common.types import JointSpec

FloatArray = npt.NDArray[np.float64]

#: Slack on joint limits, so a value computed as exactly ``upper`` is not rejected.
LIMIT_TOL = 1e-9


def rpy_to_R(rpy: npt.ArrayLike) -> FloatArray:
    """URDF roll-pitch-yaw: fixed-axis X, then Y, then Z, i.e. ``Rz @ Ry @ Rx``."""
    r, p, y = (float(v) for v in np.asarray(rpy).reshape(3))
    return rot_z(y) @ rot_y(p) @ rot_x(r)


def joint_origin(spec: JointSpec) -> FloatArray:
    """``T_parent_joint``: the hinge frame in the parent link's frame."""
    return T_from_Rt(rpy_to_R(spec.origin_rpy), np.asarray(spec.origin_xyz, dtype=np.float64))


def joint_motion(spec: JointSpec, q: float) -> FloatArray:
    """``M(q)``: the child's motion relative to the hinge frame. Enforces the limits."""
    if spec.type in ("revolute", "prismatic") and not (
        spec.limit_lower - LIMIT_TOL <= q <= spec.limit_upper + LIMIT_TOL
    ):
        raise ValueError(
            f"{spec.child_link}: q = {q} is outside [{spec.limit_lower}, {spec.limit_upper}]"
        )
    axis = np.asarray(spec.axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    if spec.type in ("revolute", "continuous"):
        return T_from_Rt(axis_angle_to_R(axis, q), np.zeros(3))
    if spec.type == "prismatic":
        return T_from_Rt(np.eye(3), axis * q)
    return np.eye(4)  # fixed


def to_child_frame(spec: JointSpec, points_veh_closed: npt.ArrayLike) -> FloatArray:
    """Scanned (closed, ``q = 0``) points -> the child link frame the URDF stores."""
    return transform_points(invert(joint_origin(spec)), np.asarray(points_veh_closed))


def posed(spec: JointSpec, points_child: npt.ArrayLike, q: float) -> FloatArray:
    """Child-frame points -> parent (``veh``) frame at joint value ``q``."""
    return transform_points(joint_origin(spec) @ joint_motion(spec, q), np.asarray(points_child))


def sweep_values(spec: JointSpec, n: int) -> FloatArray:
    """``n`` joint values across the range, for viewing the swing (P2-T7's Rerun check)."""
    if spec.type not in ("revolute", "prismatic"):
        raise ValueError(f"{spec.child_link}: a {spec.type} joint has no finite range to sweep")
    return np.linspace(spec.limit_lower, spec.limit_upper, n)


def joint_specs_from_config(
    joints_cfg: dict[str, Any], base_link: str
) -> tuple[dict[str, JointSpec], list[str]]:
    """Build joints from ``model.yaml``'s ``joints`` block.

    Returns ``(specs, unmeasured)``. A joint without a measured ``origin_xyz`` (the hinge
    position, measured on the van at P2-T7) is listed in ``unmeasured`` and left out,
    rather than being given a made-up origin.
    """
    specs: dict[str, JointSpec] = {}
    unmeasured: list[str] = []
    for name, j in joints_cfg.items():
        if not isinstance(j, dict):
            continue  # e.g. the "status" string
        if j.get("origin_xyz") is None:
            unmeasured.append(name)
            continue
        specs[name] = JointSpec(
            type=j["type"],
            parent_link=j.get("parent_link", base_link),
            child_link=j.get("child_link", name),
            axis=tuple(j.get("axis", (0.0, 0.0, 1.0))),
            origin_xyz=tuple(j["origin_xyz"]),
            origin_rpy=tuple(j.get("origin_rpy", (0.0, 0.0, 0.0))),
            limit_lower=float(j.get("limit_lower", 0.0)),
            limit_upper=float(j.get("limit_upper", 0.0)),
        )
    return specs, unmeasured


def validate_joints(joints: dict[str, JointSpec], links: set[str], base_link: str) -> None:
    """Every joint must connect links that exist, and no link may have two parents."""
    problems: list[str] = []
    children: dict[str, str] = {}
    for name, j in joints.items():
        if j.child_link not in links:
            problems.append(f"{name}: child link {j.child_link!r} is not a component")
        if j.parent_link != base_link and j.parent_link not in links:
            problems.append(f"{name}: parent link {j.parent_link!r} does not exist")
        if j.child_link in children:
            problems.append(
                f"{name}: link {j.child_link!r} already moved by joint {children[j.child_link]!r}"
            )
        children[j.child_link] = name
    if problems:
        raise ValueError(
            "joint configuration does not match the components:\n  " + "\n  ".join(problems)
        )
