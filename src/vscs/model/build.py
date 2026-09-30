"""Labelled points -> collision model: decompose every component, then export (P2-T6..T8).

Input is the cleaned 3D labels from P2-T5 as a ``.npz`` with ``points (N, 3)`` in ``veh``,
``labels (N,)`` (component index, or ``NONE_LABEL``) and ``names`` (index -> component
name). Output is a ``model`` run folder holding the URDF, meshes and components.yaml.

Hinges whose origin has not been measured yet (P2-T7) are left out and logged: their
parts are exported fixed, in the closed position they were scanned in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from vscs.common.log import get_logger
from vscs.common.types import ComponentModel
from vscs.model.decompose import DecomposedComponent, decompose_component
from vscs.model.joints import joint_specs_from_config
from vscs.model.urdf import export_model
from vscs.seg.labels import NONE_LABEL

logger = get_logger("model.build")


@dataclass(frozen=True)
class BuildResult:
    model: ComponentModel
    decomposed: list[DecomposedComponent]
    unmeasured_joints: list[str]

    @property
    def worst_volume_error(self) -> float:
        return max(d.volume_error_fraction for d in self.decomposed)

    @property
    def passed(self) -> bool:
        """P2-T6: every component within the volume gate."""
        return all(d.passed for d in self.decomposed)


def load_labelled_points(path: Path) -> tuple[npt.NDArray, npt.NDArray, list[str]]:
    with np.load(path, allow_pickle=False) as z:
        pts, labels, names = z["points"], z["labels"], [str(n) for n in z["names"]]
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) != len(labels):
        raise ValueError(f"{path}: points must be (N, 3) matching labels (N,)")
    bad = {int(v) for v in np.unique(labels)} - {NONE_LABEL} - set(range(len(names)))
    if bad:
        raise ValueError(f"{path}: labels {sorted(bad)} have no name")
    return pts, labels, names


def build_collision_model(
    points: npt.ArrayLike,
    labels: npt.ArrayLike,
    names: list[str],
    out_dir: Path,
    *,
    model_cfg: dict[str, Any],
    severity_cfg: dict[str, Any],
    scale_error_m: float,
) -> BuildResult:
    pts = np.asarray(points, dtype=np.float64)
    lab = np.asarray(labels)
    decomposed: list[DecomposedComponent] = []
    for idx, name in enumerate(names):
        sel = lab == idx
        if not sel.any():
            logger.warning("%s: no labelled points - left out of the model", name)
            continue
        d = decompose_component(name, pts[sel], model_cfg["decompose"])
        logger.info(d.summary())
        decomposed.append(d)
    if not decomposed:
        raise ValueError("no component has any labelled points")

    joints, unmeasured = joint_specs_from_config(
        model_cfg["joints"], model_cfg["urdf"]["base_link"]
    )
    present = {d.name for d in decomposed}
    for name in unmeasured:
        logger.warning(
            "joint %s: hinge origin not measured yet (P2-T7); exported fixed, closed", name
        )
    joints = {k: j for k, j in joints.items() if j.child_link in present}

    model = export_model(
        Path(out_dir),
        {d.name: d.parts for d in decomposed},
        joints,
        severity_cfg=severity_cfg,
        urdf_cfg=model_cfg["urdf"],
        scale_error_m=scale_error_m,
        display_names=model_cfg.get("display_names"),
    )
    return BuildResult(model, decomposed, unmeasured)
