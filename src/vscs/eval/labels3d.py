"""Score 3D component labels against the hand-labelled gold subset (P2-T3, P2-T4, P2-T5).

The gold subset is made in CloudCompare (or similar) on ``points_veh.ply``, the cloud
``scripts/fuse.py`` writes: select points, give them a scalar field holding the component
index (the order of ``names`` in ``fused.npz``; ``-1`` = no component, e.g. bare body
panel), and export as ASCII ``x y z label``. Because the gold points *are* points of that
cloud, each one is matched to its twin by nearest neighbour within a tolerance far smaller
than the point spacing. A gold point with no twin means it came from another cloud or
frame, which is reported, never silently scored.

The IoU itself is ``eval.metrics.per_component_iou`` over the matched points.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from vscs.eval.metrics import mean_iou, per_component_iou

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def read_gold_points(path: Path) -> tuple[FloatArray, IntArray]:
    """``x y z label`` rows (space or comma separated; ``//`` or ``#`` header lines skipped).

    Extra columns between xyz and the label (colours, normals) are allowed: the label is
    the last column.
    """
    rows = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith(("//", "#")):
            continue
        rows.append([float(v) for v in s.replace(",", " ").split()])
    if not rows:
        raise ValueError(f"{path}: no points")
    if len({len(r) for r in rows}) != 1 or len(rows[0]) < 4:
        raise ValueError(f"{path}: every row needs x y z ... label, with the same columns")
    a = np.array(rows)
    lab = a[:, -1]
    if not np.allclose(lab, np.round(lab)):
        raise ValueError(f"{path}: labels must be whole numbers (component index or -1)")
    return a[:, :3], np.round(lab).astype(np.int64)


@dataclass(frozen=True)
class LabelScore:
    iou: dict[str, float]  # per component present in gold or prediction
    mean_iou: float
    n_gold: int
    n_matched: int
    gold_per_component: dict[str, int]


def score_labels(
    points: npt.ArrayLike,
    labels: npt.ArrayLike,
    names: list[str],
    gold_points: npt.ArrayLike,
    gold_labels: npt.ArrayLike,
    match_tolerance_m: float,
) -> LabelScore:
    pts = np.asarray(points, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.int64)
    gp = np.asarray(gold_points, dtype=np.float64)
    gl = np.asarray(gold_labels, dtype=np.int64)
    bad = set(np.unique(gl).tolist()) - {-1} - set(range(len(names)))
    if bad:
        raise ValueError(f"gold labels {sorted(bad)} are not component indices 0..{len(names) - 1}")
    dist, idx = cKDTree(pts).query(gp, k=1)
    ok = dist <= match_tolerance_m
    if not ok.any():
        raise ValueError(
            "no gold point matches the labelled cloud: was the gold set made on this run's "
            "points_veh.ply (veh frame, metres)?"
        )
    ious = per_component_iou(lab[idx[ok]], gl[ok], range(len(names)))
    return LabelScore(
        iou={names[c]: v for c, v in ious.items()},
        mean_iou=mean_iou(ious),
        n_gold=len(gl),
        n_matched=int(ok.sum()),
        gold_per_component={n: int((gl == i).sum()) for i, n in enumerate(names)},
    )
