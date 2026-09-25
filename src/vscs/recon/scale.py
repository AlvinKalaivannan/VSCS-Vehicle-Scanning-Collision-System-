"""Metric scale recovery from markers of known size (P1-T6).

CORE CONTRIBUTION MODULE — pair mode (CLAUDE.md §0). The developer writes the first
draft; this file currently holds only the contract: data containers, signatures and
docstrings. Every function raises ``NotImplementedError`` until implemented.

The target is ``tests/unit/test_scale.py``. It fails until this module is complete.

The problem
-----------
Structure-from-motion from one moving camera recovers the scene's *shape* but not its
*size*. If every point and every camera position were multiplied by the same factor, every
image would look identical - so the images alone cannot tell a 5 m van from a 50 cm model
of it. The reconstruction is correct up to an unknown scale ``λ``:

    X_sfm = λ · X_true

P1-T6's ±2 cm gate is unreachable until ``λ`` is found. Markers of measured size fix it: if
a marker whose real side is ``L`` comes out ``l_rec`` long in the reconstruction, then
``s = L / l_rec = 1 / λ`` and multiplying everything by ``s`` makes the model metric.

Step 1 - where is each marker corner in 3D? (linear triangulation, DLT)
-----------------------------------------------------------------------
Each registered image ``i`` has a 3x4 projection matrix ``P_i = K [R_i | t_i]``, built from
COLMAP's world-to-camera pose. A 3D point ``X`` (homogeneous, ``X ∈ R^4``) appears at pixel
``(u, v)`` where

    u = (p1 · X) / (p3 · X),    v = (p2 · X) / (p3 · X)

with ``p1, p2, p3`` the rows of ``P_i``. Cross-multiplying removes the division and gives
two equations that are *linear* in ``X``:

    (u p3 - p1) · X = 0
    (v p3 - p2) · X = 0

Stack those two rows for every image the corner appears in, to get ``A X = 0`` with ``A``
of shape ``(2n, 4)``. With perfect data ``X`` is in the null space of ``A``. With noisy data
there is no exact null space, so take the unit vector minimising ``||A X||``: that is the
right singular vector of ``A`` with the smallest singular value (equivalently, the
eigenvector of ``A^T A`` with the smallest eigenvalue - MAT188's least-squares via the
normal equations). Divide by its fourth component to get the 3D point.

At least two views are needed: one view gives only two equations for three unknowns, which
is a ray, not a point.

Step 2 - how big is each marker?
--------------------------------
With four triangulated corners, the reconstructed side ``l_rec`` is the mean of the four edge
lengths. Averaging the four edges damps the error of any single corner.

Step 3 - combine the markers robustly
-------------------------------------
Each marker ``m`` gives its own estimate ``s_m = L / l_m``. Take the **median**, not the mean:
one misprinted or curled marker then cannot drag the answer. Report the spread as
``(max - min) / median`` of the per-marker estimates. If it exceeds
``scale.marker.max_spread_frac``, the markers disagree, and that is flagged rather than
averaged away (R-02).

Why range and not median absolute deviation: with the three markers the checklist asks
for, two agreeing markers make the MAD exactly zero, so a third marker printed 10% too
large would go unflagged. The range catches it. The *estimate* is still the median, so the
bad marker is flagged but does not move the answer. Fewer than
``scale.marker.min_markers_detected`` markers is a hard failure, because §8.2's fallback
(tape-measured dimensions) is then the right move, and that is the developer's call.

Step 4 - apply it
-----------------
Points scale directly: ``X' = s X``. A camera's world-to-camera translation scales too:
from ``X_cam = R X + t``, substituting ``X = X' / s`` and scaling the camera coordinates by
``s`` gives ``X'_cam = R X' + s t``. Rotations are unchanged - scale does not rotate anything.

Step 5 - validate against the tape
----------------------------------
After the vehicle frame is built (``vehicle_frame.py``), compare its dimensions with the
hand measurements. ``scale_error_m`` is the worst absolute error over the dimensions both
sides have - the P1-T6 acceptance figure, gate 0.02 m.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ScaleEstimate:
    """The recovered scale and how much the markers agreed about it.

    ``scale`` multiplies reconstruction units into metres. ``per_marker`` maps marker id to
    that marker's own estimate. ``spread_frac`` is ``(max - min) / median`` of those
    estimates; ``consistent`` is ``spread_frac <= max_spread_frac``.
    """

    scale: float
    per_marker: dict[int, float] = field(default_factory=dict)
    spread_frac: float = 0.0
    consistent: bool = True

    @property
    def n_markers(self) -> int:
        return len(self.per_marker)


def projection_matrix(K: npt.ArrayLike, T_cam_world: npt.ArrayLike) -> FloatArray:
    """The 3x4 camera matrix ``P = K [R | t]`` from intrinsics and a world-to-camera pose."""
    raise NotImplementedError("P1-T6: developer's first draft")


def triangulate_point(projections: list[npt.ArrayLike], uvs: list[npt.ArrayLike]) -> FloatArray:
    """Linear (DLT) triangulation of one point.

    The point is seen at pixel ``uvs[i]`` by the camera with matrix ``projections[i]``.

    Raises ``ValueError`` with fewer than two views, or if the lists differ in length.
    """
    raise NotImplementedError("P1-T6: developer's first draft")


def marker_side_length(corners: npt.ArrayLike) -> float:
    """Mean of the four edge lengths of a marker's corners, given in order around the square."""
    raise NotImplementedError("P1-T6: developer's first draft")


def estimate_scale(
    reconstructed_sides: dict[int, float],
    true_side_m: float,
    *,
    min_markers: int,
    max_spread_frac: float,
) -> ScaleEstimate:
    """Robust scale from several markers: the median of ``true_side_m / side`` per marker.

    Raises ``ValueError`` if fewer than ``min_markers`` markers are given, or if any side
    length is not positive.
    """
    raise NotImplementedError("P1-T6: developer's first draft")


def estimate_scale_from_config(
    reconstructed_sides: dict[int, float], recon_cfg: dict[str, Any]
) -> ScaleEstimate:
    """:func:`estimate_scale` with the marker size and limits from ``configs/recon.yaml``."""
    raise NotImplementedError("P1-T6: developer's first draft")


def scale_points(points: npt.ArrayLike, scale: float) -> FloatArray:
    """``X' = s X`` for ``(N, 3)`` points."""
    raise NotImplementedError("P1-T6: developer's first draft")


def scale_T_cam_world(T_cam_world: npt.ArrayLike, scale: float) -> FloatArray:
    """Rescale a world-to-camera pose: rotation unchanged, translation multiplied by ``s``."""
    raise NotImplementedError("P1-T6: developer's first draft")


def dimension_errors(
    reconstructed: dict[str, float], measured: dict[str, float]
) -> dict[str, float]:
    """Signed error ``reconstructed - measured`` for every dimension present in both."""
    raise NotImplementedError("P1-T6: developer's first draft")


def scale_error_m(reconstructed: dict[str, float], measured: dict[str, float]) -> float:
    """Worst absolute dimension error - the P1-T6 acceptance figure.

    Raises ``ValueError`` if the two share no dimension, since there is then nothing to
    validate against.
    """
    raise NotImplementedError("P1-T6: developer's first draft")
