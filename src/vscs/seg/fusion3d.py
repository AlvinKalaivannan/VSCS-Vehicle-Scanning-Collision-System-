"""3D label fusion: 2D component masks -> labels on 3D points (P2-T3).

CORE CONTRIBUTION MODULE — pair mode (CLAUDE.md §0). The developer writes the first
draft; this file holds only the contract: data containers, signatures and docstrings.
Every function raises ``NotImplementedError`` until implemented.

The target is ``tests/unit/test_fusion3d.py``. It fails until this module is complete.
Acceptance (§6 P2-T3): mean per-component IoU >= 0.70 against the hand-labelled subset.

The problem
-----------
SAM 2 gives a component mask per *image*: "pixel (u, v) of frame i is the right mirror".
The collision model needs labels on the *3D points* of the van. Each 3D point is seen by
many frames, and each frame may say something different about it: the mask is wrong near
edges, the mirror bleeds onto the door (R-05), and a point on the far side of the van
projects onto whatever is *in front of* it. So: ask every frame, discard the frames that
cannot actually see the point, and let the rest vote.

Step 1 - where does point X land in frame i? (projection)
---------------------------------------------------------
With ``T_cam_world`` the world-to-camera pose (§4.1 naming: it maps world points into
the camera frame) and ``K`` the intrinsics::

    X_c = T_cam_world @ [X; 1]                (4x4 times a homogeneous point)
    [u; v; 1] ~ K @ X_c   ->   u = fx * x/z + cx,   v = fy * y/z + cy

``common.frames.transform_points`` and ``common.frames.project`` already do this, and
``project`` returns NaN and ``valid=False`` for points behind the camera. Use them.
Round ``(u, v)`` to the nearest integer to get the pixel: pixel centres sit at integer
coordinates (OpenCV). Points that fall outside the image are not observed.

Step 2 - can frame i actually see X? (occlusion / depth test)
-------------------------------------------------------------
Projection alone is not enough. A point on the *front* bumper, seen from behind the van,
projects onto the *rear* doors: projection maps a whole ray to one pixel, and it does
not know what else is on that ray. The frame's depth map ``D_i`` (from COLMAP dense,
P1-T7) says how far away the *first* surface along that pixel's ray is. X is visible in
frame i only if it *is* that first surface::

    visible  <=>  z_X  <=  D_i[v, u] + tol

with ``z_X`` the point's camera-frame z (depth along the optical axis, the same quantity
the depth map stores) and ``tol = fusion3d.occlusion_depth_tolerance_m``. A point well
behind the surface is occluded. The tolerance absorbs depth-map noise and the fact that
a pixel covers a small patch of surface, not one point.

Without this test the vote is dominated by frames looking *through* the van, which is
the textbook failure of naive label projection. ``test_fusion3d.py`` has a test that
shows the wrong answer appearing when the tolerance is set to infinity.

Step 3 - collect votes
----------------------
Every frame that sees X casts one vote: the label under X in that frame's mask. Masks
use ``NONE_LABEL`` (-1) for "no component here" (bare body panel, background). That
counts as a vote too, a vote for "none". It is what lets a sharp mask outvote a
bleeding one (R-05). Keep the votes as a count matrix::

    votes[n, c]   = frames that see point n and label it component c,   c = 0 .. C-1
    votes[n, C]   = frames that see point n and label it "none"
    n_obs[n]      = sum over c of votes[n, c]

Step 4 - decide
---------------
For each point, the winner is the column with the most votes (ties go to the lowest
column index, so the result is deterministic). The point gets the winning component's
label only if **all** of these hold, and ``NONE_LABEL`` otherwise:

* ``n_obs >= fusion3d.min_observations_per_point``: a point seen once or twice is not
  trusted;
* the winner is a component, not the "none" column;
* ``votes[winner] / n_obs >= fusion3d.min_vote_fraction``: a narrow plurality is not
  consensus.

``confidence = votes[winner] / n_obs`` (0 where ``n_obs == 0``) is kept for P2-T5
cleanup, which can use it to decide which stray labels to drop.

Cost: ``N`` points x ``F`` frames projections. That is a few hundred thousand points
times a few hundred frames: vectorise over points (one frame at a time), not over both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

#: Mask value and fused label meaning "no component".
NONE_LABEL = -1


@dataclass(frozen=True)
class PosedView:
    """One posed frame with its 2D component mask and its depth map.

    ``labels[v, u]`` is a component index in ``0 .. n_classes-1`` or ``NONE_LABEL``.
    ``depth[v, u]`` is z-depth in metres of the first surface along that pixel's ray, and
    ``inf`` where there is none. Both are ``(H, W)`` and pixel-aligned.
    """

    K: FloatArray
    T_cam_world: FloatArray
    labels: IntArray
    depth: FloatArray

    @property
    def image_size(self) -> tuple[int, int]:
        """``(width, height)``, the order ``frames.project`` expects."""
        return int(self.labels.shape[1]), int(self.labels.shape[0])


@dataclass(frozen=True)
class FusionResult:
    labels: IntArray  # (N,) component index or NONE_LABEL
    confidence: FloatArray  # (N,) winner's vote fraction; 0 where never observed
    n_obs: IntArray  # (N,) frames that could see the point
    votes: IntArray  # (N, n_classes + 1); last column = "none" votes


def observe(
    points_world: npt.ArrayLike, view: PosedView, depth_tolerance_m: float
) -> tuple[IntArray, npt.NDArray[np.bool_]]:
    """What one frame says about each point.

    Returns ``(pixel_label (N,), visible (N,))``. ``visible`` is True only for points in
    front of the camera, inside the image, and passing the depth test (step 2).
    ``pixel_label`` is the mask value under each visible point, and ``NONE_LABEL`` where
    ``visible`` is False.
    """
    raise NotImplementedError("P2-T3: developer's first draft")


def accumulate_votes(
    points_world: npt.ArrayLike,
    views: list[PosedView],
    n_classes: int,
    depth_tolerance_m: float,
) -> IntArray:
    """The ``(N, n_classes + 1)`` vote-count matrix of step 3."""
    raise NotImplementedError("P2-T3: developer's first draft")


def decide(
    votes: npt.ArrayLike, *, min_observations: int, min_vote_fraction: float
) -> tuple[IntArray, FloatArray, IntArray]:
    """Step 4. Returns ``(labels, confidence, n_obs)``."""
    raise NotImplementedError("P2-T3: developer's first draft")


def fuse_labels(
    points_world: npt.ArrayLike,
    views: list[PosedView],
    n_classes: int,
    fusion_cfg: dict[str, Any],
) -> FusionResult:
    """Steps 1-4. ``fusion_cfg`` is the ``fusion3d`` block of ``configs/seg.yaml``."""
    raise NotImplementedError("P2-T3: developer's first draft")
