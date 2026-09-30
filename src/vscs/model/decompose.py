"""Per-component collision geometry: points -> solid -> convex parts (P2-T6).

Acceptance (CLAUDE.md §6): each component's decomposed volume is within
``decompose.max_volume_error_fraction`` (±15%) of its point-cloud-derived volume.

Why a solid first
-----------------
A scan sees only a component's outer skin: a bumper arrives as a curved *sheet* of
points. Convex decomposition needs a closed shape, and "volume" is not defined for a
sheet. So the points are turned into a voxel solid, and that one solid supplies both:

1. **Voxelize** at ``voxel_m``: a voxel is occupied if any point falls in it.
2. **Close** gaps between neighbouring points (morphological closing,
   ``close_iterations``), then **fill** enclosed interior (a closed shell becomes solid;
   an open sheet stays a sheet one or two voxels thick).
3. **Reference volume** = occupied voxels x ``voxel_m``^3. This is the "point-cloud-derived
   volume" of the P2-T6 gate.
4. **Surface mesh** = the boundary faces of the occupied voxels (a face is kept where an
   occupied voxel meets an empty one). Watertight by construction, and small: CoACD runs
   on a few thousand faces instead of a solid's worth of cubes (5 s instead of 64 s on
   the L-shaped test solid).

Why decompose at all
--------------------
Collision checks need convex pieces. One convex hull per component fills in every
concave region: on a thin L-shaped part the hull is up to 64% larger than the part. For *outward*
concavities (a wheel arch, a recess) that is geometry sticking out into space the van
does not occupy, i.e. false alarms. CoACD splits a shape into a few convex parts that
follow it closely.

Known bias (measured on the fixture, 2 cm voxels)
-------------------------------------------------
A point on a face fills the whole voxel it falls in, so the solid, and everything built
from it, is on average half a voxel (~1 cm) larger on every side than the true surface.
Fixture parts come out +23 to +25% by volume against the true boxes, while the P2-T6
gate (parts vs this same solid) reads +0.0%. The gate checks the *decomposition*, not the
voxelization. The bias is conservative (more warnings, not fewer), but it is ~1 cm of
margin nobody asked for, so it is reported in the P2-T6 log. ``voxel_m: 0.01`` halves it
at roughly 8x the compute. That trade-off is the developer's call.

Engines (``decompose.engine``) follow the §8.2 ladder: ``coacd`` (primary), ``vhacd``
(fallback 1, not available as a pip wheel here, see the error it raises), ``obb`` (fallback 2:
one oriented bounding box). **This module never switches engine on its own.** A component
that fails the gate is reported with its numbers; stepping down the ladder is the
developer's decision (§0), recorded in an ADR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import trimesh
from scipy import ndimage

FloatArray = npt.NDArray[np.float64]

#: Unit-cube corner offsets for each outward face, wound counter-clockwise seen from
#: outside, keyed by (axis, direction).
_FACE_QUADS: dict[tuple[int, int], tuple[tuple[int, int, int], ...]] = {
    (0, 1): ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)),
    (0, -1): ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),
    (1, 1): ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)),
    (1, -1): ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),
    (2, 1): ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),
    (2, -1): ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),
}


@dataclass(frozen=True)
class VoxelSolid:
    occupancy: npt.NDArray[np.bool_]  # (X, Y, Z)
    origin: FloatArray  # veh-frame position of voxel (0, 0, 0)'s corner
    voxel_m: float

    @property
    def volume(self) -> float:
        return float(self.occupancy.sum()) * self.voxel_m**3


@dataclass(frozen=True)
class DecomposedComponent:
    name: str
    engine: str
    parts: list[trimesh.Trimesh]
    reference_volume: float
    max_error: float

    @property
    def parts_volume(self) -> float:
        return float(sum(p.volume for p in self.parts))

    @property
    def volume_error_fraction(self) -> float:
        return abs(self.parts_volume - self.reference_volume) / self.reference_volume

    @property
    def passed(self) -> bool:
        """The P2-T6 gate. A failure is reported, never silently patched (§0, §8.2)."""
        return self.volume_error_fraction <= self.max_error

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"{self.name}: {len(self.parts)} part(s) via {self.engine}, volume "
            f"{self.parts_volume:.4f} vs {self.reference_volume:.4f} m^3 "
            f"({self.volume_error_fraction:+.1%}, limit {self.max_error:.0%}) {verdict}"
        )


def solidify(points: npt.ArrayLike, voxel_m: float, close_iterations: int) -> VoxelSolid:
    """Steps 1-2: voxelize, close gaps, fill enclosed interior."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 4:
        raise ValueError(f"need at least 4 points of shape (N, 3); got {pts.shape}")
    pad = close_iterations + 1  # room for closing to work at the edges
    origin = pts.min(axis=0) - pad * voxel_m
    idx = np.floor((pts - origin) / voxel_m).astype(np.int64)
    shape = idx.max(axis=0) + pad + 1
    occ = np.zeros(shape, dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    if close_iterations > 0:
        occ = ndimage.binary_closing(
            occ, structure=np.ones((3, 3, 3), dtype=bool), iterations=close_iterations
        )
    occ = ndimage.binary_fill_holes(occ)
    return VoxelSolid(occ, origin, float(voxel_m))


def boundary_mesh(solid: VoxelSolid) -> trimesh.Trimesh:
    """Step 4: the watertight surface of the occupied voxels, in the veh frame."""
    occ = np.pad(solid.occupancy, 1)
    cells = np.argwhere(occ)
    verts: list[npt.NDArray[np.int64]] = []
    for (axis, direction), quad in _FACE_QUADS.items():
        nb = cells.copy()
        nb[:, axis] += direction
        exposed = cells[~occ[nb[:, 0], nb[:, 1], nb[:, 2]]]
        if len(exposed):
            verts.append((exposed[:, None, :] + np.asarray(quad)[None, :, :]).reshape(-1, 3))
    if not verts:
        raise ValueError("solid is empty")
    v = np.vstack(verts).astype(np.float64)
    n_quads = len(v) // 4
    b = np.arange(n_quads)[:, None] * 4
    faces = np.vstack([np.hstack([b, b + 1, b + 2]), np.hstack([b, b + 2, b + 3])])
    # -1 undoes the padding; the grid index becomes a veh-frame position.
    mesh = trimesh.Trimesh((v - 1.0) * solid.voxel_m + solid.origin, faces, process=False)
    mesh.merge_vertices()
    return mesh


def _coacd_parts(mesh: trimesh.Trimesh, cfg: dict[str, Any]) -> list[trimesh.Trimesh]:
    import coacd  # imported here: only this engine needs the native library

    coacd.set_log_level("error")
    parts = coacd.run_coacd(
        coacd.Mesh(mesh.vertices, mesh.faces),
        threshold=float(cfg["threshold"]),
        max_convex_hull=int(cfg["max_convex_hulls"]),
        preprocess_resolution=int(cfg["preprocess_resolution"]),
        seed=int(cfg["seed"]),
    )
    return [trimesh.Trimesh(v, f) for v, f in parts]


def decompose_component(
    name: str, points: npt.ArrayLike, decompose_cfg: dict[str, Any]
) -> DecomposedComponent:
    """Points of one component -> convex parts, with the P2-T6 volume check."""
    solid = solidify(
        points, float(decompose_cfg["voxel_m"]), int(decompose_cfg["close_iterations"])
    )
    engine = str(decompose_cfg["engine"])
    if engine == "coacd":
        parts = _coacd_parts(boundary_mesh(solid), decompose_cfg)
    elif engine == "obb":
        # §8.2 fallback 2: one oriented box around the solid's surface.
        parts = [boundary_mesh(solid).bounding_box_oriented]
    elif engine == "vhacd":
        raise NotImplementedError(
            "V-HACD (§8.2 fallback 1) has no maintained pip wheel for this environment. "
            "If CoACD fails the gate, propose the step down to the developer with the "
            "failing summaries; installing V-HACD would itself need an ADR."
        )
    else:
        raise ValueError(f"unknown decompose engine {engine!r}; expected coacd | vhacd | obb")
    return DecomposedComponent(
        name=name,
        engine=engine,
        parts=parts,
        reference_volume=solid.volume,
        max_error=float(decompose_cfg["max_volume_error_fraction"]),
    )
