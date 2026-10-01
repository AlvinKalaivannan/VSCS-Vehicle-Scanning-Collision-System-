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

**The correction: ``decompose.surface: voxel_centres``** (the default since ADR 0014;
``voxel_faces`` is the behaviour above). The grid is offset so that flat faces' points
sit on voxel *centres*;
the true surface is therefore half a voxel inside the boundary faces. ``boundary_mesh(...,
inset=True)`` moves every boundary vertex half a voxel toward the solid on each axis, by
the sign of the vote of the eight voxels around it (occupied ones on the + side count +1,
on the - side -1). For an orthogonal voxel surface that is an exact inward offset of every
face, with no seams between later convex parts (insetting each convex part instead would
open a one-voxel groove at every cut). The reference volume is then the inset surface's
volume. A component with no voxel deeper than ``inset_min_depth_voxels`` (a scanned
*sheet*) would collapse, so it keeps the face surface and says so.

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
from scipy.spatial import cKDTree

from vscs.common.log import get_logger

logger = get_logger("model.decompose")

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
    point_spacing_m: float = float("nan")
    surface: str = "voxel_faces"  # or "voxel_centres" (the half-voxel inset prototype)

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
            f"({self.volume_error_fraction:+.1%}, limit {self.max_error:.0%}) {verdict}; "
            f"point spacing {self.point_spacing_m * 100:.1f} cm"
        )


def median_point_spacing(points: npt.ArrayLike, max_queries: int = 5000) -> float:
    """Median distance from a point to its nearest neighbour.

    At most ``max_queries`` points are *queried*, but always against the full cloud:
    thinning the cloud itself would inflate every nearest-neighbour distance.
    """
    pts = np.asarray(points, dtype=np.float64)
    queries = pts[:: max(1, int(np.ceil(len(pts) / max_queries)))]
    d, _ = cKDTree(pts).query(queries, k=2)
    return float(np.median(d[:, 1]))


def solidify(points: npt.ArrayLike, voxel_m: float, close_iterations: int) -> VoxelSolid:
    """Steps 1-2: voxelize, close gaps, fill enclosed interior."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 4:
        raise ValueError(f"need at least 4 points of shape (N, 3); got {pts.shape}")
    pad = close_iterations + 1  # room for closing to work at the edges
    # Half-voxel offset: a flat face through the extreme points then lies on voxel
    # *centres*. On a boundary instead, rounding scatters its points between two voxel
    # layers, leaving both half-empty and full of holes (found on the fixture's
    # axis-aligned faces; a van's flat sides are near axis-aligned in veh too).
    origin = pts.min(axis=0) - (pad + 0.5) * voxel_m
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


def boundary_mesh(solid: VoxelSolid, *, inset: bool = False) -> trimesh.Trimesh:
    """Step 4: the watertight surface of the occupied voxels, in the veh frame.

    ``inset=True`` puts it through the outermost voxel centres instead of their faces
    (module docstring, "The correction"; ADR 0014).
    """
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
    corners = np.vstack(verts)
    v = corners.astype(np.float64)
    if inset:
        # Corner g touches voxels g + d, d in {-1, 0}^3. A voxel with d = 0 on an axis lies on
        # that axis's + side of the corner, d = -1 on its - side.
        vote = np.zeros(v.shape)
        for d in np.array(np.meshgrid([-1, 0], [-1, 0], [-1, 0], indexing="ij")).reshape(3, -1).T:
            g = corners + d
            vote += occ[g[:, 0], g[:, 1], g[:, 2]][:, None] * np.where(d == 0, 1.0, -1.0)
        v += 0.5 * np.sign(vote)
    n_quads = len(v) // 4
    b = np.arange(n_quads)[:, None] * 4
    faces = np.vstack([np.hstack([b, b + 1, b + 2]), np.hstack([b, b + 2, b + 3])])
    # -1 undoes the padding; the grid index becomes a veh-frame position.
    mesh = trimesh.Trimesh((v - 1.0) * solid.voxel_m + solid.origin, faces, process=False)
    mesh.merge_vertices()
    if inset:
        mesh.update_faces(mesh.nondegenerate_faces())  # thin flanges collapse to zero area
        mesh.remove_unreferenced_vertices()
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
    """Points of one component -> convex parts, with the P2-T6 volume check.

    Filling a closed surface is all-or-nothing: one hole in the voxel shell and the
    interior stays hollow, so the reference volume collapses to the shell's. Holes appear
    when points are too sparse for every voxel face to catch a few (see
    ``max_spacing_fraction_of_voxel`` in ``model.yaml``). That is logged as a warning
    rather than guessed around.
    """
    voxel_m = float(decompose_cfg["voxel_m"])
    close_iterations = int(decompose_cfg["close_iterations"])
    spacing = median_point_spacing(points)
    if spacing > float(decompose_cfg["max_spacing_fraction_of_voxel"]) * voxel_m:
        logger.warning(
            "%s: median point spacing %.1f cm is coarse for %.1f cm voxels; closed surfaces "
            "may not fill, and the reference volume will then be the shell's",
            name,
            spacing * 100,
            voxel_m * 100,
        )
    solid = solidify(points, voxel_m, close_iterations)
    surface = str(decompose_cfg.get("surface", "voxel_faces"))
    if surface not in ("voxel_faces", "voxel_centres"):
        raise ValueError(f"unknown decompose.surface {surface!r}")
    if surface == "voxel_centres":
        depth = float(ndimage.distance_transform_edt(solid.occupancy).max())
        if depth < float(decompose_cfg["inset_min_depth_voxels"]):
            logger.warning(
                "%s: no voxel deeper than %.1f voxels (a scanned sheet); the inset would "
                "collapse it, so the face surface is kept",
                name,
                depth,
            )
            surface = "voxel_faces"
    mesh = boundary_mesh(solid, inset=surface == "voxel_centres")
    reference = solid.volume if surface == "voxel_faces" else float(mesh.volume)
    engine = str(decompose_cfg["engine"])
    if engine == "coacd":
        parts = _coacd_parts(mesh, decompose_cfg)
    elif engine == "obb":
        # §8.2 fallback 2: one oriented box around the solid's surface.
        parts = [mesh.bounding_box_oriented]
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
        reference_volume=reference,
        max_error=float(decompose_cfg["max_volume_error_fraction"]),
        point_spacing_m=spacing,
        surface=surface,
    )
