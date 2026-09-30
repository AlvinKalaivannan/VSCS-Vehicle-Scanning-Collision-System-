"""P2-T6 convex decomposition, on fixture components and a concave L-shaped solid."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from fixtures.synthetic import load_scene
from vscs.common.config import load_config
from vscs.model.decompose import VoxelSolid, boundary_mesh, decompose_component, solidify

SCENE = load_scene()
CFG = load_config("model")["decompose"]
H = float(CFG["voxel_m"])


def _surface_points(mesh: trimesh.Trimesh, per_m2: float = 20000, seed: int = 7) -> np.ndarray:
    """Dense surface samples (~7 mm apart), like a dense reconstruction of the part."""
    n = int(mesh.area * per_m2)
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts)


@pytest.fixture(scope="module")
def l_shape():
    """An L-shaped plate, 0.4 x 0.4 m with 0.2 m arms, 0.1 m thick: 0.012 m^3."""
    occ = np.zeros((4, 4, 1), dtype=bool)
    occ[:, :2, :] = True
    occ[:2, :, :] = True
    mesh = boundary_mesh(VoxelSolid(occ, np.zeros(3), 0.1))
    return mesh, _surface_points(mesh)


@pytest.fixture(scope="module")
def l_coacd(l_shape):
    """CoACD on the L, run once (it takes several seconds) and shared."""
    _, pts = l_shape
    return decompose_component("l_part", pts, CFG)


# --------------------------------------------------------------------------- #
# Points -> solid -> surface                                                   #
# --------------------------------------------------------------------------- #
def test_boundary_mesh_of_known_voxels_is_watertight_and_exact():
    occ = np.zeros((3, 3, 3), dtype=bool)
    occ[0, :, :] = True
    occ[:, 0, 0] = True
    mesh = boundary_mesh(VoxelSolid(occ, np.array([1.0, 2.0, 3.0]), 0.5))
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(occ.sum() * 0.5**3)
    assert mesh.bounds[0] == pytest.approx([1.0, 2.0, 3.0])


def test_a_closed_shell_of_points_becomes_a_filled_solid():
    """Points only on the corner box's surface; the interior is filled in."""
    box = SCENE.component("rear_right_bumper_corner")  # 0.15 x 0.30 x 0.35 m
    pts = box.sample_surface_points(20000, seed=1)
    solid = solidify(pts, H, int(CFG["close_iterations"]))
    # Surface voxels stick out by up to one voxel on each side: bounded, conservative.
    lo = box.volume
    hi = float(np.prod(box.extent + 2 * H))
    assert lo <= solid.volume <= hi
    assert boundary_mesh(solid).is_watertight


def test_an_open_sheet_stays_thin():
    """What a scan actually sees: a single face. It must not balloon into a slab."""
    rng = np.random.default_rng(0)
    sheet = np.column_stack([rng.random(20000) * 1.0, rng.random(20000) * 0.3, np.zeros(20000)])
    solid = solidify(sheet, H, int(CFG["close_iterations"]))
    thickness = solid.volume / (1.0 * 0.3)
    assert thickness <= 3 * H


def test_too_few_points_is_an_error():
    with pytest.raises(ValueError, match="at least 4 points"):
        solidify(np.zeros((2, 3)), H, 1)


# --------------------------------------------------------------------------- #
# Decomposition and the P2-T6 gate                                             #
# --------------------------------------------------------------------------- #
def test_one_convex_hull_overfills_a_concave_part(l_shape):
    """Why P2-T6 exists: the hull of this L is ~17% too large, over the 15% gate."""
    mesh, _ = l_shape
    assert mesh.convex_hull.volume / mesh.volume - 1 > CFG["max_volume_error_fraction"]


def test_coacd_follows_the_concave_part_and_passes_the_gate(l_coacd):
    res = l_coacd
    assert res.engine == "coacd" and len(res.parts) >= 2
    assert all(p.is_convex for p in res.parts)
    assert res.passed, res.summary()


def test_a_box_component_decomposes_to_few_parts_and_passes():
    box = SCENE.component("right_mirror")
    res = decompose_component("right_mirror", box.sample_surface_points(20000, seed=2), CFG)
    assert res.passed, res.summary()
    assert len(res.parts) <= 2
    # The parts sit where the mirror is (within one voxel), in the veh frame.
    lo, hi = (
        np.min([p.bounds[0] for p in res.parts], 0),
        np.max([p.bounds[1] for p in res.parts], 0),
    )
    assert np.all(lo >= box.lo - 1.5 * H) and np.all(hi <= box.hi + 1.5 * H)


def test_voxelization_bias_is_about_half_a_voxel_per_side():
    """The known, conservative bias recorded in decompose.py: bounded, and outward only."""
    box = SCENE.component("right_mirror")
    res = decompose_component("right_mirror", box.sample_surface_points(20000, seed=2), CFG)
    grown = res.parts_volume / box.volume - 1
    assert 0.0 < grown <= float(np.prod(box.extent + 2 * H) / box.volume - 1)


def test_obb_fallback_is_one_box(l_shape):
    _, pts = l_shape
    res = decompose_component("l_part", pts, {**CFG, "engine": "obb"})
    assert len(res.parts) == 1 and res.parts[0].is_convex
    assert not res.passed  # a box around an L overfills it - that is what the gate catches


def test_a_failing_component_is_reported_not_switched(l_shape):
    """§0: stepping down the fallback ladder is the developer's decision."""
    _, pts = l_shape
    res = decompose_component("l_part", pts, {**CFG, "engine": "obb"})
    assert res.engine == "obb"
    assert "FAIL" in res.summary() and "limit 15%" in res.summary()


def test_vhacd_explains_itself_and_unknown_engine_is_rejected(l_shape):
    _, pts = l_shape
    with pytest.raises(NotImplementedError, match="developer"):
        decompose_component("x", pts, {**CFG, "engine": "vhacd"})
    with pytest.raises(ValueError, match="unknown decompose engine"):
        decompose_component("x", pts, {**CFG, "engine": "magic"})


def test_decomposition_is_deterministic(l_shape, l_coacd):
    _, pts = l_shape
    a = l_coacd
    b = decompose_component("l_part", pts, CFG)
    assert len(a.parts) == len(b.parts)
    assert a.parts_volume == pytest.approx(b.parts_volume, rel=1e-9)
