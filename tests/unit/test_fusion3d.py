"""P2-T3 target: 3D label fusion on the synthetic fixture.

These tests are the contract for ``src/vscs/seg/fusion3d.py`` (developer's draft). They
fail with NotImplementedError until it exists.

The input is exact: ``fixtures/render.py`` ray-casts the fixture's boxes into perfect
component masks and depth maps. Ground truth for "which points can be labelled at all"
is computed independently: a ray is cast *through the point itself*, not through a
pixel centre, so it does not share the pixel-rounding logic under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from fixtures.render import NONE_LABEL, _occluders, _ray_box_t, component_names, render_views
from fixtures.synthetic import load_scene
from vscs.common.config import load_config
from vscs.common.frames import invert, look_at_T_world_cam, project, transform_points
from vscs.eval.metrics import mean_iou, per_component_iou
from vscs.seg import fusion3d as F

SCENE = load_scene()
NAMES = component_names(SCENE)
C = len(NAMES)
CFG = load_config("seg")["fusion3d"]
TOL = float(CFG["occlusion_depth_tolerance_m"])
MIN_OBS = int(CFG["min_observations_per_point"])

#: Reached by a straightforward implementation of the docstring (checked when this
#: contract was written); comfortably above the 0.70 acceptance gate, because the
#: masks and depth here are perfect.
CLEAN_MEAN_IOU = 0.95
NOISY_MEAN_IOU = 0.90


@pytest.fixture(scope="module")
def views():
    return [F.PosedView(K, T, labels, depth) for T, K, labels, depth in render_views(SCENE, 0.5)]


@pytest.fixture(scope="module")
def cloud():
    pts, names = SCENE.labelled_point_cloud(points_per_component=300)
    return pts, np.array([NAMES.index(n) for n in names])


def _exact_view_count(points, views) -> np.ndarray:
    """Views in which each point is the first surface on the ray through it (no pixels)."""
    boxes = _occluders(SCENE) + [SCENE.components[n] for n in NAMES]
    counts = np.zeros(len(points), dtype=int)
    for v in views:
        T_wc = invert(v.T_cam_world)
        origin = T_wc[:3, 3]
        dirs = points - origin
        z_axis = T_wc[:3, 2]
        in_front = dirs @ z_axis > 0
        # Scale each ray so the point sits at t = 1; the point is visible if nothing is hit
        # meaningfully before t = 1.
        first = np.min(np.stack([_ray_box_t(origin, dirs, b) for b in boxes]), axis=0)
        # The point's own face may be entered at t = 1 exactly (first == 1) or the ray may
        # graze it; anything hit clearly earlier occludes it.
        visible = in_front & (first >= 1.0 - 1e-6)
        # ...and it must land inside the image.
        w, h = v.image_size
        _, ok = project(v.K, transform_points(v.T_cam_world, points), (w, h))
        counts += visible & ok
    return counts


@pytest.fixture(scope="module")
def truth(cloud, views):
    """``(labellable (N,) bool, seen (N,))``.

    A point is *labellable* if it is exactly visible in at least MIN_OBS views. That is
    the set fusion must get right. A point just behind a visible face of its own
    component (closer than the depth tolerance) may also be labelled, and correctly so:
    a hand-labelled scan would call it the same component. So those points are judged
    only on never getting a *wrong* label, not on being left out.
    """
    pts, _ = cloud
    seen = _exact_view_count(pts, views)
    return seen >= MIN_OBS, seen


def _depth_inside_shell(points) -> np.ndarray:
    """How far each point is inside the body shell (<= 0 for points on or outside it)."""
    lo, hi = SCENE.vehicle.lo, SCENE.vehicle.hi
    return np.min(np.minimum(points - lo, hi - points), axis=1)


# --------------------------------------------------------------------------- #
# Step 1-2: one frame                                                          #
# --------------------------------------------------------------------------- #
def _behind_view():
    """A camera 3 m behind the van, looking forward along +x at bumper height."""
    from fixtures.render import render, scaled_intrinsics

    eye = np.array([-4.0, 0.0, 0.6])
    T_wc = look_at_T_world_cam(eye, eye + np.array([1.0, 0.0, 0.0]))
    K, size = scaled_intrinsics(SCENE, 0.5)
    labels, depth = render(SCENE, T_wc, K, size)
    return F.PosedView(K, invert(T_wc), labels, depth)


def test_observe_labels_a_visible_point():
    view = _behind_view()
    p = np.array([[-1.0, 0.0, 0.6]])  # centre of the rear bumper's back face
    lab, vis = F.observe(p, view, TOL)
    assert vis.tolist() == [True]
    assert NAMES[lab[0]] == "rear_bumper"


def test_observe_rejects_an_occluded_point():
    """The front bumper projects onto the rear bumper from behind, but is 5 m further."""
    view = _behind_view()
    p = np.array([[4.0, 0.0, 0.6]])
    lab, vis = F.observe(p, view, TOL)
    assert vis.tolist() == [False] and lab.tolist() == [NONE_LABEL]


def test_without_the_depth_test_the_occluded_point_gets_the_wrong_label():
    """Why step 2 exists: an infinite tolerance is naive projection."""
    view = _behind_view()
    lab, vis = F.observe(np.array([[4.0, 0.0, 0.6]]), view, np.inf)
    assert vis.tolist() == [True]
    assert NAMES[lab[0]] == "rear_bumper"  # wrong: it is the front bumper


def test_observe_rejects_points_behind_the_camera_and_outside_the_image():
    view = _behind_view()
    pts = np.array([[-6.0, 0.0, 0.6], [-1.0, 50.0, 0.6]])
    lab, vis = F.observe(pts, view, TOL)
    assert not vis.any() and (lab == NONE_LABEL).all()


# --------------------------------------------------------------------------- #
# Step 4: decision rules on hand-written votes                                 #
# --------------------------------------------------------------------------- #
def test_decide_rules():
    # columns: comp0, comp1, none
    votes = np.array(
        [
            [5, 0, 0],  # clear winner                          -> 0
            [2, 0, 0],  # too few observations (2 < 3)          -> none
            [1, 1, 4],  # "none" wins                           -> none
            [2, 2, 1],  # tie -> lowest index, 2/5 < 0.5        -> none
            [3, 2, 0],  # plurality 3/5 >= 0.5                  -> 0
            [0, 0, 0],  # never seen                            -> none, confidence 0
            [1, 3, 0],  #                                       -> 1
        ]
    )
    labels, conf, n_obs = F.decide(votes, min_observations=3, min_vote_fraction=0.5)
    assert labels.tolist() == [0, NONE_LABEL, NONE_LABEL, NONE_LABEL, 0, NONE_LABEL, 1]
    assert n_obs.tolist() == [5, 2, 6, 5, 5, 0, 4]
    assert conf == pytest.approx([1.0, 1.0, 4 / 6, 0.4, 0.6, 0.0, 0.75])


def test_accumulate_votes_counts_each_visible_frame_once(views, cloud):
    pts, _ = cloud
    votes = F.accumulate_votes(pts, views, C, TOL)
    assert votes.shape == (len(pts), C + 1)
    assert votes.dtype.kind == "i"
    assert votes.sum(axis=1).max() <= len(views)


# --------------------------------------------------------------------------- #
# End to end on the fixture                                                    #
# --------------------------------------------------------------------------- #
def _iou_on_labellable(res_labels, labels, labellable) -> float:
    ious = per_component_iou(res_labels[labellable], labels[labellable], range(C))
    return mean_iou(ious)


def test_fusion_recovers_the_components(views, cloud, truth):
    pts, labels = cloud
    labellable, _ = truth
    res = F.fuse_labels(pts, views, C, CFG)
    assert _iou_on_labellable(res.labels, labels, labellable) >= CLEAN_MEAN_IOU


def test_a_labelled_surface_point_never_gets_the_wrong_component(views, cloud, truth):
    """Judged on exterior points (seen at least once), which is all a real scan contains.

    The fixture also samples faces no camera can see, e.g. the mirror's inner face, which
    lies flat against the sliding door. Such a point sits within the depth tolerance of
    the door's visible surface, so it takes the door's label. A real dense cloud has no
    points there, so they are not held against fusion. The real-world version of this
    is where components meet, which is exactly where R-05's bleed lives.
    """
    pts, labels = cloud
    _, seen = truth
    res = F.fuse_labels(pts, views, C, CFG)
    got = (res.labels != NONE_LABEL) & (seen > 0)
    assert got.sum() > 1000
    assert np.mean(res.labels[got] == labels[got]) >= 0.99


def test_points_buried_in_the_body_stay_unlabelled(views, cloud, truth):
    """Faces deep inside the shell (the underbody, inner faces) must not be guessed."""
    pts, _ = cloud
    _, seen = truth
    res = F.fuse_labels(pts, views, C, CFG)
    buried = (seen == 0) & (_depth_inside_shell(pts) > 2 * TOL)
    assert buried.sum() > 100  # the fixture really has buried faces
    assert np.all(res.labels[buried] == NONE_LABEL)


def test_underbody_is_never_labelled_from_a_standing_height_scan(views, cloud):
    """P2-T3 cannot see under the van; that is a capture gap, not something to invent."""
    pts, _ = cloud
    res = F.fuse_labels(pts, views, C, CFG)
    assert not np.any(res.labels == NAMES.index("underbody"))


def test_voting_survives_noisy_masks(views, cloud, truth):
    """R-05: 20% of component pixels relabelled at random; the vote still holds."""
    rng = np.random.default_rng(20260929)
    noisy = []
    for v in views:
        lab = v.labels.copy()
        comp = lab != NONE_LABEL
        flip = comp & (rng.random(lab.shape) < 0.20)
        lab[flip] = rng.integers(-1, C, size=int(flip.sum()))
        noisy.append(F.PosedView(v.K, v.T_cam_world, lab, v.depth))
    pts, labels = cloud
    labellable, _ = truth
    res = F.fuse_labels(pts, noisy, C, CFG)
    assert _iou_on_labellable(res.labels, labels, labellable) >= NOISY_MEAN_IOU


def test_fusion_is_deterministic(views, cloud):
    pts, _ = cloud
    a = F.fuse_labels(pts, views, C, CFG)
    b = F.fuse_labels(pts, views, C, CFG)
    assert np.array_equal(a.labels, b.labels) and np.array_equal(a.votes, b.votes)
