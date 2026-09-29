"""P2-T5 stray-label cleanup on the synthetic fixture.

The acceptance rule is "IoU improves or stays equal versus fusion", so every test that
injects damage also checks IoU never goes down.
"""

from __future__ import annotations

import numpy as np
import pytest

from fixtures.render import component_names
from fixtures.synthetic import load_scene
from vscs.common.config import load_config
from vscs.eval.metrics import mean_iou, per_component_iou
from vscs.seg.cleanup import cleanup_labels, find_strays
from vscs.seg.labels import NONE_LABEL

SCENE = load_scene()
NAMES = component_names(SCENE)
C = len(NAMES)
CFG = load_config("seg")["cleanup"]

#: Surface points per square metre: ~2 cm spacing, well inside cluster_eps_m (5 cm),
#: as a dense reconstruction would be.
DENSITY_PER_M2 = 2500


@pytest.fixture(scope="module")
def cloud():
    pts, labels = [], []
    for idx, name in enumerate(NAMES):
        box = SCENE.components[name]
        dx, dy, dz = box.extent
        area = 2 * (dx * dy + dy * dz + dx * dz)
        n = max(200, int(area * DENSITY_PER_M2))
        pts.append(box.sample_surface_points(n, seed=1000 + idx))
        labels.append(np.full(n, idx))
    return np.vstack(pts), np.concatenate(labels)


def _miou(pred, truth) -> float:
    return mean_iou(per_component_iou(pred, truth, range(C)))


def test_clean_labels_are_left_exactly_alone(cloud):
    pts, truth = cloud
    res = cleanup_labels(pts, truth, CFG)
    assert np.array_equal(res.labels, truth)
    assert not res.stray.any()


def test_salt_noise_is_repaired(cloud):
    """5% of points flipped to a random *other* component, as isolated fusion errors."""
    pts, truth = cloud
    rng = np.random.default_rng(20260930)
    noisy = truth.copy()
    flip = rng.random(len(truth)) < 0.05
    noisy[flip] = (truth[flip] + rng.integers(1, C, size=int(flip.sum()))) % C
    before = _miou(noisy, truth)
    res = cleanup_labels(pts, noisy, CFG)
    after = _miou(res.labels, truth)
    assert after >= before
    assert after >= 0.99, (before, after)


def test_a_bleed_patch_is_returned_to_its_surface(cloud):
    """R-05: a 4 cm blob of 'right_mirror' painted onto the middle of the sliding door."""
    pts, truth = cloud
    door, mirror = NAMES.index("sliding_door_right"), NAMES.index("right_mirror")
    centre = np.array([1.8, -1.0, 1.2])  # door's outer face, far from the mirror
    patch = (truth == door) & (np.linalg.norm(pts - centre, axis=1) < 0.04)
    assert 5 <= patch.sum() < CFG["remove_clusters_smaller_than"]
    noisy = truth.copy()
    noisy[patch] = mirror
    res = cleanup_labels(pts, noisy, CFG)
    assert np.all(res.labels[patch] == door)
    assert _miou(res.labels, truth) >= _miou(noisy, truth)


def test_a_components_largest_cluster_is_never_removed():
    """A tiny but real component (only 10 points) survives, although below the size limit."""
    rng = np.random.default_rng(1)
    pts = np.vstack([rng.random((500, 3)) * 0.1, [5.0, 5.0, 5.0] + rng.random((10, 3)) * 0.02])
    labels = np.array([0] * 500 + [1] * 10)
    res = cleanup_labels(pts, labels, CFG)
    assert np.all(res.labels[500:] == 1)


def test_a_stray_with_no_trusted_neighbours_is_cleared_not_guessed():
    rng = np.random.default_rng(2)
    main = rng.random((500, 3)) * 0.1
    far_stray = np.array([[9.0, 9.0, 9.0]])
    also_1 = rng.random((300, 3)) * 0.1 + [0, 2.0, 0]  # component 1's real cluster
    pts = np.vstack([main, also_1, far_stray])
    labels = np.array([0] * 500 + [1] * 300 + [1])
    res = cleanup_labels(pts, labels, CFG)
    assert res.stray[-1] and res.labels[-1] == NONE_LABEL
    assert res.n_cleared == 1 and res.n_reassigned == 0


def test_unlabelled_points_are_never_filled(cloud):
    pts, truth = cloud
    holes = truth.copy()
    holes[::7] = NONE_LABEL
    res = cleanup_labels(pts, holes, CFG)
    assert np.all(res.labels[holes == NONE_LABEL] == NONE_LABEL)


def test_cluster_count_is_reported_for_the_log(cloud):
    pts, truth = cloud
    _, n_clusters = find_strays(
        pts, truth, eps=CFG["cluster_eps_m"], min_cluster=CFG["remove_clusters_smaller_than"]
    )
    assert set(n_clusters) == set(range(C))
    assert all(n == 1 for n in n_clusters.values()), n_clusters  # each part is one surface


def test_cleanup_is_deterministic(cloud):
    pts, truth = cloud
    rng = np.random.default_rng(3)
    noisy = truth.copy()
    flip = rng.random(len(truth)) < 0.05
    noisy[flip] = rng.integers(0, C, size=int(flip.sum()))
    a = cleanup_labels(pts, noisy, CFG)
    b = cleanup_labels(pts, noisy, CFG)
    assert np.array_equal(a.labels, b.labels)
    assert "stray points" in a.summary()


def test_shape_errors_are_reported():
    with pytest.raises(ValueError, match="points must be"):
        cleanup_labels(np.zeros((3, 2)), np.zeros(3), CFG)
