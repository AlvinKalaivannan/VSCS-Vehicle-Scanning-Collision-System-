"""Stray-label cleanup after 3D fusion (P2-T5).

Acceptance (CLAUDE.md §6): per-component IoU improves or stays equal versus the P2-T3
fusion output, and the change is logged.

Fusion (P2-T3) labels each point independently, so its errors are *spatially isolated*:
a speck of "mirror" on the door where one mask bled (R-05), a single point that lost a
close vote. Real components are connected surfaces. So:

1. **Cluster each component's points.** Two points with the same label closer than
   ``cluster_eps_m`` are linked; clusters are the connected components of that graph
   (single linkage, via a KD-tree radius query and ``scipy.sparse.csgraph``).
2. **Find strays.** A cluster smaller than ``remove_clusters_smaller_than`` is a stray,
   *except* the component's largest cluster: a mirror seen by only a few frames is small
   but real, and must not be deleted outright.
3. **Reassign strays.** Each stray point takes the majority label of the *kept* points
   within ``reassign_radius_m``, if there are at least ``reassign_min_neighbours`` of
   them (ties go to the lowest label, so the result is deterministic). Otherwise it
   becomes ``NONE_LABEL``.

Cleanup never labels a point fusion left unlabelled. Filling holes would let a component
grow into surface nobody saw, which is a worse error than a gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from vscs.seg.labels import NONE_LABEL

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class CleanupResult:
    labels: IntArray
    stray: npt.NDArray[np.bool_]  # points whose original label was judged a stray
    n_reassigned: int  # strays given another component's label
    n_cleared: int  # strays set to NONE_LABEL
    clusters_per_label: dict[int, int]  # before cleanup, for the log

    def summary(self) -> str:
        return (
            f"{int(self.stray.sum())} stray points: {self.n_reassigned} reassigned, "
            f"{self.n_cleared} cleared"
        )


def _clusters(points: FloatArray, eps: float) -> IntArray:
    """Single-linkage cluster id per point (points closer than ``eps`` are linked)."""
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    pairs = cKDTree(points).query_pairs(eps, output_type="ndarray")
    graph = coo_matrix(
        (np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
    )
    _, ids = connected_components(graph, directed=False)
    return ids.astype(np.int64)


def find_strays(
    points: FloatArray, labels: IntArray, *, eps: float, min_cluster: int
) -> tuple[npt.NDArray[np.bool_], dict[int, int]]:
    """``(stray mask, number of clusters per label)``. See module docstring, steps 1-2."""
    stray = np.zeros(len(labels), dtype=bool)
    n_clusters: dict[int, int] = {}
    for c in np.unique(labels):
        if c == NONE_LABEL:
            continue
        idx = np.flatnonzero(labels == c)
        ids = _clusters(points[idx], eps)
        sizes = np.bincount(ids)
        n_clusters[int(c)] = len(sizes)
        largest = int(np.argmax(sizes))  # lowest id on a tie: deterministic
        small = sizes < min_cluster
        small[largest] = False
        stray[idx[small[ids]]] = True
    return stray, n_clusters


def cleanup_labels(
    points: npt.ArrayLike, labels: npt.ArrayLike, cleanup_cfg: dict[str, Any]
) -> CleanupResult:
    """Steps 1-3. ``cleanup_cfg`` is the ``cleanup`` block of ``configs/seg.yaml``."""
    pts = np.asarray(points, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.int64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) != len(lab):
        raise ValueError(
            f"points must be (N, 3) matching labels (N,); got {pts.shape}, {lab.shape}"
        )

    stray, n_clusters = find_strays(
        pts,
        lab,
        eps=float(cleanup_cfg["cluster_eps_m"]),
        min_cluster=int(cleanup_cfg["remove_clusters_smaller_than"]),
    )
    out = lab.copy()
    out[stray] = NONE_LABEL

    keep = np.flatnonzero((~stray) & (lab != NONE_LABEL))
    stray_idx = np.flatnonzero(stray)
    n_reassigned = 0
    if len(keep) and len(stray_idx):
        tree = cKDTree(pts[keep])
        min_nb = int(cleanup_cfg["reassign_min_neighbours"])
        neighbours = tree.query_ball_point(
            pts[stray_idx], r=float(cleanup_cfg["reassign_radius_m"])
        )
        for i, nb in zip(stray_idx, neighbours, strict=True):
            if len(nb) < min_nb:
                continue
            votes = np.bincount(lab[keep[nb]])
            out[i] = int(np.argmax(votes))
            n_reassigned += 1

    return CleanupResult(
        labels=out,
        stray=stray,
        n_reassigned=n_reassigned,
        n_cleared=int(stray.sum()) - n_reassigned,
        clusters_per_label=n_clusters,
    )
