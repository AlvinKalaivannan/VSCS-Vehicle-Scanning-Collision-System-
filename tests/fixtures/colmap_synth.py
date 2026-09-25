"""Synthetic COLMAP models built from the fixture world, with exactly known poses.

COLMAP is not installed yet (ADR 0003), and there is no real scan. This builds what a
perfect reconstruction of the fixture scene would contain - the 20-pose camera loop and the
labelled component point cloud, projected through the fixture camera - so the reader, the
wrapper's model selection and the P1-T5 metrics can be tested against ground truth.
"""

from __future__ import annotations

import numpy as np

from fixtures.synthetic import load_scene
from vscs.common.frames import invert, project, transform_points
from vscs.recon.colmap_io import Camera, Image, Model, Point3D


def build_fixture_model(
    *,
    n_images: int | None = None,
    points_per_component: int = 30,
    point_error_px: float = 0.4,
    seed: int = 20260924,
) -> Model:
    """A COLMAP model of the fixture scene. World frame = fixture veh frame, metric.

    ``point_error_px`` is written as every point's reprojection error, so the model's mean
    error is known exactly.
    """
    scene = load_scene()
    w, h = scene.image_size
    K = scene.K
    cam = Camera(1, "PINHOLE", w, h, np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]))

    poses = scene.camera_trajectory()
    if n_images is not None:
        poses = poses[:n_images]
    pts, _ = scene.labelled_point_cloud(points_per_component, seed=seed)

    images: dict[int, Image] = {}
    obs: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(pts))}
    for idx, pose in enumerate(poses, start=1):
        p_cam = transform_points(invert(pose.T_world_cam), pts)
        uv, valid = project(K, p_cam, image_size=(w, h))
        keep = np.nonzero(valid)[0]
        xys = uv[keep]
        pids = keep.astype(np.int64) + 1
        for j, pid in enumerate(keep):
            obs[int(pid)].append((idx, j))
        images[idx] = Image.from_T_world_cam(
            idx, pose.T_world_cam, 1, f"{idx:06d}.png", xys=xys, point3D_ids=pids
        )

    points: dict[int, Point3D] = {}
    for i, xyz in enumerate(pts):
        track = obs[i]
        if len(track) < 2:
            continue  # COLMAP only keeps points triangulated from two or more views
        arr = np.array(track, dtype=np.int64)
        points[i + 1] = Point3D(
            i + 1, xyz, np.array([128, 128, 128]), point_error_px, arr[:, 0], arr[:, 1]
        )
    # An observation of a point that was never triangulated carries id -1, as in COLMAP.
    for im in images.values():
        im.point3D_ids = np.where(np.isin(im.point3D_ids, list(points)), im.point3D_ids, -1).astype(
            np.int64
        )
    return Model(cameras={1: cam}, images=images, points=points)
