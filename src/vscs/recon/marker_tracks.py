"""Collect marker-corner observations across a reconstruction's images (P1-T6 plumbing).

This is the glue between structure-from-motion and scale recovery. For every registered
image it detects the ArUco markers, undistorts their corners, and files each corner's pixel
position under ``(marker id, corner index)`` together with the image it came from. The
result is one *track* per physical corner - exactly the input ``recon/scale.py``'s
triangulation needs (projection matrices from those images, plus these pixels).

Undistortion is not optional. Detection returns positions in the real, distorted image,
while triangulation uses the pinhole model ``u = K X / Z``. On a phone lens the gap is a
pixel or more near the image edges - which is where markers often sit in a close walk-round
- and at 30-60 px per marker side that is a percent-level error in the scale, and so in
every dimension of the van. Undistorting with the calibrated ``K`` and distortion
coefficients returns each corner to where an ideal pinhole camera would have seen it.

The detection fraction is also reported, because §8.2 steps scale recovery down to
tape-measured dimensions when markers are found in fewer than half the frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from vscs.capture.markers import detect_markers
from vscs.common.log import get_logger
from vscs.recon.colmap_io import Camera, Model

logger = get_logger("recon.marker_tracks")

FloatArray = npt.NDArray[np.float64]


@dataclass
class MarkerTrack:
    """Every observation of one marker: ``corners[k]`` lists ``(image_id, uv)`` for corner k."""

    marker_id: int
    corners: list[list[tuple[int, FloatArray]]] = field(
        default_factory=lambda: [[] for _ in range(4)]
    )

    @property
    def n_views(self) -> int:
        """Images in which all four corners were seen."""
        return min(len(c) for c in self.corners)

    @property
    def image_ids(self) -> list[int]:
        return sorted({i for i, _ in self.corners[0]})


@dataclass
class TrackReport:
    tracks: dict[int, MarkerTrack]
    n_images: int
    n_images_with_markers: int
    n_rejected: int = 0  # detections dropped by the size or corner-angle gate

    @property
    def detection_fraction(self) -> float:
        return self.n_images_with_markers / self.n_images if self.n_images else 0.0

    def usable(self, min_views: int = 2) -> dict[int, MarkerTrack]:
        """Markers seen in enough images to triangulate (two is the minimum)."""
        return {m: t for m, t in self.tracks.items() if t.n_views >= min_views}


def undistort_pixels(uv: npt.ArrayLike, camera: Camera) -> FloatArray:
    """Map distorted pixels to where a pinhole camera with the same ``K`` would see them."""
    pts = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    if not np.any(camera.dist):
        return pts.reshape(-1, 2).copy()
    out = cv2.undistortPoints(pts, camera.K, camera.dist, P=camera.K)
    return out.reshape(-1, 2).astype(np.float64)


def mean_side_px(corners: npt.ArrayLike) -> float:
    """Mean edge length of a detected marker quad, in pixels."""
    c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    return float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))


def min_interior_angle_deg(corners: npt.ArrayLike) -> float:
    """Smallest interior angle of a detected marker quad, in degrees (90 when square-on)."""
    c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    angles = []
    for i in range(4):
        a, b = c[i - 1] - c[i], c[(i + 1) % 4] - c[i]
        cosang = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
        angles.append(float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))))
    return min(angles)


def collect_marker_tracks(
    model: Model,
    image_dir: Path,
    *,
    dictionary: str = "DICT_4X4_50",
    min_side_px: float = 0.0,
    min_corner_angle_deg: float = 0.0,
) -> TrackReport:
    """Detect markers in every registered image of ``model`` and build per-corner tracks.

    Two quality gates drop detections whose corners cannot be trusted, since corner errors
    go straight into the scale (``scale.marker`` in ``configs/recon.yaml``):

    * ``min_side_px`` - too few pixels per marker module to locate a corner;
    * ``min_corner_angle_deg`` - seen at a grazing angle, so the quad's corners are sharply
      acute and poorly located along their edges. This, not size, turned out to be the
      main cause of multi-pixel corner errors.
    """
    image_dir = Path(image_dir)
    tracks: dict[int, MarkerTrack] = {}
    with_markers = 0
    rejected = 0
    for image_id, im in sorted(model.images.items()):
        path = image_dir / im.name
        grey = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if grey is None:
            logger.warning("could not read %s; skipped", path)
            continue
        found = detect_markers(grey, dictionary)
        poor = [
            m
            for m, c in found.items()
            if mean_side_px(c) < min_side_px or min_interior_angle_deg(c) < min_corner_angle_deg
        ]
        rejected += len(poor)
        for m in poor:
            del found[m]
        if not found:
            continue
        with_markers += 1
        camera = model.cameras[im.camera_id]
        for mid, corners in found.items():
            ideal = undistort_pixels(corners, camera)
            track = tracks.setdefault(mid, MarkerTrack(mid))
            for k in range(4):
                track.corners[k].append((image_id, ideal[k]))
    report = TrackReport(tracks, len(model.images), with_markers, rejected)
    if rejected:
        logger.info(
            "dropped %d detection(s) below %.0f px or %.0f deg corner angle",
            rejected,
            min_side_px,
            min_corner_angle_deg,
        )
    logger.info(
        "markers found in %d/%d images (%.0f%%); ids %s",
        with_markers,
        len(model.images),
        100 * report.detection_fraction,
        sorted(tracks),
    )
    return report


def detection_fallback_triggered(report: TrackReport, recon_cfg: dict[str, Any]) -> bool:
    """True when detection falls below ``scale.marker.min_detection_frame_fraction`` (R-02).

    §8.2 then steps scale recovery down to tape-measured dimensions - a switch the developer
    decides on, so this only reports it.
    """
    limit = float(recon_cfg["scale"]["marker"]["min_detection_frame_fraction"])
    return report.detection_fraction < limit


def collect_marker_tracks_from_config(
    model: Model, image_dir: Path, recon_cfg: dict[str, Any]
) -> TrackReport:
    """:func:`collect_marker_tracks` with dictionary and gates from ``configs/recon.yaml``."""
    m = recon_cfg["scale"]["marker"]
    return collect_marker_tracks(
        model,
        image_dir,
        dictionary=str(m["dictionary"]),
        min_side_px=float(m["min_side_px"]),
        min_corner_angle_deg=float(m["min_corner_angle_deg"]),
    )
