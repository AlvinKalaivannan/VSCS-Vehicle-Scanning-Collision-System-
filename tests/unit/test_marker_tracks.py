"""Marker detection and per-corner track tests (P1-T6 plumbing).

Markers are rendered into every fixture camera view at known 3D positions, so each detected
corner has an exact projection to compare against.

What these tests encode was settled by measurement (devlog 2026-09-25):

* **The renderer is anti-aliased** (4x supersampling, area-averaged down), because a real
  lens blurs edges and a naive bilinear warp does not.
* **Marker size matches the real scan in pixels.** The fixture camera is a low-resolution
  stand-in (1280 px, f = 900 px); a real 150 mm marker at 4K (f ~ 2740 px) appears as large
  as a 0.457 m marker here.
* **Grazing views, not small markers, cause the large corner errors.** A marker lying flat on
  the ground, seen from a standing walk-round, projects to a thin trapezoid with sharply
  acute corners that are poorly located along their edges - up to 5 px off. The
  corner-angle gate removes those views, and with them most views of flat markers.
* **Boards propped at 45 deg facing the walking path fare better**: every board position
  tested kept 3-4 views through the gates (enough to triangulate), with worst corner errors
  of 0.5-1.2 px. Flat markers left most positions with at most one. Boards are used for the
  track tests; flat ground markers test the gates and pin down the finding.

The renderer culls back faces - a camera behind a marker sees nothing printed there. An
earlier version did not, and counted views of boards seen *through the van from behind*,
which inflated the board result; that is corrected. Occlusion by the van body itself is
still not modelled.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from fixtures.colmap_synth import build_fixture_model
from fixtures.synthetic import load_scene
from vscs.capture.markers import _dictionary, detect_markers
from vscs.common.config import load_config
from vscs.common.frames import project, transform_points
from vscs.recon import marker_tracks as MT
from vscs.recon.colmap_io import Camera

RECON = load_config("recon")
SIDE = 0.15 * 2740 / 900  # a real 150 mm marker at 4K, in this camera's pixels
VAN_CENTRE = np.array([1.5, 0.0])
BOARD_SPOTS = {1: (1.5, 2.6), 3: (1.5, -2.6), 4: (-2.2, 1.5)}  # 45 deg boards, facing out
FLAT_SPOTS = {0: (-2.5, -1.8), 1: (1.5, 2.6), 2: (5.2, -1.6)}  # lying on the ground
TEX = 240
SUPERSAMPLE = 4


# --------------------------------------------------------------------------- #
# Synthetic markers and rendering                                              #
# --------------------------------------------------------------------------- #
def flat_corners(cx, cy, side=SIDE):
    """ArUco order (TL, TR, BR, BL of the marker), lying flat on the ground."""
    h = side / 2
    return np.array(
        [[cx - h, cy + h, 0.0], [cx + h, cy + h, 0.0], [cx + h, cy - h, 0.0], [cx - h, cy - h, 0.0]]
    )


def board_corners(cx, cy, tilt_deg=45.0, height_m=0.35, side=SIDE):
    """A marker on a board propped at ``tilt_deg``, facing out from the van and up."""
    out = np.array([cx, cy]) - VAN_CENTRE
    out = out / np.linalg.norm(out)
    z = np.array([0.0, 0.0, 1.0])
    a = np.radians(tilt_deg)
    n = np.cos(a) * np.array([out[0], out[1], 0.0]) + np.sin(a) * z
    e1 = np.cross(z, n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    # Mirror across e1 so the READABLE face points along n (outward and up). Without this
    # the marker is printed on the board's back, and only reads from behind.
    e1 = -e1
    c, h = np.array([cx, cy, height_m]), side / 2
    return np.array([c + h * (-e1 + e2), c + h * (e1 + e2), c + h * (e1 - e2), c + h * (-e1 - e2)])


def _readable_from(C, camera_centre):
    """True if the camera is on the marker's printed side.

    For corners in ArUco order the printed face is opposite to (C1 - C0) x (C3 - C0): a flat
    marker in that order reads correctly from above, where that cross product points down.
    """
    back_normal = np.cross(C[1] - C[0], C[3] - C[0])
    return float((np.asarray(camera_centre) - C.mean(axis=0)) @ back_normal) < 0.0


def _render_raw(K, size, T_cam_world, quads):
    w, h = size
    img = np.full((h, w), 255, np.uint8)
    src = np.array(
        [[-0.5, -0.5], [TEX - 0.5, -0.5], [TEX - 0.5, TEX - 0.5], [-0.5, TEX - 0.5]], np.float32
    )
    R, t = T_cam_world[:3, :3], T_cam_world[:3, 3]
    camera_centre = -R.T @ t
    for mid, C in quads.items():
        if not _readable_from(C, camera_centre):
            continue  # back of the marker: nothing printed there to see
        uv, ok = project(K, transform_points(T_cam_world, C), image_size=size)
        if not ok.all():
            continue
        H = cv2.getPerspectiveTransform(src, uv.astype(np.float32))
        tex = cv2.aruco.generateImageMarker(_dictionary("DICT_4X4_50"), mid, TEX)
        warped = cv2.warpPerspective(tex, H, size, flags=cv2.INTER_LINEAR, borderValue=255)
        mask = cv2.warpPerspective(np.full_like(tex, 255), H, size, flags=cv2.INTER_NEAREST) > 0
        img[mask] = warped[mask]
    return img


def _render(K, size, T_cam_world, quads, ss=SUPERSAMPLE):
    """Anti-aliased view: render ss times larger, then area-average down."""
    w, h = size
    K2 = np.array(K, dtype=np.float64)
    K2[0, 0] *= ss
    K2[1, 1] *= ss
    K2[0, 2] = ss * (K[0, 2] + 0.5) - 0.5  # keep pixel centres aligned
    K2[1, 2] = ss * (K[1, 2] + 0.5) - 0.5
    big = _render_raw(K2, (w * ss, h * ss), T_cam_world, quads)
    return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)


def _views(tmp_path_factory, quads, name):
    scene = load_scene()
    model = build_fixture_model()
    d = tmp_path_factory.mktemp(name)
    for im in model.images.values():
        cv2.imwrite(str(d / im.name), _render(scene.K, scene.image_size, im.T_cam_world, quads))
    return model, d, scene, quads


@pytest.fixture(scope="module")
def boards(tmp_path_factory):
    return _views(tmp_path_factory, {m: board_corners(*xy) for m, xy in BOARD_SPOTS.items()}, "b")


@pytest.fixture(scope="module")
def flat(tmp_path_factory):
    return _views(tmp_path_factory, {m: flat_corners(*xy) for m, xy in FLAT_SPOTS.items()}, "f")


def _worst_error(report, model, scene, quads):
    worst = 0.0
    for mid, track in report.tracks.items():
        for k in range(4):
            for image_id, uv in track.corners[k]:
                T = model.images[image_id].T_cam_world
                expected, _ = project(scene.K, transform_points(T, quads[mid][k]))
                worst = max(worst, float(np.linalg.norm(uv - expected)))
    return worst


# --------------------------------------------------------------------------- #
# Detection                                                                    #
# --------------------------------------------------------------------------- #
def test_detects_a_frontal_marker_with_subpixel_corners():
    img = np.full((400, 400), 255, np.uint8)
    img[100:300, 100:300] = cv2.aruco.generateImageMarker(_dictionary("DICT_4X4_50"), 7, 200)
    found = detect_markers(img)
    assert set(found) == {7}
    expected = np.array([[99.5, 99.5], [299.5, 99.5], [299.5, 299.5], [99.5, 299.5]])
    np.testing.assert_allclose(found[7], expected, atol=0.5)


def test_blank_image_has_no_markers():
    assert detect_markers(np.full((200, 200), 255, np.uint8)) == {}


def test_colour_images_are_accepted():
    img = np.full((300, 300), 255, np.uint8)
    img[50:250, 50:250] = cv2.aruco.generateImageMarker(_dictionary("DICT_4X4_50"), 3, 200)
    assert set(detect_markers(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))) == {3}


def test_duplicated_ids_are_ignored_rather_than_guessed():
    """Two prints of the same id: no way to tell which physical marker each one is."""
    img = np.full((300, 600), 255, np.uint8)
    tex = cv2.aruco.generateImageMarker(_dictionary("DICT_4X4_50"), 4, 160)
    img[70:230, 40:200] = tex
    img[70:230, 400:560] = tex
    assert 4 not in detect_markers(img)


def test_quad_geometry_helpers():
    sq = np.array([[0.0, 0.0], [40.0, 0.0], [40.0, 40.0], [0.0, 40.0]])
    assert MT.mean_side_px(sq) == pytest.approx(40.0)
    assert MT.min_interior_angle_deg(sq) == pytest.approx(90.0)
    thin = np.array([[0.0, 0.0], [100.0, 0.0], [90.0, 10.0], [10.0, 10.0]])
    assert MT.min_interior_angle_deg(thin) == pytest.approx(45.0)


# --------------------------------------------------------------------------- #
# Tracks, with markers on 45 deg boards                                        #
# --------------------------------------------------------------------------- #
def test_every_board_marker_is_tracked_in_enough_views(boards):
    model, d, _, quads = boards
    report = MT.collect_marker_tracks_from_config(model, d, RECON)
    assert set(report.usable()) == set(quads)
    for t in report.usable().values():
        assert t.n_views >= 3
        assert all(len(c) == t.n_views for c in t.corners)


def test_board_corners_are_within_a_pixel_of_the_truth(boards):
    """Each observation must be where its known 3D corner projects - corner index included."""
    model, d, scene, quads = boards
    report = MT.collect_marker_tracks_from_config(model, d, RECON)
    assert _worst_error(report, model, scene, quads) < 1.0


# --------------------------------------------------------------------------- #
# The gates, and why flat ground markers fail them                             #
# --------------------------------------------------------------------------- #
def test_the_gates_remove_the_bad_grazing_corners(flat):
    model, d, scene, quads = flat
    ungated = MT.collect_marker_tracks(model, d)
    gated = MT.collect_marker_tracks_from_config(model, d, RECON)
    assert _worst_error(ungated, model, scene, quads) > 1.5, "no grazing error to remove"
    assert _worst_error(gated, model, scene, quads) < 1.0
    assert gated.n_rejected > 0 and ungated.n_rejected == 0


def test_flat_ground_markers_mostly_cannot_be_triangulated(flat):
    """The capture finding, pinned: once grazing views are excluded, most flat markers are
    left with fewer than the two views triangulation needs."""
    model, d, _, quads = flat
    usable = MT.collect_marker_tracks_from_config(model, d, RECON).usable()
    assert len(usable) < len(quads)


def test_detection_fraction_and_fallback(boards):
    model, d, _, _ = boards
    report = MT.collect_marker_tracks_from_config(model, d, RECON)
    assert 0.0 < report.detection_fraction <= 1.0
    assert report.n_images == len(model.images)
    starved = MT.TrackReport({}, n_images=10, n_images_with_markers=3)
    assert MT.detection_fallback_triggered(starved, RECON)  # 30% < 50%
    healthy = MT.TrackReport({}, n_images=10, n_images_with_markers=8)
    assert not MT.detection_fallback_triggered(healthy, RECON)


def test_unreadable_images_are_skipped(boards, tmp_path):
    model, _, _, _ = boards
    report = MT.collect_marker_tracks(model, tmp_path)  # empty folder
    assert report.tracks == {} and report.n_images_with_markers == 0


# --------------------------------------------------------------------------- #
# Undistortion                                                                 #
# --------------------------------------------------------------------------- #
def test_undistortion_recovers_ideal_pinhole_pixels():
    cam = Camera(1, "OPENCV", 1280, 720, [900.0, 900.0, 639.5, 359.5, -0.2, 0.05, 0.001, 0.0005])
    ideal = np.array([[100.0, 80.0], [640.0, 360.0], [1200.0, 650.0], [300.0, 600.0]])
    rays = np.column_stack([(ideal - [639.5, 359.5]) / 900.0, np.ones(len(ideal))])
    distorted, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), cam.K, cam.dist)
    back = MT.undistort_pixels(distorted.reshape(-1, 2), cam)
    np.testing.assert_allclose(back, ideal, atol=0.01)
    assert np.abs(distorted.reshape(-1, 2) - ideal).max() > 5.0  # distortion was real


def test_no_distortion_is_a_no_op():
    cam = Camera(1, "PINHOLE", 1280, 720, [900.0, 900.0, 639.5, 359.5])
    uv = np.array([[12.3, 45.6]])
    np.testing.assert_array_equal(MT.undistort_pixels(uv, cam), uv)
