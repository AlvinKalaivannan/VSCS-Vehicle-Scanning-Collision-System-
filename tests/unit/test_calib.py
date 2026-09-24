"""Calibration tests (P0-T7 tooling).

Real checkerboard images do not exist yet - that capture is the developer's to shoot.
So these tests *render* checkerboard images through a known pinhole camera and check
that :func:`calibrate_intrinsics` recovers the camera it was given. That verifies the
whole path (detection, sub-pixel refinement, solve, error computation, the gate)
before any photograph is taken.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from vscs.capture import calib
from vscs.common.frames import axis_angle_to_R

PATTERN = (9, 6)  # inner corners
SQUARES = (10, 7)  # squares, one more than inner corners in each axis
SQUARE_SIZE_M = 0.025
PX_PER_SQUARE = 80
IMAGE_SIZE = (1280, 720)

K_TRUE = np.array([[900.0, 0.0, 639.5], [0.0, 900.0, 359.5], [0.0, 0.0, 1.0]])


def _board_image() -> np.ndarray:
    """A clean checkerboard bitmap: white border, SQUARES alternating cells."""
    cols, rows = SQUARES
    board = np.zeros((rows * PX_PER_SQUARE, cols * PX_PER_SQUARE), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                board[
                    r * PX_PER_SQUARE : (r + 1) * PX_PER_SQUARE,
                    c * PX_PER_SQUARE : (c + 1) * PX_PER_SQUARE,
                ] = 255
    # A white quiet zone round the board; findChessboardCorners needs the outer
    # squares to be bounded by background.
    pad = PX_PER_SQUARE
    padded = np.full((board.shape[0] + 2 * pad, board.shape[1] + 2 * pad), 255, dtype=np.uint8)
    padded[pad : pad + board.shape[0], pad : pad + board.shape[1]] = board
    return padded


def _board_pixel_to_board_metres(px: float, py: float) -> np.ndarray:
    """Map a pixel in the board bitmap to its 3D position in the board plane.

    Inner corner (0, 0) - the object-point origin - is the corner between the first
    and second square, which in the padded bitmap sits at (2*pad, 2*pad) where
    pad = PX_PER_SQUARE.
    """
    origin = 2.0 * PX_PER_SQUARE
    scale = SQUARE_SIZE_M / PX_PER_SQUARE
    return np.array([(px - origin) * scale, (py - origin) * scale, 0.0])


def _render_view(board: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Render the board seen by K_TRUE at pose (R, t), as a grey image."""
    h, w = board.shape
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst = []
    for px, py in src:
        p_board = _board_pixel_to_board_metres(float(px), float(py))
        p_cam = R @ p_board + t
        uv = K_TRUE @ p_cam
        dst.append([uv[0] / uv[2], uv[1] / uv[2]])
    H = cv2.getPerspectiveTransform(src, np.array(dst, dtype=np.float32))
    return cv2.warpPerspective(
        board,
        H,
        IMAGE_SIZE,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def _synth_images(tmp_path, n: int = 18, seed: int = 20260924) -> list:
    """Write ``n`` synthetic checkerboard views and return their paths."""
    board = _board_image()
    rng = np.random.default_rng(seed)
    # Centre of the inner-corner grid, so poses can be aimed at the board's middle.
    center_board = np.array(
        [(PATTERN[0] - 1) / 2 * SQUARE_SIZE_M, (PATTERN[1] - 1) / 2 * SQUARE_SIZE_M, 0.0]
    )
    paths = []
    for i in range(n):
        axis = rng.normal(size=3)
        angle = float(rng.uniform(-0.45, 0.45))
        R = axis_angle_to_R(axis, angle)
        depth = float(rng.uniform(0.34, 0.58))
        offset = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.035, 0.035), depth])
        t = offset - R @ center_board
        img = _render_view(board, R, t)
        p = tmp_path / f"calib_{i:02d}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths


# --------------------------------------------------------------------------- #
# Object points                                                                #
# --------------------------------------------------------------------------- #
def test_object_points_are_metric_and_planar():
    objp = calib._object_points(PATTERN, SQUARE_SIZE_M)
    assert objp.shape == (PATTERN[0] * PATTERN[1], 3)
    assert np.all(objp[:, 2] == 0.0)
    # Neighbouring corners are exactly one square apart, in real metres.
    assert np.isclose(np.linalg.norm(objp[1] - objp[0]), SQUARE_SIZE_M)
    span_x = objp[:, 0].max() - objp[:, 0].min()
    assert np.isclose(span_x, (PATTERN[0] - 1) * SQUARE_SIZE_M)


# --------------------------------------------------------------------------- #
# Detection                                                                    #
# --------------------------------------------------------------------------- #
def test_corners_are_detected_in_a_rendered_board(tmp_path):
    paths = _synth_images(tmp_path, n=1)
    corners, size = calib.detect_corners(paths[0], PATTERN)
    assert size == IMAGE_SIZE
    assert corners is not None
    assert corners.reshape(-1, 2).shape == (PATTERN[0] * PATTERN[1], 2)


def test_detection_returns_none_on_a_blank_image(tmp_path):
    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), np.full((720, 1280), 255, dtype=np.uint8))
    corners, size = calib.detect_corners(blank, PATTERN)
    assert corners is None and size == IMAGE_SIZE


def test_detection_handles_an_unreadable_file(tmp_path):
    bad = tmp_path / "not_an_image.png"
    bad.write_bytes(b"nonsense")
    corners, size = calib.detect_corners(bad, PATTERN)
    assert corners is None and size is None


def test_find_images_sorted_and_filtered(tmp_path):
    for name in ("b.png", "a.jpg", "notes.txt", "c.PNG"):
        (tmp_path / name).write_bytes(b"x")
    found = [p.name for p in calib.find_images(tmp_path)]
    assert found == ["a.jpg", "b.png", "c.PNG"]
    assert "notes.txt" not in found


def test_find_images_rejects_a_non_directory(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        calib.find_images(f)


# --------------------------------------------------------------------------- #
# The solve                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def calibrated(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("calib_imgs")
    paths = _synth_images(tmp, n=18)
    return calib.calibrate_intrinsics(
        paths, pattern_size=PATTERN, square_size_m=SQUARE_SIZE_M, min_images=10
    )


def test_recovers_the_true_focal_length(calibrated):
    """The whole point: the numbers written to capture.yaml must be the real camera."""
    assert calibrated.fx == pytest.approx(K_TRUE[0, 0], rel=0.02)
    assert calibrated.fy == pytest.approx(K_TRUE[1, 1], rel=0.02)


def test_recovers_the_principal_point(calibrated):
    assert calibrated.K[0, 2] == pytest.approx(K_TRUE[0, 2], abs=0.03 * IMAGE_SIZE[0])
    assert calibrated.K[1, 2] == pytest.approx(K_TRUE[1, 2], abs=0.03 * IMAGE_SIZE[1])


def test_meets_the_half_pixel_gate(calibrated):
    """P0-T7's acceptance criterion, exercised end to end."""
    assert calibrated.mean_reprojection_error_px < 0.5
    assert calibrated.meets_gate(0.5)
    assert not calibrated.meets_gate(0.0)


def test_reports_per_image_errors(calibrated):
    assert len(calibrated.per_image_error_px) == calibrated.n_images_used
    assert calibrated.max_reprojection_error_px >= calibrated.mean_reprojection_error_px
    assert all(e >= 0 for e in calibrated.per_image_error_px.values())


def test_summary_mentions_the_error_and_size(calibrated):
    s = calibrated.summary()
    assert "reprojection error" in s and "1280x720" in s


def test_rejects_too_few_images(tmp_path):
    paths = _synth_images(tmp_path, n=3)
    with pytest.raises(ValueError, match="need at least"):
        calib.calibrate_intrinsics(
            paths, pattern_size=PATTERN, square_size_m=SQUARE_SIZE_M, min_images=15
        )


def test_rejects_no_images():
    with pytest.raises(ValueError, match="no images given"):
        calib.calibrate_intrinsics([])


def test_rejects_mixed_resolutions(tmp_path):
    """R-04: images from different camera settings must not be mixed."""
    paths = _synth_images(tmp_path, n=2)
    small = cv2.resize(cv2.imread(str(paths[1]), cv2.IMREAD_GRAYSCALE), (640, 360))
    cv2.imwrite(str(paths[1]), small)
    with pytest.raises(ValueError, match="same camera at the same"):
        calib.calibrate_intrinsics(
            paths, pattern_size=PATTERN, square_size_m=SQUARE_SIZE_M, min_images=1
        )


def test_rejects_unreadable_only_input(tmp_path):
    bad = tmp_path / "x.png"
    bad.write_bytes(b"nope")
    with pytest.raises(ValueError, match="could be read as images"):
        calib.calibrate_intrinsics([bad], pattern_size=PATTERN, min_images=1)


# --------------------------------------------------------------------------- #
# Writing to capture.yaml - the gate                                           #
# --------------------------------------------------------------------------- #
def _capture_yaml_copy(tmp_path):
    from vscs.common.config import config_dir

    src = config_dir() / "capture.yaml"
    dst = tmp_path / "capture.yaml"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dst


def test_write_refuses_when_the_gate_is_missed(tmp_path, calibrated):
    """A bad calibration must never reach capture.yaml."""
    dst = _capture_yaml_copy(tmp_path)
    before = dst.read_text(encoding="utf-8")
    bad = calib.CalibrationResult(
        K=calibrated.K,
        dist=calibrated.dist,
        image_size=calibrated.image_size,
        mean_reprojection_error_px=1.7,
        max_reprojection_error_px=3.0,
        n_images_used=20,
        n_images_total=20,
    )
    with pytest.raises(ValueError, match="calibration rejected"):
        calib.write_to_capture_config(bad, config_path=dst)
    assert dst.read_text(encoding="utf-8") == before, "file must be untouched"


def test_write_succeeds_and_keeps_comments(tmp_path, calibrated):
    from vscs.common.config import load_config

    dst = _capture_yaml_copy(tmp_path)
    calib.write_to_capture_config(calibrated, device="test_cam", config_path=dst)

    text = dst.read_text(encoding="utf-8")
    # The capture rules live in comments; they must survive the write.
    assert "Lock focus and exposure" in text
    assert "R-03" in text or "container timestamps" in text

    cfg = load_config(str(dst))
    intr = cfg["intrinsics"]
    assert intr["calibrated"] is True
    assert intr["device"] == "test_cam"
    assert intr["image_size"] == [1280, 720]
    assert np.array(intr["K"]).shape == (3, 3)
    assert intr["mean_reprojection_error_px"] < 0.5
    assert intr["calibrated_at"]
    # Everything else in the file must still be there.
    assert cfg["checkerboard"]["max_reprojection_error_px"] == 0.5
    assert cfg["ingest"]["timestamp_source"] == "container_pts"


def test_written_intrinsics_round_trip_through_project(tmp_path, calibrated):
    """The written K must be usable by frames.project without reshaping gymnastics."""
    from vscs.common.config import load_config
    from vscs.common.frames import project, unproject

    dst = _capture_yaml_copy(tmp_path)
    calib.write_to_capture_config(calibrated, config_path=dst)
    K = np.array(load_config(str(dst))["intrinsics"]["K"])

    p_cam = np.array([0.1, -0.05, 2.0])
    uv, valid = project(K, p_cam)
    assert valid
    np.testing.assert_allclose(unproject(K, uv, p_cam[2]), p_cam, atol=1e-9)


def test_config_block_is_json_safe(calibrated):
    import json

    block = calib.result_to_config_block(calibrated, device="d")
    json.dumps(block)  # must not raise: no numpy scalars left in it


def test_replace_block_rejects_a_missing_key(tmp_path):
    """The YAML block writer now lives in common/config.py and is shared with ingest."""
    from vscs.common.config import replace_top_level_block

    p = tmp_path / "c.yaml"
    p.write_text("other: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no top-level"):
        replace_top_level_block(p, "intrinsics", {"a": 1})
