"""Synthetic checkerboard rendering, shared by the calibration and preflight tests.

Real checkerboard photographs do not exist yet — that capture is the developer's to shoot
(P0-T7). So the tests render board views through a *known* pinhole camera, which means the
correct answer is known exactly and the whole path (detection, sub-pixel refinement, solve,
reprojection error, the acceptance gate) can be exercised before any photograph is taken.

:func:`render_view` can also apply a **rolling-shutter** distortion. That matters: a global
rigid warp of the image is indistinguishable from a different camera pose, so ``solvePnP``
simply absorbs it and the reprojection error stays low. Rolling shutter is *non-rigid* —
each row is sampled at a different instant, so the image shears progressively down the
frame — and no single pose can explain it. That is exactly why per-frame warping shows up
as a jump in reprojection error, which is what ``preflight.check_frame_warp`` measures.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vscs.common.frames import axis_angle_to_R

#: Inner corner count. A 10x7 board of squares has 9x6 inner corners.
PATTERN = (9, 6)
SQUARES = (10, 7)
SQUARE_SIZE_M = 0.025
PX_PER_SQUARE = 80
IMAGE_SIZE = (1280, 720)

#: The camera the synthetic views are rendered through. Tests assert it is recovered.
K_TRUE = np.array([[900.0, 0.0, 639.5], [0.0, 900.0, 359.5], [0.0, 0.0, 1.0]])


def board_image() -> np.ndarray:
    """A checkerboard bitmap with a white quiet zone around it."""
    cols, rows = SQUARES
    board = np.zeros((rows * PX_PER_SQUARE, cols * PX_PER_SQUARE), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                board[
                    r * PX_PER_SQUARE : (r + 1) * PX_PER_SQUARE,
                    c * PX_PER_SQUARE : (c + 1) * PX_PER_SQUARE,
                ] = 255
    pad = PX_PER_SQUARE
    padded = np.full((board.shape[0] + 2 * pad, board.shape[1] + 2 * pad), 255, dtype=np.uint8)
    padded[pad : pad + board.shape[0], pad : pad + board.shape[1]] = board
    return padded


def board_pixel_to_metres(px: float, py: float) -> np.ndarray:
    """Map a pixel of the board bitmap to its 3D position in the board plane.

    Inner corner (0, 0) — the object-point origin — is the corner between the first and
    second square, which in the padded bitmap sits at ``(2*pad, 2*pad)``.
    """
    origin = 2.0 * PX_PER_SQUARE
    scale = SQUARE_SIZE_M / PX_PER_SQUARE
    return np.array([(px - origin) * scale, (py - origin) * scale, 0.0])


def _rolling_shutter(img: np.ndarray, shift_px: float) -> np.ndarray:
    """Shear the image progressively down the frame, as a rolling shutter does.

    Row ``y`` is displaced horizontally by ``(y / H) * shift_px``. Non-rigid by
    construction, so it cannot be absorbed by a camera pose.
    """
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = xs + (ys / max(h - 1, 1)) * float(shift_px)
    return cv2.remap(
        img, map_x, ys, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def render_view(
    board: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    *,
    rolling_shutter_px: float = 0.0,
) -> np.ndarray:
    """Render the board as seen by :data:`K_TRUE` at pose ``(R, t)``."""
    h, w = board.shape[:2]
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst = []
    for px, py in src:
        p_cam = R @ board_pixel_to_metres(float(px), float(py)) + t
        uv = K_TRUE @ p_cam
        dst.append([uv[0] / uv[2], uv[1] / uv[2]])
    H = cv2.getPerspectiveTransform(src, np.array(dst, dtype=np.float32))
    view = cv2.warpPerspective(
        board,
        H,
        IMAGE_SIZE,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    if rolling_shutter_px:
        view = _rolling_shutter(view, rolling_shutter_px)
    return view


def _board_centre() -> np.ndarray:
    return np.array(
        [(PATTERN[0] - 1) / 2 * SQUARE_SIZE_M, (PATTERN[1] - 1) / 2 * SQUARE_SIZE_M, 0.0]
    )


def synth_still_images(out_dir: Path, n: int = 18, seed: int = 20260924) -> list[Path]:
    """Write ``n`` still checkerboard views at varied poses, for calibration."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    board = board_image()
    rng = np.random.default_rng(seed)
    centre = _board_centre()
    paths: list[Path] = []
    for i in range(n):
        R = axis_angle_to_R(rng.normal(size=3), float(rng.uniform(-0.45, 0.45)))
        depth = float(rng.uniform(0.34, 0.58))
        offset = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.035, 0.035), depth])
        p = out_dir / f"calib_{i:02d}.png"
        cv2.imwrite(str(p), render_view(board, R, offset - R @ centre))
        paths.append(p)
    return paths


def synth_board_video(
    path: Path,
    *,
    n_frames: int = 40,
    fps: float = 30.0,
    seed: int = 7,
    rolling_shutter_px: float = 0.0,
) -> Path:
    """Write a clip of the board while the camera moves smoothly, as if walking.

    ``rolling_shutter_px`` simulates per-frame warping: 0 is a rigid camera, and a nonzero
    value shears each frame, which is what ``check_frame_warp`` is designed to detect.
    """
    path = Path(path)
    board = board_image()
    rng = np.random.default_rng(seed)
    centre = _board_centre()
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, IMAGE_SIZE)
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an mp4v writer")
    try:
        for i in range(n_frames):
            # Smooth sweep, plus a little jitter so the motion is not perfectly regular.
            phase = i / max(n_frames - 1, 1)
            angle = 0.35 * np.sin(2 * np.pi * phase) + 0.02 * rng.normal()
            R = axis_angle_to_R([0.2, 1.0, 0.1], float(angle))
            depth = 0.45 + 0.06 * np.cos(2 * np.pi * phase)
            offset = np.array([0.03 * np.sin(2 * np.pi * phase), 0.0, depth])
            frame = render_view(
                board, R, offset - R @ centre, rolling_shutter_px=rolling_shutter_px
            )
            writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()
    return path
