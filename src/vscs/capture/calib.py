"""Phone camera intrinsics calibration from checkerboard images (P0-T7).

Acceptance criterion (CLAUDE.md section 6): reprojection error < 0.5 px, saved to
``configs/capture.yaml``. :func:`write_to_capture_config` refuses to write a result
that misses the gate, so an under-calibrated camera cannot quietly become the
foundation of every later measurement.

Capture rules that matter more than anything in this file (risk R-04):

* **Lock focus and exposure.** Autofocus changes the focal length mid-capture, which
  invalidates the calibration for every frame after it moves.
* **Main lens only, no zoom.** A digital zoom or a lens switch is a different camera.
* Use the same camera settings here as for the actual scan and drive recordings.
* Vary the board's pose: tilt it, move it to the corners of the frame. Fronto-parallel
  images alone leave the focal length and distortion poorly constrained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from vscs.common.config import config_dir, load_config
from vscs.common.log import get_logger

logger = get_logger("capture.calib")

FloatArray = npt.NDArray[np.float64]

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

#: Sub-pixel corner refinement. Without this the corner locations are quantised to
#: whole pixels and the reported error floor sits well above 0.5 px.
_SUBPIX_WINDOW = (11, 11)
_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


@dataclass
class CalibrationResult:
    """Outcome of a calibration run."""

    K: FloatArray
    dist: FloatArray
    image_size: tuple[int, int]
    mean_reprojection_error_px: float
    max_reprojection_error_px: float
    per_image_error_px: dict[str, float] = field(default_factory=dict)
    n_images_used: int = 0
    n_images_total: int = 0
    pattern_size: tuple[int, int] = (9, 6)
    square_size_m: float = 0.025

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    def meets_gate(self, max_error_px: float) -> bool:
        """True if the mean reprojection error is strictly below the gate."""
        return self.mean_reprojection_error_px < max_error_px

    def summary(self) -> str:
        w, h = self.image_size
        return (
            f"{self.n_images_used}/{self.n_images_total} images used, {w}x{h} px, "
            f"fx={self.fx:.2f} fy={self.fy:.2f}, "
            f"mean reprojection error {self.mean_reprojection_error_px:.4f} px "
            f"(worst image {self.max_reprojection_error_px:.4f} px)"
        )


def _object_points(pattern_size: tuple[int, int], square_size_m: float) -> FloatArray:
    """The board's corners in its own frame: z = 0, spaced by the square size.

    Feeding real metres here is what makes the calibration metric rather than
    scale-free.
    """
    cols, rows = pattern_size
    grid = np.zeros((rows * cols, 3), dtype=np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    # float32 throughout: cv2.calibrateCamera requires Point3f object points.
    return (grid * float(square_size_m)).astype(np.float32)


def find_images(folder: Path) -> list[Path]:
    """Every image file directly under ``folder``, sorted for determinism."""
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a directory")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def detect_corners(
    image_path: Path,
    pattern_size: tuple[int, int],
) -> tuple[FloatArray | None, tuple[int, int] | None]:
    """Detect refined checkerboard corners in one image.

    Returns ``(corners, (width, height))``, or ``(None, size)`` when the board is not
    found. ``pattern_size`` counts *inner* corners, not squares - a 10x7 board of
    squares has 9x6 inner corners.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.warning("could not read %s", image_path.name)
        return None, None
    size = (int(img.shape[1]), int(img.shape[0]))

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(img, pattern_size, flags)
    if not found:
        return None, size
    corners = cv2.cornerSubPix(img, corners, _SUBPIX_WINDOW, (-1, -1), _SUBPIX_CRITERIA)
    # Must stay float32: cv2.calibrateCamera requires Point2f and rejects float64.
    return corners.astype(np.float32), size


def calibrate_intrinsics(
    image_paths: list[Path],
    *,
    pattern_size: tuple[int, int] = (9, 6),
    square_size_m: float = 0.025,
    min_images: int = 15,
) -> CalibrationResult:
    """Calibrate from checkerboard images.

    Raises if too few images yield a detection, rather than returning a confident
    result from three photographs.
    """
    if not image_paths:
        raise ValueError("no images given")

    objp = _object_points(pattern_size, square_size_m)
    obj_points: list[FloatArray] = []
    img_points: list[FloatArray] = []
    used: list[Path] = []
    image_size: tuple[int, int] | None = None

    for path in image_paths:
        corners, size = detect_corners(path, pattern_size)
        if size is None:
            continue
        if image_size is None:
            image_size = size
        elif size != image_size:
            raise ValueError(
                f"{path.name} is {size} but earlier images are {image_size}. All "
                "calibration images must come from the same camera at the same "
                "resolution (R-04)."
            )
        if corners is None:
            logger.warning("no %dx%d board found in %s", *pattern_size, path.name)
            continue
        obj_points.append(objp)
        img_points.append(corners)
        used.append(path)

    if image_size is None:
        raise ValueError("none of the given files could be read as images")
    if len(used) < min_images:
        raise ValueError(
            f"only {len(used)} of {len(image_paths)} images yielded a "
            f"{pattern_size[0]}x{pattern_size[1]} board detection; need at least "
            f"{min_images}. Check the pattern size (inner corners, not squares), "
            "lighting, and that the whole board is visible."
        )

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)
    logger.info("cv2.calibrateCamera overall RMS: %.4f px", rms)

    per_image: dict[str, float] = {}
    per_image_inputs = zip(used, obj_points, img_points, rvecs, tvecs, strict=True)
    for path, objp_i, imgp_i, rvec, tvec in per_image_inputs:
        projected, _ = cv2.projectPoints(objp_i, rvec, tvec, K, dist)
        # RMS over the corners of this image, in pixels.
        residual = projected.reshape(-1, 2) - imgp_i.reshape(-1, 2)
        per_image[path.name] = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))

    errors = np.array(list(per_image.values()))
    return CalibrationResult(
        K=np.asarray(K, dtype=np.float64),
        dist=np.asarray(dist, dtype=np.float64).reshape(-1),
        image_size=image_size,
        mean_reprojection_error_px=float(errors.mean()),
        max_reprojection_error_px=float(errors.max()),
        per_image_error_px=per_image,
        n_images_used=len(used),
        n_images_total=len(image_paths),
        pattern_size=pattern_size,
        square_size_m=square_size_m,
    )


def result_to_config_block(
    result: CalibrationResult, *, device: str | None = None
) -> dict[str, Any]:
    """The ``intrinsics:`` block to write into ``configs/capture.yaml``."""
    return {
        "calibrated": True,
        "device": device,
        "image_size": [int(result.image_size[0]), int(result.image_size[1])],
        "K": [[float(v) for v in row] for row in result.K],
        "dist": [float(v) for v in result.dist],
        "model": "opencv_pinhole",
        "mean_reprojection_error_px": round(result.mean_reprojection_error_px, 6),
        "calibrated_at": date.today().isoformat(),
        "notes": (
            f"{result.n_images_used} images, "
            f"{result.pattern_size[0]}x{result.pattern_size[1]} inner corners, "
            f"{result.square_size_m} m squares. "
            "Lock focus and exposure, main lens only, no zoom (R-04)."
        ),
    }


def write_to_capture_config(
    result: CalibrationResult,
    *,
    max_error_px: float | None = None,
    device: str | None = None,
    config_path: Path | None = None,
) -> Path:
    """Write the intrinsics into ``configs/capture.yaml``, gated on the error.

    Refuses to write when the mean reprojection error is not below the gate. A bad
    calibration is worse than none: it silently biases every distance downstream,
    and P0-T7 exists precisely to prevent that.

    The YAML is edited as text so the file's comments survive - ``safe_dump`` would
    strip every one of them.
    """
    path = Path(config_path) if config_path is not None else config_dir() / "capture.yaml"
    cfg = load_config(str(path))
    gate = (
        max_error_px
        if max_error_px is not None
        else float(cfg["checkerboard"]["max_reprojection_error_px"])
    )

    if not result.meets_gate(gate):
        raise ValueError(
            f"calibration rejected: mean reprojection error "
            f"{result.mean_reprojection_error_px:.4f} px is not below the {gate} px gate "
            f"(P0-T7). Recapture with the board filling more of the frame, at more "
            f"varied angles, with focus and exposure locked. Nothing was written."
        )

    block = result_to_config_block(result, device=device)
    _replace_yaml_block(path, "intrinsics", block)
    logger.info("wrote intrinsics to %s (%s)", path, result.summary())
    return path


def _replace_yaml_block(path: Path, top_key: str, block: dict[str, Any]) -> None:
    """Replace one top-level mapping in a YAML file, preserving other lines.

    Deliberately a text operation. Round-tripping through ``yaml.safe_load`` and
    ``safe_dump`` would discard every comment in ``capture.yaml``, and those comments
    carry the capture rules.
    """
    import yaml

    lines = path.read_text(encoding="utf-8").splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith(f"{top_key}:")), None)
    if start is None:
        raise ValueError(f"{path} has no top-level '{top_key}:' key")

    end = len(lines)
    for i in range(start + 1, len(lines)):
        stripped = lines[i]
        # The block ends at the next line that starts in column 0 and is not a comment.
        if stripped and not stripped[0].isspace() and not stripped.startswith("#"):
            end = i
            break

    rendered = yaml.safe_dump({top_key: block}, sort_keys=False, default_flow_style=False)
    new_lines = lines[:start] + rendered.rstrip("\n").splitlines() + lines[end:]
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
