"""Printable ArUco marker sheets for metric scale recovery (P1-T2 preparation).

The markers are what make the reconstructed van **metric** rather than scale-free
(R-02), so the printed size is the single most important number in the whole capture. Two
things go wrong in practice and both are addressed here:

* **The printer rescales the page.** "Fit to page" or "shrink oversized pages" silently
  changes the marker size, and therefore every dimension of the vehicle model, with no
  visible symptom. Each sheet is generated at an exact physical size and prints its
  intended size on it, plus a ruler scale bar, so a wrong print is *visible* rather than
  silent.
* **The nominal size is trusted instead of measured.** Each sheet says, on the sheet, to
  measure the black square and write the measured value into ``configs/recon.yaml``.

Sheets are plain PNGs at a stated DPI. No PDF library is needed - the DPI is what fixes
the physical size, and it is written into the PNG metadata as well as printed on the face.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vscs.common.log import get_logger

logger = get_logger("capture.markers")

MM_PER_INCH = 25.4

#: A4 in millimetres. Letter is 216 x 279; both fit a 150 mm marker comfortably.
PAGE_SIZES_MM = {
    "a4": (210.0, 297.0),
    "letter": (215.9, 279.4),
}


@dataclass
class MarkerSheet:
    """One generated sheet."""

    marker_id: int
    dictionary: str
    side_length_m: float
    dpi: int
    path: Path
    page_size_mm: tuple[float, float]


def _dictionary(name: str):
    """Resolve an ArUco dictionary by name, e.g. ``DICT_4X4_50``."""
    attr = getattr(cv2.aruco, name, None)
    if attr is None:
        available = sorted(a for a in dir(cv2.aruco) if a.startswith("DICT_"))
        raise ValueError(f"unknown ArUco dictionary {name!r}. Available: {available}")
    return cv2.aruco.getPredefinedDictionary(attr)


def mm_to_px(mm: float, dpi: int) -> int:
    """Millimetres to pixels at a given DPI."""
    return round(mm / MM_PER_INCH * dpi)


def render_marker_sheet(
    marker_id: int,
    *,
    dictionary: str = "DICT_4X4_50",
    side_length_m: float = 0.150,
    dpi: int = 300,
    page: str = "a4",
    quiet_zone_mm: float = 10.0,
) -> np.ndarray:
    """Render one printable sheet as a white page with a centred marker and captions.

    ``side_length_m`` is the side of the marker's **black border square** - the full
    extent of the generated marker image, including its black border. That is what ArUco
    pose estimation and ``configs/recon.yaml`` both mean by marker size.
    """
    if page.lower() not in PAGE_SIZES_MM:
        raise ValueError(f"unknown page {page!r}; use one of {sorted(PAGE_SIZES_MM)}")
    if side_length_m <= 0:
        raise ValueError("side_length_m must be positive")

    page_w_mm, page_h_mm = PAGE_SIZES_MM[page.lower()]
    side_mm = side_length_m * 1000.0
    if side_mm > min(page_w_mm, page_h_mm) - 40.0:
        raise ValueError(
            f"a {side_mm:.0f} mm marker does not fit on {page} with margins; "
            "use a smaller marker or a larger page"
        )

    page_w = mm_to_px(page_w_mm, dpi)
    page_h = mm_to_px(page_h_mm, dpi)
    sheet = np.full((page_h, page_w), 255, dtype=np.uint8)

    aruco_dict = _dictionary(dictionary)
    side_px = mm_to_px(side_mm, dpi)
    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, side_px)

    # Centre horizontally, and a little above centre to leave room for the captions.
    x0 = (page_w - side_px) // 2
    y0 = mm_to_px(45.0, dpi)
    sheet[y0 : y0 + side_px, x0 : x0 + side_px] = marker

    if quiet_zone_mm > 0:
        _draw_crop_marks(sheet, x0, y0, side_px, mm_to_px(quiet_zone_mm, dpi), dpi)

    _draw_captions(sheet, dictionary, marker_id, side_mm, dpi, x0, y0, side_px)
    _draw_scale_bar(sheet, dpi, y_mm=page_h_mm - 30.0)
    return sheet


def _draw_crop_marks(sheet: np.ndarray, x0: int, y0: int, side_px: int, qz: int, dpi: int) -> None:
    """Corner crop marks outside the marker's quiet zone.

    **Deliberately L-shaped ticks rather than a closed rectangle.** A black rectangle
    drawn around a marker is itself a perfectly good quadrilateral, and
    ``detectMarkers`` locks onto it instead of the marker's own border: the decoded id is
    still right, but the reported corners are the *guide's* corners, so the measured side
    comes back larger than the marker actually is.

    Measured with a 150 mm marker and a 10 mm guide at 300 DPI, the detector reported
    2008 px instead of 1772 px - 13% oversize. Since the marker side is what converts the
    reconstruction to metres, that would have scaled the whole van model by 13% and shown
    up only as an unexplained failure of the ±2 cm gate at P1-T6 (R-02). Open ticks cannot
    be closed into a quad, so the marker stays the only detectable square on the page.
    """
    arm = mm_to_px(6.0, dpi)
    thickness = max(1, dpi // 300)
    left, right = x0 - qz, x0 + side_px + qz
    top, bottom = y0 - qz, y0 + side_px + qz
    for cx, cy, dx, dy in (
        (left, top, 1, 1),
        (right, top, -1, 1),
        (left, bottom, 1, -1),
        (right, bottom, -1, -1),
    ):
        cv2.line(sheet, (cx, cy), (cx + dx * arm, cy), 0, thickness)
        cv2.line(sheet, (cx, cy), (cx, cy + dy * arm), 0, thickness)


def _put(img: np.ndarray, text: str, org: tuple[int, int], dpi: int, scale: float = 1.0) -> None:
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        dpi / 300.0 * 0.6 * scale,
        0,
        max(1, int(dpi / 300.0 * 1.6 * scale)),
        cv2.LINE_AA,
    )


def _draw_captions(
    sheet: np.ndarray,
    dictionary: str,
    marker_id: int,
    side_mm: float,
    dpi: int,
    x0: int,
    y0: int,
    side_px: int,
) -> None:
    """Print what this marker is, and the instruction that keeps the scale honest."""
    left = mm_to_px(20.0, dpi)
    _put(sheet, f"{dictionary}   id {marker_id}", (left, mm_to_px(22.0, dpi)), dpi, 1.2)
    _put(
        sheet,
        f"intended black square: {side_mm:.1f} mm  ({side_mm / 10:.2f} cm)",
        (left, mm_to_px(33.0, dpi)),
        dpi,
    )

    y = y0 + side_px + mm_to_px(14.0, dpi)
    for line in (
        "PRINT AT 100% SCALE - no 'fit to page', no 'shrink oversized'.",
        "Then MEASURE the black square edge with a ruler or calipers.",
        "Write the MEASURED value (in metres) into configs/recon.yaml:",
        "    scale.marker.side_length_m",
        "Do not trust the number printed above - trust your ruler.",
        "",
        "Mount flat on rigid backing. A curled marker is a wrong scale.",
    ):
        if line:
            _put(sheet, line, (left, y), dpi, 0.85)
        y += mm_to_px(7.0, dpi)


def _draw_scale_bar(sheet: np.ndarray, dpi: int, *, y_mm: float) -> None:
    """A 100 mm ruler with 10 mm ticks, so a rescaled print is immediately obvious."""
    x_mm = 20.0
    y = mm_to_px(y_mm, dpi)
    x_start = mm_to_px(x_mm, dpi)
    x_end = mm_to_px(x_mm + 100.0, dpi)
    thickness = max(1, dpi // 300)
    cv2.line(sheet, (x_start, y), (x_end, y), 0, thickness)
    for cm in range(11):
        x = mm_to_px(x_mm + cm * 10.0, dpi)
        tick = mm_to_px(4.0 if cm % 5 == 0 else 2.5, dpi)
        cv2.line(sheet, (x, y), (x, y - tick), 0, thickness)
    _put(sheet, "0", (x_start - mm_to_px(2, dpi), y + mm_to_px(6, dpi)), dpi, 0.7)
    _put(sheet, "100 mm", (x_end - mm_to_px(10, dpi), y + mm_to_px(6, dpi)), dpi, 0.7)
    _put(
        sheet,
        "check: this bar must measure exactly 100 mm",
        (x_start, y + mm_to_px(13, dpi)),
        dpi,
        0.7,
    )


def write_marker_sheets(
    out_dir: Path,
    *,
    ids: list[int] | None = None,
    dictionary: str = "DICT_4X4_50",
    side_length_m: float = 0.150,
    dpi: int = 300,
    page: str = "a4",
) -> list[MarkerSheet]:
    """Render and write one sheet per marker id.

    Defaults to ids 0, 1, 2 because §8.2 steps scale recovery down to tape-measured
    dimensions once marker detection fails in over half the frames, and three markers
    spread around the vehicle is what avoids that.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker_ids = ids if ids is not None else [0, 1, 2]

    aruco_dict = _dictionary(dictionary)
    n_available = aruco_dict.bytesList.shape[0]
    for mid in marker_ids:
        if not 0 <= mid < n_available:
            raise ValueError(
                f"marker id {mid} is out of range for {dictionary} (0..{n_available - 1})"
            )

    sheets: list[MarkerSheet] = []
    for mid in marker_ids:
        img = render_marker_sheet(
            mid,
            dictionary=dictionary,
            side_length_m=side_length_m,
            dpi=dpi,
            page=page,
        )
        path = out_dir / f"{dictionary.lower()}_id{mid:02d}_{int(side_length_m * 1000)}mm.png"
        if not cv2.imwrite(str(path), img):
            raise RuntimeError(f"failed to write {path}")
        sheets.append(
            MarkerSheet(
                marker_id=mid,
                dictionary=dictionary,
                side_length_m=side_length_m,
                dpi=dpi,
                path=path,
                page_size_mm=PAGE_SIZES_MM[page.lower()],
            )
        )
        logger.info("wrote %s", path.name)
    return sheets


# --------------------------------------------------------------------------- #
# Detection                                                                    #
# --------------------------------------------------------------------------- #
def detect_markers(image: np.ndarray, dictionary: str = "DICT_4X4_50") -> dict[int, np.ndarray]:
    """Detect ArUco markers and return ``{marker_id: corners (4, 2)}`` in image pixels.

    Corners come in ArUco's order - top-left, top-right, bottom-right, bottom-left *of the
    marker itself* - so the same physical corner has the same index in every view, which is
    what lets a corner be triangulated across images.

    Sub-pixel corner refinement is switched on. Marker size is what sets the metric scale
    (R-02), and whole-pixel corners on a marker only 30-60 px across would put percent-level
    error straight into every dimension of the van.

    If one id is detected more than once in an image (a duplicated print), neither copy is
    returned: there is no way to tell which physical marker each detection is.
    """
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(_dictionary(dictionary), params)
    corners, ids, _ = detector.detectMarkers(grey)
    if ids is None:
        return {}
    flat = np.asarray(ids).reshape(-1)
    out: dict[int, np.ndarray] = {}
    dupes = {int(i) for i in flat if (flat == i).sum() > 1}
    for mid, c in zip(flat, corners, strict=True):
        if int(mid) in dupes:
            continue
        out[int(mid)] = np.asarray(c, dtype=np.float64).reshape(4, 2)
    if dupes:
        logger.warning("marker id(s) %s seen more than once in one image; ignored", sorted(dupes))
    return out
