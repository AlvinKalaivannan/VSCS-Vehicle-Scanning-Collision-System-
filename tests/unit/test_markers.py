"""Marker sheet tests.

The important property is not that a PNG appears, it is that the marker on it is
**detectable** and **the size it claims to be**. The printed marker size is what makes
the van model metric (R-02), so a sheet that renders at the wrong scale would put a
systematic error into every dimension, which the ±2 cm gate at P1-T6 would then fail for
a reason nobody could find.

So the round-trip is tested directly: render the sheet, detect the marker in it, and
measure the detected square in pixels against what the DPI says it should be.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from vscs.capture import markers as M

DPI = 300
SIDE_M = 0.150


@pytest.fixture(scope="module")
def sheet():
    return M.render_marker_sheet(0, side_length_m=SIDE_M, dpi=DPI)


def _detect(img: np.ndarray, dictionary: str = "DICT_4X4_50"):
    """Detect markers, returning ``(corners, flat_ids)``.

    OpenCV 5 returns ids with shape ``(N,)``; older versions used ``(N, 1)``.
    Flattening here keeps the tests indifferent to that.
    """
    aruco_dict = M._dictionary(dictionary)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(img)
    flat = None if ids is None else np.asarray(ids).reshape(-1)
    return corners, flat


# --------------------------------------------------------------------------- #
# Units                                                                        #
# --------------------------------------------------------------------------- #
def test_mm_to_px_at_300_dpi():
    # 25.4 mm is one inch, which is exactly DPI pixels.
    assert M.mm_to_px(25.4, 300) == 300
    assert M.mm_to_px(150.0, 300) == pytest.approx(1772, abs=1)
    assert M.mm_to_px(100.0, 600) == pytest.approx(2362, abs=1)


def test_page_is_the_right_physical_size(sheet):
    """A4 at 300 DPI is 2480 x 3508 px."""
    h, w = sheet.shape
    assert w == pytest.approx(2480, abs=2)
    assert h == pytest.approx(3508, abs=2)


def test_letter_page_size():
    img = M.render_marker_sheet(0, side_length_m=0.100, dpi=150, page="letter")
    h, w = img.shape
    assert w == pytest.approx(M.mm_to_px(215.9, 150), abs=2)
    assert h == pytest.approx(M.mm_to_px(279.4, 150), abs=2)


# --------------------------------------------------------------------------- #
# The round trip that actually matters                                         #
# --------------------------------------------------------------------------- #
def test_rendered_marker_is_detectable_and_has_the_right_id(sheet):
    _, ids = _detect(sheet)
    assert ids is not None, "the rendered marker was not detected at all"
    assert len(ids) == 1, f"expected exactly one marker, found {len(ids)}"
    assert int(ids[0]) == 0


def test_detected_marker_matches_the_requested_physical_size(sheet):
    """The measured square must equal the size the sheet claims, to under a millimetre."""
    corners, ids = _detect(sheet)
    assert ids is not None
    pts = np.asarray(corners[0]).reshape(4, 2)

    sides_px = [float(np.linalg.norm(pts[i] - pts[(i + 1) % 4])) for i in range(4)]
    expected_px = M.mm_to_px(SIDE_M * 1000.0, DPI)
    for s in sides_px:
        assert s == pytest.approx(expected_px, rel=0.01)

    # Convert back to millimetres and compare against the request.
    measured_mm = float(np.mean(sides_px)) / DPI * M.MM_PER_INCH
    assert measured_mm == pytest.approx(SIDE_M * 1000.0, abs=1.0)


@pytest.mark.parametrize("marker_id", [0, 1, 2, 7, 49])
def test_each_id_round_trips(marker_id):
    img = M.render_marker_sheet(marker_id, side_length_m=0.100, dpi=200)
    _, ids = _detect(img)
    assert ids is not None and int(ids[0]) == marker_id


@pytest.mark.parametrize("side_m", [0.050, 0.100, 0.150])
def test_size_scales_correctly(side_m):
    img = M.render_marker_sheet(0, side_length_m=side_m, dpi=DPI)
    corners, ids = _detect(img)
    assert ids is not None
    pts = np.asarray(corners[0]).reshape(4, 2)
    measured_px = float(np.linalg.norm(pts[0] - pts[1]))
    assert measured_px == pytest.approx(M.mm_to_px(side_m * 1000.0, DPI), rel=0.01)


def test_scale_bar_is_exactly_100mm_wide(sheet):
    """The bar is the reader's check that the printer did not rescale the page.

    Found by locating the long horizontal run of black pixels in the lower part of the
    sheet.
    """
    h = sheet.shape[0]
    lower = sheet[int(h * 0.85) :, :]
    dark_per_row = (lower < 128).sum(axis=1)
    row = int(np.argmax(dark_per_row))
    xs = np.nonzero(lower[row] < 128)[0]
    measured_px = int(xs.max() - xs.min())
    assert measured_px == pytest.approx(M.mm_to_px(100.0, DPI), rel=0.02)


# --------------------------------------------------------------------------- #
# Captions                                                                     #
# --------------------------------------------------------------------------- #
def test_sheet_is_mostly_white_so_it_is_printable(sheet):
    """Not a giant black rectangle; ink coverage should be modest."""
    dark_fraction = float((sheet < 128).mean())
    assert 0.01 < dark_fraction < 0.35


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
def test_rejects_an_unknown_dictionary():
    with pytest.raises(ValueError, match="unknown ArUco dictionary"):
        M.render_marker_sheet(0, dictionary="DICT_NOPE")


def test_rejects_an_unknown_page():
    with pytest.raises(ValueError, match="unknown page"):
        M.render_marker_sheet(0, page="a3")


def test_rejects_a_non_positive_size():
    with pytest.raises(ValueError, match="must be positive"):
        M.render_marker_sheet(0, side_length_m=0.0)


def test_rejects_a_marker_too_big_for_the_page():
    with pytest.raises(ValueError, match="does not fit"):
        M.render_marker_sheet(0, side_length_m=0.400)


def test_rejects_an_out_of_range_id(tmp_path):
    """DICT_4X4_50 holds ids 0..49."""
    with pytest.raises(ValueError, match="out of range"):
        M.write_marker_sheets(tmp_path, ids=[50], dictionary="DICT_4X4_50")


# --------------------------------------------------------------------------- #
# Writing sheets                                                               #
# --------------------------------------------------------------------------- #
def test_write_marker_sheets_defaults_to_three(tmp_path):
    """§8.2 falls back to tape-measured dimensions if markers fail; three is the guard."""
    sheets = M.write_marker_sheets(tmp_path, side_length_m=0.100)
    assert len(sheets) == 3
    assert [s.marker_id for s in sheets] == [0, 1, 2]
    for s in sheets:
        assert s.path.is_file() and s.path.stat().st_size > 0


def test_written_filenames_carry_the_size(tmp_path):
    sheets = M.write_marker_sheets(tmp_path, ids=[3], side_length_m=0.150)
    assert sheets[0].path.name == "dict_4x4_50_id03_150mm.png"


def test_written_sheet_still_detects_after_a_png_round_trip(tmp_path):
    """PNG is lossless, so detection must survive the write and read."""
    sheets = M.write_marker_sheets(tmp_path, ids=[5], side_length_m=0.120, dpi=DPI)
    img = cv2.imread(str(sheets[0].path), cv2.IMREAD_GRAYSCALE)
    assert img is not None
    _, ids = _detect(img)
    assert ids is not None and int(ids[0]) == 5


def test_sheets_use_the_configured_dictionary_and_size(tmp_path):
    """The defaults should match configs/recon.yaml so the CLI and config agree."""
    from vscs.common.config import load_config

    marker_cfg = load_config("recon")["scale"]["marker"]
    sheets = M.write_marker_sheets(
        tmp_path,
        ids=[0],
        dictionary=marker_cfg["dictionary"],
        side_length_m=float(marker_cfg["side_length_m"]),
    )
    assert sheets[0].dictionary == marker_cfg["dictionary"]
    assert sheets[0].side_length_m == pytest.approx(marker_cfg["side_length_m"])
