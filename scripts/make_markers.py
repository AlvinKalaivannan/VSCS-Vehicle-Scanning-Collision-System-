#!/usr/bin/env python
"""Generate printable ArUco marker sheets for metric scale recovery.

Thin CLI only - logic lives in src/vscs/capture/markers.py (CLAUDE.md §3).

    python scripts/make_markers.py --out data/raw/markers

Then, and this matters more than anything the script does:
  * print at 100% scale, with no "fit to page"
  * MEASURE the printed black square with a ruler or calipers
  * write the measured value into configs/recon.yaml -> scale.marker.side_length_m

The printed size is what makes the vehicle model metric (R-02). A printer that quietly
rescaled the page will otherwise rescale the entire van with no visible symptom.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vscs.capture.markers import write_marker_sheets
from vscs.common.config import load_config
from vscs.common.log import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]),
    )
    parser.add_argument("--out", type=Path, required=True, help="output directory for the PNGs")
    parser.add_argument(
        "--ids",
        type=int,
        nargs="+",
        default=None,
        help="marker ids (default: 0 1 2 - at least three, spread around the vehicle)",
    )
    parser.add_argument(
        "--dictionary",
        default=None,
        help="ArUco dictionary (default: from configs/recon.yaml)",
    )
    parser.add_argument(
        "--size",
        type=float,
        default=None,
        help="intended black-square side in metres (default: from configs/recon.yaml)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="print resolution (default 300)")
    parser.add_argument("--page", default="a4", choices=["a4", "letter"])
    args = parser.parse_args(argv)

    setup_logging(level=logging.INFO)
    marker_cfg = load_config("recon")["scale"]["marker"]
    dictionary = args.dictionary or marker_cfg["dictionary"]
    size = args.size if args.size is not None else float(marker_cfg["side_length_m"])

    sheets = write_marker_sheets(
        args.out,
        ids=args.ids,
        dictionary=dictionary,
        side_length_m=size,
        dpi=args.dpi,
        page=args.page,
    )

    print(f"\nwrote {len(sheets)} sheet(s) to {args.out}")
    for s in sheets:
        print(f"  id {s.marker_id:2d}  {s.path.name}")
    print(
        f"\nIntended black-square size: {size * 1000:.1f} mm at {args.dpi} DPI on {args.page}.\n"
        "Print at 100% scale. Then measure the square and put the MEASURED value in\n"
        "configs/recon.yaml -> scale.marker.side_length_m. Each sheet carries a 100 mm\n"
        "scale bar so you can confirm the printer did not rescale it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
