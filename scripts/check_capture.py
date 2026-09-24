#!/usr/bin/env python
"""Validate a test capture before committing to a capture day (P1-T1 open questions).

Thin CLI only - logic lives in src/vscs/capture/preflight.py (CLAUDE.md section 3).

    python scripts/check_capture.py --video test.mp4 --imu gyro.csv
    python scripts/check_capture.py --video test.mp4 --imu gyro.csv \
        --moving-board board_walk.mp4

What to record first (about twenty minutes, at home, not at the van):

  1. Roughly 30 s of walking slowly around a textured object, with a SHARP SHAKE or a
     clap at the very start and again at the very end. That shake is what lets the
     video and IMU clocks be aligned - without it there is nothing to correlate.
     Record the IMU log at the same time, however your chosen app does that.

  2. Optional but worth it: after calibrating (scripts/calibrate.py), shoot about 20 s
     of the checkerboard while walking slowly, and pass it as --moving-board. That is
     what detects electronic stabilisation or heavy rolling shutter warping frames.

Use the same locked settings you intend to use on capture day: focus and exposure
locked, main lens, no zoom (R-04).

Exit code is 0 when nothing failed, 1 otherwise, so this can gate a capture day.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vscs.capture.preflight import run_preflight
from vscs.common.config import load_config
from vscs.common.log import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]),
    )
    parser.add_argument("--video", type=Path, required=True, help="the test clip")
    parser.add_argument("--imu", type=Path, default=None, help="IMU log (CSV) for the same clip")
    parser.add_argument(
        "--sensor",
        default="gyro",
        help="substring selecting the sensor in a long-format log (default: gyro)",
    )
    parser.add_argument(
        "--moving-board",
        type=Path,
        default=None,
        help="clip of the checkerboard recorded while walking, for the stabilisation check",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "ffprobe", "opencv"],
        help="timestamp source (default: auto, prefers ffprobe)",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress log lines")
    args = parser.parse_args(argv)

    setup_logging(level=logging.WARNING if args.quiet else logging.INFO)

    if not args.video.is_file():
        print(f"no such video: {args.video}")
        return 2

    calibration = None
    pattern_size = None
    square_size_m = None
    if args.moving_board is not None:
        capture_cfg = load_config("capture")
        intrinsics = capture_cfg["intrinsics"]
        if not intrinsics.get("calibrated"):
            print(
                "--moving-board needs calibrated intrinsics, but configs/capture.yaml is "
                "still uncalibrated. Run scripts/calibrate.py first (P0-T7)."
            )
            return 2
        calibration = {
            "K": intrinsics["K"],
            "dist": intrinsics["dist"],
            "mean_reprojection_error_px": intrinsics["mean_reprojection_error_px"],
        }
        board = capture_cfg["checkerboard"]
        pattern_size = (int(board["pattern_size"][0]), int(board["pattern_size"][1]))
        square_size_m = float(board["square_size_m"])

    report = run_preflight(
        args.video,
        args.imu,
        sensor=args.sensor,
        backend=args.backend,
        moving_board_video=args.moving_board,
        calibration=calibration,
        pattern_size=pattern_size,
        square_size_m=square_size_m,
    )

    print()
    print(report.render())
    print()
    if report.failed:
        print("Do not book a capture day on this setup until the failures above are resolved.")
    elif report.warned:
        print("Usable, but read the warnings - each one costs you something later.")
    else:
        print("Setup looks sound. Rerun this on the morning of the capture as a go/no-go.")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
