#!/usr/bin/env python
"""Calibrate phone camera intrinsics from checkerboard images (P0-T7).

Thin CLI only - logic lives in src/vscs/capture/calib.py (CLAUDE.md section 3).

    python scripts/calibrate.py --images data/raw/calib_20260924 --device pixel_main
    python scripts/calibrate.py --images <folder> --dry-run      # report, write nothing

Unless --dry-run, every run - pass or fail - gets a data/processed/capture/<run>/ folder
with calibration_result.json, and appends reprojection_error_px (P0-T7) to
metrics/results.jsonl (--no-metrics to skip). Only a pass writes configs/capture.yaml.

Before shooting the board (risk R-04):
  * lock focus and exposure, main lens only, no zoom
  * 15+ images, board tilted and placed near all four image corners
  * same camera settings you will use for the van scan and the lot runs
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vscs.capture.calib import (
    calibrate_intrinsics,
    find_images,
    write_to_capture_config,
)
from vscs.common.config import load_config, repo_root
from vscs.common.io import append_metric, make_run_dir, write_json
from vscs.common.log import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]),
    )
    parser.add_argument("--images", type=Path, required=True, help="folder of checkerboard images")
    parser.add_argument("--device", type=str, default=None, help="e.g. pixel_8_main_lens")
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="inner corners as COLSxROWS (default: from configs/capture.yaml)",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=None,
        help="checkerboard square size in metres (default: from configs/capture.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the result; write no config, run folder or metric",
    )
    parser.add_argument("--out-root", type=Path, default=None, help="default: data/processed")
    parser.add_argument("--no-metrics", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(level=logging.INFO)
    cfg = load_config("capture")
    board = cfg["checkerboard"]

    if args.pattern:
        cols, rows = (int(v) for v in args.pattern.lower().split("x"))
        pattern = (cols, rows)
    else:
        pattern = (int(board["pattern_size"][0]), int(board["pattern_size"][1]))
    square = args.square_size if args.square_size is not None else float(board["square_size_m"])
    gate = float(board["max_reprojection_error_px"])

    images = find_images(args.images)
    if not images:
        print(f"no images found in {args.images}")
        return 2
    print(f"found {len(images)} images; board {pattern[0]}x{pattern[1]} inner corners, {square} m")

    result = calibrate_intrinsics(
        images,
        pattern_size=pattern,
        square_size_m=square,
        min_images=int(board["min_images"]),
    )
    print(result.summary())

    worst = sorted(result.per_image_error_px.items(), key=lambda kv: -kv[1])[:5]
    print("worst images:")
    for name, err in worst:
        print(f"  {err:7.4f} px  {name}")

    if not args.dry_run:
        run_dir = make_run_dir(
            "capture",
            config={"checkerboard": board, "images": str(args.images), "device": args.device},
            root=args.out_root,
        )
        write_json(
            run_dir / "calibration_result.json",
            {
                "device": args.device,
                "K": result.K.tolist(),
                "dist": result.dist.tolist(),
                "image_size": list(result.image_size),
                "mean_reprojection_error_px": result.mean_reprojection_error_px,
                "max_reprojection_error_px": result.max_reprojection_error_px,
                "per_image_error_px": result.per_image_error_px,
                "n_images_used": result.n_images_used,
                "n_images_total": result.n_images_total,
                "gate_px": gate,
                "passed": result.meets_gate(gate),
            },
        )
        print(f"run folder: {run_dir}")
        if not args.no_metrics:
            under = run_dir.is_relative_to(repo_root())
            append_metric(
                task="P0-T7",
                metric="reprojection_error_px",
                value=result.mean_reprojection_error_px,
                split="calib",
                run_dir=str(run_dir.relative_to(repo_root()) if under else run_dir),
                notes=f"{args.device or 'device not named'}; "
                f"{result.n_images_used}/{result.n_images_total} images",
            )

    if not result.meets_gate(gate):
        print(
            f"\nFAIL: mean error {result.mean_reprojection_error_px:.4f} px is not below "
            f"the {gate} px gate (P0-T7). configs/capture.yaml not modified."
        )
        print("Recapture: fill more of the frame, vary the angles, lock focus and exposure.")
        return 1

    print(f"\nPASS: {result.mean_reprojection_error_px:.4f} px < {gate} px gate")
    if args.dry_run:
        print("--dry-run: configs/capture.yaml not modified")
        return 0

    path = write_to_capture_config(result, device=args.device)
    print(f"wrote intrinsics to {path}")
    print("Next: record this in today's devlog; run scripts/report.py to refresh the report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
