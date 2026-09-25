#!/usr/bin/env python
"""Run COLMAP structure-from-motion on extracted scan frames (P1-T5).

Thin CLI only - logic lives in src/vscs/recon/sfm.py (CLAUDE.md section 3).

    python scripts/recon.py --frames-run data/processed/capture/<run>
    python scripts/recon.py --images <folder of frames> --colmap C:/COLMAP/COLMAP.bat

Writes a new run folder under data/processed/recon/, containing the COLMAP workspace,
the text model, sfm_result.json and run.log. Unless --no-metrics is given, appends
frames_registered_frac and sfm_reprojection_error_px to metrics/results.jsonl, which is
how the P1-T5 gates (>= 90% registered, < 1.5 px) reach docs/REPORT.md.

Exit code is 0 when both P1-T5 gates pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vscs.common.config import load_config, repo_root
from vscs.common.io import append_metric, make_run_dir, read_json, write_json
from vscs.common.log import setup_logging
from vscs.recon.sfm import run_sfm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]),
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames-run", type=Path, help="a capture run folder from frame extraction")
    src.add_argument("--images", type=Path, help="a folder of frames")
    parser.add_argument(
        "--config", type=Path, default=None, help="recon config (default: configs/recon.yaml)"
    )
    parser.add_argument("--colmap", default=None, help="path to the COLMAP executable")
    parser.add_argument(
        "--no-metrics", action="store_true", help="do not append to metrics/results.jsonl"
    )
    args = parser.parse_args(argv)

    recon_cfg = load_config(str(args.config) if args.config else "recon")
    intrinsics = load_config("capture")["intrinsics"]

    n_frames = None
    if args.frames_run:
        image_dir = args.frames_run / "frames"
        meta = args.frames_run / "frames_meta.json"
        if meta.is_file():
            n_frames = int(read_json(meta)["n_frames_written"])
    else:
        image_dir = args.images
    if not image_dir.is_dir():
        print(f"no frames folder at {image_dir}")
        return 2

    run_dir = make_run_dir("recon", config={"recon": recon_cfg, "intrinsics": intrinsics})
    setup_logging(run_dir=run_dir, level=logging.INFO)

    result, _ = run_sfm(
        image_dir, run_dir, recon_cfg["sfm"], intrinsics, colmap=args.colmap, n_frames=n_frames
    )
    write_json(run_dir / "sfm_result.json", result.to_dict())

    print()
    print(result.summary())
    print(f"\nrun folder: {run_dir}")

    if not args.no_metrics:
        rel = run_dir.relative_to(repo_root()) if run_dir.is_relative_to(repo_root()) else run_dir
        notes = f"{result.n_models} model(s); {result.n_points} points"
        for metric, value in (
            ("frames_registered_frac", result.registered_fraction),
            ("sfm_reprojection_error_px", result.mean_reprojection_error_px),
        ):
            append_metric(
                task="P1-T5",
                metric=metric,
                value=value,
                split="scan",
                run_dir=str(rel),
                notes=notes,
            )
        print("metrics appended - run scripts/report.py to refresh docs/REPORT.md")

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
