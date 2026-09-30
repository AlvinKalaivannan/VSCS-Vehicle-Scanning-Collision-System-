#!/usr/bin/env python
"""2D component masks for a scan: Grounding DINO -> SAM 2 (P2-T2). Runs on Colab (GPU).

Thin CLI only - logic lives in src/vscs/seg/masks2d.py (CLAUDE.md section 3).

    python scripts/seg.py --frames <frames folder> --sam2-dir <sam2 checkout>

Writes a new data/processed/seg/<run>/ folder: labels/<frame>.png (uint8, 255 = none),
check/ overlays of 20 seeded-random frames for the P2-T2 visual check, and
masks_summary.json (detections per keyframe, per-component coverage, components never
detected). Exit code 1 if any component was never detected.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vscs.common.config import load_config
from vscs.common.io import make_run_dir
from vscs.common.log import setup_logging
from vscs.seg.masks2d import GroundingDinoDetector, Sam2VideoSegmenter, run_masks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]),
    )
    parser.add_argument("--frames", type=Path, required=True, help="extracted scan frames")
    parser.add_argument("--sam2-dir", type=Path, required=True, help="the sam2 git checkout")
    parser.add_argument("--out-root", type=Path, default=None, help="default: data/processed")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    frames = sorted(p for p in args.frames.iterdir() if p.suffix.lower() in (".png", ".jpg"))
    if not frames:
        print(f"no frames in {args.frames}")
        return 2
    seg_cfg = load_config("seg")
    run_dir = make_run_dir("seg", config={"seg": seg_cfg}, root=args.out_root)
    setup_logging(run_dir=run_dir, level=logging.INFO)

    detector = GroundingDinoDetector(
        seg_cfg["detector"]["model_id"],
        float(seg_cfg["detector"]["text_threshold"]),
        device=args.device,
    )
    segmenter = Sam2VideoSegmenter(
        seg_cfg["segmenter"]["model_cfg"],
        str(args.sam2_dir / seg_cfg["segmenter"]["checkpoint"]),
        work_dir=run_dir,
        device=args.device,
    )
    summary = run_masks(frames, run_dir, seg_cfg, detector, segmenter)

    print(f"\n{summary.n_frames} frames, {len(summary.keyframes)} keyframes")
    for name, n in summary.frames_with_component.items():
        print(f"  {name:28s} in {n} frames")
    if summary.never_detected:
        print(f"\nNEVER DETECTED (log as P2-T2 failures): {', '.join(summary.never_detected)}")
    print(f"\nvisual check: {run_dir / 'check'}  ({len(summary.check_frames)} frames)")
    print(f"run folder: {run_dir}")
    return 1 if summary.never_detected else 0


if __name__ == "__main__":
    raise SystemExit(main())
