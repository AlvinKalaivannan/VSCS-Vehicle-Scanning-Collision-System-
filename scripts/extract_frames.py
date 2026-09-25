#!/usr/bin/env python
"""Extract frames with real timestamps from a raw recording (P1-T4).

Thin CLI only - logic lives in src/vscs/capture/frames.py (CLAUDE.md section 3).

    python scripts/extract_frames.py --video data/raw/<id>/<clip>.mp4
    python scripts/extract_frames.py --video <clip> --stride 5 --format jpg

Writes a new run folder under data/processed/capture/ holding frames/, frames.jsonl (each
frame's real container timestamp in int64 ns - never its index, R-03) and frames_meta.json
(timing report: VFR, jitter, dropped-frame gaps). That folder is what
scripts/recon.py --frames-run takes next.

The raw recording is only read, never modified (CLAUDE.md section 0).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vscs.capture.frames import extract_frames
from vscs.common.config import load_config
from vscs.common.io import make_run_dir
from vscs.common.log import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]),
    )
    parser.add_argument("--video", type=Path, required=True, help="the recording to extract")
    parser.add_argument(
        "--stride", type=int, default=None, help="keep every Nth frame (default: capture.yaml)"
    )
    parser.add_argument(
        "--format", default=None, choices=["png", "jpg"], help="default: capture.yaml"
    )
    parser.add_argument("--max-frames", type=int, default=None, help="stop after this many")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "ffprobe", "opencv"],
        help="timestamp source (default: auto, prefers ffprobe)",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="parent of the run folder (default: data/processed)",
    )
    args = parser.parse_args(argv)

    if not args.video.is_file():
        print(f"no such video: {args.video}")
        return 2

    ingest = load_config("capture")["ingest"]
    stride = args.stride if args.stride is not None else int(ingest["extract_stride"])
    fmt = args.format or str(ingest["frame_format"])

    resolved = {"video": str(args.video), "stride": stride, "format": fmt, "ingest": ingest}
    run_dir = make_run_dir("capture", config=resolved, root=args.out_root)
    setup_logging(run_dir=run_dir, level=logging.INFO)

    meta = extract_frames(
        args.video,
        run_dir,
        backend=args.backend,
        stride=stride,
        image_format=fmt,
        max_frames=args.max_frames,
    )
    timing = meta["timing"]
    print(
        f"\nextracted {meta['n_frames_written']} of {meta['n_frames_source']} frames "
        f"(stride {stride}) via {meta['backend']}"
    )
    print(
        f"timing: {timing['mean_fps']:.2f} fps mean, "
        f"{'VFR' if timing['is_vfr'] else 'CFR'} (jitter {timing['jitter_ratio']:.1%}), "
        f"{len(timing['gap_indices'])} dropped-frame gap(s)"
    )
    if meta["backend"] == "opencv":
        print("NOTE: timestamps came from OpenCV, not ffprobe - install ffmpeg (ADR 0003).")
    print(f"\nrun folder: {run_dir}")
    print(f"next: python scripts/recon.py --frames-run {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
