#!/usr/bin/env python
"""Per-stage throughput benchmark, tagged with the hardware it ran on (P5-T3).

Thin CLI only - logic lives in src/vscs/eval/benchmark.py (CLAUDE.md section 3).

    python scripts/benchmark.py --frames-run <capture run> [--detector] [--log-metrics]
    python scripts/benchmark.py --synthetic 60

Stages timed: image decode, the RT-DETR detector (--detector; needs torch - run on the
Colab T4 for the P5-T3 figure), perception (ground plane + height map + tracker), and the
underbody check. The risk sweep is timed once the developer's risk/sweep.py exists.

Prints p50/p95 latency and fps per stage plus the sequential pipeline rate. With
--log-metrics, appends fps_<stage> lines to metrics/results.jsonl with the hardware in the
notes; leave it off for rehearsals on the laptop (§9: never quote a figure without its
hardware, and P5-T3's figure is the T4 one).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vscs.capture.frames import read_frames_index
from vscs.common.config import load_config, repo_root
from vscs.common.frames import look_at_T_world_cam
from vscs.common.io import append_metric, make_run_dir, write_json
from vscs.common.log import setup_logging
from vscs.eval.benchmark import hardware_tag, pipeline_fps, time_stage
from vscs.perception.pipeline import Box2D, Perception


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]),
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames-run", type=Path)
    src.add_argument("--synthetic", type=int, help="N blank 1280x720 frames")
    parser.add_argument("--detector", action="store_true", help="also time RT-DETR (needs torch)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--log-metrics", action="store_true")
    parser.add_argument("--out-root", type=Path, default=None)
    args = parser.parse_args(argv)

    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    run_dir = make_run_dir("eval", config={"benchmark": settings}, root=args.out_root)
    setup_logging(run_dir=run_dir, level=logging.INFO)
    hw = hardware_tag()

    if args.frames_run:
        import cv2

        paths = [p for _, p in read_frames_index(args.frames_run)]
        decode = lambda p: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)  # noqa: E731
        inputs = paths
    else:
        inputs = [np.zeros((720, 1280, 3), np.uint8)] * args.synthetic
        decode = lambda img: img.copy()  # noqa: E731

    timings = [time_stage("decode", decode, inputs, warmup=args.warmup)]
    images = [decode(x) for x in inputs]

    if args.detector:
        from vscs.perception.detect import RTDetrDetector

        det = RTDetrDetector(load_config("perception")["detect"], device=args.device)
        timings.append(time_stage("detect", det.detect, images, warmup=args.warmup))

    # Perception on a fixed synthetic detection, so its cost is measured on its own.
    pcfg = load_config("perception")
    K = np.array([[900.0, 0, 639.5], [0, 900.0, 359.5], [0, 0, 1]])
    T_veh_cam = look_at_T_world_cam(np.array([-1.0, 0.0, 1.2]), np.array([-3.0, 0.0, 0.0]))
    perception = Perception(pcfg, K, T_veh_cam, (1280, 720))
    box = [Box2D((600.0, 300.0, 680.0, 500.0), "traffic_cone", 0.9)]
    steps = list(range(len(images)))
    timings.append(
        time_stage(
            "perception",
            lambda k: perception.step(k * 66_000_000, box, np.eye(4)),
            steps,
            warmup=args.warmup,
        )
    )

    print(f"hardware: {hw}")
    for t in timings:
        print(
            f"  {t.stage:12s} p50 {t.p50_ms:8.2f} ms  p95 {t.p95_ms:8.2f} ms"
            f"  {t.fps:8.1f} fps  (n={t.n})"
        )
    seq = pipeline_fps(timings)
    print(f"  sequential pipeline: {seq:.1f} fps")
    try:
        import vscs.risk.sweep  # noqa: F401  - the developer's core module (P3-T4)
    except (ImportError, NotImplementedError):
        print("  risk sweep: not timed (risk/sweep.py not written yet)")

    write_json(
        run_dir / "benchmark.json",
        {
            "hardware": hw,
            "stages": [t.__dict__ | {"fps": t.fps} for t in timings],
            "sequential_fps": seq,
        },
    )
    if args.log_metrics:
        rel = run_dir.relative_to(repo_root()) if run_dir.is_relative_to(repo_root()) else run_dir
        for t in timings:
            append_metric(
                task="P5-T3",
                metric=f"fps_{t.stage}",
                value=t.fps,
                split="bench",
                run_dir=str(rel),
                notes=f"hardware: {hw}; p95 {t.p95_ms:.1f} ms",
            )
        print("metrics appended (with hardware) - run scripts/report.py")
    print(f"run folder: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
