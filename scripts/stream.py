#!/usr/bin/env python
"""Run the streaming pipeline skeleton on CPU with STUB stages (Phase 6 scaffolding).

Thin CLI only - logic lives in src/vscs/stream/ (CLAUDE.md section 3).

    python scripts/stream.py --frames-run data/processed/capture/<run>
    python scripts/stream.py --synthetic 300 --fps 30 --load 1.5

Replays frames at their recorded timestamps (P6-T1) through the stages in
configs/stream.yaml, joined by drop-oldest queues (P6-T2), and prints live fps, end-to-end
p95 and drops (P6-T3). Each STUB stage just waits for --load x its budget, so --load 1.0
is a pipeline exactly on budget and --load 2.0 is twice over. Real perception and risk
stages replace the stubs at P6-T4; until then no number from this script is a
performance claim (CLAUDE.md §9). Writes summary.json to data/processed/stream/<run>/.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vscs.common.config import load_config
from vscs.common.io import make_run_dir, write_json
from vscs.common.log import setup_logging
from vscs.stream.metrics import LatencyRecorder, over_budget
from vscs.stream.pipeline import Pipeline, Stage
from vscs.stream.replay import FileReplay, frames_from_capture_run


def _stub(seconds: float):
    def fn(payload):
        time.sleep(seconds)
        return payload

    return fn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]),
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames-run", type=Path, help="an extract_frames.py run folder")
    src.add_argument("--synthetic", type=int, help="number of synthetic frames")
    parser.add_argument("--fps", type=float, default=30.0, help="synthetic frame rate")
    parser.add_argument("--load", type=float, default=0.5, help="stub time as a multiple of budget")
    parser.add_argument("--report-every", type=int, default=30, help="status line every N frames")
    parser.add_argument("--out-root", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = load_config("stream")
    if args.frames_run:
        frames = frames_from_capture_run(args.frames_run)
    else:
        period = round(1e9 / args.fps)
        frames = [(i * period, i) for i in range(args.synthetic)]

    run_dir = make_run_dir("stream", config={"stream": cfg, "load": args.load}, root=args.out_root)
    setup_logging(run_dir=run_dir, level=logging.INFO)
    budget = cfg["budget_ms"]
    stages = [Stage(name, _stub(args.load * budget[name] / 1000.0)) for name in cfg["stages"]]
    recorder = LatencyRecorder(int(cfg["metrics_window"]))

    def status(frame):
        if recorder.finished % args.report_every == 0:
            s = recorder.summary()
            e2e = s["end_to_end"]
            print(
                f"[{recorder.finished:5d}] {s['fps']:5.1f} fps | e2e p95 {e2e['p95_ms']:6.1f} ms "
                f"| dropped {sum(s['dropped'].values())}",
                flush=True,
            )

    replay = FileReplay(
        frames,
        speed=float(cfg["replay"]["speed"]),
        late_tolerance_ns=round(float(cfg["replay"]["late_tolerance_ms"]) * 1e6),
    )
    summary = Pipeline(stages, int(cfg["queue_size"]), recorder, sink=status).run(replay)
    summary["over_budget"] = over_budget(summary, budget) if summary["end_to_end"] else {}
    summary["source_late_frames"] = replay.late
    summary["stub_load"] = args.load
    summary["note"] = "STUB stages (scaffolding): not a performance claim"
    write_json(run_dir / "summary.json", summary)
    print(
        f"\nfinished {recorder.finished} of {len(frames)} frames; over budget: "
        f"{summary['over_budget'] or 'none'}\nrun folder: {run_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
