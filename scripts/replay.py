#!/usr/bin/env python
"""Driver replay of a recorded drive: silhouette frames + audio cue (P5-T4).

Thin CLI only - logic lives in src/vscs/ui/driver_replay.py (CLAUDE.md section 3).

    python scripts/replay.py --model <model run folder> --risk <risk_frames.jsonl>
                             [--obstacles <obstacles.jsonl>]

Writes frames/*.png, replay.wav and index.json into a new data/processed/eval/<run>/.
Every frame carries the "advisory research prototype - not a safety device" banner (§1).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vscs.common.config import load_config
from vscs.common.io import make_run_dir, read_jsonl
from vscs.common.log import setup_logging
from vscs.common.types import Obstacle, RiskFrame
from vscs.model.urdf import load_component_model, load_parts, plan_footprints
from vscs.ui.driver_replay import write_replay


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]),
    )
    parser.add_argument("--model", type=Path, required=True, help="P2-T8 model run folder")
    parser.add_argument("--risk", type=Path, required=True, help="RiskFrame JSON Lines")
    parser.add_argument("--obstacles", type=Path, default=None, help="Obstacle JSON Lines")
    parser.add_argument("--out-root", type=Path, default=None, help="default: data/processed")
    args = parser.parse_args(argv)

    for p in (args.model, args.risk):
        if not p.exists():
            print(f"not found: {p}")
            return 2
    ui_cfg, risk_cfg = load_config("ui"), load_config("risk")
    run_dir = make_run_dir("eval", config={"ui": ui_cfg}, root=args.out_root)
    setup_logging(run_dir=run_dir, level=logging.INFO)
    model = load_component_model(args.model)
    footprints = plan_footprints(load_parts(args.model, model))
    frames = [RiskFrame(**r) for r in read_jsonl(args.risk)]
    if not frames:
        print("no frames in the risk stream")
        return 2
    obstacles: dict[int, list[Obstacle]] = defaultdict(list)
    if args.obstacles:
        for r in read_jsonl(args.obstacles):
            ob = Obstacle(**r)
            obstacles[ob.t_ns].append(ob)
    write_replay(run_dir, footprints, frames, ui_cfg, risk_cfg, obstacles_by_t=obstacles)
    print(f"{len(frames)} frames + replay.wav -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
