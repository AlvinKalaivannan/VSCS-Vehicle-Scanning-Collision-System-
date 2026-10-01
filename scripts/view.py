#!/usr/bin/env python
"""Write a Rerun recording of a drive: vehicle model, obstacles, risk over time (P3-T1).

Thin CLI only - logic lives in src/vscs/ui/rerun_view.py (CLAUDE.md section 3).

    python scripts/view.py --model <model run folder> --risk <risk_frames.jsonl>
                           [--obstacles <obstacles.jsonl>]

--model is a P2-T8 export (components.yaml + URDF + meshes). --risk is a RiskFrame JSON
Lines stream (CLAUDE.md 4.2). --obstacles, optional, is an Obstacle JSON Lines stream;
each obstacle is drawn at its own t_ns. Writes drive.rrd into a new
data/processed/eval/<run>/ folder; open it with `rerun drive.rrd`.
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
from vscs.model.urdf import load_component_model, load_parts
from vscs.ui.rerun_view import open_recording, record_drive


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
    view_cfg, risk_cfg = load_config("ui")["view"], load_config("risk")
    run_dir = make_run_dir("eval", config={"ui": view_cfg}, root=args.out_root)
    setup_logging(run_dir=run_dir, level=logging.INFO)

    model = load_component_model(args.model)
    parts = load_parts(args.model, model)
    frames = [RiskFrame(**r) for r in read_jsonl(args.risk)]
    obstacles: dict[int, list[Obstacle]] = defaultdict(list)
    if args.obstacles:
        for r in read_jsonl(args.obstacles):
            ob = Obstacle(**r)
            obstacles[ob.t_ns].append(ob)

    rrd = run_dir / "drive.rrd"
    rec = open_recording(rrd)
    n = record_drive(rec, parts, frames, view_cfg, risk_cfg, obstacles_by_t=obstacles)
    rec.flush()
    print(f"{n} frames, {len(parts)} components -> {rrd}")
    print(f"open with: rerun {rrd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
