#!/usr/bin/env python
"""Per-component risk for a recorded drive: model + obstacles + ego -> RiskFrames (P3).

Thin CLI only - logic lives in src/vscs/risk/ (CLAUDE.md section 3).

    python scripts/risk.py --model <model run folder> --perception <perception run folder>
                           [--baseline]

--model is a P2-T8 export. --perception is a scripts/perceive.py run (obstacles.jsonl +
ego.jsonl). Writes risk_frames.jsonl (one §4.2 RiskFrame per line) into a new
data/processed/risk/<run>/; scripts/view.py and scripts/replay.py read it.

--baseline runs the identical pipeline with the van as ONE oriented box (P5-T1), so both
can be scored on the same drive. The sweep is the developer's risk/sweep.py (P3-T4); until
it exists this script stops with a clear message.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vscs.common.config import load_config
from vscs.common.io import make_run_dir, read_jsonl, write_jsonl
from vscs.common.log import setup_logging
from vscs.common.types import Obstacle
from vscs.eval.baseline_bbox import baseline_severity_cfg, single_box_footprint
from vscs.model.urdf import load_component_model, load_parts, plan_footprints
from vscs.risk.drive import Shape, assess_drive, component_shapes, ego_states

#: Replaced in tests; None means the engine's default, risk/sweep.py.
SWEEP_FN = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]),
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--perception", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true", help="the van as one box (P5-T1)")
    parser.add_argument("--out-root", type=Path, default=None)
    args = parser.parse_args(argv)

    for p in (args.model, args.perception / "obstacles.jsonl", args.perception / "ego.jsonl"):
        if not p.exists():
            print(f"not found: {p}")
            return 2
    risk_cfg, sev_cfg = load_config("risk"), load_config("severity")
    run_dir = make_run_dir(
        "risk", config={"risk": risk_cfg, "baseline": args.baseline}, root=args.out_root
    )
    setup_logging(run_dir=run_dir, level=logging.INFO)

    footprints = plan_footprints(load_parts(args.model, load_component_model(args.model)))
    if args.baseline:
        base_cfg = load_config("eval")["baseline"]
        b = single_box_footprint(footprints, base_cfg["name"])
        components = [Shape(b.name, b.polygon, b.z_min, b.z_max)]
        sev_cfg = baseline_severity_cfg(sev_cfg, base_cfg)
    else:
        components = component_shapes(footprints)

    obstacles: dict[int, list[Obstacle]] = defaultdict(list)
    for r in read_jsonl(args.perception / "obstacles.jsonl"):
        ob = Obstacle(**r)
        obstacles[ob.t_ns].append(ob)
    poses = [
        (int(r["t_ns"]), np.asarray(r["T_world_veh"]).reshape(4, 4))
        for r in read_jsonl(args.perception / "ego.jsonl")
    ]
    try:
        frames = assess_drive(
            components, obstacles, ego_states(poses), risk_cfg, sev_cfg, sweep_fn=SWEEP_FN
        )
    except (ImportError, NotImplementedError) as exc:
        # Missing on main, a stub on the draft branch: either way, not written yet.
        print(f"the sweep is not written yet ({exc}): risk/sweep.py is the developer's P3-T4 draft")
        return 3
    write_jsonl(run_dir / "risk_frames.jsonl", frames)
    worst = max(
        frames,
        key=lambda f: ("none", "caution", "warning", "critical").index(f.alert_level),
        default=None,
    )
    print(
        f"{len(frames)} frames{' (baseline)' if args.baseline else ''}; "
        f"highest alert: {worst.alert_level if worst else 'n/a'}\nrun folder: {run_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
