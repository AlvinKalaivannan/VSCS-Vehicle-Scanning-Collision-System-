#!/usr/bin/env python
"""Check the scan's wheelbase against the tape, from the segmented wheels (P1-T6, ADR 0013).

Thin CLI only - logic lives in src/vscs/eval/dimensions.py (CLAUDE.md section 3).

    python scripts/check_wheelbase.py --labelled <cleaned.npz> --measured <measurements.yaml>

--labelled is a scripts/fuse.py cleaned.npz (points in veh). --measured is the same
scan-day file scripts/scale.py read; its "wheelbase" (metres) is used. Writes
wheelbase.json into a new data/processed/eval/<run>/ and, unless --no-metrics, appends
wheelbase_error_m (P1-T6). Exit code 1 if it exceeds the gate. The left/right gaps and the
rear-axle x offset say whether a miss comes from segmentation or from the frame.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vscs.common.config import load_config, repo_root
from vscs.common.io import append_metric, make_run_dir, write_json
from vscs.common.log import setup_logging
from vscs.eval.dimensions import wheelbase_from_wheels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]),
    )
    parser.add_argument("--labelled", type=Path, required=True)
    parser.add_argument("--measured", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=None, help="default: data/processed")
    parser.add_argument("--no-metrics", action="store_true")
    args = parser.parse_args(argv)

    for p in (args.labelled, args.measured):
        if not p.is_file():
            print(f"not found: {p}")
            return 2
    ev = load_config("eval")
    wcfg = ev["wheelbase"]
    tape = (yaml.safe_load(args.measured.read_text(encoding="utf-8")) or {}).get("wheelbase")
    if tape is None or not 0.5 < float(tape) < 10.0:
        print(f"{args.measured}: needs 'wheelbase' in metres, got {tape!r}")
        return 2
    try:
        with np.load(args.labelled, allow_pickle=False) as z:
            wb = wheelbase_from_wheels(
                z["points"],
                z["labels"],
                [str(n) for n in z["names"]],
                list(wcfg["front_wheels"]),
                list(wcfg["rear_wheels"]),
                float(wcfg["trim_percentile"]),
            )
    except (KeyError, ValueError) as exc:
        print(f"cannot measure the wheelbase: {exc}")
        return 2

    run_dir = make_run_dir(
        "eval", config={"wheelbase": wcfg, "labelled": str(args.labelled)}, root=args.out_root
    )
    setup_logging(run_dir=run_dir, level=logging.INFO)
    err = wb.wheelbase_m - float(tape)
    limit = float(ev["thresholds"]["wheelbase_error_m"]["limit"])
    write_json(
        run_dir / "wheelbase.json",
        {**asdict(wb), "tape_m": float(tape), "error_m": err, "passed": abs(err) <= limit},
    )
    print(f"wheelbase {wb.wheelbase_m:.3f} m, tape {float(tape):.3f} m, error {100 * err:+.1f} cm")
    print(
        f"  left/right hub gap: front {100 * wb.front_left_right_gap_m:.1f} cm, "
        f"rear {100 * wb.rear_left_right_gap_m:.1f} cm  (large = segmentation)"
    )
    print(f"  rear axle at x = {100 * wb.rear_axle_x_m:+.1f} cm (should be 0: frame / overhang)")
    print(f"P1-T6 wheelbase: {'PASS' if abs(err) <= limit else 'FAIL'} (gate {limit} m)")
    print(f"run folder: {run_dir}")
    if not args.no_metrics:
        rel = run_dir.relative_to(repo_root()) if run_dir.is_relative_to(repo_root()) else run_dir
        append_metric(
            task="P1-T6",
            metric="wheelbase_error_m",
            value=abs(err),
            split="scan",
            run_dir=str(rel),
            notes=f"signed {err:+.4f} m; rear axle x {wb.rear_axle_x_m:+.4f} m",
        )
    return 0 if abs(err) <= limit else 1


if __name__ == "__main__":
    raise SystemExit(main())
