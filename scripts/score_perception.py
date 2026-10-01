#!/usr/bin/env python
"""Score perceived obstacle positions against the lot layout on the dev split (P3-T2).

Thin CLI only - logic lives in src/vscs/eval/lot_truth.py (CLAUDE.md section 3).

    python scripts/score_perception.py --gt data/raw/lot_<date>/gt.yaml --passes <passes.yaml>

--gt is the lot day's ground truth (format PROPOSED, ADR 0012): obstacle positions and the
van's start pose per pass, from the rear-hub chalk marks. --passes is the dev passes file
(the one scripts/evaluate.py reads), with a "perception" run folder per pass. Only dev
passes are accepted (R-09).

Each pass is scored while the van is still parked: the frames just after the start sync
clap are compared with the layout carried into veh by the start pose. Writes
perception_scores.json into a new data/processed/eval/<run>/ and, unless --no-metrics,
appends cone_position_error_m: the worst per-obstacle median error at <= 3 m. Exit code 1
if that exceeds the 0.25 m gate or any obstacle in range was missed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pydantic import ValidationError

from vscs.common.config import load_config, repo_root
from vscs.common.io import append_metric, make_run_dir, read_jsonl, write_json
from vscs.common.log import setup_logging
from vscs.common.types import Obstacle
from vscs.eval.evaluate import check_pass_split
from vscs.eval.lot_truth import cone_errors, load_lot_truth, lot_to_veh, veh_pose_in_lot

NS = 1_000_000_000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]),
    )
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--passes", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=None, help="default: data/processed")
    parser.add_argument("--no-metrics", action="store_true")
    args = parser.parse_args(argv)

    ev, cap = load_config("eval"), load_config("capture")
    lc = ev["lot_truth"]
    if not cap["mount"].get("measured"):
        print("configs/capture.yaml mount is not measured: ranges need the camera position")
        return 2
    try:
        truth = load_lot_truth(args.gt)
        spec = yaml.safe_load(args.passes.read_text(encoding="utf-8"))
        ids = [str(p["id"]) for p in spec["passes"]]
        check_pass_split(ids, ev["splits"], "dev")
    except (OSError, ValidationError, ValueError, KeyError) as exc:
        print(f"cannot read the inputs: {exc}")
        return 2
    except PermissionError as exc:
        print(f"refused: {exc}")
        return 2

    run_dir = make_run_dir(
        "eval",
        config={"lot_truth": lc, "gt": str(args.gt), "passes": str(args.passes)},
        root=args.out_root,
    )
    setup_logging(run_dir=run_dir, level=logging.INFO)
    camera_xy = np.asarray(cap["mount"]["position_veh_m"][:2], dtype=np.float64)
    kinds = set(lc["score_kinds"])
    per_pass, scored, missed = {}, [], []
    for p in spec["passes"]:
        pid = str(p["id"])
        try:
            R, o = veh_pose_in_lot(
                truth.pass_(pid).start, truth.rear_track_m, float(lc["track_tolerance_m"])
            )
        except (KeyError, ValueError) as exc:
            print(f"pass {pid}: {exc}")
            return 2
        run = Path(p["perception"])
        frames = [int(r["t_ns"]) for r in read_jsonl(run / "ego.jsonl")]
        t0 = min(frames) + round(float(lc["start_skip_s"]) * NS)
        res = cone_errors(
            [Obstacle(**r) for r in read_jsonl(run / "obstacles.jsonl")],
            {
                o_.id: lot_to_veh(o_.position_m, R, o)[0]
                for o_ in truth.obstacles
                if o_.kind in kinds
            },
            camera_xy,
            frames,
            t_from_ns=t0,
            t_to_ns=t0 + round(float(lc["start_window_s"]) * NS),
            max_range_m=float(lc["max_range_m"]),
            match_gate_m=float(lc["match_gate_m"]),
        )
        per_pass[pid] = res
        for oid, r in res.items():
            frac = r["n_detected"] / r["n_frames"] if r["n_frames"] else 0.0
            if frac < float(lc["min_detected_frac"]):
                missed.append(f"{pid}/{oid}")
            else:
                scored.append((f"{pid}/{oid}", r["median_error_m"]))

    worst = max(scored, key=lambda s: s[1]) if scored else None
    limit = float(ev["thresholds"]["cone_position_error_m"]["limit"])
    passed = worst is not None and worst[1] <= limit and not missed
    write_json(
        run_dir / "perception_scores.json",
        {"per_pass": per_pass, "missed": missed, "worst": worst, "passed": passed},
    )
    print(
        f"{len(scored)} obstacle sightings at <= {lc['max_range_m']} m scored, {len(missed)} missed"
    )
    if worst:
        print(f"worst median error {worst[1]:.3f} m ({worst[0]}), gate {limit} m")
    if missed:
        print(f"MISSED (R-07): {', '.join(missed)}")
    print(f"P3-T2: {'PASS' if passed else 'FAIL'}\nrun folder: {run_dir}")

    if worst and not args.no_metrics:
        rel = run_dir.relative_to(repo_root()) if run_dir.is_relative_to(repo_root()) else run_dir
        append_metric(
            task="P3-T2",
            metric="cone_position_error_m",
            value=worst[1],
            split="dev",
            run_dir=str(rel),
            notes=f"worst of {len(scored)} at <= {lc['max_range_m']} m; {len(missed)} missed",
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
