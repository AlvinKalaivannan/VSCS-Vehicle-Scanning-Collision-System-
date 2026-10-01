#!/usr/bin/env python
"""Score perception against the lot ground truth on the dev split (P3-T2, P4-T2).

Thin CLI only - logic lives in src/vscs/eval/lot_truth.py (CLAUDE.md section 3).

    python scripts/score_perception.py --gt data/raw/lot_<date>/gt.yaml --passes <passes.yaml>

--gt is the lot day's ground truth (format: ADR 0012): obstacle positions and the
van's start pose per pass, from the rear-hub chalk marks. --passes is the dev passes file
(the one scripts/evaluate.py reads), with a "perception" run folder per pass. Only dev
passes are accepted (R-09).

Each pass is scored while the van is still parked: the frames just after the start sync
clap are compared with the layout carried into veh by the start pose. Writes
perception_scores.json into a new data/processed/eval/<run>/ and, unless --no-metrics,
appends cone_position_error_m: the worst per-obstacle median error at <= 3 m.

Passes that also have an end pose are scored for ego-motion drift (P4-T2): the motion from
the first to the last ego.jsonl pose against the motion between the two hub-mark poses,
as a fraction of the distance moved. Appends egomotion_drift_frac (the worst pass).

Only passes tagged ``shape: straight`` in gt.yaml are gated (on a curve, displacement is
shorter than the distance driven and overstates drift); curved and untagged passes are
reported beside it.

Depth (P4-T1): in the same parked-start window, every measured obstacle's range error
(perceived minus true range from the camera), bucketed at eval.yaml lot_truth
depth_buckets_m. Appends depth_error_m_at_<b>m per bucket (median absolute error); the
3 m bucket is gated.

Exit code 1 if a gate fails, or an obstacle in range was missed.
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
from vscs.eval.lot_truth import (
    cone_errors,
    depth_bucket,
    ego_drift,
    load_lot_truth,
    lot_to_veh,
    range_errors,
    veh_pose_in_lot,
)

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
    per_pass, scored, missed, drifts = {}, [], [], {}
    depth: dict[float, list[tuple[str, float]]] = {}
    tol = float(lc["track_tolerance_m"])
    for p in spec["passes"]:
        pid = str(p["id"])
        try:
            lp = truth.pass_(pid)
            R, o = veh_pose_in_lot(lp.start, truth.rear_track_m, tol)
            end = veh_pose_in_lot(lp.end, truth.rear_track_m, tol) if lp.end else None
        except (KeyError, ValueError) as exc:
            print(f"pass {pid}: {exc}")
            return 2
        run = Path(p["perception"])
        ego = list(read_jsonl(run / "ego.jsonl"))
        frames = [int(r["t_ns"]) for r in ego]
        if end is not None:
            d = ego_drift(ego[0]["T_world_veh"], ego[-1]["T_world_veh"], (R, o), end)
            if d["displacement_m"] >= float(lc["min_drift_displacement_m"]):
                drifts[pid] = {**d, "shape": lp.shape}
        t0 = min(frames) + round(float(lc["start_skip_s"]) * NS)
        obstacles = [Obstacle(**r) for r in read_jsonl(run / "obstacles.jsonl")]
        truth_veh = {
            o_.id: lot_to_veh(o_.position_m, R, o)[0] for o_ in truth.obstacles if o_.kind in kinds
        }
        t1 = t0 + round(float(lc["start_window_s"]) * NS)
        res = cone_errors(
            obstacles,
            truth_veh,
            camera_xy,
            frames,
            t_from_ns=t0,
            t_to_ns=t1,
            max_range_m=float(lc["max_range_m"]),
            match_gate_m=float(lc["match_gate_m"]),
        )
        rng = range_errors(
            obstacles,
            truth_veh,
            camera_xy,
            frames,
            t_from_ns=t0,
            t_to_ns=t1,
            match_gate_m=float(lc["match_gate_m"]),
        )
        for oid, r in rng.items():
            b = depth_bucket(
                r["range_m"], list(lc["depth_buckets_m"]), float(lc["depth_bucket_rel_tol"])
            )
            if b is not None and r["median_signed_error_m"] is not None:
                depth.setdefault(b, []).append((f"{pid}/{oid}", r["median_signed_error_m"]))
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
    gated = {k: v for k, v in drifts.items() if v["shape"] == "straight"}
    drift_worst = max(gated.items(), key=lambda kv: kv[1]["drift_frac"]) if gated else None
    drift_limit = float(ev["thresholds"]["egomotion_drift_frac"]["limit"])
    drift_ok = drift_worst is None or drift_worst[1]["drift_frac"] <= drift_limit
    depth_err = {b: float(np.median([abs(e) for _, e in v])) for b, v in sorted(depth.items())}
    d3_limit = float(ev["thresholds"]["depth_error_m_at_3m"]["limit"])
    depth_ok = depth_err.get(3.0) is None or depth_err[3.0] <= d3_limit
    write_json(
        run_dir / "perception_scores.json",
        {
            "per_pass": per_pass,
            "missed": missed,
            "worst": worst,
            "passed": passed,
            "drift": drifts,
            "drift_passed": drift_ok if gated else None,
            "depth_signed_errors_m": {str(b): v for b, v in sorted(depth.items())},
            "depth_error_m": {str(b): e for b, e in depth_err.items()},
            "depth_3m_passed": depth_ok if 3.0 in depth_err else None,
        },
    )
    print(
        f"{len(scored)} obstacle sightings at <= {lc['max_range_m']} m scored, {len(missed)} missed"
    )
    if worst:
        print(f"worst median error {worst[1]:.3f} m ({worst[0]}), gate {limit} m")
    if missed:
        print(f"MISSED (R-07): {', '.join(missed)}")
    print(f"P3-T2: {'PASS' if passed else 'FAIL'}")
    if drift_worst:
        fr = [d["drift_frac"] for d in gated.values()]
        print(
            f"ego drift over {len(gated)} straight passes: median "
            f"{100 * float(np.median(fr)):.1f}%, worst {100 * drift_worst[1]['drift_frac']:.1f}% "
            f"({drift_worst[0]}), gate {100 * drift_limit:.0f}%. "
            f"P4-T2: {'PASS' if drift_ok else 'FAIL'}"
        )
    other = {k: v for k, v in drifts.items() if v["shape"] != "straight"}
    if other:
        print(
            f"  not gated ({len(other)} curved/untagged; displacement understates the path): "
            + ", ".join(f"{k} {100 * v['drift_frac']:.1f}%" for k, v in sorted(other.items()))
        )
    if drifts and not gated:
        print("  no pass tagged shape: straight in gt.yaml, so P4-T2 is not measured")
    for b, e in depth_err.items():
        tag = f" (gate {d3_limit} m: {'PASS' if depth_ok else 'FAIL'})" if b == 3.0 else ""
        print(f"depth error at {b:g} m: {e:.3f} m over {len(depth[b])} sightings{tag}")
    print(f"run folder: {run_dir}")

    rel = run_dir.relative_to(repo_root()) if run_dir.is_relative_to(repo_root()) else run_dir
    if worst and not args.no_metrics:
        append_metric(
            task="P3-T2",
            metric="cone_position_error_m",
            value=worst[1],
            split="dev",
            run_dir=str(rel),
            notes=f"worst of {len(scored)} at <= {lc['max_range_m']} m; {len(missed)} missed",
        )
    if drift_worst and not args.no_metrics:
        append_metric(
            task="P4-T2",
            metric="egomotion_drift_frac",
            value=drift_worst[1]["drift_frac"],
            split="dev",
            run_dir=str(rel),
            notes=f"worst of {len(gated)} straight passes ({drift_worst[0]}); "
            f"{len(other)} curved/untagged not gated",
        )
    if not args.no_metrics:
        for b, e in depth_err.items():
            append_metric(
                task="P4-T1",
                metric=f"depth_error_m_at_{b:g}m",
                value=e,
                split="dev",
                run_dir=str(rel),
                notes=f"median |range error| over {len(depth[b])} sightings, parked starts",
            )
    return 0 if passed and drift_ok and depth_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
