#!/usr/bin/env python
"""Score a fuse run's 3D labels against the hand-labelled gold subset (P2-T3, P2-T4, P2-T5).

Thin CLI only - logic lives in src/vscs/eval/labels3d.py (CLAUDE.md section 3).

    python scripts/score_labels.py --fuse-run <fuse run folder> --gold <gold.txt>

--gold is the P2-T4 subset, labelled in CloudCompare on the run's points_veh.ply and
exported as ASCII "x y z label" (label = component index in the run's names, -1 = none).
Scores fused.npz (P2-T3: mean IoU >= 0.70) and cleaned.npz (P2-T5: must not be worse),
and counts the gold points (P2-T4: >= 2000). Writes label_scores.json into a new
data/processed/eval/<run>/ and, unless --no-metrics, appends component_iou_mean,
component_iou_mean_cleaned and labeled_points_count. Exit code 1 if any gate fails.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vscs.common.config import load_config, repo_root
from vscs.common.io import append_metric, make_run_dir, write_json
from vscs.common.log import setup_logging
from vscs.eval.labels3d import read_gold_points, score_labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]),
    )
    parser.add_argument("--fuse-run", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=None, help="default: data/processed")
    parser.add_argument("--no-metrics", action="store_true")
    args = parser.parse_args(argv)

    for p in (args.fuse_run / "fused.npz", args.fuse_run / "cleaned.npz", args.gold):
        if not p.is_file():
            print(f"not found: {p}")
            return 2
    eval_cfg = load_config("eval")
    th = eval_cfg["thresholds"]
    tol = float(eval_cfg["labels3d"]["match_tolerance_m"])
    try:
        gold_pts, gold_lab = read_gold_points(args.gold)
        scores = {}
        for stage in ("fused", "cleaned"):
            with np.load(args.fuse_run / f"{stage}.npz", allow_pickle=False) as z:
                names = [str(n) for n in z["names"]]
                scores[stage] = score_labels(
                    z["points"], z["labels"], names, gold_pts, gold_lab, tol
                )
    except ValueError as exc:
        print(f"cannot score: {exc}")
        return 2

    run_dir = make_run_dir(
        "eval",
        config={"labels3d": eval_cfg["labels3d"], "fuse_run": str(args.fuse_run)},
        root=args.out_root,
    )
    setup_logging(run_dir=run_dir, level=logging.INFO)
    fused, cleaned = scores["fused"], scores["cleaned"]
    gates = {
        "P2-T3 mean IoU": fused.mean_iou >= float(th["component_iou_mean"]["limit"]),
        "P2-T4 gold points": fused.n_gold >= float(th["labeled_points_count"]["limit"]),
        "P2-T5 cleanup not worse": cleaned.mean_iou >= fused.mean_iou,
    }
    write_json(
        run_dir / "label_scores.json",
        {"fused": asdict(fused), "cleaned": asdict(cleaned), "gates": gates},
    )

    print(f"gold: {fused.n_gold} points, {fused.n_matched} matched to the cloud")
    if fused.n_matched < fused.n_gold:
        print(f"  {fused.n_gold - fused.n_matched} gold points have no twin in the cloud")
    for n in names:
        f, c = fused.iou.get(n), cleaned.iou.get(n)
        if f is None and c is None:
            print(f"  {n:28s} not in the gold set")
            continue
        fs = "-" if f is None else f"{f:.3f}"
        cs = "-" if c is None else f"{c:.3f}"
        print(f"  {n:28s} fused {fs}  cleaned {cs}  ({fused.gold_per_component[n]} gold)")
    print(f"mean IoU: fused {fused.mean_iou:.3f}, cleaned {cleaned.mean_iou:.3f}")
    for g, ok in gates.items():
        print(f"  {g}: {'PASS' if ok else 'FAIL'}")
    print(f"run folder: {run_dir}")

    if not args.no_metrics:
        rel = run_dir.relative_to(repo_root()) if run_dir.is_relative_to(repo_root()) else run_dir
        for task, metric, value, notes in (
            ("P2-T4", "labeled_points_count", fused.n_gold, f"{fused.n_matched} matched"),
            ("P2-T3", "component_iou_mean", fused.mean_iou, f"{len(fused.iou)} components"),
            (
                "P2-T5",
                "component_iou_mean_cleaned",
                cleaned.mean_iou,
                "gate: not below component_iou_mean of the same run",
            ),
        ):
            append_metric(
                task=task, metric=metric, value=value, split="scan", run_dir=str(rel), notes=notes
            )
        print("metrics appended - run scripts/report.py to refresh docs/REPORT.md")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
