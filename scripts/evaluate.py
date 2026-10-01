#!/usr/bin/env python
"""Score passes: VSCS vs the single-box baseline on the §11 risk metrics (P3-T6, P5-T2).

Thin CLI only - logic lives in src/vscs/eval/evaluate.py (CLAUDE.md section 3).

    python scripts/evaluate.py --passes <passes.yaml> --split dev [--log-metrics]
    python scripts/evaluate.py --passes <passes.yaml> --split test --final --log-metrics

passes.yaml lists, per pass: id, vscs (a scripts/risk.py run folder), baseline (the
--baseline run folder), and truth: component, t_event_ns, event_onsets_ns.

R-09: --split test runs ONCE, with --final; afterwards eval.yaml records the date and any
further test run is refused. Writes evaluation.json to data/processed/eval/<run>/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vscs.common.config import config_dir, load_config, repo_root
from vscs.common.io import append_metric, make_run_dir, read_jsonl, write_json
from vscs.common.log import setup_logging
from vscs.common.types import RiskFrame
from vscs.eval.evaluate import PassTruthFull, guard_split, mark_test_split_used, score_system


def _frames(run: Path) -> list[RiskFrame]:
    return [RiskFrame(**r) for r in read_jsonl(Path(run) / "risk_frames.jsonl")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]),
    )
    parser.add_argument("--passes", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--final", action="store_true", help="the one permitted test-split run")
    parser.add_argument("--log-metrics", action="store_true")
    parser.add_argument("--eval-config", type=Path, default=None, help="default: configs/eval.yaml")
    parser.add_argument("--out-root", type=Path, default=None)
    args = parser.parse_args(argv)

    eval_path = args.eval_config or (config_dir() / "eval.yaml")
    ev = load_config(str(eval_path))
    try:
        guard_split(args.split, ev["splits"], final=args.final)
    except PermissionError as exc:
        print(f"refused: {exc}")
        return 2

    spec = yaml.safe_load(args.passes.read_text(encoding="utf-8"))
    vscs, base = [], []
    for p in spec["passes"]:
        t = p["truth"]
        truth = PassTruthFull(
            str(p["id"]),
            t["component"],
            int(t["t_event_ns"]),
            [int(x) for x in t.get("event_onsets_ns", [])],
        )
        vscs.append((_frames(p["vscs"]), truth))
        base.append((_frames(p["baseline"]), truth))

    run_dir = make_run_dir(
        "eval", config={"split": args.split, "passes": str(args.passes)}, root=args.out_root
    )
    setup_logging(run_dir=run_dir, level=logging.INFO)
    rm = ev["risk_metrics"]
    result = {
        "split": args.split,
        "vscs": score_system(vscs, rm["flag_level"], rm["false_alarm_horizon_s"], attributes=True),
        "baseline": score_system(
            base, rm["flag_level"], rm["false_alarm_horizon_s"], attributes=False
        ),
    }
    write_json(run_dir / "evaluation.json", result)
    for system in ("vscs", "baseline"):
        print(system, {k: v for k, v in result[system].items()})

    if args.log_metrics:
        rel = run_dir.relative_to(repo_root()) if run_dir.is_relative_to(repo_root()) else run_dir
        task = "P5-T2" if args.split == "test" else "P3-T6"
        for system in ("vscs", "baseline"):
            for metric, value in result[system].items():
                if isinstance(value, (int, float)) and metric != "n_passes":
                    append_metric(
                        task=task,
                        metric=f"{metric}__{system}",
                        value=float(value),
                        split=args.split,
                        run_dir=str(rel),
                        notes=f"{result[system]['n_passes']} passes",
                    )
    if args.split == "test":
        mark_test_split_used(eval_path, ev["splits"])
        print("test split marked as used in", eval_path)
    print(f"run folder: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
