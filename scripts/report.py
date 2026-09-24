#!/usr/bin/env python
"""Regenerate docs/REPORT.md from metrics/results.jsonl.

Thin CLI only - all logic lives in src/vscs/eval/report.py (CLAUDE.md section 3).

    python scripts/report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable from a bare checkout, with no editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vscs.eval.report import generate_report, load_metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="path to results.jsonl (default: metrics/results.jsonl)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output markdown path (default: docs/REPORT.md)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if any measured metric fails its acceptance gate",
    )
    args = parser.parse_args(argv)

    out = generate_report(metrics_file=args.metrics, out_path=args.out)
    n = len(load_metrics(args.metrics))
    print(f"wrote {out} ({n} measurement{'' if n == 1 else 's'})")

    if args.check:
        from vscs.common.config import load_config
        from vscs.eval.report import summarize

        thresholds = load_config("eval").get("thresholds") or {}
        summaries = summarize(load_metrics(args.metrics), thresholds)
        failing = [s.metric for s in summaries if s.status == "fail"]
        if failing:
            print(f"FAIL: {len(failing)} metric(s) below their acceptance gate: {failing}")
            return 1
        print("all measured metrics pass their acceptance gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
