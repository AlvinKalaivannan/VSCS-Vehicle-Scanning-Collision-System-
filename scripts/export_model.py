#!/usr/bin/env python
"""Build the collision model from cleaned 3D labels: decompose + URDF export (P2-T6..T8).

Thin CLI only - logic lives in src/vscs/model/build.py (CLAUDE.md section 3).

    python scripts/export_model.py --labelled <cleaned.npz> --scale-error-m 0.013

--scale-error-m is P1-T6's validated dimensional error (metrics/results.jsonl). Writes a new
data/processed/model/<run>/ folder and, unless --no-metrics, appends the worst per-component
volume error as hull_volume_error_frac (P2-T6). Exit code 1 if any component fails the gate.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vscs.common.config import load_config, repo_root
from vscs.common.io import append_metric, make_run_dir
from vscs.common.log import setup_logging
from vscs.model.build import build_collision_model, load_labelled_points


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]),
    )
    parser.add_argument("--labelled", type=Path, required=True, help="npz: points, labels, names")
    parser.add_argument("--scale-error-m", type=float, required=True, help="P1-T6 result")
    parser.add_argument("--out-root", type=Path, default=None, help="default: data/processed")
    parser.add_argument("--no-metrics", action="store_true")
    args = parser.parse_args(argv)

    if not args.labelled.is_file():
        print(f"no labelled points at {args.labelled}")
        return 2
    model_cfg, severity_cfg = load_config("model"), load_config("severity")
    run_dir = make_run_dir(
        "model", config={"model": model_cfg, "severity": severity_cfg}, root=args.out_root
    )
    setup_logging(run_dir=run_dir, level=logging.INFO)

    points, labels, names = load_labelled_points(args.labelled)
    result = build_collision_model(
        points,
        labels,
        names,
        run_dir,
        model_cfg=model_cfg,
        severity_cfg=severity_cfg,
        scale_error_m=args.scale_error_m,
    )
    print()
    for d in result.decomposed:
        print(d.summary())
    if result.unmeasured_joints:
        print(f"\nhinges not yet measured (exported fixed): {', '.join(result.unmeasured_joints)}")
    print(f"\nrun folder: {run_dir}")

    if not args.no_metrics:
        rel = run_dir.relative_to(repo_root()) if run_dir.is_relative_to(repo_root()) else run_dir
        append_metric(
            task="P2-T6",
            metric="hull_volume_error_frac",
            value=result.worst_volume_error,
            split="scan",
            run_dir=str(rel),
            notes=f"{len(result.decomposed)} components, engine {model_cfg['decompose']['engine']}",
        )
        print("metric appended - run scripts/report.py to refresh docs/REPORT.md")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
