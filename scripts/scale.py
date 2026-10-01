#!/usr/bin/env python
"""Metric scale, ground plane and vehicle frame for a scan; checked against the tape (P1-T6).

Thin CLI only - logic lives in src/vscs/recon/metric.py (CLAUDE.md section 3).

    python scripts/scale.py --recon-run <recon run folder> --images <frames SfM ran on>
                            --measured <measurements.yaml>

--measured holds scan day's hand measurements in metres, keyed as recon.yaml
scale.validate_against (length, width, height, wheelbase), e.g. "length: 5.12". Also needs
vehicle_frame.rear_overhang_m and the measured scale.marker.side_length_m in recon.yaml.

Writes a new data/processed/recon/<run>/ folder: sfm_to_veh.json (the input of
scripts/fuse.py), metric_result.json and body_points.ply (the points taken as the van, in
veh, for a visual check). Unless --no-metrics, appends scale_error_m (P1-T6). Exit code 1
if the ±2 cm gate fails or the markers disagree. The scale maths is the developer's
recon/scale.py; until it exists this script stops with a clear message (exit code 3).
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vscs.common.config import load_config, repo_root
from vscs.common.io import append_metric, make_run_dir, read_json, write_json
from vscs.common.log import setup_logging
from vscs.recon.colmap_io import read_model
from vscs.recon.marker_tracks import collect_marker_tracks_from_config
from vscs.recon.metric import metric_frame

#: Replaced in tests; None means the developer's vscs.recon.scale (P1-T6).
SCALE = None


def _measured(path: Path, wanted: list[str]) -> dict[str, float]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = {k: float(raw[k]) for k in wanted if raw.get(k) is not None}
    if not out:
        raise ValueError(f"{path} holds none of {wanted} (metres)")
    if any(not (0.1 < v < 20.0) for v in out.values()):
        raise ValueError(f"{path}: {out} - are these metres?")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]),
    )
    parser.add_argument("--recon-run", type=Path, required=True, help="a P1-T5 recon run folder")
    parser.add_argument("--images", type=Path, required=True, help="the frames SfM was run on")
    parser.add_argument("--measured", type=Path, required=True, help="tape measurements (yaml)")
    parser.add_argument("--out-root", type=Path, default=None, help="default: data/processed")
    parser.add_argument("--no-metrics", action="store_true")
    args = parser.parse_args(argv)

    recon_cfg = load_config("recon")
    summary = args.recon_run / "sfm_result.json"
    for p in (summary, args.images, args.measured):
        if not p.exists():
            print(f"not found: {p}")
            return 2
    model_dir = Path(read_json(summary)["model_dir"])
    if not model_dir.is_dir():  # the run may have moved (laptop -> Drive)
        model_dir = args.recon_run / "sparse_txt" / model_dir.name
    if not model_dir.is_dir():
        print(f"sparse model not found at {model_dir}")
        return 2
    if recon_cfg["vehicle_frame"].get("rear_overhang_m") is None:
        print(
            "configs/recon.yaml vehicle_frame.rear_overhang_m is not set: measure the rear "
            "bumper to rear-axle-centre distance on scan day (docs/capture_checklists.md)"
        )
        return 2
    try:
        measured = _measured(args.measured, list(recon_cfg["scale"]["validate_against"]))
    except (ValueError, TypeError) as exc:
        print(f"cannot read the measurements: {exc}")
        return 2

    try:
        scale_mod = SCALE or importlib.import_module("vscs.recon.scale")
        scale_mod.scale_points(np.zeros((1, 3)), 1.0)  # a probe, before a run folder is made
    except (ImportError, NotImplementedError) as exc:
        print(f"the scale maths is not written yet ({exc}): recon/scale.py is the developer's")
        return 3

    run_dir = make_run_dir(
        "recon",
        config={"recon": recon_cfg, "recon_run": str(args.recon_run), "measured": measured},
        root=args.out_root,
    )
    setup_logging(run_dir=run_dir, level=logging.INFO)

    model = read_model(model_dir)
    report = collect_marker_tracks_from_config(model, args.images, recon_cfg)
    try:
        result, body_veh = metric_frame(model, report, recon_cfg, measured, scale_mod)
    except NotImplementedError as exc:
        print(f"the scale maths is not finished yet ({exc}): recon/scale.py (P1-T6)")
        return 3
    except ValueError as exc:
        write_json(
            run_dir / "metric_result.json",
            {"error": str(exc), "detection_fraction": report.detection_fraction},
        )
        print(f"P1-T6 failed: {exc}\nrun folder: {run_dir}")
        return 1

    import trimesh

    write_json(run_dir / "metric_result.json", result.to_json())
    write_json(
        run_dir / "sfm_to_veh.json",
        {"scale": result.scale, "T_veh_world": result.T_veh_world, "source_run": str(run_dir)},
    )
    trimesh.PointCloud(body_veh).export(run_dir / "body_points.ply")

    print(f"scale {result.scale:.6f} m per SfM unit from markers {sorted(result.per_marker_scale)}")
    print(
        f"  marker spread {100 * result.spread_frac:.2f}%  consistent: {result.markers_consistent}"
    )
    for k, v in result.dimensions_m.items():
        err = result.errors_m.get(k)
        tape = (
            f"tape {result.measured_m[k]:.3f}  error {100 * err:+.1f} cm" if err is not None else ""
        )
        print(f"  {k:9s} {v:.3f} m  {tape}")
    if result.not_validated:
        print(f"  not validated here: {', '.join(result.not_validated)}")
        if "wheelbase" in result.not_validated:
            print("    wheelbase: run scripts/check_wheelbase.py after P2-T5 (ADR 0013)")
    print(
        f"scale_error_m {result.scale_error_m:.4f} (gate {result.max_dimension_error_m}): "
        f"{'PASS' if result.passed else 'FAIL'}"
    )
    if result.detection_fallback_triggered:
        print(
            f"markers found in only {100 * result.detection_fraction:.0f}% of frames: the §8.2 "
            "scale fallback trigger fired. Switching is the developer's decision (log in RISKS)."
        )
    print(f"check {run_dir / 'body_points.ply'} is the van and only the van")
    print(f"run folder: {run_dir}")

    if not args.no_metrics:
        rel = run_dir.relative_to(repo_root()) if run_dir.is_relative_to(repo_root()) else run_dir
        append_metric(
            task="P1-T6",
            metric="scale_error_m",
            value=result.scale_error_m,
            split="scan",
            run_dir=str(rel),
            notes=f"validated {sorted(result.errors_m)}; not validated {result.not_validated}",
        )
        print("metric appended - run scripts/report.py to refresh docs/REPORT.md")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
