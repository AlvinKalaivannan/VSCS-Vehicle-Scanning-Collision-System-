#!/usr/bin/env python
"""Labelled 3D points for a scan: 2D masks -> 3D fusion -> cleanup (P2-T3, P2-T5).

Thin CLI only - logic lives in src/vscs/seg/ (CLAUDE.md section 3).

    python scripts/fuse.py --dense-run <dense run folder> --seg-run <seg run folder>
                           --sfm-to-veh <sfm_to_veh.json>

--dense-run is a scripts/dense.py run (its dense/ workspace: fused.ply, undistorted
pinhole cameras in sparse_txt/, depth maps). --seg-run is a scripts/seg.py run made on
that workspace's *undistorted* images (dense/images), so masks and depth maps line up.
--sfm-to-veh is P1-T6's result, {"scale": metres per SfM unit, "T_veh_world": 4x4}; it
puts everything in the metric veh frame.

Writes a new data/processed/seg/<run>/ folder: fused.npz (labels, confidence, n_obs),
cleaned.npz (points, labels, names - the input of scripts/export_model.py) and
fuse_summary.json. The fusion is the developer's seg/fusion3d.py (P2-T3); until it exists
this script stops with a clear message (exit code 3).
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

from vscs.common.config import load_config
from vscs.common.io import make_run_dir, write_json
from vscs.common.log import setup_logging
from vscs.recon.colmap_io import read_model
from vscs.seg.cleanup import cleanup_labels
from vscs.seg.fuse_inputs import iter_views, read_sfm_to_veh
from vscs.seg.labels import NONE_LABEL

#: Replaced in tests; None means the developer's vscs.seg.fusion3d (P2-T3).
FUSION = None


def _resolved(run: Path) -> dict:
    path = run / "resolved_config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no resolved_config.yaml in {run} - is this a run folder?")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _counts(labels: np.ndarray, names: list[str]) -> dict[str, int]:
    return {n: int((labels == i).sum()) for i, n in enumerate(names)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]),
    )
    parser.add_argument("--dense-run", type=Path, required=True)
    parser.add_argument("--seg-run", type=Path, required=True)
    parser.add_argument("--sfm-to-veh", type=Path, required=True, help="P1-T6 result (json)")
    parser.add_argument("--out-root", type=Path, default=None, help="default: data/processed")
    args = parser.parse_args(argv)

    ws = args.dense_run / "dense"
    needed = (ws / "fused.ply", args.seg_run / "labels", args.sfm_to_veh)
    for p in needed:
        if not p.exists():
            print(f"not found: {p}")
            return 2
    if not (ws / "sparse_txt" / "cameras.txt").is_file():
        print(
            f"no text model in {ws / 'sparse_txt'}. Convert the undistorted one first:\n"
            f"  colmap model_converter --input_path {ws / 'sparse'} "
            f"--output_path {ws / 'sparse_txt'} --output_type TXT"
        )
        return 2
    try:
        names = list(_resolved(args.seg_run)["seg"]["components"])
        geometric = bool(_resolved(args.dense_run)["dense"].get("geom_consistency", True))
        sfm_to_veh = read_sfm_to_veh(args.sfm_to_veh)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"cannot read the inputs: {exc}")
        return 2

    try:
        fusion = FUSION or importlib.import_module("vscs.seg.fusion3d")
        # A one-point probe, so a stub is found before a run folder is made.
        fusion.decide(np.zeros((1, 2), np.int64), min_observations=1, min_vote_fraction=0.5)
    except (ImportError, NotImplementedError) as exc:
        # Missing on main, a stub on the draft branch: either way, not written yet.
        print(f"the fusion is not written yet ({exc}): seg/fusion3d.py is the developer's P2-T3")
        return 3

    seg_cfg = load_config("seg")
    fcfg, ccfg = seg_cfg["fusion3d"], seg_cfg["cleanup"]
    run_dir = make_run_dir(
        "seg",
        config={
            "fusion3d": fcfg,
            "cleanup": ccfg,
            "inputs": {k: str(v) for k, v in vars(args).items() if k != "out_root"},
        },
        root=args.out_root,
    )
    setup_logging(run_dir=run_dir, level=logging.INFO)

    import trimesh

    points = sfm_to_veh.points(np.asarray(trimesh.load(ws / "fused.ply").vertices))
    model = read_model(ws / "sparse_txt")
    missing: list[str] = []
    views = iter_views(
        model,
        ws,
        args.seg_run / "labels",
        sfm_to_veh,
        "geometric" if geometric else "photometric",
        missing,
        n_classes=len(names),
    )
    # Votes are counts, so they add over any split of the views: batching bounds memory
    # (one mask + one depth map per view) without changing the result.
    tol = float(fcfg["occlusion_depth_tolerance_m"])
    votes = np.zeros((len(points), len(names) + 1), np.int64)
    batch, n_views = [], 0
    try:
        for v in views:
            batch.append(fusion.PosedView(v.K, v.T_cam_world, v.labels, v.depth))
            n_views += 1
            if len(batch) == int(fcfg["views_per_batch"]):
                votes += fusion.accumulate_votes(points, batch, len(names), tol)
                batch = []
        if batch:
            votes += fusion.accumulate_votes(points, batch, len(names), tol)
    except NotImplementedError as exc:
        print(f"the fusion is not finished yet ({exc}): seg/fusion3d.py (P2-T3)")
        return 3
    if n_views == 0:
        print("no view has both a mask and a depth map - check the seg run's frames")
        return 2
    labels, confidence, n_obs = fusion.decide(
        votes,
        min_observations=int(fcfg["min_observations_per_point"]),
        min_vote_fraction=float(fcfg["min_vote_fraction"]),
    )
    np.savez_compressed(
        run_dir / "fused.npz",
        points=points,
        labels=labels,
        confidence=confidence,
        n_obs=n_obs,
        names=np.array(names),
    )
    cleaned = cleanup_labels(points, labels, ccfg)
    np.savez_compressed(
        run_dir / "cleaned.npz", points=points, labels=cleaned.labels, names=np.array(names)
    )
    summary = {
        "n_points": len(points),
        "n_views": n_views,
        "n_views_missing_mask_or_depth": len(missing),
        "views_missing": missing[:20],
        "unlabelled_fraction_fused": float(np.mean(labels == NONE_LABEL)),
        "unlabelled_fraction_cleaned": float(np.mean(cleaned.labels == NONE_LABEL)),
        "points_per_component_fused": _counts(labels, names),
        "points_per_component_cleaned": _counts(cleaned.labels, names),
        "cleanup_reassigned": cleaned.n_reassigned,
        "cleanup_cleared": cleaned.n_cleared,
        "scale_m_per_sfm_unit": sfm_to_veh.scale,
    }
    write_json(run_dir / "fuse_summary.json", summary)

    print(f"{len(points):,} points from {n_views} views ({len(missing)} skipped)")
    for n in names:
        print(
            f"  {n:28s} {summary['points_per_component_fused'][n]:>8,} fused  "
            f"{summary['points_per_component_cleaned'][n]:>8,} cleaned"
        )
    never = [n for n in names if summary["points_per_component_cleaned"][n] == 0]
    if never:
        print(f"\nNO POINTS (log as P2-T3 failures): {', '.join(never)}")
    print(f"\nnext: python scripts/export_model.py --labelled {run_dir / 'cleaned.npz'} ...")
    print(f"run folder: {run_dir}")
    return 1 if never else 0


if __name__ == "__main__":
    raise SystemExit(main())
