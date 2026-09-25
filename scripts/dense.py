#!/usr/bin/env python
"""COLMAP dense reconstruction from a P1-T5 sparse model (P1-T7). Runs on Colab.

Thin CLI only - logic lives in src/vscs/recon/dense.py (CLAUDE.md section 3).

    python scripts/dense.py --recon-run <recon run folder> --images <frames folder>

Needs a CUDA-enabled COLMAP: patch_match_stereo has no CPU path, which is why this stage
runs on the Colab GPU rather than the laptop (ADR 0003). The script checks the build first
and refuses to start on one without CUDA.

Writes a new run folder under data/processed/recon/ holding the dense workspace,
fused.ply and dense_result.json.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vscs.common.config import load_config
from vscs.common.io import make_run_dir, read_json, write_json
from vscs.common.log import setup_logging
from vscs.recon.dense import run_dense


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]),
    )
    parser.add_argument("--recon-run", type=Path, required=True, help="a P1-T5 recon run folder")
    parser.add_argument("--images", type=Path, required=True, help="the frames SfM was run on")
    parser.add_argument("--colmap", default="colmap", help="COLMAP executable (CUDA build)")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="parent of the run folder (default: data/processed)",
    )
    args = parser.parse_args(argv)

    summary = args.recon_run / "sfm_result.json"
    if not summary.is_file():
        print(f"no sfm_result.json in {args.recon_run} - is this a scripts/recon.py run folder?")
        return 2
    model_dir = Path(read_json(summary)["model_dir"])
    if not model_dir.is_dir():
        # The recon run may have moved (e.g. laptop -> Drive); look for it beside the summary.
        model_dir = args.recon_run / "sparse_txt" / model_dir.name
    if not model_dir.is_dir():
        print(f"sparse model not found at {model_dir}")
        return 2

    dense_cfg = load_config("recon")["dense"]
    run_dir = make_run_dir(
        "recon", config={"dense": dense_cfg, "recon_run": str(args.recon_run)}, root=args.out_root
    )
    setup_logging(run_dir=run_dir, level=logging.INFO)

    result = run_dense(args.images, model_dir, run_dir / "dense", dense_cfg, colmap=args.colmap)
    write_json(run_dir / "dense_result.json", {**result.__dict__, "plausible": result.plausible})
    print(f"\nfused cloud: {result.n_points:,} points -> {result.ply_path}")
    print(f"run folder: {run_dir}")
    return 0 if result.plausible else 1


if __name__ == "__main__":
    raise SystemExit(main())
