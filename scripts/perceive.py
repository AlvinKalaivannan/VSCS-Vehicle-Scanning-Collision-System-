#!/usr/bin/env python
"""Per-drive perception: extracted frames -> obstacle stream (P3-T2, P4-T3, P4-T4).

Thin CLI only - logic lives in src/vscs/perception/ (CLAUDE.md section 3).

    python scripts/perceive.py --frames-run data/processed/capture/<run>
                               [--ego vo|stationary|file] [--ego-poses <poses.jsonl>]

Each frame is undistorted (P0-T7 intrinsics), run through the RT-DETR detector (ADR 0011),
then perception: static objects into the persistent height map, people/vehicles/cyclists
into the tracker. Writes obstacles.jsonl (one §4.2 Obstacle per line, in each frame's
veh frame) and detections.jsonl into a new data/processed/perception/<run>/.

Needs the camera calibrated (configs/capture.yaml intrinsics.calibrated) and its mount
measured (mount.measured), and refuses otherwise. Ego-motion (--ego): "vo" (default)
estimates it from the road surface (ground-plane visual odometry, P4-T2); "stationary"
fixes the van (the P4-T4 staged walk-behind clip); "file" reads --ego-poses. The detector
needs torch: run it on Colab or in a GPU env.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vscs.capture.frames import read_frames_index
from vscs.common.config import load_config
from vscs.common.frames import look_at_T_world_cam
from vscs.common.io import make_run_dir, read_jsonl, write_jsonl
from vscs.common.log import get_logger, setup_logging
from vscs.perception.egomotion import GroundVO
from vscs.perception.pipeline import Perception

logger = get_logger("scripts.perceive")


def make_detector(detect_cfg, device):
    """The real detector. Replaced in tests; needs torch + transformers."""
    from vscs.perception.detect import RTDetrDetector

    return RTDetrDetector(detect_cfg, device=device)


def load_image(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]),
    )
    parser.add_argument("--frames-run", type=Path, required=True)
    parser.add_argument("--ego", choices=["vo", "stationary", "file"], default="vo")
    parser.add_argument(
        "--ego-poses", type=Path, default=None, help="with --ego file: t_ns, T_world_veh (16)"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-root", type=Path, default=None)
    args = parser.parse_args(argv)

    cap = load_config("capture")
    intr, mount = cap["intrinsics"], cap["mount"]
    if not intr.get("calibrated") or intr.get("K") is None:
        print("camera not calibrated: run scripts/calibrate.py first (P0-T7)")
        return 2
    if not mount.get("measured"):
        print("camera mount not measured: fill configs/capture.yaml mount (lot day)")
        return 2
    K = np.asarray(intr["K"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(intr.get("dist") or [], dtype=np.float64)
    w, h = (int(v) for v in intr["image_size"])
    T_veh_cam = look_at_T_world_cam(
        np.asarray(mount["position_veh_m"]), np.asarray(mount["look_at_veh_m"])
    )

    pcfg = load_config("perception")
    run_dir = make_run_dir(
        "perception", config={"perception": pcfg, "capture": cap}, root=args.out_root
    )
    setup_logging(run_dir=run_dir, level=logging.INFO)

    poses = {}
    vo = None
    if args.ego == "file":
        if not args.ego_poses:
            print("--ego file needs --ego-poses")
            return 2
        poses = {
            int(r["t_ns"]): np.asarray(r["T_world_veh"], dtype=np.float64).reshape(4, 4)
            for r in read_jsonl(args.ego_poses)
        }
    elif args.ego == "vo":
        eg = pcfg["egomotion"]
        vo = GroundVO(K, T_veh_cam, float(pcfg["depth"]["max_range_m"]), eg["vo"], int(eg["seed"]))
    else:
        logger.warning("--ego stationary: the van is assumed not to move for the whole run")

    detector = make_detector(pcfg["detect"], args.device)
    perception = Perception(pcfg, K, T_veh_cam, (w, h))
    obstacles, detections = [], []
    import cv2

    for t_ns, path in read_frames_index(args.frames_run):
        img = load_image(path)
        if dist.size:
            img = cv2.undistort(img, K, dist)  # the ground-plane geometry assumes a pinhole
        boxes = detector.detect(img)
        detections.extend(
            {"t_ns": t_ns, "cls": b.cls, "score": b.score, "xyxy": list(b.xyxy)} for b in boxes
        )
        if vo is not None:
            vo.step(img)
            T_world_veh = vo.T_world_veh
        else:
            T_world_veh = poses.get(t_ns, np.eye(4))
        obstacles.extend(perception.step(t_ns, boxes, T_world_veh))

    write_jsonl(run_dir / "obstacles.jsonl", obstacles)
    write_jsonl(run_dir / "detections.jsonl", detections)
    if vo is not None and vo.lost:
        logger.warning("visual odometry lost the road on %d frame(s); motion was coasted", vo.lost)
    print(
        f"{len(detections)} detections -> {len(obstacles)} obstacle records\nrun folder: {run_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
