# VSCS runbook: from capture day to evaluation

The order to run things in once real data exists. Every command below is checked against
the script's own `--help` (2026-10-01). **Where** says which machine; **needs** says what
must exist first. The scripts refuse to run when a prerequisite is missing; they do not
guess.

Two links in the offline chain are **not runnable yet**, because they wrap the developer's
core-module drafts (pair mode, CLAUDE.md §0). They are marked ⛔ below.

## 0. Before capture day (Phase 0)

| Step | Command | Where | Needs |
|---|---|---|---|
| Print marker boards | `python scripts/make_markers.py --out <dir>` (print at 100%, measure the printed side) | laptop | — |
| Pre-flight a test clip | `python scripts/check_capture.py --video <clip> --imu <gyro.csv>` | laptop | phone clip + gyro log |
| Calibrate (P0-T7) | `python scripts/calibrate.py --images <checkerboard dir> --device <name>` | laptop | checkerboard photos, focus/exposure locked; writes `capture.yaml intrinsics` only if < 0.5 px |
| Push | `git push` (after the developer's review and the security check) | laptop | everything else on Colab clones the repo |

## 1. Capture day → raw data (P1-T2, P1-T3, P1-T8)

| Step | Command | Where | Notes |
|---|---|---|---|
| Ingest each file | `python scripts/ingest.py --src <path> --kind scan\|lot\|calib [--stratum pole\|curb\|box\|mixed]` | laptop | Checksums into `data/MANIFEST.md`; assigns lot passes to dev/test once (R-09); verify the backups |
| Record the camera mount | edit `configs/capture.yaml mount` (`position_veh_m`, `look_at_veh_m`, `measured: true`) | laptop | Tape-measured on lot day. `perceive.py` refuses without it |
| Record lot ground truth | `data/raw/lot_*/gt.yaml` (obstacle positions) | laptop | Feeds the `truth` blocks of `passes.yaml` (step 4) |

## 2. The van model, once (P1-T4 … P2-T8)

| Step | Command | Where | Produces |
|---|---|---|---|
| Frames | `python scripts/extract_frames.py --video data/raw/<id>/<clip>` | laptop | `data/processed/capture/<run>/` (frames + `frames.jsonl`, real timestamps) |
| SfM (P1-T5) | `python scripts/recon.py --frames-run data/processed/capture/<run>` | laptop (CPU COLMAP) | sparse model; gates ≥ 90% registered, < 1.5 px |
| ⛔ Scale + ground + vehicle frame (P1-T6) | not yet a command: needs `recon/scale.py` (the developer's draft) | — | `T_veh_sfm` (similarity), `scale_error_m` |
| Dense (P1-T7) | `notebooks/colab/10_dense.ipynb` → `scripts/dense.py --recon-run <run> --images <frames>` | **Colab** (CUDA COLMAP) | `dense/`: `fused.ply`, undistorted images in `images/`, depth maps, and the undistorted pinhole cameras as text in `sparse_txt/` |
| 2D masks (P2-T2) | `notebooks/colab/20_seg.ipynb` → `scripts/seg.py --frames <dense run>/dense/images --sam2-dir <sam2>` | **Colab** | per-frame label PNGs + 20 check overlays. **Run it on the dense workspace's *undistorted* images**: the depth maps are in undistorted pixels, so masks from the original frames would not line up for fusion |
| ⛔ 3D fusion + cleanup (P2-T3, P2-T5) | `python scripts/fuse.py --dense-run <dense run> --seg-run <seg run> --sfm-to-veh <sfm_to_veh.json>` | laptop | `fused.npz`, `cleaned.npz` (points in `veh`, labels, names), `fuse_summary.json`. Stops with exit 3 until `seg/fusion3d.py` (the developer's draft) exists. `sfm_to_veh.json` is `{"scale": m per SfM unit, "T_veh_world": 4×4}` from P1-T6, which has no command yet. Masks must be indexed in the seg run's component order (read from its `resolved_config.yaml`) |
| Collision model (P2-T6…T8) | `python scripts/export_model.py --labelled <cleaned.npz> --scale-error-m <P1-T6 value>` | laptop | URDF + meshes + `components.yaml`; hinges stay fixed until measured (P2-T7) |

## 3. Each recorded drive (P3, P4)

| Step | Command | Where | Produces |
|---|---|---|---|
| Frames | `python scripts/extract_frames.py --video data/raw/<lot id>/<clip>` | laptop | a capture run |
| Perception | `notebooks/colab/30_perceive.ipynb` → `scripts/perceive.py --frames-run <run> [--ego vo\|stationary\|file] [--imu <gyro.csv> --imu-offset-ms <ms>]` | **Colab** (detector needs torch) | `obstacles.jsonl`, `ego.jsonl`, `detections.jsonl`. `--ego stationary` for the P4-T4 walk-behind clip. The IMU offset is the P1-T4 sync value |
| Risk | `python scripts/risk.py --model <model run> --perception <perceive run>` and again with `--baseline` | laptop | `risk_frames.jsonl` for VSCS and for the single box. Needs `risk/sweep.py` (the developer's P3-T4 draft): stops with a clear message until then |
| Look at it | `python scripts/view.py --model <m> --risk <frames.jsonl> --obstacles <obs.jsonl>` → `rerun drive.rrd` | laptop | Rerun recording |
| Driver replay | `python scripts/replay.py --model <m> --risk <frames.jsonl>` | laptop | PNG frames + `replay.wav` (advisory banner on every frame) |

## 4. Evaluation (P3-T6, P5)

| Step | Command | Where | Notes |
|---|---|---|---|
| Dev scoring (tune here only) | `python scripts/evaluate.py --passes <passes.yaml> --split dev [--log-metrics]` | laptop | VSCS vs baseline on §11 metrics; baseline attribution is reported as n/a |
| Throughput (P5-T3) | `python scripts/benchmark.py --frames-run <run> --detector --log-metrics` | **Colab T4** | every figure tagged with its hardware (§9); laptop runs are rehearsals; leave `--log-metrics` off |
| **Final test (P5-T2), once** | `python scripts/evaluate.py --passes <passes.yaml> --split test --final --log-metrics` | laptop | R-09: allowed exactly once; afterwards `eval.yaml` records the date and refuses again |
| Report | `python scripts/report.py` | laptop | `docs/REPORT.md` from `metrics/results.jsonl`, pass/fail per §6 gate |

`passes.yaml` lists each pass: `id`, `vscs` (risk run folder), `baseline` (the `--baseline` run
folder), and `truth: {component, t_event_ns, event_onsets_ns}` from the lot ground truth.

## Privacy and licences, every time

- Nothing leaves `data/` without a face/plate blur and the developer's visual check (§9).
- External datasets go through `configs/datasets.yaml`. Check `allowed_uses` before showing
  anything (ADR 0006).
