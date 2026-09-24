# VSCS - Vehicle Scanning Collision System

> ## ⚠️ This is not a safety device.
> VSCS is an **advisory-only research prototype**. It never controls steering, throttle, or brakes, and no code
> in this repository may send commands to a vehicle. It must never be relied upon to prevent a collision.
> It runs **offline, on recordings** - it is not a deployed in-vehicle system.

VSCS scans a real vehicle from phone video into a metric 3D model, segments that model into **individual
mechanical components** (bumper corners, mirrors, wheels, doors, underbody), and then - on recorded low-speed
manoeuvring footage - estimates **per-component** collision risk instead of the usual single "is the car near
something" bounding box.

The question it exists to answer: *not* "will I hit something?" but **"which part of the vehicle is at risk,
how likely is contact, and how much would it cost?"**

## Why per-component

A single bounding box around a van treats a mirror clipping a pole and a wheel scuffing a curb as the same
event. They are not: they differ in likelihood, in which driver action avoids them, and in repair cost. VSCS
carries a severity weight per component and reports an expected-damage estimate, so the warning can name the
part.

The project's core result is therefore a **comparison against a single-bounding-box baseline** on a held-out
test split (see `docs/REPORT.md`).

## Pipeline

```
Offline, once per vehicle:   capture -> recon -> seg -> model   (URDF + per-component severity)
Per drive:                   capture -> perception -> risk -> ui / eval
```

| Stage | What it does |
|---|---|
| `capture/` | Ingest phone video + IMU; variable-frame-rate-safe frame extraction with real timestamps |
| `recon/` | Structure-from-motion, metric scale from markers, ground plane, vehicle frame |
| `seg/` | 2D component masks lifted to 3D by multi-view label fusion with occlusion tests |
| `model/` | Convex decomposition per component, joints for doors and mirrors, URDF export |
| `perception/` | Depth, occupancy, detection, tracking, ego-motion on driving footage |
| `risk/` | Predicted path fan, swept distance, time-to-contact, contact probability, aggregation |
| `ui/` | Rerun developer view and a driver replay view |
| `eval/` | Metrics, the bounding-box baseline, and report generation |

## Status

Phase 0 (setup). See **`docs/STATUS.md`** for the current task and health, `docs/RISKS.md` for open risks, and
`docs/devlog/` for the working log.

## Honest numbers

Every number quoted in this README or in any writeup traces to a line in `metrics/results.jsonl` and is
regenerated into `docs/REPORT.md` by `python scripts/report.py`. **No metrics have been measured yet** - the
table below stays empty until they are.

| Metric | Value | Split | Evidence |
|---|---|---|---|
| _(none yet)_ | | | |

## Setup

Python 3.11. On this laptop there is no NVIDIA GPU, so all GPU stages run on Colab.

```bash
python -m venv .venv
.venv\Scripts\activate                              # Windows
pip install -r envs/requirements-core.txt
pip install -e .
pytest -q && ruff check .
```

`envs/core.yml` is the equivalent conda spec for machines that have conda; `envs/colab_requirements.txt` is
installed only inside Colab. See `docs/decisions/0001-venv-instead-of-conda.md`.

## Privacy

Recordings may contain bystanders and license plates. Nothing leaves `data/` without a face + plate blur pass
and a human visual confirmation. Raw recordings, GPS tracks, and location metadata are never committed.

## Licence

MIT for this repository's own code. Third-party components keep their own licences (see CLAUDE.md §10);
AGPL-licensed detectors are deliberately avoided.
