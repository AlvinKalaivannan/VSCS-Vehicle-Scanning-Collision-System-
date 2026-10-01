# VSCS — Vehicle Scanning Collision System

Per-component collision risk estimation for low-speed vehicle manoeuvring, derived from a
metric three-dimensional reconstruction of the subject vehicle.

## Intended use and limitations

VSCS is an **advisory-only research prototype**. It is **not a safety device** and must not
be relied upon to prevent a collision.

- The system issues advisory information only. It does not actuate steering, throttle or
  braking, and no component of this repository transmits commands to a vehicle.
- Processing is performed offline against recorded sequences. VSCS is not a deployed
  in-vehicle system, and real-time feasibility is characterised by measured throughput
  rather than demonstrated by live operation.
- Reported performance figures describe evaluation on a finite, held-out set of recorded
  manoeuvres. They are not a general guarantee of detection or of accuracy.

## Overview

Conventional low-speed proximity systems represent the vehicle as a single bounding volume
and report a scalar distance to the nearest obstacle. That abstraction discards the
information a driver actually needs: a mirror contacting a post and a tyre contacting a
kerb are distinguished neither by likelihood, nor by the corrective action required, nor by
consequence.

VSCS resolves the vehicle into its individual mechanical components — bumper corners,
mirrors, wheels, doors and underbody — and estimates contact risk for each independently.
Each component carries a severity weight, allowing the system to report an expected-damage
figure and to identify the specific component at risk rather than issuing an undifferentiated
proximity warning.

The system therefore addresses a more specific question than proximity alone: **which
component of the vehicle is at risk, with what probability of contact, and at what
consequence.**

## Method

### Vehicle model acquisition

A complete exterior surface scan of the subject vehicle is acquired as a calibrated
monocular image sequence, captured at multiple heights and viewing angles with fixed
intrinsics and dimensional reference markers of known size. Structure-from-motion
reconstruction, marker-based metric scaling and ground-plane estimation yield a
dimensionally accurate model expressed in a vehicle-fixed coordinate frame.

The reconstruction is segmented into individual mechanical components by lifting
open-vocabulary two-dimensional masks into three dimensions through multi-view label fusion
with explicit occlusion testing. Components are convex-decomposed and exported as a
collision model with per-component severity weights and articulation for hinged parts.

Dimensional accuracy is validated against independent physical measurements of the vehicle,
with an acceptance threshold of ±2 cm across four independent dimensions.

### Risk estimation

Recorded manoeuvring sequences are processed to recover ego-motion, obstacle geometry and
obstacle classification. A constant-curvature motion model generates a fan of candidate
vehicle trajectories over a short horizon. For each component and each candidate
trajectory, the system computes swept clearance, time-to-contact, and a contact probability
derived from the estimated clearance and its uncertainty. Per-component risks are
aggregated into a frame-level expected-damage figure and an alert state governed by
hysteresis.

### Evaluation

The principal result is a controlled comparison against a **single-bounding-box baseline**:
an otherwise identical pipeline in which the vehicle is represented as one oriented
bounding box. Both configurations are evaluated on the same held-out test split, which is
assigned at data ingest and examined exactly once, at the final evaluation stage.

Reported metrics comprise component attribution accuracy, time-to-contact error, false
alarms per minute, warning lead time, recall on thin and low-profile obstacles,
depth error by range, and per-stage throughput.

## System architecture

```
Vehicle model (once per vehicle):   capture -> recon -> seg -> model
Per manoeuvre:                      capture -> perception -> risk -> ui / eval
```

| Stage | Function |
|---|---|
| `capture/` | Sequence ingest, frame extraction with container-accurate timestamps, calibration, inertial synchronisation |
| `recon/` | Structure-from-motion, metric scaling, ground-plane estimation, vehicle frame definition |
| `seg/` | Two-dimensional component masks lifted to three dimensions by multi-view label fusion |
| `model/` | Per-component convex decomposition, articulation, collision model export |
| `perception/` | Depth estimation, occupancy mapping, obstacle detection and tracking, ego-motion |
| `risk/` | Trajectory prediction, swept clearance, time-to-contact, contact probability, aggregation |
| `ui/` | Developer visualisation and driver replay |
| `eval/` | Metric definitions, baseline comparison, report generation |

Inter-module communication is restricted to the versioned schemas defined in `CLAUDE.md`
§4.2. Units are metres, seconds and radians throughout; timestamps are 64-bit integer
nanoseconds; the vehicle frame follows ROS REP-103.

## Results

All reported figures are generated from `metrics/results.jsonl` and regenerated into
`docs/REPORT.md` by `python scripts/report.py`. No figure is recorded by hand.

**No measurements have yet been taken.** The table below remains empty until evaluation is
performed, and `docs/REPORT.md` currently lists every acceptance threshold as unproven.

| Metric | Value | Split | Evidence |
|---|---|---|---|
| — | — | — | — |

## Acquisition constraints

The current acquisition instrument is a consumer smartphone camera. This is a deliberate
constraint rather than a limitation of the method: the pipeline assumes only a single
calibrated monocular camera and makes no use of depth sensors, structured light or stereo
rigs. Consequences that follow from it are treated explicitly in the risk register
(`docs/RISKS.md`):

- variable frame rate and rolling shutter require container-accurate timestamps, never
  frame indices;
- focus and exposure must remain locked for the duration of a capture, since autofocus
  invalidates the calibration;
- metric scale derives from printed reference markers of measured dimension, validated
  against independent physical measurement of the vehicle.

Substituting a stereo or depth-capable instrument would relax several of these constraints
and is recorded as a possible future extension.

## Environment

Requires Python 3.11. Compute-intensive stages execute on a CUDA-capable host; all
remaining stages run on CPU.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r envs/requirements-core.txt
pip install -e .
pytest -q && ruff check .
```

`envs/core.yml` provides an equivalent Conda specification. `envs/colab_requirements.txt`
covers GPU stages and is installed only on the GPU host. See
`docs/decisions/0001-venv-instead-of-conda.md`.

## Data handling

Recorded sequences may contain identifiable individuals and vehicle registration plates.
No material leaves the `data/` tree without face and plate redaction followed by human
verification. Raw sequences, positional traces and location metadata are excluded from
version control; `data/MANIFEST.md` is the only tracked file within that tree. Third-party
footage is used for private evaluation only and is not redistributed.

## Third-party models, datasets and methods

Every external model, dataset and method VSCS uses is listed here with its source and
licence. Licences were checked at the source on the dates given. Datasets are governed by
`configs/datasets.yaml`, which derives what each may be used for from its licence and
provenance (`docs/decisions/0006-external-datasets.md`).

**Models (weights and code)**

| Model | Use | Licence (checked) | Reference |
|---|---|---|---|
| Grounding DINO (`IDEA-Research/grounding-dino-tiny`) | open-vocabulary component boxes | Apache-2.0, code and weights (2026-10-01) | S. Liu et al., "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection", ECCV 2024 |
| SAM 2 (`sam2.1_hiera_small`) | component masks tracked through the scan | Apache-2.0, code and checkpoints (2026-10-01) | N. Ravi et al., "SAM 2: Segment Anything in Images and Videos", ICLR 2025 |
| RT-DETR (`PekingU/rtdetr_r18vd`) | object detection while driving | Apache-2.0, code and weights (2026-10-01); trained on COCO, no COCO images redistributed | Y. Zhao et al., "DETRs Beat YOLOs on Real-time Object Detection", CVPR 2024 |

**Methods and libraries with a paper**

| Method | Use | Licence | Reference |
|---|---|---|---|
| COLMAP | structure-from-motion, dense stereo | BSD | J. L. Schönberger and J.-M. Frahm, "Structure-from-Motion Revisited", CVPR 2016; J. L. Schönberger et al., "Pixelwise View Selection for Unstructured Multi-View Stereo", ECCV 2016 |
| CoACD | per-component convex decomposition | MIT (2026-09-29) | X. Wei et al., "Approximate Convex Decomposition for 3D Meshes with Collision-Aware Concavity and Tree Search", ACM TOG (SIGGRAPH) 2022 |
| ByteTrack (association idea only; reimplemented on the ground plane, no code used) | multi-object tracking | — | Y. Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022 |

**Datasets** (none redistributed; images never shown publicly)

| Dataset | Licence and provenance | Allowed here |
|---|---|---|
| Roboflow Universe "car parts", workspace `car-segmentation-iq9jj` — <https://universe.roboflow.com/car-segmentation-iq9jj/car-parts-9vig8> | CC BY 4.0 (reported by the Roboflow API, 2026-09-29); photo source undocumented | private evaluation, reported numbers (with this credit), training; never public images |
| Roboflow Universe "Car Parts Segmentation", workspace `person-detector` — <https://universe.roboflow.com/person-detector/car-parts-segmentation> | CC BY 4.0 claimed, but it appears to re-upload the unlicensed DSMLR set, so treated as conflicting | private rehearsal and evaluation only |

## Project status

Phase 1. Current task state, verified metrics and open risks are recorded in
`docs/STATUS.md`; architectural decisions in `docs/decisions/`; the working log in
`docs/devlog/`.

## Licence

MIT for original work in this repository. Third-party components remain under their
respective licences; copyleft-licensed detection models are excluded by policy.
