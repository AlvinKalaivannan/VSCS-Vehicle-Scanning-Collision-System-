# CLAUDE.md — VSCS (Vehicle Scanning Collision System)

This file governs how Claude works in this repository. Read it fully at the start of every session. When this file conflicts with a one-off instruction from the developer, the developer's instruction wins for that session, but flag the conflict and record it in the devlog.

---

## 0. Session protocol (do this every time, in order)

1. **Orient.** Read `docs/STATUS.md`, the newest file in `docs/devlog/`, and all rows marked `OPEN` in `docs/RISKS.md`.
2. **Confirm the task.** State the task ID (e.g. `P2-T3`) you are working on, its acceptance criteria (from §6), and your plan in ≤5 bullets. If the task is ambiguous or not in §6, ask the developer before writing code.
3. **Work in small steps.** One logical change per commit. Run tests after each meaningful change.
4. **Verify with evidence.** A task is only done when its acceptance criteria are met *and the evidence is saved* (a metric in `metrics/results.jsonl`, a screenshot/recording in `docs/evidence/`, or a passing test). Never report a number you did not compute in this session or read from `metrics/`.
5. **Report.** Before ending: update `docs/STATUS.md`, write/append today's devlog entry (§7), log any new risk or triggered fallback in `docs/RISKS.md`, and record any design decision as an ADR.
6. **Hand off.** End the session with a 3–5 line summary to the developer: what changed, what's verified, what's blocked, next task.

### Stop and ask the developer before

- Deleting, overwriting, or moving anything in `data/raw/` (raw data is irreplaceable).
- Changing a coordinate convention, units, or any schema in §4 (breaks every downstream module).
- Recommending a purchase or paid service beyond Colab Pro.
- Publishing, uploading, or committing any recording, image, or scan (privacy, §9).
- Switching to a fallback path (§8) — propose it with the trigger evidence; the developer decides.
- Adding a dependency with a copyleft license (AGPL/GPL) or unclear license (§10).
- Any capture session plan involving people near the vehicle.
- When a task's acceptance criteria are still unmet after two honest attempts.

### Collaboration mode

Default: **pair mode.** VSCS is the developer's portfolio project and they must be able to explain every core algorithm in an interview.

- For **core contribution modules** — `seg/fusion3d.py`, `recon/scale.py`, `risk/sweep.py`, `risk/probability.py`, the pseudo-label projection in `seg/train/dataset.py` (P4-T8), and the student-vs-teacher evaluation metrics in `seg/train/evaluate.py` (P4-T11) — explain the approach, the math, and pseudo-code first; let the developer write the first draft if they want to; then review and harden it. The training loop, data loaders (Dataset/DataLoader wrappers) and the training notebook count as plumbing.
- For **plumbing** (I/O, CLI, configs, tests, viz, wrappers around third-party tools), write it directly.
- When a concept leans on linear algebra (projections, rigid transforms, frames), give a short explanation tied to MAT188-level notation.
- The developer can switch to `mode: build` in `docs/STATUS.md` to have Claude write everything; respect that field.

---

## 1. Project summary

VSCS is a software-only, advisory computer-vision system that:

1. Acquires a full exterior surface scan of the subject vehicle as a calibrated monocular image sequence, and reconstructs it as a metric 3D model.
2. Segments that model into **individual mechanical components** (wheels, bumpers, mirrors, doors, underbody, etc.) and exports a collision model (URDF) with per-component **severity weights** and **joints** for articulated parts.
3. Processes recorded manoeuvring sequences (reversing and low-speed manoeuvring) to build a 3D representation of obstacles around the vehicle.
4. Computes **per-component collision risk** (distance, time-to-contact, probability, severity) and an aggregate risk, then produces warnings.
5. Establishes its value through controlled comparison against a **single-bounding-box baseline**.
6. Includes **one trained model**: a vehicle-part segmenter ("student") fine-tuned on pseudo-labels produced by the scan pipeline's Grounding DINO + SAM 2 + 3D fusion ("teacher"). The risk engine stays pure geometry and is never learned.

### Hard boundaries

- **Advisory only.** VSCS never controls steering, throttle, or brakes. No code in this repo may send commands to a vehicle.
- **Not a safety device.** The README and demo must state this. Never let a test depend on VSCS to prevent a real collision.
- **Offline-first.** The system operates on recorded sequences. Real-time *feasibility* is characterised by measured throughput on a Colab T4, not demonstrated by in-vehicle deployment.
- **Single monocular instrument.** Acquisition uses one calibrated monocular camera. No depth sensor, structured light or stereo rig is assumed, and no custom hardware is built. The current instrument is a consumer smartphone camera; this is recorded explicitly because the constraints it imposes — variable frame rate, rolling shutter, autofocus drift — are the stated cause of R-03, R-04 and R-06, and of the P0-T7 calibration gate. A USB stereo camera is an optional later upgrade (Tier 1), only with the developer's approval.

### Compute

- **Laptop (no NVIDIA GPU):** COLMAP (CPU), risk engine, collision checks, Rerun viz, tests, scripting.
- **Colab Pro (T4 preferred; do not request A100 unless justified in an ADR):** Gaussian splatting, SAM 2, Grounding DINO, depth models, detection.
- Budget awareness: log approximate GPU time per job in the devlog. Warn the developer if a planned job will exceed ~10 compute units.

---

## 2. Architecture

```
 monocular sequence + IMU              monocular sequence + IMU (mounted on van)
        │                                         │
        ▼                                         ▼
 ┌─────────────┐   ┌────────────┐          ┌──────────────┐
 │  capture/   │──▶│   recon/   │          │  capture/    │
 │ ingest, VFR │   │ SfM, scale,│          │ ingest, sync │
 │ intrinsics  │   │ vehicle frm│          └──────┬───────┘
 └─────────────┘   └─────┬──────┘                 ▼
                         ▼                 ┌──────────────┐
                   ┌────────────┐          │ perception/  │
                   │   seg/     │          │ depth, occ., │
                   │ 2D masks → │          │ detect, track│
                   │ 3D fusion  │          │ ego-motion   │
                   └─────┬──────┘          └──────┬───────┘
                         ▼                        │ Obstacle[]
                   ┌────────────┐                 ▼
                   │  model/    │  URDF +   ┌──────────────┐
                   │ convex dec,│─────────▶ │    risk/     │
                   │ URDF, sev. │ components│ predict path,│
                   └────────────┘   .yaml   │ sweep, TTC,  │
                                            │ prob, agg.   │
                                            └──────┬───────┘
                                                   │ RiskFrame
                                    ┌──────────────┼──────────────┐
                                    ▼              ▼              ▼
                               ui/ rerun      ui/ driver      eval/ metrics,
                               dev view       replay + audio  baseline, report
```

**Offline pipeline (once per vehicle):** capture → recon → seg → model.
**Per-drive pipeline:** capture → perception → risk → ui/eval.

Modules communicate only through the schemas in §4. No module imports another module's internals.

---

## 3. Repository layout

```
vscs/
├── CLAUDE.md
├── README.md                  # public-facing; includes "not a safety device"
├── pyproject.toml             # package + tool config (ruff, pytest)
├── envs/
│   ├── core.yml               # laptop env (CPU): recon wrappers, risk, ui, eval
│   └── colab_requirements.txt # GPU stages, installed in Colab
├── configs/
│   ├── capture.yaml
│   ├── recon.yaml
│   ├── seg.yaml               # prompts, thresholds, component vocabulary
│   ├── model.yaml
│   ├── severity.yaml          # component + component×obstacle severity table
│   ├── perception.yaml
│   ├── risk.yaml              # horizon, dt, path fan, margins, alert thresholds
│   ├── train_seg.yaml         # student segmenter: model, augmentation, schedule, seeds (P4-T10)
│   └── eval.yaml              # dataset splits
├── data/                      # GITIGNORED except MANIFEST.md
│   ├── MANIFEST.md            # every raw file: id, date, sha256, description, backup locations
│   ├── raw/                   # never modified after ingest
│   ├── interim/               # regenerable
│   └── processed/             # regenerable
├── src/vscs/
│   ├── common/                # types.py (schemas), frames.py (transforms), config.py, io.py, log.py
│   ├── capture/               # ingest.py, frames.py (VFR→CFR/timestamps), calib.py, sync.py
│   ├── recon/                 # sfm.py, scale.py, ground.py, vehicle_frame.py
│   ├── seg/                   # masks2d.py, fusion3d.py, cleanup.py
│   │   └── train/             # dataset.py (pseudo-labels), train.py, evaluate.py (P4-T8 to P4-T11)
│   ├── model/                 # decompose.py, severity.py, joints.py, urdf.py
│   ├── perception/            # depth.py, occupancy.py, detect.py, track.py, egomotion.py
│   ├── risk/                  # motion.py, sweep.py, metrics.py, probability.py, aggregate.py, alerts.py
│   ├── ui/                    # rerun_view.py, driver_replay.py, audio.py
│   └── eval/                  # metrics.py, baseline_bbox.py, report.py
├── scripts/                   # thin CLI entry points only (argparse/typer → src/vscs)
├── notebooks/colab/           # thin wrappers: install, mount Drive, call src/vscs. NO logic here.
│   └── train_seg.ipynb        # student training on T4 - thin wrapper only, per §4.3
├── tests/
│   ├── unit/
│   └── fixtures/              # tiny synthetic data (a box "vehicle", a pole, a 20-frame clip)
├── metrics/
│   └── results.jsonl          # append-only, written by eval scripts only
└── docs/
    ├── STATUS.md
    ├── RISKS.md
    ├── REPORT.md              # auto-generated from metrics/results.jsonl
    ├── devlog/YYYY-MM-DD.md
    ├── decisions/NNNN-title.md
    ├── phase_reports/phase-N.md
    ├── capture_checklists.md
    └── evidence/              # screenshots, short clips (privacy-scrubbed only)
```

---

## 4. Conventions and data contracts

### 4.1 Frames and units (never change without an ADR + the developer's approval)

- **Units:** meters, seconds, radians. Severity costs are unitless.
- **Vehicle frame `veh`:** right-handed, **x forward, y left, z up** (ROS REP-103). Origin on the ground plane directly below the center of the rear axle.
- **Camera frame `cam`:** OpenCV convention — x right, y down, z forward.
- **World frame `world`:** fixed at the vehicle's pose at the start of a drive (for ego-motion).
- **Transform naming:** `T_a_b` is a 4×4 homogeneous matrix that maps points expressed in frame `b` into frame `a`: `p_a = T_a_b @ p_b`. Composition: `T_a_c = T_a_b @ T_b_c`. All transforms live in `common/frames.py` with tests for inverse and composition.
- **Timestamps:** `int64` nanoseconds. Never use frame index as time (consumer capture devices record at variable frame rate).

### 4.2 Schemas (`src/vscs/common/types.py`, dataclasses or pydantic)

```python
Component:
  name: str                 # machine name, e.g. "rear_right_bumper_corner"
  display_name: str         # driver-facing, e.g. "rear right corner"
  link: str                 # URDF link name
  severity: float           # default cost, from severity.yaml
  min_z: float              # lowest point in veh frame (underbody checks)

ComponentModel:
  urdf_path: Path
  components: list[Component]
  joints: dict[str, JointSpec]   # door/mirror hinges: axis, origin, limits
  scale_error_m: float           # validation result from recon

Obstacle:
  id: int
  t_ns: int
  kind: Literal["static_geom", "person", "vehicle", "cyclist", "unknown"]
  center_veh: (x, y, z)
  extent: (dx, dy, dz)
  velocity_veh: (vx, vy, vz)
  pos_sigma_m: float        # 1-σ position uncertainty
  source: Literal["occupancy", "detector"]

ComponentRisk:
  component: str
  min_distance_m: float
  ttc_s: float | None       # None = no predicted contact within horizon
  p_contact: float          # [0, 1]
  severity: float
  risk: float               # p_contact * severity * speed_factor
  obstacle_id: int | None

RiskFrame:
  t_ns: int
  ego_speed_mps: float
  per_component: list[ComponentRisk]
  p_any_contact: float      # 1 - Π(1 - p_i)
  expected_damage: float    # Σ risk_i
  worst_component: str | None
  alert_level: Literal["none", "caution", "warning", "critical"]
```

Serialize `RiskFrame` streams as JSON Lines. Every schema change requires: version bump in `types.py`, migration note in an ADR, and updated tests.

### 4.3 Code standards

- Python 3.11 locally (conda env `vscs`). On Colab, use the provided runtime; if a dependency breaks on it, record the workaround in an ADR.
- Type hints everywhere; `ruff check` and `ruff format` must pass.
- All tunable numbers live in `configs/*.yaml`. No magic constants in code (physical constants and conventions excepted).
- Deterministic: seed every random process; log seeds.
- Every script writes outputs to a new run folder: `data/processed/<stage>/<YYYYMMDD-HHMM>_<shortsha>/` containing the resolved config, the git commit hash, and a `run.log`.
- Notebooks contain no logic — only install, mount, and a call into `src/vscs`.
- Pin dependency versions in `envs/`. Nerfstudio/COLMAP may need their own env; don't force everything into one.

### 4.4 Git

- Branch per task: `p2-t3-label-fusion`. Merge to `main` only when tests pass.
- Commit messages: `P2-T3: <imperative summary>`.
- Never commit anything in `data/` except `MANIFEST.md`. Never commit API keys, tokens, or `.env` files. Keep repo small: no binaries > 5 MB outside `docs/evidence/` (and those must be privacy-scrubbed).

---

## 5. Testing requirements

- `tests/fixtures/` holds a synthetic world: a box-shaped "vehicle" with labeled sub-boxes as components, a thin pole, a curb, and a short synthetic camera trajectory with known poses. Every module must have at least one test against it with an analytically known answer (e.g. known distance from rear-right corner to pole = 0.500 m ± 1e-6).
- Required unit tests: transform inverse/composition; pinhole projection round-trip; scale recovery from a known marker; label fusion on synthetic multi-view masks; swept-distance on the fixture; `p_any_contact` math; alert hysteresis state machine; pseudo-label projection on the fixture (a known box component projects to a known pixel mask; an occluded point is not labeled).
- `pytest -q` must pass before any commit to `main`. Coverage target for `common/` and `risk/`: ≥80%.
- Integration test (`tests/test_pipeline_smoke.py`): fixture scene end-to-end → produces a valid `RiskFrame` stream in < 60 s on laptop CPU.

---

## 6. Phases, tasks, and acceptance gates

Rough budget: ~5–8 hrs/week around first-year engineering. December is exam time: **no VSCS work planned.** Capture all real-world data in October (before snow).

A phase is complete only when every task's acceptance criterion is met and `docs/phase_reports/phase-N.md` is written with evidence links. Do not start the next phase's core work early without the developer's go-ahead (setup/scaffolding is fine).

### Phase 0 — Setup (late Sept → early Oct)

| ID | Task | Acceptance criteria |
|---|---|---|
| P0-T1 | Repo scaffold per §3, `pyproject.toml`, ruff, pytest, conda env | `pytest` runs (0 tests OK), `ruff check` clean |
| P0-T2 | Reporting skeleton: `STATUS.md`, `RISKS.md` (seeded from §8), devlog template, ADR template, `metrics/results.jsonl`, `eval/report.py` stub | `python scripts/report.py` generates `docs/REPORT.md` without error |
| P0-T3 | `common/frames.py` + `types.py` with tests | Transform + projection tests pass |
| P0-T4 | Synthetic fixture world | Fixture loads; known-distance test passes |
| P0-T5 | Colab notebook template (install, Drive mount, checkpoint dir, call into package) | Runs a hello-world GPU check and writes to Drive |
| P0-T6 | Warm-up: monocular sequence of a small object → COLMAP → splat | Viewable 3D model; steps documented in devlog |
| P0-T7 | Phone camera intrinsics calibration (checkerboard, focus/exposure locked) | Reprojection error < 0.5 px, saved to `configs/capture.yaml` |

### Phase 1 — Data capture + van scan (October)

| ID | Task | Acceptance criteria |
|---|---|---|
| P1-T1 | Write `docs/capture_checklists.md` (scan day + lot day, safety, privacy) | Developer reviewed it |
| P1-T2 | Capture day 1: van scan (2–3 loops, 3 heights, ArUco/AprilTag markers, doors closed + open, overcast) | Raw files ingested, checksummed in `MANIFEST.md`, backed up to 2 locations |
| P1-T3 | Capture day 2: parking-lot runs with cones/pole/boxes at tape-measured positions; camera mounted rear-facing, sequence + IMU | ≥10 passes recorded; ground-truth positions in `data/raw/lot_*/gt.yaml`; backed up |
| P1-T4 | Ingest: VFR-safe frame extraction with real timestamps; IMU/video sync | Timestamp monotonic check passes; sync offset estimated and logged |
| P1-T5 | SfM reconstruction of van (COLMAP) | ≥90% of extracted frames registered; mean reprojection error < 1.5 px |
| P1-T6 | Metric scale from markers + ground plane (RANSAC) + vehicle frame | Length, width, height, wheelbase each within **±2 cm** of tape measurements (`scale_error_m` logged) |
| P1-T7 | Dense geometry (point cloud/mesh) + splat for visuals | Both viewable in Rerun in `veh` frame |
| P1-T8 | Capture a short walkaround video of a second vehicle (any family car, same capture rules as P1-T2, no markers needed). Needed later for P4-T11 cross-vehicle evaluation, and must be captured before snow | Raw files ingested, checksummed in `MANIFEST.md`, backed up |

### Phase 2 — Component segmentation + model export (November; reading-week push)

| ID | Task | Acceptance criteria |
|---|---|---|
| P2-T1 | Component vocabulary + severity table in `configs/seg.yaml`, `configs/severity.yaml` | Developer approved list (≥8 components) |
| P2-T2 | 2D masks: Grounding DINO boxes → SAM 2 masks tracked across scan video | Visual check on 20 random frames; failures logged |
| P2-T3 | **3D label fusion** (project points into posed frames, depth-test occlusion, multi-view vote) | Per-component IoU ≥ 0.70 mean vs hand-labeled 3D subset (P2-T4) |
| P2-T4 | Hand-label ground-truth subset of 3D points (CloudCompare or similar) | ≥ 2,000 labeled points across all components |
| P2-T5 | Cleanup (clustering, stray-label removal) | IoU improves or stays equal vs P2-T3; logged |
| P2-T6 | Convex decomposition (CoACD) per component | Each component's hull volume within ±15% of its point-cloud-derived volume |
| P2-T7 | Joints for sliding door, rear doors, mirrors (manual axis first) | Rerun shows door swinging correctly through its range |
| P2-T8 | URDF + `components.yaml` export | Loads in a URDF viewer and in `risk/`; fixture test for loading passes |

### December — examination period. No planned work. Emergency data backups only.

### Phase 3 — MVP "VSCS v0.1" (winter break)

| ID | Task | Acceptance criteria |
|---|---|---|
| P3-T1 | Rerun dev view: van model + recorded drive replay | Any lot run replays with synced video |
| P3-T2 | Simple perception v0: ground-plane-anchored depth + static obstacle positions | Cone positions within ±25 cm at ≤3 m on dev split |
| P3-T3 | Motion model + path fan (constant curvature, 5–7 curvatures) | Fixture test: straight + arc paths correct |
| P3-T4 | **2D per-component risk** (footprint polygons, shapely) → distance, TTC | Fixture known-answer tests pass |
| P3-T5 | Severity aggregate + alert levels + hysteresis | Hysteresis test: no flicker on noisy fixture input |
| P3-T6 | End-to-end v0.1 on dev runs | On ≥80% of dev passes, the correct component is flagged before passing the nearest obstacle |

**Phase 3 is the minimum viable project.** If everything after it slips, VSCS v0.1 must be demo-able on its own.

### Phase 4 — Real perception + 3D risk (January → February)

| ID | Task | Acceptance criteria |
|---|---|---|
| P4-T1 | Depth: motion stereo + learned metric depth, fused | Depth error reported at 1/3/5/10 m on dev split |
| P4-T2 | Ego-motion (visual-inertial) | Drift < 5% of distance traveled on dev passes |
| P4-T3 | Occupancy / height map in `veh`/`world` frames, persistent after leaving FOV | Obstacles remain tracked beside wheels after exiting view |
| P4-T4 | Detection + tracking for dynamic objects (person, vehicle, cyclist) + Kalman velocity | Tracks stable on a staged walk-behind clip (person at safe distance, van stationary) |
| P4-T5 | 3D sweep with FCL/trimesh using URDF + joint states | Fixture 3D known-answer tests pass |
| P4-T6 | Underbody clearance vs height map | Curb/speed-bump fixture flagged for correct component |
| P4-T7 | Uncertainty-aware `p_contact = Φ(-d/σ)` | Unit tests; σ grows with range as configured |
| P4-T8 | **Build pseudo-label dataset**: project cleaned 3D component labels (P2-T5 output) back into every posed scan frame, using a depth test for occlusion, to produce per-pixel masks. Split train/val/test by contiguous frame segments or by scan loop, never by random frame (adjacent frames leak) | Dataset builds from one command; split files saved in `configs/`; a visual check on 20 frames is logged |
| P4-T9 | Hand-correct a gold set: at least 50 van frames from the test split, plus at least 30 frames from the second vehicle (P1-T8). Gold frames are never used for training | Gold masks saved; counts per component logged |
| P4-T10 | Fine-tune a small semantic segmentation model on Colab T4. Checkpoint to Drive every epoch, seed everything, log compute units used. Model choice follows the trained-segmenter rules in §10 and is recorded in an ADR | Training runs end to end and resumes after a disconnect; best checkpoint saved with its config and git hash |
| P4-T11 | **Evaluate student vs teacher** | Report to `metrics/results.jsonl`: per-component IoU and mIoU on van gold frames (student and teacher), mIoU on second-vehicle gold frames, and FPS on T4 (student and teacher). Pass: student mIoU is at least teacher mIoU on van gold frames, OR within 5 points while at least 5× faster. Cross-vehicle transfer is reported honestly with no pass threshold |
| P4-T12 (optional) | Use student masks as the 2D input to 3D fusion and re-run P2-T3/P2-T5 | 3D IoU compared against the teacher-based result |

### Phase 5 — Evaluate + ship (March)

| ID | Task | Acceptance criteria |
|---|---|---|
| P5-T1 | Bounding-box baseline (same pipeline, van = one box) | Runs on same splits |
| P5-T2 | **Evaluation on held-out test split** (first time test split is touched) | Report: component attribution accuracy, TTC error, false alarms/min, recall on thin poles/curbs, VSCS vs baseline |
| P5-T3 | Throughput benchmark on Colab T4 | FPS per stage and end-to-end logged |
| P5-T4 | Driver replay UI: silhouette + audio + per-component path overlay | Works on any recorded run |
| P5-T5 | User test (3–5 drivers watch replays / optional stationary demo) | Notes and quotes in `docs/evidence/` |
| P5-T6 | Demo video + README/writeup with limitations | Published only after privacy scrub (§9) and the developer's approval |

### Stretch (summer 2027, only after Phase 5)

CARLA simulation with the scanned van imported · auto hinge-axis detection from open/closed scans · stereo camera (Tier 1) · live in-vehicle compute.

### Cut order if behind schedule

1. P5-T5 user test → 2. Trained segmenter (P4-T8 to P4-T12; report teacher-only results) → 3. P4-T4 dynamic tracking (static only) → 4. P4-T7 probabilities (use clearance/TTC thresholds). **Never cut P5-T1/P5-T2 (baseline comparison)** — it is the core result. P1-T8 (second-vehicle capture) is not cut: it is cheap and cannot be redone after snow.

---

## 7. Continuous reporting system

Reporting is not optional. The goal: at any moment, the developer (or a future session) can see exactly where the project stands, what's proven, and what's at risk — without re-reading code.

### 7.1 `docs/STATUS.md` (single source of truth; overwrite each session)

```markdown
# VSCS Status — updated YYYY-MM-DD
mode: pair            # pair | build
phase: P2
current_task: P2-T3 (in progress)
health: ON TRACK | AT RISK | BLOCKED
## Done since last update
- ...
## Verified metrics (link to metrics/results.jsonl entries)
- scale_error_m: 0.013 (run 20261019-1402_ab12cd3)
## Blockers
- ...
## Open risks triggered (IDs from RISKS.md)
- R-07
## Next 3 tasks
1. ...
## GPU usage this month (approx compute units)
- ...
```

### 7.2 Devlog: `docs/devlog/YYYY-MM-DD.md` (one per working day, append if multiple sessions)

```markdown
## Session HH:MM — task P?-T?
Goal:
What I did:
Evidence: (metric IDs, screenshots, test names)
What failed / surprised me:
Decisions made (→ ADR ####):
Risks touched (R-##):
Next:
Time spent: ~Xh  | GPU: ~X CU
```

### 7.3 Architecture Decision Records: `docs/decisions/NNNN-short-title.md`

Write one whenever you choose between alternatives, change a schema/convention, switch to a fallback, or add a major dependency. Template: Context → Options considered → Decision → Consequences → Evidence.

### 7.4 Metrics: `metrics/results.jsonl` (append-only, machine-written)

Every evaluation script appends one line:

```json
{"ts": "...", "git": "abc1234", "task": "P1-T6", "metric": "scale_error_m", "value": 0.013,
 "split": "dev", "run_dir": "data/processed/recon/20261019-1402_abc1234", "notes": ""}
```

- Never hand-edit this file. Never delete lines (append a correction line instead).
- `scripts/report.py` regenerates `docs/REPORT.md`: latest value per metric, trend per metric, pass/fail against acceptance thresholds from §6.

### 7.5 Phase reports: `docs/phase_reports/phase-N.md`

At each phase gate: table of every task → criterion → result → evidence link; what changed from plan; fallbacks used; lessons; readiness for next phase. The developer signs off by writing `approved: YYYY-MM-DD` at the bottom.

### 7.6 Health rules

- **AT RISK** when: a task has exceeded 2× its expected time, a fallback trigger has fired, or a phase will slip past its month.
- **BLOCKED** when: progress is impossible without the developer's decision, new data, or money.
- On AT RISK or BLOCKED, the session summary must lead with it.

---

## 8. Risk register and fallback paths

Seed `docs/RISKS.md` from this table. Columns: ID · risk · likelihood · impact · detection signal · mitigation · fallback · status (`OPEN` / `WATCH` / `TRIGGERED` / `CLOSED`). When seeding, rate likelihood and impact as low/med/high with the developer, and link each risk to its matching row in the §8.2 fallback ladder. Re-rate at every phase gate.

### 8.1 Technical vulnerabilities

| ID | Risk | Detection signal | Mitigation (do by default) |
|---|---|---|---|
| R-01 | Reflective paint / windows break SfM | <90% frames registered; holes in body panels | Overcast capture, slow walk, 3 heights, lock AE/AF, textured markers on ground |
| R-02 | Scale error or drift | Tape-measure mismatch > 2 cm | ≥3 markers of known size spread around vehicle; check against 4 independent dimensions |
| R-03 | Phone variable frame rate + rolling shutter corrupt timing | Non-monotonic or irregular timestamps; ego-motion jitter | Use container timestamps, never frame index; slow motion during capture; record at highest fixed fps available |
| R-04 | Autofocus/zoom changes intrinsics mid-capture | Calibration reprojection error rises on capture frames | Lock focus/exposure, use main lens only, no zoom |
| R-05 | SAM/Grounding DINO labels bleed across component boundaries | IoU < 0.70; labels spill onto neighbors | Multi-view voting with occlusion test; confidence thresholds; manual prompts for weak classes |
| R-06 | Monocular depth too inaccurate for contact-level precision | Depth error > 10 cm at 3 m | Motion stereo + metric depth fusion; ground-plane anchoring; report error honestly |
| R-07 | Thin poles, low curbs, glass missed | Low recall on those categories | Dedicated test cases in lot runs; occupancy layer independent of detector |
| R-08 | Ego-motion drift corrupts persistent map | Static cones "move" over a pass | IMU fusion; short horizons; reset map per maneuver |
| R-09 | Overfitting thresholds to the data you evaluate on | Dev/test gap large at P5 | Split runs into **dev** and **held-out test** at ingest; test touched once, in P5-T2 |
| R-10 | Alert flicker / false-alarm spam | Alerts toggle > 2×/s; false alarms/min high | Hysteresis, minimum dwell time, severity weighting |
| R-11 | Colab disconnects / compute units run out | Lost runs; CU balance low | Checkpoint to Drive every stage; small resumable jobs; T4 not A100; log CU per job |
| R-12 | Dependency conflicts (nerfstudio, COLMAP, torch, CUDA) | Env install fails | Separate envs; pinned versions; record working combos in ADR |
| R-13 | Data loss | Missing/corrupt raw file | 3 copies (laptop, external drive, cloud); sha256 in MANIFEST; verify after copy |
| R-14 | Scope creep / time crunch with coursework | Health AT RISK two sessions in a row | Cut order in §6; Phase 3 MVP protected |
| R-15 | Articulated joint states unknown during drives | Wrong geometry used (door open vs closed) | Manual joint-state input per run in v0.1; auto-detect is stretch |
| R-16 | Schema/convention drift between modules | Integration test fails; frames mismatched | §4 contracts, version field, smoke test on every merge |
| R-17 | Student learns the teacher's mistakes (pseudo-label errors baked in) | Student and teacher fail on the same regions | Gold set is hand-corrected and never derived from pseudo-labels; review the worst components |
| R-18 | Data leakage between near-identical adjacent frames | Val score much higher than test score | Split by contiguous segments or loops only |
| R-19 | Overfitting to one van | Big drop on second-vehicle gold set | Augmentation (color, blur, crop, flip where the label side is swapped correctly), early stopping; report the drop as a finding |

### 8.2 Fallback ladder (propose to the developer when trigger fires; record in ADR)

| Stage | Primary | Fallback 1 | Fallback 2 | Trigger to step down |
|---|---|---|---|---|
| Reconstruction | COLMAP incremental SfM | GLOMAP (global SfM) or hloc with SuperPoint + LightGlue features | Phone app export (e.g. Polycam/Luma) as geometry source, still scaled with markers | <90% frames registered after one recapture attempt |
| Scale | ArUco/AprilTag markers | Known dimension (tape-measured wheelbase + length) | Manufacturer spec sheet dimensions | Marker detection fails in >50% of frames |
| Dense geometry | COLMAP/OpenMVS dense | Points extracted from Gaussian splat | Sparse cloud + convex hulls per component | Dense run fails or OOMs twice |
| 2D segmentation | Grounding DINO text prompts → SAM 2 | Manual click prompts into SAM 2 on keyframes | — | Mean IoU < 0.60 on a class after tuning |
| 3D labels | Multi-view fusion | Fusion + manual correction in CloudCompare | Fully manual labeling of 3D points | IoU < 0.70 after cleanup |
| Collision geometry | CoACD convex decomposition | V-HACD | Oriented bounding boxes per component | Decomposition fails or volume error > 15% |
| Depth | Motion stereo + learned metric depth | Learned metric depth + ground-plane scale only | Tier 1 stereo camera (needs the developer's approval) | Depth error > 25 cm at 3 m after fusion |
| Ego-motion | Visual-inertial odometry | Visual odometry only | IMU + ground-truth-constrained lot runs only | Drift > 10% |
| Detection | Apache/MIT-licensed detector (e.g. RT-DETR, YOLOX) | Grounding DINO at lower frame rate | Occupancy-only (no semantics) | License or accuracy problems |
| Tracking | ByteTrack + Kalman | Simple IoU tracker | Static obstacles only (cut order #2) | Unstable IDs on test clip |
| Collision checks | python-fcl | trimesh proximity queries | 2D shapely footprints (v0.1 method) | Install/perf failure |
| Visualization | Rerun | Open3D | matplotlib top-down plots | — |
| Simulation (stretch) | CARLA | Custom synthetic scenes (Blender/fixture generator) | Skip | CARLA setup > 2 sessions |
| Data (if van unavailable) | The project van | Another family vehicle | Synthetic/CARLA-only evaluation (clearly labeled) | Vehicle unavailable for capture month |
| Trained segmenter | Full fine-tune | Freeze the backbone, train only the head | Drop the student and report teacher-only results (core project unaffected) | Student mIoU more than 10 points below the teacher after tuning |

**Flip augmentation and left/right components.** Left/right components (e.g. left vs right mirror) must not be merged by horizontal-flip augmentation. Either swap the labels on flip or disable flip.

---

## 9. Safety, privacy, and legal rules

### Capture safety (write into `docs/capture_checklists.md`)

- A licensed driver operates the van; Claude never plans a session in which the camera operator is also the driver.
- Walking speed only (< 5 km/h) in an empty lot, with permission to use the lot where needed.
- **No people behind or beside the moving vehicle.** Dynamic-object clips (P4-T4) use a person walking at a safe distance with the **vehicle stationary**, or use public footage/CARLA.
- Obstacles are soft/cheap (cones, cardboard boxes, a foam or PVC pole). Nothing that damages the van if touched.
- The camera mount must be secure; the camera is never handled while the vehicle is in motion. Check current Ontario distracted-driving rules before any in-vehicle screen use.
- Stop immediately on rain/ice/poor visibility.

### Privacy

- Recordings may contain bystanders' faces and license plates. **Before anything leaves `data/`** (evidence folder, README, demo video), run a blur pass (face + plate detection) and have the developer visually confirm.
- Do not commit raw recordings, GPS tracks, or anything revealing the developer's home address or routine. Strip location metadata from any published image/video.
- Third-party footage (e.g. online videos) may be used for private testing only, never republished.

### Honest claims

- Never describe VSCS as preventing collisions or as a safety system. Use "advisory," "research prototype," "evaluated on N recorded passes."
- Every number in the README must trace to a line in `metrics/results.jsonl`.

---

## 10. Dependencies and licenses

Prefer permissive licenses (MIT, BSD, Apache-2.0). Check before adding anything; record in `docs/decisions/` if non-trivial.

| Tool | Use | License (verify at install time) |
|---|---|---|
| COLMAP | SfM / MVS | BSD |
| nerfstudio (splatfacto) | Gaussian splatting | Apache-2.0 |
| OpenCV (incl. ArUco) | calibration, markers, I/O | Apache-2.0 |
| Open3D | point clouds, RANSAC | MIT |
| SAM 2 | segmentation | Apache-2.0 |
| Grounding DINO | open-vocab detection | Apache-2.0 |
| CoACD | convex decomposition | MIT |
| python-fcl / trimesh | collision / geometry | BSD / MIT |
| shapely | 2D footprints | BSD |
| Rerun | visualization | MIT / Apache-2.0 |
| CARLA (stretch) | simulation | MIT (assets under separate terms) |
| Ultralytics YOLO | detection | **AGPL-3.0 — avoid unless the developer accepts open-sourcing obligations; prefer Apache/MIT alternatives** |
| torchvision (DeepLabV3 / LR-ASPP) | student segmenter candidate (P4-T10) | Code BSD-3-Clause; pretrained weights: verify separately |
| Mask2Former | student segmenter candidate (P4-T10) | Reported MIT; verify code and pretrained weights separately |

**Trained segmenter (P4-T10) model rules.**

- Prefer a small model with a permissive license *and* permissive pretrained weights (MIT/BSD/Apache), e.g. torchvision DeepLabV3 / LR-ASPP, or Mask2Former. Verify both the code and the weights licenses at install time.
- Avoid AGPL (Ultralytics) and non-commercial licenses. Ask the developer before adding anything with an unclear license.
- Record the model choice in an ADR.

---

## 11. Evaluation protocol

- **Splits:** assign each lot pass to `dev` or `test` at ingest (≈70/30, stratified so both include pole, curb, box cases). Store in `configs/eval.yaml`. `test` is used once, at P5-T2.
- **Metrics (definitions live in `eval/metrics.py`):**
  - Scan: dimensional error (m), per-component IoU.
  - Perception: depth error by range bucket, detection recall by obstacle type, ego-motion drift %.
  - Risk: **component attribution accuracy** (flagged component == component that actually passes closest), TTC error (s), false alarms per minute, lead time (s) of first warning before closest approach.
  - System: per-stage and end-to-end FPS on Colab T4.
  - **Trained model** (P4-T11, computed by `seg/train/evaluate.py`): per-component IoU and mIoU on van gold frames for student and teacher; mIoU on second-vehicle gold frames; FPS on T4 for student and teacher. Pass: student mIoU ≥ teacher mIoU on van gold frames, or within 5 points while ≥5× faster. Cross-vehicle transfer is reported with no pass threshold.
- **Gold sets (P4-T9):** used once, for the final student/teacher numbers, like the `test` split. Only the val split is used for tuning. Gold frames are never trained on.
- **Baseline:** identical pipeline with the vehicle modeled as one oriented bounding box. Report VSCS vs baseline on every risk metric.
- Report failures and limitations as prominently as successes.

---

## 12. Quick reference

```bash
.venv\Scripts\activate                     # venv, not conda, on the dev laptop (ADR 0001)
pytest -q && ruff check .                  # before every commit to main

# Capture preparation (Phase 0/1)
python scripts/check_capture.py --video <clip> --imu <gyro.csv> [--moving-board <clip>]
python scripts/calibrate.py --images <checkerboard_dir> --device <name>     # P0-T7
python scripts/make_markers.py --out <dir>                                  # print at 100%

# Offline pipeline (per vehicle)
python scripts/ingest.py --src <path> --kind scan|lot|calib                 # P1-T4
python scripts/extract_frames.py --video data/raw/<id>/<clip>              # P1-T4
python scripts/recon.py --frames-run data/processed/capture/<run>          # P1-T5, needs COLMAP
python scripts/scale.py --recon-run <run> --images <frames> --measured <yaml> # P1-T6: needs scale.py; writes sfm_to_veh.json
python scripts/dense.py --recon-run <run> --images <frames>              # P1-T7, Colab (notebooks/colab/10_dense.ipynb)
python scripts/seg.py --frames <frames> --sam2-dir <sam2 checkout>         # P2-T2, Colab (notebooks/colab/20_seg.ipynb)
python scripts/fuse.py --dense-run <run> --seg-run <run> --sfm-to-veh <json> # P2-T3/T5: needs fusion3d.py + P1-T6
python scripts/score_labels.py --fuse-run <run> --gold <gold.txt>          # P2-T3/T4/T5 gates vs the hand-labelled subset
python scripts/export_model.py --labelled <cleaned.npz> --scale-error-m <m> # P2-T6..T8: decompose + URDF

# Per-drive pipeline
python scripts/perceive.py --frames-run <capture run> [--ego vo|stationary|file]  # detect + perception -> obstacles.jsonl (GPU env)
    #   add --imu <gyro.csv> --imu-offset-ms <P1-T4 offset> to fuse the phone gyro with VO (P4-T2)
python scripts/risk.py --model <model run> --perception <perceive run> [--baseline]  # -> risk_frames.jsonl (needs sweep.py)
python scripts/evaluate.py --passes <passes.yaml> --split dev [--log-metrics]   # VSCS vs baseline (§11); test split: --final, once
python scripts/view.py --model <model run> --risk <frames.jsonl> [--obstacles <obs.jsonl>]  # P3-T1 -> drive.rrd
python scripts/replay.py --model <model run> --risk <frames.jsonl> [--obstacles <obs.jsonl>]  # P5-T4 -> PNG frames + replay.wav

# External datasets (ADR 0006; licence-checked, private by default)
python scripts/fetch_dataset.py <id>                                        # id from configs/datasets.yaml

python scripts/benchmark.py --frames-run <run> [--detector] [--log-metrics]  # P5-T3, per-stage fps tagged with hardware
python scripts/report.py                                                    # regenerates docs/REPORT.md
```

Every script planned for Phases 0-5 now exists; Phase 6 adds `stream.py` and `docker compose up`.

(Keep this list in sync with reality.)
