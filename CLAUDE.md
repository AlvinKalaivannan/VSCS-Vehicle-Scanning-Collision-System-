# CLAUDE.md — VSCS (Vehicle Scanning Collision System)

Learning mode: DEEP

Demo-first build spec, rewritten 2026-11-16 (ADR 0020, ADR 0021). It replaces the September 2026
plan (41 tasks, own-van phone scan + recorded lot drives). Old task IDs (`P0-T1` …
`P6-T6`) stay valid in devlogs, ADRs and branch names; new work uses the IDs in §6.

When this file conflicts with a one-off instruction from the developer, the developer
wins for that session; flag the conflict and record it in the devlog.

---

## 0. Session protocol

1. **Orient.** Read `docs/STATUS.md`, the newest `docs/devlog/` file, and `OPEN` rows in
   `docs/RISKS.md`.
2. **Confirm the task.** State the task ID, its acceptance criterion (§6) and a plan in
   ≤5 bullets. If the task is not in §6, ask first.
3. **Small steps.** One logical change per commit; run tests after each.
4. **Evidence.** A task is done when its criterion is met *and the evidence is saved*: a
   line in `metrics/results.jsonl`, a file in `docs/evidence/`, or a passing test. Never
   report a number you did not compute this session or read from `metrics/`.
5. **Report.** Update `docs/STATUS.md` (and `site.json`, §12), append the devlog, log
   new risks in `docs/RISKS.md`, record decisions as ADRs.
6. **Hand off** in 3–5 lines: what changed, what is verified, what is blocked, next task.

### Stop and ask the developer before

- Changing a frame convention, unit or schema in §4.
- Publishing or deploying anything that contains dataset media (§9), and every push to
  the public demo.
- Using a dataset or model whose licence is not `clear` in `configs/datasets.yaml`, or
  adding a copyleft (AGPL/GPL) or unclear-licence dependency (§10).
- Any purchase or paid tier beyond Colab Pro (includes paid Hugging Face hardware).
- Switching to a fallback (§8.2). Propose it with the trigger evidence.
- Starting a parked item (§7).
- A task whose criterion is still unmet after two honest attempts.

### Collaboration mode

This is a portfolio project and the developer must be able to explain every core
algorithm in an interview. Set by the developer on 2026-11-16: **Claude writes all the
code; nothing core merges until the developer has done a `/teach-back` on it.**

- **Core modules** (Claude explains the approach, the math and pseudo-code, then writes
  it; it sits on its task branch until the teach-back passes):
  - `risk/sweep.py`: swept per-component distance and time-to-contact (V0-T5).
  - `core/curve.py`, `core/budget.py`, `core/track.py`: the error-budget core (V0-T7,
    V0-T8; ADR 0021).
  - `synth/parts.py::classify_faces`: mesh faces → components (V1-T6).
  - `seg/fusion3d.py`: multi-view mask fusion (V1-T7).
  - `risk/probability.py`: `p_contact = Φ(-d/σ)` from the budget (V1-T8).
- **Teach-back gate.** A teach-back that exposes a gap is logged in `LEARNING.md` and
  repeated on the gap only. Later tasks may build on an unmerged core branch; the
  backlog of un-taught branches never exceeds two (D-13).
- **Plumbing** (adapters, I/O, CLI, configs, web app, Docker, tests, viz): written and
  merged directly.
- Linear algebra and statistics (projections, rigid transforms, frames, variances) are
  explained in MAT188-level notation.

### Limits policy

```
limits_policy: exhaust-first        # set by the developer, 2026-10-01
```

Before recording any known limitation, accepted error or "won't fix": list the candidate
fixes, **try and measure** every one that is plumbing, and record the limit only if
(a) no feasible fix exists, (b) the remaining fix is the developer's decision (then
propose it with evidence), or (c) it cannot be corrected or measured with what the
project has. Every recorded limit states what was tried, the evidence, which of (a)–(c)
applies, and who or what would unblock it. Say which way a residual error leans.

---

## 1. What VSCS is

An advisory-only computer-vision system. It:

1. isolates a vehicle into **mechanical components** (bumper corners, doors, mirrors,
   wheels, liftgate, underbody) as a collision model with per-component severity;
2. **tracks surrounding objects** from camera streams;
3. computes **collision risk for the full body and for each component** (distance,
   time-to-contact, probability, severity) and raises graded alerts;
4. reports, with every risk, an **error budget**: how uncertain that number is and where
   the uncertainty comes from (§1.1);
5. shows its value against a **single-bounding-box baseline**.

A portfolio project. Not a startup, not autonomous driving.

### Hard boundaries

- **Advisory only.** No code in this repo sends commands to a vehicle.
- **Not a safety device.** The README and the demo say so on screen.
- **Existing footage only.** The MVP runs on public datasets. No self-recorded video is
  on the critical path.
- **Cameras are the only perception input.** Dataset lidar, 3D cuboids and poses may be
  used as *ground truth* for evaluation. One exception, stated on the demo: ego-motion
  comes from the dataset's poses.
- **Stream processing.** Every per-drive stage is a stream stage: one frame in, one
  result out, bounded queues, stale frames dropped. No stage reads ahead in time.

### Compute

- **Laptop (no NVIDIA GPU):** tracking, lifting, risk, web app, tests, Docker.
- **Colab Pro T4:** detector inference and the all-stages-live throughput benchmark. No
  A100 without an ADR. Log compute units per job; warn before a job over ~10 CU.
- **Hosted demo:** Hugging Face Space, free CPU tier.

### 1.1 Error budgeting: the central method (ADR 0021)

VSCS has no standalone "confidence" score. Every statement of trust is an **error
budget**: a 1-σ uncertainty in metres on a component's distance to its nearest obstacle,
split by source.

| Source | What it is | How it is measured |
|---|---|---|
| detection | Box-edge error in the image, carried to the ground through the lifting geometry | Detected box vs projected cuboid |
| lifting | Pixel → ground-plane position error | **The lifting error curve**: lifted position vs dataset 3D cuboids, binned by range and viewing angle (bearing from the camera's optical axis; object aspect when the track has a heading) |
| tracking | Filtered position and velocity error over the prediction horizon | Track state vs cuboid tracks |
| geometry | Component model vs the true body surface | Template vs mesh; later fused surface vs mesh |
| timing | Latency jitter × closing speed | Measured stage latencies. Mean latency is a bias and is *corrected*, not budgeted |

- Combination, assuming independent sources: `σ_total² = Σ σ_source²`. V1-T4 tests that
  assumption by coverage: about 68% of true errors must fall inside 1 σ.
- **Measured, never assumed.** A source without a measurement is `None`, is shown as
  unmeasured, and makes the total a labelled lower bound. No default σ in a config.
- **Per obstacle:** every lifted obstacle gets its expected error from the curve
  (`Obstacle.pos_sigma_m`).
- **Per component:** each `ComponentRisk` carries its own budget, so the demo shows the
  risk and how far to trust it for each part.
- `p_contact = Φ(-d/σ_total)` (V1-T8) is the only route from uncertainty to probability.
- **Staging.** V0: the lifting curve and a per-component budget in the demo, other
  sources unmeasured. V1: all five sources, validated on Argoverse 2.

### 1.2 Carried from Overstride

Overstride is the developer's earlier project (human pose and gait). Its figures below
were supplied by the developer on 2026-11-16 (§11 row 9).

| From Overstride | In VSCS | Where |
|---|---|---|
| Per-joint error budget | Per-component error budget by source; replaces any "confidence" | §1.1, V0-T8, V1-T4 |
| Viewpoint error curve (3.50° → 1.77° as views went side-on) | Lifting error vs range and viewing angle, measured against cuboids | V0-T7 |
| The 3D lifter degraded 1.84× cross-lab | The Argoverse 2 test exists for this reason, and reports how the budget shifts per source, not only accuracy | V1-T9 |
| Sim-to-real plan (AMASS / BEDLAM) | Sim-to-real with CARLA, as a V2 option | V2-T3 |
| A body is tracked as joints | Tracker + budget built as a domain-neutral component-level core (`vscs/core/`) that could serve bodies and vehicles. Overstride itself is not refactored | ADR 0021, V0-T8 |

---

## 2. v0 definition

One sentence: **a public web page where a visitor picks a driving clip and watches VSCS
process it as a stream, with the ego vehicle drawn as components that light up by risk.**

| Decision | Choice | Reason |
|---|---|---|
| Subject vehicle | The dataset's ego vehicle | Its cameras see the obstacles, so risk is computed for the body the cameras ride on |
| Demo dataset | **PandaSet** (6 surround cameras, 10 Hz, calibration, poses, 3D cuboids; ego is a Chrysler Pacifica) | The only surround-camera driving set found with a licence listed as CC BY 4.0, which the project's policy needs for `publish_media`. **Verify by hand, §11** |
| Cross-dataset test (V1) | **Argoverse 2 Sensor**, ~10 validation logs, numbers only | Already licence-checked (ADR 0018); CC BY-NC-SA allows results, not media |
| Component model, v0 | **Parametric template**: the 17 approved components as boxes sized from the ego's published length, width, height, wheelbase and track | No scan, no mesh, no licence question; one config file |
| Component model, V1 | **Real component segmentation**: Grounding DINO + SAM 2 masks fused across posed views of a CC BY car mesh, scored against the mesh's exact part labels → CoACD → URDF | The core CV contribution; renders give publishable media and exact truth with no scan licence |
| Uncertainty | **Per-component error budget** (§1.1); v0 measures the lifting source | Trust is measured per part, not asserted |
| Detector | **RT-DETR r18** via `transformers` (ADR 0011) | Apache-2.0 code and weights, adapter already verified |
| Tracker | Existing `perception/track.py` (association + Kalman velocity) | Built and tested; no new dependency |
| 3D lifting | **Ground-plane contact**: back-project the box's bottom edge through the dataset calibration onto z = 0 | No depth model in v0; CPU-cheap; error is measurable against cuboids |
| Ego-motion | Dataset poses | Removes VO drift from v0; stated on the demo |
| Risk | 2D footprint sweep per component (shapely) over a constant-curvature path fan → distance, TTC, alert with hysteresis | Built except `sweep.py`; the 3D sweep is not needed for a top-down demo |
| Where the detector runs | **Precomputed on T4 by the same `detect` stage; replayed from cache on the Space.** Tracking, lifting and risk run live | Measured 750 ms/image on the laptop CPU (ADR 0011): 6 cameras cannot run live on a free CPU |
| Real-time claim | All stages live on a T4, measured and logged (V0-T12) | The claim comes from a measurement, not from the hosted page |
| Serving | **FastAPI + WebSocket**, one static page (video tiles + canvas top-down), one Dockerfile | Risk runs server-side in the same Python package; sliders change real computation |
| Hosting | **Hugging Face Space, Docker SDK, free CPU** (2 vCPU / 16 GB, sleeps after 48 h idle) | Free, runs the Python stack as-is, public URL |

**Pipeline stages (v0):** `decode → detect (cached | rtdetr) → track → lift → risk → output`.

**What the visitor sees and clicks:**

- A clip picker (3 bundled PandaSet scenes, privacy-blurred, 960 px wide).
- Camera tiles with tracked boxes and track IDs.
- A top-down view: the ego as 17 component shapes coloured by risk, tracked obstacles
  with velocity arrows, the predicted path fan.
- Alert level, worst component, per-component distance and TTC.
- **Per-component error budget:** distance ± σ for each part, which sources are measured,
  and "lower bound" on the total while any source is unmeasured. Obstacles are drawn
  with their expected-error radius from the lifting curve.
- Live throughput: processed fps, per-stage latency, dropped frames.
- Sliders that re-run risk live: safety margin, horizon.
- Toggle: components vs single box (from V1-T2).
- Always on screen: "Advisory research prototype. Not a safety device.", the dataset
  attribution, and "detections precomputed on a T4".

**Not in v0:** visitor uploads (no calibration, privacy), learned depth, visual odometry,
3D sweep, articulated joints, probabilities, segmentation-derived components, the
detection / tracking / geometry / timing budget sources, any trained model.

---

## 3. Architecture and layout

```
 once per vehicle                       per drive (stream)
 vehicle.yaml ─▶ model/ ─▶ URDF +       dataset clip ─▶ data/ adapter ─▶ frames + calib + pose
 (template, V0)  build     components                        │
 CC BY mesh ─▶ synth/ renders ─▶ seg/ ─▶ (V1)                ▼
                              │         stream/: decode ▶ detect ▶ track ▶ lift ▶ risk ▶ output
                              │                           (perception/)        (risk/)    │
                              └──────── ComponentModel ──────────────────────────▲        │ RiskFrame
                                                                                          ▼
                                                           app/ (FastAPI + WS + static page)
                                                           eval/ (metrics, baseline, report)
```

Modules talk only through the §4 schemas. No module imports another's internals.

```
src/vscs/
  core/        curve.py, budget.py, track.py: domain-neutral, imports nothing else from
               vscs (ADR 0021)
  common/      types.py (schemas), frames.py, config.py, io.py, datasets.py (licence guard)
  data/        dataset adapters → frames, calibration, poses, truth in veh frame
  model/       template.py (V0), decompose.py, joints.py, urdf.py, build.py
  synth/       parts.py, render.py (mesh → exact labels + renders, V1; on branch
               p4-multivehicle-data)
  seg/         masks2d.py, fusion3d.py, cleanup.py (V1-T7)
  perception/  detect.py, track.py (adapter over core/track.py), lift, pipeline.py
  risk/        motion.py, sweep.py, aggregate.py, alerts.py, probability.py (V1), engine.py
  stream/      queues.py, pipeline.py, metrics.py, replay.py (on branch p6-stream-scaffold)
  app/         server.py, static/
  eval/        metrics.py, baseline_bbox.py, evaluate.py, report.py
  capture/ recon/   PARKED (§7): built, fixture-tested, not on the critical path
scripts/       thin CLIs only          notebooks/colab/   thin wrappers, no logic
configs/       all tunable numbers     tests/unit, tests/fixtures (synthetic box vehicle)
metrics/results.jsonl                  docs/ STATUS, RISKS, devlog/, decisions/, evidence/
Dockerfile, docker-compose.yml, site.json
```

---

## 4. Conventions and data contracts

### 4.1 Frames and units (never change without an ADR and the developer's approval)

- **Units:** metres, seconds, radians. Severity is unitless.
- **`veh`:** right-handed, **x forward, y left, z up**; origin on the ground below the
  centre of the rear axle.
- **`cam`:** OpenCV, x right, y down, z forward.
- **`world`:** the dataset's world frame for that clip.
- **`T_a_b`** maps points in frame `b` into frame `a`: `p_a = T_a_b @ p_b`;
  `T_a_c = T_a_b @ T_b_c`. All transforms live in `common/frames.py`.
- **Timestamps:** `int64` nanoseconds. Never a frame index.
- **Every dataset adapter converts into `veh` at its boundary** and ships a test that
  proves the axes and origin against a known object. Nothing downstream knows which
  dataset a frame came from.

### 4.2 Schemas (`common/types.py` is the source of truth; `SCHEMA_VERSION = 2`)

```python
Component:      name, display_name, link, severity, min_z
ComponentModel: urdf_path, components, joints, scale_error_m, schema_version
Obstacle:       id, t_ns, kind ("static_geom"|"person"|"vehicle"|"cyclist"|"unknown"),
                center_veh, extent (full, not half), velocity_veh, pos_sigma_m,
                source ("occupancy"|"detector")
ComponentRisk:  component, min_distance_m, ttc_s | None, t_closest_s | None,
                p_contact [0,1], severity, risk (= p_contact * severity * speed_factor),
                obstacle_id | None
RiskFrame:      t_ns, ego_speed_mps, per_component, p_any_contact (= 1 - Π(1 - p_i)),
                expected_damage (= Σ risk_i), worst_component,
                alert_level ("none"|"caution"|"warning"|"critical"), schema_version
```

**Schema 3 lands in V0-T8 (ADR 0021):**

```python
ErrorBudget:    sigma_total_m, detection_m | None, lifting_m | None, tracking_m | None,
                geometry_m | None, timing_m | None,
                complete (False while any source is None: the total is a lower bound)
ComponentRisk:  + budget: ErrorBudget
Obstacle:       pos_sigma_m is filled from the lifting curve (same field, new meaning)
```

`RiskFrame` streams are JSON Lines; the WebSocket sends the same objects. A schema change
needs a version bump, an ADR with a migration note, and updated tests.

### 4.3 Code standards

- Python 3.11, venv (ADR 0001). Type hints everywhere. `ruff check` and `ruff format`
  pass.
- All tunable numbers live in `configs/*.yaml`. No magic constants.
- Deterministic: seed every random process and log the seed.
- Each script writes to a new run folder
  `data/processed/<stage>/<YYYYMMDD-HHMM>_<shortsha>/` with the resolved config, the git
  hash and a `run.log`.
- Notebooks hold no logic. Dependencies are pinned in `envs/`.
- Code that touches external data calls `require_use(dataset_id, use)`.

### 4.4 Git

- Branch per task (`v0-t5-sweep`); commit messages `V0-T5: <imperative summary>`.
- Merge to `main` only with `pytest -q` and `ruff check .` green.
- Never commit `data/` (except `MANIFEST.md`), keys, tokens or `.env`. No binary over
  5 MB outside `docs/evidence/`. Demo clips live in the Space repo, not here.

---

## 5. Testing

- `tests/fixtures/` holds the synthetic world (box vehicle with labelled sub-boxes, a
  pole, a kerb, known camera poses). Each module has at least one test with an
  analytically known answer.
- Required: transform inverse and composition; projection round-trip; ground-contact
  lifting on a known pixel; every dataset adapter's frame test; swept distance and TTC
  on the fixture; `p_any_contact`; alert hysteresis; queue backpressure; the WebSocket
  message validates as a `RiskFrame`; budget quadrature and `None` handling; curve fit
  and lookup on synthetic errors with a known σ; `core/` import isolation; label fusion
  on synthetic multi-view masks (an occluded point is not labelled).
- Smoke test: fixture clip → stream pipeline → valid `RiskFrame` stream in under 60 s on
  the laptop CPU.
- Coverage for `common/` and `risk/` stays at or above 80%.

---

## 6. Phases and tasks

Budget is about 5–8 h/week. **December is exams: no planned work.** A phase is complete
when every criterion is met with saved evidence. Rigor waits for v0: no eval splits, no
baseline, no second dataset before V0-T14 is done. The one piece of measurement that
does not wait is the lifting error curve (§1.1): the demo cannot show trust without it.

### V0-A: offline end-to-end with a measured budget (now → 2027-01-03)

| ID | Task | Acceptance criterion |
|---|---|---|
| V0-T1 | Housekeeping: merge `p6-stream-scaffold`; re-seed `docs/RISKS.md` from §8; developer accepts ADR 0020 and ADR 0021 | `pytest -q` and `ruff check .` green on `main`; STATUS shows the new task IDs |
| V0-T2 | PandaSet access and licence (developer, §11), registry entry; fetch 3 demo scenes + 10 curve scenes | Entry is `license_status: clear`; `require_use("pandaset", "publish_media")` passes; sha256 in `data/MANIFEST.md` |
| V0-T3 | PandaSet adapter → frames, intrinsics, `T_veh_cam`, ego pose, cuboids in `veh` | Projected cuboid corners land inside their image boxes on 20 sampled frames (median error < 5 px); axis test passes |
| V0-T4 | Template component model from `configs/vehicle.yaml` | 17 components; footprint union within ±2 cm of the published length and width; loads through the existing URDF loader |
| V0-T5 | **`risk/sweep.py`** (core) | The existing target tests on `p3-t4-sweep` pass, including `t_closest_s`; teach-back done |
| V0-T6 | Detections on T4 for all 13 scenes, written as a cache | Cache covers every frame of 6 cameras; CU logged; a re-run with the same seed gives the same boxes within 1 px |
| V0-T7 | **Lifting error curve** (core, `core/curve.py`): lifted position vs PandaSet cuboids on the 10 curve scenes, binned by range and viewing angle | Curve saved with its git hash and sample counts; every populated bin has ≥ 30 samples (sparser bins merged); 1-σ coverage on the 3 demo scenes **reported**; logged to `metrics/results.jsonl`; teach-back done |
| V0-T8 | **Error-budget core + schema 3** (core, `core/budget.py`, `core/track.py`; ADR 0021): every `ComponentRisk` carries an `ErrorBudget`; `Obstacle.pos_sigma_m` comes from the curve | Quadrature and per-component known-answer tests pass; unmeasured sources are `None`, never 0; `core/` imports nothing from the rest of `vscs` (import test); teach-back done |
| V0-T9 | Offline run: cache → track → lift → risk + budget on one scene | Valid schema-3 `RiskFrame` JSONL for the whole scene; every `ComponentRisk` has a budget; Rerun view saved to `docs/evidence/` |

### V0-B: stream it and ship it (2027-01-04 → 2027-01-24)

| ID | Task | Acceptance criterion |
|---|---|---|
| V0-T10 | Stream pipeline with real stages; replay at native rate; drop-stale backpressure | At 1× replay on the laptop with cached detections: processed fps ≥ 9 of 10, p95 end-to-end latency < 150 ms; at 2× overload latency stays bounded and drops are counted |
| V0-T11 | Web app: FastAPI + WebSocket + static page with everything in §2, including the per-component budget | Runs locally in a browser on all 3 clips; a slider change alters the next `RiskFrame`; each component shows distance ± σ and its measured/unmeasured sources; WebSocket schema test passes |
| V0-T12 | T4 benchmark, every stage live including the detector, 6 cameras | Per-stage and end-to-end fps and p50/p95 latency in `metrics/results.jsonl`, tagged with hardware |
| V0-T13 | Privacy pass on the demo clips: face and plate blur, metadata stripped | Developer visually confirms every bundled frame set; confirmation recorded in the devlog |
| V0-T14 | Docker + deploy to the Space | `docker compose up` works on a clean machine; the public URL plays all 3 clips; cold start < 60 s; banner and attribution visible; README links it. **v0 shipped** |

### V1: rigor (late January → mid-March 2027)

| ID | Task | Acceptance criterion |
|---|---|---|
| V1-T1 | Scene splits: dev/test ≈ 70/30 **by scene**, stratified by closest-object class, frozen in `configs/eval.yaml`; the lifting curve is refitted on dev only | Split guard test passes; `test` is read once, in V1-T5 |
| V1-T2 | Single-box baseline through the identical pipeline; demo toggle | Runs on the same splits; toggle works on the Space |
| V1-T3 | Metric definitions and truth from cuboids: closest component, true closest approach, **budget calibration** (share of true errors inside k·σ) | Fixture known-answer tests for every metric |
| V1-T4 | **Full source breakdown**: measure detection, lifting, tracking, component geometry and timing separately on dev (§1.1) | All five sources measured and logged, none `None`; 1-σ coverage of the combined budget on dev is between 58% and 78% (nominal 68%), or the miss is explained by a measured correlation between sources |
| V1-T5 | **Evaluation on the held-out test split** | Logged, VSCS vs baseline: component attribution accuracy, TTC error, false alarms/min, warning lead time, detection recall by class, budget per source and its 1-σ and 2-σ coverage |
| V1-T6 | Mesh truth: **`classify_faces`** (core) on a CC BY car mesh; orbit renders with exact part labels, poses and depth | Every face of every named part gets one of the 17 classes or `none`; left/right symmetric by test; renders reproducible from one command; teach-back done |
| V1-T7 | **Component segmentation** (the core CV contribution): Grounding DINO → SAM 2 masks → **`seg/fusion3d.py`** multi-view fusion with depth-test occlusion → cleanup → CoACD → URDF. **Replaces the template in the demo** | Mean per-component IoU ≥ 0.70 against the exact mesh labels; each hull volume within ±15% of its part; the geometry budget term is re-measured from the fused surface; attribution accuracy on dev reported against the template; one real car sequence (CO3D car, numbers only) run and checked on 20 frames; teach-back done |
| V1-T8 | **`risk/probability.py`** (core): `p_contact = Φ(-d/σ)` with σ the component's budget total | Unit tests; every σ traces to `metrics/results.jsonl`; teach-back done |
| V1-T9 | **Cross-dataset: Argoverse 2**, ~10 validation logs, same pipeline, PandaSet-fitted curve and budget applied unchanged | Logged with `split: av2`: every V1-T5 metric, plus the **budget shift**: per-source σ ratio AV2 / PandaSet and the coverage of the PandaSet budget on AV2. No pass threshold; no AV2 media published |

### V2: polish (late March → April 2027)

| ID | Task | Acceptance criterion |
|---|---|---|
| V2-T1 | `scripts/report.py` → `docs/REPORT.md`; README with results and limitations | Every README number traces to a `metrics/results.jsonl` line |
| V2-T2 | Demo video (≤ 90 s) and a metrics panel on the Space | Published after the privacy pass and the developer's approval |
| V2-T3 | One upgrade, aimed at the largest budget source from V1-T5: learned metric depth for lifting, **or** the 3D sweep with joints, **or** a live CPU detector on one camera, **or sim-to-real with CARLA** (near-contact scenes with exact truth that the datasets lack; refit the curve in sim and report the sim → real budget shift as for V1-T9) | The targeted budget source shrinks on dev, or for CARLA the shift is logged per source; before and after logged |

### Cut order if behind

1. V2-T3 → 2. V2-T2 → 3. V1-T8 → 4. the real-sequence check inside V1-T7 →
5. V1-T9 (cross-dataset).
**Never cut:** V0-T7 (the lifting curve), V0-T14 (the deployed demo), V1-T2/V1-T5 (the
baseline comparison) or V1-T7 (component segmentation). If V1-T7 misses its IoU gate,
the §8.2 ladder applies; it is not dropped.

---

## 7. Parked (built, tested on the fixture, off the critical path)

Do not extend these without the developer's go-ahead. They stay in the repo and CI.

| Parked | Why |
|---|---|
| Own-vehicle capture: phone calibration, scan day, lot day, second vehicle, `capture/` | The MVP runs on existing footage |
| `recon/` (COLMAP SfM, marker scale, dense) and `recon/scale.py` | V1-T7 fuses on known poses and depth; a real scan needs a licence-clear scan dataset first |
| Trained student segmenter (old P4-T8 … T12, ADR 0017, ADR 0019) | No demo stage consumes it; V1-T7 is its teacher, so it becomes possible after V1 |
| Visual odometry, gyro fusion, occupancy map, underbody check | Dataset poses replace ego-motion; the rest is not visible in a top-down demo |
| Driver replay UI, audio, user test | The web demo replaces them |
| RTSP replay server, zone model (ADR 0009), stereo camera | Not needed to ship |

**Not parked:** `seg/` (masks, fusion, cleanup) is V1-T7. CARLA is a V2-T3 option.

---

## 8. Risks and fallbacks

`docs/RISKS.md` holds the live register (ID, risk, likelihood, impact, signal,
mitigation, fallback, status). V0-T1 re-seeds it from this table, rated with the
developer; old `R-` rows that no longer apply are closed with a pointer to ADR 0020.

### 8.1 Register

| ID | Risk | Detection signal | Mitigation |
|---|---|---|---|
| D-01 | PandaSet's terms are not CC BY 4.0, or the download is dead | Terms at download differ from the mirrors' tag; link fails | V0-T2 is first; fallback row "Demo dataset" |
| D-02 | Ground-contact lifting too coarse for component attribution (occluded box bottoms, non-flat road) | Curve median > 0.5 m within 10 m | Use only boxes with a visible bottom edge; the curve makes the error visible per obstacle; V2-T3 depth |
| D-03 | Free CPU Space cannot hold native rate | Processed fps < 9 or p95 > 150 ms on the Space | Cached detections; 960 px JPEG; drop-stale queues; fallback row "Hosting" |
| D-04 | Faces or plates published | Any unblurred frame in a clip | V0-T13 before every deploy; bundled clips only |
| D-05 | Thresholds or the curve tuned on the data they are scored on | Large dev/test gap; coverage much better on fit scenes | V0: curve scenes and demo scenes are disjoint. V1: splits by scene, frozen; `test` read once |
| D-06 | Alert flicker | Level toggles more than 2×/s | Hysteresis and minimum dwell (built) |
| D-07 | Same object seen by two cameras becomes two tracks | Duplicate obstacles at camera seams | Track in `veh` after lifting, not per image |
| D-08 | Template boxes misattribute near corners | Attribution errors cluster at component borders | Geometry term in the budget; V1-T7 replaces the template |
| D-09 | Frame or schema drift between adapters and risk | Adapter frame test or smoke test fails | §4 contracts; one adapter boundary |
| D-10 | Colab disconnects or compute units run out | Lost runs | Caches are per scene and resumable; CU logged |
| D-11 | Scope creep around coursework | Health AT RISK two sessions running | §6 cut order; §7 stays parked |
| D-12 | A dependency or weight turns out copyleft or non-commercial | Licence check at install | §10; registry guard |
| D-13 | Code outruns understanding: core branches pile up un-taught | More than two core branches awaiting teach-back | Hard cap of two; Claude stops core work and prepares the teach-back |
| D-14 | Budget sources are not independent, so quadrature understates the total | 1-σ coverage well under 68% | V1-T4 coverage gate; measure the correlation and add the cross term |
| D-15 | A budget shown with unmeasured sources reads as more trust than is earned (v0) | Demo shows a total without the "lower bound" label | Unmeasured sources are `None` and shown as such; total labelled a lower bound until V1-T4 |
| D-16 | SAM 2 / Grounding DINO labels bleed across component borders | IoU < 0.70; labels spill onto neighbours | Multi-view vote with occlusion test; thresholds; fallback row "2D segmentation" |
| D-17 | The curve does not transfer across datasets | Coverage on Argoverse 2 far from 68% | Not mitigated: this is the V1-T9 finding, reported per source |
| D-18 | Segmentation on renders does not carry to real photos | Real-sequence check in V1-T7 visibly worse | Report it; real scan data is the unblocker (§11 row 7) |

### 8.2 Fallback ladder (propose with evidence; the developer decides; record an ADR)

| Stage | Primary | Fallback 1 | Fallback 2 | Trigger |
|---|---|---|---|---|
| Demo dataset | PandaSet | comma2k19 (reported MIT, front camera only; verify first) | Argoverse 2 with no camera media: top-down view only | D-01 fires |
| Detector | RT-DETR r18 | RT-DETR r50 on T4, cached | YOLOX (Apache-2.0) | Recall on vehicles and people within 10 m < 0.8 on dev |
| Lifting | Ground contact | Ground contact + learned metric depth | Cuboid positions, labelled "oracle perception" | Curve median > 1 m within 10 m |
| 2D segmentation | Grounding DINO text prompts → SAM 2 | Manual click prompts into SAM 2 on keyframes | — | Mean IoU < 0.60 on a class after tuning |
| Component model (V1) | Fused segmentation (V1-T7) | Exact mesh labels from `classify_faces` (V1-T6) | Template | IoU < 0.70 after cleanup, or hull volume error > 15% |
| Collision check | shapely 2D footprints | trimesh / python-fcl 3D | — | Only by choice in V2-T3 |
| Hosting | HF Space, Docker, free CPU | Same Space, lower replay rate, labelled | Static page replaying precomputed `RiskFrame`s (sliders removed, labelled) | D-03 after tuning |
| Front end | Static page + canvas | Rerun web viewer | — | V0-T11 exceeds two sessions |

---

## 9. Privacy and honest claims

- Dataset frames can show faces and plates. **Before anything is deployed or enters
  `docs/evidence/`**, run the blur pass, strip metadata, and have the developer confirm
  by eye.
- Only media from datasets whose registry entry grants `publish_media` is ever shown.
  Non-commercial datasets give numbers, never pictures.
- Attribution for every dataset and model appears on the demo page and in the README.
- Never call VSCS a safety system or say it prevents collisions. Say "advisory",
  "research prototype", "evaluated on N scenes".
- Every published number traces to `metrics/results.jsonl` and names its hardware.
- The demo states what is not live (detections) and what is borrowed (ego poses).
- Report failures and limitations as prominently as successes.

---

## 10. Dependencies and licences

Permissive only (MIT, BSD, Apache-2.0). Check code **and** weights at install time.

| Tool | Use | Licence |
|---|---|---|
| RT-DETR (`PekingU/rtdetr_r18vd`, `transformers`) | detection | Apache-2.0 code and weights (checked 2026-10-01, ADR 0011) |
| OpenCV | image I/O, projection | Apache-2.0 |
| shapely | 2D footprints | BSD |
| trimesh / CoACD | mesh parts, convex decomposition (V1) | MIT / MIT |
| Grounding DINO / SAM 2 | component masks (V1-T7), Colab T4 | Reported Apache-2.0; verify code and checkpoints at install |
| CARLA | sim-to-real option (V2-T3) | MIT code; assets under separate terms: ask first |
| FastAPI / uvicorn | web app | MIT / BSD (verify at install) |
| Rerun | developer view | MIT / Apache-2.0 |
| **Ultralytics YOLO** | — | **AGPL-3.0: do not use** |
| ffmpeg | not required (clips ship as JPEG frames) | LGPL/GPL builds: ask first |

Datasets are governed by `configs/datasets.yaml`: permitted uses are *derived* from the
licence by code, never set by hand.

---

## 11. To verify by hand (developer)

| # | Check | Blocks |
|---|---|---|
| 1 | **PandaSet terms.** Mirrors tag it CC BY 4.0; Scale's legal page refers to separate "PandaSet Terms of Use". Read the terms shown at pandaset.org when you sign up, and confirm they allow showing frames publicly with attribution | V0-T2 |
| 2 | **PandaSet download.** The official form has been reported broken; mirrors exist on Hugging Face (`georghess/pandaset`, 44.5 GB) and Kaggle. A mirror does not change the licence | V0-T2 |
| 3 | **Hugging Face account** and a free Docker Space; current free-tier limits | V0-T14 |
| 4 | **Ego dimensions** for `configs/vehicle.yaml` (Chrysler Pacifica: length, width, height, wheelbase, track) from a manufacturer spec sheet | V0-T4 |
| 5 | **CC BY car mesh**: the licence and author of the exact file (ADR 0019 lists the candidates as unverified) | V1-T6 |
| 6 | **Grounding DINO and SAM 2**: code and checkpoint licences at install time (both reported Apache-2.0) | V1-T7 |
| 7 | **CO3D car** (`co3d_car` is `unverified` in the registry): confirm CC BY-NC 4.0 at source. Numbers only, never media | V1-T7 real-sequence check |
| 8 | **Argoverse 2** terms unchanged since 2026-10-19 | V1-T9 |
| 9 | **Overstride figures quoted in §1.2** (3.50° → 1.77°, 1.84×): supplied by the developer on 2026-11-16, not read from Overstride by Claude. Correct them here if they differ | README, V2-T1 |

---

## 12. Reporting

- **`docs/STATUS.md`** (overwrite each session): `phase`, `current_task`, core branches
  awaiting teach-back,
  `health` (ON TRACK | AT RISK | BLOCKED), done since last update, verified metrics,
  blockers, triggered risks, next 3 tasks, GPU compute units this month.
- **`docs/devlog/YYYY-MM-DD.md`**: goal, what was done, evidence, surprises, decisions
  (→ ADR), risks touched, next, time and CU.
- **`docs/decisions/NNNN-title.md`**: context → options → decision → consequences →
  evidence. One per choice between alternatives, schema change, fallback or major
  dependency.
- **`metrics/results.jsonl`**: append-only, machine-written, never hand-edited:
  `{"ts", "git", "task", "metric", "value", "split", "run_dir", "notes"}`; the hardware
  goes in `notes` until a field is added by ADR.
- **Health:** AT RISK when a task passes 2× its expected time, a fallback trigger fires,
  or a phase will slip its window. BLOCKED when progress needs the developer's decision
  or data. Either one leads the hand-off summary.
- **`site.json`** (repo root): whenever STATUS changes phase, stage progress or health,
  update it in the same commit. Stages are Scan, Segment, Track, Score (do not rename);
  under this plan Scan and Segment together mean the component model. Each status is
  `{label, state}` with state one of active, partial, waiting, done; labels under 40
  characters; `currentStage` is the zero-based index of the stage being worked on.

---

## 13. Quick reference

```bash
.venv\Scripts\activate
pytest -q && ruff check .                         # before every merge to main

# Exists today
python scripts/fetch_dataset.py <id>              # id from configs/datasets.yaml
python scripts/export_model.py ...                # components -> URDF + components.yaml
python scripts/perceive.py ...                    # detect + track + lift -> obstacles.jsonl
python scripts/risk.py --model <run> --perception <run> [--baseline]   # needs sweep.py
python scripts/view.py --model <run> --risk <frames.jsonl>             # Rerun .rrd
python scripts/benchmark.py ... [--log-metrics]
python scripts/evaluate.py ... ; python scripts/report.py

# Added by this plan (move up as each lands)
python scripts/fit_curve.py --scenes <curve scenes>                    # V0-T7, lifting curve
python scripts/stream.py --clip <id> [--detect cached|rtdetr]          # V0-T10
python scripts/serve.py                                                # V0-T11, local demo
docker compose up                                                      # V0-T14
```

Keep this list in sync with reality.
