# VSCS Risk Register

Seeded 2026-09-24 from CLAUDE.md §8.1. **Likelihood and impact rated with the developer on
2026-09-24.** Re-rate at every phase gate (§8).

How the ratings were set: R-01 and R-13 were set from specific facts the developer
supplied (the van is dark but glossy with significant glazing; an external drive and cloud
storage are both already in place). The remaining fourteen were proposed by Claude,
reviewed, and accepted without change. Any of them can be revisited at the Phase 1 gate.

Status values: `OPEN` (identified, mitigation in place or planned) · `WATCH` (detection
signal trending toward the trigger) · `TRIGGERED` (trigger fired; fallback proposed to the
developer) · `CLOSED` (no longer possible).

A fallback is **never** switched to unilaterally: Claude proposes it with the trigger
evidence and the developer decides, and the decision is recorded as an ADR (§0, §8.2).

## Technical risks

| ID | Risk | L | I | Detection signal | Mitigation (default) | Fallback (§8.2 ladder row) | Status |
|---|---|---|---|---|---|---|---|
| R-01 | Reflective paint / windows break SfM | high | high | <90% frames registered; holes in body panels | Overcast capture, slow walk, 3 heights, lock AE/AF, textured markers on ground. **Dark glossy paint (this van): the failure is mirror-like reflection and low panel texture, not blown highlights** — so prioritise overlap, ground texture and extra markers over merely avoiding sun | **Reconstruction**: COLMAP → GLOMAP / hloc+SuperPoint+LightGlue → phone-app export, still marker-scaled | OPEN |
| R-02 | Scale error or drift | med | high | Tape-measure mismatch > 2 cm | ≥3 markers of known size spread around the vehicle; check against 4 independent dimensions; **measure the printed marker, never trust the nominal size**. *Adopted 2026-09-25 (developer's decision): mount markers on boards propped ~45° facing the walking path - flat ground markers are seen at grazing angles and mostly fail the corner-quality gate (devlog 2026-09-25).* | **Scale**: ArUco/AprilTag → tape-measured wheelbase+length → manufacturer spec sheet | OPEN |
| R-03 | Phone variable frame rate + rolling shutter corrupt timing | high | med | Non-monotonic or irregular timestamps; ego-motion jitter | Container timestamps, never frame index; slow motion during capture; highest fixed fps available. Enforced by `validate_timestamp_stream` and `capture/frames.py` | — (mitigation only) | OPEN |
| R-04 | Autofocus / zoom changes intrinsics mid-capture | med | med | Calibration reprojection error rises on capture frames | Lock focus and exposure, main lens only, no zoom | — (re-run P0-T7 calibration if it fires) | OPEN |
| R-05 | SAM / Grounding DINO labels bleed across component boundaries | high | high | IoU < 0.70; labels spill onto neighbours | Multi-view voting with occlusion test; confidence thresholds; manual prompts for weak classes | **2D segmentation**: text prompts → manual click prompts on keyframes. **3D labels**: fusion → fusion + CloudCompare correction → fully manual | OPEN |
| R-06 | Monocular / phone depth too inaccurate for contact-level precision | high | high | Depth error > 10 cm at 3 m | Motion stereo + metric depth fusion; ground-plane anchoring; report error honestly | **Depth**: motion stereo + learned metric → learned metric + ground-plane scale → Tier 1 stereo camera (needs approval) | OPEN |
| R-07 | Thin poles, low curbs, glass missed | med | high | Low recall on those categories | Dedicated test cases in lot runs; occupancy layer independent of the detector | **Detection**: permissive detector → Grounding DINO at lower fps → occupancy-only (no semantics) | OPEN |
| R-08 | Ego-motion drift corrupts persistent map | med | med | Static cones appear to "move" over a pass | IMU fusion; short horizons; reset map per manoeuvre | **Ego-motion**: visual-inertial → visual only → IMU + ground-truth-constrained lot runs only | OPEN |
| R-09 | Overfitting thresholds to the data being evaluated on | med | high | Dev/test gap large at P5 | Split assigned mechanically at ingest, stratified, and immovable afterwards; `test` touched once, at P5-T2 | — (protocol control, §11) | OPEN |
| R-10 | Alert flicker / false-alarm spam | med | med | Alerts toggle > 2×/s; false alarms per minute high | Hysteresis, minimum dwell time, severity weighting | — (design control; `risk.yaml` alert block) | OPEN |
| R-11 | Colab disconnects / compute units run out | med | med | Lost runs; CU balance low | Checkpoint to Drive every stage; small resumable jobs; T4 not A100; log CU per job | — (operational; warn before any job over ~10 CU) | OPEN |
| R-12 | Dependency conflicts (nerfstudio, COLMAP, torch, CUDA) | high | med | Env install fails | Separate envs; pinned versions; record working combos in an ADR | — (see ADR 0001, ADR 0003) | OPEN |
| R-13 | Data loss | low | high | Missing or corrupt raw file | **External drive and cloud are both already in place**, so the 3-copy rule is achievable on capture day itself; sha256 in `data/MANIFEST.md`; **verify after copy**, enforced by `scripts/ingest.py` | — (prevention only; raw capture cannot be recovered) | OPEN |
| R-14 | Scope creep / time crunch with coursework | high | med | Health AT RISK two sessions in a row | Cut order in §6; Phase 3 MVP protected | **Cut order**: P5-T5 user test → P4-T4 dynamic tracking → P4-T7 probabilities. Never cut P5-T1/T2 | OPEN |
| R-15 | Articulated joint states unknown during drives | med | low | Wrong geometry used (door open vs closed) | Manual joint-state input per run in v0.1. **The sliding door is now two scanned geometries rather than a joint (ADR 0004)**, so the hand-entered state selects a measured model instead of driving a kinematic approximation | — (auto-detection from open/closed scans is a stretch goal) | OPEN |
| R-16 | Schema / convention drift between modules | med | med | Integration test fails; frames mismatched | §4 contracts, `SCHEMA_VERSION`, `extra="forbid"` on every schema, smoke test on every merge | — (enforced by tests) | OPEN |

## Ratings rationale (the non-obvious ones)

- **R-01 high/high.** The van is dark maroon and glossy with significant glazing. Dark
  paint avoids white's blown-out-highlight failure, but glossy dark panels behave like
  mirrors: they reflect sky and surroundings, and those reflections move with the camera,
  which is exactly what defeats feature matching. Under the overcast the checklist
  mandates, large panels tend toward uniform low texture instead. Either way the panels
  are the weak surface, so the mitigation emphasis is texture and overlap.
- **R-06 high/high.** The hardest technical risk in the project. Contact-level precision
  from a single phone camera is genuinely difficult, and the honest outcome may be a
  reported error rather than a passing gate.
- **R-12 high/med.** Already fired twice on day one: pip backtracking on `open3d`, and
  two missing native binaries (COLMAP, ffmpeg).
- **R-13 low/high.** Low *only because* both backup destinations already exist and
  `scripts/ingest.py` verifies after copying. If either disappears, this goes straight
  back to medium.
- **R-14 high/med.** A 5–8 h/week budget around first-year engineering, with December
  written off for exams. Likelihood is high by construction; the cut order is what keeps
  impact at medium.

## Triggered log

Nothing triggered yet. When a trigger fires, append here: date · risk ID · the evidence
that fired it · what was proposed · the developer's decision · ADR number.

| Date | ID | Trigger evidence | Proposed fallback | Decision | ADR |
|---|---|---|---|---|---|
| _(none)_ | | | | | |

### Near misses worth remembering

Not triggers, but the same failure mode caught early:

| Date | ID | What happened | Caught by |
|---|---|---|---|
| 2026-09-24 | R-02 | The generated marker sheet's cut-guide rectangle was detected *instead of* the marker border, reporting the side 13% oversize. Unfixed, this would have scaled the entire van model by 13% and surfaced only as an unexplained P1-T6 failure. | A round-trip test that measured the detected square rather than only checking the decoded id |
| 2026-09-24 | R-12 | `pip install` stalled >12 min backtracking through `open3d` wheels | Watching an install that should have taken one minute |
