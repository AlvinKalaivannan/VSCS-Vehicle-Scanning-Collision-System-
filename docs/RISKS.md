# VSCS Risk Register

Seeded 2026-09-24 from CLAUDE.md §8.1. Re-rate at every phase gate.

> ### ⚠️ Likelihood / impact ratings below are PROPOSED, not agreed.
> CLAUDE.md §8 says these are rated **with the developer**. The values here are Claude's opening proposal so the
> register is usable immediately; they are not a joint decision yet. **Action for the next session: walk the 16
> rows together and confirm or change each rating.** Until then, treat the ratings as a draft and the
> detection signals / mitigations / fallbacks as authoritative (those are copied from CLAUDE.md §8).

Status values: `OPEN` (identified, mitigation in place or planned) · `WATCH` (detection signal trending toward
the trigger) · `TRIGGERED` (trigger fired; fallback proposed to the developer) · `CLOSED` (no longer possible).

A fallback is **never** switched to unilaterally: Claude proposes it with the trigger evidence and the developer
decides, and the decision is recorded as an ADR (CLAUDE.md §0, §8.2).

## Technical risks

| ID | Risk | L | I | Detection signal | Mitigation (default) | Fallback (§8.2 ladder row) | Status |
|---|---|---|---|---|---|---|---|
| R-01 | Reflective paint / windows break SfM | high | high | <90% frames registered; holes in body panels | Overcast capture, slow walk, 3 heights, lock AE/AF, textured markers on ground | **Reconstruction**: COLMAP → GLOMAP / hloc+SuperPoint+LightGlue → phone-app export, still marker-scaled | OPEN |
| R-02 | Scale error or drift | med | high | Tape-measure mismatch > 2 cm | ≥3 markers of known size spread around the vehicle; check against 4 independent dimensions | **Scale**: ArUco/AprilTag → tape-measured wheelbase+length → manufacturer spec sheet | OPEN |
| R-03 | Phone variable frame rate + rolling shutter corrupt timing | high | med | Non-monotonic or irregular timestamps; ego-motion jitter | Container timestamps, never frame index; slow motion during capture; highest fixed fps available | — (mitigation only; enforced by the ns-timestamp validator in `common/types.py`) | OPEN |
| R-04 | Autofocus / zoom changes intrinsics mid-capture | med | med | Calibration reprojection error rises on capture frames | Lock focus and exposure, main lens only, no zoom | — (mitigation only; re-run P0-T7 calibration if it fires) | OPEN |
| R-05 | SAM / Grounding DINO labels bleed across component boundaries | high | high | IoU < 0.70; labels spill onto neighbours | Multi-view voting with occlusion test; confidence thresholds; manual prompts for weak classes | **2D segmentation**: text prompts → manual click prompts on keyframes. **3D labels**: fusion → fusion + CloudCompare correction → fully manual | OPEN |
| R-06 | Monocular / phone depth too inaccurate for contact-level precision | high | high | Depth error > 10 cm at 3 m | Motion stereo + metric depth fusion; ground-plane anchoring; report error honestly | **Depth**: motion stereo + learned metric → learned metric + ground-plane scale → Tier 1 stereo camera (needs approval) | OPEN |
| R-07 | Thin poles, low curbs, glass missed | med | high | Low recall on those categories | Dedicated test cases in lot runs; occupancy layer independent of the detector | **Detection**: permissive detector → Grounding DINO at lower fps → occupancy-only (no semantics) | OPEN |
| R-08 | Ego-motion drift corrupts persistent map | med | med | Static cones appear to "move" over a pass | IMU fusion; short horizons; reset map per manoeuvre | **Ego-motion**: visual-inertial → visual only → IMU + ground-truth-constrained lot runs only | OPEN |
| R-09 | Overfitting thresholds to the data being evaluated on | med | high | Dev/test gap large at P5 | Split runs into **dev** and **held-out test** at ingest; `test` touched once, at P5-T2 | — (protocol control, CLAUDE.md §11; split recorded in `configs/eval.yaml` and `data/MANIFEST.md`) | OPEN |
| R-10 | Alert flicker / false-alarm spam | med | med | Alerts toggle > 2×/s; false alarms per minute high | Hysteresis, minimum dwell time, severity weighting | — (design control; `risk.yaml` alert block + hysteresis state machine test) | OPEN |
| R-11 | Colab disconnects / compute units run out | med | med | Lost runs; CU balance low | Checkpoint to Drive every stage; small resumable jobs; T4 not A100; log CU per job in the devlog | — (operational; warn the developer before any job over ~10 CU) | OPEN |
| R-12 | Dependency conflicts (nerfstudio, COLMAP, torch, CUDA) | high | med | Env install fails | Separate envs; pinned versions; record working combos in an ADR | — (see ADR 0001, ADR 0003; do not force one env for everything) | OPEN |
| R-13 | Data loss | low | high | Missing or corrupt raw file | 3 copies (laptop, external drive, cloud); sha256 in `data/MANIFEST.md`; **verify after copy** | — (prevention only; loss of raw capture data cannot be recovered) | OPEN |
| R-14 | Scope creep / time crunch with coursework | high | med | Health 🟡 two sessions in a row | Cut order in CLAUDE.md §6; Phase 3 MVP protected | **Cut order**: P5-T5 user test → P4-T4 dynamic tracking → P4-T7 probabilities. Never cut P5-T1/T2 | OPEN |
| R-15 | Articulated joint states unknown during drives | med | low | Wrong geometry used (door open vs closed) | Manual joint-state input per run in v0.1 | — (auto-detection from open/closed scans is a stretch goal) | OPEN |
| R-16 | Schema / convention drift between modules | med | med | Integration test fails; frames mismatched | CLAUDE.md §4 contracts, `SCHEMA_VERSION` in `common/types.py`, smoke test on every merge | — (enforced by tests; any schema change needs a version bump + ADR + updated tests) | OPEN |

## Triggered log

Nothing triggered yet. When a trigger fires, append here: date · risk ID · the evidence that fired it · what was
proposed · the developer's decision · ADR number.

| Date | ID | Trigger evidence | Proposed fallback | Decision | ADR |
|---|---|---|---|---|---|
| _(none)_ | | | | | |
