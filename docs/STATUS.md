# VSCS Status — updated 2026-09-24
mode: pair            # pair | build
phase: P1
current_task: P3-T4 risk/sweep.py - developer's first draft (pair mode), tests waiting on branch p3-t4-sweep
#             P1-T4 built but unverifiable without real footage (needs ffmpeg + a real clip)
#             P2-T1 approved 2026-09-24 - vocabulary and severity are now frozen
health: ON TRACK

## Done since last update

**Phase 3 risk engine started early on the synthetic fixture** (developer's go-ahead,
2026-09-24; §6 requires it for next-phase core work). Motivation: Phase 3 is the MVP, it
needs only the fixture world, and building it now protects it from the December exam gap.

- **P3-T3** [done] `risk/motion.py` — constant-curvature path fan. Straight paths are
  `x = v t` exactly, a quarter turn at kappa 0.2 lands at (5, 5) facing pi/2, every point
  lies on the turning circle to 1e-9. 37 tests, 100% coverage.
- **P3-T4** [pending, developer's draft] `risk/sweep.py` is a **core module**, so per pair
  mode the developer writes it. The contract (signatures + docstrings, every function
  `NotImplementedError`) and the known-answer tests are on branch `p3-t4-sweep`, deliberately
  red and unmerged. Headline target: TTC = 0.37 s for the rear-right corner against the
  fixture pole. Every number in those tests was checked against a brute-force oracle first,
  so the target is achievable. Agreed design: clearance and TTC from the estimated-curvature
  path; the whole fan is still swept for P4-T7.
- **P3-T5** [done] `risk/aggregate.py` + `risk/alerts.py`. Acceptance met: parked on the
  0.50 m warning line with 4 cm noise for 10 s, the raw level changes 150 times and the
  filtered level twice, never more than twice in a second (R-10). With hysteresis disabled
  the same input gives 150 transitions at up to 21/s, so the test genuinely discriminates.
- **Two defects in my own merged P3-T5 work, found by wiring the pipeline together, fixed
  (ADR 0005).** (1) A contact predicted 2.9 s away was graded "critical", because the
  sweep's horizon-minimum clearance is 0 for every predicted contact; contacts are now
  graded by TTC, near misses by closest approach. (2) The fixture frame named the
  *sliding door* for the pole, from a phantom 2.37 s contact behind the corner's 0.37 s
  impact; new `risk/engine.py` cuts the sweep at the first predicted impact. Neither
  touched the §4.2 schema or the `sweep.py` contract.
- **End-to-end smoke test** wired to the real `sweep.py` on `p3-t4-sweep` - the second
  target after `test_sweep.py`. Checked achievable with the brute-force oracle (20 s, inside
  the 60 s budget).
- v0.1 uses a deterministic 0/1 contact indicator until P4-T7 (§6 cut-order fallback #3),
  and `alerts.use_p_contact: false` stops that indicator making every predicted contact
  "critical".


Second session today. Phase 0 is complete except for the three tasks that need captures
(see Blockers). Phase 1's buildable half is now done.

- **P1-T1** [done] **approved 2026-09-24** — `docs/capture_checklists.md`: pre-flight,
  scan day, lot day, post-capture backup, privacy, and abort triggers. Its acceptance
  criterion is literally "developer reviewed it", so it is not done until you read it.
  Approved, but it still ends with **two open questions that block booking a capture
  day** — approving the document did not answer them (below).
- **P1-T4** [partial] built, acceptance unverifiable yet — the ingest path:
  - `capture/frames.py` — per-frame container timestamps (ffprobe preferred, OpenCV
    fallback), VFR/jitter/dropped-frame analysis, frame extraction writing real `t_ns`.
  - `capture/sync.py` — video/IMU time offset by normalised cross-correlation with
    sub-sample refinement.
  - `capture/ingest.py` + `scripts/ingest.py` — sha256, copy-then-verify, manifest rows,
    and permanent dev/test split assignment.
- **Capture validator** [done] `capture/preflight.py` + `scripts/check_capture.py` —
  turns the checklist's two open questions into a repeatable pass/fail: video timestamps,
  IMU parsing, video/IMU sync recovery, and a stabilisation/rolling-shutter check. Also
  usable as a go/no-go on the morning of a capture.
- **Marker sheets** [done] `capture/markers.py` + `scripts/make_markers.py` — printable ArUco
  sheets at an exact physical size, with a 100 mm check bar so a rescaled print is visible.
- ADR 0003 extended: **ffmpeg is also not installed**, and is needed before October.
- **P2-T1 approved** (2026-09-24): 13 components and the severity weights signed off in
  `configs/seg.yaml` and `configs/severity.yaml`. Before approving I checked the list
  against the physical van and found it had nothing for rear protrusions — on many vans a
  **tow bar** is the actual first-contact point when reversing. You confirmed the van has
  none (no tow bar, rear step, spare carrier or ladder), so the rear bumper and its two
  corners really are the rearmost points and nothing needed adding. **The list is now
  frozen for the phase**: changing it after P2-T3 runs would invalidate every
  per-component IoU number.
- **Risk ratings agreed** with you, and **ADR 0004** written: the sliding door is modelled
  as two scanned collision variants rather than a prismatic joint, because it runs on a
  curved track. `configs/model.yaml` and `configs/risk.yaml` updated accordingly, and the
  door-open scan pass is now marked load-bearing in the checklist.

## Verified metrics (link to metrics/results.jsonl entries)

**Still none, deliberately.** `metrics/results.jsonl` is empty and `docs/REPORT.md` lists
all twelve acceptance gates as unproven. Nothing measurable has been captured.

Session evidence (test-suite results, **not** metrics, and not written to `results.jsonl`):

- `pytest` — **344 passed, 1 skipped** in 7.2 s (was 238)
- `ruff check .` and `ruff format --check .` — clean across 56 files
- marker round-trip — a rendered 150 mm sheet is detected as id 0 and **measures 150 mm**
- sync — a known offset is recovered to within 10 ms, with the correct sign
- split assignment — deterministic, ~70/30, and stratified across pole/kerb/box

## Blockers

Nothing blocks me from building. Everything below needs you.

1. **Run the pre-flight test** (`scripts/check_capture.py`) to settle the checklist's two
   open questions. They are now a measurement rather than a decision: record ~30 s at home
   with a sharp shake at each end, plus an IMU log, and the validator reports whether the
   setup is usable. Twenty minutes, no van required. The questions remain open until it
   has actually been run:
   - **How is IMU recorded alongside video?** A phone's stock camera app does not log IMU,
     and P1-T3 requires video + IMU. This needs an app choice *and a test at home* before
     a capture day, or the day produces video with no inertial data.
   - **Can stabilisation be turned off?** If EIS cannot be disabled it warps frames and
     the calibration no longer describes them.
2. **Install ffmpeg and COLMAP** (both native binaries, ADR 0003). ffmpeg is needed for
   trustworthy VFR timestamps; COLMAP for P0-T6 and P1-T5.
3. **P0-T6** warm-up video, **P0-T7** checkerboard, **P0-T5** one Colab run — unchanged
   from last session.
4. ~~Two review items~~ — **both closed 2026-09-24.** All 16 risks are now rated with
   you, and reviewing `JointSpec` turned up a real modelling error (ADR 0004).

## Open risks triggered (IDs from RISKS.md)

None triggered. Two worth noting:

- **R-03 is the reason P1-T4 cannot be called done.** The VFR handling is written and
  tested, but only against *generated constant-frame-rate* video. That is not evidence
  about real phone footage. The honest status is "built, unverified".
- **R-02 nearly bit already.** The marker sheet's cut-guide rectangle was being detected
  *instead of* the marker, reporting the side 13% oversize — which would have scaled the
  whole van model by 13%. Caught by a round-trip test, fixed with open corner marks.
- All 16 ratings are now **agreed** (2026-09-24). R-01 and R-13 were set from facts you
  supplied; the other fourteen were accepted as proposed. Re-rate at the Phase 1 gate.

## Next 3 tasks

1. **You: first draft of `risk/sweep.py`** on branch `p3-t4-sweep`. Run
   `pytest tests/unit/test_sweep.py` until it is green; `tests/test_pipeline_smoke.py` on
   the same branch then checks the whole fixture pipeline end to end. I review and harden
   it after.
2. **You: the at-home capture work** — ffmpeg + COLMAP, the pre-flight test
   (`scripts/check_capture.py`), checkerboard calibration, marker printing. This is still
   the critical path: October is the whole capture window.
3. **Me: P1-T5/T6** — COLMAP wrapper, then the pair-mode walkthrough of `recon/scale.py`.

## GPU usage this month (approx compute units)

**0 CU.** No Colab session run yet.
