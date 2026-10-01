# VSCS Status — updated 2026-10-01
mode: pair            # pair | build
phase: P1 (capture prep); Phase 2 plumbing built early on the fixture (developer's go-ahead)
current_task: autonomous /goal session; Wave 2 done on the fixture; streaming scaffolding next (unmerged branch). October captures remain the critical path
health: ON TRACK

## Done since last update

- **External datasets (ADR 0006):**
  - Licence-derived use guard and a fetcher (`scripts/fetch_dataset.py`).
  - Both Roboflow car-parts sets downloaded to `data/external/`:
    - 21-class: train and report numbers, but no public images.
    - 19-class: private only; it looks like a re-upload of DSMLR's unlicensed data.
- **P2-T3:** contract, target tests and exact fixture renderer on branch `p2-t3-fusion`
  for your draft. The IoU metric is on `main`.
- **P2-T5** cleanup: done on the fixture.
- **P2-T6** decomposition: CoACD, pinned and MIT-licensed; done on the fixture.
- **P2-T7** joint kinematics: done on the fixture.
- **P2-T8** URDF export and load: done on the fixture, with `scripts/export_model.py`.
- **P2-T2** masks stage (Colab; streamed Grounding DINO + SAM 2): done on the fixture,
  with `scripts/seg.py` and `notebooks/colab/20_seg.ipynb`.
- **2026-10-01 (autonomous, logged in `docs/operations_log.md`):**
  - P3-T1 Rerun dev view, P3-T2 perception v0, P4-T3 occupancy.
  - P5-T2 risk metrics, P5-T1 single-box baseline.
  - P4-T4 tracking, P4-T6 underbody, P5-T4 driver replay.
  - Perception glue with mask-based ground contact.
  - An end-to-end fixture drive.
  - The RT-DETR detector (ADR 0011).
  - Ground visual odometry + gyro fusion (P4-T2).
  - The CLIs `perceive`, `risk` (with `--baseline`), `evaluate` (test split once, R-09)
    and `benchmark`.
  - The Phase 0 report draft.
  - **Every script planned for Phases 0–5 now exists.**
  - `docs/runbook.md`: every command from capture day to evaluation, in order.
  - **The offline chain now has no missing commands:**
    - `scripts/scale.py` (P1-T6): writes `sfm_to_veh.json`;
    - `scripts/fuse.py` (P2-T3/T5): writes `cleaned.npz` for `export_model.py`.

    Both wrap your drafts and stop with exit 3 until those exist. The dense stage now
    writes the undistorted cameras as text, and `20_seg.ipynb` masks the undistorted
    images, which fusion needs pixel-aligned.
  - All on the fixture, and all merged locally.
- **ADR 0008 / proposed ADR 0009** (streaming merge) are on branch `docs-streaming-merge`,
  **not merged**, awaiting your OK.
- None of these is *accepted*: every §6 gate needs the real van scan.

## Verified metrics (link to metrics/results.jsonl entries)

**None.** `metrics/results.jsonl` is empty; nothing has been measured on real data.

Session evidence (tests, not metrics): `main` **832 passed, 1 skipped**; ruff clean; coverage 94.6% at last measurement (common/ and risk/ above the 80% target).

## Blockers — decisions and work only you can do

1. ~~Rear doors~~ **decided 2026-09-29: split into `rear_door_left` / `rear_door_right`**
   (14 components; seg.yaml and severity.yaml amended, both 2.5).
2. **Voxel size:** the collision solid runs ~1 cm large per side at 2 cm voxels. It errs
   on the safe side. 1 cm voxels halve it at ~8× compute.
3. **Your drafts:** `seg/fusion3d.py` (`p2-t3-fusion`), `risk/sweep.py` (`p3-t4-sweep`),
   `recon/scale.py` (`p1-t6-scale`).
4. **Unchanged:**
   - Rate R-17 to R-19.
   - Push OK, after you verify (`docs/operations_log.md` lists everything done autonomously).
   - Approve ADR 0009 (zones) and merge `docs-streaming-merge`; decide the NC media policy and LGPL ffmpeg.
   - **ADR 0010 (proposed):** grade near misses by time of closest approach. The end-to-end fixture drive raises 'critical' 2.2 s early on a 0.30 m bumper near miss. Needs a schema field.
   - **Decided 2026-10-01:**
     - ADR 0012 accepted; the capture checklist now has the rear track (scan day) and the
       per-pass hub-mark poses (lot day).
     - Missing COLMAP depth means "cannot tell": no vote. `fuse_inputs` writes NaN, and the
       `fusion3d` contract on `p2-t3-fusion` says so, with a target test.
     - `body_select` approved.
     - Wheelbase is checked from the segmented wheels (`check_wheelbase.py`, ADR 0013), or
       the gate is narrowed if that proves less accurate. Decide on the first real scan.
   - Review `p6-stream-scaffold` (streaming replay, queues, metrics, pipeline): unmerged.
   - Measure the camera mount on lot day (`configs/capture.yaml mount`); `perceive.py` refuses until then.
   - Sign the Phase 0 report when P0-T5..T7 are done (`docs/phase_reports/phase-0.md`).
   - The Phase 0 captures: P0-T7 calibration, the pre-flight check, P0-T6 (a family car
     is the recommended warm-up).
   - The October captures, including P1-T8.

## Open risks triggered (IDs from RISKS.md)

None triggered.

## Next 3 tasks

1. **Me:** more plumbing and tests that need no data or decisions; keep branches current.
2. **You:** verify the autonomous work; the three drafts; the Phase 0 captures.
3. **October:** van scan, lot day, second vehicle.

## GPU usage this month (approx compute units)

**0 CU.**
