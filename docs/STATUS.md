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
  - All on the fixture, and all merged locally.
- **ADR 0008 / proposed ADR 0009** (streaming merge) are on branch `docs-streaming-merge`,
  **not merged**, awaiting your OK.
- None of these is *accepted*: every §6 gate needs the real van scan.

## Verified metrics (link to metrics/results.jsonl entries)

**None.** `metrics/results.jsonl` is empty; nothing has been measured on real data.

Session evidence (tests, not metrics): `main` **708 passed, 1 skipped**; ruff clean.

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
   - The Phase 0 captures: P0-T7 calibration, the pre-flight check, P0-T6 (a family car
     is the recommended warm-up).
   - The October captures, including P1-T8.

## Open risks triggered (IDs from RISKS.md)

None triggered.

## Next 3 tasks

1. **Me:** streaming scaffolding (replay, queues, metrics, pipeline): unmerged until you review.
2. **You:** verify the autonomous work; the three drafts; the Phase 0 captures.
3. **October:** van scan, lot day, second vehicle.

## GPU usage this month (approx compute units)

**0 CU.**
