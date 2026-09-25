# VSCS Status — updated 2026-09-25
mode: pair            # pair | build
phase: P1
current_task: waiting on the developer's two core-module drafts (P3-T4 sweep, P1-T6 scale) and the capture-prep work
health: ON TRACK

## Done since last update

Everything buildable without new permissions or real data, per the session goal. All on
`main` unless noted; nothing pushed.

- **P1-T5** [done, tooling] `recon/colmap_io.py` + `recon/sfm.py` + `scripts/recon.py`. COLMAP
  text-model I/O (pose direction and scalar-first quaternions verified by reprojecting every
  observation to 1e-6 px), the COLMAP command pipeline reusing the P0-T7 intrinsics, and the
  P1-T5 gates. A split reconstruction is counted against *all* frames, so it cannot hide:
  25 frames split 20 + 5 scores 80% and fails. Acceptance still needs COLMAP and a real scan.
- **P1-T4** [done, tooling] `scripts/extract_frames.py` — the missing link between ingest and
  SfM. `ingest.extract_stride: 10` keeps a 4K30 scan to a few hundred frames.
- **P1-T6 plumbing** [done] `recon/ground.py` (seeded RANSAC with a tilt limit so a van side
  panel cannot out-vote the ground) and `recon/vehicle_frame.py`. On a synthetic van: frame
  rotation 0.07 deg, origin 5 mm, length/width/height within 1 cm. Width excludes mirrors.
- **P1-T6 plumbing** [done] `recon/marker_tracks.py` + marker detection: undistorted
  per-corner tracks, detection fraction for the R-02 trigger, and two quality gates. See the
  marker finding under Blockers.
- **P1-T6 core** [pending, developer's draft] `recon/scale.py` contract and known-marker tests
  on branch `p1-t6-scale`: target scale = 1 / 0.37 to 1e-6, validated achievable first.
- **P3-T4 core** [pending, developer's draft] unchanged, on `p3-t4-sweep`.
- **Earlier (2026-09-24), still current:** P3-T3 motion model and P3-T5 alerts/aggregation done
  on the fixture; ADR 0005 (TTC grading, first-impact horizon); P1-T1 checklist and P2-T1
  vocabulary approved.
- **Docs**: CLAUDE.md §12 quick reference synced with the scripts that exist; ADR 0003
  addendum on COLMAP dense needing CUDA.

## Verified metrics (link to metrics/results.jsonl entries)

**None.** `metrics/results.jsonl` is empty; nothing has been measured on real data.

Session evidence (tests, not metrics): `main` **556 passed, 1 skipped**; ruff clean. Both draft
branches are green apart from their intended red targets.

## Blockers — decisions and work only you can do

1. **Your drafts:** `risk/sweep.py` on `p3-t4-sweep` and `recon/scale.py` on `p1-t6-scale`.
   Each has its maths in the module docstring and a validated test target.
2. ~~Marker placement~~ **decided 2026-09-25: propped 45° boards** (you picked the most
   accurate option). Checklist updated - please re-read its "Marker placement" section, which
   changed after your approval. Four sheets now: IDs 0-2 around the van, ID 3 at the front
   bumper (`recon.yaml vehicle_frame.front_marker_id`) so the software knows the front.
3. ~~P1-T7 dense route~~ **decided 2026-09-25: COLMAP dense on Colab** (ADR 0003).
4. **Scan day:** measure the rear overhang (in the checklist; `recon.yaml
   vehicle_frame.rear_overhang_m` refuses to run until set).
5. **Unchanged:** ffmpeg + COLMAP install; the pre-flight test (`scripts/check_capture.py`);
   P0-T5/T6/T7 captures.
6. **Push:** `main` is 20+ commits ahead of `origin`. Say the word and I'll push after the same
   security check as before.

## Open risks triggered (IDs from RISKS.md)

None triggered.
- **R-02** — the marker finding above bears directly on it; recorded as a proposal on R-02.
- **R-12** — COLMAP dense needs CUDA (ADR 0003 addendum): the planned dense primary cannot run
  on this machine.

## Next 3 tasks

1. **You:** the two drafts, and the marker-placement and dense-route decisions.
2. **You:** ffmpeg + COLMAP, the pre-flight test, then the P0 captures. October is the whole
   capture window.
3. **Me, once decided:** update the checklist for marker boards if you choose them; wire P1-T7
   for whichever dense route you pick; review and harden your drafts.

## GPU usage this month (approx compute units)

**0 CU.**
