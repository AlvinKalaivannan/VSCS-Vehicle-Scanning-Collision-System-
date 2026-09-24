# VSCS Status — updated 2026-09-24
mode: pair            # pair | build
phase: P0
current_task: P0-T6 / P0-T7 (blocked on capture — see Blockers)
health: 🟢 on track

## Done since last update

First working session. The repository was empty apart from `CLAUDE.md`, with no commits.

- **P0-T1** ✅ Repo scaffold per section 3, `pyproject.toml` (ruff + pytest + coverage),
  `.venv` on Python 3.11.9, pinned `envs/requirements-core.txt`, `envs/core.yml`,
  `envs/colab_requirements.txt`, `.gitignore`, README with the "not a safety device"
  statement, `data/MANIFEST.md`.
- **P0-T2** ✅ Reporting system: this file, `RISKS.md` seeded with all 16 risks from
  section 8.1, devlog + ADR + phase-report templates, empty `metrics/results.jsonl`,
  `eval/report.py` + `scripts/report.py` generating `docs/REPORT.md`.
- **P0-T3** ✅ `common/frames.py` (transforms, rotations, pinhole projection, the
  veh↔cam convention) and `common/types.py` (all section 4.2 schemas), plus
  `config.py`, `io.py`, `log.py`.
- **P0-T4** ✅ Synthetic fixture world: 12-component box vehicle, thin pole, kerb, box,
  20-pose camera trajectory with jittered timestamps.
- **P0-T5** ⚠️ Colab notebook template written and structurally validated locally. The
  acceptance criterion ("runs a hello-world GPU check and writes to Drive") **needs one
  Colab run from you** — I cannot execute it.
- **P0-T7** ⚠️ Tooling only: `capture/calib.py` + `scripts/calibrate.py`, gated to
  refuse writing intrinsics above 0.5 px. The acceptance criterion needs your
  checkerboard capture.

## Verified metrics (link to metrics/results.jsonl entries)

**None.** `metrics/results.jsonl` is empty, and `docs/REPORT.md` says so. Nothing has
been measured because nothing measurable has been captured yet — every gate in
section 6 is currently listed as unproven, which is the correct state on day one.

Test-suite evidence for this session (not metrics, and not written to `results.jsonl`):

- `pytest` — **238 passed, 1 skipped** in 4.1 s
- coverage on `common/` + `risk/` — **95%** (requirement: ≥80%, section 5)
- `ruff check .` — clean; `ruff format --check .` — 43 files already formatted
- `python scripts/report.py` — exits 0 and regenerates `docs/REPORT.md`
- known-distance fixture test — rear right bumper corner to pole axis is **exactly
  0.500 m**, asserted to 1e-6

## Blockers

Both remaining P0 tasks need something only you can do. Neither blocks other work.

1. **P0-T6** (warm-up: phone video → COLMAP → splat) needs (a) a short video of a small
   object, and (b) COLMAP installed — it is a native binary, not a wheel. The install
   route is decided in ADR 0003 (prebuilt Windows no-CUDA build) but not yet done.
2. **P0-T7** needs a checkerboard shot with **focus and exposure locked, main lens, no
   zoom** (R-04), 15+ images, board tilted and pushed into all four image corners. Then
   `python scripts/calibrate.py --images <folder> --device <name>`.
3. **P0-T5** needs one run of `notebooks/colab/00_template.ipynb` on a T4.

## Open risks triggered (IDs from RISKS.md)

None triggered. Two need your attention rather than mine:

- **All 16 rows carry *proposed* likelihood/impact ratings, not agreed ones.** Section 8
  says these are rated with you. Please walk the table with me next session.
- **R-12** (dependency conflicts) showed up immediately: the first dependency install
  stalled over 12 minutes on pip backtracking through `open3d`. Resolved by installing
  in tiers and deferring `open3d` and `rerun-sdk` until P1-T6/T7. ADR 0001.

## Next 3 tasks

1. **Review this session's output** — specifically `JointSpec` (I invented it; section
   4.2 references it but never defines it, ADR 0002) and the proposed risk ratings.
2. **P0-T6 + P0-T7 captures** — the warm-up video and the checkerboard, plus the COLMAP
   install. These unblock the last of Phase 0.
3. **P1-T1** — write `docs/capture_checklists.md` for scan day and lot day. Worth doing
   before October: section 6 says all real-world data is captured in October, before
   snow, and the checklist is what makes a capture day not get wasted.

Not started on purpose: the component vocabulary (P2-T1) and severity weights need your
approval, so `configs/seg.yaml` and `configs/severity.yaml` hold clearly flagged
proposals only.

## GPU usage this month (approx compute units)

**0 CU.** No Colab session run yet.
