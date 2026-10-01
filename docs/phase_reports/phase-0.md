# Phase 0 report — DRAFT (gate not yet reached)

- **Window:** 2026-09-24 → open (P0-T5, P0-T6 and P0-T7 outstanding)
- **Health at gate:** not at the gate. Four tasks met, two partial, one unmet; every open
  item is waiting on the developer, not on code.
- **Drafted:** 2026-10-01, autonomously (see `docs/operations_log.md`). The developer
  verifies it and signs it at the gate.

## Task results

| ID | Task | Acceptance criterion | Result | Evidence |
|---|---|---|---|---|
| P0-T1 | Repo scaffold, pyproject, ruff, pytest, env | `pytest` runs, `ruff check` clean | **MET** | 2026-10-01: 759 passed, 1 skipped; `ruff check .` clean. venv instead of conda: ADR 0001 |
| P0-T2 | Reporting skeleton | `scripts/report.py` generates `docs/REPORT.md` | **MET** | Re-run 2026-10-01: "wrote docs/REPORT.md (0 measurements)" |
| P0-T3 | `common/frames.py` + `types.py` with tests | Transform and projection tests pass | **MET** | `tests/unit/test_frames.py`, `tests/unit/test_types.py` |
| P0-T4 | Synthetic fixture world | Fixture loads; known-distance test passes | **MET** | `tests/unit/test_fixture_scene.py` (rear-right corner to pole = 0.500 m ± 1e-6) |
| P0-T5 | Colab notebook template | Hello-world GPU check that writes to Drive | **PARTIAL** | `notebooks/colab/00_template.ipynb` exists and passes `tests/unit/test_notebooks.py`. Never run on Colab: that needs the push, then a developer session |
| P0-T6 | Warm-up: object → COLMAP → splat | A viewable 3D model, steps in the devlog | **UNMET** | COLMAP 4.2.0 now installed (ADR 0003 addendum); `scripts/recon.py` ready. Needs the developer's warm-up video (a family car recommended, ADR 0006 gap table) |
| P0-T7 | Phone intrinsics calibration | Reprojection error < 0.5 px, saved to `configs/capture.yaml` | **PARTIAL** | Tooling done (`capture/calib.py`, `scripts/calibrate.py`, which refuses to write above 0.5 px). Needs the developer's checkerboard capture |

## What changed from the plan

- **venv, not conda** (ADR 0001); **COLMAP from the GitHub release** (ADR 0003 addendum).
- **Scope grew, by the developer's choice:**
  - the trained segmenter;
  - external datasets (ADR 0006);
  - the streaming direction (ADR 0008, unmerged).
- **Much later-phase code was built early on the fixture**, with the developer's go-ahead:
  Phase 2 plumbing, plus the Phase 3–5 perception, risk, eval and ui modules. None of it
  is *accepted*; its §6 gates need real data.

## Fallbacks used

None. No §8.2 ladder row has been stepped down.

## Risks re-rated

Not yet: re-rating happens at the gate (§8). Movements since seeding, for the developer
to confirm:

- **R-12 eased:** the toolchain is installed, and CPU verification caught five adapter
  bugs before any Colab spend (ADR 0007).
- **R-07, R-10:** a thin-pole placement limitation with boxes was found and fixed with
  masks; an early-critical alert from near-miss grading was found (ADR 0010, proposed).
- **R-20 evidence:** the detector takes ~750 ms per image on the laptop CPU (ADR 0011).
- **Unrated:** R-17–R-19 (seeded, awaiting the developer). R-20–R-24 exist only on the
  unmerged streaming branch.

## Lessons

- **Check adapters against the real library before spending GPU time.** Fakes cannot
  catch API drift, missing extras, or hardware dtype limits (ADR 0007).
- **When an end-to-end result looks wrong, find the cause before writing it down.** The
  first explanation for the early "critical" was wrong and had to be corrected (devlog,
  2026-10-01).

## Readiness for the next phase

Phase 1 is capture-driven, and October is the only window. Ready:
- capture checklists (approved);
- the pre-flight check;
- ingest and frame extraction;
- SfM and the dense path (dense on Colab);
- the marker boards.

Blocking the gate:
- the P0-T5, P0-T6 and P0-T7 captures and runs (developer);
- the push, so Colab can run anything.

---

approved:
