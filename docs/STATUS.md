# VSCS Status — updated 2026-11-16
phase: V0-A (offline end-to-end with a measured error budget). Demo-first on existing datasets (ADR 0020, ADR 0021; both proposed)
core_awaiting_teach_back: none
current_task: V0-T1 (housekeeping) and V0-T2 (PandaSet licence and access: developer)
health: BLOCKED

## Done since last update

- **2026-11-16: `CLAUDE.md` rewritten as a demo-first build spec** (ADR 0020, proposed, on
  branch `spec-demo-first`). The own-vehicle scan and lot-drive plan is parked; the MVP
  runs on PandaSet and ships as a hosted web demo. Task IDs are now `V0-`, `V1-`, `V2-`.
- **Amended the same day at your direction** (ADR 0020 amendment, ADR 0021):
  - error budgeting is the central method: per-component budget by source, lifting
    error measured as a curve against PandaSet cuboids;
  - component segmentation (Grounding DINO + SAM 2 + fusion) is back as V1-T7;
  - Argoverse 2 reports the budget shift per source; CARLA is a V2 option;
  - tracker and budget become a domain-neutral `vscs/core/`;
  - I write all code; core modules merge after your `/teach-back`. No timebox.
  - **v0 ship target moved 2027-01-10 → 2027-01-24** (two more tasks, 13 scenes not 3).
- Carried over, all built and tested on the fixture only: tracking, ground-contact
  perception, risk aggregation and alerts, single-box baseline, Rerun view, evaluation
  and benchmark CLIs, the RT-DETR adapter (ADR 0011).
- Since 2026-10-12, on branches: dataset registry leads and ADR 0018 (merged);
  ADR 0019 and synthetic part renders (`p4-multivehicle-data`, unmerged).

## Verified metrics (link to metrics/results.jsonl entries)

**None.** `metrics/results.jsonl` is empty; nothing has been measured on real data.

## Blockers — decisions and work only you can do

1. **Accept ADR 0020 and ADR 0021** by merging `spec-demo-first`.
2. **PandaSet licence and download** (CLAUDE.md §11, rows 1–2). Mirrors say CC BY 4.0;
   Scale's legal page names separate terms. Nothing can be fetched or shown until this
   is read at source. If it fails, the fallback is comma2k19 (front camera only).
3. **Teach-backs.** `risk/sweep.py` (V0-T5) is the first; I write it, you explain it
   back before it merges. At most two core branches may wait at once.
4. Review and merge `p6-stream-scaffold` (V0-T1).
5. Chrysler Pacifica dimensions from a spec sheet (V0-T4).
6. Confirm the Overstride figures quoted in CLAUDE.md §1.2 (§11 row 9).

## Open risks triggered (IDs from RISKS.md)

None triggered. `docs/RISKS.md` still holds the old `R-` register; it is re-seeded from
CLAUDE.md §8 in V0-T1.

## Next 3 tasks

1. **You:** V0-T2 licence check; ADR 0020 decision.
2. **Me:** V0-T1 housekeeping, V0-T4 (template model) and V0-T5 (`sweep.py`); none needs data.
3. **You:** teach-back on `sweep.py`.

## GPU usage this month (approx compute units)

**0 CU.**
