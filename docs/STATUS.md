# VSCS Status — updated 2026-11-16
mode: pair            # pair | build
phase: V0-A (offline end-to-end on one scene). Plan changed to demo-first on existing datasets (ADR 0020, proposed)
current_task: V0-T1 (housekeeping) and V0-T2 (PandaSet licence and access: developer)
health: BLOCKED

## Done since last update

- **2026-11-16: `CLAUDE.md` rewritten as a demo-first build spec** (ADR 0020, proposed, on
  branch `spec-demo-first`). The own-vehicle scan and lot-drive plan is parked; the MVP
  runs on PandaSet and ships as a hosted web demo. Task IDs are now `V0-`, `V1-`, `V2-`.
- Carried over, all built and tested on the fixture only: tracking, ground-contact
  perception, risk aggregation and alerts, single-box baseline, Rerun view, evaluation
  and benchmark CLIs, the RT-DETR adapter (ADR 0011).
- Since 2026-10-12, on branches: dataset registry leads and ADR 0018 (merged);
  ADR 0019 and synthetic part renders (`p4-multivehicle-data`, unmerged).

## Verified metrics (link to metrics/results.jsonl entries)

**None.** `metrics/results.jsonl` is empty; nothing has been measured on real data.

## Blockers — decisions and work only you can do

1. **Accept or amend ADR 0020** by merging `spec-demo-first`.
2. **PandaSet licence and download** (CLAUDE.md §11, rows 1–2). Mirrors say CC BY 4.0;
   Scale's legal page names separate terms. Nothing can be fetched or shown until this
   is read at source. If it fails, the fallback is comma2k19 (front camera only).
3. **`risk/sweep.py` draft** (V0-T5, branch `p3-t4-sweep`). The 7-day timebox starts when
   you start the task.
4. Review and merge `p6-stream-scaffold` (V0-T1).
5. Chrysler Pacifica dimensions from a spec sheet (V0-T4).

## Open risks triggered (IDs from RISKS.md)

None triggered. `docs/RISKS.md` still holds the old `R-` register; it is re-seeded from
CLAUDE.md §8 in V0-T1.

## Next 3 tasks

1. **You:** V0-T2 licence check; ADR 0020 decision.
2. **Me:** V0-T1 housekeeping, then V0-T4 (template component model), which needs no data.
3. **You, with me:** V0-T5 `sweep.py`.

## GPU usage this month (approx compute units)

**0 CU.**
