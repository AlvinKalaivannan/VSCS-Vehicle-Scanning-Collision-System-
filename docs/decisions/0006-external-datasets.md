# ADR 0006 - External datasets: licensed data first, gaps filled by our footage or a licence request

- **Date:** 2026-09-29
- **Status:** accepted
- **Task:** cross-cutting (P1 rehearsal, P2-T1/T2, P4-T11)
- **Deciders:** developer, Claude

## Context

The developer asked whether online footage (for example a manufacturer's 360-degree
video) could replace or supplement our own capture, and then decided: **use existing
licensed datasets for now; for whatever they do not cover, either (a) provide our own
footage or (b) contact dataset owners for a licence.**

Two constraints shape how:

- Online video from manufacturers or video sites is copyrighted, uncalibrated and often
  a CGI render or a turntable spin. CLAUDE.md §9 already limits third-party footage to
  private testing, and never republished.
- Licences in this field are often inconsistent. Two of the best candidates contradict
  themselves: a permissive claim in one place, and "All Rights Reserved" or
  "non-commercial, no redistribution" in another.

## Options considered

1. **Record licences in prose only**: easy, but nothing stops a later session from putting
   a private-only dataset's images into the README or demo.
2. **A registry with uses derived from licence class**, enforced by `require_use()` and by
   tests: slightly more machinery, and publication rights cannot be granted by accident.

## Decision

Option 2.

- `configs/datasets.yaml` is the registry. `src/vscs/common/datasets.py` holds the guard.
- Uses: `rehearsal`, `private_eval`, `publish_results`, `train`, `publish_media`.
- A licence is **classified only if it is stated consistently** (`license_status: clear`).
- A conflicting licence gets private use only. An unverified, unrecognised or missing
  licence gets no use at all. That is §0's "stop and ask" for unclear licences, in code.
- External data lives in `data/external/<id>/` (gitignored). It never enters the lot
  dev/test splits (R-09, enforced by a test) or the P4-T9 gold sets.
- `train` needs a permissive licence, because the student's weights may be published.

## Gap analysis (what the registry covers, 2026-09-29)

| Need | Clean licensed source? | Plan |
|---|---|---|
| Part vocabulary and prompt tuning (P2-T1/T2) | Yes: two Roboflow car-parts sets (CC BY 4.0, to confirm at download) | (licensed) Download via the developer's Roboflow account |
| Extra cross-vehicle test for the student (P4-T11) | Partly: the same Roboflow sets (single images, cars not vans) | (licensed) as above; (a) P1-T8 plus any extra family cars on capture day |
| Full-size vehicle 360-degree scan to rehearse SfM, scale and splats | **No.** 3DRealCar and Tanks and Temples are conflicting; uCO3D is clean but has no confirmed full-size vehicles | **(a)** film a family car now as the P0-T6 warm-up; **(b)** ask the 3DRealCar authors to confirm Apache-2.0 for the data |
| Metric-scale reference to check the scale code | **No** clean source | (a) own markers and tape (P1-T6); (b) 3DRealCar if licensed |
| Lot runs: reversing footage with obstacles at known positions (P1-T3) | **No**, and none could be: it must be our van | (a) own footage only |
| Depth / ego-motion evaluation (P4-T1/T2) | Driving datasets are mostly non-commercial and are not our van or camera | (a) own lot runs |

## Consequences

- Nothing in this ADR changes the October captures. They remain the critical path.
- Adding a dataset means adding a registry entry with its licence, a check date and the
  gap it covers. The tests fail on an incomplete entry.
- A licence resolution (for example, the 3DRealCar authors confirming Apache-2.0 for the
  data) is recorded by changing `license_status` to `clear` and citing the evidence here.

## Evidence

Licences checked on 2026-09-29 against:
- uCO3D (repo README "CC BY 4.0").
- 3DRealCar (both repos' LICENSE = Apache-2.0; the download page says All Rights Reserved).
- Tanks and Temples (licence page has both CC BY 4.0 and a non-commercial research licence).
- CO3D and MVImgNet (CC BY-NC 4.0).
- DSMLR (no licence).

uCO3D's `personal_vehicles` category: 289 GB of RGB video in 18 archives. The category
names are only in the 13 GB `metadata.sqlite`, so the category could not be inspected
without a large download. Tests: `tests/unit/test_datasets.py`.
