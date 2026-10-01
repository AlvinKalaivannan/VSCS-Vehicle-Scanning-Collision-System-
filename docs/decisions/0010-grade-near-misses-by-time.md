# ADR 0010 - Grade near misses by *when* as well as *how close*

- **Date:** 2026-10-01
- **Status:** **accepted** by the developer, 2026-10-01 (option 2, implemented the same day)
- **Task:** P3-T5 (alerts), P5-T2 (false alarms per minute)
- **Deciders:** developer (pending), Claude

## Context

ADR 0005 grades a component two ways:
- **Predicted contact:** by time to contact (TTC ladder: 3 / 2 / 1 s).
- **No contact predicted:** by closest approach (distance ladder: 1.0 / 0.5 / 0.25 m).

The sweep's `min_distance_m` is the minimum over the *whole* horizon (3 s).

**Found on the end-to-end fixture drive (2026-10-01):**
- The van reverses at 1 m/s toward a pole 2.5 m behind its rear-right corner.
- The rear bumper passes the pole 0.30 m away, at t ≈ 2.2 s. After sweep margins it reads
  0.215 m, so it is graded **critical at t = 0.2 s**, 2.2 s before the near miss happens.
- It happens with mask-accurate perception too, so perception is not the cause.

**Why it matters:**
- A critical alert should mean "act now". Raising it 2 s early for a near miss inflates
  false alarms per minute (P5-T2) and erodes trust (R-10).
- The ADR 0005 reasoning (a contact's 0 distance must not grade critical regardless of
  when) applies to near misses as well, and was only solved for contacts.

## Options considered

1. **Keep as is:** simple, but near misses grade critical too early, as shown.
2. **Add `ComponentRisk.t_closest_s`** (time of closest approach, `None` if no obstacle in
   range). A near miss meets a level only if both its distance is below that level's
   threshold *and* `t_closest_s` is below the level's TTC threshold. Same ladders, no new
   tuning numbers. Requires a schema field, a version bump, and the developer's sweep to
   report it.
3. **Grade near misses only by distance at the current pose:** loses look-ahead entirely.

## Proposal

Option 2. With it, the fixture drive's bumper near miss (0.215 m at 2.2 s) would grade
"warning"-by-distance only once `t_closest_s` falls below 2 s, and "critical" below 1 s.
That is the same timing logic contacts already get.

## Decision

Option 2, approved by the developer on 2026-10-01.

## Migration note (schema 1 -> 2)

- `ComponentRisk.t_closest_s: float | None` (seconds from now, ≥ 0): the time of
  `min_distance_m`. It equals `ttc_s` for a predicted contact, and is the earliest time of
  the minimum otherwise. `SCHEMA_VERSION` = 2.
- **Version-1 streams still load.** The field defaults to `None`, and a near miss with
  `t_closest_s = None` is graded by distance alone, exactly as in version 1. So old
  `risk_frames.jsonl` files replay unchanged, and a sweep that does not yet report the
  field degrades to the old behaviour rather than failing.
- `alerts._meets`, near-miss branch: `close and soon`, where soon = `t_closest_s` below
  the level's TTC threshold. The release offsets widen both thresholds.
- `aggregate.component_risks` copies `t_closest_s` from the sweep result if present.
- `risk/sweep.py` (your core module): `SweepResult.t_closest_s`, with a red target test
  on `p3-t4-sweep`. The test oracle (`tests/fixtures/oracle_sweep.py`) reports it.

## Consequences

- Schema `SCHEMA_VERSION` bump; migration note; `sweep.py` (the developer's core module)
  reports the time of minimum clearance alongside the clearance.
- `alerts._meets` gains a time check for the near-miss branch.
- Hysteresis needs no change.

## Evidence

- **Before:** `tests/unit/test_drive_end_to_end.py`: critical from t = 0.2 s with boxes and
  with masks; the rear bumper reads 0.215 m with `ttc_s = None`.
- **After** (`test_a_near_miss_far_ahead_no_longer_grades_critical_adr_0010`):
  - t = 0.2 s is `caution` (the bumper's closest approach is 2.2 s away);
  - `warning` from 0.6 s;
  - `critical` from 1.6 s, when the real corner contact is 0.75 s away (the raw level
    crosses at 1.4 s, plus the 2-frame escalation);
  - attribution still names the rear-right corner, with the lead time above 1.5 s.
- `tests/unit/test_alerts.py`: the near-miss ladder by `t_closest_s` (None / 2.5 / 1.5 /
  0.5 / 3.5 s) and release through the time deadband.
