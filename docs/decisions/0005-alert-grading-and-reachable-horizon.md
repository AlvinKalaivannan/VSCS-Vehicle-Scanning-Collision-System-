# ADR 0005 - Grade contacts by TTC, and cut the sweep at the first predicted impact

- **Date:** 2026-09-24
- **Status:** accepted
- **Task:** P3-T5 (with consequences for P3-T4 and P3-T6)
- **Deciders:** developer, Claude

## Context

Wiring the Phase 3 pieces together for the end-to-end smoke test exposed two defects in
work that had already been merged and had passing tests. Both came from treating the
sweep's output as something it is not.

**1. Every predicted contact graded "critical".** The sweep contract defines
`min_distance_m` as the minimum inflated clearance over the *whole horizon*, which is 0
whenever contact is predicted. The alert rule compared that value against the distance
thresholds (critical below 0.25 m) *or* the TTC thresholds. So a contact predicted 2.9 s
ahead - which the TTC ladder grades as caution - was reported as critical on distance
alone, and the TTC thresholds never had any effect on a moving approach. Measured:
`raw_level(min_distance_m=0.0, ttc_s=2.9) == "critical"`.

The P3-T5 tests had passed because they fed the alert logic the *current* gap as
`min_distance_m`, which is not what the sweep produces. The unit tests were internally
consistent and wrong about the interface.

**2. The wrong component named.** Each component is swept independently. On the fixture
the sliding door "reaches" the pole at 2.37 s, but the rear corner strikes that pole at
0.37 s, and the van cannot pass through it. The phantom door contact both inflated the
risk and, because the door's severity (2.5) exceeds the corner's (1.5), made the frame name
the **door** as the worst component. Component attribution is the metric P3-T6 and P5-T2
are judged on.

## Options considered

For the alert grading:

1. **Grade a predicted contact by TTC and a near miss by closest approach.** No schema
   change. Distance thresholds come to mean "how close will it pass", TTC "how soon will it
   hit".
2. **Add a `current_distance_m` field to `ComponentRisk`** and grade distance against that.
   More information, but a §4.2 schema change: version bump, migration note, updated
   tests, and the developer's approval under §0.
3. **Redefine `min_distance_m` as the clearance at t = 0.** Loses the closest-approach
   figure, which is exactly what matters for a pass-by (a pole 0.17 m beside the path).

For attribution:

1. **Cut the sweep at the first predicted contact** (sweep, find the earliest contact `t*`,
   re-sweep with the horizon ending at `t*`). Physically grounded: nothing later is
   reachable without the first impact happening.
2. **Weight risk by urgency** (for example, divide by TTC). Heuristic, introduces a new
   tunable, and still counts impossible contacts - just with less weight.
3. **Per-obstacle shadowing** (only the first component to reach a given obstacle keeps
   the contact). Handles the pole case, but not a component whose closest approach to a
   *different* obstacle occurs after the first impact.

## Decision

Grading **option 1**. Attribution **option 1**, in a new orchestration module
`risk/engine.py`.

- `alerts._meets`: a predicted contact (`ttc_s` set) is graded by TTC alone; otherwise by
  `min_distance_m`. Contact probability, when `alerts.use_p_contact` is enabled at P4-T7,
  still meets a level on its own.
- A TTC deadband, `alerts.hysteresis.deescalate_ttc_hysteresis_s: 0.30`, mirrors the
  distance deadband so a TTC hovering on a threshold cannot flicker.
- `PathFan.truncated(t)` cuts a fan exactly at `t` (not snapped to a step), with the final
  pose evaluated from the continuous model.
- `engine.reachable_clearances` sweeps once, and if anything is struck, sweeps again with
  the horizon cut at `t* + ttc_tolerance_s`. The pad ensures a contact reported at the top
  of its tolerance bracket is still captured on the second pass.
- The sweep function is **injected** into the engine, so the engine is tested now against
  a brute-force reference (`tests/fixtures/oracle_sweep.py`) without waiting for the
  developer's `sweep.py`.

## Consequences

- No change to the §4.2 schema and no change to the developer's `sweep.py` contract.
- A frame with a predicted contact costs two sweeps instead of one. Acceptable in v0.1;
  worth measuring at P5-T3.
- Truncating at the first impact assumes the vehicle follows the predicted path until then.
  That is the constant-curvature assumption already made; steering changes are the fan's job
  and, from P4-T7, the probability model's.
- After P4-T7 the question returns in a softer form: with a real probability, a contact
  that is *likely but not certain* at `t*` does not make later contacts impossible, only
  less likely. The hard cut will need revisiting then.
- Lesson recorded: unit tests of a consumer should be fed what the producer actually emits,
  not what the consumer's author assumes it emits. The integration test found this, not the
  unit tests.

## Evidence

- `tests/unit/test_alerts.py::test_distant_predicted_contact_is_caution_not_critical` -
  regression test for defect 1.
- `tests/unit/test_engine.py::test_without_truncation_the_wrong_component_is_named` - shows
  the full-horizon sweep names the sliding door (2.37 s).
- `tests/unit/test_engine.py::test_frame_names_the_corner_for_the_pole` - with truncation,
  the rear-right corner is named, TTC 0.37 s.
- `tests/unit/test_alerts.py::test_no_flicker_when_ttc_hovers_on_a_threshold` - TTC
  deadband.
