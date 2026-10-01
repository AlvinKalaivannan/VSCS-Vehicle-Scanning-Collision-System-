# ADR 0013 - Validate P1-T6's wheelbase from the segmented wheels, or narrow the gate

- **Date:** 2026-10-01
- **Status:** accepted (the developer, 2026-10-01: "segmented wheels or narrow the gate,
  whichever gives more accurate results")
- **Task:** P1-T6 (metric scale; gate: length, width, height, wheelbase each within ±2 cm)
- **Deciders:** developer, Claude

## Context

`scripts/scale.py` builds the vehicle frame from the body points. That gives length, width
and height, but no wheel centres, so P1-T6's fourth dimension, the wheelbase, cannot be
checked at that step. It reports it as `not_validated` (devlog 2026-10-01).

## Options considered

1. **Wheel hubs from the segmented wheels (after P2-T5).** In `veh`, each wheel's hub x is
   the middle of its trimmed x-extent; wheelbase = mean front hub x - mean rear hub x.
   It costs nothing extra on the day. Its accuracy depends on segmentation quality at the
   wheels.
2. **Narrow the P1-T6 gate** to length, width and height, and report the wheelbase as a
   finding. Simple, but the gate checks less.
3. **Hand-placed hub points in CloudCompare.** Accurate, but manual, and it is the
   developer's time.

## Decision

Build option 1 now and decide between 1 and 2 **on the first real scan**, by whichever is
more accurate, as the developer asked:

- `scripts/check_wheelbase.py --labelled <cleaned.npz> --measured <measurements.yaml>`
  appends `wheelbase_error_m` (task P1-T6, gate 0.02 m in `eval.yaml`). It also prints two
  diagnostics:
  - the **left/right hub gap** per axle, which is large when segmentation is at fault;
  - the **rear axle x**, which should be 0 in `veh`; a large offset means the frame or the
    rear-overhang tape is at fault.
- **Reading the result:**
  - Error ≤ 2 cm: keep the full four-dimension gate (option 1 stands).
  - Error > 2 cm with a large left/right gap but a rear axle x near 0: the wheel
    segmentation is less accurate than the gate, so narrow the gate (option 2). That is a
    §6 change: the developer edits CLAUDE.md and amends this ADR, and the wheelbase error
    stays reported as a finding.
  - Error > 2 cm with a rear axle x well off 0: the frame itself is wrong. That is a
    genuine P1-T6 failure, not a reason to narrow the gate.

## Consequences

- Full P1-T6 sign-off waits for P2-T5 (cleaned labels), unless the gate is narrowed.
- `wheelbase_error_m` is a separate metric, so `scale_error_m` stays comparable across runs.

## Evidence

`tests/unit/test_wheelbase.py`:
- the fixture wheelbase (3.0 m) is recovered to 1 cm, with the rear axle at x = 0;
- trimming absorbs 1.2% stray labels, while the untrimmed estimate moves by more than 5 cm;
- a wheel segmented 10 cm off shows up as a 10 cm left/right gap;
- the CLI passes and fails the gate, and logs the metric.
