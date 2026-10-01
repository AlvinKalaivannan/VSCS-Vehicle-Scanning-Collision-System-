# ADR 0012 - Lot-day ground truth must include where the van is

- **Date:** 2026-10-01
- **Status:** **accepted** by the developer, 2026-10-01. Checklist updated the same day.
- **Task:** P1-T3 (lot day), P3-T2 (cone positions), P3-T6 / P5-T2 (component attribution)
- **Deciders:** developer (pending), Claude

## Context

The lot-day checklist (`docs/capture_checklists.md` §3) records every obstacle's position
from a chalk origin, "tape-measured in two axes". Perception reports obstacles in the
**van's** frame `veh`. Nothing in the checklist records where the van stood relative to
that origin, so the two cannot be compared:

- **P3-T2 ("cone positions within ±25 cm at ≤3 m"):** the error is `perceived − true`, and
  "true" in `veh` needs the van's pose in the lot frame.
- **P3-T6 / P5-T2 (`passes.yaml truth`):** which component passes closest and *when* needs
  the van's position relative to the obstacle at the closest approach.

Found while building the scorer for `cone_position_error_m`, the one P3 gate with no
producer (`tests/unit/test_metric_producers.py`). Lot day is in October, before snow, and
cannot be cheaply repeated, so this is time-critical.

## Options considered

1. **Tape-measure the van's pose at the start and end of every pass (proposed).**
   - With the van stopped, drop a plumb line from the centre of each **rear wheel hub** to
     the ground, chalk the two points, and tape-measure each from the origin in two axes.
   - The midpoint of the two marks *is* the `veh` origin (§4.1: on the ground below the
     rear-axle centre). The line from the right mark to the left mark is the `veh` y axis.
     No model and no bumper-corner guesswork are involved.
   - The distance between the marks should equal the rear track, measured once on scan day
     (to add to the "Measure the van first" list). That gives a built-in check on every
     pose.
   - Cost: about 2 minutes per pass, so about 25-30 minutes for 12-15 passes.
   - Accuracy: ±1 cm per mark over a ~1.7 m baseline gives about 0.6° of heading, or about
     3 cm at 3 m. That is well inside P3-T2's 25 cm gate.
2. **A marker board at the origin, with the van pose read by the camera.** No tape work,
   but the truth comes from the system under test (its own camera and calibration), so
   its errors would cancel out of the score. Rejected.
3. **Score relative geometry only.** Rigidly align the detected cones to the measured
   layout (Procrustes) and score the residuals. No van pose is needed, but the alignment
   absorbs any error common to all cones: a range bias that pushes every cone the same
   way would vanish from the score. That flatters VSCS. Rejected.
4. **Do nothing on lot day; reconstruct the van pose later from ego-motion.** Circular for
   the same reason as option 2, and there is nothing to anchor the start. Rejected.

## Decision

Option 1, approved by the developer on 2026-10-01.

**Checklist additions** (applied to `docs/capture_checklists.md`):

- Scan day, "Measure the van first": add **rear track** (centre of the left rear hub to
  centre of the right rear hub), measured twice.
- Lot day, every pass: before moving and after stopping, plumb from both rear hub centres,
  chalk, and tape-measure both marks from the origin (x and y). Note the pass's raw id.
  This is done only with the **van stationary and the engine off or in park**; the person
  measuring is never beside a moving vehicle (§9). **Read each tape twice**, and **left means the van's own
  left**: the track check catches a misread *along* the hub line, but not one *across* it
  (10 cm turns the heading by about 3°), and not a left/right swap (which turns the van
  around). Both limits are shown in `tests/unit/test_lot_truth.py`.
- If a pass is meant to take a component *past* an obstacle (mirror or wheel passes), also
  note which component and obstacle it was designed for. That is the truth `component`.

**`gt.yaml` format** (`data/raw/lot_<date>/gt.yaml`; loaded by `vscs.eval.lot_truth`):

```yaml
schema_version: 1
origin: "chalk X by the light pole; x toward the fence; y 90 deg left of x"
rear_track_m: 1.72            # scan-day measurement; each pose is checked against it
obstacles:
  - {id: pole_1, kind: pole, position_m: [3.10, -0.45]}
  - {id: cone_1, kind: cone, position_m: [2.40, 1.20]}
passes:
  - id: lot_20261018_03       # the raw id ingest gave the clip
    start: {rear_left_hub_m: [5.21, 0.86], rear_right_hub_m: [5.20, -0.86]}
    end:   {rear_left_hub_m: [4.02, 0.85], rear_right_hub_m: [4.01, -0.87]}
    designed_nearest: {component: rear_right_bumper_corner, obstacle: pole_1}
    shape: straight           # straight | curve (amendment 2026-10-01): P4-T2 gates straight only
    notes: ""
```

The lot frame is right-handed and in metres, with z up and its origin at the chalk X.

## Consequences

- **P3-T2 becomes scoreable without ego-motion.** The van is stationary at the start, so
  the frames after the start sync clap are compared directly with the measured layout
  transformed into `veh` by the start pose. `scripts/score_perception.py` does this and
  appends `cone_position_error_m` (worst matched cone at ≤3 m, the gate's reading of
  "within ±25 cm"). It also reports the cones it missed (R-07), so a missed pole cannot
  hide behind a good error.
- **The start and end poses give P4-T2 its truth too:** the true motion over the pass,
  against the first and last ego-motion poses. `egomotion_drift_frac` is endpoint error
  over displacement, which overstates drift on curved passes (the cautious direction).
- **The end pose gives the closest approach** on the "reverse toward X and stop short"
  passes, which are most of the plan. For pass-by passes the end pose alone does not
  give the moment of closest approach. Two options for the developer: annotate it from a
  tripod side camera (no operator near the vehicle), or score those passes on component
  attribution only.
- **Measurement time on lot day:** about 30 minutes for 12-15 passes, at walking-speed
  safety rules.

## Evidence

- `docs/capture_checklists.md` §3 ("Obstacles, at measured positions"): obstacle positions
  only.
- `tests/unit/test_metric_producers.py`: `cone_position_error_m` is listed as having no
  producer.
- `tests/unit/test_lot_truth.py` (with this ADR): the pose from two hub marks, the
  track-length check, and the P3-T2 scorer on a synthetic layout with known answers.

## Amendment 2026-10-01 - pass shape, and P4-T1 depth

- **`shape: straight | curve`** per pass. P4-T2's drift is endpoint error over
  displacement, and on a curve the displacement is shorter than the path driven. So only
  `straight` passes gate `egomotion_drift_frac`. Curved and untagged passes are reported
  beside it, never hidden.
- The same parked-start window now also gives **P4-T1 depth error by range bucket**: the
  perceived minus true range from the camera, bucketed at 1/3/5/10 m (±25%).
  `depth_error_m_at_3m` is gated at 0.25 m; the other buckets are reported.
- Evidence: `tests/unit/test_lot_truth.py` (curved pass reported but not gated, range
  error and bucket known answers, CLI depth metric).
