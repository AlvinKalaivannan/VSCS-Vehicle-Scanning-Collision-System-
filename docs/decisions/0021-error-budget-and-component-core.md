# ADR 0021 - Error budgeting as the central method; a reusable component-level core

- **Date:** 2026-11-16
- **Status:** proposed. Directed by the developer on 2026-11-16; accepted when the
  developer merges branch `spec-demo-first`. The schema change it describes is
  implemented in V0-T8, not in this commit.
- **Task:** V0-T7, V0-T8, V1-T4, V1-T9
- **Deciders:** developer, Claude

## Context

The first draft of the demo-first spec (ADR 0020) left uncertainty as a single assumed
`pos_sigma_m` and a V1 probability. The developer's earlier project, Overstride, used a
per-joint error budget and a measured viewpoint error curve, and found that its 3D
lifter degraded cross-lab. The developer asked for the same method to be the centre of
VSCS, and for the tracker and budget to be built so they could serve both projects.

Overstride figures quoted in CLAUDE.md §1.2 (3.50° → 1.77° as views went side-on; 1.84×
cross-lab degradation) were supplied by the developer. Claude has not read Overstride.

## Options considered

1. **Per-component error budget split by source, measured (chosen).**
2. A single learned or heuristic confidence per alert: says how sure, not why, and
   cannot say what to fix next.
3. Full covariance propagation through every stage (EKF-style Jacobians end to end):
   more exact, but far more code and harder to explain per source in a demo.

For the shared code:

1. **A domain-neutral package inside this repo, `vscs/core/` (chosen).**
2. A separate pip package shared by both repos now: a second repo, versioning and CI
   before there is one working user.
3. Refactor Overstride onto it: ruled out by the developer.

## Decision

### The budget

- The budgeted quantity is the 1-σ uncertainty, in metres, of one component's distance
  to its nearest obstacle.
- Five sources: detection, lifting, tracking, geometry, timing (definitions and how each
  is measured are in CLAUDE.md §1.1).
- Combination: `σ_total² = Σ σ_source²`, which holds when the sources are independent.
  That assumption is tested, not trusted: V1-T4 requires 1-σ coverage near 68%, and a
  miss has to be explained by a measured correlation (then a cross term is added).
- Timing: latency jitter `σ_τ` becomes distance through the closing speed,
  `σ_timing = |v_closing| · σ_τ`. Mean latency is a bias; it is corrected by predicting
  the obstacle forward, not put in the budget.
- A source with no measurement is `None`. The total is then a lower bound and is
  labelled so. No default σ lives in a config. This understates uncertainty in v0, and
  the demo says which sources are missing.
- **The lifting curve.** Lifted positions are compared with dataset 3D cuboids and
  binned by range and viewing angle (bearing from the optical axis; object aspect when
  the track has a heading). Each bin stores a robust σ and its sample count; bins under
  30 samples are merged with a neighbour. Lookup gives every obstacle its expected
  error. In v0 the curve is fitted on 10 PandaSet scenes disjoint from the 3 demo
  scenes; in V1 it is refitted on the dev split only.
- `p_contact = Φ(-d/σ_total)` replaces any other route to probability.
- **Staging.** V0: lifting curve and per-component budget in the demo. V1: all five
  sources; Argoverse 2 reports the budget shift per source (σ ratio and coverage of the
  PandaSet budget), not only accuracy.

### Schema 3 (migration note)

- New `ErrorBudget`: `sigma_total_m`, `detection_m`, `lifting_m`, `tracking_m`,
  `geometry_m`, `timing_m` (each `float | None`), `complete: bool`.
- `ComponentRisk` gains `budget: ErrorBudget`.
- `Obstacle.pos_sigma_m` keeps its name and type; its value now comes from the curve.
- `SCHEMA_VERSION` 2 → 3. A schema-2 stream has no budget; readers treat it as
  all-`None`, `complete = False`. Tests updated in V0-T8.

### The reusable core: `vscs/core/`

| Module | Holds | Knows nothing about |
|---|---|---|
| `core/curve.py` | An empirical error curve: fit from (covariates, error) samples, merge sparse bins, look up σ | Cameras, ranges, vehicles |
| `core/budget.py` | Named sources, `None` handling, quadrature, per-component budgets | Which sources exist |
| `core/track.py` | Kalman tracking of a set of named components of one entity, with per-component covariance | Whether a component is a bumper corner or a knee |

- A "component" is a named part of a tracked entity: a vehicle part here, a joint in
  Overstride. The curve's covariates are plain named axes: (range, viewing angle) here,
  (viewpoint angle) there.
- **Rule:** `core/` imports only numpy and the standard library, nothing else from
  `vscs`. A test enforces it. VSCS-specific code (the `veh` frame, the component
  vocabulary, the five source names) lives in adapters: `perception/track.py`,
  `risk/`.
- Overstride is not touched. Extracting `core/` into its own package is a later step,
  taken only when Overstride actually adopts it.

## Consequences

- Two new v0 tasks (V0-T7 curve, V0-T8 budget core + schema 3) and 10 extra PandaSet
  scenes. v0 ship date moves from 2027-01-10 to 2027-01-24.
- The existing tracker is split: generic filter into `core/track.py`, VSCS glue stays in
  `perception/track.py`. Existing tracker tests must stay green through the move.
- V1 gains the source breakdown (V1-T4) and a larger cross-dataset task (V1-T9).
- The v0 demo shows a budget that is known to be incomplete (risk D-15).

## Evidence

None yet: nothing was measured in this session. The first evidence is the V0-T7 curve
in `metrics/results.jsonl`.
