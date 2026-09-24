# ADR 0002 - Pydantic v2 for the section 4.2 schemas, and a definition for `JointSpec`

- **Date:** 2026-09-24
- **Status:** accepted
- **Task:** P0-T3
- **Deciders:** developer, Claude

## Context

CLAUDE.md section 4.2 defines the inter-module data contracts and says "dataclasses or
pydantic", leaving the choice open. Section 2 states modules communicate *only* through
these schemas, and R-16 names schema/convention drift between modules as a live risk
whose detection signal is "integration test fails; frames mismatched".

While implementing them, one gap turned up: **`ComponentModel.joints` is typed
`dict[str, JointSpec]`, but `JointSpec` is never defined anywhere in CLAUDE.md.** It is
referenced and never specified. Something had to be invented to make the schema
loadable, so it is recorded here rather than slipped in silently.

## Options considered

### Schema library

1. **Plain `@dataclass`.** Zero dependencies, fastest construction. No validation:
   a float where an int64 nanosecond timestamp belongs, or a probability of 1.4, would
   pass straight through into the risk maths.
2. **Pydantic v2.** Validation at construction, clean JSON round-trip for the
   `RiskFrame` JSON Lines streams section 4.2 requires, and `extra="forbid"` turns a
   typo'd or stale field into an immediate error. Costs a dependency and some
   construction overhead.

### `JointSpec` shape

1. Minimal: `axis`, `origin`, `limits`, as section 4.2's prose hints ("axis, origin,
   limits").
2. URDF-shaped: add `type`, split origin into `xyz`/`rpy`, name the parent and child
   links.

## Decision

**Pydantic v2**, with a shared base setting `extra="forbid"` and
`validate_assignment=True`.

Concretely, the validation that earns the dependency:

- `t_ns` is `StrictInt`, so a float timestamp is rejected outright. This is R-03
  ("phone variable frame rate ... never use frame index as time") enforced in the type
  system rather than in review.
- `p_contact` is constrained to `[0, 1]`; `severity`, `risk`, `pos_sigma_m` and
  `scale_error_m` to be non-negative; `Obstacle.extent` to be positive in every axis.
- `RiskFrame.worst_component` must actually name a component present in
  `per_component`.
- `ComponentModel` rejects duplicate component names.
- `SCHEMA_VERSION = 1` is stamped onto `ComponentModel` and `RiskFrame`, so a stale
  serialized stream is recognisable instead of being misread.

**`JointSpec` is defined URDF-shaped** (option 2), because P2-T8 exports a URDF and a
direct field-for-field mapping removes a translation layer that could introduce a
convention error:

```
type: revolute | prismatic | fixed | continuous
parent_link, child_link: str
axis: Vec3                  (direction; rotate it, never translate it)
origin_xyz, origin_rpy: Vec3
limit_lower, limit_upper: float     radians for revolute, metres for prismatic
```

with validators rejecting a zero axis and inverted limits.

Two supporting decisions worth naming:

- **`validate_timestamp_stream()`** lives in `types.py` and rejects non-monotonic
  timestamps *and* sequences that look like frame indices (a whole clip spanning under
  a millisecond). This serves P1-T4's "timestamp monotonic check passes".
- **`p_any_contact()` and `RiskFrame.from_components()`** live in `types.py`, not in
  `risk/`. Only the aggregation the schema itself defines is here - section 4.2 writes
  out `p_any_contact = 1 - prod(1 - p_i)` and `expected_damage = sum(risk_i)` as part of
  the contract. Choosing the speed factor, looking up severity, and running the alert
  hysteresis remain `risk/aggregate.py` and `risk/alerts.py` at P3-T5.

## Consequences

- Schema drift (R-16) now fails loudly at construction, in the process that made the
  mistake, instead of surfacing as a wrong distance several stages downstream.
- Pydantic validation costs time per object. `RiskFrame` construction happens once per
  frame, not per point, so this is not on a hot path. If profiling later shows
  otherwise, `model_construct()` bypasses validation for trusted internal paths.
- `JointSpec` was **Claude's invention, not the developer's specification.** It was
  reviewed with the developer on 2026-09-24 and **kept as defined** — but that review
  found that the *sliding door should not be a joint at all*, because it runs on a curved
  track. `JointSpec` remains correct for the rear doors and mirrors, which are genuine
  revolute hinges; the sliding door became two scanned collision variants instead. See
  **ADR 0004**. Reviewing the schema before the URDF existed is what made that a cheap
  change rather than an expensive one.
- Any future schema change must bump `SCHEMA_VERSION`, add an ADR, and update tests, per
  section 4.2.
- `p_any_contact` assumes per-component independence, which is false: adjacent
  components usually face the same obstacle, so the true probability is lower. It
  therefore over-warns rather than under-warns, which is the right direction for an
  advisory system. The assumption is documented in the function's docstring and must be
  stated in the writeup (section 9, honest claims).

## Evidence

- `tests/unit/test_types.py`, 35 tests: float and negative timestamps rejected, unknown
  fields rejected, frame-index sequences rejected, `p_any_contact([0.5, 0.5]) == 0.75`
  and agreement with an explicit product, `JointSpec` axis and limit validation,
  duplicate component names rejected.
- `tests/unit/test_io.py::test_riskframe_jsonl_round_trip`: a `RiskFrame` stream
  survives a JSON Lines write and read unchanged.
