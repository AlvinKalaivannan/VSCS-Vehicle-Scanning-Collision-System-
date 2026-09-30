# ADR 0009 - ZoneModel: one interface for scanned, template and single-box vehicles

- **Date:** 2026-09-30
- **Status:** **proposed** (a §4 schema change: needs the developer's approval before any code)
- **Task:** P4-T13 (enables P4-T14 and Phase 6)
- **Deciders:** developer (pending), Claude

## Context

The streaming direction (ADR 0008) asks for per-**zone** risk on the ego vehicle *and on
tracked vehicles*: zone risk = worst TTC in the zone, body risk = worst zone. VSCS already
computes per-**component** TTC and severity for the scanned van (P2-T8 URDF, P3 risk
engine). What is missing:

1. **A way to give zones to vehicles that were never scanned:** the nuScenes ego car, and
   every car VSCS tracks.
2. **A way to say *which zone of the other vehicle* is at risk.**

## Proposal

**One `ZoneModel` interface, three fidelities:**

| Fidelity | Source | Used for |
|---|---|---|
| Scanned | P2-T8 URDF + components.yaml | The van (the core contribution) |
| Template | A box of given length/width/height split into named zones by `configs/zones.yaml` (front/rear bumper corners, front/rear centre, left/right sides, mirrors, wheels) | The nuScenes ego car; every tracked vehicle, sized from its 3D box |
| Single box | One zone = the whole box | The P5-T1 baseline |

A `ZoneModel` yields, for each zone: name, severity, plan footprint and z-range (the same
shape `model.urdf.plan_footprints` already returns). So the existing risk engine can take
any of the three unchanged.

**Proposed schema additions** (version bump, migration note, updated tests, per §4.2):

- `ComponentRisk.target_vehicle_id: int | None` and `ComponentRisk.target_zone: str | None`:
  for vehicle obstacles, which zone of the *other* vehicle the contact is predicted on.
  `None` for static geometry.
- `RiskFrame.body_risk_level`: the worst zone's alert level. This equals today's
  `alert_level`, so it may simply be a documented alias rather than a new field.
- `Component` keeps its fields; "zone" is documentation, not a rename. That avoids
  churning P2 and P3 code.

## Options considered

1. **This proposal:** reuses the engine and the baseline, and adds two optional fields.
2. **Separate zone types beside Component:** duplicates the engine paths.
3. **Template zones only, and drop the scanned model:** loses the core contribution.
   Rejected by ADR 0008.

## Decision

**Pending the developer.** Until approved, nothing in `types.py` changes, and P4-T13 does
not start.

## Consequences if approved

- The comparison becomes scanned vs template vs single box, which directly measures what
  a scan buys over a generic zone template.
- Template zones for tracked vehicles depend on their 3D box accuracy (R-06). Report zone
  attribution with that caveat.
- `configs/zones.yaml` needs a template the developer approves, just as P2-T1 approved
  the van's vocabulary.

## Evidence

None yet. The existing pieces it builds on: `model/urdf.py::plan_footprints`, the P3 risk
engine, and `eval/metrics.py`.
