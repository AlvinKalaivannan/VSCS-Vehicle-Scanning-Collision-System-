# ADR 0015 - Subject vehicle: a 2025 Toyota RAV4 replaces the van

- **Date:** 2026-10-11
- **Status:** accepted by the developer, 2026-10-11
- **Task:** P1-T2 / P1-T3 (capture), P2-T1 (vocabulary), P2-T7 (joints)
- **Deciders:** developer, Claude

## Context

The developer will record a **2025 Toyota RAV4**, which replaces the van as the scanned
and driven vehicle. This is CLAUDE.md §8.2's data fallback ("another family vehicle"),
chosen by the developer. Everything up to P2-T1 had been built for the van:
- a 14-component vocabulary with a sliding door and two cargo doors;
- ADR 0004 (the sliding door as scanned variants);
- hinge placeholders for the cargo doors.

No real data exists yet, so nothing measured is invalidated.

## Options considered

1. **The RAV4 as the main vehicle, with a new vocabulary (chosen).**
2. **The RAV4 as the second vehicle only (P1-T8):** not the developer's intent.
3. **Scan one vehicle and drive the other:** unworkable, because the collision model must
   be the vehicle that is driven.

## Decision

The 2025 RAV4 is the subject vehicle. The **component vocabulary, re-approved by the
developer**, has 17 components (`configs/seg.yaml`, `configs/severity.yaml`):

| Components | Severity |
|---|---|
| front / rear bumper | 1.2 |
| **front** left/right bumper corner (new: an SUV parks nose-in too) | 1.5 |
| rear left/right bumper corner | 1.5 |
| left / right mirror | 2.0 |
| **front** left/right side door (new) | 2.5 |
| rear left/right side door (now passenger doors) | 2.5 |
| **liftgate** (new; top-hinged rear hatch) | 2.5 |
| 4 wheels (0.4 against static geometry) | 0.6 |
| underbody (4.0 against static geometry) | 3.0 |

Removed: `sliding_door_right`. **ADR 0004 is superseded:** the RAV4 has no sliding door,
so `model.yaml variants` is empty.

**Joints** (`model.yaml`; placeholders until measured at P2-T7):
- side doors: revolute about +z, hinged at the front edge. Left doors open at negative
  angles (rear edge to +y), right doors at positive angles; about 70°.
- liftgate: revolute about +y, hinged at the top, opening positive (bottom edge up and
  back); about 80°.
- mirrors: unchanged.

`risk.yaml default_joint_states` holds every door and the liftgate at 0 (closed).

## Consequences

- **Unchanged:** the capture procedure, hub-mark poses (ADR 0012), marker placement, the
  scale / ground / frame pipeline and the risk engine. None of them assume a van.
- **Changed:**
  - P1-T2's "doors open" scan now means the four side doors and the liftgate;
  - capture-day measurements replace the van's door throw with each door's and the
    liftgate's opening;
  - P2-T7 joints now cover four doors, the liftgate and the mirrors.
- **The second vehicle (P1-T8)** can now be the van, if it is available, or any other
  car.
- **Fixture world:** `tests/fixtures/scene.yaml` keeps its van-like placeholder
  components (it says so itself). It is synthetic geometry for known-answer tests, not
  the vocabulary.
- **Older documents** (devlogs, earlier ADRs) say "van"; read that as the subject
  vehicle. The CLAUDE.md §1 note records the switch, and P2-T7's line is updated.
- **Privacy:** the RAV4's own plate, like any bystander's, is blurred before anything
  leaves `data/` (§9).

## Evidence

- The developer's message and answers, 2026-10-11 (operations log).
- `tests/unit/test_joints.py::test_the_shipped_hinges_match_the_approved_components`
  checks that every hinge in `model.yaml` attaches to a component in the new `seg.yaml`.
