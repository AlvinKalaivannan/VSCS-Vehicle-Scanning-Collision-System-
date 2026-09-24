# ADR 0004 - Model the sliding door as two scanned variants, not a joint

- **Date:** 2026-09-24
- **Status:** accepted
- **Task:** P2-T7 (decided early, because it changes what P1-T2 must capture)
- **Deciders:** developer, Claude

## Context

`configs/model.yaml` had the sliding door as a `prismatic` joint along the vehicle x
axis, and `JointSpec` (ADR 0002) supports exactly that.

**A van's sliding door does not slide along a straight line.** It runs on a curved rail:
the door first pops *outward* from the body, then travels rearward, then settles back in.
A pure prismatic joint along x would place the door roughly the track's outward throw
(order 10-15 cm) away from its true position through the whole middle of its travel.

That error lands on a component with severity 2.5 in `configs/severity.yaml` - one of the
highest in the table - and on the side of the vehicle where a driver is most likely to be
manoeuvring near a kerb, post or another car. A collision model that is 10 cm wrong about
a costly component is worse than not modelling that state at all, because it produces
confident wrong answers.

This was noticed while reviewing `JointSpec` with the developer, before any URDF existed.

## Options considered

1. **Two scanned variants - door closed and door fully open.** P1-T2 already requires
   capturing the van with doors closed *and* open. Use whichever collision geometry
   matches the configuration recorded for that run. No joint, no interpolation, exact
   measured geometry at both ends.
2. **Keep the prismatic joint and document the error.** Simplest URDF, animates smoothly
   in Rerun, and wrong by 10-15 cm through mid-travel on a high-severity component.
3. **Model the curved track properly.** A path-constrained or 2-DOF parameterisation
   following the real rail. Most accurate; requires measuring the track geometry, and
   URDF has no native curved-path joint, so it needs a custom representation that every
   downstream consumer would have to understand.

## Decision

**Option 1**, chosen by the developer.

The sliding door is represented as **two collision geometries**, selected per run from the
joint state the operator recorded, rather than as an articulated joint.

- `configs/model.yaml` drops `joints.sliding_door_right` and gains a `variants:` section
  naming the two scans.
- `risk.yaml`'s `default_joint_states` keeps `sliding_door_right`, but its value becomes a
  variant selector (`closed` / `open`) rather than a displacement in metres.
- `JointSpec` is unchanged and still correct for the **rear doors and mirrors**, which are
  genuine revolute hinges. It is simply not used for the sliding door.
- The capture checklist already asks for a slow pass with the sliding door fully open;
  that pass is now load-bearing rather than optional, so it is called out as such.

This also matches R-15's chosen mitigation exactly: joint states are entered by hand per
run in v0.1. If the state is a hand-entered label anyway, a label that selects a measured
geometry is strictly better than a label that feeds a wrong kinematic model.

## Consequences

- **No intermediate door positions.** Acceptable: nobody manoeuvres a van with the
  sliding door half open, and VSCS only ever runs on recorded low-speed manoeuvres.
- The offline pipeline must produce **two collision models per vehicle**, so P2-T6
  (convex decomposition) and P2-T8 (URDF export) each run twice. Modest extra cost,
  already-captured input.
- A capture-day dependency: the door-open pass must be good enough to reconstruct on its
  own, not just a quick sweep. Recorded in the checklist.
- The door-open variant is only as good as the scan, whereas a joint would have
  generalised from one scan. Given the joint would have generalised *incorrectly*, this
  is a gain.
- If articulated mid-travel geometry is ever genuinely needed, option 3 remains open and
  nothing here blocks it. Automatic hinge-axis detection from open/closed scans is
  already a stretch goal in §6.

## Evidence

None yet - the van has not been scanned. The 10-15 cm figure is the typical outward throw
of a van sliding-door track and is an estimate, not a measurement. **Measure the actual
outward throw on capture day** (door half open, body to door skin, versus closed) and
record it in the devlog; it is the number that justifies this decision, and it costs one
minute with a tape measure.
