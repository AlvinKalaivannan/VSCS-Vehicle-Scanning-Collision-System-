# ADR 0018 - Evaluate thin-obstacle recall and depth-by-range on an Argoverse 2 subset

- **Date:** 2026-10-19
- **Status:** **proposed.** The developer approved planning it on 2026-10-19; it is built in
  Phase 4 alongside P4-T1, and accepted with evidence.
- **Task:** P4-T1 (depth error at 1/3/5/10 m), R-07 (thin poles and kerbs missed), with
  P3-T2 as a cross-check.
- **Deciders:** developer, Claude

## Context

Lot day gives about 10–15 passes, with a handful of cones and poles each. That is enough to
pass gates, but thin as evidence for R-07 (recall of thin and low obstacles) and for depth
error by range bucket. Argoverse 2 Sensor (`argoverse2_sensor`) has thousands of
annotated `CONSTRUCTION_CONE`, `CONSTRUCTION_BARREL` and `BOLLARD` objects with 3D
cuboids and lidar, so it gives independent ground truth at many ranges.

Its licence was verified at source on 2026-10-19: CC BY-NC-SA 4.0, non-commercial. The
registry derives `private_eval`, `publish_results` and `rehearsal`; **not**
`publish_media`, and not `train` under project policy. So it is used to *measure*, never
to train, and no Argoverse frames appear in demos or the README.

## Proposal

1. **Subset, not the whole dataset** (the full set is ~1 TB):
   - about 10 logs from the **validation** split that contain cones or bollards close to
     the vehicle;
   - only `ring_rear_left` / `ring_rear_right` images, the lidar sweeps, the calibration
     and `annotations.feather`.

   Log ids are chosen by a script, stored in the registry entry's `source`, and fetched
   only after `require_use("argoverse2_sensor", "private_eval")`. Exact download tooling
   follows the official Argoverse instructions at download time; it is not yet verified.
2. **Frame check first.** The Argoverse docs give cuboid and sensor poses in the
   "egovehicle" frame, but don't state its origin or axis directions (checked
   2026-10-19).
   - Before any comparison, verify the axes against the VSCS `veh` frame (x forward,
     y left, z up; origin on the ground below the rear-axle centre).
   - Recover the **ground plane from the lidar** with the existing `recon/ground.py`
     RANSAC, instead of assuming the camera height of about 1.43 m is above ground.
3. **Adapter, which is plumbing:** Argoverse cuboids become the same truth that
   `eval/lot_truth.py` scores, i.e. obstacle centres in `veh`. Perception runs on the
   rear-camera frames with Argoverse's calibration as the mount. The existing
   `cone_errors` / `range_errors` / `depth_bucket` functions then report:
   - **recall** of cones, bollards and barrels by range bucket (R-07);
   - **depth error** at 1/3/5/10 m (P4-T1).

   Metrics are tagged `split: av2_private`, so they never mix with VSCS dev or test.
4. **Honest framing:** roof-mounted road cameras (~1.4 m, on-road) differ from a
   low rear mount in a parking lot. This *supplements* the lot-day numbers and never
   replaces them; the gates in §6 stay on the developer's own recordings.

## Consequences

- Storage: a few logs, tens of GB. Laptop CPU is enough for ground-plane depth; the
  detector runs on Colab.
- Attribution "(c) 2021 Argo AI, LLC" and a CC BY-NC-SA 4.0 link wherever results are
  reported.

## Evidence

To be added: recall and depth-by-range metric lines from the subset.
