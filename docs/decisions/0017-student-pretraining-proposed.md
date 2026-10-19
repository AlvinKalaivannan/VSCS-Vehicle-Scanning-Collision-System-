# ADR 0017 - Pretrain the student segmenter on licensed car-part data, then fine-tune

- **Date:** 2026-10-19
- **Status:** **proposed.** The design was approved by the developer on 2026-10-19; it gets
  built when Phase 4 starts (P4-T10) and is accepted with evidence.
- **Task:** P4-T10, P4-T11
- **Deciders:** developer, Claude

## Context

The developer asked whether training on existing data first would make the model more
accurate on their own recording. VSCS has exactly one learned component, the student
segmenter. Everything else is either geometry (scale, frame, fusion, sweep, TTC, alerts)
or already-pretrained detectors used as they are (Grounding DINO, SAM 2, RT-DETR).

Transfer learning can therefore help only the student. Its main target is R-19
(overfitting to one vehicle), which the second-vehicle gold set measures.

## Options considered

1. **Train from ImageNet initialisation on the RAV4 pseudo-labels only** (the original
   plan).
2. **Pretrain on licensed car-part data, then fine-tune on the RAV4 pseudo-labels
   (proposed).**
3. **Train on external data only:** rejected. The evaluation is on the developer's own
   vehicle, and the teacher's pseudo-labels are the supervision the project is designed
   around.

## Proposal

- **Stage A (multi-vehicle pretraining):** fine-tune the chosen permissive model (§10) on
  **every licensed multi-vehicle car-part dataset** in the registry whose licence allows
  `train`. Today that is only `roboflow_car_parts_21`; any dataset later added through a
  licence request or our own recordings joins Stage A (checked with
  `datasets.allowed_uses`, 2026-10-19). Each dataset's classes map to coarse groups (bumper,
  door, mirror, wheel, rear hatch, other), because external taxonomies differ from ours.
  The Roboflow set, for example, has one mirror class and no liftgate.
- **Stage B:** a new 17-class head, fine-tuned on the P4-T8 pseudo-labels. Left/right
  comes from the 3D labels, and flip augmentation swaps it.
- **Ablation:** the same seed, schedule and splits, with and without Stage A. Both go to
  `metrics/results.jsonl`: van gold mIoU, second-vehicle gold mIoU, and T4 FPS.
  P4-T11's pass rule is unchanged.
- **Guards:**
  - `require_use(..., "train")`;
  - gold frames and the test split are never used for training or tuning;
  - no external images are published (`publish_media` is not allowed).

## Developer's choice (2026-10-19)

**Hybrid**: pretrain on multi-vehicle data, fine-tune on the RAV4 *training* frames, and
test **cross-domain** on held-out RAV4 gold frames plus the second vehicle (never trained
on).

External-only training was considered and rejected:
- it cannot learn the 17-class taxonomy (liftgate, left/right, door split);
- it drops the teacher-student design;
- the collision geometry must be the driven vehicle's own in any case.

## Consequences

- **Expected:** a larger gain on the second vehicle than on the RAV4 itself. Domain shift
  (web photos versus a phone walk-round) limits the gain. It must be measured, not
  assumed.
- **Compute:** Stage A is about 1–2 Colab CU (an estimate, logged per run, R-11).
- **October captures are unaffected.**

## Evidence

To be added at P4-T10/T11: the ablation metric lines.
