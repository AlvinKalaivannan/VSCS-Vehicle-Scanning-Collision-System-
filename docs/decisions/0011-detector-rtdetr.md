# ADR 0011 - Object detector: RT-DETR (Apache-2.0), via Hugging Face transformers

- **Date:** 2026-10-01
- **Status:** accepted (§8.2 primary row "Apache/MIT-licensed detector (e.g. RT-DETR, YOLOX)")
- **Task:** P4-T4 (detection half; tracking is `perception/track.py`)
- **Deciders:** Claude, within the §8.2 primary already approved in CLAUDE.md;
  the developer can override

## Context

P4-T4 needs a detector for people, vehicles and cyclists. CLAUDE.md §10 excludes Ultralytics
YOLO (AGPL-3.0) and asks for Apache/MIT code *and* weights. §8.2 names RT-DETR or YOLOX
as the primary.

## Options considered

1. **RT-DETR** (Zhao et al., "DETRs Beat YOLOs on Real-time Object Detection", CVPR 2024):
   - Code: `lyuwenyu/RT-DETR`, Apache-2.0.
   - Weights: `PekingU/rtdetr_*`, Apache-2.0 on the model cards.
   - Already in `transformers`, which is in the Colab environment for Grounding DINO.
2. **YOLOX** (Apache-2.0): also eligible; it would need its own package and checkpoint
   handling.
3. **Ultralytics YOLO:** excluded (AGPL-3.0, §10).

## Decision

RT-DETR, `PekingU/rtdetr_r18vd` (the smallest ResNet-18 variant: speed matters for
Phase 6), through `transformers`.

**Licences checked on 2026-10-01:**
- The GitHub repo's LICENSE is Apache-2.0.
- The Hugging Face model cards for `rtdetr_r18vd`, `rtdetr_v2_r18vd` and `rtdetr_r50vd`
  are all tagged `license:apache-2.0`, ungated.
- Provenance note: the weights were trained on COCO, whose images carry their own Flickr
  licences. VSCS uses the *published weights* under their Apache-2.0 licence and
  redistributes no COCO images.

## Evidence (CPU, isolated environment; private evaluation per ADR 0006)

- Environment: torch 2.14.0 (CPU), transformers 5.17.0 (as ADR 0007).
- The post-processing signature matches: `post_process_object_detection(outputs,
  threshold, target_sizes)`.
- **Roboflow 21-class test split, first 20 photos:**
  - Car (or truck/bus) found in 18/20.
  - IoU against the union of each photo's labelled part boxes: median 0.98; ≥ 0.5 in
    18/20.
- **Latency on the developer's laptop CPU** (Intel64 Family 6 Model 197): median
  ~750 ms per image, about 1.3 fps.
  - This is R-20 evidence: model stages cannot meet 15 fps on this CPU. They are
    benchmarked on the Colab T4 (P6-T4).
  - This number is tagged with its hardware and is not a performance claim (§9).

## Consequences

- `perception/detect.py`:
  - `filter_detections`: tested logic.
  - `RTDetrDetector`: the adapter, verified on CPU.
- `perception.yaml detect.model_id` records the choice.
- Output feeds `perception/pipeline.py` as `Box2D`. Masks, when wanted, come from SAM 2,
  prompted with these boxes.
- If accuracy on real lot footage is poor, the §8.2 ladder steps to Grounding DINO, then
  occupancy-only. That step is the developer's decision.
