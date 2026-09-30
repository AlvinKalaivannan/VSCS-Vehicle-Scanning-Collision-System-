# ADR 0007 - Grounding DINO and SAM 2 adapters verified on CPU before the first Colab run

- **Date:** 2026-09-29
- **Status:** accepted
- **Task:** P2-T2
- **Deciders:** developer (asked for the verification), Claude

## Context

The P2-T2 masks stage (`seg/masks2d.py`) wraps Grounding DINO and SAM 2. It was
unit-tested with fakes only; the real adapters had never run. The laptop has no NVIDIA
GPU, but both models run on CPU (slowly), which is enough to check that the code is
correct before any Colab compute is spent (R-11, R-12).

## Options considered

1. **Wait for the first Colab run:** costs compute units per bug, and each failure
   happens in the middle of a GPU session.
2. **Run the real libraries on CPU in an isolated environment**, against real car photos
   with hand-drawn part masks (Roboflow 21-class test split, private evaluation under
   ADR 0006). This is free and catches the bugs before Colab.

## Decision

Option 2. The environment was built outside the repo at `%TEMP%\vgc`, because the
scratchpad path exceeds Windows' 260-character limit during `pip install`. The laptop's
core environment was not touched.

## Versions verified to work (record for R-12)

| Package | Version |
|---|---|
| torch | 2.14.0 (CPU build; Colab keeps its own CUDA build) |
| transformers | 5.17.0 |
| sam2 | 1.0 (facebookresearch/sam2 main, 2026-09-29; built without CUDA extension) |
| hf_xet | required by transformers to download the Grounding DINO weights |

Model weights: `IDEA-Research/grounding-dino-tiny`, and SAM 2.1 hiera small
(`sam2.1_hiera_small.pt`).

## Bugs found and fixed (none were visible to the fake-based tests)

1. **SAM 2 would crash on the T4.** The adapter used `torch.autocast("cuda",
   dtype=bfloat16)`, copied from SAM 2's examples. The T4 has no bfloat16, and torch
   refuses it there. `autocast_for()` now picks bfloat16 / float16 / none to suit the
   hardware.
2. **Whole-vehicle boxes.** Every `seg.yaml` prompt ended "... of a van". On real photos,
   Grounding DINO grounds the vehicle word: for "side mirror of a car" the top box (0.64)
   was the whole car, labelled `car`. The prompts now name only the part, and
   `detector.ignore_labels` drops boxes matched to only a vehicle word. A test keeps
   vehicle words out of the prompts.
3. **900 boxes per query.** Passing `threshold=0.0` (so that the threshold lived in one
   place) returned all 900 query boxes, some malformed. The box threshold is now applied
   in post-processing, and boxes are clipped to the image.
4. **Label/box mismatch in transformers 5.17.** With zero boxes above the threshold, it
   returns `text_labels == ['']`. `grounded_to_hits()` treats "no boxes" as no detections,
   and any other mismatch as an error.
5. **Missing `hf_xet`.** The weights download fails without it. Added to the Colab
   notebook's install cell.

## Measured (CPU, 3 test photos; private evaluation, not a §6 metric)

- **SAM 2 video tracking**, prompted with boxes on one frame of an 8-frame shifted
  sequence:
  - Every frame was covered (forward + reverse passes).
  - Logits came back at full image resolution.
  - Mask IoU against the hand-drawn polygons: mirror 0.84, wheel 0.76, bumper 0.51.
- **Grounding DINO tiny, part-only prompts:**
  - Front bumper box IoU 0.98 and 0.83 (1 of 3 photos: 0.02).
  - Front wheel 0.44 / 0.00 / 0.00.
  - Mirror 0.01 / none / 0.01 (scores ~0.33, just above the 0.30 threshold).
  - Mean 0.25.
- **The whole stage** (`run_masks`) with both real adapters ran end to end.

## Consequences

- The adapters are correct against the real libraries. What remains is **detection
  quality on small parts** (wheels, mirrors) with the tiny model. That is P2-T2 tuning
  on the real scan: prompt wording, `grounding-dino-base`, and thresholds. SAM 2 tracking
  helps: a component needs only one good box on one keyframe.
- If detection stays weak after tuning, the §8.2 fallback is manual click prompts into
  SAM 2 on keyframes. That is the developer's call.
- On Colab, confirm again that `transformers` resolves to a version with the same
  post-processing signature (`threshold`, `text_threshold`, `target_sizes`).

## Evidence

- Verification script and probes (scratchpad, not committed; they read private data).
- Tests: `test_autocast_matches_the_hardware`,
  `test_filter_drops_whole_vehicle_boxes_and_clips`, `test_no_prompt_names_the_whole_vehicle`,
  `test_empty_result_with_a_stray_label_is_no_detections`,
  `test_mismatched_labels_are_an_error_not_a_guess`.
