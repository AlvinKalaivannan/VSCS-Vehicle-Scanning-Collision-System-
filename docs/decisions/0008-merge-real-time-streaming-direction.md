# ADR 0008 - Merge the real-time streaming direction into VSCS

- **Date:** 2026-09-30
- **Status:** accepted
- **Task:** cross-cutting (CLAUDE.md §1, §2, §3, §5, §6, §8–§12)
- **Deciders:** developer (chose "merge"), Claude

## Context

The developer set out a new direction for VSCS:

- per-zone collision risk as a near-real-time streaming system;
- a real, deployable demo, built the way production CV pipelines are built;
- public datasets as the development and evaluation base: nuScenes primary; KITTI,
  Waymo and Argoverse 2 optional;
- a demo on crash and near-miss footage (DoTA, CCD, DAD);
- replay as live RTSP streams;
- a staged concurrent pipeline with queues, backpressure and a latency budget (~15 fps,
  < 150 ms);
- observability, Docker packaging, then ONNX/TensorRT and a Jetson Orin.

This conflicted with CLAUDE.md in six places:

1. **Core data:** the developer's own van scan versus public datasets.
2. **Offline-first:** the old boundary versus live streaming.
3. **Hardware:** the paste assumes a desktop GPU or Jetson; none is available.
4. **Licences:** the public datasets are non-commercial, and the crash datasets are
   compiled from public dashcam videos.
5. **Schema:** zones on tracked vehicles need new §4 fields.
6. **Scope:** 5–8 h/week.

## Options considered

1. **Merge:** keep the scanned per-component van as the ego vehicle and the core
   contribution, and add streaming, datasets and latency engineering on top.
2. **Full pivot:** public datasets as the core; the van becomes a bonus.
3. **Assess first:** write up and decide later.
4. **Separate project:** a new repository.

## Decision

**Option 1 (the developer's choice).** Hardware: **Colab only**; no NVIDIA GPU is
available or planned.

What changes in CLAUDE.md:

- **§1:** new item 7 (streaming, latency budget measured on available hardware). "Offline-first"
  becomes **"Recorded input only"** (live = replayed recordings; no in-vehicle camera or
  deployment without approval). "Single monocular instrument" is scoped to the van
  capture. Advisory-only and not-a-safety-device are unchanged and apply to the demo.
- **Compute:** streaming is developed on the laptop CPU; model stages are benchmarked on
  Colab; a desktop GPU or Jetson is conditional (a purchase is the developer's decision, §0).
- **§2 and §3:** the streaming path; new `stream/`, `zones/`, `docker/`, and
  `configs/stream.yaml` and `configs/zones.yaml`.
- **§4.2:** zones described as a **proposal** (ADR 0009). No schema changes yet.
- **§5:** new required tests (backpressure, latency accounting, zone template, replay
  timing).
- **§6:**
  - P4-T13 to P4-T15: zones, nuScenes-mini offline, and the crash datasets (private).
  - New Phase 6, streaming (April → May).
  - New Phase 7, acceleration (conditional on hardware).
  - Cut order: Phase 7 first, and Phase 6 is trimmed before the Phase 3 MVP is touched.
- **§8:** R-20 to R-24; fallback rows for stream ingest and acceleration.
- **§9:** crash datasets are private evaluation only. Public datasets appear in a demo
  only within their licence terms and with the developer's approval. Latency claims name
  their hardware.
- **§10:** licence rows for the datasets and tools. ffmpeg and GStreamer (LGPL) are
  flagged for review.
- **§11:** streaming and zone metrics.
- **§12:** planned scripts.

## Consequences

- **The core result gets stronger rather than being replaced:** scanned zones versus
  template zones versus a single box, on the van and on nuScenes.
- **The paste's own phases map onto the plan:** its Phase 1 ("offline, correct") is
  VSCS P0–P5; its Phase 2 (real-time) is VSCS Phase 6; its Phase 3 (TensorRT, Jetson) is
  VSCS Phase 7, conditional.
- **Real-time results will be honest about hardware** (R-20). No real-time number is
  claimed for hardware VSCS was not run on.
- **Open decision for the developer:** the registry's policy blocks `publish_media` for
  non-commercial licences. CC BY-NC-SA itself allows non-commercial sharing with
  attribution, so a portfolio demo showing nuScenes frames would need that policy
  loosened. That is the developer's call and is not assumed here.
- **Open decisions from §10:** ffmpeg and GStreamer are LGPL (run as separate processes,
  not linked); the developer should confirm this is acceptable.
- **The October captures stay the critical path.** The van scan cannot be redone after
  snow.

## Evidence

- nuScenes licence confirmed on 2026-09-30: CC BY-NC-SA 4.0, non-commercial only,
  commercial licences through Motional (nuscenes.org terms of use).
- DoTA, CCD and DAD registered as `unverified` (no uses allowed until checked).
