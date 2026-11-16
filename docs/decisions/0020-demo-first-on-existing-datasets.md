# ADR 0020 - Demo-first plan on existing datasets; CLAUDE.md rewritten

- **Date:** 2026-11-16
- **Status:** proposed. Requested by the developer on 2026-11-16; accepted when the
  developer merges branch `spec-demo-first`.
- **Task:** V0-T1
- **Deciders:** developer, Claude

## Context

The September plan assumed a phone scan of the developer's own vehicle and recorded lot
drives. By 2026-11-16 no capture had happened, `metrics/results.jsonl` was empty, and
every gate waited on real data. The developer set new constraints:

- the MVP runs on existing footage and datasets, not self-recorded video;
- near-real-time stream processing;
- a real, deployed, clickable demo;
- a rough end-to-end v0 first, rigor (splits, cross-dataset test, box baseline) after;
- laptop without an NVIDIA GPU, plus Colab Pro T4;
- permissive licences only (Ultralytics is AGPL).

## Options considered and decisions (one line each)

| Decision | Chosen | Rejected, and why |
|---|---|---|
| Subject vehicle | The dataset's ego vehicle | A vehicle seen in third-person footage: 3D risk from one outside view is far harder, and nothing built so far would be reused |
| Demo dataset | PandaSet (listed as CC BY 4.0) | Argoverse 2, nuScenes, KITTI: non-commercial, so project policy (`configs/datasets.yaml`) gives results but not `publish_media`. comma2k19: front camera only, so side and rear components would never be at risk |
| Component model | Parametric template first, CC BY mesh in V1 | A scan-derived model first: needs a scan dataset with a clear licence (3DRealCar is `conflicting`) and the whole recon + fusion chain before any demo |
| 3D lifting | Ground-plane contact with dataset calibration | A learned depth model: more GPU work and a weights licence before the first end-to-end run |
| Ego-motion | Dataset poses | Visual odometry: drift would be debugged before anything is visible |
| Detector on the hosted demo | Cached RT-DETR detections from a T4 | Live on the free CPU: 750 ms/image measured (ADR 0011), times 6 cameras. A paid GPU Space: a purchase |
| Serving and hosting | FastAPI + WebSocket in Docker on a free Hugging Face Space | Vercel: not built for a long-lived Python CV process. Gradio: awkward for a custom synchronized canvas. A static replay: sliders would be fake |
| Trained student segmenter | Parked | No demo stage consumes its output |

## Consequences

- `CLAUDE.md` is rewritten (41 tasks → 22; 44 kB → 29 kB). Old task IDs
  remain valid in history; new IDs are `V0-`, `V1-`, `V2-`.
- Parked, not deleted: `capture/`, `recon/`, `seg/`, the student segmenter, VO, occupancy,
  underbody, driver replay. They stay under test.
- Core pair-mode modules are now `risk/sweep.py`, `synth/parts.py::classify_faces` and
  `risk/probability.py`. `recon/scale.py` and `seg/fusion3d.py` are parked with their
  branches.
- A 7-day draft timebox is added for core modules, with a teach-back before merge.
- The "single monocular instrument" boundary becomes "cameras are the only perception
  input"; ego-motion from dataset poses is a stated exception.
- ADR 0018 (Argoverse 2 subset) is reused as the V1 cross-dataset test. ADR 0017 and
  ADR 0019 stay proposed and parked. ADR 0015 (RAV4) no longer drives any task.
- The risk register is re-seeded with `D-` IDs at V0-T1.
- Unmerged work this plan depends on: `p6-stream-scaffold` (stream package),
  `p3-t4-sweep` (sweep contract and tests), `p4-multivehicle-data` (`synth/parts.py`).

## Evidence

- Licence and access findings of 2026-11-16 are web-search level, not read at source:
  PandaSet is tagged CC BY 4.0 on its mirrors while Scale's legal page names separate
  "PandaSet Terms of Use". It is **unverified** until the developer reads the terms
  (CLAUDE.md §11). No dataset was downloaded and nothing was measured in this session.
- Hugging Face free CPU tier (2 vCPU, 16 GB, sleeps after 48 h idle): from the Hub
  documentation as found by search on 2026-11-16.
