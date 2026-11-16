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
| Component model | Parametric template in v0; **real component segmentation in V1** (Grounding DINO + SAM 2 + multi-view mask fusion on posed renders of a CC BY mesh, scored against exact mesh labels) replaces it | A scan-derived model first: needs a scan dataset with a clear licence (3DRealCar is `conflicting`) and the recon chain before any demo. Mesh labels alone (first draft): leaves out the core CV contribution |
| Uncertainty | Per-component error budget by source, measured (ADR 0021) | A single confidence value or an assumed σ |
| 3D lifting | Ground-plane contact with dataset calibration | A learned depth model: more GPU work and a weights licence before the first end-to-end run |
| Ego-motion | Dataset poses | Visual odometry: drift would be debugged before anything is visible |
| Detector on the hosted demo | Cached RT-DETR detections from a T4 | Live on the free CPU: 750 ms/image measured (ADR 0011), times 6 cameras. A paid GPU Space: a purchase |
| Serving and hosting | FastAPI + WebSocket in Docker on a free Hugging Face Space | Vercel: not built for a long-lived Python CV process. Gradio: awkward for a custom synchronized canvas. A static replay: sliders would be fake |
| Trained student segmenter | Parked | No demo stage consumes its output |

## Consequences

- `CLAUDE.md` is rewritten (41 tasks → 26; 44 kB → 38 kB after the amendment below). Old task IDs
  remain valid in history; new IDs are `V0-`, `V1-`, `V2-`.
- Parked, not deleted: `capture/`, `recon/`, the student segmenter, VO, occupancy,
  underbody, driver replay. They stay under test.
- Claude writes every module. Core modules (`risk/sweep.py`, `core/*`,
  `classify_faces`, `seg/fusion3d.py`, `risk/probability.py`) merge only after the
  developer's `/teach-back`. `recon/scale.py` stays parked with its branch.
- The "single monocular instrument" boundary becomes "cameras are the only perception
  input"; ego-motion from dataset poses is a stated exception.
- ADR 0018 (Argoverse 2 subset) is reused as the V1 cross-dataset test. ADR 0017 and
  ADR 0019 stay proposed and parked. ADR 0015 (RAV4) no longer drives any task.
- The risk register is re-seeded with `D-` IDs at V0-T1.
- Unmerged work this plan depends on: `p6-stream-scaffold` (stream package),
  `p3-t4-sweep` (sweep contract and tests), `p4-multivehicle-data` (`synth/parts.py`).

## Amendment, 2026-11-16 (same day, before merge; directed by the developer)

1. Error budgeting is the central method and replaces any standalone confidence
   (ADR 0021). v0 gains the lifting error curve and a per-component budget in the demo.
2. The Argoverse 2 test is kept because Overstride's 3D lifter degraded 1.84× cross-lab
   (developer's figure); it reports the budget shift per source, not only accuracy.
3. Component segmentation is un-parked as V1-T7 and replaces the template.
4. Sim-to-real with CARLA is a V2-T3 option; CARLA leaves the parked list.
5. Tracker and budget are built as a domain-neutral core (ADR 0021). Overstride is not
   refactored.
6. The 7-day draft timebox is dropped. Claude writes `sweep.py`, `classify_faces` and
   `probability.py`; the developer does a teach-back on each before merge. Claude
   extended the same gate to `core/*` and `seg/fusion3d.py`; the developer can narrow it.

Effect on scope and dates: v0 grows from 12 to 14 tasks and fetches 13 scenes instead of
3; v0 ship moves 2027-01-10 → 2027-01-24; V1 grows from 7 to 9 tasks and ends mid-March
instead of end of February; V2 runs into April.

## Evidence

- Licence and access findings of 2026-11-16 are web-search level, not read at source:
  PandaSet is tagged CC BY 4.0 on its mirrors while Scale's legal page names separate
  "PandaSet Terms of Use". It is **unverified** until the developer reads the terms
  (CLAUDE.md §11). No dataset was downloaded and nothing was measured in this session.
- Hugging Face free CPU tier (2 vCPU, 16 GB, sleeps after 48 h idle): from the Hub
  documentation as found by search on 2026-11-16.
