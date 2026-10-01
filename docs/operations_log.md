# Autonomous operations log

Append-only record of every operation Claude performs without step-by-step supervision
(developer's /goal of 2026-10-01). One line per operation, written as it happens and
committed with the work it describes. **Nothing here has been pushed**: the developer
verifies all of it first.

Format: `- YYYY-MM-DD HH:MM | kind | detail`
Kinds: branch, write, test, commit, merge, install, data, deferred, note.

## 2026-10-01

- 2026-10-01 02:32 | note | session start under /goal; main 59 commits ahead of origin; no push this session
- 2026-10-01 02:32 | note | docs-streaming-merge (ADR 0008/0009) left UNMERGED pending developer OK
- 2026-10-01 02:32 | branch | ops-log from main
- 2026-10-01 02:32 | write | docs/operations_log.md (this file)
- 2026-10-01 02:32 | commit | ops-log: add operations log
- 2026-10-01 02:32 | merge | ops-log -> main (local only)
- 2026-10-01 02:33 | branch | p3-t1-rerun-view from main
- 2026-10-01 02:34 | write | src/vscs/ui/rerun_view.py, configs/ui.yaml (+ KNOWN_CONFIGS 'ui')
- 2026-10-01 02:35 | write | scripts/view.py, tests/unit/test_rerun_view.py; CLAUDE.md §12 synced (view.py)
- 2026-10-01 02:35 | test | full suite green; real .rrd verified by 'rerun rrd verify'
- 2026-10-01 02:35 | commit | P3-T1: Rerun dev view
- 2026-10-01 02:35 | merge | p3-t1-rerun-view -> main (local only)
- 2026-10-01 02:36 | branch | p3-t2-perception-v0 from main
- 2026-10-01 02:39 | write | src/vscs/perception/depth.py (ground-plane v0), tests/unit/test_depth_v0.py, configs/perception.yaml depth.pixel_sigma_px
- 2026-10-01 02:39 | note | P3-T2 fixture finding: box bottom edge = near face; near-face anchoring puts cone near face within 2 cm (centre 5-13 cm); width/height read high (conservative)
- 2026-10-01 02:39 | commit | P3-T2: perception v0
- 2026-10-01 02:39 | merge | p3-t2-perception-v0 -> main (local only)
- 2026-10-01 02:40 | branch | p4-t3-occupancy from main
- 2026-10-01 02:40 | write | src/vscs/perception/occupancy.py (world height map: insert, carve, persist, expire, reset), tests/unit/test_occupancy.py
- 2026-10-01 02:40 | commit | P4-T3: persistent height map
- 2026-10-01 02:40 | merge | p4-t3-occupancy -> main (local only)
- 2026-10-01 02:41 | branch | p5-eval-risk-metrics from main
- 2026-10-01 02:42 | write | eval/metrics.py: attribution accuracy, lead time, TTC error, alert episodes, false alarms/min; configs/eval.yaml risk_metrics; tests
- 2026-10-01 02:42 | commit | P5 eval risk metrics
- 2026-10-01 02:42 | merge | p5-eval-risk-metrics -> main (local only)
- 2026-10-01 02:42 | branch | p5-t1-baseline from main
- 2026-10-01 02:44 | write | src/vscs/eval/baseline_bbox.py (single oriented box + baseline severity), tests/unit/test_baseline_bbox.py
- 2026-10-01 02:44 | note | P5-T1 fixture finding: pole 0.40 m from sliding door reads 0.15 m to the single box (mirror protrusion 0.25 m); same engine, geometry only
- 2026-10-01 02:44 | commit | P5-T1: single-box baseline
- 2026-10-01 02:44 | merge | p5-t1-baseline -> main (local only)
- 2026-10-01 02:44 | write | docs/devlog/2026-10-01.md, docs/STATUS.md checkpoint
- 2026-10-01 02:44 | commit | docs checkpoint
- 2026-10-01 02:44 | merge | docs-1001 -> main (local only)
- 2026-10-01 02:44 | note | streaming scaffolding (Phase 6 setup) is done on branch p6-stream-scaffold (from docs-streaming-merge + main), UNMERGED; its operations are logged in that branch's copy of this file
- 2026-10-01 02:45 | branch | p6-stream-scaffold from docs-streaming-merge; merged main in (resolved CLAUDE.md §12 conflict: kept view.py line and both planned lists)
- 2026-10-01 02:46 | write | src/vscs/stream/queues.py (DropOldestQueue), tests/unit/test_stream_queues.py (P6-T2 required backpressure tests)
- 2026-10-01 02:46 | commit | P6-T2 queue (branch p6-stream-scaffold, unmerged)
- 2026-10-01 02:47 | write | src/vscs/stream/metrics.py (StreamFrame, breakdown, LatencyRecorder, over_budget), tests/unit/test_stream_metrics.py (exact latency accounting)
- 2026-10-01 02:47 | commit | P6-T3 metrics (branch p6-stream-scaffold, unmerged)
