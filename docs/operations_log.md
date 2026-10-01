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
