# VSCS Status — updated 2026-09-26
mode: pair            # pair | build
phase: P1
current_task: finish Phase 0 (P0-T5/T6/T7) and prepare for the October captures; the developer's sweep and scale drafts
health: ON TRACK

## Done since last update

- **Spec: trained segmenter added to CLAUDE.md** (developer's request). New tasks P1-T8
  (second-vehicle walkaround, October) and P4-T8 to P4-T12; risks R-17 to R-19; fallback
  row; §10 license rules; §11 gold-set protocol; cut-order slot #2. Spec only.
- **Toolchain installed:** ffmpeg/ffprobe 9.0.1, COLMAP 4.2.0 (CPU build, sha256
  verified), open3d 0.20.0, rerun-sdk 0.38.1. Pinned in `envs/`; ADR 0001/0003 addenda.
- **P1-T7 guard fix:** the dense stage now recognises COLMAP 4.x's "GPU support" banner
  wording. The real 4.2 binary is correctly refused as `no_cuda`.
- Still current from 2026-09-25: P1-T4/T5/T6/T7 tooling built and tested on synthetic data;
  P3-T3 and P3-T5 done on the fixture; marker boards and Colab dense adopted.

## Verified metrics (link to metrics/results.jsonl entries)

**None.** `metrics/results.jsonl` is empty; nothing has been measured on real data.

Session evidence (tests, not metrics): `main` **576 passed, 1 skipped**; ruff clean.

## Blockers — decisions and work only you can do

1. **Rate R-17 to R-19** (likelihood and impact) so they can go into `docs/RISKS.md`.
2. **Push:** `main` is 30+ commits ahead of `origin`. Say the word and I'll push after the
   security check. This is needed before P0-T5, because Colab clones the repo.
3. **Your drafts:** `risk/sweep.py` (`p3-t4-sweep`) and `recon/scale.py` (`p1-t6-scale`).
4. **Phase 0 captures:** P0-T7 checkerboard, the pre-flight test clip, and the P0-T6
   warm-up object. COLMAP and ffmpeg are now installed, so these can run as soon as the
   footage exists.
5. **Scan day:** measure the rear overhang (`recon.yaml vehicle_frame.rear_overhang_m`).

## Open risks triggered (IDs from RISKS.md)

None triggered. R-12 eased: the laptop toolchain is installed. The Colab CUDA COLMAP route
is still unverified.

## Next 3 tasks

1. **You:** rate R-17 to R-19; the push OK; P0-T7 calibration and the pre-flight check.
2. **You:** P0-T6 warm-up (a small object → `extract_frames.py` → `recon.py`, now
   runnable locally); the sweep and scale drafts.
3. **October (critical path):** P1-T2 van scan, P1-T3 lot day, P1-T8 second vehicle.
   Then P1-T4 to P1-T7 on the real data, together.

## GPU usage this month (approx compute units)

**0 CU.**
