# ADR 0003 - Windows-first local toolchain, and how COLMAP gets installed

- **Date:** 2026-09-24
- **Status:** proposed
- **Task:** P0-T1 (recorded now; acted on at P0-T6)
- **Deciders:** developer, Claude

## Context

The laptop is Windows 11 with no NVIDIA GPU. CLAUDE.md section 1 assigns COLMAP (CPU),
the risk engine, collision checks, Rerun visualisation, tests and scripting to the
laptop, and the GPU stages to Colab.

**COLMAP is not installed**, and it is not a Python wheel - it is a native binary with
its own dependency chain. P0-T6 (warm-up: phone video to COLMAP to splat) and P1-T5 (SfM
reconstruction of the van) both need it. This is written down now so the install is not
rediscovered under time pressure in October, when the capture window is open and the
weather is closing.

Section 10 lists COLMAP as BSD-licensed, which is compatible and needs no further
review.

## Options considered

1. **Official prebuilt Windows binary** from the COLMAP releases page. A zip containing
   `COLMAP.bat` and the executables; there are CUDA and no-CUDA builds, and the no-CUDA
   build is the correct one here. No build toolchain needed. Feature extraction and
   matching run on CPU, which is slower but workable for a few hundred scan frames.
2. **conda-forge `colmap`.** Clean dependency handling, but conda is not installed on
   this machine (ADR 0001), so this would pull the whole conda decision back open.
3. **Build from source with vcpkg.** Full control, and genuinely painful on Windows.
   Hours of build time and a real chance of a dependency failure. Not justified.
4. **Run COLMAP on Colab instead.** Possible, and it would be GPU-accelerated. But it
   means uploading raw capture footage to Drive before the privacy scrub, and it
   contradicts section 1's assignment of COLMAP to the laptop.

## Decision

**Option 1**: the official prebuilt Windows **no-CUDA** binary, extracted to a known
location and put on PATH, with the resolved version recorded here once installed.

Not yet done - it is not needed until P0-T6, which is blocked on the developer shooting
the warm-up video anyway. Recording it now so the two blocked tasks have one owner and
one known path.

### ffmpeg / ffprobe is also missing, and P1-T4 needs it

*Added 2026-09-24, second session.*

`ffprobe` is the authoritative source of per-frame container presentation timestamps,
which is exactly what R-03 and CLAUDE.md §4.1 require ("never use frame index as time").
It is not installed either.

`capture/frames.py` therefore has two backends: `ffprobe` (preferred) and an OpenCV
`CAP_PROP_POS_MSEC` fallback, chosen automatically with a warning when it falls back. The
fallback works and is tested, but it is backend-dependent and is not trustworthy for real
variable-frame-rate phone footage - which is the only kind this project will ever have.

**Install ffmpeg before the October capture day**, alongside COLMAP. Both are native
binaries on the same shopping list. Until then, P1-T4's acceptance criterion
("timestamp monotonic check passes; sync offset estimated and logged") is only verified
against generated constant-frame-rate video, which is not evidence about VFR behaviour and
is recorded as such in STATUS.md.

### COLMAP's dense stereo needs CUDA, so P1-T7 cannot run on the laptop as planned

*Added 2026-09-25.*

`colmap patch_match_stereo` - the dense step `configs/recon.yaml` names as the primary
(`dense.engine: colmap_patch_match`) - requires a CUDA GPU. The no-CUDA Windows build chosen
above does sparse SfM (P1-T5) but cannot do dense reconstruction, and the laptop has no
NVIDIA GPU. P1-T7 therefore needs one of:

1. **COLMAP dense on Colab** (T4 has CUDA): upload the undistorted scan frames and the sparse
   model, run `image_undistorter` / `patch_match_stereo` / `stereo_fusion` there. The scan
   frames show only the van, so the privacy concern of uploading footage is small.
2. **OpenMVS on the laptop** - the §8.2 dense fallback 1, which runs on CPU.
3. **Points from the Gaussian splat** (§8.2 dense fallback 2), already planned on Colab for
   P1-T7's visuals.

**Decided 2026-09-25 by the developer: option 1, COLMAP dense on Colab.** It is also the
most accurate of the three: full multi-view stereo on a CUDA GPU, the planned §8.2 primary.
OpenMVS would work on the laptop but slowly; splat-derived points are the least
geometrically accurate. Scan frames are uploaded to the developer's own Google Drive for
this - private storage, not publication, and scan frames show only the van. Note it is not a fallback *trigger* (§8.2's trigger is a dense run failing or running
out of memory twice); the primary simply cannot be executed on this machine.

Related toolchain facts settled this session:

- **`open3d` and `rerun-sdk` are not installed yet**, deliberately. See ADR 0001. They
  are first needed at P1-T6/T7.
- **No `torch` locally.** GPU-stage dependencies live only in
  `envs/colab_requirements.txt`. `common/log.py::seed_everything` seeds torch only if it
  is already imported, so nothing on the laptop imports it as a side effect.
- **Colab's own torch build is never overridden.** `envs/colab_requirements.txt` does not
  pin torch, and `tests/unit/test_notebooks.py::test_template_does_not_pin_torch`
  enforces that, because replacing Colab's torch is the usual way to break the runtime
  (R-12).
- **Ultralytics YOLO is excluded** (AGPL-3.0, section 10). A test asserts it is absent
  from `envs/colab_requirements.txt` rather than relying on memory.

## Consequences

- P0-T6 stays blocked on two things - the developer's warm-up video, and this install -
  and both are now written down instead of implied.
- CPU-only SfM will be slow. If P1-T5 turns out to be impractically slow on the laptop,
  that is a trigger to reconsider option 4 for the *scan* footage specifically, which
  contains the van and not bystanders, so the privacy objection is weaker there. That
  would need the developer's decision and an ADR.
- The fallback ladder for reconstruction (section 8.2: COLMAP, then GLOMAP or
  hloc+SuperPoint+LightGlue, then a phone-app export) is unaffected by the install
  method. Note that the fallbacks are Python/GPU tools and would likely run on Colab.

## Evidence

Nothing measured yet. Update this ADR with the installed COLMAP version and the
observed P1-T5 runtime once P0-T6 runs.

## Addendum 2026-09-26: installed

- **COLMAP 4.2.0** (commit be5e291), `colmap-x64-windows-nocuda.zip` from the official
  GitHub release, sha256 `7dd1e72f9632199b7f6d80ab6f545fa8d2a2ea023e4802cab3a4624e309f18a1`
  matching the digest the release publishes. Extracted to
  `%LOCALAPPDATA%\Programs\colmap`, with `bin\` on the user PATH. COLMAP is no longer
  listed in winget, so the release zip is the route (option 1, as decided).
- Every option `recon/sfm.py` and `recon/dense.py` pass exists in the 4.2 help text.
- **COLMAP 4.x changed its banner** from "with/without CUDA" to "with/without GPU
  support". `parse_cuda_status` read the 4.2 banner as `unknown`, so the dense guard
  warned instead of refusing. Fixed on `p1-t7-colmap42-banner`; the real binary now
  reads as `no_cuda`. The Colab CUDA build's exact wording is still unverified until the
  first P1-T7 run.
- **ffmpeg / ffprobe 9.0.1** (`Gyan.FFmpeg.Essentials`, winget). `ffprobe_available()` is
  True. On an ffmpeg-generated variable-frame-rate clip, the ffprobe and OpenCV backends
  agree and both report VFR. That is not evidence about real iPhone footage; P1-T4
  still needs that.
- P1-T5 runtime is not measured yet (needs the P0-T6 warm-up video).
