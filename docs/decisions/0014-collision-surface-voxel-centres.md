# ADR 0014 - Collision surface through the outermost voxel centres, 2 cm voxels

- **Date:** 2026-10-01
- **Status:** accepted. The developer asked for "whatever is the most accurate option";
  this is the measured answer.
- **Task:** P2-T6 (convex decomposition)
- **Deciders:** developer, Claude

## Context

The voxel solid's face surface sits about half a voxel (1 cm) outside the true surface
on every side (decompose.py, "Known bias"). The developer chose to remove it. The full
reasoning is in the private technical guide (`docs/private/TECHNICAL_GUIDE.md`, 6.1).

## Options measured (5 fixture components; `measure_surface.py` in the session scratchpad)

Dense points (~1 mm apart):

| Option | Mean side error | Worst out / in | Mean volume error | Time | Gate |
|---|---|---|---|---|---|
| faces, 2 cm (old default) | 1.13 cm | 2.0 / 0.0 cm | 33.0% | 8.9 s | pass |
| **centres, 2 cm (chosen)** | **0.20 cm** | 1.0 / 1.0 cm | **7.5%** | 8.7 s | pass |
| faces, 1 cm | 0.50 cm | 0.5 / 0 cm | 13.0% | 20.0 s | pass |
| centres, 1 cm | 0.00 cm | 0 / 0 cm | 0.0% | 13.7 s | pass |

At a realistic dense-cloud spacing (the density `test_decompose.py` uses), 2 cm results
are unchanged. **Centres at 1 cm fail:** the points are too sparse to close 1 cm shells,
so the solids stay hollow, giving 17% volume error, a failed gate, and 151 s.

## Decision

`decompose.surface: voxel_centres` with `voxel_m: 0.02`. It is the most accurate option
that holds at realistic density. 1 cm voxels are more accurate only when the dense cloud
has points about 3 mm apart or closer; `decompose.py` already warns when it does not.
Re-check on the first real scan.

## Consequences

- The remaining error is quantization, about ±1 cm, and can lean *inside* on a side. The
  sweep inflates every component and obstacle by 5 cm each (`risk.yaml sweep`), so a 1 cm
  inward error never removes a warning by itself.
- Thin sheets (no voxel at least 2 deep) keep the face surface, rather than being
  collapsed by the inset.

## Evidence

- `tests/unit/test_decompose.py`:
  - `test_inset_removes_the_voxel_bias_on_the_fixture_wheel`;
  - `test_inset_mesh_is_half_a_voxel_inside_on_every_side`;
  - `test_a_scanned_sheet_keeps_its_faces`;
  - `test_default_surface_is_the_voxel_centres_adr_0014`.
- Full suite: 860 passed, 1 skipped.
