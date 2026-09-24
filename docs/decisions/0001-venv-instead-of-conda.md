# ADR 0001 - Use a venv on the laptop instead of the conda env `vscs`

- **Date:** 2026-09-24
- **Status:** accepted
- **Task:** P0-T1
- **Deciders:** developer, Claude

## Context

CLAUDE.md section 4.3 specifies "Python 3.11 locally (conda env `vscs`)", and section 3
lists `envs/core.yml` as the laptop environment spec.

Conda is not installed on the development machine (Windows 11, `conda` not on PATH). A
system Python 3.11.9 is present, which already satisfies the version requirement. The
choice was put to the developer at the start of the session.

A second constraint showed up during setup and shaped the result: the network here
downloads at roughly 190 kB/s. A single `pip install` of the full intended dependency
list stalled for over twelve minutes with nothing installed, because pip was
backtracking through repeated `open3d` candidate wheels (each of which is large).

## Options considered

1. **venv from the existing Python 3.11.9, keep `envs/core.yml` for parity.** No
   install step, correct Python version immediately, tests runnable in the same
   session. Diverges from the letter of section 4.3.
2. **Install Miniconda and follow section 4.3 exactly.** Matches the document, and
   conda is genuinely better for the native-binary stages later (COLMAP is available on
   conda-forge). Costs a manual installer step and a session restart before any code
   could be verified.
3. **Author the env files and create nothing.** Would have left P0-T1..T4 acceptance
   criteria unverifiable this session.

## Decision

Option 1, chosen by the developer.

- `.venv/` is created from the system Python 3.11.9 and is gitignored.
- Dependencies are pinned in `envs/requirements-core.txt`.
- `envs/core.yml` is still written, with the same pins, so a machine that *does* have
  conda reproduces the environment as section 4.3 intended.
- Installation is **split into two tiers** rather than one command:
  `numpy pyyaml pydantic pytest pytest-cov ruff` first (about one minute, and enough to
  run tests and lint), then `opencv-python shapely trimesh typer nbformat scipy`.
- **`open3d` and `rerun-sdk` are deliberately not installed yet.** Neither is needed by
  anything built in Phase 0; `open3d` is first required for dense geometry and RANSAC
  ground-plane fitting (P1-T6/T7), `rerun-sdk` for visualisation (P1-T7, P3-T1). They
  are declared in the `viz` extra in `pyproject.toml` and in `envs/core.yml`, to be
  installed when the task that needs them starts.

## Consequences

- P0-T1 through P0-T5 were verifiable in the same session. This was the point.
- The documented conda path is now *aspirational* on this machine. Anyone following
  CLAUDE.md section 4.3 literally will not reproduce this setup; they should read this
  ADR. CLAUDE.md itself has not been edited - that is the developer's call, and is worth
  making if the venv path proves permanent.
- COLMAP still has to come from somewhere, and it is a native binary rather than a
  wheel. See ADR 0003.
- Installing `open3d` later may well hit the same pip backtracking problem. When it
  does, install it alone and pin the exact version that resolves, rather than letting
  pip solve it alongside everything else.
- The two-tier install is recorded in `README.md` as a single
  `pip install -r envs/requirements-core.txt`, which is the pinned form and does not
  backtrack. The tiering was a bootstrapping detail, not a permanent workflow.

## Evidence

- `conda` absent: checked on PATH at session start; `python --version` reports 3.11.9.
- pip backtracking: first combined install ran > 12 min with `site-packages` still
  holding only `pip` and `setuptools`; killed and re-run in tiers, tier 1 completed in
  about one minute.
- Result: `pytest -q` 238 passed / 1 skipped, `ruff check .` clean, coverage 95% on
  `common/` + `risk/` against the 80% requirement in section 5.
