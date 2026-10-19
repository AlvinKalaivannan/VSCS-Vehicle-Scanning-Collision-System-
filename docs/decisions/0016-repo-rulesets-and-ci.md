# ADR 0016 - Repository rulesets and CI

- **Date:** 2026-10-19
- **Status:** accepted by the developer, 2026-10-19 ("protect + CI, no PR gate")
- **Deciders:** developer, Claude

## Context

The repository is public, and `main` carries rewritten history (the trailer removal) that
must not be lost. Tests and lint ran only on the laptop, before each commit.

## Options considered

1. **Protect `main` + informational CI (chosen).** No deletion and no force-push on the
   default branch. A CI workflow runs on every push but is not a required check, so
   direct pushes to `main` stay possible.
2. **Strict:** also require a pull request and green CI to change `main`. Safest, but
   every change becomes branch → PR → merge.
3. **Protect only**, with no CI.

Linear history is not required: the project merges with `--no-ff` merge commits.

## Decision

- `.github/rulesets/main-protection.json`: a branch ruleset on `~DEFAULT_BRANCH` with the
  `deletion` and `non_fast_forward` rules and no bypass actors. It is applied on GitHub
  through the import in Settings → Rules → Rulesets, or via `gh api`; the file is the
  record of what was applied.
- `.github/workflows/ci.yml`:
  - Windows (matching the laptop), Python 3.11;
  - installs the pinned `envs/requirements-core.txt`;
  - runs `ruff check`, `ruff format --check` and `pytest -q`.
- **Pillow** is pinned explicitly. It had been arriving only through `rerun-sdk`, so a
  fresh install could silently lose it.

## Consequences

- A red CI run is visible on every push but blocks nothing. Fix it on the next commit.
- If CI becomes required later, that means moving to option 2 and amending this ADR.
