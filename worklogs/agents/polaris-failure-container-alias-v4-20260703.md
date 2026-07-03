# EEF failure container-alias verifier fix — 2026-07-03

- Worktree: `/home/lzha/code/PolaRiS-worktrees/eef-failure-container-alias-v4-20260703`
- Branch: `codex/eef-failure-container-alias-v4-20260703`
- Exact base: `6c4781feacf0d92955496fa6141465894f6dd7d4`
- State: intentionally uncommitted; no push, deployment, simulator run, or job launch.

## Root cause

The candidate runner records the lexical absolute container path supplied by
the wrapper, such as `/lustre/fsw/...`. Successful validation preserved that
lexical path. Failure validation instead called `Path.resolve()`, producing the
same file's canonical `/lustre/fs11/...` path and then requiring strict record
equality. The independently verified failure transaction therefore stopped at
a false path mismatch before checking the retained diagnostic capture.

## Changes

- Added one shared container-file helper used by both successful and failed
  candidate validation. It requires an absolute regular final-component
  nonsymlink, verifies exact size and SHA-256, and preserves the lexical path.
- Added one shared recorded-container helper used by both paths. It closes the
  record schema, compares profile/size/digest with type-strict equality,
  validates the recorded lexical path, and requires `os.path.samefile`.
- Samefile filesystem aliases and hardlinks are accepted only with matching
  content identity. Final-component symlink inputs, independent same-content
  copies, relative/missing paths, and schema/profile/size/digest/type/path
  tampering are rejected.
- Added a realistic parent-directory symlink fixture modeling the live
  `/lustre/fsw` alias. The full independent failure validator crosses the
  container comparison and reaches a deliberately invalid later production
  evidence check.

No controller implementation, controller profile, target-slew/interlock
configuration, fixture, action plan, or evaluation behavior changed.

## Validation

- Focused host suite: `169 passed`.
- Broad host suite: `741 passed, 30 subtests passed`; only
  `tests/test_robust_differential_ik.py` was excluded because the host lacks the
  Isaac Lab import. The isolated robust/gripper host stub passed in the focused
  suite.
- Ruff check and format check: passed.
- Python compileall: passed.
- `git diff --check`: passed.
- Leakage audit: only the independent validator, its host test module, and this
  worklog differ from the exact base; `src/`, controllers, configs, fixtures,
  action plans, and eval code are unchanged.

## Handoff

The diff is frozen for independent review. A separately approved official
candidate rerun should retain the lexical Lustre alias, pass the hash-bound
failure verifier, and then expose the complete arm/gripper failure evidence.
