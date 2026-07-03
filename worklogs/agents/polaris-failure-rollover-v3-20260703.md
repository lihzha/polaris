# EEF failure rollover evidence fix — 2026-07-03

- Worktree: `/home/lzha/code/PolaRiS-worktrees/eef-failure-rollover-v3-20260703`
- Branch: `codex/eef-failure-rollover-v3-20260703`
- Exact base: `00a6d0098b29eb4a9374ffc745a5b6b8e9df6d17`
- State: intentionally uncommitted; no push, deployment, simulator run, or job launch.

## Scope and evidence

Official candidate job 1098473 aborted on the attempted policy-121 physics
substep 0 arm apply. The gripper had 968 completed applies, so its terminal
entry was apply 967 at policy 120, substep 7. The tail validator instead
required policy 121, substep -1 and raised a secondary validation error. The
complete diagnostic payload had already been built, but validation occurred in
the builder's return expression before assignment into persistent state, so the
immutable failure raw recorded a null capture.

## Changes

- Derive terminal gripper cadence from
  `expected_final_apply_index = expected_apply_entries - 1`, including rollover
  through integer division and modulo at substep 0.
- Require exact integer types for failure coordinates, tail cadence counters,
  joint indices, and every per-entry apply/policy/substep index.
- Split controller-abort capture construction from validation. Retain the built
  payload in transaction state before secondary validation, so a validation
  exception cannot discard diagnostics.
- Preserve the retained dictionary in an incomplete, non-promotable failure
  payload together with the primary and secondary exceptions.
- Add rollover, bool-alias, cadence-tamper, forced-validation-failure, and
  retained-incomplete-payload regression coverage. Existing nonzero-substep
  fixture tests remain green.

No controller implementation, controller profile, target-slew/interlock
configuration, fixture, or action plan changed.

## Validation

- Focused host suite: `164 passed`.
- Broad host suite: `736 passed, 30 subtests passed`; only
  `tests/test_robust_differential_ik.py` was excluded because the host lacks the
  Isaac Lab import. The isolated robust/gripper host stub passed in the focused
  suite.
- Ruff check and format check: passed.
- Python compileall: passed.
- `git diff --check`: passed.
- Leakage audit: the diff contains only two evidence scripts, their two test
  modules, and this worklog; `src/`, controller behavior, configs, fixtures,
  action plans, and eval code are unchanged.

## Handoff

The diff is frozen for independent review. A later separately approved launch
should rerun only the official model-free candidate and require a complete,
independently verified failure raw before interpreting the retained arm ring
and all-six gripper tail.
