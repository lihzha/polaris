# PolaRiS concurrent arm/gripper controller v6

Date: 2026-07-04

## Scope and identity

- Base commit: `c71b3d78bfd2f01fc6788b22f61270fc3bfb8a3a`
- Worktree: `/home/lzha/code/PolaRiS-worktrees/eef-concurrent-arm-gripper-v6-20260704`
- Branch: `codex/eef-concurrent-arm-gripper-v6-20260704`
- Public profile:
  `arm_slew_0p95_gripper_rate0p25_concurrent_arm_velocity_recovery8_clean2_mimic100_damping1p2_v6`
- No Slurm, simulator, GPU, registry, checkpoint, or evaluation job was launched.

## Controller behavior

- V6 has an explicit disabled close-interlock identity:
  `concurrent_arm_no_close_interlock_v1`, configured substeps `0`, fixed anchor
  `false`, and zero/null lifecycle evidence.
- Every normal apply computes a fresh DLS target and passes it through the
  existing 0.95 arm slew. Closing or opening the gripper does not hold,
  overwrite, defer, ramp, or replay an arm target.
- Abnormal measured arm velocity retains the fail-closed float32 envelope,
  current/predicted PhysX hard-limit guards, eight-substep maximum, clean-2
  requirement, and position/velocity/effort setter-readback transaction.
- The second clean sample is the final current-position hold and closes with
  `clean2_concurrent_resume`. The following apply is fresh DLS; v6 has no
  release-ramp phase, target, counter, deferred endpoint accounting, or stale
  lower target.
- V5 transition constants, event reason, release-ramp path, report schema, and
  output fields remain on their legacy branches. Existing v5-focused tests are
  unchanged and passing.

## Evidence and runtime contract

- The controller report adds v6-only fresh-DLS, normal-setter,
  closed-endpoint, distinct-desired-pose, recovery-owned, deferred-transition,
  and stored-replay counters. It closes every committed apply as either fresh
  DLS or recovery-owned.
- V6 recovery uses schema 4 and explicitly records a null release-ramp profile
  and its distinct concurrent transaction identity.
- Gripper dynamic evidence adds v6-only open-endpoint contact/mimic telemetry.
  Passive follower velocity above float32 `5.001 rad/s` is telemetry only.
  Completion fails only when the same open-endpoint sample also has any arm
  measured velocity above that joint's existing float32 recovery envelope.
  Non-finite evidence remains fail-closed.
- A failing coupled sample retains an independent first-failure diagnostic,
  even when the maximum-follower diagnostic occurred at a different sample.
- Sidecars and runtime contracts use schema 8 for v6, preserve schema 7 for
  v5, and aggregate the v6 telemetry counters and maxima across episodes.

## Target-surface smoke

- `scripts/smoke_eef_pose_controller.py` accepts
  `--eef-controller-profile` and applies the selected profile before
  `gym.make`; baseline output remains schema 2 and unchanged.
- The v6 discriminator performs one open step, ten distinct moving-EEF close
  policy steps (80 physics applies for the rate-0.25 transition), and ten
  distinct moving-EEF reopen steps.
- It requires every committed normal apply to be fresh DLS/slew, at least ten
  distinct desired poses while closed, zero close-interlock/release-ramp/
  deferred/replay evidence, and a passing emitted open-endpoint telemetry gate.

## Validation

Focused host-safe regression set:

```text
404 passed, 1 warning
```

Broad host-safe set, excluding the Isaac-Lab-only module and deselecting only
the eight immutable v5 source-identity checks listed below:

```text
1035 passed, 1 skipped, 8 deselected, 1 warning, 30 subtests passed
```

Ruff lint, Ruff formatting, Python byte compilation, and `git diff --check`
were clean. The warning is the pre-existing Torch `pynvml` deprecation warning.

The eight expected immutable-v5 identity failures are intentionally not
weakened or rewritten in this implementation descendant:

1. `tests/test_eef_velocity_recovery_promotion.py::test_exact_producer_sources_remain_unchanged`
   - `ValueError: V5 producer source digest drift: src/polaris/config.py`
2. `tests/test_eef_velocity_recovery_standard_promotion.py::test_exact_predecessor_and_controller_sources_remain_unchanged`
   - `ValueError: V5 producer source digest drift: src/polaris/config.py`
3. `tests/test_finalize_eef_pose_canary_trace_replay.py::test_status_writer_and_finalizer_bind_exact_srun_lifecycle`
   - `Gate0ReplayValidationError: production policy config source identity drift`
4. `tests/test_finalize_eef_pose_canary_trace_replay.py::test_status_writer_accepts_publisher_visible_parent_alias`
   - `Gate0ReplayValidationError: production policy config source identity drift`
5. `tests/test_finalize_eef_pose_canary_trace_replay.py::test_production_eval_evidence_allows_only_same_inode_path_aliases`
   - `Gate0ReplayValidationError: production policy config source identity drift`
6. `tests/test_smoke_eef_pose_canary_trace_replay.py::test_production_reset_source_is_seed_none_and_default_render`
   - `Gate0ReplayValidationError: production policy config source identity drift`
7. `tests/test_smoke_eef_pose_canary_trace_replay.py::test_capture_validator_binds_runtime_contract_tail_and_failure`
   - `Gate0ReplayValidationError: production policy config source identity drift`
8. `tests/test_validate_eef_pose_canary_controller_candidate.py::test_recursive_production_and_asset_comparisons_are_type_strict`
   - `Gate0ReplayValidationError: production policy config source identity drift`

Those checks continue to protect the exact v5 producer bytes. A distinct v6
implementation/evidence lineage must be added separately after live smoke
evidence exists.

## Independent review correction

Independent pre-launch review found three evidence-accounting defects without
finding a defect in the concurrent control path itself:

- the delayed-close smoke label still said `close5` even though the v6 0.25-rate
  profile executes ten close policy steps;
- open-endpoint sample counts were not bounded by total recorded runtime
  samples; and
- the controller report did not cross-bind recovery-owned target applies to
  recovery hold/active-substep counters.

The follow-up revision gives v6 the distinct
`eef_open115_then_close10_same_arm_pose_v2` identity while preserving the
legacy `close5` identity, adds the missing runtime-sample bound, and requires
all three recovery ownership counters to agree. New negative regressions cover
each finding.

Post-correction validation:

```text
focused: 406 passed, 1 warning
broad host-safe: 1037 passed, 1 skipped, 8 deselected, 1 warning, 30 subtests passed
```

The same eight immutable-v5 source-identity checks remain the only deliberate
deselections. Ruff lint/format, Python compilation, and `git diff --check`
passed. No GPU, simulator, Slurm, checkpoint, or evaluation job was launched
before this correction was committed.

## L40S smoke attempt 1098921

The first exact-commit target run (`b6dec3d`, Slurm `1098921`) reached a fully
initialized FoodBussing environment and validated the initial v6 safety
capture, then failed before its first physics action with
`PolaRiS EEF gripper trace process lacks policy context`. The failure was in
the standalone harness: v6 enables the production all-six gripper trace, while
the smoke called `env.step` without the `begin_eef_policy_step` metadata that
`scripts/eval.py` installs before every policy step. No controller result or
checkpoint evidence was produced; the immutable failure raw and log are kept.

The follow-up makes every one of the smoke's five `env.step` call sites install
the exact episode/policy-step trace context when the selected profile enables
that trace. Policy indices restart at zero after each environment reset and
advance once per outer step, matching production evaluation cadence. Profiles
without the trace remain unchanged. A source-level regression binds all five
call sites to the helper before the bounded rerun.

Independent read-only review verified the implementation at all five call
sites, the episode identities `0..15`, reset behavior, and per-outer-step
cadence. The regression parses the harness AST and now binds each `env.step` to
its immediately preceding helper call, exact episode and policy-step
expressions, zero initialization, exactly-one increments, and the
profile-conditional delegate to `finger_term.begin_eef_policy_step`.

Post-fix host-safe validation remains:

```text
focused: 406 passed, 1 warning
broad host-safe: 1037 passed, 1 skipped, 8 deselected, 1 warning, 30 subtests passed
```

Ruff lint/format, Python compilation, and `git diff --check` pass. The eight
deselections are the same immutable-v5 source-identity gates documented above.
