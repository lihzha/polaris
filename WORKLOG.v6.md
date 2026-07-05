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

## Reviewed post-job finalizer

Job `1098922` completed `0:0` after 317 seconds on `pool0-00016`. Its exact
raw/ready/inline-attestation/source/wrapper/log bundle was fetched and passed
an independent strict-JSON, finite-value, source-byte, mode/hash, scheduler,
counter, and log audit. The inline wrapper attestation is retained as an input,
not used as the promotion attestation, because it does not bind the producer
parent, saved wrapper and log, terminal Slurm lifecycle, or a reviewed
finalizer identity.

`scripts/finalize_eef_pose_controller_v6_smoke.py` is a new stdlib-only,
v6-specific evidence finalizer. It changes no simulator/controller source and
does not touch either immutable v5 promotion module. It pins all six capture
artifacts, producer commit/tree/parent and ten source hashes, the exact image
and FoodBussing scene, the allocation/batch/extern/srun terminal `sacct` rows,
and the evidence-descendant commit/tree/finalizer hash. Image and scene
size/mode/link/time metadata are pinned; both mtime and ctime predate the srun,
recovering temporal provenance for the otherwise post-job scene digest. Input
leaves use lstat/open/fstat/read/fstat/lstat binding and reject symlinks,
hardlinks, mode drift, and parent replacement. The fixed l401 aliases must
resolve to the pinned canonical user root, and both Git inputs must be clean,
standalone, top-level detached checkouts with in-root `.git` directories.

The semantic verifier independently closes every nested result/safety/
gripper/recovery/controller schema, reconstructs the 13 target geometries and
the 21-pose concurrent discriminator, binds per-report cadence and aggregates,
and proves finite adversarial q/dq plus genuine bounded slew saturation. It
publishes only one non-overwriting mode-0444 promotion attestation and labels
its scope as a standalone controller smoke with no checkpoint or task metric.

Host-safe finalizer validation before commit:

```text
focused finalizer: 47 passed
broad host-safe: 1084 passed, 1 skipped, 8 deselected, 1 warning, 30 subtests passed
```

The eight deselections are the same immutable-v5 source-identity gates listed
above. Ruff lint/format, Python compilation, and `git diff --check` pass. The
separate promotion manifest and any checkpoint-canary authorization remain
blocked until this committed finalizer is run and its immutable output is
verified.

## Controller-smoke promotion to paired checkpoint canaries

The committed finalizer at evidence commit
`f4a27ce2bdbbaf2b87a38b4850390f9697ce8f9e`, tree
`4c4ce225bdfd57564e2e90db7657f9dc807a93f8`, was run against producer
`6e4b7c5be5ff6db670970774be3250c5d5ffa4d2`. Its finalizer source is exactly
106,246 bytes with SHA-256
`f9ab24398286d5e4db2af816cfa86c9b0b355c13eeb246e307331b5e14720c4c`.
The resulting promotion attestation is exactly 10,423 bytes, mode `0444`, one
hard link, and SHA-256
`c359e978bf4aede7555fd3d6118a2abf5f7f4c2e5cf058326d7c3304bda2305a`.

`src/polaris/eef_concurrent_arm_gripper_v6_promotion.py` is a separate,
stdlib-only, v6 evidence gate. It embeds those exact attestation bytes, parses
them with duplicate-key/non-finite rejection, and freshly checks a regular,
single-link, mode-0444 attestation through lstat/open/fstat/read/fstat/lstat.
Production validation requires the literal pinned result path. An offline
content-addressed mirror is allowed only through an explicit opt-in and is
reported in the authorization result. The gate also freshly binds the ten
producer/controller source hashes, the Commit-A finalizer bytes, and both v5
promotion modules. The v5 bytes remain:

- canary promotion:
  `f98f6d3ae6eb06f0127e3ec686fa70e3bb524ea892582b0ee3461b0dd6d84df4`;
- standard promotion:
  `8cb836645dde741876ff5d10b285761bc8f47c822732e9a2e1c469fb79ee0e06`.

The sealed provenance identities remain mode `0444`: `sacct.json` is 1,823
bytes / `13ec9313b8a593463e23a649798871338ca59e672a530c43849dd9125938996c`;
the saved job script is 9,548 bytes /
`a74f46d5e6d1e359b8df9bc02209e1b08c10b400cfd710826c6203fdeec55669`;
the Slurm log is 44,136 bytes /
`64179ecdc1ae32b51fe12a88b11987df955e7053a03e2b26cfad30c08a0621f6`;
and source identity is 800 bytes /
`52c9a5c00506886e68dd394950fb84e80cbd9bfb18c91290e3abd201162561ad`.

The closed validation summary is 17 safety reports, 5,856 total controller
applies, 732 post-policy samples, 6,003 total open-endpoint samples, 13
ordinary pose cases / 4,680 ordinary applies, 1,000 delayed-close applies, 168
concurrent applies, 80 fresh-DLS closed applies, 10 distinct closed desired
poses, 99 discriminator open samples, and eight adversarial applies with eight
slew events. Maximum position error is 0.001071291510015726 m, maximum rotation
error is 0.4384913281711513 degrees, and maximum follower velocity is
0.7730712294578552 rad/s. Controller aborts, coupled-impulse failure samples,
and recovery events are all zero.

The only authorized next request is a paired canary on `DROID-FoodBussing`,
one rollout per checkpoint (two total):

- official LAP-3B at HF revision
  `601db9c1ab4bcaf6dddb160c7b2dec589a67b730`, content manifest
  `567cc3ff7d20f3f03913a6f11c3fa151f789e1c0118ed5af0eea24d9cc48f20e`,
  public two-image `[external, wrist]` legacy order and train-matched rows-R6;
- reasoning checkpoint
  `gs://v6_east1d/checkpoints/lap_oxe_magic_soup_reasoning_full/oxe_magic_soup_reasoning_full_v2_flow_pred0_cf0_ckpt25_v6_32_b512_s42_20260630/43075`,
  inference subset
  `bb9ea5bb041f689a08f914cac7dfe5d061c822ddbe87e292f9c7878a9d3bfc4d`,
  three-image `[wrist, external, blank]` order and train-matched columns-R6.

Both requests are pinned to FLOW with ten integration steps, response horizon
16 / execution horizon 8, 224x224 RGB uint8 input, wrist resize-with-pad then
180-degree rotation, and train-matched float32 global-Q99 formulas. The
checkpoint metadata may say `single_arm`, but effective category selection is
null because global statistics are required. The v6 absolute EEF controller
and IK profiles are exact.

These are required prelaunch identities, not claims produced by the standalone
controller smoke. The attestation and promotion gate still say checkpoint
loaded `false`, policy serving validated `false`, camera/image contract
validated `false`, image order/resolution validated `false`, normalization
validated `false`, and task-success metric validated `false`. No smoke-suite
or standard evaluation is authorized. The scene digest was not logged by the
smoke job; only its post-job digest plus pre-srun metadata were validated. No
live recovery event or follower-threshold crossing was observed.

Host-safe promotion-gate validation before commit:

```text
focused promotion gate: 69 passed
promotion gate + committed finalizer regression: 116 passed
broad host-safe: 1153 passed, 1 skipped, 8 deselected, 1 warning, 30 subtests passed
Ruff lint/format: passed
Python byte compilation: passed
git diff --check: passed
promotion evidence SHA-256:
714b22a185ff06135cdc84d03a17347943c405b3d782f3a0141455f0194eb937
```

No GPU, simulator, Slurm, checkpoint, canary, smoke-suite, or standard job was
launched by this evidence-only change. The paired canary is merely the next
permitted stage and still requires a separate exact launch review.
