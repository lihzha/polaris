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

At that review point, before the later camera/image audit documented below,
the recorded next request was a paired canary on `DROID-FoodBussing`, one
rollout per checkpoint (two total):

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
launched by this evidence-only change. Its then-recorded paired-canary next
stage is superseded by the corrected camera/image-contract requirement below.

## Cadence-correct controller smoke and evidence finalizer

The cadence repair was committed as producer
`39418400493cdcf8cd8272608980a798f7929a20`, tree
`7fc1ff24053e3aeab5ed3e06068089b5aa596bc6`, with direct parent
`ee6d09351bed75e32db93ecf59c039a8e99fac9f`. Its fresh standalone L40S
controller smoke, Slurm job `1098975` (`pol_v6_cad_3941840`), completed
`0:0` on `pool0-00010`: the allocation ran from
`2026-07-04T22:37:09-07:00` through `22:42:19`, and the main srun ran for
300 seconds from `22:37:19` through `22:42:19`.

The exact remote inputs are:

- producer repo:
  `/lustre/fsw/portfolios/nvr/users/lzha/src/PolaRiS-concurrent-v6-cadence-3941840-20260705T053510Z`;
- result root:
  `/lustre/fsw/portfolios/nvr/users/lzha/results/polaris_eval/controller_concurrent_v6_cadence_smoke/3941840-20260705T053510Z`;
- saved wrapper:
  `/lustre/fsw/portfolios/nvr/users/lzha/launchers/polaris_eval/polaris_v6_cadence_smoke_3941840_20260705T053510Z.sbatch`;
- Slurm log:
  `/lustre/fsw/portfolios/nvr/users/lzha/slurm_logs/polaris_eval/pol_v6_cad_3941840-1098975.out`.

The copied immutable capture is
`/home/lzha/code/ego-lap/.codex_artifacts/polaris-v6-cadence-smoke-3941840-job1098975-success`.
Its pinned leaves are:

- raw: 793,098 bytes, mode `0444`,
  `393f0a57f409beb249635214ab2d7efb66783625048ddb18a5dc57426eaef2a5`;
- ready: 380 bytes, mode `0444`,
  `9e0a6826601a9d7019f6a4836a6524e259551bdc048b11e6184de0cf6dafc576`;
- inline attestation: 1,923 bytes, mode `0444`,
  `dfb0d40593241b85ea2af261e3de70d3c4d75fc6331109f38d32c305adefee42`;
- source identity: 800 bytes, mode `0644`,
  `d18a718f402d539d031ae699da1230144e8f9f016874530189d31916f508e2d1`;
- saved wrapper: 10,396 bytes, mode `0444`,
  `2215a73434d5c0f76368238932a8a18ebfd18125afb3b447f9396b4187fa18d4`;
- Slurm log: 43,539 bytes, mode `0644`,
  `115a7d83b887a3403138626cd85429615955ab99a050e07e9c710f093b772b56`.

The producer source set is pinned to `b5b1b621...` for the smoke,
`47f1a5af...` for config, `fa55f0b1...` for controller profile,
`b2a4df4c...` for controller repair, `0687434b...` for gripper runtime,
`bc34d745...` for IK safety, `fb7094a3...` for runtime contract, and
`a07a62f0...` for robust differential IK. The additional production sources
remain `b5464158...` for `scripts/eval.py` and `f66af500...` for the all-six
gripper trace.

All six exact Slurm rows are terminal and pinned: allocation, batch, extern,
main srun `.0`, and the two read-only `nvidia-smi` monitoring steps `.1` and
`.2`. The monitoring steps completed `0:0` on the same node at
`22:38:49..22:38:50` and `22:39:11`, respectively. The canonical sealed
sacct payload is 2,675 bytes with SHA-256
`05464ee37834c59e55bef41eabd489cc85fb93b545594041d9e21e3eb92dabb1`.

The smoke independently passed all 16 checks. Its discriminator records 168
fresh-DLS applies, 80 closed-endpoint fresh applies, ten distinct closed
desired poses, and exactly two driver endpoint changes. The disabled arm-side
cursor also records exactly two changes. Every actual close-interlock,
anchor, hold, release, deferred, and replay counter remains zero; no recovery,
controller abort, non-finite sample, or coupled-impulse failure occurred. The
closed semantic summary remains 17 safety reports, 5,856 applies, 732
post-policy samples, 6,003 open-endpoint samples, and eight adversarial slew
events.

The adapted stdlib-only finalizer is 108,170 bytes with pre-commit SHA-256
`953eb9e43c93a4aa3525bcd3001019348d30b38c0279be6ea51cfd88cd6c6d81`.
It retains the exact arm/finger `2 == 2` cross-binding and the zero-control-
state gate. The intended evidence commit changes only this worklog, the
finalizer, and its focused test, and must be a direct child of producer
`3941840`. No promotion attestation is claimed until that evidence-only child
is committed and the finalizer is run against its clean detached checkout.

A separate camera/image audit found an extra PolaRiS down/up filtering stage.
This controller-only smoke therefore does not validate the camera/image
contract and cannot authorize the paired checkpoint canaries directly. Its
truthful status requires a corrected camera/image-contract smoke first, and
only a passing result from that gate may advance to the paired official and
reasoning FoodBussing canaries. The finalizer keeps
`camera_image_contract_validated=false` and retains its controller-only scope.

## Corrected FoodBussing image-contract smoke and evidence finalizer

The image-contract producer is commit
`9d296361bb323b2e309a3b92a204c102908c61a6`, tree
`2e868fd4a31a55c9cedfb3221e4c2bc1fbbb9310`, with exact parent
`42e266353df71d5906e98975165f8aa021020dad`. It changes only the standalone
smoke, its focused host test, and its owned worklog. The seven producer source
digests are pinned by the evidence finalizer, including
`29db9e302179bb3ca4b05c14cae92e697376bb26066e4583b752bf9f6ce8d202`
for `scripts/smoke_splat_image_contract.py`.

The first wrapper attempt, Slurm job `1098980`, is preserved as a
non-authoritative failed evidence attempt. Its primary simulator srun completed
`0:0` and created a valid raw/ready/28-leaf bundle, but the host wrapper
canonicalized `/lustre/fsw` to `/lustre/fs11` while comparing the ready
marker's intentionally literal producer path. The post-srun assertion failed,
so allocation and batch accounting ended `FAILED 1:0`. No evidence attestation
may accept that attempt.

The corrected fresh attempt is job `1098982` (`pol_img_9d29636`) at result
root
`/lustre/fsw/portfolios/nvr/users/lzha/results/polaris_eval/image_contract_smoke/9d29636-20260705T061730Z`.
Allocation, batch, extern, and main srun `.0` all completed `0:0` on one
`pool0-00010` L40S. The allocation ran from `2026-07-04T23:19:11-07:00`
through `23:22:02` (171 seconds), and the srun ran from `23:19:22` through
`23:22:02` (160 seconds). Its reviewed capture identities are:

- raw: 25,039 bytes, mode `0444`, SHA-256
  `5f7f03d59728a54fd78fc10638886f9053bd3c2754c3aac80bf73960c2641a84`;
- ready: 427 bytes, mode `0444`, SHA-256
  `2c423b14a72fa6f7cc526928f99d62ebbb20f7c11dcaeb380d95129f96905be3`;
- source identity: 763 bytes, mode `0444`, SHA-256
  `0b3cec33eb34e48dd411a67ac022ff9b432677c68230018b74a49d0568be13d3`;
- post-srun validation: 1,066 bytes, mode `0444`, SHA-256
  `823bc029b4c3630df36a75344d835d61813276d509e9c50580931e5502ce3787`;
- saved wrapper: 13,018 bytes, mode `0444`, SHA-256
  `71b6dabef645c1807c4ad07fcdcbb360e7f6ae3238ce4836fdd3e96b18528934`;
- Slurm log: 21,283 bytes, mode `0644`, SHA-256
  `886c91b5d5385515c50228673b53c058bb9952236857fe918069cab05af06fc4`.

The raw record closes all 28 array/PNG/MessagePack leaves. Their canonical
path/size/SHA/mode/kind manifest is 9,456 bytes with SHA-256
`d14bdb03f67b2b105d21c8544ef19c1671c1dc437b9240fc60c1e3e0798d763b`.
The runtime remains pinned to container SHA-256 `ad566a3a...`, FoodBussing
scene `82cd641e...`, initial conditions `40091fae...` at index zero, Hub
revision `8c7e410...`, and both metadata digests. The live request exercised
only `base_0_rgb` and `left_wrist_0_rgb`; no checkpoint, server, reasoning
blank image, normalization, policy action, task metric, or success-rate
surface was exercised.

`scripts/finalize_splat_image_contract_smoke.py` is a separate stdlib-only
evidence finalizer. It does not import the producer. It independently parses
NPY, PNG, and restricted MessagePack bytes; recomputes the float renderer
conversion, RGB channel-order discriminator, robot-mask compositing, removed
resize difference, float32 half-pixel resize/pad, wrist rotation ordering,
noncommuting odd probe, request image bytes, and every array/PNG summary. It
uses lstat/open/fstat/read/fstat/lstat binding and rejects symlinks, hardlinks,
mode drift, replacement races, non-strict JSON, malformed binary formats, and
source/Git drift.

The reviewed finalizer source is 81,744 bytes with pre-commit SHA-256
`4faf9d6edbea18e2761b333b15e23bab57a6443f81cb31a111d3db91d27b1e7c`.

Volatile job/result identities are explicit CLI inputs. The finalizer requires
their expected sizes and SHA-256 values, a reviewer-supplied canonical digest
over all 28 leaf identities, and an immutable terminal sacct snapshot. It also
queries live sacct and requires exact equality with the allocation/batch/
extern/srun snapshot. Both `finalize` and `verify` reconstruct the same closed
attestation bytes; `finalize` is non-overwriting and publishes mode `0444`
with one hard link. The attestation's authorization object keeps checkpoint,
policy serving, task metric, benchmark, controller behavior, canary, smoke
suite, standard suite, and promotion fields false.

Host-safe pre-commit validation:

```text
focused image finalizer: 31 passed
image producer/finalizer/client integration: 103 passed, 30 subtests passed
Ruff lint/format: passed
Python byte compilation: passed
full retry raw/ready/28-leaf semantic replay: passed
git diff --check: passed
```

This evidence-only change launches no simulator, GPU, Slurm, checkpoint, or
evaluation job and publishes no attestation. The intended commit changes only
this append-only worklog, the image finalizer, and its focused test, and must
remain a direct child of producer `9d29636`.

## Finalized dual-evidence checkpoint-canary promotion gate

On 2026-07-05, `CODEX_AGENT_ID=polaris_v6_promotion_design` finalized the
schema-2 promotion gate in an isolated worktree at exact image-evidence C5
`5c9a2c50f564fb58d58777fbe34fb831ba362ec3`, tree
`707a5d7b659e7c4dfc13d19ede9ce8a8077aeec7`, direct parent image-smoke
producer C4 `9d296361bb323b2e309a3b92a204c102908c61a6`. The change is restricted to
this worklog, `src/polaris/eef_concurrent_arm_gripper_v6_promotion.py`, and
`tests/test_eef_concurrent_arm_gripper_v6_promotion.py`; it changes no
controller, evaluator, image, policy, or serving behavior.

The gate embeds and validates only the cadence-correct job-1098975 controller
attestation (11,481 bytes, SHA-256
`4b5f53524590711874a06fa3d2f47b1b430df7ff7b82b445d14e44db2c4e1e90`)
and freshly binds the immutable job-1098982 image attestation at
`/lustre/fsw/portfolios/nvr/users/lzha/results/polaris_eval/image_contract_smoke/9d29636-20260705T061730Z/smoke-1098982.image-evidence-attestation.json`
(28,839 bytes, mode `0444`, one hard link, SHA-256
`f85125e27c00bab0173a2f78642555bc4cdcf7d72ab5bb4cc9c2948cb84e4212`).
The image attestation is independently parsed in its actual finalized schema:
producer C4, reviewer C5, finalizer SHA-256
`4faf9d6edbea18e2761b333b15e23bab57a6443f81cb31a111d3db91d27b1e7c`,
terminal allocation/batch/extern/srun records, real FoodBussing environment,
production renderer call path, exact RGB conversion/compositing pixels,
resize-pad-before-wrist-rotation evidence, MessagePack image bytes, and the
removed-resize counterfactual. Its standalone authorization fields remain all
false; the C6 composition gate, not the evidence artifact itself, grants the
narrow next step. Job 1098982 exercised the public `base_0_rgb`/wrist request
only; it did not runtime-exercise the reasoning blank-image/model-key mapping,
either checkpoint, or normalization. Those remain closed static request
constraints whose first permitted runtime exercise is the paired canary.

The complete lineage binds controller C1 `3941840`, controller evidence C2
`be2f608`, image implementation `f1d32a3`, integration tip `42e2663`, image
producer C4 `9d29636`, and image evidence C5 `5c9a2c5`, while preserving the
exact v5 promotion-module bytes. The earlier controller job and promotion
evidence are not authority; `ee6d093` remains only the historical Git parent
of C1. Local validation also hashes all seven image-attested runtime sources,
including the production splat renderer.

The only authorized request is exactly one official-LAP3B and one
reasoning-43075 FLOW-10 canary on `DROID-FoodBussing`, response horizon 16 and
execution horizon 8, with native 720x1280 RGB uint8 images and model 224x224
images. Official uses `[external,wrist]` and row-R6; reasoning uses
`[wrist,external,blank]` and column-R6. Both bind resize/pad before the
180-degree wrist rotation and global float32 train-matched Q99, with
`single_arm` configured metadata and effective category null. Smoke-suite,
standard, native joint-position, pi0.5, and any other task/checkpoint/scale
remain explicitly unauthorized.

Host validation of the finalized gate:

```text
focused schema/promotion tests: 59 passed
promotion + image producer/finalizer/client + v5 preservation selection:
245 passed, 1 skipped, 30 subtests passed; the 2 closed historical v5 identity
tests fail only on `src/polaris/config.py`, as before
broad host-safe suite (Isaac-only robust-DIK test excluded):
1202 passed, 1 skipped, 30 subtests passed, 8 closed historical identity
tests failed; exact C5 baseline has the same 8 node IDs plus 2 superseded old
v6 promotion failures, so C6 introduces no new broad-suite failure
exact content-addressed dual-attestation/source authorizer: authorized=true,
eval_scale=canary, stage=paired_official_and_reasoning_foodbussing_canaries
Ruff lint/format: passed
Python byte compilation: passed
git diff --check: passed
final promotion-manifest SHA-256:
5a03e09fc5dd8d3c3d196909695d58e7ae589a99d3db9fe730877481858ab301
paired canary-request SHA-256:
6963d2cc0f3c02ee9a4e5f2f3c3718a027bbbfe36a97e884c48e83e94122be28
```

No GPU, simulator, checkpoint server, policy, Slurm job, or evaluation was
launched while finalizing this evidence-only authorization gate.
