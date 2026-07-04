# Production V4 Core Full-Trace Replay Gate

Agent: `release_ramp16_core_fulltrace_gate`

Branch: `codex/release-ramp16-core-fulltrace-gate-v1-20260703`

Date: 2026-07-03

## Goal and scope

Implement, but do not launch, one model-free PolaRiS controller replay that
feeds the exact 294-action reasoning fixture plus a 64-physics-substep frozen
tail through the production v4 release-ramp controller. The replay must retain
the complete 13-DOF causal trace and video while proving that the production
core is the only target writer.

No production controller source, canonical worktree, cluster state, evaluation
registry, checkpoint, or policy server is changed by this branch.

## Ancestry

The production controller implementation is
`7fc74d648328432a7f9f06d13c0e82a03f73a0c1`. Target-runtime reset coverage
landed as the test-only direct child
`e18b8ebbc26fd309d8e45bd58bef9c867948098a`; it has no `src/` or `scripts/`
changes relative to `7fc74d6`. This replay branch was fast-forwarded to
`e18b8eb` before its owned files were committed.

Required final first-parent ancestry is therefore:

```text
<container-image identity mount fix commit>
  -> d32115d36f2dea510dee86edeaddcc58309afc2e
  -> 585ab6f72098fd67118fd8b33cdd90be809bed3a
  -> 2ebfe7db5b2a31887481781b214608976e8023db
  -> e18b8ebbc26fd309d8e45bd58bef9c867948098a
  -> 7fc74d648328432a7f9f06d13c0e82a03f73a0c1
  -> 0611d384f5f26ef9bd8ff114be273e875c3fe719
```

Commit `2ebfe7d` is the original replay implementation, `585ab6f` is its
independent validator/publication hardening child, and `d32115d` is the
synchronous publication / lexical-root P2 child. The image-mount fix child is
the launch checkout, so the runner and wrapper explicitly require `d32115d`
as `HEAD^`, `585ab6f` as `HEAD^^`, `2ebfe7d` as `HEAD^^^`, `e18b8eb` as
`HEAD^^^^`, and `7fc74d6` as `HEAD^^^^^`.

The four production-core SHA-256 identities remain:

- `src/polaris/config.py`: `ea38e87ab20f204929e39454bd9edf6b321d419cb3cebb61c7a6b9487f12373a`
- `src/polaris/eef_controller_profile.py`: `af5c7d73a0b1bd5bf229c1b54f3c271d46fbdaafc7b81b9a1bd2799133420ec1`
- `src/polaris/eef_controller_repair.py`: `3233945b7a70f1c93612fd1dab13fabf6b79591ea17d610282b6650b2d08f567`
- `src/polaris/robust_differential_ik.py`: `8add3b6bc3f33e2797a2c4cab2aa2ebf4c67c2ab07c197dd9a0cd004bfde49dc`

The upstream target-runtime gate for `e18b8eb` is job `1098636`, reported by
the root orchestrator as passing all `84/84` Isaac controller tests. This
branch does not launch another job.

## Implementation

Added:

- `scripts/fixtures/reasoning_43075_job1098523_fulltrace_actions.json`
- `scripts/build_reasoning_fulltrace_replay_fixture.py`
- `scripts/smoke_eef_pose_reasoning_production_v4_core_replay.py`
- `scripts/validate_eef_pose_reasoning_production_v4_core_replay.py`
- `scripts/run_eef_pose_reasoning_production_v4_core_replay_srun.sh`
- `scripts/eef_pose_reasoning_production_v4_core_gate_io.py`
- `tests/test_smoke_eef_pose_reasoning_production_v4_core_replay.py`
- this worklog

The runner configures
`arm_slew_0p95_gripper_rate0p25_fixed_anchor86_release_ramp16_mimic100_damping1p2_v4`
on the pristine action config, validates it, and only then substitutes
observation-only descendants.

The arm observer intercepts the core transaction argument, clones it, and
delegates exactly once to
`super()._set_targets_and_commit_gripper_close_arm_interlock(...)`. It never
calls an articulation target setter. After the core returns, it independently
re-runs the production nominal-bound and release-ramp helpers from the pending
failure-trace current/raw-DLS tensors. It requires exact `torch.equal`
identity among:

1. the core transaction argument;
2. the independent helper result;
3. the pending failure-trace final target;
4. the live articulation target readback.

Each record also binds little-endian float32 SHA-256 identities, exact endpoint
behavior (index 0 equals current; index 15 equals nominal), limited-joint masks,
and the matching causal 13-DOF command snapshot. The observer has zero target
setter calls, failure-trace writes, release-ramp state writes, and gripper
target/state writes. Static AST tests reject any such write.

The production all-six gripper trace remains in the inheritance chain. The
full-trace subclass only finalizes both observational traces before the next
arm command and adds synchronized 13-DOF snapshots.

## Independent post-Kit hardening

The fix child closes the independent validation and publication gaps without
changing `src/polaris`:

- The post-Kit validator calls the exact pure
  `validate_eef_all_six_gripper_trace` contract for episode 0, 302 policy
  steps, 2,416 applies, and no numerical failure. It retains the validated
  object and canonical digest in the manifest and cross-binds retained applies
  `2352..2415` to the full 13-DOF trace.
- Every full-trace gripper action/request is independently compared with the
  immutable 294-action fixture plus eight repeated tail actions. Exact endpoint
  changes are policy steps `198, 200, 265, 272, 281`, or applies
  `1584, 1600, 2120, 2176, 2248`.
- The official public episode-cadence validator, controller safety/profile
  validator, apply-count validator, and controller-report validator all run in
  the post-Kit process. The established finalizer validates every safety field,
  nested schema, digest, counter, maximum, diagnostic, gripper static contract,
  and gripper dynamic contract.
- The finalizer gained one optional cumulative limited-apply expectation. Its
  default remains the original single-close value; this replay passes the
  independently derived exact `217` limited / `2199` reached counts for five
  endpoint changes.
- Strict JSON uses a checked float parser and a recursive numeric audit, so
  nested exponent overflow and non-finite values fail even in otherwise unused
  fields. Lexical paths are `lstat`-checked before resolution and opened with
  `O_NOFOLLOW`.
- Output and cache roots must resolve to disjoint trees. Attempt and cache
  namespaces are created exclusively, cache cleanup is exact-path scoped, and
  failure handling never deletes a pre-existing success object.
- `SUCCESS` uses exclusive temp creation, file fsync, mode sealing, a
  non-replacing hard-link publication, temp unlink, directory fsync, and
  independent reread/hash/mode/link-count checks.

## Synchronous publication and lexical-root P2 fix

The current child factors the host-only namespace and `SUCCESS` operations
into a pinned, standard-library-only helper. The wrapper executes both helper
commands synchronously and checks their statuses directly; the `SUCCESS`
publisher is no longer hidden inside process substitution.

The publisher records the exclusive temporary inode before metadata checks,
tracks the hard-link attempt and successful return, and treats link completion
as uncertain when `link(2)` raises. Every exception removes the temporary only
when `lstat` still matches that owned inode. After a link attempt it likewise
removes `SUCCESS` only when its device and inode match, fsyncs the directory,
and re-raises. A pre-existing or concurrently replaced unrelated marker is
never removed. On success the helper seals mode `0444`, removes the temporary,
fsyncs, and validates exact inode, regular-file, size, single-link, and payload
identity. The wrapper has no fallible command after publication that can turn
an authoritative success into wrapper failure.

Root validation strips only trailing separators before the first `lstat`, so
a final-component directory symlink is rejected both with and without a
trailing slash. It requires an absolute normalized non-symlink directory,
rechecks the lexical inode around canonicalization, and only then compares the
canonical output/cache trees for equality or nesting.

Behavioral tests cover forced post-link failure with synchronous nonzero
propagation and no marker/temp, pre-existing marker byte/inode preservation,
unrelated replacement preservation, successful `0444`/single-link
publication, link-success-before-error cleanup, initial metadata rejection
cleanup, directory symlink rejection with and without a trailing slash, and
normal disjoint roots.

## First target replay and container-image identity mount fix

The first exact replay attempt, Slurm job `1098637`, passed the deliberate
Isaac exit-5 negative probe, the full `84/84` Isaac controller gate, and the
`97/97` in-container focused replay/sidecar tests. It then failed closed before
executing any controller action. The runner's first live provenance check
could not see the host squashfs path from inside Pyxis:

```text
ProductionV4ReplayError: missing/linked file
/lustre/fsw/portfolios/nvr/users/lzha/cache/polaris/
polaris-eval-cuda13-fd00a51.sqsh
```

The failure result has `completed_replay_payload=null`,
`controller_failure_evidence=null`, and `execution_segment=null`; therefore it
contains no controller-behavior evidence. The wrapper published immutable
`FAILED`, did not publish `SUCCESS`, removed the scoped cache, and left the
detached source clean. Scheduler state is `FAILED 1:0`, as required by the
fail-closed gate.

The minimal fix adds the already host-validated squashfs to the Pyxis mount
list at the same absolute path, read-only. This lets the runner independently
re-open the bind as a regular file, size-check it, and hash the exact image from
inside the container. The wrapper still passes the same image to
`--container-image`; the new bind is read-only and does not expose the broader
host cache directory.
The exact ancestry gate is shifted by one commit to bind this child to
`d32115d`.

The focused gate now requires exactly one
`${FULLTRACE_CONTAINER_IMAGE}:${FULLTRACE_CONTAINER_IMAGE}:ro` mount, rejects
an `:rw` variant, and checks the shifted five-generation ancestry chain in
the shell wrapper, runner, and validator. It exercises the runner's exact
ancestry capture and independently rejects drift at each of the five parent
depths.
Post-fix host validation is:

```text
focused replay gate: 47 passed
selected controller/gripper regression: 219 passed
full host-safe suite: 914 passed, 30 subtests passed
Ruff check / format check / Python compile: passed
bash -n / ShellCheck / git diff --check: passed
wrapper SHA-256: 33913db072df861e287d590004be519fbe9cc5b42cb091c10bcec77740cd473d
runner SHA-256: 1eba3493cb43f647f1831a28d401910491f180b9b6c0733a09878ab05de9b641
validator SHA-256: 8405d4de795e26c688578c51821357c78b7b8f5365bcb79fd1c5e8bd53305459
focused-test SHA-256: 31678318d3ee01e82aa5269b16635fd30bddbfd5cc0f7003394708f1dd5ebe01
```

Historical component SHA-256 values used by the `d32115d` checkout and failed
job `1098637` were:

- finalizer: `74dccfeb25c9522e5741eb72510f3f7940abd64678be8a357aca102fe2038fc7`
- wrapper: `cf5351fcc666c7fb6fa3abd0f742867f43d99bebff9e9a1000eca2e5d3316b5d`
- runner: `8a60694870d8e8efeddcc21bea063d1f900b39e8b53d4a4942c95527342d6c8b`
- validator: `7131c69a098ac0f512e6437efa45f96dc397e03927a650eee6a885952b8967c4`
- fixture: `daf2aa682f2296a93170f842a5adb13a4fbc6b2694fa5dca28de7ac7ad83d7cb`
- fixture builder: `d9151df864a447fc5e18a900188125f89a49fcd04e0bf703f1b7dc2e8a5872e4`
- focused test: `dd1cd389c1071774888a3923a4f937ec2e3bf1f8b425ef40a575b6a6c35fd9cf`
- gate I/O helper: `34b1cc6b493d2e0e078bd5769fb44a68d824a4deac8f58a54aefc41f83641cdb`

## Frozen replay contract

- Fixture size/SHA-256: `14478` /
  `daf2aa682f2296a93170f842a5adb13a4fbc6b2694fa5dca28de7ac7ad83d7cb`
- 294 width-8 little-endian float32 actions
- Action bytes SHA-256:
  `0e781cd1df2d00f3496c1feb2bf079e9194ad664710ac988cc9f7e8bcde11bce`
- Tail action 293 SHA-256:
  `b938c1ae7f29d0d762b48502af53789a7117e364514f3bdf9887ff5e3e36ab50`
- Tail: 8 policy steps / 64 physics substeps
- Total: `294 * 8 + 64 = 2416` contiguous applies
- Ramp windows: `1600..1615`, `2176..2191`, `2334..2349`
- Ramp indices: `0..15` exactly three times
- Required core counts: releases/starts/completions `3/3/3`, targets `48`,
  limited applies `38`, limited joints `221`, cancellation counts `0`
- Per-ramp limited applies/joints: `15/81`, `8/35`, `15/105`
- Interlock terminals: one natural anchor completion, two open cancellations
- Terminal ramp state: `release`, no next index, last apply/index `2349/15`

## Host validation

Commands and results:

```bash
ruff check \
  scripts/build_reasoning_fulltrace_replay_fixture.py \
  scripts/eef_pose_reasoning_production_v4_core_gate_io.py \
  scripts/smoke_eef_pose_reasoning_production_v4_core_replay.py \
  scripts/validate_eef_pose_reasoning_production_v4_core_replay.py \
  tests/test_smoke_eef_pose_reasoning_production_v4_core_replay.py

ruff format --check <the same five files>
python3 -m py_compile <the four Python scripts and focused test>
bash -n scripts/run_eef_pose_reasoning_production_v4_core_replay_srun.sh
git diff --check

PYTHONPATH=$PWD:$PWD/scripts:$PWD/src \
  /home/lzha/code/ego-lap/.venv/bin/python -m pytest -q \
  tests/test_eef_controller_repair.py \
  tests/test_eef_controller_profile.py \
  tests/test_eef_gripper_failure_trace.py \
  tests/test_robust_gripper_target_slew_host_stub.py \
  tests/test_run_isaac_pytest.py \
  tests/test_smoke_eef_pose_reasoning_production_v4_core_replay.py \
  -p no:cacheprovider
# 191 passed

PYTHONPATH=$PWD:$PWD/scripts:$PWD/src \
  /home/lzha/code/ego-lap/.venv/bin/python -m pytest -q \
  --ignore=tests/test_robust_differential_ik.py \
  -p no:cacheprovider
# 886 passed, 30 subtests passed
```

The independent hardening child then passed:

```text
focused adversarial gate: 33 passed
selected safety/controller/gripper regression: 366 passed
full host-safe suite: 900 passed, 30 subtests passed
Ruff check: passed
Ruff format --check: passed
Python compile: passed
bash -n: passed
ShellCheck: passed
git diff --check: passed
```

The synchronous publication / lexical-root P2 child then passed:

```text
focused behavioral gate: 41 passed
selected controller/gripper regression: 213 passed
full host-safe suite: 908 passed, 30 subtests passed
Ruff check: passed
Ruff format --check: passed
Python compile: passed
bash -n: passed
ShellCheck: passed
git diff --check: passed
```

The unrestricted host suite stops during collection only because the local
environment lacks `isaaclab`. The launch wrapper runs the omitted
`tests/test_robust_differential_ik.py` first in the pinned Isaac image. It
requires both the deliberate exit-5 negative probe and immutable exit-code
sidecar, then the exit-0 full test sidecar. This defends against Pyxis masking a
test failure as scheduler success. Matching successful target job `1098636`,
the container `PYTHONPATH` includes the repository `src` and `scripts`
directories plus the pinned Isaac Lab, Isaac Lab Tasks, and Isaac Lab Assets
source roots under `/.venv/lib/python3.11/site-packages/isaaclab/source`.
That same path is used by both `run_isaac_pytest.py` and the direct replay
runner.

## Exact launch design (not executed; independent literals required)

This repository must not derive its own commit, component hashes, or launch ID
at submission time. The prior dynamic example was removed because a checkout
can otherwise attest to itself after an unreviewed change. The independently
reviewed outer `sbatch` wrapper must contain literal values for all of:

- `FULLTRACE_REPLAY_COMMIT`
- `FULLTRACE_LAUNCH_ID`
- `FULLTRACE_RUNNER_SHA256`
- `FULLTRACE_VALIDATOR_SHA256`
- `FULLTRACE_SAFETY_VALIDATOR_SHA256`
- `FULLTRACE_FIXTURE_SHA256`
- `FULLTRACE_FIXTURE_BUILDER_SHA256`
- `FULLTRACE_TEST_SHA256`
- `FULLTRACE_GATE_IO_SHA256`
- `FULLTRACE_WRAPPER_SHA256`

No command substitution, Git lookup, or hash computation from the launch
checkout is permitted for those values. The most recent historical,
independently reviewed outer wrapper is SHA-256
`be8f5e12308c711d99c156ac6763ce093cf36faecac731ef88b5b58bfe3f2905`;
it targets `d32115d` and was used by failed job `1098637`, so it is invalid for
the new image-mount child. Older rejected wrappers are SHA-256
`343f9d2126b5549964985ba794be01c5dfce17b649b5cfb2646ff5abe841edb1`,
which targets `585ab6f` and omits the gate-I/O literal, and
`724867f6227ba4100db1c3f7d4e7f946312d9785c860499745e4f4cd519cf504`,
which targets `2ebfe7d`. A new outer wrapper may be reviewed only after the
final child commit and component hashes are frozen.

The resource contract remains one ordinary, non-array, non-requeue L40S job:
account `nvr_lpr_rvp`, partition `batch`, one node/task/GPU, 16 CPUs, 96 GiB,
and 45 minutes. The source must be a clean detached checkout. Output and cache
roots must already exist as distinct non-symlink directories. The wrapper
requires a fresh exclusive attempt namespace and a separate fresh exclusive
cache namespace.

The wrapper pins image SHA-256
`ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a`,
FoodBussing IC0 and scene hashes, read-only source/data mounts, fresh
attempt/cache namespaces, target-runtime tests, result/video, post-Kit
validation, and immutable success/failure markers. Its EXIT trap checks that
the detached source commit is still exact and completely clean, then removes
the uniquely scoped cache on both success and failure while preserving the
original exit status unless cleanup or source attestation fails.

## Remaining target-runtime risks

- The observer intentionally reads private failure-trace buffers and core ramp
  fields. Exact parent/production-source hashes make interface drift fail
  closed.
- CUDA/PhysX replay has not been executed for this child. Host tests validate
  logic and tamper rejection, not simulator behavior.
- Per-substep CUDA-to-CPU trace synchronization changes wall time but not target
  values. The job requests 96 GB and 45 minutes.
- Cross-run whole-physics bit identity is not the authority. Within-run core
  argument/helper/failure-trace/live-readback equality and exact aggregate
  counts are the mandatory gate.
- This is a controller replay, not a task-success or checkpoint evaluation.
