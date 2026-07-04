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
<replay implementation commit>
  -> e18b8ebbc26fd309d8e45bd58bef9c867948098a
  -> 7fc74d648328432a7f9f06d13c0e82a03f73a0c1
  -> 0611d384f5f26ef9bd8ff114be273e875c3fe719
```

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
  scripts/smoke_eef_pose_reasoning_production_v4_core_replay.py \
  scripts/validate_eef_pose_reasoning_production_v4_core_replay.py \
  tests/test_smoke_eef_pose_reasoning_production_v4_core_replay.py

ruff format --check <the same four files>
python3 -m py_compile <the three Python scripts and focused test>
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

## Exact launch design (not executed)

First deploy the pushed replay commit as a clean detached Git checkout on
Lustre. Resolve the source hashes from that checkout and submit one ordinary
L40S job:

```bash
POLARIS_REPO=/lustre/fsw/portfolios/nvr/users/lzha/src/PolaRiS-release-ramp16-core-fulltrace-gate-v1-20260703
COMMIT=$(git -C "$POLARIS_REPO" rev-parse HEAD)
RUNNER_SHA=$(sha256sum "$POLARIS_REPO/scripts/smoke_eef_pose_reasoning_production_v4_core_replay.py" | awk '{print $1}')
VALIDATOR_SHA=$(sha256sum "$POLARIS_REPO/scripts/validate_eef_pose_reasoning_production_v4_core_replay.py" | awk '{print $1}')
FIXTURE_SHA=$(sha256sum "$POLARIS_REPO/scripts/fixtures/reasoning_43075_job1098523_fulltrace_actions.json" | awk '{print $1}')
BUILDER_SHA=$(sha256sum "$POLARIS_REPO/scripts/build_reasoning_fulltrace_replay_fixture.py" | awk '{print $1}')
TEST_SHA=$(sha256sum "$POLARIS_REPO/tests/test_smoke_eef_pose_reasoning_production_v4_core_replay.py" | awk '{print $1}')
WRAPPER_SHA=$(sha256sum "$POLARIS_REPO/scripts/run_eef_pose_reasoning_production_v4_core_replay_srun.sh" | awk '{print $1}')
LAUNCH_ID=$(printf '%s\n' "$COMMIT|production_v4_core_ramp16|294x8+64" | sha256sum | awk '{print $1}')

sbatch --parsable \
  --account=nvr_lpr_rvp \
  --partition=batch \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=1 \
  --cpus-per-task=16 \
  --mem=96G \
  --time=00:45:00 \
  --job-name=pol_v4_core_fulltrace \
  --output=/lustre/fsw/portfolios/nvr/users/lzha/slurm_logs/polaris/%j.out \
  --export=ALL,FULLTRACE_POLARIS_REPO="$POLARIS_REPO",FULLTRACE_REPLAY_COMMIT="$COMMIT",FULLTRACE_CONTAINER_IMAGE=/lustre/fsw/portfolios/nvr/users/lzha/cache/polaris/polaris-eval-cuda13-fd00a51.sqsh,FULLTRACE_CONTAINER_SHA256=ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a,FULLTRACE_POLARIS_DATA_PATH=/lustre/fsw/portfolios/nvr/users/lzha/data/PolaRiS-Hub,FULLTRACE_OUTPUT_ROOT=/lustre/fsw/portfolios/nvr/users/lzha/results/polaris_eval/release_ramp16_core_fulltrace_gate_v1,FULLTRACE_HOST_CACHE_ROOT=/lustre/fsw/portfolios/nvr/users/lzha/cache/polaris/release_ramp16_core_fulltrace_gate_v1,FULLTRACE_LAUNCH_ID="$LAUNCH_ID",FULLTRACE_RUNNER_SHA256="$RUNNER_SHA",FULLTRACE_VALIDATOR_SHA256="$VALIDATOR_SHA",FULLTRACE_FIXTURE_SHA256="$FIXTURE_SHA",FULLTRACE_FIXTURE_BUILDER_SHA256="$BUILDER_SHA",FULLTRACE_TEST_SHA256="$TEST_SHA",FULLTRACE_WRAPPER_SHA256="$WRAPPER_SHA" \
  --wrap="bash $POLARIS_REPO/scripts/run_eef_pose_reasoning_production_v4_core_replay_srun.sh"
```

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
