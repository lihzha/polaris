# PolaRiS EEF production-v4 integration — 2026-07-03

- Agent: `polaris-production-v4`
- Base commit: `557ef7217de7fd4980fee8305bbe53cd3c3b0dd1`
- Worktree:
  `/home/lzha/code/PolaRiS-worktrees/eef-iksafety-terminal-v4-production-20260703`
- Branch: `codex/eef-iksafety-terminal-v4-production-20260703`
- Scope is isolated to the dedicated worktree. No canonical checkout,
  diagnostic implementation, Pi policy path, Ego-LAP consumer checkout,
  cluster wrapper, or shared launch state was changed. No simulator or cluster
  job was launched from this worktree.

## Production gripper adaptation

- Added runtime profile
  `implicit_gripper_physx_velocity_limit5_followers5_cuda_actuator_cpu_static_physx_v1`.
- The evaluator selects the already probed 5 rad/s / 200 Nm driver actuator
  configuration in EEF mode. After the first explicit reset and before the
  first controller apply, it requires the exact 13-DOF CPU PhysX velocity
  tensor, changes only passive follower indices 8 through 12 from
  `174.53292846679688` to `5.0`, makes one full-tensor
  `root_physx_view.set_dof_max_velocities` call, and requires exact CPU
  readback. Later resets read and validate without another write.
- Static evidence pins the robot USD hash/size, driver and five follower names
  and indices, PhysX mimic axes/references/gearing/frequency/damping, action and
  actuator ownership, CUDA actuator tensors, CPU static tensors, full setter
  input, and exact before/after values.
- Dynamic controller evidence samples all six gripper joints at every 120-Hz
  apply entry and after each completed policy step. It records measured
  velocity/acceleration maxima, a max-velocity causal diagnostic, and the last
  completed post-policy state. The contract explicitly records that the PhysX
  velocity setting is not a hard bound on measured passive velocity.
- The max diagnostic is bound to the aggregate float32 maximum and a valid
  interleaved sample index. A post-policy terminal sample has exact index
  `post_policy_step_samples * 9 - 1`, including a numerical-failure tail.
  Empty evidence requires zero maxima.
- An apply-entry gripper non-finite is counted before raising, so its sample
  cadence still equals `apply_calls` and the normal numerical-failure sidecar
  preserves the fault. A non-finite first observed only after `env.step`
  completed raises the distinct hard-stop `GripperRuntimePostStepError`; it is
  never mislabeled as an unexecuted-action failure tail.

The production profile is based on the approved diagnostic matrix: the
follower write reduced `left_inner_finger_knuckle_joint` from a measured
55.622322 rad/s peak to 5.000020 rad/s, while
`right_outer_knuckle_joint` still measured 7.174964 rad/s. This change does not
misrepresent the 5 rad/s setting as a measured hard bound. Promotion still
requires the final integrated canary.

## Outer-450 / internal-451 and terminal contract

- Environment config is changed before `gym.make` to an internal timeout of
  451 steps. The evaluator owns exactly 450 outer steps at 15 Hz with 120-Hz
  physics and decimation 8, preventing Isaac's final-step auto-reset.
- Each successful outer step must return both terminal flags false and advance
  episode/common/sim counters by exactly 1/1/8 and both external/wrist camera
  frame counters by exactly one. The complete before/after chain is recorded.
- Terminal profile is `ego_lap_eef_terminal_rollout_v1`; environment profile is
  `ego_lap_eef_outer450_internal451_no_autoreset_v1`; state profile is
  `isaaclab_single_env_episode_sim_common_camera_counters_v1`.
- A completed rollout requires 450 actions, 450 completed steps, last index
  449, 3600 sim/apply ticks, and 450 episode/common/camera increments.
- A numerical failure retains the attempted policy action but has one fewer
  completed outer transition. The terminal snapshot is captured after the
  failed apply: episode/common/camera counters stay at the completed count,
  while the sim counter has a strict one-to-eight-tick tail. Sidecar schema 3
  binds that exact sim delta to `safety.counters.apply_calls` and existing
  abort step/substep evidence. Stale last-success, zero-tail, over-eight-tail,
  sim/apply mismatch, and completed-counter advancement mutations reject.

## Trace, transaction, and aggregate schemas

- Trace profile is `ego_lap_eef_pose_runtime_trace_v2`, schema version 2.
  Reset carries the initial live environment snapshot; no extra rollout-start
  event is emitted.
- For execute horizon 8, a completed rollout has exactly 959 records:
  reset 1, query 57, action 450, execution 450, episode_complete 1. Query
  placement is checked online against emitted actions; moving queries to the
  front rejects. Action/query/chunk identities, exact nested transition keys,
  and the full environment chain reject extra keys and broken links.
- Every query now independently revalidates the checkpoint/contract hashes,
  flow-or-AR sampler, normalization profile/dtype/formulas, frame and state-R6
  layout, state reconstruction, server/raw/base chunks, and anchored float32
  actions. Every emitted action is exactly bound to its query chunk; immutable
  query identity must not drift; the failure-event reason must equal the
  terminal result.
- Episode sidecar schema 3 has exact top-level keys
  `schema_version, transaction_state, episode_index, episode_result,
  artifact_identity, cadence_evidence, terminal_rollout, safety`.
  Trace identity includes its schema/profile, canonical result, and the same
  terminal object. Safety includes gripper static/dynamic evidence.
- Runtime contract schema 3 aggregates immutable sidecars. `ik_safety` carries
  the gripper static contract and all-six maxima; each runtime episode carries
  its dynamic gripper evidence and terminal object. Publication independently
  recomputes the completed count, contiguous episode indices, all counters,
  arm maxima, and gripper maxima from `episodes[]` and rejects drift.

## Validation

Host/fake gates completed after the final terminal-tail changes:

```text
PYTHONPATH=src .venv/bin/pytest -q \
  tests/test_eef_gripper_runtime.py \
  tests/test_eef_terminal_contract.py \
  tests/test_eval_artifacts.py \
  tests/test_eef_runtime_contract.py \
  tests/test_lap_eef_pose_client.py
=> 121 passed, 22 subtests passed

PYTHONPATH=.:src .venv/bin/pytest -q tests \
  --ignore=tests/test_robust_differential_ik.py \
  --ignore=tests/test_smoke_eef_pose_gripper_impulse_diagnostic.py
=> 303 passed, 2 skipped, 22 subtests passed

PYTHONPATH=.:src /home/lzha/code/.venvs/dextrah-isaaclab/bin/python \
  -m pytest -q tests/test_smoke_eef_pose_gripper_impulse_diagnostic.py
=> 232 passed
```

Ruff on every changed Python file, Python byte compilation, and
`git diff --check` pass. The production installer unit uses fakes to prove one
setter call and read-only later-reset validation; other host tests use fake
states/tensors. They do not prove real `Camera.frame`, real CUDA actuator
tensors, or the real CPU PhysX setter.

## Required pending real-Isaac gate

No redundant one-step smoke is added. The existing diagnostic script uses its
own evidence path and cannot validate this production integration. The first
one-rollout official LAP `scripts/eval.py` canary is the authoritative real
Isaac gate because it is the only exact path through config, reset, production
install, all 450 steps, trace, terminal, sidecar, runtime aggregate, and video.
Promotion remains pending until that canary proves:

- one exact CPU follower-only write and persistent readback;
- 3600 controller applies and 450 exact counter/camera transitions;
- zero unexpected termination/truncation or hidden reset;
- complete all-six velocity/acceleration evidence with no drops/non-finites;
- exact 959-record trace, schema-3 sidecar/runtime contract, canonical CSV row,
  and complete 450-frame video.
