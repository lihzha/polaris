# PolaRiS EEF gripper target slew — 2026-07-03

- Agent: `polaris_gripper_slew_candidate_20260703`
- Branch: `codex/eef-gripper-target-slew-v1-20260703`
- Worktree:
  `/home/lzha/code/PolaRiS-worktrees/eef-gripper-target-slew-v1-20260703`
- Exact base: `95f57bb043aa7385f396d6c82c30c0f5738c6ae7`
- Scope: code and host validation only. No local simulator, GPU, Slurm,
  registry, deployment, or shared-checkout mutation is authorized in this
  task.

## Goal and hypothesis

The official LAP-3B FoodBussing condition-0 canary reached a measured
`panda_joint7` velocity of `+2.76075697 rad/s` at policy action 117, physics
substep 3, after the model changed the gripper from open to closed at action
115. The smallest next physical candidate is an EEF-only driven-finger
position-target slew toward the unchanged binary `0`/`pi/4` endpoint. The
per-physics-substep cap must be derived from the exact live configured gripper
driver limit of `5 rad/s` at `120 Hz`; the five passive follower caps remain
unchanged.

This candidate may reduce the coupled arm transient by removing the immediate
`pi/4` driver-target jump. It does not establish task success or physical
promotion without a separately reviewed real-Isaac controller smoke and a
bounded checkpoint canary.

## Implemented contract

- Select the new action term only in `EgoLapEefPoseActionCfg`; leave the
  default joint-position/native `ActionCfg`, generic `EefPoseActionCfg`, and
  client paths unchanged.
- Preserve the existing closed-positive `>=0.5` binary endpoint semantics and
  exact open/closed endpoints.
- Anchor the first post-reset applied target to the live driven-finger joint
  position, then move at most `float32(5 * (1/120))` radians per physics apply.
- Reject nonfinite actions/state, tensor device/dtype drift, action-term
  profile drift, endpoint drift, external target drift, and live configured
  driver-limit drift before writing a new target.
- Bind a closed, versioned static profile and per-episode counters/maxima to the
  existing gripper runtime evidence. Bump resumable sidecar/runtime schemas for
  the changed closed contract.
- Test open/close, threshold equality, exact cap, no overshoot, repeated
  commands, reset, device/dtype, schema/identity rejection, cadence, and
  native-path noninterference.

## Status

- Worktree created cleanly from the exact requested base.
- Canonical Ego-LAP remained the primary `main` worktree; registry doctor
  passed before development.
- Implemented `EefBinaryJointPositionTargetSlewAction` as an EEF-only mixin
  over the unchanged binary action. It derives the exact float32 target cap
  from the live implicit-actuator `velocity_limit_sim` tensor and the pinned
  120 Hz physics cadence on every apply.
- The first apply after each action reset anchors from live `joint_pos`, not a
  potentially stale target. It rejects an anchor outside the endpoint range
  (with the named float32 tolerance) before any write. Subsequent writes
  require exact readback of the previously written target. Endpoint, profile,
  ownership, device, dtype, live-limit, finiteness, monotonic-error, and
  no-overshoot checks fail closed.
- Apply state is transactional: initialization, anchor/current target,
  live-limit count, apply/classification counters, and maxima are committed
  only after setter readback succeeds. Pre-write and readback failures retain
  a self-validating report for the last completed apply.
- Added closed static and per-reset dynamic target-slew evidence to the
  existing all-six gripper report, and cross-bound arm/finger action-manager
  ordering in episode cadence validation. Resumable sidecar and runtime
  schemas are now version 5.
- Kept the native `ActionCfg` and generic `EefPoseActionCfg` action classes
  unchanged. An AST snapshot from the exact base guards the native action
  classes; only Ego-LAP evaluation and the dedicated controller smoke select
  the candidate.
- Extended the standalone smoke/finalizer contract. Case zero requests close
  for 45 policy steps and must independently prove 18 cap-limited applies and
  exact endpoint arrival on apply 19. All other ordinary and adversarial cases
  hold the exact open endpoint. The finalizer independently validates the
  complete gripper static/dynamic schemas and physical evidence.

## Host validation

All commands used `CUDA_VISIBLE_DEVICES=`; no GPU, simulator, Slurm, registry,
or deployment action was launched.

- `pytest -q tests/test_eef_gripper_target_slew.py
  tests/test_eef_gripper_runtime.py
  tests/test_robust_gripper_target_slew_host_stub.py`: `60 passed`.
- `pytest -q tests/test_eef_runtime_contract.py
  tests/test_finalize_eef_pose_smoke.py`: `74 passed`.
- `pytest -q tests --ignore=tests/test_robust_differential_ik.py`:
  `619 passed, 30 subtests passed` (one environment `pynvml` deprecation
  warning).
- Ruff check, Ruff format check, in-memory compilation of all 13 changed
  Python files, and `git diff --check`: passed.
- The full real-Isaac robust-action test module cannot be collected in this
  host shell because the installed Isaac environment lacks the Omniverse
  runtime (`omni.log`). An isolated host-stub subprocess exercises the actual
  robust module's target-slew binding/report path instead. The required next
  gate remains a separately authorized L40S real-Isaac controller smoke.

## Independent audit

The first frozen-diff audit reproduced the host suites and checked the exact
Isaac Lab 2.3.0 wheel API. It found one P1 closure defect: first-apply state and
the live-limit counter were mutated before setter/readback, and an out-of-range
anchor could therefore produce a write followed by a report that rejected its
own evidence. The implementation now stages the transition locally, validates
both anchor bounds before writing, and commits all state only after exact
readback. Regression tests cover below-open and above-closed anchors, first and
initialized setter/readback failures, other pre-write failures, and rapid
mid-slew endpoint reversal. The first repair re-audit found a remaining P1 at
the tolerance boundary: float32 transition arithmetic could accept an anchor
and produce an endpoint-error maximum one ULP above the Python-double
validator bound. Precheck, production dynamic validation, and the independent
finalizer now share explicit float32-expanded minimum/maximum bounds. Tests
prove both exact representable bounds write and self-validate, while each
immediately adjacent out-of-bound float32 value fails with zero committed
evidence. The final fresh audit on staged review hash
`260a24129bc61bbc138211bf013e33665adb91a8c3b1264268f2d42b20c303dc`
independently reproduced both boundary directions and all three test suites,
verified exact Isaac Lab 2.3.0 action APIs and ordering, and returned GO with
no remaining P0/P1/P2 findings. Its only residual is the intentionally pending
real Isaac/Omniverse/CUDA/PhysX promotion gate.

## Physical promotion status

Not promoted. The code only establishes a bounded candidate and durable gate.
A real-Isaac controller smoke must pass and its artifacts must be independently
finalized before any checkpoint canary. A canary must then show that the
original `panda_joint7 +2.76075697 rad/s` action-117/substep-3 abort is removed
without a new arm or all-six-gripper safety violation. Task success is not
claimed.

## Git handoff

- Implementation commit:
  `dc1426d4c711c657e2e213dda9ab752fc6d83364`
- Independently audited staged review hash:
  `260a24129bc61bbc138211bf013e33665adb91a8c3b1264268f2d42b20c303dc`
- No simulator, GPU, Slurm, evaluation, or other persistent job was launched
  from this worktree. No cleanup is pending.
