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
115. The initial physical candidate was an EEF-only driven-finger
position-target slew toward the unchanged binary `0`/`pi/4` endpoint at the
full live driver rate. Job `1098286` subsequently disproved that command rate;
the corrected contract separates a 2.5 rad/s EEF target rate from the unchanged
physical 5 rad/s driver/follower limits, as recorded below.

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
  position, then move at most `float32(2.5 * float32(1/120))` radians per
  physics apply. Bind source, `0.5` factor, and 2.5 rad/s rate separately from
  the live physical 5 rad/s limit.
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
  over the unchanged binary action. It validates the exact live implicit-
  actuator `velocity_limit_sim` tensor on every apply, derives the versioned
  2.5 rad/s EEF command rate as its float32 `0.5` factor, and binds that rate
  to the pinned 120 Hz physics cadence.
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
  for 45 policy steps and must independently prove 37 cap-limited applies and
  exact endpoint arrival on apply 38. All other ordinary pose cases hold the
  exact open endpoint. The finalizer independently validates the complete
  gripper static/dynamic schemas and physical evidence; the delayed-close and
  velocity-headroom additions are detailed below.

## Original pre-job host validation (historical)

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

## Failed physical gate and corrected candidate

The original 5 rad/s target-slew candidate was deployed and exercised by the
separately reviewed controller-smoke wrapper. Slurm job `1098286` failed
closed (`FAILED 1:0`, `srun` 1) in the first close-hold case at policy step 0,
physics substep 4. The signed live `panda_joint5` velocity was
`2.610020160675049 rad/s`; the float32 physical limit was
`2.609999895095825 rad/s`, for the producer-recorded limit excess
`2.0265579223632812e-5 rad/s`. The float32 guard threshold after its frozen
`1e-5` tolerance was `2.6100099086761475 rad/s`, so the threshold overage was
still `1.0251998901367188e-5 rad/s`. The current-abort evidence digest was
`60f6ecaea1046e94bbb8b0d4995e130c7a390c242215f96478dfe7b5ea28a46a`.

Preserved job evidence is exact:

- raw failure JSON SHA-256:
  `dc17ac7e91f59308b9ea4e60ded20b10fe60ac6069d5625edf3ce11a1688457b`;
- complete log SHA-256:
  `89932dd90d9d35716c9fe85026976543e665e8a62e2d02168e4b5421ba3c66e5`;
- immutable saved wrapper SHA-256:
  `7a07e760061863d14b98a0488993c1398e6d75696002ed7f4f9f44dcc7267afb`.

The failure is not a reset-only or stale-state artifact. The arm action term is
first in action-manager order; four successful finger writes occurred before
the fifth arm entry rejected the live velocity, so the old target was already
approximately `0.16666666 rad`. The old command cap equaled the physical limit
with no tracking/acceleration margin:
`float32(5 * float32(1/120)) = 0.0416666679084301 rad`. With the probed live
driver stiffness `5729.578125 N*m/rad`, that first-step position error implies
an unconstrained proportional demand of `238.732421875 N*m`, above the exact
`200 N*m` effort cap. This explains the saturated impulse and measured arm
coupling; relaxing the arm guard would hide it rather than fix it.

The corrected EEF-only contract therefore keeps the independently validated
physical driver and five follower velocity limits at exactly `5 rad/s`, but
introduces a separate versioned target-rate source/factor/rate identity:
factor `0.5`, rate `2.5 rad/s`. Its exact float32 target step is
`0.02083333395421505 rad`, whose corresponding proportional demand is
`119.3662109375 N*m`. Closing from exact zero reaches float32 `pi/4` on apply
38: exactly 37 writes are slew-limited and apply 38 reaches the unchanged
endpoint.

The corrected smoke/evidence surface now additionally:

- preserves all 13 ordinary pose cases;
- runs a production-boundary episode that holds the same reset-anchored arm
  pose open for 115 policy steps and closes for five, requiring exact
  process/apply cadence `120/960`, one endpoint change, 37 limited writes,
  exact endpoint arrival, and zero arm abort;
- moves the one-step adversarial check to episode 14;
- records arm velocity maxima/limits/ratios for both immediate-close and
  delayed-close episodes and rejects promotion unless
  `max_j |dq_j| / limit_j <= 0.95` for each;
- captures, before environment close on failure, the live active safety and
  current-abort report, signed seven-joint arm velocity, target-slew state, and
  all-six gripper position/velocity/acceleration/target vectors. A secondary
  capture error is recorded without masking the original exception.

Physical relaunch is **pending**. No corrected real-Isaac smoke or checkpoint
canary has been launched from this implementation task; promotion and task
success remain unclaimed until the revised immutable wrapper passes review,
the controller smoke satisfies both headroom gates, and artifacts are fully
inspected and finalized.

## Corrected host validation

Validation used the existing Ego-LAP host environment with `PYTHONPATH=src:.`;
no GPU, simulator, Slurm, registry, deployment, or shared checkout was touched.

- Focused runtime/target-slew/finalizer/episode-contract suite:
  `137 passed`.
- Isolated robust-action production binding stub: `1 passed`.
- Full host suite excluding only the real-Isaac
  `tests/test_robust_differential_ik.py` module: `623 passed` (one unrelated
  `pynvml` deprecation warning).
- Ruff check and format check on all eight changed Python files, Python compile
  checks, and `git diff --check`: passed.
- A repository-wide Ruff invocation still reports ten pre-existing findings in
  unrelated splat-renderer/environment/client files; none is in this diff.

## Git handoff

- Original 5 rad/s candidate implementation commit:
  `dc1426d4c711c657e2e213dda9ab752fc6d83364`
- Independently audited staged review hash:
  `260a24129bc61bbc138211bf013e33665adb91a8c3b1264268f2d42b20c303dc`
- Corrected 2.5 rad/s implementation commit:
  `86ce5a9f4b154b13f51e10921c40178a0373a719`.
- Corrected staged diff SHA-256 before commit:
  `9ce615a7457e39f753d557a7f9ea4d4c620ab96095a73bc2331b3614423c7077`.
- No corrected simulator, GPU, Slurm, evaluation, or other persistent job was
  launched from this implementation task. Relaunch is pending; no task-owned
  process or local artifact cleanup remains.

## 2026-07-03 corrected L40S smoke and independent promotion audit

- The clean pushed runtime head
  `9098cac31713d5f9b3742de28f94402a92d0553f` was deployed detached and clean
  on l401. The frozen wrapper SHA-256 was
  `d756a0dabc78ec576c17b9f070c00b8d6814b91fcb767cae1965d30e80c07775`.
- Slurm job `1098288` completed `0:0` on `pool0-00005` in 304 seconds. All 13
  ordinary cases, the delayed close replay, and the adversarial case passed
  (`15/15`). Immediate and delayed close each had exactly 37 slew-limited
  writes and reached the endpoint on apply 38, with zero arm abort, fallback,
  nonfinite, dropped-diagnostic, or post-clamp violation counters.
- Maximum arm velocity/limit ratios were `0.028715338070707645` for immediate
  close and `0.02872080468466036` for delayed close, far below the `0.95`
  promotion threshold.
- Immutable artifact SHA-256 values: raw
  `553ba7301b91fb96a637a41c6acf7fb35f96de04b556fb91614c142753b04e20`;
  ready `03dfdf2f11eecfd14414e96d8b3c45d243d9a149d8b6668af728946cda7b3ac1`;
  attestation
  `ecf7765db3c02d71d7c8f39193f7f467a7a4a16675d38d47a0916645d158d647`;
  full log
  `9590035422bd554c200fb22152bb559221f6396492db27a4f859510e0dc2d7ea`.
- Independent post-run audit: **GO**, no P0/P1/P2. It reproduced the strict
  closed-schema validator and durable metadata byte-for-byte, checked the
  exact four-file mode-0444 result root, independently verified all cadence,
  target-slew, headroom, and adversarial invariants, and rehashed the full
  runtime image and relevant sources.
- This promotes the standalone controller smoke only. Checkpoint/task behavior
  remains pending an artifact-complete, video-inspected full-horizon canary.
