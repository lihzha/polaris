# PolaRiS current-velocity abort evidence — 2026-07-03

- Agent: `gripper_diag_impl`
- Branch: `codex/eef-current-velocity-abort-evidence-v5-20260703`
- Worktree:
  `/home/lzha/code/PolaRiS-worktrees/current-velocity-abort-evidence-v5-20260703`
- Exact base: `310d3f31d88e34d67f92857b7b0fcc34b3e22d51`
- Frozen producer branch and canonical checkouts are not modified.

## Failed production canary 1098256

- L40S job `1098256`, official LAP-3B, `DROID-FoodBussing`, official initial
  condition 0, flow/execute-8, aborted on attempted outer action 117.
- Final trace evidence is structurally correct: 118 actions, 117 completed
  transitions, one execution failure, 940 simulation/apply ticks, and a
  three-zero-indexed failed physics substep (`(940 - 1) % 8 == 3`).
- The policy changed the gripper from open to closed at action 115. The new
  production follower limit delayed the known coupled arm transient but did
  not eliminate it: the arm guard raised
  `current_joint_velocity_limit_abort` at action 117.
- `eef_episode_safety_report()` obtained the controller report and then called
  the live validator, whose unconditional measured-velocity maximum assertion
  rejected the evidence that caused the guard. Consequently no episode safety
  sidecar, CSV row, or post-episode runtime aggregate was published.
- The generic guard diagnostic did not record signed arm velocity, configured
  limit, or excess. Those exact values cannot be reconstructed after process
  exit.
- A local read-only copy of the 18 attempt/log files is under
  `cluster_results/l401/job1098256`, with a mode-0444 SHA-256 manifest at
  `cluster_results/l401/job1098256.sha256`. The remote attempt is read-only for
  this task.

## Approved evidence-only repair

- Add one nullable, closed current-joint-velocity abort object to episode
  safety and runtime-episode evidence.
- Record ordered joint names, signed measured velocity, configured PhysX
  velocity limit, exact tolerance, clamped per-joint excess, and exceeded mask
  at the aborting apply entry.
- Bind the object bidirectionally to the existing guard diagnostic, abort
  counter, per-joint maxima, episode result, and failed policy/substep cadence.
- The raised `DifferentialIKInvariantError` carries a SHA-256 of the exact
  object using canonical JSON (`sort_keys=True`, compact separators,
  `ensure_ascii=False`, `allow_nan=False`, UTF-8). Episode and runtime
  validators reproduce that encoding and require the exact caught-exception
  reason, so changing the sign of any velocity cannot preserve validity.
- Use one float32 predicate everywhere: `abs(dq) > float32(limit + tolerance)`.
  Excess remains a separate nonnegative `float32(abs(dq) - limit)` report.
  This avoids subtraction-rounding mask drift at the exact threshold.
- After current state finiteness and maxima capture, dispatch the live velocity
  guard before desired-pose, quaternion, and position guards. This defines
  deterministic simultaneous-failure precedence so an over-limit maximum can
  never be retained without its matching object.
- Permit an over-limit measured maximum only for this exact terminal numerical
  failure. Completed rollouts retain the strict velocity-bound requirement.
- Validate the exact report returned by `eef_episode_safety_report()` instead
  of fetching a second report.
- Bump only the episode safety sidecar and runtime-contract outer schemas from
  v3 to v4. Terminal rollout stays v1 and policy trace stays v2. Load and
  recovery reject old v3 sidecars.
- Update every independent controller/boundary smoke closed field set. Initial,
  adversarial, ordinary, and completed promotion reports require the nullable
  field to be exactly null. Failure-trace validation instead binds the non-null
  object to the independently captured cached/direct-PhysX arm velocity.
- Do not change policy actions, target slew, joint/effort limits, gains,
  armature, solver configuration, or gripper physics in this branch.

The standalone boundary/controller/gripper diagnostic raw envelopes remain
schema v1. They are not resume/runtime interchange formats: each is
source-commit, exact-helper-size/SHA, runner-SHA, and finalizer-SHA pinned. The
closed nested validator and its bootstrap identity were updated atomically.
The transactional episode sidecar/runtime formats are the externally resumed
formats and therefore received the explicit v4 bump.

## Physical interpretation and next gate

The pinned Isaac Lab actuator contract describes `velocity_limit_sim` as a
PhysX braking constraint: when a joint is already faster, the solver attempts
to brake it, and tight limits can fail to converge without gain/damping
tuning. It is not a post-step sample clamp. Prior direct-PhysX/cache-equal
replays and canary 1098256 therefore establish a real coupled-dynamics failure,
not a reporting false positive. Evidence preservation is not physical
promotion.

Ranked physical candidates, none implemented here:

1. Smallest next experiment: EEF-only per-physics-substep slew of the driven
   finger target toward the unchanged binary `0`/`pi/4` endpoint, bounded by
   the live 5 rad/s driver limit while retaining follower caps. This preserves
   eventual action semantics and models physical gripper latency, but still
   requires exact fixture A/B because terminal contact may remain impulsive.
2. Gripper drive damping/effort retuning only after live inertia/force evidence;
   it can alter grasp strength and is therefore not the first intervention.
3. Contact, depenetration, or solver tuning is broader and changes task
   dynamics.

Already falsified/rejected: one-policy-step gripper delay, wrist energy brake,
scalar arm-target slew, extra velocity iterations, arm-limit relaxation, and
guard removal.

The proposed real gate is one reviewed L40S job on the unchanged public
outer450/internal451 path, one official LAP-3B FoodBussing condition-0 rollout,
`EVAL_TIME=00:10:00`, and `Requeue=0`. There is deliberately no diagnostic
action-cap code. Acceptance requires reproducing the durable velocity guard on
attempted action 117 / physics substep 3 with 118 actions, 117 completed outer
steps, 940 simulation/apply ticks, and a complete trace/video/v4-sidecar/CSV/v4-
runtime chain whose exact signed velocity object matches the exception digest.
If it continues instead of reproducing, the reproduction gate fails and its
artifacts are inspected; it is not silently promoted. This gate will not launch
until frozen producer and consumer commits receive independent compatibility
review.

## Local validation

- `tests/test_eef_runtime_contract.py`: 70 passed. Includes exact v4 acceptance,
  v3 loader/reconciler rejection, returned-report/no-refetch, all closed-object
  mutations, all seven individual sign flips, unrelated/digest-drifted reasons,
  fixed consumer digest oracle
  `f7c69e1fa8ae3a36cdd17ad511de65ba1795f2ee3391d3ddb181d2369602d86d`,
  and both float32 velocity-limit families at exact-threshold/nextafter.
- Lightweight host-stub producer/order selection: 9 passed, 75 deselected.
  This checks exact capture/reset/report, both threshold families, pre-DLS/
  pre-PhysX ordering, simultaneous-invariant precedence, and that ordinary
  action reset cannot erase terminal evidence. It is semantic host coverage,
  not a substitute for the mandatory reviewed real-Isaac gate.
- Boundary/controller/finalizer affected host suites: 125 passed.
- Full gripper impulse diagnostic host suite: 232 passed (one unrelated
  `pynvml` deprecation warning).
- Wide non-Isaac suite: 340 passed, 2 skipped, 22 subtests passed. Ruff check,
  Ruff format check, Python byte compilation, and `git diff --check` pass.
- Refreshed content chain after the closed boundary-helper edit: boundary
  helper 112869 bytes / SHA-256
  `edc62b33f6e5edb7737e121fb60cf801cb9964cbd62a92ccf26d292cc3937209`;
  gripper diagnostic SHA-256
  `1edcf11245433954836d8ebae6acf02b6380082b1a55b1c25b2d454ed5fcb8d8`;
  gripper finalizer SHA-256
  `5ec5afc8e2627bcceb4cc2c4e9e8897cb74a295f1412689a8103e3570a6851d9`.
- Independent frozen-diff results are recorded before commit.

## Independent frozen review

- Final review: GO; no outstanding P0, P1, or P2 findings.
- Reviewed base:
  `310d3f31d88e34d67f92857b7b0fcc34b3e22d51`.
- Reviewed tracked source/test diff SHA-256:
  `42072521b0300980e75dbf206c77c2250df60c41a9465fd1b402acab2663b06c`.
- Independent gates reproduced 70 runtime-contract tests, 9 focused host-stub
  producer/order tests (75 deselected), 572 wider host tests with 2 skips and
  22 subtests, Ruff check/format, in-memory compilation of all 11 changed
  Python files, `git diff --check`, and stable source/test digest.
- Real-Isaac promotion remains blocked on the reviewed bounded diagnostic gate;
  no job was launched from this branch.
