# PolaRiS EEF Current-Velocity Recovery V5 Worklog

Agent: `polaris_v5_recovery_impl`

Branch: `codex/eef-contact-velocity-recovery-v5-20260704`

Exact base: `7fc74d648328432a7f9f06d13c0e82a03f73a0c1`

Date: 2026-07-04

## Goal

Add an opt-in v5 PolaRiS EEF controller profile that treats the small measured
PhysX velocity residual seen in the official LAP-3B canary as DLS-eligible,
while recovering the large reasoning-checkpoint residual through a bounded,
transactional current-position hold. Preserve the v4 controller path and its
setter order byte-for-byte when the v5 flag is disabled.

Public controller profile:

`arm_slew_0p95_gripper_rate0p25_fixed_anchor86_release_ramp16_velocity_recovery8_clean2_mimic100_damping1p2_v5`

Safety profile:

`panda_velocity_physxlimit_solveriter1_residual_recovery8_clean2_v5`

## Implemented contract

- The recovery envelope is exactly
  `float32(L + float32(L * float32(1e-4)))`. The exact envelope remains
  DLS-eligible; the next float32 value enters recovery.
- Every velocity residual above the live PhysX limit is counted. The observed
  official LAP value `2.6102163791656494` is residual evidence but remains on
  the normal DLS path. The reasoning value `11.743` enters recovery.
- Recovery skips DLS and owns one all-arm transaction in the order position,
  exact-zero velocity, exact-zero effort, readback, failure-trace staging, then
  state commit. Normal/inactive v5 DLS retains the v4 velocity/position setters,
  performs no effort write, and records the live effort surface in its trace.
- The hold lasts at most eight physics substeps. The ninth consecutive recovery
  substep is a named terminal abort. Two consecutive in-envelope samples hand
  off through the existing inclusive 16-substep release ramp; clean sample 2 is
  both the final active hold apply and release-ramp index 0.
- A re-exceedance during the recovery ramp returns to HOLD without opening a
  second event. `recovered_events` increments only after ramp index 15 commits.
- While recovery owns the target, the lower close-interlock and lower release
  ramp are frozen without consuming counters. The recovery ramp targets the
  lower controller's current owner: fixed close anchor, frozen underlying ramp,
  or nominal target. The lower lifecycle resumes on the next inactive apply.
- Current installed hard-envelope violation, float32 `q + dq * physics_dt`
  predicted hard crossing, sustained recovery, and setter/readback/trace
  transaction failure close the event, terminalize the recovery state, emit one
  named diagnostic, and bind the exact exception to canonical JSON SHA-256.
  The predicted crossing guard applies throughout v5, even below the recovery
  envelope. An unrelated existing terminal invariant may truthfully interrupt
  an open event; a normal horizon may also end with one open event before reset.
- The v5 report is a closed `current_joint_velocity_recovery` sibling in both
  safety and controller evidence. Event indices/cadence, exact float32 snapshot
  arithmetic, committed readbacks, end-reason counters, nested/top-level maxima,
  and terminal result digests are independently revalidated.
- V5 episode sidecars and runtime contracts use outer schema version 7. Legacy
  profiles retain schema version 6 and do not gain the nested field.

## Important audit correction

The first runtime-contract implementation still called the legacy v4 rule that
requires a terminal `current_joint_velocity_abort` whenever the per-joint
maximum exceeds `limit + 1e-5`. That would reject every valid v5 recovery after
the controller had safely handled it. The legacy inference is now disabled only
for the v5 safety profile; the legacy abort object remains required-null in v5,
the nested v5 maxima must equal the independently accumulated top-level maxima,
and v4 retains its original fail-closed behavior.

The exact `config.py` source pin used by the model-free canary replay was updated
for the intentional new profile. `scripts/eval.py` was left unchanged so its
historical production source identity remains intact.

## Host validation

The PolaRiS worktree did not contain a populated standalone test environment.
Host-safe validation therefore used the already validated Ego-LAP environment
with this worktree's `src` first on `PYTHONPATH`; no dependency installation or
cluster mutation was performed.

Commands included:

```bash
PYTHONPATH=$PWD/src /home/lzha/code/ego-lap/.venv/bin/python -m pytest -q \
  tests/test_eef_current_velocity_recovery.py \
  tests/test_eef_controller_repair.py \
  tests/test_eef_controller_profile.py \
  tests/test_robust_gripper_target_slew_host_stub.py \
  tests/test_eef_runtime_contract.py \
  tests/test_eval_mode_contract.py \
  tests/test_eval_artifacts.py \
  tests/test_eef_terminal_contract.py \
  tests/test_smoke_eef_pose_canary_controller_candidate.py \
  tests/test_validate_eef_pose_canary_controller_candidate.py \
  tests/test_finalize_eef_pose_canary_controller_candidate.py \
  tests/test_finalize_eef_pose_smoke.py \
  tests/test_finalize_eef_pose_canary_trace_replay.py \
  tests/test_smoke_eef_pose_canary_trace_replay.py
ruff check <changed Python files>
ruff format --check <changed Python files>
python -m py_compile <changed Python files>
git diff --check
```

Focused producer/runtime/artifact validation passed `330` tests. The final
recovery/controller/runtime subset passed `209` tests after the maxima and
horizon-open-event corrections. Independent wide validation reported `892`
passes plus `30` subtests. Ruff, formatting, byte compilation, and diff checks
were clean; only the existing `pynvml` deprecation warning was emitted.

Coverage includes exact-envelope/nextafter boundaries for both Panda velocity
limit families, official/reasoning observed fixtures, unconditional predicted
crossing below the envelope, abort-on-9, clean-2/ramp-16/re-exceed semantics,
lower lifecycle suspension, v4 setter preservation, v5 effort setter/readback
failure rollback, schema-7 sidecar/runtime propagation, exact event SHA/result
binding, unrelated digest-bearing interruption, normal-horizon open state, and
legacy-v4 maxima rejection.

## Cluster status and next gate

No Slurm or Isaac Sim job was launched from this implementation branch. The
next gate is review of the exact producer/consumer source pins, followed by a
pinned L40S target-runtime smoke and parallel official LAP-3B/reasoning canaries.
Those jobs must inspect logs, sidecars, runtime contracts, traces, and videos
before any promotion.

## P1/P2 pre-launch review closure

The follow-up revision is an additive commit above feature commit
`048bfd20ac1ba69051a8c628349a7407c01335c8`; that commit and its parent were not
rewritten. The review found and closed five launch-blocking integration gaps:

- Nested recovery evidence is now schema 2 and binds the exact float32 physics
  timestep, the installed canonical PhysX hard-limit matrix, and its little-
  endian float32 SHA-256. For both the start and last snapshot, the consumer
  independently recomputes `float32(q + float32(dq * dt))`, signed minimum hard-
  limit clearance, velocity residual/ratio, and start/end-reason predicates.
  Mutation tests cover timestep, hard-limit bytes/digest, and every recomputed
  state vector on both snapshots. Impossible historical fixtures were replaced
  with physically consistent Panda states.
- Clean sample 2 now bypasses the Jacobian and DLS path while transactionally
  applying current position as recovery release-ramp index 0. An isolated host
  integration test uses Jacobian and controller failure sentinels to prove that
  neither path is entered.
- A lower arm release ramp suspended by v5 may carry exactly one stale target
  into the first inactive apply only when the last recovery event completed at
  the immediately preceding apply. That apply resumes the exact frozen lower
  ramp index and restores strict latest-target validation. The overlap test
  freezes index 5 for 16 recovery-owned applies, resumes index 5, and validates
  the resulting index-6 state strictly.
- Every recovery-owned apply observes the frozen lower endpoint-transition
  counter without consuming it. One deferred flip remains frozen and is
  processed on the first inactive resume. A second flip closes the active event
  with `lower_endpoint_transition_overflow_abort`, increments the nested
  `lower_endpoint_transition_aborts` counter, emits
  `measured_velocity_recovery_lower_endpoint_transition_abort`, and raises a
  canonical-event-digest-bound `DifferentialIKInvariantError` before DLS or a
  PhysX target setter. Sidecar validation now requires the uncommitted finger
  endpoint count to lead the frozen arm count by exactly two for this terminal
  reason; all other failure/completion count rules remain unchanged.
- Recovery-disabled legacy profiles again stage lower endpoint validation before
  frame/Jacobian/DLS work. Runtime sentinels cover both release-ramp-disabled
  and v4-style release-ramp-enabled paths, proving malformed endpoint evidence
  wins over a later DLS failure.

The previously outer-only current-hard abort count is also represented by the
closed nested `current_hard_limit_aborts` counter and bound one-for-one to a
`current_hard_limit_abort` event.

Post-review validation used:

```bash
PYTHONPATH=$PWD/src /home/lzha/code/ego-lap/.venv/bin/python -m pytest -q \
  tests/test_eef_current_velocity_recovery.py \
  tests/test_eef_runtime_contract.py \
  tests/test_eef_controller_repair.py \
  tests/test_eef_controller_profile.py \
  tests/test_robust_gripper_target_slew_host_stub.py
PYTHONPATH=$PWD/src /home/lzha/code/ego-lap/.venv/bin/python -m pytest -q \
  tests --ignore=tests/test_robust_differential_ik.py
```

The focused suite passed `217` tests. The broad host-safe suite passed `906`
tests plus `30` subtests. The only emitted warning was the existing Torch
`pynvml` deprecation warning. No cluster job was launched during this review
closure.

## Final-review chronology and stale-resume closure

Independent review of pushed revision
`ebd5f9483d4bb098d3944c0faec92678ef2c7609` reproduced two additional
pre-launch fail-open cases. First, the nested validator accepted a fabricated
same-apply `sustained_recovery_abort` with only one committed active apply; the
same missing lifecycle proof also allowed an impossibly short clean completion
and illegal start/end pairings. Second, two unconsumed gripper endpoint changes
arriving on the one post-completion stale-target resume apply bypassed the
named lower-overflow guard because the lower controller was no longer marked
suspended, then fell into a plain missed-transition `ValueError`.

The additive final-review repair upgrades only the nested recovery evidence to
schema 3; legacy outer sidecars/runtime contracts remain schema 6 and v5 outer
artifacts remain schema 7. Each event now records closed lifecycle evidence:

- exact deferred lower endpoint-transition count when terminal;
- lower overflow context (`active_recovery` or `post_recovery_resume`);
- the exact recovery-completion apply index for a completed ramp.

The independent validator now applies one explicit legal start/end matrix.
Open, clean, and sustained events require a measured-velocity start. Immediate
current-hard, predicted-hard, and target-transaction owners may end only in
their matching reason on the same apply; measured starts may reach only the
runtime-reachable later/immediate terminal combinations. A measured event that
survives its start apply must carry a fully committed start transaction. Clean
completion requires `end-start >= 17`, at least three committed active applies,
and sixteen recovery-ramp target applies per recovered event. Sustained abort
requires `end-start >= 8`, at least eight committed active applies per event,
and exact global consecutive maximum 8. The global active count must cover the
sum of all per-event minima and remain consistent with its maximum.

On the immediately following stale-target resume apply, the overflow guard is
now active before Jacobian/DLS or any target setter. Zero or one deferred change
retains the prior resume behavior. More than one reopens only the terminal
classification of the just-completed measured event, preserves its truthful
recovered-ramp count and completion apply, records the exact deferred count and
post-recovery context, then emits the existing named diagnostic and canonical-
event-digest-bound `DifferentialIKInvariantError`. Sidecar validation binds the
recorded deferred count to the exact frozen-arm versus live-finger endpoint-
count difference. Active-recovery and post-recovery variants both traverse the
full result, sidecar, aggregate, and runtime-contract path; count drift is
rejected.

Final-review validation commands included:

```bash
PYTHONPATH=$PWD/src /home/lzha/code/ego-lap/.venv/bin/python -m pytest -q \
  tests/test_eef_current_velocity_recovery.py \
  tests/test_eef_runtime_contract.py \
  tests/test_eef_controller_repair.py \
  tests/test_eef_controller_profile.py \
  tests/test_robust_gripper_target_slew_host_stub.py
PYTHONPATH=$PWD/src /home/lzha/code/ego-lap/.venv/bin/python -m pytest -q \
  tests --ignore=tests/test_robust_differential_ik.py
```

The final focused suite passed `237` tests. The broad host-safe suite passed
`926` tests plus `30` subtests. Only the existing Torch `pynvml` deprecation
warning was emitted. No local GPU, simulator, Slurm, evaluation, or checkpoint
job was launched, and no canonical worktree was modified.

## Terminal-guard/deferred-endpoint collision closure

Final review of pushed revision
`99fb191f4fe0238f8f9950cfd8887d3a2ba35c43` found one remaining artifact
finalization mismatch. During active recovery or the first post-recovery resume
apply, a current-hard or predicted-hard guard correctly precedes the deferred
lower endpoint-overflow guard. When both conditions occurred together, the
hard-limit guard therefore won, but its terminal event did not record the
unconsumed lower endpoint count. The sidecar then applied the ordinary
numerical-failure allowance and rejected the truthful two-count finger/arm
difference.

The additive repair preserves safety precedence and schema 3 while making the
collision evidence guard-agnostic:

- Before any current-hard, predicted-hard, lower-overflow, DLS, or setter path,
  the producer observes the exact deferred endpoint count. Counts zero or one
  retain the prior behavior. A count greater than one is attached to whichever
  current-hard or predicted-hard event wins, together with
  `active_recovery` or `post_recovery_resume` context.
- Post-recovery hard-limit events are separate same-apply terminal events and
  bind the immediately preceding clean completion exactly at
  `completion == start - 1`. Active collisions retain no completion index.
  The three existing schema-3 fields are included in the canonical event JSON
  before its exception SHA-256 is computed.
- The nested validator accepts collision metadata only on lower-overflow,
  current-hard, or predicted-hard terminal events, enforces the exact active or
  post chronology, and preserves ordinary hard-limit events without metadata.
- Sidecar validation now binds the exact recorded deferred count to
  `finger_endpoint_changes - arm_endpoint_changes` for all three terminal
  collision reasons. Lower-only active/post behavior and the legacy one-count
  numerical-failure allowance are unchanged.
- Producer and consumer tests cover active/post x current/predicted through
  guard ordering, event digest generation, nested validation, cadence/result
  binding, atomic sidecar publication, aggregate reconstruction, and runtime
  publication. Count, context, completion, and endpoint-difference mutations
  fail closed.

The first focused run exposed a compatibility regression in the no-overflow
path: zero/one deferred transitions were unnecessarily required to carry a
terminal snapshot apply index. The cadence check is now delayed until an actual
count-greater-than-one collision, preserving the prior zero/one resume path.

Final host validation from the dedicated PolaRiS worktree used:

```bash
PYTHONPATH=$PWD/src /home/lzha/code/ego-lap/.venv/bin/python -m pytest -q \
  tests/test_eef_current_velocity_recovery.py \
  tests/test_eef_runtime_contract.py \
  tests/test_eef_controller_repair.py \
  tests/test_eef_controller_profile.py \
  tests/test_robust_gripper_target_slew_host_stub.py
PYTHONPATH=$PWD/src /home/lzha/code/ego-lap/.venv/bin/python -m pytest -q \
  tests --ignore=tests/test_robust_differential_ik.py
ruff check <six changed Python files>
ruff format --check <six changed Python files>
python -m py_compile <six changed Python files>
git diff --check
```

The focused suite passed `245` tests. The broad host-safe suite passed `934`
tests plus `30` subtests. Static checks were clean; the only warning was the
existing Torch `pynvml` deprecation warning. This work remained local and
model-free: no GPU process, simulator, Slurm job, evaluation, watcher, or
checkpoint job was launched. The canonical Ego-LAP primary checkout remained
on `main`, and registry doctor passed before development.
