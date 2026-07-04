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
