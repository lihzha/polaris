# EEF mimic-compliance candidate — 2026-07-03

## Goal

Add one default-off PolaRiS controller candidate on top of
edf053e4ae4e2dcfeecae5e9ff9b857fabe9020b: retain the rate-0.25 gripper
target slew, fixed activation-anchor arm interlock, and every existing
arm/gripper position, velocity, effort, and abort limit while changing only
the five passive Robotiq PhysX mimic followers to natural angular frequency
float32(100.0) rad/s and damping ratio float32(1.2).

No commit, push, deployment, or job launch is part of this implementation
handoff.

## Guidance and runtime ordering

NVIDIA's Omni Physics articulation guide recommends mimic compliance for the
same gripping/contact instability, demonstrates natural frequency 100.0 and
damping ratio 1.2, recommends a timestep-frequency product near one, and
discourages damping ratios below one:

- https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/dev_guide/rigid_bodies_articulations/articulations.html#articulation-mimic-joint-compliance
- https://docs.omniverse.nvidia.com/kit/docs/omni_physics/106.5/dev_guide/guides/articulation_stability_guide.html#mimic-joint-compliance

The pinned Isaac Lab source was independently checked in the evaluation
container. UsdFileCfg.func is the decorated spawn_from_usd; AssetBase calls it
before matching prims and registering initialization callbacks. The @clone
wrapper completes all matching spawn/clone work before returning. The
candidate therefore wraps the exact original callable, calls it once, then
authors the composed clone before articulation/PhysX initialization. Evidence
binds the original module/qualname/name, wrapper identity, one exact
/World/envs/env_0/robot root, one original call, and one overlay call.

## Implementation

- src/polaris/eef_gripper_runtime.py
  - Candidate-only pre-articulation spawn overlay; baseline returns the exact
    original callable and remains unmodified.
  - Immutable source USD is captured before spawn and after the overlay and
    compared exactly; the existing source contract remains a separate field.
  - Source and composed followers require the exact
    PhysxMimicJointAPI:<axis> instance.
  - Before any write, all five followers must pass prim/type, applied API,
    remapped referenceJoint, float32 gearing, zero offset, articulation
    exclusion, attribute type, and source frequency/damping checks.
  - Only after all-five prevalidation, ten scalar UsdAttribute.Set calls are
    made. All ten values and untouched structure are then reread.
  - After the first explicit reset, runtime installation performs composed-USD
    readback only; it makes zero hot writes. Later resets repeat the same
    read-only check.
  - Closed candidate evidence records source, before, after, and explicitly
    named post_reset_composed_usd_* snapshots; dt=1/120 and
    dt*frequency=5/6.
  - The extra static field is accepted only for the existing rate-0.25
    profile. The baseline static schema remains byte-for-byte unchanged.
- scripts/smoke_eef_pose_canary_controller_candidate.py
  - Installs the overlay before gym.make.
  - Bumps the scientific candidate/replay/failure identities.
  - Independently validates the full compliance transaction in initial,
    success, and controller-abort evidence.
- scripts/finalize_eef_pose_smoke.py
  - Host validator accepts and fully validates the extra field only when the
    expected target-slew profile is the rate-0.25 candidate; baseline remains
    closed and unchanged.
- src/polaris/eef_runtime_contract.py
  - Every durable static/dynamic validation call now derives and forwards the
    exact bound target-slew profile instead of silently defaulting to baseline.
- Tests add fake composed-stage coverage for exact writes/readback, original
  callable identity, successful exact-root/zero-offset binding,
  API/reference/gearing/exclusion tampering, late-follower pre-value drift
  with zero writes, partial schemas, type/value tampering, candidate-only host
  validation, failure verification, and explicit absence from scripts/eval.py.

## Validation

All commands used the host-safe Ego-LAP Python environment with this worktree
first on PYTHONPATH.

    python3 -m py_compile <all changed Python files>
    PASS

    ruff check <all changed Python files>
    PASS

    ruff format --check <all changed Python files>
    PASS

    pytest focused runtime/runner/finalizer/runtime-contract set
    178 passed, 1 expected pynvml deprecation warning

    pytest candidate success/failure verifier/finalizer set
    69 passed

    pytest broad EEF host-safe set
    650 passed, 1 expected pynvml deprecation warning

    pytest all 23 host-safe test modules
    771 passed, 30 subtests passed, 1 expected pynvml deprecation warning

tests/test_robust_differential_ik.py is the sole excluded test module because
it imports Isaac Lab during collection and the host test environment does not
contain Isaac Lab. An initial broad glob hit that known collection dependency;
it was not an implementation failure. The module must be exercised by the
pinned Isaac container during the eventual runtime gate.

## Residual gate

Host tests establish schema, transaction ordering, source/live identity,
baseline isolation, and downstream verifier behavior. They cannot establish
the physical response of the pinned GPU PhysX runtime. The next owner must run
the paired same-commit official/reasoning L40S controller gate and inspect both
the formally verified success/failure artifact and measured arm/gripper
dynamics before promotion.
