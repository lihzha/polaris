# EEF gripper rate-0.25 controller candidate — 2026-07-03

- Agent: `gripper-rate025-impl`
- Worktree: `/home/lzha/code/PolaRiS-worktrees/eef-gripper-target-slew-rate025-v2-20260703`
- Branch: `codex/eef-gripper-target-slew-rate025-v2-20260703`
- Base: `bb44752f89a03cd60165ae691f797e19a9d911d4`
- Version state: intentionally uncommitted; no push, deployment, simulator run, or cluster launch.

## Goal and hypothesis

Add one default-off EEF-only gripper target-slew candidate at factor 0.25 of
the unchanged 5 rad/s physical driver/follower limit. The exact float32 command
rate is 1.25 rad/s and the exact per-120 Hz target step is
0.010416666977107525 rad. The production arithmetic reaches the close endpoint
on apply 76 after 75 limited applies and 41 represented-overshoot `nextafter`
corrections. Holding the arm for that 76-apply ramp plus 10 settling substeps
should isolate the close transient while retaining the 0.95 arm target-slew
headroom candidate.

## Changes

- Preserved the existing factor-0.5 / 2.5 rad/s target-slew profile and default.
- Added a closed profile mapping and one exact boolean config defaulting false.
- Derived the baseline and candidate transition/interlock contracts by bounded
  simulations of the production float32 loop: `(38, 37, 15, 48)` and
  `(76, 75, 41, 86)` for endpoint, limited, `nextafter`, and interlock counts.
- Bound the pure interlock transition to an explicit validated substep count,
  and cross-bound the robust controller's interlock identity to the installed
  target-slew profile.
- Configured both paired model-free fixtures with the same controller profile.
  Official executes 96 close applies (`86 active + 10 released`); reasoning
  remains open and proves the same static profile without activation.
- Increased the exact replay plan from 122 to 127 actions (seven final-action
  repeats after the 120-action fixture).
- Added transactional controller-abort capture with pre-replay provenance,
  arm failure ring/current-velocity evidence, all-six gripper tail, active
  safety/candidate/target-slew state, and primary/secondary exception isolation.
- Added an independently hash-bound host failure verifier. On nonzero `srun`,
  the wrapper invokes it before returning the original nonzero code and never
  publishes a ready marker, srun-success status, or promotion attestation.
- Closed the successful lifecycle over that verifier as well: the finalizer
  validates its expected 64-hex digest, hashes its exact repository source,
  and records the identity under attestation provenance.
- Kept `set +e` through the immediate post-`srun` failure branch. Failure-side
  timestamp, log-mode, and verifier errors are each reported without skipping
  the next diagnostic or replacing the original `srun` return code; the
  success path restores `set -e` before requiring timestamp and log-mode work.
- Closed every failure-context lifecycle field over exact scalar types. The
  independent validator now shares one strict lifecycle helper between success
  and failure, including full schema, job/launch binding, step/nodelist checks,
  and bool-rejecting single-rank integer checks.
- Added regression/tamper/leakage coverage; native joint-position and pi0.5
  paths do not select the new flag.

## Validation

- Focused host suite: `144 passed`.
- Broad host suite: `733 passed, 30 subtests passed`; only
  `tests/test_robust_differential_ik.py` was excluded because the host test
  environment has no Isaac Lab import. Its isolated robust/gripper host stub
  passed in the focused suite.
- Ruff check: passed.
- Ruff format check: passed.
- Shellcheck on the candidate wrapper: passed.
- `bash -n` on the candidate wrapper: passed.
- Stdlib failure-verifier `--help` import/preflight: passed.
- Python `compileall`: passed.
- `git diff --check`: passed.
- The finalizer lifecycle timestamp fixture uses a deterministic one-second
  pre-write margin; its formerly flaky rejection test passed 20 consecutive
  isolated invocations before the focused and broad reruns.
- Production leakage search: the opt-in flag occurs only in its config/mixin
  and the isolated candidate runner; `scripts/eval.py`, native `ActionCfg`,
  `config.py`, and `DroidJointPos` contain no opt-in.

## Final status and risks

Implementation and host validation are complete. No local process or job was
launched. Target-surface Isaac/Pyxis behavior remains deliberately untested in
this implementation handoff; the next authorized step is independent diff
review followed by a separately approved committed/deployed model-free replay.
