# PolaRiS EEF Release-Ramp16 Production V1 Worklog

Agent: `root`

Branch: `codex/eef-release-ramp16-production-v1-20260703`

Base: `0611d384f5f26ef9bd8ff114be273e875c3fe719`

Date: 2026-07-03

## Goal

Replace the diagnostic post-setter arm release overlay with a production-grade, profile-bound controller implementation, then rerun the exact reasoning replay and full-horizon LAP checkpoints before promotion.

## Diagnostic decision

The immutable three-arm L40S sweep at diagnostic commit `25cef0cdbcd249f4679564a6ca643c0a4d8b4972` selected the production follower cap of 5 rad/s plus an inclusive 16-physics-substep arm slew-cap release ramp:

| Job | Variant | Source actions | Frozen tail | Outcome | Peak arm velocity |
| --- | --- | ---: | ---: | --- | ---: |
| 1098570 | cap-5 abrupt control | 293/294 | 0/64 | J7 abort, policy 293/substep 2 | 2.923666 rad/s |
| 1098629 | cap-8 abrupt | 293/294 | 0/64 | J7 abort, policy 293/substep 3 | 2.610134 rad/s |
| 1098631 | cap-24 abrupt | 294/294 | 12/64 | J7 abort, policy 295/substep 4 | 2.617754 rad/s |
| 1098633 | cap-5 + ramp16 | 294/294 | 64/64 | completed | 0.817634 rad/s |

The selected replay completed three ramps, all 2416 physics applies, and had zero target guard-band violations. It is bit-identical to the cap-5 abrupt control through apply 1599, immediately before the first intervention. This isolates the causal benefit for the pinned replay but does not establish cross-task or cross-initial-condition promotion.

Evidence:

- `/home/lzha/code/ego-lap/.codex_artifacts/polaris-reasoning-fulltrace-cap-release-followup-25cef0c-20260703/comparison.md`
- `/home/lzha/code/ego-lap/.codex_artifacts/polaris-reasoning-fulltrace-cap-release-followup-25cef0c-20260703/sweep_manifest.json`

## Implementation

- Added a pure float32 release-target helper and explicit `HOLD -> RAMP -> RELEASE` transition in `src/polaris/eef_controller_repair.py`.
- Defined natural hold completion to schedule index 0 on the next apply, open-cancel during HOLD to apply index 0 immediately, OPEN during RAMP to continue without restart/skip, and CLOSE during RAMP to cancel atomically and return to HOLD.
- Integrated the ramp before applied-delta, soft-limit, guard-band, and slew validation in `src/polaris/robust_differential_ik.py`.
- Kept one position-target setter. Exact velocity/position readback and final failure-trace target staging precede the joint interlock/ramp lifecycle commit.
- Added reset coverage through both `begin_safety_episode()` and `reset()`.
- Preserved baseline and durable v3 report schemas. Added distinct v4 profile `arm_slew_0p95_gripper_rate0p25_fixed_anchor86_release_ramp16_mimic100_damping1p2_v4` with a conditional closed `arm_release_ramp` report sibling.
- Bound v4 through controller config validation, sidecars, aggregates, runtime contracts, and the public eval log.
- Updated exact production-source identities used by the model-free replay gate after the intentional `config.py` and `eval.py` changes.

## Host validation

The new worktree initialized the pinned `third_party/openpi` and GLM submodules. A direct `uv sync` was blocked by the known `flatdict==4.0.1` undeclared `pkg_resources` build dependency, so host-safe tests used the already validated Ego-LAP Python environment with this worktree first on `PYTHONPATH`.

Commands:

```bash
/home/lzha/code/ego-lap/.venv/bin/ruff format <changed Python files>
/home/lzha/code/ego-lap/.venv/bin/ruff check <changed Python files>
python3 -m py_compile <changed source files>
PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
  /home/lzha/code/ego-lap/.venv/bin/python -m pytest -q \
  tests --ignore=tests/test_robust_differential_ik.py
```

Current result: `867 passed`, `30 subtests passed`, Ruff clean, source compilation clean, and `git diff --check` clean. The omitted test imports Isaac Lab and is reserved for the pinned target runtime. Pure coverage includes the exact 86-HOLD/16-RAMP sequence, natural/open release edges, OPEN continuation, CLOSE reactivation, randomized float32 formula checks, endpoint bit identity/no-alias checks, v3/v4 schema separation, tamper rejection, transaction setter/readback/trace failure atomicity, and v4 sidecar/aggregate propagation.

Independent review initially found two P1s: disabled-path device synchronizations/readbacks that changed the durable baseline/v3 hot path, and insufficiently closed ramp report equations. Both were fixed. The non-ramp transaction now executes only the parent velocity and position setters and retains the parent failure-trace ordering. The v4 validator requires exact integer-zero gripper writes, one non-limited endpoint per completed ramp, phase-consistent last index/apply evidence, and positive target-change maxima if and only if a target was limited. Re-review on code diff SHA `56c864fa2bfdac1ea8b86909230fecfe698b177c7977eca158916f83280813b4` returned GO with no P0/P1; independent validation reproduced `867 passed`, `30 subtests passed`, 184 focused passes, Ruff, compilation, and diff checks.

## L40S target-runtime gate

Fail-closed job `1098635` ran the pinned CUDA 13/Isaac image on one L40S against commit `7fc74d648328432a7f9f06d13c0e82a03f73a0c1`. Its deliberate no-tests-selected probe correctly produced immutable `5\n` evidence. The authoritative test run then reported `83 passed, 1 failed` and immutable `1\n`, so the wrapper and Slurm job failed as intended.

The sole failure was in `test_standard_action_reset_clears_selected_candidate_state`: the test constructs an object with `object.__new__` and bypasses `__init__`, but did not provide `_max_delta_joint_pos`, which real construction installs before the interlock and release-ramp lifecycle resets. The production lifecycle and controller math were unchanged. The test now supplies that initialized dependency explicitly and verifies that standard reset clears both the inherited interlock anchor and the new release-ramp phase/evidence/transaction latch. A fresh host validation and L40S gate are required before promotion.

The repair is test-only. The Isaac test uses the exact float32 Panda velocity-limit vector divided by 120 Hz, matching real construction, and the host stub now invokes the same reset lifecycle with a faithful parent raw-action reset plus an initialization-order assertion. Ruff, Python compilation, `git diff --check`, the focused host stub, and the full host-safe suite pass: `867 passed`, `30 subtests passed`. Production source hashes remain identical to commit `7fc74d6`.

## Next gate

1. Preserve the independent GO and commit/push the test-only child revision without changing production source.
2. Deploy that exact child commit and rerun the fail-closed L40S gate.
3. Build and review a pinned replay-only child revision that observes the production v4 path without a post-setter overlay.
4. Run and inspect the exact 2,416-apply production-core replay on L40S, including traces and video.
5. Run fresh full-horizon official LAP-3B and reasoning-checkpoint canaries in parallel.
6. Promote to the standard suite only after every preceding target-runtime gate passes.
