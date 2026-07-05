# Official pi0.5-DROID position-adapter worklog

## 2026-07-05 — implementation start

- Agent: `pi05-droid-position-adapter-20260705`.
- Goal: replace the non-authoritative direct-rad/s PolaRiS execution path with
  the official DROID command semantics: for every executed action, read fresh
  measured Panda joints, clip the seven normalized arm commands to `[-1, 1]`,
  compute `q_target = measured_q + 0.2 * command`, and hold that absolute
  target for all eight 120 Hz physics substeps. The gripper remains an absolute
  closed-positive binary command.
- Isolated worktree:
  `/home/lzha/code/PolaRiS-worktrees/pi05-droid-position-adapter-20260705`.
- Branch: `codex/pi05-droid-position-adapter-20260705`.
- Deliberate base: `bf344db3554ed22624c2960154da21d7b233d683`
  (`codex/pi05-host-media-tools-fix-20260704`).
- Official OpenPI inference compatibility commit remains
  `bd70b8f4011e85b3f3b0f039f12113f78718e7bf`.
- Official DROID source pin: repository `droid-dataset/droid`, commit
  `33ae6a67274f36d2e29525b86f23a56616ef43a7`. The three control sources are
  byte-identical at `c5737e40a6b18859b5b78dbcdbf1e3b3f5e461be`:
  - `droid/robot_env.py`: SHA-256
    `41cff898b9e3c3b465c3465fd4b9889db70edd9da4c083af1789ea335ad6e116`,
    Git blob `f555b501b5b91fb53f82d54c43684078cef15892`.
  - `droid/franka/robot.py`: SHA-256
    `25f2edf095b13f590371a4c53c8fbf0b8948d5c08b42600542a579917442ec38`,
    Git blob `e4f02202542ffcc050b089aa9ff0fde16323d289`.
  - `droid/robot_ik/robot_ik_solver.py`: SHA-256
    `c32df1ea7e8c56fc32b8560c0a057892c505d1a05fa0a850540b50b6e964c57d`,
    Git blob `a073699c20b4bef3e5941454c0d6a9aaf7b05534`.
- Scope restriction: local source and CPU/golden validation only. No Slurm
  submission, cluster mutation, registry publication, main-branch update, or
  canonical-checkout mutation is authorized for this implementation handoff.

## 2026-07-05 — implementation complete, CPU-validated

- Added a distinct `openpi_pi05_droid_fresh_jointdelta_position_v1` path. It
  preserves the official `pi05_droid` checkpoint, global DROID normalization
  (explicitly rejects `single_arm` substitution), FLOW sampler, state, model
  image order, 15x8 response, and execute-eight behavior. It changes only the
  simulator execution adaptation to the official per-step formula.
- The CPU oracle and live client now binarize the closed-positive gripper,
  clip all eight command dimensions, read fresh live float32 Panda joint
  positions for every executed action, prove policy/live equality, compute
  `q_target = q_measured + 0.2 * clip(command)`, and reject out-of-soft-limit
  targets before an environment step or articulation setter.
- Added an absolute `JointPositionAction` term with a typed independent
  pre-setter guard and an exact float32 eight-physics-substep target-hold
  witness. Existing all-13-DOF velocity/acceleration monitoring remains active.
- Closed train/eval parity for inputs: global `droid` stats; float32 ordered
  `panda_joint1..7` plus closed-positive gripper; no observation noise, clip,
  corruption, or concatenation; exact native external/wrist RGB inputs of
  `[720,1280,3] uint8`; OpenPI resize-with-pad to `[224,224,3]`; no wrist
  rotation; external, wrist, then zero-masked right-wrist model order. Query
  traces bind SHA-256 identities for both native and resized images.
- Live runtime evidence binds Isaac Lab 2.3 source hashes, actual imported
  `droid_cfg.py` observation functions, action/reset order, position and
  gripper buffers, buffered/direct PhysX drive values, soft/hard position
  limits, all-six gripper reset/mimic/ownership/limits, and a closed runtime
  digest. CPU round-trip tests reject direct-drive, gripper, and policy-state
  tampering.
- Added exact serving/handshake tooling and closed success/numerical-failure
  traces, incidents, episode sidecars, and close-ready lifecycle artifacts.
- Added a controller-only L40S smoke scaffold. Its host attestation binds the
  final/child/ready smoke artifacts, exact DROID formula in a two-step fresh
  re-anchor probe, target-limit guard, saved Slurm spool script, srun zero,
  one-L40S inventory, pinned Pyxis image, PolaRiS-Hub assets/revision, live
  package/source identities, and governed committed files by relative path,
  Git blob SHA-1, SHA-256, and size. It is explicitly non-promotable without a
  separate checkpoint canary. `/lustre/fsw` inputs are resolved to physical
  paths before comparison, and cleanup is restricted to the exact per-job
  cache directory.

### Validation

- Focused adapter/runtime/client plus inherited-contract tests:

  ```bash
  PYTHONPATH="$PWD/src:$PWD/third_party/openpi/packages/openpi-client/src:$PWD/third_party/openpi/src" \
    /home/lzha/code/ego-lap/.venv/bin/python -m pytest -q \
    tests/test_pi05_droid_position_adapter.py \
    tests/test_pi05_droid_jointvelocity_contract.py::test_eval_profile_intent_is_required_only_for_native_jointvelocity \
    tests/test_joint_velocity_smoke.py
  ```

  Result before the final camera/state test addition: `13 passed`; the final
  position-adapter file alone then passed `9 passed`.
- CPU-safe repository suite:

  ```bash
  PYTHONPATH="$PWD/src:$PWD/third_party/openpi/packages/openpi-client/src:$PWD/third_party/openpi/src" \
    /home/lzha/code/ego-lap/.venv/bin/python -m pytest -q tests \
    --ignore=tests/test_robust_differential_ik.py
  ```

  Result: `323 passed, 5 subtests passed, 3 warnings in 13.56s`.
- All owned Python files passed `python -m py_compile`; both shell launch files
  passed `bash -n`; all owned Python files passed
  `/home/lzha/code/ego-lap/.venv/bin/python -m ruff check`; `git diff --check`
  passed.
- An earlier unfiltered repository-root `pytest -q` was intentionally not used
  as the acceptance suite: this local CPU environment has no Isaac Lab for
  `tests/test_robust_differential_ik.py`, and repository-root discovery also
  collects the initialized OpenPI submodule's own training test. The scoped
  `tests/` command above excludes only the known Isaac-only test and passes.

### Handoff and remaining execution

- No Slurm job, simulation, policy server, checkpoint evaluation, registry
  update, shared-doc write, branch merge, or canonical-checkout mutation was
  performed in this worktree. In particular, scheduler acceptance is not being
  represented as controller validation.
- Required next gate: check out the final commit as a clean detached standalone
  PolaRiS clone on l401, create one fresh empty result directory, and invoke:

  ```bash
  POLARIS_DIR=<canonical-detached-clone> \
  EXPECTED_POLARIS_COMMIT=<final-commit> \
  RUN_DIR=<fresh-empty-canonical-result-dir> \
    scripts/polaris/submit_pi05_droid_position_controller_smoke.sh
  ```

  Monitor through immutable final attestation and inspect the controller
  artifacts; only then launch the separately governed official-checkpoint
  canary/evaluation wrapper.
