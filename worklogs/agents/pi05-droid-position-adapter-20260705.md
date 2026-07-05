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

## 2026-07-05 — job 1099012 hard/soft limit diagnosis and correction

- Inherited failure evidence (the job was launched by the orchestrator, not by
  this implementation agent): controller smoke job `1099012`, source commit
  `921a3e25999cfedd337411e1e7d63f5864bd2316`, node `pool0-00020`, state
  `FAILED`, exit `1:0`, runtime `00:02:49`. Log:
  `/lustre/fsw/portfolios/nvr/users/lzha/slurm_logs/polaris-pi05-position/pi05_pos_smoke-1099012.out`.
- The smoke failed closed before any controller case in
  `capture_position_adapter_runtime`. It incorrectly required direct PhysX
  hard limits to equal `robot.data.soft_joint_pos_limits`.
- Exact diagnosis from the pinned image's Isaac Lab 2.3
  `articulation.py`: the buffered soft matrix is computed in float32 as
  `center=(lower+upper)/2`, `range=upper-lower`, then
  `soft=center +/- 0.5*range*factor`. With factor exactly 1, this is not a raw
  copy. Float32 center/range reconstruction changes q4 upper from raw hard
  `-0.0697999969124794` to soft `-0.06979990005493164`, and q6 lower from raw
  hard `-0.017500000074505806` to soft `-0.017499923706054688`. This is
  arithmetic rounding, not a comparison tolerance. In particular, q4's soft
  upper is outside the raw hard upper.
- The corrected closed contract binds three separate exact float32 layers:

  - buffered and direct PhysX hard limits, exact digest
    `d7ec7ea6108d670f910c43a9fba370e5023c7a5b9aa31df06b89ffc172529e00`;
  - factor-1 Isaac soft derivation and live buffered result, exact digest
    `fbf7535901c042fea5d901812ecd02c5fd81ade06c23c1499c32d66a859104de`;
  - inclusive zero-inset elementwise intersection guard
    `intersection(live_joint_pos_limits,live_soft_joint_pos_limits)`, exact
    digest
    `558f0c01f992abe1e6c60559665047d115b4891d25cd627c7bd68d9e9cbcfedb`.

- No `allclose`, epsilon, clipping, or new 1e-5 behavior was introduced.
  Runtime validation requires exact serialized float32 values, exact CUDA
  buffered-hard equality with CPU direct-hard values, exact soft derivation,
  factor 1, and the closed guard contract.
- Client and action-term guards now call the same pure helper. The client first
  casts the float64 reference formula target to the exact float32 actuator
  command, then performs the same inclusive zero-tolerance intersection check
  as the action term. Traces/incidents preserve both the float64 reference and
  float32 guarded target plus hard, soft, guard, source, inset, and contract
  digest.
- The actual controller smoke adversarial probe now targets q4 exactly one
  float32 step above the hard/intersection upper bound while remaining inside
  Isaac's rounded soft bound. It must raise before the setter and leave the
  articulation target unchanged. CPU tests also cover exact inclusive q4/q6
  boundaries, one-float32-step violations, and sub-ULP serialized evidence
  tampering.

### Correction validation

- Focused command:

  ```bash
  PYTHONPATH="$PWD/src:$PWD/third_party/openpi/packages/openpi-client/src:$PWD/third_party/openpi/src" \
    /home/lzha/code/ego-lap/.venv/bin/python -m pytest -q \
    tests/test_pi05_droid_position_adapter.py
  ```

  Result: `11 passed in 0.36s`.
- CPU-safe repository command:

  ```bash
  PYTHONPATH="$PWD/src:$PWD/third_party/openpi/packages/openpi-client/src:$PWD/third_party/openpi/src" \
    /home/lzha/code/ego-lap/.venv/bin/python -m pytest -q tests \
    --ignore=tests/test_robust_differential_ik.py
  ```

  Result: `325 passed, 5 subtests passed, 3 warnings in 14.02s`.
- All owned Python files passed `python -m py_compile` and Ruff; both launch
  shell files passed `bash -n`; `git diff --check` passed.
- This correction launched no job and changed no registry, shared document,
  main branch, canonical checkout, or remote cluster checkout. A fresh
  controller smoke is still required from the corrected committed descendant;
  job `1099012` remains preserved as failed diagnostic evidence.
