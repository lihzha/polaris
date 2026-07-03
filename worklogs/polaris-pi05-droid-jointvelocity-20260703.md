# Native pi0.5-DROID joint-velocity controller gate

## 2026-07-03 — controller-only smoke 1098163

- Scope: controller/runtime validation only; no policy server, model, or
  checkpoint was loaded. Promotion to checkpoint evaluation remained
  forbidden.
- Frozen source: detached clean remote checkout
  `bce3041f12eca3ce6ed2e38ba96b5cb36d9f5907`, tree
  `2600c1fab66dc839178af0790d4b4e8fbfad8e69`, with OpenPI at
  `bd70b8f4011e85b3f3b0f039f12113f78718e7bf`.
- Runtime: one NVIDIA L40S on `pool0-00005`, driver `580.105.08`, container
  SHA-256
  `ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a`.
- Slurm job `1098163` ended `FAILED`, exit `1:0`, after 2m53s. The Kit child
  constructed the 8-action joint-velocity environment but exited zero without
  publishing its raw capture. The stdlib parent rejected the missing capture,
  so no smoke pass or completion attestation was published.
- Full log:
  `/lustre/fsw/portfolios/nvr/users/lzha/slurm_logs/polaris-pi05-jointvelocity/pi05_jv_ctrl_smoke-1098163.out`,
  15,728 bytes, SHA-256
  `829e743957f023f2567330dd7da3082c3ef2c9eb4b86b797b2c6a354c4ea26fc`.
- Preserved immutable failure evidence in
  `/lustre/fsw/portfolios/nvr/users/lzha/results/polaris-pi05-jointvelocity/controller-smoke/bce3041-controller-only-v1`:
  submission, GPU inventory, saved sbatch, and
  `{"job_id":1098163,"srun_exit_code":1}`. The pass artifact and completion
  are absent as required.
- Job-specific cache residue was inspected at
  `/lustre/fsw/portfolios/nvr/users/lzha/cache/polaris/runtime/jointvelocity-smoke-1098163`
  (30,293,094 bytes, 56 files, including `_cache.lock`), then removed after
  the job was terminal. Its absence was verified before the replacement
  candidate was frozen.

## Failure diagnosis and lifecycle hardening

- The original failure handler called `SimulationApp.close()` before
  re-raising setup, runtime-validation, capture, environment-close, or
  persistence exceptions. The pinned Kit close can terminate the process with
  status zero, making the original exception unreachable. Therefore job
  `1098163` did not prove that `env.close()` itself failed.
- Pinned Isaac Lab source establishes a concrete runtime-validation mismatch:
  `BinaryJointAction` stores `_open_command` and `_close_command` as one
  dimensional `[num_joints]` tensors. For the one DROID finger joint their
  shape is `[1]`, while commit `bce3041` incorrectly required `[1,1]`. This
  check precedes the robot/PhysX validation and is the first deterministic
  shape mismatch in the observed path. Raw and processed gripper action
  buffers remain `[1,1]` and are validated separately.
- Failure paths now publish a canonical, fsynced, non-overwriting mode-0444
  failure status with stage, exception type/message, and traceback; they skip
  Kit close and force `os._exit(1)`, including for `SystemExit(0)`.
- Success now requires: controller capture, successful `env.close()`, full
  semantic validation, immutable mode-0444 raw capture, immutable mode-0444
  ready marker binding the raw absolute path/size/SHA/mode, immediate
  `SimulationApp.close()` invocation, and child exit zero. The stdlib parent
  rejects missing, partial, linked, noncanonical, wrongly permissioned,
  tampered, failure-bearing, or nonzero-child transactions before publishing.
  Final lifecycle wording records close invocation followed by zero child exit;
  it no longer claims that Kit close returned.

## Device and source contract hardening

- Authoritative diagnostic job `1098162` completed `0:0` and established
  CUDA:0 float32 dynamic/cached/actuator tensors versus CPU float32 static
  PhysX stiffness, damping, max-force, and max-velocity getters. Its emitted
  device-probe JSON SHA-256 is
  `d3c8ccfcb16cd523f084f5c7c82f41a03c1c2ab0f58487f45ff4c2a59066283c`.
- Runtime and independent finalizer validation now require, field by field:
  arm clip/raw/processed tensors, buffered drive tensors, and gripper
  command/raw/processed tensors on `cuda:0`; direct static PhysX drive tensors
  on `cpu`; every tensor remains exact `torch.float32` with closed shape/value
  schemas.
- Gripper command/action device evidence is grounded in the pinned Isaac Lab
  implementation, which constructs all four tensors on `self.device`, and the
  live environment's `cuda:0` device. The previously omitted
  `binary_joint_actions.py` source is now SHA-256 pinned as
  `84bf343dc4a609d2327f1ee8b965439f49f3167f9a45f652e9aa6b652c9c0630`.

## Standalone deployment checkout gate

- A later diagnostic exposed that the job-`1098163` checkout was a linked Git
  worktree: its `.git` was a gitfile pointing into a common directory outside
  the source mount. The controller smoke failed earlier, but a successful
  rerun from that layout would reach the strict finalizer and fail its Git
  provenance queries inside the container.
- The submitter, sbatch worker, and Python finalizer now all require `.git` to
  be a real, non-symlink directory at exactly `<POLARIS_DIR>/.git`. Both
  `--absolute-git-dir` and the absolute Git common directory must resolve to
  that same in-root directory, and `HEAD` must be detached. Linked worktrees,
  gitfiles, external common directories, and branch checkouts fail before a
  pass can be published.
- The next deployment must be a fresh standalone clone followed by an explicit
  detached checkout of the reviewed commit. The existing linked checkout is
  not eligible for reuse.

## Local validation and launch state

- Focused lifecycle/runtime/finalizer suite: 32 passed.
- All host-runnable repository tests with the worktree-local OpenPI origins:
  69 passed plus 5 subtests. The single Isaac Lab controller test requires the
  pinned Isaac container and was not run on the host.
- Ruff passes for every changed Python file; Python compilation, Bash syntax,
  ShellCheck, and `git diff --check` pass.
- No replacement cluster job has been launched. Relaunch is intentionally
  paused for main-agent review of this dirty candidate.
