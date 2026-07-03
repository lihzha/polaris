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

## 2026-07-03 — replacement controller-only smoke 1098166

- Scope remained controller/runtime validation only. No policy server, model,
  or checkpoint was loaded, and this run cannot support a promotion claim.
- Frozen remote source was the clean standalone clone at detached commit
  `46e5eec62e66b45f04943c84d8fa5eb3dd71a4db`; both its Git directory and
  common directory were the clone-local `.git` directory.
- Slurm job `1098166` ended `FAILED`, exit `1:0`, after 3m11s on
  `pool0-00005`. Its container step `1098166.0` completed `0:0` after 3m and
  produced the child raw capture, child ready marker, and parent-published smoke
  artifact. The subsequent host finalizer failed, so the strict completion
  attestation is absent.
- Full log:
  `/lustre/fsw/portfolios/nvr/users/lzha/slurm_logs/polaris-pi05-jointvelocity/pi05_jv_ctrl_smoke-1098166.out`,
  15,295 bytes, mode `0644`, one link, SHA-256
  `b70cd04f7bc1bf4d2801c9ecec1bf498397681b42158962ff101b42295d4f737`.
- Preserved one-link, mode-`0444` evidence in
  `/lustre/fsw/portfolios/nvr/users/lzha/results/polaris-pi05-jointvelocity/controller-smoke/46e5eec-controller-only-v2`:
  parent-published smoke, 28,032 bytes, SHA-256
  `f9c6fa137d95f7b60fb5b240ce983d667293208358b7a58f0239b0d2f27c211f`;
  child raw capture, 27,276 bytes, SHA-256
  `c1e119b1ef5e3341d6d392a98f04248f6eb5eeacf26742bdaed04bf96e494f66`;
  child ready marker, 405 bytes, SHA-256
  `e17b3508ff3828670a574559bef755caff3c63620bf2b18630afac02bee4a3e0`;
  zero-exit srun status, 38 bytes, SHA-256
  `4e5e751b54b196606cff2a1b81695d339a69c9d1769cdf9aa7b72166375fc429`;
  GPU inventory, 66 bytes, SHA-256
  `52a36f3a1122d8813c9b9e279c4c1e681aa9db3fd3f74292937c07a73712e9d0`;
  saved sbatch, 8,776 bytes, SHA-256
  `056950ad4e607557dca22d92d65540e6c0de8a7f99a0483eb02de6c68959bb14`;
  and submission record, 952 bytes, SHA-256
  `478ea1d008171a1a12c8c30c00ea0c3fad042ac9ec6bc8315a42097be6c65126`.
- The captured runtime payload passed its 20-case semantic validator: signed
  joint directions and measured velocities were coherent, velocity targets
  stayed within the pinned limits, gripper open/closed/boundary commands
  matched the float32 contract, and reset positions, velocities, and targets
  matched the reset contract. CUDA/CPU tensor placement also matched the
  pinned runtime contract. These are preserved controller observations, not a
  completed strict gate, because no host completion was published.
- Root cause of the host failure was path spelling, not controller behavior.
  The child and ready records preserved `/lustre/fsw/...`; the old finalizer
  resolved its expected child path to `/lustre/fs11/...` and rejected the
  lexical mismatch. Both spellings were independently verified to resolve to
  the same regular file with device `f7980bf2`, inode
  `324260256274187394`, size 27,276, mode `0444`, one link, identical
  timestamps, and identical SHA-256.
- The pending fix preserves normalized absolute producer/host spellings and
  records their resolved spelling. An ancestor alias is accepted only when
  both spellings retain the expected final filename and identify the same
  regular one-link file by device, inode, size, mode, link count, mtime,
  ctime, and resolved target. Final-component symlinks and same-content files
  with different inodes remain rejected. Ready-marker validation now uses the
  producer-recorded raw path, and source, data, container, run-artifact, and
  submission records retain declared and resolved spellings consistently.
- Expanded lifecycle/runtime/client/finalizer validation after the alias fix:
  44 focused tests passed. All host-runnable repository tests passed with 71
  tests plus 5 subtests; the single pinned-container Isaac Lab controller test
  remains intentionally excluded from the host run. Ruff format/check, Python
  compilation, Bash syntax, ShellCheck, and `git diff --check` all passed.
- Independent review first rejected conversion through `Path` before lexical
  validation because it could hide `/./`, repeated-slash, and `//` producer
  spellings. Validation now operates on the raw producer and CLI strings, the
  submitter enforces the same normalized-absolute contract before `sbatch`, and
  adversarial spelling tests pass. A later main review then found that raw
  `POLARIS_DIR` was canonicalized before that submitter check. Both submitter
  and worker now validate it before `realpath`, retain separate declared and
  resolved source paths, and pass executable malformed-path tests. The prior
  independent GO was invalidated; final independent re-audit of the corrected
  tree returned GO.
- No further job has been launched. The alias fix remains an uncommitted
  candidate at base commit
  `46e5eec62e66b45f04943c84d8fa5eb3dd71a4db`, pending main-agent review.
