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

## 2026-07-03 — completed controller-only smoke 1098174

- The reviewed alias fix was committed as
  `90d56b3b8d0a93ad7c48319a377d325790b89144`, tree
  `e6f27fb6c6da31f96b296417cd98ebc3be0c8775`, and pushed to
  `lihzha/codex/pi05-droid-native-jointvelocity-v1-20260703` before launch.
  Its six reviewed file hashes matched the frozen candidate.
- Launch source was a new full standalone clone declared at
  `/lustre/fsw/portfolios/nvr/users/lzha/src/PolaRiS-pi05-jv-90d56b3b8d0a-standalone-v3`
  and resolved at
  `/lustre/fs11/portfolios/nvr/projects/nvr_lpr_rvp/users/lzha/src/PolaRiS-pi05-jv-90d56b3b8d0a-standalone-v3`.
  It had a real in-root `.git` directory, no alternates, detached `HEAD`, a
  clean tracked/untracked status, and OpenPI initialized over HTTPS at
  `bd70b8f4011e85b3f3b0f039f12113f78718e7bf`. GLM and OpenPI's unrelated
  nested submodules remained uninitialized.
- The committed public submitter launched Slurm job `1098174` with namespace
  `90d56b3b8d0a-controller-only-v3`. It completed `0:0` in 3m17s on one
  NVIDIA L40S at `pool0-00005`; the container step completed `0:0` in 2m58s
  with maximum RSS 31,054,664 KiB. The full 15,992-byte mode-`0644`, one-link
  log has SHA-256
  `857fba68ad5cec4c033fc0759086eb5e813ce3d9d2d877e82737ed40b4dc8a5e`
  at
  `/lustre/fsw/portfolios/nvr/users/lzha/slurm_logs/polaris-pi05-jointvelocity/controller-v3-90d56b3b8d0a/pi05_jv_ctrl_smoke-1098174.out`.
  No traceback, finalization error, child failure, or nonzero-srun marker was
  present.
- The strict finalizer published and live-reverified the 13,947-byte
  mode-`0444`, one-link controller-only completion with SHA-256
  `05403d0aabf3ebc8111cecf64d33f56f50a3a5673e7a84653ae096e7f4027ad3`.
  The remaining immutable one-link artifacts are: parent-published smoke,
  28,042 bytes, SHA-256
  `ef046c07e844ec15accf8cacc0178c67d587ee497047b416e575464054f2c611`;
  child raw capture, 27,276 bytes, SHA-256
  `c1e119b1ef5e3341d6d392a98f04248f6eb5eeacf26742bdaed04bf96e494f66`;
  ready marker, 410 bytes, SHA-256
  `5e8897bc67fef727f399ca8b15f3575de1939409f0a69e5b99b6aa00083ad5b4`;
  zero-exit srun status, 38 bytes, SHA-256
  `ece5d9dfa33e3dc4d23de0680407585bb8010e954a194a6c9b376703b41eb765`;
  GPU inventory, 66 bytes, SHA-256
  `52a36f3a1122d8813c9b9e279c4c1e681aa9db3fd3f74292937c07a73712e9d0`;
  saved sbatch, 9,643 bytes, SHA-256
  `5d257bc2df44446307f9c07da5e1a51084930c554b7d9abdaa2445222fc4c482`;
  and submission record, 1,629 bytes, SHA-256
  `7d050bf80b413b0546971dab42d20315bd71f9c863268578bc09b16c7df1fb00`.
- Every completion path retained its normalized `/lustre/fsw/...` declared
  spelling and separately recorded its `/lustre/fs11/...` resolved spelling.
  Smoke, raw, ready, status, GPU, saved-sbatch, and image bindings reported
  alias equivalence, producer/host spelling agreement, and the exact device
  and inode. Source, data root, every asset and metadata file, run directory,
  and submission record also retained declared/resolved paths. Live
  re-verification reproduced the completion hash.
- Runtime provenance pinned image SHA-256
  `ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a`,
  PolaRiS-Hub revision `8c7e4103e266ef83d8b1ad2e9a63116edd5f155b`,
  FoodBussing initial-condition SHA-256
  `40091faee14f692350220871d30705294f21f17ae3d2974cd3c09a34d560f5de`,
  scene SHA-256
  `82cd641e422935b394ce7ea7b6be55214c9952a2544000222921e544c409b489`,
  and DROID USD SHA-256
  `d8379925b103963dbf3e7c85bcc4ae101b81b7c1d7dabe7d2e964f41d069ec44`.
- All 20 controller cases passed: hold remained zero; every signed
  `panda_joint1..7` command produced the requested position and measured-
  velocity direction near 0.25 rad/s; positive limit-case velocities ranged
  from 0.9656 to 1.0133 rad/s and negative limit-case velocities from -1.0065
  to -0.9896 rad/s; action processing exactly matched articulation velocity
  targets. Gripper open and boundary `0.5` produced target `0`, closed `1.0`
  produced float32 pi/4, and reset exactly restored default position, zero
  velocity, and zero velocity target. Arm/gripper action and buffered drive
  tensors were float32 on `cuda:0`; direct PhysX arm-drive evidence was
  float32 on CPU. Runtime SHA-256 was
  `495ce92226ad0d1840138fc2b315fc2531d0ff50953fb16d70172080a8ee0b71`.
  Lifecycle evidence records completed `env.close()`, immediate
  `SimulationApp.close()` invocation, zero child exit, parent publication, and
  no failure sidecar.
- The job-specific cache contained 30,703,440 bytes, 74 files, 46 directories,
  and a zero-byte `ov/_cache.lock`. After terminal artifact inspection it was
  removed, and absence was verified through both `/lustre/fsw` and
  `/lustre/fs11` spellings.

## Remaining gripper simulation-limit gate

- The successful gate validates the existing native arm velocity drive and
  binary gripper command/target contract. It does not establish a physical
  gripper slew limit: `NVIDIA_DROID_JOINT_VELOCITY` copies `NVIDIA_DROID` and
  replaces only the two Panda arm actuators, so it inherits the gripper's
  legacy `ImplicitActuatorCfg(velocity_limit=5.0)`. The job's live Isaac Lab
  warning states that this legacy field is ignored; the runtime attestation
  currently contains gripper command/action tensors but no gripper PhysX
  max-velocity or measured-slew evidence.
- This is therefore not EEF-only. It affects the same simulated gripper used
  by native joint-velocity evaluation. Job `1098174` remains valid within its
  explicitly attested controller-only scope, but authoritative native full
  checkpoint evaluation should remain blocked until the gripper uses and
  attests the intended simulation-side limit (and the legacy effort-limit
  path is audited), gripper slew is covered by the smoke, and the native
  controller gate is rerun. An EEF-only canary cannot independently qualify
  the native action mode or its changed source.
- No model, checkpoint canary, or full checkpoint evaluation was launched.

## Native gripper-limit remediation candidate (not launched)

- The native-only articulation copy now replaces its gripper actuator with an
  explicit simulation-side velocity limit of `5.0` rad/s and effort limit of
  `200.0`. Both legacy and `_sim` fields are authored for pinned Isaac Lab
  2.3 compatibility. The shared `NVIDIA_DROID` object used by joint-position
  and EEF evaluation is unchanged; an Isaac-stub regression test proves the
  copied native actuator is distinct and that the shared config has no newly
  authored `_sim` fields.
- Candidate intent is named
  `implicit_gripper_physx_velocity_limit5_cuda_actuator_cpu_static_physx_v1`.
  The serving contract declares the configured fields and the exact live
  float32 `[1,1]` CUDA actuator and CPU direct-PhysX stiffness, damping,
  effort, and velocity surfaces. Runtime validation independently reads both
  surfaces and rejects missing, device-swapped, or value-swapped fields. The
  exact `robot_cfg.py` source digest is pinned alongside `droid_cfg.py`.
- Controller smoke profile v2 preconditions and physically moves the gripper
  for open, closed, and boundary-`0.5` cases. It records before/after finger
  position and velocity, recomputes one-step slew at the attested 15 Hz policy
  rate, checks the 5 rad/s limit, checks motion direction, and verifies reset
  state and target. The independent finalizer repeats these checks rather than
  importing the runtime validator.
- The candidate profile is explicit and required across the submitter export,
  sbatch worker, controller CLI, smoke artifact, finalizer CLI, and completion
  `candidate_intent`. Omission and the legacy ignored-limit profile are covered
  by fail-closed runtime, smoke, CLI, and static launch-contract tests. Full
  joint-velocity evaluation also requires the exact candidate profile, while
  joint-position and EEF modes reject that profile argument.
- Host validation after the contract expansion: 49 focused tests passed; all
  host-runnable tests passed with 81 tests plus 5 subtests. The one
  `robust_differential_ik` module still requires Isaac Lab and cannot collect
  in the host venv. Ruff check/format for all changed Python files, Python
  compilation, Bash syntax, ShellCheck, and `git diff --check` passed. The
  whole-tree Ruff baseline still has nine unrelated pre-existing findings in
  the splat renderer and environment package initializer.
- This candidate remains uncommitted and unpushed at the independent-review
  boundary. No source deployment, controller rerun, policy server, model,
  checkpoint canary, or full evaluation has been launched. Job `1098174`
  remains valid only for its earlier arm/native-target controller scope and
  remains non-promotable for authoritative checkpoint evaluation.

## 2026-07-03 — native gripper-cap controller gate 1098204

- The independently reviewed candidate was committed and pushed as
  `5e2947fc900838715859cf8d2476410527737924`, tree
  `e1d9f72c9fc61362ba5b42c9b9ab2c525f6e9857`, on
  `lihzha/codex/pi05-droid-native-jointvelocity-v1-20260703`. The accepted
  combined tracked-plus-new-test candidate hash was
  `8121586b4383c15484ab09127e0b6a989d918b51ec68e6d3ae1574404096760e`.
- Launch source was a new detached, clean standalone clone declared at
  `/lustre/fsw/portfolios/nvr/users/lzha/src/PolaRiS-pi05-jv-5e2947fc9008-standalone-v4`
  and resolved at
  `/lustre/fs11/portfolios/nvr/projects/nvr_lpr_rvp/users/lzha/src/PolaRiS-pi05-jv-5e2947fc9008-standalone-v4`.
  Its Git and common directories were the clone-local `.git`, no alternates
  were present, and OpenPI was explicitly active over HTTPS at
  `bd70b8f4011e85b3f3b0f039f12113f78718e7bf`.
  Unrelated GLM and OpenPI nested submodules remained uninitialized.
- The mode-`0444`, one-link, 3,633-byte frozen launch wrapper has SHA-256
  `105e103e6b12b1fbb4c4b4ef7d82af70f3988e9713508fa05f6058db47041527`
  at
  `/lustre/fsw/portfolios/nvr/users/lzha/launchers/polaris-pi05-jointvelocity/5e2947fc9008-controller-v2-gripper-cap/submit.sh`.
  It bound the commit, tree, OpenPI pin, candidate profile, and fresh result,
  log, and cache namespaces before invoking the committed public submitter.
- Slurm job `1098204` completed `0:0` in 3m16s on one NVIDIA L40S at
  `pool0-00005`; its container step completed `0:0` in 2m57s with MaxRSS
  30,782,140 KiB. The one-link 15,333-byte log has SHA-256
  `f4c21d955c2673f479d6c9a02821fe0b7ba8a1fee3d7d37231831ffa59842cea`
  at
  `/lustre/fsw/portfolios/nvr/users/lzha/slurm_logs/polaris-pi05-jointvelocity/controller-v2-5e2947fc9008-gripper-cap/pi05_jv_ctrl_smoke-1098204.out`.
  Its only errors were the established headless NGX-context messages; no
  traceback, child failure, contract mismatch, or nonzero status occurred.
- The strict finalizer published and live-reverified the mode-`0444`, one-link,
  15,744-byte completion, SHA-256
  `778594b8eea64d6c2fb031d43af53539e07014af218e8e2c60751cb0d399a657`.
  It records profile `pi05_droid_native_jointvelocity_l40s_controller_smoke_v2`,
  exact candidate intent
  `implicit_gripper_physx_velocity_limit5_cuda_actuator_cpu_static_physx_v1`,
  scope `controller_only_no_model_or_checkpoint`, and promotion forbidden
  without a separate checkpoint canary. The failure sidecar is absent.
- Live runtime SHA-256
  `c7e932ca9f697cd02825fb06ee5fa5c0f168af73309026e91d844f16fd3729eb`
  proves configured gripper legacy and simulation effort/velocity limits of
  `200.0` and `5.0`; CUDA float32 `[1,1]` actuator stiffness
  `5729.578125`, damping `0.011459155939519405`, effort legacy/sim `200.0`,
  and velocity legacy/sim `5.0`; and the same four effective static values
  from the CPU direct-PhysX view. Isaac Lab was exactly `2.3.0`, both PolaRiS
  source hashes matched, and the arm CUDA/CPU velocity-drive contract remained
  unchanged.
- All 20 cases passed. Gripper open and boundary `0.5` moved from
  `0.7853981853` toward target `0` with endpoint velocity `-4.9728875160`
  rad/s and average slew `-4.9784392118` rad/s. Closed moved from `0` toward
  float32 pi/4 with endpoint velocity `4.9801592827` rad/s and average slew
  `4.9713331461` rad/s. Directions, targets, and the 5 rad/s cap all matched;
  reset restored finger position/target/velocity to exact zero. Hold was exact
  zero, all fourteen signed arm cases had correct position and measured-
  velocity directions, and the positive/negative limit cases remained near
  +/-1 rad/s without termination or truncation.
- Lifecycle evidence binds completed `env.close()`, immediate
  `SimulationApp.close()` invocation, child exit zero, immutable child/ready
  evidence, stdlib-parent publication, zero srun status, exact L40S inventory,
  and the saved committed sbatch. Image SHA-256 remained
  `ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a`;
  PolaRiS-Hub revision and all three asset/metadata hashes revalidated.
- All eight immutable result files and the authoritative log were copied with
  identical hashes to
  `/home/lzha/code/shared_artifacts/polaris-pi05-native-controller-cap-1098204/`.
  Its mode-`0444` manifest has SHA-256
  `50fcd8ab562e8dc5c1cda18d04443f11ab7b46458fd5460f24e403ab958ec0db`.
  The job cache contained 30,703,445 bytes, 74 files, 45 directories, no
  symlinks, and a zero-byte `ov/_cache.lock`; it was removed and verified
  absent through both NFS aliases after terminal inspection. Remote source and
  OpenPI remained clean, detached, exact, and alternates-free.
- This completes the native controller/gripper-cap gate only. No policy
  server, model, checkpoint inference, checkpoint canary, or full evaluation
  was launched. Checkpoint work remains blocked pending separate main and
  native-evaluation review; job `1098204` cannot itself establish policy
  success or checkpoint promotion.
