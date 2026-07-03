# pi0.5 native all-six controller gate — 2026-07-03

## Scope and blocker

- Isolated descendant branch: `codex/pi05-native-all-six-v1-20260703` from
  PolaRiS `3e9df7f605baa75848a0ad8edd2783d629d105c5`.
- Job `1098204` attested only actuator-owned `finger_joint`: configured,
  CUDA actuator, and CPU PhysX velocity limits were each scalar `5 rad/s`.
  It did not enumerate or dynamically sample the five passive Robotiq mimic
  followers.
- The same pinned robot USD (`d8379925...`) and driver-only/no-write semantics
  were measured by the later follower matrix: the live indices 7:13 were
  `[5, 174.53292846679688, 174.53292846679688, 174.53292846679688,
  174.53292846679688, 174.53292846679688] rad/s`. Uncapped followers reached
  `55.62232208251953 rad/s` and `7886.1845703125 rad/s^2`. The official model
  canary therefore remains blocked.

## Candidate implementation

- A native-only reset event uses Isaac Lab's public
  `Articulation.write_joint_velocity_limit_to_sim` path after
  `reset_scene_to_default` on every reset. It performs the proven full-13-DOF
  read/modify/write, preserves arm plus driven-finger values bitwise, writes
  follower indices 8:13 to `5 rad/s`, and requires exact CUDA-buffer/CPU-PhysX
  readback and write/reset counts. The public writer receives the complete
  `1x13` replacement tensor with `joint_ids=None`; arm plus driver are checked
  unchanged before and after the call.
- The exact six names/order are `finger_joint`, `right_outer_knuckle_joint`,
  `left_inner_finger_joint`, `right_inner_finger_joint`,
  `left_inner_finger_knuckle_joint`, and
  `right_inner_finger_knuckle_joint`. The source-USD PhysX mimic identity and
  actuator ownership are closed contracts.
- An audit-only subclass retains upstream `JointVelocityAction` processing.
  The existing binary finger action records after both action terms have
  applied. The recorder samples all 7 arm and 6 gripper DOFs at each of eight
  physics apply entries plus the post-policy boundary, checks finite
  positions/velocities/targets/accelerations, and fails on current physical
  velocity-limit violations. Its named `5e-5 rad/s` PhysX allowance covers the
  already observed healthy `5.000018119812012 rad/s` float/solver boundary but
  remains far below the prior `55.622322 rad/s` follower failure.
- The no-model coupled smoke has four fixed 12-step cases: immediate/delayed
  close and immediate/delayed open while all seven arm commands move. It
  requires 96 apply calls plus 12 post-policy samples per case, immutable
  child-close/ready/final lifecycle evidence, measured motion of all seven arm
  joints, signed nontrivial driver/follower displacement, approximately 1:1
  mimic coupling, and endpoint approach. The post-srun host finalizer reopens
  and cross-binds the canonical mode-0444 raw, ready, and final files and binds
  source, one L40S, container, asset bytes plus Hub revision metadata, wrapper,
  and srun exit.
- Runtime attestation additionally pins the exact Isaac Lab 2.3
  `ActionManager`, `EventManager`, `ManagerBasedEnv`, and `ManagerBasedRLEnv`
  sources plus the active PolaRiS splat-env override, so reset ordering and the
  eight-action-apply cadence are part of the closed contract.
- Policy serving, `DroidJointVelocityClient`, OpenPI gitlink, checkpoint
  manifest, normalization/environment contract, image transforms/order, and
  model runtime capture remain byte-identical to the base commit. The new
  completion finalizer rechecks those identities against `3e9df7f`.

## Host validation before freeze

- Focused native/controller/model-contract suite after the independent-review
  fixes: `60 passed`.
- Full non-Isaac PolaRiS suite after those fixes:
  `151 passed, 5 subtests passed`; the Isaac-only robust-DIK test is excluded
  because the host environment does not provide Isaac Lab.
- New mutation coverage rejects a frozen robot, a decoupled follower, velocity
  values outside the narrow PhysX allowance, missing child captures, and a
  post-publication-mutated ready marker.
- Ruff check/format, Python compilation, Bash syntax, ShellCheck, Git diff
  check, exact OpenPI gitlink/checkout, forbidden model/checkpoint/network
  token scan, and base-byte comparison for the official model-I/O paths pass.
- The first independent review was `NO-GO` and all reported P1/P2 findings were
  patched. The second read-only review returned `GO` with no residual P0/P1/P2
  findings, scoped only to the no-model controller smoke. No L40S coupled smoke
  and no checkpoint/model evaluation has been submitted from this branch.

## First exact-commit L40S attempt and source-identity repair

- Exact deployed commit `ad3274a47e7c4c515b48d7296b1e742a02fd6e82`
  ran as the sole no-model job `1098328` on one L40S. It terminated `FAILED`
  with exit `1:0` after 2:53 at `validate_runtime`; no model or checkpoint was
  loaded.
- Immutable failure evidence reports that the `actions_cfg.py` check observed
  SHA `19ceacee...`, the exact PolaRiS `droid_cfg.py` digest. The verifier had
  passed the custom `AuditedDroidJointVelocityActionCfg` directly to the
  upstream Isaac Lab source check, so it compared the correct custom source to
  the distinct pinned upstream `actions_cfg.py` digest `94722a...`.
- The minimal repair now resolves
  `isaaclab.envs.mdp.actions.actions_cfg.JointVelocityActionCfg` from the custom
  config's MRO for the upstream source hash. The PolaRiS source check explicitly
  requires the custom arm term, custom config, and binary finger term to share
  the pinned `droid_cfg.py` source. Runtime and serving contracts now record the
  custom action-config identity separately from its upstream base identity.
- Focused source/runtime/contract/all-six suite: `60 passed`. Broad non-Isaac
  suite: `152 passed, 5 subtests passed`. Changed-file Ruff, formatting,
  compilation, and `git diff --check` pass.
- Job evidence is mirrored under
  `/home/lzha/code/ego-lap/.codex_artifacts/pi05-native-all-six-controller-ad3274a/job_1098328`;
  the detailed operational log is the sibling `worklog.md`. A relaunch remains
  forbidden pending an independently reviewed frozen repair.
- Main-agent frozen-diff review added an explicit `#SBATCH --no-requeue` gate
  (plus regression assertion), so a failed smoke cannot silently recreate an
  allocation. The correctly rooted broad suite independently reproduced
  `151 passed, 5 subtests`; the targeted all-six/runtime/OpenPI suite passed 57
  tests. Bash, ShellCheck, Ruff, format, diff, exact OpenPI checkout, and
  byte-unchanged official serving/client/manifest/native-contract checks were
  rerun after that scheduler-only change.

## Accepted rerun and model-canary gate preparation

- Minimal source-identity repair commit
  `93083d2694b8638de30e970e3bea450526593e7e` passed as sole rerun job
  `1098349` on one L40S (`COMPLETED 0:0`, 3:08). The immutable controller-only
  completion is
  `/lustre/fsw/portfolios/nvr/users/lzha/results/polaris-pi05-native/all-six-controller-smoke/20260703T201500Z-93083d2-all6-smoke2/native-all-six-smoke-1098349.completion.json`,
  SHA-256
  `a03ffbf0745327ce604db7be5928a94dce910eac91a8766c21f8efcb71fea867`,
  size 7502, mode 0444, one link. Its runtime SHA-256 is
  `9a0597d62debc01fbde064360f9845a28a2df06fd2853ff0b3556dff48c14efc`.
- The follow-up official model-canary contract now replaces obsolete job
  `1098204` as descendant authority with job `1098349`. It reopens the exact
  all-six final/raw/ready artifacts, all 432 per-substep samples, all four
  open/close scenarios, all-six velocity/coupling evidence, srun status, and
  one-L40S inventory. It also requires the current descendant to retain every
  one of the 12 controller-critical source hashes and the unchanged official
  serving/client/checkpoint-manifest bytes attested by job `1098349`.
- The model-canary launcher/run-record interface is renamed from the stale
  `GRIPPER_CAP_*` vocabulary to `ALL_SIX_*`; its completion will expose the
  gate as `controllers.native_all_six_coupled`. The live canary runtime must
  exactly equal the accepted all-six runtime SHA above. Historical job
  `1098174` remains a separate arm-controller prerequisite, but no longer acts
  as descendant source authority.
- Host validation: focused gate/lifecycle suite `22 passed`; broad non-Isaac
  suite `152 passed, 5 subtests passed`; Ruff, formatting, Python compilation,
  Bash syntax, and Git diff checks pass. A read-only preflight using temporary
  audit code on `l401` reopened the actual job `1098174` and `1098349`
  completions and passed every source, lifecycle, runtime, GPU, and policy-I/O
  binding. The temporary audit directory was removed. No model, checkpoint,
  GPU allocation, or Slurm job was launched for this preparation.
- Independent frozen review returned GO with no P0/P1 findings. Its sole P2
  noted that live l401 defaults to requeue; the model-canary wrapper now pins
  `#SBATCH --no-requeue`, with an exact one-occurrence regression assertion,
  so an infrastructure failure cannot consume an automatic second allocation.
