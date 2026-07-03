# Official pi0.5-DROID native PolaRiS canary handoff — 2026-07-03

## Scope and launch state

- This eval-only candidate was developed in the isolated worktree
  `/home/lzha/code/PolaRiS-worktrees/pi05-native-eval-canary-v1-20260703`
  from exact base `90d56b3b8d0a93ad7c48319a377d325790b89144`.
- It defines one fresh, non-resumable `DROID-FoodBussing` rollout: 450 policy
  steps at 15 Hz, official untouched
  `gs://openpi-assets/checkpoints/pi05_droid`, native 7-DoF Panda joint
  velocity plus closed-positive gripper, 15x8 response, execute-first-8.
- No checkpoint download, allocation, Slurm submission, inference server,
  simulator, evaluation, commit, push, or deployment was performed from this
  worktree. The candidate remains uncommitted pending independent review.
- Launch must occur only after this eval-only patch is integrated onto
  `5e2947fc900838715859cf8d2476410527737924`. That descendant contains the
  controller-critical native gripper cap attested by job 1098204. The
  integrated source must match job 1098204 for every controller-critical file.

## Train/eval contract

- Checkpoint content manifest SHA-256:
  `6f9ccfa5695c669962ad10dbe0dcb7d44bf903918e5fffe33e5d1ff531287922`;
  20 objects; 12,429,488,598 bytes; every object is fully MD5-verified.
- Normalization is the checkpoint-local global DROID file
  `assets/droid/norm_stats.json`, SHA-256
  `403b3a22f897e9ae5dd617966a3c8f7d1835ac79dfd5a8993179514be26a3b8b`.
  `single_arm` and related category substitutions are explicitly forbidden.
- Exact norm probes bind joint-velocity semantics: action q01/q99 first seven
  are `[-0.458, -0.8076, -0.4472, -0.9268, -0.6456, -0.646, -0.7616]` /
  `[0.4476, 0.7652, 0.448, 0.7944, 0.6484, 0.6628, 0.7344]`; gripper action
  q01/q99 is `0/0.9998`. State first seven probes are Panda joint-position
  ranges and gripper state q01/q99 is `0/0.991`. DeltaActions,
  AbsoluteActions, and joint-position action interpretation are rejected.
- Input state is exactly seven float32 Panda joint positions plus one
  closed-positive gripper value. Images use OpenPI's pinned 224x224
  aspect-preserving resize-with-pad: external to `base_0_rgb`, unrotated wrist
  to `left_wrist_0_rgb`, and zero/masked `right_wrist_0_rgb`.
- The render fix does not alter policy query cadence. Expensive rendering is
  `render_every_step OR needs_next_policy_render`; the client's execute-8
  counter/query logic is unchanged. A complete trace must contain exactly 57
  queries, 450 emitted actions, and 450 measured-execution records.

## Mandatory controller prerequisites

- Base controller-only gate: exact path
  `/lustre/fsw/portfolios/nvr/users/lzha/results/polaris-pi05-jointvelocity/controller-smoke/90d56b3b8d0a-controller-only-v3/controller-smoke-1098174.completion.json`,
  SHA-256
  `05403d0aabf3ebc8111cecf64d33f56f50a3a5673e7a84653ae096e7f4027ad3`,
  13,947 bytes, mode 0444, one link, runtime SHA-256
  `495ce92226ad0d1840138fc2b315fc2531d0ff50953fb16d70172080a8ee0b71`.
- Descendant gripper-cap gate: exact path
  `/lustre/fsw/portfolios/nvr/users/lzha/results/polaris-pi05-jointvelocity/controller-smoke/5e2947fc9008-controller-v2-gripper-cap/controller-smoke-1098204.completion.json`,
  SHA-256
  `778594b8eea64d6c2fb031d43af53539e07014af218e8e2c60751cb0d399a657`,
  15,744 bytes, mode 0444, one link, source `5e2947fc9008...`, runtime SHA-256
  `c7e932ca9f697cd02825fb06ee5fa5c0f168af73309026e91d844f16fd3729eb`.
- The job 1098204 validator independently recomputes the runtime digest and
  checks configured, CUDA actuator, and CPU direct-PhysX gripper velocity
  limits of 5 rad/s. It follows and hashes the immutable smoke, child-close,
  and ready artifacts; validates exact reset; and checks open/closed/boundary
  average slew magnitudes `4.978439211845398`, `4.971333146095276`, and
  `4.978439211845398` rad/s with arm residual velocity at most 0.001 rad/s.
  Both controller completions remain non-promotable controller-only evidence;
  they are prerequisites, not checkpoint evaluation results.

## New evaluation artifacts and provenance

- Pure immutable eval/model contract and mode-0444 canonical JSON helpers.
- Strict official-checkpoint verifier and explicit norm probe report.
- Exact OpenPI checkout/uv.lock, Python package inventory, JAX/JAXlib/NumPy,
  single-L40S, and x64-disabled inference environment capture.
- Immutable serving handshake, runtime contract, evaluator-close-ready marker,
  run/submission records, saved batch script, exact commands, srun status, GPU
  inventory, source hashes, container hash, and PolaRiS-Hub asset hashes.
- Strict 959-record trace audit: one post-reset counter record, 57 queries,
  450 emitted actions, 450 measured executions, and one terminal record.
  Image/state/action/target/continuity checks remain fail-closed.
- Every action/execution trace pair also binds the measured normalized gripper
  position before/after, enforces `[0, 1]`, and proves exact per-step and query
  continuity. Physical target/drive/slew semantics remain independently bound
  by the live runtime contract and job 1098204.
- An immutable server-side model-runtime artifact records the resolved official
  train config, checkpoint identity, exact transform classes/parameters,
  checkpoint-global quantile normalization, empty sample kwargs, initial RNG,
  and imported OpenPI module attestation.
- Episode and summary video validation: H.264, yuv420p, progressive, 448x224,
  15 fps, 450 frames, 30 seconds, full decode; summary MP4 additionally
  requires `moov` before `mdat` fast-start ordering.

## Terminal-boundary and sensor-liveness correction

- The evaluator's outer contract remains exactly 450 actions. Before
  `gym.make`, it sets `episode_length_s=451/15`, and immediately after
  construction it requires the live `max_episode_length` to be exactly 451.
  The configured/live timeout contract and digest are carried by the runtime,
  trace, evaluator-close, trace-validation, and final completion artifacts.
- All 450 `terminated` and `truncated` values must be exact false values.
  `record_execution` checks those flags before accepting the returned
  observation, then requires live episode length `step+1`; this rejects the
  Isaac Lab auto-reset result at the step-450 boundary instead of recording a
  reset observation as the final state.
- The final measured-execution record, environment counters, and rubric are
  captured and cross-bound while the environment is still live, before
  `env.close()`. The terminal episode length must be 450 while the internal
  timeout remains 451.
- Sensor liveness uses the simulator's integer
  `isaaclab.sensors.camera.Camera.frame` counters for `external_cam` and
  `wrist_cam`. Each must advance exactly once per outer action for all 450
  actions. Query-image hash diversity remains descriptive and is explicitly
  non-authoritative, so a visually static but freshly rendered scene passes.

## Validation completed locally

- `ruff format --check` and `ruff check` on all candidate Python files: pass.
- Byte compilation via `compile(...)` on all candidate implementation and
  focused-test sources (without writing bytecode): pass.
- `bash -n` and `shellcheck` on eval shell, sbatch, and submitter: pass.
- Focused timeout/client/trace/finalizer suite: 49 passed. This includes
  synthetic step-450 timeout/auto-reset rejection, stale/reset camera-counter
  rejection, static image hashes with live camera counters, terminal-rubric
  cross-binding, and internal-max mismatch failures.
- Repository `tests/` excluding the local-environment-only Isaac Lab import
  test: 117 passed, 5 subtests passed, 3 third-party warnings. Both candidate
  OpenPI source roots were placed first on `PYTHONPATH` so provenance tests
  attested this worktree rather than the shared venv's checkout.
- Unconstrained bare `pytest` cannot be a valid local gate: the available
  non-Isaac venv lacks `isaaclab` for `test_robust_differential_ik.py`, and
  unconstrained discovery also enters the OpenPI submodule. No test failure
  was observed after collection was correctly scoped.
- The exact cached official norm file and all four reference-probe vectors
  were verified live. Remote artifacts 1098174/1098204 were read-only checked
  for exact size/mode/link/canonical bytes; job 1098204 runtime digest and
  drive profile were recomputed, and its child-close/ready derivation and arm
  residual limits were independently confirmed.

## Required integration validation before launch

1. Integrate this eval-only candidate after `5e2947fc9008...` in a clean,
   detached, standalone checkout and resolve the intentional `scripts/eval.py`
   and `src/polaris/config.py` overlap explicitly.
2. Rerun the complete host suite plus the Isaac/container tests against the
   integrated tree. Confirm all controller-critical hashes equal job 1098204,
   and run a real one-step/short-rollout Isaac canary proving the pinned
   `Camera.frame` counters follow the exact one-increment policy cadence.
3. Independently review the frozen diff. Only after GO may it be committed and
   pushed for handoff. A separate GO is required before the one-rollout L40S
   canary is submitted.

## Clean promoted-base integration — 2026-07-03T17:13:34Z

- CODEX agent: `pi05-native-eval-integrated-v1-20260703`.
- Fresh integration worktree:
  `/home/lzha/code/PolaRiS-worktrees/pi05-native-eval-integrated-v1-20260703`;
  branch `codex/pi05-native-eval-integrated-v1-20260703`; exact base
  `12bfc723b6c4c67df0e98e7339961ca2e6a0b216`, whose implementation ancestor
  `5e2947fc900838715859cf8d2476410527737924` is the job-1098204 promoted
  native-gripper-cap source.
- The 18 candidate files were ported deliberately from the untouched dirty
  candidate. Before the integration-only correction, every destination file
  was byte-identical to its source. The candidate's tracked diff SHA-256
  remained
  `809b106c3318bafb6c4ddcbecaa0eb1ef341964fe25bea642e17f2091fe37721`.
- The overlapping `scripts/eval.py` and `src/polaris/config.py` were resolved
  against the promoted base. Both retain the job-1098204 gripper-drive-profile
  validation while adding the eval runtime and close-lifecycle paths.
  `PI05_DROID_GRIPPER_DRIVE_PROFILE` now aliases the imported promoted
  `NATIVE_GRIPPER_DRIVE_PROFILE`, so an integration on the pre-cap base fails
  at import instead of silently duplicating a string.
- The candidate's rendering expression was narrowed after review:
  `render_every_step OR policy_client.rerender` is used only when the exact
  client is `DroidJointVelocity`. Every other client returns
  `policy_client.rerender` unchanged; the regression test verifies object
  identity for `DroidJointPos` and `EgoLAPEefPose`.
- The pinned OpenPI submodule was initialized at exact commit
  `bd70b8f4011e85b3f3b0f039f12113f78718e7bf`. No new virtual environment was
  created; host tests used the existing Python 3.11 environment from
  `pi05-polaris-eval-20260701` with this integration tree first on
  `PYTHONPATH`.
- First focused attempt: 95 passed and 8 failed closed because the fresh
  worktree's OpenPI submodule path was not yet a Git checkout. After exact
  submodule initialization, the same focused command passed: 103 passed,
  one third-party deprecation warning.
- Broad host suite, scoped to `tests/` and excluding only the real-Isaac
  `test_robust_differential_ik.py`: 129 passed, 1 skipped, one third-party
  deprecation warning.
- Ruff check and format-check passed on 14 changed Python files; all 14 compiled
  in memory; `bash -n` and ShellCheck passed on all three changed shell/sbatch
  files; `git diff --check` passed.
- All nine controller-critical files that are logically unchanged match the
  local immutable copy of job 1098204's completion attestation by both size and
  SHA-256, and `git diff 12bfc723 -- <nine paths>` is empty. Their attested
  hashes are:
  - controller finalizer:
    `e3af1618e15808f71022ea493d36031f1307ccb02fbf6204d4cf0b3ccdfc07c8`
  - controller sbatch:
    `addfe38768c1ee6ee358151b03dcba7fe0cd1919a955286656ca89770910ce0e`
  - controller submitter:
    `61bd5294ee0dc07a74b5b0a765f7e1a504b4f0a471f6d7ed9d1ca4e0bdf04ce4`
  - controller simulator:
    `ed2149b325e82f0d5f0e1bfecdb0ccf33ccae12464296f3d809cb6d23043cc2c`
  - DROID environment config:
    `111c34d8d707f6edf31e9166c9aafd999ff3e7ea72344fc31fe5c9b8d6e175ee`
  - robot config:
    `d514b32e07b54f98deb6d9dbc7a5201fff5337cdc4600d9351ee2a95e5c4c4c5`
  - joint-velocity runtime:
    `5bb779b99c4df115b7fca326f6e8c8ff71fa89c207c51e0a90bffe55a96a1ccb`
  - joint-velocity smoke contract:
    `f74b42660301e062e63371edee2591b77478e58063ee5d152e81fa192c6c1261`
  - native joint-velocity model/controller contract:
    `cacf2ea79bc2bced0c1ea402c00be62ac1b5ea2f9d25b5f8b00f5476549fccee`
- No checkpoint was downloaded and no allocation, Slurm job, server,
  simulator, evaluation, transfer to a cluster, launch, or cancellation was
  performed. The next runtime action remains a separately approved one-rollout
  L40S canary after this host-only branch is reviewed and promoted.
- An independent read-only review reported no P0, P1, or P2 findings in the
  eval/config overlap, exact-client render isolation, gripper-cap integration,
  or spot-checked launcher/finalizer chain. Its independent focused selection
  passed 54 tests plus diff, Bash-syntax, and in-memory compilation checks.
