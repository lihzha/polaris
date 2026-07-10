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
