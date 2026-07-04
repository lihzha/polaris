# pi0.5 native artifact recovery v2 — 2026-07-04

## Scope and immutable base

- Agent: `pi05-native-artifact-recovery-v2-20260704`.
- Worktree: `/home/lzha/code/PolaRiS-worktrees/pi05-native-artifact-recovery-v2-20260704`.
- Branch: `codex/pi05-native-artifact-recovery-v2-20260704`.
- Exact base and failed model-canary source:
  `7aed30fa1825fd39ee1d0227e7ca4c4f4f6f6154`.
- Exact OpenPI gitlink and initialized checkout:
  `bd70b8f4011e85b3f3b0f039f12113f78718e7bf`.
- Historical job `1098368` is read-only evidence. This branch will not launch,
  cancel, or update any job or shared registry record.

## Failure reproduced from preserved evidence

- The canary wrote 30 queries, 239 emitted actions, and 238 completed
  executions, then raised a generic `ValueError` from the all-13-DOF dynamic
  recorder during action 239.
- The recorder checked the velocity before updating maxima, appending the
  sample, or incrementing its apply counter. The offending joint, value, and
  physics substep were therefore not persisted.
- The evaluator handled only IK/linear-algebra numerical exceptions and kept
  environment and SimulationApp cleanup in straight-line tail code. The
  exception escaped, Kit did not shut down, the shell remained inside
  `srun | tee`, and video, metrics, terminal trace, close marker, srun status,
  completion, and eval-success artifacts were never published.

## Approved repair

- Add a typed all-joint velocity-limit exception and closed terminal incident
  evidence before raising.
- Treat only that type as a per-rollout numerical failure; preserve all other
  `ValueError` contract failures as fatal.
- Transactionally finalize failure trace, incident, video, sidecar, and CSV,
  with two exact terminal forms: full success-path completion or typed
  numerical failure.
- Put environment and SimulationApp shutdown in nested `finally` paths and
  preserve close-ready evidence for either complete terminal form.
- Keep checkpoint, normalization, images, state, action, sampler, and native
  controller semantics unchanged.

## Controller-gate consequence

The repair necessarily changes `scripts/eval.py` and
`src/polaris/native_gripper_runtime.py`, both critical bytes bound by accepted
no-model all-six controller job `1098349`. That completion cannot authorize a
model rerun from this branch. After independent review, a new exact-commit
no-model all-six L40S smoke and immutable completion must be produced and
rebound before another official pi0.5 model canary.

## Implemented recovery contract

- `NativeAllJointVelocityLimitError` is the only recoverable partial terminal
  type. The recorder captures and validates all 13 positions, velocities,
  accelerations, position/velocity targets, live and expected limits,
  thresholds, excess mask and magnitude, violating joint identities, and the
  exact policy/apply/physics-substep cadence. It publishes the immutable
  incident before raising.
- Healthy dynamic reports require exactly eight apply-entry samples plus one
  post-policy sample per outer step. A failed report permits exactly one
  terminal partial cadence and binds the missing failing sample through the
  immutable incident. Aggregate and sampled validators independently
  recompute the cadence and violation arithmetic.
- The policy client accepts only the exact typed failure, binds it to the
  pending emitted action, immutable incident, dynamic report, live simulator
  counters, and one terminal trace record, then seals the trace. Plain
  `ValueError`, observation/schema failures, and every other numerical error
  remain fatal.
- Each accepted terminal form publishes immutable trace and video artifacts,
  then an episode sidecar binding those artifacts, the dynamic report, the
  terminal outcome, and optional incident, followed by the immutable metrics
  CSV. The evaluator-close marker is published only after `env.close()`.
- Environment close, close-marker publication, and `SimulationApp.close()` are
  ordered through nested cleanup. Kit close is attempted even when environment
  close or publication raises, and multiple cleanup failures are preserved.
- The trace auditor and host finalizer accept exactly two terminal forms: a
  complete 450-action rollout, or the typed all-joint velocity-limit failure.
  A short typed-failure video is retained at its exact attempted-action frame
  count, while the shared summary is padded to the existing three-second
  minimum without changing the scientific result.

## Train/eval and source invariants

- Official checkpoint URI, OpenPI commit, global DROID normalization asset and
  probes, 224x224 unrotated external/wrist image order, zero blank third image,
  joint-position state, 15x8 response, execute-first-8 behavior, native
  velocity action transform, and sampler/server path are unchanged.
- An AST comparison against integrated base
  `3e9df7f605baa75848a0ad8edd2783d629d105c5` proves the checkpoint-facing
  image contract, action transform, argument contract, visualization,
  inference, resize, and observation-extraction symbols are unchanged.
- Exact runtime-source SHA-256 values were rebound to:
  - `src/polaris/environments/droid_cfg.py`:
    `2947a19d75d75229462debd0b7faddd4cce75e73ea67e2dbf41eefb3ae90467f`
  - `src/polaris/native_gripper_runtime.py`:
    `4130fd5e3e0a929627d002a002b5915f08da776f26e21a7980627d28e78e03f2`
- The next all-six completion will bind the repaired evaluator, lifecycle, and
  policy client in its exact source manifest. The eval-contract module remains
  part of the exact model-canary commit provenance because it contains the
  completion constants that must be rebound after the smoke; requiring its
  pre- and post-smoke bytes to match would create a circular gate. The official
  GCS checkpoint manifest and server entry point remain byte-identical to the
  integrated base.

## Validation

- PolaRiS Ruff check and Ruff format check: pass for every changed Python file.
- Host test suite, excluding only the Isaac-import-only
  `tests/test_robust_differential_ik.py`: `165 passed, 1 skipped`.
- Focused recovery/attestation suite: pass, including typed incident
  persistence, partial cadence mutation rejection, trace/sidecar mutation
  rejection, close ordering/error propagation, exact source manifests, and
  full and short H.264 video probes.
- `bash -n`, ShellCheck, Python compilation, and `git diff --check`: pass.
- No Slurm or model job was launched. The mandatory next execution is one new
  exact-commit, one-L40S, no-model all-six controller smoke. Only after its
  immutable completion is reviewed and rebound may an official pi0.5-DROID
  model canary be submitted.
