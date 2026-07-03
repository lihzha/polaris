# Main-branch Ego-LAP adapter rebuild

## 2026-07-02 — implementation

- Agent: `polaris-adapter-20260702`.
- Branch: `codex/main-lap-adapter-20260702`.
- Worktree: `/home/lzha/code/PolaRiS-worktrees/main-lap-adapter-20260702`.
- Base: official PolaRiS `origin/main` commit
  `2f4046bfe9e0b6a7ce5f86f76c7964e16c3238b4`.
- Goal: rebuild the validated Ego-LAP absolute EEF adapter from reference
  `b53b2db3a20e4d858351eba0926d1eb7cf9b3839` without replay diagnostics,
  historical result logs, cluster launches, or external repository changes.

### Implemented scope

- Restored the validated `panda_link8` observation/control frame, one-anchor
  absolute action conversion, exact wrist/image preprocessing, robust DLS
  failure containment, durable episode CSV/video loop, and controller smoke.
- Added authoritative websocket metadata validation and exact contract
  persistence before rollout. CLI values are assertions only.
- Added distinct mode protocols: flow `16x7 -> execute 8`; AR `1x7` total
  endpoint -> 8 inclusive cumulative targets -> execute all 8.
- Kept `DroidJointPos` plus `joint-position` as defaults and added pure
  regression tests for that path.
- Added only the direct SciPy dependency edges required by the pose adapter;
  no Docker/Pyxis/runtime recipe was changed.

### Validation

- The PolaRiS local environment does not include `pytest`; the pure suite was
  run with the existing Ego-LAP Python environment and this worktree on
  `PYTHONPATH`.
- Final pure adapter suite: `26 passed, 5 subtests passed`. This includes exact
  TensorFlow resize oracles, frame/action conversion, flow/AR protocols,
  metadata fail-closed checks, exact contract persistence, top-level contract
  digest tamper detection, and the legacy joint-mode regression.
- Generated six contracts directly from the current Ego-LAP contract
  implementation and passed them through the PolaRiS validator: public LAP
  train-matched and legacy-Q99 profiles, plus a modern manifest profile, each
  in flow and AR mode. Their response/execution horizons validated as `16/8`
  and `1/8`, respectively, including recomputed top-level identities.
- Ruff format/check, Python byte compilation, and `git diff --check` passed for
  the changed Python/source files.
- `uv lock --check` reaches the known upstream `flatdict==4.0.1` source-build
  failure (`ModuleNotFoundError: pkg_resources`). The lockfile therefore keeps
  only the already-validated direct SciPy dependency edges; no unrelated
  runtime recipe workaround was added.
- Isaac-only robust-DLS unit tests and the controller smoke were not run because
  the local environment has no Isaac Lab runtime and this scoped handoff did
  not authorize a cluster/container launch. They remain required canary checks
  before production rollouts.
- No simulator, GPU, cluster, policy-server, monitor, shared-registry, or
  external repository process was launched. No branch was pushed or merged.

### Handoff

- Implementation commit: `b2654c379d8c1d04ebfb6a7ae7c122de101bd3e3`.
- The orchestrator can review and integrate the branch
  `codex/main-lap-adapter-20260702`; this agent intentionally did not push,
  merge, or modify the canonical PolaRiS checkout.

## 2026-07-02 — independent production-hardening review

- Agent: `polaris-adapter-review`.
- Reviewed implementation commits `b2654c3` and `cf9b505` against the strict
  checkpoint parity, EEF-control, and requeue artifact contracts.
- Made PolaRiS numeric actions unconditionally robot-base so an egocentric
  language frame can never trigger a second EEF-to-base conversion.
- Bound Q99 profiles to exact formula IDs/constants and recomputed the nested
  normalization-formula, normalization-stats, live-execution, and top-level
  canonical JSON digests. Global normalization preserves configured category
  metadata while keeping the effective PolaRiS category null.
- Added atomic episode video/trace/CSV publication, contiguous artifact-aware
  resume validation, global episode IDs, and partial-trace reconciliation for
  Slurm requeue.
- Added automatic 450-step/15-Hz checks and a first-reset live articulation
  check against physical `panda_link8`, the installed action term, and identity
  controller offset.
- Pure validation after the changes: `42 passed, 5 subtests passed` across the
  adapter, contract, image oracle, eval mode, artifact, and runtime suites.
  Six live contracts built from the paired Ego-LAP implementation also passed:
  public LAP train-matched and legacy-Q99 plus modern manifest train-matched,
  each in flow and AR mode. Ruff check/format and `git diff --check` passed.
  Isaac-only controller smoke remains a required cluster canary; no simulator
  or cluster job was launched in this scoped implementation task.

## 2026-07-02 — checkpoint-specific R6 and runtime evidence fix

- The public `original_lap_public_3b_v1` profile now fails closed unless both
  contract locations select `xyz+r6_first_two_rows+gripper_open` with mode
  `public_lap_train_matched_rows_v1`, matching the public checkpoint's training
  implementation. Manifest-backed profiles require the newer first-two-column
  layout and `manifest_train_matched_columns_v1` mode.
- The client builds state from the validated layout rather than a fixed helper
  convention and records the layout plus mode in every policy-query trace.
  An asymmetric cyclic rotation regression proves the exact, distinct six-value
  row and column order; client tests prove public and manifest profiles send the
  corresponding state bytes.
- `scripts/eval.py` now atomically writes schema-1 runtime evidence containing
  the exact 450-step/15-Hz protocol and live `panda_link0 -> panda_link8`
  observation/controller facts. A fully resumed task resets and validates the
  live controller, writes the evidence, and only then takes its early return.
- Pure validation: `45 passed, 7 subtests passed` across the adapter/contract,
  resize oracle, eval mode, artifact, and runtime suites. Ruff format/check,
  Python byte compilation, and `git diff --check` passed. The unfiltered suite
  still cannot collect the Isaac-only robust-IK test in the local non-Isaac
  environment; no simulator or cluster job was launched by this scoped fix.

## 2026-07-02 — official AR endpoint interpolation parity

- Agent: `ar-parity-audit`.
- Audit evidence: official LAP commit
  `3958d1466d5b92445b67de7d4202c19608ad4d56` constructs a single-action
  translation chunk with inclusive `np.linspace(current, target, steps)` and
  samples rotation SLERP at inclusive `np.linspace(0, 1, steps)`. The PolaRiS
  implementation instead used `1 / steps .. 1`, so its first four targets
  reached `4/16 = 25%` of an AR endpoint instead of the official-style
  `3/15 = 20%` on a 16-target grid.
- Implementation commit `0907409d3215263f3c2cfe30a31b70d84e05331a`
  changes only the AR interpolation fractions to inclusive `0..1`. The first
  target is now the unchanged query-time anchor and the last remains the full
  endpoint. Existing one-time anchoring and `R_anchor * R_delta`
  right-multiplication are unchanged, and the endpoint gripper target remains
  held across all targets.
- Focused client assertions now cover the zero-motion first target, `1/15` and
  `3/15` intermediate fractions, full endpoint, held gripper, and the four
  absolute actions actually emitted before replanning. Documentation records
  the inclusive grid and retained rotation composition.
- Validation at `2026-07-02T11:28:59-07:00`: the complete non-Isaac PolaRiS
  adapter/contract/artifact suite passed with `50 passed, 23 subtests passed`;
  Ruff check passed; Ruff format check reported both Python files formatted;
  Python byte compilation and `git diff --check` passed.
- Live-impact audit of the shared Ego-LAP registry found every requested or
  active evaluation in flow mode: 129 planned, 4 queued, and 2 running, with
  zero AR evaluations. The four live queued PolaRiS children
  `1096052`, `1096057`, `1096062`, and `1096065` are flow canaries. No live
  flow job, watcher, simulator, registry record, or external checkout was
  modified or relaunched for this AR-only fix.

## 2026-07-02 — final shared-contract parity audit

- Corrected the prior AR timing assumption after checking the official LAP
  runner end to end: one AR endpoint is expanded to 8 inclusive targets and
  all 8 are executed before replanning. The prior 16-target/4-action profile
  reached only 20% of the endpoint and is no longer accepted.
- The named AR profile now uses SO(3) identity-to-delta interpolation followed
  by query-anchor right multiplication. This matches the training definition
  `R(anchor) @ R(delta)` and is explicitly distinct from the legacy official
  real-robot helper's coupled-axis-incorrect Euler-add endpoint.
- Added fail-closed policy-input float32 state metadata and request dtype
  assertions. Flow traces now record null AR interpolation metadata instead of
  the confusing internal model horizon.
- The server handshake now binds the binary model-open `>0.5` gripper execution
  profile and threshold. This is cross-checked with the existing PolaRiS
  closed-positive `>=0.5` runtime profile. The client itself now emits only
  binary closed-positive values, including equality-close boundary coverage,
  so anchored chunks, traces, and controller inputs all agree.
- Final non-Isaac validation: `60 passed, 2 skipped`; Ruff format/check and
  `git diff --check` pass. The two skips and the uncollected robust-IK module
  require the pinned Isaac/Pyxis cluster runtime and remain part of the clean
  deployment canary gate.

## 2026-07-02 — EEF IK safety v3 remediation (in progress)

- Three preserved v2 BlockStack rollouts independently reached a non-finite
  Jacobian after an unbounded DLS joint target: episode 12/action 360,
  episode 38/action 242, and episode 47/action 216. Historical v2 results are
  not rewritten and are explicitly superseded for future publication.
- The requested v2 diagnostic jobs are now terminal: BlockStack completed
  50/50 with 0 successes, mean progress 0.1429, and three numerical failures;
  FoodBussing completed 50/50 with 0 successes, mean progress 0.1467, and no
  numerical failures. Both marker trees remain preserved as non-publishable v2
  evidence, and the diagnostic monitor was stopped after terminal inspection.
- The opt-in EEF controller now installs explicit Panda arm simulation
  velocity/effort limits, bounds every target at the 120-Hz physics-substep
  cadence, clamps to the exact float32 7x2 soft limits, and aborts current-state,
  post-clamp, or non-finite violations before the PhysX target setter. The
  inactive guard path preserves the inherited target bit-for-bit.
- Per-episode reports isolate apply/slew/position/abort/fallback counters,
  per-joint maxima, the worst raw DLS update, and bounded finite-or-null guard
  diagnostics. One named `1e-6` rad float32 tolerance is shared by controller
  and contract checks.
- Schema-2 runtime evidence is reconstructed from immutable per-episode safety
  transactions binding the exact CSV row, video, and finalized terminal trace.
  Resume can commit one prepared next row after validation and archives any
  uncommitted artifacts by content hash rather than deleting them.
- Pure validation currently passes the focused runtime/artifact suite. The
  standalone Isaac controller smoke, exact live soft-limit recapture,
  3600-apply-call full-horizon canaries, wall-time comparison, and downstream
  Ego-LAP completion review remain mandatory before any v3 checkpoint result is
  publishable.
- Prelaunch review expanded the standalone smoke to hold plus all ±XYZ/±RXYZ
  directions and a bounded one-step oversized-target phase. The adversarial
  phase must record exactly eight apply calls, at least one slew event, finite
  state/strict-JSON evidence, no abort/post-clamp violations, and per-joint
  applied maxima no larger than `velocity/120 + 1e-6`, followed immediately by
  reset. The final gate now also captures live Panda arm joint position and
  velocity values/masks and requires both joint and EEF state to remain finite.
- The controller now rejects finite but non-unit current or desired EEF
  quaternions before pose-error computation using the profile-bound `1e-3`
  norm tolerance. Strict evidence validation closes the sidecar, aggregate,
  artifact-identity, diagnostic, and counter schemas; completed rollouts also
  bind the max-raw diagnostic to aggregate maxima. Float32 state/action casts
  and JSONL writes are rechecked so huge finite inputs cannot become Inf/NaN
  artifacts.

## 2026-07-02 — pinned Isaac robust-IK test audit

- Tested code commit: `4f1ce3c7c95ee298291bb6860f2c67f8015a2abc` in pinned
  Pyxis image SHA-256
  `ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a`.
- Infra attempt `1097385` failed before collection because `sbatch --wrap`
  generated `/bin/sh` and rejected `set -o pipefail`. Attempt `1097391`
  reached collection but exposed only the outer IsaacLab CLI namespace and was
  cancelled after `isaaclab.controllers` was missing. Explicit-Bash attempt
  `1097393` reproduced that missing nested source-root failure. Bounded
  preflight `1097396` then showed that adding the core source root still leaves
  `omni` unavailable until Isaac Sim starts.
- Final tests-only job `1097402` used an immutable external bootstrap (SHA-256
  `836cf8131b42165b8525d0c673e76f570ed057f8d5f2e5172a2eba6879c063f8`),
  all three nested Isaac source roots, and `AppLauncher(headless=True)` before
  pytest collection. It completed `17 passed`, exit `0:0`, on `pool0-00005`.
  JUnit:
  `/lustre/fsw/portfolios/nvr/users/lzha/results/polaris_eval/controller_safety_v3_review/4f1ce3c/pytest-1097402.xml`
  (SHA-256 `88f0a47d53cc0456808a6b9772d586c7d76654831cbc72b6567347bb79ee07b2`).
  Full log:
  `/lustre/fsw/portfolios/nvr/users/lzha/slurm_logs/polaris_eval/pol_ik_v3_test-1097402.out`
  (SHA-256 `0a0c55214d597a7d9c0e9ca55ed5decfe0e141530e2f1cc2dceacd046829bbaa`).
  Isaac emitted headless Vulkan/display warnings, including an incompatible
  Vulkan-driver message, but the CPU-focused unit suite and JUnit were clean.
  No standalone controller smoke or checkpoint evaluation was chained.

## 2026-07-02 — first standalone controller-smoke capture

- Reviewer-approved standalone smoke job `1097411` ran PolaRiS commit
  `4f1ce3c7c95ee298291bb6860f2c67f8015a2abc` in the same pinned image,
  with the production NVIDIA Vulkan ICD mapping and PolaRiS data/cache mounts.
  It did not start a policy server, load a checkpoint, or chain an evaluator.
- The job failed closed after `00:02:56` because no result JSON was produced.
  Its Python `srun` step reported `0:0`, but Isaac shutdown suppressed the
  exception that occurred after the initial live-limit capture. The log is
  `/lustre/fsw/portfolios/nvr/users/lzha/slurm_logs/polaris_eval/pol_ik_v3_smoke-1097411.out`
  (SHA-256 `6a2ba56e10bfd420bcb81489d5203666327c80f172007e5b7ba157721a1ff7d7`).
- The live float32 Panda limits differed from the candidate only at joint 4's
  upper bound (`-0.06979990005493164`) and joint 6's lower bound
  (`-0.017499923706054688`). Independently hashing the 14 little-endian
  float32 endpoints reproduced the live digest
  `fbf7535901c042fea5d901812ecd02c5fd81ade06c23c1499c32d66a859104de`.
  Those exact values and digest are now the candidate constants, but their
  status remains `pending_controller_smoke` until a clean rerun validates the
  full matrix and writes strict JSON.
- The smoke now validates the raw capture immediately before any motion and
  records the active stage/case. Its required output is atomically written as
  strict JSON before teardown and finalized after teardown; failures retain
  the raw capture, completed partial cases, exception type/message/traceback,
  and separate environment/SimulationApp close errors. Tracebacks are flushed
  before close and the intended nonzero exit survives teardown. The next
  immutable sbatch must parse the JSON after Python and fail unless it is
  finalized, passed, and free of run/close/persistence errors; file existence
  alone is deliberately insufficient.

## 2026-07-02 — standalone smoke `1097498` (failed publication gate)

- Job `1097498` ran exact PolaRiS commit
  `dbfc41760cd2d9c5f8f97a5ab7fb8a33d488302d` on one L40S with the pinned
  image/Vulkan/data contract and no model, checkpoint, server, evaluator, or
  downstream chain. All 13 ordered hold/±XYZ/±RXYZ cases passed with 360
  physics-substep apply calls each. The one-step adversarial case also passed:
  eight apply calls, eight slew events, finite EEF/q/dq, q inside captured
  limits, and zero abort or post-clamp violations.
- The attempt remains failed and non-promotable. `SimulationApp.close()`
  cleanly terminated the Python process before its post-close rewrite, so the
  saved JSON correctly remained `finalized=false`, `passed=false`, stage
  `close_environment`. The external verifier rejected it, and the batch ended
  `FAILED 1:0` after `00:04:18`; its `srun` step was `COMPLETED 0:0`.
- Preserved raw JSON:
  `/lustre/fsw/portfolios/nvr/users/lzha/results/polaris_eval/controller_safety_v3_smoke/dbfc417/smoke-1097498.json`
  (99,784 bytes, SHA-256
  `b0907b132c0872e26245fae8f2ad99c2ce95247abc46b5d0c22f52efd075f861`).
  Preserved log:
  `/lustre/fsw/portfolios/nvr/users/lzha/slurm_logs/polaris_eval/pol_ik_v3_smoke-1097498.out`
  (SHA-256 `b1e8d6bb5650a879fda3af9064f12d1fdd1b3be218dc99725ba64977430a8482`).
  The immutable job script SHA-256 was
  `0f137d8471bdb0c47c39cc4e85bf5dcb92948b54540a8aca859a44414088124a`;
  verifier SHA-256 was
  `34d4289543a103a26324f9ab9e59597a6ca59cfb495536b7242d20504268ee17`.
- The next revision uses two-phase teardown. After `env.close()`, a clean run
  atomically publishes exactly one 0444 raw JSON at stage
  `simulation_app_close_pending`, fsyncs the file and parent directory, and
  atomically publishes a 0444 ready marker binding its path, size, and SHA-256.
  Only that exact durable path calls `SimulationApp.close()`; all earlier
  failures skip the hard-exit call so their nonzero status cannot be masked.
  After `srun` returns zero, a saved hashed host finalizer must independently
  validate the raw bytes, ready marker, full smoke evidence, commit/source/
  image/job provenance, and then create a separate non-overwriting 0444 final
  attestation. It never rewrites the raw result. Capture status remains
  `pending_controller_smoke`, and no rerun may start before renewed review.
- Reusable source finalizer `scripts/finalize_eef_pose_smoke.py` is stdlib-only
  for host execution. It enforces closed raw/result/controller/diagnostic/
  adversarial schemas; exact cadence, controller constants, limits, digest,
  errors, frame checks, finite evidence, maxima and diagnostic consistency;
  zero fallback/abort/post-clamp/drop counters for promotion; ready-marker and
  0444 modes; zero `srun` status; and commit/source/image/job provenance. It
  atomically publishes a non-overwriting 0444 attestation and rereads/hashes it
  in both `finalize` and `verify` modes. Unit tests cover lifecycle, schema,
  limits, counters, errors, diagnostics, raw/marker mutation, modes, `srun`,
  non-overwrite behavior, and provenance tampering.
- Follow-up review rejected the first two-phase revision before launch and
  tightened it further. The ready-marker publish now calls SimulationApp close
  on the immediately following statement; every diagnostic print/flush occurs
  before marker publication, every failure log is best-effort, and live-app
  failure paths end through guaranteed `os._exit(1)` so Isaac teardown cannot
  mask them. A missing app is never eligible. The finalizer now enforces exact
  JSON scalar types, event-to-limited-joint bounds, quaternion angular-error
  reconstruction, max-raw scalar/vector/target/slew/limit identities, and a
  semantically saturated adversarial slew event. It also binds the actual
  `SLURM_JOB_ID`, expected finalizer SHA, and externally supplied expected
  saved-sbatch SHA. It also reconstructs the actual ±0.04-m translation and
  right-multiplied, sign-aware ±15° rotation target matrix from the hold case,
  and bounds terminal adversarial dq by the configured velocity limits.
  The finalizer additionally requires exact bidirectional agreement between
  positive slew-event counters and a raw per-joint delta strictly above its
  configured slew bound. Probe-specific mutation tests cover both mismatch
  directions and every other rejected case.

## 2026-07-02 — controller-safety v3 promotion evidence

- Preflight-only job `1097684` failed before `srun` because it pinned the login
  node's stale Vulkan ICD. It produced no raw, ready, attestation, controller,
  or simulator evidence and is non-promotable. Replacement `1097716` pinned
  the allocated L40S node's driver-580.105.08 ICD and completed job, batch,
  extern, and `srun` with `0:0` in 4m39s.
- All 13 ordered hold/±XYZ/±RXYZ cases passed with exactly 360 apply calls each;
  maximum error was 0.538 mm / 0.218 degrees. The fresh-reset oversized +X
  case passed with eight apply calls, eight slew events affecting 24 joints,
  finite q/dq, maximum terminal |dq| 0.10463 rad/s, q inside the captured soft
  limits, and zero abort, fallback, dropped-diagnostic, current-limit, or
  post-clamp violations.
- Immutable mode-0444 evidence is under
  `results/polaris_eval/controller_safety_v3_smoke/c6939d7`: raw SHA-256
  `545048264844c2d18d594f574b87073a0ab98071697c402993bd3915ad741890`,
  ready-marker SHA-256
  `5f5771d5ce8109392a4ee97c6921603dcd44718188a22a1f1a18de797ecd567a`,
  final attestation SHA-256
  `36fb7338c1d438bc9af34ef3bff2e983134b5ee1cae3f8e61b8707da365eb5c9`,
  and saved/recovered sbatch SHA-256
  `dd4cb04c5487125e1b8b4a448938b5fdad00d6c330f982ffb775486da80e85e7`.
  Both finalize and verify passes agree on the attestation hash.
- Independent strict parsing, typed attestation reconstruction, target-geometry
  recomputation, counter/maxima audit, and little-endian float32 reconstruction
  reproduced soft-limit digest
  `fbf7535901c042fea5d901812ecd02c5fd81ade06c23c1499c32d66a859104de`.
  This promotes the exact captured limits for EEF IK-safety v3 only; native
  joint-position/pi05 semantics are unchanged. A full-horizon checkpoint
  canary remains mandatory before any standard evaluation.
