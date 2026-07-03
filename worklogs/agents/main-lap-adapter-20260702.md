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

## 2026-07-02 — v4 boundary replay is non-promotable

- V3 official and reasoning canaries later reached joint-5's upper boundary
  and failed closed. V4 candidate `e47481458719a0b07637e8c62b5980ea498062e4`
  therefore inset every commanded joint target by one maximum 120-Hz velocity
  step and added an exact boundary replay of the 378 official actions plus an
  adaptive outward dwell.
- Pinned L40S job `1097984` passed all 23 Isaac-dependent and 45 host/runtime
  tests. It completed the exact 378-action fixture, but its first adaptive
  step (`policy_step=378`) found a current joint already outside the live outer
  soft limit and aborted before DLS/PhysX. Parent/batch/replay failed `1:0`;
  no ready marker or attestation exists. The immutable mode-0444 failure JSON
  SHA-256 is
  `f0e07058bd6d1c6dfdd05cb1039c553582f1d5f658e9ab7f8e26ae0d054103a0`.
- This disproves the one-physics-substep static target inset as a sufficient
  implicit-actuator state bound. V4 remains pending and no checkpoint rerun is
  authorized. The boundary runner now captures finite-or-null live arm q/dq/
  position-target vectors plus the full controller safety report before
  failure teardown so the replacement can be based on measured terminal
  dynamics rather than an assumed margin.
- Diagnostic rerun `1097986` captured the exact residual dynamics. It failed
  at policy step 378/substep 6 after 3,031 apply calls with joint 5 at
  `2.8973512649536133` rad, `5.125999450683594e-05` above the canonical outer
  limit, while its position target was the bit-exact inner upper limit
  `2.8755500316619873` and velocity remained outward at
  `0.07716280221939087` rad/s. All targets and vectors were finite and every
  post-clamp invariant passed. Failure JSON SHA-256 is
  `8b014f4f3a4620aa010062e43cf51b7e4ee08817a8b3b6be96ca3358447a204d`.
- V5 therefore promotes the already-forbidden target envelope to an EEF-only
  PhysX hard-position envelope. The action term freezes the original Panda
  limits as the canonical outer safety contract, writes `outer +/- v/120`
  exactly once through Isaac Lab's position-limit writer, and requires exact
  PhysX, articulation-mirror, and derived-soft-limit readback. It explicitly
  writes a zero arm velocity target before each position target. No joint state
  is teleported, healthy DLS/target bits remain unchanged, and native joint/
  pi05 setup never instantiates this mutation. Static/runtime evidence now
  binds the hard-limit profile/digest/write count/readback and zero-velocity
  profile; per-episode maxima bind hard-limit solver slop, absolute velocity,
  and canonical outer clearance.

## 2026-07-02 — velocity-limit transient diagnosis

- The v5 standalone matrix passed, but exact official-action boundary replay
  `1098049` on the retained TGS 64-position/1-velocity-iteration controller
  failed closed at apply call 923 (policy step 115, physics substep 2): joint
  7 reached `-2.8927373886` rad/s against the live `2.6099998951` limit.
  Direct PhysX q/dq matched the Isaac cache bit-for-bit. Four velocity
  iterations (`1098051`) were worse, reaching joint-5/joint-7 ratios of
  1.132/1.295 times their limits; this solver variant is rejected.
- A separately reviewed fixed 0.8 position-target slew candidate, PolaRiS
  `bc4890854e8c758b9278de772f09a229b2b95764`, preserved the physical hard
  envelope, zero velocity target, 400/80 gains, effort limits and strict
  pre-DLS qdot abort. Its boundary job `1098074` passed 38 Isaac tests and 105
  host tests plus 22 subtests, then failed at apply 936 (policy 116/substep 7)
  with joint 5 at `-3.7024555206` rad/s and joint 7 at `+3.6261839867` rad/s.
  The immutable mode-0444 JSON SHA-256 is
  `9d812bdf05d1fc4e0284d1c1860c6a4f37b4b1e22cb3c7287b871e1301bafebc`.
  It had zero position/recovery events and exact target bounds, disproving a
  monotonic fixed-slew repair; no lower scalar slew or checkpoint rerun is
  authorized.
- Isaac Lab 2.3 source inspection established the missing phase information:
  captured q/dq is the current post-physics state, while `computed_torque` and
  `applied_torque` are the preceding `write_data_to_sim` call's approximate
  implicit-PD preclip/postclip buffers, not actual PhysX drive torque. The
  terminal artifact alone therefore cannot select a brake threshold or
  distinguish coherent velocity-reference control from a qdot-opposing target
  projection.
- This branch restores the exact retained factor-1/TGS-64/1 source revision
  and adds a boundary-only, default-disabled 64-substep device ring. Each
  completed entry causally binds pre/post q/dq and float32 deltas, previous/raw/
  accepted position targets, zero velocity and feed-forward targets, current/
  desired EEF poses and pose error, and Isaac's approximate preclip/postclip
  efforts. The producer binds direct PhysX and mirror readbacks for 400/80
  gains and `[87,87,87,87,12,12,12]` effort limits. The runner independently
  reconstructs every float32 PD computation and clip, enforces contiguous
  ring/index/transition identities, the exact velocity-abort guard, cached vs
  direct PhysX terminal state, and terminal target/effort-buffer identity.
  Standard safety reports, sidecars, success payloads and native/pi05 control
  remain unchanged.
- Local gates pass: 37 boundary-runner tests, 14 focused controller tests under
  lightweight Isaac stubs, Ruff, Ruff format, `py_compile`, and `git diff
  --check`. Two independent read-only reviews accept the causal producer/ring;
  one initial NO-GO on stale-effort validation was resolved by the exact live
  gain/target/effort reconstruction above. Full Isaac tests and one immutable
  L40S diagnostic replay remain mandatory before choosing any numeric braking
  law or launching official/reasoning checkpoint canaries.
- Committed and pushed the reviewed diagnostic as PolaRiS
  `9a3c12d6188906bab504d5de8bc346218c4ee71e`; the clean detached L40S deploy is
  `/lustre/fsw/portfolios/nvr/users/lzha/src/PolaRiS-trace-v9-9a3c12d`.
  Test-only job `1098093` was canceled while still pending after its pinned
  pool3 node was consumed by a four-GPU job; it never started and produced no
  runtime artifact. The independently reviewed replacement wrapper changes
  only the node pin to idle `pool0-00013`, is mode 0444 with SHA-256
  `acac1e8ef869bb0a10064d7a444ee174c60ebc4ff70946b8fdd6c5a2c50687cc`,
  and launched as job `1098094`. It runs the full Isaac/host gates and the
  exact replay only; its success condition is an immutable, internally valid
  expected velocity-abort trace with no teardown error, ready marker, or
  attestation. No model, checkpoint, server, or downstream chain is involved.
- Job `1098094` is non-promotable regardless of its replay result: its Isaac
  pytest process found that Isaac preloads a conflicting top-level `scripts`
  module, so collection failed when the new cross-contract test used a normal
  `from scripts ...` import. The existing bootstrap then returned step status
  zero despite pytest's collection error, allowing the wrapper to continue.
  The repair loads the runner contract by exact file path under a unique module
  name and makes the bootstrap capture pytest/teardown failures, flush logs,
  and use `os._exit` with the exact resulting code after Kit closes. Six host
  regressions bind pytest exit codes 0/1/2/4/5 and close-failure override;
  updated local gates pass 118 host tests plus 22 subtests and 14 focused
  controller tests. A clean-commit L40S rerun is required.

## 2026-07-02 — fail-closed Isaac test transport repair

- Although job `1098094` cannot certify the test gate, its replay produced a
  scientifically useful immutable expected-failure trace at
  `controller_diagnostics_v9/9a3c12d-factor1-trace64/boundary-trace-1098094.json`,
  mode 0444, SHA-256
  `8c926bc8421e27414631f5266399dc52121ffb3b53e4419c2cfa39eaecbaf5e7`.
  The 64 entries cover apply calls 858 through 921 before the velocity guard
  aborted apply 922 (policy 115/substep 2). The terminal transition is an
  abrupt coupled wrist impulse despite small position deltas and unsaturated
  approximate PD effort: joint 5 `-0.090587 -> +2.223634`, joint 6
  `+0.103990 -> -0.651813`, and joint 7 `+0.077781 -> -2.892737` rad/s.
  Only two of 448 joint entries saturated, both during an earlier healthy
  reversal, so neither magnitude headroom nor coherent velocity targets
  explain or safely repair the transient.
- Clean follow-up job `1098095` used PolaRiS commit
  `3b3dcc1a6ec4db8c8a143a8826efd0fdf6f9716e`. Real-Isaac collection ran and
  exposed two test-fixture defects: assigning read-only `ActionTerm.num_envs`
  and missing `_debug_vis_handle` on bare allocations. It reported `12 failed,
  37 passed` and printed `ISAAC_PYTEST_EXIT_CODE=1`, but Slurm/Pyxis still
  recorded the step as `COMPLETED|0:0`; the wrapper incorrectly printed
  `TRACE_ISAAC_UNIT_TEST_RC=0`. Host tests then passed (`124 passed, 22
  subtests`). The replay step was canceled before completion, and no raw trace,
  ready marker, or attestation exists. The only result artifact is immutable
  wrapper `boundary-trace-1098095.sbatch`, SHA-256
  `1cecc0a0e6335b18defbbb1febdb63f2c85d6e08b2aa7e7bca4e2586e3df69b1`;
  log SHA-256 is
  `fdf4bc4ffd614cb9347f629506ba6430d9f9ec9637f45e1679fec23966bb35af`.
- Commit `058694d` centralizes all bare action fixtures around the real Isaac
  `_env` property contract and initializes `_debug_vis_handle` first. It also
  makes the bootstrap publish a final post-close/post-flush exit code through
  an exclusive, fsynced, mode-0444 hard-link commit point. The fixed temporary
  path and final path are non-overwriting; any missing, leftover, malformed,
  or nonzero artifact is reserved for wrapper rejection even if Pyxis masks
  the process code. Unit tests cover exact pytest codes 0/1/2/4/5, import,
  pytest, close, reporting and flush failures, write/fsync/link failure,
  existing and dangling destinations, and unlink failure. Two independent
  read-only reviews accepted the real-Isaac fixture and sidecar producer.
  Local gates pass 140 host tests plus 22 subtests, Ruff format/check,
  `py_compile`, and `git diff --check`. A new immutable wrapper must require
  both zero `srun` status and exact `0\n` sidecar after host directory fsync,
  then run the full real-Isaac suite and expected-failure replay.
- The independently reviewed v11 wrapper is mode 0444 at
  `/lustre/fsw/portfolios/nvr/users/lzha/staging/polaris_v11_9b67e14/boundary-trace-v11.sbatch`,
  SHA-256
  `77f1b84f6fa8fdd322c3b281a3c11dbbca73c8a2e56afcff1d4cca341f3e8991`.
  It adds an intentional no-tests-selected exit-5 probe, proves that the same
  strict zero validator rejects its immutable `5\n` sidecar, then requires
  both zero outer `srun` status and an immutable exact `0\n` sidecar for the
  full real-Isaac suite. The sidecar validator rejects missing, symlinked,
  non-0444, empty, truncated, extra-newline, nonzero, wrong-hash, leftover-temp,
  and multi-link artifacts. It then runs all 141 host tests plus 22 subtests
  and the exact controller-only expected-failure replay. Clean detached source
  is `/lustre/fsw/portfolios/nvr/users/lzha/src/PolaRiS-trace-v11-9b67e14`
  at commit `9b67e1496f5ecd6ce4f40953ca112f7b670ff423`.
- Launched ordinary one-GPU Slurm job `1098101` on L40S `pool0-00013` at
  2026-07-02 23:03 PDT. Log:
  `/lustre/fsw/portfolios/nvr/users/lzha/slurm_logs/polaris_eval/pol_v11_trace-1098101.out`.
  Result root:
  `/lustre/fsw/portfolios/nvr/users/lzha/results/polaris_eval/controller_diagnostics_v11/9b67e14-factor1-trace64`.
  Success requires both sidecar probes, all target-surface tests, an immutable
  strictly accepted velocity-abort trace, no ready marker/attestation, and
  wrapper exit zero; scheduler completion alone is insufficient.
- Job `1098101` failed closed after the intentional pytest probe reported exit
  5 while Pyxis again returned outer step status zero. The strict gate rejected
  the missing sidecar and stopped before the positive suite or replay. The only
  result artifact is its immutable saved wrapper; no raw trace, ready marker,
  or attestation exists. Log SHA-256:
  `d143bcf7e20e74f97aba0a388bd6dd96c25e5e9d0f751b5219bb9aab8cd1ad1c`.
  A direct exact-container publisher probe, job `1098102`, completed and wrote
  immutable `5\n` SHA-256
  `f0b5c2c2211c8d67ed15e75e656c7862d086e9245420892a7de62cd9ec582a06`,
  proving env transport, mount writability, and the publisher independent of
  Kit. Minimal AppLauncher probes `1098103`/`1098104` segfaulted before launch
  returned because their reduced runtime surface did not reproduce the full
  validated container environment; they are infrastructure diagnostics only,
  produced no sidecar, and are not controller evidence.
- Early-capture rerun `1098107` at commit `841f4d6` printed the correct
  sidecar path and pytest exit 5, but again produced no sidecar and failed the
  strict gate before positive tests/replay. This proves the issue is not a
  missing env value: pinned Kit termination prevents Python after
  `SimulationApp.close()` from executing. Its only result is the immutable
  saved wrapper; log SHA-256 is
  `86b890b3c85f5e7bca452602f9edcf2ad640c9714c25c7150561117250d05823`.
- Commit `2fca845f0e42c089dd6820cb8e0da21cc8a5c4a2` moves the final transaction
  outside Kit. A standard-library parent spawns the Isaac child with an
  anonymous write-only pipe; the child scrubs parent-only env, validates and
  de-inherits the FD before imports, flushes pytest diagnostics, commits one
  exact result byte, and only then closes Kit. The parent reaps native teardown,
  requires one allowed byte plus EOF and a coherent direct-child status, kills
  and rejects lingering process groups, handles timeout and TERM/INT cleanup,
  and alone publishes the immutable final sidecar. Native fake-Kit regressions
  reproduce `close() -> os._exit(0)` for pytest 0 and 5. Post-report close,
  marker-flush, pipe, descriptor, signal, timeout, unreapable-child, and
  sidecar failures are fail-closed. Three independent reviews accept the
  boundary. Local gates pass 168 host tests plus 22 subtests, 56 focused
  bootstrap tests, Ruff, `py_compile`, and `git diff --check`.
- The independently reviewed v13 wrapper differs from v12 only in its six
  provenance/output bindings and is staged mode 0444 at
  `/lustre/fsw/portfolios/nvr/users/lzha/staging/polaris_v13_2fca845/boundary-trace-v13.sbatch`,
  SHA-256
  `eadeffbb02a09aaac60189cafdc91020ad393ded5f07bc2e71c34d46f5dc91ff`.
  Clean detached source is
  `/lustre/fsw/portfolios/nvr/users/lzha/src/PolaRiS-trace-v13-2fca845`.
  Launched ordinary one-GPU job `1098116` on `pool0-00013`; log is
  `/lustre/fsw/portfolios/nvr/users/lzha/slurm_logs/polaris_eval/pol_v13_trace-1098116.out`,
  result root is
  `/lustre/fsw/portfolios/nvr/users/lzha/results/polaris_eval/controller_diagnostics_v13/2fca845-factor1-trace64`.
  It remains controller-only with no model, checkpoint, policy server, or
  downstream chain.
- Job `1098116` completed the clean diagnostic contract in 257 seconds. The
  intentional no-tests-selected probe returned pytest/child/outer status 5,
  published immutable exact `5\n` SHA-256
  `f0b5c2c2211c8d67ed15e75e656c7862d086e9245420892a7de62cd9ec582a06`,
  and was rejected by the same zero gate. The positive real-Isaac suite passed
  all 50 tests, returned child/outer status 0, and published immutable exact
  `0\n` SHA-256
  `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`.
  The host suite then passed 168 tests plus 22 subtests. The expected-failure
  boundary replay exited 1 at policy step 115 and produced immutable mode-0444
  raw trace SHA-256
  `00486e0012753b71ecd6c79c1c00ec9b9cf84e7d0ae11e1143a9536d3f2776ff`;
  strict validation accepted 64 consecutive completed entries 858--921 with
  no pending entry, close failure, temporary, ready, attestation, or hidden
  result. The wrapper exited 0, all Slurm processes reaped, and the clean
  source remained exactly `2fca845f0e42c089dd6820cb8e0da21cc8a5c4a2`.
  The log SHA-256 is
  `1f96b9281212e170e6944bd76b19df6c2925e38ac17fd6405d470bba628eeb2e`.
  Apart from its source-path traceback, this trace is byte-for-byte equal as a
  parsed JSON object to the earlier job `1098094` trace. It independently
  confirms a one-substep coupled wrist impulse at apply index 921: joints 5--7
  change velocity by `+2.314222`, `-0.755803`, and `-2.970518` rad/s while the
  post-clip efforts are only `-0.642102`, `+0.380754`, and `+0.454144` Nm.
  Only two of 448 traced effort samples saturated, both 25 substeps earlier;
  all arm velocity targets and feed-forward effort targets are exactly zero.
  An independent post-run audit also executed the source runner's complete
  failure-trace validator and accepted all transition, live-state, PD, counter,
  permission, link-count, and publication-boundary identities.
- Started isolated branch `codex/eef-wrist-energy-brake-v1-20260703` from the
  clean v13 evidence commit `09a8afd`. Two independent trace reviews agree the
  smallest useful next experiment is an opt-in diagnostic canary, not a
  proven controller repair: a near-full-substep applied-target error reversal
  on any of Panda joints 5--7 arms a group latch for the trigger substep and
  one following physics substep. While active, only wrist targets whose
  nominal spring term has positive power (`error * velocity > 0`) are replaced
  by the ordinary slew/guard-bounded hold-at-current-position target; all arm
  velocity targets remain exactly zero. The post-hoc velocity-jump extension
  was intentionally omitted. The existing v4 behavior and report schema stay
  the default; the experiment has a separate candidate profile and must pass
  the deterministic boundary replay before it can reach a full-horizon model
  canary.
- Implemented the candidate as an exact-name-bound, single-environment
  two-substep state machine. Reset starts reversal detection disarmed; after
  one ordinary nominal command it is armed. A trigger and its immediate
  follow-up are both disarmed, followed by one ordinary refractory command
  before re-arming. This prevents a moving joint from treating a brake-created
  hold target as a fresh reversal. Normal action-term resets and explicit
  episode resets both clear the latch, stored target, validity, and arming
  state. Both PhysX target setters must finish before any candidate state,
  counter, or diagnostic commit.
- Split projection attempts from effective brakes. A projected wrist joint is
  counted effective only when the applied target differs from nominal and the
  residual float32 spring power is nonpositive. The bounded evidence tail now
  records every active latch substep, not only triggers. Runtime and host
  validators recompute float32 trigger/attempt/effective masks, enforce exact
  `active + latch = 2 * trigger` history, trigger/follow-up alternation,
  earliest initial/refractory arming cadence, immediate follow-up cadence,
  float32-exact previous-target chaining, open-latch/last-successful-apply
  identity, ordered apply indices, and exact or loss-bounded counter sums.
  Hidden effective totals cannot exceed hidden projection attempts. The
  deterministic boundary promotion gate rejects any dropped causal
  diagnostics; a larger immutable capacity and rerun are required if the
  current 32-record tail is insufficient.
- Closed the launch/attestation surface after two initial independent NO-GO
  reviews: the candidate now fails fast outside one environment, runtime frame
  and safety profiles are cross-bound, initial/final/frame profiles must agree,
  initial latch/diagnostics must be empty, and the finalizer requires the exact
  expected base or candidate profile. Tamper tests cover mixed profiles, stale
  latch state, impossible counters and apply indices, forged attempted/effective
  totals, non-causal follow-up timing, and wrong-profile finalization.
- Current host gates pass all 190 non-Isaac tests plus 30 subtests. A separate
  lightweight Isaac-module stub run passes 19 focused wrist controller/helper/
  reset tests with 55 deselected. Ruff format/check, Python byte compilation,
  and `git diff --check` pass. Independent controller and artifact-contract
  re-reviews are code-GO for an isolated L40S diagnostic canary; real-Isaac
  collection and the immutable baseline/candidate target-surface A/B remain
  required before this experiment can be called a repair or used for
  checkpoint evaluation.

## 2026-07-03 — wrist-energy brake A/B is valid negative evidence

- Commit `426ad821ec300e13febd4895151ae2c959d6d6a0` passed its fail-closed
  L40S test gate in job `1098125`: the deliberate pytest-exit-5 probe was
  rejected, all 74 real-Isaac tests passed with an immutable exact `0\n`
  sidecar, and the host suite passed 188 tests plus 2 skips and 22 subtests.
  The saved wrapper SHA-256 is
  `924b5862a52243e4f248aacadda2677800ab7dda51f0d8e02a40f4e49e1a2d62`;
  the log SHA-256 is
  `9ce0b86f1f79aac8543e755f4480c61454ebba711edc75430f4c4519b1fb3ce5`.
- Immutable parallel L40S jobs `1098126` (baseline, `pool0-00013`) and
  `1098127` (candidate, `pool0-00027`) both started at 00:35 PDT and ended in
  188/190 seconds. Their parent/batch jobs completed `0:0`; each replay step
  intentionally returned `1:0`, and the corresponding wrapper accepted only
  a strictly validated `DifferentialIKInvariantError` velocity-abort artifact
  before exiting zero. Both nodes are idle, all job processes are reaped, and
  the clean deployed checkout remains exactly `426ad821`.
- Baseline raw JSON is immutable mode 0444, single-link, SHA-256
  `d24ede878efe7b841bbaa1fd097c4595a6b5c9c5e2b4a4068b8e12753efdb13f`;
  its saved wrapper SHA-256 is
  `54f21abb9e94ab2903518d50ec8c28760416e198ad19ea965cb2e5d93d57ad6f`.
  It reproduces the preserved v13 runtime evidence and 64-entry causal trace
  exactly: the only full-payload difference is the source path/line in the
  traceback. The guard aborts apply 922 at policy step 115/substep 2 after
  joint 7 reaches `-2.8927373886` rad/s against its `2.6099998951` limit.
- Candidate raw JSON is immutable mode 0444, single-link, SHA-256
  `a73b584b1c46a505c6cd8812e74841aa882b3c39bb4992a19bdc26be8f07d163`;
  its saved wrapper SHA-256 is
  `3d41260a8526f36c983b4a88727462e0ccf2db5c26e2c11940d894c91623d610`.
  Strict candidate-history and failure-trace validation accepts 14 triggers,
  28 active substeps, 49 attempted/effective projected targets, a closed
  latch, and zero dropped evidence. Nevertheless it aborts at the same policy
  step and apply: joint 7 reaches `-2.8293285370` rad/s and joint 5 reaches
  `+2.4241445065` rad/s. The joint-7 limit excess falls by 22.43 percent, but
  remains unsafe; no ready marker or attestation exists and the candidate is
  rejected for a model canary.
- The terminal candidate transition falsifies positive per-joint spring power
  as a sufficient causal model. At apply 921 all three approximate wrist drive
  contributions have net dissipative power, yet PhysX produces the coupled
  velocity impulse. The committed 378-action fixture adds a more specific
  diagnostic lead: policy step 115 is its first and only gripper command
  transition (`0 -> 1`, open to closed). The next experiment must isolate the
  gripper drive/contact impulse and record live gripper targets/state/drive and
  link-contact evidence; another threshold-only wrist-brake tune is not
  justified by these results.

## 2026-07-03 — authoritative gripper articulation metadata probe

- Model-free L40S job `1098158` completed `0:0` on `pool0-00013` in 258
  seconds using the pinned PolaRiS source `426ad821`, pinned container, and
  pinned FoodBussing assets. The job published only the saved wrapper and an
  immutable mode-0444, single-link JSON artifact. The wrapper SHA-256 is
  `2de33f05955aba82d7a9cdd4c7f69eb449dc8c0c5e29342f9e3ef356ef407ab4`;
  the JSON SHA-256 is
  `839b04a5226ec2d1a0b3171696a24bc97494fa116e16de866bf758c35cebdb1c`.
  Strict post-run parsing passed, the deployed checkout remained clean, the
  per-job cache was removed, and the node returned idle. The two logged error
  lines are the known non-fatal headless NGX initialization messages; there is
  no traceback, crash, or killed process.
- The live articulation has exactly 13 ordered DOFs: Panda joints 1--7,
  `finger_joint`, `right_outer_knuckle_joint`, `left_inner_finger_joint`,
  `right_inner_finger_joint`, `left_inner_finger_knuckle_joint`, and
  `right_inner_finger_knuckle_joint`. It has exactly 18 ordered bodies: Panda
  links 0--8, `base_link`, then left/right outer knuckles, left/right outer
  fingers, left/right inner fingers, and left/right inner knuckles. Joint and
  PhysX DOF shapes are `[1, 13]`; body state is `[1, 18, 13]`; PhysX link
  velocity is `[1, 18, 6]`. The robot USD is 14,156,155 bytes with SHA-256
  `d8379925b103963dbf3e7c85bcc4ae101b81b7c1d7dabe7d2e964f41d069ec44`.
  These live ordered lists replace the provisional USD traversal order and
  are mandatory cross-bindings for the gripper impulse diagnostic.

## 2026-07-03 — first gripper impulse pair failed closed on static readback device

- Independently reviewed wrappers launched immutable parallel L40S jobs
  `1098160` (exact, `pool0-00013`) and `1098161` (delay-one-step,
  `pool0-00027`) from PolaRiS `6e9b7be`. Both scheduler records exactly
  matched the pinned command, source checkout, node, account, partition,
  16-CPU/96-GiB/one-GPU allocation, stdout, and generic GPU TRES contract.
- Both jobs rejected before replay because PhysX
  `get_dof_max_velocities()` returned a CPU tensor while the diagnostic had
  incorrectly required every direct PhysX getter to share the articulation's
  `cuda:0` device. Exact raw invalid-capture SHA-256 is
  `52afe48a2e10cdd38d430097188dda0f3fd969005919bfe4240194b59a31290c`;
  delayed raw invalid-capture SHA-256 is
  `fcfbf9847134d37e5d81e39fcf229f54f2469cf407d6f56a99265194759d762c`.
  Each attempt contains only immutable mode-0444, single-link
  `capture.json`, `runtime.exit`, and `outer-srun.exit`; both status files are
  exact `1\n` with SHA-256
  `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`.
  Parent and batch jobs are `FAILED 1:0`, no ready marker, video, validator,
  or attestation exists, caches/staging are empty, the deployed source remains
  clean, and both nodes returned idle. This is valid negative infrastructure
  evidence only and says nothing about close-command causality.
- The repair must not generically weaken device validation. A pinned live
  device/dtype/shape probe is required to classify dynamic GPU readbacks,
  static CPU drive-property readbacks, cached articulation tensors, and
  actuator tensors field by field before the pair is relaunched.
- Model-free L40S job `1098162` supplied that closed classification and
  completed `0:0` on `pool0-00013` in 255 seconds. Its immutable JSON is mode
  0444/single-link, 11,403 bytes, SHA-256
  `d3c8ccfcb16cd523f084f5c7c82f41a03c1c2ab0f58487f45ff4c2a59066283c`;
  the saved wrapper SHA-256 is
  `7bf346c05b676d16db0f102990efba9c481be01e2fb57ea96115313c200d48d1`.
  The log has no traceback or crash, the cache was removed, source stayed
  clean, and the node returned idle.
- Direct PhysX positions, velocities, projected joint forces, link
  velocities, link accelerations, and link incoming-joint wrenches are
  `cuda:0` float32. Direct PhysX maximum velocities, maximum forces,
  stiffnesses, and dampings are instead `cpu` float32. Every captured cached
  articulation tensor and every resolved gripper actuator tensor is
  `cuda:0` float32. All observed shapes, element counts, and finiteness checks
  match the 13-DOF/18-body articulation. The diagnostic must now enforce this
  exact field partition and compare actuator/static-PhysX values while
  excluding only their intentionally different device labels.

## 2026-07-03 — corrected gripper pair 1098164/1098165 failed closed in publication

- Parallel L40S jobs `1098164` (`pol_gi_exact2`, `pool0-00013`) and `1098165`
  (`pol_gi_delay2`, `pool0-00027`) are terminal `FAILED 1:0` and no longer in
  `squeue`. Job `1098164` ran from 03:46:16 to 03:49:44 PDT; job `1098165`
  ran from 03:46:17 to 03:49:29 PDT. Their authoritative log SHA-256 values
  are respectively
  `22d6ed001bd53290acbf4092f8c1a9afae813d6d54b8591712c6b489ca7f593b`
  and
  `722d3c530bb1e909f3c7a86cde3c0f71a75305515629d7f5daf1b2ecdbc1c6e5`.
- The exact runtime in `1098164` completed and published immutable raw, ready,
  video, runtime-exit, and outer-srun evidence. Its capture records the allowed
  velocity-guard outcome and a 117-frame 448x224 video at 15 fps. Host
  finalization then failed before attestation because the deployed source was
  a linked Git worktree: its `.git` file referenced an administrative gitdir
  under an unmounted external checkout, so `git rev-parse --show-toplevel`
  exited 128 in the finalizer container. The validator status is `1`; no
  attestation exists. This is usable raw diagnostic evidence but not a
  provenance-complete promoted capture.
- The delayed runtime in `1098165` reached temporary video publication, where
  the old host probe rejected the frame-count/container-duration predicate.
  It published only an immutable invalid `capture.json` plus exact failure
  runtime/outer-srun status; no video, ready marker, validator status, or
  attestation survives. The temporary MP4 was unlinked, and the failed-capture
  schema does not preserve its frame count or ffprobe duration. The expected
  119-frame count is source-derived rather than cryptographically bound by
  this attempt and is not reported as observed job evidence.
- Both attempts therefore failed closed, and the exact-versus-delay causal
  pair is incomplete: there is no authoritative conclusion about whether the
  close command caused the velocity guard. The next deployment must use a
  truly standalone clean detached clone with a real in-root `.git` directory,
  and both launch preflight and finalization must reject linked worktrees or
  external Git/common-Git directories. Video cadence validation must bind
  decoded stream ticks and time base rationally while accepting only the
  canonical ffprobe stream-duration or MP4 millisecond-ceiling container
  representation.

## 2026-07-03 — fully attested gripper-delay pair implicates close timing but remains unsafe

- Independently reviewed mode-0444 wrappers were submitted 79.7 milliseconds
  apart from one SSH session with their reviewed SHA-256 values bound through
  `EXPECTED_SAVED_JOB_SCRIPT_SHA256`. Exact job `1098167` ran on
  `pool0-00013` from 04:16:41 to 04:21:21 PDT and delay-one job `1098168` ran
  on `pool0-00027` from 04:16:41 to 04:21:23 PDT. Both root jobs and all five
  steps completed `0:0`; the nodes are idle and neither job remains queued.
  The final exact and delayed log SHA-256 values are respectively
  `d5ec6689325df2d646dee5a1b5f84d1608d4194a9444fe46a41b078494c79675`
  and
  `c17b80b0e89f005fb60ab5525682c2b055635d326038ef167f8c5cf07d074136`.
- Each attempt has exactly ten regular, mode-0444, single-link files and no
  extras. Runtime, outer-srun, and validator sidecars are exact `0\n`.
  Exact capture/video/ready/attestation SHA-256 values are
  `5ec6a5cc5fb85e9f28fc42bf322bd3e9341099e9da3b6dac6e0ef9fa22facff9`,
  `351126b771e11c1fe8b403ab8ff443607f33a01fc782e1cd8a9d309a1f18f508`,
  `f36d74011f270f4ae3dc7b69c2a20c2fca8c2ed803171cb114e9f7cabc670f3b`,
  and
  `0851b171ed6d7a532c7508d3a6af3ddeaaaba6f40b8f258664a6bbd325261599`.
  Delayed values are
  `510c3cd8c7493694fd4ffec54fc9fa4851f2db0c1402e3216a1ee944d8508966`,
  `dd3a5475fdb6acedcc39591652a7f639d0aa4bca9b4ba5c6912eabcd93e5ad3d`,
  `0e1cff0ab51ec536d3d18156b18b18a03edd833a9a9a19940e6146691e8fe221`,
  and
  `b30c9668779d43236d7665835fb83d7b7b0c1e959164837ce10284a3012fcdef`.
  Runtime and Slurm wrapper snapshots equal the reviewed exact wrapper
  `32fc35a4ea6ce684eb1b9ad86204e9f1f8f042ad4fbcebe2d344b1ce9f05a78e`
  or delayed wrapper
  `fc98de4dbb6bef279798aa6c3eb06aa7836057ec00f2976180a046b38c954e89`.
- Both attestation-v6 records bind the clean detached standalone commit
  `04d9ed6e4f7ec3e418a0927f990c6b30bc4ade8f`, its real in-root Git/common-Git
  directory, exact source and asset identities, assigned node, command, and
  Slurm snapshot. Per-job cache and staging directories are absent after
  completion; the source remains clean and byte-exact.
- Local ffprobe and complete ffmpeg decoding accept both H.264 High/yuv420p
  videos at 448x224 and exactly 15 fps. Exact is 117 frames/7.8 seconds;
  delayed is 119 frames/7.933333 seconds. First, middle, and terminal frames
  are nonblank, show continuous arm/object motion, and have the intended
  external-plus-rotated-wrist layout.
- In exact mode, the only gripper transition is close at policy step 115.
  Finger velocity jumps from approximately zero to `+8.022388` rad/s on its
  first physics substep, then to `-8.726713` rad/s while applied gripper torque
  is clipped at `+200` Nm. The arm guard aborts apply 922 at policy step
  115/substep 2: joint 5 is `+2.223634` rad/s and joint 7 is `-2.892737` rad/s
  against their `2.61` rad/s limit.
- The delayed plan changes only action index 7 at policy step 115 and preserves
  all arm dimensions bitwise. The gripper remains effectively stationary and
  arm velocity stays below `0.106` rad/s through that step. Closing at step
  116 then drives the finger to `+8.726656` rad/s, later oscillating through
  `-8.727228` rad/s under the same `+200` Nm clip. The arm eventually aborts
  at apply 943, policy step 117/substep 7, with joints 5 and 7 at
  `+4.827284` and `-4.899215` rad/s.
- This intervention establishes that the original policy-115/substep-2 guard
  is contingent on executing the close at policy step 115: suppressing only
  that close eliminates the reference-timed failure. It falsifies the simpler
  prediction that delaying close by one policy step merely shifts the guard by
  one policy step, because the delayed run instead fails 21 physics substeps
  later at the diagnostic horizon. The controller is still unsafe, so this is
  not a production repair. The next isolated default-off candidate should set
  the implicit gripper's simulation velocity limit to 5 rad/s while preserving
  legacy behavior byte-for-byte when disabled, then repeat the fully attested
  A/B under the corrected CPU-static/CUDA-actuator evidence contract.

## 2026-07-03 — isolated 5 rad/s gripper PhysX-cap canary is code-complete locally

- Ported the smallest default-off candidate onto the current attested base
  `04d9ed6`: `configure_eef_pose_joint_safety` now accepts the exact boolean
  `enable_gripper_velocity_limit=False`. Only the opt-in gripper diagnostic CLI
  passes it true. The candidate exact-name-checks the one `finger_joint`, its
  null configured gains, legacy 5 rad/s velocity, and 200 Nm effort before
  authoring `velocity_limit_sim=5.0` and `effort_limit_sim=200.0`. Normal
  `scripts/eval.py`, the action config, arm controller, and arm runtime-contract
  schemas are unchanged. A source gate asserts that `scripts/eval.py` does not
  name the candidate flag; the full diff audit separately confirms the action
  config, arm controller, and production runtime-contract files are unchanged.
- The disabled diagnostic keeps its existing payload schema and drive profile,
  and still requires the job-1098162 live value of
  `8.726646423339844` rad/s. The candidate has a separate nested drive profile
  and requires exact float32 5.0 rad/s readback from both the CUDA actuator
  tensor and the CPU direct-PhysX tensor. It preserves the authoritative
  job-1098162 device partition: actuator/cached evidence is `cuda:0`, static
  direct PhysX evidence is CPU, dtype/shape/finiteness/value equality remains
  exact, and only the device label is excluded when cross-comparing those two
  representations. Effort remains exactly 200 Nm; stiffness and damping remain
  the probed `5729.578125` and `0.011459155939519405` values. Every retained
  pre/post physics snapshot cross-binds its selected finger drive values to
  this startup contract.
- Isaac Lab v2.3.0 upstream source was checked before pinning the live config
  expectation: when legacy and `_sim` velocity fields are both present and
  equal, `ImplicitActuator` leaves both at 5; the candidate therefore requires
  live `cfg.velocity_limit_sim == 5.0` rather than assuming the legacy unset
  representation.
- Local gates currently pass: 258 focused gripper diagnostic/finalizer tests;
  the wider non-Isaac suite at 448 tests plus 30 subtests; three focused robot
  config/helper tests under lightweight Isaac-module stubs; Ruff format/check,
  Python byte compilation, and `git diff --check`. Tamper cases reject a legacy
  profile paired with candidate config, the old 8.7266 value, swapped CPU/CUDA
  evidence, an unset live simulation cap, and a full-snapshot finger drive that
  disagrees with the startup contract.
- The first independent pre-commit audit was an explicit NO-GO despite all
  value/device checks passing: candidate intent was derived only inside the
  runtime capture, so accidentally omitting the runtime candidate flag could
  produce legacy 8.7266 behavior and still satisfy a finalizer that expected
  only mode `exact`. The repaired contract now derives and checks the expected
  profile independently in the stdlib parent; requires the finalizer's closed
  `--expected-gripper-drive-profile`; rejects candidate/legacy swaps; securely
  checks that diagnostic and finalizer profile sets agree; and records the
  expected/capture-matched profile in the v7 attestation. Adversarial tests
  cover candidate-flag omission, both profile-swap directions, open-ended
  profile input, missing/non-closed finalizer intent, forwarding into artifact
  validation, and attestation/capture mismatch.
- Fresh independent re-audit of the repaired source/test diff SHA-256
  `24c76a321c18eaef96e16c11129ce5cb4915a53ece9a7a662f7b1a5cd9eab36b`
  is GO for commit, push, deployment, and the mandatory real-Isaac canary, with
  no P0/P1 findings. It independently reran all reported gates and confirmed
  the prior profile-intent issue is closed end to end. This is explicitly not
  a promotion GO: the fully attested real-Isaac canary remains the integration
  gate. A single monolithic synthetic candidate-capture/finalizer test is a
  nonblocking P2 gap because each component boundary is adversarially covered
  and the real canary is authoritative.
- This is not yet a repair and has not been committed, deployed, or launched.
  Independent code/contract GO, a real-Isaac gate, immutable provenance, and a
  fully attested exact-action canary are still required. Promotion remains
  blocked unless the capture proves the live 5 rad/s cap and completes the
  diagnostic horizon without an arm velocity-guard failure.
