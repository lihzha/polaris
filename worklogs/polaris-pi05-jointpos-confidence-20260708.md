# pi0.5 native joint-position confidence audit

## 2026-07-08 — plan

- Agent: `codex-pi05-jointpos-confidence-20260708`.
- Goal: close the remaining reproducibility and statistical-confidence gaps for
  `gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris` on the
  latest PolaRiS native `DroidJointPos` evaluator.
- Development base: `378f3a7d6db99bb1a3cbc807f0e91c1585b48e6f`; runtime
  behavior base: `25563f0b99ff03191aa7cc28c6947c60b4e6cafc`; OpenPI:
  `bd70b8f4011e85b3f3b0f039f12113f78718e7bf`.
- Intended changes: add an explicit evaluator environment-seed contract, pass
  and persist it through the native joint-position launcher, fail closed when
  the seed is absent, and add focused unit/static wrapper tests. Do not alter
  checkpoint normalization, image routing, model sampling, action conversion,
  gripper semantics, controller targets, or official task assets.
- Validation sequence:
  1. local Ruff, shell syntax, focused pytest, and dry-run launch validation;
  2. two fresh seed-0 two-condition FoodBussing runs, requiring matching seed
     provenance and comparing initial images, actions, metrics, traces, and
     fully decoded videos;
  3. paired ordinary one-L40S 50-condition FoodBussing evaluations for the
     historical and current evaluator trees under the same seed contract;
  4. paired success/progress statistics, state/target joint-bound audit, raw
     trace accounting, and representative success/failure video inspection.
- Scientific acceptance: no static train/eval mismatch; reproducibility
  behavior explicitly characterized; current success/progress statistically
  compatible with the historical control; no unexplained state-order,
  image-order, normalization, action, or controller regression. Target-only
  excursions remain separately reported because adding a clamp would define a
  different benchmark protocol.
- Expected cluster jobs: one bounded repeatability canary at a time until the
  seed contract is validated, then two ordinary FoodBussing full-50 jobs in
  parallel on l401 L40S. No arrays.

## 2026-07-08 — local seed-contract implementation

- Added `environment_seed` to the evaluator configuration and a lightweight
  `isaaclab_env_cfg_seed_v1` contract that validates an unsigned 32-bit seed,
  binds `env_cfg.seed` before `gym.make`, and emits one canonical log marker.
- Native `DroidJointPos` now fails before Isaac launch if no environment seed
  is supplied. The public worker, sbatch wrapper, submitter manifest, resolved
  command, and run metadata all carry seed 0 explicitly and completion-gate the
  exact runtime marker.
- This patch does not change the OpenPI policy RNG, checkpoint/config,
  normalization, image preprocessing, controller, action chunk, or gripper
  path. It only closes the environment-seed provenance gap identified by jobs
  `1090783`, `1090806`, and `1100641`.
- Local validation: Ruff passed; `bash -n` and ShellCheck passed; 61 focused
  tests passed using the Ego-LAP test environment, covering the new seed
  contract plus native client, lifecycle, and evaluator-contract behavior.

## 2026-07-08 — prelaunch hardening after independent review

- Independent review confirmed that construction-time `env_cfg.seed=0` is a
  useful control but cannot by itself prove a resumable or bitwise-deterministic
  protocol. Isaac seeds logical RNGs with nondeterministic torch settings,
  PhysX enhanced determinism remains false, and RTX/CUDA splat rendering may
  still vary at the pixel level.
- Replaced the minimal marker with live post-`gym.make` readback. The evaluator
  now requires the constructed environment to retain the base seed, records
  the live PhysX flag, and truthfully labels the claim `rng_bound_not_bitwise`.
- Added `base_plus_episode_index_v1`: every rollout explicitly calls
  `env.reset(seed=base_seed+episode_index, ...)`. The client binds the live
  contract, requires the matching episode/reset index and derived seed, and
  writes closed seed provenance on every schema-2 policy query.
- The trace validator now requires schema 2 and the expected base seed for
  seeded runs, rejects missing/mixed/tampered provenance, and reports the
  ordered episode seeds. Completion also rejects the old Isaac `Seed not set`
  warning and requires Isaac's own live `Environment seed: 0` report.
- Seeded resume is deliberately disabled: restarting the separate OpenPI
  server would restart its JAX request RNG stream, so a copied episode prefix
  would not yet be an exact continuation. A failed full run must restart in a
  fresh attempt rather than silently combine policy-noise streams.
- Added exact task asset SHA checks and the pinned PolaRiS-Hub revision to run
  metadata, an honest single-task `foodbussing50` submitter mode, manifest
  header validation, and a one-job/two-process sequential repeatability wrapper
  so the first seed test uses the same physical L40S.
- Updated validation: Ruff, `bash -n`, ShellCheck, and `git diff --check` pass;
  73 focused tests pass across environment seed derivation/live readback,
  client trace binding, trace rejection cases, native lifecycle, and evaluator
  contracts.

## 2026-07-08 — frozen-source setup preflight

- Frozen detached l401 sources were prepared for the then-current seed-only
  candidates at `734e162bfdacf60cd81a6a2b33388f468c54373c` and
  `4645b4e092c7f3264f07e4b8216cd1641b289765`, both with OpenPI
  `bd70b8f4011e85b3f3b0f039f12113f78718e7bf`.
- The first two setup submissions (`1101747`, `1101748`) failed before setup
  because an inline Slurm wrapper ran under `/bin/sh` and rejected `pipefail`.
  The next two (`1101749`, `1101750`) also failed before setup because nested
  shell quoting produced an unterminated string. These are preserved as
  launcher-only failed attempts; neither executed setup or evaluation.
- Standalone-Bash replacements `1101751` and `1101752` completed `0:0` on the
  CPU partition. Each independently installed the pinned OpenPI environment,
  fully MD5-verified all 27 checkpoint objects (12,434,530,837 bytes), matched
  manifest SHA-256 `7abd0c2294d442d429a77655783232206b2b30d95c508d435503135a5523a11c`,
  matched checkpoint-owned DROID normalization SHA-256
  `57ce9956f9e07d65f8a8205aabec72d436a2c8927f53edb40c7a77b14a5a90c7`,
  and resolved the expected pi0.5 joint-position config.
- Exact-source, non-submitting FoodBussing50 seed-0 dry runs then passed for
  both trees. Evidence is retained under
  `/lustre/fsw/portfolios/nvr/users/lzha/results/polaris-pi05/seed0-confidence-20260708T170848Z-freeze-remote-sources`.
  No GPU evaluation was launched from these superseded candidates.

## 2026-07-08 — full-confidence adversarial review and scoring correction

- A second independent review found no static mismatch in checkpoint,
  checkpoint-global DROID quantile normalization, image order/resolution,
  state order, FLOW sampling, delta-to-absolute action reconstruction, or
  execute-eight semantics. It did find that the live WebSocket server and the
  live Isaac joint-position controller were not independently attested: the
  old server returned empty metadata, while the trace stopped at the client
  emission and could not prove processed/articulation targets, cadence, joint
  order, or camera freshness.
- Direct inspection of the exact pinned simulator image found an additional
  scoring bug. Isaac Lab's `ManagerBasedRLEnv.step` resets timed-out
  environments before returning observations. PolaRiS evaluates its rubric
  after that call, while the old native joint-position profile timed out at
  action 450. The terminal metric could therefore miss action-450 state or
  observe reset-scene state. The historical 12/50 FoodBussing result is not
  accepted as full-confidence evidence under this boundary behavior.
- The corrected protocol retains exactly 450 policy actions but configures an
  internal 451-step timeout and requires all 450 returned terminated/truncated
  flags to be false. The action-450 live rubric and state are captured before
  any autoreset and cross-checked against the CSV metric.
- A behavior-preserving audited action subclass now calls the upstream
  `JointPositionAction` processing/setter unchanged, with no clamp or guard,
  while recording raw/processed float32 targets, all eight setter holds, and
  post-step articulation targets. Runtime evidence also binds exact arm state
  and action order, gripper semantics, direct PhysX and buffered drive/limit
  values, 120/15 Hz cadence, camera shape/dtype/frame counters, and episode
  counters.
- The replacement server still constructs the official unwrapped OpenPI
  policy and official WebSocket server. Before listening it fully verifies the
  checkpoint and attests the live `pi05_droid_jointpos_polaris` FLOW config,
  bfloat16/15x32 model, default Euler-10 sampler, JAX key 0, global DROID
  quantile statistics, exact transform order including DeltaActions and
  AbsoluteActions, image routing, leading-eight output, and imported OpenPI
  source identities. The client rejects missing or altered metadata and every
  schema-3 trace record cross-binds server, environment-seed, and Isaac runtime
  contract hashes.
- The worker now verifies Hugging Face metadata identity and the pinned Hub
  revision for every task's initial conditions and scene plus the shared robot
  asset. Seed derivation is prevalidated across the complete rollout count;
  seeded resume remains forbidden. Global policy request indices and per-
  episode cumulative query counts make any later paired comparison invalid
  after an early-termination RNG-stream shift instead of silently treating it
  as paired.

## 2026-07-08 — final prelaunch adversarial closure

- A final independent review rejected the first attestation candidate before
  launch because its dedicated observation config had inadvertently removed
  the historical gripper clip. The corrected config now preserves the exact
  effective historical observation surface: ordered seven-joint state,
  closed-positive gripper with configured Gaussian `std=0.05`, group
  corruption disabled, clip `[0, 1]`, and the existing EEF terms. The live
  runtime contract requires the generated observation-manager term order,
  functions, noise/clip settings, and arm/gripper state indices.
- The official unwrapped OpenPI policy and official WebSocket server are now
  bound to an owned IPv4 `127.0.0.1` listener. After evaluation, `SIGUSR1`
  snapshots the policy's stored JAX key and exits the server. An independent
  verifier recomputes the exact official `split(key)[0]` recurrence for 57
  requests per 450-step rollout and joins it to contiguous trace request
  indices. Any extra local connection that calls inference, or any missing
  evaluator request, changes the final key and fails the run.
- Server startup also attests CPython, `uv.lock`, `pyproject.toml`, all installed
  distribution versions against the lock, SHA-256 `RECORD` integrity for the
  inference-critical JAX/Flax/NumPy/Orbax/WebSocket wheels, live JAX flags,
  CUDA backend, and one L40S device. The checkpoint and normalization numeric
  digest remain independently bound to the model runtime and handshake.
- The complete selected task tree plus shared `nvidia_droid` tree is hashed as
  one closed dependency manifest, including USD/USDZ payloads, textures,
  splats, and every robot `SEGMENTED` mesh. The tree is scanned before serving
  and again after evaluation. For FoodBussing this is 36 files, 380,426,267
  bytes, tree SHA-256
  `36b80c4a9499b0643bec2775e6c33a0517212b494065cba6c1f94e511f9fd094`;
  an independent local and l401 scan matched.
- Completion now stops the finalized server, rechecks both Git trees, seals
  the trace, CSV, trace summary, request/RNG proofs, runtime/model/serving and
  asset contracts, checkpoint report, commands, logs, GPU/source records, and
  every numbered video as mode `0444`, single-link files. A canonical evidence
  manifest reopens and semantically revalidates every identity before either
  task or run `SUCCESS` is created.
- Final local validation after integration: 383 tests passed plus 5 subtests;
  the Isaac-only local test remains intentionally excluded. Ruff formatting
  and lint, Python compilation, Bash syntax, ShellCheck, and Git whitespace
  checks pass. No GPU rollout has yet been launched from this candidate; the
  pinned-container canary remains the live acceptance gate.

## 2026-07-08 — terminal and media evidence closure

- Dry runs now publish an immutable `DRY_RUN` marker and can never enter the
  normal run-success branch. `SUCCESS`, `FAILED`, and `DRY_RUN` are created by
  a non-overwriting, fsynced hard-link transaction with mode/link/readback
  validation; a publication or cleanup failure leaves the scheduler exit
  nonzero. Normal success additionally requires the evidence transaction to
  have completed and returned its manifest identity.
- The final evidence transaction now seals the raw policy trace and metrics
  CSV, independently reruns the schema-3 trace validator against those sealed
  inputs and the sealed runtime/server contracts, and requires byte-canonical
  equality with the persisted trace summary before publishing the manifest.
- Every expected `episode_N.mp4` is independently probed and fully decoded in
  the pinned Pyxis image. The gate requires exactly one H.264/yuv420p
  progressive 448x224 stream, 450 decoded frames at 15 fps, and 30 seconds,
  then joins the pre-seal decode identity to each sealed video identity.
  Execution is bound to Pyxis image SHA-256
  `ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a`,
  in-image `/usr/bin/ffprobe` SHA-256
  `d4f3ef9c12be756793cad83dd2004d89f49c1c4094053bfbbe7e28925c8fa4fd`,
  and `/usr/bin/ffmpeg` SHA-256
  `36d94a605d612e4090d1b8aec889d0c0801c6eafb1593c90f5c0dfd2e2966a45`.
- The exact probe/full-decode implementation was exercised on the retained
  native FoodBussing canary video and observed H.264/yuv420p progressive
  448x224, 450/450 frames, 15/1 fps, and 30.0 seconds with an error-fatal full
  decode ending at frame 450. The terminal-marker publisher was also executed
  in tests and proved mode `0444`, single-link, and non-overwriting behavior.
- Final local regression: 401 tests passed and one unrelated test skipped with
  only the Isaac-dependent local test excluded. The focused evidence/media
  suite passed 43 tests. Ruff, Bash syntax, ShellCheck, and Git whitespace
  checks pass. This closure launched no new GPU rollout; the fresh pinned-image
  canary remains the final live-system acceptance gate.

## 2026-07-08 — training-matched server-side image resize

- A final scientific audit measured a small but real pixel-level mismatch in
  the released evaluation path: client-side PIL bilinear resize and the
  training-time OpenPI/JAX linear resize differed by at most one intensity
  value on a synthetic native frame. The native joint-position client now
  sends byte-identical 720x1280 uint8 external and wrist images to the official
  unwrapped policy server. `DroidInputs` preserves external/wrist/masked-third
  routing, and the existing live `openpi.transforms.ResizeImages` performs the
  sole spatial model transform with JAX linear resize and zero padding to
  224x224, matching training.
- PIL resize remains only for the side-by-side rollout video. Handshake,
  model-runtime, schema-3 query traces, trace summaries, and immutable evidence
  now distinguish native request hashes from non-model visualization hashes
  and fail if a client-resized request is supplied. A synthetic client test
  proves both native request byte identity and 224x224 visualization isolation.
- Validation after the change: the expanded focused surface passed 79 tests;
  the full non-Isaac suite passed 404 tests plus 5 subtests. Ruff, Bash syntax,
  Python compilation, and Git whitespace checks pass. No GPU rollout was
  launched by this patch.

## 2026-07-08 — hermetic setup and historical-control portability closure

- The clean setup path now requires an explicit unique, unused absolute setup
  record directory, removes only a validated non-symlink OpenPI `.venv`, and
  rebuilds it with frozen, no-cache, copy-mode reinstalls. CPU setup and live
  GPU startup both validate every noneditable installed distribution against
  `uv.lock` and every installed `RECORD`; only the separately source-attested
  editable `openpi` and `openpi-client` distributions are exempt. The live
  PaliGemma tokenizer additionally binds its exact GCS generation, remote
  size/MD5, local SHA-256, serialized SentencePiece proto, vocabulary, and
  special-token IDs. `sentencepiece==0.2.0` and `tensorstore==0.1.74` are
  explicit runtime requirements.
- Correction to the earlier shutdown description: `SIGUSR1` cancels the
  official server's `run()` task first; the WebSocket async context closes the
  listener and drains all handlers before the policy RNG key is read and
  published. A live server test verified that the owned loopback port was
  closed before the RNG snapshot appeared. The independent recurrence proof
  therefore cannot race an in-flight request.
- The worker now hashes the selected Vulkan ICD in addition to the pinned
  Pyxis image. Setup and evaluation publication remain invalid until source is
  committed and clean; exact submitted Slurm script and launch argv will be
  preserved with each real attempt.
- A historical-port audit found that the first evidence implementation
  transitively imported the unrelated native joint-velocity contract. Seven
  immutable-artifact primitives were copied AST-identically into the new
  lightweight `pi05_droid_jointpos_immutable` module, and the joint-position
  evidence, video, and RNG verifier now depend only on the portable
  joint-position surface. Imports pass with the native joint-velocity module
  explicitly unavailable.
- Final combined-tree local validation: 404 tests passed plus 5 subtests with
  only the Isaac-dependent test excluded. Ruff format/lint, Python
  compilation, Bash syntax, ShellCheck, and Git whitespace checks all pass.
  No GPU job has been launched from the final candidate; the clean l401 setup
  and one-rollout full-horizon canary remain mandatory before scale-up.

## 2026-07-08 — resize correction, full-frame rendering, and terminal visual proof

- Correction to the preceding image-resize entry: the pinned OpenPI
  `ResizeImages` transform imports
  `openpi_client.image_tools.resize_with_pad`, whose live implementation is
  PIL bilinear resize with symmetric zero padding. It does not call the JAX
  helper. The transform produces uint8 224x224 images, then
  `Observation.from_dict` maps them to float32 `[-1, 1]`; the model-level JAX
  resize is inactive because the images are already 224x224. Training uses the
  same data-config transform sequence.
- The native 720x1280 request change remains valid: a deterministic
  production-shaped probe proved the server-side PIL output byte-identical to
  the historical client-PIL/server-identity path. The live model-runtime gate
  now binds the transform's actual imported helper and a pinned input/output
  byte digest, and the OpenPI source attestation explicitly includes
  `openpi_client.image_tools`. All handshake, trace, worker, and evidence
  labels now describe the real PIL pipeline; the superseded JAX claim above
  must not be used.
- `DroidJointPos` is now included in the render-every-step path. This makes all
  450 saved frames use the composed splat-plus-robot renderer instead of
  falling back to raw Isaac RGB on seven of every eight open-loop actions.
- The rollout MP4 remains exactly 450 pre-action frames for protocol
  continuity. Schema-4 execution traces additionally hash a 448x224 uint8
  visualization from the returned post-action-450 expensive splat
  observation. The evaluator writes `episode_N_terminal.png`; pinned ffprobe
  and ffmpeg validate PNG/rgb24/448x224 and hash the fully decoded RGB bytes.
  The immutable evidence transaction seals every terminal PNG and requires
  its decoded pixel hash to equal the action-450 trace before success.
- The public submitters now non-overwritingly preserve the exact Slurm spool
  batch script and shell-escaped submission argv, hash both, record their
  provenance directory, and cancel a newly submitted job if capture fails.
  The repeatability job requests two hours by default.
- Final combined local regression after these corrections: 408 tests passed
  plus 5 subtests with only the Isaac-dependent test excluded. Ruff
  format/lint, Python compilation, Bash syntax, ShellCheck, and Git whitespace
  checks pass. The final trace/video/evidence protocols are schema 4 / full
  decode v2 / evidence transaction v3. No final-candidate GPU job has yet run.
- Frozen local implementation commits: portable joint-position contract
  `2dc2a66cca6e7e60a7c9afee0ac50a4bfa27d3e4`; current-tree evaluator lifecycle
  integration `2e2bdfa15c74d6901a66f986d0a13dc6b2ef23ca`. The historical control will
  consume the portable commit and a separately reviewed hand-port of its
  intentionally older evaluator lifecycle.

## 2026-07-08 — schema-4 physical-audit gate

- The first clean setup attempts, current job `1101794` and historical job
  `1101795`, were intentionally canceled before any GPU launch. A final
  prelaunch check found that the separate state/joint-bound analysis script
  still accepted only legacy query/action trace records and rejected the new
  schema-4 per-action execution records. Rollout, rubric scoring, trace
  validation, and immutable evidence were unaffected, but publishing a
  state-valid success rate from that analyzer would not have been possible.
- `audit_pi05_joint_bounds.py` now fail-closes on one-to-one action/execution
  identity, requires execution indices `0..episode_length-1`, and audits the
  measured post-action state at every step in addition to the initial query
  state. Schema-4 state-OOB counts are therefore exact over all recorded
  rollout states, including the returned post-action-450 state. Legacy traces
  remain supported and are explicitly labeled as policy-query-only lower
  bounds. Target-only excursions remain separate from measured-state
  excursions.
- New tests prove that a transient between-query state violation is detected,
  a success with such a violation is excluded from the state-valid numerator,
  mismatched action/execution targets are rejected, and legacy traces retain
  their lower-bound label. The analyzer also reproduced the existing
  query-only result on the retained two-episode canary trace.

- A second adversarial review rejected the first analyzer fix before GPU
  launch. Setup retry `1101798` was canceled after seven seconds. The review
  showed that deleting every execution record could falsely downgrade a
  schema-4 trace to legacy, orphaned query references and swapped execution
  indices were not rejected, and the aggregate summarizer trusted stale audit
  JSON. No evaluation ran from that commit.
- The closed analyzer now requires one homogeneous trace schema, makes
  execution records mandatory for schema 4, reconstructs every expected
  `(episode, query, chunk)` key from the metrics, checks each emitted arm
  target against its query response, and binds execution step `q*8+chunk`.
  It hashes the source trace and CSV into audit schema 2. A realistic 450-step
  test proves 57 queries, 450 actions/executions, and correct detection of a
  violation present only in the post-action-450 state; downgrade, missing-
  query, swapped-step, missing-legacy-action, and stale-target attacks fail.
- The aggregate summarizer now re-hashes trace/CSV inputs and validates trace
  and physical-audit status, schemas, episode lengths, query/action/execution
  counts, success/numerical/state/target episode sets, and the state-valid
  numerator before aggregation. State-OOB-minus-numerical is a set difference,
  so the reported count cannot become negative, and impossible rates above
  one are rejected upstream rather than rendered.
- The final adversarial closure also binds the canonical Panda joint table and
  exactly `1e-3` rad tolerance, prevents a schema-4 audit from claiming legacy
  query-only coverage, and adds the sealed metrics-CSV SHA-256 to the primary
  trace validator summary. Immutable evidence now joins that hash to the
  separately sealed CSV identity, so editing metrics and rerunning only the
  physical audit cannot preserve a passing trace-validation claim.
- The physical analyzer additionally enforces contiguous JSONL query/action/
  execution order, including rejecting reordered complete action/execution
  pairs, and exact equality between each noninitial policy-query joint state
  and the preceding post-action state. Final current-tree host regression after
  these changes: 427 tests passed and one unrelated test was skipped with only the
  Isaac-dependent test excluded; focused trace/physical/evidence tests passed
  38/38, and Ruff format/lint, Python compilation, and whitespace checks pass.

## 2026-07-08 — Lustre checkout-alias attestation fix

- Final clean setup job `1101806` rebuilt all 242 locked packages, then failed
  closed before checkpoint evaluation because Git canonicalized the checkout
  from `/lustre/fsw/...` to its identical `/lustre/fs11/...` backing path while
  `sys.executable` retained the `/lustre/fsw/...` spelling. The verifier used a
  lexical path comparison. Historical setup `1101807` was canceled after the
  current failure established the shared cause; no GPU evaluation launched.
- Checkout-local interpreter verification now requires both the resolved
  `sys.prefix` to equal the resolved checkout `.venv` and `samefile()` identity
  between the declared and expected interpreter. This accepts only aliases of
  the same environment while still rejecting a different virtual environment
  whose Python symlink happens to share the same base interpreter. After that
  proof, it canonicalizes the persisted declared-executable path to the Git
  checkout root, preserving downstream model-runtime lexical binding.
  Dedicated helper and end-to-end model-runtime alias/adversarial tests pass;
  full current host regression is 429 passed and one unrelated test skipped.

## 2026-07-08 — pinned-wheel RECORD encoding failure

- Replacement setup job `1101808`, from immutable PolaRiS commit
  `8edce3ed6b0672b387898f09c5b950a5821cfbf3` and OpenPI commit
  `bd70b8f4011e85b3f3b0f039f12113f78718e7bf`, rebuilt and installed all 242
  locked packages. It then failed closed in the package-integrity gate before
  tokenizer/checkpoint validation or any GPU evaluation. The exact exception
  was `Installed distribution RECORD mismatch: augmax
  augmax-0.4.1.dist-info/METADATA`.
- The terminal Slurm state is `FAILED` with exit `2:0`, from
  `2026-07-08T20:05:45Z` through `2026-07-08T20:09:03Z`. The log is
  `/lustre/fsw/portfolios/nvr/users/lzha/slurm_logs/polaris-pi05/pi05-confidence-final-v4-20260708T200401Z/setup-current-1101808.out`,
  12,764 bytes, SHA-256
  `0d21c4b08187c9e0c60148c3ec412411ae16c65d0600d27d98925eb1af545f37`.
  Its setup-record directory is empty, so no incomplete setup artifact can be
  mistaken for a validated environment.
- Historical setup `1101809` was canceled after the current failure identified
  the shared gate. It had prepared 242 packages but had not independently
  reached the RECORD verifier. No rollout job launched from either setup, and
  the namespace has no remaining Slurm job or login-host process.
- Initial forensic comparison shows this is not an installed-byte mutation:
  the exact wheel pinned by `uv.lock` has SHA-256
  `60f9711a4ffc08f27d1ff0783f7c51c01e6f78e20d4581d075ebf2d904ab2d14`
  and itself stores lowercase hexadecimal SHA-256 values in its RECORD, while
  the wheel specification requires URL-safe base64. The bytes and hexadecimal
  digests agree. The verifier currently interprets every value as base64 and
  therefore rejects the malformed but content-matching upstream RECORD. A
  correction must stay fail-closed by pinning the exact nonstandard entries;
  a package-wide exemption is not acceptable.
- The complete clean-environment census covered 242 distributions and 45,714
  noneditable hashed files. Exactly 11 nonstandard hashes exist, all in the
  locked Augmax wheel and all matching their path, size, and SHA-256. It also
  exposed three genuine last-writer-wins collisions: co-installed
  `opencv-python-headless==4.11.0.86` and `opencv-python==4.11.0.86` claim
  different bytes for `cv2/cv2.abi3.so`, `cv2/typing/__init__.py`, and
  `cv2/version.py`; the live files exactly match the locked non-headless wheel.
  No other encoding, size, or content discrepancy exists.
- The verifier now binds the exact Linux x86_64 Augmax and both OpenCV wheel
  URLs, sizes, and lock SHA-256 values. It accepts hexadecimal RECORD values
  only for the exact 11 Augmax `(name, version, path, size, digest)` tuples and
  requires the complete tuple set. The three OpenCV conflicts are explicit
  loser-to-active-winner resolutions: each requires the exact losing RECORD
  claim, exact live winner digest/size, the exact active distribution/version/
  RECORD claim, and `samefile()` identity. Extra, missing, reversed, or changed
  overlaps fail. Every direct RECORD entry also requires both its SHA-256 and
  recorded size.
- The package report now states the honest invariant that every noneditable
  file is bound either to its locked RECORD or to the pinned co-install winner,
  and seals per-mode counts plus the complete three-resolution list. The setup
  verifier additionally requires all three pinned wheel distributions to be
  installed at the exact version; an independent review found and closed the
  initial fail-open case where deleting a pinned distribution's metadata could
  otherwise skip its completeness check.
- Focused contract regression after closure: 24 tests passed. The complete
  current host-safe suite before the final completeness test was added passed
  434 tests with one unrelated skip; Ruff format/lint, Python compilation, and
  whitespace checks pass. Direct execution against an existing 242-package
  OpenPI environment validates all 13 hashed Augmax entries and the exact three
  OpenCV overlap resolutions; that older environment separately fails closed
  on an unrelated stale `numpydantic` file, as expected rather than being
  silently accepted.

## 2026-07-08 — deterministic OpenCV active-provider closure

- Clean setup v5 current job `1101810` rebuilt all 242 packages and then
  failed closed on `opencv-python cv2/cv2.abi3.so`. This was the reverse of
  the preceding clean v4 environment: v5 ended with the locked headless wheel
  active at all three shared paths, while v4 ended with the locked full wheel
  active. Exact inspection showed every live path matched the headless RECORD;
  no byte was outside the two locked artifacts. The failure therefore exposed
  nondeterministic last-writer order in `uv sync`, not corruption.
- A diagnostic frozen-lock reinstall using
  `uv sync --frozen --no-cache --reinstall-package opencv-python --link-mode
  copy` deterministically restored the direct OpenPI dependency as the active
  provider. The full wheel then verifies 130/130 direct hashes, and the
  headless wheel verifies 95 direct hashes plus exactly the three sealed
  overlap resolutions. No package is exempted or removed.
- Clean setup now performs that exact lock-bound one-package reinstall after
  the complete 242-package rebuild and before editable OpenPI reinstall and
  package attestation. The resulting active runtime is deterministic and the
  already reviewed overlap profile remains narrow; accepting either provider
  nondeterministically is explicitly rejected.
