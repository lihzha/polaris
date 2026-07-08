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
