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
