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
  endpoint -> 16 cumulative targets -> execute 4.
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
  and `1/4`, respectively, including recomputed top-level identities.
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
