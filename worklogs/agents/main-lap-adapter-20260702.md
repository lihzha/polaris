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
