# pi0.5 native host-alias finalizer repair v3 — 2026-07-04

- Agent: `pi05_official_model_canary`.
- Base: `f9c82d75cc3dd880c53ae9a3d196f7c355527f10`.
- Trigger: official model canary job `1098704` produced a complete typed-failure transaction but the host finalizer rejected the valid incident descriptor after resolving container path `/lustre/fsw/...` to host alias `/lustre/fs11/...`.
- Scope: artifact identity/path comparison; correction of stale user-visible
  all-six provenance from job `1098349` to the already-bound recovery smoke job
  `1098682`; and an explicit raw DROID gripper-observation boundary audit. No
  checkpoint, normalization, image order/resolution, action, sampler,
  controller, evaluator physics, or model-input transform changes.
- Required behavior: bind resolved path target plus exact size/SHA/mode/nlink while retaining the recorded lexical path so enclosing immutable contracts rebuild byte-for-byte. Reject a different resolved target and any content/identity drift.
- Rerun gate: host tests, static checks, read-only replay of job `1098704` close-ready/trace/video evidence, commit/push, and independent review before a fresh GPU attempt.

## Local validation

- Focused contract/finalizer tests: `33 passed`.
- Broader pi0.5/native tests with the exact pinned OpenPI submodule first on
  `PYTHONPATH`: `127 passed`; only external dependency deprecation warnings.
- Ruff format/check, `bash -n`, ShellCheck, and `git diff --check`: pass.
- Alias regressions bind the resolved target and exact size/SHA-256/mode/nlink,
  preserve the recorded lexical path, and reject wrong targets or identity
  drift.
- Gate-provenance regressions require job `1098682`, reject stale job
  `1098349`, and ensure the base-controller descendant authority names the
  current all-six job.

## Masked raw-gripper audit defect

- Once close-ready validation passed, preserved job `1098704` reached the trace
  audit and exposed a second masked host defect: PhysX produced a minimum raw
  normalized open-gripper value of `-1.701161989053901e-9`, while the audit
  required an exact mathematical `[0, 1]` bound.
- Pinned OpenPI `docs/norm_stats.md` defines open/closed gripper state as
  `[0, 1]`, but pinned official DROID commit
  `33ae6a67274f36d2e29525b86f23a56616ef43a7` computes observation state as
  `1 - width / max_width` without clipping. Pinned OpenPI `DroidInputs` also
  forwards it unchanged before checkpoint quantile normalization. Only DROID
  gripper commands are clipped.
- The repair therefore preserves the raw float32 value and adds a named bound
  tolerance of exactly eight float32 epsilons (`2^-20`,
  `9.5367431640625e-7`). The client rejects anything outside that envelope
  before the server; serving, live runtime, model-eval, and trace validators all
  bind the same contract.
- The all-six source gate permits only this exact additive, non-transforming
  guard and still compares the remaining request/image/action semantics to the
  integrated official-model base AST.
- Updated broader host suite: `163 passed`; Ruff format/check and
  `git diff --check` pass. A fresh no-model all-six L40S gate and rebind are
  mandatory because the contract, runtime, gate finalizer, and client are
  all-six-critical sources.
