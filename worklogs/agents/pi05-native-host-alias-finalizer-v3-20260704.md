# pi0.5 native host-alias finalizer repair v3 — 2026-07-04

- Agent: `pi05_official_model_canary`.
- Base: `f9c82d75cc3dd880c53ae9a3d196f7c355527f10`.
- Trigger: official model canary job `1098704` produced a complete typed-failure transaction but the host finalizer rejected the valid incident descriptor after resolving container path `/lustre/fsw/...` to host alias `/lustre/fs11/...`.
- Scope: artifact identity/path comparison plus correction of stale user-visible
  all-six provenance from job `1098349` to the already-bound recovery smoke job
  `1098682`. No checkpoint, normalization, image, state, action, sampler,
  controller, evaluator physics, or policy semantics change.
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
