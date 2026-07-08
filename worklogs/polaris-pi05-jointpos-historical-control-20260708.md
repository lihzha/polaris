# Official pi0.5 joint-position historical-control rerun

## 2026-07-08 — purpose and preserved control

- Goal: rerun the prior FoodBussing seed-0 implementation as a historical
  control after fixing the action-450 autoreset/scoring boundary and adding
  full train/eval/evidence attestation. This branch starts from historical
  seed-control commit `4645b4e092c7f3264f07e4b8216cd1641b289765`.
- The control intentionally retains its older direct evaluator/Isaac shutdown
  lifecycle and excludes the newer joint-velocity, delta-position, and
  `NativeEvaluatorLifecycle` integration. It is therefore not a copy of the
  current evaluator.
- Portable contract commit
  `5a575e6` carries the exact official checkpoint/config, global checkpoint
  DROID normalization, native-image/server-PIL resize probe, tokenizer and
  package attestation, joint-position client/controller/runtime checks,
  schema-4 trace/RNG closure, pinned asset/media validation, terminal PNG, and
  immutable evidence transaction used by the current arm.
- The branch-local hand-port adds only `runtime_contract_path` plus the native
  joint-position action/observation config, internal-451/outer-450 timeout,
  live runtime publication, per-rollout seed/reset binding, exact 450-step
  execution recording, render-every-step composite frames, and the trace-bound
  post-action-450 PNG. It preserves the historical direct `env.close()` and
  `SimulationApp.close()` behavior.
- Validation: the complete historical host-runnable suite passes 103 tests plus
  5 subtests with only the Isaac-dependent test excluded; the focused
  joint-position/evidence surface passes 65 tests. Ruff format/lint, Python
  compilation, and Git whitespace checks pass. The portable shell surface was
  already validated with Bash syntax and ShellCheck before cherry-pick. No GPU
  job has yet been launched from this final control candidate.
