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
  `5a575e6194b8bdf3a3a25e6b5d39bf8802d0e0eb` carries the exact official checkpoint/config, global checkpoint
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
- Branch-local integration commit:
  `94c48f525a53a67fb8c03841682b9e9bf16db3a2`.

## 2026-07-08 — shared schema-4 physical audit

- CPU setup job `1101795` was intentionally canceled before evaluation after
  the final prelaunch review found that the current tree's optional physical
  analyzer had not yet learned the schema-4 execution records. No L40S job was
  launched from that attempt.
- This historical control now carries the same byte-identical schema-4
  joint-bound analyzer, aggregate summarizer, and tests as the current arm.
  The analyzer requires one execution record per emitted action and audits
  every measured post-action state, including action 450; legacy query-only
  traces are retained only with an explicit lower-bound label. These files do
  not participate in rollout execution, so the intended historical evaluator
  lifecycle remains unchanged while paired raw and state-valid results use the
  same committed analysis implementation.

- The subsequent adversarial closure is also copied byte-identically from the
  reviewed current arm: homogeneous schema and complete query/action/execution
  ordering, query/post-action state continuity, canonical Panda limits and
  `1e-3` rad tolerance, trace/CSV SHA-256 binding, sealed-evidence metrics
  identity, and fail-closed aggregation of success, numerical, state, and
  target episode sets. It rejects execution deletion, schema downgrade,
  reordered pairs, missing queries, swapped steps, stale metrics, noncanonical
  tolerance, and impossible state-valid counts. These additions remain
  analysis/evidence-only and do not change the historical rollout lifecycle.
