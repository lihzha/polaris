# PolaRiS EEF full-eval controller profile — 2026-07-03

- Agent: `polaris_mimic_eval_integration`
- Branch: `codex/eef-full-eval-profile-v7-20260703`
- Worktree:
  `/home/lzha/code/PolaRiS-worktrees/eef-full-eval-profile-v7-20260703`
- Exact base: `ad19745a97ed7c9b0330325324cc82e23575c896`
- Scope: implementation and host validation only. No commit, push, simulator,
  policy server, GPU, Slurm job, registry write, or canonical-checkout mutation
  was performed.

## Public selection and accepted stack

`EvalArgs.eef_controller_profile` and CLI
`--eef-controller-profile` now expose a closed, default-off selection:

- `baseline` is the default and performs no config mutation;
- `arm_slew_0p95_gripper_rate0p25_fixed_anchor86_mimic100_damping1p2_v3`
  is accepted only for `EgoLAPEefPose` with `--control-mode eef-pose`.

The candidate is configured before `gym.make` and enables only the reviewed
stack: failure-only arm ring, 0.95 nominal arm target slew, rate-0.25 gripper
target slew, fixed activation anchor for 86 physics substeps, and the live
Robotiq 2F-85 mimic overlay at frequency 100/damping 1.2. The wrist-energy
brake remains off. Configuration validation binds every flag, target/interlock
profile, overlay identity and pre-spawn zero-call state, and tracing action
identity.

## Durable evidence

The episode safety sidecar and runtime contract advance to schema version 6.
Every sidecar binds the controller profile, full arm/interlock candidate
report, optional failure-only arm ring, and candidate-only all-six gripper
trace. The runtime contract binds the profile and an initial-plus-per-episode
controller-report aggregate. Resume, reconciliation, aggregation, and runtime
serialization reject schema/profile/target-slew drift.

The controller-report validator closes arm ratio/vector identity, attempted
versus transactionally committed arm applies, endpoint observation,
activation/capture/target counts, exact 86-substep completion/cancel/current
countdown arithmetic, fixed-anchor residual and slew bounds, exact float32
anchor values, and the little-endian anchor digest. A first-substep numerical
abort is represented by one attempted arm apply but zero committed applies.

The reusable candidate gripper action wrapper is observational. It records an
exact initial snapshot, terminal snapshot, and last 64 all-six pre/post
substeps with action/endpoint/setter-target evidence. Validation binds strict
integer cadence, success `episode_length * 8` applies, a partial failed final
step, retained-entry continuity, setter target to post target, and empty-trace
initial/terminal identity.

Failure-only arm rings bind the exact effort/phase semantics, strict integer
identity, contiguous q/dq/target transitions, float32 post-minus-pre deltas,
and the established pre-physics-state float32 implicit-PD reconstruction and
effort-limit clamp. This preserves the producer contract that Isaac's
`computed_torque`/`applied_torque` are preceding `write_data_to_sim` buffers,
not post-state recomputations.

## Host validation

No CUDA/Isaac/Slurm process was launched.

- Focused config/profile/trace/runtime/eval contract tests: `123 passed`.
- Full host suite excluding only the real-Isaac robust-controller module:
  `819 passed`, `30 subtests passed`; one known `pynvml` deprecation warning.
- Ruff check and format check, Python byte compilation, and
  `git diff --check`: passed.
- `tests/test_robust_differential_ik.py` cannot collect in the host shell
  because Isaac Lab is unavailable. No production robust-controller source
  was changed; the accepted reporter and failure ring already exist at the
  exact base.

The exact broad command was:

```bash
mapfile -t TESTS < <(rg --files tests \
  | rg '^tests/test_.*\.py$' \
  | rg -v 'tests/test_robust_differential_ik.py$')
PYTHONPATH=$PWD/src /home/lzha/code/ego-lap/.venv/bin/python \
  -m pytest -q "${TESTS[@]}"
```

The intentional `scripts/eval.py` and lightweight config edits were rebound
in the existing Gate-0 production-source validator; its six affected host
tests pass without admitting an alternate source identity.

## Frozen pre-commit source identities

- `scripts/eval.py`:
  `4737a1a40cc1f30cd9bcbdb1e93d4a03ce61c9a92b6010ddf3651ce4dc446a83`
- `src/polaris/config.py`:
  `ec14fcac50f371b49731946b03fbc97518d6c651febeecc6d295f156b33b8763`
- `src/polaris/eef_controller_profile.py`:
  `07addcf57f81135986607bf916b8d6c68d258c540b10cce41d0070c7935b8746`
- `src/polaris/eef_gripper_failure_trace.py`:
  `8a9c29fb8c24db5df52330f717cbaed1c20dc4c9fbd7d56af7ad96e7d794aa8d`
- `src/polaris/eef_runtime_contract.py`:
  `b835e89a04c536ecb38feb5573a380fd059a80857a37581bdfac816be8e69918`

These are pre-commit identities. Re-run source pins and all gates after any
review edit. Real-Isaac validation and one-rollout checkpoint canaries remain
explicitly outside this implementation-only handoff. The remaining technical
risk is therefore runtime-only: the dynamically wrapped production finger
action, pre-spawn mimic overlay, and live Isaac tensor cadence have strong
host stubs and closed evidence validation but have not yet been exercised in
the real Isaac environment.

## Git handoff

- Implementation commit: intentionally not created.
- Push: intentionally not performed.
- Cluster evaluation: intentionally not launched.

## Independent-review repair cycle

The first independent review rejected the implementation for two P1 and three
P2 fail-closed gaps. No simulator or checkpoint job was launched while these
gaps were open.

- P1: configuration validation now binds the exact production arm action
  class, exact production finger action class, and exact original
  `spawn_from_usd` function before any candidate mutation. The candidate
  wrapper must have the single production finger base plus the closed
  production module/name/qualname identity; double-wrapped and arbitrary
  classes are rejected. Baseline validation also checks the original spawn
  identity without writing an overlay.
- P1: a numerical failure's arm ring now requires at least one attempted apply,
  zero pending entries, exactly `apply_calls - 1` completed entries before
  last-64 retention, and the exact dropped-prefix count. This closes the
  previously accepted empty late-failure trace while retaining the valid
  first-apply empty trace.
- P2: all-six gripper evidence now binds raw actions to exactly `0.0` or `1.0`
  and the requested endpoint to the corresponding exact float32 open/closed
  target. Successful episodes require identical arm/finger endpoint-change
  counts; a failed final apply may let the finger lead by at most one.
- P2: every released arm target is bounded by the nominal slew vector and the
  global live applied maximum is independently bounded by the candidate's
  nominal 0.95 vector plus the named `1e-6` safety allowance.

Post-repair host validation, with bytecode and pytest cache writes disabled:

- focused profile/trace/runtime/eval contract: `135 passed`;
- full host suite excluding only the real-Isaac robust-controller module:
  `831 passed`, `30 subtests passed`;
- Ruff check, Ruff format check, Python byte compilation, and
  `git diff --check`: passed.

The earlier frozen pre-commit hashes are superseded. A final source freeze will
be recorded only after the repaired tree receives a second independent
read-only review. The next permissible execution remains a short real-Isaac
L40S smoke; checkpoint canaries and standard evaluation remain blocked.

## Second independent review and signed-zero repair

The second read-only producer review found no P0/P1 and one final P2 before
launch: Python numeric equality admitted IEEE-754 `-0.0` as the exact open raw
action and requested endpoint even though its float32 bytes differ from
`+0.0`. The reviewer therefore returned NO-GO and no simulator was launched.

The all-six validator now compares little-endian float32 bytes for both the
binary raw action and its selected endpoint. It accepts only the exact `+0.0`
or `1.0` raw encodings and exact matching open/closed endpoint encoding.
Dedicated raw-action and endpoint `-0.0` regressions were added.

Post-repair validation:

- focused profile/trace/runtime/eval contract: `137 passed`;
- full host suite excluding only the real-Isaac robust-controller module:
  `833 passed`, `30 subtests passed`;
- Ruff check, Ruff format check, Python byte compilation, and
  `git diff --check`: passed.

A narrow independent signed-zero recheck passed with no findings: the reviewer
reproduced rejection for both fields, reran the focused suite (`137 passed`),
and returned GO for the short real-Isaac L40S smoke only. Checkpoint canaries
and standard evaluation remain blocked.

Final reviewed pre-commit source identities:

- `scripts/eval.py`:
  `4737a1a40cc1f30cd9bcbdb1e93d4a03ce61c9a92b6010ddf3651ce4dc446a83`
- `src/polaris/config.py`:
  `ec14fcac50f371b49731946b03fbc97518d6c651febeecc6d295f156b33b8763`
- `src/polaris/eef_controller_profile.py`:
  `71b4964496843fca73020e10b10c7bbe48e1639d04364689fe0a62c5d756a145`
- `src/polaris/eef_gripper_failure_trace.py`:
  `f66af5001f8636333f6db00948a64214909a43b7d6afd1af968397dea33280b0`
- `src/polaris/eef_runtime_contract.py`:
  `a5f868f58be6850b9b7a4f9f2c1b719d0c867ad969b2fbec26a843e25db90cef`

The reviewed implementation was committed as
`0611d384f5f26ef9bd8ff114be273e875c3fe719` with tree
`197858347b9f783a62d282e9edc66a354ec5b424`, exact parent
`ad19745a97ed7c9b0330325324cc82e23575c896`, and full binary commit-diff
SHA-256
`9572152853bfd3e251a88c22c30d0f2887565df8fa99a14e9dfc6a9c24faf67c`.
The implementation commit is the exact producer identity for Ego-LAP and the
cluster smoke. This following worklog-only evidence commit is not a substitute
for that source identity.

The agent branch was pushed to `lihzha` as
`codex/eef-full-eval-profile-v7-20260703`; its first worklog-only evidence head
was `f9847215734bec3ca2c003945cfa13cf4ebefd1b`. No simulator or checkpoint job
had been launched at that point.
