# pi05 position canary wrappers — 2026-07-05

- Agent: `pi05-position-canary-wrappers-20260705`
- Branch: `codex/pi05-position-canary-wrappers-20260705`
- Deliberate original base: `bf344db3554ed22624c2960154da21d7b233d683`
- Current core ancestor: `921a3e25999cfedd337411e1e7d63f5864bd2316`
- Scope: new production canary wrappers, host finalization/validation, pure tests,
  and this worklog only. No registry, canonical checkout, core adapter, or cluster
  mutation is owned by this agent.

## Goal and contract

Build a fail-closed, ordinary one-L40S FoodBussing canary for the immutable
official `gs://openpi-assets/checkpoints/pi05_droid` checkpoint. The only
accepted execution path is OpenPI FLOW-10 seed 0, response `15x8`, execute 8,
with each model arm command converted at execution time to
`q_target[t] = fresh_measured_q[t] + 0.2 * clip(command[t], -1, 1)` and sent as
an absolute position target for eight physics substeps. The protocol identifier
is `polaris-native-droid-freshq-delta0p2-position-h8-canary1-v1`.

The checkpoint contract is global DROID normalization (a `single_arm` override
is forbidden), native external/wrist RGB at `720x1280`, OpenPI resize-with-pad
to `224x224`, image order external/wrist/blank, and zero wrist rotation.

## Implementation

Implementation commit:
`3812df4ebb7649981a80a42f697b44749678db30`.

- Added a position-specific submitter, ordinary sbatch worker, and evaluator
  wrapper. All three reject the historical joint-velocity controller-gate
  variables before any launch work.
- Added a run-specific checkpoint snapshot transaction. It independently
  copies bytes (no hardlinks or reflinks), requires every directory mode `0555`
  and every object mode `0444` with link count one, verifies all 20 manifest
  MD5 values, and compares pre/post root and object identities excluding only
  the explicit verification phase.
- Added a position-specific inference-environment wrapper so no persisted
  artifact carries the obsolete direct-rad/s canary profile.
- Added a trace/CSV/runtime/sidecar/close-ready auditor. A nonfailed rollout
  requires exactly 450 executions, 57 queries, 3600 apply calls, and exact
  external/wrist camera cadence. Typed target-limit or all-DOF monitor
  numerical failures remain distinct.
- Added host finalization for source/OpenPI/DROID/checkpoint/image/asset/Slurm
  provenance, atomic bound-port plus real WebSocket handshake, saved-spool
  equality, full raw/summary video decode, H.264 yuv420p progressive fast-start
  summary creation, completion, `eval_success.txt`, and a non-authoritative
  canary registry candidate.
- Hardened the sole controller authorization to the full controller-smoke
  schema. The smoke commit may be an ancestor of a wrapper-only canary commit,
  but every governed source size, SHA-256, and Git blob must be identical. The
  same image, Hub assets, L40S/srun/spool, child close/ready, and runtime
  identities are mandatory. The obsolete minimal schema is rejected.
- Added explicit mounting/binding of external OpenPI Git metadata for the
  evaluator container, so a submodule checkout remains verifiable when its
  `.git` directory lives outside the OpenPI worktree mount.

## Validation

Passed:

```text
bash -n scripts/polaris/eval_pi05_droid_position.sh \
  scripts/polaris/l40s_pi05_droid_position_canary.sbatch \
  scripts/polaris/submit_pi05_droid_position_canary.sh

python -m py_compile \
  scripts/polaris/capture_pi05_droid_position_environment.py \
  scripts/polaris/verify_pi05_droid_position_checkpoint.py \
  scripts/polaris/validate_pi05_droid_position_trace.py \
  scripts/polaris/finalize_pi05_droid_position_eval.py

ruff check <all new Python files and focused test>

ruff format --check <all new Python files and focused test>

shellcheck scripts/polaris/eval_pi05_droid_position.sh \
  scripts/polaris/l40s_pi05_droid_position_canary.sbatch \
  scripts/polaris/submit_pi05_droid_position_canary.sh

pytest -q tests/test_pi05_droid_position_adapter.py \
  tests/test_pi05_droid_position_canary_wrappers.py
# 16 passed

git diff --check
```

The focused tests cover hardened-schema acceptance, ancestor plus identical
governed-byte acceptance, governed-byte divergence rejection, obsolete-schema
rejection, independent immutable checkpoint copying, strict pre/post comparison,
and non-array/old-gate launcher guards.

## External dependency and handoff

The orchestrator-owned controller smoke job `1099012` failed safely before
target execution because the first core runtime report conflated raw PhysX hard
limits with tolerance-adjusted guard limits. This agent did not launch, cancel,
or mutate that job. The core agent is preparing a corrected descendant. Per
orchestrator direction, this wrapper implementation is handed off against
`921a3e2`; the orchestrator will cherry-pick it onto the reviewed corrected-core
descendant and own the cluster canary. No cluster job was launched, cancelled,
or mutated by this agent, and this agent has no job requiring monitoring or
cleanup.
