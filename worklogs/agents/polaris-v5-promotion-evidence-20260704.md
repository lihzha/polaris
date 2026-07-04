# PolaRiS V5 Full-Horizon Evidence Promotion

Agent: `polaris-v5-promotion-impl-20260704`

Branch: `codex/eef-velocity-recovery-v5-promotion-20260704`

Base: `f11ae45a64b2f839dcb3325459ab06776d1dd81a`

## Goal

Pin the already inspected official LAP-3B and reasoning-checkpoint full-horizon
canaries without changing the v5 controller. Authorize only the next lifecycle
stage, the canonical six-task one-rollout smoke suite. Keep standard evaluation
blocked until that suite is complete and visually inspected.

## Evidence design

`src/polaris/eef_velocity_recovery_promotion.py` is a closed schema-1 manifest.
It binds Ego-LAP `74bb225d...`, PolaRiS producer `f11ae45a...` / tree
`7d68beea...`, all seven unchanged production-source digests, both checkpoint
identities and distinct protocols, jobs `1098707` through `1098710`, exact
runtime/sidecar/trace/rollout hashes, completion-audit/candidate/suite/summary
hashes, full-horizon cadence, video properties and visual findings.

Official completed 3,600 applies with zero recovery events. Reasoning recovered
one measured-velocity event from apply 2386 through apply 2404 using four hold
targets and all 16 release-ramp targets, then completed 3,600 applies with zero
abort. Its authoritative nested counters record two residual samples across two
joint exceedances, one recovery event, and one recovered event. Both task
results remain raw/task-valid `0/1`; this is controller and runtime evidence,
not evidence of task success or a success-rate estimate.

The manifest authorizes `canary` and `smoke_suite`. It explicitly rejects
`standard`; the named blocker is the missing completed and inspected six-task
one-rollout smoke suite.

## Validation

Focused evidence validation:

```bash
PYTHONPATH=$PWD/src /home/lzha/code/ego-lap/.venv/bin/python -m pytest -q \
  tests/test_eef_velocity_recovery_promotion.py
```

passed 30/30 tests, including either-canary omission, every completion artifact
hash, cross-job artifact swaps, checkpoint-specific protocol separation,
production-source hashing, manifest mutation, and scale authorization.

The complete host-safe PolaRiS suite passed 964 tests plus 30 subtests:

```bash
PYTHONPATH=$PWD/src /home/lzha/code/ego-lap/.venv/bin/python -m pytest -q \
  tests --ignore=tests/test_robust_differential_ik.py
```

The sole excluded module imports `isaaclab` at collection time, which is not
installed in this host environment. An unfiltered collection attempt failed
only at that known dependency boundary with `ModuleNotFoundError: isaaclab`.
No controller or runtime source changed, so no new Isaac/Slurm execution is
required or authorized for this evidence-only revision.

Ruff lint and formatting, byte compilation, `git diff --check`, canonical
manifest digest verification, and all seven producer-source digest checks
passed. A direct reconciliation against both fetched authoritative sidecars
and runtime contracts also matched every embedded recovery counter and maximum,
the reasoning event chronology, full-horizon cadence, result fields, and all
eight artifact hashes for each canary. The final commit is the evidence-only
commit containing this worklog;
its exact commit/tree/parent/diff identities are reported in the handoff.

No cluster job, local long-running process, deployment, registry write, or
shared artifact was created or mutated by this branch.
