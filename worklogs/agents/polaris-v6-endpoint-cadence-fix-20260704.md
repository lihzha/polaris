# PolaRiS v6 disabled-interlock cadence repair

## Scope

The paired one-rollout FoodBussing canaries from PolaRiS Commit B
`ee6d09351bed75e32db93ecf59c039a8e99fac9f` completed all 450 policy steps,
but both failed closed while constructing the schema-8 episode safety sidecar.
The official LAP-3B rollout recorded 15 finger endpoint changes and the
reasoning-43075 rollout recorded one; both arm reports remained at zero.
Neither metric is authoritative.

The root cause is checkpoint-independent.  The v6 concurrent path correctly
used fresh DLS targets and kept the close interlock disabled, but selected the
immutable disabled transition on every physics apply.  That transition also
froze `observed_endpoint_change_count` at zero, so the final arm/finger cadence
equality gate rejected truthful finger telemetry.

## Repair

Disabled concurrent mode now reads the installed finger driver's nonnegative
endpoint-change counter and stages it through the existing arm target
transaction.  The transition rejects counter regression or a jump larger than
one.  It remains inactive with zero remaining substeps, zero activation/
completion/cancel deltas, and `endpoint_observed=false`.  Recovery-owned target
applies synchronize the same cadence instead of suspending it; enabled v5
interlock suspension is unchanged.

The controller-report validator distinguishes the cadence cursor from actual
disabled-interlock control counters.  All hold, anchor, release, and maxima
evidence must remain zero.  The live moving-close/reopen smoke now cross-binds
the arm cursor to `driver_target_slew.endpoint_change_count`, so the historical
job-1098922 payload (`arm=0`, `finger=2`) is no longer promotable.

## Host validation before implementation commit

- Modified-path suite: 251 passed.
- Broad host-safe suite: 1,157 passed, one skipped, 30 subtests passed.
- The ten broad failures are closed historical identity gates: eight already
  present at Commit B (six old reset-replay `config.py` pins and two immutable
  v5 lineage pins), plus two current v6-promotion source pins that must remain
  closed until a new smoke and promotion descendant exist.
- Ruff lint/format, Python byte compilation, and `git diff --check`: passed.

## Promotion state

No checkpoint result, smoke-suite, or standard evaluation is authorized by
this implementation alone.  A fresh standalone L40S controller smoke must
exercise close/reopen endpoint changes, prove arm/finger cadence equality and
fresh DLS target application, and be finalized from a direct evidence-only
descendant before the same paired canaries may be retried.
