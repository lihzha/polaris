# LAP `panda_link8` frame correction and PolaRiS rerun

## 2026-07-01 — Root-cause audit and implementation

- Agent: `lap-link8-fix-rerun-20260701`; branch
  `codex/lap-link8-rerun-20260701`; base commit
  `c32b28732cda59eaf76a04a83e78aca1feffd092`.
- Scope: correct the physical end-effector frame used by the official LAP-3B
  and OXE magic-soup reasoning checkpoint evaluations, then rerun both cohorts
  on PolaRiS. Prior result trees remain immutable and scientifically excluded.
- Root cause: the prior integration observed and controlled the attached
  Robotiq `base_link`. DROID records `cartesian_position` from its configured
  Franka end effector, `panda_link8`, and LAP trains its state and deltas from
  that field. The old client therefore normalized one physical frame as if it
  were another and anchored every predicted chunk to the wrong link.
- Asset audit: the fixed joint in
  `PolaRiS-Hub/nvidia_droid/noninstanceable.usd` places the Robotiq frame
  18.1740224 mm from `panda_link8` with a 179.99999-degree relative rotation.
  This is not a naming-only discrepancy.
- Fix: the frame transformer now observes `panda_link8`, and absolute
  differential IK directly controls `panda_link8`. The policy request and
  trace carry an explicit `eef_frame=panda_link8`; both the policy client and
  evaluator fail closed on any different frame. The evaluator emits a marker
  containing the configured observation prim and controller body.
- The evaluator also asserts the `panda_link0` reference and identity source,
  target, and controller offsets. The controller smoke independently compares
  the policy observation against the articulation's direct `panda_link8` body
  transform at reset and after every axis test.
- Benchmark isolation: the original Robotiq `ee_frame` remains in place for
  PolaRiS rubric scoring. LAP state uses a separate `lap_ee_frame`, avoiding an
  unintended 18 mm shift in reach predicates.
- Initial CPU validation: ten LAP client tests and three TensorFlow-equivalent
  image-resize tests pass. Ruff lint/format, `py_compile`, and `git diff
  --check` pass.

## Pending execution gates

1. Run the Isaac-launched robust-controller tests and seven-axis controller
   smoke in the pinned CUDA 13 runtime. Inspect final pose errors and frame
   metadata.
2. Deploy the committed source through a complete Git bundle to a fresh L40S
   checkout. Run one FoodBussing rollout for each checkpoint in parallel.
3. Require exact checkpoint/config/normalization/image contracts, explicit
   `panda_link8` markers, finite 508-event traces, one complete CSV row, and a
   visually inspected 450-frame video before scale-out.
4. Submit six independent 50-rollout task jobs per checkpoint. Aggregate only
   complete frame-corrected cohorts and retain all failed attempts.
