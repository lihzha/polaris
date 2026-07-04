# Reasoning full-trace cap/release follow-up (2026-07-03)

## Scope and evidence

This diagnostic-only follow-up is rooted at PolaRiS commit
`26f75a1aeb2e6342d45f96d746ee101be02764f5`, whose first parent is the
production controller candidate `0611d384f5f26ef9bd8ff114be273e875c3fe719`.
The completed source diagnostic showed:

- follower cap 5 rad/s: the first released arm target was apply 2334; Panda J7
  reached 2.592 rad/s at apply 2344 and 2.924 rad/s at apply 2345, over its
  2.610-rad/s live limit;
- source/default follower limit 174.533 rad/s: the trace completed, but a
  passive follower reached 20.514 rad/s and 2590.13 rad/s^2, so that arm is not
  a production-safe answer;
- extending the fixed anchor by 16 substeps moved release to apply 2350, only
  two traced substeps before the original horizon ended. It established causal
  timing but did not test sustained post-release safety.

## Pre-registered follow-up

Exactly three variants are admitted by both CLI and wrapper:

1. `cap8_abrupt_release`: follower limits 8 rad/s, production abrupt release;
2. `cap24_abrupt_release`: follower limits 24 rad/s, production abrupt release;
3. `cap5_release_ramp16`: follower limits remain exactly 5 rad/s; only the
   seven arm position targets receive a 16-substep inclusive linear release
   ramp with fractions 0/15 through 15/15 of the nominal 0.95-physical-limit
   arm slew.

The 8-rad/s arm probes just below the 8.218-rad/s cap-5 observed transient. The
24-rad/s arm is above the 20.514-rad/s source/default-limit observed transient.
The cap arms are diagnostic; the cap-5 ramp is the production-oriented arm.

All variants replay the 294 fixture actions byte-for-byte. If those actions
complete, they then repeat source action 293 for exactly eight policy steps,
or 64 physics substeps at decimation 8. The result reports source-action and
tail cadence separately. A fail-closed numerical abort stops the tail and is
accepted only when the parsed global policy/substep identity exactly binds to
the recorded trace length.

## Contracts retained

- byte-pinned job-1098523 action fixture and source trace hashes;
- exact FoodBussing IC0 scene, initial-condition, robot USD, PolaRiS-Hub
  revision, production reset/render/client sources, container, and repository
  ancestry;
- 120-Hz physics, 15-Hz policy, decimation 8, absolute panda-link8 EEF DLS
  controller, factor-0.25 driver slew, fixed-anchor-86 interlock, and
  frequency-100/damping-1.2 live mimic overlay;
- causal full 13-DOF `pre`, `command_after_setters`, and `post` snapshots for
  every completed physics substep;
- immutable result/video publication, fast-start/progressive full video decode,
  zero simulator `srun` exit, and separate post-Kit finalization.

## Implementation status

- Worktree: `/home/lzha/code/PolaRiS-worktrees/reasoning-fulltrace-cap-release-followup-v1-20260703`
- Branch: `codex/reasoning-fulltrace-cap-release-followup-v1-20260703`
- No commit, push, or cluster launch was performed.

Host-safe validation completed:

- `ruff check` on the runner, post-Kit validator, and focused tests: pass;
- `ruff format --check` on the same files: pass;
- `shellcheck` and `bash -n` on the Slurm wrapper: pass;
- `git diff --check`: pass;
- 72 host tests spanning this follow-up, the inherited controller-candidate
  contract, target-slew host stub, finalizer, and failure verifier: pass in
  1.30 seconds.

An independent pre-launch review then returned NO-GO on four evidence gaps:

- counters did not prove the live ramped arm target or its float32 formula;
- wrapper state advanced before the diagnostic target setter;
- a synthetic numerical-failure object could be promoted without the live
  controller abort report and digest binding;
- the final tail physics state had no following controller apply, so a final
  arm velocity excursion could escape the next-apply guard.

The follow-up was hardened before any launch:

- every ramp apply now records arm IDs, float32 fraction, live current joint
  position, nominal pre-overlay target, nominal slew vector, setter readback,
  and final target; validation independently recomputes the endpoint-exact
  float32 clamp formula and binds it to the three-phase command snapshot;
- ramp state/counters/overlay records commit only after the second setter,
  exact live readback, and rewrite of the pending controller failure-trace
  target all succeed; a setter-failure test proves no diagnostic ramp state is
  advanced;
- numerical failures now require the exact
  `DifferentialIKInvariantError`, independently captured controller safety and
  substep trace, canonical abort-evidence digest/message, terminal full-trace
  identity, and a post-Kit recomputation of the outcome from trace length;
- all traced arm pre/command/post positions and velocities and every actual arm
  target are checked against the pinned live soft-position, velocity, and slew
  limits. A completed result must include a safe final post-tail state; only a
  closed numerical-failure boundary may carry the terminal velocity excess;
- diagnostic ramp state is explicitly cleared after the base controller reset,
  and the profile is named as a per-substep slew-cap ramp rather than
  release-origin interpolation.

After hardening, 858 host-safe tests plus 30 subtests pass in 8.36 seconds.
The focused follow-up file has 25 passing tests, including target-formula
tamper rejection, setter-failure transaction behavior, reset, synthetic abort
rejection, outcome recomputation, and final-tail safety. Ruff, formatting,
ShellCheck, `bash -n`, Python compilation, and `git diff --check` pass.

The final independent review returned GO for commit and the pinned diagnostic
launch with no remaining P0/P1. Its additional adversarial checks included 150
combined tests, 21 failure-ring boundary tests, and 20,000 real-PyTorch
float32 formula cases. It explicitly remains NO-GO for direct production
promotion until the selected behavior is implemented inside the core
controller transaction.

Isaac/PhysX execution was intentionally not run or submitted in this review
stage.

## Design caveats

- The release ramp is a diagnostic wrapper after the production arm action. It
  changes the live articulation's seven arm position targets only; the
  production controller's internal pre-wrapper aggregate applied-delta
  counters still describe its nominal target. The pending controller failure
  trace is rewritten to the setter-verified actual target, and the closed
  overlay plus full command-after-setters trace are authoritative for the
  ramped target. A wrapper-originated setter/readback/trace failure aborts the
  entire diagnostic and cannot publish SUCCESS; production integration must
  still place the chosen release behavior inside the controller's single
  staged setter/commit transaction.
- The 64-substep tail freezes the final source command, not the physical state.
  It is a safety-observation extension and is not counted as additional source
  policy actions or task success evidence.
- Passing cap 8 or cap 24 would locate a solver-regime threshold but would not,
  by itself, authorize a higher passive-joint limit for production.
- An OPEN endpoint change during an already-running ramp is not separately
  consumed by this diagnostic state machine. The pinned fixture has no such
  timing: its first two OPEN changes start their ramps, and each ramp completes
  before the next CLOSE. Production integration must make that endpoint policy
  explicit and test it across resets and multiple episodes.
