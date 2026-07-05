# PolaRiS development worklog

## 2026-06-30 — Ego-LAP absolute EEF-pose evaluation integration

- Agent: `implement_polaris_eef`; branch
  `codex/ego-lap-eef-eval-20260630`; base commit `2f4046b`.
- Goal: add a selectable absolute end-effector pose evaluation path for
  Ego-LAP without changing the existing joint-position default.
- Controller/observation change: added Isaac Lab absolute differential IK for
  Robotiq `base_link` (`pose`, `use_relative_mode=False`, DLS, scale 1) and
  exposed `panda_link0 -> base_link` position/quaternion observations.
- Client change: registered `EgoLAPEefPose`; added DROID external/wrist image
  mapping with 180-degree wrist rotation, 10-D `xyz + rot6d + open-gripper`
  state, strict finite `T x 7` response validation, one-anchor chunk conversion,
  SciPy `xyzw` to Isaac `wxyz` conversion, open-to-closed gripper inversion,
  full-step splat visualization, and optional JSONL traces.
- Evaluation change: controller selection now occurs before `gym.make`; known
  policy/controller mismatches fail early; instruction overrides are honored;
  initial-condition indexing no longer repeats an episode; and videos request
  one frame per executed step.
- Supporting change: added a headless scripted axis/rotation controller smoke,
  pure unit tests, dependency metadata, and usage documentation.
- CPU validation:
  - `PYTHONPATH=... /home/lzha/code/ego-lap/.venv/bin/python -m unittest discover -s tests -v`:
    6 tests passed.
  - `ruff check` on all changed Python files: passed.
  - `ruff format --check` on all changed Python files: passed.
  - `python3 -m py_compile` on all changed Python files: passed.
  - `git diff --check`: passed.
- Dependency note: SciPy 1.15.3 was already resolved in `uv.lock`; it is now a
  direct PolaRiS dependency for quaternion and rotation conversion. A fresh
  `uv lock` remains blocked by upstream packaging: the broad Python matrix
  cannot resolve Isaac Lab 2.3 on unsupported interpreters, and a 3.11-only
  attempt reaches `flatdict==4.0.1`'s undeclared `pkg_resources` build need.
  The existing resolved versions were therefore preserved and only the
  PolaRiS-to-SciPy lock edges were added.
- Execution status: no Isaac Sim, policy-server, GPU, or cluster job was
  launched, as requested. The controller smoke and an end-to-end rollout remain
  target-runtime validation steps.

## 2026-06-30 — CUDA runtime and end-to-end validation

- Built a local CUDA 13 / Isaac Lab 2.3 runtime for the RTX 6000 Ada. The first
  base image exposed a dangling `/.venv/bin/python` symlink because uv's managed
  interpreter tree was not copied into the runtime stage. The base recipe now
  copies `/root/.local/share/uv/python`; the working base tag used for validation
  is `polaris-isaaclab-base:cu130-isaaclab230-r1`.
- Reworked the final recipe to build wheel-only copies of the two splat CUDA
  extensions in a CUDA 13 builder, pin `arhanjain/splat_kernels` commit
  `be51fb1ddc3618474c88fc3d2b529397700097f2`, and assert with `cuobjdump` that
  both wheels contain `sm_89`. This avoids copying the roughly 17 GiB virtual
  environment through an extra build/final layer.
- Final image: `polaris-eval:cuda13`, digest
  `sha256:e32265e25ae2d61b88cf0c530d5561d3fd48e6b2655f48cb583d18fb18851006`.
  Container validation reported Python 3.11.15, Torch 2.9.1+cu130, CUDA 13.0,
  SciPy 1.15.3, RTX 6000 Ada compute capability 8.9, and successful imports of
  both CUDA extensions.
- `uv pip check` reports the known CUDA-13 override mismatch: Isaac Sim's
  metadata pins Torch 2.7/Torchvision 0.22 and torchaudio pins Torch 2.7, while
  the upstream dockerize dependency set selects Torch 2.9.1+cu130/Torchvision
  0.24.1. This was retained because the real Isaac/PhysX/rasterizer smokes below
  passed; the mismatch is documented rather than hidden.
- The first controller launch initialized Isaac and then exited before the
  Python test body because a source-layout bind mount lacked
  `PYTHONPATH=/workspace/polaris/src`. A standalone import reproduced the cause.
  The Docker documentation now sets `PYTHONPATH` and invokes the isolated image
  interpreter directly instead of `uv run`.
- Corrected controller smoke passed hold, +x/+y/+z translation, and +x/+y/+z
  rotation (7/7). Non-hold position errors were 0.0277--0.0885 mm and rotation
  errors were 0.0017--0.0127 degrees. Docker exited 0 with no OOM or fatal Kit
  error. Artifacts:
  `/home/lzha/code/shared_artifacts/polaris-eef-eval-20260630/controller-smoke-20260701T012259Z`.
- End-to-end validation used public LAP-3B revision
  `601db9c1ab4bcaf6dddb160c7b2dec589a67b730`, `lap_original`, flow decoding,
  robot-base-frame deltas, open-loop horizon 8, and one headless FoodBussing
  rollout. The 12.5 GiB checkpoint restored in 6.81 seconds with global action
  and state normalization. Isaac executed the standard 450-step episode in 103
  seconds and the launcher exited 0 after cleaning up server/container state.
- Trace audit: 57 returned `16 x 7` chunks, 450 sequential actions, all finite;
  state shape was 10; each query executed at most 8 actions; every absolute
  position/quaternion/gripper target reproduced the one-anchor equations within
  `4.1e-8`; quaternion norm error was at most `3.9e-8`.
- Artifact audit: CSV reports episode length 450, `success=False`, progress
  `1/6`. The bounded rollout therefore exercised one rubric stage but did not
  solve FoodBussing; this is a policy result, not an integration failure. Video
  is H.264 448x224 at 15 fps for 30 seconds with exactly 450 frames. Sampled
  frames/contact sheet show populated, coherent external and wrist streams;
  ffmpeg found no black or >=2-second frozen interval. No fatal, CUDA, OOM, or
  Python runtime error was found. Artifacts:
  `/home/lzha/code/shared_artifacts/polaris-eef-eval-20260630/e2e/lap3b-foodbussing-eef-20260701T012711Z`.

## 2026-06-30 — Official-contract audit and dual-checkpoint preparation

- Compared the public LAP-3B request/transform stack against official LAP
  commit `3958d1466d5b92445b67de7d4202c19608ad4d56`. Matching non-dropout
  inputs produced identical model-facing image tensors/masks, normalized
  state, and prompt tokens. Corrected the remaining state mismatch by
  binarizing the open-positive gripper observation at the official 0.5
  threshold.
- Added an explicit numeric `action_frame`, separate from the language prompt's
  `frame_description`. This matters for the OXE magic-soup reasoning checkpoint:
  its prompt target says `egocentric frame`, but its flow action target is the
  unchanged base-frame dataset action. The production setting is therefore
  `frame_description=egocentric frame`, `action_frame=robot_base`.
- Added a guarded inverse for genuinely egocentric DROID numeric action chunks.
  A cross-repository randomized check against Ego-LAP's authoritative forward
  transform recovered 200/200 base-frame chunks within `1e-7`; this path is
  not enabled for either current evaluation cohort.
- Validation in `polaris-eval:cuda13`: all 9 client tests passed. The exact
  Pyxis dependency image was published privately as
  `ghcr.io/lihzha/polaris-eval:cuda13-fd00a51`, digest
  `sha256:e32265e25ae2d61b88cf0c530d5561d3fd48e6b2655f48cb583d18fb18851006`,
  for import to the L40S cluster.

## 2026-07-01 — Robust DLS containment for pathological simulation states

- The first 50-reset reasoning-checkpoint BlockStackKitchen evaluation reached
  reset 46, step 328, then Isaac Lab's DLS controller raised
  `torch._C._LinAlgError` while directly inverting its float32 damped normal
  matrix. The job was cancelled after the simulator exited but its policy
  server kept the allocation alive. The failed run remains immutable as job
  `1080837`, with 45 complete CSV/video episodes and one partial trace.
- Isaac Lab 2.3 applies its default `lambda_val=0.01`; an ordinary finite,
  moderate-scale Jacobian is therefore not singular. The failed trace showed
  severe EEF dynamics excursions and a 56-step frozen pose before the error.
  A rank-one float32 Jacobian of magnitude 100 reproduces the numerical mode:
  adding `1e-4 I` to `J J^T` rounds away, leaving the direct inverse singular.
- Added a custom task-space action/controller that preserves the upstream DLS
  implementation exactly on healthy inputs. Only a linear-algebra exception
  uses the same damped normal matrix with a float64 pseudo-inverse; non-finite
  inputs return zero joint delta. Throttled warnings record damping, dtype,
  Jacobian scale, finite-input count, and whether each environment recovered or
  held. A zero delta assumes the current articulation joint state is finite; a
  fallback event still flags that rollout for artifact review.
- Validation in the exact `polaris-eval:cuda13` image: all 9 existing LAP client
  tests passed; all 4 robust-controller tests passed after launching Isaac Sim.
  The latter cover bitwise equality with healthy upstream DLS, production-scale
  float32 damping loss, non-finite input containment, and action-config wiring.
  Ruff formatting/lint and `git diff --check` also passed.
- Recovery policy: do not resume job `1080837` in place. The evaluator would
  resume CSV/video numbering correctly, but the client would append colliding
  trace episode IDs and the launcher would overwrite attempt metadata. Launch a
  fresh 50-reset run directory after staging this patch instead.

### Follow-up: fail-fast numerical rollouts

- A later reasoning MoveLatteCup run (`1081586`) demonstrated that returning a
  zero joint delta is insufficient when the entire Jacobian is already
  non-finite. The controller logged `finite_inputs=0/1`, the EEF pose snapped to
  `[0.088, 0, 0.907826]`, and the simulator stopped advancing after one more
  traced action while consuming GPU/CPU. The job was cancelled with 15 durable
  episodes; it is excluded from metrics.
- Non-finite IK inputs now raise `DifferentialIKNumericalError` before the
  physics step. The evaluator catches that error (and other Torch linear-algebra
  failures), ends the affected rollout, records success/progress as zero plus
  explicit `numerical_failure` columns, writes the partial diagnostic video,
  and resets for the next official initial condition. Finite direct-inverse
  failures still use the float64 damped pseudo-inverse and healthy DLS remains
  bitwise unchanged.
- Validation after this change: all 9 LAP client tests and all 4 Isaac-launched
  robust-controller tests pass in `polaris-eval:cuda13`; Ruff formatting/lint
  and `git diff --check` pass. A clean reasoning MoveLatteCup run is required to
  validate per-episode abort/reset behavior in the full simulator.

### Final full-evaluation validation

- Clean replacement job `1081819`, running commit `88aa4c0`, completed all 50
  reasoning-checkpoint MoveLatteCup resets with Slurm `COMPLETED 0:0`. It
  crossed the old episode-15/step-351 corruption window, completed every
  rollout at 450 steps, and recorded zero numerical failures, DLS fallbacks,
  non-finite policy values, or fatal signatures. Its task result is 0/50
  success and mean normalized progress `0.0133333`.
- The corresponding reasoning BlockStack clean retry (`1081412`) also completed
  50/50 and crossed its former episode-45/step-328 failure point without a
  fallback. Both incomplete first attempts remain preserved and excluded.
- The final twelve accepted task/checkpoint cohorts pass the reusable strict
  artifact auditor: 600/600 rows and videos, 34,200 policy queries, and 270,000
  finite executed actions. Every video decodes to 450 frames/30 seconds at
  15 fps; sampled midpoint/final and critical-window videos pass visual,
  black-frame, and freeze checks. Accepted runs contain zero numerical failure,
  DLS fallback, or unexpected traceback.
- Official LAP-3B achieves 0/300 success and six-task mean progress `0.0665079`;
  reasoning step 10000 achieves 0/300 and `0.0264286`. All evaluation jobs are
  off queue and their isolated runtime caches were removed after checksum-copy
  and audit. Full records live under
  `/home/lzha/code/shared_artifacts/polaris-eef-eval-20260630`.

## 2026-07-05 — Official pi0.5-DROID position canary runtime repair

- Owner: `pi05_position_eval_orchestrator`; branch
  `codex/pi05-position-runtime-fix-20260705`; base PolaRiS commit
  `7d83ea21778d8a0de68ea4dd82a209f6f8d53632`.
- Job `1099014` failed non-scientifically after one model query and one guarded
  action but before a durable execution record. The trace remained mode `0644`
  with only rollout-start/query/action records; no metric, video, sidecar,
  close-ready marker, completion, or eval-success artifact exists.
- The first request and action matched the intended train/eval contract:
  checkpoint-global DROID normalization, external/wrist/blank ordering,
  OpenPI pad-to-224, zero wrist rotation, FLOW Euler-10 seed 0, and
  `q_target=fresh_q+0.2*clip(command)` within the exact hard/soft intersection.
- Root cause: the new position client passed Isaac Lab CUDA termination tensors
  through `numpy.asarray`; the duplicated boolean helper cannot convert CUDA
  tensors. The resulting post-step exception was then masked because the outer
  `finally` invoked pinned Kit teardown, which can hard-exit the child with
  status zero.
- Repair: reuse the promoted tensor-safe outer-step flag validator; print and
  convert any evaluator-body failure to exit 1 before entering Kit cleanup;
  forbid Kit close after an env-close or close-ready publication error; and
  make the shell reject a zero evaluator return without an immutable mode-0444
  close-ready marker.
- Governance: added the shared native lifecycle module to the position
  controller-smoke source set and canary controller-governed set. The old job
  `1099013` attestation cannot authorize the repaired canary; a fresh exact-tip
  L40S controller smoke is mandatory.
- Local validation: 25 focused tests passed. The broader non-Isaac suite passed
  334 tests plus 5 subtests; the single omitted test imports Isaac Lab and is
  covered by the target-runtime controller smoke. Ruff lint/format, ShellCheck,
  `bash -n`, Python bytecode compilation, and `git diff --check` passed.
- Shared Ego-LAP external-checkpoint registry revision 18 records job `1099014`
  as failed/non-scientific with zero completed episodes and no video.
- Next: commit and independently review this repair, run the fresh controller
  smoke on one L40S, then submit a fresh same-protocol model canary and inspect
  the full trace/metric/video transaction.
