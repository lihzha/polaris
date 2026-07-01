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
