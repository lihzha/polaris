# Evaluating Ego-LAP with end-effector pose control

PolaRiS can evaluate an Ego-LAP websocket policy with absolute differential-IK
control while retaining joint-position control as the default for existing
policies.

## Local Ada Docker runtime

The checked-in Docker recipes isolate the CUDA 13 compiler used for the splat
kernels from the host CUDA toolkit and compile the kernels for Ada GPUs
(`TORCH_CUDA_ARCH_LIST=8.9`). From the repository root:

```bash
docker build --progress=plain \
  -f docker/Dockerfile.base-ada \
  -t polaris-isaaclab-base:cu130-isaaclab230-r1 .
docker build --progress=plain \
  -f docker/Dockerfile.polaris-ada \
  -t polaris-eval:cuda13 .
```

The final image has an empty entrypoint and `python` from its isolated virtual
environment on `PATH`, so it is compatible with the ego-lap launcher described
below. Restrict runtime access to the RTX 6000 Ada with `--gpus device=0` and
`CUDA_VISIBLE_DEVICES=0` on this two-GPU workstation.

The image contains dependencies, not the PolaRiS checkout or Hub assets. A
bind-mounted source checkout must therefore set `PYTHONPATH` to its `src`
directory. Do not use `uv run` inside this image: invoke its isolated `python`
directly so uv does not attempt to create or synchronize a second environment.

## Interface contract

- The observed and controlled Cartesian frame is Franka `panda_link8`,
  expressed relative to the robot root (`panda_link0`). This matches DROID's
  recorded `cartesian_position` contract. The attached Robotiq `base_link`
  differs by a fixed 18.174 mm / approximately 180-degree transform and must
  not be used as the LAP state or action anchor. Isaac Lab receives absolute
  actions as `[x, y, z, qw, qx, qy, qz, gripper_closed]`.
- The benchmark's pre-existing Robotiq `ee_frame` sensor remains unchanged for
  rubric scoring. LAP observations use a separate `lap_ee_frame`, so correcting
  the policy contract does not shift PolaRiS success thresholds.
- The policy state is `[x, y, z, R[:, 0], R[:, 1], gripper_open]` (10 values).
  Isaac quaternions are `wxyz`; SciPy conversion is performed in `xyzw`.
- The external image is sent as `base_0_rgb`. The wrist image is rotated 180
  degrees and sent as `left_wrist_0_rgb`, matching DROID preprocessing.
- Ego-LAP responses must be finite, non-empty `T x 7` arrays containing
  `[dx, dy, dz, droll, dpitch, dyaw, gripper_open]`. Invalid responses stop the
  rollout instead of silently substituting an action.
- Every pose delta in a returned chunk is anchored once to the query-time pose.
  `--policy.action-frame robot_base` anchors the numeric flow delta directly.
  `--policy.action-frame egocentric` first applies the exact inverse of Ego-LAP's single-arm
  DROID semantic EEF transform (including its wrist-view axis convention and
  rotation conjugation), then anchors the recovered base-frame delta. Targets
  are not accumulated against the robot pose at each execution step.

## Start the Ego-LAP server

From an ego-lap checkout with its environment installed, serve a flow
checkpoint on a free port. Replace the checkpoint path as needed:

```bash
cd /path/to/ego-lap
uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config lap_original \
  --policy.dir /path/to/LAP-3B-checkpoint \
  --policy.type flow
```

Wait until the server reports that it is listening before starting PolaRiS.
The server and client may use separate Python environments.

## Run a bounded evaluation

Download the PolaRiS Hub assets as described in the main README, then run:

```bash
uv run scripts/eval.py \
  --environment DROID-FoodBussing \
  --control-mode eef-pose \
  --policy.client EgoLAPEefPose \
  --policy.host 127.0.0.1 \
  --policy.port 8000 \
  --policy.open-loop-horizon 8 \
  --policy.frame-description "robot base frame" \
  --policy.trace-path runs/ego_lap_food_bussing/action_trace.jsonl \
  --run-folder runs/ego_lap_food_bussing \
  --rollouts 1
```

`--policy.open-loop-horizon` controls how many actions are executed before a
new query. It must not exceed the returned chunk length. Pass
`--policy.open-loop-horizon None` to execute the complete chunk. The optional
JSONL trace records each query anchor, raw delta chunk, anchored absolute
chunk, decoded base-frame delta chunk, numeric action frame, and executed
action. Omit `--policy.trace-path` to disable it. Prompt wording and numeric
flow-action coordinates are intentionally configured separately: current
reasoning checkpoints use `--policy.frame-description "egocentric frame"`, but
their flow targets remain the unmodified base-frame dataset actions and thus
use `--policy.action-frame robot_base`.

Use `--instruction "..."` to override the instruction stored with an
environment's initial conditions. Evaluation videos contain one visualization
for every executed step, including open-loop steps between policy queries. The
Ego-LAP client renders the splat composite every step by default so those frames
have a consistent appearance; `--policy.no-render-every-step` is a
performance-oriented opt-out.

## Controller-only smoke test

The scripted smoke drives translations and rotations about all three axes and
checks final pose error. It launches Isaac Sim and therefore requires a working
GPU installation and downloaded PolaRiS assets:

```bash
mkdir -p runs/eef_pose_controller_smoke
docker run --rm --gpus device=0 --network host --ipc host --shm-size=16g \
  -e PYTHONPATH=/workspace/polaris/src \
  -e POLARIS_DATA_PATH=/workspace/polaris/PolaRiS-Hub \
  -v "$PWD:/workspace/polaris:ro" \
  -v "$PWD/runs/eef_pose_controller_smoke:/outputs:rw" \
  -w /workspace/polaris polaris-eval:cuda13 \
  python scripts/smoke_eef_pose_controller.py \
    --headless --device cuda:0 --output-json /outputs/results.json
```

This smoke is intentionally separate from the CPU unit tests. Run it before a
production evaluation after changing the robot USD, controlled link, actuator
gains, Isaac Lab version, or differential-IK configuration.
