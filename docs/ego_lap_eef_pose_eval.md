# Evaluating Ego-LAP with end-effector pose control

PolaRiS can evaluate an Ego-LAP websocket policy with absolute differential-IK
control while retaining joint-position control as the default for existing
policies.

## Interface contract

- The controlled frame is the Robotiq `base_link`, expressed relative to the
  robot root (`panda_link0`). Isaac Lab receives absolute actions as
  `[x, y, z, qw, qx, qy, qz, gripper_closed]`.
- The policy state is `[x, y, z, R[:, 0], R[:, 1], gripper_open]` (10 values).
  Isaac quaternions are `wxyz`; SciPy conversion is performed in `xyzw`.
- The external image is sent as `base_0_rgb`. The wrist image is rotated 180
  degrees and sent as `left_wrist_0_rgb`, matching DROID preprocessing.
- Ego-LAP responses must be finite, non-empty `T x 7` arrays containing
  `[dx, dy, dz, droll, dpitch, dyaw, gripper_open]`. Invalid responses stop the
  rollout instead of silently substituting an action.
- Every pose delta in a returned chunk is anchored once to the query-time pose:
  `p_target = p_anchor + dp` and `R_target = R_anchor * R_delta`. Targets are
  not accumulated against the robot pose at each execution step.

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
chunk, and executed action. Omit `--policy.trace-path` to disable it.

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
uv run scripts/smoke_eef_pose_controller.py --headless
```

This smoke is intentionally separate from the CPU unit tests. Run it before a
production evaluation after changing the robot USD, controlled link, actuator
gains, Isaac Lab version, or differential-IK configuration.
