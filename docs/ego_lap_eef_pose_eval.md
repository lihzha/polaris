# Ego-LAP end-effector pose evaluation

PolaRiS provides an opt-in `EgoLAPEefPose` client and absolute end-effector
pose controller. The existing `DroidJointPos` client and `joint-position`
control mode remain the defaults.

## Physical and action contract

- The policy state is the 10-D DROID state `[xyz, rotation-6D, gripper-open]`.
  The state layout is checkpoint-specific and versioned in serving metadata:
  `original_lap_public_3b_v1` uses the first two rotation-matrix rows with
  mode `public_lap_train_matched_rows_v1`, matching its training code, while
  manifest-backed checkpoints use the first two columns with mode
  `manifest_train_matched_columns_v1`.
- State, chunk anchoring, and absolute differential IK all use Franka
  `panda_link8` relative to `panda_link0`, with identity frame offsets.
- LAP returns 7-D deltas
  `[dx, dy, dz, droll, dpitch, dyaw, gripper-open]`. PolaRiS anchors every
  cumulative target in a chunk once at query time and sends absolute
  `[xyz, qw, qx, qy, qz, gripper-closed]` commands to Isaac Lab.
- The original Robotiq `ee_frame` remains unchanged for benchmark rubric
  scoring. It is not used as the LAP state or controller frame.
- External and wrist images are RGB uint8. Both are resized to 224 square with
  the TensorFlow-compatible half-pixel bilinear resize-and-pad path; the wrist
  image is rotated 180 degrees before resizing.

Do not feed LAP deltas directly to relative IK and do not use the Robotiq
`base_link` as the state anchor. Either change violates the checkpoint
contract.

## Authoritative server metadata

The websocket server must publish `ego_lap_serving_contract`. The client reads
it through `WebsocketClientPolicy.get_server_metadata()`, validates it before
the first rollout, and atomically writes the exact contract to
`--policy.contract-output`. An existing output may be reused only when its JSON
is identical, so resumed evaluations cannot silently change checkpoints or
transforms. The validator also recomputes the required top-level `sha256`
(defined over the contract excluding that identity field) before accepting it.

Validation covers:

- schema, checkpoint profile/path, and flow versus AR mode;
- 10-D state, 7-D action, 16-step model horizon, 224x224 RGB uint8 images,
  legacy or canonical image routing, and 180-degree wrist rotation;
- the checkpoint-specific R6 layout and mode in both `policy_input` and
  `polaris`; disagreement or an opposite-profile layout fails closed;
- checkpoint normalization digest, Q99 type, and global/category scope;
  category scope must select `single_arm`, while global scope stays global;
- versioned Q99 input/output formula IDs plus passing input, round-trip, and
  extrapolation probes;
- zero wrist-image, zero-image-mask, and state-token dropout at inference;
- language/prompt frame separately from the numeric action frame;
- an unconditional robot-base numeric action frame (an egocentric language
  frame does not change the server's decoded numeric output frame);
- response horizon and semantics, right-multiplied extrinsic-XYZ rotation
  deltas, and the exact action layout.
- `polaris.compatible=true`, an empty incompatibility list, `panda_link8`, and
  consistency of the PolaRiS profile with the Q99 and numeric-frame fields.

The validator recomputes the top-level, live-execution, normalization-stats,
and normalization-formula SHA-256 identities with the same canonical JSON
rules as Ego-LAP. It also binds every Q99 profile to its exact versioned
constants and formula IDs. `normalization.policy_category` is configured
checkpoint metadata; for global stats the effective
`polaris.normalization_category` remains null.

CLI fields are assertions, not alternate sources of truth. Omit an optional
assertion to derive it from metadata. Production jobs should provide exact
checkpoint/profile/digest assertions so a wrong server fails closed.

Production jobs may additionally assert the required Q99 fields with
`--policy.normalization-profile`,
`--policy.normalization-input-formula`, and
`--policy.normalization-output-formula`. A published failing formula probe is
always rejected.

Modern manifest checkpoints require `q99_train_matched_v1`. The public LAP
profile may explicitly reproduce `q99_legacy_upstream_v1` or use the named
train-matched repair; these are different evaluation protocols and must not be
combined under one result identity.

## Flow evaluation

Flow serving must return 16 cumulative `7`-D targets. PolaRiS executes the
first 8 and then replans:

```bash
uv run scripts/eval.py \
  --environment DROID-FoodBussing \
  --control-mode eef-pose \
  --run-folder runs/lap-flow-food-bussing \
  --runtime-contract-output runs/lap-flow-food-bussing/polaris_runtime_contract.json \
  --policy.client EgoLAPEefPose \
  --policy.host 127.0.0.1 \
  --policy.port 8000 \
  --policy.policy-type flow \
  --policy.open-loop-horizon 8 \
  --policy.checkpoint-profile original_lap_public_3b_v1 \
  --policy.checkpoint-path /exact/checkpoint/LAP-3B \
  --policy.normalization-scope global \
  --policy.normalization-stats-sha256 <selected-stats-sha256> \
  --policy.frame-description "robot base frame" \
  --policy.action-frame robot_base \
  --policy.dataset-name droid \
  --policy.state-type eef_pose \
  --policy.eef-frame panda_link8 \
  --policy.contract-output runs/lap-flow-food-bussing/serving_contract.json \
  --policy.trace-dir runs/lap-flow-food-bussing/policy_traces
```

For a manifest-backed modern checkpoint, use the profile and exact checkpoint
path published by that server. Canonical image routing is validated from
metadata; the client still sends the stable source keys `base_0_rgb` and
`left_wrist_0_rgb`.

## AR evaluation

AR serving must return one `7`-D total-delta endpoint. PolaRiS linearly expands
its translation and Euler deltas into 16 cumulative targets, holds the endpoint
gripper target across all 16, executes the first 4, and replans. Use:

```bash
  --policy.policy-type ar \
  --policy.open-loop-horizon 4 \
  --policy.ar-interpolation-steps 16
```

The client rejects a flow response that is not `16x7`, an AR response that is
not `1x7`, and any non-finite value.

## Rollout durability and numerical containment

`scripts/eval.py` atomically publishes one video, finalized per-episode policy
trace, and then one CSV row per completed official initial condition. Resume is
allowed only from a contiguous CSV prefix whose videos decode with the recorded
frame counts and whose `episode_XXXXXX.jsonl` traces contain the same global
episode ID, action count, and terminal `episode_complete` record. A preempted
episode's hidden temporary trace is replaced on retry, so Slurm requeue does not
duplicate or renumber trace records. EEF evaluations use the robust DLS action
term:

- healthy finite inputs call Isaac Lab's implementation unchanged;
- a finite direct-inverse failure retries the same damped system with a
  float64 pseudo-inverse;
- non-finite IK inputs abort only the affected rollout before another physics
  step and are recorded as `numerical_failure=True`, success false, progress
  zero.

Before rollout, the evaluator also requires the live environment to expose
exactly 450 policy steps at 15 Hz. After the first reset it compares the policy
observation to the articulation's direct `panda_link0 -> panda_link8` transform
and verifies that the installed 7-D action term is absolute-pose differential
IK on `panda_link8` with an identity body offset. These protocol, frame,
controller-body, offset, command-mode, and action-dimension facts are written
atomically to `--runtime-contract-output` (defaulting to
`RUN_FOLDER/polaris_runtime_contract.json`). A fully resumed task performs the
same live reset, validation, and write before its completed-path early return.

Run a one-rollout canary and inspect its contract JSON, trace, CSV, and complete
video before scaling out.

## Controller smoke and focused tests

The headless controller smoke checks hold, XYZ translation, and XYZ rotation
against the direct articulation `panda_link8` transform:

```bash
python scripts/smoke_eef_pose_controller.py --headless
```

Pure adapter tests can be run without Isaac Sim:

```bash
PYTHONPATH="$PWD/src:$PWD/third_party/openpi/packages/openpi-client/src" \
  python -m pytest -q \
  tests/test_lap_eef_pose_client.py \
  tests/test_lap_image_resize.py \
  tests/test_eval_mode_contract.py
```

`tests/test_robust_differential_ik.py` requires the pinned Isaac Lab runtime.
SciPy is a direct dependency because pose conversion and composition use its
well-tested quaternion/Euler implementation. No runtime image recipe is part of
this source change.
