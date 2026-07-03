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
  image is rotated 180 degrees only after that resize-and-pad transform.

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
its translation delta and right-multiplied SO(3) rotation into 8 cumulative
targets on the inclusive
`0..1` grid used by LAP, so the first target is the unchanged query-time
anchor and the last is the full endpoint. It holds the endpoint gripper target
across all 8, executes all 8, and replans. Rotation targets retain the
training-contracted right-multiplication by each interpolated delta rather than
the legacy real-robot helper's Euler-add endpoint. Use:

```bash
  --policy.policy-type ar \
  --policy.open-loop-horizon 8 \
  --policy.ar-interpolation-steps 8
```

The client rejects a flow response that is not `16x7`, an AR response that is
not `1x7`, and any non-finite value.

## Production Robotiq adaptation

EEF evaluation installs the named
`implicit_gripper_physx_velocity_limit5_followers5_cuda_actuator_cpu_static_physx_v1`
runtime profile after the first explicit reset and before the first controller
apply. The driven `finger_joint` retains its configured 5 rad/s and 200 Nm
implicit-actuator limits on CUDA. The direct PhysX articulation view exposes
the static 13-DOF maximum-velocity tensor on CPU: the evaluator requires the
seven arm entries and driven finger entry to match their pinned pre-write
values, changes only the five passive Robotiq mimic followers from
`174.53292846679688` to `5.0`, calls
`root_physx_view.set_dof_max_velocities` once with the full tensor, and requires
an exact CPU readback. Later resets verify persistence and never rewrite it.

The static contract also pins the source robot USD identity, every mimic axis,
reference, gearing, natural frequency, damping ratio, exact joint order, and
action/actuator ownership. Dynamic evidence samples all six gripper joints at
every 120-Hz arm-controller entry and after every completed policy step. It
records measured velocity and acceleration maxima plus their causal diagnostic
and terminal vectors. The PhysX velocity setting is not represented as a hard
bound on measured passive-joint velocity; values above 5 rad/s remain valid
measurements and must not be hidden.

Apply-entry samples are counted before finiteness rejection, so a gripper
non-finite preserves exact controller cadence in the numerical-failure
sidecar. A gripper non-finite first discovered only after a completed
`env.step` is a hard job stop: the evaluator does not misrepresent that
executed transition as an unexecuted numerical-failure tail.

## Rollout durability and numerical containment

The v3 evaluator publishes one video, finalized per-episode policy trace,
immutable `ik_safety/episode_XXXXXX.json` transaction, and then one CSV row per
completed official initial condition. The sidecar binds the exact row, video
size/hash/frame identity, terminal-trace size/hash/result, and isolated
controller report. If termination occurs after the sidecar but before the CSV,
resume validates all three artifacts and deterministically replays only that
next contiguous row. Uncommitted video/trace/temporary artifacts are moved to a
content-addressed `recovery_orphans/` archive before retry; evidence is never
silently deleted. Resume reconstructs the aggregate runtime contract from all
immutable sidecars, so a crash after CSV publication cannot erase earlier
controller evidence. EEF evaluations use the robust DLS action term:

- healthy finite inputs call Isaac Lab's implementation unchanged;
- a finite direct-inverse failure retries the same damped system with a
  float64 pseudo-inverse;
- each 120-Hz physics-substep target is clamped to the live Panda velocity
  bound and canonical 7x2 soft joint limits before the PhysX target setter;
- healthy joints on which no guard activates retain Isaac Lab's DLS target
  bit-for-bit, including float32 ULP identity;
- non-finite IK inputs abort only the affected rollout before another physics
  step and are recorded as `numerical_failure=True`, success false, progress
  zero.

Before environment construction, the evaluator configures an internal timeout
of 451 policy steps at 15 Hz, then owns an exact 450-step outer loop. This
one-step margin prevents Isaac Lab's step-450 timeout path from auto-resetting
the terminal observation. After the first reset it compares the policy
observation to the articulation's direct `panda_link0 -> panda_link8` transform
and verifies that the installed 7-D action term is absolute-pose differential
IK on `panda_link8` with an identity body offset. These protocol, frame,
controller-body, offset, command-mode, and action-dimension facts are written
atomically to schema-3 `--runtime-contract-output` (defaulting to
`RUN_FOLDER/polaris_runtime_contract.json`). A fully resumed task performs the
same live reset, validation, and write before its completed-path early return.
The contract binds 120-Hz physics with decimation 8, explicit arm
velocity/effort simulation limits, one named float32 slew tolerance, exact
float32 Panda soft-limit values/digest, per-episode apply/abort/fallback
counters, finite-or-null diagnostics, and aggregate maxima. A nonfailed
450-step rollout must record exactly 3600 controller apply calls; a failed
attempt records its exact policy step and physics substep.

Every successful outer step must return `terminated=false` and
`truncated=false`, advance `episode_length_buf` and `common_step_counter` by
one, `_sim_step_counter` by eight, and both `external_cam.frame` and
`wrist_cam.frame` by one. These exact before/after transitions are written to
the schema-2 `ego_lap_eef_pose_runtime_trace_v2`. With execute horizon 8, a
completed rollout has exactly 959 records: one reset carrying the initial
environment snapshot, 57 queries, 450 actions, 450 execution records, and one
terminal record. Query placement, action/chunk identity, transition schemas,
and the full counter chain are validated during artifact publication and
resume.

Trace-v2 query payloads are recomputed rather than accepted as opaque JSON:
checkpoint hashes, sampler mode, normalization formula/dtype, state-R6,
server/raw/base chunks, and anchored float32 actions must agree. Each action
must equal its selected query chunk, query-static identity cannot drift within
an episode, and an execution-failure reason must equal the terminal result.
Runtime schema 3 likewise recomputes completed episode count, contiguous
indices, counters, arm maxima, and all-six gripper maxima from the immutable
episode entries before publication.

The immutable episode sidecar is schema 3 and carries the same terminal object
as the finalized trace. A normal terminal state has outer/common/camera deltas
of 450 and a simulation-counter delta of 3600. On a numerical failure, the
terminal snapshot is the actual live state after the failed controller apply:
outer/common/camera deltas remain the number of completed steps, while the
simulation-counter tail is strictly one through eight ticks. The sidecar binds
that simulation delta exactly to the safety report's `apply_calls`; a stale
last-success snapshot or a synthetic full-step terminal is rejected.

The older v2 diagnostic runs remain preserved as historical evidence but are
superseded for publication by the `panda_velocity_softlimit_v1` v3 contract.

First run the standalone controller smoke and pin its exact live soft-limit
bytes/digest in the paired Ego-LAP revision. Only that verified immutable
revision may launch a checkpoint canary. Inspect the canary's schema-3 contract,
sidecar, trace, CSV, complete video, 3600-call cadence, all-six gripper maxima,
and wall time before
scaling out.

## Controller smoke and focused tests

The headless controller smoke checks hold, both signs of XYZ translation, and
both signs of XYZ rotation against the direct articulation `panda_link8`
transform. A dedicated final phase resets, sends one deliberately oversized
absolute +X target for exactly one policy step/eight physics substeps, captures
the report, and resets immediately. It passes only when the state/report stay
finite, the slew guard activates, all applied maxima remain within
`velocity/120 + 1e-6` rad, and abort/post-clamp violation counts remain zero.
The post-step finiteness gate reads both the EEF observation and live
`panda_joint1..7` position/velocity tensors, recording strict value/mask
and maximum-absolute-value evidence for the joint state before the immediate
reset. Final joint positions must also remain inside the exact captured soft
limits with only the named `1e-5` float allowance.

Before any DLS pose-error computation, both the measured and commanded EEF
quaternions must be finite and have norm within exactly `1e-3` of one. Current
and desired violations have distinct invariant-abort diagnostics. Durable
episode evidence uses closed key schemas and exact diagnostic-to-counter
mapping; unknown aborts, dropped diagnostics, or a missing/inconsistent
max-raw-delta record invalidate the episode. State/action arrays are checked
again after float32 conversion, and JSONL traces forbid non-finite constants.

```bash
python scripts/smoke_eef_pose_controller.py \
  --headless \
  --output-json /tmp/polaris-eef-controller-smoke.json
```

`--output-json` is required. After environment teardown, the smoke atomically
leaves one immutable strict failure record or clean
`simulation_app_close_pending` raw record after the environment closes. The
host wrapper may create a separate finalized attestation only after the Isaac
process returns zero and it independently verifies the raw record; it never
rewrites the raw JSON. Both the raw JSON and its path/size/SHA-bound ready
marker are non-overwriting, mode `0444`, and file/parent-directory fsynced.
Failures retain the raw safety capture, completed partial cases, current
stage/case, and flushed exception traceback; a failure path skips the explicit
SimulationApp hard-exit call so it cannot mask the intended nonzero status.
The host uses `scripts/finalize_eef_pose_smoke.py` in `finalize` and then
`verify` mode; its attestation is a separate immutable object bound to the raw
bytes, ready marker, Slurm job, `srun` status, commit, source, image, and saved
job script.

Pure adapter tests can be run without Isaac Sim:

```bash
PYTHONPATH="$PWD/src:$PWD/third_party/openpi/packages/openpi-client/src" \
  python -m pytest -q \
  tests/test_lap_eef_pose_client.py \
  tests/test_lap_image_resize.py \
  tests/test_eval_mode_contract.py
```

The new terminal, artifact, client, and gripper contract tests use fake
environments/tensors. They prove schema, mutation rejection, one-call control
flow, and resume binding, but they do not prove the real camera counters or the
CPU/CUDA PhysX device split. Do not promote this profile from host tests. The
first one-rollout official LAP `scripts/eval.py` canary is the authoritative
integrated real-Isaac gate and must validate all 450 transitions, 3600 applies,
all-six gripper evidence, trace/sidecar/runtime artifacts, and the complete
video before any scale-out.

`tests/test_robust_differential_ik.py` requires the pinned Isaac Lab runtime.
SciPy is a direct dependency because pose conversion and composition use its
well-tested quaternion/Euler implementation. No runtime image recipe is part of
this source change.
