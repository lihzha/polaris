"""Official OpenPI :math:`pi0.5` DROID native joint-velocity client."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy

from polaris.pi05_droid_jointvelocity_contract import (
    PANDA_ARM_JOINT_NAMES,
    PI05_DROID_JOINTVELOCITY_PROFILE,
    validate_persisted_serving_contract,
    validate_pi05_droid_server_metadata,
    verify_openpi_git_checkout,
)
from polaris.pi05_droid_native_eval_contract import (
    fsync_directory,
    PI05_DROID_NATIVE_DECIMATION,
    PI05_DROID_NATIVE_ENVIRONMENT_RUNTIME_PROFILE,
    PI05_DROID_NATIVE_EPISODE_STEPS,
    PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS,
    PI05_DROID_NATIVE_SENSOR_LIVENESS_PROFILE,
    PI05_DROID_NATIVE_SENSOR_NAMES,
    PI05_DROID_NATIVE_TERMINAL_FAILURE_PROFILE,
    PI05_DROID_NATIVE_TRACE_SCHEMA_VERSION,
    validate_environment_runtime_contract,
    validate_outer_step_flags,
    validate_immutable_file,
    validate_terminal_numerical_failure_evidence,
    validate_terminal_rollout_evidence,
)
from polaris.policy.abstract_client import InferenceClient, PolicyArgs
from polaris.policy.droid_jointpos_client import (
    _latest_trace_reset_index,
    validate_joint_action_chunk,
)


PI05_DROID_JOINTVELOCITY_CONTRACT_MARKER = "POLARIS_PI05_DROID_JOINTVELOCITY_CONTRACT="


class JointVelocityObservationNumericalError(FloatingPointError):
    """Raised when simulator joint-velocity proprioception is invalid."""


def _image_contract(image: np.ndarray) -> dict[str, Any]:
    image = np.ascontiguousarray(np.asarray(image))
    if image.shape != (224, 224, 3) or image.dtype != np.uint8:
        raise ValueError(
            "pi0.5-DROID images must be 224x224 uint8 RGB; "
            f"got shape={image.shape}, dtype={image.dtype}"
        )
    return {
        "shape": [224, 224, 3],
        "dtype": "uint8",
        "sha256": hashlib.sha256(image.tobytes()).hexdigest(),
    }


def process_native_jointvelocity_action(
    raw_action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply upstream DROID gripper binarization, then clip all eight values."""

    raw_action = np.asarray(raw_action)
    if raw_action.shape != (8,):
        raise ValueError(f"Expected one 8-D DROID action, got {raw_action.shape}")
    if not np.isfinite(raw_action).all():
        raise ValueError("DROID joint-velocity action contains non-finite values")
    # This intentionally matches examples/droid/main.py.  np.ones/np.zeros
    # preserve the upstream float64 promotion before the all-dimension clip.
    if raw_action[-1].item() > 0.5:
        binary_action = np.concatenate([raw_action[:-1], np.ones((1,))])
    else:
        binary_action = np.concatenate([raw_action[:-1], np.zeros((1,))])
    clipped_action = np.clip(binary_action, -1.0, 1.0)
    return binary_action, clipped_action


def _tensor_numpy(value: Any, *, field: str) -> np.ndarray:
    try:
        value = value.detach().cpu().numpy()
    except AttributeError:
        value = np.asarray(value)
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{field} must be numeric, got {array.dtype}")
    return array


def _exact_nonnegative_int(value: Any, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be one nonnegative exact integer")
    return value


def _single_integer_tensor(value: Any, *, field: str) -> int:
    array = _tensor_numpy(value, field=field)
    if array.shape != (1,) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(
            f"{field} must be one integer tensor, got {array.dtype} {array.shape}"
        )
    return _exact_nonnegative_int(int(array[0]), field=field)


def _capture_environment_state(
    env: Any, environment_runtime_contract: dict[str, Any]
) -> dict[str, Any]:
    runtime = validate_environment_runtime_contract(environment_runtime_contract)
    root_env = getattr(env, "unwrapped", env)
    live_max_episode_length = getattr(root_env, "max_episode_length", None)
    if (
        type(live_max_episode_length) is not int
        or live_max_episode_length != runtime["live_max_episode_length"]
    ):
        raise ValueError("Live environment max_episode_length drifted during rollout")
    episode_length = _single_integer_tensor(
        getattr(root_env, "episode_length_buf", None), field="episode length buffer"
    )
    sim_step_counter = _exact_nonnegative_int(
        getattr(root_env, "_sim_step_counter", None), field="sim step counter"
    )
    common_step_counter = _exact_nonnegative_int(
        getattr(root_env, "common_step_counter", None), field="common step counter"
    )
    scene = getattr(root_env, "scene", None)
    sensors = getattr(scene, "sensors", None)
    if not isinstance(sensors, dict):
        raise ValueError("Live environment has no closed camera-sensor mapping")
    sensor_frame_counters = {}
    for sensor_name in PI05_DROID_NATIVE_SENSOR_NAMES:
        if sensor_name not in sensors:
            raise ValueError(f"Missing required live camera sensor: {sensor_name}")
        sensor_frame_counters[sensor_name] = _single_integer_tensor(
            getattr(sensors[sensor_name], "frame", None),
            field=f"{sensor_name} camera frame counter",
        )
    return {
        "live_max_episode_length": live_max_episode_length,
        "episode_length": episode_length,
        "sim_step_counter": sim_step_counter,
        "common_step_counter": common_step_counter,
        "sensor_frame_counters": sensor_frame_counters,
    }


@InferenceClient.register(client_name="DroidJointVelocity")
class DroidJointVelocityClient(InferenceClient):
    """Serve the immutable official ``pi05_droid`` velocity contract."""

    def __init__(self, args: PolicyArgs) -> None:
        self.args = args
        self._validate_args()
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host, port=args.port
        )
        server_metadata = self.client.get_server_metadata()
        self.serving_contract = validate_pi05_droid_server_metadata(server_metadata)
        if (
            not isinstance(args.serving_contract_path, str)
            or not args.serving_contract_path
        ):
            raise ValueError("DroidJointVelocity requires serving_contract_path")
        self.serving_contract_artifact = validate_persisted_serving_contract(
            Path(args.serving_contract_path), server_metadata
        )
        self.client_runtime_attestation = self._validate_client_runtime_origin()
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk: np.ndarray | None = None
        self.open_loop_horizon = args.open_loop_horizon
        self.query_index = 0
        self.active_query_index: int | None = None
        self.trace_path = Path(args.trace_path) if args.trace_path else None
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self.reset_index = _latest_trace_reset_index(self.trace_path)
        else:
            self.reset_index = -1
        self._pending_execution: dict[str, Any] | None = None
        self._environment_runtime_contract: dict[str, Any] | None = None
        self._rollout_environment_before: dict[str, Any] | None = None
        self._last_environment_after: dict[str, Any] | None = None
        self._execution_step_index = 0
        self._terminated_false_count = 0
        self._truncated_false_count = 0
        self._terminal_rollout: dict[str, Any] | None = None
        self._finalized_trace_artifact: dict[str, Any] | None = None

        marker = {
            "client": "DroidJointVelocity",
            "profile": PI05_DROID_JOINTVELOCITY_PROFILE,
            "serving_contract_sha256": self.serving_contract["contract_sha256"],
            "serving_contract_artifact_sha256": self.serving_contract_artifact[
                "sha256"
            ],
            "serving_contract_artifact_size": self.serving_contract_artifact["size"],
            "serving_contract_path": self.serving_contract_artifact["path"],
            "client_runtime_attestation": self.client_runtime_attestation,
            "open_loop_horizon": self.open_loop_horizon,
            "expected_action_horizon": args.expected_action_horizon,
            "expected_action_dim": args.expected_action_dim,
            "image_resolution": [224, 224],
            "wrist_rotation_degrees": 0,
            "initial_reset_index": self.reset_index,
        }
        print(
            PI05_DROID_JOINTVELOCITY_CONTRACT_MARKER
            + json.dumps(marker, sort_keys=True),
            flush=True,
        )

    def _validate_args(self) -> None:
        expected = {
            "policy_profile": PI05_DROID_JOINTVELOCITY_PROFILE,
            "open_loop_horizon": 8,
            "expected_action_horizon": 15,
            "expected_action_dim": 8,
            "state_type": "joint_position",
            "frame_description": "robot base frame",
            "action_frame": "robot_base",
            "dataset_name": "droid",
            "rotate_wrist_180": False,
            "render_every_step": True,
        }
        for field, expected_value in expected.items():
            actual = getattr(self.args, field, None)
            if actual != expected_value:
                raise ValueError(
                    f"DroidJointVelocity requires {field}={expected_value!r}; "
                    f"got {actual!r}"
                )

    def _validate_client_runtime_origin(self) -> dict[str, Any]:
        if not isinstance(self.args.openpi_dir, str) or not self.args.openpi_dir:
            raise ValueError("DroidJointVelocity requires openpi_dir")
        checkout = verify_openpi_git_checkout(Path(self.args.openpi_dir))
        root = Path(checkout["root"])
        expected = {
            "openpi_client.image_tools": (
                image_tools,
                "packages/openpi-client/src/openpi_client/image_tools.py",
                "d48b4bd7f44e79fe6db8a8e07c9161144fa250be686e1245014a8b47e6171977",
            ),
            "openpi_client.websocket_client_policy": (
                websocket_client_policy,
                "packages/openpi-client/src/openpi_client/websocket_client_policy.py",
                "36557cb0b91ccf31cd4fb4b508306850d76ed0feb4028dac5182d0f5a5d88005",
            ),
        }
        records = []
        for module_name, (module, relative_path, expected_digest) in expected.items():
            module_file = getattr(module, "__file__", None)
            if not isinstance(module_file, str):
                raise ValueError(f"Imported {module_name} has no source origin")
            raw_module_path = Path(module_file)
            raw_expected_path = root / relative_path
            if raw_module_path.is_symlink() or raw_expected_path.is_symlink():
                raise ValueError(f"Imported {module_name} source must not be a symlink")
            module_path = raw_module_path.resolve()
            expected_path = raw_expected_path.resolve()
            if module_path != expected_path or not module_path.is_file():
                raise ValueError(
                    f"Imported {module_name} escaped DroidJointVelocity openpi_dir"
                )
            digest = hashlib.sha256(module_path.read_bytes()).hexdigest()
            if digest != expected_digest:
                raise ValueError(f"Imported {module_name} source digest mismatch")
            records.append(
                {
                    "module": module_name,
                    "relative_path": relative_path,
                    "sha256": digest,
                }
            )
        records.sort(key=lambda record: record["module"])
        identity = {
            "schema_version": 1,
            "openpi_dir": str(root),
            "git_head": checkout["git_head"],
            "git_tracked_and_untracked_clean": checkout[
                "git_tracked_and_untracked_clean"
            ],
            "modules": records,
        }
        identity["sha256"] = hashlib.sha256(
            json.dumps(
                identity,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()
        return identity

    @property
    def rerender(self) -> bool:
        return (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        )

    def visualize(self, request: dict) -> np.ndarray:
        current = self._extract_observation(request)
        external, wrist = self._resize_images(current)
        return np.concatenate([external, wrist], axis=1)

    def reset(self) -> None:
        if self._environment_runtime_contract is None:
            raise RuntimeError(
                "DroidJointVelocity evaluation runtime must be bound before reset"
            )
        if self._pending_execution is not None:
            raise RuntimeError(
                "Cannot reset DroidJointVelocity before recording the prior execution"
            )
        if self._rollout_environment_before is not None:
            raise RuntimeError(
                "DroidJointVelocity native canary forbids a second rollout"
            )
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.reset_index += 1
        self.query_index = 0
        self.active_query_index = None

    def bind_evaluation_runtime(self, value: dict[str, Any]) -> None:
        """Bind the live timeout and sensor-liveness contract before reset."""

        if self._environment_runtime_contract is not None:
            raise RuntimeError("DroidJointVelocity evaluation runtime is already bound")
        self._environment_runtime_contract = validate_environment_runtime_contract(
            value
        )

    def begin_rollout(self, env: Any) -> dict[str, Any]:
        """Capture the exact counters after explicit reset and before action zero."""

        if self._environment_runtime_contract is None:
            raise RuntimeError("DroidJointVelocity evaluation runtime is not bound")
        if self.reset_index != 0:
            raise ValueError("Official native canary requires exactly reset index zero")
        if self._rollout_environment_before is not None:
            raise RuntimeError("DroidJointVelocity rollout already began")
        before = _capture_environment_state(env, self._environment_runtime_contract)
        if before["episode_length"] != 0:
            raise ValueError(
                "Explicit native rollout reset did not zero episode length"
            )
        self._rollout_environment_before = before
        self._last_environment_after = None
        self._execution_step_index = 0
        self._terminated_false_count = 0
        self._truncated_false_count = 0
        self._terminal_rollout = None
        self._finalized_trace_artifact = None
        if self.trace_path is not None:
            self._append_trace(
                {
                    "schema_version": PI05_DROID_NATIVE_TRACE_SCHEMA_VERSION,
                    "record_type": "openpi_joint_velocity_rollout_start",
                    "profile": PI05_DROID_JOINTVELOCITY_PROFILE,
                    **self._trace_contract_identity(),
                    "reset_index": self.reset_index,
                    "environment_before": before,
                }
            )
        return dict(before)

    def infer(
        self, obs: dict, instruction: str, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if self._pending_execution is not None:
            raise RuntimeError(
                "Previous DroidJointVelocity action has no measured execution record"
            )
        current = self._extract_observation(obs)
        visualization = None
        if self.rerender:
            self.actions_from_chunk_completed = 0
            external, wrist = self._resize_images(current)
            request_data = {
                "observation/exterior_image_1_left": external,
                "observation/wrist_image_left": wrist,
                "observation/joint_position": current["joint_position"],
                "observation/gripper_position": current["gripper_position"],
                "prompt": instruction,
            }
            response = self.client.infer(request_data)
            self.pred_action_chunk = validate_joint_action_chunk(
                response,
                open_loop_horizon=self.open_loop_horizon,
                expected_action_horizon=15,
                expected_action_dim=8,
            )
            if self.pred_action_chunk.dtype != np.float64:
                raise ValueError(
                    "Official pi05_droid response must be float64 after checkpoint "
                    f"unnormalization; got {self.pred_action_chunk.dtype}"
                )
            self.active_query_index = self.query_index
            self._trace_query(request_data, self.pred_action_chunk)
            self.query_index += 1
            visualization = np.concatenate([external, wrist], axis=1)
        elif return_viz:
            external, wrist = self._resize_images(current)
            visualization = np.concatenate([external, wrist], axis=1)

        if self.pred_action_chunk is None or self.active_query_index is None:
            raise ValueError("No pi0.5-DROID action chunk predicted")

        action_index = self.actions_from_chunk_completed
        raw_action = self.pred_action_chunk[action_index].copy()
        binary_action, clipped_action = process_native_jointvelocity_action(raw_action)
        self.actions_from_chunk_completed += 1
        self._pending_execution = {
            "query_index": self.active_query_index,
            "chunk_action_index": action_index,
            "raw_action": raw_action,
            "binary_action": binary_action,
            "clipped_action": clipped_action,
            "pre_joint_position": current["joint_position"],
            "pre_joint_velocity": current["joint_velocity"],
            "pre_normalized_gripper_position": current["gripper_position"],
        }
        self._trace_emitted_action()
        return clipped_action, visualization

    def record_execution(
        self, obs: dict, env: Any, *, terminated: Any, truncated: Any
    ) -> None:
        """Validate live velocity targets and trace measured post-step q/dq."""

        if self._pending_execution is None:
            raise RuntimeError("No pending DroidJointVelocity action to record")
        if (
            self._environment_runtime_contract is None
            or self._rollout_environment_before is None
        ):
            raise RuntimeError("DroidJointVelocity rollout runtime was not started")
        if self._execution_step_index >= PI05_DROID_NATIVE_EPISODE_STEPS:
            raise RuntimeError("DroidJointVelocity recorded more than 450 executions")
        step_boundary = validate_outer_step_flags(
            terminated,
            truncated,
            outer_step_index=self._execution_step_index,
        )
        environment_after = _capture_environment_state(
            env, self._environment_runtime_contract
        )
        expected_episode_length = self._execution_step_index + 1
        expected_sim_step_counter = (
            self._rollout_environment_before["sim_step_counter"]
            + expected_episode_length * PI05_DROID_NATIVE_DECIMATION
        )
        expected_common_step_counter = (
            self._rollout_environment_before["common_step_counter"]
            + expected_episode_length
        )
        if environment_after["episode_length"] != expected_episode_length:
            raise ValueError(
                "Live episode length proves an internal auto-reset before the outer "
                f"450-step contract: expected={expected_episode_length}, "
                f"actual={environment_after['episode_length']}"
            )
        if (
            environment_after["sim_step_counter"] != expected_sim_step_counter
            or environment_after["common_step_counter"] != expected_common_step_counter
        ):
            raise ValueError("Live simulator counters do not match one policy action")
        for sensor_name in PI05_DROID_NATIVE_SENSOR_NAMES:
            expected_frame = (
                self._rollout_environment_before["sensor_frame_counters"][sensor_name]
                + expected_episode_length
            )
            if (
                environment_after["sensor_frame_counters"][sensor_name]
                != expected_frame
            ):
                raise ValueError(
                    "Live camera frame counter did not advance exactly once for "
                    f"{sensor_name} at outer step {self._execution_step_index}"
                )
        current = self._extract_observation(obs)
        root_env = getattr(env, "unwrapped", env)
        arm_term = root_env.action_manager._terms["arm"]
        finger_term = root_env.action_manager._terms["finger_joint"]
        robot = root_env.scene["robot"]
        joint_ids, joint_names = robot.find_joints(
            list(PANDA_ARM_JOINT_NAMES), preserve_order=True
        )
        if tuple(joint_names) != PANDA_ARM_JOINT_NAMES:
            raise ValueError(f"Live Panda joint order mismatch: {joint_names}")
        finger_ids, finger_names = robot.find_joints(
            ["finger_joint"], preserve_order=True
        )
        if finger_names != ["finger_joint"]:
            raise ValueError(f"Live finger joint mismatch: {finger_names}")

        processed = _tensor_numpy(
            arm_term.processed_actions, field="arm processed actions"
        )
        targets = _tensor_numpy(
            robot.data.joint_vel_target[:, joint_ids], field="joint velocity targets"
        )
        if processed.dtype != np.float32 or targets.dtype != np.float32:
            raise ValueError(
                "Live processed and target velocities must both be float32; "
                f"got {processed.dtype} and {targets.dtype}"
            )
        if processed.shape != (1, 7) or targets.shape != (1, 7):
            raise ValueError(
                "Live velocity target shape mismatch: "
                f"processed={processed.shape}, target={targets.shape}"
            )
        expected = np.asarray(
            self._pending_execution["clipped_action"][:7], dtype=targets.dtype
        )[None, :]
        if not np.array_equal(processed.astype(targets.dtype, copy=False), expected):
            raise ValueError("Action manager changed emitted joint velocity")
        if not np.array_equal(targets, expected):
            raise ValueError("Articulation joint-velocity target differs from emission")
        processed_finger = _tensor_numpy(
            finger_term.processed_actions, field="processed finger target"
        )
        finger_target = _tensor_numpy(
            robot.data.joint_pos_target[:, finger_ids], field="finger position target"
        )
        if processed_finger.dtype != np.float32 or finger_target.dtype != np.float32:
            raise ValueError("Live finger targets must be float32")
        emitted_binary = self._pending_execution["binary_action"][7]
        if emitted_binary not in (0.0, 1.0):
            raise ValueError("Pending gripper emission is not binary")
        expected_finger_value = (
            np.float32(np.pi / 4.0) if emitted_binary == 1.0 else np.float32(0.0)
        )
        expected_finger = np.asarray([[expected_finger_value]], dtype=np.float32)
        if processed_finger.shape != (1, 1) or not np.array_equal(
            processed_finger, expected_finger
        ):
            raise ValueError("Action manager changed emitted binary gripper target")
        if finger_target.shape != (1, 1) or not np.array_equal(
            finger_target, expected_finger
        ):
            raise ValueError("Articulation finger target differs from binary emission")

        if self.trace_path is not None:
            pending = self._pending_execution
            self._append_trace(
                {
                    "schema_version": PI05_DROID_NATIVE_TRACE_SCHEMA_VERSION,
                    "record_type": "openpi_joint_velocity_execution",
                    "profile": PI05_DROID_JOINTVELOCITY_PROFILE,
                    **self._trace_contract_identity(),
                    "reset_index": self.reset_index,
                    "query_index": pending["query_index"],
                    "chunk_action_index": pending["chunk_action_index"],
                    **step_boundary,
                    "environment_after": environment_after,
                    "processed_joint_velocity": processed[0].tolist(),
                    "articulation_joint_velocity_target": targets[0].tolist(),
                    "processed_finger_position_target": processed_finger[0].tolist(),
                    "articulation_finger_position_target": finger_target[0].tolist(),
                    "measured_joint_position_after": current["joint_position"].tolist(),
                    "measured_joint_velocity_after": current["joint_velocity"].tolist(),
                    "measured_normalized_gripper_position_after": current[
                        "gripper_position"
                    ].tolist(),
                }
            )
        self._last_environment_after = environment_after
        self._execution_step_index += 1
        self._terminated_false_count += 1
        self._truncated_false_count += 1
        self._pending_execution = None

    def finish_rollout(self, env: Any, rubric: Any) -> dict[str, Any]:
        """Freeze the true post-action state and rubric before environment close."""

        if (
            self._environment_runtime_contract is None
            or self._rollout_environment_before is None
            or self._last_environment_after is None
        ):
            raise RuntimeError("DroidJointVelocity rollout is incomplete")
        if self._pending_execution is not None:
            raise RuntimeError("DroidJointVelocity has an unmeasured terminal action")
        if self._execution_step_index != PI05_DROID_NATIVE_EPISODE_STEPS:
            raise ValueError("DroidJointVelocity did not execute exactly 450 actions")
        current_environment = _capture_environment_state(
            env, self._environment_runtime_contract
        )
        if current_environment != self._last_environment_after:
            raise ValueError(
                "Terminal environment changed after final execution capture"
            )
        if not isinstance(rubric, dict):
            raise ValueError("Terminal rubric is not an object")
        success = rubric.get("success")
        progress = rubric.get("progress")
        if type(success) is not bool:
            raise ValueError("Terminal rubric success is not an exact boolean")
        if hasattr(progress, "item"):
            progress = progress.item()
        if (
            type(progress) not in (int, float)
            or isinstance(progress, bool)
            or not np.isfinite(progress)
        ):
            raise ValueError("Terminal rubric progress is not finite")
        terminal = {
            "schema_version": 1,
            "profile": PI05_DROID_NATIVE_ENVIRONMENT_RUNTIME_PROFILE,
            "environment_runtime_sha256": self._environment_runtime_contract["sha256"],
            "outer_steps_completed": self._execution_step_index,
            "last_outer_step_index": self._execution_step_index - 1,
            "terminated_false_count": self._terminated_false_count,
            "truncated_false_count": self._truncated_false_count,
            "environment_before": self._rollout_environment_before,
            "environment_after": current_environment,
            "rubric": {"success": success, "progress": float(progress)},
        }
        terminal = validate_terminal_rollout_evidence(
            terminal, self._environment_runtime_contract
        )
        if self.trace_path is not None:
            self._append_trace(
                {
                    "schema_version": PI05_DROID_NATIVE_TRACE_SCHEMA_VERSION,
                    "record_type": "openpi_joint_velocity_rollout_end",
                    "profile": PI05_DROID_JOINTVELOCITY_PROFILE,
                    **self._trace_contract_identity(),
                    "reset_index": self.reset_index,
                    "terminal_rollout": terminal,
                }
            )
            self._finalized_trace_artifact = self._seal_trace()
        self._terminal_rollout = terminal
        return terminal

    def record_execution_failure(
        self, error: BaseException, env: Any, dynamic_report: dict[str, Any]
    ) -> dict[str, Any]:
        """Finalize the only allowed partial native rollout failure."""

        from polaris.native_gripper_runtime import (  # noqa: PLC0415
            NativeAllJointVelocityLimitError,
        )

        if type(error) is not NativeAllJointVelocityLimitError:
            raise TypeError(
                "Native failure finalization requires the exact typed error"
            )
        if self._pending_execution is None:
            raise RuntimeError("Native failure has no pending emitted action")
        if (
            self._environment_runtime_contract is None
            or self._rollout_environment_before is None
        ):
            raise RuntimeError("Native failure has no bound rollout runtime")
        if error.incident_artifact is None:
            raise RuntimeError("Native velocity failure was not durably published")
        if dynamic_report.get("terminal_velocity_failure") != error.evidence:
            raise RuntimeError("Native velocity failure differs from dynamic evidence")
        pending = self._pending_execution
        environment_after_failure = _capture_environment_state(
            env, self._environment_runtime_contract
        )
        last_completed = (
            self._last_environment_after
            if self._last_environment_after is not None
            else self._rollout_environment_before
        )
        reason = f"{type(error).__name__}: {error}"
        episode_result = {
            "episode": 0,
            "episode_length": self._execution_step_index + 1,
            "success": False,
            "progress": 0.0,
            "numerical_failure": True,
            "numerical_failure_reason": reason,
        }
        terminal = validate_terminal_numerical_failure_evidence(
            {
                "schema_version": 1,
                "profile": PI05_DROID_NATIVE_TERMINAL_FAILURE_PROFILE,
                "terminal_form": "native_all_joint_velocity_limit_failure",
                "environment_runtime_sha256": self._environment_runtime_contract[
                    "sha256"
                ],
                "failure_type": type(error).__name__,
                "episode_result": episode_result,
                "actions_attempted": self._execution_step_index + 1,
                "outer_steps_completed": self._execution_step_index,
                "failed_outer_step_index": self._execution_step_index,
                "terminated_false_count": self._terminated_false_count,
                "truncated_false_count": self._truncated_false_count,
                "environment_before": self._rollout_environment_before,
                "last_completed_environment": last_completed,
                "environment_after_failure": environment_after_failure,
                "incident_artifact": error.incident_artifact,
                "dynamic_report": dynamic_report,
            },
            self._environment_runtime_contract,
        )
        if self.trace_path is not None:
            self._append_trace(
                {
                    "schema_version": PI05_DROID_NATIVE_TRACE_SCHEMA_VERSION,
                    "record_type": "openpi_joint_velocity_rollout_failure",
                    "profile": PI05_DROID_JOINTVELOCITY_PROFILE,
                    **self._trace_contract_identity(),
                    "reset_index": self.reset_index,
                    "query_index": pending["query_index"],
                    "chunk_action_index": pending["chunk_action_index"],
                    "outer_step_index": self._execution_step_index,
                    "terminal_failure": terminal,
                }
            )
            self._finalized_trace_artifact = self._seal_trace()
        self._pending_execution = None
        self._terminal_rollout = terminal
        return terminal

    @property
    def finalized_trace_artifact(self) -> dict[str, Any] | None:
        return (
            None
            if self._finalized_trace_artifact is None
            else dict(self._finalized_trace_artifact)
        )

    def _seal_trace(self) -> dict[str, Any]:
        if self.trace_path is None:
            raise RuntimeError("Cannot seal an unconfigured native trace")
        self.trace_path.chmod(0o444)
        with self.trace_path.open("rb") as source:
            import os  # noqa: PLC0415

            os.fsync(source.fileno())
        fsync_directory(self.trace_path.parent)
        return validate_immutable_file(self.trace_path)

    def _resize_images(
        self, current: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        external = image_tools.resize_with_pad(current["right_image"], 224, 224)
        wrist = image_tools.resize_with_pad(current["wrist_image"], 224, 224)
        _image_contract(external)
        _image_contract(wrist)
        return external, wrist

    def _trace_contract_identity(self) -> dict[str, str | int]:
        if self._environment_runtime_contract is None:
            raise RuntimeError("Trace requested before environment runtime binding")
        return {
            "serving_contract_sha256": self.serving_contract["contract_sha256"],
            "serving_contract_artifact_sha256": self.serving_contract_artifact[
                "sha256"
            ],
            "serving_contract_artifact_size": self.serving_contract_artifact["size"],
            "client_runtime_attestation_sha256": self.client_runtime_attestation[
                "sha256"
            ],
            "environment_runtime_sha256": self._environment_runtime_contract["sha256"],
            "outer_episode_steps": PI05_DROID_NATIVE_EPISODE_STEPS,
            "internal_max_episode_length": (
                PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS
            ),
            "sensor_liveness_profile": PI05_DROID_NATIVE_SENSOR_LIVENESS_PROFILE,
        }

    def _trace_query(self, request: dict, action_chunk: np.ndarray) -> None:
        if self.trace_path is None:
            return
        blank = np.zeros((224, 224, 3), dtype=np.uint8)
        planned_binary = []
        planned_clipped = []
        for raw_action in action_chunk[: self.open_loop_horizon]:
            binary, clipped = process_native_jointvelocity_action(raw_action)
            planned_binary.append(binary.tolist())
            planned_clipped.append(clipped.tolist())
        self._append_trace(
            {
                "schema_version": PI05_DROID_NATIVE_TRACE_SCHEMA_VERSION,
                "record_type": "openpi_joint_velocity_query",
                "profile": PI05_DROID_JOINTVELOCITY_PROFILE,
                **self._trace_contract_identity(),
                "reset_index": self.reset_index,
                "query_index": self.query_index,
                "prompt": request["prompt"],
                "state": {
                    "joint_position": np.asarray(
                        request["observation/joint_position"]
                    ).tolist(),
                    "gripper_position": np.asarray(
                        request["observation/gripper_position"]
                    ).tolist(),
                },
                "images": {
                    "external": _image_contract(
                        request["observation/exterior_image_1_left"]
                    ),
                    "wrist": _image_contract(request["observation/wrist_image_left"]),
                    "blank_masked_right_wrist": _image_contract(blank),
                    "model_order": [
                        "base_0_rgb",
                        "left_wrist_0_rgb",
                        "right_wrist_0_rgb_masked",
                    ],
                    "wrist_rotation_degrees": 0,
                },
                "response_action_shape": list(action_chunk.shape),
                "response_action_chunk_raw": action_chunk.tolist(),
                "execution_horizon": self.open_loop_horizon,
                "planned_action_chunk_binary_gripper": planned_binary,
                "planned_action_chunk_clipped": planned_clipped,
            }
        )

    def _trace_emitted_action(self) -> None:
        if self.trace_path is None:
            return
        if self._pending_execution is None:
            raise RuntimeError("Cannot trace an action without pending execution")
        pending = self._pending_execution
        self._append_trace(
            {
                "schema_version": PI05_DROID_NATIVE_TRACE_SCHEMA_VERSION,
                "record_type": "openpi_joint_velocity_action",
                "profile": PI05_DROID_JOINTVELOCITY_PROFILE,
                **self._trace_contract_identity(),
                "reset_index": self.reset_index,
                "query_index": pending["query_index"],
                "chunk_action_index": pending["chunk_action_index"],
                "raw_action": np.asarray(pending["raw_action"]).tolist(),
                "binary_gripper_action": np.asarray(pending["binary_action"]).tolist(),
                "clipped_action": np.asarray(pending["clipped_action"]).tolist(),
                "emitted_joint_velocity": np.asarray(
                    pending["clipped_action"][:7]
                ).tolist(),
                "emitted_gripper_closed": float(pending["clipped_action"][7]),
                "measured_joint_position_before": np.asarray(
                    pending["pre_joint_position"]
                ).tolist(),
                "measured_joint_velocity_before": np.asarray(
                    pending["pre_joint_velocity"]
                ).tolist(),
                "measured_normalized_gripper_position_before": np.asarray(
                    pending["pre_normalized_gripper_position"]
                ).tolist(),
            }
        )

    def _append_trace(self, record: dict[str, Any]) -> None:
        if self.trace_path is None:
            return
        with self.trace_path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(
                json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
            )

    def _extract_observation(self, obs_dict: dict) -> dict[str, np.ndarray]:
        right_image = np.asarray(obs_dict["splat"]["external_cam"])
        wrist_image = np.asarray(obs_dict["splat"]["wrist_cam"])
        state = obs_dict["policy"]
        joint_position = _tensor_numpy(
            state["arm_joint_pos"], field="arm joint position"
        )[0]
        joint_velocity = _tensor_numpy(
            state["arm_joint_vel"], field="arm joint velocity"
        )[0]
        gripper_position = _tensor_numpy(
            state["gripper_pos"], field="gripper position"
        )[0]
        expected_shapes = {
            "joint position": (joint_position, (7,)),
            "joint velocity": (joint_velocity, (7,)),
            "gripper position": (gripper_position, (1,)),
        }
        for field, (array, shape) in expected_shapes.items():
            if array.shape != shape:
                raise ValueError(f"Expected {field} shape {shape}, got {array.shape}")
            if array.dtype != np.float32:
                raise ValueError(f"DROID {field} must be float32, got {array.dtype}")
            if not np.isfinite(array).all():
                raise JointVelocityObservationNumericalError(
                    f"DROID {field} contains non-finite values"
                )
        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "joint_velocity": joint_velocity,
            "gripper_position": gripper_position,
        }
