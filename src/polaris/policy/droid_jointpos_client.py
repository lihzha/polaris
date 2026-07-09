"""Official pi0.5 PolaRiS absolute joint-position client and live evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy

from polaris.evaluation_seed import (
    environment_seed_contract_sha256,
    make_episode_environment_rng,
    validate_live_environment_seed_contract,
)
from polaris.pi05_droid_jointpos_image_contract import (
    CLIENT_RESIZE_PROFILE,
    IMAGE_PROFILE,
    get_jointpos_image_evidence,
    resize_final_composite_for_wire,
    static_image_contract,
)
from polaris.pi05_droid_jointpos_runtime import (
    PANDA_ARM_JOINT_NAMES,
    PI05_DROID_JOINTPOS_DECIMATION,
    PI05_DROID_JOINTPOS_INTERNAL_MAX_EPISODE_STEPS,
    PI05_DROID_JOINTPOS_NATIVE_IMAGE_SHAPE,
    PI05_DROID_JOINTPOS_OUTER_STEPS,
    PI05_DROID_JOINTPOS_PROFILE,
    PI05_DROID_JOINTPOS_SENSOR_NAMES,
    PI05_DROID_JOINTPOS_TRACE_SCHEMA_VERSION,
    capture_jointpos_environment_state,
    validate_jointpos_runtime_report,
)
from polaris.pi05_droid_jointpos_serving_contract import (
    PI05_DROID_JOINTPOS_PROFILE as PI05_DROID_JOINTPOS_SERVING_PROFILE,
    PI05_DROID_JOINTPOS_SERVER_MODEL_RESIZE,
    pi05_droid_jointpos_server_contract_sha256,
    validate_pi05_droid_jointpos_server_metadata,
)
from polaris.policy.abstract_client import InferenceClient, PolicyArgs


PI05_DROID_CONTRACT_MARKER = "POLARIS_PI05_DROID_CONTRACT="


class JointPositionObservationNumericalError(FloatingPointError):
    """Raised when simulator corruption makes joint proprioception non-finite."""


def _latest_trace_reset_index(trace_path: Path) -> int:
    """Return the greatest reset index in an existing resumable trace."""

    if not trace_path.exists():
        return -1
    latest_reset_index = -1
    with trace_path.open(encoding="utf-8") as trace_file:
        for line_number, line in enumerate(trace_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid existing trace JSON at line {line_number}: {error}"
                ) from error
            reset_index = record.get("reset_index")
            if not isinstance(reset_index, int) or reset_index < 0:
                raise ValueError(
                    f"Invalid existing trace reset_index at line {line_number}"
                )
            latest_reset_index = max(latest_reset_index, reset_index)
    return latest_reset_index


def _next_global_query_index(trace_path: Path) -> int:
    if not trace_path.exists():
        return 0
    indices = []
    with trace_path.open(encoding="utf-8") as trace_file:
        for line_number, line in enumerate(trace_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type") != "openpi_joint_position_query":
                continue
            value = record.get("global_query_index")
            if type(value) is not int or value < 0:
                raise ValueError(
                    "Existing attested query lacks a valid global_query_index at "
                    f"line {line_number}"
                )
            indices.append(value)
    if not indices:
        return 0
    if indices != list(range(len(indices))):
        raise ValueError("Existing global query indices are not contiguous from zero")
    return len(indices)


def _image_contract(
    image: Any, *, expected_shape: tuple[int, int, int], field: str
) -> dict[str, Any]:
    image = np.ascontiguousarray(np.asarray(image))
    if image.shape != expected_shape or image.dtype != np.uint8:
        raise ValueError(
            f"{field} must be uint8 {list(expected_shape)}, got "
            f"{image.dtype} {list(image.shape)}"
        )
    return {
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "sha256": hashlib.sha256(image.tobytes()).hexdigest(),
    }


def _tensor_numpy(value: Any, *, field: str) -> np.ndarray:
    try:
        value = value.detach().cpu().numpy()
    except AttributeError:
        value = np.asarray(value)
    result = np.asarray(value)
    if (
        not np.issubdtype(result.dtype, np.number)
        or np.issubdtype(result.dtype, np.bool_)
        or not np.isfinite(result).all()
    ):
        raise ValueError(f"{field} must be finite numeric data")
    return result


def validate_joint_action_chunk(
    response: dict,
    *,
    open_loop_horizon: int,
    expected_action_horizon: int | None = None,
    expected_action_dim: int | None = None,
) -> np.ndarray:
    """Validate the absolute joint-position chunk returned by OpenPI."""

    if "actions" not in response:
        raise KeyError("OpenPI response is missing 'actions'")
    actions = np.asarray(response["actions"])
    if actions.ndim != 2:
        raise ValueError(
            f"Expected OpenPI actions with shape (T, D), got {actions.shape}"
        )
    if (
        expected_action_horizon is not None
        and actions.shape[0] != expected_action_horizon
    ):
        raise ValueError(
            "OpenPI action horizon mismatch: "
            f"expected {expected_action_horizon}, got {actions.shape[0]}"
        )
    if expected_action_dim is not None and actions.shape[1] != expected_action_dim:
        raise ValueError(
            "OpenPI action width mismatch: "
            f"expected {expected_action_dim}, got {actions.shape[1]}"
        )
    if actions.shape[0] < open_loop_horizon:
        raise ValueError(
            "OpenPI action chunk is shorter than the requested execution horizon: "
            f"{actions.shape[0]} < {open_loop_horizon}"
        )
    if not np.isfinite(actions).all():
        raise ValueError("OpenPI action chunk contains non-finite values")
    return actions


def _binarize_gripper(action_chunk: np.ndarray) -> np.ndarray:
    action_chunk = np.asarray(action_chunk)
    gripper = np.where(action_chunk[..., -1:] > 0.5, 1.0, 0.0)
    return np.concatenate([action_chunk[..., :-1], gripper], axis=-1)


def _validate_rubric(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"success", "progress", "metrics"}:
        raise ValueError("terminal rubric schema mismatch")
    if type(value["success"]) is not bool:
        raise ValueError("terminal rubric success must be boolean")
    if (
        not isinstance(value["progress"], int | float)
        or isinstance(value["progress"], bool)
        or not math.isfinite(value["progress"])
        or not isinstance(value["metrics"], dict)
    ):
        raise ValueError("terminal rubric progress/metrics mismatch")
    return json.loads(json.dumps(value, allow_nan=False))


@InferenceClient.register(client_name="DroidJointPos")
class DroidJointPosClient(InferenceClient):
    def __init__(self, args: PolicyArgs) -> None:
        self.args = args
        self._validate_args()
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host, port=args.port
        )
        server_metadata = self.client.get_server_metadata()
        self.server_contract = validate_pi05_droid_jointpos_server_metadata(
            server_metadata
        )
        self.server_contract_sha256 = pi05_droid_jointpos_server_contract_sha256(
            self.server_contract
        )
        if self.server_contract.get("contract_sha256") != self.server_contract_sha256:
            raise ValueError("validated server contract hash mismatch")

        self.open_loop_horizon = args.open_loop_horizon
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk: np.ndarray | None = None
        self.query_index = 0
        self.active_query_index: int | None = None
        self.active_global_query_index: int | None = None
        self.environment_seed_contract: dict[str, Any] | None = None
        self.environment_seed_contract_sha256: str | None = None
        self.active_environment_rng: dict[str, Any] | None = None
        self.runtime_contract: dict[str, Any] | None = None
        self.runtime_contract_sha256: str | None = None
        self._rollout_environment_before: dict[str, Any] | None = None
        self._last_environment_after: dict[str, Any] | None = None
        self._image_evidence_env: Any | None = None
        self._pending_execution: dict[str, Any] | None = None
        self._terminal_visualization: np.ndarray | None = None
        self.outer_step_index = 0
        self.trace_path = Path(args.trace_path) if args.trace_path else None
        if self.trace_path is None:
            raise ValueError("DroidJointPos requires trace_path")
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.reset_index = _latest_trace_reset_index(self.trace_path)
        self.global_query_index = _next_global_query_index(self.trace_path)

        marker = {
            "client": "DroidJointPos",
            "profile": PI05_DROID_JOINTPOS_PROFILE,
            "serving_profile": PI05_DROID_JOINTPOS_SERVING_PROFILE,
            "server_contract_sha256": self.server_contract_sha256,
            "state": "ordered_7_panda_joint_radians_plus_closed_positive_gripper",
            "action": "7_absolute_panda_joint_targets_plus_closed_positive_gripper",
            "image_slots": [
                "base_0_rgb",
                "left_wrist_0_rgb",
                "right_wrist_0_rgb_masked",
            ],
            "final_composite_image_shape": list(PI05_DROID_JOINTPOS_NATIVE_IMAGE_SHAPE),
            "final_composite_image_source": (
                "post_manager_filtered_splat_then_sim_mask_composite"
            ),
            "environment_image_profile": IMAGE_PROFILE,
            "environment_image_contract": static_image_contract(),
            "request_image_shape": [224, 224, 3],
            "request_image_dtype": "uint8",
            "request_image_source": "client_resize_with_pad_224_of_final_composite",
            "client_model_spatial_transform": CLIENT_RESIZE_PROFILE,
            "server_model_resize": PI05_DROID_JOINTPOS_SERVER_MODEL_RESIZE,
            "model_image_resolution": [224, 224],
            "visualization_image_resolution": [224, 224],
            "query_visualization_source": "byte_identical_client224_wire_model_input",
            "interquery_visualization_source": (
                "client224_resize_of_nonexpensive_sim_camera_non_model_input"
            ),
            "wrist_rotation_degrees": 0,
            "open_loop_horizon": self.open_loop_horizon,
            "response_horizon": args.expected_action_horizon,
            "outer_steps": PI05_DROID_JOINTPOS_OUTER_STEPS,
            "internal_max_episode_steps": (
                PI05_DROID_JOINTPOS_INTERNAL_MAX_EPISODE_STEPS
            ),
            "initial_reset_index": self.reset_index,
            "initial_global_query_index": self.global_query_index,
        }
        print(
            PI05_DROID_CONTRACT_MARKER
            + json.dumps(marker, sort_keys=True, separators=(",", ":")),
            flush=True,
        )

    def _validate_args(self) -> None:
        expected = {
            "open_loop_horizon": 8,
            "expected_action_horizon": 15,
            "expected_action_dim": 8,
            "state_type": "joint_position",
            "frame_description": "robot base frame",
            "action_frame": "robot_base",
            "dataset_name": "droid",
            "rotate_wrist_180": False,
            "render_every_step": False,
        }
        for name, expected_value in expected.items():
            actual = getattr(self.args, name, None)
            if actual != expected_value:
                raise ValueError(
                    f"DroidJointPos requires {name}={expected_value!r}; got {actual!r}"
                )

    @property
    def rerender(self) -> bool:
        return (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        )

    def bind_environment_seed_contract(self, contract: dict[str, Any]) -> None:
        if self.environment_seed_contract is not None:
            raise RuntimeError("Environment seed contract was bound more than once")
        self.environment_seed_contract = validate_live_environment_seed_contract(
            contract
        )
        self.environment_seed_contract_sha256 = environment_seed_contract_sha256(
            self.environment_seed_contract
        )

    def bind_jointpos_runtime(self, report: dict[str, Any]) -> None:
        canonical = validate_jointpos_runtime_report(report)
        digest = canonical["runtime_sha256"]
        if self.runtime_contract is None:
            self.runtime_contract = canonical
            self.runtime_contract_sha256 = digest
        elif self.runtime_contract_sha256 != digest:
            raise ValueError("joint-position runtime drifted across episode reset")

    def reset(
        self,
        *,
        episode_index: int | None = None,
        episode_seed: int | None = None,
    ) -> None:
        if self.environment_seed_contract is None:
            raise RuntimeError("DroidJointPos environment seed contract is not bound")
        if self.runtime_contract is None:
            raise RuntimeError("DroidJointPos live runtime contract is not bound")
        if self._pending_execution is not None:
            raise RuntimeError("cannot reset with an unrecorded joint-position action")
        next_reset_index = self.reset_index + 1
        if episode_index != next_reset_index:
            raise ValueError("Episode index does not match the next trace reset index")
        expected_rng = make_episode_environment_rng(
            self.environment_seed_contract, episode_index
        )
        if episode_seed != expected_rng["episode_seed"]:
            raise ValueError("Episode seed does not match the bound seed scheme")
        self.active_environment_rng = expected_rng
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.reset_index = next_reset_index
        self.query_index = 0
        self.active_query_index = None
        self.active_global_query_index = None
        self.outer_step_index = 0
        self._rollout_environment_before = None
        self._last_environment_after = None
        self._image_evidence_env = None
        self._terminal_visualization = None

    def begin_rollout(self, env: Any) -> dict[str, Any]:
        if self.runtime_contract is None:
            raise RuntimeError("joint-position runtime is not bound")
        if self._rollout_environment_before is not None:
            raise RuntimeError("joint-position rollout already began")
        before = capture_jointpos_environment_state(env)
        if before["episode_length"] != 0:
            raise ValueError("explicit episode reset did not zero episode length")
        self._rollout_environment_before = before
        self._last_environment_after = before
        self._image_evidence_env = env
        return dict(before)

    def infer(
        self, obs: dict, instruction: str, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if self._pending_execution is not None:
            raise RuntimeError("prior emitted action has no live execution record")
        if self._last_environment_after is None:
            raise RuntimeError("joint-position rollout did not begin")
        current = self._extract_observation(obs)
        visualization = None
        query_now = self.rerender
        if query_now:
            self.actions_from_chunk_completed = 0
            if self._image_evidence_env is None:
                raise RuntimeError(
                    "joint-position image evidence environment is unbound"
                )
            image_evidence = get_jointpos_image_evidence(self._image_evidence_env, obs)
            (
                external,
                wrist,
                external_resize_evidence,
                wrist_resize_evidence,
            ) = self._wire_request_images(current)
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
                expected_action_horizon=self.args.expected_action_horizon,
                expected_action_dim=self.args.expected_action_dim,
            )
            if self.pred_action_chunk.dtype != np.float64:
                raise ValueError(
                    "official pi0.5 joint-position response must be float64 after "
                    f"unnormalization; got {self.pred_action_chunk.dtype}"
                )
            self.active_query_index = self.query_index
            self.active_global_query_index = self.global_query_index
            self._trace_query(
                request_data,
                self.pred_action_chunk,
                current=current,
                image_evidence=image_evidence,
                client_resize_external=external_resize_evidence,
                client_resize_wrist=wrist_resize_evidence,
            )
            self.query_index += 1
            self.global_query_index += 1
            visualization = np.concatenate([external, wrist], axis=1)
        elif return_viz:
            viz_external, viz_wrist = self._diagnostic_images(current)
            visualization = np.concatenate([viz_external, viz_wrist], axis=1)

        if (
            self.pred_action_chunk is None
            or self.active_query_index is None
            or self.active_global_query_index is None
        ):
            raise ValueError("No pi0.5 joint-position action chunk predicted")
        action_index = self.actions_from_chunk_completed
        raw_action = self.pred_action_chunk[action_index].copy()
        emitted_action = _binarize_gripper(raw_action)
        self.actions_from_chunk_completed += 1
        self._pending_execution = {
            "query_index": self.active_query_index,
            "global_query_index": self.active_global_query_index,
            "chunk_action_index": action_index,
            "raw_action": raw_action,
            "emitted_action": emitted_action,
            "environment_before": self._last_environment_after,
        }
        self._trace_emitted_action()
        return emitted_action, visualization

    def record_execution(
        self,
        obs: dict,
        env: Any,
        *,
        terminated: Any,
        truncated: Any,
        terminal_rubric: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._pending_execution is None:
            raise RuntimeError("no pending joint-position action to record")
        if self._rollout_environment_before is None:
            raise RuntimeError("joint-position rollout counters are not bound")
        if self.outer_step_index >= PI05_DROID_JOINTPOS_OUTER_STEPS:
            raise RuntimeError("more than 450 joint-position actions were recorded")
        try:
            terminated_array = terminated.detach().cpu().numpy()
        except AttributeError:
            terminated_array = np.asarray(terminated)
        try:
            truncated_array = truncated.detach().cpu().numpy()
        except AttributeError:
            truncated_array = np.asarray(truncated)
        if (
            terminated_array.shape != (1,)
            or truncated_array.shape != (1,)
            or terminated_array.dtype != np.bool_
            or truncated_array.dtype != np.bool_
            or bool(terminated_array[0])
            or bool(truncated_array[0])
        ):
            raise ValueError(
                "outer450_internal451_no_autoreset requires every returned "
                "terminated/truncated flag to be false"
            )
        completed = self.outer_step_index + 1
        environment_after = capture_jointpos_environment_state(env)
        before = self._rollout_environment_before
        if (
            environment_after["episode_length"] != completed
            or environment_after["sim_step_counter"]
            != before["sim_step_counter"] + completed * PI05_DROID_JOINTPOS_DECIMATION
            or environment_after["common_step_counter"]
            != before["common_step_counter"] + completed
            or any(
                environment_after["sensor_frame_counters"][name]
                != before["sensor_frame_counters"][name] + completed
                for name in PI05_DROID_JOINTPOS_SENSOR_NAMES
            )
        ):
            raise ValueError("joint-position simulator/camera cadence mismatch")

        root = getattr(env, "unwrapped", env)
        arm = root.action_manager._terms["arm"]
        finger = root.action_manager._terms["finger_joint"]
        robot = root.scene["robot"]
        joint_ids, joint_names = robot.find_joints(
            list(PANDA_ARM_JOINT_NAMES), preserve_order=True
        )
        if tuple(joint_names) != PANDA_ARM_JOINT_NAMES:
            raise ValueError(f"live Panda target order mismatch: {joint_names}")
        finger_ids, finger_names = robot.find_joints(
            ["finger_joint"], preserve_order=True
        )
        if finger_names != ["finger_joint"]:
            raise ValueError("live finger target order mismatch")
        execution = arm.consume_joint_position_execution_report()
        pending = self._pending_execution
        expected = np.asarray(pending["emitted_action"][:7], dtype=np.float32)
        if not np.array_equal(
            np.asarray(execution["raw_action_buffer"], dtype=np.float32), expected
        ):
            raise ValueError("upstream raw action buffer differs from emitted target")
        processed_finger = _tensor_numpy(
            finger.processed_actions, field="processed finger target"
        )
        articulation_finger = _tensor_numpy(
            robot.data.joint_pos_target[:, finger_ids],
            field="articulation finger target",
        )
        expected_finger_value = (
            np.float32(np.pi / 4.0)
            if pending["emitted_action"][7] == 1.0
            else np.float32(0.0)
        )
        expected_finger = np.asarray([[expected_finger_value]], dtype=np.float32)
        if (
            processed_finger.dtype != np.float32
            or articulation_finger.dtype != np.float32
            or not np.array_equal(processed_finger, expected_finger)
            or not np.array_equal(articulation_finger, expected_finger)
        ):
            raise ValueError(
                "live gripper target differs from closed-positive emission"
            )
        current = self._extract_observation(obs)
        if completed == PI05_DROID_JOINTPOS_OUTER_STEPS:
            rubric = _validate_rubric(terminal_rubric)
            terminal_external, terminal_wrist = self._diagnostic_images(current)
            self._terminal_visualization = np.ascontiguousarray(
                np.concatenate([terminal_external, terminal_wrist], axis=1)
            )
            terminal_visualization = {
                **_image_contract(
                    self._terminal_visualization,
                    expected_shape=(224, 448, 3),
                    field="post-action-450 terminal visualization",
                ),
                "source": (
                    "post_action450_returned_nonexpensive_sim_camera_observation"
                ),
            }
        elif terminal_rubric is not None:
            raise ValueError("terminal rubric may only be attached to action 450")
        else:
            rubric = None
            terminal_visualization = None
        record = {
            **self._trace_identity(),
            "record_type": "openpi_joint_position_execution",
            "reset_index": self.reset_index,
            "query_index": pending["query_index"],
            "global_query_index": pending["global_query_index"],
            "chunk_action_index": pending["chunk_action_index"],
            "outer_step_index": self.outer_step_index,
            "emitted_action": pending["emitted_action"].tolist(),
            "action_execution": execution,
            "processed_finger_position_target": processed_finger[0].tolist(),
            "articulation_finger_position_target": articulation_finger[0].tolist(),
            "measured_joint_position_after": current["joint_position"].tolist(),
            "measured_closed_positive_gripper_after": current[
                "gripper_position"
            ].tolist(),
            "environment_before": pending["environment_before"],
            "environment_after": environment_after,
            "terminated": False,
            "truncated": False,
            "terminal_rubric": rubric,
            "terminal_visualization": terminal_visualization,
        }
        self._append_trace(record)
        self._pending_execution = None
        self._last_environment_after = environment_after
        self.outer_step_index = completed
        return record

    def final_terminal_visualization(self) -> np.ndarray:
        """Return the trace-bound post-action-450 image for durable PNG evidence."""

        if (
            self.outer_step_index != PI05_DROID_JOINTPOS_OUTER_STEPS
            or self._pending_execution is not None
            or self._terminal_visualization is None
        ):
            raise RuntimeError(
                "terminal visualization is unavailable before action 450"
            )
        _image_contract(
            self._terminal_visualization,
            expected_shape=(224, 448, 3),
            field="post-action-450 terminal visualization",
        )
        return self._terminal_visualization.copy()

    def visualize(self, request: dict) -> np.ndarray:
        current = self._extract_observation(request)
        external, wrist = self._diagnostic_images(current)
        return np.concatenate([external, wrist], axis=1)

    def _trace_identity(self) -> dict[str, Any]:
        if (
            self.environment_seed_contract is None
            or self.environment_seed_contract_sha256 is None
            or self.runtime_contract_sha256 is None
        ):
            raise RuntimeError("trace identity requested before live contracts")
        return {
            "schema_version": PI05_DROID_JOINTPOS_TRACE_SCHEMA_VERSION,
            "profile": PI05_DROID_JOINTPOS_PROFILE,
            "server_contract_sha256": self.server_contract_sha256,
            "environment_seed_contract_sha256": (self.environment_seed_contract_sha256),
            "runtime_contract_sha256": self.runtime_contract_sha256,
            "physx_enhanced_determinism": self.environment_seed_contract[
                "physx_enhanced_determinism"
            ],
        }

    def _trace_query(
        self,
        request: dict,
        action_chunk: np.ndarray,
        *,
        current: dict[str, np.ndarray],
        image_evidence: dict[str, Any],
        client_resize_external: dict[str, Any],
        client_resize_wrist: dict[str, Any],
    ) -> None:
        if self.active_environment_rng is None:
            raise RuntimeError("query has no environment RNG provenance")
        if self.active_global_query_index is None:
            raise RuntimeError("query has no global request identity")
        planned = _binarize_gripper(action_chunk[: self.open_loop_horizon])
        final_external = _image_contract(
            current["right_image"],
            expected_shape=PI05_DROID_JOINTPOS_NATIVE_IMAGE_SHAPE,
            field="final external composite",
        )
        final_wrist = _image_contract(
            current["wrist_image"],
            expected_shape=PI05_DROID_JOINTPOS_NATIVE_IMAGE_SHAPE,
            field="final wrist composite",
        )
        request_external = _image_contract(
            request["observation/exterior_image_1_left"],
            expected_shape=(224, 224, 3),
            field="external client224 wire request",
        )
        request_wrist = _image_contract(
            request["observation/wrist_image_left"],
            expected_shape=(224, 224, 3),
            field="wrist client224 wire request",
        )
        if (
            client_resize_external["input_final_composite"] != final_external
            or client_resize_external["wire_request"] != request_external
        ):
            raise ValueError("external final720/client224 evidence mismatch")
        if (
            client_resize_wrist["input_final_composite"] != final_wrist
            or client_resize_wrist["wire_request"] != request_wrist
        ):
            raise ValueError("wrist final720/client224 evidence mismatch")
        self._append_trace(
            {
                **self._trace_identity(),
                "record_type": "openpi_joint_position_query",
                "reset_index": self.reset_index,
                "query_index": self.query_index,
                "global_query_index": self.active_global_query_index,
                "environment_rng": self.active_environment_rng,
                "sensor_frame_counters": self._last_environment_after[
                    "sensor_frame_counters"
                ],
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
                    "environment_image_contract": static_image_contract(),
                    "external_camera_pipeline": image_evidence["external_cam"],
                    "wrist_camera_pipeline": image_evidence["wrist_cam"],
                    "final_composite_external": final_external,
                    "final_composite_wrist": final_wrist,
                    "client_resize_external": client_resize_external,
                    "client_resize_wrist": client_resize_wrist,
                    "request_external": request_external,
                    "request_wrist": request_wrist,
                    "server224_external_idempotent": request_external,
                    "server224_wrist_idempotent": request_wrist,
                    "query_visualization_external": request_external,
                    "query_visualization_wrist": request_wrist,
                    "model_order": [
                        "base_0_rgb",
                        "left_wrist_0_rgb",
                        "right_wrist_0_rgb_masked",
                    ],
                    "client_model_spatial_transform": CLIENT_RESIZE_PROFILE,
                    "server_model_resize": PI05_DROID_JOINTPOS_SERVER_MODEL_RESIZE,
                    "masked_third_slot": (
                        "server_DroidInputs_zeros_like_base_mask_false"
                    ),
                    "query_visualization_source": (
                        "byte_identical_client224_wire_model_input"
                    ),
                    "wrist_rotation_degrees": 0,
                },
                "response_action_shape": list(action_chunk.shape),
                "response_action_dtype": str(action_chunk.dtype),
                "response_action_chunk": action_chunk.tolist(),
                "execution_horizon": self.open_loop_horizon,
                "planned_action_chunk": planned.tolist(),
            }
        )

    def _trace_emitted_action(self) -> None:
        if self._pending_execution is None:
            raise RuntimeError("cannot trace an action without pending execution")
        pending = self._pending_execution
        self._append_trace(
            {
                **self._trace_identity(),
                "record_type": "openpi_joint_position_action",
                "reset_index": self.reset_index,
                "query_index": pending["query_index"],
                "global_query_index": pending["global_query_index"],
                "chunk_action_index": pending["chunk_action_index"],
                "raw_action": pending["raw_action"].tolist(),
                "emitted_action": pending["emitted_action"].tolist(),
            }
        )

    def _append_trace(self, record: dict[str, Any]) -> None:
        if self.trace_path is None:
            raise RuntimeError("DroidJointPos trace path is not configured")
        with self.trace_path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(
                json.dumps(
                    record,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )

    def _wire_request_images(
        self, current: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
        """Apply the official client-side resize before WebSocket transport."""

        external, external_evidence = resize_final_composite_for_wire(
            current["right_image"],
            image_tools_module=image_tools,
            camera_name="external_cam",
        )
        wrist, wrist_evidence = resize_final_composite_for_wire(
            current["wrist_image"],
            image_tools_module=image_tools,
            camera_name="wrist_cam",
        )
        _image_contract(
            external,
            expected_shape=(224, 224, 3),
            field="external client224 wire request",
        )
        _image_contract(
            wrist,
            expected_shape=(224, 224, 3),
            field="wrist client224 wire request",
        )
        return external, wrist, external_evidence, wrist_evidence

    def _diagnostic_images(
        self, current: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resize a sim-only inter-query or terminal diagnostic frame."""

        external = image_tools.resize_with_pad(current["right_image"], 224, 224)
        wrist = image_tools.resize_with_pad(current["wrist_image"], 224, 224)
        _image_contract(
            external,
            expected_shape=(224, 224, 3),
            field="non-model external visualization",
        )
        _image_contract(
            wrist,
            expected_shape=(224, 224, 3),
            field="non-model wrist visualization",
        )
        return external, wrist

    def _extract_observation(self, obs_dict: dict[str, Any]) -> dict[str, np.ndarray]:
        right_image = np.asarray(obs_dict["splat"]["external_cam"])
        wrist_image = np.asarray(obs_dict["splat"]["wrist_cam"])
        _image_contract(
            right_image,
            expected_shape=PI05_DROID_JOINTPOS_NATIVE_IMAGE_SHAPE,
            field="final external composite",
        )
        _image_contract(
            wrist_image,
            expected_shape=PI05_DROID_JOINTPOS_NATIVE_IMAGE_SHAPE,
            field="final wrist composite",
        )
        robot_state = obs_dict["policy"]
        try:
            joint_position = robot_state["arm_joint_pos"].detach().cpu().numpy()[0]
            gripper_position = robot_state["gripper_pos"].detach().cpu().numpy()[0]
        except AttributeError:
            joint_position = np.asarray(robot_state["arm_joint_pos"])[0]
            gripper_position = np.asarray(robot_state["gripper_pos"])[0]
        if joint_position.shape != (7,):
            raise ValueError(
                f"Expected seven ordered Panda joint positions, got {joint_position.shape}"
            )
        if gripper_position.shape != (1,):
            raise ValueError(
                f"Expected one closed-positive gripper position, got {gripper_position.shape}"
            )
        if joint_position.dtype != np.float32 or gripper_position.dtype != np.float32:
            raise ValueError(
                "joint-position and closed-positive gripper observations must be "
                "float32"
            )
        if bool((gripper_position < 0.0).any()) or bool((gripper_position > 1.0).any()):
            raise ValueError(
                "historical closed-positive gripper observation was not clipped "
                "to [0, 1]"
            )
        if (
            not np.isfinite(joint_position).all()
            or not np.isfinite(gripper_position).all()
        ):
            raise JointPositionObservationNumericalError(
                "Joint-position observation contains non-finite values"
            )
        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }
