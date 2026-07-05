"""Official ``pi05_droid`` client with DROID-faithful position execution."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy

from polaris.pi05_droid_position_adapter import (
    PI05_DROID_EXECUTION_HORIZON,
    PI05_DROID_POSITION_ADAPTER_PROFILE,
    PI05_DROID_POSITION_TRACE_SCHEMA_VERSION,
    PI05_DROID_RESPONSE_HORIZON,
    adapt_official_droid_action,
    canonical_json_bytes,
    evaluate_position_target_guard,
    expected_position_limit_contract,
    validate_position_adapter_evidence,
    validate_position_limit_contract,
    validate_position_target_hold_report,
)
from polaris.pi05_droid_position_contract import (
    PANDA_ARM_JOINT_NAMES,
    PI05_DROID_GRIPPER_OBSERVATION_BOUND_TOLERANCE,
    validate_persisted_position_serving_contract,
    validate_pi05_droid_position_server_metadata,
    verify_openpi_git_checkout,
)
from polaris.pi05_droid_native_eval_contract import (
    PI05_DROID_NATIVE_DECIMATION,
    PI05_DROID_NATIVE_EPISODE_STEPS,
    PI05_DROID_NATIVE_SENSOR_NAMES,
    publish_immutable_json,
    validate_environment_runtime_contract,
    validate_outer_step_flags,
)
from polaris.native_gripper_runtime import (
    NativeAllJointVelocityLimitError,
    validate_native_all_joint_dynamic_report,
    validate_native_all_joint_velocity_failure,
)
from polaris.policy.abstract_client import InferenceClient, PolicyArgs
from polaris.policy.droid_jointpos_client import (
    JointPositionObservationNumericalError,
    validate_joint_action_chunk,
)


PI05_DROID_POSITION_CLIENT_MARKER = "POLARIS_PI05_DROID_POSITION_CLIENT="
MODEL_IMAGE_ORDER = [
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb_masked",
]


class PositionTargetLimitError(RuntimeError):
    """Typed pre-setter target-limit abort with immutable evidence."""

    def __init__(self, evidence: dict[str, Any], incident_artifact: dict[str, Any]):
        super().__init__(
            "adapted absolute joint target exceeds exact zero-inset live hard/soft "
            "intersection guard before setter"
        )
        self.evidence = copy.deepcopy(evidence)
        self.incident_artifact = copy.deepcopy(incident_artifact)


def _serialized_float32(
    value: Any, *, shape: tuple[int, ...], field: str
) -> np.ndarray:
    serialized = np.asarray(value)
    if (
        serialized.shape != shape
        or not np.issubdtype(serialized.dtype, np.number)
        or not np.isfinite(serialized).all()
    ):
        raise ValueError(f"{field} must be one finite numeric {shape} value")
    as_float32 = serialized.astype(np.float32)
    if not np.array_equal(serialized.astype(np.float64), as_float32.astype(np.float64)):
        raise ValueError(f"{field} is not an exact serialized float32 value")
    return as_float32


def validate_position_target_limit_incident(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "profile",
        "reason",
        "outer_step_index",
        "query_index",
        "chunk_action_index",
        "measured_joint_position",
        "clipped_arm_command",
        "reference_float64_absolute_joint_position_target",
        "guarded_float32_joint_position_target",
        "position_limit_contract_sha256",
        "live_buffered_hard_joint_position_limits",
        "live_isaaclab_soft_joint_position_limits",
        "derived_target_guard_joint_position_limits",
        "guard_source",
        "guard_inset_rad",
        "violating_joint_indices",
        "setter_calls_for_rejected_target",
    }:
        raise ValueError("position target-limit incident schema mismatch")
    reference_target = np.asarray(
        value["reference_float64_absolute_joint_position_target"]
    )
    guarded_target = _serialized_float32(
        value["guarded_float32_joint_position_target"],
        shape=(7,),
        field="incident guarded target",
    )
    live_hard = _serialized_float32(
        value["live_buffered_hard_joint_position_limits"],
        shape=(7, 2),
        field="incident live hard limits",
    )
    live_soft = _serialized_float32(
        value["live_isaaclab_soft_joint_position_limits"],
        shape=(7, 2),
        field="incident live soft limits",
    )
    recomputed_target, guard, mask = evaluate_position_target_guard(
        reference_target, live_hard, live_soft
    )
    recorded_guard = _serialized_float32(
        value["derived_target_guard_joint_position_limits"],
        shape=(7, 2),
        field="incident derived target guard",
    )
    expected_limit_contract = expected_position_limit_contract()
    if (
        value["schema_version"] != 1
        or value["profile"] != "openpi_pi05_droid_position_target_limit_failure_v1"
        or value["reason"] != "absolute_target_outside_zero_inset_live_hard_soft_guard"
        or reference_target.shape != (7,)
        or reference_target.dtype != np.float64
        or not np.array_equal(guarded_target, recomputed_target[0])
        or not np.array_equal(recorded_guard, guard[0])
        or value["position_limit_contract_sha256"]
        != expected_limit_contract["contract_sha256"]
        or value["guard_source"]
        != "intersection(live_joint_pos_limits,live_soft_joint_pos_limits)"
        or value["guard_inset_rad"] != 0.0
        or value["violating_joint_indices"] != np.flatnonzero(mask).tolist()
        or not bool(mask.any())
        or value["setter_calls_for_rejected_target"] != 0
    ):
        raise ValueError("position target-limit incident identity mismatch")
    return copy.deepcopy(value)


def _tensor_numpy(value: Any, *, field: str) -> np.ndarray:
    try:
        value = value.detach().cpu().numpy()
    except AttributeError:
        value = np.asarray(value)
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{field} must be numeric")
    return array


def _image_contract(image: Any) -> dict[str, Any]:
    image = np.ascontiguousarray(np.asarray(image))
    if image.shape != (224, 224, 3) or image.dtype != np.uint8:
        raise ValueError("pi0.5-DROID images must be 224x224 uint8 RGB")
    return {
        "shape": [224, 224, 3],
        "dtype": "uint8",
        "sha256": hashlib.sha256(image.tobytes()).hexdigest(),
    }


def _native_image_contract(image: Any, *, field: str) -> dict[str, Any]:
    image = np.ascontiguousarray(np.asarray(image))
    if image.shape != (720, 1280, 3) or image.dtype != np.uint8:
        raise ValueError(f"{field} must be native uint8 RGB with shape [720,1280,3]")
    return {
        "shape": [720, 1280, 3],
        "dtype": "uint8",
        "sha256": hashlib.sha256(image.tobytes()).hexdigest(),
    }


def _validate_image_identity(
    value: Any, *, expected_shape: list[int], field: str
) -> None:
    if not isinstance(value, dict) or set(value) != {"shape", "dtype", "sha256"}:
        raise ValueError(f"{field} identity schema mismatch")
    digest = value["sha256"]
    if (
        value["shape"] != expected_shape
        or value["dtype"] != "uint8"
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{field} identity mismatch")


def _single_nonnegative_integer(value: Any, *, field: str) -> int:
    array = _tensor_numpy(value, field=field)
    if array.shape != (1,) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{field} must be one integer tensor")
    result = int(array[0])
    if result < 0:
        raise ValueError(f"{field} must be nonnegative")
    return result


def _capture_environment_state(
    env: Any, environment_runtime_contract: dict[str, Any]
) -> dict[str, Any]:
    runtime = validate_environment_runtime_contract(environment_runtime_contract)
    root = getattr(env, "unwrapped", env)
    if root.max_episode_length != runtime["live_max_episode_length"]:
        raise ValueError("position environment max_episode_length drifted")
    scene = root.scene
    sensors = getattr(scene, "sensors", None)
    if not isinstance(sensors, dict) or set(PI05_DROID_NATIVE_SENSOR_NAMES) - set(
        sensors
    ):
        raise ValueError("position environment camera mapping drifted")
    return {
        "live_max_episode_length": root.max_episode_length,
        "episode_length": _single_nonnegative_integer(
            root.episode_length_buf, field="episode length"
        ),
        "sim_step_counter": int(root._sim_step_counter),
        "common_step_counter": int(root.common_step_counter),
        "sensor_frame_counters": {
            name: _single_nonnegative_integer(
                sensors[name].frame, field=f"{name} frame"
            )
            for name in PI05_DROID_NATIVE_SENSOR_NAMES
        },
    }


def _validate_environment_trace_state(value: Any) -> dict[str, Any]:
    required = {
        "live_max_episode_length",
        "episode_length",
        "sim_step_counter",
        "common_step_counter",
        "sensor_frame_counters",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("position environment trace schema mismatch")
    for field in (
        "live_max_episode_length",
        "episode_length",
        "sim_step_counter",
        "common_step_counter",
    ):
        if type(value[field]) is not int or value[field] < 0:
            raise ValueError(f"position environment {field} mismatch")
    counters = value["sensor_frame_counters"]
    if (
        not isinstance(counters, dict)
        or set(counters) != set(PI05_DROID_NATIVE_SENSOR_NAMES)
        or any(type(item) is not int or item < 0 for item in counters.values())
    ):
        raise ValueError("position environment sensor counters mismatch")
    return copy.deepcopy(value)


def validate_position_trace_record(value: Any) -> dict[str, Any]:
    """Validate one closed-schema position-adapter trace record."""

    if not isinstance(value, dict):
        raise ValueError("position trace record must be an object")
    common = {
        "schema_version",
        "record_type",
        "profile",
        "serving_contract_sha256",
        "serving_contract_artifact_sha256",
        "reset_index",
    }
    record_type = value.get("record_type")
    specific = {
        "openpi_droid_position_rollout_start": {"environment_before"},
        "openpi_droid_position_query": {
            "query_index",
            "prompt",
            "request_state",
            "images",
            "model_image_order",
            "normalization_scope",
            "normalization_asset_id",
            "normalization_sha256",
            "sampler",
            "response_action_shape",
            "response_action_dtype",
            "response_action_chunk",
            "execution_horizon",
        },
        "openpi_droid_position_action": {
            "query_index",
            "chunk_action_index",
            "live_pre_step_joint_position",
            "policy_observation_joint_position",
            "guarded_float32_joint_position_target",
            "position_limit_contract_sha256",
            "live_buffered_hard_joint_position_limits",
            "live_isaaclab_soft_joint_position_limits",
            "derived_target_guard_joint_position_limits",
            "guard_source",
            "guard_inset_rad",
            "target_limit_guard",
            "adapter",
        },
        "openpi_droid_position_execution": {
            "query_index",
            "chunk_action_index",
            "outer_step_index",
            "terminated",
            "truncated",
            "processed_joint_position_target",
            "articulation_joint_position_target",
            "processed_finger_position_target",
            "articulation_finger_position_target",
            "target_hold",
            "measured_joint_position_after",
            "measured_joint_velocity_after",
            "measured_closed_positive_gripper_after",
            "environment_after",
        },
        "openpi_droid_position_rollout_end": {
            "outer_steps_completed",
            "query_count",
            "terminal_rollout",
        },
        "openpi_droid_position_rollout_failure": {"terminal_failure"},
    }
    if record_type not in specific or set(value) != common | specific[record_type]:
        raise ValueError("position trace record schema mismatch")
    if (
        value["schema_version"] != PI05_DROID_POSITION_TRACE_SCHEMA_VERSION
        or value["profile"] != PI05_DROID_POSITION_ADAPTER_PROFILE
        or type(value["reset_index"]) is not int
        or value["reset_index"] < 0
    ):
        raise ValueError("position trace record identity mismatch")
    for field in ("serving_contract_sha256", "serving_contract_artifact_sha256"):
        digest = value[field]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"position trace {field} is invalid")

    if record_type == "openpi_droid_position_query":
        if (
            type(value["query_index"]) is not int
            or value["query_index"] < 0
            or not isinstance(value["prompt"], str)
            or value["model_image_order"] != MODEL_IMAGE_ORDER
            or value["normalization_scope"] != "checkpoint_global_droid"
            or value["normalization_asset_id"] != "droid"
            or value["sampler"] != "flow_euler_t1_to_t0_num_steps10_rng_key0_v1"
            or value["response_action_shape"] != [15, 8]
            or value["response_action_dtype"] != "float64"
            or value["execution_horizon"] != 8
        ):
            raise ValueError("position query contract mismatch")
        request_state = value["request_state"]
        if not isinstance(request_state, dict) or set(request_state) != {
            "joint_position",
            "closed_positive_gripper",
        }:
            raise ValueError("position query state schema mismatch")
        if np.asarray(request_state["joint_position"]).shape != (7,) or np.asarray(
            request_state["closed_positive_gripper"]
        ).shape != (1,):
            raise ValueError("position query state shape mismatch")
        images = value["images"]
        if not isinstance(images, dict) or set(images) != {
            "native_external",
            "native_wrist",
            "external",
            "wrist",
            "blank_masked_right_wrist",
            "resize",
            "wrist_rotation_degrees",
        }:
            raise ValueError("position query image schema mismatch")
        if (
            images["resize"] != "openpi_image_tools_resize_with_pad_224_v1"
            or images["wrist_rotation_degrees"] != 0
        ):
            raise ValueError("position query image preprocessing mismatch")
        for name in ("native_external", "native_wrist"):
            _validate_image_identity(
                images[name],
                expected_shape=[720, 1280, 3],
                field=f"position query {name}",
            )
        for name in ("external", "wrist", "blank_masked_right_wrist"):
            _validate_image_identity(
                images[name],
                expected_shape=[224, 224, 3],
                field=f"position query {name}",
            )
        actions = np.asarray(value["response_action_chunk"])
        if actions.shape != (15, 8) or not np.isfinite(actions).all():
            raise ValueError("position query response chunk mismatch")
    elif record_type == "openpi_droid_position_action":
        if (
            type(value["query_index"]) is not int
            or type(value["chunk_action_index"]) is not int
        ):
            raise ValueError("position action indices must be exact integers")
        validate_position_adapter_evidence(value["adapter"])
        live_pre = np.asarray(value["live_pre_step_joint_position"])
        policy_q = np.asarray(value["policy_observation_joint_position"])
        if (
            live_pre.shape != (7,)
            or policy_q.shape != (7,)
            or not np.array_equal(live_pre, policy_q)
            or not np.array_equal(
                live_pre,
                np.asarray(value["adapter"]["measured_joint_position"]),
            )
        ):
            raise ValueError("position action fresh live anchor mismatch")
        limit_contract = expected_position_limit_contract()
        live_hard = _serialized_float32(
            value["live_buffered_hard_joint_position_limits"],
            shape=(7, 2),
            field="position action live hard limits",
        )
        live_soft = _serialized_float32(
            value["live_isaaclab_soft_joint_position_limits"],
            shape=(7, 2),
            field="position action live soft limits",
        )
        reference_target = np.asarray(
            value["adapter"]["absolute_joint_position_target_rad"], dtype=np.float64
        )
        guarded_target, guard, violation = evaluate_position_target_guard(
            reference_target, live_hard, live_soft
        )
        recorded_guard = _serialized_float32(
            value["derived_target_guard_joint_position_limits"],
            shape=(7, 2),
            field="position action derived target guard",
        )
        recorded_target = _serialized_float32(
            value["guarded_float32_joint_position_target"],
            shape=(7,),
            field="position action guarded target",
        )
        if (
            not np.array_equal(recorded_target, guarded_target[0])
            or not np.array_equal(recorded_guard, guard[0])
            or value["position_limit_contract_sha256"]
            != limit_contract["contract_sha256"]
            or value["guard_source"]
            != "intersection(live_joint_pos_limits,live_soft_joint_pos_limits)"
            or value["guard_inset_rad"] != 0.0
            or value["target_limit_guard"]
            != (
                "passed_exact_zero_inset_live_hard_soft_intersection_guard_"
                "before_env_step_and_setter"
            )
            or bool(violation.any())
        ):
            raise ValueError("position action target-limit guard mismatch")
    elif record_type == "openpi_droid_position_execution":
        for field in ("query_index", "chunk_action_index", "outer_step_index"):
            if type(value[field]) is not int or value[field] < 0:
                raise ValueError(f"position execution {field} is invalid")
        if (
            type(value["terminated"]) is not bool
            or type(value["truncated"]) is not bool
        ):
            raise ValueError("position execution flags must be booleans")
        for field, width in (
            ("processed_joint_position_target", 7),
            ("articulation_joint_position_target", 7),
            ("processed_finger_position_target", 1),
            ("articulation_finger_position_target", 1),
            ("measured_joint_position_after", 7),
            ("measured_joint_velocity_after", 7),
            ("measured_closed_positive_gripper_after", 1),
        ):
            array = np.asarray(value[field])
            if array.shape != (width,) or not np.isfinite(array).all():
                raise ValueError(f"position execution {field} mismatch")
        hold = validate_position_target_hold_report(value["target_hold"])
        expected = np.asarray(
            value["articulation_joint_position_target"], dtype=np.float32
        )
        if not np.array_equal(
            np.asarray(hold["absolute_joint_position_target_rad"], dtype=np.float32),
            expected,
        ):
            raise ValueError("position execution hold target mismatch")
        _validate_environment_trace_state(value["environment_after"])
    elif record_type == "openpi_droid_position_rollout_start":
        _validate_environment_trace_state(value["environment_before"])
    elif record_type == "openpi_droid_position_rollout_end":
        if (
            type(value["outer_steps_completed"]) is not int
            or value["outer_steps_completed"] < 0
            or type(value["query_count"]) is not int
            or value["query_count"] < 0
        ):
            raise ValueError("position rollout-end counters are invalid")
        terminal = value["terminal_rollout"]
        if not isinstance(terminal, dict) or set(terminal) != {
            "environment_before",
            "environment_after",
            "outer_steps_completed",
            "rubric",
        }:
            raise ValueError("position terminal rollout schema mismatch")
        _validate_environment_trace_state(terminal["environment_before"])
        _validate_environment_trace_state(terminal["environment_after"])
        rubric = terminal["rubric"]
        if (
            terminal["outer_steps_completed"] != value["outer_steps_completed"]
            or not isinstance(rubric, dict)
            or set(rubric) != {"success", "progress"}
            or type(rubric["success"]) is not bool
            or type(rubric["progress"]) not in (int, float)
            or not np.isfinite(rubric["progress"])
        ):
            raise ValueError("position terminal rollout mismatch")
    elif record_type == "openpi_droid_position_rollout_failure":
        failure = value["terminal_failure"]
        if not isinstance(failure, dict) or set(failure) != {
            "schema_version",
            "profile",
            "failure_type",
            "reason",
            "sample_kind",
            "actions_attempted",
            "outer_steps_completed",
            "incident_artifact",
            "incident",
            "dynamic_report",
            "environment_after_failure",
            "rubric",
        }:
            raise ValueError("position terminal failure schema mismatch")
        if (
            failure["schema_version"] != 1
            or failure["profile"] != "openpi_pi05_droid_position_numerical_failure_v1"
            or failure["failure_type"]
            not in {"NativeAllJointVelocityLimitError", "PositionTargetLimitError"}
            or failure["sample_kind"]
            not in {"apply_entry", "post_policy_step", "pre_setter_target_limit"}
            or failure["rubric"] != {"success": False, "progress": 0.0}
        ):
            raise ValueError("position terminal failure identity mismatch")
        dynamic = validate_native_all_joint_dynamic_report(
            failure["dynamic_report"], require_samples=False
        )
        if failure["failure_type"] == "NativeAllJointVelocityLimitError":
            validate_native_all_joint_velocity_failure(failure["incident"])
            if dynamic["terminal_velocity_failure"] != failure["incident"]:
                raise ValueError("position velocity terminal/dynamic mismatch")
        else:
            validate_position_target_limit_incident(failure["incident"])
            if (
                failure["sample_kind"] != "pre_setter_target_limit"
                or dynamic["terminal_velocity_failure"] is not None
            ):
                raise ValueError("position target-limit terminal mismatch")
        _validate_environment_trace_state(failure["environment_after_failure"])
        artifact = failure["incident_artifact"]
        if (
            not isinstance(artifact, dict)
            or set(artifact) != {"path", "size", "sha256", "mode", "nlink"}
            or artifact["mode"] != "0444"
            or artifact["nlink"] != 1
        ):
            raise ValueError("position terminal incident artifact mismatch")
    return copy.deepcopy(value)


def validate_position_trace(
    records: Any, *, expected_executions: int | None = None
) -> dict[str, Any]:
    """Validate ordering, execute-eight grouping, and fresh-anchor records."""

    if not isinstance(records, list) or len(records) < 2:
        raise ValueError("position trace must be a nonempty record list")
    records = [validate_position_trace_record(record) for record in records]
    if records[0]["record_type"] != "openpi_droid_position_rollout_start":
        raise ValueError("position trace must start with rollout_start")
    if records[-1]["record_type"] != "openpi_droid_position_rollout_end":
        raise ValueError("position trace must end with rollout_end")
    identity = {
        key: records[0][key]
        for key in (
            "profile",
            "serving_contract_sha256",
            "serving_contract_artifact_sha256",
            "reset_index",
        )
    }
    active_query = -1
    next_chunk_index = 0
    pending_action: dict[str, Any] | None = None
    executions = 0
    query_count = 0
    environment_before = records[0]["environment_before"]
    for record in records[1:-1]:
        if any(record[key] != expected for key, expected in identity.items()):
            raise ValueError("position trace identity drift")
        if record["record_type"] == "openpi_droid_position_query":
            if pending_action is not None or next_chunk_index not in (0, 8):
                raise ValueError("position query broke execute-eight grouping")
            active_query += 1
            query_count += 1
            if record["query_index"] != active_query:
                raise ValueError("position query index is not contiguous")
            next_chunk_index = 0
        elif record["record_type"] == "openpi_droid_position_action":
            if pending_action is not None or active_query < 0:
                raise ValueError("position action ordering mismatch")
            if (
                record["query_index"] != active_query
                or record["chunk_action_index"] != next_chunk_index
            ):
                raise ValueError("position action chunk index mismatch")
            pending_action = record
        elif record["record_type"] == "openpi_droid_position_execution":
            if pending_action is None:
                raise ValueError("position execution has no action")
            if (
                record["query_index"] != pending_action["query_index"]
                or record["chunk_action_index"] != pending_action["chunk_action_index"]
                or record["outer_step_index"] != executions
            ):
                raise ValueError("position execution/action identity mismatch")
            target = np.asarray(
                pending_action["adapter"]["absolute_joint_position_target_rad"],
                dtype=np.float32,
            )
            if not np.array_equal(
                np.asarray(record["processed_joint_position_target"], dtype=np.float32),
                target,
            ) or not np.array_equal(
                np.asarray(
                    record["articulation_joint_position_target"], dtype=np.float32
                ),
                target,
            ):
                raise ValueError("position execution differs from adapted target")
            environment_after = record["environment_after"]
            expected_completed = executions + 1
            if (
                record["terminated"]
                or record["truncated"]
                or environment_after["episode_length"] != expected_completed
                or environment_after["sim_step_counter"]
                != environment_before["sim_step_counter"] + expected_completed * 8
                or environment_after["common_step_counter"]
                != environment_before["common_step_counter"] + expected_completed
                or any(
                    environment_after["sensor_frame_counters"][name]
                    != environment_before["sensor_frame_counters"][name]
                    + expected_completed
                    for name in PI05_DROID_NATIVE_SENSOR_NAMES
                )
            ):
                raise ValueError("position execution environment cadence mismatch")
            pending_action = None
            executions += 1
            next_chunk_index += 1
        else:
            raise ValueError("unexpected record inside position rollout")
    if pending_action is not None:
        raise ValueError("position trace ends with an unexecuted action")
    if expected_executions is not None and executions != expected_executions:
        raise ValueError("position trace execution count mismatch")
    end = records[-1]
    if end["outer_steps_completed"] != executions or end["query_count"] != query_count:
        raise ValueError("position rollout-end counters mismatch")
    terminal = end["terminal_rollout"]
    expected_environment_after = (
        environment_before if executions == 0 else records[-2]["environment_after"]
    )
    if (
        terminal["environment_before"] != environment_before
        or terminal["environment_after"] != expected_environment_after
    ):
        raise ValueError("position terminal environment mismatch")
    expected_queries = (executions + 7) // 8
    if query_count != expected_queries:
        raise ValueError("position trace query count does not match execute-eight")
    return {"executions": executions, "queries": query_count, "status": "pass"}


@InferenceClient.register(client_name="DroidDeltaJointPosition")
class DroidDeltaJointPositionClient(InferenceClient):
    """Execute official model commands through fresh DROID position targets."""

    def __init__(self, args: PolicyArgs) -> None:
        self.args = args
        self._validate_args()
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host, port=args.port
        )
        server_metadata = self.client.get_server_metadata()
        self.serving_contract = validate_pi05_droid_position_server_metadata(
            server_metadata
        )
        self.position_limit_contract = validate_position_limit_contract(
            self.serving_contract["control"]["position_limits"]
        )
        if (
            not isinstance(args.serving_contract_path, str)
            or not args.serving_contract_path
        ):
            raise ValueError("DroidDeltaJointPosition requires serving_contract_path")
        self.serving_contract_artifact = validate_persisted_position_serving_contract(
            Path(args.serving_contract_path), server_metadata
        )
        self.client_runtime_attestation = self._validate_client_runtime_origin()
        self.open_loop_horizon = args.open_loop_horizon
        self.pred_action_chunk: np.ndarray | None = None
        self.actions_from_chunk_completed = 0
        self.query_index = 0
        self.active_query_index: int | None = None
        self.reset_index = -1
        self.outer_step_index = 0
        self._pending_execution: dict[str, Any] | None = None
        self._trace_artifact: dict[str, Any] | None = None
        self._environment_runtime_contract: dict[str, Any] | None = None
        self._rollout_environment_before: dict[str, Any] | None = None
        self._last_environment_after: dict[str, Any] | None = None
        self._live_robot: Any | None = None
        self._live_arm_joint_ids: Any | None = None
        self.trace_path = Path(args.trace_path) if args.trace_path else None
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            if self.trace_path.exists() or self.trace_path.is_symlink():
                raise FileExistsError(
                    "official position-adapter canary requires one fresh trace"
                )

        marker = {
            "client": "DroidDeltaJointPosition",
            "profile": PI05_DROID_POSITION_ADAPTER_PROFILE,
            "serving_contract_sha256": self.serving_contract["contract_sha256"],
            "serving_contract_artifact_sha256": self.serving_contract_artifact[
                "sha256"
            ],
            "response_horizon": 15,
            "execution_horizon": 8,
            "fresh_measurement_anchor": True,
            "wrist_rotation_degrees": 0,
            "position_limit_contract_sha256": self.position_limit_contract[
                "contract_sha256"
            ],
        }
        print(
            PI05_DROID_POSITION_CLIENT_MARKER
            + json.dumps(marker, sort_keys=True, separators=(",", ":")),
            flush=True,
        )

    def _validate_args(self) -> None:
        expected = {
            "policy_profile": PI05_DROID_POSITION_ADAPTER_PROFILE,
            "open_loop_horizon": PI05_DROID_EXECUTION_HORIZON,
            "expected_action_horizon": PI05_DROID_RESPONSE_HORIZON,
            "expected_action_dim": 8,
            "state_type": "joint_position",
            "frame_description": "robot base frame",
            "action_frame": "robot_base",
            "dataset_name": "droid",
            "rotate_wrist_180": False,
            "render_every_step": True,
        }
        for field, expected_value in expected.items():
            if getattr(self.args, field, None) != expected_value:
                raise ValueError(
                    f"DroidDeltaJointPosition requires {field}={expected_value!r}"
                )
        if not isinstance(self.args.trace_path, str) or not self.args.trace_path:
            raise ValueError("DroidDeltaJointPosition requires trace_path")

    def _validate_client_runtime_origin(self) -> dict[str, Any]:
        if not isinstance(self.args.openpi_dir, str) or not self.args.openpi_dir:
            raise ValueError("DroidDeltaJointPosition requires openpi_dir")
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
        for module_name, (module, relative_path, expected_sha256) in expected.items():
            module_file = getattr(module, "__file__", None)
            if not isinstance(module_file, str):
                raise ValueError(f"imported {module_name} has no source origin")
            raw_module_path = Path(module_file)
            raw_expected_path = root / relative_path
            if raw_module_path.is_symlink() or raw_expected_path.is_symlink():
                raise ValueError(f"imported {module_name} source must not be a symlink")
            module_path = raw_module_path.resolve()
            expected_path = raw_expected_path.resolve()
            if module_path != expected_path or not module_path.is_file():
                raise ValueError(f"imported {module_name} escaped --openpi-dir")
            digest = hashlib.sha256(module_path.read_bytes()).hexdigest()
            if digest != expected_sha256:
                raise ValueError(f"imported {module_name} source digest mismatch")
            records.append(
                {
                    "module": module_name,
                    "relative_path": relative_path,
                    "sha256": digest,
                }
            )
        records.sort(key=lambda item: item["module"])
        identity = {
            "schema_version": 1,
            "openpi_dir": str(root),
            "git_head": checkout["git_head"],
            "git_tracked_and_untracked_clean": checkout[
                "git_tracked_and_untracked_clean"
            ],
            "modules": records,
        }
        identity["sha256"] = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        return identity

    @property
    def rerender(self) -> bool:
        return (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        )

    def reset(self) -> None:
        if self._environment_runtime_contract is None:
            raise RuntimeError("position evaluation runtime must be bound before reset")
        if self._pending_execution is not None:
            raise RuntimeError("cannot reset with an unrecorded position target")
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.query_index = 0
        self.active_query_index = None
        self.outer_step_index = 0
        self.reset_index += 1
        if self.reset_index != 0:
            raise RuntimeError("official position-adapter canary allows one rollout")

    def bind_evaluation_runtime(self, value: dict[str, Any]) -> None:
        if self._environment_runtime_contract is not None:
            raise RuntimeError("position evaluation runtime is already bound")
        self._environment_runtime_contract = validate_environment_runtime_contract(
            value
        )

    def begin_rollout(self, env: Any) -> dict[str, Any]:
        if self._environment_runtime_contract is None or self.reset_index != 0:
            raise RuntimeError("position rollout runtime/reset is invalid")
        if self._rollout_environment_before is not None:
            raise RuntimeError("position rollout already began")
        before = _capture_environment_state(env, self._environment_runtime_contract)
        if before["episode_length"] != 0:
            raise ValueError("position rollout did not begin at episode step zero")
        self._rollout_environment_before = before
        root = getattr(env, "unwrapped", env)
        robot = root.scene["robot"]
        joint_ids, joint_names = robot.find_joints(
            list(PANDA_ARM_JOINT_NAMES), preserve_order=True
        )
        if tuple(joint_names) != PANDA_ARM_JOINT_NAMES:
            raise ValueError("position live robot joint order mismatch")
        self._live_robot = robot
        self._live_arm_joint_ids = joint_ids
        self._last_environment_after = None
        self._append_trace(
            {
                **self._record("openpi_droid_position_rollout_start"),
                "environment_before": before,
            }
        )
        return copy.deepcopy(before)

    def visualize(self, request: dict) -> np.ndarray:
        current = self._extract_observation(request)
        external, wrist = self._resize_images(current)
        return np.concatenate([external, wrist], axis=1)

    def infer(
        self, obs: dict, instruction: str, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if self._pending_execution is not None:
            raise RuntimeError("previous position target has no execution evidence")
        if self._rollout_environment_before is None:
            raise RuntimeError("position rollout did not begin")
        current = self._extract_observation(obs)
        if self._live_robot is None or self._live_arm_joint_ids is None:
            raise RuntimeError("position live robot was not bound")
        live_q = _tensor_numpy(
            self._live_robot.data.joint_pos[:, self._live_arm_joint_ids],
            field="live pre-step joint position",
        )
        if live_q.dtype != np.float32 or live_q.shape != (1, 7):
            raise ValueError("live pre-step joint readback must be float32 [1,7]")
        if not np.array_equal(live_q[0], current["joint_position"]):
            raise ValueError("policy observation differs from fresh live Panda q")
        visualization = None
        if self.rerender:
            self.actions_from_chunk_completed = 0
            external, wrist = self._resize_images(current)
            request = {
                "observation/exterior_image_1_left": external,
                "observation/wrist_image_left": wrist,
                "observation/joint_position": current["joint_position"],
                "observation/gripper_position": current["gripper_position"],
                "prompt": instruction,
            }
            response = self.client.infer(request)
            self.pred_action_chunk = validate_joint_action_chunk(
                response,
                open_loop_horizon=8,
                expected_action_horizon=15,
                expected_action_dim=8,
            )
            if self.pred_action_chunk.dtype != np.float64:
                raise ValueError("official pi05_droid response must be float64")
            self.active_query_index = self.query_index
            self._trace_query(request, self.pred_action_chunk, current)
            self.query_index += 1
            visualization = np.concatenate([external, wrist], axis=1)
        elif return_viz:
            external, wrist = self._resize_images(current)
            visualization = np.concatenate([external, wrist], axis=1)
        if self.pred_action_chunk is None or self.active_query_index is None:
            raise RuntimeError("no active pi0.5-DROID action chunk")

        action_index = self.actions_from_chunk_completed
        adapter = adapt_official_droid_action(
            self.pred_action_chunk[action_index].copy(), live_q[0]
        )
        live_hard_limits = _tensor_numpy(
            self._live_robot.data.joint_pos_limits[:, self._live_arm_joint_ids],
            field="live buffered hard joint-position limits",
        )
        live_soft_limits = _tensor_numpy(
            self._live_robot.data.soft_joint_pos_limits[:, self._live_arm_joint_ids],
            field="live Isaac Lab soft joint-position limits",
        )
        if (
            live_hard_limits.dtype != np.float32
            or live_hard_limits.shape != (1, 7, 2)
            or live_soft_limits.dtype != np.float32
            or live_soft_limits.shape != (1, 7, 2)
        ):
            raise ValueError("live hard/soft joint limits must be float32 [1,7,2]")
        reference_target = np.asarray(
            adapter["absolute_joint_position_target_rad"], dtype=np.float64
        )
        guarded_target, target_guard_limits, violation = evaluate_position_target_guard(
            reference_target, live_hard_limits, live_soft_limits
        )
        if bool(violation.any()):
            incident = validate_position_target_limit_incident(
                {
                    "schema_version": 1,
                    "profile": ("openpi_pi05_droid_position_target_limit_failure_v1"),
                    "reason": (
                        "absolute_target_outside_zero_inset_live_hard_soft_guard"
                    ),
                    "outer_step_index": self.outer_step_index,
                    "query_index": self.active_query_index,
                    "chunk_action_index": action_index,
                    "measured_joint_position": live_q[0].tolist(),
                    "clipped_arm_command": adapter["clipped_action"][:7],
                    "reference_float64_absolute_joint_position_target": (
                        reference_target.tolist()
                    ),
                    "guarded_float32_joint_position_target": (
                        guarded_target[0].tolist()
                    ),
                    "position_limit_contract_sha256": self.position_limit_contract[
                        "contract_sha256"
                    ],
                    "live_buffered_hard_joint_position_limits": (
                        live_hard_limits[0].tolist()
                    ),
                    "live_isaaclab_soft_joint_position_limits": (
                        live_soft_limits[0].tolist()
                    ),
                    "derived_target_guard_joint_position_limits": (
                        target_guard_limits[0].tolist()
                    ),
                    "guard_source": (
                        "intersection(live_joint_pos_limits,live_soft_joint_pos_limits)"
                    ),
                    "guard_inset_rad": 0.0,
                    "violating_joint_indices": np.flatnonzero(violation[0]).tolist(),
                    "setter_calls_for_rejected_target": 0,
                }
            )
            if self.trace_path is None:
                raise RuntimeError("position target-limit incident has no trace root")
            artifact = publish_immutable_json(
                self.trace_path.parent
                / "position_target_failures"
                / "episode_000000.json",
                incident,
            )
            identity = {
                key: artifact[key]
                for key in ("path", "size", "sha256", "mode", "nlink")
            }
            raise PositionTargetLimitError(incident, identity)
        emitted = np.concatenate(
            [
                guarded_target[0].astype(np.float64),
                np.asarray(
                    [adapter["absolute_closed_positive_gripper"]], dtype=np.float64
                ),
            ]
        )
        self._pending_execution = {
            "query_index": self.active_query_index,
            "chunk_action_index": action_index,
            "adapter": adapter,
        }
        self.actions_from_chunk_completed += 1
        self._append_trace(
            {
                **self._record("openpi_droid_position_action"),
                "query_index": self.active_query_index,
                "chunk_action_index": action_index,
                "live_pre_step_joint_position": live_q[0].tolist(),
                "policy_observation_joint_position": current["joint_position"].tolist(),
                "guarded_float32_joint_position_target": guarded_target[0].tolist(),
                "position_limit_contract_sha256": self.position_limit_contract[
                    "contract_sha256"
                ],
                "live_buffered_hard_joint_position_limits": (
                    live_hard_limits[0].tolist()
                ),
                "live_isaaclab_soft_joint_position_limits": (
                    live_soft_limits[0].tolist()
                ),
                "derived_target_guard_joint_position_limits": (
                    target_guard_limits[0].tolist()
                ),
                "guard_source": (
                    "intersection(live_joint_pos_limits,live_soft_joint_pos_limits)"
                ),
                "guard_inset_rad": 0.0,
                "target_limit_guard": (
                    "passed_exact_zero_inset_live_hard_soft_intersection_guard_"
                    "before_env_step_and_setter"
                ),
                "adapter": adapter,
            }
        )
        return emitted, visualization

    def record_execution(
        self, obs: dict, env: Any, *, terminated: Any, truncated: Any
    ) -> dict[str, Any]:
        if self._pending_execution is None:
            raise RuntimeError("no pending position target to record")
        pending = self._pending_execution
        if (
            self._environment_runtime_contract is None
            or self._rollout_environment_before is None
        ):
            raise RuntimeError("position environment cadence was not bound")
        flags = validate_outer_step_flags(
            terminated,
            truncated,
            outer_step_index=self.outer_step_index,
        )
        terminated_bool = flags["terminated"]
        truncated_bool = flags["truncated"]
        environment_after = _capture_environment_state(
            env, self._environment_runtime_contract
        )
        completed = self.outer_step_index + 1
        before = self._rollout_environment_before
        if (
            environment_after["episode_length"] != completed
            or environment_after["sim_step_counter"]
            != before["sim_step_counter"] + completed * PI05_DROID_NATIVE_DECIMATION
            or environment_after["common_step_counter"]
            != before["common_step_counter"] + completed
            or any(
                environment_after["sensor_frame_counters"][name]
                != before["sensor_frame_counters"][name] + completed
                for name in PI05_DROID_NATIVE_SENSOR_NAMES
            )
        ):
            raise ValueError("position environment/camera cadence mismatch")
        root = getattr(env, "unwrapped", env)
        arm_term = root.action_manager._terms["arm"]
        finger_term = root.action_manager._terms["finger_joint"]
        robot = root.scene["robot"]
        joint_ids, joint_names = robot.find_joints(
            list(PANDA_ARM_JOINT_NAMES), preserve_order=True
        )
        if tuple(joint_names) != PANDA_ARM_JOINT_NAMES:
            raise ValueError(f"live Panda joint order mismatch: {joint_names}")
        finger_ids, finger_names = robot.find_joints(
            ["finger_joint"], preserve_order=True
        )
        if finger_names != ["finger_joint"]:
            raise ValueError("live finger joint mismatch")

        processed = _tensor_numpy(
            arm_term.processed_actions, field="processed joint-position target"
        )
        target = _tensor_numpy(
            robot.data.joint_pos_target[:, joint_ids],
            field="articulation joint-position target",
        )
        expected = np.asarray(
            pending["adapter"]["absolute_joint_position_target_rad"],
            dtype=np.float32,
        )[None, :]
        if (
            processed.dtype != np.float32
            or target.dtype != np.float32
            or processed.shape != (1, 7)
            or target.shape != (1, 7)
            or not np.array_equal(processed, expected)
            or not np.array_equal(target, expected)
        ):
            raise ValueError("live absolute joint-position target mismatch")

        processed_finger = _tensor_numpy(
            finger_term.processed_actions, field="processed finger target"
        )
        finger_target = _tensor_numpy(
            robot.data.joint_pos_target[:, finger_ids], field="finger target"
        )
        closed = pending["adapter"]["absolute_closed_positive_gripper"]
        expected_finger = np.asarray(
            [[np.float32(np.pi / 4.0) if closed == 1.0 else np.float32(0.0)]],
            dtype=np.float32,
        )
        if (
            processed_finger.dtype != np.float32
            or finger_target.dtype != np.float32
            or not np.array_equal(processed_finger, expected_finger)
            or not np.array_equal(finger_target, expected_finger)
        ):
            raise ValueError("live absolute closed-positive gripper target mismatch")
        hold = validate_position_target_hold_report(
            arm_term.consume_position_target_hold_report()
        )
        current = self._extract_observation(obs)
        record = {
            **self._record("openpi_droid_position_execution"),
            "query_index": pending["query_index"],
            "chunk_action_index": pending["chunk_action_index"],
            "outer_step_index": self.outer_step_index,
            "terminated": terminated_bool,
            "truncated": truncated_bool,
            "processed_joint_position_target": processed[0].tolist(),
            "articulation_joint_position_target": target[0].tolist(),
            "processed_finger_position_target": processed_finger[0].tolist(),
            "articulation_finger_position_target": finger_target[0].tolist(),
            "target_hold": hold,
            "measured_joint_position_after": current["joint_position"].tolist(),
            "measured_joint_velocity_after": current["joint_velocity"].tolist(),
            "measured_closed_positive_gripper_after": current[
                "gripper_position"
            ].tolist(),
            "environment_after": environment_after,
        }
        self._append_trace(record)
        self._pending_execution = None
        self._last_environment_after = environment_after
        self.outer_step_index += 1
        return record

    def record_execution_failure(
        self,
        error: BaseException,
        env: Any,
        dynamic_report: dict[str, Any],
    ) -> dict[str, Any]:
        """Seal the distinct position-profile all-DOF numerical terminal."""

        if type(error) is not NativeAllJointVelocityLimitError:
            raise TypeError("position numerical terminal requires exact monitor error")
        if error.incident_artifact is None:
            raise RuntimeError("position numerical incident was not durable")
        incident = validate_native_all_joint_velocity_failure(error.evidence)
        dynamic = validate_native_all_joint_dynamic_report(
            dynamic_report, require_samples=False
        )
        if dynamic["terminal_velocity_failure"] != incident:
            raise ValueError("position dynamic report differs from incident")
        sample_kind = incident["sample_kind"]
        if sample_kind == "apply_entry":
            if self._pending_execution is None:
                raise RuntimeError("position apply-entry failure has no pending action")
            actions_attempted = self.outer_step_index + 1
        else:
            if self._pending_execution is not None:
                raise RuntimeError(
                    "position post-policy failure retained pending action"
                )
            actions_attempted = self.outer_step_index
        if self._environment_runtime_contract is None:
            raise RuntimeError("position failure has no environment runtime")
        environment_after_failure = _capture_environment_state(
            env, self._environment_runtime_contract
        )
        terminal = {
            "schema_version": 1,
            "profile": "openpi_pi05_droid_position_numerical_failure_v1",
            "failure_type": "NativeAllJointVelocityLimitError",
            "reason": f"{type(error).__name__}: {error}",
            "sample_kind": sample_kind,
            "actions_attempted": actions_attempted,
            "outer_steps_completed": self.outer_step_index,
            "incident_artifact": copy.deepcopy(error.incident_artifact),
            "incident": incident,
            "dynamic_report": dynamic,
            "environment_after_failure": environment_after_failure,
            "rubric": {"success": False, "progress": 0.0},
        }
        self._append_trace(
            {
                **self._record("openpi_droid_position_rollout_failure"),
                "terminal_failure": terminal,
            }
        )
        self._trace_artifact = self._seal_trace()
        self._pending_execution = None
        return terminal

    def record_target_limit_failure(
        self,
        error: BaseException,
        env: Any,
        dynamic_report: dict[str, Any],
    ) -> dict[str, Any]:
        if type(error) is not PositionTargetLimitError:
            raise TypeError("target-limit terminal requires exact typed error")
        incident = validate_position_target_limit_incident(error.evidence)
        dynamic = validate_native_all_joint_dynamic_report(
            dynamic_report, require_samples=False
        )
        if dynamic["terminal_velocity_failure"] is not None:
            raise ValueError("target-limit terminal contains velocity failure")
        if self._environment_runtime_contract is None:
            raise RuntimeError("target-limit terminal has no runtime")
        terminal = {
            "schema_version": 1,
            "profile": "openpi_pi05_droid_position_numerical_failure_v1",
            "failure_type": "PositionTargetLimitError",
            "reason": f"{type(error).__name__}: {error}",
            "sample_kind": "pre_setter_target_limit",
            "actions_attempted": self.outer_step_index + 1,
            "outer_steps_completed": self.outer_step_index,
            "incident_artifact": copy.deepcopy(error.incident_artifact),
            "incident": incident,
            "dynamic_report": dynamic,
            "environment_after_failure": _capture_environment_state(
                env, self._environment_runtime_contract
            ),
            "rubric": {"success": False, "progress": 0.0},
        }
        self._append_trace(
            {
                **self._record("openpi_droid_position_rollout_failure"),
                "terminal_failure": terminal,
            }
        )
        self._trace_artifact = self._seal_trace()
        return terminal

    def finish_rollout(self, env: Any, rubric: Any) -> dict[str, Any]:
        if self._pending_execution is not None:
            raise RuntimeError("rollout ended with an unrecorded position target")
        if (
            self.outer_step_index != PI05_DROID_NATIVE_EPISODE_STEPS
            or self._environment_runtime_contract is None
            or self._rollout_environment_before is None
            or self._last_environment_after is None
        ):
            raise RuntimeError("position rollout is not one complete 450-step episode")
        current_environment = _capture_environment_state(
            env, self._environment_runtime_contract
        )
        if current_environment != self._last_environment_after:
            raise ValueError("position terminal environment changed after execution")
        if not isinstance(rubric, dict) or type(rubric.get("success")) is not bool:
            raise ValueError("position terminal rubric schema mismatch")
        progress = rubric.get("progress")
        try:
            progress = progress.item()
        except AttributeError:
            pass
        if (
            type(progress) not in (int, float)
            or isinstance(progress, bool)
            or not np.isfinite(progress)
        ):
            raise ValueError("position terminal progress is not finite")
        terminal_rollout = {
            "environment_before": self._rollout_environment_before,
            "environment_after": current_environment,
            "outer_steps_completed": self.outer_step_index,
            "rubric": {
                "success": rubric["success"],
                "progress": float(progress),
            },
        }
        end = {
            **self._record("openpi_droid_position_rollout_end"),
            "outer_steps_completed": self.outer_step_index,
            "query_count": self.query_index,
            "terminal_rollout": terminal_rollout,
        }
        self._append_trace(end)
        self._trace_artifact = self._seal_trace()
        return terminal_rollout

    @property
    def finalized_trace_artifact(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._trace_artifact)

    def _seal_trace(self) -> dict[str, Any]:
        if self.trace_path is None or not self.trace_path.is_file():
            raise RuntimeError("position trace is missing")
        records = [
            json.loads(line)
            for line in self.trace_path.read_text(encoding="ascii").splitlines()
            if line
        ]
        if records[-1]["record_type"] == "openpi_droid_position_rollout_failure":
            for record in records:
                validate_position_trace_record(record)
            if records[0]["record_type"] != "openpi_droid_position_rollout_start":
                raise ValueError("position failure trace has no rollout start")
            summary = {
                "executions": self.outer_step_index,
                "queries": self.query_index,
                "status": "numerical_failure",
            }
        else:
            summary = validate_position_trace(
                records, expected_executions=self.outer_step_index
            )
        os.chmod(self.trace_path, 0o444)
        with self.trace_path.open("rb") as source:
            os.fsync(source.fileno())
        stat_result = self.trace_path.stat()
        payload = self.trace_path.read_bytes()
        return {
            "path": str(self.trace_path.resolve()),
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "mode": "0444",
            "nlink": stat_result.st_nlink,
            "summary": summary,
        }

    def _record(self, record_type: str) -> dict[str, Any]:
        return {
            "schema_version": PI05_DROID_POSITION_TRACE_SCHEMA_VERSION,
            "record_type": record_type,
            "profile": PI05_DROID_POSITION_ADAPTER_PROFILE,
            "serving_contract_sha256": self.serving_contract["contract_sha256"],
            "serving_contract_artifact_sha256": self.serving_contract_artifact[
                "sha256"
            ],
            "reset_index": self.reset_index,
        }

    def _trace_query(
        self,
        request: dict[str, Any],
        actions: np.ndarray,
        current: dict[str, np.ndarray | dict[str, Any]],
    ) -> None:
        blank = np.zeros((224, 224, 3), dtype=np.uint8)
        normalization = self.serving_contract["normalization"]
        openpi = self.serving_contract["openpi"]
        self._append_trace(
            {
                **self._record("openpi_droid_position_query"),
                "query_index": self.query_index,
                "prompt": request["prompt"],
                "request_state": {
                    "joint_position": np.asarray(
                        request["observation/joint_position"]
                    ).tolist(),
                    "closed_positive_gripper": np.asarray(
                        request["observation/gripper_position"]
                    ).tolist(),
                },
                "images": {
                    "native_external": copy.deepcopy(
                        current["external_image_identity"]
                    ),
                    "native_wrist": copy.deepcopy(current["wrist_image_identity"]),
                    "external": _image_contract(
                        request["observation/exterior_image_1_left"]
                    ),
                    "wrist": _image_contract(request["observation/wrist_image_left"]),
                    "blank_masked_right_wrist": _image_contract(blank),
                    "resize": "openpi_image_tools_resize_with_pad_224_v1",
                    "wrist_rotation_degrees": 0,
                },
                "model_image_order": list(MODEL_IMAGE_ORDER),
                "normalization_scope": normalization["scope"],
                "normalization_asset_id": normalization["asset_id"],
                "normalization_sha256": normalization["sha256"],
                "sampler": openpi["sampler"],
                "response_action_shape": list(actions.shape),
                "response_action_dtype": str(actions.dtype),
                "response_action_chunk": actions.tolist(),
                "execution_horizon": self.open_loop_horizon,
            }
        )

    def _append_trace(self, record: dict[str, Any]) -> None:
        if self.trace_path is None:
            return
        validate_position_trace_record(record)
        payload = canonical_json_bytes(record) + b"\n"
        with self.trace_path.open("ab") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())

    @staticmethod
    def _resize_images(current: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        external = image_tools.resize_with_pad(current["external_image"], 224, 224)
        wrist = image_tools.resize_with_pad(current["wrist_image"], 224, 224)
        external = np.asarray(external)
        wrist = np.asarray(wrist)
        _image_contract(external)
        _image_contract(wrist)
        return external, wrist

    @staticmethod
    def _extract_observation(obs: dict[str, Any]) -> dict[str, Any]:
        external = np.asarray(obs["splat"]["external_cam"])
        wrist = np.asarray(obs["splat"]["wrist_cam"])
        external_identity = _native_image_contract(
            external, field="external camera observation"
        )
        wrist_identity = _native_image_contract(wrist, field="wrist camera observation")
        state = obs["policy"]
        joint_position = _tensor_numpy(
            state["arm_joint_pos"], field="joint-position observation"
        )[0]
        joint_velocity = _tensor_numpy(
            state["arm_joint_vel"], field="joint-velocity observation"
        )[0]
        gripper = _tensor_numpy(state["gripper_pos"], field="gripper observation")[0]
        if (
            joint_position.shape != (7,)
            or joint_velocity.shape != (7,)
            or joint_position.dtype != np.float32
            or joint_velocity.dtype != np.float32
        ):
            raise ValueError("official DROID state requires seven ordered Panda joints")
        if gripper.shape != (1,) or gripper.dtype != np.float32:
            raise ValueError("official DROID state requires one float32 gripper value")
        if not (
            np.isfinite(joint_position).all()
            and np.isfinite(joint_velocity).all()
            and np.isfinite(gripper).all()
        ):
            raise JointPositionObservationNumericalError(
                "official DROID proprioception contains non-finite values"
            )
        tolerance = PI05_DROID_GRIPPER_OBSERVATION_BOUND_TOLERANCE
        if gripper[0] < -tolerance or gripper[0] > 1.0 + tolerance:
            raise JointPositionObservationNumericalError(
                "closed-positive gripper escaped official [0, 1] domain"
            )
        return {
            "external_image": external,
            "wrist_image": wrist,
            "external_image_identity": external_identity,
            "wrist_image_identity": wrist_identity,
            "joint_position": joint_position,
            "joint_velocity": joint_velocity,
            "gripper_position": gripper,
        }


__all__ = [
    "DroidDeltaJointPositionClient",
    "MODEL_IMAGE_ORDER",
    "validate_position_trace",
    "validate_position_trace_record",
]
