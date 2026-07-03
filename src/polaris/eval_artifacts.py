"""Transactional PolaRiS rollout artifacts and fail-closed resume checks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


EVAL_RESULT_COLUMNS = (
    "episode",
    "episode_length",
    "success",
    "progress",
    "numerical_failure",
    "numerical_failure_reason",
)
EGO_LAP_TRACE_SCHEMA_VERSION = 2
EGO_LAP_TRACE_PROFILE = "ego_lap_eef_pose_runtime_trace_v2"
EGO_LAP_ENVIRONMENT_RUNTIME_PROFILE = "ego_lap_eef_outer450_internal451_no_autoreset_v1"
EGO_LAP_TERMINAL_ROLLOUT_PROFILE = "ego_lap_eef_terminal_rollout_v1"
EGO_LAP_ENVIRONMENT_STATE_PROFILE = (
    "isaaclab_single_env_episode_sim_common_camera_counters_v1"
)
EGO_LAP_CAMERA_SENSOR_NAMES = ("external_cam", "wrist_cam")
TRACE_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
TRACE_R6_ROWS_LAYOUT = "xyz+r6_first_two_rows+gripper_open"
TRACE_R6_COLUMNS_LAYOUT = "xyz+r6_first_two_columns+gripper_open"
TRACE_R6_ROWS_MODE = "public_lap_train_matched_rows_v1"
TRACE_R6_COLUMNS_MODE = "manifest_train_matched_columns_v1"
TRACE_AR_INTERPOLATION_PROFILE = (
    "so3_right_multiply_slerp_identity_to_delta_inclusive_0_1_8_v1"
)
TRACE_ENVIRONMENT_STATE_FIELDS = {
    "profile",
    "live_max_episode_length",
    "episode_length",
    "sim_step_counter",
    "common_step_counter",
    "sensor_frame_counters",
}
TRACE_TRANSITION_FIELDS = {
    "step_index",
    "terminated",
    "truncated",
    "environment_before",
    "environment_after",
    "counter_deltas",
    "camera_frame_deltas",
}
TRACE_COUNTER_DELTA_FIELDS = {
    "episode_length",
    "sim_step_counter",
    "common_step_counter",
}
TRACE_TERMINAL_ROLLOUT_FIELDS = {
    "schema_version",
    "profile",
    "environment_runtime_profile",
    "episode_index",
    "expected_outer_steps",
    "actions_attempted",
    "outer_steps_completed",
    "last_outer_step_index",
    "terminated_false_count",
    "truncated_false_count",
    "environment_before",
    "environment_after",
    "counter_deltas",
    "camera_frame_deltas",
    "episode_result",
}
TRACE_COMMON_FIELDS = {
    "schema_version",
    "trace_profile",
    "timestamp",
    "event",
    "episode",
}
TRACE_QUERY_FIELDS = TRACE_COMMON_FIELDS | {
    "query",
    "step",
    "instruction",
    "checkpoint_profile",
    "checkpoint_path",
    "contract_sha256",
    "policy_type",
    "response_semantics",
    "execution_horizon",
    "ar_endpoint_interpolation_profile",
    "ar_endpoint_interpolation_steps",
    "gripper_execution_profile",
    "gripper_threshold",
    "action_sampler_profile",
    "flow_num_steps",
    "initial_rng_seed",
    "ar_max_decoding_steps",
    "ar_temperature",
    "ar_stop_at_eos",
    "frame_description",
    "eef_frame",
    "numeric_action_frame",
    "normalization_scope",
    "normalization_stats_sha256",
    "normalization_profile",
    "normalization_compute_dtype",
    "normalization_input_formula",
    "normalization_output_formula",
    "normalization_formula_probe_sha256",
    "state_layout",
    "state_layout_mode",
    "polaris_profile",
    "anchor_position",
    "anchor_quaternion_wxyz",
    "state",
    "server_delta_chunk",
    "raw_delta_chunk",
    "base_delta_chunk",
    "anchored_action_chunk",
    "reasoning",
}
TRACE_ACTION_FIELDS = TRACE_COMMON_FIELDS | {
    "query",
    "step",
    "chunk_index",
    "raw_delta",
    "polaris_action",
}
TRACE_EXECUTION_FIELDS = TRACE_COMMON_FIELDS | {
    "query",
    "step",
    "chunk_index",
    "transition",
}
TRACE_EXECUTION_FAILURE_FIELDS = TRACE_COMMON_FIELDS | {
    "query",
    "step",
    "chunk_index",
    "numerical_failure_reason",
}
TRACE_COMPLETE_FIELDS = TRACE_COMMON_FIELDS | {
    *EVAL_RESULT_COLUMNS,
    "status",
    "terminal_rollout",
}
TRACE_QUERY_STATIC_IDENTITY_FIELDS = {
    "instruction",
    "checkpoint_profile",
    "checkpoint_path",
    "contract_sha256",
    "policy_type",
    "response_semantics",
    "execution_horizon",
    "ar_endpoint_interpolation_profile",
    "ar_endpoint_interpolation_steps",
    "gripper_execution_profile",
    "gripper_threshold",
    "action_sampler_profile",
    "flow_num_steps",
    "initial_rng_seed",
    "ar_max_decoding_steps",
    "ar_temperature",
    "ar_stop_at_eos",
    "frame_description",
    "eef_frame",
    "numeric_action_frame",
    "normalization_scope",
    "normalization_stats_sha256",
    "normalization_profile",
    "normalization_compute_dtype",
    "normalization_input_formula",
    "normalization_output_formula",
    "normalization_formula_probe_sha256",
    "state_layout",
    "state_layout_mode",
    "polaris_profile",
}


def empty_eval_results() -> pd.DataFrame:
    """Return the canonical empty episode table."""

    return pd.DataFrame(
        {
            "episode": pd.Series(dtype="int64"),
            "episode_length": pd.Series(dtype="int64"),
            "success": pd.Series(dtype="bool"),
            "progress": pd.Series(dtype="float64"),
            "numerical_failure": pd.Series(dtype="bool"),
            "numerical_failure_reason": pd.Series(dtype="str"),
        }
    )


def probe_episode_video(path: Path) -> dict[str, int]:
    """Decode an entire rollout video and return its structural identity."""

    import mediapy  # noqa: PLC0415 - simulator runtime dependency

    try:
        frames = np.asarray(mediapy.read_video(path))
    except Exception as error:
        raise ValueError(f"Rollout video is not decodable: {path}: {error}") from error
    if frames.ndim != 4 or frames.shape[0] < 1 or frames.shape[-1] < 3:
        raise ValueError(
            f"Rollout video must decode as T x H x W x C>=3; got {frames.shape}: {path}"
        )
    return {
        "frame_count": int(frames.shape[0]),
        "height": int(frames.shape[1]),
        "width": int(frames.shape[2]),
    }


def validate_episode_video(
    path: Path,
    *,
    expected_frames: int,
    probe_fn: Callable[[Path], Mapping[str, Any]] = probe_episode_video,
) -> dict[str, int]:
    """Require one nonempty, decodable 448x224 video matching its CSV row."""

    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing nonempty completed rollout video: {path}")
    probe = probe_fn(path)
    expected = {"frame_count": expected_frames, "height": 224, "width": 448}
    for key, expected_value in expected.items():
        if probe.get(key) != expected_value:
            raise ValueError(
                f"Completed rollout video {key} mismatch for {path}: "
                f"expected={expected_value!r}, actual={probe.get(key)!r}"
            )
    return expected


def sha256_file(path: Path) -> str:
    """Return the SHA-256 identity of one durable rollout artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_episode_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and strictly validate the row shared by CSV/trace/sidecar."""

    required = set(EVAL_RESULT_COLUMNS)
    if set(result) != required:
        raise ValueError(
            "Episode result must have exactly the canonical columns: "
            f"expected={sorted(required)!r}, actual={sorted(result)!r}"
        )
    episode_value = result["episode"]
    length_value = result["episode_length"]
    if (
        not isinstance(episode_value, numbers.Integral)
        or isinstance(episode_value, (bool, np.bool_))
        or int(episode_value) < 0
    ):
        raise ValueError(f"Episode result has invalid episode: {episode_value!r}")
    if (
        not isinstance(length_value, numbers.Integral)
        or isinstance(length_value, (bool, np.bool_))
        or int(length_value) < 1
    ):
        raise ValueError(f"Episode result has invalid length: {length_value!r}")
    success_value = result["success"]
    failure_value = result["numerical_failure"]
    if not isinstance(success_value, (bool, np.bool_)) or not isinstance(
        failure_value, (bool, np.bool_)
    ):
        raise ValueError("Episode success/failure fields must be booleans")
    progress_value = result["progress"]
    if not isinstance(progress_value, numbers.Real) or isinstance(
        progress_value, (bool, np.bool_)
    ):
        raise ValueError(f"Episode progress must be numeric: {progress_value!r}")
    progress = float(progress_value)
    if not math.isfinite(progress):
        raise ValueError(f"Episode progress must be finite: {progress!r}")
    if not 0.0 <= progress <= 1.0:
        raise ValueError(f"Episode progress must be in [0, 1]: {progress!r}")
    reason = result["numerical_failure_reason"]
    if not isinstance(reason, str):
        raise ValueError(
            f"Episode numerical failure reason must be a string: {reason!r}"
        )
    numerical_failure = bool(failure_value)
    if numerical_failure != bool(reason):
        raise ValueError(
            "Episode numerical failure flag and reason disagree: "
            f"flag={numerical_failure!r}, reason={reason!r}"
        )
    if numerical_failure and bool(success_value):
        raise ValueError("A numerical-failure episode cannot be successful")
    if numerical_failure and progress != 0.0:
        raise ValueError("A numerical-failure episode must have progress=0.0")
    return {
        "episode": int(episode_value),
        "episode_length": int(length_value),
        "success": bool(success_value),
        "progress": progress,
        "numerical_failure": numerical_failure,
        "numerical_failure_reason": reason,
    }


def _read_trace_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing nonempty completed policy trace: {path}")
    records: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            record = json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant {value!r}")
                ),
            )
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number} is not an object")
            records.append(record)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Policy trace is not valid JSONL: {path}: {error}") from error
    if not records:
        raise ValueError(f"Policy trace contains no records: {path}")
    return records


def _validate_trace_environment_state(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != TRACE_ENVIRONMENT_STATE_FIELDS:
        raise ValueError(f"{field} environment-state schema drift")
    if value.get("profile") != EGO_LAP_ENVIRONMENT_STATE_PROFILE:
        raise ValueError(f"{field} environment-state profile drift")
    if value.get("live_max_episode_length") != 451:
        raise ValueError(f"{field} environment-state live horizon drift")
    for name in ("episode_length", "sim_step_counter", "common_step_counter"):
        if type(value.get(name)) is not int or value[name] < 0:
            raise ValueError(f"{field} environment-state {name} drift")
    frames = value.get("sensor_frame_counters")
    if not isinstance(frames, Mapping) or set(frames) != set(
        EGO_LAP_CAMERA_SENSOR_NAMES
    ):
        raise ValueError(f"{field} camera-frame schema drift")
    if any(type(item) is not int or item < 0 for item in frames.values()):
        raise ValueError(f"{field} camera-frame value drift")
    return dict(value)


def _validate_trace_transition(
    value: Any,
    *,
    step: int,
    expected_before: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != TRACE_TRANSITION_FIELDS:
        raise ValueError(f"Policy trace transition schema drift at step {step}: {path}")
    if (
        type(value.get("step_index")) is not int
        or value.get("step_index") != step
        or value.get("terminated") is not False
        or value.get("truncated") is not False
    ):
        raise ValueError(f"Policy trace transition identity/flags drift: {path}")
    before = _validate_trace_environment_state(
        value.get("environment_before"), field=f"transition {step} before"
    )
    after = _validate_trace_environment_state(
        value.get("environment_after"), field=f"transition {step} after"
    )
    if before != dict(expected_before):
        raise ValueError(f"Policy trace environment transition chain drift: {path}")
    if before["episode_length"] != step or after["episode_length"] != step + 1:
        raise ValueError(f"Policy trace episode counter cadence drift: {path}")
    expected_counter_deltas = {
        "episode_length": 1,
        "sim_step_counter": 8,
        "common_step_counter": 1,
    }
    counter_deltas = value.get("counter_deltas")
    if (
        not isinstance(counter_deltas, Mapping)
        or set(counter_deltas) != TRACE_COUNTER_DELTA_FIELDS
        or any(type(item) is not int for item in counter_deltas.values())
        or dict(counter_deltas) != expected_counter_deltas
    ):
        raise ValueError(f"Policy trace transition counter-delta drift: {path}")
    actual_counter_deltas = {
        name: after[name] - before[name] for name in TRACE_COUNTER_DELTA_FIELDS
    }
    if actual_counter_deltas != expected_counter_deltas:
        raise ValueError(f"Policy trace transition counter snapshot drift: {path}")
    expected_camera_deltas = {name: 1 for name in EGO_LAP_CAMERA_SENSOR_NAMES}
    camera_deltas = value.get("camera_frame_deltas")
    if (
        not isinstance(camera_deltas, Mapping)
        or set(camera_deltas) != set(EGO_LAP_CAMERA_SENSOR_NAMES)
        or any(type(item) is not int for item in camera_deltas.values())
        or dict(camera_deltas) != expected_camera_deltas
    ):
        raise ValueError(f"Policy trace transition camera-delta drift: {path}")
    actual_camera_deltas = {
        name: (
            after["sensor_frame_counters"][name] - before["sensor_frame_counters"][name]
        )
        for name in EGO_LAP_CAMERA_SENSOR_NAMES
    }
    if actual_camera_deltas != expected_camera_deltas:
        raise ValueError(f"Policy trace transition camera snapshot drift: {path}")
    return after


def _validate_trace_terminal_rollout(
    value: Any,
    *,
    result: Mapping[str, Any],
    initial_environment: Mapping[str, Any],
    final_environment: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != TRACE_TERMINAL_ROLLOUT_FIELDS:
        raise ValueError(f"Policy trace terminal rollout schema drift: {path}")
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("profile") != EGO_LAP_TERMINAL_ROLLOUT_PROFILE
        or value.get("environment_runtime_profile")
        != EGO_LAP_ENVIRONMENT_RUNTIME_PROFILE
        or type(value.get("episode_index")) is not int
        or value.get("episode_index") != result["episode"]
        or value.get("expected_outer_steps") != 450
        or value.get("episode_result") != dict(result)
    ):
        raise ValueError(f"Policy trace terminal rollout identity drift: {path}")
    completed = result["episode_length"] - int(result["numerical_failure"])
    last_step = value.get("last_outer_step_index")
    if (
        type(value.get("actions_attempted")) is not int
        or value.get("actions_attempted") != result["episode_length"]
        or type(value.get("outer_steps_completed")) is not int
        or value.get("outer_steps_completed") != completed
        or (
            completed == 0
            and last_step is not None
            or completed > 0
            and (type(last_step) is not int or last_step != completed - 1)
        )
        or type(value.get("terminated_false_count")) is not int
        or value.get("terminated_false_count") != completed
        or type(value.get("truncated_false_count")) is not int
        or value.get("truncated_false_count") != completed
    ):
        raise ValueError(f"Policy trace terminal rollout cadence drift: {path}")
    if not result["numerical_failure"] and completed != 450:
        raise ValueError(f"Policy trace completed terminal horizon drift: {path}")
    terminal_before = _validate_trace_environment_state(
        value.get("environment_before"), field="terminal before"
    )
    terminal_after = _validate_trace_environment_state(
        value.get("environment_after"), field="terminal after"
    )
    if terminal_before != dict(initial_environment):
        raise ValueError(f"Policy trace terminal environment-chain drift: {path}")
    if result["numerical_failure"]:
        if (
            any(
                terminal_after[field] != final_environment[field]
                for field in ("episode_length", "common_step_counter")
            )
            or terminal_after["sensor_frame_counters"]
            != final_environment["sensor_frame_counters"]
        ):
            raise ValueError(
                f"Policy trace failure advanced a completed-step counter: {path}"
            )
        sim_tail = (
            terminal_after["sim_step_counter"] - final_environment["sim_step_counter"]
        )
        if not 1 <= sim_tail <= 8:
            raise ValueError(f"Policy trace numerical-failure sim tail drift: {path}")
    elif terminal_after != dict(final_environment):
        raise ValueError(f"Policy trace terminal environment-chain drift: {path}")
    actual_counter_deltas = {
        name: terminal_after[name] - terminal_before[name]
        for name in TRACE_COUNTER_DELTA_FIELDS
    }
    recorded_counter_deltas = value.get("counter_deltas")
    if (
        not isinstance(recorded_counter_deltas, Mapping)
        or set(recorded_counter_deltas) != TRACE_COUNTER_DELTA_FIELDS
        or any(type(item) is not int for item in recorded_counter_deltas.values())
        or dict(recorded_counter_deltas) != actual_counter_deltas
    ):
        raise ValueError(f"Policy trace terminal counter/snapshot drift: {path}")
    expected_non_sim_counter_deltas = {
        "episode_length": completed,
        "common_step_counter": completed,
    }
    if any(
        actual_counter_deltas[name] != expected
        for name, expected in expected_non_sim_counter_deltas.items()
    ):
        raise ValueError(f"Policy trace terminal non-sim counter drift: {path}")
    completed_sim_steps = completed * 8
    sim_delta = actual_counter_deltas["sim_step_counter"]
    if result["numerical_failure"]:
        if not completed_sim_steps < sim_delta <= completed_sim_steps + 8:
            raise ValueError(f"Policy trace terminal sim-counter tail drift: {path}")
    elif sim_delta != completed_sim_steps:
        raise ValueError(f"Policy trace terminal sim-counter drift: {path}")
    expected_camera_deltas = {name: completed for name in EGO_LAP_CAMERA_SENSOR_NAMES}
    recorded_camera_deltas = value.get("camera_frame_deltas")
    if (
        not isinstance(recorded_camera_deltas, Mapping)
        or set(recorded_camera_deltas) != set(EGO_LAP_CAMERA_SENSOR_NAMES)
        or any(type(item) is not int for item in recorded_camera_deltas.values())
        or dict(recorded_camera_deltas) != expected_camera_deltas
    ):
        raise ValueError(f"Policy trace terminal camera-delta drift: {path}")
    actual_camera_deltas = {
        name: (
            terminal_after["sensor_frame_counters"][name]
            - terminal_before["sensor_frame_counters"][name]
        )
        for name in EGO_LAP_CAMERA_SENSOR_NAMES
    }
    if actual_camera_deltas != expected_camera_deltas:
        raise ValueError(f"Policy trace terminal camera/snapshot drift: {path}")
    return dict(value)


def _trace_finite_array(
    value: Any, *, shape: tuple[int, ...], field: str, path: Path
) -> np.ndarray:
    if not isinstance(value, list):
        raise ValueError(f"Policy trace {field} is not a JSON array: {path}")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Policy trace {field} is not numeric: {path}") from error
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(
            f"Policy trace {field} shape/finiteness drift: "
            f"expected={shape}, actual={array.shape}: {path}"
        )
    return array


def _validate_trace_query_payload(
    record: Mapping[str, Any], *, path: Path
) -> dict[str, Any]:
    """Validate and independently recompute one policy-query payload."""

    for field in ("instruction", "checkpoint_profile", "checkpoint_path"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Policy trace query {field} drift: {path}")
    for field in (
        "contract_sha256",
        "normalization_stats_sha256",
        "normalization_formula_probe_sha256",
    ):
        value = record.get(field)
        if not isinstance(value, str) or TRACE_SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"Policy trace query {field} drift: {path}")
    if (
        record.get("eef_frame") != "panda_link8"
        or record.get("numeric_action_frame") != "robot_base"
        or record.get("gripper_execution_profile")
        != "binary_model_open_gt_0p5_else_closed_v1"
        or record.get("gripper_threshold") != 0.5
        or record.get("polaris_profile") != "panda_link8_eef_pose_single_arm_v1"
    ):
        raise ValueError(f"Policy trace query frame/gripper profile drift: {path}")
    frame_description = record.get("frame_description")
    if not isinstance(frame_description, str) or not frame_description.strip():
        raise ValueError(f"Policy trace query frame description drift: {path}")
    if record.get("execution_horizon") != 8:
        raise ValueError(f"Policy trace query execution horizon drift: {path}")

    policy_type = record.get("policy_type")
    if policy_type == "flow":
        expected_sampler = {
            "action_sampler_profile": "flow_explicit_euler_t1_to_t0_v1",
            "flow_num_steps": 10,
            "initial_rng_seed": 0,
            "ar_max_decoding_steps": None,
            "ar_temperature": None,
            "ar_stop_at_eos": None,
            "response_semantics": "cumulative_delta_targets",
            "ar_endpoint_interpolation_profile": None,
            "ar_endpoint_interpolation_steps": None,
        }
        server_horizon = raw_horizon = 16
    elif policy_type == "ar":
        expected_sampler = {
            "action_sampler_profile": "autoregressive_max500_temp0_eos_v1",
            "flow_num_steps": None,
            "initial_rng_seed": 0,
            "ar_max_decoding_steps": 500,
            "ar_temperature": 0.0,
            "ar_stop_at_eos": True,
            "response_semantics": "total_delta_endpoint",
            "ar_endpoint_interpolation_profile": TRACE_AR_INTERPOLATION_PROFILE,
            "ar_endpoint_interpolation_steps": 8,
        }
        server_horizon, raw_horizon = 1, 8
    else:
        raise ValueError(f"Policy trace query policy type drift: {path}")
    if any(
        record.get(field) != expected for field, expected in expected_sampler.items()
    ):
        raise ValueError(f"Policy trace query sampler/response drift: {path}")

    normalization_profile = record.get("normalization_profile")
    normalization_formulas = {
        "q99_train_matched_v1": (
            "q99_input_eps1e-8_clip_zero0_v1",
            "q99_output_eps1e-8_zeroq01_extrapolate_v1",
        ),
        "q99_legacy_upstream_v1": (
            "q99_input_eps1e-6_no_clip_zero0_v1",
            "q99_output_eps1e-6_no_zero_override_extrapolate_v1",
        ),
    }
    expected_normalization_dtype = {
        "q99_train_matched_v1": "float32",
        "q99_legacy_upstream_v1": "float64",
    }.get(normalization_profile)
    if (
        normalization_profile not in normalization_formulas
        or record.get("normalization_scope") not in {"global", "category"}
        or record.get("normalization_compute_dtype") != expected_normalization_dtype
        or (
            record.get("normalization_input_formula"),
            record.get("normalization_output_formula"),
        )
        != normalization_formulas[normalization_profile]
    ):
        raise ValueError(f"Policy trace query normalization drift: {path}")

    state_layout = record.get("state_layout")
    expected_layout_mode = {
        TRACE_R6_ROWS_LAYOUT: TRACE_R6_ROWS_MODE,
        TRACE_R6_COLUMNS_LAYOUT: TRACE_R6_COLUMNS_MODE,
    }.get(state_layout)
    if (
        expected_layout_mode is None
        or record.get("state_layout_mode") != expected_layout_mode
    ):
        raise ValueError(f"Policy trace query state-layout drift: {path}")

    anchor_position = _trace_finite_array(
        record.get("anchor_position"),
        shape=(3,),
        field="query anchor_position",
        path=path,
    )
    anchor_quaternion = _trace_finite_array(
        record.get("anchor_quaternion_wxyz"),
        shape=(4,),
        field="query anchor_quaternion_wxyz",
        path=path,
    )
    quaternion_norm = float(np.linalg.norm(anchor_quaternion))
    if abs(quaternion_norm - 1.0) > 1e-3:
        raise ValueError(f"Policy trace query anchor quaternion norm drift: {path}")
    anchor_quaternion = anchor_quaternion / quaternion_norm
    state = _trace_finite_array(
        record.get("state"), shape=(10,), field="query state", path=path
    )
    rotation = Rotation.from_quat(anchor_quaternion[[1, 2, 3, 0]])
    matrix = rotation.as_matrix()
    r6 = (
        np.concatenate((matrix[0, :], matrix[1, :]))
        if state_layout == TRACE_R6_ROWS_LAYOUT
        else np.concatenate((matrix[:, 0], matrix[:, 1]))
    )
    if state[-1] not in (0.0, 1.0) or not np.array_equal(
        state.astype(np.float32),
        np.concatenate((anchor_position, r6, state[-1:])).astype(np.float32),
    ):
        raise ValueError(f"Policy trace query state/R6 recompute drift: {path}")

    server = _trace_finite_array(
        record.get("server_delta_chunk"),
        shape=(server_horizon, 7),
        field="query server_delta_chunk",
        path=path,
    )
    raw = _trace_finite_array(
        record.get("raw_delta_chunk"),
        shape=(raw_horizon, 7),
        field="query raw_delta_chunk",
        path=path,
    )
    if policy_type == "flow":
        expected_raw = server
    else:
        fractions = np.linspace(0.0, 1.0, 8, endpoint=True, dtype=np.float64)[:, None]
        expected_raw = np.repeat(server, 8, axis=0)
        expected_raw[:, :3] *= fractions
        endpoint_rotvec = Rotation.from_euler("xyz", server[0, 3:6]).as_rotvec()
        expected_raw[:, 3:6] = Rotation.from_rotvec(
            fractions * endpoint_rotvec
        ).as_euler("xyz")
    if not np.allclose(raw, expected_raw, rtol=0.0, atol=1e-12):
        raise ValueError(f"Policy trace query server/raw chunk drift: {path}")
    base = _trace_finite_array(
        record.get("base_delta_chunk"),
        shape=(raw_horizon, 7),
        field="query base_delta_chunk",
        path=path,
    )
    if not np.array_equal(base, raw):
        raise ValueError(f"Policy trace query numeric-frame chunk drift: {path}")
    anchored = _trace_finite_array(
        record.get("anchored_action_chunk"),
        shape=(raw_horizon, 8),
        field="query anchored_action_chunk",
        path=path,
    )
    expected_position = anchor_position[None, :] + base[:, :3]
    expected_rotation = (rotation * Rotation.from_euler("xyz", base[:, 3:6])).as_quat()[
        :, [3, 0, 1, 2]
    ]
    expected_closed = (np.clip(base[:, 6:7], 0.0, 1.0) <= 0.5).astype(np.float64)
    expected_anchored = np.concatenate(
        (expected_position, expected_rotation, expected_closed), axis=1
    ).astype(np.float32)
    if not np.array_equal(anchored.astype(np.float32), expected_anchored):
        raise ValueError(f"Policy trace query anchored-action recompute drift: {path}")
    return {
        "raw": raw,
        "anchored": anchored.astype(np.float32),
        "execution_horizon": 8,
    }


def validate_episode_trace(
    path: Path,
    *,
    episode: int,
    expected_length: int,
    expected_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require one finalized, internally consistent per-episode policy trace."""

    records = _read_trace_records(path)
    if len(records) < 2 or (
        records[0].get("event") != "reset"
        or records[-1].get("event") != "episode_complete"
    ):
        raise ValueError(
            f"Policy trace must start with reset and end with episode_complete: {path}"
        )
    for record_index, record in enumerate(records):
        if (
            type(record.get("schema_version")) is not int
            or record.get("schema_version") != EGO_LAP_TRACE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"Policy trace schema version drift at record {record_index}: {path}"
            )
        if record.get("trace_profile") != EGO_LAP_TRACE_PROFILE:
            raise ValueError(
                f"Policy trace profile drift at record {record_index}: {path}"
            )
        if type(record.get("episode")) is not int:
            raise ValueError(
                f"Policy trace episode type drift at record {record_index}: {path}"
            )
        timestamp = record.get("timestamp")
        if (
            not isinstance(timestamp, numbers.Real)
            or isinstance(timestamp, (bool, np.bool_))
            or not math.isfinite(float(timestamp))
        ):
            raise ValueError(
                f"Policy trace timestamp drift at record {record_index}: {path}"
            )
    wrong_episode = [
        record.get("episode") for record in records if record.get("episode") != episode
    ]
    if wrong_episode:
        raise ValueError(
            f"Policy trace contains records for the wrong episode {episode}: "
            f"{wrong_episode[:5]} in {path}"
        )
    if records[-1].get("episode_length") != expected_length:
        raise ValueError(
            f"Policy trace length mismatch for episode {episode}: "
            f"expected={expected_length}, actual={records[-1].get('episode_length')!r}"
        )
    reset_fields = TRACE_COMMON_FIELDS | {
        "environment_runtime_profile",
        "environment_before",
    }
    if set(records[0]) != reset_fields or (
        records[0].get("environment_runtime_profile")
        != EGO_LAP_ENVIRONMENT_RUNTIME_PROFILE
    ):
        raise ValueError(f"Policy trace reset schema drift: {path}")
    initial_environment = _validate_trace_environment_state(
        records[0].get("environment_before"), field="reset"
    )
    if initial_environment["episode_length"] != 0:
        raise ValueError(f"Policy trace reset episode counter is not zero: {path}")
    action_records = [record for record in records if record.get("event") == "action"]
    action_count = len(action_records)
    if action_count != expected_length:
        raise ValueError(
            f"Policy trace action count mismatch for episode {episode}: "
            f"expected={expected_length}, actual={action_count}"
        )
    terminal = records[-1]
    if set(terminal) != TRACE_COMPLETE_FIELDS:
        raise ValueError(f"Policy trace episode-complete schema drift: {path}")
    terminal_result = canonical_episode_result(
        {field: terminal.get(field) for field in EVAL_RESULT_COLUMNS}
    )
    expected_status = (
        "numerical_failure" if terminal_result["numerical_failure"] else "completed"
    )
    if terminal.get("status") != expected_status:
        raise ValueError(
            f"Policy trace terminal status mismatch for episode {episode}: "
            f"expected={expected_status!r}, actual={terminal.get('status')!r}"
        )
    if expected_result is not None:
        canonical_expected = canonical_episode_result(expected_result)
        if terminal_result != canonical_expected:
            raise ValueError(
                f"Policy trace terminal result mismatch for episode {episode}: "
                f"expected={canonical_expected!r}, actual={terminal_result!r}"
            )
    terminal_rollout = terminal.get("terminal_rollout")
    if not isinstance(terminal_rollout, Mapping):
        raise ValueError(f"Policy trace has no terminal rollout evidence: {path}")
    if terminal_rollout.get("profile") != EGO_LAP_TERMINAL_ROLLOUT_PROFILE or (
        terminal_rollout.get("environment_runtime_profile")
        != EGO_LAP_ENVIRONMENT_RUNTIME_PROFILE
    ):
        raise ValueError(f"Policy trace terminal rollout profile drift: {path}")
    if terminal_rollout.get("episode_result") != terminal_result:
        raise ValueError(f"Policy trace terminal rollout/result binding drift: {path}")

    expected_execution_count = expected_length - int(
        terminal_result["numerical_failure"]
    )
    action_steps: list[int] = []
    execution_steps: list[int] = []
    failure_steps: list[int] = []
    pending_action: tuple[int, int, int] | None = None
    query_records: list[Mapping[str, Any]] = []
    validated_queries: list[dict[str, Any]] = []
    query_static_identity: dict[str, Any] | None = None
    execution_horizon: int | None = None
    current_environment = initial_environment
    for record_index, record in enumerate(records[1:-1], start=1):
        event = record.get("event")
        if event == "query":
            if set(record) != TRACE_QUERY_FIELDS or pending_action is not None:
                raise ValueError(
                    f"Policy trace query schema/order drift at {record_index}: {path}"
                )
            validated_query = _validate_trace_query_payload(record, path=path)
            candidate_horizon = validated_query["execution_horizon"]
            if any(
                type(record.get(field)) is not int or record[field] < 0
                for field in ("query", "step")
            ):
                raise ValueError(f"Policy trace query identity drift: {path}")
            if execution_horizon is None:
                execution_horizon = candidate_horizon
            if (
                candidate_horizon != execution_horizon
                or record.get("query") != len(query_records)
                or record.get("step") != len(action_steps)
                or record.get("step") != record.get("query") * execution_horizon
            ):
                raise ValueError(
                    f"Policy trace query placement/cadence drift at {record_index}: {path}"
                )
            candidate_identity = {
                field: record[field] for field in TRACE_QUERY_STATIC_IDENTITY_FIELDS
            }
            if query_static_identity is None:
                query_static_identity = candidate_identity
            elif candidate_identity != query_static_identity:
                raise ValueError(
                    f"Policy trace query static identity drift at {record_index}: "
                    f"{path}"
                )
            query_records.append(record)
            validated_queries.append(validated_query)
            continue
        if event == "action":
            if set(record) != TRACE_ACTION_FIELDS or pending_action is not None:
                raise ValueError(
                    f"Policy trace action schema/order drift at {record_index}: {path}"
                )
            step = record.get("step")
            query = record.get("query")
            chunk = record.get("chunk_index")
            if any(
                type(value) is not int or value < 0 for value in (step, query, chunk)
            ):
                raise ValueError(f"Policy trace action identity drift: {path}")
            if (
                execution_horizon is None
                or not query_records
                or step != len(action_steps)
                or query != len(query_records) - 1
                or chunk != step % execution_horizon
            ):
                raise ValueError(f"Policy trace action/query cadence drift: {path}")
            raw_delta = _trace_finite_array(
                record.get("raw_delta"),
                shape=(7,),
                field="action raw_delta",
                path=path,
            )
            polaris_action = _trace_finite_array(
                record.get("polaris_action"),
                shape=(8,),
                field="action polaris_action",
                path=path,
            )
            query_payload = validated_queries[query]
            if not np.array_equal(raw_delta, query_payload["raw"][chunk]):
                raise ValueError(f"Policy trace action/raw-query binding drift: {path}")
            if not np.array_equal(
                polaris_action.astype(np.float32),
                query_payload["anchored"][chunk],
            ):
                raise ValueError(
                    f"Policy trace action/anchored-query binding drift: {path}"
                )
            pending_action = (step, query, chunk)
            action_steps.append(step)
            continue
        if event == "execution":
            if set(record) != TRACE_EXECUTION_FIELDS or pending_action is None:
                raise ValueError(
                    f"Policy trace execution schema/order drift at {record_index}: {path}"
                )
            identity = (
                record.get("step"),
                record.get("query"),
                record.get("chunk_index"),
            )
            if identity != pending_action:
                raise ValueError(
                    f"Policy trace action/execution identity drift: {path}"
                )
            current_environment = _validate_trace_transition(
                record.get("transition"),
                step=pending_action[0],
                expected_before=current_environment,
                path=path,
            )
            execution_steps.append(pending_action[0])
            pending_action = None
            continue
        if event == "execution_failure":
            if set(record) != TRACE_EXECUTION_FAILURE_FIELDS or pending_action is None:
                raise ValueError(
                    "Policy trace execution-failure schema/order drift at "
                    f"{record_index}: {path}"
                )
            identity = (
                record.get("step"),
                record.get("query"),
                record.get("chunk_index"),
            )
            reason = record.get("numerical_failure_reason")
            if (
                identity != pending_action
                or not isinstance(reason, str)
                or not reason
                or reason != terminal_result["numerical_failure_reason"]
            ):
                raise ValueError(
                    f"Policy trace execution-failure identity drift: {path}"
                )
            failure_steps.append(pending_action[0])
            pending_action = None
            continue
        raise ValueError(f"Unsupported policy trace event {event!r}: {path}")
    if pending_action is not None:
        raise ValueError(f"Policy trace has an unaccounted action: {path}")
    if action_steps != list(range(expected_length)):
        raise ValueError(f"Policy trace action steps are not contiguous: {path}")
    if execution_steps != list(range(expected_execution_count)):
        raise ValueError(f"Policy trace execution steps are not contiguous: {path}")
    expected_failure_steps = (
        [expected_length - 1] if terminal_result["numerical_failure"] else []
    )
    if failure_steps != expected_failure_steps:
        raise ValueError(f"Policy trace execution-failure cadence drift: {path}")
    if not query_records:
        raise ValueError(f"Policy trace has no policy queries: {path}")
    if execution_horizon is None:
        raise ValueError(f"Policy trace execution horizon drift: {path}")
    expected_query_count = math.ceil(expected_length / execution_horizon)
    if len(query_records) != expected_query_count:
        raise ValueError(
            "Policy trace query count drift: "
            f"expected={expected_query_count}, actual={len(query_records)}: {path}"
        )
    for query_index, query_record in enumerate(query_records):
        if (
            query_record.get("query") != query_index
            or query_record.get("step") != query_index * execution_horizon
            or query_record.get("execution_horizon") != execution_horizon
        ):
            raise ValueError(f"Policy trace query cadence drift: {path}")
    for action_record in action_records:
        step = action_record["step"]
        if (
            action_record["query"] != step // execution_horizon
            or action_record["chunk_index"] != step % execution_horizon
        ):
            raise ValueError(f"Policy trace action chunk cadence drift: {path}")
    expected_record_count = 2 + expected_query_count + 2 * expected_length
    if len(records) != expected_record_count:
        raise ValueError(
            "Policy trace record multiplicity drift: "
            f"expected={expected_record_count}, actual={len(records)}: {path}"
        )
    if terminal_rollout.get("actions_attempted") != expected_length or (
        terminal_rollout.get("outer_steps_completed") != expected_execution_count
    ):
        raise ValueError(f"Policy trace terminal cadence binding drift: {path}")
    validated_terminal_rollout = _validate_trace_terminal_rollout(
        terminal_rollout,
        result=terminal_result,
        initial_environment=initial_environment,
        final_environment=current_environment,
        path=path,
    )
    return {
        "episode_result": terminal_result,
        "terminal_rollout": validated_terminal_rollout,
    }


def build_episode_artifact_identity(
    *,
    run_folder: Path,
    trace_path: Path,
    episode_result: Mapping[str, Any],
    video_probe_fn: Callable[[Path], Mapping[str, Any]] = probe_episode_video,
) -> dict[str, Any]:
    """Capture immutable video and finalized terminal-trace identities."""

    result = canonical_episode_result(episode_result)
    episode = result["episode"]
    video_path = run_folder / f"episode_{episode}.mp4"
    expected_trace_path = trace_path.parent / f"episode_{episode:06d}.jsonl"
    if trace_path != expected_trace_path:
        raise ValueError(
            "Transactional Ego-LAP evidence requires one per-episode trace: "
            f"expected={expected_trace_path}, actual={trace_path}"
        )
    video_shape = validate_episode_video(
        video_path,
        expected_frames=result["episode_length"],
        probe_fn=video_probe_fn,
    )
    trace_terminal = validate_episode_trace(
        trace_path,
        episode=episode,
        expected_length=result["episode_length"],
        expected_result=result,
    )
    return {
        "video": {
            "filename": video_path.name,
            "size_bytes": video_path.stat().st_size,
            "sha256": sha256_file(video_path),
            **video_shape,
        },
        "terminal_trace": {
            "filename": trace_path.name,
            "size_bytes": trace_path.stat().st_size,
            "sha256": sha256_file(trace_path),
            "schema_version": EGO_LAP_TRACE_SCHEMA_VERSION,
            "trace_profile": EGO_LAP_TRACE_PROFILE,
            "episode_result": trace_terminal["episode_result"],
            "terminal_rollout": trace_terminal["terminal_rollout"],
        },
    }


def validate_episode_artifact_identity(
    identity: Mapping[str, Any],
    *,
    run_folder: Path,
    trace_dir: Path,
    episode_result: Mapping[str, Any],
    video_probe_fn: Callable[[Path], Mapping[str, Any]] = probe_episode_video,
) -> dict[str, Any]:
    """Recompute and exactly match one sidecar's artifact identities."""

    result = canonical_episode_result(episode_result)
    actual = build_episode_artifact_identity(
        run_folder=run_folder,
        trace_path=trace_dir / f"episode_{result['episode']:06d}.jsonl",
        episode_result=result,
        video_probe_fn=video_probe_fn,
    )
    if dict(identity) != actual:
        raise ValueError(
            "Episode sidecar artifact identity drift: "
            f"recorded={dict(identity)!r}, actual={actual!r}"
        )
    return actual


def _validate_legacy_trace(
    path: Path,
    *,
    completed_rows: Sequence[tuple[int, int]],
) -> None:
    records = _read_trace_records(path)
    for episode, expected_length in completed_rows:
        episode_records = [
            record for record in records if record.get("episode") == episode
        ]
        if not episode_records:
            raise ValueError(
                f"Legacy policy trace has no records for episode {episode}: {path}"
            )
        if (
            episode_records[0].get("event") != "reset"
            or episode_records[-1].get("event") != "episode_complete"
        ):
            raise ValueError(
                f"Legacy policy trace episode {episode} is incomplete: {path}"
            )
        action_count = sum(
            record.get("event") == "action" for record in episode_records
        )
        if action_count != expected_length:
            raise ValueError(
                f"Legacy policy trace action count mismatch for episode {episode}: "
                f"expected={expected_length}, actual={action_count}"
            )


def load_resume_results(
    csv_path: Path,
    *,
    run_folder: Path,
    expected_rollouts: int,
    expected_horizon: int,
    require_episode_artifacts: bool,
    trace_dir: Path | None = None,
    trace_path: Path | None = None,
    video_probe_fn: Callable[[Path], Mapping[str, Any]] = probe_episode_video,
) -> pd.DataFrame:
    """Load only a contiguous, artifact-complete prefix of rollout results."""

    if trace_dir is not None and trace_path is not None:
        raise ValueError("Configure either trace_dir or trace_path, not both")
    if not csv_path.exists():
        return empty_eval_results()
    try:
        frame = pd.read_csv(csv_path)
    except Exception as error:
        raise ValueError(f"Could not read resume CSV {csv_path}: {error}") from error
    required = {"episode", "episode_length", "success", "progress"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Resume CSV is missing required columns {missing}: {csv_path}"
        )
    if len(frame) > expected_rollouts:
        raise ValueError(
            f"Resume CSV has {len(frame)} rows but only {expected_rollouts} rollouts were requested"
        )

    episodes: list[int] = []
    lengths: list[int] = []
    for row_index, (episode_value, length_value) in enumerate(
        zip(frame["episode"], frame["episode_length"], strict=True)
    ):
        try:
            episode = int(episode_value)
            length = int(length_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Resume CSV row {row_index} has a non-integer identity"
            ) from error
        if float(episode_value) != episode or float(length_value) != length:
            raise ValueError(f"Resume CSV row {row_index} has a non-integral identity")
        if not 1 <= length <= expected_horizon:
            raise ValueError(
                f"Resume CSV episode {episode} length must be in [1, {expected_horizon}], got {length}"
            )
        episodes.append(episode)
        lengths.append(length)
    expected_episodes = list(range(len(frame)))
    if episodes != expected_episodes:
        raise ValueError(
            f"Resume CSV episode IDs must be the contiguous prefix {expected_episodes}; got {episodes}"
        )

    if "numerical_failure" not in frame:
        frame["numerical_failure"] = False
    if "numerical_failure_reason" not in frame:
        frame["numerical_failure_reason"] = ""
    frame["numerical_failure_reason"] = frame["numerical_failure_reason"].fillna("")
    frame = frame.loc[:, list(EVAL_RESULT_COLUMNS)]
    normalized_rows = [
        canonical_episode_result(row) for row in frame.to_dict(orient="records")
    ]
    frame = (
        pd.DataFrame(normalized_rows, columns=EVAL_RESULT_COLUMNS)
        if normalized_rows
        else empty_eval_results()
    )

    if require_episode_artifacts:
        if trace_dir is None and trace_path is None:
            raise ValueError("Ego-LAP resume requires per-episode policy traces")
        for episode, length in zip(episodes, lengths, strict=True):
            result = canonical_episode_result(
                frame.loc[frame["episode"] == episode].iloc[0].to_dict()
            )
            validate_episode_video(
                run_folder / f"episode_{episode}.mp4",
                expected_frames=length,
                probe_fn=video_probe_fn,
            )
            if trace_dir is not None:
                validate_episode_trace(
                    trace_dir / f"episode_{episode:06d}.jsonl",
                    episode=episode,
                    expected_length=length,
                    expected_result=result,
                )
        if trace_path is not None and episodes:
            _validate_legacy_trace(
                trace_path,
                completed_rows=list(zip(episodes, lengths, strict=True)),
            )
    return frame


def atomic_write_episode_video(
    path: Path,
    frames: Sequence[np.ndarray],
    *,
    fps: int,
    writer: Callable[..., Any] | None = None,
    probe_fn: Callable[[Path], Mapping[str, Any]] = probe_episode_video,
) -> None:
    """Write, decode-check, and atomically publish one episode video."""

    if not frames:
        raise ValueError("Cannot finalize an empty rollout video")
    if writer is None:
        import mediapy  # noqa: PLC0415 - simulator runtime dependency

        writer = mediapy.write_video
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        writer(temporary, frames, fps=fps)
        validate_episode_video(
            temporary,
            expected_frames=len(frames),
            probe_fn=probe_fn,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_results(frame: pd.DataFrame, path: Path) -> None:
    """Atomically replace the episode CSV after all row artifacts are durable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as output:
            frame.to_csv(output, index=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
