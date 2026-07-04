#!/usr/bin/env python3
"""Fail-closed trace audit for the official native pi0.5-DROID canary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any

from polaris.pi05_droid_jointvelocity_contract import (
    PI05_DROID_GRIPPER_OBSERVATION_BOUND_TOLERANCE,
    PI05_DROID_JOINTVELOCITY_PROFILE,
)
from polaris.pi05_droid_native_eval_contract import (
    PI05_DROID_NATIVE_ACTION_WIDTH,
    PI05_DROID_NATIVE_DECIMATION,
    PI05_DROID_NATIVE_EPISODE_STEPS,
    PI05_DROID_NATIVE_EXECUTION_HORIZON,
    PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS,
    PI05_DROID_NATIVE_RESPONSE_HORIZON,
    PI05_DROID_NATIVE_SENSOR_LIVENESS_PROFILE,
    PI05_DROID_NATIVE_SENSOR_NAMES,
    PI05_DROID_NATIVE_TASK,
    PI05_DROID_NATIVE_TRACE_SCHEMA_VERSION,
    canonical_json_bytes,
    make_environment_runtime_contract,
    publish_immutable_json,
    validate_immutable_json,
    validate_terminal_numerical_failure_evidence,
    validate_terminal_rollout_evidence,
)


EXPECTED_PROMPT = "Put all the foods in the bowl"
PANDA_SOFT_LIMITS = (
    (-2.8973000049591064, 2.8973000049591064),
    (-1.7627999782562256, 1.7627999782562256),
    (-2.8973000049591064, 2.8973000049591064),
    (-3.0717999935150146, -0.06979990005493164),
    (-2.8973000049591064, 2.8973000049591064),
    (-0.017499923706054688, 3.752500057220459),
    (-2.8973000049591064, 2.8973000049591064),
)
PANDA_VELOCITY_LIMITS = (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)
BLANK_IMAGE_SHA256 = hashlib.sha256(bytes(224 * 224 * 3)).hexdigest()

CONTRACT_KEYS = {
    "serving_contract_sha256",
    "serving_contract_artifact_sha256",
    "serving_contract_artifact_size",
    "client_runtime_attestation_sha256",
    "environment_runtime_sha256",
    "outer_episode_steps",
    "internal_max_episode_length",
    "sensor_liveness_profile",
}
ENVIRONMENT_STATE_KEYS = {
    "live_max_episode_length",
    "episode_length",
    "sim_step_counter",
    "common_step_counter",
    "sensor_frame_counters",
}
ROLLOUT_START_KEYS = {
    "schema_version",
    "record_type",
    "profile",
    *CONTRACT_KEYS,
    "reset_index",
    "environment_before",
}
QUERY_KEYS = {
    "schema_version",
    "record_type",
    "profile",
    *CONTRACT_KEYS,
    "reset_index",
    "query_index",
    "prompt",
    "state",
    "images",
    "response_action_shape",
    "response_action_chunk_raw",
    "execution_horizon",
    "planned_action_chunk_binary_gripper",
    "planned_action_chunk_clipped",
}
ACTION_KEYS = {
    "schema_version",
    "record_type",
    "profile",
    *CONTRACT_KEYS,
    "reset_index",
    "query_index",
    "chunk_action_index",
    "raw_action",
    "binary_gripper_action",
    "clipped_action",
    "emitted_joint_velocity",
    "emitted_gripper_closed",
    "measured_joint_position_before",
    "measured_joint_velocity_before",
    "measured_normalized_gripper_position_before",
}
EXECUTION_KEYS = {
    "schema_version",
    "record_type",
    "profile",
    *CONTRACT_KEYS,
    "reset_index",
    "query_index",
    "chunk_action_index",
    "outer_step_index",
    "terminated",
    "truncated",
    "environment_after",
    "processed_joint_velocity",
    "articulation_joint_velocity_target",
    "processed_finger_position_target",
    "articulation_finger_position_target",
    "measured_joint_position_after",
    "measured_joint_velocity_after",
    "measured_normalized_gripper_position_after",
}
ROLLOUT_END_KEYS = {
    "schema_version",
    "record_type",
    "profile",
    *CONTRACT_KEYS,
    "reset_index",
    "terminal_rollout",
}
ROLLOUT_FAILURE_KEYS = {
    "schema_version",
    "record_type",
    "profile",
    *CONTRACT_KEYS,
    "reset_index",
    "query_index",
    "chunk_action_index",
    "outer_step_index",
    "terminal_failure",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_string(value: Any, field: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{field} must be one lowercase SHA-256",
    )
    return value


def _finite_number(value: Any, field: str) -> float:
    _require(
        type(value) in (int, float) and math.isfinite(value),
        f"{field} must be finite and numeric",
    )
    return float(value)


def _finite_vector(value: Any, width: int, field: str) -> list[float]:
    _require(isinstance(value, list) and len(value) == width, f"{field} width mismatch")
    return [
        _finite_number(item, f"{field}[{index}]") for index, item in enumerate(value)
    ]


def _normalized_gripper_vector(value: Any, field: str) -> list[float]:
    result = _finite_vector(value, 1, field)
    tolerance = PI05_DROID_GRIPPER_OBSERVATION_BOUND_TOLERANCE
    _require(
        -tolerance <= result[0] <= 1.0 + tolerance,
        f"{field} is outside [0, 1] plus {tolerance} audit tolerance",
    )
    return result


def _finite_matrix(
    value: Any, rows: int, columns: int, field: str
) -> list[list[float]]:
    _require(
        isinstance(value, list) and len(value) == rows, f"{field} row count mismatch"
    )
    return [
        _finite_vector(row, columns, f"{field}[{index}]")
        for index, row in enumerate(value)
    ]


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _parse_bool(value: str, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{field} must be True or False")


def _validate_metrics(metrics_csv: Path) -> dict[str, Any]:
    with metrics_csv.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        expected_fields = [
            "episode",
            "episode_length",
            "success",
            "progress",
            "numerical_failure",
            "numerical_failure_reason",
        ]
        _require(reader.fieldnames == expected_fields, "Metrics CSV schema mismatch")
        rows = list(reader)
    _require(len(rows) == 1, "Canary metrics must contain exactly one episode")
    row = rows[0]
    _require(int(row["episode"]) == 0, "Canary episode must be zero")
    episode_length = int(row["episode_length"])
    _require(
        1 <= episode_length <= PI05_DROID_NATIVE_EPISODE_STEPS,
        "Canary episode length is invalid",
    )
    success = _parse_bool(row["success"], "success")
    numerical_failure = _parse_bool(row["numerical_failure"], "numerical_failure")
    progress = _finite_number(float(row["progress"]), "progress")
    _require(0.0 <= progress <= 1.0, "Canary progress must be in [0, 1]")
    reason = row["numerical_failure_reason"].strip()
    if numerical_failure:
        _require(
            success is False
            and progress == 0.0
            and reason.startswith("NativeAllJointVelocityLimitError: "),
            "Canary numerical-failure metrics mismatch",
        )
    else:
        _require(
            episode_length == PI05_DROID_NATIVE_EPISODE_STEPS,
            "Canary must complete all 450 policy steps",
        )
        _require(not reason, "Canary has a failure reason")
    return {
        "episode": 0,
        "episode_length": episode_length,
        "success": success,
        "progress": progress,
        "numerical_failure": numerical_failure,
        "numerical_failure_reason": reason,
    }


def _validate_contract_identity(
    record: dict[str, Any], expected: dict[str, Any] | None, field: str
) -> dict[str, Any]:
    identity = {key: record[key] for key in CONTRACT_KEYS}
    for key in (
        "serving_contract_sha256",
        "serving_contract_artifact_sha256",
        "client_runtime_attestation_sha256",
    ):
        _sha256_string(identity[key], f"{field} {key}")
    _require(
        type(identity["serving_contract_artifact_size"]) is int
        and identity["serving_contract_artifact_size"] > 0,
        f"{field} serving-contract size is invalid",
    )
    _sha256_string(
        identity["environment_runtime_sha256"],
        f"{field} environment runtime digest",
    )
    _require(
        identity["outer_episode_steps"] == PI05_DROID_NATIVE_EPISODE_STEPS
        and identity["internal_max_episode_length"]
        == PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS
        and identity["sensor_liveness_profile"]
        == PI05_DROID_NATIVE_SENSOR_LIVENESS_PROFILE,
        f"{field} environment runtime identity drift",
    )
    if expected is not None:
        _require(identity == expected, f"{field} contract identity drift")
    return identity


def _validate_environment_state(value: Any, field: str) -> dict[str, Any]:
    _require(
        isinstance(value, dict) and set(value) == ENVIRONMENT_STATE_KEYS,
        f"{field} schema mismatch",
    )
    for counter_field in (
        "live_max_episode_length",
        "episode_length",
        "sim_step_counter",
        "common_step_counter",
    ):
        _require(
            type(value[counter_field]) is int and value[counter_field] >= 0,
            f"{field} {counter_field} is invalid",
        )
    _require(
        value["live_max_episode_length"]
        == PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS,
        f"{field} live max_episode_length mismatch",
    )
    sensor_frames = value["sensor_frame_counters"]
    _require(
        isinstance(sensor_frames, dict)
        and set(sensor_frames) == set(PI05_DROID_NATIVE_SENSOR_NAMES)
        and all(type(frame) is int and frame >= 0 for frame in sensor_frames.values()),
        f"{field} sensor frame counters mismatch",
    )
    return value


def _validate_image(value: Any, field: str, *, blank: bool = False) -> str:
    _require(
        isinstance(value, dict) and set(value) == {"shape", "dtype", "sha256"},
        f"{field} schema mismatch",
    )
    _require(value["shape"] == [224, 224, 3], f"{field} shape mismatch")
    _require(value["dtype"] == "uint8", f"{field} dtype mismatch")
    digest = _sha256_string(value["sha256"], f"{field} digest")
    if blank:
        _require(digest == BLANK_IMAGE_SHA256, "Masked blank image bytes changed")
    return digest


def _processed_action(raw: list[float]) -> tuple[list[float], list[float]]:
    binary = [*raw[:7], 1.0 if raw[7] > 0.5 else 0.0]
    clipped = [min(1.0, max(-1.0, value)) for value in binary]
    return binary, clipped


def _validate_joint_state(
    q: list[float],
    dq: list[float],
    field: str,
    *,
    allow_incident_velocity: bool = False,
) -> None:
    for index, (value, (lower, upper)) in enumerate(
        zip(q, PANDA_SOFT_LIMITS, strict=True)
    ):
        _require(
            lower - 1e-5 <= value <= upper + 1e-5,
            f"{field} joint {index + 1} is outside the live soft limit",
        )
    if not allow_incident_velocity:
        for index, (value, limit) in enumerate(
            zip(dq, PANDA_VELOCITY_LIMITS, strict=True)
        ):
            _require(
                abs(value) <= limit + 1e-4,
                f"{field} joint {index + 1} exceeds its velocity limit",
            )


def _validate_incident_bound_arm_state(
    q: list[float],
    dq: list[float],
    terminal: dict[str, Any],
    *,
    expected_sample_kind: str,
    expected_physics_substep: int,
    field: str,
) -> None:
    incident_path = Path(terminal["incident_artifact"]["path"])
    incident = validate_immutable_json(incident_path)["value"]
    _require(
        incident == terminal["dynamic_report"]["terminal_velocity_failure"],
        f"{field} immutable incident differs from terminal evidence",
    )
    _require(
        incident["sample_kind"] == expected_sample_kind,
        f"{field} incident sample kind mismatch",
    )
    _require(
        incident["physics_substep_index"] == expected_physics_substep,
        f"{field} incident physics substep mismatch",
    )
    incident_q = _finite_vector(
        incident["joint_position"][:7], 7, f"{field} incident arm position"
    )
    incident_dq = _finite_vector(
        incident["joint_velocity"][:7], 7, f"{field} incident arm velocity"
    )
    _require(q == incident_q, f"{field} position differs from immutable incident")
    _require(dq == incident_dq, f"{field} velocity differs from immutable incident")
    _validate_joint_state(q, dq, field, allow_incident_velocity=True)


def audit_trace(trace_path: Path, metrics_csv: Path) -> dict[str, Any]:
    """Validate the exact 450-step query/action/execution sequence."""

    metrics = _validate_metrics(Path(metrics_csv))
    records: list[dict[str, Any]] = []
    trace_digest = hashlib.sha256()
    with Path(trace_path).open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            trace_digest.update(raw_line)
            _require(
                raw_line.endswith(b"\n"), f"Trace line {line_number} lacks newline"
            )
            _require(raw_line.strip(), f"Trace line {line_number} is empty")
            try:
                record = json.loads(
                    raw_line,
                    parse_constant=lambda constant: (_ for _ in ()).throw(
                        ValueError(f"Non-finite JSON constant: {constant}")
                    ),
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid JSON at trace line {line_number}") from error
            _require(
                isinstance(record, dict), f"Trace line {line_number} is not an object"
            )
            records.append(record)

    actions_attempted = metrics["episode_length"]
    failure_sample_kind = None
    if metrics["numerical_failure"]:
        try:
            failure_sample_kind = records[-1]["terminal_failure"]["failure_sample_kind"]
        except (IndexError, KeyError, TypeError) as error:
            raise ValueError("Trace lacks a closed terminal failure kind") from error
        _require(
            failure_sample_kind in {"apply_entry", "post_policy_step"},
            "Trace terminal failure sample kind mismatch",
        )
    execution_records = actions_attempted - int(failure_sample_kind == "apply_entry")
    expected_queries = math.ceil(
        actions_attempted / PI05_DROID_NATIVE_EXECUTION_HORIZON
    )
    expected_records = expected_queries + actions_attempted + execution_records + 2
    _require(len(records) == expected_records, "Trace record count mismatch")

    cursor = 0
    rollout_start = records[cursor]
    cursor += 1
    _require(set(rollout_start) == ROLLOUT_START_KEYS, "Rollout start schema mismatch")
    _require(
        rollout_start["schema_version"] == PI05_DROID_NATIVE_TRACE_SCHEMA_VERSION,
        "Rollout start schema version mismatch",
    )
    _require(
        rollout_start["record_type"] == "openpi_joint_velocity_rollout_start"
        and rollout_start["profile"] == PI05_DROID_JOINTVELOCITY_PROFILE
        and rollout_start["reset_index"] == 0,
        "Rollout start identity mismatch",
    )
    contract_identity = _validate_contract_identity(
        rollout_start, None, "rollout start"
    )
    environment_before = _validate_environment_state(
        rollout_start["environment_before"], "rollout start environment"
    )
    _require(
        environment_before["episode_length"] == 0,
        "Rollout start episode length was not reset to zero",
    )
    environment_runtime_contract = make_environment_runtime_contract(
        configured_episode_length_seconds=(
            PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS / 15
        ),
        live_max_episode_length=PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS,
    )
    _require(
        environment_runtime_contract["sha256"]
        == contract_identity["environment_runtime_sha256"],
        "Trace environment runtime digest mismatch",
    )
    environment_after = environment_before
    active_chunk: list[list[float]] | None = None
    active_query_q: list[float] | None = None
    active_query_gripper: list[float] | None = None
    previous_q: list[float] | None = None
    previous_dq: list[float] | None = None
    previous_gripper: list[float] | None = None
    external_hashes: set[str] = set()
    wrist_hashes: set[str] = set()
    max_abs_measured_velocity = [0.0] * 7
    normalized_gripper_min = math.inf
    normalized_gripper_max = -math.inf

    terminal_outcome = None

    def consume_failure_record(
        *, step: int, query_index: int, action_index: int, last_environment: dict
    ) -> dict[str, Any]:
        nonlocal cursor
        failure_record = records[cursor]
        cursor += 1
        _require(
            set(failure_record) == ROLLOUT_FAILURE_KEYS,
            "Rollout failure schema mismatch",
        )
        _require(
            failure_record["schema_version"] == PI05_DROID_NATIVE_TRACE_SCHEMA_VERSION
            and failure_record["record_type"] == "openpi_joint_velocity_rollout_failure"
            and failure_record["profile"] == PI05_DROID_JOINTVELOCITY_PROFILE
            and failure_record["reset_index"] == 0,
            "Rollout failure identity mismatch",
        )
        _validate_contract_identity(
            failure_record, contract_identity, "rollout failure"
        )
        _require(
            failure_record["query_index"] == query_index
            and failure_record["chunk_action_index"] == action_index
            and failure_record["outer_step_index"] == step,
            "Rollout failure action identity mismatch",
        )
        terminal = validate_terminal_numerical_failure_evidence(
            failure_record["terminal_failure"], environment_runtime_contract
        )
        _require(
            terminal["failure_sample_kind"] == failure_sample_kind,
            "Rollout failure sample-kind drift",
        )
        _require(
            terminal["episode_result"] == metrics,
            "Terminal failure differs from metrics",
        )
        _require(
            terminal["environment_before"] == environment_before
            and terminal["last_completed_environment"] == last_environment,
            "Terminal failure environment endpoints drift",
        )
        return terminal

    for step in range(actions_attempted):
        query_index = step // PI05_DROID_NATIVE_EXECUTION_HORIZON
        action_index = step % PI05_DROID_NATIVE_EXECUTION_HORIZON
        if action_index == 0:
            query = records[cursor]
            cursor += 1
            _require(set(query) == QUERY_KEYS, f"Query {query_index} schema mismatch")
            _require(
                query["schema_version"] == PI05_DROID_NATIVE_TRACE_SCHEMA_VERSION,
                "Query schema version mismatch",
            )
            _require(
                query["record_type"] == "openpi_joint_velocity_query",
                "Query type mismatch",
            )
            _require(
                query["profile"] == PI05_DROID_JOINTVELOCITY_PROFILE,
                "Query profile mismatch",
            )
            contract_identity = _validate_contract_identity(
                query, contract_identity, f"query {query_index}"
            )
            _require(query["reset_index"] == 0, "Canary reset index mismatch")
            _require(query["query_index"] == query_index, "Query index mismatch")
            _require(query["prompt"] == EXPECTED_PROMPT, "Canary prompt mismatch")
            _require(
                query["response_action_shape"]
                == [PI05_DROID_NATIVE_RESPONSE_HORIZON, PI05_DROID_NATIVE_ACTION_WIDTH],
                "Response shape mismatch",
            )
            _require(
                query["execution_horizon"] == PI05_DROID_NATIVE_EXECUTION_HORIZON,
                "Execution horizon mismatch",
            )
            state = query["state"]
            _require(
                isinstance(state, dict)
                and set(state) == {"joint_position", "gripper_position"},
                "Query state schema mismatch",
            )
            query_q = _finite_vector(state["joint_position"], 7, "query joint position")
            active_query_q = query_q
            gripper = _normalized_gripper_vector(
                state["gripper_position"], "Query normalized gripper"
            )
            active_query_gripper = gripper
            if previous_q is not None:
                _require(
                    query_q == previous_q,
                    "Query state does not match prior measured state",
                )
            if previous_gripper is not None:
                _require(
                    gripper == previous_gripper,
                    "Query gripper does not match prior measured state",
                )

            images = query["images"]
            _require(
                isinstance(images, dict)
                and set(images)
                == {
                    "external",
                    "wrist",
                    "blank_masked_right_wrist",
                    "model_order",
                    "wrist_rotation_degrees",
                },
                "Query image schema mismatch",
            )
            _require(
                images["model_order"]
                == ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb_masked"],
                "Model image order mismatch",
            )
            _require(images["wrist_rotation_degrees"] == 0, "Wrist image was rotated")
            external_hashes.add(_validate_image(images["external"], "external image"))
            wrist_hashes.add(_validate_image(images["wrist"], "wrist image"))
            _validate_image(
                images["blank_masked_right_wrist"], "blank image", blank=True
            )

            active_chunk = _finite_matrix(
                query["response_action_chunk_raw"],
                PI05_DROID_NATIVE_RESPONSE_HORIZON,
                PI05_DROID_NATIVE_ACTION_WIDTH,
                "response action chunk",
            )
            planned_binary = _finite_matrix(
                query["planned_action_chunk_binary_gripper"],
                PI05_DROID_NATIVE_EXECUTION_HORIZON,
                PI05_DROID_NATIVE_ACTION_WIDTH,
                "planned binary chunk",
            )
            planned_clipped = _finite_matrix(
                query["planned_action_chunk_clipped"],
                PI05_DROID_NATIVE_EXECUTION_HORIZON,
                PI05_DROID_NATIVE_ACTION_WIDTH,
                "planned clipped chunk",
            )
            for index in range(PI05_DROID_NATIVE_EXECUTION_HORIZON):
                expected_binary, expected_clipped = _processed_action(
                    active_chunk[index]
                )
                _require(
                    planned_binary[index] == expected_binary,
                    "Planned gripper processing mismatch",
                )
                _require(
                    planned_clipped[index] == expected_clipped,
                    "Planned action clipping mismatch",
                )

        _require(
            active_chunk is not None and contract_identity is not None,
            "Action lacks query",
        )
        raw = active_chunk[action_index]
        expected_binary, expected_clipped = _processed_action(raw)

        action = records[cursor]
        cursor += 1
        _require(set(action) == ACTION_KEYS, f"Action {step} schema mismatch")
        _require(
            action["schema_version"] == PI05_DROID_NATIVE_TRACE_SCHEMA_VERSION,
            "Action schema version mismatch",
        )
        _require(
            action["record_type"] == "openpi_joint_velocity_action",
            "Action type mismatch",
        )
        _require(
            action["profile"] == PI05_DROID_JOINTVELOCITY_PROFILE,
            "Action profile mismatch",
        )
        _validate_contract_identity(action, contract_identity, f"action {step}")
        _require(action["reset_index"] == 0, "Action reset index mismatch")
        _require(action["query_index"] == query_index, "Action query index mismatch")
        _require(
            action["chunk_action_index"] == action_index, "Chunk action index mismatch"
        )
        _require(
            _finite_vector(action["raw_action"], 8, "raw action") == raw,
            "Raw action drift",
        )
        _require(
            _finite_vector(action["binary_gripper_action"], 8, "binary action")
            == expected_binary,
            "Binary action mismatch",
        )
        _require(
            _finite_vector(action["clipped_action"], 8, "clipped action")
            == expected_clipped,
            "Clipped action mismatch",
        )
        _require(
            _finite_vector(action["emitted_joint_velocity"], 7, "emitted velocity")
            == expected_clipped[:7],
            "Emitted velocity mismatch",
        )
        _require(
            _finite_number(action["emitted_gripper_closed"], "emitted gripper")
            == expected_clipped[7],
            "Emitted gripper mismatch",
        )
        q_before = _finite_vector(
            action["measured_joint_position_before"], 7, "q before"
        )
        dq_before = _finite_vector(
            action["measured_joint_velocity_before"], 7, "dq before"
        )
        gripper_before = _normalized_gripper_vector(
            action["measured_normalized_gripper_position_before"],
            "Pre-action normalized gripper",
        )
        normalized_gripper_min = min(normalized_gripper_min, gripper_before[0])
        normalized_gripper_max = max(normalized_gripper_max, gripper_before[0])
        is_terminal_apply_failure = (
            failure_sample_kind == "apply_entry" and step == actions_attempted - 1
        )
        if is_terminal_apply_failure:
            terminal_outcome = consume_failure_record(
                step=step,
                query_index=query_index,
                action_index=action_index,
                last_environment=environment_after,
            )
            failure = terminal_outcome["dynamic_report"]["terminal_velocity_failure"]
            if failure["physics_substep_index"] == 0:
                _validate_incident_bound_arm_state(
                    q_before,
                    dq_before,
                    terminal_outcome,
                    expected_sample_kind="apply_entry",
                    expected_physics_substep=0,
                    field=f"action {step} pre-state",
                )
            else:
                _validate_joint_state(q_before, dq_before, f"action {step} pre-state")
        else:
            _validate_joint_state(q_before, dq_before, f"action {step} pre-state")
        if action_index == 0:
            _require(
                q_before == active_query_q, "Query and emitted-action state mismatch"
            )
            _require(
                gripper_before == active_query_gripper,
                "Query and emitted-action gripper state mismatch",
            )
        if previous_q is not None:
            _require(
                q_before == previous_q and dq_before == previous_dq,
                "Measured state continuity mismatch",
            )
        if previous_gripper is not None:
            _require(
                gripper_before == previous_gripper,
                "Measured gripper state continuity mismatch",
            )

        if is_terminal_apply_failure:
            break

        execution = records[cursor]
        cursor += 1
        _require(set(execution) == EXECUTION_KEYS, f"Execution {step} schema mismatch")
        _require(
            execution["schema_version"] == PI05_DROID_NATIVE_TRACE_SCHEMA_VERSION,
            "Execution schema version mismatch",
        )
        _require(
            execution["record_type"] == "openpi_joint_velocity_execution",
            "Execution type mismatch",
        )
        _require(
            execution["profile"] == PI05_DROID_JOINTVELOCITY_PROFILE,
            "Execution profile mismatch",
        )
        _validate_contract_identity(execution, contract_identity, f"execution {step}")
        _require(execution["reset_index"] == 0, "Execution reset index mismatch")
        _require(
            execution["query_index"] == query_index, "Execution query index mismatch"
        )
        _require(
            execution["chunk_action_index"] == action_index,
            "Execution action index mismatch",
        )
        _require(
            execution["outer_step_index"] == step,
            "Execution outer step index mismatch",
        )
        _require(
            execution["terminated"] is False and execution["truncated"] is False,
            "Execution crossed a terminal or auto-reset boundary",
        )
        environment_after = _validate_environment_state(
            execution["environment_after"], f"execution {step} environment"
        )
        expected_episode_length = step + 1
        _require(
            environment_after["episode_length"] == expected_episode_length,
            "Execution environment episode length proves an auto-reset",
        )
        _require(
            environment_after["sim_step_counter"]
            == environment_before["sim_step_counter"]
            + expected_episode_length * PI05_DROID_NATIVE_DECIMATION
            and environment_after["common_step_counter"]
            == environment_before["common_step_counter"] + expected_episode_length,
            "Execution simulator counters do not match policy cadence",
        )
        for sensor_name in PI05_DROID_NATIVE_SENSOR_NAMES:
            _require(
                environment_after["sensor_frame_counters"][sensor_name]
                == environment_before["sensor_frame_counters"][sensor_name]
                + expected_episode_length,
                "Execution camera frame counter is stale or reset",
            )
        expected_target = [_float32(value) for value in expected_clipped[:7]]
        for key in ("processed_joint_velocity", "articulation_joint_velocity_target"):
            _require(
                _finite_vector(execution[key], 7, key) == expected_target,
                f"{key} differs from emitted velocity",
            )
        expected_finger = _float32(math.pi / 4.0) if expected_clipped[7] == 1.0 else 0.0
        for key in (
            "processed_finger_position_target",
            "articulation_finger_position_target",
        ):
            _require(
                _finite_vector(execution[key], 1, key) == [expected_finger],
                f"{key} differs from binary gripper target",
            )
        previous_q = _finite_vector(
            execution["measured_joint_position_after"], 7, "q after"
        )
        previous_dq = _finite_vector(
            execution["measured_joint_velocity_after"], 7, "dq after"
        )
        previous_gripper = _normalized_gripper_vector(
            execution["measured_normalized_gripper_position_after"],
            "Post-action normalized gripper",
        )
        normalized_gripper_min = min(normalized_gripper_min, previous_gripper[0])
        normalized_gripper_max = max(normalized_gripper_max, previous_gripper[0])
        is_terminal_post_failure = (
            failure_sample_kind == "post_policy_step" and step == actions_attempted - 1
        )
        if is_terminal_post_failure:
            terminal_outcome = consume_failure_record(
                step=step,
                query_index=query_index,
                action_index=action_index,
                last_environment=environment_after,
            )
            _validate_incident_bound_arm_state(
                previous_q,
                previous_dq,
                terminal_outcome,
                expected_sample_kind="post_policy_step",
                expected_physics_substep=8,
                field=f"execution {step} post-state",
            )
        else:
            _validate_joint_state(
                previous_q, previous_dq, f"execution {step} post-state"
            )
        for index, value in enumerate(previous_dq):
            max_abs_measured_velocity[index] = max(
                max_abs_measured_velocity[index], abs(value)
            )
        if is_terminal_post_failure:
            break

    if not metrics["numerical_failure"]:
        rollout_end = records[cursor]
        cursor += 1
        _require(set(rollout_end) == ROLLOUT_END_KEYS, "Rollout end schema mismatch")
        _require(
            rollout_end["schema_version"] == PI05_DROID_NATIVE_TRACE_SCHEMA_VERSION,
            "Rollout end schema version mismatch",
        )
        _require(
            rollout_end["record_type"] == "openpi_joint_velocity_rollout_end"
            and rollout_end["profile"] == PI05_DROID_JOINTVELOCITY_PROFILE
            and rollout_end["reset_index"] == 0,
            "Rollout end identity mismatch",
        )
        _validate_contract_identity(rollout_end, contract_identity, "rollout end")
        terminal_outcome = validate_terminal_rollout_evidence(
            rollout_end["terminal_rollout"], environment_runtime_contract
        )
        _require(
            terminal_outcome["environment_before"] == environment_before
            and terminal_outcome["environment_after"] == environment_after,
            "Terminal rollout counters differ from trace endpoints",
        )
        _require(
            terminal_outcome["rubric"]["success"] == metrics["success"]
            and terminal_outcome["rubric"]["progress"] == metrics["progress"],
            "Terminal post-action rubric differs from metrics",
        )

    _require(cursor == len(records), "Trace contains trailing records")
    _require(
        contract_identity is not None and terminal_outcome is not None,
        "Trace contains no terminal outcome",
    )
    return {
        "schema_version": 1,
        "status": "pass",
        "profile": PI05_DROID_JOINTVELOCITY_PROFILE,
        "environment": PI05_DROID_NATIVE_TASK,
        "metrics": metrics,
        "trace_path": str(Path(trace_path).resolve()),
        "trace_sha256": trace_digest.hexdigest(),
        "trace_record_count": len(records),
        "query_records": expected_queries,
        "action_records": actions_attempted,
        "execution_records": execution_records,
        "response_shape": [
            PI05_DROID_NATIVE_RESPONSE_HORIZON,
            PI05_DROID_NATIVE_ACTION_WIDTH,
        ],
        "execution_horizon": PI05_DROID_NATIVE_EXECUTION_HORIZON,
        "contract_identity": contract_identity,
        "environment_runtime_contract": environment_runtime_contract,
        "terminal_outcome": terminal_outcome,
        "sensor_liveness": {
            "status": "pass",
            "profile": PI05_DROID_NATIVE_SENSOR_LIVENESS_PROFILE,
            "sensor_names": list(PI05_DROID_NATIVE_SENSOR_NAMES),
            "counter_source": "isaaclab.sensors.camera.Camera.frame",
            "validated_outer_steps": execution_records,
            "required_counter_increment_per_outer_step": 1,
            "image_hash_variation_authoritative": False,
        },
        "image_order": ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb_masked"],
        "image_shape": [224, 224, 3],
        "wrist_rotation_degrees": 0,
        "distinct_external_query_frames": len(external_hashes),
        "distinct_wrist_query_frames": len(wrist_hashes),
        "max_abs_measured_joint_velocity": max_abs_measured_velocity,
        "measured_normalized_gripper_position_range": [
            normalized_gripper_min,
            normalized_gripper_max,
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = audit_trace(args.trace, args.metrics_csv)
    if args.output is not None:
        publish_immutable_json(args.output, summary)
    print(canonical_json_bytes(summary).decode("ascii"), end="")


if __name__ == "__main__":
    main()
