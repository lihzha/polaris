import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import struct

import pytest

from scripts.polaris import validate_pi05_droid_jointvelocity_trace as validator

from polaris.pi05_droid_jointvelocity_contract import (
    PI05_DROID_JOINTVELOCITY_PROFILE,
)
from polaris.pi05_droid_native_eval_contract import (
    PI05_DROID_NATIVE_CONFIGURED_EPISODE_LENGTH_SECONDS,
    PI05_DROID_NATIVE_EPISODE_STEPS,
    PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS,
    PI05_DROID_NATIVE_SENSOR_LIVENESS_PROFILE,
    PI05_DROID_NATIVE_TRACE_SCHEMA_VERSION,
    make_environment_runtime_contract,
    make_episode_sidecar,
    publish_immutable_file_from_temporary,
    publish_immutable_json,
    validate_episode_sidecar,
    validate_immutable_file,
)
from polaris.native_gripper_runtime import (
    EXPECTED_DROID_JOINT_NAMES,
    EXPECTED_FULL_LIMITS_CAPPED,
    NATIVE_ALL_JOINT_VELOCITY_FAILURE_PROFILE,
    NATIVE_GRIPPER_DYNAMIC_PROFILE,
    NativeAllJointVelocityLimitError,
    PHYSX_VELOCITY_LIMIT_ABSOLUTE_TOLERANCE_RAD_S,
)


Q = [0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0]
DQ = [0.0] * 7
ENVIRONMENT_RUNTIME = make_environment_runtime_contract(
    configured_episode_length_seconds=(
        PI05_DROID_NATIVE_CONFIGURED_EPISODE_LENGTH_SECONDS
    ),
    live_max_episode_length=PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS,
)
CONTRACT = {
    "serving_contract_sha256": "1" * 64,
    "serving_contract_artifact_sha256": "2" * 64,
    "serving_contract_artifact_size": 1234,
    "client_runtime_attestation_sha256": "3" * 64,
    "environment_runtime_sha256": ENVIRONMENT_RUNTIME["sha256"],
    "outer_episode_steps": PI05_DROID_NATIVE_EPISODE_STEPS,
    "internal_max_episode_length": PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS,
    "sensor_liveness_profile": PI05_DROID_NATIVE_SENSOR_LIVENESS_PROFILE,
}
ENVIRONMENT_BEFORE = {
    "live_max_episode_length": PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS,
    "episode_length": 0,
    "sim_step_counter": 17,
    "common_step_counter": 3,
    "sensor_frame_counters": {"external_cam": 1, "wrist_cam": 1},
}


def _float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _processed(raw):
    binary = [*raw[:7], 1.0 if raw[7] > 0.5 else 0.0]
    return binary, [min(1.0, max(-1.0, value)) for value in binary]


def _chunk(query_index):
    rows = []
    for row in range(15):
        phase = (query_index * 15 + row) % 7
        rows.append(
            [
                1.25 if phase == 0 else 0.01 * phase,
                -1.25 if phase == 1 else -0.01 * phase,
                0.02,
                -0.03,
                0.04,
                -0.05,
                0.06,
                0.75 if row % 2 else 0.5,
            ]
        )
    return rows


def _records():
    records = [
        {
            "schema_version": PI05_DROID_NATIVE_TRACE_SCHEMA_VERSION,
            "record_type": "openpi_joint_velocity_rollout_start",
            "profile": PI05_DROID_JOINTVELOCITY_PROFILE,
            **CONTRACT,
            "reset_index": 0,
            "environment_before": copy.deepcopy(ENVIRONMENT_BEFORE),
        }
    ]
    active_chunk = None
    for step in range(450):
        query_index = step // 8
        action_index = step % 8
        if action_index == 0:
            active_chunk = _chunk(query_index)
            planned = [_processed(row) for row in active_chunk[:8]]
            records.append(
                {
                    "schema_version": PI05_DROID_NATIVE_TRACE_SCHEMA_VERSION,
                    "record_type": "openpi_joint_velocity_query",
                    "profile": PI05_DROID_JOINTVELOCITY_PROFILE,
                    **CONTRACT,
                    "reset_index": 0,
                    "query_index": query_index,
                    "prompt": "Put all the foods in the bowl",
                    "state": {
                        "joint_position": Q,
                        "gripper_position": [0.25],
                    },
                    "images": {
                        "external": {
                            "shape": [224, 224, 3],
                            "dtype": "uint8",
                            "sha256": "4" * 64,
                        },
                        "wrist": {
                            "shape": [224, 224, 3],
                            "dtype": "uint8",
                            "sha256": "5" * 64,
                        },
                        "blank_masked_right_wrist": {
                            "shape": [224, 224, 3],
                            "dtype": "uint8",
                            "sha256": hashlib.sha256(bytes(224 * 224 * 3)).hexdigest(),
                        },
                        "model_order": [
                            "base_0_rgb",
                            "left_wrist_0_rgb",
                            "right_wrist_0_rgb_masked",
                        ],
                        "wrist_rotation_degrees": 0,
                    },
                    "response_action_shape": [15, 8],
                    "response_action_chunk_raw": active_chunk,
                    "execution_horizon": 8,
                    "planned_action_chunk_binary_gripper": [
                        item[0] for item in planned
                    ],
                    "planned_action_chunk_clipped": [item[1] for item in planned],
                }
            )
        assert active_chunk is not None
        raw = active_chunk[action_index]
        binary, clipped = _processed(raw)
        records.append(
            {
                "schema_version": PI05_DROID_NATIVE_TRACE_SCHEMA_VERSION,
                "record_type": "openpi_joint_velocity_action",
                "profile": PI05_DROID_JOINTVELOCITY_PROFILE,
                **CONTRACT,
                "reset_index": 0,
                "query_index": query_index,
                "chunk_action_index": action_index,
                "raw_action": raw,
                "binary_gripper_action": binary,
                "clipped_action": clipped,
                "emitted_joint_velocity": clipped[:7],
                "emitted_gripper_closed": clipped[7],
                "measured_joint_position_before": Q,
                "measured_joint_velocity_before": DQ,
                "measured_normalized_gripper_position_before": [0.25],
            }
        )
        finger = _float32(math.pi / 4.0) if clipped[7] == 1.0 else 0.0
        records.append(
            {
                "schema_version": PI05_DROID_NATIVE_TRACE_SCHEMA_VERSION,
                "record_type": "openpi_joint_velocity_execution",
                "profile": PI05_DROID_JOINTVELOCITY_PROFILE,
                **CONTRACT,
                "reset_index": 0,
                "query_index": query_index,
                "chunk_action_index": action_index,
                "outer_step_index": step,
                "terminated": False,
                "truncated": False,
                "environment_after": {
                    "live_max_episode_length": (
                        PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS
                    ),
                    "episode_length": step + 1,
                    "sim_step_counter": 17 + (step + 1) * 8,
                    "common_step_counter": 3 + step + 1,
                    "sensor_frame_counters": {
                        "external_cam": 1 + step + 1,
                        "wrist_cam": 1 + step + 1,
                    },
                },
                "processed_joint_velocity": [_float32(value) for value in clipped[:7]],
                "articulation_joint_velocity_target": [
                    _float32(value) for value in clipped[:7]
                ],
                "processed_finger_position_target": [finger],
                "articulation_finger_position_target": [finger],
                "measured_joint_position_after": Q,
                "measured_joint_velocity_after": DQ,
                "measured_normalized_gripper_position_after": [0.25],
            }
        )
    terminal = {
        "schema_version": 1,
        "profile": ENVIRONMENT_RUNTIME["profile"],
        "environment_runtime_sha256": ENVIRONMENT_RUNTIME["sha256"],
        "outer_steps_completed": PI05_DROID_NATIVE_EPISODE_STEPS,
        "last_outer_step_index": PI05_DROID_NATIVE_EPISODE_STEPS - 1,
        "terminated_false_count": PI05_DROID_NATIVE_EPISODE_STEPS,
        "truncated_false_count": PI05_DROID_NATIVE_EPISODE_STEPS,
        "environment_before": copy.deepcopy(ENVIRONMENT_BEFORE),
        "environment_after": copy.deepcopy(records[-1]["environment_after"]),
        "rubric": {"success": False, "progress": 0.25},
    }
    records.append(
        {
            "schema_version": PI05_DROID_NATIVE_TRACE_SCHEMA_VERSION,
            "record_type": "openpi_joint_velocity_rollout_end",
            "profile": PI05_DROID_JOINTVELOCITY_PROFILE,
            **CONTRACT,
            "reset_index": 0,
            "terminal_rollout": terminal,
        }
    )
    return records


def _failure_records(tmp_path, *, failed_step=238, substep=3):
    prefix = []
    action_count = 0
    for record in _records():
        prefix.append(record)
        if record["record_type"] == "openpi_joint_velocity_action":
            action_count += 1
            if action_count == failed_step + 1:
                break
    action = prefix[-1]
    completed = failed_step
    expected_limits = [float(_float32(value)) for value in EXPECTED_FULL_LIMITS_CAPPED]
    velocity = [0.0] * 13
    velocity[12] = 5.25
    thresholds = [
        value + PHYSX_VELOCITY_LIMIT_ABSOLUTE_TOLERANCE_RAD_S
        for value in expected_limits
    ]
    excess_mask = [
        abs(value) > threshold
        for value, threshold in zip(velocity, thresholds, strict=True)
    ]
    evidence = {
        "schema_version": 1,
        "profile": NATIVE_ALL_JOINT_VELOCITY_FAILURE_PROFILE,
        "reason": "measured_all_joint_velocity_limit_exceeded",
        "kind": "apply_entry",
        "joint_names": list(EXPECTED_DROID_JOINT_NAMES),
        "joint_indices": list(range(13)),
        "policy_step_index": completed,
        "physics_substep_index": substep,
        "failed_apply_call_index": completed * 8 + substep,
        "completed_apply_calls": completed * 8 + substep,
        "completed_policy_steps": completed,
        "joint_position": [0.0] * 13,
        "joint_velocity": velocity,
        "joint_acceleration": [0.0] * 13,
        "joint_velocity_target": [0.0] * 13,
        "joint_position_target": [0.0] * 13,
        "absolute_joint_velocity": [abs(value) for value in velocity],
        "expected_joint_velocity_limit": expected_limits,
        "live_joint_velocity_limit": expected_limits,
        "absolute_tolerance_rad_s": PHYSX_VELOCITY_LIMIT_ABSOLUTE_TOLERANCE_RAD_S,
        "effective_joint_velocity_threshold": thresholds,
        "excess_mask": excess_mask,
        "excess_rad_s": [
            max(abs(value) - threshold, 0.0)
            for value, threshold in zip(velocity, thresholds, strict=True)
        ],
        "violating_joint_indices": [12],
        "violating_joint_names": [EXPECTED_DROID_JOINT_NAMES[12]],
    }
    incident = publish_immutable_json(tmp_path / "incident.json", evidence)
    incident_identity = {
        key: incident[key] for key in ("path", "size", "sha256", "mode", "nlink")
    }
    error = NativeAllJointVelocityLimitError(evidence, incident_identity)
    reason = f"{type(error).__name__}: {error}"
    result = {
        "episode": 0,
        "episode_length": completed + 1,
        "success": False,
        "progress": 0.0,
        "numerical_failure": True,
        "numerical_failure_reason": reason,
    }
    last_environment = (
        copy.deepcopy(ENVIRONMENT_BEFORE)
        if completed == 0
        else copy.deepcopy(
            next(
                record["environment_after"]
                for record in reversed(prefix)
                if record["record_type"] == "openpi_joint_velocity_execution"
            )
        )
    )
    dynamic = {
        "schema_version": 2,
        "profile": NATIVE_GRIPPER_DYNAMIC_PROFILE,
        "joint_names": list(EXPECTED_DROID_JOINT_NAMES),
        "joint_indices": list(range(13)),
        "apply_calls": completed * 8 + substep,
        "post_policy_step_samples": completed,
        "sample_count": completed * 9 + substep,
        "max_abs_joint_velocity_rad_s": [0.0] * 13,
        "max_abs_joint_acceleration_rad_s2": [0.0] * 13,
        "terminal_velocity_failure": evidence,
        "samples": None,
    }
    after_failure = copy.deepcopy(last_environment)
    after_failure["sim_step_counter"] += substep + 1
    terminal = {
        "schema_version": 1,
        "profile": "openpi_pi05_droid_native_jointvelocity_numerical_failure_v1",
        "terminal_form": "native_all_joint_velocity_limit_failure",
        "environment_runtime_sha256": ENVIRONMENT_RUNTIME["sha256"],
        "failure_type": "NativeAllJointVelocityLimitError",
        "episode_result": result,
        "actions_attempted": completed + 1,
        "outer_steps_completed": completed,
        "failed_outer_step_index": completed,
        "terminated_false_count": completed,
        "truncated_false_count": completed,
        "environment_before": copy.deepcopy(ENVIRONMENT_BEFORE),
        "last_completed_environment": last_environment,
        "environment_after_failure": after_failure,
        "incident_artifact": incident_identity,
        "dynamic_report": dynamic,
    }
    prefix.append(
        {
            "schema_version": PI05_DROID_NATIVE_TRACE_SCHEMA_VERSION,
            "record_type": "openpi_joint_velocity_rollout_failure",
            "profile": PI05_DROID_JOINTVELOCITY_PROFILE,
            **CONTRACT,
            "reset_index": 0,
            "query_index": action["query_index"],
            "chunk_action_index": action["chunk_action_index"],
            "outer_step_index": completed,
            "terminal_failure": terminal,
        }
    )
    return prefix, result


def _write_case(
    tmp_path: Path,
    records,
    *,
    episode_length=450,
    numerical_failure=False,
    numerical_failure_reason="",
    progress=0.25,
):
    trace = tmp_path / "policy_traces.jsonl"
    with trace.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, separators=(",", ":"), allow_nan=False))
            output.write("\n")
    metrics = tmp_path / "eval_results.csv"
    with metrics.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "episode",
                "episode_length",
                "success",
                "progress",
                "numerical_failure",
                "numerical_failure_reason",
            ]
        )
        writer.writerow(
            [
                0,
                episode_length,
                False,
                progress,
                numerical_failure,
                numerical_failure_reason,
            ]
        )
    return trace, metrics


def test_full_450_step_execute8_trace_passes_and_binds_57_queries(tmp_path):
    trace, metrics = _write_case(tmp_path, _records())
    result = validator.audit_trace(trace, metrics)

    assert result["status"] == "pass"
    assert result["trace_record_count"] == 959
    assert result["query_records"] == 57
    assert result["action_records"] == 450
    assert result["execution_records"] == 450
    assert result["response_shape"] == [15, 8]
    assert result["execution_horizon"] == 8
    assert result["image_order"] == [
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb_masked",
    ]
    assert result["wrist_rotation_degrees"] == 0
    assert result["distinct_external_query_frames"] == 1
    assert result["distinct_wrist_query_frames"] == 1
    assert result["sensor_liveness"]["status"] == "pass"
    assert result["sensor_liveness"]["image_hash_variation_authoritative"] is False
    assert result["environment_runtime_contract"]["live_max_episode_length"] == 451
    assert result["terminal_outcome"]["environment_after"]["episode_length"] == 450
    assert result["measured_normalized_gripper_position_range"] == [0.25, 0.25]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda records: records[1]["images"].update(
                {
                    "model_order": [
                        "left_wrist_0_rgb",
                        "base_0_rgb",
                        "right_wrist_0_rgb_masked",
                    ]
                }
            ),
            "Model image order mismatch",
        ),
        (
            lambda records: records[1]["images"].update(
                {"wrist_rotation_degrees": 180}
            ),
            "Wrist image was rotated",
        ),
        (
            lambda records: records[18].update({"query_index": 2}),
            "Query index mismatch",
        ),
        (
            lambda records: records[3]["processed_joint_velocity"].__setitem__(0, 0.5),
            "processed_joint_velocity differs",
        ),
        (
            lambda records: records[4].update(
                {"measured_joint_position_before": [0.01, *Q[1:]]}
            ),
            "Measured state continuity mismatch",
        ),
        (
            lambda records: records[2].update({"serving_contract_sha256": "f" * 64}),
            "contract identity drift",
        ),
        (
            lambda records: records[4].update(
                {"measured_normalized_gripper_position_before": [0.5]}
            ),
            "Measured gripper state continuity mismatch",
        ),
        (
            lambda records: records[1]["images"]["blank_masked_right_wrist"].update(
                {"sha256": "f" * 64}
            ),
            "Masked blank image bytes changed",
        ),
    ],
)
def test_trace_semantic_tampering_fails_closed(tmp_path, mutation, message):
    records = _records()
    mutation(records)
    trace, metrics = _write_case(tmp_path, records)
    with pytest.raises(ValueError, match=message):
        validator.audit_trace(trace, metrics)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda execution: execution.update({"truncated": True}),
            "terminal or auto-reset boundary",
        ),
        (
            lambda execution: execution["environment_after"].update(
                {"episode_length": 0}
            ),
            "episode length proves an auto-reset",
        ),
        (
            lambda execution: execution["environment_after"][
                "sensor_frame_counters"
            ].update({"external_cam": 1}),
            "camera frame counter is stale or reset",
        ),
    ],
)
def test_step_450_auto_reset_or_stale_sensor_counter_is_rejected(
    tmp_path, mutation, message
):
    records = _records()
    terminal_execution = next(
        record
        for record in records
        if record["record_type"] == "openpi_joint_velocity_execution"
        and record["outer_step_index"] == 449
    )
    mutation(terminal_execution)
    trace, metrics = _write_case(tmp_path, records)
    with pytest.raises(ValueError, match=message):
        validator.audit_trace(trace, metrics)


def test_terminal_rubric_must_be_from_the_true_final_post_action_state(tmp_path):
    records = _records()
    records[-1]["terminal_rollout"]["rubric"]["progress"] = 0.75
    trace, metrics = _write_case(tmp_path, records)
    with pytest.raises(ValueError, match="rubric differs from metrics"):
        validator.audit_trace(trace, metrics)


def test_trace_rejects_prefix_or_failed_episode_metrics(tmp_path):
    records = copy.deepcopy(_records())
    trace, metrics = _write_case(tmp_path, records, episode_length=449)
    with pytest.raises(ValueError, match="complete all 450"):
        validator.audit_trace(trace, metrics)


def test_typed_partial_failure_trace_passes_and_binds_incident(tmp_path):
    records, result = _failure_records(tmp_path)
    trace, metrics = _write_case(
        tmp_path,
        records,
        episode_length=result["episode_length"],
        numerical_failure=True,
        numerical_failure_reason=result["numerical_failure_reason"],
        progress=0.0,
    )
    summary = validator.audit_trace(trace, metrics)
    assert summary["action_records"] == 239
    assert summary["execution_records"] == 238
    assert summary["query_records"] == 30
    assert summary["trace_record_count"] == 509
    assert summary["terminal_outcome"]["terminal_form"] == (
        "native_all_joint_velocity_limit_failure"
    )
    assert summary["terminal_outcome"]["dynamic_report"]["apply_calls"] == 1907
    trace.chmod(0o444)
    trace_artifact = validate_immutable_file(trace)
    video_temporary = tmp_path / ".video.partial"
    video_temporary.write_bytes(b"synthetic-video")
    video_artifact = publish_immutable_file_from_temporary(
        video_temporary, tmp_path / "episode_0.mp4"
    )
    terminal = summary["terminal_outcome"]
    sidecar_path = tmp_path / "episode_000000.json"
    publish_immutable_json(
        sidecar_path,
        make_episode_sidecar(
            episode_result=result,
            terminal_outcome=terminal,
            environment_runtime_contract=ENVIRONMENT_RUNTIME,
            dynamic_report=terminal["dynamic_report"],
            trace_artifact=trace_artifact,
            video_artifact=video_artifact,
            incident_artifact=terminal["incident_artifact"],
        ),
    )
    sidecar = validate_episode_sidecar(sidecar_path, ENVIRONMENT_RUNTIME)
    assert sidecar["value"]["episode_result"] == result
    assert sidecar["value"]["artifacts"]["incident"] == terminal["incident_artifact"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda terminal: terminal["dynamic_report"]["terminal_velocity_failure"][
                "excess_mask"
            ].__setitem__(12, False),
            "mask drift",
        ),
        (
            lambda terminal: terminal["dynamic_report"][
                "terminal_velocity_failure"
            ].update({"physics_substep_index": 2}),
            "cadence drift",
        ),
        (
            lambda terminal: terminal["environment_after_failure"].update(
                {
                    "sim_step_counter": terminal["environment_after_failure"][
                        "sim_step_counter"
                    ]
                    + 1
                }
            ),
            "simulator tail drift",
        ),
    ],
)
def test_typed_partial_failure_mutations_fail_closed(tmp_path, mutation, message):
    records, result = _failure_records(tmp_path)
    mutation(records[-1]["terminal_failure"])
    trace, metrics = _write_case(
        tmp_path,
        records,
        episode_length=result["episode_length"],
        numerical_failure=True,
        numerical_failure_reason=result["numerical_failure_reason"],
        progress=0.0,
    )
    with pytest.raises(ValueError, match=message):
        validator.audit_trace(trace, metrics)
