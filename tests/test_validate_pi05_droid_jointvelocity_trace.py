import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import struct

import pytest

from scripts.polaris import (
    finalize_pi05_droid_native_jointvelocity_eval as finalizer,
)
from scripts.polaris import validate_pi05_droid_jointvelocity_trace as validator

from polaris import pi05_droid_native_eval_contract as native_eval_contract
from polaris.pi05_droid_jointvelocity_contract import (
    PI05_DROID_GRIPPER_OBSERVATION_BOUND_TOLERANCE,
    PI05_DROID_JOINTVELOCITY_PROFILE,
)
from polaris.pi05_droid_native_eval_contract import (
    PI05_DROID_NATIVE_CONFIGURED_EPISODE_LENGTH_SECONDS,
    PI05_DROID_NATIVE_EPISODE_STEPS,
    PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS,
    PI05_DROID_NATIVE_SENSOR_LIVENESS_PROFILE,
    PI05_DROID_NATIVE_TRACE_SCHEMA_VERSION,
    make_environment_runtime_contract,
    make_close_ready_artifact,
    make_episode_sidecar,
    publish_immutable_file_from_temporary,
    publish_immutable_json,
    validate_episode_sidecar,
    validate_immutable_file,
    validate_immutable_json,
)
from polaris.pi05_droid_native_lifecycle import NativeEvaluatorLifecycle
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
PANDA_VELOCITY_LIMITS = validator.PANDA_VELOCITY_LIMITS
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
                        "joint_position": Q.copy(),
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
                "measured_joint_position_before": Q.copy(),
                "measured_joint_velocity_before": DQ.copy(),
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
                "measured_joint_position_after": Q.copy(),
                "measured_joint_velocity_after": DQ.copy(),
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


def _set_normalized_gripper(records, value):
    for record in records:
        record_type = record["record_type"]
        if record_type == "openpi_joint_velocity_query":
            record["state"]["gripper_position"] = [value]
        elif record_type == "openpi_joint_velocity_action":
            record["measured_normalized_gripper_position_before"] = [value]
        elif record_type == "openpi_joint_velocity_execution":
            record["measured_normalized_gripper_position_after"] = [value]


def _failure_records(
    tmp_path,
    *,
    failed_step=238,
    sample_kind="apply_entry",
    substep=3,
    violating_joint_index=12,
    incident_path=None,
):
    assert sample_kind in {"apply_entry", "post_policy_step"}
    assert 0 <= violating_joint_index < 13
    if sample_kind == "post_policy_step":
        substep = 8
    prefix = []
    action_count = 0
    action = None
    for record in _records():
        prefix.append(record)
        if record["record_type"] == "openpi_joint_velocity_action":
            action_count += 1
            if action_count == failed_step + 1:
                action = record
                if sample_kind == "apply_entry":
                    break
        if (
            sample_kind == "post_policy_step"
            and record["record_type"] == "openpi_joint_velocity_execution"
            and record["outer_step_index"] == failed_step
        ):
            break
    assert action is not None
    attempts = failed_step + 1
    completed_post = failed_step
    completed_apply = (
        failed_step * 8 + substep if sample_kind == "apply_entry" else attempts * 8
    )
    expected_limits = [float(_float32(value)) for value in EXPECTED_FULL_LIMITS_CAPPED]
    joint_position = [*Q, *([0.0] * 6)]
    velocity = [*DQ, *([0.0] * 6)]
    velocity[violating_joint_index] = expected_limits[violating_joint_index] + 0.25
    thresholds = [
        value + PHYSX_VELOCITY_LIMIT_ABSOLUTE_TOLERANCE_RAD_S
        for value in expected_limits
    ]
    excess_mask = [
        abs(value) > threshold
        for value, threshold in zip(velocity, thresholds, strict=True)
    ]
    evidence = {
        "schema_version": 2,
        "profile": NATIVE_ALL_JOINT_VELOCITY_FAILURE_PROFILE,
        "reason": "measured_all_joint_velocity_limit_exceeded",
        "sample_kind": sample_kind,
        "joint_names": list(EXPECTED_DROID_JOINT_NAMES),
        "joint_indices": list(range(13)),
        "policy_step_index": completed_post,
        "physics_substep_index": substep,
        "failed_sample_index": completed_apply + completed_post,
        "completed_apply_calls": completed_apply,
        "completed_post_policy_step_samples": completed_post,
        "outer_step_physics_complete": sample_kind == "post_policy_step",
        "joint_position": joint_position,
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
        "violating_joint_indices": [violating_joint_index],
        "violating_joint_names": [EXPECTED_DROID_JOINT_NAMES[violating_joint_index]],
    }
    if sample_kind == "post_policy_step":
        execution = prefix[-1]
        assert execution["record_type"] == "openpi_joint_velocity_execution"
        execution["measured_joint_position_after"] = joint_position[:7]
        execution["measured_joint_velocity_after"] = velocity[:7]
    elif substep == 0:
        action["measured_joint_position_before"] = joint_position[:7]
        action["measured_joint_velocity_before"] = velocity[:7]
    incident = publish_immutable_json(
        Path(incident_path)
        if incident_path is not None
        else tmp_path / "incident.json",
        evidence,
    )
    incident_identity = {
        key: incident[key] for key in ("path", "size", "sha256", "mode", "nlink")
    }
    error = NativeAllJointVelocityLimitError(evidence, incident_identity)
    reason = f"{type(error).__name__}: {error}"
    result = {
        "episode": 0,
        "episode_length": attempts,
        "success": False,
        "progress": 0.0,
        "numerical_failure": True,
        "numerical_failure_reason": reason,
    }
    last_environment = (
        copy.deepcopy(ENVIRONMENT_BEFORE)
        if failed_step == 0 and sample_kind == "apply_entry"
        else copy.deepcopy(
            next(
                record["environment_after"]
                for record in reversed(prefix)
                if record["record_type"] == "openpi_joint_velocity_execution"
            )
        )
    )
    dynamic = {
        "schema_version": 3,
        "profile": NATIVE_GRIPPER_DYNAMIC_PROFILE,
        "joint_names": list(EXPECTED_DROID_JOINT_NAMES),
        "joint_indices": list(range(13)),
        "apply_calls": completed_apply,
        "post_policy_step_samples": completed_post,
        "sample_count": completed_apply + completed_post,
        "max_abs_joint_velocity_rad_s": [0.0] * 13,
        "max_abs_joint_acceleration_rad_s2": [0.0] * 13,
        "terminal_velocity_failure": evidence,
        "samples": None,
    }
    after_failure = copy.deepcopy(last_environment)
    if sample_kind == "apply_entry":
        after_failure["sim_step_counter"] += substep + 1
    completed_outer = failed_step + int(sample_kind == "post_policy_step")
    terminal = {
        "schema_version": 2,
        "profile": "openpi_pi05_droid_native_jointvelocity_numerical_failure_v1",
        "terminal_form": "native_all_joint_velocity_limit_failure",
        "environment_runtime_sha256": ENVIRONMENT_RUNTIME["sha256"],
        "failure_type": "NativeAllJointVelocityLimitError",
        "failure_sample_kind": sample_kind,
        "episode_result": result,
        "actions_attempted": attempts,
        "outer_steps_completed": completed_outer,
        "failed_outer_step_index": failed_step,
        "terminated_false_count": completed_outer,
        "truncated_false_count": completed_outer,
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
            "outer_step_index": failed_step,
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
    "value",
    [
        -PI05_DROID_GRIPPER_OBSERVATION_BOUND_TOLERANCE,
        1.0 + PI05_DROID_GRIPPER_OBSERVATION_BOUND_TOLERANCE,
    ],
)
def test_gripper_trace_exact_tolerance_boundaries_pass_and_remain_raw(tmp_path, value):
    records = _records()
    _set_normalized_gripper(records, value)
    trace, metrics = _write_case(tmp_path, records)

    result = validator.audit_trace(trace, metrics)
    assert result["measured_normalized_gripper_position_range"] == [value, value]


@pytest.mark.parametrize(
    "value",
    [
        math.nextafter(-PI05_DROID_GRIPPER_OBSERVATION_BOUND_TOLERANCE, -math.inf),
        math.nextafter(1.0 + PI05_DROID_GRIPPER_OBSERVATION_BOUND_TOLERANCE, math.inf),
    ],
)
def test_gripper_trace_nextafter_outside_tolerance_fails_closed(tmp_path, value):
    records = _records()
    _set_normalized_gripper(records, value)
    trace, metrics = _write_case(tmp_path, records)

    with pytest.raises(ValueError, match=r"outside \[0, 1\] plus"):
        validator.audit_trace(trace, metrics)


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


def test_terminal_failure_consumes_incident_from_stable_bound_read(
    tmp_path, monkeypatch
):
    original_root = tmp_path / "original"
    original_root.mkdir()
    wrong_root = tmp_path / "wrong"
    wrong_root.mkdir()
    alias_root = tmp_path / "alias"
    alias_root.symlink_to(original_root, target_is_directory=True)
    filename = "incident.json"
    records, _ = _failure_records(
        original_root,
        incident_path=original_root / filename,
    )
    terminal = records[-1]["terminal_failure"]
    terminal["incident_artifact"]["path"] = str(alias_root / filename)
    original_failure = copy.deepcopy(
        terminal["dynamic_report"]["terminal_velocity_failure"]
    )
    substituted_failure = copy.deepcopy(original_failure)
    substituted_failure["joint_position"][0] = 0.125
    substituted = publish_immutable_json(wrong_root / filename, substituted_failure)
    assert substituted["sha256"] != terminal["incident_artifact"]["sha256"]
    terminal["dynamic_report"]["terminal_velocity_failure"] = substituted_failure

    stable_validate = native_eval_contract.validate_bound_json_artifact
    retargeted = False

    def retarget_after_stable_read(*args, **kwargs):
        nonlocal retargeted
        artifact = stable_validate(*args, **kwargs)
        alias_root.unlink()
        alias_root.symlink_to(wrong_root, target_is_directory=True)
        retargeted = True
        return artifact

    monkeypatch.setattr(
        native_eval_contract,
        "validate_bound_json_artifact",
        retarget_after_stable_read,
    )
    with pytest.raises(ValueError, match="dynamic report incident drift"):
        native_eval_contract.validate_terminal_numerical_failure_evidence(
            terminal, ENVIRONMENT_RUNTIME
        )
    assert retargeted is True
    assert (alias_root / filename).resolve() == wrong_root / filename


@pytest.mark.parametrize(
    ("sample_kind", "failed_step", "substep", "expected_executions"),
    [
        ("post_policy_step", 238, 8, 239),
        ("apply_entry", 0, 0, 0),
    ],
)
def test_arm_joint5_terminal_state_is_exactly_cross_bound_to_incident(
    tmp_path, sample_kind, failed_step, substep, expected_executions
):
    records, result = _failure_records(
        tmp_path,
        failed_step=failed_step,
        sample_kind=sample_kind,
        substep=substep,
        violating_joint_index=4,
    )
    trace, metrics = _write_case(
        tmp_path,
        records,
        episode_length=result["episode_length"],
        numerical_failure=True,
        numerical_failure_reason=result["numerical_failure_reason"],
        progress=0.0,
    )
    summary = validator.audit_trace(trace, metrics)
    assert summary["execution_records"] == expected_executions
    failure = summary["terminal_outcome"]["dynamic_report"]["terminal_velocity_failure"]
    assert failure["violating_joint_indices"] == [4]
    assert failure["violating_joint_names"] == ["panda_joint5"]
    assert failure["joint_velocity"][4] > PANDA_VELOCITY_LIMITS[4]


@pytest.mark.parametrize(
    ("sample_kind", "state_key", "vector_key", "index", "message"),
    [
        (
            "post_policy_step",
            "openpi_joint_velocity_execution",
            "measured_joint_position_after",
            0,
            "position differs from immutable incident",
        ),
        (
            "post_policy_step",
            "openpi_joint_velocity_execution",
            "measured_joint_velocity_after",
            4,
            "velocity differs from immutable incident",
        ),
        (
            "apply_entry",
            "openpi_joint_velocity_action",
            "measured_joint_position_before",
            0,
            "position differs from immutable incident",
        ),
        (
            "apply_entry",
            "openpi_joint_velocity_action",
            "measured_joint_velocity_before",
            4,
            "velocity differs from immutable incident",
        ),
    ],
)
def test_terminal_arm_state_q_and_dq_mutations_fail_cross_binding(
    tmp_path, sample_kind, state_key, vector_key, index, message
):
    failed_step = 238 if sample_kind == "post_policy_step" else 0
    records, result = _failure_records(
        tmp_path,
        failed_step=failed_step,
        sample_kind=sample_kind,
        substep=0,
        violating_joint_index=4,
    )
    state = next(
        record for record in reversed(records) if record["record_type"] == state_key
    )
    state[vector_key][index] += 0.01
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


def test_apply_entry_substep_greater_than_zero_keeps_trace_pre_state_strict(tmp_path):
    records, result = _failure_records(
        tmp_path,
        failed_step=0,
        sample_kind="apply_entry",
        substep=1,
        violating_joint_index=4,
    )
    trace, metrics = _write_case(
        tmp_path,
        records,
        episode_length=result["episode_length"],
        numerical_failure=True,
        numerical_failure_reason=result["numerical_failure_reason"],
        progress=0.0,
    )
    assert validator.audit_trace(trace, metrics)["execution_records"] == 0

    records, result = _failure_records(
        tmp_path / "over-limit-pre-state",
        failed_step=0,
        sample_kind="apply_entry",
        substep=1,
        violating_joint_index=4,
    )
    action = next(
        record
        for record in reversed(records)
        if record["record_type"] == "openpi_joint_velocity_action"
    )
    action["measured_joint_velocity_before"][4] = PANDA_VELOCITY_LIMITS[4] + 0.25
    case_dir = tmp_path / "strict-mid-step"
    case_dir.mkdir()
    trace, metrics = _write_case(
        case_dir,
        records,
        episode_length=result["episode_length"],
        numerical_failure=True,
        numerical_failure_reason=result["numerical_failure_reason"],
        progress=0.0,
    )
    with pytest.raises(ValueError, match="pre-state joint 5 exceeds"):
        validator.audit_trace(trace, metrics)


def test_terminal_failure_index_kind_and_healthy_neighbor_mutations_fail(tmp_path):
    records, result = _failure_records(
        tmp_path,
        failed_step=238,
        sample_kind="post_policy_step",
        violating_joint_index=4,
    )
    records[-1]["outer_step_index"] = 237
    trace, metrics = _write_case(
        tmp_path,
        records,
        episode_length=result["episode_length"],
        numerical_failure=True,
        numerical_failure_reason=result["numerical_failure_reason"],
        progress=0.0,
    )
    with pytest.raises(ValueError, match="action identity mismatch"):
        validator.audit_trace(trace, metrics)

    kind_dir = tmp_path / "kind"
    kind_dir.mkdir()
    records, result = _failure_records(
        kind_dir,
        failed_step=238,
        sample_kind="post_policy_step",
        violating_joint_index=4,
    )
    records[-1]["terminal_failure"]["failure_sample_kind"] = "apply_entry"
    trace, metrics = _write_case(
        kind_dir,
        records,
        episode_length=result["episode_length"],
        numerical_failure=True,
        numerical_failure_reason=result["numerical_failure_reason"],
        progress=0.0,
    )
    with pytest.raises(ValueError, match="count mismatch"):
        validator.audit_trace(trace, metrics)

    neighbor_dir = tmp_path / "neighbor"
    neighbor_dir.mkdir()
    records, result = _failure_records(
        neighbor_dir,
        failed_step=238,
        sample_kind="post_policy_step",
        violating_joint_index=4,
    )
    neighboring_execution = next(
        record
        for record in records
        if record["record_type"] == "openpi_joint_velocity_execution"
        and record["outer_step_index"] == 237
    )
    neighboring_execution["measured_joint_velocity_after"][4] = (
        PANDA_VELOCITY_LIMITS[4] + 0.25
    )
    trace, metrics = _write_case(
        neighbor_dir,
        records,
        episode_length=result["episode_length"],
        numerical_failure=True,
        numerical_failure_reason=result["numerical_failure_reason"],
        progress=0.0,
    )
    with pytest.raises(ValueError, match="post-state joint 5 exceeds"):
        validator.audit_trace(trace, metrics)


def test_post_policy_failure_closes_full_artifact_transaction_and_cleanup(tmp_path):
    run_dir = tmp_path / "run"
    task_dir = run_dir / "DROID-FoodBussing"
    task_dir.mkdir(parents=True)
    records, result = _failure_records(
        task_dir,
        sample_kind="post_policy_step",
        violating_joint_index=4,
        incident_path=task_dir / "native_failures" / "episode_000000.json",
    )
    trace, metrics = _write_case(
        task_dir,
        records,
        episode_length=result["episode_length"],
        numerical_failure=True,
        numerical_failure_reason=result["numerical_failure_reason"],
        progress=0.0,
    )
    summary = validator.audit_trace(trace, metrics)
    assert finalizer.audit_trace(trace, metrics) == summary
    terminal = summary["terminal_outcome"]
    assert summary["action_records"] == 239
    assert summary["execution_records"] == 239
    assert summary["trace_record_count"] == 510
    assert terminal["failure_sample_kind"] == "post_policy_step"
    assert terminal["outer_steps_completed"] == 239
    assert terminal["dynamic_report"]["apply_calls"] == 1912
    assert terminal["dynamic_report"]["post_policy_step_samples"] == 238
    assert (
        terminal["last_completed_environment"] == terminal["environment_after_failure"]
    )

    trace.chmod(0o444)
    trace_artifact = validate_immutable_file(trace)
    metrics_temporary = metrics.with_name(".eval_results.partial.csv")
    metrics.replace(metrics_temporary)
    metrics_artifact = publish_immutable_file_from_temporary(metrics_temporary, metrics)
    assert metrics_artifact["path"] == str(metrics.resolve())
    video_temporary = task_dir / ".episode_0.partial.mp4"
    video_temporary.write_bytes(b"synthetic-post-policy-video")
    video_artifact = publish_immutable_file_from_temporary(
        video_temporary, task_dir / "episode_0.mp4"
    )
    sidecar_path = task_dir / "native_runtime" / "episode_000000.json"
    sidecar = publish_immutable_json(
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
    sidecar_identity = {
        key: sidecar[key] for key in ("path", "size", "sha256", "mode", "nlink")
    }
    runtime_path = task_dir / "joint_velocity_runtime.json"
    runtime = publish_immutable_json(runtime_path, {"runtime": "synthetic"})
    close_payload = make_close_ready_artifact(
        runtime_artifact=runtime,
        runtime_path=runtime_path,
        metrics_path=metrics,
        trace_path=trace,
        video_path=Path(video_artifact["path"]),
        environment_runtime_contract=ENVIRONMENT_RUNTIME,
        terminal_outcome=terminal,
        episode_sidecar=sidecar_identity,
    )

    events = []

    class _Closer:
        def __init__(self, name):
            self.name = name

        def close(self):
            events.append(self.name)

    close_path = task_dir / "evaluator_close_ready.json"

    def publish_close(path, payload):
        events.append("publish")
        return publish_immutable_json(path, payload)

    lifecycle = NativeEvaluatorLifecycle(_Closer("simulation.close"))
    lifecycle.bind_environment(_Closer("env.close"))
    lifecycle.prepare_close_ready(publish_close, close_path, close_payload)
    lifecycle.close()
    assert events == ["env.close", "publish", "simulation.close"]
    close = validate_immutable_json(close_path)["value"]
    assert close["terminal_outcome"] == terminal
    validated_sidecar = validate_episode_sidecar(sidecar_path, ENVIRONMENT_RUNTIME)
    assert validated_sidecar["value"]["episode_result"] == result
    runtime_for_finalizer = {
        **{key: runtime[key] for key in ("path", "size", "sha256", "mode", "nlink")},
        "environment_runtime_contract": ENVIRONMENT_RUNTIME,
    }
    finalized_close = finalizer._validate_close_ready(
        close_path, runtime_for_finalizer, run_dir
    )
    assert finalized_close["terminal_outcome"] == terminal
    assert finalized_close["episode_sidecar"]["value"] == validated_sidecar["value"]


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
            lambda terminal: terminal.update(
                {"failure_sample_kind": "post_policy_step"}
            ),
            "count mismatch",
        ),
        (
            lambda terminal: terminal["incident_artifact"].update({"sha256": "0" * 64}),
            "artifact identity drift",
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
