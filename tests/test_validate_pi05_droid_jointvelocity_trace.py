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


def _write_case(tmp_path: Path, records, *, episode_length=450):
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
        writer.writerow([0, episode_length, False, 0.25, False, ""])
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
    assert result["terminal_rollout"]["environment_after"]["episode_length"] == 450
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
