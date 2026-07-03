import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from polaris.eef_runtime_contract import build_terminal_rollout_evidence
from polaris.eef_runtime_contract import configure_ego_lap_environment_timeout
from polaris.eef_runtime_contract import validate_eef_outer_step_transition
from polaris.eef_runtime_contract import validate_ego_lap_runtime_protocol
from polaris.eef_runtime_contract import validate_terminal_rollout_evidence
from polaris.eval_artifacts import EGO_LAP_ENVIRONMENT_RUNTIME_PROFILE
from polaris.eval_artifacts import EGO_LAP_TRACE_PROFILE
from polaris.eval_artifacts import EGO_LAP_TRACE_SCHEMA_VERSION
from polaris.eval_artifacts import TRACE_QUERY_FIELDS
from polaris.eval_artifacts import validate_episode_trace


def _state(step: int, *, sim_start: int = 96, common_start: int = 12) -> dict:
    return {
        "profile": "isaaclab_single_env_episode_sim_common_camera_counters_v1",
        "live_max_episode_length": 451,
        "episode_length": step,
        "sim_step_counter": sim_start + 8 * step,
        "common_step_counter": common_start + step,
        "sensor_frame_counters": {
            "external_cam": 21 + step,
            "wrist_cam": 34 + step,
        },
    }


def _failure_state(completed: int, *, sim_tail: int = 1) -> dict:
    state = _state(completed)
    state["sim_step_counter"] += sim_tail
    return state


def _result(*, length: int = 450, numerical_failure: bool = False) -> dict:
    return {
        "episode": 0,
        "episode_length": length,
        "success": False,
        "progress": 0.0 if numerical_failure else 0.5,
        "numerical_failure": numerical_failure,
        "numerical_failure_reason": "guard" if numerical_failure else "",
    }


def _common(event: str) -> dict:
    return {
        "schema_version": EGO_LAP_TRACE_SCHEMA_VERSION,
        "trace_profile": EGO_LAP_TRACE_PROFILE,
        "timestamp": 1.0,
        "event": event,
        "episode": 0,
    }


def _query(query_index: int) -> dict:
    zero_chunk = [[0.0] * 7 for _ in range(16)]
    anchored_chunk = [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] for _ in range(16)]
    record = {field: None for field in TRACE_QUERY_FIELDS}
    record.update(_common("query"))
    record.update(
        {
            "query": query_index,
            "step": query_index * 8,
            "instruction": "perform the task",
            "checkpoint_profile": "original_lap_public_3b_v1",
            "checkpoint_path": "/checkpoints/LAP-3B",
            "contract_sha256": "0" * 64,
            "policy_type": "flow",
            "response_semantics": "cumulative_delta_targets",
            "execution_horizon": 8,
            "ar_endpoint_interpolation_profile": None,
            "ar_endpoint_interpolation_steps": None,
            "gripper_execution_profile": "binary_model_open_gt_0p5_else_closed_v1",
            "gripper_threshold": 0.5,
            "action_sampler_profile": "flow_explicit_euler_t1_to_t0_v1",
            "flow_num_steps": 10,
            "initial_rng_seed": 0,
            "ar_max_decoding_steps": None,
            "ar_temperature": None,
            "ar_stop_at_eos": None,
            "frame_description": "robot base frame",
            "eef_frame": "panda_link8",
            "numeric_action_frame": "robot_base",
            "normalization_scope": "category",
            "normalization_stats_sha256": "1" * 64,
            "normalization_profile": "q99_train_matched_v1",
            "normalization_compute_dtype": "float32",
            "normalization_input_formula": "q99_input_eps1e-8_clip_zero0_v1",
            "normalization_output_formula": (
                "q99_output_eps1e-8_zeroq01_extrapolate_v1"
            ),
            "normalization_formula_probe_sha256": "2" * 64,
            "state_layout": "xyz+r6_first_two_rows+gripper_open",
            "state_layout_mode": "public_lap_train_matched_rows_v1",
            "polaris_profile": "panda_link8_eef_pose_single_arm_v1",
            "anchor_position": [0.0, 0.0, 0.0],
            "anchor_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "state": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            "server_delta_chunk": zero_chunk,
            "raw_delta_chunk": zero_chunk,
            "base_delta_chunk": zero_chunk,
            "anchored_action_chunk": anchored_chunk,
            "reasoning": None,
        }
    )
    return record


def _write_trace(path: Path, *, length: int = 450, failure: bool = False) -> dict:
    result = _result(length=length, numerical_failure=failure)
    completed = length - int(failure)
    terminal = build_terminal_rollout_evidence(
        episode_result=result,
        environment_before=_state(0),
        environment_after=(
            _failure_state(completed, sim_tail=3) if failure else _state(completed)
        ),
        terminated_false_count=completed,
        truncated_false_count=completed,
    )
    records = [
        {
            **_common("reset"),
            "environment_runtime_profile": EGO_LAP_ENVIRONMENT_RUNTIME_PROFILE,
            "environment_before": _state(0),
        }
    ]
    for step in range(length):
        if step % 8 == 0:
            records.append(_query(step // 8))
        identity = {
            "query": step // 8,
            "step": step,
            "chunk_index": step % 8,
        }
        records.append(
            {
                **_common("action"),
                **identity,
                "raw_delta": [0.0] * 7,
                "polaris_action": [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            }
        )
        if failure and step == length - 1:
            records.append(
                {
                    **_common("execution_failure"),
                    **identity,
                    "numerical_failure_reason": "guard",
                }
            )
        else:
            transition = validate_eef_outer_step_transition(
                step_index=step,
                environment_before=_state(step),
                environment_after=_state(step + 1),
                terminated=np.array([False]),
                truncated=np.array([False]),
            )
            records.append(
                {**_common("execution"), **identity, "transition": transition}
            )
    records.append(
        {
            **_common("episode_complete"),
            **result,
            "status": "numerical_failure" if failure else "completed",
            "terminal_rollout": terminal,
        }
    )
    path.write_text(
        "".join(json.dumps(record, allow_nan=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return {"records": records, "result": result, "terminal": terminal}


def test_timeout_configuration_and_live_protocol_reserve_one_step():
    cfg = SimpleNamespace(
        sim=SimpleNamespace(dt=1.0 / 120.0),
        decimation=8,
        episode_length_s=30.0,
    )
    configured = configure_ego_lap_environment_timeout(cfg)
    assert configured["outer_episode_steps"] == 450
    assert configured["configured_internal_episode_steps"] == 451
    assert math.isclose(cfg.episode_length_s, 451.0 / 15.0)

    runtime = SimpleNamespace(
        max_episode_length=451,
        step_dt=1.0 / 15.0,
        physics_dt=1.0 / 120.0,
        cfg=cfg,
    )
    env = SimpleNamespace(unwrapped=runtime, max_episode_length=451)
    protocol = validate_ego_lap_runtime_protocol(env)
    assert protocol["episode_steps"] == 450
    assert protocol["live_max_episode_length"] == 451
    assert protocol["autoreset_margin_steps"] == 1


def test_outer_step_transition_binds_flags_all_counters_and_both_cameras():
    transition = validate_eef_outer_step_transition(
        step_index=0,
        environment_before=_state(0),
        environment_after=_state(1),
        terminated=np.array([False]),
        truncated=np.array([False]),
    )
    assert transition["counter_deltas"] == {
        "episode_length": 1,
        "sim_step_counter": 8,
        "common_step_counter": 1,
    }
    assert transition["camera_frame_deltas"] == {
        "external_cam": 1,
        "wrist_cam": 1,
    }

    with pytest.raises(ValueError, match="termination/timeout"):
        validate_eef_outer_step_transition(
            step_index=0,
            environment_before=_state(0),
            environment_after=_state(1),
            terminated=np.array([True]),
            truncated=np.array([False]),
        )
    camera_drift = _state(1)
    camera_drift["sensor_frame_counters"]["wrist_cam"] += 1
    with pytest.raises(ValueError, match="camera cadence"):
        validate_eef_outer_step_transition(
            step_index=0,
            environment_before=_state(0),
            environment_after=camera_drift,
            terminated=np.array([False]),
            truncated=np.array([False]),
        )


def test_terminal_rollout_binds_normal_and_failure_cadence():
    completed = build_terminal_rollout_evidence(
        episode_result=_result(),
        environment_before=_state(0),
        environment_after=_state(450),
        terminated_false_count=450,
        truncated_false_count=450,
    )
    assert completed["actions_attempted"] == 450
    assert completed["outer_steps_completed"] == 450
    assert completed["last_outer_step_index"] == 449
    assert completed["counter_deltas"]["sim_step_counter"] == 3600

    failed = build_terminal_rollout_evidence(
        episode_result=_result(length=17, numerical_failure=True),
        environment_before=_state(0),
        environment_after=_failure_state(16, sim_tail=3),
        terminated_false_count=16,
        truncated_false_count=16,
    )
    assert failed["actions_attempted"] == 17
    assert failed["outer_steps_completed"] == 16
    assert failed["last_outer_step_index"] == 15

    assert failed["counter_deltas"]["sim_step_counter"] == 16 * 8 + 3

    failed["counter_deltas"]["sim_step_counter"] += 1
    with pytest.raises(ValueError, match="snapshots disagree"):
        validate_terminal_rollout_evidence(failed)


def test_numerical_failure_terminal_rejects_stale_or_overlong_sim_tail():
    result = _result(length=17, numerical_failure=True)
    for environment_after in (_state(16), _failure_state(16, sim_tail=9)):
        with pytest.raises(ValueError, match="sim-counter tail"):
            build_terminal_rollout_evidence(
                episode_result=result,
                environment_before=_state(0),
                environment_after=environment_after,
                terminated_false_count=16,
                truncated_false_count=16,
            )


@pytest.mark.parametrize("field", ["episode_length", "common_step_counter"])
def test_numerical_failure_terminal_rejects_completed_counter_advance(field):
    after = _failure_state(16, sim_tail=1)
    after[field] += 1
    with pytest.raises(ValueError, match="hidden reset|non-sim counter"):
        build_terminal_rollout_evidence(
            episode_result=_result(length=17, numerical_failure=True),
            environment_before=_state(0),
            environment_after=after,
            terminated_false_count=16,
            truncated_false_count=16,
        )


def test_numerical_failure_terminal_rejects_camera_advance():
    after = _failure_state(16, sim_tail=1)
    after["sensor_frame_counters"]["external_cam"] += 1
    with pytest.raises(ValueError, match="camera"):
        build_terminal_rollout_evidence(
            episode_result=_result(length=17, numerical_failure=True),
            environment_before=_state(0),
            environment_after=after,
            terminated_false_count=16,
            truncated_false_count=16,
        )


def test_trace_v2_completed_execute8_has_exact_audited_959_records(tmp_path: Path):
    path = tmp_path / "episode_000000.jsonl"
    evidence = _write_trace(path)
    validated = validate_episode_trace(
        path,
        episode=0,
        expected_length=450,
        expected_result=evidence["result"],
    )
    events = [record["event"] for record in evidence["records"]]
    assert len(events) == 959
    assert events.count("reset") == 1
    assert events.count("query") == 57
    assert events.count("action") == 450
    assert events.count("execution") == 450
    assert events.count("execution_failure") == 0
    assert events.count("episode_complete") == 1
    assert validated["terminal_rollout"] == evidence["terminal"]


def test_trace_v2_numerical_failure_has_one_failed_execution(tmp_path: Path):
    path = tmp_path / "episode_000000.jsonl"
    evidence = _write_trace(path, length=17, failure=True)
    validate_episode_trace(
        path,
        episode=0,
        expected_length=17,
        expected_result=evidence["result"],
    )
    events = [record["event"] for record in evidence["records"]]
    assert len(events) == 39
    assert events.count("query") == 3
    assert events.count("action") == 17
    assert events.count("execution") == 16
    assert events.count("execution_failure") == 1


def test_trace_v2_rejects_query_moved_ahead_of_replan_boundary(tmp_path: Path):
    path = tmp_path / "episode_000000.jsonl"
    evidence = _write_trace(path, length=17, failure=True)
    records = evidence["records"]
    moved_index = next(
        index
        for index, record in enumerate(records)
        if record["event"] == "query" and record["query"] == 1
    )
    moved = records.pop(moved_index)
    records.insert(2, moved)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="query placement/cadence"):
        validate_episode_trace(
            path,
            episode=0,
            expected_length=17,
            expected_result=evidence["result"],
        )


def test_trace_v2_rejects_nested_transition_extra_key_and_broken_chain(
    tmp_path: Path,
):
    path = tmp_path / "episode_000000.jsonl"
    evidence = _write_trace(path, length=17, failure=True)
    execution = next(
        record for record in evidence["records"] if record["event"] == "execution"
    )
    execution["transition"]["extra"] = 1
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in evidence["records"]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="transition schema"):
        validate_episode_trace(
            path,
            episode=0,
            expected_length=17,
            expected_result=evidence["result"],
        )

    evidence = _write_trace(path, length=17, failure=True)
    second_execution = next(
        record
        for record in evidence["records"]
        if record["event"] == "execution" and record["step"] == 1
    )
    second_execution["transition"]["environment_before"]["common_step_counter"] += 1
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in evidence["records"]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="transition chain"):
        validate_episode_trace(
            path,
            episode=0,
            expected_length=17,
            expected_result=evidence["result"],
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda records: next(
                record for record in records if record["event"] == "query"
            )["state"].__setitem__(3, 0.0),
            "state/R6 recompute",
        ),
        (
            lambda records: next(
                record for record in records if record["event"] == "query"
            ).__setitem__("normalization_compute_dtype", "float64"),
            "normalization drift",
        ),
        (
            lambda records: next(
                record for record in records if record["event"] == "action"
            )["raw_delta"].__setitem__(0, 0.1),
            "action/raw-query binding",
        ),
        (
            lambda records: next(
                record for record in records if record["event"] == "action"
            )["polaris_action"].__setitem__(0, 0.1),
            "action/anchored-query binding",
        ),
    ],
)
def test_trace_v2_rejects_query_payload_and_action_binding_tamper(
    tmp_path: Path, mutation, match
):
    path = tmp_path / "episode_000000.jsonl"
    evidence = _write_trace(path, length=17, failure=True)
    mutation(evidence["records"])
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in evidence["records"]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=match):
        validate_episode_trace(
            path,
            episode=0,
            expected_length=17,
            expected_result=evidence["result"],
        )


def test_trace_v2_rejects_query_static_identity_and_failure_reason_drift(
    tmp_path: Path,
):
    path = tmp_path / "episode_000000.jsonl"
    evidence = _write_trace(path, length=17, failure=True)
    second_query = next(
        record
        for record in evidence["records"]
        if record["event"] == "query" and record["query"] == 1
    )
    second_query["instruction"] = "a different task"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in evidence["records"]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="query static identity"):
        validate_episode_trace(
            path,
            episode=0,
            expected_length=17,
            expected_result=evidence["result"],
        )

    evidence = _write_trace(path, length=17, failure=True)
    failure = next(
        record
        for record in evidence["records"]
        if record["event"] == "execution_failure"
    )
    failure["numerical_failure_reason"] = "different failure"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in evidence["records"]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="execution-failure identity"):
        validate_episode_trace(
            path,
            episode=0,
            expected_length=17,
            expected_result=evidence["result"],
        )
