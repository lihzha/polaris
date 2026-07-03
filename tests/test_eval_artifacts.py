import json
from pathlib import Path

import pandas as pd
import pytest

from polaris.eval_artifacts import atomic_write_episode_video
from polaris.eval_artifacts import atomic_write_results
from polaris.eval_artifacts import canonical_episode_result
from polaris.eval_artifacts import EGO_LAP_ENVIRONMENT_RUNTIME_PROFILE
from polaris.eval_artifacts import EGO_LAP_TRACE_PROFILE
from polaris.eval_artifacts import EGO_LAP_TRACE_SCHEMA_VERSION
from polaris.eval_artifacts import load_resume_results
from polaris.eval_artifacts import TRACE_QUERY_FIELDS
from polaris.eef_runtime_contract import build_terminal_rollout_evidence


def _state(step: int) -> dict:
    return {
        "profile": "isaaclab_single_env_episode_sim_common_camera_counters_v1",
        "live_max_episode_length": 451,
        "episode_length": step,
        "sim_step_counter": 8 * step,
        "common_step_counter": step,
        "sensor_frame_counters": {
            "external_cam": step,
            "wrist_cam": step,
        },
    }


def _common(event: str, episode: int) -> dict:
    return {
        "schema_version": EGO_LAP_TRACE_SCHEMA_VERSION,
        "trace_profile": EGO_LAP_TRACE_PROFILE,
        "timestamp": 1.0,
        "event": event,
        "episode": episode,
    }


def _query(episode: int, query_index: int) -> dict:
    zero_chunk = [[0.0] * 7 for _ in range(16)]
    anchored_chunk = [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] for _ in range(16)]
    record = {field: None for field in TRACE_QUERY_FIELDS}
    record.update(_common("query", episode))
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


def _write_trace(path: Path, *, episode: int, length: int) -> None:
    result = {
        "episode": episode,
        "episode_length": length,
        "success": False,
        "progress": 0.25,
        "numerical_failure": False,
        "numerical_failure_reason": "",
    }
    terminal = build_terminal_rollout_evidence(
        episode_result=result,
        environment_before=_state(0),
        environment_after=_state(length),
        terminated_false_count=length,
        truncated_false_count=length,
    )
    records = [
        {
            **_common("reset", episode),
            "environment_runtime_profile": EGO_LAP_ENVIRONMENT_RUNTIME_PROFILE,
            "environment_before": _state(0),
        },
    ]
    for step in range(length):
        if step % 8 == 0:
            records.append(_query(episode, step // 8))
        identity = {
            "query": step // 8,
            "step": step,
            "chunk_index": step % 8,
        }
        records.extend(
            [
                {
                    **_common("action", episode),
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
                },
                {
                    **_common("execution", episode),
                    **identity,
                    "transition": {
                        "step_index": step,
                        "terminated": False,
                        "truncated": False,
                        "environment_before": _state(step),
                        "environment_after": _state(step + 1),
                        "counter_deltas": {
                            "episode_length": 1,
                            "sim_step_counter": 8,
                            "common_step_counter": 1,
                        },
                        "camera_frame_deltas": {
                            "external_cam": 1,
                            "wrist_cam": 1,
                        },
                    },
                },
            ]
        )
    records.append(
        {
            **_common("episode_complete", episode),
            **result,
            "status": "completed",
            "terminal_rollout": terminal,
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _frame(episode: int = 0, length: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "episode": episode,
                "episode_length": length,
                "success": False,
                "progress": 0.25,
                "numerical_failure": False,
                "numerical_failure_reason": "",
            }
        ]
    )


def _video_probe(_path: Path, *, frames: int = 2):
    return {"frame_count": frames, "height": 224, "width": 448}


def test_resume_requires_contiguous_rows_video_and_finalized_trace(tmp_path: Path):
    csv_path = tmp_path / "eval_results.csv"
    _frame(length=450).to_csv(csv_path, index=False)
    (tmp_path / "episode_0.mp4").write_bytes(b"video")
    trace_dir = tmp_path / "policy_traces"
    _write_trace(trace_dir / "episode_000000.jsonl", episode=0, length=450)

    actual = load_resume_results(
        csv_path,
        run_folder=tmp_path,
        expected_rollouts=50,
        expected_horizon=450,
        require_episode_artifacts=True,
        trace_dir=trace_dir,
        video_probe_fn=lambda path: _video_probe(path, frames=450),
    )

    assert actual["episode"].tolist() == [0]


def test_resume_rejects_noncontiguous_episode_ids(tmp_path: Path):
    csv_path = tmp_path / "eval_results.csv"
    _frame(episode=1).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="contiguous prefix"):
        load_resume_results(
            csv_path,
            run_folder=tmp_path,
            expected_rollouts=50,
            expected_horizon=450,
            require_episode_artifacts=False,
        )


def test_resume_rejects_missing_video_and_incomplete_trace(tmp_path: Path):
    csv_path = tmp_path / "eval_results.csv"
    _frame(length=450).to_csv(csv_path, index=False)
    trace_dir = tmp_path / "policy_traces"
    _write_trace(trace_dir / "episode_000000.jsonl", episode=0, length=450)
    with pytest.raises(ValueError, match="Missing nonempty completed rollout video"):
        load_resume_results(
            csv_path,
            run_folder=tmp_path,
            expected_rollouts=50,
            expected_horizon=450,
            require_episode_artifacts=True,
            trace_dir=trace_dir,
            video_probe_fn=lambda path: _video_probe(path, frames=450),
        )

    (tmp_path / "episode_0.mp4").write_bytes(b"video")
    trace_path = trace_dir / "episode_000000.jsonl"
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    records = [
        record
        for record in records
        if not (record.get("event") == "action" and record.get("step") == 1)
    ]
    trace_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ValueError, match="action count"):
        load_resume_results(
            csv_path,
            run_folder=tmp_path,
            expected_rollouts=50,
            expected_horizon=450,
            require_episode_artifacts=True,
            trace_dir=trace_dir,
            video_probe_fn=lambda path: _video_probe(path, frames=450),
        )


def test_episode_video_and_csv_publish_atomically(tmp_path: Path):
    video_path = tmp_path / "episode_0.mp4"
    calls = []

    def writer(path, frames, *, fps):
        calls.append((path, len(frames), fps))
        path.write_bytes(b"complete-video")

    atomic_write_episode_video(
        video_path,
        [object(), object()],
        fps=15,
        writer=writer,
        probe_fn=_video_probe,
    )
    assert video_path.read_bytes() == b"complete-video"
    assert calls[0][1:] == (2, 15)
    assert not list(tmp_path.glob(".*.tmp.mp4"))

    csv_path = tmp_path / "eval_results.csv"
    atomic_write_results(_frame(), csv_path)
    assert pd.read_csv(csv_path)["episode"].tolist() == [0]
    assert not list(tmp_path.glob(".*.tmp.csv"))


def test_failed_video_publish_preserves_existing_final(tmp_path: Path):
    video_path = tmp_path / "episode_0.mp4"
    video_path.write_bytes(b"old-complete")

    def failing_writer(path, frames, *, fps):
        path.write_bytes(b"partial")
        raise RuntimeError("encoder failed")

    with pytest.raises(RuntimeError, match="encoder failed"):
        atomic_write_episode_video(
            video_path,
            [object()],
            fps=15,
            writer=failing_writer,
            probe_fn=lambda _path: {
                "frame_count": 1,
                "height": 224,
                "width": 448,
            },
        )
    assert video_path.read_bytes() == b"old-complete"
    assert not list(tmp_path.glob(".*.tmp.mp4"))


def test_numerical_failure_result_requires_zero_progress_and_no_success():
    result = _frame().iloc[0].to_dict()
    result["numerical_failure"] = True
    result["numerical_failure_reason"] = "controller abort"
    result["progress"] = 0.25
    with pytest.raises(ValueError, match="progress=0.0"):
        canonical_episode_result(result)

    result["progress"] = 0.0
    result["success"] = True
    with pytest.raises(ValueError, match="cannot be successful"):
        canonical_episode_result(result)
