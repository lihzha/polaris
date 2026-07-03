import ast
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from polaris.eef_runtime_contract import atomic_write_runtime_contract
from polaris.eef_runtime_contract import atomic_write_episode_safety
from polaris.eef_runtime_contract import aggregate_episode_safety
from polaris.eef_runtime_contract import load_episode_safety_sidecars
from polaris.eef_runtime_contract import reconcile_episode_safety_transactions
from polaris.eef_runtime_contract import validate_episode_safety_cadence
from polaris.eef_runtime_contract import validate_eef_runtime_frame
from polaris.eef_runtime_contract import validate_eef_runtime_safety
from polaris.eef_runtime_contract import validate_ego_lap_runtime_protocol
from polaris.eef_ik_safety import EEF_IK_APPLY_CADENCE
from polaris.eef_ik_safety import EEF_IK_SAFETY_PROFILE
from polaris.eef_ik_safety import EEF_QUATERNION_UNIT_NORM_TOLERANCE
from polaris.eef_ik_safety import JOINT_SLEW_FLOAT32_TOLERANCE_RAD
from polaris.eef_ik_safety import PANDA_EEF_JOINT_EFFORT_LIMITS
from polaris.eef_ik_safety import PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S
from polaris.eef_ik_safety import PANDA_SOFT_JOINT_POS_LIMITS_FLOAT32_SHA256
from polaris.eef_ik_safety import PANDA_SOFT_JOINT_POS_LIMITS_RAD
from polaris.eef_ik_safety import validate_one_step_adversarial_report
from polaris.gripper_semantics import GRIPPER_THRESHOLD_PROFILE
from polaris.eval_artifacts import build_episode_artifact_identity
from polaris.eval_artifacts import empty_eval_results


def _wxyz(rotation: Rotation) -> np.ndarray:
    return rotation.as_quat()[[3, 0, 1, 2]]


def _runtime_fixture():
    link0_rotation = Rotation.from_euler("z", 20, degrees=True)
    relative_rotation = Rotation.from_euler("xyz", [10, -5, 30], degrees=True)
    link8_rotation = link0_rotation * relative_rotation
    link0_position = np.array([0.1, -0.2, 0.3])
    relative_position = np.array([0.4, 0.05, 0.2])
    link8_position = link0_position + link0_rotation.apply(relative_position)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_names=["panda_link0", "panda_link8"],
            body_pos_w=np.array([[link0_position, link8_position]]),
            body_quat_w=np.array([[_wxyz(link0_rotation), _wxyz(link8_rotation)]]),
        )
    )
    offset = SimpleNamespace(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0))
    controller = SimpleNamespace(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.01},
    )
    arm_term = SimpleNamespace(
        cfg=SimpleNamespace(
            body_name="panda_link8",
            body_offset=offset,
            controller=controller,
            scale=1.0,
        ),
        action_dim=7,
        _body_idx=1,
        _joint_names=[f"panda_joint{index}" for index in range(1, 8)],
    )
    max_delta = [value / 120.0 for value in PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S]
    soft_limits = [list(values) for values in PANDA_SOFT_JOINT_POS_LIMITS_RAD]
    soft_limit_sha256 = PANDA_SOFT_JOINT_POS_LIMITS_FLOAT32_SHA256
    arm_term.safety_report = lambda: {
        "episode_index": None,
        "profile": EEF_IK_SAFETY_PROFILE,
        "apply_actions_cadence": EEF_IK_APPLY_CADENCE,
        "physics_dt": 1.0 / 120.0,
        "control_dt": 1.0 / 15.0,
        "decimation": 8,
        "current_joint_soft_limit_tolerance_rad": 1e-5,
        "eef_quaternion_unit_norm_tolerance": EEF_QUATERNION_UNIT_NORM_TOLERANCE,
        "joint_slew_float32_tolerance_rad": JOINT_SLEW_FLOAT32_TOLERANCE_RAD,
        "soft_joint_pos_limit_factor": 1.0,
        "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
        "joint_velocity_limits_rad_s": list(PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S),
        "joint_effort_limits": list(PANDA_EEF_JOINT_EFFORT_LIMITS),
        "max_delta_joint_pos_rad": max_delta,
        "soft_joint_pos_limits_rad": soft_limits,
        "soft_joint_pos_limits_float32_sha256": soft_limit_sha256,
        "counters": {
            "apply_calls": 0,
            "environment_substeps": 0,
            "slew_limit_events": 0,
            "slew_limited_joints": 0,
            "position_limit_events": 0,
            "position_limited_joints": 0,
            "post_clamp_target_violations": 0,
            "current_joint_limit_aborts": 0,
            "invariant_aborts": 0,
            "nonfinite_aborts": 0,
            "dls_fallbacks": 0,
            "guard_diagnostics_dropped": 0,
        },
        "maxima": {
            "raw_delta_joint_pos_rad": [0.0] * 7,
            "applied_delta_joint_pos_rad": [0.0] * 7,
            "raw_target_soft_limit_violation_rad": [0.0] * 7,
            "post_clamp_target_soft_limit_violation_rad": [0.0] * 7,
            "current_joint_soft_limit_violation_rad": [0.0] * 7,
        },
        "guard_diagnostics": [],
        "max_raw_delta_diagnostic": None,
    }
    finger_term = SimpleNamespace(gripper_threshold_profile=GRIPPER_THRESHOLD_PROFILE)
    runtime = SimpleNamespace(
        max_episode_length=450,
        step_dt=1.0 / 15.0,
        physics_dt=1.0 / 120.0,
        cfg=SimpleNamespace(sim=SimpleNamespace(dt=1.0 / 120.0), decimation=8),
        scene={"robot": robot},
        action_manager=SimpleNamespace(
            _terms={"arm": arm_term, "finger_joint": finger_term}
        ),
    )
    env = SimpleNamespace(unwrapped=runtime, max_episode_length=450)
    observation = {
        "policy": {
            "eef_pos": relative_position[None, :],
            "eef_quat": _wxyz(relative_rotation)[None, :],
        }
    }
    return env, observation


def _episode_result(*, episode=0, length=2, numerical_failure=False):
    return {
        "episode": episode,
        "episode_length": length,
        "success": False,
        "progress": 0.0 if numerical_failure else 0.25,
        "numerical_failure": numerical_failure,
        "numerical_failure_reason": (
            "DifferentialIKNumericalError: guard" if numerical_failure else ""
        ),
    }


def _episode_safety(*, episode=0, length=2, numerical_failure=False):
    env, _ = _runtime_fixture()
    report = env.unwrapped.action_manager._terms["arm"].safety_report()
    report["episode_index"] = episode
    apply_calls = length * 8 if not numerical_failure else (length - 1) * 8 + 3
    report["counters"]["apply_calls"] = apply_calls
    report["counters"]["environment_substeps"] = apply_calls
    zero_vector = {
        "values": [0.0] * 7,
        "finite_mask": [True] * 7,
        "finite_count": 7,
    }
    report["max_raw_delta_diagnostic"] = {
        "kind": "max_raw_delta",
        "episode_index": episode,
        "policy_step": 0,
        "physics_substep": 0,
        "joint_pos_rad": zero_vector,
        "raw_delta_joint_pos_rad": zero_vector,
        "raw_joint_pos_target_rad": zero_vector,
        "safe_joint_pos_target_rad": zero_vector,
        "pose_error_norm": 0.0,
        "jacobian_finite": True,
        "jacobian_max_abs": 0.0,
        "eef_quaternion_norm": None,
    }
    if numerical_failure:
        report["counters"]["nonfinite_aborts"] = 1
        report["guard_diagnostics"] = [
            {
                "kind": "nonfinite_abort",
                "episode_index": episode,
                "policy_step": length - 1,
                "physics_substep": 2,
                "joint_pos_rad": None,
                "raw_delta_joint_pos_rad": None,
                "raw_joint_pos_target_rad": None,
                "safe_joint_pos_target_rad": None,
                "pose_error_norm": None,
                "jacobian_finite": None,
                "jacobian_max_abs": None,
                "eef_quaternion_norm": None,
            }
        ]
    return report


def _write_completed_trace(path: Path, result):
    records = [
        {"event": "reset", "episode": result["episode"]},
        *(
            {"event": "action", "episode": result["episode"], "step": step}
            for step in range(result["episode_length"])
        ),
        {
            "event": "episode_complete",
            **result,
            "status": (
                "numerical_failure" if result["numerical_failure"] else "completed"
            ),
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _video_probe(_path: Path, *, frames=2):
    return {"frame_count": frames, "height": 224, "width": 448}


def _prepare_episode_transaction(tmp_path: Path, *, episode: int):
    result = _episode_result(episode=episode)
    safety = _episode_safety(episode=episode)
    video_path = tmp_path / f"episode_{episode}.mp4"
    trace_dir = tmp_path / "policy_traces"
    trace_path = trace_dir / f"episode_{episode:06d}.jsonl"
    video_path.write_bytes(f"complete-video-{episode}".encode())
    _write_completed_trace(trace_path, result)
    identity = build_episode_artifact_identity(
        run_folder=tmp_path,
        trace_path=trace_path,
        episode_result=result,
        video_probe_fn=_video_probe,
    )
    sidecar_path = tmp_path / "ik_safety" / f"episode_{episode:06d}.json"
    payload = atomic_write_episode_safety(
        sidecar_path,
        episode_index=episode,
        episode_result=result,
        safety=safety,
        artifact_identity=identity,
    )
    return result, sidecar_path, payload


def test_runtime_protocol_requires_exact_450_steps_at_15hz():
    env, _ = _runtime_fixture()
    resolved = validate_ego_lap_runtime_protocol(env)
    assert resolved["episode_steps"] == 450
    assert resolved["policy_hz"] == 15.0
    assert resolved["physics_hz"] == 120.0
    assert resolved["decimation"] == 8

    env.max_episode_length = 449
    with pytest.raises(ValueError, match="450"):
        validate_ego_lap_runtime_protocol(env)
    env.max_episode_length = 450
    env.unwrapped.step_dt = 1.0 / 10.0
    with pytest.raises(ValueError, match="15 Hz"):
        validate_ego_lap_runtime_protocol(env)


def test_runtime_frame_matches_direct_link8_and_absolute_action_term():
    env, observation = _runtime_fixture()
    result = validate_eef_runtime_frame(env, observation)
    assert result["eef_frame"] == "panda_link8"
    assert result["position_error_m"] < 1e-12
    assert result["rotation_error_rad"] < 1e-12
    assert result["reference_frame"] == "panda_link0"
    assert result["controlled_body"] == "panda_link8"
    assert result["body_offset"] == "identity"
    assert result["command_type"] == "pose"
    assert result["use_relative_mode"] is False
    assert result["ik_method"] == "dls"
    assert result["dls_damping"] == 0.01
    assert result["arm_scale"] == 1.0
    assert result["arm_joint_names"] == [f"panda_joint{index}" for index in range(1, 8)]
    assert result["gripper_threshold_profile"] == GRIPPER_THRESHOLD_PROFILE
    assert result["ik_safety_profile"] == EEF_IK_SAFETY_PROFILE
    assert result["action_dim"] == 7

    safety = validate_eef_runtime_safety(env)
    assert safety["profile"] == EEF_IK_SAFETY_PROFILE
    assert safety["max_delta_joint_pos_rad"] == [
        value / 120.0 for value in PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S
    ]


def test_runtime_contract_is_atomic_and_has_exact_evidence_schema():
    env, observation = _runtime_fixture()
    protocol = validate_ego_lap_runtime_protocol(env)
    frame = validate_eef_runtime_frame(env, observation)
    safety = validate_eef_runtime_safety(env)
    aggregate_safety = aggregate_episode_safety(safety, [])

    with tempfile.TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / "nested" / "runtime.json"
        path.parent.mkdir(parents=True)
        path.write_text('{"stale": true}\n', encoding="utf-8")
        atomic_write_runtime_contract(
            path, protocol=protocol, frame=frame, ik_safety=aggregate_safety
        )
        payload = json.loads(path.read_text(encoding="utf-8"))

        assert payload == {
            "schema_version": 2,
            "protocol": {
                "episode_steps": 450,
                "policy_hz": 15.0,
                "step_dt": 1.0 / 15.0,
                "physics_hz": 120.0,
                "physics_dt": 1.0 / 120.0,
                "decimation": 8,
            },
            "frame": frame,
            "ik_safety": aggregate_safety,
        }
        assert not list(path.parent.glob(".*.tmp"))


def test_prepared_sidecar_recovers_exact_missing_csv_row(tmp_path: Path):
    result = _episode_result()
    safety = _episode_safety()
    video_path = tmp_path / "episode_0.mp4"
    trace_dir = tmp_path / "policy_traces"
    trace_path = trace_dir / "episode_000000.jsonl"
    video_path.write_bytes(b"complete-video")
    _write_completed_trace(trace_path, result)
    identity = build_episode_artifact_identity(
        run_folder=tmp_path,
        trace_path=trace_path,
        episode_result=result,
        video_probe_fn=_video_probe,
    )
    sidecar_path = tmp_path / "ik_safety" / "episode_000000.json"
    atomic_write_episode_safety(
        sidecar_path,
        episode_index=0,
        episode_result=result,
        safety=safety,
        artifact_identity=identity,
    )

    recovered, changed = reconcile_episode_safety_transactions(
        empty_eval_results(),
        directory=sidecar_path.parent,
        run_folder=tmp_path,
        trace_dir=trace_dir,
        expected_rollouts=50,
        expected_horizon=450,
        video_probe_fn=_video_probe,
    )

    assert changed is True
    assert recovered.to_dict(orient="records") == [result]
    assert sidecar_path.is_file()


def test_transaction_recovery_is_idempotent_after_csv_commit(tmp_path: Path):
    result, sidecar_path, payload = _prepare_episode_transaction(tmp_path, episode=0)
    committed, changed = reconcile_episode_safety_transactions(
        pd.DataFrame([result]),
        directory=sidecar_path.parent,
        run_folder=tmp_path,
        trace_dir=tmp_path / "policy_traces",
        expected_rollouts=50,
        expected_horizon=450,
        video_probe_fn=_video_probe,
    )
    assert changed is False
    assert committed.to_dict(orient="records") == [result]
    assert json.loads(sidecar_path.read_text()) == payload

    drifted_safety = _episode_safety()
    drifted_safety["counters"]["slew_limit_events"] = 1
    drifted_safety["counters"]["slew_limited_joints"] = 1
    with pytest.raises(ValueError, match="Refusing to overwrite drifted"):
        atomic_write_episode_safety(
            sidecar_path,
            episode_index=0,
            episode_result=result,
            safety=drifted_safety,
            artifact_identity=payload["artifact_identity"],
        )


def test_transaction_recovery_rejects_missing_or_multiple_prepared_sidecars(
    tmp_path: Path,
):
    result = _episode_result()
    with pytest.raises(ValueError, match="must equal the CSV prefix"):
        reconcile_episode_safety_transactions(
            pd.DataFrame([result]),
            directory=tmp_path / "ik_safety",
            run_folder=tmp_path,
            trace_dir=tmp_path / "policy_traces",
            expected_rollouts=50,
            expected_horizon=450,
            video_probe_fn=_video_probe,
        )

    _prepare_episode_transaction(tmp_path, episode=0)
    _prepare_episode_transaction(tmp_path, episode=1)
    with pytest.raises(ValueError, match="add exactly its next episode"):
        reconcile_episode_safety_transactions(
            empty_eval_results(),
            directory=tmp_path / "ik_safety",
            run_folder=tmp_path,
            trace_dir=tmp_path / "policy_traces",
            expected_rollouts=50,
            expected_horizon=450,
            video_probe_fn=_video_probe,
        )


def test_transaction_recovery_archives_uncommitted_artifacts_without_deleting_evidence(
    tmp_path: Path,
):
    video = tmp_path / "episode_0.mp4"
    trace = tmp_path / "policy_traces" / "episode_000000.jsonl"
    temporary_video = tmp_path / ".episode_0.tmp.mp4"
    trace.parent.mkdir(parents=True)
    video.write_bytes(b"uncommitted-video")
    trace.write_text("partial trace evidence\n")
    temporary_video.write_bytes(b"partial-video-evidence")

    frame, changed = reconcile_episode_safety_transactions(
        empty_eval_results(),
        directory=tmp_path / "ik_safety",
        run_folder=tmp_path,
        trace_dir=trace.parent,
        expected_rollouts=50,
        expected_horizon=450,
        video_probe_fn=_video_probe,
    )

    assert frame.empty
    assert changed is False
    assert not video.exists()
    assert not trace.exists()
    assert not temporary_video.exists()
    archived = list((tmp_path / "recovery_orphans" / "episode_000000").iterdir())
    assert len(archived) == 3
    assert sorted(path.read_bytes() for path in archived) == sorted(
        [
            b"uncommitted-video",
            b"partial trace evidence\n",
            b"partial-video-evidence",
        ]
    )


def test_runtime_aggregate_reconstructs_all_resume_history(tmp_path: Path):
    for episode in range(2):
        _prepare_episode_transaction(tmp_path, episode=episode)
    sidecars = load_episode_safety_sidecars(tmp_path / "ik_safety", [0, 1])
    env, _ = _runtime_fixture()
    live = env.unwrapped.action_manager._terms["arm"].safety_report()
    aggregate = aggregate_episode_safety(live, sidecars)

    assert aggregate["episodes_completed"] == 2
    assert aggregate["counters"]["apply_calls"] == 32
    assert aggregate["counters"]["environment_substeps"] == 32
    assert [item["episode_index"] for item in aggregate["episodes"]] == [0, 1]
    assert all(item["sidecar_sha256"] for item in aggregate["episodes"])


def test_prepared_sidecar_recovery_rejects_csv_and_trace_drift(tmp_path: Path):
    result = _episode_result()
    safety = _episode_safety()
    (tmp_path / "episode_0.mp4").write_bytes(b"complete-video")
    trace_dir = tmp_path / "policy_traces"
    trace_path = trace_dir / "episode_000000.jsonl"
    _write_completed_trace(trace_path, result)
    identity = build_episode_artifact_identity(
        run_folder=tmp_path,
        trace_path=trace_path,
        episode_result=result,
        video_probe_fn=_video_probe,
    )
    sidecar_path = tmp_path / "ik_safety" / "episode_000000.json"
    atomic_write_episode_safety(
        sidecar_path,
        episode_index=0,
        episode_result=result,
        safety=safety,
        artifact_identity=identity,
    )

    drifted_row = {**result, "progress": 0.5}
    with pytest.raises(ValueError, match="CSV row differs"):
        reconcile_episode_safety_transactions(
            pd.DataFrame([drifted_row]),
            directory=sidecar_path.parent,
            run_folder=tmp_path,
            trace_dir=trace_dir,
            expected_rollouts=50,
            expected_horizon=450,
            video_probe_fn=_video_probe,
        )

    trace_path.write_text(trace_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact identity drift"):
        reconcile_episode_safety_transactions(
            empty_eval_results(),
            directory=sidecar_path.parent,
            run_folder=tmp_path,
            trace_dir=trace_dir,
            expected_rollouts=50,
            expected_horizon=450,
            video_probe_fn=_video_probe,
        )


def test_episode_safety_cadence_binds_success_and_failure_substep():
    completed = validate_episode_safety_cadence(
        safety=_episode_safety(), episode_result=_episode_result()
    )
    assert completed["apply_calls"] == 16
    assert completed["failed_policy_step"] is None

    failed = validate_episode_safety_cadence(
        safety=_episode_safety(numerical_failure=True),
        episode_result=_episode_result(numerical_failure=True),
    )
    assert failed["apply_calls"] == 11
    assert failed["failed_policy_step"] == 1
    assert failed["failed_physics_substep"] == 2

    invalid = _episode_safety()
    invalid["counters"]["apply_calls"] -= 1
    invalid["counters"]["environment_substeps"] -= 1
    with pytest.raises(ValueError, match="Completed episode controller cadence"):
        validate_episode_safety_cadence(
            safety=invalid, episode_result=_episode_result()
        )


def test_episode_safety_rejects_schema_counter_and_unknown_abort_tamper():
    extra = _episode_safety()
    extra["unexpected"] = True
    with pytest.raises(ValueError, match="schema drift"):
        validate_episode_safety_cadence(safety=extra, episode_result=_episode_result())

    mismatched = _episode_safety(numerical_failure=True)
    mismatched["counters"]["nonfinite_aborts"] = 2
    with pytest.raises(ValueError, match="counter history is impossible"):
        validate_episode_safety_cadence(
            safety=mismatched,
            episode_result=_episode_result(numerical_failure=True),
        )

    unknown = _episode_safety(numerical_failure=True)
    unknown["guard_diagnostics"][0]["kind"] = "unknown_abort"
    with pytest.raises(ValueError, match="kind is not allowed"):
        validate_episode_safety_cadence(
            safety=unknown,
            episode_result=_episode_result(numerical_failure=True),
        )

    dropped = _episode_safety()
    dropped["counters"]["guard_diagnostics_dropped"] = 1
    with pytest.raises(ValueError, match="dropped durable"):
        validate_episode_safety_cadence(
            safety=dropped, episode_result=_episode_result()
        )


def test_completed_episode_requires_consistent_max_raw_diagnostic():
    missing = _episode_safety()
    missing["max_raw_delta_diagnostic"] = None
    with pytest.raises(ValueError, match="lacks max-raw-delta"):
        validate_episode_safety_cadence(
            safety=missing, episode_result=_episode_result()
        )

    inconsistent = _episode_safety()
    inconsistent["maxima"]["raw_delta_joint_pos_rad"][0] = 0.25
    with pytest.raises(ValueError, match="disagrees with maxima"):
        validate_episode_safety_cadence(
            safety=inconsistent, episode_result=_episode_result()
        )

    incomplete_vectors = _episode_safety()
    incomplete_vectors["max_raw_delta_diagnostic"]["joint_pos_rad"] = None
    with pytest.raises(ValueError, match="max-raw diagnostic is non-finite"):
        validate_episode_safety_cadence(
            safety=incomplete_vectors, episode_result=_episode_result()
        )


def test_episode_safety_rejects_impossible_counter_and_diagnostic_history():
    impossible_events = _episode_safety()
    impossible_events["counters"]["slew_limit_events"] = 17
    impossible_events["counters"]["slew_limited_joints"] = 17
    with pytest.raises(ValueError, match="history is impossible"):
        validate_episode_safety_cadence(
            safety=impossible_events, episode_result=_episode_result()
        )

    multiple_abort = _episode_safety(numerical_failure=True)
    duplicate = json.loads(json.dumps(multiple_abort["guard_diagnostics"][0]))
    multiple_abort["guard_diagnostics"].append(duplicate)
    multiple_abort["counters"]["nonfinite_aborts"] = 2
    with pytest.raises(ValueError, match="history is impossible|exactly one"):
        validate_episode_safety_cadence(
            safety=multiple_abort,
            episode_result=_episode_result(numerical_failure=True),
        )

    out_of_order = _episode_safety(numerical_failure=True)
    fallback = json.loads(json.dumps(out_of_order["guard_diagnostics"][0]))
    fallback["kind"] = "dls_pseudoinverse_fallback"
    fallback["policy_step"] = 0
    fallback["physics_substep"] = 0
    out_of_order["guard_diagnostics"].append(fallback)
    out_of_order["counters"]["dls_fallbacks"] = 1
    with pytest.raises(ValueError, match="out of order"):
        validate_episode_safety_cadence(
            safety=out_of_order,
            episode_result=_episode_result(numerical_failure=True),
        )


def test_episode_sidecar_strict_json_rejects_nan(tmp_path: Path):
    safety = _episode_safety()
    safety["maxima"]["raw_delta_joint_pos_rad"][0] = float("nan")
    with pytest.raises(ValueError, match="maximum raw_delta_joint_pos_rad is invalid"):
        atomic_write_episode_safety(
            tmp_path / "episode_000000.json",
            episode_index=0,
            episode_result=_episode_result(),
            safety=safety,
            artifact_identity={
                "video": {
                    "filename": "episode_0.mp4",
                    "size_bytes": 1,
                    "sha256": "0" * 64,
                    "frame_count": 2,
                    "height": 224,
                    "width": 448,
                },
                "terminal_trace": {
                    "filename": "episode_000000.jsonl",
                    "size_bytes": 1,
                    "sha256": "0" * 64,
                    "episode_result": _episode_result(),
                },
            },
        )
    assert not (tmp_path / "episode_000000.json").exists()


def test_runtime_frame_rejects_observation_and_controller_drift():
    env, observation = _runtime_fixture()
    observation["policy"]["eef_pos"] = observation["policy"]["eef_pos"].copy()
    observation["policy"]["eef_pos"][0, 0] += 0.01
    with pytest.raises(ValueError, match="direct panda_link0->panda_link8"):
        validate_eef_runtime_frame(env, observation)

    env, observation = _runtime_fixture()
    env.unwrapped.action_manager._terms["arm"].cfg.body_name = "base_link"
    with pytest.raises(ValueError, match="does not control physical panda_link8"):
        validate_eef_runtime_frame(env, observation)


def test_runtime_frame_rejects_nonidentity_offset_and_relative_mode():
    env, observation = _runtime_fixture()
    env.unwrapped.action_manager._terms["arm"].cfg.body_offset.pos = (0.0, 0.0, 0.01)
    with pytest.raises(ValueError, match="offset is not identity"):
        validate_eef_runtime_frame(env, observation)

    env, observation = _runtime_fixture()
    env.unwrapped.action_manager._terms["arm"].cfg.controller.use_relative_mode = True
    with pytest.raises(ValueError, match="not absolute pose"):
        validate_eef_runtime_frame(env, observation)


def test_runtime_safety_rejects_drift_and_unbounded_applied_delta():
    env, _ = _runtime_fixture()
    original_reporter = env.unwrapped.action_manager._terms["arm"].safety_report
    drifted = original_reporter()
    drifted["apply_actions_cadence"] = "policy_step"
    env.unwrapped.action_manager._terms["arm"].safety_report = lambda: drifted
    with pytest.raises(ValueError, match="apply_actions_cadence"):
        validate_eef_runtime_safety(env)

    env, _ = _runtime_fixture()
    unsafe = env.unwrapped.action_manager._terms["arm"].safety_report()
    unsafe["maxima"]["applied_delta_joint_pos_rad"][0] = 1.0
    env.unwrapped.action_manager._terms["arm"].safety_report = lambda: unsafe
    with pytest.raises(ValueError, match="exceeds its physics-substep bound"):
        validate_eef_runtime_safety(env)

    env, _ = _runtime_fixture()
    tampered = env.unwrapped.action_manager._terms["arm"].safety_report()
    tampered["soft_joint_pos_limits_rad"][0][0] += 1e-3
    tampered["soft_joint_pos_limits_float32_sha256"] = hashlib.sha256(
        np.asarray(tampered["soft_joint_pos_limits_rad"], dtype="<f4").tobytes()
    ).hexdigest()
    env.unwrapped.action_manager._terms["arm"].safety_report = lambda: tampered
    with pytest.raises(ValueError, match="canonical Panda float32"):
        validate_eef_runtime_safety(env)


def test_runtime_safety_uses_one_exact_float32_slew_tolerance():
    env, _ = _runtime_fixture()
    report = env.unwrapped.action_manager._terms["arm"].safety_report()
    bound = report["max_delta_joint_pos_rad"][0]
    report["maxima"]["applied_delta_joint_pos_rad"][0] = (
        bound + JOINT_SLEW_FLOAT32_TOLERANCE_RAD
    )
    env.unwrapped.action_manager._terms["arm"].safety_report = lambda: report
    validate_eef_runtime_safety(env)

    report["maxima"]["applied_delta_joint_pos_rad"][0] = (
        bound + 2 * JOINT_SLEW_FLOAT32_TOLERANCE_RAD
    )
    with pytest.raises(ValueError, match="exceeds its physics-substep bound"):
        validate_eef_runtime_safety(env)


def test_runtime_safety_diagnostic_requires_strict_null_and_finite_mask():
    env, _ = _runtime_fixture()
    report = env.unwrapped.action_manager._terms["arm"].safety_report()
    report["episode_index"] = 0
    diagnostic = {
        "kind": "nonfinite_abort",
        "episode_index": 0,
        "policy_step": 3,
        "physics_substep": 2,
        "joint_pos_rad": {
            "values": [0.0, None, 0.0, 0.0, 0.0, 0.0, 0.0],
            "finite_mask": [True, False, True, True, True, True, True],
            "finite_count": 6,
        },
        "raw_delta_joint_pos_rad": None,
        "raw_joint_pos_target_rad": None,
        "safe_joint_pos_target_rad": None,
        "pose_error_norm": None,
        "jacobian_finite": False,
        "jacobian_max_abs": None,
        "eef_quaternion_norm": None,
    }
    report["guard_diagnostics"] = [diagnostic]
    report["counters"]["nonfinite_aborts"] = 1
    env.unwrapped.action_manager._terms["arm"].safety_report = lambda: report
    validated = validate_eef_runtime_safety(env)
    json.dumps(validated, allow_nan=False)

    diagnostic["joint_pos_rad"]["values"][1] = 1.0
    with pytest.raises(ValueError, match="nonfinite value must be null"):
        validate_eef_runtime_safety(env)

    diagnostic["joint_pos_rad"]["values"][1] = None
    diagnostic["eef_quaternion_norm"] = float("nan")
    with pytest.raises(ValueError, match="eef_quaternion_norm is non-finite"):
        validate_eef_runtime_safety(env)


def test_runtime_safety_binds_exact_quaternion_unit_norm_tolerance():
    env, _ = _runtime_fixture()
    report = env.unwrapped.action_manager._terms["arm"].safety_report()
    validate_eef_runtime_safety(env)

    report["eef_quaternion_unit_norm_tolerance"] = (
        EEF_QUATERNION_UNIT_NORM_TOLERANCE * 2
    )
    env.unwrapped.action_manager._terms["arm"].safety_report = lambda: report
    with pytest.raises(ValueError, match="quaternion_tolerance"):
        validate_eef_runtime_safety(env)


def test_one_step_adversarial_smoke_requires_bounded_active_slew_guard():
    report = _episode_safety()
    report["counters"]["apply_calls"] = 8
    report["counters"]["environment_substeps"] = 8
    report["counters"]["slew_limit_events"] = 1
    report["counters"]["slew_limited_joints"] = 1
    report["maxima"]["applied_delta_joint_pos_rad"] = list(
        report["max_delta_joint_pos_rad"]
    )

    evidence = validate_one_step_adversarial_report(report)
    assert evidence["apply_calls"] == 8
    assert evidence["slew_limit_events"] == 1
    assert evidence["applied_within_bounds"] is True

    no_guard = json.loads(json.dumps(report))
    no_guard["counters"]["slew_limit_events"] = 0
    with pytest.raises(ValueError, match="did not activate"):
        validate_one_step_adversarial_report(no_guard)

    unsafe = json.loads(json.dumps(report))
    unsafe["maxima"]["applied_delta_joint_pos_rad"][0] += 2e-6
    with pytest.raises(ValueError, match="exceeded"):
        validate_one_step_adversarial_report(unsafe)

    smoke_source = (
        Path(__file__).parents[1] / "scripts" / "smoke_eef_pose_controller.py"
    ).read_text()
    for axis in "xyz":
        assert f'"translate +{axis}"' in smoke_source
        assert f'"translate -{axis}"' in smoke_source
        assert f'"rotate +{axis}"' in smoke_source
        assert f'"rotate -{axis}"' in smoke_source
    assert "robot.data.joint_pos[:, arm_term._joint_ids]" in smoke_source
    assert "robot.data.joint_vel[:, arm_term._joint_ids]" in smoke_source
    assert '"joint_state_is_finite"' in smoke_source
    assert '"joint_pos_rad"' in smoke_source
    assert '"joint_vel_rad_s"' in smoke_source
    assert '"max_abs"' in smoke_source
    assert '"position_within_captured_soft_limits"' in smoke_source
    assert "CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD" in smoke_source
    assert (
        smoke_source.index('state["raw_capture"] = initial_capture')
        < (
            smoke_source.index(
                "validated_initial_capture = validate_eef_runtime_safety(env)"
            )
        )
        < smoke_source.index("for case_index, (label, pose_delta)")
    )
    assert "os.link(temporary_path, path)" in smoke_source
    assert 'temporary_path.open("xb"' in smoke_source
    assert "path.chmod(0o444)" in smoke_source
    assert "os.fsync(directory_fd)" in smoke_source
    assert "allow_nan=False" in smoke_source
    assert 'required=True,\n    help="Required atomic' in smoke_source
    assert '"raw_ik_safety_capture"' in smoke_source
    assert '"stage": state["stage"]' in smoke_source
    assert '"case": state["case"]' in smoke_source
    assert 'formatted_traceback = "".join(' in smoke_source
    assert '"traceback": formatted_traceback' in smoke_source
    assert "except BaseException as run_error:" in smoke_source
    assert "except BaseException as close_error:" in smoke_source
    assert 'close_evidence["component"] = "environment"' in smoke_source
    assert "_result_payload(state, finalized=False" in smoke_source
    assert "_result_payload(state, finalized=True" not in smoke_source
    assert '"failure": state["failure"]' in smoke_source
    assert '"close_failures": state["close_failures"]' in smoke_source
    assert '"persistence_failures": state["persistence_failures"]' in smoke_source
    assert smoke_source.index("_print_exception(run_error)") < smoke_source.index(
        "simulation_app.close()"
    )
    pending_index = smoke_source.index(
        'state["stage"] = "simulation_app_close_pending"'
    )
    raw_write_index = smoke_source.index("_atomic_write_strict_json(", pending_index)
    prepared_index = smoke_source.index("POLARIS_SMOKE_RAW_PREPARED=", raw_write_index)
    marker_publish_index = smoke_source.index(
        "_atomic_write_strict_json(ready_marker, marker_payload)", prepared_index
    )
    simulation_close_index = smoke_source.index(
        "simulation_app.close()", marker_publish_index
    )
    assert (
        smoke_source.index("env.close()")
        < pending_index
        < raw_write_index
        < prepared_index
        < marker_publish_index
        < simulation_close_index
    )
    assert (
        "_atomic_write_strict_json(ready_marker, marker_payload)\n"
        "            simulation_app.close()"
    ) in smoke_source
    assert "POLARIS_SMOKE_RAW_READY=" not in smoke_source
    assert "POLARIS_SMOKE_RAW_PREPARED=" in smoke_source
    assert "POLARIS_SMOKE_RAW_SHA256=" in smoke_source
    assert "POLARIS_SMOKE_READY_MARKER_PATH=" in smoke_source
    assert "POLARIS_SMOKE_READY_MARKER_EXPECTED_SHA256=" in smoke_source
    assert "POLARIS_SIMULATION_APP_CLOSE_SKIPPED=raw_not_ready" in smoke_source
    assert "os._exit(1)" in smoke_source
    assert "def _best_effort_failure_log" in smoke_source
    assert (
        "else:\n        exit_code = 1\n        _best_effort_failure_log(\n"
        '            "POLARIS_SMOKE_RAW_FAILURE="'
    ) in smoke_source
    assert "finally:\n            os._exit(1)" in smoke_source
    assert "sys.stderr.flush()" in smoke_source


def test_smoke_raw_json_publication_is_strict_and_nonoverwriting(tmp_path):
    smoke_path = Path(__file__).parents[1] / "scripts" / "smoke_eef_pose_controller.py"
    parsed = ast.parse(smoke_path.read_text())
    helper_names = {
        "_strict_json_value",
        "_strict_json_bytes",
        "_atomic_write_strict_json",
        "_raw_is_eligible_for_close",
    }
    helper_nodes = [
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    assert {node.name for node in helper_nodes} == helper_names
    namespace = {"Path": Path, "json": json, "math": math, "os": os}
    exec(
        compile(ast.Module(helper_nodes, type_ignores=[]), str(smoke_path), "exec"),
        namespace,
    )
    writer = namespace["_atomic_write_strict_json"]
    eligible = namespace["_raw_is_eligible_for_close"]

    clean_state = {
        "stage": "simulation_app_close_pending",
        "case": None,
        "failure": None,
        "close_failures": [],
        "persistence_failures": [],
    }
    assert not eligible(
        clean_state, exit_code=0, raw_published=True, simulation_app=None
    )
    assert eligible(
        clean_state, exit_code=0, raw_published=True, simulation_app=object()
    )

    output = tmp_path / "raw.json"
    writer(output, {"value": 1.0, "nonfinite": math.inf})
    original = output.read_bytes()
    assert output.stat().st_mode & 0o777 == 0o444
    assert json.loads(original) == {"value": 1.0, "nonfinite": None}

    with pytest.raises(FileExistsError):
        writer(output, {"value": 2.0})
    assert output.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("ik_method", "pinv", "damped least-squares"),
        ("damping", 0.1, "DLS damping"),
        ("scale", 0.5, "action scale"),
        (
            "joint_names",
            list(reversed([f"panda_joint{i}" for i in range(1, 8)])),
            "joint order",
        ),
        ("gripper_profile", "closed_positive_gt_0p5", "gripper threshold semantics"),
    ],
)
def test_runtime_frame_rejects_controller_semantics_drift(field, value, match):
    env, observation = _runtime_fixture()
    arm = env.unwrapped.action_manager._terms["arm"]
    if field == "ik_method":
        arm.cfg.controller.ik_method = value
    elif field == "damping":
        arm.cfg.controller.ik_params["lambda_val"] = value
    elif field == "scale":
        arm.cfg.scale = value
    elif field == "joint_names":
        arm._joint_names = value
    else:
        env.unwrapped.action_manager._terms[
            "finger_joint"
        ].gripper_threshold_profile = value

    with pytest.raises(ValueError, match=match):
        validate_eef_runtime_frame(env, observation)
