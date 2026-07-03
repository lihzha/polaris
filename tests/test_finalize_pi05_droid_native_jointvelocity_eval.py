import copy
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.polaris import capture_pi05_droid_native_environment as environment
from scripts.polaris import finalize_pi05_droid_native_jointvelocity_eval as finalizer

from polaris.pi05_droid_jointvelocity_contract import (
    PI05_DROID_OPENPI_INFERENCE_COMPATIBILITY_COMMIT,
    reference_openpi_runtime_attestation,
)
from polaris.pi05_droid_native_eval_contract import (
    PI05_DROID_CONTROLLER_CRITICAL_PATHS,
    PI05_DROID_NATIVE_CANARY_PROFILE,
    PI05_DROID_NATIVE_CONFIGURED_EPISODE_LENGTH_SECONDS,
    PI05_DROID_NATIVE_EPISODE_STEPS,
    PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS,
    PI05_DROID_PYXIS_SHA256,
    PI05_DROID_NATIVE_MODEL_EVAL_CONTRACT,
    PI05_DROID_NATIVE_TRANSFORM_RUNTIME_CONTRACT,
    file_sha256,
    make_close_ready_artifact,
    make_environment_runtime_contract,
    make_runtime_artifact,
    publish_immutable_json,
)


ROOT = Path(__file__).parents[1]


def test_base_controller_gate_binds_job1098174_runtime_image_and_sources(
    tmp_path, monkeypatch, valid_joint_velocity_smoke_payload
):
    repository = tmp_path / "polaris"
    source_records = {}
    for relative_path in PI05_DROID_CONTROLLER_CRITICAL_PATHS:
        path = repository / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"controller source: {relative_path}\n", encoding="utf-8")
        source_records[relative_path] = {
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    completion_path = tmp_path / "controller-completion.json"
    artifact = publish_immutable_json(
        completion_path,
        {
            "schema_version": 1,
            "profile": finalizer.CONTROLLER_PROFILE,
            "status": "pass",
            "scope": "controller_only_no_model_or_checkpoint",
            "slurm": {"job_id": 1098174, "srun_exit_code": 0},
            "source": {
                "commit": finalizer.PI05_DROID_BASE_CONTROLLER_SOURCE_COMMIT,
                "detached_head": True,
                "tracked_and_untracked_clean": True,
                "head_reference": "HEAD",
                "standalone_git_directory": True,
                "files": source_records,
            },
            "runtime_contract": valid_joint_velocity_smoke_payload["runtime_contract"],
            "runtime": {"container_image": {"sha256": PI05_DROID_PYXIS_SHA256}},
        },
    )
    monkeypatch.setattr(
        finalizer, "PI05_DROID_BASE_CONTROLLER_COMPLETION_PATH", str(completion_path)
    )
    monkeypatch.setattr(
        finalizer,
        "PI05_DROID_BASE_CONTROLLER_COMPLETION_SHA256",
        artifact["sha256"],
    )
    monkeypatch.setattr(
        finalizer, "PI05_DROID_BASE_CONTROLLER_COMPLETION_SIZE", artifact["size"]
    )
    monkeypatch.setattr(
        finalizer,
        "PI05_DROID_BASE_CONTROLLER_RUNTIME_SHA256",
        valid_joint_velocity_smoke_payload["runtime_contract"]["runtime_sha256"],
    )

    result = finalizer.validate_base_controller_completion(
        completion_path, artifact["sha256"], repository
    )
    assert result["job_id"] == 1098174
    assert result["profile"] == finalizer.CONTROLLER_PROFILE
    assert set(result["critical_source_files"]) == set(
        PI05_DROID_CONTROLLER_CRITICAL_PATHS
    )
    assert result["descendant_source_authority"] == "required_job1098204_gate"
    with pytest.raises(ValueError, match="Unexpected job1098174 completion SHA"):
        finalizer.validate_base_controller_completion(
            completion_path, "0" * 64, repository
        )


def _scalar(value, device):
    return {
        "shape": [1, 1],
        "dtype": "torch.float32",
        "device": device,
        "values": [[value]],
    }


def _gripper_runtime(base_runtime):
    runtime = copy.deepcopy(base_runtime)
    runtime["gripper"]["drive"] = {
        "profile": finalizer.PI05_DROID_GRIPPER_DRIVE_PROFILE,
        "configured": {
            "joint_names_expr": ["finger_joint"],
            "stiffness": None,
            "damping": None,
            "effort_limit": 200.0,
            "effort_limit_sim": 200.0,
            "velocity_limit": 5.0,
            "velocity_limit_sim": 5.0,
        },
        "actuator": {
            "stiffness": _scalar(5729.578125, "cuda:0"),
            "damping": _scalar(0.011459155939519405, "cuda:0"),
            "effort_limit": _scalar(200.0, "cuda:0"),
            "effort_limit_sim": _scalar(200.0, "cuda:0"),
            "velocity_limit": _scalar(5.0, "cuda:0"),
            "velocity_limit_sim": _scalar(5.0, "cuda:0"),
        },
        "direct_physx": {
            "stiffness": _scalar(5729.578125, "cpu"),
            "damping": _scalar(0.011459155939519405, "cpu"),
            "effort_limit": _scalar(200.0, "cpu"),
            "velocity_limit": _scalar(5.0, "cpu"),
        },
    }
    runtime["runtime_sha256"] = finalizer._runtime_sha256(runtime)
    return runtime


def _gripper_case(label, command, slew, target, processed_target, sign):
    return {
        "label": label,
        "kind": "gripper",
        "action": [0.0] * 7 + [command],
        "processed_joint_velocity": [0.0] * 7,
        "articulation_joint_velocity_target": [0.0] * 7,
        "expected_finger_target": target,
        "processed_finger_position_target": processed_target,
        "expected_motion_sign": sign,
        "finger_average_slew_rad_s": slew,
        "finger_velocity_after": sign * 4.97,
        "finger_position_before": 0.0 if sign > 0 else 0.7853981852531433,
        "finger_position_after": 0.33 if sign > 0 else 0.45,
        "joint_velocity_before": [0.0] * 7,
        "joint_velocity_after": [0.0001] * 7,
        "terminated": False,
        "truncated": False,
    }


def _publish_gripper_smoke(tmp_path, runtime):
    labels = [
        "hold",
        *(
            f"panda_joint{joint}_{sign}"
            for joint in range(1, 8)
            for sign in ("positive", "negative")
        ),
        "positive_action_limit",
        "negative_action_limit",
    ]
    cases = [{"label": label, "kind": "other"} for label in labels]
    cases.extend(
        [
            _gripper_case("gripper_open", 0.0, -4.978439211845398, 0.0, 0.0, -1),
            _gripper_case(
                "gripper_closed",
                1.0,
                4.971333146095276,
                0.7853981633974483,
                0.7853981852531433,
                1,
            ),
            _gripper_case(
                "gripper_boundary_0p5",
                0.5,
                -4.978439211845398,
                0.0,
                0.0,
                -1,
            ),
        ]
    )
    default_q = [
        0.0,
        -0.6283185482025146,
        0.0,
        -2.5132741928100586,
        0.0,
        1.884955644607544,
        0.0,
    ]
    parent = {
        "schema_version": 1,
        "smoke_profile": "pi05_droid_native_jointvelocity_controller_smoke_v2",
        "controller_profile": "openpi_pi05_droid_native_jointvelocity_v1",
        "environment": "DROID-FoodBussing",
        "command_magnitude": 0.25,
        "settle_steps": 5,
        "expected_gripper_drive_profile": finalizer.PI05_DROID_GRIPPER_DRIVE_PROFILE,
        "gripper_precondition_steps": 5,
        "runtime_contract": runtime,
        "cases": cases,
        "case_count": 20,
        "reset_probe": {
            "default_joint_position": default_q,
            "joint_position": default_q,
            "joint_velocity": [0.0] * 7,
            "joint_velocity_target": [0.0] * 7,
            "default_finger_position": 0.0,
            "finger_position": 0.0,
            "finger_velocity": 0.0,
            "finger_position_target": 0.0,
        },
        "lifecycle": {
            "env_close": "complete",
            "simulation_app_close": "invoked_then_child_exited_zero",
            "capture_stage": "stdlib_parent_after_kit_child_exit",
        },
        "status": "pass",
    }
    child_value = copy.deepcopy(parent)
    child_value["lifecycle"] = {
        "env_close": "complete",
        "simulation_app_close": "pending_child_exit",
        "capture_stage": "kit_child_after_env_close_before_simulation_app_close",
    }
    child_value["status"] = "close_validated_pending_parent"
    child = publish_immutable_json(tmp_path / "smoke.child-close.json", child_value)
    ready_value = {
        "schema_version": 1,
        "status": "success",
        "stage": "kit_child_after_env_close_before_simulation_app_close",
        "raw_result": {
            "path": str((tmp_path / "smoke.child-close.json")),
            "size_bytes": child["size"],
            "sha256": child["sha256"],
            "mode": "0444",
        },
    }
    ready = publish_immutable_json(
        tmp_path / "smoke.child-close.ready.json", ready_value
    )
    parent["completion"] = {
        "child_exit_code": 0,
        "publication_stage": "stdlib_parent_after_child_exit",
        "child_capture_sha256": child["sha256"],
        "child_capture_size": child["size"],
        "child_capture_mode": "0444",
        "child_capture_path": str(tmp_path / "smoke.child-close.json"),
        "child_ready_marker_sha256": ready["sha256"],
        "child_ready_marker_size": ready["size"],
        "child_ready_marker_mode": "0444",
        "child_ready_marker_path": str(tmp_path / "smoke.child-close.ready.json"),
    }
    smoke = publish_immutable_json(tmp_path / "smoke.json", parent)
    return {
        "path": str(tmp_path / "smoke.json"),
        "size": smoke["size"],
        "sha256": smoke["sha256"],
        "mode": "0444",
        "nlink": 1,
        "status": "pass",
        "runtime_sha256": runtime["runtime_sha256"],
    }


def test_gripper_cap_gate_revalidates_drive_slew_reset_lifecycle_and_source(
    tmp_path, monkeypatch, valid_joint_velocity_smoke_payload
):
    runtime = _gripper_runtime(valid_joint_velocity_smoke_payload["runtime_contract"])
    monkeypatch.setattr(
        finalizer, "PI05_DROID_GRIPPER_RUNTIME_SHA256", runtime["runtime_sha256"]
    )
    smoke_record = _publish_gripper_smoke(tmp_path, runtime)
    repository = tmp_path / "polaris"
    source_files = {}
    for relative_path in PI05_DROID_CONTROLLER_CRITICAL_PATHS:
        path = repository / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"job1098204:{relative_path}\n", encoding="utf-8")
        source_files[relative_path] = {
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    completion_path = tmp_path / "controller-smoke-1098204.completion.json"
    completion = publish_immutable_json(
        completion_path,
        {
            "schema_version": 1,
            "profile": finalizer.PI05_DROID_GRIPPER_CONTROLLER_PROFILE,
            "status": "pass",
            "scope": "controller_only_no_model_or_checkpoint",
            "promotion": "forbidden_without_separate_checkpoint_canary",
            "candidate_intent": {
                "expected_gripper_drive_profile": finalizer.PI05_DROID_GRIPPER_DRIVE_PROFILE
            },
            "source": {
                "commit": finalizer.PI05_DROID_GRIPPER_CONTROLLER_SOURCE_COMMIT,
                "detached_head": True,
                "tracked_and_untracked_clean": True,
                "head_reference": "HEAD",
                "standalone_git_directory": True,
                "files": source_files,
            },
            "runtime": {"container_image": {"sha256": PI05_DROID_PYXIS_SHA256}},
            "runtime_contract": runtime,
            "slurm": {"job_id": 1098204, "srun_exit_code": 0},
            "smoke_artifact": smoke_record,
        },
    )
    monkeypatch.setattr(
        finalizer, "PI05_DROID_GRIPPER_CONTROLLER_COMPLETION_PATH", str(completion_path)
    )
    monkeypatch.setattr(
        finalizer,
        "PI05_DROID_GRIPPER_CONTROLLER_COMPLETION_SHA256",
        completion["sha256"],
    )
    monkeypatch.setattr(
        finalizer,
        "PI05_DROID_GRIPPER_CONTROLLER_COMPLETION_SIZE",
        completion["size"],
    )

    result = finalizer.validate_gripper_cap_controller_completion(
        completion_path,
        completion["sha256"],
        finalizer.PI05_DROID_GRIPPER_DRIVE_PROFILE,
        repository,
    )
    assert result["job_id"] == 1098204
    assert result["runtime_sha256"] == runtime["runtime_sha256"]
    assert result["smoke"]["measured_average_slew_magnitude_rad_s"] == {
        "gripper_open": 4.978439211845398,
        "gripper_closed": 4.971333146095276,
        "gripper_boundary_0p5": 4.978439211845398,
    }
    assert result["smoke"]["exact_reset"] is True

    changed = repository / PI05_DROID_CONTROLLER_CRITICAL_PATHS[0]
    changed.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from job1098204"):
        finalizer.validate_gripper_cap_controller_completion(
            completion_path,
            completion["sha256"],
            finalizer.PI05_DROID_GRIPPER_DRIVE_PROFILE,
            repository,
        )


def _environment_value(openpi_dir: Path, python: Path):
    installed = [
        {"name": name, "version": "1.0"}
        for name in sorted(environment.RELEVANT_PACKAGES)
    ]
    relevant = {
        name: {"installed_version": "1.0", "locked_versions": ["1.0"]}
        for name in environment.RELEVANT_PACKAGES
    }
    return installed, {
        "schema_version": 1,
        "profile": PI05_DROID_NATIVE_CANARY_PROFILE,
        "status": "pass",
        "python": {
            "executable": str(python),
            "resolved_executable": str(python.resolve()),
            "version": environment.platform.python_version(),
            "implementation": "CPython",
        },
        "openpi": {
            "root": str(openpi_dir.resolve()),
            "git_head": PI05_DROID_OPENPI_INFERENCE_COMPATIBILITY_COMMIT,
            "git_tracked_and_untracked_clean": True,
            "uv_lock_sha256": file_sha256(openpi_dir / "uv.lock"),
        },
        "jax": {
            "version": "1.0",
            "jaxlib_version": "1.0",
            "numpy_version": "1.0",
            "enable_x64": False,
            "default_backend": "gpu",
            "devices": [{"id": 0, "platform": "gpu", "device_kind": "NVIDIA L40S"}],
        },
        "relevant_packages": relevant,
        "installed_packages": installed,
        "installed_packages_sha256": hashlib.sha256(
            environment.canonical_json_bytes(installed)
        ).hexdigest(),
    }


def test_inference_environment_recomputes_lock_and_installed_inventory(
    tmp_path, monkeypatch
):
    openpi_dir = tmp_path / "openpi"
    openpi_dir.mkdir()
    lock = "".join(
        f'[[package]]\nname = "{name}"\nversion = "1.0"\n'
        for name in environment.RELEVANT_PACKAGES
    )
    (openpi_dir / "uv.lock").write_text(lock, encoding="utf-8")
    python = Path(sys.executable)
    installed, value = _environment_value(openpi_dir, python)
    monkeypatch.setattr(environment, "_installed_packages", lambda: installed)

    assert environment.validate_environment(value, openpi_dir, python) == value

    wrong_category = copy.deepcopy(value)
    wrong_category["relevant_packages"]["numpy"]["locked_versions"] = ["2.0"]
    with pytest.raises(ValueError, match="package provenance mismatch: numpy"):
        environment.validate_environment(wrong_category, openpi_dir, python)

    wrong_jax = copy.deepcopy(value)
    wrong_jax["jax"]["jaxlib_version"] = "9.9"
    with pytest.raises(ValueError, match="JAX runtime mismatch"):
        environment.validate_environment(wrong_jax, openpi_dir, python)


def test_model_runtime_artifact_binds_official_config_transforms_and_rng(tmp_path):
    checkpoint_value = {
        "sha256": "6f9ccfa5695c669962ad10dbe0dcb7d44bf903918e5fffe33e5d1ff531287922",
        "object_count": 20,
        "total_bytes": 12_429_488_598,
        "checkpoint_dir": "/cache/pi05_droid",
        "norm_stats_sha256": "403b3a22f897e9ae5dd617966a3c8f7d1835ac79dfd5a8993179514be26a3b8b",
        "full_md5": True,
    }
    checkpoint = {"checkpoint": checkpoint_value}
    value = {
        "schema_version": 1,
        "profile": PI05_DROID_NATIVE_CANARY_PROFILE,
        "status": "pass",
        "checkpoint": checkpoint_value,
        "train_config": {
            "name": "pi05_droid",
            "model_type": "pi05",
            "pi05": True,
            "dtype": "bfloat16",
            "action_horizon": 15,
            "action_dim": 32,
            "asset_id": "droid",
            "policy_metadata": None,
        },
        "transform_runtime": PI05_DROID_NATIVE_TRANSFORM_RUNTIME_CONTRACT,
        "policy": {"metadata": {}, "sample_kwargs": {}, "rng_key_data": [0, 0]},
        "official_model_eval_contract": PI05_DROID_NATIVE_MODEL_EVAL_CONTRACT,
        "openpi_runtime_attestation": reference_openpi_runtime_attestation(),
    }
    path = tmp_path / "model-runtime.json"
    publish_immutable_json(path, value)
    result = finalizer._validate_model_runtime_artifact(path, checkpoint)
    assert result["transform_runtime"]["asset_id"] == "droid"
    assert result["policy"]["rng_key_data"] == [0, 0]

    tampered = copy.deepcopy(value)
    tampered["transform_runtime"]["asset_id"] = "single_arm"
    tampered_path = tmp_path / "tampered-model-runtime.json"
    publish_immutable_json(tampered_path, tampered)
    with pytest.raises(ValueError, match="model-runtime artifact mismatch"):
        finalizer._validate_model_runtime_artifact(tampered_path, checkpoint)


def test_finalizer_binds_internal_timeout_terminal_state_and_close_artifact(
    tmp_path, valid_joint_velocity_smoke_payload
):
    environment_runtime = make_environment_runtime_contract(
        configured_episode_length_seconds=(
            PI05_DROID_NATIVE_CONFIGURED_EPISODE_LENGTH_SECONDS
        ),
        live_max_episode_length=PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS,
    )
    run_dir = tmp_path / "run"
    task_dir = run_dir / "DROID-FoodBussing"
    task_dir.mkdir(parents=True)
    runtime_path = task_dir / "joint_velocity_runtime.json"
    runtime_identity = publish_immutable_json(
        runtime_path,
        make_runtime_artifact(
            valid_joint_velocity_smoke_payload["runtime_contract"],
            environment_runtime,
        ),
    )
    before = {
        "live_max_episode_length": PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS,
        "episode_length": 0,
        "sim_step_counter": 9,
        "common_step_counter": 2,
        "sensor_frame_counters": {"external_cam": 1, "wrist_cam": 1},
    }
    terminal = {
        "schema_version": 1,
        "profile": environment_runtime["profile"],
        "environment_runtime_sha256": environment_runtime["sha256"],
        "outer_steps_completed": PI05_DROID_NATIVE_EPISODE_STEPS,
        "last_outer_step_index": PI05_DROID_NATIVE_EPISODE_STEPS - 1,
        "terminated_false_count": PI05_DROID_NATIVE_EPISODE_STEPS,
        "truncated_false_count": PI05_DROID_NATIVE_EPISODE_STEPS,
        "environment_before": before,
        "environment_after": {
            "live_max_episode_length": (PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS),
            "episode_length": PI05_DROID_NATIVE_EPISODE_STEPS,
            "sim_step_counter": 9 + PI05_DROID_NATIVE_EPISODE_STEPS * 8,
            "common_step_counter": 2 + PI05_DROID_NATIVE_EPISODE_STEPS,
            "sensor_frame_counters": {
                "external_cam": 1 + PI05_DROID_NATIVE_EPISODE_STEPS,
                "wrist_cam": 1 + PI05_DROID_NATIVE_EPISODE_STEPS,
            },
        },
        "rubric": {"success": False, "progress": 0.25},
    }
    close_path = task_dir / "evaluator_close_ready.json"
    publish_immutable_json(
        close_path,
        make_close_ready_artifact(
            runtime_artifact=runtime_identity,
            runtime_path=runtime_path,
            metrics_path=task_dir / "eval_results.csv",
            trace_path=task_dir / "policy_traces.jsonl",
            video_path=task_dir / "episode_0.mp4",
            environment_runtime_contract=environment_runtime,
            terminal_rollout=terminal,
        ),
    )

    runtime = finalizer._validate_runtime_artifact(runtime_path)
    close = finalizer._validate_close_ready(close_path, runtime, run_dir)
    assert runtime["environment_runtime_contract"] == environment_runtime
    assert close["terminal_rollout"] == terminal

    bad_runtime = make_runtime_artifact(
        valid_joint_velocity_smoke_payload["runtime_contract"], environment_runtime
    )
    bad_runtime["environment_runtime_contract"]["live_max_episode_length"] = 450
    bad_path = task_dir / "bad-runtime.json"
    publish_immutable_json(bad_path, bad_runtime)
    with pytest.raises(ValueError, match="environment runtime contract mismatch"):
        finalizer._validate_runtime_artifact(bad_path)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg tools are unavailable",
)
def test_summary_video_is_exact_h264_yuv420p_faststart_and_fully_decodable(tmp_path):
    source = tmp_path / "episode_0.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=448x224:r=15",
            "-frames:v",
            "450",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    source_probe = finalizer.probe_video(source, require_faststart=False)
    assert source_probe["frame_count"] == 450
    assert source_probe["full_decode"] == "pass"

    summary = tmp_path / "summary.mp4"
    summary_probe = finalizer.create_summary_video(source, summary)
    assert summary_probe["codec"] == "h264"
    assert summary_probe["pixel_format"] == "yuv420p"
    assert summary_probe["frame_count"] == 450
    assert summary_probe["duration_seconds"] == 30.0
    assert summary_probe["faststart"] is True
    assert summary_probe["top_level_boxes"].index("moov") < summary_probe[
        "top_level_boxes"
    ].index("mdat")
    assert (summary.stat().st_mode & 0o777) == 0o444


def test_launchers_block_before_download_or_sbatch_and_forbid_resume():
    eval_source = (
        ROOT / "scripts/polaris/eval_pi05_droid_native_jointvelocity.sh"
    ).read_text(encoding="utf-8")
    submit_source = (
        ROOT / "scripts/polaris/submit_pi05_droid_native_jointvelocity_canary.sh"
    ).read_text(encoding="utf-8")
    evaluator_source = (ROOT / "scripts/eval.py").read_text(encoding="utf-8")

    assert eval_source.index("finalize_pi05_droid_native_jointvelocity_eval.py") < (
        eval_source.index("maybe_download")
    )
    assert submit_source.index(
        "finalize_pi05_droid_native_jointvelocity_eval.py"
    ) < submit_source.index("sbatch --parsable")
    for source in (eval_source, submit_source):
        assert "CONTROLLER_COMPLETION" in source
        assert "GRIPPER_CAP_CONTROLLER_COMPLETION" in source
        assert "EXPECTED_GRIPPER_CAP_COMPLETION_SHA256" in source
        assert "RESUME_FROM_TASK_DIR" in source
    timeout_configuration = evaluator_source.index(
        "configured_episode_length_seconds = configure_native_environment_timeout"
    )
    environment_construction = evaluator_source.index(
        "env: ManagerBasedRLSplatEnv = gym.make"
    )
    assert timeout_configuration < environment_construction
    assert "PI05_DROID_NATIVE_EPISODE_STEPS" in evaluator_source
    assert "terminated=term" in evaluator_source
    assert "truncated=trunc" in evaluator_source
    assert evaluator_source.index("finish_rollout") < evaluator_source.index(
        "env.close()", evaluator_source.index("finish_rollout")
    )
    assert "official_model_eval_contract" in finalizer.finalize.__code__.co_consts
