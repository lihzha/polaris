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
    PI05_DROID_ALL_SIX_CONTROLLER_CRITICAL_PATHS,
    PI05_DROID_ALL_SIX_CONTROLLER_JOB_ID,
    PI05_DROID_ALL_SIX_UNCHANGED_POLICY_IO_PATHS,
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
    make_episode_sidecar,
    make_environment_runtime_contract,
    make_runtime_artifact,
    publish_immutable_file_from_temporary,
    publish_immutable_json,
)
from polaris.native_gripper_runtime import (
    EXPECTED_DROID_JOINT_NAMES,
    NATIVE_GRIPPER_DYNAMIC_PROFILE,
)


ROOT = Path(__file__).parents[1]


def test_bound_json_record_expected_value_requires_canonical_identity(tmp_path):
    artifact = publish_immutable_json(tmp_path / "record.json", {"job_id": 1.0})
    record = {key: artifact[key] for key in ("path", "size", "sha256", "mode", "nlink")}
    assert artifact["value"] == {"job_id": 1}
    assert finalizer._validate_bound_json_record(
        record,
        field="typed record",
        expected_value={"job_id": 1.0},
    )["value"] == {"job_id": 1.0}
    with pytest.raises(ValueError, match="typed record identity mismatch"):
        finalizer._validate_bound_json_record(
            record,
            field="typed record",
            expected_value={"job_id": 1},
        )


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
    assert result["descendant_source_authority"] == (
        f"required_job{PI05_DROID_ALL_SIX_CONTROLLER_JOB_ID}_all_six_gate"
    )
    assert result["descendant_source_authority"] != ("required_job1098349_all_six_gate")
    with pytest.raises(ValueError, match="Unexpected job1098174 completion SHA"):
        finalizer.validate_base_controller_completion(
            completion_path, "0" * 64, repository
        )


def test_all_six_gate_revalidates_coupling_lifecycle_runtime_and_source(
    tmp_path, monkeypatch
):
    repository = tmp_path / "polaris"
    source_files = {}
    for relative_path in PI05_DROID_ALL_SIX_CONTROLLER_CRITICAL_PATHS:
        path = repository / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"job{PI05_DROID_ALL_SIX_CONTROLLER_JOB_ID}:{relative_path}\n",
            encoding="utf-8",
        )
        source_files[relative_path] = {
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }

    model_io = {}
    for relative_path in PI05_DROID_ALL_SIX_UNCHANGED_POLICY_IO_PATHS:
        path = repository / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"official-policy-io:{relative_path}\n", encoding="utf-8")
        model_io[relative_path] = {
            "base_commit": "3e9df7f605baa75848a0ad8edd2783d629d105c5",
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }

    runtime_sha256 = "9" * 64
    smoke = {
        "path": str(
            tmp_path
            / f"native-all-six-smoke-{PI05_DROID_ALL_SIX_CONTROLLER_JOB_ID}.json"
        ),
        "size": 559_362,
        "sha256": "2" * 64,
        "mode": "0444",
        "nlink": 1,
        "device": "f7980bf2",
        "inode": 1,
        "status": "pass",
        "runtime_sha256": runtime_sha256,
        "child_artifacts": {
            "raw": {"sha256": "b" * 64},
            "ready": {"sha256": "8" * 64},
        },
    }
    srun_status = publish_immutable_json(
        tmp_path / f"srun-{PI05_DROID_ALL_SIX_CONTROLLER_JOB_ID}.status.json",
        {"job_id": PI05_DROID_ALL_SIX_CONTROLLER_JOB_ID, "srun_exit_code": 0},
    )
    gpu_inventory = publish_immutable_json(
        tmp_path / f"gpu-{PI05_DROID_ALL_SIX_CONTROLLER_JOB_ID}.json",
        {
            "schema_version": 1,
            "job_id": PI05_DROID_ALL_SIX_CONTROLLER_JOB_ID,
            "gpus": [
                {
                    "uuid": "GPU-8688921b-a641-2ae1-1dc9-494501f1f422",
                    "name": "NVIDIA L40S",
                    "driver_version": "580.95.05",
                }
            ],
        },
    )
    assets = {
        relative_path: {
            "asset": {
                "path": str(tmp_path / relative_path),
                "size": 1,
                "sha256": expected["sha256"],
                "mode": "0640",
                "nlink": 1,
            },
            "metadata": {
                "path": str(tmp_path / f"{relative_path}.metadata"),
                "size": 1,
                "sha256": expected["metadata_sha256"],
                "mode": "0640",
                "nlink": 1,
            },
            "hub_revision": finalizer.PI05_DROID_HUB_REVISION,
        }
        for relative_path, expected in finalizer.PI05_DROID_CANARY_ASSETS.items()
    }
    completion_path = (
        tmp_path
        / f"native-all-six-smoke-{PI05_DROID_ALL_SIX_CONTROLLER_JOB_ID}.completion.json"
    )
    completion = publish_immutable_json(
        completion_path,
        {
            "schema_version": 1,
            "profile": finalizer.PI05_DROID_ALL_SIX_CONTROLLER_PROFILE,
            "status": "pass",
            "job_id": PI05_DROID_ALL_SIX_CONTROLLER_JOB_ID,
            "scope": "controller_only_no_model_no_checkpoint",
            "task": "DROID-FoodBussing",
            "official_policy_io_changed": False,
            "checkpoint_loaded": False,
            "model_server_started": False,
            "source": {
                "repository": (
                    f"/immutable/job{PI05_DROID_ALL_SIX_CONTROLLER_JOB_ID}/source"
                ),
                "commit": finalizer.PI05_DROID_ALL_SIX_CONTROLLER_SOURCE_COMMIT,
                "detached_and_clean": True,
                "openpi_commit": PI05_DROID_OPENPI_INFERENCE_COMPATIBILITY_COMMIT,
                "files": source_files,
                "official_model_io_unchanged_from_base": model_io,
            },
            "smoke": smoke,
            "saved_wrapper": {
                "path": f"/immutable/job-{PI05_DROID_ALL_SIX_CONTROLLER_JOB_ID}.sbatch",
                "size": 8_146,
                "sha256": "c" * 64,
                "mode": "0444",
                "nlink": 1,
            },
            "srun_status": {
                key: srun_status[key]
                for key in ("path", "size", "sha256", "mode", "nlink")
            },
            "gpu_inventory": {
                key: gpu_inventory[key]
                for key in ("path", "size", "sha256", "mode", "nlink")
            },
            "container": {
                "path": "/immutable/polaris.sqsh",
                "size": 7_183_130_624,
                "sha256": PI05_DROID_PYXIS_SHA256,
                "mode": "0644",
                "nlink": 1,
            },
            "polaris_hub_revision": finalizer.PI05_DROID_HUB_REVISION,
            "assets": assets,
            "promotion": "forbidden_without_separate_official_checkpoint_canary",
        },
    )
    monkeypatch.setattr(
        finalizer,
        "PI05_DROID_ALL_SIX_CONTROLLER_COMPLETION_PATH",
        str(completion_path),
    )
    monkeypatch.setattr(
        finalizer,
        "PI05_DROID_ALL_SIX_CONTROLLER_COMPLETION_SHA256",
        completion["sha256"],
    )
    monkeypatch.setattr(
        finalizer,
        "PI05_DROID_ALL_SIX_CONTROLLER_COMPLETION_SIZE",
        completion["size"],
    )
    monkeypatch.setattr(finalizer, "PI05_DROID_ALL_SIX_RUNTIME_SHA256", runtime_sha256)
    smoke_calls = []
    monkeypatch.setattr(
        finalizer,
        "validate_immutable_native_all_six_smoke",
        lambda path: smoke_calls.append(path) or smoke,
    )

    result = finalizer.validate_all_six_controller_completion(
        completion_path,
        completion["sha256"],
        finalizer.PI05_DROID_ALL_SIX_CONTROLLER_PROFILE,
        repository,
    )
    assert result["job_id"] == PI05_DROID_ALL_SIX_CONTROLLER_JOB_ID
    assert result["runtime_sha256"] == runtime_sha256
    assert result["smoke"] == smoke
    assert smoke_calls == [Path(smoke["path"])]
    assert set(result["critical_source_files"]) == set(
        PI05_DROID_ALL_SIX_CONTROLLER_CRITICAL_PATHS
    )
    assert set(result["unchanged_policy_io_files"]) == set(
        PI05_DROID_ALL_SIX_UNCHANGED_POLICY_IO_PATHS
    )

    stale_path = tmp_path / "stale-job1098349.completion.json"
    stale_value = copy.deepcopy(completion["value"])
    stale_value["job_id"] = 1098349
    stale = publish_immutable_json(stale_path, stale_value)
    monkeypatch.setattr(
        finalizer, "PI05_DROID_ALL_SIX_CONTROLLER_COMPLETION_PATH", str(stale_path)
    )
    monkeypatch.setattr(
        finalizer,
        "PI05_DROID_ALL_SIX_CONTROLLER_COMPLETION_SHA256",
        stale["sha256"],
    )
    monkeypatch.setattr(
        finalizer, "PI05_DROID_ALL_SIX_CONTROLLER_COMPLETION_SIZE", stale["size"]
    )
    with pytest.raises(ValueError, match="schema or identity mismatch"):
        finalizer.validate_all_six_controller_completion(
            stale_path,
            stale["sha256"],
            finalizer.PI05_DROID_ALL_SIX_CONTROLLER_PROFILE,
            repository,
        )

    monkeypatch.setattr(
        finalizer, "PI05_DROID_ALL_SIX_CONTROLLER_COMPLETION_PATH", str(completion_path)
    )
    monkeypatch.setattr(
        finalizer,
        "PI05_DROID_ALL_SIX_CONTROLLER_COMPLETION_SHA256",
        completion["sha256"],
    )
    monkeypatch.setattr(
        finalizer,
        "PI05_DROID_ALL_SIX_CONTROLLER_COMPLETION_SIZE",
        completion["size"],
    )
    changed = repository / PI05_DROID_ALL_SIX_CONTROLLER_CRITICAL_PATHS[0]
    changed.write_text("changed\n", encoding="utf-8")
    with pytest.raises(
        ValueError,
        match=f"differs from job{PI05_DROID_ALL_SIX_CONTROLLER_JOB_ID}",
    ):
        finalizer.validate_all_six_controller_completion(
            completion_path,
            completion["sha256"],
            finalizer.PI05_DROID_ALL_SIX_CONTROLLER_PROFILE,
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
    alias_run_dir = tmp_path / "run_alias"
    alias_run_dir.symlink_to(run_dir, target_is_directory=True)
    alias_task_dir = alias_run_dir / "DROID-FoodBussing"
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
        "rubric": {"success": False, "progress": 0.0},
    }
    result = {
        "episode": 0,
        "episode_length": 450,
        "success": False,
        "progress": 0.0,
        "numerical_failure": False,
        "numerical_failure_reason": "",
    }
    bound = {}
    for label in ("trace", "video"):
        temporary = task_dir / f".{label}.partial"
        temporary.write_bytes(label.encode("ascii"))
        bound[label] = publish_immutable_file_from_temporary(
            temporary, task_dir / f"{label}.bin"
        )
        bound[label]["path"] = str(alias_task_dir / f"{label}.bin")
    dynamic = {
        "schema_version": 3,
        "profile": NATIVE_GRIPPER_DYNAMIC_PROFILE,
        "joint_names": list(EXPECTED_DROID_JOINT_NAMES),
        "joint_indices": list(range(13)),
        "apply_calls": 3600,
        "post_policy_step_samples": 450,
        "sample_count": 4050,
        "max_abs_joint_velocity_rad_s": [0.0] * 13,
        "max_abs_joint_acceleration_rad_s2": [0.0] * 13,
        "terminal_velocity_failure": None,
        "samples": None,
    }
    sidecar_path = task_dir / "native_runtime" / "episode_000000.json"
    sidecar = publish_immutable_json(
        sidecar_path,
        make_episode_sidecar(
            episode_result=result,
            terminal_outcome=terminal,
            environment_runtime_contract=environment_runtime,
            dynamic_report=dynamic,
            trace_artifact=bound["trace"],
            video_artifact=bound["video"],
            incident_artifact=None,
        ),
    )
    sidecar_identity = {
        key: sidecar[key] for key in ("path", "size", "sha256", "mode", "nlink")
    }
    sidecar_identity["path"] = str(
        alias_task_dir / "native_runtime" / "episode_000000.json"
    )
    close_path = task_dir / "evaluator_close_ready.json"
    close_payload = make_close_ready_artifact(
        runtime_artifact={
            **runtime_identity,
            "path": str(alias_task_dir / "joint_velocity_runtime.json"),
        },
        runtime_path=alias_task_dir / "joint_velocity_runtime.json",
        metrics_path=task_dir / "eval_results.csv",
        trace_path=task_dir / "policy_traces.jsonl",
        video_path=task_dir / "episode_0.mp4",
        environment_runtime_contract=environment_runtime,
        terminal_outcome=terminal,
        episode_sidecar=sidecar_identity,
    )
    close_payload["metrics_path"] = str(alias_task_dir / "eval_results.csv")
    close_payload["trace_path"] = str(alias_task_dir / "policy_traces.jsonl")
    close_payload["video_path"] = str(alias_task_dir / "episode_0.mp4")
    publish_immutable_json(
        close_path,
        close_payload,
    )

    runtime = finalizer._validate_runtime_artifact(runtime_path)
    close = finalizer._validate_close_ready(close_path, runtime, run_dir)
    assert runtime["environment_runtime_contract"] == environment_runtime
    assert close["terminal_outcome"] == terminal
    assert close["episode_sidecar"]["value"]["episode_result"] == result

    type_drifted_close = copy.deepcopy(close_payload)
    type_drifted_close["terminal_outcome"]["rubric"]["progress"] = 0
    assert type_drifted_close["terminal_outcome"] == terminal
    assert finalizer.canonical_json_bytes(type_drifted_close["terminal_outcome"]) != (
        finalizer.canonical_json_bytes(terminal)
    )
    type_drifted_close_path = task_dir / "type-drifted-close.json"
    publish_immutable_json(type_drifted_close_path, type_drifted_close)
    with pytest.raises(ValueError, match="Evaluator close-ready identity mismatch"):
        finalizer._validate_close_ready(type_drifted_close_path, runtime, run_dir)

    wrong_close = copy.deepcopy(close_payload)
    wrong_close["runtime_artifact"]["path"] = str(tmp_path / "wrong-runtime.json")
    wrong_close_path = task_dir / "wrong-close.json"
    publish_immutable_json(wrong_close_path, wrong_close)
    with pytest.raises(ValueError, match="artifact path drift"):
        finalizer._validate_close_ready(wrong_close_path, runtime, run_dir)

    bad_runtime = make_runtime_artifact(
        valid_joint_velocity_smoke_payload["runtime_contract"], environment_runtime
    )
    bad_runtime["environment_runtime_contract"]["live_max_episode_length"] = 450
    bad_path = task_dir / "bad-runtime.json"
    publish_immutable_json(bad_path, bad_runtime)
    with pytest.raises(ValueError, match="environment runtime contract mismatch"):
        finalizer._validate_runtime_artifact(bad_path)


@pytest.mark.parametrize(
    ("label", "filename"),
    [
        ("trace", "policy_traces.jsonl"),
        ("video", "episode_0.mp4"),
    ],
)
def test_sealed_sidecar_artifacts_accept_fsw_fs11_alias_and_reject_drift(
    tmp_path,
    label,
    filename,
):
    fs11 = tmp_path / "lustre" / "fs11" / "attempt"
    fs11.mkdir(parents=True)
    fsw = tmp_path / "lustre" / "fsw"
    fsw.symlink_to(fs11, target_is_directory=True)
    artifact_path = fs11 / filename
    artifact_path.write_bytes(f"{label}-artifact\n".encode("ascii"))
    sealed = finalizer._seal_file(artifact_path, f"sealed {label}")
    sidecar = {**sealed, "path": str(fsw / filename)}

    assert (
        finalizer._validate_sealed_sidecar_artifact(
            sidecar,
            sealed,
            expected_path=artifact_path,
            field=f"episode sidecar {label}",
        )
        == sidecar
    )

    wrong_root = tmp_path / "wrong"
    wrong_root.mkdir()
    wrong_path = wrong_root / filename
    wrong_path.write_bytes(artifact_path.read_bytes())
    finalizer._seal_file(wrong_path, f"wrong {label}")
    with pytest.raises(ValueError, match="artifact path drift"):
        finalizer._validate_sealed_sidecar_artifact(
            {**sidecar, "path": str(wrong_path)},
            sealed,
            expected_path=artifact_path,
            field=f"episode sidecar {label}",
        )

    with pytest.raises(ValueError, match="artifact identity drift"):
        finalizer._validate_sealed_sidecar_artifact(
            {**sidecar, "sha256": "0" * 64},
            sealed,
            expected_path=artifact_path,
            field=f"episode sidecar {label}",
        )


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
    source_probe = finalizer.probe_video(
        source, require_faststart=False, expected_frame_count=450
    )
    assert source_probe["frame_count"] == 450
    assert source_probe["full_decode"] == "pass"

    summary = tmp_path / "summary.mp4"
    summary_probe = finalizer.create_summary_video(
        source, summary, source_frame_count=450
    )
    assert summary_probe["codec"] == "h264"
    assert summary_probe["pixel_format"] == "yuv420p"
    assert summary_probe["frame_count"] == 450
    assert summary_probe["duration_seconds"] == 30.0
    assert summary_probe["faststart"] is True
    assert summary_probe["top_level_boxes"].index("moov") < summary_probe[
        "top_level_boxes"
    ].index("mdat")
    assert (summary.stat().st_mode & 0o777) == 0o444


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg tools are unavailable",
)
def test_short_numerical_failure_summary_is_padded_to_three_seconds(tmp_path):
    source = tmp_path / "failure.mp4"
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
            "8",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    raw = finalizer.probe_video(source, require_faststart=False, expected_frame_count=8)
    assert raw["frame_count"] == 8
    summary = tmp_path / "failure-summary.mp4"
    summary_probe = finalizer.create_summary_video(
        source, summary, source_frame_count=8
    )
    assert summary_probe["frame_count"] == 45
    assert summary_probe["duration_seconds"] == 3.0
    assert summary_probe["faststart"] is True


def test_launchers_block_before_download_or_sbatch_and_forbid_resume():
    eval_source = (
        ROOT / "scripts/polaris/eval_pi05_droid_native_jointvelocity.sh"
    ).read_text(encoding="utf-8")
    submit_source = (
        ROOT / "scripts/polaris/submit_pi05_droid_native_jointvelocity_canary.sh"
    ).read_text(encoding="utf-8")
    sbatch_source = (
        ROOT / "scripts/polaris/l40s_pi05_droid_native_jointvelocity_canary.sbatch"
    ).read_text(encoding="utf-8")
    evaluator_source = (ROOT / "scripts/eval.py").read_text(encoding="utf-8")
    lifecycle_source = (ROOT / "src/polaris/pi05_droid_native_lifecycle.py").read_text(
        encoding="utf-8"
    )

    assert eval_source.index("finalize_pi05_droid_native_jointvelocity_eval.py") < (
        eval_source.index("maybe_download")
    )
    assert submit_source.index(
        "finalize_pi05_droid_native_jointvelocity_eval.py"
    ) < submit_source.index("sbatch --parsable")
    for source in (eval_source, submit_source):
        assert "CONTROLLER_COMPLETION" in source
        assert "ALL_SIX_CONTROLLER_COMPLETION" in source
        assert "EXPECTED_ALL_SIX_COMPLETION_SHA256" in source
        assert "EXPECTED_ALL_SIX_PROFILE" in source
        assert "1098204" not in source
        assert "RESUME_FROM_TASK_DIR" in source
    assert sbatch_source.count("#SBATCH --no-requeue") == 1
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
    assert evaluator_source.index("bind_native_all_joint_failure_path") < (
        evaluator_source.index("env.step(")
    )
    assert (
        evaluator_source.count("except NativeAllJointVelocityLimitError as error:") == 2
    )
    assert evaluator_source.count("finalize_native_velocity_failure(error)") == 3
    post_sample = evaluator_source.index(
        "native_arm_term.record_native_all_joint_post_policy_step()"
    )
    post_failure = evaluator_source.index(
        "except NativeAllJointVelocityLimitError as error:", post_sample
    )
    post_execution = evaluator_source.index(
        "policy_client.record_execution(", post_failure
    )
    post_terminal = evaluator_source.index(
        "finalize_native_velocity_failure(error)", post_failure
    )
    assert post_sample < post_failure < post_execution < post_terminal
    assert evaluator_source.index("make_episode_sidecar(") < evaluator_source.index(
        "episode_df.to_csv"
    )
    assert evaluator_source.index("finish_rollout") < evaluator_source.index(
        "lifecycle.prepare_close_ready"
    )
    assert "finally:\n        lifecycle.close()" in evaluator_source
    assert (
        lifecycle_source.index("self._env.close()")
        < lifecycle_source.index("publisher(path, payload)")
        < lifecycle_source.index("self._simulation_app.close()")
    )
    assert "official_model_eval_contract" in finalizer.finalize.__code__.co_consts
