#!/usr/bin/env python3
"""Independently validate one immutable controller-candidate replay result."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any

import finalize_eef_pose_smoke as safety_validator
import smoke_eef_pose_canary_controller_candidate as candidate
import smoke_eef_pose_canary_trace_replay as gate0


RAW_FIELDS = {
    "schema_version",
    "profile",
    "finalized",
    "passed",
    "stage",
    "environment",
    "variant",
    "candidate",
    "lifecycle",
    "repository",
    "container_image",
    "production_eval",
    "fixture",
    "action_plan",
    "boundary_helper",
    "assets",
    "runtime_protocol",
    "runtime_frame",
    "gripper_runtime_contract",
    "initial_safety",
    "initial_candidate",
    "final_safety",
    "final_candidate",
    "candidate_replay_validation",
    "velocity_headroom",
    "outcome",
    "close_failures",
}
READY_FIELDS = {"schema_version", "profile", "stage", "raw_result"}
FILE_IDENTITY_FIELDS = {"path", "size_bytes", "sha256", "mode"}
ACTION_PLAN = {
    "profile": "exact_fixture_then_repeat_final_recorded_action_v1",
    "fixture_action_count": candidate.FIXTURE_ACTION_COUNT,
    "post_fixture_repeat_count": candidate.POST_FIXTURE_REPEAT_COUNT,
    "total_action_count": candidate.TOTAL_ACTION_COUNT,
}
LIFECYCLE_FIELDS = {
    "profile",
    "launch_id",
    "job_id",
    "step_id",
    "nodelist",
    "procid",
    "localid",
    "ntasks",
}
REPOSITORY_FIELDS = {"path", "commit", "clean_tracked"}
FIXTURE_FIELDS = {
    "path",
    "size_bytes",
    "sha256",
    "mode",
    "source_trace_sha256",
    "action_float32_sha256",
    "fixture_action_count",
}
RUNTIME_FRAME_FIELDS = {
    "eef_frame",
    "reference_frame",
    "controlled_body",
    "command_type",
    "use_relative_mode",
    "ik_method",
    "dls_damping",
    "arm_scale",
    "body_offset",
    "action_dim",
    "arm_joint_names",
    "gripper_threshold_profile",
    "ik_safety_profile",
    "position_error_m",
    "rotation_error_rad",
}
RUNTIME_PROTOCOL_FIELDS = {
    "profile",
    "episode_steps",
    "live_max_episode_length",
    "autoreset_margin_steps",
    "policy_hz",
    "step_dt",
    "physics_hz",
    "physics_dt",
    "decimation",
    "camera_sensor_names",
}


class CandidateArtifactValidationError(ValueError):
    """An immutable candidate artifact violated the independent contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateArtifactValidationError(message)


def _typed_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(right, dict):
        return set(left) == set(right) and all(
            _typed_equal(left[name], value) for name, value in right.items()
        )
    if isinstance(right, list):
        return len(left) == len(right) and all(
            _typed_equal(actual, expected)
            for actual, expected in zip(left, right, strict=True)
        )
    return bool(left == right)


def _validate_lifecycle(
    value: Any,
    *,
    launch_id: str,
    job_id: int,
    field: str,
) -> dict[str, Any]:
    """Close one single-rank Slurm lifecycle with exact JSON scalar types."""

    _require(
        isinstance(value, dict)
        and set(value) == LIFECYCLE_FIELDS
        and type(value.get("profile")) is str
        and value.get("profile") == "slurm_single_task_srun_lifecycle_v1"
        and type(value.get("launch_id")) is str
        and value.get("launch_id") == launch_id
        and type(value.get("job_id")) is int
        and value.get("job_id") == job_id
        and type(value.get("step_id")) is int
        and value["step_id"] >= 0
        and type(value.get("nodelist")) is str
        and bool(value["nodelist"].strip())
        and type(value.get("procid")) is int
        and value.get("procid") == 0
        and type(value.get("localid")) is int
        and value.get("localid") == 0
        and type(value.get("ntasks")) is int
        and value.get("ntasks") == 1,
        f"{field} drift",
    )
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _immutable_file(path: Path, *, expected_mode: int = 0o444) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing/linked file: {path}")
    metadata = path.stat()
    _require(metadata.st_nlink == 1, f"file link count drift: {path}")
    _require(
        stat.S_IMODE(metadata.st_mode) == expected_mode,
        f"file mode drift: {path}",
    )
    return {
        "path": str(path.resolve()),
        "size_bytes": metadata.st_size,
        "sha256": _sha256(path),
        "mode": f"{expected_mode:04o}",
    }


def _same_recorded_file(record: Any, actual: dict[str, Any], *, field: str) -> None:
    _require(
        isinstance(record, dict) and set(record) == FILE_IDENTITY_FIELDS,
        f"{field} identity schema",
    )
    _require(
        all(
            _typed_equal(record.get(name), actual[name])
            for name in ("size_bytes", "sha256", "mode")
        ),
        f"{field} content identity drift",
    )
    recorded_path = record.get("path")
    _require(isinstance(recorded_path, str) and recorded_path.startswith("/"), field)
    _require(
        os.path.samefile(recorded_path, actual["path"]),
        f"{field} path is not the same file",
    )


def _compare_with_samefile_paths(recorded: Any, expected: Any, *, field: str) -> None:
    if isinstance(expected, dict):
        _require(isinstance(recorded, dict), f"{field} object type drift")
        _require(set(recorded) == set(expected), f"{field} object schema drift")
        for key in expected:
            child = f"{field}.{key}"
            if key == "path":
                _require(
                    isinstance(recorded[key], str)
                    and isinstance(expected[key], str)
                    and os.path.samefile(recorded[key], expected[key]),
                    f"{child} is not the same file",
                )
            else:
                _compare_with_samefile_paths(recorded[key], expected[key], field=child)
        return
    if isinstance(expected, list):
        _require(isinstance(recorded, list), f"{field} list type drift")
        _require(len(recorded) == len(expected), f"{field} list length drift")
        for index, (left, right) in enumerate(zip(recorded, expected, strict=True)):
            _compare_with_samefile_paths(left, right, field=f"{field}[{index}]")
        return
    _require(_typed_equal(recorded, expected), f"{field} value/type drift")


def _repository_identity(path: Path, commit: str) -> dict[str, Any]:
    _require(path.is_dir() and (path / ".git").exists(), "repository path")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=path,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    _require(head == commit and status == "", "repository commit/cleanliness drift")
    return {"path": str(path.resolve()), "commit": head, "clean_tracked": True}


def _validate_production_eval(value: Any) -> dict[str, Any]:
    _require(isinstance(value, dict), "production eval must be an object")
    live = gate0.validate_production_reset_source()
    _require(set(value) == set(live), "production eval schema drift")
    recorded = dict(value)
    expected = dict(live)
    for name in (None, "policy_config_source", "lap_client_source"):
        if name is None:
            recorded_identity = {
                field: recorded[field] for field in FILE_IDENTITY_FIELDS
            }
            expected_identity = {
                field: expected[field] for field in FILE_IDENTITY_FIELDS
            }
            _same_recorded_file(
                recorded_identity, expected_identity, field="production eval"
            )
            recorded["path"] = expected["path"]
        else:
            _same_recorded_file(recorded[name], expected[name], field=name)
            recorded[name] = expected[name]
    _require(
        _typed_equal(recorded, expected),
        "production eval content contract value/type drift",
    )
    return value


def _validate_runtime_protocol(value: Any) -> dict[str, Any]:
    _require(isinstance(value, dict), "runtime protocol must be an object")
    _require(
        set(value) == RUNTIME_PROTOCOL_FIELDS,
        "runtime protocol schema drift",
    )
    expected = {
        "profile": "ego_lap_eef_outer450_internal451_no_autoreset_v1",
        "episode_steps": 450,
        "live_max_episode_length": 451,
        "autoreset_margin_steps": 1,
        "policy_hz": 15.0,
        "step_dt": 1.0 / 15.0,
        "physics_hz": 120.0,
        "physics_dt": 1.0 / 120.0,
        "decimation": 8,
        "camera_sensor_names": ["external_cam", "wrist_cam"],
    }
    _require(_typed_equal(value, expected), "runtime protocol value/type drift")
    return value


def _validate_runtime_frame(value: Any) -> dict[str, Any]:
    _require(isinstance(value, dict), "runtime frame must be an object")
    _require(set(value) == RUNTIME_FRAME_FIELDS, "runtime frame schema drift")
    expected = {
        "eef_frame": "panda_link8",
        "reference_frame": "panda_link0",
        "controlled_body": "panda_link8",
        "command_type": "pose",
        "use_relative_mode": False,
        "ik_method": "dls",
        "dls_damping": 0.01,
        "arm_scale": 1.0,
        "body_offset": "identity",
        "action_dim": 7,
        "arm_joint_names": [f"panda_joint{index}" for index in range(1, 8)],
        "gripper_threshold_profile": ("closed_positive_ge_0p5_inverse_open_gt_0p5_v1"),
        "ik_safety_profile": "panda_velocity_physxlimit_solveriter1_v4",
    }
    for field, wanted in expected.items():
        _require(
            _typed_equal(value.get(field), wanted),
            f"runtime frame {field} value/type drift",
        )
    position_error = value.get("position_error_m")
    rotation_error = value.get("rotation_error_rad")
    _require(
        isinstance(position_error, (int, float))
        and not isinstance(position_error, bool)
        and math.isfinite(float(position_error))
        and 0.0 <= float(position_error) <= 1e-5,
        "runtime frame position error",
    )
    _require(
        isinstance(rotation_error, (int, float))
        and not isinstance(rotation_error, bool)
        and math.isfinite(float(rotation_error))
        and 0.0 <= float(rotation_error) <= math.radians(0.01),
        "runtime frame rotation error",
    )
    return value


def _validate_offline_safety(
    value: Any,
    *,
    field: str,
    apply_calls: int,
    expect_closed_target: bool,
    expected_endpoint_change_count: int,
    expected_gripper_target_slew_profile: str,
) -> dict[str, Any]:
    try:
        safety_validator._validate_safety_report(
            value,
            field=field,
            episode_index=0,
            apply_calls=apply_calls,
            expect_closed_target=expect_closed_target,
            expected_endpoint_change_count=expected_endpoint_change_count,
            expected_gripper_target_slew_profile=(expected_gripper_target_slew_profile),
        )
    except (TypeError, ValueError) as error:
        raise CandidateArtifactValidationError(
            f"{field} is invalid: {error}"
        ) from error
    return value


def validate(args: argparse.Namespace) -> dict[str, Any]:
    for name, value, width in (
        ("launch ID", args.launch_id, 64),
        ("PolaRiS commit", args.expected_polaris_commit, 40),
        ("runner SHA-256", args.expected_runner_sha256, 64),
        ("validator SHA-256", args.expected_validator_sha256, 64),
        ("safety validator SHA-256", args.expected_safety_validator_sha256, 64),
        ("Gate0 helper SHA-256", args.expected_gate0_helper_sha256, 64),
        ("fixture SHA-256", args.expected_fixture_sha256, 64),
        ("container SHA-256", args.expected_container_sha256, 64),
    ):
        _require(
            isinstance(value, str)
            and re.fullmatch(rf"[0-9a-f]{{{width}}}", value) is not None,
            f"invalid {name}",
        )
    _require(type(args.job_id) is int and args.job_id > 0, "job ID")
    _require(
        type(args.expected_container_size_bytes) is int
        and args.expected_container_size_bytes > 0,
        "container size",
    )

    repo = args.polaris_repo.resolve()
    repository = _repository_identity(repo, args.expected_polaris_commit)
    source_specs = {
        "runner": (
            repo / "scripts/smoke_eef_pose_canary_controller_candidate.py",
            args.expected_runner_sha256,
        ),
        "validator": (
            repo / "scripts/validate_eef_pose_canary_controller_candidate.py",
            args.expected_validator_sha256,
        ),
        "safety_validator": (
            repo / "scripts/finalize_eef_pose_smoke.py",
            args.expected_safety_validator_sha256,
        ),
        "gate0_helper": (
            repo / "scripts/smoke_eef_pose_canary_trace_replay.py",
            args.expected_gate0_helper_sha256,
        ),
        "fixture": (
            repo
            / "scripts/fixtures"
            / gate0.EXPECTED_FIXTURES[args.variant]["filename"],
            args.expected_fixture_sha256,
        ),
    }
    source_identities: dict[str, Any] = {}
    for name, (path, expected_sha256) in source_specs.items():
        _require(path.is_file() and not path.is_symlink(), f"missing source {name}")
        identity = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        _require(identity["sha256"] == expected_sha256, f"source hash drift: {name}")
        source_identities[name] = identity
    for module, source_name in (
        (candidate, "runner"),
        (safety_validator, "safety_validator"),
        (gate0, "gate0_helper"),
    ):
        _require(
            os.path.samefile(module.__file__, source_identities[source_name]["path"]),
            f"imported module differs from hashed source: {source_name}",
        )

    container = args.container_image
    _require(
        container.is_file() and not container.is_symlink(),
        "container must be a regular non-symlink file",
    )
    _require(
        container.stat().st_size == args.expected_container_size_bytes
        and _sha256(container) == args.expected_container_sha256,
        "container content identity drift",
    )
    container_record = candidate.validate_container_argument(
        str(container),
        size_bytes=args.expected_container_size_bytes,
        sha256=args.expected_container_sha256,
    )

    raw_identity = _immutable_file(args.raw_result)
    ready_identity = _immutable_file(args.ready_marker)
    raw = gate0.strict_json_loads(args.raw_result.read_bytes(), field="candidate raw")
    ready = gate0.strict_json_loads(
        args.ready_marker.read_bytes(), field="candidate ready marker"
    )
    _require(set(raw) == RAW_FIELDS, "candidate raw top-level schema drift")
    _require(set(ready) == READY_FIELDS, "candidate ready schema drift")
    _require(
        type(raw.get("schema_version")) is int
        and raw.get("schema_version") == 1
        and raw.get("profile") == candidate.PROFILE
        and raw.get("finalized") is False
        and raw.get("passed") is True
        and raw.get("stage") == "simulation_app_close_pending"
        and raw.get("environment") == gate0.ENVIRONMENT
        and raw.get("variant") == args.variant
        and raw.get("candidate") == candidate.CANDIDATE_BY_VARIANT[args.variant]
        and raw.get("close_failures") == [],
        "candidate raw identity/outcome drift",
    )
    _require(
        type(ready.get("schema_version")) is int
        and ready.get("schema_version") == 1
        and ready.get("profile") == candidate.PROFILE
        and ready.get("stage") == "simulation_app_close_pending",
        "candidate ready identity drift",
    )
    _same_recorded_file(ready.get("raw_result"), raw_identity, field="ready raw")

    _validate_lifecycle(
        raw.get("lifecycle"),
        launch_id=args.launch_id,
        job_id=args.job_id,
        field="candidate Slurm lifecycle",
    )
    recorded_repository = raw.get("repository")
    _require(
        isinstance(recorded_repository, dict)
        and set(recorded_repository) == REPOSITORY_FIELDS,
        "recorded repository schema",
    )
    _require(
        recorded_repository.get("commit") == repository["commit"]
        and recorded_repository.get("clean_tracked") is True
        and os.path.samefile(recorded_repository.get("path"), repository["path"]),
        "recorded repository identity drift",
    )
    _require(
        _typed_equal(raw.get("container_image"), container_record),
        "container record value/type drift",
    )

    _validate_production_eval(raw.get("production_eval"))
    fixture = raw.get("fixture")
    live_fixture_identity, live_fixture_payload, live_actions = (
        gate0.load_replay_fixture(args.variant)
    )
    _require(
        isinstance(fixture, dict)
        and set(fixture) == FIXTURE_FIELDS
        and fixture.get("sha256") == args.expected_fixture_sha256
        and type(fixture.get("fixture_action_count")) is int
        and fixture.get("fixture_action_count") == candidate.FIXTURE_ACTION_COUNT
        and len(live_actions) == candidate.FIXTURE_ACTION_COUNT
        and fixture.get("source_trace_sha256")
        == live_fixture_payload["source"]["trace_sha256"]
        and fixture.get("action_float32_sha256")
        == live_fixture_payload["action_encoding"]["uncompressed_sha256"],
        "candidate fixture identity drift",
    )
    _same_recorded_file(
        {field: fixture[field] for field in FILE_IDENTITY_FIELDS},
        live_fixture_identity,
        field="candidate fixture",
    )
    _require(
        _typed_equal(raw.get("action_plan"), ACTION_PLAN),
        "candidate action plan value/type drift",
    )

    boundary, live_boundary_identity = gate0._load_boundary_helper()
    _same_recorded_file(
        raw.get("boundary_helper"),
        live_boundary_identity,
        field="boundary helper",
    )
    assets = raw.get("assets")
    _require(
        isinstance(assets, dict)
        and set(assets) == {"contract", "scene", "robot_usd"}
        and _typed_equal(assets.get("contract"), gate0.EXPECTED_ASSET_CONTRACT),
        "candidate asset contract drift",
    )
    recorded_scene = assets.get("scene")
    _require(isinstance(recorded_scene, dict), "candidate scene evidence")
    scene_identity = recorded_scene.get("scene")
    _require(isinstance(scene_identity, dict), "candidate scene file evidence")
    scene_path = scene_identity.get("path")
    _require(isinstance(scene_path, str), "candidate scene path")
    live_scene = boundary.validate_asset_contract(Path(scene_path))
    _compare_with_samefile_paths(recorded_scene, live_scene, field="candidate scene")
    robot_record = assets.get("robot_usd")
    _require(isinstance(robot_record, dict), "candidate robot USD evidence")
    robot_path = robot_record.get("path")
    _require(isinstance(robot_path, str), "candidate robot USD path")
    live_robot = gate0._file_identity(Path(robot_path))
    _same_recorded_file(robot_record, live_robot, field="candidate robot USD")
    _require(
        live_robot["sha256"] == gate0.EXPECTED_ROBOT_USD_SHA256,
        "candidate robot USD digest drift",
    )

    _validate_runtime_protocol(raw.get("runtime_protocol"))
    _validate_runtime_frame(raw.get("runtime_frame"))
    try:
        safety_validator._validate_gripper_static(
            raw.get("gripper_runtime_contract"),
            field="candidate gripper runtime contract",
            expected_target_slew_profile=(candidate.CANDIDATE_TARGET_SLEW_PROFILE),
        )
    except (TypeError, ValueError) as error:
        raise CandidateArtifactValidationError(
            f"candidate gripper runtime contract is invalid: {error}"
        ) from error

    initial_safety = _validate_offline_safety(
        raw.get("initial_safety"),
        field="initial safety",
        apply_calls=0,
        expect_closed_target=False,
        expected_endpoint_change_count=0,
        expected_gripper_target_slew_profile=(candidate.CANDIDATE_TARGET_SLEW_PROFILE),
    )
    final_safety = _validate_offline_safety(
        raw.get("final_safety"),
        field="final safety",
        apply_calls=candidate.TOTAL_ACTION_COUNT * gate0.DECIMATION,
        expect_closed_target=args.variant == "official_lap3b",
        expected_endpoint_change_count=int(args.variant == "official_lap3b"),
        expected_gripper_target_slew_profile=(candidate.CANDIDATE_TARGET_SLEW_PROFILE),
    )
    _require(
        _typed_equal(
            raw.get("gripper_runtime_contract"),
            initial_safety["gripper_runtime_static"],
        )
        and _typed_equal(
            raw.get("gripper_runtime_contract"),
            final_safety["gripper_runtime_static"],
        ),
        "candidate gripper static contract changed across replay",
    )

    initial_candidate = candidate.validate_candidate_report(
        raw.get("initial_candidate"), variant=args.variant, final=False
    )
    final_candidate = candidate.validate_candidate_report(
        raw.get("final_candidate"), variant=args.variant, final=True
    )
    replay_validation = candidate.validate_candidate_replay_evidence(
        final_safety, final_candidate, variant=args.variant
    )
    _require(
        _typed_equal(raw.get("candidate_replay_validation"), replay_validation),
        "candidate replay validation summary drift",
    )
    velocity = candidate.validate_velocity_headroom(final_safety)
    _require(
        _typed_equal(raw.get("velocity_headroom"), velocity),
        "velocity summary value/type drift",
    )
    expected_outcome = {
        "status": "candidate_replay_completed_without_controller_abort",
        "actions_completed": candidate.TOTAL_ACTION_COUNT,
        "original_failure_step_crossed": gate0.EXPECTED_FIXTURES[args.variant][
            "failure"
        ]["policy_step"],
        "post_fixture_release_probe_completed": True,
    }
    _require(
        _typed_equal(raw.get("outcome"), expected_outcome),
        "candidate outcome value/type drift",
    )
    return {
        "profile": "polaris_eef_candidate_independent_artifact_validation_v1",
        "variant": args.variant,
        "job_id": args.job_id,
        "launch_id": args.launch_id,
        "raw_result": raw_identity,
        "ready_marker": ready_identity,
        "repository": repository,
        "container_image": container_record,
        "sources": source_identities,
        "initial_candidate": initial_candidate,
        "final_candidate": final_candidate,
        "replay_validation": replay_validation,
        "velocity_headroom": velocity,
    }


def validate_failure(args: argparse.Namespace) -> dict[str, Any]:
    """Independently verify one immutable, non-promotable failed raw."""

    for name, value, width in (
        ("launch ID", args.launch_id, 64),
        ("PolaRiS commit", args.expected_polaris_commit, 40),
        ("runner SHA-256", args.expected_runner_sha256, 64),
        ("validator SHA-256", args.expected_validator_sha256, 64),
        ("failure verifier SHA-256", args.expected_failure_verifier_sha256, 64),
        ("safety validator SHA-256", args.expected_safety_validator_sha256, 64),
        ("Gate0 helper SHA-256", args.expected_gate0_helper_sha256, 64),
        ("fixture SHA-256", args.expected_fixture_sha256, 64),
        ("container SHA-256", args.expected_container_sha256, 64),
    ):
        _require(
            isinstance(value, str)
            and re.fullmatch(rf"[0-9a-f]{{{width}}}", value) is not None,
            f"invalid {name}",
        )
    _require(type(args.job_id) is int and args.job_id > 0, "job ID")
    _require(
        type(args.expected_container_size_bytes) is int
        and args.expected_container_size_bytes > 0,
        "container size",
    )

    repo = args.polaris_repo.resolve()
    repository = _repository_identity(repo, args.expected_polaris_commit)
    source_specs = {
        "runner": (
            repo / "scripts/smoke_eef_pose_canary_controller_candidate.py",
            args.expected_runner_sha256,
        ),
        "validator": (
            repo / "scripts/validate_eef_pose_canary_controller_candidate.py",
            args.expected_validator_sha256,
        ),
        "failure_verifier": (
            args.failure_verifier.resolve(),
            args.expected_failure_verifier_sha256,
        ),
        "safety_validator": (
            repo / "scripts/finalize_eef_pose_smoke.py",
            args.expected_safety_validator_sha256,
        ),
        "gate0_helper": (
            repo / "scripts/smoke_eef_pose_canary_trace_replay.py",
            args.expected_gate0_helper_sha256,
        ),
        "fixture": (
            repo
            / "scripts/fixtures"
            / gate0.EXPECTED_FIXTURES[args.variant]["filename"],
            args.expected_fixture_sha256,
        ),
    }
    source_identities: dict[str, Any] = {}
    for name, (path, expected_sha256) in source_specs.items():
        _require(path.is_file() and not path.is_symlink(), f"missing source {name}")
        identity = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        _require(identity["sha256"] == expected_sha256, f"source hash drift: {name}")
        source_identities[name] = identity
    for module, source_name in (
        (candidate, "runner"),
        (safety_validator, "safety_validator"),
        (gate0, "gate0_helper"),
    ):
        _require(
            os.path.samefile(module.__file__, source_identities[source_name]["path"]),
            f"imported module differs from hashed source: {source_name}",
        )

    container = args.container_image.resolve()
    _require(
        container.is_file() and not container.is_symlink(),
        "container must be a regular non-symlink file",
    )
    _require(
        container.stat().st_size == args.expected_container_size_bytes
        and _sha256(container) == args.expected_container_sha256,
        "container content identity drift",
    )
    container_record = candidate.validate_container_argument(
        str(container),
        size_bytes=args.expected_container_size_bytes,
        sha256=args.expected_container_sha256,
    )

    raw_identity = _immutable_file(args.raw_result)
    namespace = args.raw_result.resolve().parent
    _require(
        args.raw_result.name == f"candidate-{args.variant}.raw.json"
        and namespace.name == f"launch_{args.launch_id}"
        and namespace.parent.name == f"job_{args.job_id}"
        and namespace.parent.parent.name == args.variant,
        "failed raw variant/job/launch namespace",
    )
    for forbidden in (
        args.raw_result.with_name(args.raw_result.name + ".ready.json"),
        namespace / f"candidate-{args.variant}.srun-status.json",
        namespace / f"candidate-{args.variant}.attestation.json",
    ):
        _require(
            not forbidden.exists() and not forbidden.is_symlink(),
            f"failed raw has a promotion artifact: {forbidden.name}",
        )
    raw = gate0.strict_json_loads(args.raw_result.read_bytes(), field="failure raw")
    candidate.validate_failure_payload(
        raw,
        variant=args.variant,
        require_complete_capture=True,
    )
    context = raw.get("failure_context")
    _require(isinstance(context, dict), "failed raw context drift")
    _validate_lifecycle(
        context.get("lifecycle"),
        launch_id=args.launch_id,
        job_id=args.job_id,
        field="failed raw lifecycle",
    )
    recorded_repository = context["repository"]
    _require(
        recorded_repository["commit"] == repository["commit"]
        and recorded_repository["clean_tracked"] is True
        and os.path.samefile(recorded_repository["path"], repository["path"]),
        "failed raw repository drift",
    )
    _require(
        _typed_equal(context["container_image"], container_record),
        "failed raw container drift",
    )
    _validate_production_eval(context["production_eval"])
    live_fixture_identity, live_fixture_payload, live_actions = (
        gate0.load_replay_fixture(args.variant)
    )
    fixture = context["fixture"]
    _require(
        fixture.get("sha256") == args.expected_fixture_sha256
        and fixture.get("fixture_action_count") == candidate.FIXTURE_ACTION_COUNT
        and len(live_actions) == candidate.FIXTURE_ACTION_COUNT
        and fixture.get("source_trace_sha256")
        == live_fixture_payload["source"]["trace_sha256"]
        and fixture.get("action_float32_sha256")
        == live_fixture_payload["action_encoding"]["uncompressed_sha256"],
        "failed raw fixture drift",
    )
    _same_recorded_file(
        {field: fixture[field] for field in FILE_IDENTITY_FIELDS},
        live_fixture_identity,
        field="failed raw fixture",
    )
    _require(
        _typed_equal(context["action_plan"], ACTION_PLAN),
        "failed raw action plan drift",
    )
    boundary, live_boundary_identity = gate0._load_boundary_helper()  # noqa: SLF001
    _same_recorded_file(
        context["boundary_helper"],
        live_boundary_identity,
        field="failed raw boundary helper",
    )
    assets = context["assets"]
    _require(
        isinstance(assets, dict)
        and set(assets) == {"contract", "scene", "robot_usd"}
        and _typed_equal(assets["contract"], gate0.EXPECTED_ASSET_CONTRACT),
        "failed raw asset contract",
    )
    scene_path = assets.get("scene", {}).get("scene", {}).get("path")
    _require(isinstance(scene_path, str), "failed raw scene path")
    _compare_with_samefile_paths(
        assets["scene"],
        boundary.validate_asset_contract(Path(scene_path)),
        field="failed raw scene",
    )
    robot_path = assets.get("robot_usd", {}).get("path")
    _require(isinstance(robot_path, str), "failed raw robot path")
    _same_recorded_file(
        assets["robot_usd"],
        gate0._file_identity(Path(robot_path)),  # noqa: SLF001
        field="failed raw robot USD",
    )
    _validate_runtime_protocol(context["runtime_protocol"])
    _validate_runtime_frame(context["runtime_frame"])
    safety_validator._validate_gripper_static(
        context["gripper_runtime_contract"],
        field="failed raw gripper runtime contract",
        expected_target_slew_profile=candidate.CANDIDATE_TARGET_SLEW_PROFILE,
    )
    initial_safety = _validate_offline_safety(
        context["initial_safety"],
        field="failed raw initial safety",
        apply_calls=0,
        expect_closed_target=False,
        expected_endpoint_change_count=0,
        expected_gripper_target_slew_profile=(candidate.CANDIDATE_TARGET_SLEW_PROFILE),
    )
    _require(
        _typed_equal(
            initial_safety["gripper_runtime_static"],
            context["gripper_runtime_contract"],
        ),
        "failed raw initial gripper binding",
    )
    candidate.validate_candidate_report(
        context["initial_candidate"], variant=args.variant, final=False
    )

    capture = raw["controller_abort_capture"]
    parsed = capture["parsed_failure"]
    arm_failure = capture["arm_failure_runtime_evidence"]
    gate0._validate_arm_failure_runtime_evidence(  # noqa: SLF001
        arm_failure,
        expected_failure=parsed,
    )
    gate0.validate_gripper_tail(
        capture["all_six_gripper_tail"],
        expected_failure=parsed,
    )
    active_safety = capture["active_safety"]
    counters = active_safety.get("counters")
    dynamic = active_safety.get("gripper_runtime_dynamic")
    target_slew = capture["active_target_slew"]
    _require(
        isinstance(counters, dict)
        and type(counters.get("apply_calls")) is int
        and counters["apply_calls"] >= 2
        and isinstance(dynamic, dict)
        and dynamic.get("apply_entry_samples") == counters["apply_calls"]
        and dynamic.get("dropped_diagnostics") == 0
        and target_slew == dynamic.get("driver_target_slew")
        and target_slew.get("profile") == candidate.CANDIDATE_TARGET_SLEW_PROFILE
        and target_slew.get("process_action_calls") == raw["policy_step"] + 1
        and target_slew.get("apply_calls") == counters["apply_calls"] - 1
        and target_slew.get("live_limit_validation_count")
        == target_slew.get("apply_calls")
        and target_slew.get("slew_limited_apply_count")
        + target_slew.get("endpoint_reached_apply_count")
        == target_slew.get("apply_calls"),
        "failed raw active gripper/target-slew cadence",
    )
    _require(
        isinstance(target_slew.get("max_abs_target_step_rad"), (int, float))
        and not isinstance(target_slew["max_abs_target_step_rad"], bool)
        and 0.0
        <= float(target_slew["max_abs_target_step_rad"])
        <= candidate.CANDIDATE_TARGET_SLEW_MAX_STEP_RAD + 1e-6,
        "failed raw target-slew maximum",
    )
    _require(
        active_safety.get("gripper_runtime_static")
        == context["gripper_runtime_contract"],
        "failed raw active gripper static drift",
    )
    candidate.validate_candidate_report(
        capture["active_candidate"], variant=args.variant, final=None
    )
    return {
        "profile": "polaris_eef_controller_candidate_failure_verification_v1",
        "variant": args.variant,
        "job_id": args.job_id,
        "launch_id": args.launch_id,
        "raw_result": raw_identity,
        "repository": repository,
        "container_image": container_record,
        "sources": source_identities,
        "policy_step": raw["policy_step"],
        "physics_substep": parsed["physics_substep"],
        "joint_name": parsed["joint_name"],
        "evidence_sha256": parsed["evidence_sha256"],
        "arm_apply_calls": counters["apply_calls"],
        "gripper_apply_calls": target_slew["apply_calls"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=sorted(candidate.CANDIDATE_BY_VARIANT), required=True
    )
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--raw-result", type=Path, required=True)
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--polaris-repo", type=Path, required=True)
    parser.add_argument("--expected-polaris-commit", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-validator-sha256", required=True)
    parser.add_argument("--expected-safety-validator-sha256", required=True)
    parser.add_argument("--expected-gate0-helper-sha256", required=True)
    parser.add_argument("--expected-fixture-sha256", required=True)
    parser.add_argument("--container-image", type=Path, required=True)
    parser.add_argument("--expected-container-size-bytes", type=int, required=True)
    parser.add_argument("--expected-container-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    result = validate(_parse_args())
    print(
        "POLARIS_CONTROLLER_CANDIDATE_VALIDATION_PASS="
        f"{result['variant']};job={result['job_id']};"
        f"raw_sha256={result['raw_result']['sha256']};"
        f"max_velocity_ratio={result['velocity_headroom']['maximum_ratio']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
