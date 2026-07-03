from __future__ import annotations

from argparse import Namespace
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPTS))
    try:
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPTS))
    return module


validator = _load_script("validate_eef_pose_canary_controller_candidate")


def _load_test_module(filename: str, module_name: str):
    path = ROOT / "tests" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


runtime_tests = _load_test_module(
    "test_eef_runtime_contract.py", "test_eef_runtime_contract_for_candidate_consumer"
)
gripper_tests = _load_test_module(
    "test_eef_gripper_runtime.py", "test_eef_gripper_runtime_for_candidate_consumer"
)


def _failure_validation_args(
    tmp_path: Path,
    lifecycle: dict,
    *,
    container_image: Path | None = None,
    context_fields: dict | None = None,
) -> Namespace:
    variant = "official_lap3b"
    launch_id = "a" * 64
    job_id = 123
    namespace = tmp_path / variant / f"job_{job_id}" / f"launch_{launch_id}"
    raw_result = namespace / f"candidate-{variant}.raw.json"
    context = {"lifecycle": lifecycle}
    if context_fields is not None:
        context.update(context_fields)
    validator.gate0._atomic_write_immutable(  # noqa: SLF001
        raw_result,
        {"failure_context": context},
    )
    container = container_image or tmp_path / "container.sqsh"
    if container_image is None:
        container.write_bytes(b"lifecycle validation container")
    fixture = (
        SCRIPTS / "fixtures" / validator.gate0.EXPECTED_FIXTURES[variant]["filename"]
    )
    return Namespace(
        variant=variant,
        launch_id=launch_id,
        job_id=job_id,
        raw_result=raw_result,
        polaris_repo=ROOT,
        expected_polaris_commit="b" * 40,
        expected_runner_sha256=validator._sha256(  # noqa: SLF001
            SCRIPTS / "smoke_eef_pose_canary_controller_candidate.py"
        ),
        expected_validator_sha256=validator._sha256(  # noqa: SLF001
            SCRIPTS / "validate_eef_pose_canary_controller_candidate.py"
        ),
        expected_failure_verifier_sha256=validator._sha256(  # noqa: SLF001
            SCRIPTS / "verify_eef_pose_canary_controller_candidate_failure.py"
        ),
        expected_safety_validator_sha256=validator._sha256(  # noqa: SLF001
            SCRIPTS / "finalize_eef_pose_smoke.py"
        ),
        expected_gate0_helper_sha256=validator._sha256(  # noqa: SLF001
            SCRIPTS / "smoke_eef_pose_canary_trace_replay.py"
        ),
        expected_fixture_sha256=validator._sha256(fixture),  # noqa: SLF001
        failure_verifier=(
            SCRIPTS / "verify_eef_pose_canary_controller_candidate_failure.py"
        ),
        container_image=container,
        expected_container_size_bytes=container.stat().st_size,
        expected_container_sha256=validator._sha256(container),  # noqa: SLF001
    )


def _container_file(path: Path, content: bytes = b"immutable container") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_container_contract_preserves_parent_symlink_lexical_alias(tmp_path):
    canonical = _container_file(tmp_path / "fs11" / "container.sqsh")
    alias_directory = tmp_path / "fsw"
    alias_directory.symlink_to(canonical.parent, target_is_directory=True)
    lexical_alias = alias_directory / canonical.name
    assert lexical_alias.is_file()
    assert not lexical_alias.is_symlink()
    assert lexical_alias.resolve() == canonical
    digest = validator._sha256(canonical)  # noqa: SLF001
    live = validator._validate_container_file(  # noqa: SLF001
        lexical_alias,
        expected_size_bytes=canonical.stat().st_size,
        expected_sha256=digest,
    )
    recorded = validator.candidate.validate_container_argument(
        str(canonical),
        size_bytes=canonical.stat().st_size,
        sha256=digest,
    )
    assert live["path"] == str(lexical_alias)
    assert (
        validator._validate_recorded_container(  # noqa: SLF001
            recorded,
            live,
            field="container",
        )
        is recorded
    )


def test_container_contract_accepts_hardlink_samefile_alias(tmp_path):
    recorded_path = _container_file(tmp_path / "fsw" / "container.sqsh")
    live_alias = tmp_path / "fs11" / "container.sqsh"
    live_alias.parent.mkdir(parents=True)
    live_alias.hardlink_to(recorded_path)
    digest = validator._sha256(recorded_path)  # noqa: SLF001
    live = validator._validate_container_file(  # noqa: SLF001
        live_alias,
        expected_size_bytes=recorded_path.stat().st_size,
        expected_sha256=digest,
    )
    recorded = validator.candidate.validate_container_argument(
        str(recorded_path),
        size_bytes=recorded_path.stat().st_size,
        sha256=digest,
    )
    assert live["path"] == str(live_alias)
    assert (
        validator._validate_recorded_container(  # noqa: SLF001
            recorded,
            live,
            field="container",
        )
        is recorded
    )


def test_container_contract_rejects_symlink_input_and_wrong_identity(tmp_path):
    container = _container_file(tmp_path / "container.sqsh")
    symlink = tmp_path / "container-link.sqsh"
    symlink.symlink_to(container)
    digest = validator._sha256(container)  # noqa: SLF001
    with pytest.raises(
        validator.CandidateArtifactValidationError,
        match="regular non-symlink",
    ):
        validator._validate_container_file(  # noqa: SLF001
            symlink,
            expected_size_bytes=container.stat().st_size,
            expected_sha256=digest,
        )
    with pytest.raises(
        validator.CandidateArtifactValidationError,
        match="path must be absolute",
    ):
        validator._validate_container_file(  # noqa: SLF001
            Path("container.sqsh"),
            expected_size_bytes=container.stat().st_size,
            expected_sha256=digest,
        )
    for size_bytes, sha256 in (
        (container.stat().st_size + 1, digest),
        (container.stat().st_size, "f" * 64),
    ):
        with pytest.raises(
            validator.CandidateArtifactValidationError,
            match="content identity drift",
        ):
            validator._validate_container_file(  # noqa: SLF001
                container,
                expected_size_bytes=size_bytes,
                expected_sha256=sha256,
            )


def test_container_contract_rejects_copy_and_record_tampering(tmp_path):
    recorded_path = _container_file(tmp_path / "recorded" / "container.sqsh")
    alias = tmp_path / "alias" / "container.sqsh"
    alias.parent.mkdir(parents=True)
    alias.hardlink_to(recorded_path)
    independent_copy = _container_file(
        tmp_path / "copy" / "container.sqsh",
        recorded_path.read_bytes(),
    )
    symlink = tmp_path / "recorded-symlink.sqsh"
    symlink.symlink_to(recorded_path)
    digest = validator._sha256(recorded_path)  # noqa: SLF001
    live = validator._validate_container_file(  # noqa: SLF001
        alias,
        expected_size_bytes=alias.stat().st_size,
        expected_sha256=digest,
    )
    recorded = validator.candidate.validate_container_argument(
        str(recorded_path),
        size_bytes=recorded_path.stat().st_size,
        sha256=digest,
    )
    mutations = (
        lambda value: value.update(profile="wrong-profile"),
        lambda value: value.update(size_bytes=value["size_bytes"] + 1),
        lambda value: value.update(size_bytes=True),
        lambda value: value.update(sha256="f" * 64),
        lambda value: value.update(path="relative/container.sqsh"),
        lambda value: value.update(path=123),
        lambda value: value.update(path=str(symlink)),
        lambda value: value.update(hidden=True),
    )
    for mutation in mutations:
        tampered = copy.deepcopy(recorded)
        mutation(tampered)
        with pytest.raises(validator.CandidateArtifactValidationError):
            validator._validate_recorded_container(  # noqa: SLF001
                tampered,
                live,
                field="container",
            )

    copied_live = validator._validate_container_file(  # noqa: SLF001
        independent_copy,
        expected_size_bytes=independent_copy.stat().st_size,
        expected_sha256=digest,
    )
    with pytest.raises(
        validator.CandidateArtifactValidationError,
        match="not the same file",
    ):
        validator._validate_recorded_container(  # noqa: SLF001
            recorded,
            copied_live,
            field="container",
        )


def test_failure_validation_accepts_container_alias_before_later_checks(
    tmp_path,
    monkeypatch,
):
    canonical = _container_file(
        tmp_path / "fs11" / "projects" / "users" / "container.sqsh"
    )
    alias_directory = tmp_path / "fsw" / "users"
    alias_directory.parent.mkdir(parents=True)
    alias_directory.symlink_to(canonical.parent, target_is_directory=True)
    lexical_alias = alias_directory / canonical.name
    assert lexical_alias.resolve() == canonical
    digest = validator._sha256(canonical)  # noqa: SLF001
    lifecycle = {
        "profile": "slurm_single_task_srun_lifecycle_v1",
        "launch_id": "a" * 64,
        "job_id": 123,
        "step_id": 4,
        "nodelist": "l401",
        "procid": 0,
        "localid": 0,
        "ntasks": 1,
    }
    recorded_container = validator.candidate.validate_container_argument(
        str(lexical_alias),
        size_bytes=canonical.stat().st_size,
        sha256=digest,
    )
    args = _failure_validation_args(
        tmp_path,
        lifecycle,
        container_image=lexical_alias,
        context_fields={
            "repository": {
                "path": str(ROOT.resolve()),
                "commit": "b" * 40,
                "clean_tracked": True,
            },
            "container_image": recorded_container,
            "production_eval": None,
        },
    )
    monkeypatch.setattr(
        validator,
        "_repository_identity",
        lambda path, commit: {
            "path": str(path.resolve()),
            "commit": commit,
            "clean_tracked": True,
        },
    )
    monkeypatch.setattr(
        validator.candidate,
        "validate_failure_payload",
        lambda raw, *, variant, require_complete_capture: raw,
    )
    with pytest.raises(
        validator.CandidateArtifactValidationError,
        match="production eval must be an object",
    ):
        validator.validate_failure(args)


@pytest.mark.parametrize(
    ("field", "boolean_alias"),
    (("procid", False), ("localid", False), ("ntasks", True)),
)
def test_independent_failure_validation_rejects_boolean_rank_and_task_aliases(
    tmp_path,
    monkeypatch,
    field,
    boolean_alias,
):
    lifecycle = {
        "profile": "slurm_single_task_srun_lifecycle_v1",
        "launch_id": "a" * 64,
        "job_id": 123,
        "step_id": 4,
        "nodelist": "l401",
        "procid": 0,
        "localid": 0,
        "ntasks": 1,
    }
    assert (
        validator._validate_lifecycle(  # noqa: SLF001
            lifecycle,
            launch_id="a" * 64,
            job_id=123,
            field="test lifecycle",
        )
        == lifecycle
    )
    lifecycle[field] = boolean_alias
    args = _failure_validation_args(tmp_path, lifecycle)
    monkeypatch.setattr(
        validator,
        "_repository_identity",
        lambda path, commit: {
            "path": str(path.resolve()),
            "commit": commit,
            "clean_tracked": True,
        },
    )
    monkeypatch.setattr(
        validator.candidate,
        "validate_failure_payload",
        lambda raw, *, variant, require_complete_capture: raw,
    )
    with pytest.raises(
        validator.CandidateArtifactValidationError,
        match="failed raw lifecycle drift",
    ):
        validator.validate_failure(args)


def _initial_safety():
    env, _ = runtime_tests._runtime_fixture()
    safety = env.unwrapped.action_manager._terms["arm"].safety_report()
    safety["episode_index"] = 0
    safety["joint_velocity_limits_rad_s"] = list(
        validator.safety_validator.EXPECTED_VELOCITY_LIMITS
    )
    safety["gripper_runtime_static"] = gripper_tests._static_contract()
    dynamic = gripper_tests._dynamic_evidence()
    dynamic.update(
        {
            "apply_entry_samples": 0,
            "post_policy_step_samples": 0,
            "max_abs_joint_velocity_rad_s": [0.0] * 6,
            "max_abs_joint_acceleration_rad_s2": [0.0] * 6,
            "max_velocity_diagnostic": None,
            "terminal_state": None,
            "driver_target_slew": gripper_tests._target_slew_dynamic(
                process_calls=0, apply_calls=0
            ),
        }
    )
    safety["gripper_runtime_dynamic"] = dynamic
    return safety


def test_independent_validator_closes_full_raw_and_ready_schemas():
    assert validator.READY_FIELDS == {
        "schema_version",
        "profile",
        "stage",
        "raw_result",
    }
    assert validator.RAW_FIELDS == {
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
    assert validator.ACTION_PLAN["total_action_count"] == 127


def test_validator_rehashes_every_launch_source_and_container():
    source = (SCRIPTS / "validate_eef_pose_canary_controller_candidate.py").read_text()
    for token in (
        '"runner": (',
        '"validator": (',
        '"safety_validator": (',
        '"gate0_helper": (',
        '"fixture": (',
        "_sha256(path) == expected_sha256",
        "set(raw) == RAW_FIELDS",
        "set(ready) == READY_FIELDS",
        "validate_candidate_replay_evidence(",
        "validate_velocity_headroom(",
        '_validate_offline_safety(\n        raw.get("initial_safety")',
        '_validate_offline_safety(\n        raw.get("final_safety")',
        "safety_validator._validate_safety_report(",
        "safety_validator._validate_gripper_static(",
        "_validate_production_eval(",
        "_validate_runtime_protocol(",
        "_validate_runtime_frame(",
    ):
        assert token in source
    assert source.count("container_record = _validate_container_file(") == 2
    assert source.count("_validate_recorded_container(") == 3


def test_failure_verifier_rehashes_context_and_rejects_promotion_artifacts():
    source = (SCRIPTS / "validate_eef_pose_canary_controller_candidate.py").read_text()
    function = source[source.index("def validate_failure(") :]
    for token in (
        "raw_identity = _immutable_file(args.raw_result)",
        "candidate.validate_failure_payload(",
        "require_complete_capture=True",
        'args.raw_result.with_name(args.raw_result.name + ".ready.json")',
        'f"candidate-{args.variant}.srun-status.json"',
        'f"candidate-{args.variant}.attestation.json"',
        '_validate_production_eval(context["production_eval"])',
        "gate0._validate_arm_failure_runtime_evidence(",
        "gate0.validate_gripper_tail(",
        'target_slew.get("apply_calls") == counters["apply_calls"] - 1',
        "expected_target_slew_profile=candidate.CANDIDATE_TARGET_SLEW_PROFILE",
    ):
        assert token in function
    verifier = SCRIPTS / "verify_eef_pose_canary_controller_candidate_failure.py"
    assert verifier.is_file()


def test_offline_safety_consumer_closes_all_nested_schemas():
    safety = _initial_safety()
    kwargs = {
        "field": "initial",
        "apply_calls": 0,
        "expect_closed_target": False,
        "expected_endpoint_change_count": 0,
        "expected_gripper_target_slew_profile": (
            validator.safety_validator.GRIPPER_TARGET_SLEW_PROFILE
        ),
    }
    assert validator._validate_offline_safety(safety, **kwargs) is safety
    mutations = [
        lambda value: value.update(hidden=True),
        lambda value: value["counters"].update(hidden=0),
        lambda value: value["maxima"].update(hidden=[0.0] * 7),
        lambda value: value["gripper_runtime_dynamic"].update(hidden=0),
        lambda value: value["gripper_runtime_dynamic"]["driver_target_slew"].update(
            hidden=0
        ),
    ]
    for mutation in mutations:
        tampered = copy.deepcopy(safety)
        mutation(tampered)
        with pytest.raises(validator.CandidateArtifactValidationError):
            validator._validate_offline_safety(tampered, **kwargs)


def _open_final_safety():
    safety = _initial_safety()
    apply_calls = 8
    safety["counters"]["apply_calls"] = apply_calls
    safety["counters"]["environment_substeps"] = apply_calls
    zero_gripper = [0.0] * 6
    gripper_diagnostic = {
        "sample_phase": "post_policy_step",
        "sample_index": 8,
        "joint_position_rad": zero_gripper,
        "joint_velocity_rad_s": zero_gripper,
        "joint_acceleration_rad_s2": zero_gripper,
        "joint_position_target_rad": zero_gripper,
        "joint_velocity_target_rad_s": zero_gripper,
    }
    safety["gripper_runtime_dynamic"].update(
        {
            "apply_entry_samples": apply_calls,
            "post_policy_step_samples": 1,
            "max_abs_joint_velocity_rad_s": zero_gripper,
            "max_abs_joint_acceleration_rad_s2": zero_gripper,
            "max_velocity_diagnostic": gripper_diagnostic,
            "terminal_state": {
                name: copy.deepcopy(value)
                for name, value in gripper_diagnostic.items()
                if name != "sample_phase"
            },
        }
    )
    target_slew = safety["gripper_runtime_dynamic"]["driver_target_slew"]
    target_slew.update(
        {
            "process_action_calls": 1,
            "apply_calls": apply_calls,
            "initialization_count": 1,
            "endpoint_change_count": 0,
            "repeated_endpoint_process_count": 0,
            "slew_limited_apply_count": 0,
            "endpoint_reached_apply_count": apply_calls,
            "live_limit_validation_count": apply_calls,
            "max_abs_target_step_rad": 0.0,
            "max_abs_endpoint_error_before_step_rad": 0.0,
            "max_abs_endpoint_error_after_step_rad": 0.0,
            "initial_anchor_rad": 0.0,
            "last_requested_endpoint_rad": 0.0,
            "last_applied_target_rad": 0.0,
        }
    )
    joint_pos = [
        (lower + upper) / 2.0 for lower, upper in safety["target_joint_pos_limits_rad"]
    ]

    def vector(values):
        return {
            "values": list(values),
            "finite_mask": [True] * 7,
            "finite_count": 7,
        }

    safety["max_raw_delta_diagnostic"] = {
        "kind": "max_raw_delta",
        "episode_index": 0,
        "policy_step": 0,
        "physics_substep": 0,
        "joint_pos_rad": vector(joint_pos),
        "raw_delta_joint_pos_rad": vector([0.0] * 7),
        "raw_joint_pos_target_rad": vector(joint_pos),
        "safe_joint_pos_target_rad": vector(joint_pos),
        "pose_error_norm": 0.0,
        "jacobian_finite": True,
        "jacobian_max_abs": 0.0,
        "eef_quaternion_norm": None,
    }
    return safety


def test_offline_safety_matches_production_promotion_invariants():
    safety = _open_final_safety()
    kwargs = {
        "field": "final",
        "apply_calls": 8,
        "expect_closed_target": False,
        "expected_endpoint_change_count": 0,
        "expected_gripper_target_slew_profile": (
            validator.safety_validator.GRIPPER_TARGET_SLEW_PROFILE
        ),
    }
    assert validator._validate_offline_safety(safety, **kwargs) is safety
    mutations = [
        lambda value: value["maxima"]["applied_delta_joint_pos_rad"].__setitem__(
            slice(None), [99.0] * 7
        ),
        lambda value: value["maxima"]["abs_joint_vel_rad_s"].__setitem__(
            slice(None), [99.0] * 7
        ),
        lambda value: value["maxima"][
            "current_physx_hard_limit_violation_rad"
        ].__setitem__(slice(None), [99.0] * 7),
        lambda value: value["maxima"]["minimum_outer_joint_clearance_rad"].__setitem__(
            0, -1e-3
        ),
        lambda value: value["max_raw_delta_diagnostic"].__setitem__(
            "episode_index", "not-an-int"
        ),
        lambda value: value["max_raw_delta_diagnostic"].__setitem__(
            "jacobian_finite", "not-a-bool"
        ),
        lambda value: value["max_raw_delta_diagnostic"].__setitem__(
            "joint_pos_rad", "not-a-vector"
        ),
    ]
    for mutation in mutations:
        tampered = copy.deepcopy(safety)
        mutation(tampered)
        with pytest.raises(validator.CandidateArtifactValidationError):
            validator._validate_offline_safety(tampered, **kwargs)


def test_runtime_protocol_and_frame_schemas_are_closed():
    protocol = {
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
    frame = {
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
        "position_error_m": 0.0,
        "rotation_error_rad": 0.0,
    }
    assert validator._validate_runtime_protocol(protocol) is protocol
    assert validator._validate_runtime_frame(frame) is frame
    for value, function in (
        (protocol, validator._validate_runtime_protocol),
        (frame, validator._validate_runtime_frame),
    ):
        tampered = copy.deepcopy(value)
        tampered["hidden"] = True
        with pytest.raises(validator.CandidateArtifactValidationError):
            function(tampered)


def test_recursive_production_and_asset_comparisons_are_type_strict():
    production = validator.gate0.validate_production_reset_source()
    assert validator._validate_production_eval(production) is production
    tampered = copy.deepcopy(production)
    tampered["initial_condition_index"] = False
    with pytest.raises(validator.CandidateArtifactValidationError):
        validator._validate_production_eval(tampered)
    with pytest.raises(validator.CandidateArtifactValidationError):
        validator._compare_with_samefile_paths(
            {"nested": {"count": False}},
            {"nested": {"count": 0}},
            field="asset",
        )


def test_wrapper_hashes_image_before_srun_and_uses_isolated_cache():
    source = (SCRIPTS / "run_eef_pose_canary_controller_candidate_srun.sh").read_text()
    image_hash = "sha256sum \"${CANDIDATE_CONTAINER_IMAGE}\" | awk '{print $1}'"
    assert image_hash in source
    assert source.index(image_hash) < source.index("srun \\")
    assert '[[ -f "${CANDIDATE_CONTAINER_IMAGE}" && ! -L' in source
    assert (
        'cache_suffix="${CANDIDATE_VARIANT}/job_${SLURM_JOB_ID}/'
        'launch_${CANDIDATE_LAUNCH_ID}"'
    ) in source
    assert (
        'host_cache_namespace="${CANDIDATE_HOST_CACHE_ROOT%/}/${cache_suffix}"'
        in source
    )
    assert (
        'container_cache_namespace="${CANDIDATE_CACHE_ROOT%/}/${cache_suffix}"'
        in source
    )
    assert 'XDG_CACHE_HOME="${container_cache_namespace}"' in source
    assert 'HF_HOME="${container_cache_namespace}/huggingface"' in source
    assert 'HOME="${container_cache_namespace}/home"' in source
    assert "${CANDIDATE_HOST_CACHE_ROOT}:${CANDIDATE_CACHE_ROOT}" in source
    assert "trap cleanup_cache EXIT" in source
    assert "validate_eef_pose_canary_controller_candidate.py" in source
    assert '--expected-validator-sha256 "${CANDIDATE_VALIDATOR_SHA256}"' in source
    assert (
        '"${CANDIDATE_FAILURE_VERIFIER_SHA256:?required failure-verifier digest}"'
        in source
    )
    verifier_hash = (
        'sha256sum "${CANDIDATE_POLARIS_REPO}/scripts/'
        'verify_eef_pose_canary_controller_candidate_failure.py"'
    )
    assert verifier_hash in source
    handler_start = source.index("handle_failed_srun() {")
    handler_end = source.index("\n}\n\nstarted_at_ns=", handler_start) + 2
    failure_handler = source[handler_start:handler_end]
    verifier_call = failure_handler.index(
        "verify_eef_pose_canary_controller_candidate_failure.py"
    )
    failure_branch = source.index('if [[ "${srun_rc}" -ne 0 ]]')
    status_call = source.index(
        "write_eef_pose_canary_controller_candidate_srun_status.py",
        failure_branch,
    )
    assert verifier_call >= 0
    assert failure_branch < status_call
    assert (
        '--expected-failure-verifier-sha256 "${CANDIDATE_FAILURE_VERIFIER_SHA256}"'
        in source
    )
    srun_rc_capture = source.index("srun_rc=$?")
    success_set_e = source.index("set -e", failure_branch)
    success_timestamp = source.index('returned_at_ns="$(date +%s%N)"', success_set_e)
    assert srun_rc_capture < failure_branch < success_set_e < success_timestamp
    assert source[srun_rc_capture:failure_branch].strip() == "srun_rc=$?"
    assert 'handle_failed_srun "${srun_rc}"' in source[failure_branch:success_set_e]
    assert 'exit "${original_srun_rc}"' in failure_handler
    assert "failure_verify_rc=$?" in failure_handler


def test_wrapper_failure_handler_attempts_all_diagnostics_and_preserves_srun_rc():
    source = (SCRIPTS / "run_eef_pose_canary_controller_candidate_srun.sh").read_text()
    handler_start = source.index("handle_failed_srun() {")
    handler_end = source.index("\n}\n\nstarted_at_ns=", handler_start) + 2
    handler = source[handler_start:handler_end]
    digest = "a" * 64
    shell = f"""
set -uo pipefail
set +e
date() {{ return 31; }}
chmod() {{ return 41; }}
python3() {{ return 51; }}
CANDIDATE_POLARIS_REPO=/repo
CANDIDATE_VARIANT=official_lap3b
CANDIDATE_LAUNCH_ID={digest}
SLURM_JOB_ID=123
raw_result=/output/raw.json
CANDIDATE_POLARIS_COMMIT={"b" * 40}
CANDIDATE_RUNNER_SHA256={digest}
CANDIDATE_VALIDATOR_SHA256={digest}
CANDIDATE_FAILURE_VERIFIER_SHA256={digest}
CANDIDATE_SAFETY_VALIDATOR_SHA256={digest}
CANDIDATE_GATE0_HELPER_SHA256={digest}
CANDIDATE_FIXTURE_SHA256={digest}
CANDIDATE_CONTAINER_IMAGE=/container.sqsh
container_size_bytes=1234
CANDIDATE_CONTAINER_SHA256={digest}
stdout_log=/output/stdout.log
stderr_log=/output/stderr.log
{handler}
handle_failed_srun 7
"""
    result = subprocess.run(
        ["bash"],
        input=shell,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 7
    assert "failure timestamp failed rc=31" in result.stderr
    assert "failure log chmod failed rc=41" in result.stderr
    assert "failure verification failed rc=51" in result.stderr
    assert "srun failed rc=7; returning original rc" in result.stderr
