from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
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
    assert validator.ACTION_PLAN["total_action_count"] == 122


def test_validator_rehashes_every_launch_source_and_container():
    source = (SCRIPTS / "validate_eef_pose_canary_controller_candidate.py").read_text()
    for token in (
        '"runner": (',
        '"validator": (',
        '"safety_validator": (',
        '"gate0_helper": (',
        '"fixture": (',
        "_sha256(container) == args.expected_container_sha256",
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


def test_offline_safety_consumer_closes_all_nested_schemas():
    safety = _initial_safety()
    kwargs = {
        "field": "initial",
        "apply_calls": 0,
        "expect_closed_target": False,
        "expected_endpoint_change_count": 0,
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
