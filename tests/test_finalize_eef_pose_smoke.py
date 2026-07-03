from argparse import Namespace
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess

import pytest

from scripts import finalize_eef_pose_smoke as finalizer


def _vector_evidence(values):
    return {
        "values": values,
        "finite_mask": [True] * 7,
        "finite_count": 7,
        "max_abs": max(map(abs, values)),
    }


def _diagnostic_vector(values):
    return {
        "values": list(values),
        "finite_mask": [True] * 7,
        "finite_count": 7,
    }


def _gripper_tensor(values, *, shape, device):
    return {
        "dtype": "torch.float32",
        "device": device,
        "shape": list(shape),
        "values": list(values),
        "finite_mask": [True] * len(values),
        "finite_count": len(values),
    }


def _gripper_static():
    before_values = [
        *finalizer.EXPECTED_VELOCITY_LIMITS,
        5.0,
        *([174.53292846679688] * 5),
    ]
    after_values = [*finalizer.EXPECTED_VELOCITY_LIMITS, *([5.0] * 6)]
    before = _gripper_tensor(before_values, shape=(1, 13), device="cpu")
    after = _gripper_tensor(after_values, shape=(1, 13), device="cpu")
    root = "/panda/Gripper/Robotiq_2F_85/Joints"
    driver_path = f"{root}/finger_joint"
    follower_specs = [
        ("right_outer_knuckle_joint", 8, "rotZ", -1.0, 1_000_000.0, 0.0),
        ("left_inner_finger_joint", 9, "rotX", 1.0, 1_000.0, 0.05),
        ("right_inner_finger_joint", 10, "rotX", -1.0, 1_000.0, 0.05),
        ("left_inner_finger_knuckle_joint", 11, "rotX", 1.0, 1_000.0, 0.05),
        ("right_inner_finger_knuckle_joint", 12, "rotX", 1.0, 1_000.0, 0.05),
    ]
    actuator = {
        "cfg_velocity_limit": 5.0,
        "cfg_velocity_limit_sim": 5.0,
        "cfg_effort_limit": 200.0,
        "cfg_effort_limit_sim": 200.0,
        "resolved_velocity_limit": _gripper_tensor(
            [5.0], shape=(1, 1), device="cuda:0"
        ),
        "resolved_velocity_limit_sim": _gripper_tensor(
            [5.0], shape=(1, 1), device="cuda:0"
        ),
        "resolved_effort_limit": _gripper_tensor(
            [200.0], shape=(1, 1), device="cuda:0"
        ),
        "resolved_effort_limit_sim": _gripper_tensor(
            [200.0], shape=(1, 1), device="cuda:0"
        ),
    }
    return {
        "profile": finalizer.GRIPPER_RUNTIME_PROFILE,
        "joint_names": finalizer.EXPECTED_DROID_JOINT_NAMES,
        "gripper_joint_names": finalizer.GRIPPER_JOINT_NAMES,
        "gripper_joint_indices": list(range(7, 13)),
        "driver_joint_name": "finger_joint",
        "driver_joint_index": 7,
        "follower_joint_names": finalizer.GRIPPER_JOINT_NAMES[1:],
        "follower_joint_indices": list(range(8, 13)),
        "actuator_joint_ownership": {
            "panda_shoulder": {
                "joint_names": finalizer.EXPECTED_JOINT_NAMES[:4],
                "joint_indices": list(range(4)),
            },
            "panda_forearm": {
                "joint_names": finalizer.EXPECTED_JOINT_NAMES[4:],
                "joint_indices": list(range(4, 7)),
            },
            "gripper": {"joint_names": ["finger_joint"], "joint_indices": [7]},
        },
        "device_partition": {
            "profile": "nvidia_droid_cuda_dynamic_actuator_cpu_static_physx_v1",
            "dynamic_articulation": "cuda:0",
            "implicit_actuator": "cuda:0",
            "static_physx": "cpu",
            "dtype": "torch.float32",
        },
        "driver_actuator": actuator,
        "mimic_joint_contract": {
            "profile": "robotiq_2f85_source_usd_physx_mimic_joint_v1",
            "robot_usd_sha256": (
                "d8379925b103963dbf3e7c85bcc4ae101b81b7c1d7dabe7d2e964f41d069ec44"
            ),
            "driver_joint_name": "finger_joint",
            "driver_joint_index": 7,
            "driver_joint_prim_path": driver_path,
            "driver_physics_joint_type": "PhysicsRevoluteJoint",
            "driver_exclude_from_articulation": False,
            "followers": [
                {
                    "joint_name": name,
                    "joint_index": index,
                    "prim_path": f"{root}/{name}",
                    "physics_joint_type": "PhysicsRevoluteJoint",
                    "exclude_from_articulation": False,
                    "mimic_axis": axis,
                    "reference_joint_path": driver_path,
                    "gearing": gearing,
                    "natural_frequency_hz": frequency,
                    "damping_ratio": damping,
                }
                for name, index, axis, gearing, frequency, damping in follower_specs
            ],
        },
        "velocity_limits_before_write": before,
        "velocity_limits_after_write": after,
        "velocity_limit_write_contract": {
            "profile": (
                "live_root_physx_view_full_tensor_five_mimic_dofs_"
                "velocity_limit5_eef_production_v1"
            ),
            "setter": "root_physx_view.set_dof_max_velocities",
            "timing": "after_first_explicit_reset_before_first_apply_v1",
            "call_count": 1,
            "articulation_indices": [0],
            "full_input": copy.deepcopy(after),
        },
        "driver_target_slew": {
            "profile": finalizer.GRIPPER_TARGET_SLEW_PROFILE,
            "scope": "eef_pose_only_native_joint_position_unchanged_v1",
            "action_class": finalizer.GRIPPER_TARGET_SLEW_ACTION_CLASS,
            "driver_joint_name": "finger_joint",
            "driver_joint_index": 7,
            "endpoint_semantics_profile": (
                "closed_positive_ge_0p5_inverse_open_gt_0p5_v1"
            ),
            "open_target_rad": 0.0,
            "closed_target_rad": finalizer.GRIPPER_CLOSED_TARGET,
            "physical_velocity_limit_source": (
                "live_implicit_actuator_velocity_limit_sim_float32_v1"
            ),
            "physical_velocity_limit_rad_s": 5.0,
            "target_slew_rate_source": (
                "eef_profile_fraction_of_live_physical_velocity_limit_float32_v1"
            ),
            "target_slew_rate_factor": 0.5,
            "target_slew_rate_rad_s": 2.5,
            "physics_hz": 120.0,
            "physics_dt": 1.0 / 120.0,
            "max_target_step_rad": finalizer.GRIPPER_MAX_TARGET_STEP,
            "float32_tolerance_rad": 1e-6,
            "reset_profile": finalizer.GRIPPER_TARGET_SLEW_RESET_PROFILE,
            "tensor_dtype": "torch.float32",
            "tensor_device": "cuda:0",
        },
        "measured_velocity_is_hard_bounded_by_limit": False,
    }


def _candidate_gripper_static():
    contract = _gripper_static()
    profile = finalizer.GRIPPER_TARGET_SLEW_PROFILES[
        finalizer.GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
    ]
    contract["driver_target_slew"].update(
        {
            "profile": profile["profile"],
            "target_slew_rate_factor": profile["rate_factor"],
            "target_slew_rate_rad_s": profile["rate_rad_s"],
            "max_target_step_rad": profile["max_target_step_rad"],
        }
    )
    followers = []
    for source in contract["mimic_joint_contract"]["followers"]:
        source_snapshot = {
            "natural_frequency_rad_s": source["natural_frequency_hz"],
            "damping_ratio": source["damping_ratio"],
        }
        target_snapshot = {
            "natural_frequency_rad_s": 100.0,
            "damping_ratio": finalizer._float32(1.2),  # noqa: SLF001
        }
        axis = source["mimic_axis"]
        structure = {
            "applied_mimic_api": f"PhysxMimicJointAPI:{axis}",
            "reference_joint_path": (
                "/World/envs/env_0/robot/Gripper/Robotiq_2F_85/Joints/finger_joint"
            ),
            "gearing": source["gearing"],
            "offset": 0.0,
            "exclude_from_articulation": source["exclude_from_articulation"],
        }
        followers.append(
            {
                "joint_name": source["joint_name"],
                "joint_index": source["joint_index"],
                "live_prim_path": (
                    "/World/envs/env_0/robot"
                    + source["prim_path"].removeprefix("/panda")
                ),
                "mimic_axis": axis,
                "natural_frequency_attribute": (
                    f"physxMimicJoint:{axis}:naturalFrequency"
                ),
                "damping_ratio_attribute": (f"physxMimicJoint:{axis}:dampingRatio"),
                "source": copy.deepcopy(source_snapshot),
                "before_spawn_write": copy.deepcopy(source_snapshot),
                "before_spawn_structure": copy.deepcopy(structure),
                "natural_frequency_write_count": 1,
                "damping_ratio_write_count": 1,
                "after_spawn_write": copy.deepcopy(target_snapshot),
                "after_spawn_structure": copy.deepcopy(structure),
                "post_reset_composed_usd_readback": copy.deepcopy(target_snapshot),
                "post_reset_composed_usd_structure": copy.deepcopy(structure),
            }
        )
    contract["mimic_compliance"] = {
        "profile": finalizer.GRIPPER_MIMIC_COMPLIANCE_PROFILE,
        "enabled": True,
        "scope": "eef_rate0p25_candidate_only_source_usd_immutable_v1",
        "timing": "after_original_usd_spawn_before_articulation_initialization_v1",
        "setter": "UsdAttribute.Set_default_float_v1",
        "live_root_profile": "single_composed_world_env0_robot_root_v1",
        "live_root_path": "/World/envs/env_0/robot",
        "original_spawn_func": {
            "module": "isaaclab.sim.spawners.from_files.from_files",
            "qualname": "spawn_from_usd",
            "name": "spawn_from_usd",
        },
        "overlay_func": {
            "module": "polaris.eef_gripper_runtime",
            "qualname": "eef_mimic_compliance_spawn_overlay",
            "name": "eef_mimic_compliance_spawn_overlay",
        },
        "original_spawn_call_count": 1,
        "overlay_call_count": 1,
        "physics_hz": 120.0,
        "physics_dt": 1.0 / 120.0,
        "target_natural_frequency_rad_s": 100.0,
        "target_damping_ratio": finalizer._float32(1.2),  # noqa: SLF001
        "frequency_timestep_product": 5.0 / 6.0,
        "follower_count": 5,
        "natural_frequency_write_count": 5,
        "damping_ratio_write_count": 5,
        "total_write_count": 10,
        "source_usd_sha256": contract["mimic_joint_contract"]["robot_usd_sha256"],
        "source_usd_unchanged_after_spawn_overlay": True,
        "followers": followers,
    }
    return contract


def _gripper_dynamic(apply_calls, *, closed, endpoint_changes=0):
    post_samples = apply_calls // 8
    vector = [0.0] * 6
    diagnostic = None
    terminal = None
    if apply_calls:
        diagnostic = {
            "sample_phase": "apply_entry",
            "sample_index": 0,
            "joint_position_rad": vector,
            "joint_velocity_rad_s": vector,
            "joint_acceleration_rad_s2": vector,
            "joint_position_target_rad": vector,
            "joint_velocity_target_rad_s": vector,
        }
        terminal = {
            "sample_index": post_samples * 9 - 1,
            "joint_position_rad": vector,
            "joint_velocity_rad_s": vector,
            "joint_acceleration_rad_s2": vector,
            "joint_position_target_rad": vector,
            "joint_velocity_target_rad_s": vector,
        }
    process_calls = post_samples
    limited = min(finalizer.GRIPPER_CLOSE_LIMITED_APPLIES, apply_calls) if closed else 0
    return {
        "profile": finalizer.GRIPPER_RUNTIME_PROFILE,
        "joint_names": finalizer.GRIPPER_JOINT_NAMES,
        "joint_indices": list(range(7, 13)),
        "apply_entry_samples": apply_calls,
        "post_policy_step_samples": post_samples,
        "max_abs_joint_velocity_rad_s": vector,
        "max_abs_joint_acceleration_rad_s2": vector,
        "max_velocity_diagnostic": diagnostic,
        "terminal_state": terminal,
        "driver_target_slew": {
            "profile": finalizer.GRIPPER_TARGET_SLEW_PROFILE,
            "process_action_calls": process_calls,
            "apply_calls": apply_calls,
            "initialization_count": int(apply_calls > 0),
            "endpoint_change_count": endpoint_changes,
            "repeated_endpoint_process_count": max(
                process_calls - 1 - endpoint_changes, 0
            ),
            "slew_limited_apply_count": limited,
            "endpoint_reached_apply_count": apply_calls - limited,
            "live_limit_validation_count": apply_calls,
            "max_abs_target_step_rad": (
                finalizer.GRIPPER_MAX_TARGET_STEP if apply_calls and closed else 0.0
            ),
            "max_abs_endpoint_error_before_step_rad": (
                finalizer.GRIPPER_CLOSED_TARGET if apply_calls and closed else 0.0
            ),
            "max_abs_endpoint_error_after_step_rad": (
                finalizer.GRIPPER_CLOSED_TARGET - finalizer.GRIPPER_MAX_TARGET_STEP
                if apply_calls and closed
                else 0.0
            ),
            "initial_anchor_rad": 0.0 if apply_calls else None,
            "last_requested_endpoint_rad": (
                finalizer.GRIPPER_CLOSED_TARGET
                if apply_calls and closed
                else 0.0
                if apply_calls
                else None
            ),
            "last_applied_target_rad": (
                finalizer.GRIPPER_CLOSED_TARGET
                if apply_calls >= finalizer.GRIPPER_CLOSE_TRANSITION_APPLIES and closed
                else finalizer.GRIPPER_MAX_TARGET_STEP * apply_calls
                if apply_calls and closed
                else 0.0
                if apply_calls
                else None
            ),
        },
        "nonfinite_samples": 0,
        "dropped_diagnostics": 0,
    }


def _safety_report(
    episode_index,
    apply_calls,
    *,
    adversarial=False,
    closed=None,
    endpoint_changes=0,
):
    if closed is None:
        closed = episode_index == 0
    counters = {name: 0 for name in finalizer.COUNTER_FIELDS}
    counters["apply_calls"] = apply_calls
    counters["environment_substeps"] = apply_calls
    if adversarial:
        counters["slew_limit_events"] = 1
        counters["slew_limited_joints"] = 1
    maxima = {name: [0.0] * 7 for name in finalizer.MAXIMA_FIELDS}
    q = [(lower + upper) / 2 for lower, upper in finalizer.EXPECTED_LIMITS]
    max_raw = None
    if apply_calls:
        max_raw = {
            "kind": "max_raw_delta",
            "episode_index": episode_index,
            "policy_step": 0,
            "physics_substep": 0,
            "joint_pos_rad": _diagnostic_vector(q),
            "raw_delta_joint_pos_rad": _diagnostic_vector([0.0] * 7),
            "raw_joint_pos_target_rad": _diagnostic_vector(q),
            "safe_joint_pos_target_rad": _diagnostic_vector(q),
            "pose_error_norm": 0.0,
            "jacobian_finite": True,
            "jacobian_max_abs": 1.0,
            "eef_quaternion_norm": None,
        }
    if adversarial:
        raw_delta = finalizer.EXPECTED_MAX_DELTA[0] + 0.01
        maxima["raw_delta_joint_pos_rad"][0] = raw_delta
        maxima["applied_delta_joint_pos_rad"][0] = finalizer.EXPECTED_MAX_DELTA[0]
        raw_delta_vector = [raw_delta] + [0.0] * 6
        raw_target = [q[0] + raw_delta] + q[1:]
        safe_target = [q[0] + finalizer.EXPECTED_MAX_DELTA[0]] + q[1:]
        max_raw["raw_delta_joint_pos_rad"] = _diagnostic_vector(raw_delta_vector)
        max_raw["raw_joint_pos_target_rad"] = _diagnostic_vector(raw_target)
        max_raw["safe_joint_pos_target_rad"] = _diagnostic_vector(safe_target)
    return {
        "episode_index": episode_index,
        "profile": "panda_velocity_physxlimit_solveriter1_v4",
        "apply_actions_cadence": "physics_substep",
        "physics_dt": 1.0 / 120.0,
        "control_dt": 1.0 / 15.0,
        "decimation": 8,
        "current_joint_soft_limit_tolerance_rad": 1e-5,
        "target_soft_limit_guard_band_profile": (
            "eef_physx_inner_hardlimit_one_substep_v2"
        ),
        "physx_hard_limit_profile": "outer_minus_one_velocity_substep_v1",
        "physx_derived_soft_limit_profile": (
            "isaaclab_midpoint_range_factor1_float32_v1"
        ),
        "physx_hard_limit_write_count": 1,
        "arm_velocity_target_profile": "zero_per_physics_substep_v1",
        "articulation_solver_profile": "tgs_position64_velocity1_eef_only_v1",
        "articulation_solver_readback": (
            "composed_usd_physx_articulation_api_all_env_roots_v1"
        ),
        "physx_solver_type": 1,
        "solver_position_iteration_count": 64,
        "solver_velocity_iteration_count": 1,
        "joint_velocity_limit_tolerance_rad_s": 1e-5,
        "eef_quaternion_unit_norm_tolerance": 1e-3,
        "joint_slew_float32_tolerance_rad": 1e-6,
        "soft_joint_pos_limit_factor": 1.0,
        "joint_names": finalizer.EXPECTED_JOINT_NAMES,
        "joint_velocity_limits_rad_s": finalizer.EXPECTED_VELOCITY_LIMITS,
        "joint_effort_limits": finalizer.EXPECTED_EFFORT_LIMITS,
        "max_delta_joint_pos_rad": finalizer.EXPECTED_MAX_DELTA,
        "target_soft_limit_margin_rad": finalizer.EXPECTED_MAX_DELTA,
        "target_joint_pos_limits_rad": finalizer.EXPECTED_TARGET_LIMITS,
        "target_joint_pos_limits_float32_sha256": (finalizer.EXPECTED_TARGET_DIGEST),
        "physx_hard_joint_pos_limits_rad": finalizer.EXPECTED_TARGET_LIMITS,
        "physx_hard_joint_pos_limits_float32_sha256": (
            finalizer.EXPECTED_TARGET_DIGEST
        ),
        "physx_derived_soft_joint_pos_limits_rad": (
            [list(pair) for pair in finalizer.EXPECTED_PHYSX_DERIVED_SOFT_LIMITS]
        ),
        "physx_derived_soft_joint_pos_limits_float32_sha256": (
            finalizer.EXPECTED_PHYSX_DERIVED_SOFT_DIGEST
        ),
        "arm_velocity_target_rad_s": [0.0] * 7,
        "soft_joint_pos_limits_rad": finalizer.EXPECTED_LIMITS,
        "soft_joint_pos_limits_float32_sha256": finalizer.EXPECTED_DIGEST,
        "counters": counters,
        "maxima": maxima,
        "guard_diagnostics": [],
        "max_raw_delta_diagnostic": max_raw,
        "current_joint_velocity_abort": None,
        "gripper_runtime_static": _gripper_static(),
        "gripper_runtime_dynamic": _gripper_dynamic(
            apply_calls,
            closed=closed,
            endpoint_changes=endpoint_changes,
        ),
    }


def _case_results():
    hold_position = [0.3, 0.0, 0.5]
    hold_quaternion = [1.0, 0.0, 0.0, 0.0]
    targets = [(hold_position.copy(), hold_quaternion.copy())]
    for axis, sign in ((0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0), (2, 1.0), (2, -1.0)):
        position = hold_position.copy()
        position[axis] += sign * 0.04
        targets.append((position, hold_quaternion.copy()))
    half_angle = math.radians(15.0) / 2.0
    for axis, sign in ((0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0), (2, 1.0), (2, -1.0)):
        delta = [math.cos(half_angle), 0.0, 0.0, 0.0]
        delta[axis + 1] = sign * math.sin(half_angle)
        targets.append(
            (
                hold_position.copy(),
                finalizer._quaternion_multiply_wxyz(hold_quaternion, delta),
            )
        )
    return [
        {
            "case": case,
            "passed": True,
            "position_error_m": 0.0,
            "rotation_error_rad": 0.0,
            "target_position": position,
            "actual_position": position.copy(),
            "target_quaternion_wxyz": quaternion,
            "actual_quaternion_wxyz": quaternion.copy(),
            "reset_frame_position_error_m": 0.0,
            "reset_frame_rotation_error_rad": 0.0,
            "final_frame_position_error_m": 0.0,
            "final_frame_rotation_error_rad": 0.0,
        }
        for case, (position, quaternion) in zip(
            finalizer.EXPECTED_CASES, targets, strict=True
        )
    ]


def _valid_raw_result():
    q = [(lower + upper) / 2 for lower, upper in finalizer.EXPECTED_LIMITS]
    reports = [_safety_report(index, 360) for index in range(13)]
    delayed_close_safety = _safety_report(
        13,
        finalizer.DELAYED_CLOSE_APPLY_CALLS,
        closed=True,
        endpoint_changes=1,
    )
    adversarial_safety = _safety_report(14, 8, adversarial=True)

    def headroom_entry(safety):
        maxima = safety["maxima"]["abs_joint_vel_rad_s"]
        limits = safety["joint_velocity_limits_rad_s"]
        ratios = [
            maximum / limit for maximum, limit in zip(maxima, limits, strict=True)
        ]
        return {
            "episode_index": safety["episode_index"],
            "joint_names": safety["joint_names"],
            "max_abs_joint_velocity_rad_s": maxima,
            "joint_velocity_limit_rad_s": limits,
            "velocity_to_limit_ratio": ratios,
            "maximum_ratio": max(ratios),
            "threshold_ratio": finalizer.CLOSE_ARM_VELOCITY_HEADROOM_MAX_RATIO,
            "passed": True,
        }

    payload = {
        "schema_version": 2,
        "finalized": False,
        "passed": False,
        "stage": "simulation_app_close_pending",
        "case": None,
        "exit_code": 0,
        "failure": None,
        "terminal_failure_evidence": None,
        "close_failures": [],
        "persistence_failures": [],
        "environment": "DROID-FoodBussing",
        "eef_frame": "panda_link8",
        "hold_steps": 45,
        "position_delta_m": 0.04,
        "rotation_delta_deg": 15.0,
        "position_tolerance_m": 0.01,
        "rotation_tolerance_deg": 5.0,
        "frame_position_tolerance_m": 1e-5,
        "frame_rotation_tolerance_deg": 0.01,
        "raw_ik_safety_capture": _safety_report(None, 0),
        "results": _case_results(),
        "ik_safety_episodes": reports,
        "gripper_delayed_close_replay": {
            "profile": finalizer.DELAYED_CLOSE_REPLAY_PROFILE,
            "case": "open 115 policy steps then close at the same arm pose",
            "passed": True,
            "episode_index": 13,
            "open_policy_steps": finalizer.DELAYED_CLOSE_OPEN_POLICY_STEPS,
            "close_policy_steps": finalizer.DELAYED_CLOSE_CLOSE_POLICY_STEPS,
            "close_transition_substeps": (finalizer.GRIPPER_CLOSE_TRANSITION_APPLIES),
            "terminated": False,
            "truncated": False,
            "arm_abort_count": 0,
            "ik_safety": delayed_close_safety,
        },
        "gripper_close_velocity_headroom": {
            "profile": finalizer.CLOSE_ARM_VELOCITY_HEADROOM_PROFILE,
            "threshold_ratio": finalizer.CLOSE_ARM_VELOCITY_HEADROOM_MAX_RATIO,
            "passed": True,
            "immediate_close_hold": headroom_entry(reports[0]),
            "delayed_close_replay": headroom_entry(delayed_close_safety),
        },
        "ik_safety_adversarial": {
            "case": "oversized absolute +x target for one policy step",
            "passed": True,
            "state_is_finite": True,
            "eef_state_is_finite": True,
            "joint_state_is_finite": True,
            "joint_pos_within_captured_soft_limits": True,
            "terminated": False,
            "truncated": False,
            "guard_error": "",
            "joint_state": {
                "joint_names": finalizer.EXPECTED_JOINT_NAMES,
                "position_within_captured_soft_limits": True,
                "soft_limit_tolerance_rad": 1e-5,
                "joint_pos_rad": _vector_evidence(q),
                "joint_vel_rad_s": _vector_evidence([0.0] * 7),
                "soft_limit_violation_rad": _vector_evidence([0.0] * 7),
            },
            "ik_safety": adversarial_safety,
            "guard_evidence": {
                "apply_calls": 8,
                "slew_limit_events": 1,
                "abort_count": 0,
                "post_clamp_target_violations": 0,
                "applied_within_bounds": True,
            },
        },
    }
    return copy.deepcopy(payload)


def _write_immutable_json(path: Path, payload) -> bytes:
    data = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
    path.write_bytes(data)
    path.chmod(0o444)
    return data


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments], text=True
    ).strip()


def _attestation_args(tmp_path: Path, monkeypatch) -> Namespace:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    smoke_source = repo / "scripts" / "smoke_eef_pose_controller.py"
    smoke_source.write_text("# reviewed synthetic smoke\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "synthetic"], check=True)

    image = tmp_path / "image.sqsh"
    image.write_bytes(b"synthetic pinned image")
    saved_job = tmp_path / "saved.sbatch"
    runtime_job = tmp_path / "runtime.sbatch"
    saved_job.write_text("#!/bin/bash\n# immutable\n")
    runtime_job.write_bytes(saved_job.read_bytes())

    job_id = 12345
    monkeypatch.setenv("SLURM_JOB_ID", str(job_id))
    raw_path = tmp_path / f"smoke-{job_id}.json"
    raw_bytes = _write_immutable_json(raw_path, _valid_raw_result())
    marker_path = raw_path.with_name(raw_path.name + ".ready.json")
    _write_immutable_json(
        marker_path,
        {
            "schema_version": 1,
            "stage": "simulation_app_close_pending",
            "raw_result": {
                "path": str(raw_path),
                "size_bytes": len(raw_bytes),
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "mode": "0444",
            },
        },
    )
    return Namespace(
        raw_result=raw_path,
        attestation=tmp_path / f"smoke-{job_id}.attestation.json",
        srun_rc=0,
        job_id=job_id,
        runtime_job_script=runtime_job,
        saved_job_script=saved_job,
        polaris_repo=repo,
        expected_polaris_commit=_git(repo, "rev-parse", "HEAD"),
        expected_smoke_sha256=hashlib.sha256(smoke_source.read_bytes()).hexdigest(),
        container_image=image,
        expected_image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        expected_finalizer_sha256=hashlib.sha256(
            Path(finalizer.__file__).resolve().read_bytes()
        ).hexdigest(),
        expected_saved_job_script_sha256=hashlib.sha256(
            saved_job.read_bytes()
        ).hexdigest(),
    )


def test_host_gripper_static_validator_accepts_only_candidate_bound_compliance():
    baseline = _gripper_static()
    finalizer._validate_gripper_static(baseline, field="baseline")
    with pytest.raises(finalizer.VerificationError, match="schema drift"):
        finalizer._validate_gripper_static(
            baseline,
            field="candidate",
            expected_target_slew_profile=(
                finalizer.GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
            ),
        )

    candidate_contract = _candidate_gripper_static()
    finalizer._validate_gripper_static(
        candidate_contract,
        field="candidate",
        expected_target_slew_profile=(
            finalizer.GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
        ),
    )
    with pytest.raises(finalizer.VerificationError, match="schema drift"):
        finalizer._validate_gripper_static(candidate_contract, field="baseline")


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("total_write_count",), True, "total_write_count"),
        (("target_damping_ratio",), 1.0, "target/cadence"),
        (
            ("followers", 0, "natural_frequency_write_count"),
            0,
            "natural_frequency_write_count",
        ),
        (
            ("followers", 0, "post_reset_composed_usd_readback", "damping_ratio"),
            0.0,
            "post_reset_composed_usd_readback values",
        ),
    ],
)
def test_host_gripper_static_validator_rejects_compliance_tampering(path, value, match):
    contract = _candidate_gripper_static()
    target = contract["mimic_compliance"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(finalizer.VerificationError, match=match):
        finalizer._validate_gripper_static(
            contract,
            field="candidate",
            expected_target_slew_profile=(
                finalizer.GRIPPER_TARGET_SLEW_RATE_0P25_CANDIDATE_PROFILE
            ),
        )


def test_raw_smoke_gate_requires_pending_full_evidence():
    summary = finalizer._verify_raw(_valid_raw_result())
    assert summary["ordinary_pass_count"] == 13
    assert summary["delayed_close"]["apply_calls"] == 960
    assert summary["delayed_close"]["slew_limited_apply_count"] == 37
    assert summary["immediate_close_arm_velocity_headroom"]["maximum_ratio"] == 0.0
    assert summary["adversarial"]["slew_limit_events"] == 1

    raw = _valid_raw_result()
    raw["finalized"] = True
    with pytest.raises(finalizer.VerificationError, match="finalized"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["terminal_failure_evidence"] = {"status": "captured"}
    with pytest.raises(finalizer.VerificationError, match="failure evidence"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["gripper_delayed_close_replay"]["ik_safety"]["gripper_runtime_dynamic"][
        "driver_target_slew"
    ]["slew_limited_apply_count"] = 36
    with pytest.raises(finalizer.VerificationError, match="cadence/history"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    limit = finalizer.EXPECTED_VELOCITY_LIMITS[0]
    maximum = limit * 0.96
    raw["ik_safety_episodes"][0]["maxima"]["abs_joint_vel_rad_s"][0] = maximum
    headroom = raw["gripper_close_velocity_headroom"]["immediate_close_hold"]
    headroom["max_abs_joint_velocity_rad_s"][0] = maximum
    headroom["velocity_to_limit_ratio"][0] = 0.96
    headroom["maximum_ratio"] = 0.96
    with pytest.raises(finalizer.VerificationError, match="five-percent"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["ik_safety_episodes"][4]["counters"]["apply_calls"] = 359
    with pytest.raises(finalizer.VerificationError, match="apply_calls"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["raw_ik_safety_capture"]["soft_joint_pos_limits_rad"][0][0] += 0.01
    with pytest.raises(finalizer.VerificationError, match="limits"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["raw_ik_safety_capture"]["physx_derived_soft_joint_pos_limits_rad"][3][1] = (
        finalizer.EXPECTED_TARGET_LIMITS[3][1]
    )
    with pytest.raises(finalizer.VerificationError, match="physx_derived_soft"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["raw_ik_safety_capture"]["arm_velocity_target_rad_s"][4] = 1e-3
    with pytest.raises(finalizer.VerificationError, match="velocity_target"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["raw_ik_safety_capture"]["solver_velocity_iteration_count"] = 0
    with pytest.raises(finalizer.VerificationError, match="solver_velocity"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["raw_ik_safety_capture"]["physx_solver_type"] = 0
    with pytest.raises(finalizer.VerificationError, match="physx_solver_type"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["raw_ik_safety_capture"]["current_joint_velocity_abort"] = {}
    with pytest.raises(finalizer.VerificationError, match="must be null"):
        finalizer._verify_raw(raw)

    for path, value, match in (
        (("gripper_runtime_static", "profile"), "wrong", "gripper_static.profile"),
        (
            (
                "gripper_runtime_static",
                "driver_target_slew",
                "max_target_step_rad",
            ),
            0.1,
            "target_slew.max_target_step",
        ),
        (
            (
                "gripper_runtime_static",
                "driver_actuator",
                "resolved_velocity_limit_sim",
                "device",
            ),
            "cpu",
            "device",
        ),
        (
            ("gripper_runtime_dynamic", "driver_target_slew", "apply_calls"),
            359,
            "cadence/history",
        ),
        (
            (
                "gripper_runtime_dynamic",
                "driver_target_slew",
                "max_abs_target_step_rad",
            ),
            0.1,
            "maxima",
        ),
        (
            (
                "gripper_runtime_dynamic",
                "driver_target_slew",
                "last_requested_endpoint_rad",
            ),
            0.0,
            "state",
        ),
    ):
        raw = _valid_raw_result()
        target = raw["ik_safety_episodes"][0]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(finalizer.VerificationError, match=match):
            finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    open_slew = raw["ik_safety_episodes"][1]["gripper_runtime_dynamic"][
        "driver_target_slew"
    ]
    open_slew["slew_limited_apply_count"] = 1
    open_slew["endpoint_reached_apply_count"] -= 1
    with pytest.raises(finalizer.VerificationError, match="open hold"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["unexpected"] = True
    with pytest.raises(finalizer.VerificationError, match="top-level schema"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["results"][2]["position_error_m"] = 0.02
    with pytest.raises(finalizer.VerificationError, match="position error"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["ik_safety_episodes"][3]["max_raw_delta_diagnostic"]["raw_delta_joint_pos_rad"][
        "values"
    ][0] = 0.1
    with pytest.raises(finalizer.VerificationError, match="max-raw/maxima"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["ik_safety_episodes"][5]["counters"]["dls_fallbacks"] = 1
    with pytest.raises(finalizer.VerificationError, match="dls_fallbacks"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["schema_version"] = True
    with pytest.raises(finalizer.VerificationError, match="schema_version"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["exit_code"] = False
    with pytest.raises(finalizer.VerificationError, match="exit_code"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["ik_safety_episodes"][0]["counters"]["apply_calls"] = True
    with pytest.raises(finalizer.VerificationError, match="counters invalid"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    counters = raw["ik_safety_episodes"][0]["counters"]
    counters["slew_limit_events"] = 1
    counters["slew_limited_joints"] = 100
    with pytest.raises(finalizer.VerificationError, match="impossible"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    counters = raw["ik_safety_episodes"][0]["counters"]
    counters["slew_limit_events"] = 1
    counters["slew_limited_joints"] = 1
    with pytest.raises(finalizer.VerificationError, match="activation mismatch"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    safety = raw["ik_safety_episodes"][0]
    diagnostic = safety["max_raw_delta_diagnostic"]
    q0 = diagnostic["joint_pos_rad"]["values"][0]
    bound = finalizer.EXPECTED_MAX_DELTA[0]
    raw_delta = bound + 0.01
    safety["maxima"]["raw_delta_joint_pos_rad"][0] = raw_delta
    safety["maxima"]["applied_delta_joint_pos_rad"][0] = bound
    diagnostic["raw_delta_joint_pos_rad"]["values"][0] = raw_delta
    diagnostic["raw_joint_pos_target_rad"]["values"][0] = q0 + raw_delta
    diagnostic["safe_joint_pos_target_rad"]["values"][0] = q0 + bound
    with pytest.raises(finalizer.VerificationError, match="activation mismatch"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["ik_safety_episodes"][0]["max_raw_delta_diagnostic"]["pose_error_norm"] = None
    with pytest.raises(finalizer.VerificationError, match="pose_error_norm"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["ik_safety_episodes"][0]["max_raw_delta_diagnostic"]["jacobian_max_abs"] = None
    with pytest.raises(finalizer.VerificationError, match="jacobian_max_abs"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["ik_safety_episodes"][0]["max_raw_delta_diagnostic"]["eef_quaternion_norm"] = (
        1.0
    )
    with pytest.raises(finalizer.VerificationError, match="eef_quaternion_norm"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["results"][0]["actual_quaternion_wxyz"] = [
        math.cos(0.1),
        math.sin(0.1),
        0.0,
        0.0,
    ]
    with pytest.raises(
        finalizer.VerificationError, match="rotation error inconsistent"
    ):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["results"][1]["target_position"] = raw["results"][0]["target_position"].copy()
    raw["results"][1]["actual_position"] = raw["results"][1]["target_position"].copy()
    with pytest.raises(
        finalizer.VerificationError, match="translation target geometry"
    ):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    wrong_sign = raw["results"][8]["target_quaternion_wxyz"].copy()
    raw["results"][7]["target_quaternion_wxyz"] = wrong_sign
    raw["results"][7]["actual_quaternion_wxyz"] = wrong_sign.copy()
    with pytest.raises(finalizer.VerificationError, match="rotation target geometry"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["ik_safety_episodes"][0]["max_raw_delta_diagnostic"][
        "raw_joint_pos_target_rad"
    ]["values"][0] += 0.01
    with pytest.raises(finalizer.VerificationError, match="raw target identity"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["ik_safety_episodes"][0]["max_raw_delta_diagnostic"][
        "safe_joint_pos_target_rad"
    ]["values"][0] += 0.1
    with pytest.raises(finalizer.VerificationError, match="safe slew"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["ik_safety_episodes"][0]["target_joint_pos_limits_float32_sha256"] = "0" * 64
    with pytest.raises(finalizer.VerificationError, match="target_joint_pos_limits"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["ik_safety_episodes"][0]["maxima"][
        "post_clamp_target_guard_band_violation_rad"
    ][0] = 2e-5
    with pytest.raises(finalizer.VerificationError, match="target guard-band maxima"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    raw["ik_safety_episodes"][0]["maxima"][
        "post_clamp_target_guard_band_violation_rad"
    ][0] = 5e-6
    with pytest.raises(
        finalizer.VerificationError,
        match="target guard-band recovery attribution",
    ):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    diagnostic = raw["ik_safety_adversarial"]["ik_safety"]["max_raw_delta_diagnostic"]
    safety = raw["ik_safety_adversarial"]["ik_safety"]
    safety["maxima"]["raw_delta_joint_pos_rad"] = [0.0] * 7
    safety["maxima"]["applied_delta_joint_pos_rad"] = [0.0] * 7
    diagnostic["raw_delta_joint_pos_rad"] = _diagnostic_vector([0.0] * 7)
    diagnostic["raw_joint_pos_target_rad"] = copy.deepcopy(diagnostic["joint_pos_rad"])
    diagnostic["safe_joint_pos_target_rad"] = copy.deepcopy(diagnostic["joint_pos_rad"])
    with pytest.raises(finalizer.VerificationError, match="activation mismatch"):
        finalizer._verify_raw(raw)

    raw = _valid_raw_result()
    dq = raw["ik_safety_adversarial"]["joint_state"]["joint_vel_rad_s"]
    dq["values"][0] = finalizer.EXPECTED_VELOCITY_LIMITS[0] + 0.1
    dq["max_abs"] = dq["values"][0]
    with pytest.raises(finalizer.VerificationError, match="terminal dq"):
        finalizer._verify_raw(raw)


def test_attestation_is_bound_verified_and_nonoverwriting(tmp_path, monkeypatch):
    args = _attestation_args(tmp_path, monkeypatch)
    expected = finalizer._build_expected(args)
    finalizer._publish_nonoverwriting(args.attestation, expected)

    attestation, _, _ = finalizer._read_json_once(args.attestation, "attestation")
    assert attestation == finalizer._build_expected(args)
    assert args.attestation.stat().st_mode & 0o777 == 0o444
    with pytest.raises(finalizer.VerificationError, match="already exists"):
        finalizer._publish_nonoverwriting(args.attestation, expected)

    args.srun_rc = 9
    with pytest.raises(finalizer.VerificationError, match="srun_rc"):
        finalizer._build_expected(args)


def test_attestation_rejects_writable_or_mutated_evidence(tmp_path, monkeypatch):
    args = _attestation_args(tmp_path, monkeypatch)
    args.raw_result.chmod(0o644)
    with pytest.raises(finalizer.VerificationError, match="mode"):
        finalizer._build_expected(args)

    args = _attestation_args(tmp_path / "second", monkeypatch)
    marker = args.raw_result.with_name(args.raw_result.name + ".ready.json")
    marker.chmod(0o644)
    with pytest.raises(finalizer.VerificationError, match="ready marker mode"):
        finalizer._build_expected(args)

    args = _attestation_args(tmp_path / "third", monkeypatch)
    args.raw_result.chmod(0o644)
    args.raw_result.write_bytes(args.raw_result.read_bytes() + b" ")
    args.raw_result.chmod(0o444)
    with pytest.raises(finalizer.VerificationError, match="ready marker"):
        finalizer._build_expected(args)

    args = _attestation_args(tmp_path / "fourth", monkeypatch)
    args.expected_image_sha256 = "0" * 64
    with pytest.raises(finalizer.VerificationError, match="image digest"):
        finalizer._build_expected(args)

    args = _attestation_args(tmp_path / "fifth", monkeypatch)
    args.saved_job_script.write_text("tampered\n")
    with pytest.raises(finalizer.VerificationError, match="job script digest"):
        finalizer._build_expected(args)

    args = _attestation_args(tmp_path / "sixth", monkeypatch)
    dirty_path = args.polaris_repo / "dirty.txt"
    dirty_path.write_text("dirty\n")
    with pytest.raises(finalizer.VerificationError, match="repo dirty"):
        finalizer._build_expected(args)

    args = _attestation_args(tmp_path / "seventh", monkeypatch)
    args.expected_finalizer_sha256 = "0" * 64
    with pytest.raises(finalizer.VerificationError, match="finalizer expected digest"):
        finalizer._build_expected(args)

    args = _attestation_args(tmp_path / "eighth", monkeypatch)
    args.expected_saved_job_script_sha256 = "0" * 64
    with pytest.raises(finalizer.VerificationError, match="saved job script expected"):
        finalizer._build_expected(args)

    args = _attestation_args(tmp_path / "ninth", monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "99999")
    with pytest.raises(finalizer.VerificationError, match="SLURM_JOB_ID"):
        finalizer._build_expected(args)

    args = _attestation_args(tmp_path / "tenth", monkeypatch)
    marker = args.raw_result.with_name(args.raw_result.name + ".ready.json")
    marker_payload = json.loads(marker.read_text())
    marker_payload["schema_version"] = True
    marker.chmod(0o644)
    marker.write_text(json.dumps(marker_payload, indent=2) + "\n")
    marker.chmod(0o444)
    with pytest.raises(finalizer.VerificationError, match="ready marker"):
        finalizer._build_expected(args)

    assert not finalizer._typed_equal({"schema_version": True}, {"schema_version": 1})
