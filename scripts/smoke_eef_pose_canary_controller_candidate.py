#!/usr/bin/env python3
"""Replay exact failed canary actions through isolated controller candidates.

This is a model-free promotion gate. Both variants use one identical default-
off controller profile: 0.95 nominal arm slew, factor-0.25 gripper target slew,
the profile-bound 86-substep fixed activation-anchor close interlock, and the
pre-articulation PhysX mimic-compliance overlay (100 rad/s, damping ratio 1.2).
After the 120 content-pinned actions, seven repeats of the final recorded
action give the official close transition exactly 96 applies: 86 anchored and
10 released. The open reasoning fixture proves the same static controller and
mimic-compliance identity without activating the close interlock.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import struct
import sys
import traceback
from typing import Any

import smoke_eef_pose_canary_trace_replay as gate0


PROFILE = "polaris_eef_canary_controller_candidate_replay_v3"
FIXTURE_ACTION_COUNT = 120
POST_FIXTURE_REPEAT_COUNT = 7
TOTAL_ACTION_COUNT = FIXTURE_ACTION_COUNT + POST_FIXTURE_REPEAT_COUNT
CANDIDATE_TARGET_SLEW_PROFILE = (
    "eef_binary_driver_target_slew_rate1p25_from_live_limit5_"
    "per_120hz_substep_candidate_v1"
)
CANDIDATE_TARGET_SLEW_MAX_STEP_RAD = 0.010416666977107525
CANDIDATE_MIMIC_COMPLIANCE_PROFILE = (
    "robotiq_2f85_live_physx_mimic_frequency100_damping1p2_candidate_v1"
)
MIMIC_COMPLIANCE_FIELDS = {
    "profile",
    "enabled",
    "scope",
    "timing",
    "setter",
    "live_root_profile",
    "live_root_path",
    "original_spawn_func",
    "overlay_func",
    "original_spawn_call_count",
    "overlay_call_count",
    "physics_hz",
    "physics_dt",
    "target_natural_frequency_rad_s",
    "target_damping_ratio",
    "frequency_timestep_product",
    "follower_count",
    "natural_frequency_write_count",
    "damping_ratio_write_count",
    "total_write_count",
    "source_usd_sha256",
    "source_usd_unchanged_after_spawn_overlay",
    "followers",
}
MIMIC_COMPLIANCE_FOLLOWER_FIELDS = {
    "joint_name",
    "joint_index",
    "live_prim_path",
    "mimic_axis",
    "natural_frequency_attribute",
    "damping_ratio_attribute",
    "source",
    "before_spawn_write",
    "before_spawn_structure",
    "natural_frequency_write_count",
    "damping_ratio_write_count",
    "after_spawn_write",
    "after_spawn_structure",
    "post_reset_composed_usd_readback",
    "post_reset_composed_usd_structure",
}
MIMIC_COMPLIANCE_SNAPSHOT_FIELDS = {
    "natural_frequency_rad_s",
    "damping_ratio",
}


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _next_float32_toward(left: float, right: float) -> float:
    left = _float32(left)
    right = _float32(right)
    if left == right:
        return right
    bits = struct.unpack("<I", struct.pack("<f", left))[0]
    bits = bits - 1 if left > right and left > 0.0 else bits + 1
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _simulate_candidate_close_transition() -> dict[str, int]:
    endpoint = _float32(math.pi / 4.0)
    maximum = _float32(CANDIDATE_TARGET_SLEW_MAX_STEP_RAD)
    previous = _float32(0.0)
    applies = 0
    limited_applies = 0
    nextafter_corrections = 0
    while previous != endpoint:
        if applies >= 1024:
            raise RuntimeError("Candidate close simulation did not end")
        delta = _float32(endpoint - previous)
        limited = abs(delta) > maximum
        step = min(max(delta, -maximum), maximum)
        next_target = _float32(previous + step) if limited else endpoint
        applied = _float32(next_target - previous)
        if abs(applied) > maximum:
            next_target = _next_float32_toward(next_target, previous)
            nextafter_corrections += 1
            applied = _float32(next_target - previous)
        if not (
            math.isfinite(next_target)
            and math.isfinite(applied)
            and abs(applied) <= maximum
            and previous <= next_target <= endpoint
        ):
            raise RuntimeError("Candidate close simulation invariant")
        previous = next_target
        applies += 1
        limited_applies += int(limited)
    return {
        "endpoint_applies": applies,
        "limited_applies": limited_applies,
        "nextafter_corrections": nextafter_corrections,
    }


CANDIDATE_CLOSE_SIMULATION = _simulate_candidate_close_transition()
if CANDIDATE_CLOSE_SIMULATION != {
    "endpoint_applies": 76,
    "limited_applies": 75,
    "nextafter_corrections": 41,
}:
    raise RuntimeError("Candidate float32 close-transition profile drift")
CANDIDATE_CLOSE_TRANSITION_APPLIES = CANDIDATE_CLOSE_SIMULATION["endpoint_applies"]
CANDIDATE_CLOSE_LIMITED_APPLIES = CANDIDATE_CLOSE_SIMULATION["limited_applies"]
CANDIDATE_CLOSE_INTERLOCK_SUBSTEPS = CANDIDATE_CLOSE_TRANSITION_APPLIES + 10
CANDIDATE_CLOSE_INTERLOCK_PROFILE = (
    "eef_gripper_close_fixed_activation_anchor_86_physics_substeps_v2"
)
CANDIDATE_PROFILE = (
    "arm_slew_0p95_gripper_rate0p25_fixed_anchor86_mimic100_damping1p2_v3"
)
CANDIDATE_BY_VARIANT = {
    "official_lap3b": CANDIDATE_PROFILE,
    "reasoning_43075": CANDIDATE_PROFILE,
}
ARM_CANDIDATE_FIELDS = {
    "enabled",
    "profile",
    "ratio",
    "physical_max_delta_joint_pos_rad",
    "nominal_max_delta_joint_pos_rad",
}
INTERLOCK_CANDIDATE_FIELDS = {
    "enabled",
    "profile",
    "configured_substeps",
    "remaining_substeps",
    "observed_endpoint_change_count",
    "endpoint_observed",
    "activation_count",
    "active_apply_count",
    "anchor_valid",
    "anchor_capture_count",
    "anchor_target_apply_count",
    "anchor_first_exact_target_count",
    "anchor_refresh_count",
    "anchor_slew_limit_event_count",
    "anchor_slew_limited_joint_count",
    "anchor_position_limit_event_count",
    "anchor_position_limited_joint_count",
    "anchor_completion_count",
    "anchor_open_cancel_count",
    "last_activation_apply_index",
    "last_anchor_joint_pos_rad",
    "last_anchor_little_endian_float32_sha256",
    "max_abs_current_anchor_residual_rad",
    "max_abs_target_anchor_residual_rad",
    "max_abs_active_delta_joint_pos_rad",
    "released_apply_count",
    "max_abs_released_delta_joint_pos_rad",
}
ZERO_SAFETY_COUNTERS = {
    "current_joint_limit_aborts",
    "dls_fallbacks",
    "guard_diagnostics_dropped",
    "invariant_aborts",
    "nonfinite_aborts",
    "position_limit_events",
    "position_limited_joints",
    "post_clamp_target_violations",
}
CONTROLLER_ABORT_CAPTURE_PROFILE = (
    "polaris_eef_controller_candidate_transactional_abort_capture_v3"
)
CONTROLLER_ABORT_CAPTURE_FIELDS = {
    "profile",
    "failure_exception",
    "parsed_failure",
    "arm_failure_runtime_evidence",
    "all_six_gripper_tail",
    "active_safety",
    "active_candidate",
    "active_target_slew",
}
FAILURE_CONTEXT_FIELDS = {
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
}
FAILURE_RAW_FIELDS = {
    "schema_version",
    "profile",
    "finalized",
    "passed",
    "stage",
    "environment",
    "variant",
    "candidate",
    "policy_step",
    "failure_context",
    "failure",
    "controller_abort_capture",
    "controller_abort_capture_failure",
    "close_failures",
}


class CandidateReplayValidationError(ValueError):
    """The candidate replay or its evidence violated the promotion contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateReplayValidationError(message)


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


def validate_candidate_mimic_compliance(
    gripper_contract: Any,
) -> dict[str, Any]:
    """Independently bind the pre-articulation compliance write transaction."""

    _require(isinstance(gripper_contract, dict), "candidate gripper static contract")
    source = gripper_contract.get("mimic_joint_contract")
    compliance = gripper_contract.get("mimic_compliance")
    _require(
        isinstance(source, dict)
        and isinstance(source.get("followers"), list)
        and len(source["followers"]) == 5,
        "candidate mimic source contract",
    )
    _require(
        isinstance(compliance, dict)
        and set(compliance) == MIMIC_COMPLIANCE_FIELDS
        and compliance.get("profile") == CANDIDATE_MIMIC_COMPLIANCE_PROFILE
        and compliance.get("enabled") is True
        and compliance.get("scope")
        == "eef_rate0p25_candidate_only_source_usd_immutable_v1"
        and compliance.get("timing")
        == "after_original_usd_spawn_before_articulation_initialization_v1"
        and compliance.get("setter") == "UsdAttribute.Set_default_float_v1"
        and compliance.get("live_root_profile")
        == "single_composed_world_env0_robot_root_v1"
        and compliance.get("live_root_path") == "/World/envs/env_0/robot"
        and compliance.get("original_spawn_func")
        == {
            "module": "isaaclab.sim.spawners.from_files.from_files",
            "qualname": "spawn_from_usd",
            "name": "spawn_from_usd",
        }
        and compliance.get("overlay_func")
        == {
            "module": "polaris.eef_gripper_runtime",
            "qualname": "eef_mimic_compliance_spawn_overlay",
            "name": "eef_mimic_compliance_spawn_overlay",
        }
        and compliance.get("original_spawn_call_count") == 1
        and compliance.get("overlay_call_count") == 1
        and compliance.get("follower_count") == 5
        and compliance.get("natural_frequency_write_count") == 5
        and compliance.get("damping_ratio_write_count") == 5
        and compliance.get("total_write_count") == 10
        and compliance.get("source_usd_sha256") == source.get("robot_usd_sha256")
        and compliance.get("source_usd_unchanged_after_spawn_overlay") is True,
        "candidate mimic-compliance transaction identity drift",
    )
    _require(
        all(
            type(compliance.get(name)) is float
            for name in (
                "physics_hz",
                "physics_dt",
                "target_natural_frequency_rad_s",
                "target_damping_ratio",
                "frequency_timestep_product",
            )
        )
        and compliance.get("physics_hz") == 120.0
        and compliance.get("physics_dt") == 1.0 / 120.0
        and _float32(compliance.get("target_natural_frequency_rad_s")) == 100.0
        and _float32(compliance.get("target_damping_ratio")) == _float32(1.2)
        and compliance.get("frequency_timestep_product") == 5.0 / 6.0
        and compliance["physics_dt"] * compliance["target_natural_frequency_rad_s"]
        == compliance["frequency_timestep_product"],
        "candidate mimic-compliance cadence/value drift",
    )
    followers = compliance.get("followers")
    _require(
        isinstance(followers, list) and len(followers) == 5,
        "candidate mimic-compliance followers",
    )
    for index, (source_follower, follower) in enumerate(
        zip(source["followers"], followers, strict=True)
    ):
        axis = source_follower.get("mimic_axis")
        expected_live_path = "/World/envs/env_0/robot" + source_follower.get(
            "prim_path", ""
        ).removeprefix("/panda")
        expected_frequency_attribute = f"physxMimicJoint:{axis}:naturalFrequency"
        expected_damping_attribute = f"physxMimicJoint:{axis}:dampingRatio"
        expected_structure = {
            "applied_mimic_api": f"PhysxMimicJointAPI:{axis}",
            "reference_joint_path": (
                "/World/envs/env_0/robot/Gripper/Robotiq_2F_85/Joints/finger_joint"
            ),
            "gearing": source_follower.get("gearing"),
            "offset": 0.0,
            "exclude_from_articulation": False,
        }
        _require(
            isinstance(follower, dict)
            and set(follower) == MIMIC_COMPLIANCE_FOLLOWER_FIELDS
            and follower.get("joint_name") == source_follower.get("joint_name")
            and follower.get("joint_index") == source_follower.get("joint_index")
            and follower.get("live_prim_path") == expected_live_path
            and follower.get("mimic_axis") == axis
            and follower.get("natural_frequency_attribute")
            == expected_frequency_attribute
            and follower.get("damping_ratio_attribute") == expected_damping_attribute
            and follower.get("natural_frequency_write_count") == 1
            and follower.get("damping_ratio_write_count") == 1,
            f"candidate mimic-compliance follower {index} identity drift",
        )
        for field in (
            "before_spawn_structure",
            "after_spawn_structure",
            "post_reset_composed_usd_structure",
        ):
            _require(
                _typed_equal(follower.get(field), expected_structure),
                f"candidate mimic-compliance follower {index} {field} drift",
            )
        for field in ("source", "before_spawn_write"):
            snapshot = follower.get(field)
            _require(
                isinstance(snapshot, dict)
                and set(snapshot) == MIMIC_COMPLIANCE_SNAPSHOT_FIELDS
                and type(snapshot.get("natural_frequency_rad_s")) is float
                and type(snapshot.get("damping_ratio")) is float
                and type(source_follower.get("natural_frequency_hz")) is float
                and type(source_follower.get("damping_ratio")) is float
                and _float32(snapshot.get("natural_frequency_rad_s"))
                == _float32(source_follower.get("natural_frequency_hz"))
                and _float32(snapshot.get("damping_ratio"))
                == _float32(source_follower.get("damping_ratio")),
                f"candidate mimic-compliance follower {index} {field} drift",
            )
        for field in ("after_spawn_write", "post_reset_composed_usd_readback"):
            snapshot = follower.get(field)
            _require(
                isinstance(snapshot, dict)
                and set(snapshot) == MIMIC_COMPLIANCE_SNAPSHOT_FIELDS
                and type(snapshot.get("natural_frequency_rad_s")) is float
                and type(snapshot.get("damping_ratio")) is float
                and _float32(snapshot.get("natural_frequency_rad_s")) == 100.0
                and _float32(snapshot.get("damping_ratio")) == _float32(1.2),
                f"candidate mimic-compliance follower {index} {field} drift",
            )
    return dict(compliance)


def validate_failure_context(value: Any, *, variant: str) -> dict[str, Any]:
    """Validate the closed pre-replay context retained by a failed raw."""

    _require(variant in CANDIDATE_BY_VARIANT, "failure-context variant")
    _require(
        isinstance(value, dict) and set(value) == FAILURE_CONTEXT_FIELDS,
        "failure-context schema drift",
    )
    lifecycle = value.get("lifecycle")
    _require(
        isinstance(lifecycle, dict)
        and set(lifecycle)
        == {
            "profile",
            "launch_id",
            "job_id",
            "step_id",
            "nodelist",
            "procid",
            "localid",
            "ntasks",
        }
        and type(lifecycle.get("profile")) is str
        and lifecycle.get("profile") == "slurm_single_task_srun_lifecycle_v1"
        and type(lifecycle.get("launch_id")) is str
        and len(lifecycle["launch_id"]) == 64
        and all(character in "0123456789abcdef" for character in lifecycle["launch_id"])
        and type(lifecycle.get("job_id")) is int
        and lifecycle["job_id"] > 0
        and type(lifecycle.get("step_id")) is int
        and lifecycle["step_id"] >= 0
        and type(lifecycle.get("nodelist")) is str
        and bool(lifecycle["nodelist"].strip())
        and type(lifecycle.get("procid")) is int
        and type(lifecycle.get("localid")) is int
        and lifecycle.get("procid") == lifecycle.get("localid") == 0
        and type(lifecycle.get("ntasks")) is int
        and lifecycle.get("ntasks") == 1,
        "failure-context lifecycle drift",
    )
    repository = value.get("repository")
    _require(
        isinstance(repository, dict)
        and set(repository) == {"path", "commit", "clean_tracked"}
        and isinstance(repository.get("path"), str)
        and isinstance(repository.get("commit"), str)
        and len(repository["commit"]) == 40
        and repository.get("clean_tracked") is True,
        "failure-context repository drift",
    )
    container = value.get("container_image")
    _require(
        isinstance(container, dict)
        and _typed_equal(
            container,
            validate_container_argument(
                container.get("path"),
                size_bytes=container.get("size_bytes"),
                sha256=container.get("sha256"),
            ),
        ),
        "failure-context container drift",
    )
    fixture = value.get("fixture")
    _require(
        isinstance(fixture, dict)
        and type(fixture.get("fixture_action_count")) is int
        and fixture["fixture_action_count"] == FIXTURE_ACTION_COUNT,
        "failure-context fixture drift",
    )
    expected_action_plan = {
        "profile": "exact_fixture_then_repeat_final_recorded_action_v1",
        "fixture_action_count": FIXTURE_ACTION_COUNT,
        "post_fixture_repeat_count": POST_FIXTURE_REPEAT_COUNT,
        "total_action_count": TOTAL_ACTION_COUNT,
    }
    _require(
        _typed_equal(value.get("action_plan"), expected_action_plan),
        "failure-context action plan drift",
    )
    _require(
        isinstance(value.get("production_eval"), dict)
        and isinstance(value.get("boundary_helper"), dict)
        and isinstance(value.get("assets"), dict)
        and value["assets"].get("contract") == gate0.EXPECTED_ASSET_CONTRACT,
        "failure-context source/asset drift",
    )
    runtime_protocol = value.get("runtime_protocol")
    _require(
        isinstance(runtime_protocol, dict)
        and runtime_protocol.get("decimation") == gate0.DECIMATION
        and runtime_protocol.get("physics_hz") == 120.0
        and runtime_protocol.get("policy_hz") == 15.0,
        "failure-context runtime protocol drift",
    )
    runtime_frame = value.get("runtime_frame")
    _require(
        isinstance(runtime_frame, dict)
        and runtime_frame.get("eef_frame") == "panda_link8"
        and runtime_frame.get("reference_frame") == "panda_link0"
        and runtime_frame.get("controlled_body") == "panda_link8",
        "failure-context runtime frame drift",
    )
    gripper_contract = value.get("gripper_runtime_contract")
    _require(
        isinstance(gripper_contract, dict)
        and gripper_contract.get("driver_target_slew", {}).get("profile")
        == CANDIDATE_TARGET_SLEW_PROFILE,
        "failure-context gripper target-slew drift",
    )
    validate_candidate_mimic_compliance(gripper_contract)
    initial_safety = value.get("initial_safety")
    _require(
        isinstance(initial_safety, dict)
        and initial_safety.get("counters", {}).get("apply_calls") == 0
        and initial_safety.get("current_joint_velocity_abort") is None
        and initial_safety.get("gripper_runtime_static") == gripper_contract,
        "failure-context initial safety drift",
    )
    validate_candidate_report(
        value.get("initial_candidate"), variant=variant, final=False
    )
    return dict(value)


def validate_candidate_report(
    report: Any, *, variant: str, final: bool | None
) -> dict[str, Any]:
    _require(isinstance(report, dict), "candidate report must be an object")
    _require(
        set(report) == {"arm_slew_headroom", "gripper_close_arm_interlock"},
        "candidate report schema drift",
    )
    arm = report["arm_slew_headroom"]
    interlock = report["gripper_close_arm_interlock"]
    _require(
        isinstance(arm, dict) and isinstance(interlock, dict), "candidate sections"
    )
    _require(set(arm) == ARM_CANDIDATE_FIELDS, "arm candidate schema drift")
    _require(
        set(interlock) == INTERLOCK_CANDIDATE_FIELDS,
        "interlock candidate schema drift",
    )
    _require(
        arm.get("enabled") is True
        and arm.get("profile") == "panda_nominal_target_slew_0p95_physical_limit_v1"
        and arm.get("ratio") == 0.95,
        "arm-slew candidate identity drift",
    )
    physical = arm.get("physical_max_delta_joint_pos_rad")
    nominal = arm.get("nominal_max_delta_joint_pos_rad")
    _require(
        isinstance(physical, list)
        and isinstance(nominal, list)
        and len(physical) == len(nominal) == 7,
        "arm-slew candidate vector shape",
    )
    for index, (outer, inner) in enumerate(zip(physical, nominal, strict=True)):
        _require(
            isinstance(outer, (int, float))
            and not isinstance(outer, bool)
            and isinstance(inner, (int, float))
            and not isinstance(inner, bool)
            and math.isfinite(float(outer))
            and math.isfinite(float(inner))
            and 0.0 < float(inner) < float(outer),
            f"arm-slew candidate bound {index}",
        )
        _require(
            math.isclose(float(inner), float(outer) * 0.95, rel_tol=2e-7),
            f"arm-slew candidate ratio {index}",
        )

    _require(
        interlock.get("enabled") is True
        and interlock.get("profile") == CANDIDATE_CLOSE_INTERLOCK_PROFILE
        and type(interlock.get("configured_substeps")) is int
        and interlock.get("configured_substeps") == CANDIDATE_CLOSE_INTERLOCK_SUBSTEPS,
        "close-interlock candidate identity drift",
    )
    counter_fields = (
        "remaining_substeps",
        "observed_endpoint_change_count",
        "activation_count",
        "active_apply_count",
        "released_apply_count",
        "anchor_capture_count",
        "anchor_target_apply_count",
        "anchor_first_exact_target_count",
        "anchor_refresh_count",
        "anchor_slew_limit_event_count",
        "anchor_slew_limited_joint_count",
        "anchor_position_limit_event_count",
        "anchor_position_limited_joint_count",
        "anchor_completion_count",
        "anchor_open_cancel_count",
    )
    for name in counter_fields:
        _require(
            type(interlock.get(name)) is int and interlock[name] >= 0,
            f"close-interlock {name} type/range drift",
        )
    _require(
        type(interlock.get("endpoint_observed")) is bool,
        "close-interlock endpoint_observed type drift",
    )
    _require(
        type(interlock.get("anchor_valid")) is bool,
        "close-interlock anchor_valid type drift",
    )
    active_vector = interlock.get("max_abs_active_delta_joint_pos_rad")
    released_vector = interlock.get("max_abs_released_delta_joint_pos_rad")
    current_anchor_residual = interlock.get("max_abs_current_anchor_residual_rad")
    target_anchor_residual = interlock.get("max_abs_target_anchor_residual_rad")
    for field, vector in (
        ("active", active_vector),
        ("released", released_vector),
        ("current-anchor residual", current_anchor_residual),
        ("target-anchor residual", target_anchor_residual),
    ):
        _require(
            isinstance(vector, list)
            and len(vector) == 7
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) >= 0.0
                for value in vector
            ),
            f"close-interlock {field} vector",
        )
    for index, (active_delta, bound) in enumerate(
        zip(active_vector, nominal, strict=True)
    ):
        _require(
            float(active_delta) <= float(bound) + 1e-6,
            f"close-interlock active slew bound {index}",
        )
    for index, (target_residual, current_residual) in enumerate(
        zip(target_anchor_residual, current_anchor_residual, strict=True)
    ):
        _require(
            float(target_residual) <= float(current_residual) + 1e-6,
            f"close-interlock anchor direction {index}",
        )

    activation_count = interlock["activation_count"]
    active_count = interlock["active_apply_count"]
    released_count = interlock["released_apply_count"]
    remaining = interlock["remaining_substeps"]
    anchor_valid = interlock["anchor_valid"]
    capture_count = interlock["anchor_capture_count"]
    anchor_target_count = interlock["anchor_target_apply_count"]
    first_exact_count = interlock["anchor_first_exact_target_count"]
    completion_count = interlock["anchor_completion_count"]
    open_cancel_count = interlock["anchor_open_cancel_count"]
    _require(
        remaining <= CANDIDATE_CLOSE_INTERLOCK_SUBSTEPS
        and activation_count in (0, 1)
        and active_count <= CANDIDATE_CLOSE_INTERLOCK_SUBSTEPS
        and capture_count == activation_count
        and anchor_target_count == active_count
        and first_exact_count == capture_count
        and interlock["anchor_refresh_count"] == 0
        and anchor_valid == (remaining > 0)
        and completion_count + open_cancel_count + int(anchor_valid) == capture_count,
        "close-interlock fixed activation-anchor lifecycle drift",
    )
    for prefix in ("slew", "position"):
        event_count = interlock[f"anchor_{prefix}_limit_event_count"]
        joint_count = interlock[f"anchor_{prefix}_limited_joint_count"]
        _require(
            event_count <= anchor_target_count
            and event_count <= joint_count <= 7 * event_count,
            f"close-interlock anchor {prefix}-limit counter drift",
        )

    last_activation_apply_index = interlock.get("last_activation_apply_index")
    last_anchor = interlock.get("last_anchor_joint_pos_rad")
    last_anchor_digest = interlock.get("last_anchor_little_endian_float32_sha256")
    if capture_count == 0:
        _require(
            last_activation_apply_index is None
            and last_anchor is None
            and last_anchor_digest is None,
            "close-interlock inactive anchor evidence drift",
        )
    else:
        _require(
            type(last_activation_apply_index) is int
            and last_activation_apply_index == 920,
            "close-interlock activation apply index drift",
        )
        _require(
            isinstance(last_anchor, list)
            and len(last_anchor) == 7
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and _float32(float(value)) == float(value)
                for value in last_anchor
            ),
            "close-interlock last anchor vector drift",
        )
        _require(
            isinstance(last_anchor_digest, str)
            and len(last_anchor_digest) == 64
            and all(
                character in "0123456789abcdef" for character in last_anchor_digest
            ),
            "close-interlock last anchor digest shape drift",
        )
        expected_anchor_digest = hashlib.sha256(
            struct.pack("<7f", *(float(value) for value in last_anchor))
        ).hexdigest()
        _require(
            last_anchor_digest == expected_anchor_digest,
            "close-interlock last anchor digest mismatch",
        )

    zero_anchor_vectors = all(
        float(value) == 0.0
        for vector in (current_anchor_residual, target_anchor_residual)
        for value in vector
    )
    if final is False:
        _require(
            remaining == 0
            and interlock.get("observed_endpoint_change_count") == 0
            and interlock.get("endpoint_observed") is False
            and activation_count == 0
            and active_count == 0
            and released_count == 0
            and zero_anchor_vectors
            and all(float(value) == 0.0 for value in active_vector)
            and all(float(value) == 0.0 for value in released_vector),
            "initial close-interlock state is not empty",
        )
    elif final is True and variant == "official_lap3b":
        _require(
            remaining == 0
            and interlock.get("observed_endpoint_change_count") == 1
            and interlock.get("endpoint_observed") is True
            and activation_count == 1
            and active_count == CANDIDATE_CLOSE_INTERLOCK_SUBSTEPS
            and released_count == 10
            and capture_count == 1
            and anchor_target_count == CANDIDATE_CLOSE_INTERLOCK_SUBSTEPS
            and first_exact_count == 1
            and completion_count == 1
            and open_cancel_count == 0
            and anchor_valid is False
            and interlock["anchor_position_limit_event_count"] == 0
            and interlock["anchor_position_limited_joint_count"] == 0
            and any(float(value) > 0.0 for value in released_vector),
            "official fixed-anchor interlock did not complete and release",
        )
    elif final is True:
        _require(
            remaining == 0
            and interlock.get("observed_endpoint_change_count") == 0
            and interlock.get("endpoint_observed") is True
            and activation_count == 0
            and active_count == 0
            and released_count == 0
            and zero_anchor_vectors
            and all(float(value) == 0.0 for value in active_vector)
            and all(float(value) == 0.0 for value in released_vector),
            "reasoning open replay unexpectedly activated the close interlock",
        )
    else:
        _require(
            open_cancel_count == 0
            and (variant == "official_lap3b" or activation_count == 0),
            "captured close-interlock state is impossible",
        )
        if activation_count == 0:
            _require(
                active_count == released_count == remaining == 0
                and zero_anchor_vectors
                and all(float(value) == 0.0 for value in active_vector)
                and all(float(value) == 0.0 for value in released_vector),
                "inactive captured close-interlock state retained evidence",
            )
        elif remaining > 0:
            _require(
                released_count == 0
                and completion_count == 0
                and anchor_valid is True
                and active_count + remaining == CANDIDATE_CLOSE_INTERLOCK_SUBSTEPS,
                "active captured close-interlock countdown drift",
            )
        else:
            _require(
                active_count == CANDIDATE_CLOSE_INTERLOCK_SUBSTEPS
                and completion_count == 1
                and anchor_valid is False,
                "released captured close-interlock countdown drift",
            )
    return dict(report)


def validate_controller_abort_capture(value: Any, *, variant: str) -> dict[str, Any]:
    """Validate one failure-only, non-promotable controller transaction."""

    _require(variant in CANDIDATE_BY_VARIANT, "controller-abort variant")
    _require(
        isinstance(value, dict) and set(value) == CONTROLLER_ABORT_CAPTURE_FIELDS,
        "controller-abort capture schema drift",
    )
    _require(
        value.get("profile") == CONTROLLER_ABORT_CAPTURE_PROFILE,
        "controller-abort capture profile drift",
    )
    failure = value.get("failure_exception")
    _require(
        isinstance(failure, dict)
        and set(failure) == {"type", "message", "traceback"}
        and isinstance(failure.get("type"), str)
        and failure["type"].endswith(".DifferentialIKInvariantError")
        and isinstance(failure.get("message"), str)
        and isinstance(failure.get("traceback"), str),
        "controller-abort exception evidence",
    )
    parsed = gate0.parse_failure_exception(failure["message"])
    _require(
        value.get("parsed_failure") == parsed,
        "controller-abort parsed exception drift",
    )
    arm_failure = value.get("arm_failure_runtime_evidence")
    gate0._validate_arm_failure_runtime_evidence(  # noqa: SLF001
        arm_failure,
        expected_failure=parsed,
    )
    gate0.validate_gripper_tail(
        value.get("all_six_gripper_tail"),
        expected_failure=parsed,
    )
    _require(isinstance(arm_failure, dict), "controller-abort arm evidence")
    active_safety = value.get("active_safety")
    _require(
        isinstance(active_safety, dict)
        and active_safety == arm_failure.get("ik_safety")
        and isinstance(active_safety.get("current_joint_velocity_abort"), dict),
        "controller-abort active/current-abort safety drift",
    )
    validate_candidate_mimic_compliance(active_safety.get("gripper_runtime_static"))
    active_candidate = validate_candidate_report(
        value.get("active_candidate"), variant=variant, final=None
    )
    target_slew = value.get("active_target_slew")
    dynamic = active_safety.get("gripper_runtime_dynamic")
    _require(
        isinstance(dynamic, dict)
        and isinstance(target_slew, dict)
        and target_slew == dynamic.get("driver_target_slew")
        and target_slew.get("profile") == CANDIDATE_TARGET_SLEW_PROFILE,
        "controller-abort active target-slew state drift",
    )
    _require(
        active_candidate["gripper_close_arm_interlock"]["enabled"] is True,
        "controller-abort close interlock was not enabled",
    )
    if (
        variant == "official_lap3b"
        and parsed["policy_step"] == 121
        and parsed["physics_substep"] == 0
    ):
        interlock = active_candidate["gripper_close_arm_interlock"]
        counters = active_safety.get("counters")
        _require(
            isinstance(counters, dict)
            and counters.get("apply_calls") == 969
            and target_slew.get("apply_calls") == 968
            and target_slew.get("slew_limited_apply_count") == 48
            and target_slew.get("endpoint_reached_apply_count") == 920
            and interlock["activation_count"] == 1
            and interlock["active_apply_count"] == 48
            and interlock["remaining_substeps"] == 38
            and interlock["anchor_valid"] is True
            and interlock["anchor_capture_count"] == 1
            and interlock["anchor_target_apply_count"] == 48
            and interlock["anchor_completion_count"] == 0
            and interlock["anchor_open_cancel_count"] == 0
            and interlock["last_activation_apply_index"] == 920,
            "policy121/substep0 transactional fixed-anchor state drift",
        )
    return dict(value)


def validate_failure_payload(
    value: Any, *, variant: str, require_complete_capture: bool
) -> dict[str, Any]:
    """Validate a closed non-promotable failure payload before publication."""

    _require(type(require_complete_capture) is bool, "failure capture requirement")
    _require(
        isinstance(value, dict) and set(value) == FAILURE_RAW_FIELDS,
        "failure raw schema drift",
    )
    _require(
        type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and value.get("profile") == PROFILE
        and value.get("finalized") is False
        and value.get("passed") is False
        and value.get("environment") == gate0.ENVIRONMENT
        and value.get("variant") == variant
        and value.get("candidate") == CANDIDATE_BY_VARIANT[variant]
        and value.get("close_failures") == [],
        "failure raw identity drift",
    )
    context = validate_failure_context(value.get("failure_context"), variant=variant)
    failure = value.get("failure")
    failure_type = failure.get("type") if isinstance(failure, dict) else None
    _require(
        isinstance(failure, dict)
        and set(failure) == {"type", "message", "traceback"}
        and isinstance(failure_type, str)
        and failure_type.endswith(
            ("DifferentialIKNumericalError", "DifferentialIKInvariantError")
        )
        and isinstance(failure.get("message"), str)
        and isinstance(failure.get("traceback"), str),
        "failure raw primary exception drift",
    )
    _require(
        type(value.get("policy_step")) is int and value["policy_step"] >= 0,
        "failure raw policy step",
    )
    capture = value.get("controller_abort_capture")
    capture_failure = value.get("controller_abort_capture_failure")
    if require_complete_capture:
        validated_capture = validate_controller_abort_capture(capture, variant=variant)
        _require(
            value.get("stage") == "failed_controller_abort_captured"
            and capture_failure is None
            and validated_capture["failure_exception"] == failure
            and validated_capture["parsed_failure"]["policy_step"]
            == value["policy_step"],
            "complete failure raw transaction drift",
        )
    else:
        _require(
            value.get("stage") == "failed_controller_abort_capture_incomplete"
            and isinstance(capture_failure, dict)
            and set(capture_failure) == {"type", "message", "traceback"}
            and all(
                isinstance(capture_failure.get(field), str)
                for field in ("type", "message", "traceback")
            ),
            "incomplete failure raw transaction drift",
        )
        if capture is not None:
            _require(isinstance(capture, dict), "incomplete failure capture type")
    _require(
        context["initial_candidate"]["gripper_close_arm_interlock"]["enabled"] is True,
        "failure raw controller context was not enabled",
    )
    return dict(value)


def validate_candidate_replay_evidence(
    safety: Any, candidate_report: Any, *, variant: str
) -> dict[str, Any]:
    """Independently bind exact cadence and nominal candidate invariants."""

    report = validate_candidate_report(candidate_report, variant=variant, final=True)
    _require(isinstance(safety, dict), "candidate final safety report")
    counters = safety.get("counters")
    maxima = safety.get("maxima")
    _require(isinstance(counters, dict), "candidate safety counters")
    _require(isinstance(maxima, dict), "candidate safety maxima")
    _require(
        counters.get("apply_calls") == TOTAL_ACTION_COUNT * gate0.DECIMATION
        and counters.get("environment_substeps")
        == TOTAL_ACTION_COUNT * gate0.DECIMATION,
        "candidate arm apply cadence drift",
    )
    for field in ZERO_SAFETY_COUNTERS:
        _require(
            type(counters.get(field)) is int and counters[field] == 0,
            f"candidate safety counter {field}",
        )
    _require(
        type(counters.get("slew_limit_events")) is int
        and counters["slew_limit_events"] >= 1
        and type(counters.get("slew_limited_joints")) is int
        and counters["slew_limited_joints"] >= counters["slew_limit_events"],
        "candidate did not exercise nominal slew limiting",
    )
    _require(safety.get("guard_diagnostics") == [], "candidate guard diagnostics")
    _require(
        safety.get("current_joint_velocity_abort") is None,
        "candidate retained a velocity abort",
    )

    nominal = report["arm_slew_headroom"]["nominal_max_delta_joint_pos_rad"]
    applied = maxima.get("applied_delta_joint_pos_rad")
    _require(
        isinstance(applied, list) and len(applied) == len(nominal) == 7,
        "candidate applied-delta maxima shape",
    )
    for index, (actual, bound) in enumerate(zip(applied, nominal, strict=True)):
        _require(
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(float(actual))
            and 0.0 <= float(actual) <= float(bound) + 1e-6,
            f"candidate nominal applied-delta bound {index}",
        )
    for field in (
        "current_joint_soft_limit_violation_rad",
        "current_physx_hard_limit_violation_rad",
        "post_clamp_target_guard_band_violation_rad",
        "post_clamp_target_soft_limit_violation_rad",
    ):
        vector = maxima.get(field)
        _require(
            isinstance(vector, list)
            and len(vector) == 7
            and all(float(value) == 0.0 for value in vector),
            f"candidate safety maximum {field}",
        )

    gripper = safety.get("gripper_runtime_dynamic")
    _require(isinstance(gripper, dict), "candidate gripper dynamic evidence")
    mimic_compliance = validate_candidate_mimic_compliance(
        safety.get("gripper_runtime_static")
    )
    _require(
        gripper.get("apply_entry_samples") == TOTAL_ACTION_COUNT * gate0.DECIMATION
        and gripper.get("post_policy_step_samples") == TOTAL_ACTION_COUNT
        and gripper.get("nonfinite_samples") == 0
        and gripper.get("dropped_diagnostics") == 0,
        "candidate gripper sample cadence drift",
    )
    driver = gripper.get("driver_target_slew")
    _require(isinstance(driver, dict), "candidate gripper target-slew evidence")
    expected_changes = 1 if variant == "official_lap3b" else 0
    expected_limited = (
        CANDIDATE_CLOSE_LIMITED_APPLIES if variant == "official_lap3b" else 0
    )
    _require(
        driver.get("profile") == CANDIDATE_TARGET_SLEW_PROFILE
        and driver.get("slew_limited_apply_count") == expected_limited
        and driver.get("endpoint_reached_apply_count")
        == TOTAL_ACTION_COUNT * gate0.DECIMATION - expected_limited
        and driver.get("apply_calls") == TOTAL_ACTION_COUNT * gate0.DECIMATION
        and driver.get("live_limit_validation_count")
        == TOTAL_ACTION_COUNT * gate0.DECIMATION
        and driver.get("process_action_calls") == TOTAL_ACTION_COUNT
        and driver.get("initialization_count") == 1
        and driver.get("endpoint_change_count") == expected_changes
        and driver.get("repeated_endpoint_process_count")
        == TOTAL_ACTION_COUNT - 1 - expected_changes,
        "candidate gripper target-slew cadence drift",
    )
    interlock = report["gripper_close_arm_interlock"]
    return {
        "profile": ("polaris_eef_candidate_exact_cadence_anchor_bound_mimic100_1p2_v3"),
        "arm_apply_calls": counters["apply_calls"],
        "gripper_apply_calls": driver["apply_calls"],
        "process_action_calls": driver["process_action_calls"],
        "post_policy_step_samples": gripper["post_policy_step_samples"],
        "target_slew_profile": driver["profile"],
        "target_slew_limited_apply_count": expected_limited,
        "target_slew_endpoint_reached_apply_count": driver[
            "endpoint_reached_apply_count"
        ],
        "mimic_compliance_profile": mimic_compliance["profile"],
        "mimic_compliance_total_write_count": mimic_compliance["total_write_count"],
        "mimic_compliance_post_reset_composed_usd_readback_count": len(
            mimic_compliance["followers"]
        ),
        "mimic_compliance_frequency_timestep_product": mimic_compliance[
            "frequency_timestep_product"
        ],
        "slew_limit_events": counters["slew_limit_events"],
        "anchor_capture_count": interlock["anchor_capture_count"],
        "anchor_target_apply_count": interlock["anchor_target_apply_count"],
        "anchor_completion_count": interlock["anchor_completion_count"],
        "anchor_open_cancel_count": interlock["anchor_open_cancel_count"],
        "last_activation_apply_index": interlock["last_activation_apply_index"],
        "anchor_digest_verified": True,
        "dls_fallbacks": 0,
        "abort_count": 0,
        "nominal_applied_delta_bound_passed": True,
        "guard_diagnostics_empty": True,
    }


def validate_velocity_headroom(safety: Any) -> dict[str, Any]:
    _require(isinstance(safety, dict), "final safety report")
    maxima = safety.get("maxima")
    limits = safety.get("joint_velocity_limits_rad_s")
    _require(isinstance(maxima, dict) and isinstance(limits, list), "velocity evidence")
    velocities = maxima.get("abs_joint_vel_rad_s")
    _require(
        isinstance(velocities, list) and len(velocities) == len(limits) == 7,
        "velocity evidence vector shape",
    )
    ratios: list[float] = []
    for index, (velocity, limit) in enumerate(zip(velocities, limits, strict=True)):
        _require(
            isinstance(velocity, (int, float))
            and isinstance(limit, (int, float))
            and not isinstance(velocity, bool)
            and not isinstance(limit, bool)
            and math.isfinite(float(velocity))
            and math.isfinite(float(limit))
            and 0.0 < float(limit)
            and 0.0 <= float(velocity) <= float(limit) + 1e-5,
            f"live arm velocity bound {index}",
        )
        ratios.append(float(velocity) / float(limit))
    return {
        "profile": "max_observed_abs_velocity_over_live_physical_limit_v1",
        "ratios": ratios,
        "maximum_ratio": max(ratios),
        "passed": True,
    }


def _validate_output_namespace(
    path: Path, *, variant: str, lifecycle: dict[str, Any]
) -> None:
    resolved = path.resolve()
    _require(
        resolved.name == f"candidate-{variant}.raw.json",
        "candidate raw result filename drift",
    )
    _require(
        resolved.parent.name == f"launch_{lifecycle['launch_id']}"
        and resolved.parent.parent.name == f"job_{lifecycle['job_id']}"
        and resolved.parent.parent.parent.name == variant,
        "candidate raw result namespace drift",
    )


def validate_container_argument(
    image: str, *, size_bytes: int, sha256: str
) -> dict[str, Any]:
    _require(
        isinstance(image, str) and image.startswith("/") and "\x00" not in image,
        "candidate container image path",
    )
    _require(
        type(size_bytes) is int and size_bytes > 0,
        "candidate container image size",
    )
    _require(
        isinstance(sha256, str)
        and len(sha256) == 64
        and all(character in "0123456789abcdef" for character in sha256),
        "candidate container image digest",
    )
    return {
        "profile": "host_regular_nonsymlink_sha256_verified_before_pyxis_v1",
        "path": image,
        "size_bytes": size_bytes,
        "sha256": sha256,
    }


def _build_controller_abort_capture(
    *,
    error: BaseException,
    policy_step: int,
    env: Any,
    boundary: Any,
    finger_term: Any,
    candidate_reporter: Any,
) -> dict[str, Any]:
    """Build all controller/gripper evidence before teardown or validation."""

    failure_exception = gate0._exception_evidence(error)  # noqa: SLF001
    parsed_failure = gate0.parse_failure_exception(failure_exception["message"])
    _require(
        parsed_failure["policy_step"] == policy_step,
        "controller-abort loop/exception policy-step drift",
    )
    finger_term.finalize_gate0_failure()
    arm_failure = boundary._capture_failure_runtime_evidence(  # noqa: SLF001
        env,
        policy_step=policy_step,
    )
    gripper_tail = finger_term.gate0_gripper_tail()
    active_safety = arm_failure["ik_safety"]
    active_candidate = candidate_reporter()
    target_reporter = getattr(finger_term, "gripper_target_slew_dynamic_report", None)
    _require(callable(target_reporter), "controller-abort target-slew reporter")
    payload = {
        "profile": CONTROLLER_ABORT_CAPTURE_PROFILE,
        "failure_exception": failure_exception,
        "parsed_failure": parsed_failure,
        "arm_failure_runtime_evidence": arm_failure,
        "all_six_gripper_tail": gripper_tail,
        "active_safety": active_safety,
        "active_candidate": active_candidate,
        "active_target_slew": target_reporter(),
    }
    return payload


def _retain_then_validate_controller_abort_capture(
    state: dict[str, Any],
    payload: dict[str, Any],
    *,
    variant: str,
) -> dict[str, Any]:
    """Retain built evidence even if its secondary validation fails."""

    state["controller_abort_capture"] = payload
    return validate_controller_abort_capture(payload, variant=variant)


def _run_live(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    import gymnasium as gym  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: PLC0415

    import polaris.environments  # noqa: F401, PLC0415
    from polaris.eef_gripper_runtime import (  # noqa: PLC0415
        configure_eef_gripper_mimic_compliance_spawn_overlay,
        install_eef_gripper_runtime,
        record_eef_gripper_post_policy_step,
    )
    from polaris.eef_runtime_contract import (  # noqa: PLC0415
        begin_eef_safety_episode,
        configure_ego_lap_environment_timeout,
        validate_eef_runtime_frame,
        validate_eef_runtime_safety,
        validate_ego_lap_runtime_protocol,
    )
    from polaris.environments.droid_cfg import (  # noqa: PLC0415
        EefBinaryJointPositionTargetSlewAction,
        EgoLapEefPoseActionCfg,
    )
    from polaris.environments.robot_cfg import (  # noqa: PLC0415
        configure_eef_pose_joint_safety,
    )
    from polaris.robust_differential_ik import (  # noqa: PLC0415
        DifferentialIKNumericalError,
    )
    from polaris.utils import load_eval_initial_conditions  # noqa: PLC0415

    state["stage"] = "bind_repository"
    container_image = validate_container_argument(
        args.container_image,
        size_bytes=args.expected_container_size_bytes,
        sha256=args.expected_container_sha256,
    )
    repository = gate0._repository_provenance(args.expected_polaris_commit)
    production_eval = gate0.validate_production_reset_source()
    lifecycle = gate0._slurm_lifecycle(args.launch_id)
    _validate_output_namespace(
        args.output_json, variant=args.variant, lifecycle=lifecycle
    )
    state["stage"] = "load_fixture"
    fixture_identity, fixture_payload, actions = gate0.load_replay_fixture(args.variant)
    _require(len(actions) == FIXTURE_ACTION_COUNT, "fixture action count drift")
    boundary, helper_identity = gate0._load_boundary_helper()

    state["stage"] = "build_environment"
    env_cfg = parse_env_cfg(
        gate0.ENVIRONMENT,
        device=args.device,
        num_envs=1,
        use_fabric=True,
    )
    configure_ego_lap_environment_timeout(env_cfg)
    env_cfg.actions = EgoLapEefPoseActionCfg()
    env_cfg.actions.arm.enable_failure_substep_trace = True
    env_cfg.actions.arm.enable_wrist_energy_brake = False
    env_cfg.actions.arm.enable_arm_slew_headroom = True
    env_cfg.actions.arm.enable_gripper_close_arm_interlock = True
    env_cfg.actions.finger_joint.enable_target_slew_rate_0p25_candidate = True
    tracing_class = gate0._make_tracing_gripper_class(
        EefBinaryJointPositionTargetSlewAction
    )
    env_cfg.actions.finger_joint.class_type = tracing_class
    configure_eef_pose_joint_safety(
        env_cfg.scene.robot,
        physx_cfg=env_cfg.sim.physx,
        enable_gripper_velocity_limit=True,
    )
    configure_eef_gripper_mimic_compliance_spawn_overlay(
        env_cfg.scene.robot.spawn,
        target_slew_profile=CANDIDATE_TARGET_SLEW_PROFILE,
    )
    robot_usd_path = Path(env_cfg.scene.robot.spawn.usd_path)
    env = gym.make(gate0.ENVIRONMENT, cfg=env_cfg)
    state["env"] = env
    runtime_protocol = validate_ego_lap_runtime_protocol(env)

    state["stage"] = "validate_assets"
    assets = gate0._capture_assets(
        boundary,
        scene_path=Path(env.unwrapped.usd_file),
        robot_usd_path=robot_usd_path,
    )
    _, initial_conditions = load_eval_initial_conditions(
        usd=env.unwrapped.usd_file,
        rollouts=1,
    )
    _require(
        isinstance(initial_conditions, list)
        and len(initial_conditions) == 1
        and isinstance(initial_conditions[gate0.INITIAL_CONDITION_INDEX], dict),
        "FoodBussing IC0 loader drift",
    )

    state["stage"] = "reset_ic0"
    observation, _ = env.reset(
        object_positions=initial_conditions[gate0.INITIAL_CONDITION_INDEX]
    )
    gripper_runtime_contract = install_eef_gripper_runtime(
        env, robot_usd_path=robot_usd_path
    )
    runtime_frame = validate_eef_runtime_frame(env, observation)
    begin_eef_safety_episode(env, 0)
    initial_safety = validate_eef_runtime_safety(
        env,
        require_gripper_runtime=True,
        expected_gripper_target_slew_profile=CANDIDATE_TARGET_SLEW_PROFILE,
    )
    terms = env.unwrapped.action_manager._terms
    _require(list(terms) == ["arm", "finger_joint"], "live action order drift")
    arm_term = terms["arm"]
    finger_term = terms["finger_joint"]
    _require(type(finger_term) is tracing_class, "tracing gripper class not installed")
    reporter = getattr(arm_term, "controller_repair_candidate_report", None)
    _require(callable(reporter), "controller candidate reporter is absent")
    initial_candidate = validate_candidate_report(
        reporter(), variant=args.variant, final=False
    )

    failure_context = {
        "lifecycle": lifecycle,
        "repository": repository,
        "container_image": container_image,
        "production_eval": production_eval,
        "fixture": {
            **fixture_identity,
            "source_trace_sha256": fixture_payload["source"]["trace_sha256"],
            "action_float32_sha256": fixture_payload["action_encoding"][
                "uncompressed_sha256"
            ],
            "fixture_action_count": len(actions),
        },
        "action_plan": {
            "profile": "exact_fixture_then_repeat_final_recorded_action_v1",
            "fixture_action_count": FIXTURE_ACTION_COUNT,
            "post_fixture_repeat_count": POST_FIXTURE_REPEAT_COUNT,
            "total_action_count": TOTAL_ACTION_COUNT,
        },
        "boundary_helper": helper_identity,
        "assets": assets,
        "runtime_protocol": runtime_protocol,
        "runtime_frame": runtime_frame,
        "gripper_runtime_contract": gripper_runtime_contract,
        "initial_safety": initial_safety,
        "initial_candidate": initial_candidate,
    }
    state["failure_context"] = validate_failure_context(
        failure_context, variant=args.variant
    )

    replay_actions = list(actions) + [list(actions[-1])] * POST_FIXTURE_REPEAT_COUNT
    state["stage"] = "replay_actions"
    for step, action_values in enumerate(replay_actions):
        state["policy_step"] = step
        finger_term.begin_gate0_policy_step(step)
        action = torch.tensor(
            action_values, dtype=torch.float32, device=env.device
        ).reshape(1, -1)
        try:
            observation, _, terminated, truncated, _ = env.step(action, expensive=True)
        except DifferentialIKNumericalError as error:
            # Preserve the primary controller exception before any secondary
            # diagnostic operation can fail. The outer failure transaction
            # writes this object without a ready marker or promotion artifact.
            state["controller_abort_original_failure"] = gate0._exception_evidence(
                error
            )  # noqa: SLF001
            state["stage"] = "capture_controller_abort"
            try:
                capture = _build_controller_abort_capture(
                    error=error,
                    policy_step=step,
                    env=env,
                    boundary=boundary,
                    finger_term=finger_term,
                    candidate_reporter=reporter,
                )
                _retain_then_validate_controller_abort_capture(
                    state,
                    capture,
                    variant=args.variant,
                )
            except BaseException as capture_error:
                state["controller_abort_capture_failure"] = gate0._exception_evidence(
                    capture_error
                )  # noqa: SLF001
            raise
        _require(not bool(terminated[0]), f"candidate replay terminated at step {step}")
        _require(not bool(truncated[0]), f"candidate replay truncated at step {step}")
        record_eef_gripper_post_policy_step(env)

    _require(len(replay_actions) == TOTAL_ACTION_COUNT, "candidate replay action count")
    final_safety = validate_eef_runtime_safety(
        env,
        require_gripper_runtime=True,
        expected_gripper_target_slew_profile=CANDIDATE_TARGET_SLEW_PROFILE,
    )
    final_candidate = validate_candidate_report(
        reporter(), variant=args.variant, final=True
    )
    candidate_replay_validation = validate_candidate_replay_evidence(
        final_safety, final_candidate, variant=args.variant
    )
    velocity_headroom = validate_velocity_headroom(final_safety)
    state["policy_step"] = None
    return {
        **failure_context,
        "final_safety": final_safety,
        "final_candidate": final_candidate,
        "candidate_replay_validation": candidate_replay_validation,
        "velocity_headroom": velocity_headroom,
        "outcome": {
            "status": "candidate_replay_completed_without_controller_abort",
            "actions_completed": TOTAL_ACTION_COUNT,
            "original_failure_step_crossed": gate0.EXPECTED_FIXTURES[args.variant][
                "failure"
            ]["policy_step"],
            "post_fixture_release_probe_completed": True,
        },
    }


def _parse_args() -> tuple[argparse.Namespace, Any]:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=sorted(CANDIDATE_BY_VARIANT), required=True
    )
    parser.add_argument("--expected-polaris-commit", required=True)
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--expected-container-size-bytes", type=int, required=True)
    parser.add_argument("--expected-container-sha256", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.headless = True
    return args, AppLauncher


def main() -> int:
    args, app_launcher_type = _parse_args()
    state: dict[str, Any] = {
        "stage": "launch_simulation_app",
        "policy_step": None,
        "env": None,
        "controller_abort_original_failure": None,
        "controller_abort_capture": None,
        "controller_abort_capture_failure": None,
        "failure_context": None,
    }
    simulation_app = None
    close_failures: list[dict[str, str]] = []
    try:
        app_launcher = app_launcher_type(args)
        simulation_app = app_launcher.app
        evidence = _run_live(args, state)
        env = state.get("env")
        if env is not None:
            state["stage"] = "close_environment"
            try:
                env.close()
            except BaseException as error:
                close_failures.append(gate0._exception_evidence(error))
                raise
        state["stage"] = "simulation_app_close_pending"
        payload = {
            "schema_version": 1,
            "profile": PROFILE,
            "finalized": False,
            "passed": True,
            "stage": state["stage"],
            "environment": gate0.ENVIRONMENT,
            "variant": args.variant,
            "candidate": CANDIDATE_BY_VARIANT[args.variant],
            **evidence,
            "close_failures": close_failures,
        }
        identity = gate0._atomic_write_immutable(args.output_json, payload)
        marker_path = args.output_json.resolve().with_name(
            args.output_json.name + ".ready.json"
        )
        gate0._atomic_write_immutable(
            marker_path,
            {
                "schema_version": 1,
                "profile": PROFILE,
                "stage": "simulation_app_close_pending",
                "raw_result": identity,
            },
        )
        print(
            "POLARIS_CONTROLLER_CANDIDATE_RAW="
            f"{identity['path']};size={identity['size_bytes']};"
            f"sha256={identity['sha256']};mode={identity['mode']}",
            flush=True,
        )
        print(f"POLARIS_CONTROLLER_CANDIDATE_READY={marker_path}", flush=True)
        simulation_app.close()
        return 0
    except BaseException as error:
        traceback.print_exception(
            type(error), error, error.__traceback__, file=sys.stderr
        )
        controller_abort_capture = state.get("controller_abort_capture")
        controller_abort_capture_failure = state.get("controller_abort_capture_failure")
        original_failure = state.get("controller_abort_original_failure")
        if controller_abort_capture is not None:
            try:
                validate_controller_abort_capture(
                    controller_abort_capture,
                    variant=getattr(args, "variant", None),
                )
            except BaseException as capture_validation_error:
                controller_abort_capture_failure = gate0._exception_evidence(  # noqa: SLF001
                    capture_validation_error
                )
        failure_payload = {
            "schema_version": 1,
            "profile": PROFILE,
            "finalized": False,
            "passed": False,
            "stage": (
                "failed_controller_abort_captured"
                if controller_abort_capture is not None
                and controller_abort_capture_failure is None
                else "failed_controller_abort_capture_incomplete"
                if original_failure is not None
                else "failed"
            ),
            "environment": gate0.ENVIRONMENT,
            "variant": getattr(args, "variant", None),
            "candidate": CANDIDATE_BY_VARIANT.get(getattr(args, "variant", None)),
            "policy_step": state.get("policy_step"),
            "failure_context": state.get("failure_context"),
            "failure": original_failure or gate0._exception_evidence(error),
            "controller_abort_capture": controller_abort_capture,
            "controller_abort_capture_failure": controller_abort_capture_failure,
            "close_failures": close_failures,
        }
        try:
            if original_failure is not None:
                try:
                    validate_failure_payload(
                        failure_payload,
                        variant=args.variant,
                        require_complete_capture=(
                            controller_abort_capture is not None
                            and controller_abort_capture_failure is None
                        ),
                    )
                except BaseException as transaction_validation_error:
                    failure_payload["stage"] = (
                        "failed_controller_abort_capture_incomplete"
                    )
                    failure_payload["controller_abort_capture_failure"] = (
                        gate0._exception_evidence(  # noqa: SLF001
                            transaction_validation_error
                        )
                    )
                    validate_failure_payload(
                        failure_payload,
                        variant=args.variant,
                        require_complete_capture=False,
                    )
            identity = gate0._atomic_write_immutable(args.output_json, failure_payload)
            print(
                "POLARIS_CONTROLLER_CANDIDATE_FAILURE_RAW="
                f"{identity['path']};size={identity['size_bytes']};"
                f"sha256={identity['sha256']};mode={identity['mode']}",
                flush=True,
            )
        except BaseException as persistence_error:
            traceback.print_exception(
                type(persistence_error),
                persistence_error,
                persistence_error.__traceback__,
                file=sys.stderr,
            )
        if simulation_app is not None:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
