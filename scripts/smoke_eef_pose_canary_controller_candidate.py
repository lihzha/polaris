#!/usr/bin/env python3
"""Replay exact failed canary actions through isolated controller candidates.

This is a model-free promotion gate. Both variants enable the 0.95 nominal arm
slew bound. The official LAP-3B variant additionally enables the bounded
gripper-close/arm interlock. After the 120 content-pinned actions, two repeats
of the final recorded action prove that the 48-substep interlock releases and
ordinary arm motion resumes without crossing a live physical velocity limit.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import smoke_eef_pose_canary_trace_replay as gate0


PROFILE = "polaris_eef_canary_controller_candidate_replay_v1"
FIXTURE_ACTION_COUNT = 120
POST_FIXTURE_REPEAT_COUNT = 2
TOTAL_ACTION_COUNT = FIXTURE_ACTION_COUNT + POST_FIXTURE_REPEAT_COUNT
CANDIDATE_BY_VARIANT = {
    "official_lap3b": "arm_slew_0p95_plus_gripper_close_interlock48_v1",
    "reasoning_43075": "arm_slew_0p95_only_v1",
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


class CandidateReplayValidationError(ValueError):
    """The candidate replay or its evidence violated the promotion contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateReplayValidationError(message)


def validate_candidate_report(
    report: Any, *, variant: str, final: bool
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

    enabled = variant == "official_lap3b"
    _require(
        interlock.get("enabled") is enabled
        and interlock.get("profile")
        == "eef_gripper_close_hold_arm_48_physics_substeps_v1"
        and type(interlock.get("configured_substeps")) is int
        and interlock.get("configured_substeps") == 48,
        "close-interlock candidate identity drift",
    )
    for name in (
        "remaining_substeps",
        "observed_endpoint_change_count",
        "activation_count",
        "active_apply_count",
        "released_apply_count",
    ):
        _require(
            type(interlock.get(name)) is int and interlock[name] >= 0,
            f"close-interlock {name} type/range drift",
        )
    _require(
        type(interlock.get("endpoint_observed")) is bool,
        "close-interlock endpoint_observed type drift",
    )
    active_vector = interlock.get("max_abs_active_delta_joint_pos_rad")
    released_vector = interlock.get("max_abs_released_delta_joint_pos_rad")
    for field, vector in (
        ("active", active_vector),
        ("released", released_vector),
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
    if not final:
        _require(
            interlock.get("remaining_substeps") == 0
            and interlock.get("observed_endpoint_change_count") == 0
            and interlock.get("endpoint_observed") is False
            and interlock.get("activation_count") == 0
            and interlock.get("active_apply_count") == 0
            and interlock.get("released_apply_count") == 0
            and all(float(value) == 0.0 for value in active_vector)
            and all(float(value) == 0.0 for value in released_vector),
            "initial close-interlock state is not empty",
        )
    elif enabled:
        _require(
            interlock.get("remaining_substeps") == 0
            and interlock.get("observed_endpoint_change_count") == 1
            and interlock.get("endpoint_observed") is True
            and interlock.get("activation_count") == 1
            and interlock.get("active_apply_count") == 48
            and interlock.get("released_apply_count") == 8
            and all(float(value) == 0.0 for value in active_vector)
            and any(float(value) > 0.0 for value in released_vector),
            "official close interlock did not activate, release, and resume",
        )
    else:
        _require(
            interlock.get("remaining_substeps") == 0
            and interlock.get("observed_endpoint_change_count") == 0
            and interlock.get("endpoint_observed") is False
            and interlock.get("activation_count") == 0
            and interlock.get("active_apply_count") == 0
            and interlock.get("released_apply_count") == 0
            and all(float(value) == 0.0 for value in active_vector)
            and all(float(value) == 0.0 for value in released_vector),
            "reasoning replay unexpectedly used the close interlock",
        )
    return dict(report)


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
    _require(
        driver.get("apply_calls") == TOTAL_ACTION_COUNT * gate0.DECIMATION
        and driver.get("live_limit_validation_count")
        == TOTAL_ACTION_COUNT * gate0.DECIMATION
        and driver.get("process_action_calls") == TOTAL_ACTION_COUNT
        and driver.get("initialization_count") == 1
        and driver.get("endpoint_change_count") == expected_changes
        and driver.get("repeated_endpoint_process_count")
        == TOTAL_ACTION_COUNT - 1 - expected_changes,
        "candidate gripper target-slew cadence drift",
    )
    return {
        "profile": "polaris_eef_candidate_exact_cadence_and_nominal_bound_v1",
        "arm_apply_calls": counters["apply_calls"],
        "gripper_apply_calls": driver["apply_calls"],
        "process_action_calls": driver["process_action_calls"],
        "post_policy_step_samples": gripper["post_policy_step_samples"],
        "slew_limit_events": counters["slew_limit_events"],
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


def _run_live(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    import gymnasium as gym  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: PLC0415

    import polaris.environments  # noqa: F401, PLC0415
    from polaris.eef_gripper_runtime import (  # noqa: PLC0415
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
    env_cfg.actions.arm.enable_gripper_close_arm_interlock = (
        args.variant == "official_lap3b"
    )
    tracing_class = gate0._make_tracing_gripper_class(
        EefBinaryJointPositionTargetSlewAction
    )
    env_cfg.actions.finger_joint.class_type = tracing_class
    configure_eef_pose_joint_safety(
        env_cfg.scene.robot,
        physx_cfg=env_cfg.sim.physx,
        enable_gripper_velocity_limit=True,
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
    initial_safety = validate_eef_runtime_safety(env, require_gripper_runtime=True)
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

    replay_actions = list(actions) + [list(actions[-1])] * POST_FIXTURE_REPEAT_COUNT
    state["stage"] = "replay_actions"
    for step, action_values in enumerate(replay_actions):
        state["policy_step"] = step
        finger_term.begin_gate0_policy_step(step)
        action = torch.tensor(
            action_values, dtype=torch.float32, device=env.device
        ).reshape(1, -1)
        observation, _, terminated, truncated, _ = env.step(action, expensive=True)
        _require(not bool(terminated[0]), f"candidate replay terminated at step {step}")
        _require(not bool(truncated[0]), f"candidate replay truncated at step {step}")
        record_eef_gripper_post_policy_step(env)

    _require(len(replay_actions) == TOTAL_ACTION_COUNT, "candidate replay action count")
    final_safety = validate_eef_runtime_safety(env, require_gripper_runtime=True)
    final_candidate = validate_candidate_report(
        reporter(), variant=args.variant, final=True
    )
    candidate_replay_validation = validate_candidate_replay_evidence(
        final_safety, final_candidate, variant=args.variant
    )
    velocity_headroom = validate_velocity_headroom(final_safety)
    state["policy_step"] = None
    return {
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
        failure_payload = {
            "schema_version": 1,
            "profile": PROFILE,
            "finalized": False,
            "passed": False,
            "stage": "failed",
            "environment": gate0.ENVIRONMENT,
            "variant": getattr(args, "variant", None),
            "candidate": CANDIDATE_BY_VARIANT.get(getattr(args, "variant", None)),
            "policy_step": state.get("policy_step"),
            "failure": gate0._exception_evidence(error),
            "close_failures": close_failures,
        }
        try:
            gate0._atomic_write_immutable(args.output_json, failure_payload)
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
