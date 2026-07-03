"""Pure smoke-plan and result validation for native DROID velocity control."""

from __future__ import annotations

import copy
import math
from typing import Any

from polaris.pi05_droid_jointvelocity_contract import (
    PANDA_ARM_JOINT_NAMES,
    PANDA_ARM_VELOCITY_LIMITS,
    PI05_DROID_JOINTVELOCITY_PROFILE,
)


SMOKE_PROFILE = "pi05_droid_native_jointvelocity_controller_smoke_v1"


def build_joint_velocity_smoke_cases(
    command_magnitude: float = 0.25,
) -> list[dict[str, Any]]:
    """Return deterministic hold, signed-axis, gripper, and limit cases."""

    if not math.isfinite(command_magnitude) or not 0.0 < command_magnitude <= 1.0:
        raise ValueError("command_magnitude must be finite and in (0, 1]")
    cases: list[dict[str, Any]] = [
        {"label": "hold", "action": [0.0] * 8, "kind": "hold"}
    ]
    for joint_index, joint_name in enumerate(PANDA_ARM_JOINT_NAMES):
        for sign_name, sign in (("positive", 1.0), ("negative", -1.0)):
            action = [0.0] * 8
            action[joint_index] = sign * command_magnitude
            cases.append(
                {
                    "label": f"{joint_name}_{sign_name}",
                    "action": action,
                    "kind": "signed_joint",
                    "joint_index": joint_index,
                    "sign": int(sign),
                }
            )
    cases.extend(
        [
            {
                "label": "gripper_open",
                "action": [0.0] * 8,
                "kind": "gripper",
                "expected_finger_target": 0.0,
            },
            {
                "label": "gripper_closed",
                "action": [0.0] * 7 + [1.0],
                "kind": "gripper",
                "expected_finger_target": math.pi / 4.0,
            },
            {
                "label": "positive_action_limit",
                "action": [1.0] * 7 + [0.0],
                "kind": "limit",
            },
            {
                "label": "negative_action_limit",
                "action": [-1.0] * 7 + [0.0],
                "kind": "limit",
            },
        ]
    )
    return cases


def _finite_vector(value: Any, length: int, *, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field} must have {length} values")
    if not all(isinstance(item, int | float) and math.isfinite(item) for item in value):
        raise ValueError(f"{field} must contain only finite numbers")
    return [float(item) for item in value]


def validate_joint_velocity_smoke(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate one standalone Isaac capture without trusting its pass boolean."""

    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Velocity smoke must be a schema-1 object")
    if payload.get("smoke_profile") != SMOKE_PROFILE:
        raise ValueError("Velocity smoke profile mismatch")
    if payload.get("controller_profile") != PI05_DROID_JOINTVELOCITY_PROFILE:
        raise ValueError("Velocity controller profile mismatch")
    runtime = payload.get("runtime_contract")
    if not isinstance(runtime, dict) or runtime.get("status") != "pass":
        raise ValueError("Velocity smoke is missing a passing runtime contract")
    if runtime.get("profile") != PI05_DROID_JOINTVELOCITY_PROFILE:
        raise ValueError("Velocity runtime profile mismatch")

    expected_cases = build_joint_velocity_smoke_cases(payload.get("command_magnitude"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != len(expected_cases):
        raise ValueError("Velocity smoke case set mismatch")

    velocity_limits = list(PANDA_ARM_VELOCITY_LIMITS)
    for index, (expected_case, case) in enumerate(
        zip(expected_cases, cases, strict=True)
    ):
        if not isinstance(case, dict):
            raise ValueError(f"Velocity smoke case {index} must be an object")
        for key in ("label", "kind", "action"):
            if case.get(key) != expected_case[key]:
                raise ValueError(f"Velocity smoke case {index} {key} mismatch")
        if case.get("terminated") is not False or case.get("truncated") is not False:
            raise ValueError(f"Velocity smoke case {index} ended the episode")
        action = _finite_vector(case["action"], 8, field=f"case {index} action")
        q_before = _finite_vector(
            case.get("joint_position_before"), 7, field=f"case {index} q before"
        )
        q_after = _finite_vector(
            case.get("joint_position_after"), 7, field=f"case {index} q after"
        )
        _finite_vector(
            case.get("joint_velocity_before"), 7, field=f"case {index} dq before"
        )
        dq_after = _finite_vector(
            case.get("joint_velocity_after"), 7, field=f"case {index} dq after"
        )
        processed = _finite_vector(
            case.get("processed_joint_velocity"),
            7,
            field=f"case {index} processed velocity",
        )
        target = _finite_vector(
            case.get("articulation_joint_velocity_target"),
            7,
            field=f"case {index} velocity target",
        )
        expected_velocity = action[:7]
        if processed != expected_velocity or target != expected_velocity:
            raise ValueError(f"Velocity smoke case {index} command path mismatch")
        if any(
            abs(value) > limit + 1e-4
            for value, limit in zip(dq_after, velocity_limits, strict=True)
        ):
            raise ValueError(f"Velocity smoke case {index} exceeded a velocity limit")
        soft_limits = case.get("soft_joint_position_limits")
        if (
            not isinstance(soft_limits, list)
            or len(soft_limits) != 7
            or any(
                not isinstance(limit, list)
                or len(limit) != 2
                or not all(
                    isinstance(value, int | float) and math.isfinite(value)
                    for value in limit
                )
                for limit in soft_limits
            )
        ):
            raise ValueError(f"Velocity smoke case {index} has invalid position limits")
        if any(
            value < float(limit[0]) - 1e-5 or value > float(limit[1]) + 1e-5
            for value, limit in zip(q_after, soft_limits, strict=True)
        ):
            raise ValueError(f"Velocity smoke case {index} exceeded a position limit")

        if expected_case["kind"] == "signed_joint":
            joint_index = expected_case["joint_index"]
            sign = expected_case["sign"]
            displacement = q_after[joint_index] - q_before[joint_index]
            if sign * displacement <= 1e-7:
                raise ValueError(
                    f"Velocity smoke case {index} did not move in the commanded direction"
                )
            if sign * dq_after[joint_index] <= 1e-6:
                raise ValueError(
                    f"Velocity smoke case {index} has wrong signed measured velocity"
                )
        if expected_case["kind"] == "gripper":
            finger_target = case.get("finger_position_target")
            if not isinstance(finger_target, int | float) or not math.isclose(
                float(finger_target),
                expected_case["expected_finger_target"],
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError(f"Velocity smoke case {index} gripper target mismatch")

    reset = payload.get("reset_probe")
    if not isinstance(reset, dict):
        raise ValueError("Velocity smoke is missing its reset probe")
    default_q = _finite_vector(
        reset.get("default_joint_position"), 7, field="reset default q"
    )
    reset_q = _finite_vector(reset.get("joint_position"), 7, field="reset q")
    reset_dq = _finite_vector(reset.get("joint_velocity"), 7, field="reset dq")
    reset_target = _finite_vector(
        reset.get("joint_velocity_target"), 7, field="reset target"
    )
    if any(
        abs(actual - expected) > 2e-3
        for actual, expected in zip(reset_q, default_q, strict=True)
    ):
        raise ValueError("Velocity smoke reset did not restore default joint positions")
    if max(abs(value) for value in reset_dq) > 2e-2:
        raise ValueError("Velocity smoke reset left excessive joint velocity")
    if reset_target != [0.0] * 7:
        raise ValueError("Velocity smoke reset did not install a zero velocity target")

    validated = copy.deepcopy(payload)
    validated["status"] = "pass"
    validated["case_count"] = len(cases)
    return validated
