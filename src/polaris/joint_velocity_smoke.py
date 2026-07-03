"""Pure smoke-plan and result validation for native DROID velocity control."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any

from polaris.pi05_droid_jointvelocity_contract import (
    NATIVE_GRIPPER_DRIVE_PROFILE,
    NATIVE_GRIPPER_MEASURED_VELOCITY_TOLERANCE,
    NATIVE_GRIPPER_PRECONDITION_POSITION_TOLERANCE,
    NATIVE_GRIPPER_PRECONDITION_STEPS,
    NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S,
    PANDA_ARM_JOINT_NAMES,
    PANDA_ARM_VELOCITY_LIMITS,
    PI05_DROID_JOINTVELOCITY_PROFILE,
)
from polaris.joint_velocity_runtime import validate_joint_velocity_runtime_report


SMOKE_PROFILE = "pi05_droid_native_jointvelocity_controller_smoke_v2"


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
                "precondition_finger_target": math.pi / 4.0,
                "expected_finger_target": 0.0,
                "expected_motion_sign": -1,
            },
            {
                "label": "gripper_closed",
                "action": [0.0] * 7 + [1.0],
                "kind": "gripper",
                "precondition_finger_target": 0.0,
                "expected_finger_target": math.pi / 4.0,
                "expected_motion_sign": 1,
            },
            {
                "label": "gripper_boundary_0p5",
                "action": [0.0] * 7 + [0.5],
                "kind": "gripper",
                "precondition_finger_target": math.pi / 4.0,
                "expected_finger_target": 0.0,
                "expected_motion_sign": -1,
                "threshold_boundary": 0.5,
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
    if not all(type(item) in (int, float) and math.isfinite(item) for item in value):
        raise ValueError(f"{field} must contain only finite numbers")
    return [float(item) for item in value]


def validate_joint_velocity_smoke(
    payload: dict[str, Any], *, require_parent_completion: bool = True
) -> dict[str, Any]:
    """Validate one standalone Isaac capture without trusting its pass boolean."""

    if (
        not isinstance(payload, dict)
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
    ):
        raise ValueError("Velocity smoke must be a schema-1 object")
    required_keys = {
        "schema_version",
        "smoke_profile",
        "controller_profile",
        "environment",
        "command_magnitude",
        "settle_steps",
        "expected_gripper_drive_profile",
        "gripper_precondition_steps",
        "runtime_contract",
        "cases",
        "reset_probe",
        "lifecycle",
    }
    if require_parent_completion:
        required_keys.add("completion")
    derived_keys = {"status", "case_count"}
    present_derived = set(payload) & derived_keys
    if present_derived not in (set(), derived_keys):
        raise ValueError("Velocity smoke derived fields must appear together")
    if set(payload) != required_keys | present_derived:
        raise ValueError("Velocity smoke schema mismatch")
    if payload.get("smoke_profile") != SMOKE_PROFILE:
        raise ValueError("Velocity smoke profile mismatch")
    if payload.get("controller_profile") != PI05_DROID_JOINTVELOCITY_PROFILE:
        raise ValueError("Velocity controller profile mismatch")
    if payload.get("environment") != "DROID-FoodBussing":
        raise ValueError("Velocity smoke environment mismatch")
    if (
        type(payload.get("command_magnitude")) is not float
        or payload.get("command_magnitude") != 0.25
    ):
        raise ValueError("Velocity smoke command magnitude mismatch")
    if type(payload.get("settle_steps")) is not int or payload.get("settle_steps") != 5:
        raise ValueError("Velocity smoke settle-step contract mismatch")
    if payload.get("expected_gripper_drive_profile") != NATIVE_GRIPPER_DRIVE_PROFILE:
        raise ValueError("Velocity smoke expected gripper drive profile mismatch")
    if (
        type(payload.get("gripper_precondition_steps")) is not int
        or payload.get("gripper_precondition_steps")
        != NATIVE_GRIPPER_PRECONDITION_STEPS
    ):
        raise ValueError("Velocity smoke gripper precondition-step mismatch")
    validate_joint_velocity_runtime_report(payload.get("runtime_contract"))
    if (
        payload["runtime_contract"]["gripper"]["drive"]["profile"]
        != payload["expected_gripper_drive_profile"]
    ):
        raise ValueError("Velocity smoke runtime gripper profile mismatch")
    lifecycle = payload.get("lifecycle")
    if require_parent_completion:
        if lifecycle != {
            "env_close": "complete",
            "simulation_app_close": "invoked_then_child_exited_zero",
            "capture_stage": "stdlib_parent_after_kit_child_exit",
        }:
            raise ValueError("Velocity smoke lifecycle is not close-complete")
        completion = payload.get("completion")
        if not isinstance(completion, dict) or set(completion) != {
            "child_exit_code",
            "publication_stage",
            "child_capture_sha256",
            "child_capture_size",
            "child_capture_mode",
            "child_capture_path",
            "child_ready_marker_sha256",
            "child_ready_marker_size",
            "child_ready_marker_mode",
            "child_ready_marker_path",
        }:
            raise ValueError("Velocity smoke lacks parent completion evidence")
        if (
            completion["child_exit_code"] != 0
            or type(completion["child_exit_code"]) is not int
            or completion["publication_stage"] != "stdlib_parent_after_child_exit"
            or not isinstance(completion["child_capture_sha256"], str)
            or len(completion["child_capture_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in completion["child_capture_sha256"]
            )
            or type(completion["child_capture_size"]) is not int
            or completion["child_capture_size"] <= 0
            or completion["child_capture_mode"] != "0444"
            or not isinstance(completion["child_capture_path"], str)
            or not completion["child_capture_path"]
            or not Path(completion["child_capture_path"]).is_absolute()
            or not isinstance(completion["child_ready_marker_sha256"], str)
            or len(completion["child_ready_marker_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in completion["child_ready_marker_sha256"]
            )
            or type(completion["child_ready_marker_size"]) is not int
            or completion["child_ready_marker_size"] <= 0
            or completion["child_ready_marker_mode"] != "0444"
            or not isinstance(completion["child_ready_marker_path"], str)
            or not completion["child_ready_marker_path"]
            or not Path(completion["child_ready_marker_path"]).is_absolute()
        ):
            raise ValueError("Velocity smoke lacks parent completion evidence")
    else:
        if lifecycle != {
            "env_close": "complete",
            "simulation_app_close": "pending_child_exit",
            "capture_stage": "kit_child_after_env_close_before_simulation_app_close",
        }:
            raise ValueError("Kit-child smoke lifecycle is not close-pending")
        if "completion" in payload:
            raise ValueError("Kit-child capture must not claim parent completion")

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
        expected_case_keys = set(expected_case) | {
            "joint_position_before",
            "joint_velocity_before",
            "joint_position_after",
            "joint_velocity_after",
            "processed_joint_velocity",
            "articulation_joint_velocity_target",
            "soft_joint_position_limits",
            "finger_position_target",
            "processed_finger_position_target",
            "finger_position_before",
            "finger_velocity_before",
            "finger_position_after",
            "finger_velocity_after",
            "finger_average_slew_rad_s",
            "terminated",
            "truncated",
        }
        if set(case) != expected_case_keys:
            raise ValueError(f"Velocity smoke case {index} schema mismatch")
        for key, expected_value in expected_case.items():
            if case.get(key) != expected_value or (
                key != "action" and type(case.get(key)) is not type(expected_value)
            ):
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
                    type(value) in (int, float) and math.isfinite(value)
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
        expected_finger_target = math.pi / 4.0 if action[7] > 0.5 else 0.0
        for field in (
            "processed_finger_position_target",
            "finger_position_target",
        ):
            finger_target = case.get(field)
            if type(finger_target) not in (int, float) or not math.isclose(
                float(finger_target),
                expected_finger_target,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError(f"Velocity smoke case {index} {field} mismatch")
        finger_position_before = _finite_vector(
            [case.get("finger_position_before")],
            1,
            field=f"case {index} finger position before",
        )[0]
        finger_velocity_before = _finite_vector(
            [case.get("finger_velocity_before")],
            1,
            field=f"case {index} finger velocity before",
        )[0]
        finger_position_after = _finite_vector(
            [case.get("finger_position_after")],
            1,
            field=f"case {index} finger position after",
        )[0]
        finger_velocity_after = _finite_vector(
            [case.get("finger_velocity_after")],
            1,
            field=f"case {index} finger velocity after",
        )[0]
        finger_average_slew = _finite_vector(
            [case.get("finger_average_slew_rad_s")],
            1,
            field=f"case {index} finger average slew",
        )[0]
        recomputed_slew = (finger_position_after - finger_position_before) * 15.0
        if not math.isclose(
            finger_average_slew,
            recomputed_slew,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(f"Velocity smoke case {index} finger slew mismatch")
        maximum_gripper_velocity = (
            NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S
            + NATIVE_GRIPPER_MEASURED_VELOCITY_TOLERANCE
        )
        if any(
            abs(value) > maximum_gripper_velocity
            for value in (
                finger_velocity_before,
                finger_velocity_after,
                finger_average_slew,
            )
        ):
            raise ValueError(
                f"Velocity smoke case {index} exceeded the gripper velocity limit"
            )
        if expected_case["kind"] == "gripper" and not math.isclose(
            expected_finger_target,
            expected_case["expected_finger_target"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Velocity smoke case {index} gripper plan mismatch")
        if expected_case["kind"] == "gripper":
            if not math.isclose(
                finger_position_before,
                expected_case["precondition_finger_target"],
                rel_tol=0.0,
                abs_tol=NATIVE_GRIPPER_PRECONDITION_POSITION_TOLERANCE,
            ):
                raise ValueError(
                    f"Velocity smoke case {index} gripper precondition mismatch"
                )
            motion_sign = expected_case["expected_motion_sign"]
            if motion_sign * (finger_position_after - finger_position_before) <= 1e-4:
                raise ValueError(
                    f"Velocity smoke case {index} gripper position direction mismatch"
                )
            if motion_sign * finger_velocity_after <= 1e-4:
                raise ValueError(
                    f"Velocity smoke case {index} gripper velocity direction mismatch"
                )

    reset = payload.get("reset_probe")
    if not isinstance(reset, dict) or set(reset) != {
        "default_joint_position",
        "joint_position",
        "joint_velocity",
        "joint_velocity_target",
        "default_finger_position",
        "finger_position",
        "finger_velocity",
        "finger_position_target",
    }:
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
    default_finger = _finite_vector(
        [reset.get("default_finger_position")], 1, field="reset default finger"
    )[0]
    reset_finger = _finite_vector(
        [reset.get("finger_position")], 1, field="reset finger position"
    )[0]
    reset_finger_velocity = _finite_vector(
        [reset.get("finger_velocity")], 1, field="reset finger velocity"
    )[0]
    reset_finger_target = _finite_vector(
        [reset.get("finger_position_target")], 1, field="reset finger target"
    )[0]
    if (
        abs(reset_finger - default_finger)
        > NATIVE_GRIPPER_PRECONDITION_POSITION_TOLERANCE
        or abs(reset_finger_velocity) > NATIVE_GRIPPER_MEASURED_VELOCITY_TOLERANCE
        or not math.isclose(reset_finger_target, 0.0, rel_tol=0.0, abs_tol=1e-6)
    ):
        raise ValueError("Velocity smoke reset did not restore the gripper")

    validated = copy.deepcopy(payload)
    expected_status = (
        "pass" if require_parent_completion else "close_validated_pending_parent"
    )
    if present_derived and (
        payload["status"] != expected_status
        or payload["case_count"] != len(cases)
        or type(payload["case_count"]) is not int
    ):
        raise ValueError("Velocity smoke derived status mismatch")
    validated["status"] = expected_status
    validated["case_count"] = len(cases)
    return validated


def _smoke_bytes(payload: dict[str, Any]) -> bytes:
    validated = validate_joint_velocity_smoke(payload)
    return (
        json.dumps(
            validated,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_immutable_joint_velocity_smoke(
    path: Path, payload: dict[str, Any]
) -> dict[str, Any]:
    """Publish the only accepted pass artifact after child exit zero."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = _smoke_bytes(payload)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(rendered)
            output.flush()
            os.fsync(output.fileno())
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return validate_immutable_joint_velocity_smoke(path)


def validate_immutable_joint_velocity_smoke(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Velocity smoke artifact must not be a symlink")
    file_stat = path.stat()
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
        raise ValueError("Velocity smoke artifact must be one regular link")
    if stat.S_IMODE(file_stat.st_mode) != 0o444:
        raise ValueError("Velocity smoke artifact must have mode 0444")
    rendered = path.read_bytes()
    try:
        payload = json.loads(rendered)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Velocity smoke artifact is not strict JSON") from error
    if rendered != _smoke_bytes(payload):
        raise ValueError("Velocity smoke artifact is not canonical JSON")
    return {
        "path": str(path.resolve()),
        "size": len(rendered),
        "sha256": hashlib.sha256(rendered).hexdigest(),
        "runtime_sha256": payload["runtime_contract"]["runtime_sha256"],
        "status": "pass",
        "mode": "0444",
        "nlink": 1,
    }
