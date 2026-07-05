"""Closed controller-smoke schema for the DROID position adapter."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from polaris.pi05_droid_position_adapter import (
    PI05_DROID_POSITION_ADAPTER_PROFILE,
    canonical_json_bytes,
    validate_position_adapter_evidence,
    validate_position_target_hold_report,
)
from polaris.pi05_droid_position_contract import PANDA_ARM_JOINT_NAMES
from polaris.pi05_droid_position_runtime import (
    validate_position_adapter_runtime_report,
    validate_position_safety_report,
)


POSITION_SMOKE_PROFILE = "openpi_pi05_droid_position_controller_smoke_v1"


def build_position_smoke_cases(command_magnitude: float = 1.5) -> list[dict[str, Any]]:
    if (
        type(command_magnitude) not in (int, float)
        or isinstance(command_magnitude, bool)
        or not np.isfinite(command_magnitude)
        or command_magnitude <= 1.0
    ):
        raise ValueError("position smoke magnitude must be finite and exceed one")
    cases = []
    for index, joint_name in enumerate(PANDA_ARM_JOINT_NAMES):
        for sign, suffix in ((-1.0, "negative"), (1.0, "positive")):
            action = np.zeros(8, dtype=np.float64)
            action[index] = sign * command_magnitude
            cases.append(
                {
                    "label": f"{joint_name}_{suffix}_clipped",
                    "kind": "arm",
                    "joint_index": index,
                    "raw_action": action.tolist(),
                }
            )
    for value, label in ((0.0, "gripper_open"), (1.0, "gripper_closed")):
        action = np.zeros(8, dtype=np.float64)
        action[-1] = value
        cases.append(
            {
                "label": label,
                "kind": "gripper",
                "joint_index": None,
                "raw_action": action.tolist(),
            }
        )
    return cases


def validate_position_smoke(value: Any, *, require_parent_completion: bool) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("position smoke must be an object")
    required = {
        "schema_version",
        "smoke_profile",
        "controller_profile",
        "environment",
        "command_magnitude",
        "runtime_contract",
        "cases",
        "limit_guard_probe",
        "fresh_reanchor_probe",
        "lifecycle",
    }
    if require_parent_completion:
        required |= {"completion", "status", "case_count"}
    if set(value) != required:
        raise ValueError("position smoke schema mismatch")
    if (
        value["schema_version"] != 1
        or value["smoke_profile"] != POSITION_SMOKE_PROFILE
        or value["controller_profile"] != PI05_DROID_POSITION_ADAPTER_PROFILE
        or value["environment"] != "DROID-FoodBussing"
        or type(value["command_magnitude"]) not in (int, float)
        or value["command_magnitude"] <= 1.0
    ):
        raise ValueError("position smoke identity mismatch")
    validate_position_adapter_runtime_report(value["runtime_contract"])
    cases = value["cases"]
    expected_cases = build_position_smoke_cases(float(value["command_magnitude"]))
    if not isinstance(cases, list) or len(cases) != len(expected_cases):
        raise ValueError("position smoke case count mismatch")
    for case, expected_case in zip(cases, expected_cases, strict=True):
        required_case = {
            "label",
            "kind",
            "joint_index",
            "raw_action",
            "adapter",
            "processed_joint_position_target",
            "articulation_joint_position_target",
            "processed_finger_position_target",
            "articulation_finger_position_target",
            "target_hold",
            "safety",
            "measured_joint_position_after",
            "measured_joint_velocity_after",
            "terminated",
            "truncated",
        }
        if not isinstance(case, dict) or set(case) != required_case:
            raise ValueError("position smoke case schema mismatch")
        for field in ("label", "kind", "joint_index", "raw_action"):
            if case[field] != expected_case[field]:
                raise ValueError(f"position smoke {field} mismatch")
        adapter = validate_position_adapter_evidence(case["adapter"])
        if adapter["raw_action"] != expected_case["raw_action"]:
            raise ValueError("position smoke adapter/raw action mismatch")
        expected_target = np.asarray(
            adapter["absolute_joint_position_target_rad"], dtype=np.float32
        )
        for field in (
            "processed_joint_position_target",
            "articulation_joint_position_target",
        ):
            actual = np.asarray(case[field], dtype=np.float32)
            if actual.shape != (7,) or not np.array_equal(actual, expected_target):
                raise ValueError(f"position smoke {field} mismatch")
        expected_finger = np.float32(
            np.pi / 4.0
            if adapter["absolute_closed_positive_gripper"] == 1.0
            else 0.0
        )
        for field in (
            "processed_finger_position_target",
            "articulation_finger_position_target",
        ):
            if np.float32(case[field]) != expected_finger:
                raise ValueError(f"position smoke {field} mismatch")
        hold = validate_position_target_hold_report(case["target_hold"])
        if not np.array_equal(
            np.asarray(hold["absolute_joint_position_target_rad"], dtype=np.float32),
            expected_target,
        ):
            raise ValueError("position smoke hold target mismatch")
        safety = validate_position_safety_report(case["safety"])
        if safety["outer_steps"] != 1:
            raise ValueError("position smoke safety cadence mismatch")
        for field in ("measured_joint_position_after", "measured_joint_velocity_after"):
            array = np.asarray(case[field])
            if array.shape != (7,) or not np.isfinite(array).all():
                raise ValueError(f"position smoke {field} mismatch")
        if type(case["terminated"]) is not bool or type(case["truncated"]) is not bool:
            raise ValueError("position smoke terminal flags mismatch")
        if case["terminated"] or case["truncated"]:
            raise ValueError("position smoke case terminated")
    guard = value["limit_guard_probe"]
    if not isinstance(guard, dict) or set(guard) != {
        "joint_index",
        "upper_limit_rad",
        "adversarial_target_rad",
        "articulation_target_before",
        "articulation_target_after",
        "exception_type",
        "exception_message",
        "setter_unchanged",
    }:
        raise ValueError("position smoke limit guard schema mismatch")
    if (
        guard["joint_index"] != 0
        or not guard["adversarial_target_rad"] > guard["upper_limit_rad"]
        or guard["articulation_target_before"] != guard["articulation_target_after"]
        or guard["exception_type"] != "PositionActionTargetLimitError"
        or "before setter" not in guard["exception_message"]
        or guard["setter_unchanged"] is not True
    ):
        raise ValueError("position smoke limit guard did not fail before setter")
    reanchor = value["fresh_reanchor_probe"]
    required_reanchor = {
        "raw_action",
        "step1_measured_joint_position",
        "step1_absolute_target",
        "step1_measured_joint_position_after",
        "step1_target_hold",
        "step2_measured_joint_position",
        "step2_absolute_target",
        "stale_prior_target_anchor_result",
        "step2_target_hold",
        "fresh_measurement_equals_step1_after",
        "step2_differs_from_stale_target_anchor",
        "safety",
    }
    if not isinstance(reanchor, dict) or set(reanchor) != required_reanchor:
        raise ValueError("position smoke fresh-reanchor schema mismatch")
    q1_after = np.asarray(reanchor["step1_measured_joint_position_after"])
    raw_action = np.asarray(reanchor["raw_action"])
    q1 = np.asarray(reanchor["step1_measured_joint_position"])
    target1 = np.asarray(reanchor["step1_absolute_target"])
    q2 = np.asarray(reanchor["step2_measured_joint_position"])
    target2 = np.asarray(reanchor["step2_absolute_target"])
    stale = np.asarray(reanchor["stale_prior_target_anchor_result"])
    clipped_arm_action = np.clip(raw_action[:7], -1.0, 1.0)
    expected_target1 = q1.astype(np.float64) + 0.2 * clipped_arm_action
    expected_target2 = q2.astype(np.float64) + 0.2 * clipped_arm_action
    expected_stale = target1.astype(np.float64) + 0.2 * clipped_arm_action
    if (
        raw_action.shape != (8,)
        or not np.isfinite(raw_action).all()
        or q1.shape != (7,)
        or q1_after.shape != (7,)
        or q2.shape != (7,)
        or target1.shape != (7,)
        or target2.shape != (7,)
        or stale.shape != (7,)
        or not np.isfinite(q1).all()
        or not np.isfinite(q1_after).all()
        or not np.isfinite(q2).all()
        or not np.isfinite(target1).all()
        or not np.isfinite(target2).all()
        or not np.isfinite(stale).all()
        or reanchor["fresh_measurement_equals_step1_after"] is not True
        or reanchor["step2_differs_from_stale_target_anchor"] is not True
        or not np.array_equal(q1_after, q2)
        or not np.array_equal(target1.astype(np.float64), expected_target1)
        or not np.array_equal(target2.astype(np.float64), expected_target2)
        or not np.array_equal(stale.astype(np.float64), expected_stale)
        or np.array_equal(target2.astype(np.float32), stale.astype(np.float32))
    ):
        raise ValueError(
            "position smoke did not prove exact DROID formula and fresh re-anchoring"
        )
    for field in ("step1_target_hold", "step2_target_hold"):
        validate_position_target_hold_report(reanchor[field])
    if validate_position_safety_report(reanchor["safety"])["outer_steps"] != 2:
        raise ValueError("position smoke fresh-reanchor safety cadence mismatch")
    lifecycle = value["lifecycle"]
    expected_lifecycle = (
        {
            "env_close": "complete",
            "simulation_app_close": "invoked_then_child_exited_zero",
            "capture_stage": "stdlib_parent_after_kit_child_exit",
        }
        if require_parent_completion
        else {
            "env_close": "complete",
            "simulation_app_close": "pending_child_exit",
            "capture_stage": "kit_child_after_env_close_before_simulation_app_close",
        }
    )
    if lifecycle != expected_lifecycle:
        raise ValueError("position smoke lifecycle mismatch")
    if require_parent_completion:
        if value["status"] != "pass" or value["case_count"] != len(cases):
            raise ValueError("position smoke completion status mismatch")
        completion = value["completion"]
        if not isinstance(completion, dict) or set(completion) != {
            "child_exit_code",
            "raw_sha256",
            "ready_sha256",
        }:
            raise ValueError("position smoke completion schema mismatch")
        if completion["child_exit_code"] != 0:
            raise ValueError("position smoke child failed")
    return json.loads(canonical_json_bytes(value))


def publish_position_smoke(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    validated = validate_position_smoke(payload, require_parent_completion=True)
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing position smoke: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = canonical_json_bytes(validated) + b"\n"
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
    return {
        "path": str(path.resolve()),
        "size": len(rendered),
        "sha256": hashlib.sha256(rendered).hexdigest(),
        "mode": "0444",
        "nlink": 1,
    }


__all__ = [
    "POSITION_SMOKE_PROFILE",
    "build_position_smoke_cases",
    "publish_position_smoke",
    "validate_position_smoke",
]
