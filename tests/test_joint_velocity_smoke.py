import copy
import math

import pytest

from polaris.joint_velocity_smoke import (
    SMOKE_PROFILE,
    build_joint_velocity_smoke_cases,
    validate_joint_velocity_smoke,
)
from polaris.pi05_droid_jointvelocity_contract import (
    PI05_DROID_JOINTVELOCITY_PROFILE,
)


def _payload():
    command_magnitude = 0.25
    cases = []
    for case in build_joint_velocity_smoke_cases(command_magnitude):
        action = case["action"]
        q_before = [0.0] * 7
        q_after = [0.01 * value for value in action[:7]]
        dq_after = [0.5 * value for value in action[:7]]
        result = {
            **case,
            "joint_position_before": q_before,
            "joint_velocity_before": [0.0] * 7,
            "joint_position_after": q_after,
            "joint_velocity_after": dq_after,
            "processed_joint_velocity": action[:7],
            "articulation_joint_velocity_target": action[:7],
            "soft_joint_position_limits": [[-3.0, 3.0] for _ in range(7)],
            "finger_position_target": (case.get("expected_finger_target", 0.0)),
            "terminated": False,
            "truncated": False,
        }
        cases.append(result)
    return {
        "schema_version": 1,
        "smoke_profile": SMOKE_PROFILE,
        "controller_profile": PI05_DROID_JOINTVELOCITY_PROFILE,
        "environment": "DROID-FoodBussing",
        "command_magnitude": command_magnitude,
        "settle_steps": 5,
        "runtime_contract": {
            "status": "pass",
            "profile": PI05_DROID_JOINTVELOCITY_PROFILE,
        },
        "cases": cases,
        "reset_probe": {
            "default_joint_position": [0.0] * 7,
            "joint_position": [0.0] * 7,
            "joint_velocity": [0.0] * 7,
            "joint_velocity_target": [0.0] * 7,
        },
    }


def test_smoke_plan_covers_hold_signed_joints_gripper_reset_and_limits():
    cases = build_joint_velocity_smoke_cases()
    assert len(cases) == 19
    assert cases[0]["label"] == "hold"
    assert sum(case["kind"] == "signed_joint" for case in cases) == 14
    assert sum(case["kind"] == "gripper" for case in cases) == 2
    assert sum(case["kind"] == "limit" for case in cases) == 2
    assert cases[-3]["expected_finger_target"] == pytest.approx(math.pi / 4)


def test_smoke_validator_recomputes_direction_targets_limits_and_reset():
    validated = validate_joint_velocity_smoke(_payload())
    assert validated["status"] == "pass"
    assert validated["case_count"] == 19

    wrong_direction = _payload()
    wrong_direction["cases"][1]["joint_position_after"][0] *= -1
    with pytest.raises(ValueError, match="commanded direction"):
        validate_joint_velocity_smoke(wrong_direction)

    wrong_gripper = _payload()
    wrong_gripper["cases"][-3]["finger_position_target"] = 0.0
    with pytest.raises(ValueError, match="gripper target"):
        validate_joint_velocity_smoke(wrong_gripper)

    leaked_reset = copy.deepcopy(_payload())
    leaked_reset["reset_probe"]["joint_velocity_target"][0] = 0.25
    with pytest.raises(ValueError, match="zero velocity target"):
        validate_joint_velocity_smoke(leaked_reset)
