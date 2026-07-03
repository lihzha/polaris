import copy
import math

import pytest

from polaris.joint_velocity_smoke import (
    build_joint_velocity_smoke_cases,
    validate_joint_velocity_smoke,
)


def test_smoke_plan_covers_hold_signed_joints_gripper_boundary_reset_and_limits():
    cases = build_joint_velocity_smoke_cases()
    assert len(cases) == 20
    assert cases[0]["label"] == "hold"
    assert sum(case["kind"] == "signed_joint" for case in cases) == 14
    assert sum(case["kind"] == "gripper" for case in cases) == 3
    assert sum(case["kind"] == "limit" for case in cases) == 2
    assert cases[-4]["expected_finger_target"] == pytest.approx(math.pi / 4)
    assert cases[-3] == {
        "label": "gripper_boundary_0p5",
        "action": [0.0] * 7 + [0.5],
        "kind": "gripper",
        "expected_finger_target": 0.0,
        "threshold_boundary": 0.5,
    }


def test_smoke_validator_recomputes_full_runtime_direction_targets_and_reset(
    valid_joint_velocity_smoke_payload,
):
    validated = validate_joint_velocity_smoke(valid_joint_velocity_smoke_payload)
    assert validated["status"] == "pass"
    assert validated["case_count"] == 20

    wrong_direction = copy.deepcopy(valid_joint_velocity_smoke_payload)
    wrong_direction["cases"][1]["joint_position_after"][0] *= -1
    with pytest.raises(ValueError, match="commanded direction"):
        validate_joint_velocity_smoke(wrong_direction)

    wrong_gripper = copy.deepcopy(valid_joint_velocity_smoke_payload)
    boundary = next(
        case
        for case in wrong_gripper["cases"]
        if case["label"] == "gripper_boundary_0p5"
    )
    boundary["finger_position_target"] = math.pi / 4
    with pytest.raises(ValueError, match="finger_position_target mismatch"):
        validate_joint_velocity_smoke(wrong_gripper)

    wrong_processed_gripper = copy.deepcopy(valid_joint_velocity_smoke_payload)
    wrong_processed_gripper["cases"][0]["processed_finger_position_target"] = 0.1
    with pytest.raises(ValueError, match="processed_finger_position_target mismatch"):
        validate_joint_velocity_smoke(wrong_processed_gripper)

    leaked_reset = copy.deepcopy(valid_joint_velocity_smoke_payload)
    leaked_reset["reset_probe"]["joint_velocity_target"][0] = 0.25
    with pytest.raises(ValueError, match="zero velocity target"):
        validate_joint_velocity_smoke(leaked_reset)

    tampered_runtime = copy.deepcopy(valid_joint_velocity_smoke_payload)
    tampered_runtime["runtime_contract"]["clip"]["values"][0][0][0] = -0.5
    with pytest.raises(ValueError, match="runtime contract SHA-256 mismatch"):
        validate_joint_velocity_smoke(tampered_runtime)


def test_smoke_child_stage_cannot_claim_parent_completion_or_extra_fields(
    valid_joint_velocity_smoke_payload,
):
    child = copy.deepcopy(valid_joint_velocity_smoke_payload)
    child.pop("completion")
    child["lifecycle"] = {
        "env_close": "complete",
        "simulation_app_close": "pending_child_exit",
        "capture_stage": "kit_child_after_env_close_before_simulation_app_close",
    }
    validated = validate_joint_velocity_smoke(child, require_parent_completion=False)
    assert validated["status"] == "close_validated_pending_parent"

    child["completion"] = {
        "child_exit_code": 0,
        "publication_stage": "stdlib_parent_after_child_exit",
        "child_capture_sha256": "a" * 64,
        "child_capture_size": 1,
        "child_capture_mode": "0400",
        "child_capture_path": "/tmp/test-child-close.json",
    }
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_joint_velocity_smoke(child, require_parent_completion=False)

    extra = copy.deepcopy(valid_joint_velocity_smoke_payload)
    extra["unbound"] = True
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_joint_velocity_smoke(extra)
