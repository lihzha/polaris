from __future__ import annotations

import pytest

from polaris.config import EvalArgs, PolicyArgs, validate_policy_control_mode


def _eval_args(*, client: str, control_mode: str) -> EvalArgs:
    return EvalArgs(
        policy=PolicyArgs(client=client),
        environment="DROID-FoodBussing",
        run_folder="/tmp/polaris-test",
        control_mode=control_mode,  # type: ignore[arg-type]
    )


def test_joint_position_defaults_are_unchanged() -> None:
    args = _eval_args(client="DroidJointPos", control_mode="joint-position")

    assert args.policy.open_loop_horizon == 8
    assert args.control_mode == "joint-position"
    validate_policy_control_mode(args)


def test_joint_policy_rejects_eef_controller() -> None:
    with pytest.raises(ValueError, match="DroidJointPos"):
        validate_policy_control_mode(
            _eval_args(client="DroidJointPos", control_mode="eef-pose")
        )


def test_ego_lap_requires_eef_controller() -> None:
    with pytest.raises(ValueError, match="EgoLAPEefPose"):
        validate_policy_control_mode(
            _eval_args(client="EgoLAPEefPose", control_mode="joint-position")
        )

    validate_policy_control_mode(
        _eval_args(client="EgoLAPEefPose", control_mode="eef-pose")
    )
