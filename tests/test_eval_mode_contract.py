from __future__ import annotations

import ast
from pathlib import Path

import pytest

from polaris.config import EEF_CONTROLLER_BASELINE_PROFILE
from polaris.config import EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE
from polaris.config import EEF_CONTROLLER_RELEASE_RAMP_CANDIDATE_PROFILE
from polaris.config import EvalArgs, PolicyArgs, validate_policy_control_mode


def _eval_args(
    *,
    client: str,
    control_mode: str,
    eef_controller_profile: str = EEF_CONTROLLER_BASELINE_PROFILE,
) -> EvalArgs:
    return EvalArgs(
        policy=PolicyArgs(client=client),
        environment="DROID-FoodBussing",
        run_folder="/tmp/polaris-test",
        control_mode=control_mode,  # type: ignore[arg-type]
        eef_controller_profile=eef_controller_profile,  # type: ignore[arg-type]
    )


def test_joint_position_defaults_are_unchanged() -> None:
    args = _eval_args(client="DroidJointPos", control_mode="joint-position")

    assert args.policy.open_loop_horizon == 8
    assert args.control_mode == "joint-position"
    assert args.eef_controller_profile == EEF_CONTROLLER_BASELINE_PROFILE
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


def test_candidate_profile_requires_exact_ego_lap_eef_pairing() -> None:
    for profile in (
        EEF_CONTROLLER_MIMIC_COMPLIANCE_CANDIDATE_PROFILE,
        EEF_CONTROLLER_RELEASE_RAMP_CANDIDATE_PROFILE,
    ):
        validate_policy_control_mode(
            _eval_args(
                client="EgoLAPEefPose",
                control_mode="eef-pose",
                eef_controller_profile=profile,
            )
        )
        with pytest.raises(ValueError, match="mimic-compliance"):
            validate_policy_control_mode(
                _eval_args(
                    client="DroidJointPos",
                    control_mode="joint-position",
                    eef_controller_profile=profile,
                )
            )


def test_unknown_controller_profile_fails_before_isaac_launch() -> None:
    with pytest.raises(ValueError, match="Unknown PolaRiS EEF controller profile"):
        validate_policy_control_mode(
            _eval_args(
                client="EgoLAPEefPose",
                control_mode="eef-pose",
                eef_controller_profile="unreviewed-controller",
            )
        )


def test_transactional_trace_dir_guard_is_one_exact_comparison() -> None:
    source = (Path(__file__).parents[1] / "scripts" / "eval.py").read_text()
    tree = ast.parse(source)
    guarded = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Constant)
            and isinstance(child.value, str)
            and "trace_dir inside the" in child.value
            for child in ast.walk(node)
        )
    ]
    assert len(guarded) == 1
    comparisons = [
        node for node in ast.walk(guarded[0].test) if isinstance(node, ast.Compare)
    ]
    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert len(comparison.ops) == 1
    assert len(comparison.comparators) == 1
    assert isinstance(comparison.ops[0], ast.NotEq)
    assert ast.unparse(comparison.left) == (
        "Path(eval_args.policy.trace_dir).resolve()"
    )
    assert ast.unparse(comparison.comparators[0]) == (
        "(run_folder / 'policy_traces').resolve()"
    )
