from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from polaris.eef_gripper_failure_trace import EEF_ALL_SIX_GRIPPER_TRACE_CAPACITY
from polaris.eef_gripper_failure_trace import EEF_ALL_SIX_GRIPPER_TRACE_PROFILE
from polaris.eef_gripper_failure_trace import (
    make_eef_all_six_gripper_failure_trace_class,
)
from polaris.eef_gripper_failure_trace import validate_eef_all_six_gripper_trace
from polaris.eef_gripper_runtime import EXPECTED_DROID_JOINT_NAMES
from polaris.eef_gripper_runtime import GRIPPER_CLOSED_TARGET_FLOAT32
from polaris.eef_gripper_runtime import GRIPPER_OPEN_TARGET_FLOAT32


class _BaseAction:
    def __init__(self, _cfg, _env) -> None:
        shape = (1, len(EXPECTED_DROID_JOINT_NAMES))
        data = SimpleNamespace(
            joint_names=list(EXPECTED_DROID_JOINT_NAMES),
            joint_pos=torch.zeros(shape, dtype=torch.float32),
            joint_vel=torch.zeros(shape, dtype=torch.float32),
            joint_acc=torch.zeros(shape, dtype=torch.float32),
            joint_pos_target=torch.zeros(shape, dtype=torch.float32),
            joint_vel_target=torch.zeros(shape, dtype=torch.float32),
            joint_effort_target=torch.zeros(shape, dtype=torch.float32),
        )
        self._asset = SimpleNamespace(data=data)
        self._raw_actions = torch.zeros((1, 1), dtype=torch.float32)
        self._processed_actions = torch.zeros((1, 1), dtype=torch.float32)
        self._open_command = torch.tensor(
            [GRIPPER_OPEN_TARGET_FLOAT32], dtype=torch.float32
        )
        self._close_command = torch.tensor(
            [GRIPPER_CLOSED_TARGET_FLOAT32], dtype=torch.float32
        )
        self.base_apply_calls = 0

    def reset(self, env_ids=None) -> None:
        del env_ids
        self._raw_actions.zero_()
        self._processed_actions.zero_()
        self.base_apply_calls = 0

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions.copy_(actions)
        self._processed_actions.copy_(
            torch.where(actions > 0.5, self._close_command, self._open_command)
        )

    def apply_actions(self) -> None:
        self.base_apply_calls += 1
        data = self._asset.data
        data.joint_pos[:, 7:] += 0.001
        data.joint_vel[:, 7:] = float(self.base_apply_calls) * 0.01
        data.joint_acc[:, 7:] = 0.1
        data.joint_pos_target[:, 7] = self._processed_actions[:, 0]
        data.joint_vel_target[:, 7:] = 0.0
        data.joint_effort_target[:, 7:] = 0.0


def _rollout(*, policy_steps: int, final_substeps: int = 8, failure: bool = False):
    action_class = make_eef_all_six_gripper_failure_trace_class(_BaseAction)
    action = action_class(None, None)
    for step in range(policy_steps):
        action.begin_eef_policy_step(episode_index=3, policy_step=step)
        action.process_actions(torch.tensor([[float(step % 2)]], dtype=torch.float32))
        substeps = final_substeps if step == policy_steps - 1 else 8
        for _ in range(substeps):
            action.apply_actions()
    action.finalize_eef_rollout_trace(numerical_failure=failure)
    return action, action.eef_all_six_gripper_trace()


def test_success_trace_is_observational_and_retains_exact_last_64() -> None:
    action, trace = _rollout(policy_steps=10)

    assert action.base_apply_calls == 80
    assert trace["profile"] == EEF_ALL_SIX_GRIPPER_TRACE_PROFILE
    assert trace["total_apply_entries"] == 80
    assert trace["dropped_entries"] == 16
    assert trace["initial_snapshot"]["joint_pos_rad"] == [0.0] * 6
    assert len(trace["entries"]) == EEF_ALL_SIX_GRIPPER_TRACE_CAPACITY
    assert trace["entries"][0]["apply_index"] == 16
    assert trace["entries"][-1]["apply_index"] == 79
    validate_eef_all_six_gripper_trace(
        trace,
        episode_index=3,
        episode_length=10,
        numerical_failure=False,
        expected_apply_calls=80,
    )


def test_failure_trace_closes_partial_final_policy_step() -> None:
    action, trace = _rollout(policy_steps=2, final_substeps=3, failure=True)

    assert action.base_apply_calls == 11
    assert trace["process_action_calls"] == 2
    assert trace["total_apply_entries"] == 11
    assert trace["entries"][-1]["policy_step"] == 1
    assert trace["entries"][-1]["physics_substep"] == 2
    assert trace["terminal_snapshot"] == trace["entries"][-1]["post"]
    validate_eef_all_six_gripper_trace(
        trace,
        episode_index=3,
        episode_length=2,
        numerical_failure=True,
        expected_apply_calls=11,
    )


def test_first_arm_apply_failure_binds_empty_terminal_to_initial() -> None:
    _action, trace = _rollout(policy_steps=1, final_substeps=0, failure=True)
    assert trace["total_apply_entries"] == 0
    assert trace["entries"] == []
    assert trace["terminal_snapshot"] == trace["initial_snapshot"]
    validate_eef_all_six_gripper_trace(
        trace,
        episode_index=3,
        episode_length=1,
        numerical_failure=True,
        expected_apply_calls=0,
    )

    trace["terminal_snapshot"]["joint_pos_rad"][0] = 1.0
    with pytest.raises(ValueError, match="empty terminal identity"):
        validate_eef_all_six_gripper_trace(
            trace,
            episode_index=3,
            episode_length=1,
            numerical_failure=True,
            expected_apply_calls=0,
        )


def test_trace_validator_rejects_cadence_and_terminal_tamper() -> None:
    _action, trace = _rollout(policy_steps=1)
    drifted = copy.deepcopy(trace)
    drifted["entries"][-1]["physics_substep"] = 0
    try:
        validate_eef_all_six_gripper_trace(
            drifted,
            episode_index=3,
            episode_length=1,
            numerical_failure=False,
            expected_apply_calls=8,
        )
    except ValueError as error:
        assert "entry cadence" in str(error)
    else:
        raise AssertionError("Trace cadence tamper was accepted")

    drifted = copy.deepcopy(trace)
    drifted["terminal_snapshot"]["joint_pos_rad"][0] += 1.0
    try:
        validate_eef_all_six_gripper_trace(
            drifted,
            episode_index=3,
            episode_length=1,
            numerical_failure=False,
            expected_apply_calls=8,
        )
    except ValueError as error:
        assert "terminal identity" in str(error)
    else:
        raise AssertionError("Trace terminal tamper was accepted")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda trace: trace["initial_snapshot"]["joint_pos_rad"].__setitem__(
                0, 1.0
            ),
            "initial identity",
        ),
        (
            lambda trace: trace["entries"][1]["pre"]["joint_pos_rad"].__setitem__(
                0, 1.0
            ),
            "transition continuity",
        ),
        (
            lambda trace: trace["entries"][1].__setitem__(
                "target_after_setter_rad", 0.123
            ),
            "setter target",
        ),
        (
            lambda trace: trace["entries"][1].__setitem__("raw_action", 0.25),
            "binary endpoint",
        ),
        (
            lambda trace: trace["entries"][1].__setitem__("raw_action", -0.0),
            "binary endpoint",
        ),
        (
            lambda trace: trace["entries"][1].__setitem__(
                "requested_endpoint_rad", 0.25
            ),
            "binary endpoint",
        ),
        (
            lambda trace: trace["entries"][1].__setitem__(
                "requested_endpoint_rad", -0.0
            ),
            "binary endpoint",
        ),
    ],
)
def test_trace_validator_rejects_initial_and_middle_tail_tamper(
    mutation, message
) -> None:
    _action, trace = _rollout(policy_steps=1)
    mutation(trace)

    with pytest.raises(ValueError, match=message):
        validate_eef_all_six_gripper_trace(
            trace,
            episode_index=3,
            episode_length=1,
            numerical_failure=False,
            expected_apply_calls=8,
        )


def test_trace_validator_closes_success_and_failure_apply_cadence() -> None:
    _action, complete_failure = _rollout(policy_steps=1, failure=True)
    with pytest.raises(ValueError, match="failure cadence"):
        validate_eef_all_six_gripper_trace(
            complete_failure,
            episode_index=3,
            episode_length=1,
            numerical_failure=True,
            expected_apply_calls=8,
        )

    _action, partial_success = _rollout(policy_steps=1, final_substeps=3, failure=False)
    with pytest.raises(ValueError, match="success cadence"):
        validate_eef_all_six_gripper_trace(
            partial_success,
            episode_index=3,
            episode_length=1,
            numerical_failure=False,
            expected_apply_calls=3,
        )

    _action, trace = _rollout(policy_steps=1)
    with pytest.raises(ValueError, match="validation inputs"):
        validate_eef_all_six_gripper_trace(
            trace,
            episode_index=3,
            episode_length=True,
            numerical_failure=False,
            expected_apply_calls=8,
        )
