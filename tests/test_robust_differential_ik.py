import inspect
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
import polaris.robust_differential_ik as robust_ik
from polaris.robust_differential_ik import (
    DifferentialIKNumericalError,
    RobustDifferentialIKController,
    RobustDifferentialInverseKinematicsAction,
    _apply_wrist_energy_brake_target,
    _bound_joint_position_target,
    _derive_isaac_soft_joint_position_limits,
    _eef_quaternion_norm_is_valid,
    _install_eef_physx_position_limits,
    _require_current_joint_position_in_soft_limits,
    _require_finite,
)
from polaris.eef_ik_safety import CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD
from polaris.eef_ik_safety import CURRENT_JOINT_VELOCITY_ABORT_EVIDENCE_PROFILE
from polaris.eef_ik_safety import current_joint_velocity_abort_evidence_sha256
from polaris.eef_ik_safety import EEF_QUATERNION_UNIT_NORM_TOLERANCE
from polaris.eef_ik_safety import format_current_joint_velocity_abort_message
from polaris.eef_ik_safety import JOINT_VELOCITY_LIMIT_TOLERANCE_RAD_S
from polaris.eef_ik_safety import PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S
from polaris.eef_ik_safety import PANDA_EEF_JOINT_EFFORT_LIMITS
from polaris.eef_ik_safety import PANDA_PHYSX_DERIVED_SOFT_JOINT_POS_LIMITS_RAD
from polaris.eef_ik_safety import PANDA_SOFT_JOINT_POS_LIMITS_RAD
from polaris.eef_ik_safety import WRIST_ENERGY_BRAKE_LATCH_SUBSTEPS
from polaris.eef_ik_safety import WRIST_ENERGY_BRAKE_TARGET_SHIFT_FRACTION


_BOUNDARY_SMOKE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "smoke_eef_pose_boundary_replay.py"
)
_BOUNDARY_SMOKE_SPEC = importlib.util.spec_from_file_location(
    "polaris_boundary_smoke_contract",
    _BOUNDARY_SMOKE_PATH,
)
if _BOUNDARY_SMOKE_SPEC is None or _BOUNDARY_SMOKE_SPEC.loader is None:
    raise RuntimeError(f"Cannot load boundary smoke contract: {_BOUNDARY_SMOKE_PATH}")
boundary_smoke = importlib.util.module_from_spec(_BOUNDARY_SMOKE_SPEC)
_BOUNDARY_SMOKE_SPEC.loader.exec_module(boundary_smoke)


def _bare_robust_action():
    action = object.__new__(RobustDifferentialInverseKinematicsAction)
    # ActionTerm.__del__ reads this field even when __init__ never ran.
    action._debug_vis_handle = None
    action._env = SimpleNamespace(num_envs=1, device="cpu")
    return action


def test_bare_robust_action_uses_real_action_term_properties_and_tears_down():
    action = _bare_robust_action()
    assert action.num_envs == 1
    assert action.device == "cpu"
    action.__del__()


def _controller(controller_type, *, damping=0.01):
    cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": damping},
    )
    return controller_type(cfg=cfg, num_envs=1, device="cpu")


def test_healthy_dls_path_matches_isaac_lab_exactly():
    torch.manual_seed(7)
    jacobian = torch.randn(1, 6, 7)
    delta_pose = torch.randn(1, 6)
    expected = _controller(DifferentialIKController)._compute_delta_joint_pos(
        delta_pose, jacobian
    )
    controller = _controller(RobustDifferentialIKController)

    actual = controller._compute_delta_joint_pos(delta_pose, jacobian)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert controller.fallback_count == 0


def test_float32_damping_loss_uses_finite_pseudoinverse_fallback():
    jacobian = torch.full((1, 6, 7), 100.0)
    delta_pose = torch.ones(1, 6)
    with pytest.raises(torch.linalg.LinAlgError):
        _controller(DifferentialIKController)._compute_delta_joint_pos(
            delta_pose, jacobian
        )

    controller = _controller(RobustDifferentialIKController)

    actual = controller._compute_delta_joint_pos(delta_pose, jacobian)

    assert torch.isfinite(actual).all()
    assert controller.fallback_count == 1


def test_nonfinite_input_aborts_rollout_before_physics_step():
    controller = _controller(RobustDifferentialIKController, damping=0.0)
    jacobian = torch.zeros(1, 6, 7)
    jacobian[0, 0, 0] = torch.nan
    delta_pose = torch.ones(1, 6)

    with pytest.raises(DifferentialIKNumericalError, match="non-finite input"):
        controller._compute_delta_joint_pos(delta_pose, jacobian)

    assert controller.fallback_count == 0


def test_joint_target_safety_preserves_healthy_target_and_bounds_outlier():
    joint_pos = torch.zeros(1, 7)
    max_delta = torch.full((1, 7), 0.02)
    soft_limits = torch.tensor([[[-1.0, 1.0]] * 7])

    healthy = torch.tensor([[0.01, -0.01, 0.0, 0.005, 0.0, 0.01, -0.01]])
    safe, raw_delta, slew_limited, position_limited = _bound_joint_position_target(
        joint_pos, healthy, max_delta, soft_limits
    )
    torch.testing.assert_close(safe, healthy, rtol=0.0, atol=0.0)
    torch.testing.assert_close(raw_delta, healthy, rtol=0.0, atol=0.0)
    assert not slew_limited.any()
    assert not position_limited.any()

    # q + (target - q) rounds one ULP away for this finite float32 pair.
    # An inactive safety guard must still preserve the inherited DLS target.
    ulp_joint_pos = torch.full((1, 7), 1.0941112, dtype=torch.float32)
    ulp_target = torch.full((1, 7), 0.4359291, dtype=torch.float32)
    ulp_safe, _, ulp_slew, ulp_position = _bound_joint_position_target(
        ulp_joint_pos,
        ulp_target,
        torch.ones((1, 7), dtype=torch.float32),
        torch.tensor([[[-3.0, 3.0]] * 7], dtype=torch.float32),
    )
    assert not ulp_slew.any()
    assert not ulp_position.any()
    assert torch.equal(ulp_safe.view(torch.int32), ulp_target.view(torch.int32))

    outlier = torch.tensor([[0.5, -0.5, 0.03, -0.03, 0.0, 0.02, -0.02]])
    safe, _, slew_limited, position_limited = _bound_joint_position_target(
        joint_pos, outlier, max_delta, soft_limits
    )
    assert torch.all(safe.abs() <= max_delta)
    assert slew_limited[0, :4].tolist() == [True, True, True, True]
    assert not position_limited.any()


def test_joint_target_safety_intersects_slew_and_soft_position_limits():
    joint_pos = torch.tensor([[0.99] * 7])
    raw_target = torch.tensor([[1.5] * 7])
    max_delta = torch.full((1, 7), 0.02)
    soft_limits = torch.tensor([[[-1.0, 1.0]] * 7])

    safe, _, slew_limited, position_limited = _bound_joint_position_target(
        joint_pos, raw_target, max_delta, soft_limits
    )

    # The command remains one maximum physics-substep motion inside the live
    # articulation limit, so the implicit actuator has room to brake.
    torch.testing.assert_close(safe, torch.full_like(safe, 0.98), rtol=0.0, atol=0.0)
    assert slew_limited.all()
    assert position_limited.all()
    assert torch.all((safe - joint_pos).abs() <= max_delta)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_joint_target_guard_band_prevents_exact_bound_actuator_command(direction):
    soft_limits = torch.tensor([PANDA_SOFT_JOINT_POS_LIMITS_RAD], dtype=torch.float32)
    max_delta = torch.tensor(
        [PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S], dtype=torch.float32
    ) * (1.0 / 120.0)
    boundary = soft_limits[..., 1] if direction > 0 else soft_limits[..., 0]
    joint_pos = boundary - direction * max_delta / 4.0
    raw_target = boundary + direction * max_delta

    safe, _, slew_limited, position_limited = _bound_joint_position_target(
        joint_pos, raw_target, max_delta, soft_limits
    )

    expected = boundary - direction * max_delta
    torch.testing.assert_close(safe, expected, rtol=0.0, atol=0.0)
    assert slew_limited.all()
    assert position_limited.all()
    if direction > 0:
        assert torch.all(safe < soft_limits[..., 1])
    else:
        assert torch.all(safe > soft_limits[..., 0])
    assert torch.all((safe - joint_pos).abs() <= max_delta)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_joint_target_guard_band_recovers_outer_tolerance_without_slew_violation(
    direction,
):
    soft_limits = torch.tensor([PANDA_SOFT_JOINT_POS_LIMITS_RAD], dtype=torch.float32)
    max_delta = torch.tensor(
        [PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S], dtype=torch.float32
    ) * (1.0 / 120.0)
    boundary = soft_limits[..., 1] if direction > 0 else soft_limits[..., 0]
    outer_tolerance_offset = torch.full_like(boundary, 5e-6)
    joint_pos = boundary + direction * outer_tolerance_offset
    raw_target = boundary + direction

    safe, _, slew_limited, position_limited = _bound_joint_position_target(
        joint_pos, raw_target, max_delta, soft_limits
    )

    strict_inner = boundary - direction * max_delta
    guard_band_violation = (safe - strict_inner) * direction
    assert slew_limited.all()
    assert position_limited.all()
    assert torch.all((safe - joint_pos).abs() <= max_delta + 1e-6)
    assert torch.all(guard_band_violation >= 0.0)
    assert torch.all(guard_band_violation <= CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD)
    assert torch.all(safe <= soft_limits[..., 1])
    assert torch.all(safe >= soft_limits[..., 0])


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_joint_target_guard_band_does_not_consume_recovery_for_in_range_state(
    direction,
):
    soft_limits = torch.tensor([PANDA_SOFT_JOINT_POS_LIMITS_RAD], dtype=torch.float32)
    max_delta = torch.tensor(
        [PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S], dtype=torch.float32
    ) * (1.0 / 120.0)
    boundary = soft_limits[..., 1] if direction > 0 else soft_limits[..., 0]
    joint_pos = boundary - direction * torch.full_like(boundary, 5e-6)
    raw_target = boundary + direction

    safe, _, _, position_limited = _bound_joint_position_target(
        joint_pos, raw_target, max_delta, soft_limits
    )

    strict_inner = boundary - direction * max_delta
    assert position_limited.all()
    torch.testing.assert_close(safe, strict_inner, rtol=0.0, atol=0.0)


_WRIST_JOINT_INDICES = (4, 5, 6)


def _wrist_energy_brake_inputs(num_envs=1):
    return {
        "joint_pos": torch.zeros((num_envs, 7), dtype=torch.float32),
        "joint_vel": torch.zeros((num_envs, 7), dtype=torch.float32),
        "previous_applied_target": torch.zeros((num_envs, 7), dtype=torch.float32),
        "reversal_detection_armed": torch.ones((num_envs,), dtype=torch.bool),
        "nominal_safe_target": torch.zeros((num_envs, 7), dtype=torch.float32),
        "max_delta_joint_pos": torch.full((num_envs, 7), 0.02, dtype=torch.float32),
        "soft_joint_pos_limits": torch.tensor(
            [[[-1.0, 1.0]] * 7] * num_envs,
            dtype=torch.float32,
        ),
        "latch_remaining": torch.zeros((num_envs,), dtype=torch.int64),
        "wrist_joint_indices": _WRIST_JOINT_INDICES,
    }


def test_wrist_energy_brake_disarmed_reversal_detection_is_bitwise_noop():
    inputs = _wrist_energy_brake_inputs()
    inputs["reversal_detection_armed"][0] = False
    inputs["previous_applied_target"][0, 4] = -0.01
    inputs["nominal_safe_target"][0, 4] = 0.01
    nominal_bits = inputs["nominal_safe_target"].view(torch.int32).clone()

    result = _apply_wrist_energy_brake_target(**inputs)

    assert torch.equal(result.applied_target.view(torch.int32), nominal_bits)
    assert result.next_latch_remaining.tolist() == [0]
    assert not result.trigger_joint_mask.any()
    assert not result.active_environment_mask.any()
    assert not result.attempted_joint_mask.any()
    assert not result.braked_joint_mask.any()
    torch.testing.assert_close(
        result.target_shift,
        torch.tensor([[0.02, 0.0, 0.0]], dtype=torch.float32),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("target_shift", "expected_trigger"),
    [(8.5, False), (9.0, True), (9.5, True)],
)
def test_wrist_energy_brake_target_shift_threshold_is_inclusive(
    target_shift, expected_trigger
):
    inputs = _wrist_energy_brake_inputs()
    inputs["max_delta_joint_pos"].fill_(10.0)
    inputs["soft_joint_pos_limits"] = torch.tensor(
        [[[-30.0, 30.0]] * 7], dtype=torch.float32
    )
    inputs["previous_applied_target"][0, 4] = -target_shift / 2
    inputs["nominal_safe_target"][0, 4] = target_shift / 2
    threshold = (
        inputs["max_delta_joint_pos"][0, 4] * WRIST_ENERGY_BRAKE_TARGET_SHIFT_FRACTION
    )
    assert threshold.item() == 9.0
    assert (
        inputs["nominal_safe_target"][0, 4] - inputs["previous_applied_target"][0, 4]
    ).item() == target_shift

    result = _apply_wrist_energy_brake_target(**inputs)

    assert result.trigger_joint_mask[0, 0].item() is expected_trigger
    assert result.trigger_joint_mask.sum().item() == int(expected_trigger)
    assert result.active_environment_mask.tolist() == [expected_trigger]
    assert result.next_latch_remaining.tolist() == [
        WRIST_ENERGY_BRAKE_LATCH_SUBSTEPS - 1 if expected_trigger else 0
    ]


@pytest.mark.parametrize(
    ("previous_error", "nominal_error"),
    [
        (0.0, 1.0),
        (-1.0, 0.0),
        (0.1, 1.0),
        (-1.0, -0.1),
    ],
)
def test_wrist_energy_brake_requires_strict_error_sign_reversal(
    previous_error, nominal_error
):
    inputs = _wrist_energy_brake_inputs()
    inputs["max_delta_joint_pos"].fill_(1.0)
    inputs["soft_joint_pos_limits"] = torch.tensor(
        [[[-3.0, 3.0]] * 7], dtype=torch.float32
    )
    inputs["previous_applied_target"][0, 4] = previous_error
    inputs["nominal_safe_target"][0, 4] = nominal_error

    result = _apply_wrist_energy_brake_target(**inputs)

    assert not result.trigger_joint_mask.any()
    assert not result.active_environment_mask.any()
    assert result.next_latch_remaining.tolist() == [0]
    assert torch.equal(
        result.applied_target.view(torch.int32),
        inputs["nominal_safe_target"].view(torch.int32),
    )


@pytest.mark.parametrize("trigger_joint", _WRIST_JOINT_INDICES)
def test_each_wrist_reversal_activates_group_brake_without_changing_arm_joints(
    trigger_joint,
):
    inputs = _wrist_energy_brake_inputs()
    inputs["nominal_safe_target"][0, :4] = torch.tensor(
        [0.001, -0.002, 0.003, -0.004], dtype=torch.float32
    )
    inputs["nominal_safe_target"][0, 4:] = 0.01
    inputs["previous_applied_target"].copy_(inputs["nominal_safe_target"])
    inputs["previous_applied_target"][0, trigger_joint] = -0.01
    inputs["joint_vel"][0, 4:] = 0.2
    nominal_bits = inputs["nominal_safe_target"].view(torch.int32).clone()

    result = _apply_wrist_energy_brake_target(**inputs)

    expected_trigger = torch.zeros((1, 3), dtype=torch.bool)
    expected_trigger[0, trigger_joint - _WRIST_JOINT_INDICES[0]] = True
    expected_braked = torch.ones((1, 3), dtype=torch.bool)
    assert torch.equal(result.trigger_joint_mask, expected_trigger)
    assert result.active_environment_mask.tolist() == [True]
    assert result.next_latch_remaining.tolist() == [
        WRIST_ENERGY_BRAKE_LATCH_SUBSTEPS - 1
    ]
    assert torch.equal(result.attempted_joint_mask, expected_braked)
    assert torch.equal(result.braked_joint_mask, expected_braked)
    assert torch.equal(
        result.applied_target[0, :4].view(torch.int32), nominal_bits[0, :4]
    )
    assert torch.equal(result.applied_target[0, 4:], inputs["joint_pos"][0, 4:])
    assert torch.equal(
        result.target_shift,
        (
            inputs["nominal_safe_target"][:, 4:]
            - inputs["previous_applied_target"][:, 4:]
        ).abs(),
    )


def test_wrist_energy_brake_projects_only_strictly_positive_spring_power():
    inputs = _wrist_energy_brake_inputs()
    inputs["joint_pos"][0, 4:] = torch.tensor([0.1, 0.2, -0.1], dtype=torch.float32)
    inputs["nominal_safe_target"][0, 4:] = inputs["joint_pos"][0, 4:] + torch.tensor(
        [0.01, 0.01, -0.01], dtype=torch.float32
    )
    inputs["previous_applied_target"].copy_(inputs["nominal_safe_target"])
    inputs["previous_applied_target"][0, 4] = inputs["joint_pos"][0, 4] - 0.01
    inputs["joint_vel"][0, 4:] = torch.tensor([0.2, 0.0, -0.2], dtype=torch.float32)

    result = _apply_wrist_energy_brake_target(**inputs)

    assert result.active_environment_mask.tolist() == [True]
    assert result.attempted_joint_mask[0].tolist() == [True, False, True]
    assert result.braked_joint_mask[0].tolist() == [True, False, True]
    torch.testing.assert_close(
        result.applied_target[0, [4, 6]],
        inputs["joint_pos"][0, [4, 6]],
        rtol=0.0,
        atol=0.0,
    )
    assert result.applied_target[0, 5].view(torch.int32) == inputs[
        "nominal_safe_target"
    ][0, 5].view(torch.int32)
    assert torch.equal(
        result.target_shift,
        (
            inputs["nominal_safe_target"][:, 4:]
            - inputs["previous_applied_target"][:, 4:]
        ).abs(),
    )


def test_wrist_energy_brake_latch_covers_trigger_and_next_then_expires_and_retriggers():
    inputs = _wrist_energy_brake_inputs()
    inputs["nominal_safe_target"][0, 4:] = 0.01
    inputs["previous_applied_target"].copy_(inputs["nominal_safe_target"])
    inputs["previous_applied_target"][0, 4] = -0.01
    inputs["joint_vel"][0, 4:] = 0.2

    trigger = _apply_wrist_energy_brake_target(**inputs)
    assert trigger.active_environment_mask.tolist() == [True]
    assert trigger.next_latch_remaining.tolist() == [1]

    inputs["previous_applied_target"] = trigger.applied_target
    inputs["latch_remaining"] = trigger.next_latch_remaining
    next_substep = _apply_wrist_energy_brake_target(**inputs)
    assert not next_substep.trigger_joint_mask.any()
    assert next_substep.active_environment_mask.tolist() == [True]
    assert next_substep.next_latch_remaining.tolist() == [0]
    assert next_substep.attempted_joint_mask[0].tolist() == [True, True, True]
    assert next_substep.braked_joint_mask[0].tolist() == [True, True, True]

    inputs["previous_applied_target"] = next_substep.applied_target
    inputs["latch_remaining"] = next_substep.next_latch_remaining
    expired = _apply_wrist_energy_brake_target(**inputs)
    assert not expired.trigger_joint_mask.any()
    assert not expired.active_environment_mask.any()
    assert expired.next_latch_remaining.tolist() == [0]
    assert torch.equal(
        expired.applied_target.view(torch.int32),
        inputs["nominal_safe_target"].view(torch.int32),
    )

    inputs["previous_applied_target"] = inputs["nominal_safe_target"].clone()
    inputs["previous_applied_target"][0, 6] = -0.01
    inputs["latch_remaining"] = torch.ones((1,), dtype=torch.int64)
    retriggered = _apply_wrist_energy_brake_target(**inputs)
    assert retriggered.trigger_joint_mask[0, 2]
    assert retriggered.active_environment_mask.tolist() == [True]
    assert retriggered.next_latch_remaining.tolist() == [1]


def test_wrist_energy_brake_refractory_command_suppresses_hold_induced_retrigger():
    inputs = _wrist_energy_brake_inputs()
    inputs["previous_applied_target"][0, 4] = -0.01
    inputs["nominal_safe_target"][0, 4] = 0.01
    inputs["joint_vel"][0, 4] = 0.2

    trigger = _apply_wrist_energy_brake_target(**inputs)
    assert trigger.trigger_joint_mask[0, 0]
    assert trigger.next_latch_remaining.tolist() == [1]

    inputs["joint_pos"][0, 4] = 0.001
    inputs["joint_vel"][0, 4] = 0.2
    inputs["previous_applied_target"] = trigger.applied_target
    inputs["reversal_detection_armed"] = torch.zeros((1,), dtype=torch.bool)
    inputs["nominal_safe_target"][0, 4] = 0.021
    inputs["latch_remaining"] = trigger.next_latch_remaining
    follow_up = _apply_wrist_energy_brake_target(**inputs)
    assert not follow_up.trigger_joint_mask.any()
    assert follow_up.active_environment_mask.tolist() == [True]
    assert follow_up.next_latch_remaining.tolist() == [0]

    inputs["joint_pos"][0, 4] = 0.002
    inputs["previous_applied_target"] = follow_up.applied_target
    inputs["nominal_safe_target"][0, 4] = 0.022
    inputs["latch_remaining"] = follow_up.next_latch_remaining
    refractory = _apply_wrist_energy_brake_target(**inputs)
    assert not refractory.trigger_joint_mask.any()
    assert not refractory.active_environment_mask.any()
    assert refractory.next_latch_remaining.tolist() == [0]
    assert torch.equal(
        refractory.applied_target.view(torch.int32),
        inputs["nominal_safe_target"].view(torch.int32),
    )


def test_wrist_energy_brake_state_is_isolated_per_environment():
    inputs = _wrist_energy_brake_inputs(num_envs=2)
    inputs["nominal_safe_target"][:, 4:] = 0.01
    inputs["previous_applied_target"].copy_(inputs["nominal_safe_target"])
    inputs["previous_applied_target"][0, 6] = -0.01
    inputs["joint_vel"][:, 4:] = 0.2

    result = _apply_wrist_energy_brake_target(**inputs)

    assert result.active_environment_mask.tolist() == [True, False]
    assert result.next_latch_remaining.tolist() == [1, 0]
    assert result.trigger_joint_mask[0].tolist() == [
        False,
        False,
        True,
    ]
    assert not result.trigger_joint_mask[1].any()
    assert result.attempted_joint_mask[0].tolist() == [True, True, True]
    assert not result.attempted_joint_mask[1].any()
    assert result.braked_joint_mask[0].tolist() == [True, True, True]
    assert not result.braked_joint_mask[1].any()
    assert torch.equal(result.applied_target[0, 4:], inputs["joint_pos"][0, 4:])
    assert torch.equal(
        result.applied_target[1].view(torch.int32),
        inputs["nominal_safe_target"][1].view(torch.int32),
    )


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_wrist_energy_brake_near_limit_hold_remains_slew_and_guard_safe(direction):
    inputs = _wrist_energy_brake_inputs()
    inputs["latch_remaining"][0] = 1
    inputs["joint_pos"][0, 4:] = direction * 0.995
    inputs["nominal_safe_target"][0, 4:] = direction * 0.98
    inputs["previous_applied_target"].copy_(inputs["nominal_safe_target"])
    inputs["joint_vel"][0, 4:] = -direction * 0.2

    result = _apply_wrist_energy_brake_target(**inputs)

    strict_lower = (
        inputs["soft_joint_pos_limits"][..., 0] + inputs["max_delta_joint_pos"]
    )
    strict_upper = (
        inputs["soft_joint_pos_limits"][..., 1] - inputs["max_delta_joint_pos"]
    )
    assert result.active_environment_mask.tolist() == [True]
    assert result.next_latch_remaining.tolist() == [0]
    assert result.attempted_joint_mask[0].tolist() == [True, True, True]
    assert result.braked_joint_mask[0].tolist() == [False, False, False]
    assert torch.all(
        (result.applied_target - inputs["joint_pos"]).abs()
        <= inputs["max_delta_joint_pos"] + 1e-6
    )
    assert torch.all(result.applied_target >= strict_lower)
    assert torch.all(result.applied_target <= strict_upper)
    torch.testing.assert_close(
        result.applied_target[0, 4:],
        torch.full((3,), direction * 0.98, dtype=torch.float32),
        rtol=0.0,
        atol=0.0,
    )


def test_wrist_energy_brake_no_trigger_preserves_nominal_target_bits():
    inputs = _wrist_energy_brake_inputs()
    inputs["joint_pos"].fill_(1.0941112)
    inputs["nominal_safe_target"].fill_(0.4359291)
    inputs["previous_applied_target"].copy_(inputs["nominal_safe_target"])
    # Even a large non-wrist sign reversal must not arm the wrist-group latch.
    inputs["previous_applied_target"][0, 0] = 1.7522933
    inputs["max_delta_joint_pos"].fill_(1.0)
    inputs["soft_joint_pos_limits"] = torch.tensor(
        [[[-3.0, 3.0]] * 7], dtype=torch.float32
    )
    nominal_bits = inputs["nominal_safe_target"].view(torch.int32).clone()

    result = _apply_wrist_energy_brake_target(**inputs)

    assert not result.trigger_joint_mask.any()
    assert not result.active_environment_mask.any()
    assert not result.attempted_joint_mask.any()
    assert not result.braked_joint_mask.any()
    assert torch.equal(result.applied_target.view(torch.int32), nominal_bits)
    assert torch.equal(result.target_shift, torch.zeros_like(result.target_shift))


@pytest.mark.parametrize(
    (
        "apply_index",
        "joint_pos_wrist",
        "joint_vel_wrist",
        "previous_target_wrist",
        "nominal_target_wrist",
        "expected_trigger",
        "expected_braked",
    ),
    [
        (
            896,
            [-0.4672307, 2.4151399, -0.1759258],
            [-0.1043425, 0.1041249, 0.0930457],
            [-0.4880932, 2.4360042, -0.1573215],
            [-0.4485214, 2.4252367, -0.1976758],
            [True, False, True],
            [False, True, False],
        ),
        (
            912,
            [-0.4592690, 2.4250145, -0.1891724],
            [0.0340723, 0.1042351, -0.0813407],
            [-0.4524480, 2.4458785, -0.2054473],
            [-0.4754810, 2.4467645, -0.1938981],
            [True, False, False],
            [False, True, True],
        ),
        (
            920,
            [-0.4638891, 2.4321313, -0.1904860],
            [-0.0588444, 0.1042782, -0.0162449],
            [-0.4756514, 2.4529948, -0.1937413],
            [-0.4842108, 2.4538813, -0.1732677],
            [False, False, True],
            [True, True, False],
        ),
    ],
)
def test_wrist_energy_brake_exact_v13_precursor_masks(
    apply_index,
    joint_pos_wrist,
    joint_vel_wrist,
    previous_target_wrist,
    nominal_target_wrist,
    expected_trigger,
    expected_braked,
):
    del apply_index
    inputs = _wrist_energy_brake_inputs()
    inputs["max_delta_joint_pos"][0, 4:] = torch.tensor(
        [2.61 / 120.0] * 3,
        dtype=torch.float32,
    )
    inputs["joint_pos"][0, 4:] = torch.tensor(
        joint_pos_wrist,
        dtype=torch.float32,
    )
    inputs["joint_vel"][0, 4:] = torch.tensor(
        joint_vel_wrist,
        dtype=torch.float32,
    )
    inputs["previous_applied_target"][0, 4:] = torch.tensor(
        previous_target_wrist,
        dtype=torch.float32,
    )
    inputs["nominal_safe_target"][0, 4:] = torch.tensor(
        nominal_target_wrist,
        dtype=torch.float32,
    )
    inputs["soft_joint_pos_limits"] = torch.tensor(
        [[[-4.0, 4.0]] * 7],
        dtype=torch.float32,
    )

    result = _apply_wrist_energy_brake_target(**inputs)

    assert result.trigger_joint_mask[0].tolist() == expected_trigger
    assert result.attempted_joint_mask[0].tolist() == expected_braked
    assert result.braked_joint_mask[0].tolist() == expected_braked
    assert result.active_environment_mask.tolist() == [True]
    assert result.next_latch_remaining.tolist() == [1]


class _LimitData:
    pass


class _LimitRootView:
    def __init__(self, limits):
        self.limits = limits
        self.max_velocities = torch.tensor(
            [PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S], dtype=torch.float32
        )
        self.max_forces = torch.tensor(
            [PANDA_EEF_JOINT_EFFORT_LIMITS], dtype=torch.float32
        )

    def get_dof_limits(self):
        return self.limits

    def get_dof_max_velocities(self):
        return self.max_velocities

    def get_dof_max_forces(self):
        return self.max_forces


class _LimitCfg:
    soft_joint_pos_limit_factor = 1.0


class _LimitAsset:
    device = "cpu"
    cfg = _LimitCfg()

    def __init__(self, hard_limits, *, corrupt_physx=False, corrupt_soft=False):
        self.data = _LimitData()
        self.data.joint_pos_limits = hard_limits.clone()
        self.data.soft_joint_pos_limits = _derive_isaac_soft_joint_position_limits(
            hard_limits,
            soft_limit_factor=self.cfg.soft_joint_pos_limit_factor,
        )
        self.root_physx_view = _LimitRootView(hard_limits.clone())
        self.write_count = 0
        self.corrupt_physx = corrupt_physx
        self.corrupt_soft = corrupt_soft

    def write_joint_position_limit_to_sim(
        self, limits, *, joint_ids, env_ids, warn_limit_violation
    ):
        assert env_ids is None
        assert warn_limit_violation is True
        self.write_count += 1
        self.data.joint_pos_limits[:, joint_ids, :] = limits
        self.root_physx_view.limits[:, joint_ids, :] = limits

        # Reproduce the pinned Isaac Lab implementation, including the
        # float32 midpoint/range roundoff that makes this buffer distinct from
        # the requested hard limits for panda_joint4 and panda_joint6.
        hard = self.data.joint_pos_limits
        joint_pos_mean = (hard[..., 0] + hard[..., 1]) / 2
        joint_pos_range = hard[..., 1] - hard[..., 0]
        factor = self.cfg.soft_joint_pos_limit_factor
        self.data.soft_joint_pos_limits[..., 0] = (
            joint_pos_mean - 0.5 * joint_pos_range * factor
        )
        self.data.soft_joint_pos_limits[..., 1] = (
            joint_pos_mean + 0.5 * joint_pos_range * factor
        )
        if self.corrupt_physx:
            self.root_physx_view.limits[0, 0, 0] += 1e-3
        if self.corrupt_soft:
            self.data.soft_joint_pos_limits[0, 0, 0] += 1e-3


class _SolverAttr:
    def __init__(self, value, *, authored=True):
        self.value = value
        self.authored = authored

    def HasAuthoredValueOpinion(self):
        return self.authored

    def Get(self):
        return self.value


class _SolverApi:
    def __init__(
        self, position, velocity, *, position_authored=True, velocity_authored=True
    ):
        self.position = _SolverAttr(position, authored=position_authored)
        self.velocity = _SolverAttr(velocity, authored=velocity_authored)

    def GetSolverPositionIterationCountAttr(self):
        return self.position

    def GetSolverVelocityIterationCountAttr(self):
        return self.velocity


class _SolverPrim:
    def __init__(self, path, api=None):
        self.path = path
        self.api = api

    def GetPath(self):
        return SimpleNamespace(pathString=self.path)

    def HasAPI(self, _schema):
        return self.api is not None


def _install_solver_schema_fakes(monkeypatch, apis):
    stage = object()
    asset_prims = [
        _SolverPrim(f"/World/envs/env_{index}/robot") for index in range(len(apis))
    ]
    roots = {
        prim.path: _SolverPrim(f"{prim.path}/panda_link0", api)
        for prim, api in zip(asset_prims, apis, strict=True)
    }
    monkeypatch.setattr(
        robust_ik.omni.usd,
        "get_context",
        lambda: SimpleNamespace(get_stage=lambda: stage),
    )
    monkeypatch.setattr(
        robust_ik.sim_utils,
        "find_matching_prims",
        lambda _expression, *, stage: asset_prims,
    )
    monkeypatch.setattr(
        robust_ik.sim_utils,
        "get_all_matching_child_prims",
        lambda path, *, predicate, stage: [roots[path]],
    )
    monkeypatch.setattr(
        robust_ik,
        "PhysxSchema",
        SimpleNamespace(PhysxArticulationAPI=lambda prim: prim.api),
    )
    monkeypatch.setattr(
        robust_ik,
        "UsdPhysics",
        SimpleNamespace(ArticulationRootAPI=object()),
    )
    return SimpleNamespace(
        cfg=SimpleNamespace(
            prim_path="/World/envs/env_.*/robot",
            articulation_root_prim_path=None,
        ),
        root_physx_view=SimpleNamespace(count=len(apis)),
    )


def test_solver_schema_readback_covers_every_authored_articulation_root(monkeypatch):
    asset = _install_solver_schema_fakes(
        monkeypatch,
        [_SolverApi(64, 1), _SolverApi(64, 1)],
    )

    position, velocity = robust_ik._read_articulation_solver_iteration_counts(asset)

    assert position == (64, 64)
    assert velocity == (1, 1)


@pytest.mark.parametrize(
    "api",
    [
        _SolverApi(64, 1, position_authored=False),
        _SolverApi(64, 1, velocity_authored=False),
    ],
)
def test_solver_schema_readback_rejects_fallback_but_unauthored_values(
    monkeypatch, api
):
    asset = _install_solver_schema_fakes(monkeypatch, [api])

    with pytest.raises(ValueError, match="unauthored"):
        robust_ik._read_articulation_solver_iteration_counts(asset)


def _canonical_limit_inputs():
    prewrite_hard = torch.tensor(
        [
            [
                [-2.8973, 2.8973],
                [-1.7628, 1.7628],
                [-2.8973, 2.8973],
                [-3.0718, -0.0698],
                [-2.8973, 2.8973],
                [-0.0175, 3.7525],
                [-2.8973, 2.8973],
            ]
        ],
        dtype=torch.float32,
    )
    outer = torch.tensor([PANDA_SOFT_JOINT_POS_LIMITS_RAD], dtype=torch.float32)
    assert not torch.equal(prewrite_hard, outer)
    assert prewrite_hard[0, 3, 1] != outer[0, 3, 1]
    assert prewrite_hard[0, 5, 0] != outer[0, 5, 0]
    assert torch.equal(
        _derive_isaac_soft_joint_position_limits(
            prewrite_hard,
            soft_limit_factor=1.0,
        ),
        outer,
    )
    max_delta = torch.tensor(
        [PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S], dtype=torch.float32
    ) * torch.tensor(1.0 / 120.0, dtype=torch.float32)
    return prewrite_hard, outer, max_delta


def test_eef_physx_inner_limits_are_written_once_and_read_back_exactly():
    prewrite_hard, outer, max_delta = _canonical_limit_inputs()
    asset = _LimitAsset(prewrite_hard)

    inner, derived_soft = _install_eef_physx_position_limits(
        asset,
        joint_ids=list(range(7)),
        outer_limits=outer,
        max_delta_joint_pos=max_delta,
        soft_limit_factor=1.0,
    )

    expected = torch.stack(
        (outer[..., 0] + max_delta, outer[..., 1] - max_delta), dim=-1
    )
    assert asset.write_count == 1
    assert torch.equal(inner, expected)
    assert torch.equal(asset.root_physx_view.get_dof_limits(), expected)
    assert torch.equal(asset.data.joint_pos_limits, expected)
    assert not torch.equal(asset.data.soft_joint_pos_limits, expected)
    assert torch.equal(asset.data.soft_joint_pos_limits, derived_soft)
    assert torch.equal(
        derived_soft,
        torch.tensor(
            [PANDA_PHYSX_DERIVED_SOFT_JOINT_POS_LIMITS_RAD],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(
        derived_soft,
        _derive_isaac_soft_joint_position_limits(
            expected,
            soft_limit_factor=1.0,
        ),
    )
    assert torch.equal(outer, torch.tensor([PANDA_SOFT_JOINT_POS_LIMITS_RAD]))


@pytest.mark.parametrize(
    ("asset_kwargs", "match"),
    [
        ({"corrupt_physx": True}, "PhysX position-limit readback"),
        ({"corrupt_soft": True}, "derived-soft position-limit readback"),
    ],
)
def test_eef_physx_limit_install_rejects_readback_mutation(asset_kwargs, match):
    prewrite_hard, outer, max_delta = _canonical_limit_inputs()
    with pytest.raises(ValueError, match=match):
        _install_eef_physx_position_limits(
            _LimitAsset(prewrite_hard, **asset_kwargs),
            joint_ids=list(range(7)),
            outer_limits=outer,
            max_delta_joint_pos=max_delta,
            soft_limit_factor=1.0,
        )


def test_eef_physx_limit_install_rejects_prewrite_outer_identity_drift():
    prewrite_hard, outer, max_delta = _canonical_limit_inputs()
    asset = _LimitAsset(prewrite_hard)
    asset.data.joint_pos_limits[0, 0, 0] += 1e-3
    with pytest.raises(ValueError, match="do not derive to the captured outer"):
        _install_eef_physx_position_limits(
            asset,
            joint_ids=list(range(7)),
            outer_limits=outer,
            max_delta_joint_pos=max_delta,
            soft_limit_factor=1.0,
        )


def test_safety_report_rejects_live_velocity_target_mutation(monkeypatch):
    prewrite_hard, outer, max_delta = _canonical_limit_inputs()
    asset = _LimitAsset(prewrite_hard)
    hard, derived_soft = _install_eef_physx_position_limits(
        asset,
        joint_ids=list(range(7)),
        outer_limits=outer,
        max_delta_joint_pos=max_delta,
        soft_limit_factor=1.0,
    )
    asset.data.joint_vel_target = torch.zeros((1, 7), dtype=torch.float32)
    asset.data.joint_vel_target[0, 4] = 1e-3

    action = _bare_robust_action()
    action._asset = asset
    action._physx_cfg = SimpleNamespace(solver_type=1)
    action._physx_solver_type = 1
    action._joint_ids = list(range(7))
    action._soft_joint_position_limits = outer
    action._physx_hard_joint_position_limits = hard
    action._physx_derived_soft_joint_position_limits = derived_soft
    action._zero_joint_velocity_target = torch.zeros((1, 7), dtype=torch.float32)
    action._joint_velocity_limits = asset.root_physx_view.max_velocities.clone()
    action._joint_effort_limits = asset.root_physx_view.max_forces.clone()
    action._solver_position_iteration_counts = (64,)
    action._solver_velocity_iteration_counts = (1,)
    monkeypatch.setattr(
        robust_ik,
        "_read_articulation_solver_iteration_counts",
        lambda _asset: ((64,), (1,)),
    )

    with pytest.raises(
        ValueError, match="live arm velocity target is not exactly zero"
    ):
        action.safety_report()


def test_safety_report_rejects_physx_solver_type_drift():
    action = _bare_robust_action()
    action._physx_cfg = SimpleNamespace(solver_type=0)
    action._physx_solver_type = 1

    with pytest.raises(ValueError, match="PhysX solver type drifted"):
        action.safety_report()


@pytest.mark.parametrize("limit_field", ["max_velocities", "max_forces"])
def test_safety_report_rejects_live_physx_joint_limit_drift(monkeypatch, limit_field):
    prewrite_hard, outer, max_delta = _canonical_limit_inputs()
    asset = _LimitAsset(prewrite_hard)
    hard, derived_soft = _install_eef_physx_position_limits(
        asset,
        joint_ids=list(range(7)),
        outer_limits=outer,
        max_delta_joint_pos=max_delta,
        soft_limit_factor=1.0,
    )
    asset.data.joint_vel_target = torch.zeros((1, 7), dtype=torch.float32)

    action = _bare_robust_action()
    action._asset = asset
    action._physx_cfg = SimpleNamespace(solver_type=1)
    action._physx_solver_type = 1
    action._joint_ids = list(range(7))
    action._soft_joint_position_limits = outer
    action._physx_hard_joint_position_limits = hard
    action._physx_derived_soft_joint_position_limits = derived_soft
    action._zero_joint_velocity_target = torch.zeros((1, 7), dtype=torch.float32)
    action._joint_velocity_limits = asset.root_physx_view.max_velocities.clone()
    action._joint_effort_limits = asset.root_physx_view.max_forces.clone()
    action._solver_position_iteration_counts = (64,)
    action._solver_velocity_iteration_counts = (1,)
    monkeypatch.setattr(
        robust_ik,
        "_read_articulation_solver_iteration_counts",
        lambda _asset: ((64,), (1,)),
    )
    getattr(asset.root_physx_view, limit_field)[0, 0] += 1e-3

    with pytest.raises(ValueError, match="live PhysX joint-limit readback drifted"):
        action.safety_report()


def test_joint_target_safety_rejects_nonfinite_and_out_of_limit_current_state():
    with pytest.raises(DifferentialIKNumericalError, match="non-finite raw target"):
        _require_finite(torch.tensor([float("nan")]), field="raw target")

    soft_limits = torch.tensor([[[-1.0, 1.0]] * 7])
    within_float_tolerance = torch.tensor([[-1.0 - 1e-6] + [0.0] * 6])
    violation = _require_current_joint_position_in_soft_limits(
        within_float_tolerance, soft_limits
    )
    assert violation[0, 0] > 0.0

    outside = torch.tensor([[-1.0 - 1e-3] + [0.0] * 6])
    with pytest.raises(DifferentialIKNumericalError, match="outside live soft"):
        _require_current_joint_position_in_soft_limits(outside, soft_limits)


@pytest.mark.parametrize(
    ("norm", "expected"),
    [
        (0.0, False),
        (1e-12, False),
        (1.0 + EEF_QUATERNION_UNIT_NORM_TOLERANCE + 1e-4, False),
        (1.0 - EEF_QUATERNION_UNIT_NORM_TOLERANCE / 2, True),
        (1.0 + EEF_QUATERNION_UNIT_NORM_TOLERANCE / 2, True),
        (1.0, True),
    ],
)
def test_eef_quaternion_named_unit_norm_guard(norm, expected):
    quaternion = torch.tensor([[norm, 0.0, 0.0, 0.0]], dtype=torch.float64)

    norms, valid = _eef_quaternion_norm_is_valid(quaternion)

    assert norms.item() == norm
    assert valid.item() is expected


def test_eef_quaternion_named_unit_norm_guard_rejects_nonfinite():
    for value in (float("nan"), float("inf"), torch.finfo(torch.float32).max):
        _, valid = _eef_quaternion_norm_is_valid(torch.tensor([[value, 0.0, 0.0, 0.0]]))
        assert not valid.item()


def test_huge_finite_diagnostic_scalars_remain_strict_json_finite():
    action = _bare_robust_action()
    action._apply_call_count = 1
    action._active_episode_index = 0
    action._decimation = 8
    diagnostic = action._diagnostic_record(
        kind="current_eef_quaternion_invariant_abort",
        joint_pos=torch.zeros(1, 7),
        raw_delta=None,
        raw_target=None,
        safe_target=None,
        pose_error=torch.full((1, 6), torch.finfo(torch.float32).max),
        jacobian=torch.full((1, 6, 7), torch.finfo(torch.float32).max),
        eef_quaternion_norm=torch.tensor(
            [torch.finfo(torch.float32).max], dtype=torch.float64
        ),
    )

    assert torch.isfinite(
        torch.tensor(diagnostic["pose_error_norm"], dtype=torch.float64)
    )
    assert torch.isfinite(
        torch.tensor(diagnostic["jacobian_max_abs"], dtype=torch.float64)
    )
    assert torch.isfinite(
        torch.tensor(diagnostic["eef_quaternion_norm"], dtype=torch.float64)
    )


def _current_joint_velocity_abort_action():
    action = _bare_robust_action()
    action._current_joint_velocity_abort = None
    action._active_episode_index = 0
    action._decimation = 8
    action._apply_call_count = 940
    action._joint_names = [f"panda_joint{index}" for index in range(1, 8)]
    action._joint_velocity_limits = torch.tensor(
        [PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S], dtype=torch.float32
    )
    return action


def test_current_joint_velocity_abort_records_exact_signed_terminal_evidence():
    action = _current_joint_velocity_abort_action()
    joint_vel = torch.tensor(
        [[0.25, -0.5, 1.0, -1.5, -2.75, 0.0, 2.9]], dtype=torch.float32
    )
    expected_excess = torch.clamp(
        joint_vel.abs() - action._joint_velocity_limits, min=0.0
    )
    exceeded = joint_vel.abs() > (
        action._joint_velocity_limits + JOINT_VELOCITY_LIMIT_TOLERANCE_RAD_S
    )

    evidence = action._record_current_joint_velocity_abort(
        joint_vel=joint_vel,
        exceeded_joint_mask=exceeded,
    )

    assert set(evidence) == {
        "profile",
        "episode_index",
        "policy_step",
        "physics_substep",
        "joint_names",
        "joint_velocity_rad_s",
        "joint_velocity_limit_rad_s",
        "joint_velocity_limit_tolerance_rad_s",
        "joint_velocity_limit_excess_rad_s",
        "exceeded_joint_mask",
    }
    assert evidence["profile"] == CURRENT_JOINT_VELOCITY_ABORT_EVIDENCE_PROFILE
    assert evidence["episode_index"] == 0
    assert evidence["policy_step"] == 117
    assert evidence["physics_substep"] == 3
    assert evidence["joint_names"] == action._joint_names
    torch.testing.assert_close(
        torch.tensor(evidence["joint_velocity_rad_s"]),
        joint_vel[0],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        torch.tensor(evidence["joint_velocity_limit_rad_s"]),
        action._joint_velocity_limits[0],
        rtol=0.0,
        atol=0.0,
    )
    assert evidence["joint_velocity_limit_tolerance_rad_s"] == (
        JOINT_VELOCITY_LIMIT_TOLERANCE_RAD_S
    )
    torch.testing.assert_close(
        torch.tensor(evidence["joint_velocity_limit_excess_rad_s"]),
        expected_excess[0],
        rtol=0.0,
        atol=0.0,
    )
    assert evidence["exceeded_joint_mask"] == exceeded[0].tolist()
    assert evidence["joint_velocity_rad_s"][4] < 0.0
    assert evidence["exceeded_joint_mask"] == [
        False,
        False,
        False,
        False,
        True,
        False,
        True,
    ]
    digest = current_joint_velocity_abort_evidence_sha256(evidence)
    message = format_current_joint_velocity_abort_message(evidence)
    assert len(digest) == 64
    assert message.endswith(f"evidence_sha256={digest})")
    json.dumps(evidence, allow_nan=False)

    with pytest.raises(ValueError, match="recorded twice"):
        action._record_current_joint_velocity_abort(
            joint_vel=joint_vel,
            exceeded_joint_mask=exceeded,
        )


def test_current_joint_velocity_abort_rejects_mask_drift_and_nonfinite_state():
    action = _current_joint_velocity_abort_action()
    joint_vel = action._joint_velocity_limits.clone()
    joint_vel[0, 0] += 0.1
    exceeded = torch.zeros_like(joint_vel, dtype=torch.bool)

    with pytest.raises(ValueError, match="mask drift"):
        action._record_current_joint_velocity_abort(
            joint_vel=joint_vel,
            exceeded_joint_mask=exceeded,
        )
    assert action._current_joint_velocity_abort is None

    joint_vel[0, 0] = torch.nan
    with pytest.raises(DifferentialIKNumericalError, match="evidence is non-finite"):
        action._record_current_joint_velocity_abort(
            joint_vel=joint_vel,
            exceeded_joint_mask=exceeded,
        )
    assert action._current_joint_velocity_abort is None


@pytest.mark.parametrize("joint_index", [0, 4])
def test_current_joint_velocity_abort_uses_direct_float32_threshold(joint_index):
    action = _current_joint_velocity_abort_action()
    threshold = action._joint_velocity_limits + (JOINT_VELOCITY_LIMIT_TOLERANCE_RAD_S)
    joint_vel = torch.zeros_like(action._joint_velocity_limits)
    joint_vel[0, joint_index] = threshold[0, joint_index]
    trigger_index = 1 if joint_index == 0 else 6
    joint_vel[0, trigger_index] = torch.nextafter(
        threshold[0, trigger_index],
        torch.tensor(float("inf"), dtype=torch.float32),
    )
    exceeded = joint_vel.abs() > threshold

    evidence = action._record_current_joint_velocity_abort(
        joint_vel=joint_vel,
        exceeded_joint_mask=exceeded,
    )

    assert evidence["exceeded_joint_mask"][joint_index] is False
    assert evidence["exceeded_joint_mask"][trigger_index] is True
    assert evidence["joint_velocity_limit_excess_rad_s"][joint_index] > (
        JOINT_VELOCITY_LIMIT_TOLERANCE_RAD_S
    )


def test_current_limit_and_slew_invariants_abort_before_physx_target_setter():
    source = inspect.getsource(RobustDifferentialInverseKinematicsAction.apply_actions)
    setter = source.index("self._set_targets_and_commit_gripper_close_arm_interlock(")
    transaction_source = inspect.getsource(
        RobustDifferentialInverseKinematicsAction._set_targets_and_commit_gripper_close_arm_interlock
    )
    assert source.index("if not current_quaternion_valid:") < setter
    assert source.index("if not desired_quaternion_valid:") < setter
    assert source.index("self._ik_controller.ee_quat_des") < setter
    assert source.index(
        '_require_finite(current_state, field="current EEF/joint state")'
    ) < source.index("self._max_current_joint_soft_limit_violation")
    current_guard = source.index("if not current_joint_valid:")
    current_counter = source.index(
        "self._current_joint_limit_abort_count += current_joint_invalid.sum()"
    )
    current_diagnostic = source.index('kind="current_joint_limit_abort"')
    jacobian_compute = source.index("jacobian = self._compute_frame_jacobian()")
    assert (
        current_guard < current_counter < current_diagnostic < jacobian_compute < setter
    )
    velocity_guard = source.index("if not current_joint_velocity_valid:")
    velocity_counter = source.index(
        "self._invariant_abort_count += current_joint_velocity_invalid.sum()"
    )
    velocity_record = source.index("self._record_current_joint_velocity_abort(")
    velocity_diagnostic = source.index('kind="current_joint_velocity_limit_abort"')
    velocity_raise = source.index(
        "format_current_joint_velocity_abort_message(velocity_abort)"
    )
    velocity_maximum = source.index("self._max_abs_joint_vel = torch.maximum(")
    current_finite_guard = source.index("if not current_finite:")
    desired_finite_guard = source.index("if not desired_finite:")
    current_quaternion_guard = source.index("if not current_quaternion_valid:")
    desired_quaternion_guard = source.index("if not desired_quaternion_valid:")
    assert (
        current_finite_guard
        < velocity_maximum
        < velocity_guard
        < velocity_counter
        < velocity_record
        < velocity_diagnostic
        < velocity_raise
        < desired_finite_guard
        < current_quaternion_guard
        < desired_quaternion_guard
        < current_guard
        < jacobian_compute
        < setter
    )
    assert source.index("if target_invalid:") < setter
    assert source.index("if slew_invalid:") < setter
    assert transaction_source.index("self._asset.set_joint_velocity_target") < (
        transaction_source.index("self._asset.set_joint_position_target")
    )
    assert "write_joint_state_to_sim" not in source

    init_source = inspect.getsource(RobustDifferentialInverseKinematicsAction.__init__)
    assert "_install_eef_physx_position_limits" in init_source
    assert "write_joint_state_to_sim" not in init_source

    report_source = inspect.getsource(
        RobustDifferentialInverseKinematicsAction.safety_report
    )
    assert "self._asset.data.joint_vel_target" in report_source
    assert "torch.equal(live_velocity_target" in report_source
    assert '"arm_velocity_target_rad_s": live_velocity_target[0]' in report_source
    assert '"current_joint_velocity_abort": (' in report_source

    reset_source = inspect.getsource(
        RobustDifferentialInverseKinematicsAction._reset_episode_safety_state
    )
    assert "self._current_joint_velocity_abort: dict[str, object] | None = None" in (
        reset_source
    )


@pytest.mark.parametrize(
    "simultaneous_later_guard",
    [
        "if not desired_finite:",
        "if not current_quaternion_valid:",
        "if not desired_quaternion_valid:",
        "if not current_joint_valid:",
    ],
)
def test_overlimit_velocity_has_durable_precedence_over_simultaneous_finite_guard(
    simultaneous_later_guard,
):
    source = inspect.getsource(RobustDifferentialInverseKinematicsAction.apply_actions)
    current_finite = source.index("if not current_finite:")
    maxima = source.index("self._max_abs_joint_vel = torch.maximum(")
    velocity_guard = source.index("if not current_joint_velocity_valid:")
    evidence_capture = source.index("self._record_current_joint_velocity_abort(")
    durable_guard = source.index('kind="current_joint_velocity_limit_abort"')
    digest_bound_raise = source.index(
        "format_current_joint_velocity_abort_message(velocity_abort)"
    )
    later_guard = source.index(simultaneous_later_guard)

    assert (
        current_finite
        < maxima
        < velocity_guard
        < evidence_capture
        < durable_guard
        < digest_bound_raise
        < later_guard
    )


def test_eef_pose_config_installs_robust_action_term():
    from polaris.config import LAP_EEF_FRAME
    from polaris.environments.droid_cfg import EefPoseActionCfg, SceneCfg

    cfg = EefPoseActionCfg()
    scene_cfg = SceneCfg()

    assert cfg.arm.class_type is RobustDifferentialInverseKinematicsAction
    assert cfg.arm.enable_failure_substep_trace is False
    assert cfg.arm.body_name == LAP_EEF_FRAME == "panda_link8"
    frame_cfg = scene_cfg.lap_ee_frame
    target_cfg = frame_cfg.target_frames[0]
    assert frame_cfg.prim_path.endswith("/robot/panda_link0")
    assert tuple(frame_cfg.source_frame_offset.pos) == (0.0, 0.0, 0.0)
    assert tuple(frame_cfg.source_frame_offset.rot) == (1.0, 0.0, 0.0, 0.0)
    assert target_cfg.prim_path.endswith(f"/robot/{LAP_EEF_FRAME}")
    assert tuple(target_cfg.offset.pos) == (0.0, 0.0, 0.0)
    assert tuple(target_cfg.offset.rot) == (1.0, 0.0, 0.0, 0.0)
    assert tuple(cfg.arm.body_offset.pos) == (0.0, 0.0, 0.0)
    assert tuple(cfg.arm.body_offset.rot) == (1.0, 0.0, 0.0, 0.0)
    assert scene_cfg.ee_frame.target_frames[0].prim_path.endswith(
        "/robot/Gripper/Robotiq_2F_85/base_link"
    )


def test_eef_wrist_energy_brake_config_defaults_disabled():
    from polaris.environments.droid_cfg import EefPoseActionCfg

    assert EefPoseActionCfg().arm.enable_wrist_energy_brake is False


def test_wrist_energy_brake_state_resets_and_commits_only_after_target_setters():
    episode_reset_source = inspect.getsource(
        RobustDifferentialInverseKinematicsAction._reset_episode_safety_state
    )
    state_reset_source = inspect.getsource(
        RobustDifferentialInverseKinematicsAction._reset_wrist_energy_brake_state
    )
    standard_reset_source = inspect.getsource(
        RobustDifferentialInverseKinematicsAction.reset
    )
    begin_source = inspect.getsource(
        RobustDifferentialInverseKinematicsAction.begin_safety_episode
    )
    apply_source = inspect.getsource(
        RobustDifferentialInverseKinematicsAction.apply_actions
    )

    assert "self._reset_wrist_energy_brake_state()" in episode_reset_source
    assert "self._wrist_energy_brake_latch_remaining[selected] = 0" in (
        state_reset_source
    )
    assert "self._wrist_energy_brake_previous_applied_target[selected] = 0.0" in (
        state_reset_source
    )
    assert "self._wrist_energy_brake_previous_target_valid[selected] = False" in (
        state_reset_source
    )
    assert "self._wrist_energy_brake_reversal_detection_armed[selected] = False" in (
        state_reset_source
    )
    assert "super().reset(env_ids)" in standard_reset_source
    assert "self._reset_wrist_energy_brake_state(env_ids)" in standard_reset_source
    assert "self._reset_episode_safety_state(episode_index=episode_index)" in (
        begin_source
    )
    transaction = apply_source.index(
        "self._set_targets_and_commit_gripper_close_arm_interlock("
    )
    latch_commit = apply_source.index("self._wrist_energy_brake_latch_remaining.copy_(")
    target_commit = apply_source.index(
        "self._wrist_energy_brake_previous_applied_target.copy_(safe_target)"
    )
    validity_commit = apply_source.index(
        "self._wrist_energy_brake_previous_target_valid.fill_(True)"
    )
    assert transaction < latch_commit < target_commit
    assert target_commit < validity_commit


def test_standard_action_reset_clears_selected_candidate_state():
    reset_source = inspect.getsource(RobustDifferentialInverseKinematicsAction.reset)
    assert "_current_joint_velocity_abort" not in reset_source
    assert "_reset_episode_safety_state" not in reset_source

    action = _bare_robust_action()
    action._raw_actions = torch.ones((1, 7), dtype=torch.float32)
    action._wrist_energy_brake_enabled = True
    action._wrist_energy_brake_latch_remaining = torch.ones(1, dtype=torch.int64)
    action._wrist_energy_brake_previous_applied_target = torch.ones((1, 7))
    action._wrist_energy_brake_previous_target_valid = torch.ones(1, dtype=torch.bool)
    action._wrist_energy_brake_reversal_detection_armed = torch.ones(
        1, dtype=torch.bool
    )

    action.reset([0])

    assert torch.equal(action._raw_actions, torch.zeros_like(action._raw_actions))
    assert action._wrist_energy_brake_latch_remaining.tolist() == [0]
    assert not action._wrist_energy_brake_previous_applied_target.any()
    assert not action._wrist_energy_brake_previous_target_valid.any()
    assert not action._wrist_energy_brake_reversal_detection_armed.any()
    action.__del__()


def test_eef_velocity_and_effort_limits_are_scoped_to_eef_setup():
    from polaris.environments.robot_cfg import NVIDIA_DROID
    from polaris.environments.robot_cfg import configure_eef_pose_joint_safety

    native_cfg = NVIDIA_DROID.copy()
    eef_cfg = NVIDIA_DROID.copy()
    assert native_cfg.actuators["panda_shoulder"].velocity_limit_sim is None
    assert native_cfg.actuators["panda_forearm"].velocity_limit_sim is None
    assert native_cfg.actuators["gripper"].velocity_limit_sim is None
    assert native_cfg.spawn.articulation_props.solver_position_iteration_count == 64
    assert native_cfg.spawn.articulation_props.solver_velocity_iteration_count == 0

    eval_source = (Path(__file__).parents[1] / "scripts" / "eval.py").read_text()
    eef_branch = eval_source.index('if eval_args.control_mode == "eef-pose":')
    configure_call = eval_source.index(
        "configure_eef_pose_joint_safety(\n            env_cfg.scene.robot,"
    )
    native_branch = eval_source.index(
        'elif eval_args.control_mode != "joint-position":'
    )
    assert eef_branch < configure_call < native_branch
    gripper_limit_enable = eval_source.index("enable_gripper_velocity_limit=is_ego_lap")
    assert configure_call < gripper_limit_enable < native_branch
    reset = eval_source.index("obs, info = env.reset(")
    gripper_install = eval_source.index("install_or_validate_gripper_runtime()", reset)
    first_step = eval_source.index("obs, rew, term, trunc, info = env.step(")
    assert reset < gripper_install < first_step

    native_physx_cfg = SimpleNamespace(solver_type=0)
    eef_physx_cfg = SimpleNamespace(solver_type=0)
    configure_eef_pose_joint_safety(eef_cfg, physx_cfg=eef_physx_cfg)

    assert eef_cfg.actuators["panda_shoulder"].velocity_limit_sim == 2.175
    assert eef_cfg.actuators["panda_shoulder"].effort_limit_sim == 87.0
    assert eef_cfg.actuators["panda_forearm"].velocity_limit_sim == 2.61
    assert eef_cfg.actuators["panda_forearm"].effort_limit_sim == 12.0
    assert eef_cfg.actuators["gripper"].velocity_limit_sim is None
    assert eef_cfg.spawn.articulation_props.solver_position_iteration_count == 64
    assert eef_cfg.spawn.articulation_props.solver_velocity_iteration_count == 1
    assert eef_physx_cfg.solver_type == 1
    assert native_cfg.actuators["panda_shoulder"].velocity_limit_sim is None
    assert native_cfg.actuators["panda_forearm"].velocity_limit_sim is None
    assert native_cfg.actuators["gripper"].velocity_limit_sim is None
    assert native_cfg.spawn.articulation_props.solver_position_iteration_count == 64
    assert native_cfg.spawn.articulation_props.solver_velocity_iteration_count == 0
    assert native_physx_cfg.solver_type == 0


def test_eef_gripper_velocity_limit_candidate_is_explicit_and_opt_in():
    from polaris.environments.robot_cfg import NVIDIA_DROID
    from polaris.environments.robot_cfg import configure_eef_pose_joint_safety

    native_cfg = NVIDIA_DROID.copy()
    candidate_cfg = NVIDIA_DROID.copy()
    candidate_physx = SimpleNamespace(solver_type=0)

    configure_eef_pose_joint_safety(
        candidate_cfg,
        physx_cfg=candidate_physx,
        enable_gripper_velocity_limit=True,
    )

    assert candidate_cfg.actuators["gripper"].velocity_limit == 5.0
    assert candidate_cfg.actuators["gripper"].velocity_limit_sim == 5.0
    assert candidate_cfg.actuators["gripper"].effort_limit == 200.0
    assert candidate_cfg.actuators["gripper"].effort_limit_sim == 200.0
    assert candidate_cfg.actuators["gripper"].stiffness is None
    assert candidate_cfg.actuators["gripper"].damping is None
    assert candidate_cfg.spawn.articulation_props.solver_position_iteration_count == 64
    assert candidate_cfg.spawn.articulation_props.solver_velocity_iteration_count == 1
    assert candidate_physx.solver_type == 1
    assert native_cfg.actuators["gripper"].velocity_limit_sim is None
    assert native_cfg.spawn.articulation_props.solver_velocity_iteration_count == 0


def test_eef_gripper_velocity_limit_candidate_rejects_config_drift():
    from polaris.environments.robot_cfg import NVIDIA_DROID
    from polaris.environments.robot_cfg import configure_eef_pose_joint_safety

    with pytest.raises(ValueError, match="enable flag must be bool"):
        configure_eef_pose_joint_safety(
            NVIDIA_DROID.copy(),
            physx_cfg=SimpleNamespace(solver_type=0),
            enable_gripper_velocity_limit="true",
        )

    cfg = NVIDIA_DROID.copy()
    cfg.actuators["gripper"].joint_names_expr = ["wrong_joint"]
    with pytest.raises(ValueError, match="pinned EEF"):
        configure_eef_pose_joint_safety(
            cfg,
            physx_cfg=SimpleNamespace(solver_type=0),
            enable_gripper_velocity_limit=True,
        )


def test_eef_pose_config_rejects_missing_articulation_properties():
    from polaris.environments.robot_cfg import NVIDIA_DROID
    from polaris.environments.robot_cfg import configure_eef_pose_joint_safety

    cfg = NVIDIA_DROID.copy()
    cfg.spawn.articulation_props = None
    with pytest.raises(ValueError, match="no articulation properties"):
        configure_eef_pose_joint_safety(
            cfg,
            physx_cfg=SimpleNamespace(solver_type=0),
        )


def _failure_substep_trace_action(
    *, episode_index=0, data_overrides=None, readback_overrides=None
):
    action = _bare_robust_action()
    action._failure_substep_trace_enabled = True
    action._num_joints = 7
    action._joint_ids = list(range(7))
    action._joint_names = [f"panda_joint{index + 1}" for index in range(7)]
    action._decimation = robust_ik.FAILURE_SUBSTEP_TRACE_DECIMATION
    action._active_episode_index = episode_index
    action._apply_call_count = 0
    data = SimpleNamespace(
        joint_pos=torch.zeros((1, 7), dtype=torch.float32),
        joint_stiffness=torch.full((1, 7), 400.0, dtype=torch.float32),
        joint_damping=torch.full((1, 7), 80.0, dtype=torch.float32),
        joint_effort_limits=torch.tensor(
            [PANDA_EEF_JOINT_EFFORT_LIMITS], dtype=torch.float32
        ),
        joint_effort_target=torch.zeros((1, 7), dtype=torch.float32),
        computed_torque=torch.zeros((1, 7), dtype=torch.float32),
        applied_torque=torch.zeros((1, 7), dtype=torch.float32),
    )
    for field, value in (data_overrides or {}).items():
        setattr(data, field, value)
    readbacks = {
        "joint_stiffness": data.joint_stiffness.clone(),
        "joint_damping": data.joint_damping.clone(),
        "joint_effort_limits": data.joint_effort_limits.clone(),
    }
    readbacks.update(readback_overrides or {})
    root_physx_view = SimpleNamespace(
        get_dof_stiffnesses=lambda: readbacks["joint_stiffness"],
        get_dof_dampings=lambda: readbacks["joint_damping"],
        get_dof_max_forces=lambda: readbacks["joint_effort_limits"],
    )
    action._asset = SimpleNamespace(
        data=data,
        device="cpu",
        root_physx_view=root_physx_view,
    )
    action._initialize_failure_substep_trace()
    action._reset_failure_substep_trace_state()
    return action


def _stage_failure_substep_trace(action, apply_index):
    offset = float(apply_index * 10)
    joint_pos = torch.arange(7, dtype=torch.float32).unsqueeze(0) + offset
    joint_vel = joint_pos + 10.0
    previous_target = joint_pos + 20.0
    raw_target = joint_pos + 30.0
    new_target = joint_pos + 40.0
    new_velocity_target = torch.zeros_like(joint_pos)
    current_position = torch.tensor(
        [[offset + 0.1, offset + 0.2, offset + 0.3]], dtype=torch.float32
    )
    current_quaternion = torch.tensor(
        [[1.0, offset + 0.4, offset + 0.5, offset + 0.6]],
        dtype=torch.float32,
    )
    desired_position = current_position + 1.0
    desired_quaternion = current_quaternion + 2.0
    pose_error = torch.arange(6, dtype=torch.float32).unsqueeze(0) + offset
    action._apply_call_count = apply_index + 1
    action._stage_failure_substep_trace(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        previous_joint_pos_target=previous_target,
        raw_dls_joint_pos_target=raw_target,
        new_joint_pos_target=new_target,
        new_joint_vel_target=new_velocity_target,
        new_joint_effort_target=action._asset.data.joint_effort_target,
        current_eef_position=current_position,
        current_eef_quaternion=current_quaternion,
        desired_eef_position=desired_position,
        desired_eef_quaternion=desired_quaternion,
        pose_error=pose_error,
    )
    return SimpleNamespace(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        previous_target=previous_target,
        raw_target=raw_target,
        new_target=new_target,
        new_velocity_target=new_velocity_target,
        new_effort_target=action._asset.data.joint_effort_target.clone(),
        current_position=current_position,
        current_quaternion=current_quaternion,
        desired_position=desired_position,
        desired_quaternion=desired_quaternion,
        pose_error=pose_error,
    )


def _finalize_failure_substep_trace(action, staged, apply_index):
    del apply_index
    preclip_effort = (
        action._asset.data.joint_stiffness * (staged.new_target - staged.joint_pos)
        + action._asset.data.joint_damping
        * (staged.new_velocity_target - staged.joint_vel)
        + staged.new_effort_target
    )
    postclip_effort = torch.clamp(
        preclip_effort,
        min=-action._asset.data.joint_effort_limits,
        max=action._asset.data.joint_effort_limits,
    )
    action._asset.data.computed_torque = preclip_effort
    action._asset.data.applied_torque = postclip_effort
    post_joint_pos = staged.joint_pos + 0.25
    post_joint_vel = staged.joint_vel - 0.5
    action._finalize_pending_failure_substep_trace(
        post_joint_pos=post_joint_pos,
        post_joint_vel=post_joint_vel,
    )
    return SimpleNamespace(
        post_joint_pos=post_joint_pos,
        post_joint_vel=post_joint_vel,
        preclip_effort=preclip_effort,
        postclip_effort=postclip_effort,
    )


def _trace_values(entry, field):
    return torch.tensor(entry[field]["values"], dtype=torch.float32).unsqueeze(0)


def test_failure_substep_trace_is_disabled_by_default_and_separate_from_safety():
    action = _bare_robust_action()
    action._failure_substep_trace_enabled = False
    with pytest.raises(ValueError, match="trace is disabled"):
        action.failure_substep_trace(episode_index=0)

    report_source = inspect.getsource(
        RobustDifferentialInverseKinematicsAction.safety_report
    )
    assert "failure_substep_trace" not in report_source
    apply_source = inspect.getsource(
        RobustDifferentialInverseKinematicsAction.apply_actions
    )
    assert apply_source.index("self._finalize_pending_failure_substep_trace(") < (
        apply_source.index("self._apply_call_count += 1")
    )
    assert apply_source.index(
        "self._set_targets_and_commit_gripper_close_arm_interlock("
    ) < apply_source.index("self._stage_failure_substep_trace(")
    transaction_source = inspect.getsource(
        RobustDifferentialInverseKinematicsAction._set_targets_and_commit_gripper_close_arm_interlock
    )
    assert transaction_source.index("self._asset.set_joint_velocity_target(") < (
        transaction_source.index("self._asset.set_joint_position_target(")
    )
    assert "new_joint_vel_target=self._zero_joint_velocity_target" in apply_source
    assert "new_joint_effort_target=self._asset.data.joint_effort_target[" in (
        apply_source
    )
    for method in (
        RobustDifferentialInverseKinematicsAction._stage_failure_substep_trace,
        RobustDifferentialInverseKinematicsAction._finalize_pending_failure_substep_trace,
    ):
        hot_path_source = inspect.getsource(method)
        assert ".cpu(" not in hot_path_source
        assert ".tolist(" not in hot_path_source
        assert ".item(" not in hot_path_source


def test_failure_substep_trace_runner_contract_matches_controller():
    assert (
        boundary_smoke.FAILURE_SUBSTEP_TRACE_PROFILE
        == robust_ik.FAILURE_SUBSTEP_TRACE_PROFILE
    )
    assert (
        boundary_smoke.FAILURE_SUBSTEP_TRACE_CAPACITY
        == robust_ik.FAILURE_SUBSTEP_TRACE_CAPACITY
    )
    assert (
        boundary_smoke.FAILURE_SUBSTEP_TRACE_EFFORT_SEMANTICS
        == robust_ik.FAILURE_SUBSTEP_TRACE_EFFORT_SEMANTICS
    )
    assert (
        boundary_smoke.FAILURE_SUBSTEP_TRACE_PHASE_CONTRACT
        == robust_ik.FAILURE_SUBSTEP_TRACE_PHASE_CONTRACT
    )
    assert (
        boundary_smoke.FAILURE_SUBSTEP_TRACE_VECTOR_WIDTHS
        == robust_ik.FAILURE_SUBSTEP_TRACE_VECTOR_WIDTHS
    )


def test_failure_substep_trace_exports_closed_causal_schema():
    action = _failure_substep_trace_action(episode_index=3)
    staged = _stage_failure_substep_trace(action, apply_index=0)
    finalized = _finalize_failure_substep_trace(action, staged, apply_index=0)

    report = action.failure_substep_trace(episode_index=3)

    assert set(report) == {
        "schema_version",
        "profile",
        "episode_index",
        "capacity",
        "policy_step_capacity",
        "decimation",
        "joint_names",
        "joint_drive_stiffness",
        "joint_drive_damping",
        "joint_effort_limits",
        "effort_semantics",
        "phase_contract",
        "completed_entry_count",
        "total_completed_entry_count",
        "dropped_prefix_entry_count",
        "pending_entry_count",
        "pending_apply_index",
        "entries",
    }
    assert report["schema_version"] == 1
    assert report["profile"] == robust_ik.FAILURE_SUBSTEP_TRACE_PROFILE
    assert report["capacity"] == 64
    assert report["policy_step_capacity"] == 8
    assert report["decimation"] == 8
    assert report["joint_drive_stiffness"] == [400.0] * 7
    assert report["joint_drive_damping"] == [80.0] * 7
    assert report["joint_effort_limits"] == list(PANDA_EEF_JOINT_EFFORT_LIMITS)
    assert report["phase_contract"] == robust_ik.FAILURE_SUBSTEP_TRACE_PHASE_CONTRACT
    assert report["phase_contract"]["new_effort_target"] == (
        "zero_feedforward_live_at_write_data_to_sim_v1"
    )
    assert report["effort_semantics"] == (
        robust_ik.FAILURE_SUBSTEP_TRACE_EFFORT_SEMANTICS
    )
    assert report["completed_entry_count"] == 1
    assert report["total_completed_entry_count"] == 1
    assert report["dropped_prefix_entry_count"] == 0
    assert report["pending_entry_count"] == 0
    assert report["pending_apply_index"] is None

    entry = report["entries"][0]
    assert set(entry) == {
        "apply_index",
        "policy_step",
        "physics_substep",
        *robust_ik.FAILURE_SUBSTEP_TRACE_VECTOR_WIDTHS,
    }
    assert (entry["apply_index"], entry["policy_step"], entry["physics_substep"]) == (
        0,
        0,
        0,
    )
    for field, width in robust_ik.FAILURE_SUBSTEP_TRACE_VECTOR_WIDTHS.items():
        assert set(entry[field]) == {"values", "finite_mask", "finite_count"}
        assert len(entry[field]["values"]) == width
        assert entry[field]["finite_mask"] == [True] * width
        assert entry[field]["finite_count"] == width

    expected_vectors = {
        "joint_pos_rad": staged.joint_pos,
        "joint_vel_rad_s": staged.joint_vel,
        "post_joint_pos_rad": finalized.post_joint_pos,
        "post_joint_vel_rad_s": finalized.post_joint_vel,
        "delta_joint_pos_rad": finalized.post_joint_pos - staged.joint_pos,
        "delta_joint_vel_rad_s": finalized.post_joint_vel - staged.joint_vel,
        "previous_joint_pos_target_rad": staged.previous_target,
        "raw_dls_joint_pos_target_rad": staged.raw_target,
        "new_joint_pos_target_rad": staged.new_target,
        "new_joint_vel_target_rad_s": staged.new_velocity_target,
        "new_joint_effort_target_nm": staged.new_effort_target,
        "current_eef_position_m": staged.current_position,
        "current_eef_quaternion_wxyz": staged.current_quaternion,
        "desired_eef_position_m": staged.desired_position,
        "desired_eef_quaternion_wxyz": staged.desired_quaternion,
        "pose_error_position_m_axis_angle_rad": staged.pose_error,
        "approximate_pd_effort_preclip_nm": finalized.preclip_effort,
        "approximate_pd_effort_postclip_nm": finalized.postclip_effort,
    }
    for field, expected in expected_vectors.items():
        torch.testing.assert_close(
            _trace_values(entry, field), expected, rtol=0.0, atol=0.0
        )
    json.dumps(report, allow_nan=False)


def test_failure_substep_trace_producer_passes_boundary_failure_validator():
    action = _failure_substep_trace_action(episode_index=0)
    staged = _stage_failure_substep_trace(action, apply_index=0)
    finalized = _finalize_failure_substep_trace(action, staged, apply_index=0)
    action._apply_call_count = 2
    report = action.failure_substep_trace(episode_index=0)

    joint_pos = boundary_smoke._finite_vector_evidence(  # noqa: SLF001
        finalized.post_joint_pos
    )
    joint_vel = boundary_smoke._finite_vector_evidence(  # noqa: SLF001
        finalized.post_joint_vel
    )
    joint_pos_target = boundary_smoke._finite_vector_evidence(  # noqa: SLF001
        staged.new_target
    )
    joint_vel_target = boundary_smoke._finite_vector_evidence(  # noqa: SLF001
        staged.new_velocity_target
    )
    joint_effort_target = boundary_smoke._finite_vector_evidence(  # noqa: SLF001
        staged.new_effort_target
    )
    guard = {
        "kind": "current_joint_velocity_limit_abort",
        "episode_index": 0,
        "policy_step": 0,
        "physics_substep": 1,
        "joint_pos_rad": joint_pos,
        "raw_delta_joint_pos_rad": None,
        "raw_joint_pos_target_rad": None,
        "safe_joint_pos_target_rad": None,
        "pose_error_norm": None,
        "jacobian_finite": None,
        "jacobian_max_abs": None,
        "eef_quaternion_norm": None,
    }
    joint_velocity_values = joint_vel["values"]
    velocity_limits = list(PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S)
    velocity_excess = [
        max(
            boundary_smoke._float32_subtract(abs(value), limit),  # noqa: SLF001
            0.0,
        )
        for value, limit in zip(joint_velocity_values, velocity_limits, strict=True)
    ]
    velocity_mask = [
        abs(value)
        > boundary_smoke._float32_add(  # noqa: SLF001
            limit, JOINT_VELOCITY_LIMIT_TOLERANCE_RAD_S
        )
        for value, limit in zip(joint_velocity_values, velocity_limits, strict=True)
    ]
    safety = {
        "counters": {
            "apply_calls": 2,
            "invariant_aborts": 1,
            "current_joint_limit_aborts": 0,
            "nonfinite_aborts": 0,
        },
        "guard_diagnostics": [guard],
        "current_joint_velocity_abort": {
            "profile": CURRENT_JOINT_VELOCITY_ABORT_EVIDENCE_PROFILE,
            "episode_index": 0,
            "policy_step": 0,
            "physics_substep": 1,
            "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
            "joint_velocity_rad_s": joint_velocity_values,
            "joint_velocity_limit_rad_s": velocity_limits,
            "joint_velocity_limit_tolerance_rad_s": (
                JOINT_VELOCITY_LIMIT_TOLERANCE_RAD_S
            ),
            "joint_velocity_limit_excess_rad_s": velocity_excess,
            "exceeded_joint_mask": velocity_mask,
        },
    }

    assert (
        boundary_smoke.validate_failure_substep_trace(
            report,
            safety=safety,
            failure_policy_step=0,
            current_joint_pos=joint_pos,
            current_joint_vel=joint_vel,
            current_joint_pos_target=joint_pos_target,
            current_joint_vel_target=joint_vel_target,
            current_joint_effort_target=joint_effort_target,
            current_approximate_pd_effort_preclip=report["entries"][-1][
                "approximate_pd_effort_preclip_nm"
            ],
            current_approximate_pd_effort_postclip=report["entries"][-1][
                "approximate_pd_effort_postclip_nm"
            ],
            physx_joint_pos=joint_pos,
            physx_joint_vel=joint_vel,
        )
        is report
    )


@pytest.mark.parametrize(
    ("field", "expected_value"),
    [
        ("joint_stiffness", 400.0),
        ("joint_damping", 80.0),
        ("joint_effort_limits", None),
    ],
)
@pytest.mark.parametrize("failure_mode", ["shape", "value"])
def test_failure_substep_trace_rejects_live_drive_contract_drift(
    field, expected_value, failure_mode
):
    if expected_value is None:
        value = torch.tensor([PANDA_EEF_JOINT_EFFORT_LIMITS], dtype=torch.float32)
    else:
        value = torch.full((1, 7), expected_value, dtype=torch.float32)
    if failure_mode == "shape":
        value = value[:, :6]
        message = "shape/device/dtype drift"
    else:
        value[0, 0] += 1.0
        message = "live drive value drift"

    with pytest.raises(ValueError, match=message):
        _failure_substep_trace_action(data_overrides={field: value})


def test_failure_substep_trace_rejects_nonzero_effort_target_and_export_drift():
    with pytest.raises(ValueError, match="exactly zero live joint effort target"):
        _failure_substep_trace_action(
            data_overrides={
                "joint_effort_target": torch.ones((1, 7), dtype=torch.float32)
            }
        )

    action = _failure_substep_trace_action()
    action._asset.data.joint_stiffness[0, 0] = 399.0
    with pytest.raises(ValueError, match="mirror/readback mismatch"):
        action.failure_substep_trace(episode_index=0)

    effort_action = _failure_substep_trace_action()
    effort_action._asset.data.joint_effort_target[0, 0] = 1.0
    with pytest.raises(ValueError, match="exactly zero live joint effort target"):
        effort_action.failure_substep_trace(episode_index=0)


def test_failure_substep_trace_rejects_direct_physx_drive_readback_mismatch():
    with pytest.raises(ValueError, match="mirror/readback mismatch"):
        _failure_substep_trace_action(
            readback_overrides={
                "joint_damping": torch.full(
                    (1, 7),
                    79.0,
                    dtype=torch.float32,
                )
            }
        )


def test_failure_substep_trace_wrap_preserves_chronology_and_pending_capacity():
    action = _failure_substep_trace_action()
    for apply_index in range(72):
        staged = _stage_failure_substep_trace(action, apply_index)
        _finalize_failure_substep_trace(action, staged, apply_index)

    report = action.failure_substep_trace(episode_index=0)
    assert report["completed_entry_count"] == 64
    assert report["total_completed_entry_count"] == 72
    assert report["dropped_prefix_entry_count"] == 8
    assert [entry["apply_index"] for entry in report["entries"]] == list(range(8, 72))
    assert (
        report["entries"][0]["policy_step"],
        report["entries"][0]["physics_substep"],
    ) == (
        1,
        0,
    )
    assert (
        report["entries"][-1]["policy_step"],
        report["entries"][-1]["physics_substep"],
    ) == (
        8,
        7,
    )

    _stage_failure_substep_trace(action, apply_index=72)
    pending_report = action.failure_substep_trace(episode_index=0)
    assert pending_report["completed_entry_count"] == 63
    assert pending_report["total_completed_entry_count"] == 72
    assert pending_report["dropped_prefix_entry_count"] == 9
    assert pending_report["pending_entry_count"] == 1
    assert pending_report["pending_apply_index"] == 72
    assert [entry["apply_index"] for entry in pending_report["entries"]] == list(
        range(9, 72)
    )


def test_failure_substep_trace_masks_nonfinite_effort_and_resets_lifecycle():
    action = _failure_substep_trace_action(episode_index=4)
    staged = _stage_failure_substep_trace(action, apply_index=0)
    action._asset.data.computed_torque = torch.tensor(
        [[float("nan"), float("inf"), -float("inf"), 1.0, 2.0, 3.0, 4.0]],
        dtype=torch.float32,
    )
    action._asset.data.applied_torque = torch.zeros((1, 7), dtype=torch.float32)
    action._finalize_pending_failure_substep_trace(
        post_joint_pos=staged.joint_pos,
        post_joint_vel=staged.joint_vel,
    )

    report = action.failure_substep_trace(episode_index=4)
    effort = report["entries"][0]["approximate_pd_effort_preclip_nm"]
    assert effort["values"][:3] == [None, None, None]
    assert effort["finite_mask"] == [False, False, False, True, True, True, True]
    assert effort["finite_count"] == 4
    json.dumps(report, allow_nan=False)

    action._reset_failure_substep_trace_state()
    action._apply_call_count = 0
    reset_report = action.failure_substep_trace(episode_index=4)
    assert reset_report["completed_entry_count"] == 0
    assert reset_report["pending_entry_count"] == 0
    assert reset_report["pending_apply_index"] is None
    assert reset_report["entries"] == []
    with pytest.raises(ValueError, match="episode lifecycle mismatch"):
        action.failure_substep_trace(episode_index=5)
