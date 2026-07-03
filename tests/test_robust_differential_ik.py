import inspect
from pathlib import Path

import torch
import pytest

from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from polaris.robust_differential_ik import (
    DifferentialIKNumericalError,
    RobustDifferentialIKController,
    RobustDifferentialInverseKinematicsAction,
    _bound_joint_position_target,
    _eef_quaternion_norm_is_valid,
    _require_current_joint_position_in_soft_limits,
    _require_finite,
)
from polaris.eef_ik_safety import CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD
from polaris.eef_ik_safety import EEF_QUATERNION_UNIT_NORM_TOLERANCE
from polaris.eef_ik_safety import PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S
from polaris.eef_ik_safety import PANDA_SOFT_JOINT_POS_LIMITS_RAD


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
    action = object.__new__(RobustDifferentialInverseKinematicsAction)
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


def test_current_limit_and_slew_invariants_abort_before_physx_target_setter():
    source = inspect.getsource(RobustDifferentialInverseKinematicsAction.apply_actions)
    setter = source.index("self._asset.set_joint_position_target")
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
    assert source.index("if target_invalid:") < setter
    assert source.index("if slew_invalid:") < setter


def test_eef_pose_config_installs_robust_action_term():
    from polaris.config import LAP_EEF_FRAME
    from polaris.environments.droid_cfg import EefPoseActionCfg, SceneCfg

    cfg = EefPoseActionCfg()
    scene_cfg = SceneCfg()

    assert cfg.arm.class_type is RobustDifferentialInverseKinematicsAction
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


def test_eef_velocity_and_effort_limits_are_scoped_to_eef_setup():
    from polaris.environments.robot_cfg import NVIDIA_DROID
    from polaris.environments.robot_cfg import configure_eef_pose_joint_safety

    native_cfg = NVIDIA_DROID.copy()
    eef_cfg = NVIDIA_DROID.copy()
    assert native_cfg.actuators["panda_shoulder"].velocity_limit_sim is None
    assert native_cfg.actuators["panda_forearm"].velocity_limit_sim is None

    eval_source = (Path(__file__).parents[1] / "scripts" / "eval.py").read_text()
    eef_branch = eval_source.index('if eval_args.control_mode == "eef-pose":')
    configure_call = eval_source.index(
        "configure_eef_pose_joint_safety(env_cfg.scene.robot)"
    )
    native_branch = eval_source.index(
        'elif eval_args.control_mode != "joint-position":'
    )
    assert eef_branch < configure_call < native_branch

    configure_eef_pose_joint_safety(eef_cfg)

    assert eef_cfg.actuators["panda_shoulder"].velocity_limit_sim == 2.175
    assert eef_cfg.actuators["panda_shoulder"].effort_limit_sim == 87.0
    assert eef_cfg.actuators["panda_forearm"].velocity_limit_sim == 2.61
    assert eef_cfg.actuators["panda_forearm"].effort_limit_sim == 12.0
    assert native_cfg.actuators["panda_shoulder"].velocity_limit_sim is None
    assert native_cfg.actuators["panda_forearm"].velocity_limit_sim is None
