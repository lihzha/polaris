import torch
import pytest

from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from polaris.robust_differential_ik import (
    RobustDifferentialIKController,
    RobustDifferentialInverseKinematicsAction,
)


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


def test_nonfinite_input_holds_joint_target_after_inverse_failure():
    controller = _controller(RobustDifferentialIKController, damping=0.0)
    jacobian = torch.zeros(1, 6, 7)
    jacobian[0, 0, 0] = torch.nan
    delta_pose = torch.ones(1, 6)

    actual = controller._compute_delta_joint_pos(delta_pose, jacobian)

    torch.testing.assert_close(actual, torch.zeros(1, 7))
    assert torch.isfinite(actual).all()
    assert controller.fallback_count == 1


def test_eef_pose_config_installs_robust_action_term():
    from polaris.environments.droid_cfg import EefPoseActionCfg

    cfg = EefPoseActionCfg()

    assert cfg.arm.class_type is RobustDifferentialInverseKinematicsAction
