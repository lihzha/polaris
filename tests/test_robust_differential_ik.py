import torch
import pytest

from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from polaris.robust_differential_ik import (
    DifferentialIKNumericalError,
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


def test_nonfinite_input_aborts_rollout_before_physics_step():
    controller = _controller(RobustDifferentialIKController, damping=0.0)
    jacobian = torch.zeros(1, 6, 7)
    jacobian[0, 0, 0] = torch.nan
    delta_pose = torch.ones(1, 6)

    with pytest.raises(DifferentialIKNumericalError, match="non-finite input"):
        controller._compute_delta_joint_pos(delta_pose, jacobian)

    assert controller.fallback_count == 0


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
