import inspect
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
    _bound_joint_position_target,
    _classify_hard_limit_inward_recovery,
    _derive_isaac_soft_joint_position_limits,
    _eef_quaternion_norm_is_valid,
    _install_eef_physx_position_limits,
    _require_current_joint_position_in_soft_limits,
    _require_finite,
)
from polaris.eef_ik_safety import CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD
from polaris.eef_ik_safety import EEF_QUATERNION_UNIT_NORM_TOLERANCE
from polaris.eef_ik_safety import PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S
from polaris.eef_ik_safety import PANDA_EEF_JOINT_EFFORT_LIMITS
from polaris.eef_ik_safety import PANDA_EEF_JOINT_DRIVE_DAMPING
from polaris.eef_ik_safety import PANDA_EEF_JOINT_DRIVE_STIFFNESS
from polaris.eef_ik_safety import PANDA_PHYSX_DERIVED_SOFT_JOINT_POS_LIMITS_RAD
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
        joint_pos, healthy, max_delta, max_delta, soft_limits
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
        torch.ones((1, 7), dtype=torch.float32),
        torch.tensor([[[-3.0, 3.0]] * 7], dtype=torch.float32),
    )
    assert not ulp_slew.any()
    assert not ulp_position.any()
    assert torch.equal(ulp_safe.view(torch.int32), ulp_target.view(torch.int32))

    outlier = torch.tensor([[0.5, -0.5, 0.03, -0.03, 0.0, 0.02, -0.02]])
    safe, _, slew_limited, position_limited = _bound_joint_position_target(
        joint_pos, outlier, max_delta, max_delta, soft_limits
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
        joint_pos, raw_target, max_delta, max_delta, soft_limits
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
    physical_margin = torch.tensor(
        [PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S], dtype=torch.float32
    ) * (1.0 / 120.0)
    max_delta = physical_margin * 0.8
    boundary = soft_limits[..., 1] if direction > 0 else soft_limits[..., 0]
    joint_pos = boundary - direction * physical_margin / 4.0
    raw_target = boundary + direction * physical_margin

    safe, _, slew_limited, position_limited = _bound_joint_position_target(
        joint_pos, raw_target, max_delta, physical_margin, soft_limits
    )

    expected = boundary - direction * physical_margin
    torch.testing.assert_close(safe, expected, rtol=0.0, atol=0.0)
    assert slew_limited.all()
    assert position_limited.all()
    if direction > 0:
        assert torch.all(safe < soft_limits[..., 1])
    else:
        assert torch.all(safe > soft_limits[..., 0])
    assert torch.all((safe - joint_pos).abs() <= physical_margin)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_joint_target_guard_band_recovers_outer_tolerance_without_slew_violation(
    direction,
):
    soft_limits = torch.tensor([PANDA_SOFT_JOINT_POS_LIMITS_RAD], dtype=torch.float32)
    physical_margin = torch.tensor(
        [PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S], dtype=torch.float32
    ) * (1.0 / 120.0)
    max_delta = physical_margin * 0.8
    boundary = soft_limits[..., 1] if direction > 0 else soft_limits[..., 0]
    outer_tolerance_offset = torch.full_like(boundary, 5e-6)
    joint_pos = boundary + direction * outer_tolerance_offset
    raw_target = boundary + direction

    safe, _, slew_limited, position_limited = _bound_joint_position_target(
        joint_pos, raw_target, max_delta, physical_margin, soft_limits
    )

    strict_inner = boundary - direction * physical_margin
    guard_band_violation = (safe - strict_inner) * direction
    assert slew_limited.all()
    assert position_limited.all()
    assert torch.all((safe - joint_pos).abs() <= physical_margin + 1e-6)
    assert torch.all(guard_band_violation >= 0.0)
    assert torch.all(guard_band_violation <= CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD)
    assert torch.all(safe <= soft_limits[..., 1])
    assert torch.all(safe >= soft_limits[..., 0])


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_joint_target_guard_band_does_not_consume_recovery_for_in_range_state(
    direction,
):
    soft_limits = torch.tensor([PANDA_SOFT_JOINT_POS_LIMITS_RAD], dtype=torch.float32)
    physical_margin = torch.tensor(
        [PANDA_EEF_JOINT_VELOCITY_LIMITS_RAD_S], dtype=torch.float32
    ) * (1.0 / 120.0)
    max_delta = physical_margin * 0.8
    boundary = soft_limits[..., 1] if direction > 0 else soft_limits[..., 0]
    joint_pos = boundary - direction * torch.full_like(boundary, 5e-6)
    raw_target = boundary + direction

    safe, _, _, position_limited = _bound_joint_position_target(
        joint_pos, raw_target, max_delta, physical_margin, soft_limits
    )

    strict_inner = boundary - direction * physical_margin
    assert position_limited.all()
    torch.testing.assert_close(safe, strict_inner, rtol=0.0, atol=0.0)


def test_hard_limit_recovery_classifier_is_above_nominal_and_inward_only():
    joint_pos = torch.tensor([[-0.99, 0.99, -0.99, 0.99]])
    target_lower = torch.full((1, 4), -0.98)
    target_upper = torch.full((1, 4), 0.98)
    nominal = torch.full((1, 4), 0.008)
    position_limited = torch.ones((1, 4), dtype=torch.bool)

    recovery = _classify_hard_limit_inward_recovery(
        joint_pos=joint_pos,
        applied_delta=torch.tensor([[0.01, -0.01, -0.01, 0.01]]),
        position_limited=position_limited,
        target_lower=target_lower,
        target_upper=target_upper,
        nominal_max_delta=nominal,
    )
    assert recovery.tolist() == [[True, True, False, False]]

    not_limited = _classify_hard_limit_inward_recovery(
        joint_pos=joint_pos,
        applied_delta=torch.tensor([[0.01, -0.01, 0.01, -0.01]]),
        position_limited=torch.zeros_like(position_limited),
        target_lower=target_lower,
        target_upper=target_upper,
        nominal_max_delta=nominal,
    )
    assert not not_limited.any()

    float32_roundoff_only = _classify_hard_limit_inward_recovery(
        joint_pos=joint_pos,
        applied_delta=nominal + 2e-8,
        position_limited=position_limited,
        target_lower=target_lower,
        target_upper=target_upper,
        nominal_max_delta=nominal,
    )
    assert not float32_roundoff_only.any()


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
        self.stiffnesses = torch.tensor(
            [PANDA_EEF_JOINT_DRIVE_STIFFNESS], dtype=torch.float32
        )
        self.dampings = torch.tensor(
            [PANDA_EEF_JOINT_DRIVE_DAMPING], dtype=torch.float32
        )

    def get_dof_limits(self):
        return self.limits

    def get_dof_max_velocities(self):
        return self.max_velocities

    def get_dof_max_forces(self):
        return self.max_forces

    def get_dof_stiffnesses(self):
        return self.stiffnesses

    def get_dof_dampings(self):
        return self.dampings


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

    action = object.__new__(RobustDifferentialInverseKinematicsAction)
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
    action._joint_drive_stiffness = asset.root_physx_view.stiffnesses.clone()
    action._joint_drive_damping = asset.root_physx_view.dampings.clone()
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
    action = object.__new__(RobustDifferentialInverseKinematicsAction)
    action._physx_cfg = SimpleNamespace(solver_type=0)
    action._physx_solver_type = 1

    with pytest.raises(ValueError, match="PhysX solver type drifted"):
        action.safety_report()


@pytest.mark.parametrize(
    "limit_field", ["max_velocities", "max_forces", "stiffnesses", "dampings"]
)
def test_safety_report_rejects_live_physx_joint_limit_or_gain_drift(
    monkeypatch, limit_field
):
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

    action = object.__new__(RobustDifferentialInverseKinematicsAction)
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
    action._joint_drive_stiffness = asset.root_physx_view.stiffnesses.clone()
    action._joint_drive_damping = asset.root_physx_view.dampings.clone()
    action._solver_velocity_iteration_counts = (1,)
    monkeypatch.setattr(
        robust_ik,
        "_read_articulation_solver_iteration_counts",
        lambda _asset: ((64,), (1,)),
    )
    getattr(asset.root_physx_view, limit_field)[0, 0] += 1e-3

    with pytest.raises(ValueError, match="joint-limit or drive-gain readback drifted"):
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
    velocity_setter = source.index("self._asset.set_joint_velocity_target")
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
    assert velocity_setter < setter
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

    native_physx_cfg = SimpleNamespace(solver_type=0)
    eef_physx_cfg = SimpleNamespace(solver_type=0)
    configure_eef_pose_joint_safety(eef_cfg, physx_cfg=eef_physx_cfg)

    assert eef_cfg.actuators["panda_shoulder"].velocity_limit_sim == 2.175
    assert eef_cfg.actuators["panda_shoulder"].effort_limit_sim == 87.0
    assert eef_cfg.actuators["panda_forearm"].velocity_limit_sim == 2.61
    assert eef_cfg.actuators["panda_forearm"].effort_limit_sim == 12.0
    assert eef_cfg.spawn.articulation_props.solver_position_iteration_count == 64
    assert eef_cfg.spawn.articulation_props.solver_velocity_iteration_count == 1
    assert eef_physx_cfg.solver_type == 1
    assert native_cfg.actuators["panda_shoulder"].velocity_limit_sim is None
    assert native_cfg.actuators["panda_forearm"].velocity_limit_sim is None
    assert native_cfg.spawn.articulation_props.solver_position_iteration_count == 64
    assert native_cfg.spawn.articulation_props.solver_velocity_iteration_count == 0
    assert native_physx_cfg.solver_type == 0


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
