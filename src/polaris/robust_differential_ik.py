"""Numerically robust differential-IK action components.

Isaac Lab's damped-least-squares implementation uses a direct float32 matrix
inverse. That is normally appropriate, but a pathological Jacobian after a
dynamics excursion can make the configured damping round away and leave the
normal matrix singular. The classes in this module preserve Isaac Lab's normal
DLS path exactly and use a double-precision pseudo-inverse only after that
direct inverse raises a linear-algebra error.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import omni.log
import omni.usd
import torch

import isaaclab.sim as sim_utils
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.envs.mdp.actions.task_space_actions import (
    DifferentialInverseKinematicsAction,
)
from isaaclab.utils import configclass
from isaaclab.utils.math import compute_pose_error
from pxr import PhysxSchema
from pxr import UsdPhysics

from polaris.eef_controller_repair import advance_gripper_close_arm_interlock
from polaris.eef_controller_repair import (
    bound_joint_position_target as _bound_joint_position_target,
)
from polaris.eef_controller_repair import (
    DISABLED_GRIPPER_CLOSE_ARM_INTERLOCK_TRANSITION,
)
from polaris.eef_controller_repair import GripperCloseArmInterlockTransition
from polaris.eef_ik_safety import CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD
from polaris.eef_ik_safety import CURRENT_JOINT_VELOCITY_ABORT_EVIDENCE_PROFILE
from polaris.eef_ik_safety import ARM_VELOCITY_TARGET_PROFILE
from polaris.eef_ik_safety import ARM_SLEW_HEADROOM_CANDIDATE_PROFILE
from polaris.eef_ik_safety import ARM_SLEW_HEADROOM_RATIO
from polaris.eef_ik_safety import ARTICULATION_SOLVER_PROFILE
from polaris.eef_ik_safety import ARTICULATION_SOLVER_READBACK
from polaris.eef_ik_safety import EEF_IK_APPLY_CADENCE
from polaris.eef_ik_safety import EEF_IK_SAFETY_PROFILE
from polaris.eef_ik_safety import EEF_IK_WRIST_ENERGY_BRAKE_CANDIDATE_PROFILE
from polaris.eef_ik_safety import GRIPPER_CLOSE_ARM_INTERLOCK_CANDIDATE_PROFILE
from polaris.eef_ik_safety import GRIPPER_CLOSE_ARM_INTERLOCK_SUBSTEPS
from polaris.eef_ik_safety import EEF_QUATERNION_UNIT_NORM_TOLERANCE
from polaris.eef_ik_safety import format_current_joint_velocity_abort_message
from polaris.eef_ik_safety import JOINT_SLEW_FLOAT32_TOLERANCE_RAD
from polaris.eef_ik_safety import JOINT_VELOCITY_LIMIT_TOLERANCE_RAD_S
from polaris.eef_ik_safety import PHYSX_DERIVED_SOFT_LIMIT_PROFILE
from polaris.eef_ik_safety import PHYSX_HARD_LIMIT_PROFILE
from polaris.eef_ik_safety import PANDA_EEF_JOINT_EFFORT_LIMITS
from polaris.eef_ik_safety import PANDA_EEF_SOLVER_POSITION_ITERATION_COUNT
from polaris.eef_ik_safety import PANDA_EEF_SOLVER_VELOCITY_ITERATION_COUNT
from polaris.eef_ik_safety import PANDA_EEF_PHYSX_SOLVER_TYPE
from polaris.eef_ik_safety import TARGET_SOFT_LIMIT_GUARD_BAND_PROFILE
from polaris.eef_ik_safety import WRIST_ENERGY_BRAKE_JOINT_NAMES
from polaris.eef_ik_safety import WRIST_ENERGY_BRAKE_LATCH_SUBSTEPS
from polaris.eef_ik_safety import WRIST_ENERGY_BRAKE_PROFILE
from polaris.eef_ik_safety import WRIST_ENERGY_BRAKE_TARGET_SHIFT_FRACTION
from polaris.eef_gripper_runtime import EEF_GRIPPER_RUNTIME_PROFILE
from polaris.eef_gripper_runtime import GRIPPER_CLOSED_TARGET_FLOAT32
from polaris.eef_gripper_runtime import GRIPPER_JOINT_INDICES
from polaris.eef_gripper_runtime import GRIPPER_JOINT_NAMES
from polaris.eef_gripper_runtime import GRIPPER_OPEN_TARGET_FLOAT32
from polaris.eef_gripper_runtime import PINNED_DYNAMIC_DEVICE
from polaris.eef_gripper_runtime import PINNED_TENSOR_DTYPE
from polaris.eef_gripper_runtime import validate_eef_gripper_dynamic_evidence
from polaris.eef_gripper_runtime import validate_eef_gripper_static_contract


FAILURE_SUBSTEP_TRACE_CAPACITY = 64
FAILURE_SUBSTEP_TRACE_DECIMATION = 8
FAILURE_SUBSTEP_TRACE_PROFILE = "eef_applied_substep_ring_last64_v1"
FAILURE_SUBSTEP_TRACE_EFFORT_SEMANTICS = (
    "isaaclab_implicit_actuator_approximate_pd_preclip_and_effortlimit_clipped_v1"
)
FAILURE_SUBSTEP_TRACE_JOINT_DRIVE_STIFFNESS = (400.0,) * 7
FAILURE_SUBSTEP_TRACE_JOINT_DRIVE_DAMPING = (80.0,) * 7
FAILURE_SUBSTEP_TRACE_PHASE_CONTRACT = {
    "joint_state": "apply_actions_entry_cached_after_previous_scene_update_v1",
    "post_joint_state": (
        "next_apply_actions_entry_cached_after_command_physics_and_scene_update_v1"
    ),
    "joint_state_delta": "post_joint_state_minus_pre_joint_state_v1",
    "current_eef_pose": "apply_actions_entry_before_pose_error_and_dls_v1",
    "desired_eef_pose": "controller_command_live_at_apply_actions_entry_v1",
    "pose_error": "position_and_axis_angle_after_pose_error_before_dls_v1",
    "previous_target": "apply_actions_entry_before_current_target_setters_v1",
    "raw_dls_target": "after_dls_before_safety_bounding_v1",
    "new_target": "after_safety_bounding_and_both_target_setters_returned_v1",
    "new_velocity_target": "zero_target_after_velocity_setter_returned_v1",
    "new_effort_target": "zero_feedforward_live_at_write_data_to_sim_v1",
    "effort": (
        "isaaclab_write_data_to_sim_for_new_target_before_physics_"
        "observed_at_next_apply_actions_entry_v1"
    ),
}
FAILURE_SUBSTEP_TRACE_VECTOR_WIDTHS = {
    "joint_pos_rad": 7,
    "joint_vel_rad_s": 7,
    "post_joint_pos_rad": 7,
    "post_joint_vel_rad_s": 7,
    "delta_joint_pos_rad": 7,
    "delta_joint_vel_rad_s": 7,
    "previous_joint_pos_target_rad": 7,
    "raw_dls_joint_pos_target_rad": 7,
    "new_joint_pos_target_rad": 7,
    "new_joint_vel_target_rad_s": 7,
    "new_joint_effort_target_nm": 7,
    "current_eef_position_m": 3,
    "current_eef_quaternion_wxyz": 4,
    "desired_eef_position_m": 3,
    "desired_eef_quaternion_wxyz": 4,
    "pose_error_position_m_axis_angle_rad": 6,
    "approximate_pd_effort_preclip_nm": 7,
    "approximate_pd_effort_postclip_nm": 7,
}


class DifferentialIKNumericalError(RuntimeError):
    """Raised when an invalid IK state requires aborting the current rollout."""


class DifferentialIKInvariantError(DifferentialIKNumericalError):
    """Raised when a finite controller state violates a safety invariant."""


class GripperRuntimePostStepError(RuntimeError):
    """Hard-stop when gripper evidence fails after an environment step completed."""


def _require_finite(value: torch.Tensor, *, field: str) -> None:
    """Abort before PhysX when an IK input or output is non-finite."""

    if not torch.isfinite(value).all():
        finite_values = value[torch.isfinite(value)]
        max_abs = (
            float(finite_values.abs().max().item())
            if finite_values.numel() > 0
            else float("nan")
        )
        raise DifferentialIKNumericalError(
            f"PolaRiS EEF IK safety received non-finite {field}; "
            f"aborting rollout (max_abs_finite={max_abs:g})"
        )


def _read_articulation_solver_iteration_counts(
    asset,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Read composed PhysX articulation solver attributes for every env root.

    The pinned ``omni.physics.tensors`` articulation view does not expose a
    post-parser solver-iteration getter. Querying every concrete composed USD
    root therefore verifies the exact parser input rather than pretending the
    higher-level Isaac Sim Core wrapper is a tensor-state readback.
    """

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise ValueError("PolaRiS EEF IK could not access the live USD stage")
    asset_prims = sim_utils.find_matching_prims(asset.cfg.prim_path, stage=stage)
    expected_count = int(asset.root_physx_view.count)
    if len(asset_prims) != expected_count:
        raise ValueError(
            "PolaRiS EEF IK articulation asset-root count mismatch: "
            f"expected={expected_count}, actual={len(asset_prims)}"
        )

    relative_root_path = getattr(asset.cfg, "articulation_root_prim_path", None)
    articulation_roots = []
    for asset_prim in asset_prims:
        asset_path = asset_prim.GetPath().pathString
        if relative_root_path is None:
            matches = sim_utils.get_all_matching_child_prims(
                asset_path,
                predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI),
                stage=stage,
            )
            if len(matches) != 1:
                raise ValueError(
                    "PolaRiS EEF IK requires exactly one articulation root "
                    f"under {asset_path!r}; found={len(matches)}"
                )
            articulation_root = matches[0]
        else:
            root_path = (
                f"{asset_path.rstrip('/')}/{str(relative_root_path).lstrip('/')}"
            )
            articulation_root = stage.GetPrimAtPath(root_path)
            if not articulation_root.IsValid() or not articulation_root.HasAPI(
                UsdPhysics.ArticulationRootAPI
            ):
                raise ValueError(
                    "PolaRiS EEF IK configured articulation root is invalid: "
                    f"{root_path!r}"
                )
        if not articulation_root.HasAPI(PhysxSchema.PhysxArticulationAPI):
            raise ValueError(
                "PolaRiS EEF IK articulation root lacks PhysxArticulationAPI: "
                f"{articulation_root.GetPath().pathString!r}"
            )
        articulation_roots.append(articulation_root)

    articulation_roots.sort(key=lambda prim: prim.GetPath().pathString)
    root_paths = [prim.GetPath().pathString for prim in articulation_roots]
    if len(root_paths) != expected_count or len(set(root_paths)) != expected_count:
        raise ValueError("PolaRiS EEF IK articulation-root identity drift")

    position_counts: list[int] = []
    velocity_counts: list[int] = []
    for articulation_root in articulation_roots:
        schema = PhysxSchema.PhysxArticulationAPI(articulation_root)
        position_attr = schema.GetSolverPositionIterationCountAttr()
        velocity_attr = schema.GetSolverVelocityIterationCountAttr()
        if not (
            position_attr.HasAuthoredValueOpinion()
            and velocity_attr.HasAuthoredValueOpinion()
        ):
            raise ValueError(
                "PolaRiS EEF IK articulation solver attributes are unauthored"
            )
        position_value = position_attr.Get()
        velocity_value = velocity_attr.Get()
        if position_value is None or velocity_value is None:
            raise ValueError("PolaRiS EEF IK articulation solver values are absent")
        position_counts.append(int(position_value))
        velocity_counts.append(int(velocity_value))
    return tuple(position_counts), tuple(velocity_counts)


def _eef_quaternion_norm_is_valid(
    quaternion: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-environment quaternion norms and the named unit-norm guard."""

    # Float32 norm accumulation can overflow for finite policy outputs. Use
    # float64 so the diagnostic remains strict-JSON finite while still
    # rejecting the wildly non-unit command before pose-error computation.
    norms = quaternion.to(torch.float64).norm(dim=-1)
    valid = torch.isfinite(norms) & (
        (norms - 1.0).abs() <= EEF_QUATERNION_UNIT_NORM_TOLERANCE
    )
    return norms, valid


@dataclass(frozen=True)
class _WristEnergyBrakeTarget:
    """One pure, vectorized wrist energy-brake transition."""

    applied_target: torch.Tensor
    next_latch_remaining: torch.Tensor
    trigger_joint_mask: torch.Tensor
    active_environment_mask: torch.Tensor
    attempted_joint_mask: torch.Tensor
    braked_joint_mask: torch.Tensor
    target_shift: torch.Tensor


def _apply_wrist_energy_brake_target(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    previous_applied_target: torch.Tensor,
    reversal_detection_armed: torch.Tensor,
    nominal_safe_target: torch.Tensor,
    max_delta_joint_pos: torch.Tensor,
    soft_joint_pos_limits: torch.Tensor,
    latch_remaining: torch.Tensor,
    wrist_joint_indices: tuple[int, int, int],
) -> _WristEnergyBrakeTarget:
    """Apply the opt-in two-substep group wrist energy brake.

    A near-full-substep sign reversal of any *previously applied* wrist target
    arms one group latch for the trigger call and the immediately following
    physics substep.  While active, only nominal wrist spring terms with
    ``position_error * velocity > 0`` are energy-injecting; those targets are
    replaced by the ordinary slew/guard-bounded hold-at-current-position
    target.  Dissipative or exactly neutral nominal targets remain bitwise
    unchanged, as do all non-wrist targets.

    The caller owns lifecycle validation and commits the returned latch state
    only after both PhysX target setters succeed.
    """

    wrist_indices = list(wrist_joint_indices)
    previous_error = (
        previous_applied_target[:, wrist_indices] - joint_pos[:, wrist_indices]
    )
    nominal_error = nominal_safe_target[:, wrist_indices] - joint_pos[:, wrist_indices]
    target_shift = (
        nominal_safe_target[:, wrist_indices]
        - previous_applied_target[:, wrist_indices]
    ).abs()
    trigger_threshold = (
        max_delta_joint_pos[:, wrist_indices] * WRIST_ENERGY_BRAKE_TARGET_SHIFT_FRACTION
    )
    trigger_joint_mask = (
        reversal_detection_armed.unsqueeze(-1)
        & ((previous_error * nominal_error) < 0.0)
        & (target_shift >= trigger_threshold)
    )
    trigger_environment_mask = trigger_joint_mask.any(dim=-1)
    refreshed_latch = torch.where(
        trigger_environment_mask,
        torch.full_like(latch_remaining, WRIST_ENERGY_BRAKE_LATCH_SUBSTEPS),
        latch_remaining,
    )
    active_environment_mask = refreshed_latch > 0
    attempted_joint_mask = active_environment_mask.unsqueeze(-1) & (
        nominal_error * joint_vel[:, wrist_indices] > 0.0
    )

    hold_target, _, _, _ = _bound_joint_position_target(
        joint_pos,
        joint_pos,
        max_delta_joint_pos,
        soft_joint_pos_limits,
    )
    full_brake_mask = torch.zeros_like(nominal_safe_target, dtype=torch.bool)
    full_brake_mask[:, wrist_indices] = attempted_joint_mask
    applied_target = torch.where(
        full_brake_mask,
        hold_target,
        nominal_safe_target,
    )
    applied_wrist_target = applied_target[:, wrist_indices]
    braked_joint_mask = (
        attempted_joint_mask
        & (applied_wrist_target != nominal_safe_target[:, wrist_indices])
        & (
            (applied_wrist_target - joint_pos[:, wrist_indices])
            * joint_vel[:, wrist_indices]
            <= 0.0
        )
    )
    next_latch_remaining = torch.clamp(refreshed_latch - 1, min=0)
    return _WristEnergyBrakeTarget(
        applied_target=applied_target,
        next_latch_remaining=next_latch_remaining,
        trigger_joint_mask=trigger_joint_mask,
        active_environment_mask=active_environment_mask,
        attempted_joint_mask=attempted_joint_mask,
        braked_joint_mask=braked_joint_mask,
        target_shift=target_shift,
    )


def _require_current_joint_position_in_soft_limits(
    joint_pos: torch.Tensor,
    soft_joint_pos_limits: torch.Tensor,
) -> torch.Tensor:
    """Return current-position violation, aborting beyond float tolerance."""

    lower = soft_joint_pos_limits[..., 0]
    upper = soft_joint_pos_limits[..., 1]
    violation = torch.maximum(
        torch.clamp(lower - joint_pos, min=0.0),
        torch.clamp(joint_pos - upper, min=0.0),
    )
    if (violation > CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD).any():
        raise DifferentialIKNumericalError(
            "PolaRiS EEF IK current joint position is outside live soft "
            "limits; aborting before PhysX"
        )
    return violation


def _derive_isaac_soft_joint_position_limits(
    hard_limits: torch.Tensor,
    *,
    soft_limit_factor: float,
) -> torch.Tensor:
    """Reproduce Isaac Lab's float32 hard-to-soft limit derivation exactly."""

    joint_pos_mean = (hard_limits[..., 0] + hard_limits[..., 1]) / 2
    joint_pos_range = hard_limits[..., 1] - hard_limits[..., 0]
    return torch.stack(
        (
            joint_pos_mean - 0.5 * joint_pos_range * soft_limit_factor,
            joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor,
        ),
        dim=-1,
    )


def _install_eef_physx_position_limits(
    asset,
    *,
    joint_ids,
    outer_limits: torch.Tensor,
    max_delta_joint_pos: torch.Tensor,
    soft_limit_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Install and exactly verify the EEF-only inner PhysX joint envelope."""

    inner_limits = torch.stack(
        (
            outer_limits[..., 0] + max_delta_joint_pos,
            outer_limits[..., 1] - max_delta_joint_pos,
        ),
        dim=-1,
    )
    _require_finite(inner_limits, field="EEF PhysX hard joint position limits")
    if (inner_limits[..., 0] >= inner_limits[..., 1]).any():
        raise ValueError("PolaRiS EEF PhysX hard-limit envelope is not ordered")
    expected_derived_soft_limits = _derive_isaac_soft_joint_position_limits(
        inner_limits,
        soft_limit_factor=soft_limit_factor,
    )
    _require_finite(
        expected_derived_soft_limits,
        field="EEF PhysX-derived soft joint position limits",
    )
    prewrite_hard_limits = asset.data.joint_pos_limits[:, joint_ids, :].clone()
    _require_finite(
        prewrite_hard_limits,
        field="pre-install EEF PhysX hard joint position limits",
    )
    prewrite_derived_soft_limits = _derive_isaac_soft_joint_position_limits(
        prewrite_hard_limits,
        soft_limit_factor=soft_limit_factor,
    )
    if not torch.equal(prewrite_derived_soft_limits, outer_limits):
        raise ValueError(
            "PolaRiS EEF pre-install hard limits do not derive to the captured "
            "outer soft envelope"
        )
    asset.write_joint_position_limit_to_sim(
        inner_limits,
        joint_ids=joint_ids,
        env_ids=None,
        warn_limit_violation=True,
    )
    physx_limits = asset.root_physx_view.get_dof_limits().to(asset.device)[
        :, joint_ids, :
    ]
    mirror_limits = asset.data.joint_pos_limits[:, joint_ids, :]
    derived_soft_limits = asset.data.soft_joint_pos_limits[:, joint_ids, :]
    for field, actual in (
        ("PhysX", physx_limits),
        ("articulation mirror", mirror_limits),
    ):
        if not torch.equal(actual, inner_limits):
            raise ValueError(
                f"PolaRiS EEF {field} position-limit readback does not exactly "
                "match the requested inner envelope"
            )
    if not torch.equal(derived_soft_limits, expected_derived_soft_limits):
        raise ValueError(
            "PolaRiS EEF derived-soft position-limit readback does not exactly "
            "match pinned Isaac Lab midpoint/range arithmetic"
        )
    return inner_limits.clone(), expected_derived_soft_limits.clone()


class RobustDifferentialIKController(DifferentialIKController):
    """Isaac Lab DLS controller with an exception-only pseudo-inverse fallback."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fallback_count = 0

    def _compute_delta_joint_pos(
        self, delta_pose: torch.Tensor, jacobian: torch.Tensor
    ) -> torch.Tensor:
        inputs_are_finite = torch.isfinite(
            torch.cat((jacobian.flatten(start_dim=1), delta_pose), dim=-1)
        ).all()
        if self.cfg.ik_method == "dls" and not inputs_are_finite:
            finite_jacobian_values = jacobian[torch.isfinite(jacobian)]
            max_abs_jacobian = (
                float(finite_jacobian_values.abs().max().item())
                if finite_jacobian_values.numel() > 0
                else float("nan")
            )
            message = (
                "PolaRiS DLS received non-finite input; aborting rollout "
                f"(lambda={float(self.cfg.ik_params['lambda_val']):g}, "
                f"dtype={jacobian.dtype}, max_abs_jacobian={max_abs_jacobian:g})"
            )
            omni.log.error(message)
            raise DifferentialIKNumericalError(message)
        try:
            return super()._compute_delta_joint_pos(delta_pose, jacobian)
        except torch.linalg.LinAlgError as error:
            if self.cfg.ik_method != "dls":
                raise
            return self._compute_dls_pinv_fallback(delta_pose, jacobian, error)

    def _compute_dls_pinv_fallback(
        self,
        delta_pose: torch.Tensor,
        jacobian: torch.Tensor,
        error: Exception,
    ) -> torch.Tensor:
        """Recover from a failed DLS inverse without changing healthy steps.

        Finite environments use the same damped normal matrix as Isaac Lab,
        evaluated in float64 with ``pinv``. A second linear-algebra failure or
        non-finite fallback result aborts the rollout instead of allowing an
        invalid physics state to hang the simulator.
        """

        delta_joint_pos = torch.zeros(
            (jacobian.shape[0], jacobian.shape[2]),
            dtype=jacobian.dtype,
            device=jacobian.device,
        )
        finite = torch.isfinite(jacobian).all(dim=(-2, -1))
        finite &= torch.isfinite(delta_pose).all(dim=-1)
        valid_indices = finite.nonzero(as_tuple=False).flatten()
        recovered = 0
        lambda_val = float(self.cfg.ik_params["lambda_val"])

        if valid_indices.numel() > 0:
            valid_jacobian = jacobian[valid_indices]
            valid_delta_pose = delta_pose[valid_indices]
            work_dtype = (
                torch.float64
                if valid_jacobian.dtype
                in (torch.float16, torch.bfloat16, torch.float32)
                else valid_jacobian.dtype
            )
            valid_jacobian = valid_jacobian.to(dtype=work_dtype)
            valid_delta_pose = valid_delta_pose.to(dtype=work_dtype)
            jacobian_t = valid_jacobian.transpose(1, 2)
            damping = (lambda_val**2) * torch.eye(
                valid_jacobian.shape[1],
                dtype=work_dtype,
                device=valid_jacobian.device,
            )
            normal_matrix = valid_jacobian @ jacobian_t + damping

            try:
                candidate = (
                    jacobian_t
                    @ torch.linalg.pinv(normal_matrix, hermitian=True)
                    @ valid_delta_pose.unsqueeze(-1)
                ).squeeze(-1)
                candidate_is_finite = torch.isfinite(candidate).all(dim=-1)
                recovered_indices = valid_indices[candidate_is_finite]
                delta_joint_pos[recovered_indices] = candidate[candidate_is_finite].to(
                    dtype=jacobian.dtype
                )
                recovered = int(candidate_is_finite.sum().item())
            except torch.linalg.LinAlgError as fallback_error:
                raise DifferentialIKNumericalError(
                    "PolaRiS damped pseudo-inverse fallback also failed"
                ) from fallback_error

        if recovered != jacobian.shape[0]:
            raise DifferentialIKNumericalError(
                "PolaRiS damped pseudo-inverse fallback returned a non-finite result"
            )

        self.fallback_count += 1
        if self.fallback_count <= 5 or self.fallback_count % 100 == 0:
            held = jacobian.shape[0] - recovered
            finite_jacobian_values = jacobian[torch.isfinite(jacobian)]
            max_abs_jacobian = (
                float(finite_jacobian_values.abs().max().item())
                if finite_jacobian_values.numel() > 0
                else float("nan")
            )
            omni.log.warn(
                "PolaRiS DLS inverse failed; used pseudo-inverse fallback "
                f"for {recovered} environment(s) and held {held} "
                f"environment(s) (fallback event {self.fallback_count}, "
                f"lambda={lambda_val:g}, dtype={jacobian.dtype}, "
                f"max_abs_jacobian={max_abs_jacobian:g}, "
                f"finite_inputs={int(finite.sum().item())}/{jacobian.shape[0]}): "
                f"{error}"
            )

        return delta_joint_pos


class RobustDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """Task-space action that installs :class:`RobustDifferentialIKController`."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        trace_enabled = self.cfg.enable_failure_substep_trace
        if type(trace_enabled) is not bool:
            raise ValueError(
                "PolaRiS EEF failure substep trace enable flag must be bool"
            )
        self._failure_substep_trace_enabled = trace_enabled
        if self._failure_substep_trace_enabled and self.num_envs != 1:
            raise ValueError(
                "PolaRiS EEF failure substep trace requires exactly one environment"
            )
        wrist_energy_brake_enabled = self.cfg.enable_wrist_energy_brake
        if type(wrist_energy_brake_enabled) is not bool:
            raise ValueError("PolaRiS EEF wrist energy-brake enable flag must be bool")
        self._wrist_energy_brake_enabled = wrist_energy_brake_enabled
        if self._wrist_energy_brake_enabled and self.num_envs != 1:
            raise ValueError(
                "PolaRiS EEF wrist energy-brake candidate requires exactly "
                "one environment"
            )
        arm_slew_headroom_enabled = self.cfg.enable_arm_slew_headroom
        if type(arm_slew_headroom_enabled) is not bool:
            raise ValueError("PolaRiS EEF arm-slew headroom enable flag must be bool")
        self._arm_slew_headroom_enabled = arm_slew_headroom_enabled
        close_interlock_enabled = self.cfg.enable_gripper_close_arm_interlock
        if type(close_interlock_enabled) is not bool:
            raise ValueError("PolaRiS EEF close-interlock enable flag must be bool")
        self._gripper_close_arm_interlock_enabled = close_interlock_enabled
        if self._gripper_close_arm_interlock_enabled and self.num_envs != 1:
            raise ValueError(
                "PolaRiS EEF close-interlock candidate requires exactly one environment"
            )
        if (
            self._gripper_close_arm_interlock_enabled
            and self._wrist_energy_brake_enabled
        ):
            raise ValueError(
                "PolaRiS EEF close-interlock and wrist-energy-brake candidates "
                "cannot be combined"
            )
        self._safety_profile = (
            EEF_IK_WRIST_ENERGY_BRAKE_CANDIDATE_PROFILE
            if self._wrist_energy_brake_enabled
            else EEF_IK_SAFETY_PROFILE
        )
        self._gripper_runtime_static: dict[str, object] | None = None
        self._ik_controller = RobustDifferentialIKController(
            cfg=self.cfg.controller,
            num_envs=self.num_envs,
            device=self.device,
        )
        self._physics_dt = float(env.physics_dt)
        self._control_dt = float(env.step_dt)
        self._decimation = int(env.cfg.decimation)
        self._physx_cfg = env.cfg.sim.physx
        self._physx_solver_type = int(self._physx_cfg.solver_type)
        if self._physx_solver_type != PANDA_EEF_PHYSX_SOLVER_TYPE:
            raise ValueError(
                "PolaRiS EEF IK requires the TGS PhysX solver "
                f"(type={PANDA_EEF_PHYSX_SOLVER_TYPE}); "
                f"live={self._physx_solver_type}"
            )
        if (
            self._physics_dt <= 0.0
            or self._decimation <= 0
            or not math.isclose(
                self._physics_dt * self._decimation,
                self._control_dt,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "PolaRiS EEF IK safety requires apply_actions at the physics "
                "substep cadence: "
                f"physics_dt={self._physics_dt!r}, control_dt={self._control_dt!r}, "
                f"decimation={self._decimation!r}"
            )

        self._joint_velocity_limits = self._asset.data.joint_vel_limits[
            :, self._joint_ids
        ].clone()
        self._joint_effort_limits = self._asset.data.joint_effort_limits[
            :, self._joint_ids
        ].clone()
        physx_joint_velocity_limits = (
            self._asset.root_physx_view.get_dof_max_velocities()
            .to(self._asset.device)[:, self._joint_ids]
            .clone()
        )
        physx_joint_effort_limits = (
            self._asset.root_physx_view.get_dof_max_forces()
            .to(self._asset.device)[:, self._joint_ids]
            .clone()
        )
        _require_finite(
            self._joint_velocity_limits, field="configured joint velocity limits"
        )
        _require_finite(
            self._joint_effort_limits, field="configured joint effort limits"
        )
        if (self._joint_velocity_limits <= 0.0).any() or (
            self._joint_effort_limits <= 0.0
        ).any():
            raise ValueError(
                "PolaRiS EEF IK safety requires positive live joint velocity "
                "and effort limits"
            )
        if not (
            torch.equal(physx_joint_velocity_limits, self._joint_velocity_limits)
            and torch.equal(physx_joint_effort_limits, self._joint_effort_limits)
        ):
            raise ValueError(
                "PolaRiS EEF IK cached joint limits do not match PhysX readback"
            )
        self._max_delta_joint_pos = self._joint_velocity_limits * self._physics_dt
        self._nominal_max_delta_joint_pos = self._max_delta_joint_pos * (
            ARM_SLEW_HEADROOM_RATIO if self._arm_slew_headroom_enabled else 1.0
        )
        if not (
            torch.isfinite(self._nominal_max_delta_joint_pos).all()
            and (self._nominal_max_delta_joint_pos > 0.0).all()
            and (self._nominal_max_delta_joint_pos <= self._max_delta_joint_pos).all()
        ):
            raise ValueError("PolaRiS EEF nominal arm-slew bound is invalid")

        articulation_props = getattr(
            getattr(self._asset.cfg, "spawn", None),
            "articulation_props",
            None,
        )
        if articulation_props is None:
            raise ValueError(
                "PolaRiS EEF IK safety requires articulation solver properties"
            )
        configured_position_iterations = int(
            articulation_props.solver_position_iteration_count
        )
        configured_velocity_iterations = int(
            articulation_props.solver_velocity_iteration_count
        )
        if (
            configured_position_iterations != PANDA_EEF_SOLVER_POSITION_ITERATION_COUNT
            or configured_velocity_iterations
            != PANDA_EEF_SOLVER_VELOCITY_ITERATION_COUNT
        ):
            raise ValueError(
                "PolaRiS EEF IK articulation solver configuration mismatch: "
                f"position={configured_position_iterations!r}, "
                f"velocity={configured_velocity_iterations!r}"
            )
        (
            self._solver_position_iteration_counts,
            self._solver_velocity_iteration_counts,
        ) = _read_articulation_solver_iteration_counts(self._asset)
        if (
            len(self._solver_position_iteration_counts) != self.num_envs
            or len(self._solver_velocity_iteration_counts) != self.num_envs
            or any(
                count != PANDA_EEF_SOLVER_POSITION_ITERATION_COUNT
                for count in self._solver_position_iteration_counts
            )
            or any(
                count != PANDA_EEF_SOLVER_VELOCITY_ITERATION_COUNT
                for count in self._solver_velocity_iteration_counts
            )
        ):
            raise ValueError(
                "PolaRiS EEF IK composed articulation solver readback mismatch"
            )

        soft_limits = self._soft_joint_pos_limits()
        _require_finite(soft_limits, field="soft joint position limits")
        if (soft_limits[..., 0] >= soft_limits[..., 1]).any():
            raise ValueError(
                "PolaRiS EEF IK safety requires ordered live soft joint limits"
            )
        self._soft_joint_pos_limit_factor = float(
            self._asset.cfg.soft_joint_pos_limit_factor
        )
        if self._soft_joint_pos_limit_factor != 1.0:
            raise ValueError(
                "PolaRiS EEF IK safety requires soft_joint_pos_limit_factor=1"
            )
        self._soft_joint_position_limits = soft_limits.clone()
        (
            self._physx_hard_joint_position_limits,
            self._physx_derived_soft_joint_position_limits,
        ) = _install_eef_physx_position_limits(
            self._asset,
            joint_ids=self._joint_ids,
            outer_limits=self._soft_joint_position_limits,
            max_delta_joint_pos=self._max_delta_joint_pos,
            soft_limit_factor=self._soft_joint_pos_limit_factor,
        )
        self._physx_hard_limit_write_count = 1
        self._zero_joint_velocity_target = torch.zeros_like(self._max_delta_joint_pos)
        self._max_guard_diagnostics = 32
        if self._wrist_energy_brake_enabled:
            resolved_indices = []
            for joint_name in WRIST_ENERGY_BRAKE_JOINT_NAMES:
                matches = [
                    index
                    for index, resolved_name in enumerate(self._joint_names)
                    if resolved_name == joint_name
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "PolaRiS EEF wrist energy brake requires one exact "
                        f"{joint_name!r} joint; matches={matches!r}"
                    )
                resolved_indices.append(matches[0])
            self._wrist_energy_brake_joint_indices = tuple(resolved_indices)
            self._wrist_energy_brake_latch_remaining = torch.zeros(
                self.num_envs,
                dtype=torch.int64,
                device=self.device,
            )
            self._wrist_energy_brake_previous_applied_target = torch.zeros_like(
                self._max_delta_joint_pos
            )
            self._wrist_energy_brake_previous_target_valid = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
            self._wrist_energy_brake_reversal_detection_armed = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
            self._wrist_energy_brake_target_shift_threshold = (
                self._max_delta_joint_pos[
                    :, list(self._wrist_energy_brake_joint_indices)
                ]
                * WRIST_ENERGY_BRAKE_TARGET_SHIFT_FRACTION
            )
        if self._failure_substep_trace_enabled:
            self._initialize_failure_substep_trace()
        self._reset_episode_safety_state(episode_index=None)

    def _validated_failure_substep_trace_drive_tensors(
        self,
    ) -> dict[str, torch.Tensor]:
        """Return exact live Panda drive tensors or reject contract drift."""

        expected_shape = (self.num_envs, self._num_joints)
        expected_device = torch.device(self.device)
        expected_dtype = self._asset.data.joint_pos.dtype
        specifications = {
            "joint_drive_stiffness": (
                "joint_stiffness",
                "get_dof_stiffnesses",
                FAILURE_SUBSTEP_TRACE_JOINT_DRIVE_STIFFNESS,
            ),
            "joint_drive_damping": (
                "joint_damping",
                "get_dof_dampings",
                FAILURE_SUBSTEP_TRACE_JOINT_DRIVE_DAMPING,
            ),
            "joint_effort_limits": (
                "joint_effort_limits",
                "get_dof_max_forces",
                PANDA_EEF_JOINT_EFFORT_LIMITS,
            ),
        }
        live_tensors: dict[str, torch.Tensor] = {}
        for field, (
            asset_field,
            readback_method,
            expected_values,
        ) in specifications.items():
            full_value = getattr(self._asset.data, asset_field, None)
            if not isinstance(full_value, torch.Tensor):
                raise ValueError(
                    "PolaRiS EEF failure substep trace requires live "
                    f"{asset_field} tensor"
                )
            try:
                mirror_value = full_value[:, self._joint_ids]
            except (IndexError, RuntimeError) as error:
                raise ValueError(
                    "PolaRiS EEF failure substep trace live drive tensor "
                    f"shape/device/dtype drift: field={field!r}"
                ) from error
            getter = getattr(self._asset.root_physx_view, readback_method, None)
            if not callable(getter):
                raise ValueError(
                    "PolaRiS EEF failure substep trace requires direct PhysX "
                    f"drive readback: field={field!r}"
                )
            try:
                readback_value = getter().to(self._asset.device)[:, self._joint_ids]
            except (AttributeError, IndexError, RuntimeError) as error:
                raise ValueError(
                    "PolaRiS EEF failure substep trace live drive tensor "
                    f"shape/device/dtype drift: field={field!r}"
                ) from error
            if (
                tuple(mirror_value.shape) != expected_shape
                or mirror_value.device != expected_device
                or mirror_value.dtype != expected_dtype
                or tuple(readback_value.shape) != expected_shape
                or readback_value.device != expected_device
                or readback_value.dtype != expected_dtype
            ):
                raise ValueError(
                    "PolaRiS EEF failure substep trace live drive tensor "
                    f"shape/device/dtype drift: field={field!r}, "
                    f"expected_shape={expected_shape!r}, "
                    f"mirror_shape={tuple(mirror_value.shape)!r}, "
                    f"readback_shape={tuple(readback_value.shape)!r}, "
                    f"expected_device={expected_device!r}, "
                    f"mirror_device={mirror_value.device!r}, "
                    f"readback_device={readback_value.device!r}, "
                    f"expected_dtype={expected_dtype!r}, "
                    f"mirror_dtype={mirror_value.dtype!r}, "
                    f"readback_dtype={readback_value.dtype!r}"
                )
            if not torch.equal(mirror_value, readback_value):
                raise ValueError(
                    "PolaRiS EEF failure substep trace live drive "
                    f"mirror/readback mismatch: field={field!r}"
                )
            expected = torch.tensor(
                expected_values,
                dtype=expected_dtype,
                device=expected_device,
            ).expand(self.num_envs, -1)
            if not torch.equal(readback_value, expected):
                raise ValueError(
                    "PolaRiS EEF failure substep trace live drive value drift: "
                    f"field={field!r}"
                )
            live_tensors[field] = readback_value
        return live_tensors

    def _validated_failure_substep_trace_zero_effort_target(self) -> torch.Tensor:
        """Return the live zero feed-forward target or reject contract drift."""

        full_value = getattr(self._asset.data, "joint_effort_target", None)
        if not isinstance(full_value, torch.Tensor):
            raise ValueError(
                "PolaRiS EEF failure substep trace requires a live "
                "joint_effort_target tensor"
            )
        try:
            selected = full_value[:, self._joint_ids]
        except (IndexError, RuntimeError) as error:
            raise ValueError(
                "PolaRiS EEF failure substep trace requires live "
                "joint_effort_target with controller shape/device/dtype"
            ) from error
        if (
            tuple(selected.shape) != (self.num_envs, self._num_joints)
            or selected.device != torch.device(self.device)
            or selected.dtype != self._asset.data.joint_pos.dtype
        ):
            raise ValueError(
                "PolaRiS EEF failure substep trace requires live "
                "joint_effort_target with controller shape/device/dtype"
            )
        if not torch.equal(selected, torch.zeros_like(selected)):
            raise ValueError(
                "PolaRiS EEF failure substep trace requires an exactly "
                "zero live joint effort target"
            )
        return selected

    def _initialize_failure_substep_trace(self) -> None:
        """Allocate the optional failure-only ring without any host transfers."""

        if self._decimation != FAILURE_SUBSTEP_TRACE_DECIMATION:
            raise ValueError(
                "PolaRiS EEF failure substep trace requires exactly eight "
                f"physics substeps per policy step; live={self._decimation!r}"
            )
        if self._num_joints != 7 or len(self._joint_names) != 7:
            raise ValueError(
                "PolaRiS EEF failure substep trace requires exactly seven "
                "ordered Panda arm joints"
            )
        self._validated_failure_substep_trace_drive_tensors()
        self._validated_failure_substep_trace_zero_effort_target()
        joint_dtype = self._asset.data.joint_pos.dtype
        self._failure_substep_trace_buffers = {
            field: torch.empty(
                (
                    FAILURE_SUBSTEP_TRACE_CAPACITY,
                    self.num_envs,
                    width,
                ),
                dtype=joint_dtype,
                device=self.device,
            )
            for field, width in FAILURE_SUBSTEP_TRACE_VECTOR_WIDTHS.items()
        }
        self._failure_substep_trace_apply_indices = torch.empty(
            FAILURE_SUBSTEP_TRACE_CAPACITY,
            dtype=torch.int64,
            device=self.device,
        )

        for field in ("computed_torque", "applied_torque"):
            value = getattr(self._asset.data, field, None)
            selected = (
                value[:, self._joint_ids] if isinstance(value, torch.Tensor) else None
            )
            if (
                selected is None
                or tuple(selected.shape) != (self.num_envs, self._num_joints)
                or selected.device != torch.device(self.device)
                or selected.dtype != joint_dtype
            ):
                raise ValueError(
                    "PolaRiS EEF failure substep trace requires live "
                    f"{field} with controller shape/device/dtype"
                )

    def _reset_failure_substep_trace_state(self) -> None:
        """Clear ring lifecycle state for one isolated rollout."""

        self._failure_substep_trace_total_completed = 0
        self._failure_substep_trace_pending_slot: int | None = None
        self._failure_substep_trace_pending_apply_index: int | None = None
        self._failure_substep_trace_apply_indices.fill_(-1)
        for buffer in self._failure_substep_trace_buffers.values():
            buffer.fill_(float("nan"))

    def _reset_wrist_energy_brake_state(self, env_ids=None) -> None:
        """Clear candidate controller state for the selected environments."""

        selected = slice(None) if env_ids is None else env_ids
        self._wrist_energy_brake_latch_remaining[selected] = 0
        self._wrist_energy_brake_previous_applied_target[selected] = 0.0
        self._wrist_energy_brake_previous_target_valid[selected] = False
        self._wrist_energy_brake_reversal_detection_armed[selected] = False

    def _reset_gripper_close_arm_interlock_state(self) -> None:
        """Clear the default-off close/motion interlock lifecycle."""

        self._gripper_close_arm_interlock_remaining = 0
        self._gripper_close_arm_interlock_observed_endpoint_change_count = 0
        self._gripper_close_arm_interlock_endpoint_observed = False
        self._gripper_close_arm_interlock_activation_count = 0
        self._gripper_close_arm_interlock_active_apply_count = 0
        self._gripper_close_arm_interlock_released_apply_count = 0
        self._gripper_close_arm_interlock_max_abs_active_delta = torch.zeros_like(
            self._max_delta_joint_pos[0]
        )
        self._gripper_close_arm_interlock_max_abs_released_delta = torch.zeros_like(
            self._max_delta_joint_pos[0]
        )

    def _reset_episode_safety_state(self, episode_index: int | None) -> None:
        counter_dtype = torch.int64
        self._active_episode_index = episode_index
        self._reset_gripper_close_arm_interlock_state()
        self._apply_call_count = 0
        self._slew_limit_event_count = torch.zeros(
            (), dtype=counter_dtype, device=self.device
        )
        self._slew_limit_joint_count = torch.zeros(
            (), dtype=counter_dtype, device=self.device
        )
        self._position_limit_event_count = torch.zeros(
            (), dtype=counter_dtype, device=self.device
        )
        self._position_limit_joint_count = torch.zeros(
            (), dtype=counter_dtype, device=self.device
        )
        self._post_clamp_target_violation_count = torch.zeros(
            (), dtype=counter_dtype, device=self.device
        )
        self._current_joint_limit_abort_count = torch.zeros(
            (), dtype=counter_dtype, device=self.device
        )
        self._nonfinite_abort_count = torch.zeros(
            (), dtype=counter_dtype, device=self.device
        )
        self._invariant_abort_count = torch.zeros(
            (), dtype=counter_dtype, device=self.device
        )
        if self._wrist_energy_brake_enabled:
            self._wrist_energy_brake_trigger_event_count = torch.zeros(
                (), dtype=counter_dtype, device=self.device
            )
            self._wrist_energy_brake_active_substep_count = torch.zeros(
                (), dtype=counter_dtype, device=self.device
            )
            self._wrist_energy_brake_attempted_joint_target_count = torch.zeros(
                (), dtype=counter_dtype, device=self.device
            )
            self._wrist_energy_brake_braked_joint_target_count = torch.zeros(
                (), dtype=counter_dtype, device=self.device
            )
            self._reset_wrist_energy_brake_state()
            self._wrist_energy_brake_diagnostics: list[dict[str, object]] = []
            self._wrist_energy_brake_diagnostics_dropped = 0
        self._max_raw_delta_joint_pos = torch.zeros(
            self._num_joints, dtype=torch.float32, device=self.device
        )
        self._max_applied_delta_joint_pos = torch.zeros_like(
            self._max_raw_delta_joint_pos
        )
        self._max_raw_target_soft_limit_violation = torch.zeros_like(
            self._max_raw_delta_joint_pos
        )
        self._max_post_clamp_target_soft_limit_violation = torch.zeros_like(
            self._max_raw_delta_joint_pos
        )
        self._max_post_clamp_target_guard_band_violation = torch.zeros_like(
            self._max_raw_delta_joint_pos
        )
        self._max_current_joint_soft_limit_violation = torch.zeros_like(
            self._max_raw_delta_joint_pos
        )
        self._max_current_physx_hard_limit_violation = torch.zeros_like(
            self._max_raw_delta_joint_pos
        )
        self._max_abs_joint_vel = torch.zeros_like(self._max_raw_delta_joint_pos)
        self._current_joint_velocity_abort: dict[str, object] | None = None
        self._minimum_outer_joint_clearance = torch.full_like(
            self._max_raw_delta_joint_pos,
            float("inf"),
        )
        self._fallback_count_at_episode_start = self._ik_controller.fallback_count
        self._guard_diagnostics: list[dict[str, object]] = []
        self._guard_diagnostics_dropped = 0
        self._max_raw_delta_diagnostic_value = torch.tensor(
            -1.0, dtype=torch.float32, device=self.device
        )
        self._max_raw_delta_diagnostic_apply_index = torch.tensor(
            -1, dtype=torch.int64, device=self.device
        )
        self._max_raw_delta_diagnostic_joint_pos = torch.zeros(
            self._num_joints, dtype=torch.float32, device=self.device
        )
        self._max_raw_delta_diagnostic_raw_delta = torch.zeros_like(
            self._max_raw_delta_diagnostic_joint_pos
        )
        self._max_raw_delta_diagnostic_raw_target = torch.zeros_like(
            self._max_raw_delta_diagnostic_joint_pos
        )
        self._max_raw_delta_diagnostic_safe_target = torch.zeros_like(
            self._max_raw_delta_diagnostic_joint_pos
        )
        self._max_raw_delta_diagnostic_pose_error_norm = torch.tensor(
            0.0, dtype=torch.float64, device=self.device
        )
        self._max_raw_delta_diagnostic_jacobian_max_abs = torch.tensor(
            0.0, dtype=torch.float64, device=self.device
        )
        if getattr(self, "_gripper_runtime_static", None) is not None:
            self._reset_gripper_runtime_evidence()
        if self._failure_substep_trace_enabled:
            self._reset_failure_substep_trace_state()

    def install_gripper_runtime_contract(
        self, contract: dict[str, object], *, finger_term: object
    ) -> None:
        """Bind the one-call production follower write to this action term."""

        if getattr(self, "_gripper_runtime_static", None) is not None:
            raise ValueError(
                "PolaRiS EEF gripper runtime contract is already installed"
            )
        if self._apply_call_count != 0:
            raise ValueError(
                "PolaRiS EEF gripper contract must be installed before apply"
            )
        validated = validate_eef_gripper_static_contract(contract)
        target_slew_static = getattr(
            finger_term, "gripper_target_slew_static_contract", None
        )
        target_slew_dynamic = getattr(
            finger_term, "gripper_target_slew_dynamic_report", None
        )
        if not callable(target_slew_static) or not callable(target_slew_dynamic):
            raise ValueError(
                "PolaRiS EEF gripper term lacks target-slew runtime evidence"
            )
        if target_slew_static() != validated["driver_target_slew"]:
            raise ValueError("PolaRiS EEF gripper target-slew static evidence drifted")
        for field in (
            "joint_pos",
            "joint_vel",
            "joint_acc",
            "joint_pos_target",
            "joint_vel_target",
        ):
            tensor = getattr(self._asset.data, field, None)
            if (
                not isinstance(tensor, torch.Tensor)
                or str(tensor.device) != PINNED_DYNAMIC_DEVICE
                or str(tensor.dtype) != PINNED_TENSOR_DTYPE
                or tuple(tensor.shape) != (self.num_envs, len(self._asset.joint_names))
            ):
                raise ValueError(
                    "PolaRiS EEF gripper dynamic tensor contract drift: "
                    f"field={field!r}, shape={getattr(tensor, 'shape', None)!r}, "
                    f"device={getattr(tensor, 'device', None)!r}, "
                    f"dtype={getattr(tensor, 'dtype', None)!r}"
                )
        self._gripper_runtime_static = validated
        self._gripper_target_slew_term = finger_term
        self._reset_gripper_runtime_evidence()

    def _next_gripper_close_arm_interlock_transition(
        self,
    ) -> GripperCloseArmInterlockTransition:
        """Read the bound binary endpoint and stage one interlock transition."""

        if not self._gripper_close_arm_interlock_enabled:
            return advance_gripper_close_arm_interlock(
                enabled=False,
                previous_endpoint_change_count=(
                    self._gripper_close_arm_interlock_observed_endpoint_change_count
                ),
                current_endpoint_change_count=(
                    self._gripper_close_arm_interlock_observed_endpoint_change_count
                ),
                endpoint_observed_before_apply=(
                    self._gripper_close_arm_interlock_endpoint_observed
                ),
                endpoint_is_closed=False,
                remaining_before_apply=self._gripper_close_arm_interlock_remaining,
            )
        finger_term = getattr(self, "_gripper_target_slew_term", None)
        if finger_term is None:
            raise ValueError(
                "PolaRiS EEF close interlock requires the installed gripper term"
            )
        endpoint = getattr(finger_term, "_gripper_target_slew_endpoint", None)
        endpoint_seen = getattr(finger_term, "_gripper_target_slew_endpoint_seen", None)
        close_command = getattr(finger_term, "_close_command", None)
        open_command = getattr(finger_term, "_open_command", None)
        endpoint_change_count = getattr(
            finger_term, "_gripper_target_slew_endpoint_change_count", None
        )
        if (
            endpoint_seen is not True
            or not isinstance(endpoint, torch.Tensor)
            or not isinstance(close_command, torch.Tensor)
            or not isinstance(open_command, torch.Tensor)
            or endpoint.shape != (self.num_envs, 1)
            or close_command.shape != (1,)
            or open_command.shape != (1,)
            or endpoint.dtype != self._asset.data.joint_pos.dtype
            or close_command.dtype != endpoint.dtype
            or open_command.dtype != endpoint.dtype
            or endpoint.device != torch.device(self.device)
            or close_command.device != endpoint.device
            or open_command.device != endpoint.device
            or not bool(torch.isfinite(endpoint).all().item())
            or not bool(torch.isfinite(close_command).all().item())
            or not bool(torch.isfinite(open_command).all().item())
            or type(endpoint_change_count) is not int
            or endpoint_change_count < 0
        ):
            raise ValueError("PolaRiS EEF close-interlock endpoint state drift")
        expected_close = torch.full_like(close_command, GRIPPER_CLOSED_TARGET_FLOAT32)
        expected_open = torch.full_like(open_command, GRIPPER_OPEN_TARGET_FLOAT32)
        if not torch.equal(close_command, expected_close) or not torch.equal(
            open_command, expected_open
        ):
            raise ValueError("PolaRiS EEF close-interlock endpoint state drift")
        # Isaac Lab stores the per-environment processed endpoint as ``(N, 1)``
        # while BinaryJointPositionAction stores each endpoint command as
        # ``(1,)``.  The interlock is itself restricted to one environment;
        # reshape the now-validated constants to the exact live endpoint shape
        # before using shape-strict equality.
        endpoint_is_closed = torch.equal(endpoint, close_command.reshape(1, 1))
        endpoint_is_open = torch.equal(endpoint, open_command.reshape(1, 1))
        if not endpoint_is_closed and not endpoint_is_open:
            raise ValueError("PolaRiS EEF close interlock requires a binary endpoint")
        return advance_gripper_close_arm_interlock(
            enabled=True,
            previous_endpoint_change_count=(
                self._gripper_close_arm_interlock_observed_endpoint_change_count
            ),
            current_endpoint_change_count=endpoint_change_count,
            endpoint_observed_before_apply=(
                self._gripper_close_arm_interlock_endpoint_observed
            ),
            endpoint_is_closed=endpoint_is_closed,
            remaining_before_apply=self._gripper_close_arm_interlock_remaining,
        )

    def _reset_gripper_runtime_evidence(self) -> None:
        dtype = self._asset.data.joint_pos.dtype
        width = len(GRIPPER_JOINT_INDICES)
        self._gripper_apply_entry_samples = 0
        self._gripper_post_policy_step_samples = 0
        self._gripper_nonfinite_samples = 0
        self._gripper_dropped_diagnostics = 0
        self._gripper_max_abs_joint_velocity = torch.zeros(
            width, dtype=dtype, device=self.device
        )
        self._gripper_max_abs_joint_acceleration = torch.zeros_like(
            self._gripper_max_abs_joint_velocity
        )
        self._gripper_max_velocity_value = torch.tensor(
            -1.0, dtype=dtype, device=self.device
        )
        self._gripper_max_velocity_phase = torch.tensor(
            -1, dtype=torch.int64, device=self.device
        )
        self._gripper_max_velocity_sample_index = torch.tensor(
            -1, dtype=torch.int64, device=self.device
        )
        self._gripper_max_velocity_vectors = {
            field: torch.zeros(width, dtype=dtype, device=self.device)
            for field in (
                "joint_position_rad",
                "joint_velocity_rad_s",
                "joint_acceleration_rad_s2",
                "joint_position_target_rad",
                "joint_velocity_target_rad_s",
            )
        }
        self._gripper_terminal_sample_index = torch.tensor(
            -1, dtype=torch.int64, device=self.device
        )
        self._gripper_terminal_vectors = {
            field: torch.zeros(width, dtype=dtype, device=self.device)
            for field in (
                "joint_position_rad",
                "joint_velocity_rad_s",
                "joint_acceleration_rad_s2",
                "joint_position_target_rad",
                "joint_velocity_target_rad_s",
            )
        }

    def _record_gripper_runtime_sample(self, *, phase: str) -> None:
        if getattr(self, "_gripper_runtime_static", None) is None:
            return
        if phase not in {"apply_entry", "post_policy_step"}:
            raise ValueError(f"Unknown EEF gripper sample phase: {phase!r}")
        indices = list(GRIPPER_JOINT_INDICES)
        vectors = {
            "joint_position_rad": self._asset.data.joint_pos[:, indices],
            "joint_velocity_rad_s": self._asset.data.joint_vel[:, indices],
            "joint_acceleration_rad_s2": self._asset.data.joint_acc[:, indices],
            "joint_position_target_rad": self._asset.data.joint_pos_target[:, indices],
            "joint_velocity_target_rad_s": self._asset.data.joint_vel_target[
                :, indices
            ],
        }
        sample_index = (
            self._gripper_apply_entry_samples + self._gripper_post_policy_step_samples
        )
        if phase == "apply_entry":
            self._gripper_apply_entry_samples += 1
        else:
            self._gripper_post_policy_step_samples += 1
        combined = torch.cat(tuple(vectors.values()), dim=-1)
        if not torch.isfinite(combined).all():
            self._gripper_nonfinite_samples += 1
            try:
                _require_finite(combined, field="all-six gripper runtime state")
            except DifferentialIKNumericalError as error:
                if phase == "post_policy_step":
                    raise GripperRuntimePostStepError(
                        "PolaRiS all-six gripper state became non-finite only "
                        "after env.step completed; refusing to misclassify the "
                        "executed action as an unexecuted numerical-failure tail"
                    ) from error
                raise
        velocity = vectors["joint_velocity_rad_s"]
        acceleration = vectors["joint_acceleration_rad_s2"]
        self._gripper_max_abs_joint_velocity = torch.maximum(
            self._gripper_max_abs_joint_velocity,
            velocity.abs().amax(dim=0),
        )
        self._gripper_max_abs_joint_acceleration = torch.maximum(
            self._gripper_max_abs_joint_acceleration,
            acceleration.abs().amax(dim=0),
        )
        candidate = velocity[0].abs().amax()
        replace = candidate > self._gripper_max_velocity_value
        self._gripper_max_velocity_value = torch.where(
            replace, candidate, self._gripper_max_velocity_value
        )
        self._gripper_max_velocity_phase = torch.where(
            replace,
            torch.tensor(
                0 if phase == "apply_entry" else 1,
                dtype=torch.int64,
                device=self.device,
            ),
            self._gripper_max_velocity_phase,
        )
        self._gripper_max_velocity_sample_index = torch.where(
            replace,
            torch.tensor(sample_index, dtype=torch.int64, device=self.device),
            self._gripper_max_velocity_sample_index,
        )
        for field, vector in vectors.items():
            self._gripper_max_velocity_vectors[field] = torch.where(
                replace, vector[0], self._gripper_max_velocity_vectors[field]
            )
        if phase == "post_policy_step":
            self._gripper_terminal_sample_index.fill_(sample_index)
            for field, vector in vectors.items():
                self._gripper_terminal_vectors[field].copy_(vector[0])

    def record_gripper_post_policy_step(self) -> None:
        """Capture the state after the eighth physics step and scene update."""

        if self._active_episode_index is None:
            raise ValueError(
                "PolaRiS EEF gripper post-step sample has no active episode"
            )
        self._record_gripper_runtime_sample(phase="post_policy_step")

    def _gripper_runtime_dynamic_report(self) -> dict[str, object]:
        if getattr(self, "_gripper_runtime_static", None) is None:
            raise ValueError("PolaRiS EEF gripper runtime contract is not installed")
        total_samples = (
            self._gripper_apply_entry_samples + self._gripper_post_policy_step_samples
        )
        finite_samples = total_samples - self._gripper_nonfinite_samples
        if finite_samples == 0:
            maximum = None
        else:
            phase_code = int(self._gripper_max_velocity_phase.detach().cpu().item())
            maximum = {
                "sample_phase": "apply_entry"
                if phase_code == 0
                else "post_policy_step",
                "sample_index": int(
                    self._gripper_max_velocity_sample_index.detach().cpu().item()
                ),
                **{
                    field: vector.detach().cpu().tolist()
                    for field, vector in self._gripper_max_velocity_vectors.items()
                },
            }
        terminal = (
            None
            if self._gripper_post_policy_step_samples == 0
            else {
                "sample_index": int(
                    self._gripper_terminal_sample_index.detach().cpu().item()
                ),
                **{
                    field: vector.detach().cpu().tolist()
                    for field, vector in self._gripper_terminal_vectors.items()
                },
            }
        )
        target_slew_reporter = getattr(
            getattr(self, "_gripper_target_slew_term", None),
            "gripper_target_slew_dynamic_report",
            None,
        )
        if not callable(target_slew_reporter):
            raise ValueError("PolaRiS EEF gripper target-slew reporter is absent")
        return validate_eef_gripper_dynamic_evidence(
            {
                "profile": EEF_GRIPPER_RUNTIME_PROFILE,
                "joint_names": list(GRIPPER_JOINT_NAMES),
                "joint_indices": list(GRIPPER_JOINT_INDICES),
                "apply_entry_samples": self._gripper_apply_entry_samples,
                "post_policy_step_samples": self._gripper_post_policy_step_samples,
                "max_abs_joint_velocity_rad_s": self._gripper_max_abs_joint_velocity.detach()
                .cpu()
                .tolist(),
                "max_abs_joint_acceleration_rad_s2": self._gripper_max_abs_joint_acceleration.detach()
                .cpu()
                .tolist(),
                "max_velocity_diagnostic": maximum,
                "terminal_state": terminal,
                "driver_target_slew": target_slew_reporter(),
                "nonfinite_samples": self._gripper_nonfinite_samples,
                "dropped_diagnostics": self._gripper_dropped_diagnostics,
            }
        )

    def begin_safety_episode(self, episode_index: int) -> None:
        """Start isolated safety accounting for one rollout."""

        if type(episode_index) is not int or episode_index < 0:
            raise ValueError(f"Invalid EEF safety episode index: {episode_index!r}")
        self._reset_episode_safety_state(episode_index=episode_index)

    def reset(self, env_ids=None) -> None:
        """Reset base actions and all opt-in candidate state."""

        super().reset(env_ids)
        if self._wrist_energy_brake_enabled:
            self._reset_wrist_energy_brake_state(env_ids)
        self._reset_gripper_close_arm_interlock_state()

    def _soft_joint_pos_limits(self) -> torch.Tensor:
        return self._asset.data.soft_joint_pos_limits[:, self._joint_ids, :]

    def _record_nonfinite_abort(self) -> None:
        self._nonfinite_abort_count += self.num_envs

    @staticmethod
    def _first_vector(value: torch.Tensor | None) -> dict[str, object] | None:
        if value is None:
            return None
        array = value[0] if value.ndim > 1 else value
        raw_values = [float(item) for item in array.detach().cpu().tolist()]
        finite_mask = [math.isfinite(item) for item in raw_values]
        return {
            "values": [
                item if finite else None
                for item, finite in zip(raw_values, finite_mask, strict=True)
            ],
            "finite_mask": finite_mask,
            "finite_count": sum(finite_mask),
        }

    @staticmethod
    def _failure_substep_trace_vector(
        values: list[float],
    ) -> dict[str, object]:
        raw_values = [float(item) for item in values]
        finite_mask = [math.isfinite(item) for item in raw_values]
        return {
            "values": [
                item if finite else None
                for item, finite in zip(raw_values, finite_mask, strict=True)
            ],
            "finite_mask": finite_mask,
            "finite_count": sum(finite_mask),
        }

    def _copy_failure_substep_trace_value(
        self,
        *,
        field: str,
        slot: int,
        value: torch.Tensor,
    ) -> None:
        buffer = self._failure_substep_trace_buffers[field]
        expected_shape = (self.num_envs, buffer.shape[-1])
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                "PolaRiS EEF failure substep trace tensor shape drift: "
                f"field={field!r}, expected={expected_shape!r}, "
                f"actual={tuple(value.shape)!r}"
            )
        if value.device != buffer.device or value.dtype != buffer.dtype:
            raise ValueError(
                "PolaRiS EEF failure substep trace tensor device/dtype drift: "
                f"field={field!r}, expected_device={buffer.device!r}, "
                f"actual_device={value.device!r}, expected_dtype={buffer.dtype!r}, "
                f"actual_dtype={value.dtype!r}"
            )
        buffer[slot].copy_(value)

    def _finalize_pending_failure_substep_trace(
        self,
        *,
        post_joint_pos: torch.Tensor,
        post_joint_vel: torch.Tensor,
    ) -> None:
        """Attach causal post-physics state and prior actuator effort."""

        if not self._failure_substep_trace_enabled:
            return
        slot = self._failure_substep_trace_pending_slot
        if slot is None:
            if self._failure_substep_trace_pending_apply_index is not None:
                raise ValueError(
                    "PolaRiS EEF failure substep trace pending identity drift"
                )
            return
        if self._failure_substep_trace_pending_apply_index is None:
            raise ValueError(
                "PolaRiS EEF failure substep trace pending index is absent"
            )
        if (
            slot
            != self._failure_substep_trace_total_completed
            % FAILURE_SUBSTEP_TRACE_CAPACITY
            or self._failure_substep_trace_pending_apply_index
            != self._failure_substep_trace_total_completed
        ):
            raise ValueError(
                "PolaRiS EEF failure substep trace pending lifecycle drift"
            )

        pre_joint_pos = self._failure_substep_trace_buffers["joint_pos_rad"][slot]
        pre_joint_vel = self._failure_substep_trace_buffers["joint_vel_rad_s"][slot]
        computed_effort = self._asset.data.computed_torque[:, self._joint_ids]
        applied_effort = self._asset.data.applied_torque[:, self._joint_ids]
        for field, value in (
            ("post_joint_pos_rad", post_joint_pos),
            ("post_joint_vel_rad_s", post_joint_vel),
            ("delta_joint_pos_rad", post_joint_pos - pre_joint_pos),
            ("delta_joint_vel_rad_s", post_joint_vel - pre_joint_vel),
            ("approximate_pd_effort_preclip_nm", computed_effort),
            ("approximate_pd_effort_postclip_nm", applied_effort),
        ):
            self._copy_failure_substep_trace_value(
                field=field,
                slot=slot,
                value=value,
            )

        self._failure_substep_trace_total_completed += 1
        self._failure_substep_trace_pending_slot = None
        self._failure_substep_trace_pending_apply_index = None

    def _stage_failure_substep_trace(
        self,
        *,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        previous_joint_pos_target: torch.Tensor,
        raw_dls_joint_pos_target: torch.Tensor,
        new_joint_pos_target: torch.Tensor,
        new_joint_vel_target: torch.Tensor,
        new_joint_effort_target: torch.Tensor,
        current_eef_position: torch.Tensor,
        current_eef_quaternion: torch.Tensor,
        desired_eef_position: torch.Tensor,
        desired_eef_quaternion: torch.Tensor,
        pose_error: torch.Tensor,
    ) -> None:
        """Stage one accepted command; effort is attached at the next call."""

        if not self._failure_substep_trace_enabled:
            return
        if self._failure_substep_trace_pending_slot is not None:
            raise ValueError(
                "PolaRiS EEF failure substep trace has an unfinalized command"
            )
        slot = (
            self._failure_substep_trace_total_completed % FAILURE_SUBSTEP_TRACE_CAPACITY
        )
        apply_index = self._apply_call_count - 1
        if apply_index != self._failure_substep_trace_total_completed:
            raise ValueError(
                "PolaRiS EEF failure substep trace apply lifecycle drift: "
                f"expected={self._failure_substep_trace_total_completed!r}, "
                f"actual={apply_index!r}"
            )
        for buffer in self._failure_substep_trace_buffers.values():
            buffer[slot].fill_(float("nan"))
        for field, value in (
            ("joint_pos_rad", joint_pos),
            ("joint_vel_rad_s", joint_vel),
            ("previous_joint_pos_target_rad", previous_joint_pos_target),
            ("raw_dls_joint_pos_target_rad", raw_dls_joint_pos_target),
            ("new_joint_pos_target_rad", new_joint_pos_target),
            ("new_joint_vel_target_rad_s", new_joint_vel_target),
            ("new_joint_effort_target_nm", new_joint_effort_target),
            ("current_eef_position_m", current_eef_position),
            ("current_eef_quaternion_wxyz", current_eef_quaternion),
            ("desired_eef_position_m", desired_eef_position),
            ("desired_eef_quaternion_wxyz", desired_eef_quaternion),
            ("pose_error_position_m_axis_angle_rad", pose_error),
        ):
            self._copy_failure_substep_trace_value(
                field=field,
                slot=slot,
                value=value,
            )
        self._failure_substep_trace_apply_indices[slot] = apply_index
        self._failure_substep_trace_pending_slot = slot
        self._failure_substep_trace_pending_apply_index = apply_index

    def failure_substep_trace(self, episode_index: int) -> dict[str, object]:
        """Export the failure-only ring without changing the safety report schema."""

        if not self._failure_substep_trace_enabled:
            raise ValueError("PolaRiS EEF failure substep trace is disabled")
        if (
            type(episode_index) is not int
            or self._active_episode_index != episode_index
        ):
            raise ValueError(
                "PolaRiS EEF failure substep trace episode lifecycle mismatch: "
                f"active={self._active_episode_index!r}, requested={episode_index!r}"
            )
        live_drive_tensors = self._validated_failure_substep_trace_drive_tensors()
        self._validated_failure_substep_trace_zero_effort_target()
        total_completed = self._failure_substep_trace_total_completed
        pending_apply_index = self._failure_substep_trace_pending_apply_index
        pending_entry_count = int(pending_apply_index is not None)
        # A pending command owns its slot but is intentionally excluded until
        # the next apply call attaches causal post-physics state. Once full,
        # staging therefore evicts one completed prefix entry temporarily.
        completed_capacity = FAILURE_SUBSTEP_TRACE_CAPACITY - pending_entry_count
        entry_count = min(total_completed, completed_capacity)
        first_sequence = total_completed - entry_count
        logical_sequences = list(range(first_sequence, total_completed))
        slots = [
            sequence % FAILURE_SUBSTEP_TRACE_CAPACITY for sequence in logical_sequences
        ]

        if entry_count == 0:
            apply_indices: list[int] = []
            vector_values: dict[str, list[list[float]]] = {
                field: [] for field in FAILURE_SUBSTEP_TRACE_VECTOR_WIDTHS
            }
        else:
            slot_tensor = torch.tensor(
                slots,
                dtype=torch.int64,
                device=self.device,
            )
            apply_indices = [
                int(value)
                for value in self._failure_substep_trace_apply_indices.index_select(
                    0, slot_tensor
                )
                .detach()
                .cpu()
                .tolist()
            ]
            vector_values = {
                field: [
                    [float(item) for item in vector]
                    for vector in buffer.index_select(0, slot_tensor)[:, 0, :]
                    .detach()
                    .cpu()
                    .tolist()
                ]
                for field, buffer in self._failure_substep_trace_buffers.items()
            }

        if apply_indices != logical_sequences:
            raise ValueError(
                "PolaRiS EEF failure substep trace apply-index ordering drift: "
                f"expected={logical_sequences!r}, actual={apply_indices!r}"
            )
        entries: list[dict[str, object]] = []
        for entry_offset, apply_index in enumerate(apply_indices):
            entry: dict[str, object] = {
                "apply_index": apply_index,
                "policy_step": apply_index // self._decimation,
                "physics_substep": apply_index % self._decimation,
            }
            entry.update(
                {
                    field: self._failure_substep_trace_vector(
                        vector_values[field][entry_offset]
                    )
                    for field in FAILURE_SUBSTEP_TRACE_VECTOR_WIDTHS
                }
            )
            entries.append(entry)

        if pending_apply_index is not None and (
            type(pending_apply_index) is not int
            or pending_apply_index != total_completed
            or pending_apply_index >= self._apply_call_count
        ):
            raise ValueError(
                "PolaRiS EEF failure substep trace pending apply-index drift"
            )
        return {
            "schema_version": 1,
            "profile": FAILURE_SUBSTEP_TRACE_PROFILE,
            "episode_index": episode_index,
            "capacity": FAILURE_SUBSTEP_TRACE_CAPACITY,
            "policy_step_capacity": FAILURE_SUBSTEP_TRACE_CAPACITY
            // FAILURE_SUBSTEP_TRACE_DECIMATION,
            "decimation": self._decimation,
            "joint_names": list(self._joint_names),
            "joint_drive_stiffness": live_drive_tensors["joint_drive_stiffness"][0]
            .detach()
            .cpu()
            .tolist(),
            "joint_drive_damping": live_drive_tensors["joint_drive_damping"][0]
            .detach()
            .cpu()
            .tolist(),
            "joint_effort_limits": live_drive_tensors["joint_effort_limits"][0]
            .detach()
            .cpu()
            .tolist(),
            "effort_semantics": FAILURE_SUBSTEP_TRACE_EFFORT_SEMANTICS,
            "phase_contract": dict(FAILURE_SUBSTEP_TRACE_PHASE_CONTRACT),
            "completed_entry_count": entry_count,
            "total_completed_entry_count": total_completed,
            "dropped_prefix_entry_count": first_sequence,
            "pending_entry_count": pending_entry_count,
            "pending_apply_index": pending_apply_index,
            "entries": entries,
        }

    def _diagnostic_record(
        self,
        *,
        kind: str,
        joint_pos: torch.Tensor,
        raw_delta: torch.Tensor | None,
        raw_target: torch.Tensor | None,
        safe_target: torch.Tensor | None,
        pose_error: torch.Tensor | None,
        jacobian: torch.Tensor | None,
        eef_quaternion_norm: torch.Tensor | None = None,
    ) -> dict[str, object]:
        apply_index = self._apply_call_count - 1
        pose_error_norm = None
        if pose_error is not None and torch.isfinite(pose_error).all():
            raw_pose_error_norm = float(
                pose_error[0].to(torch.float64).norm().detach().cpu().item()
            )
            if math.isfinite(raw_pose_error_norm):
                pose_error_norm = raw_pose_error_norm
        jacobian_finite = (
            None
            if jacobian is None
            else bool(torch.isfinite(jacobian).all().detach().cpu().item())
        )
        jacobian_max_abs = None
        if jacobian is not None:
            finite_values = jacobian[torch.isfinite(jacobian)]
            if finite_values.numel() > 0:
                raw_jacobian_max_abs = float(
                    finite_values.to(torch.float64).abs().max().detach().cpu().item()
                )
                if math.isfinite(raw_jacobian_max_abs):
                    jacobian_max_abs = raw_jacobian_max_abs
        quaternion_norm = None
        if eef_quaternion_norm is not None:
            raw_quaternion_norm = float(eef_quaternion_norm[0].detach().cpu().item())
            if math.isfinite(raw_quaternion_norm):
                quaternion_norm = raw_quaternion_norm
        return {
            "kind": kind,
            "episode_index": self._active_episode_index,
            "policy_step": apply_index // self._decimation,
            "physics_substep": apply_index % self._decimation,
            "joint_pos_rad": self._first_vector(joint_pos),
            "raw_delta_joint_pos_rad": self._first_vector(raw_delta),
            "raw_joint_pos_target_rad": self._first_vector(raw_target),
            "safe_joint_pos_target_rad": self._first_vector(safe_target),
            "pose_error_norm": pose_error_norm,
            "jacobian_finite": jacobian_finite,
            "jacobian_max_abs": jacobian_max_abs,
            "eef_quaternion_norm": quaternion_norm,
        }

    def _append_guard_diagnostic(self, **kwargs) -> dict[str, object]:
        record = self._diagnostic_record(**kwargs)
        if len(self._guard_diagnostics) < self._max_guard_diagnostics:
            self._guard_diagnostics.append(record)
        else:
            self._guard_diagnostics_dropped += 1
        return record

    def _record_current_joint_velocity_abort(
        self,
        *,
        joint_vel: torch.Tensor,
        exceeded_joint_mask: torch.Tensor,
    ) -> dict[str, object]:
        """Capture the exact finite arm velocity state that triggered an abort."""

        if self._current_joint_velocity_abort is not None:
            raise ValueError("PolaRiS EEF current-velocity abort was recorded twice")
        if not (
            torch.isfinite(joint_vel).all()
            and torch.isfinite(self._joint_velocity_limits).all()
        ):
            raise DifferentialIKNumericalError(
                "PolaRiS EEF current-velocity abort evidence is non-finite"
            )
        excess = torch.clamp(
            joint_vel.abs() - self._joint_velocity_limits,
            min=0.0,
        )
        expected_mask = joint_vel.abs() > (
            self._joint_velocity_limits + JOINT_VELOCITY_LIMIT_TOLERANCE_RAD_S
        )
        if not torch.equal(exceeded_joint_mask, expected_mask):
            raise ValueError("PolaRiS EEF current-velocity abort mask drift")
        apply_index = self._apply_call_count - 1
        evidence: dict[str, object] = {
            "profile": CURRENT_JOINT_VELOCITY_ABORT_EVIDENCE_PROFILE,
            "episode_index": self._active_episode_index,
            "policy_step": apply_index // self._decimation,
            "physics_substep": apply_index % self._decimation,
            "joint_names": list(self._joint_names),
            "joint_velocity_rad_s": joint_vel[0].detach().cpu().tolist(),
            "joint_velocity_limit_rad_s": self._joint_velocity_limits[0]
            .detach()
            .cpu()
            .tolist(),
            "joint_velocity_limit_tolerance_rad_s": (
                JOINT_VELOCITY_LIMIT_TOLERANCE_RAD_S
            ),
            "joint_velocity_limit_excess_rad_s": excess[0].detach().cpu().tolist(),
            "exceeded_joint_mask": [
                bool(value) for value in exceeded_joint_mask[0].detach().cpu().tolist()
            ],
        }
        self._current_joint_velocity_abort = evidence
        return evidence

    def _append_wrist_energy_brake_diagnostics(
        self,
        *,
        result: _WristEnergyBrakeTarget,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        previous_applied_target: torch.Tensor,
        nominal_safe_target: torch.Tensor,
    ) -> None:
        """Capture a bounded tail of active-latch evidence after setters succeed."""

        active_environments = (
            result.active_environment_mask.nonzero(as_tuple=False)
            .flatten()
            .detach()
            .cpu()
            .tolist()
        )
        apply_index = self._apply_call_count - 1
        for environment_index in active_environments:
            if len(self._wrist_energy_brake_diagnostics) >= self._max_guard_diagnostics:
                self._wrist_energy_brake_diagnostics.pop(0)
                self._wrist_energy_brake_diagnostics_dropped += 1
            self._wrist_energy_brake_diagnostics.append(
                {
                    "episode_index": self._active_episode_index,
                    "apply_index": apply_index,
                    "policy_step": apply_index // self._decimation,
                    "physics_substep": apply_index % self._decimation,
                    "environment_index": int(environment_index),
                    "reversal_detection_armed": bool(
                        self._wrist_energy_brake_reversal_detection_armed[
                            environment_index
                        ]
                        .detach()
                        .cpu()
                        .item()
                    ),
                    "trigger_joint_mask": [
                        bool(value)
                        for value in result.trigger_joint_mask[environment_index]
                        .detach()
                        .cpu()
                        .tolist()
                    ],
                    "attempted_joint_mask": [
                        bool(value)
                        for value in result.attempted_joint_mask[environment_index]
                        .detach()
                        .cpu()
                        .tolist()
                    ],
                    "braked_joint_mask": [
                        bool(value)
                        for value in result.braked_joint_mask[environment_index]
                        .detach()
                        .cpu()
                        .tolist()
                    ],
                    "joint_pos_rad": [
                        float(value)
                        for value in joint_pos[environment_index]
                        .detach()
                        .cpu()
                        .tolist()
                    ],
                    "joint_vel_rad_s": [
                        float(value)
                        for value in joint_vel[environment_index]
                        .detach()
                        .cpu()
                        .tolist()
                    ],
                    "previous_applied_target_rad": [
                        float(value)
                        for value in previous_applied_target[environment_index]
                        .detach()
                        .cpu()
                        .tolist()
                    ],
                    "nominal_safe_target_rad": [
                        float(value)
                        for value in nominal_safe_target[environment_index]
                        .detach()
                        .cpu()
                        .tolist()
                    ],
                    "applied_target_rad": [
                        float(value)
                        for value in result.applied_target[environment_index]
                        .detach()
                        .cpu()
                        .tolist()
                    ],
                    "target_shift_rad": [
                        float(value)
                        for value in result.target_shift[environment_index]
                        .detach()
                        .cpu()
                        .tolist()
                    ],
                }
            )

    def _update_max_raw_delta_diagnostic(
        self,
        *,
        joint_pos: torch.Tensor,
        raw_delta: torch.Tensor,
        raw_target: torch.Tensor,
        safe_target: torch.Tensor,
        pose_error: torch.Tensor,
        jacobian: torch.Tensor,
    ) -> None:
        """Track the worst finite DLS update entirely on device."""

        raw_delta_is_finite = torch.isfinite(raw_delta).all()
        raw_max = torch.nan_to_num(
            raw_delta.abs(), nan=0.0, posinf=0.0, neginf=0.0
        ).amax()
        replace = raw_delta_is_finite & (raw_max > self._max_raw_delta_diagnostic_value)
        self._max_raw_delta_diagnostic_value = torch.maximum(
            self._max_raw_delta_diagnostic_value, raw_max
        )
        apply_index = torch.tensor(
            self._apply_call_count - 1, dtype=torch.int64, device=self.device
        )
        self._max_raw_delta_diagnostic_apply_index = torch.where(
            replace,
            apply_index,
            self._max_raw_delta_diagnostic_apply_index,
        )
        for attribute, value in (
            ("_max_raw_delta_diagnostic_joint_pos", joint_pos[0]),
            ("_max_raw_delta_diagnostic_raw_delta", raw_delta[0]),
            ("_max_raw_delta_diagnostic_raw_target", raw_target[0]),
            ("_max_raw_delta_diagnostic_safe_target", safe_target[0]),
        ):
            current = getattr(self, attribute)
            setattr(self, attribute, torch.where(replace, value, current))
        pose_error_norm = pose_error[0].to(torch.float64).norm()
        jacobian_max_abs = jacobian[0].to(torch.float64).abs().amax()
        self._max_raw_delta_diagnostic_pose_error_norm = torch.where(
            replace,
            pose_error_norm,
            self._max_raw_delta_diagnostic_pose_error_norm,
        )
        self._max_raw_delta_diagnostic_jacobian_max_abs = torch.where(
            replace,
            jacobian_max_abs,
            self._max_raw_delta_diagnostic_jacobian_max_abs,
        )

    def apply_actions(self):
        """Apply finite, velocity-slewed, soft-limited EEF IK joint targets."""

        if self._failure_substep_trace_enabled:
            self._finalize_pending_failure_substep_trace(
                post_joint_pos=self._asset.data.joint_pos[:, self._joint_ids],
                post_joint_vel=self._asset.data.joint_vel[:, self._joint_ids],
            )
        self._apply_call_count += 1
        close_interlock_transition = (
            self._next_gripper_close_arm_interlock_transition()
            if self._gripper_close_arm_interlock_enabled
            else DISABLED_GRIPPER_CLOSE_ARM_INTERLOCK_TRANSITION
        )
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        joint_vel = self._asset.data.joint_vel[:, self._joint_ids]
        live_joint_pos_target = (
            self._asset.data.joint_pos_target[:, self._joint_ids]
            if (self._failure_substep_trace_enabled or self._wrist_energy_brake_enabled)
            else None
        )
        previous_joint_pos_target = (
            live_joint_pos_target.clone()
            if self._failure_substep_trace_enabled
            else None
        )
        soft_limits = self._soft_joint_position_limits
        lower = soft_limits[..., 0]
        upper = soft_limits[..., 1]
        current_joint_violation = torch.maximum(
            torch.clamp(lower - joint_pos, min=0.0),
            torch.clamp(joint_pos - upper, min=0.0),
        )
        current_joint_invalid = (
            current_joint_violation > CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD
        ).any(dim=-1)
        hard_limits = self._physx_hard_joint_position_limits
        hard_lower = hard_limits[..., 0]
        hard_upper = hard_limits[..., 1]
        current_hard_limit_violation = torch.maximum(
            torch.clamp(hard_lower - joint_pos, min=0.0),
            torch.clamp(joint_pos - hard_upper, min=0.0),
        )
        current_outer_clearance = torch.minimum(joint_pos - lower, upper - joint_pos)
        current_joint_velocity_exceeded = (
            joint_vel.abs()
            > self._joint_velocity_limits + JOINT_VELOCITY_LIMIT_TOLERANCE_RAD_S
        )
        current_joint_velocity_invalid = current_joint_velocity_exceeded.any(dim=-1)
        jacobian = None
        pose_error = None
        raw_joint_pos_target = None
        wrist_target_state_invalid = None
        fallback_count_before = self._ik_controller.fallback_count
        try:
            self._record_gripper_runtime_sample(phase="apply_entry")
            current_state = torch.cat(
                (ee_pos_curr, ee_quat_curr, joint_pos, joint_vel), dim=-1
            )
            desired_state = torch.cat(
                (
                    self._ik_controller.ee_pos_des,
                    self._ik_controller.ee_quat_des,
                ),
                dim=-1,
            )
            current_quaternion_norms, current_quaternion_norm_valid = (
                _eef_quaternion_norm_is_valid(ee_quat_curr)
            )
            desired_quaternion_norms, desired_quaternion_norm_valid = (
                _eef_quaternion_norm_is_valid(self._ik_controller.ee_quat_des)
            )
            status_tensors = [
                torch.isfinite(current_state).all(),
                torch.isfinite(desired_state).all(),
                current_quaternion_norm_valid.all(),
                desired_quaternion_norm_valid.all(),
                ~current_joint_invalid.any(),
                ~current_joint_velocity_invalid.any(),
            ]
            if self._wrist_energy_brake_enabled:
                if live_joint_pos_target is None:
                    raise ValueError(
                        "PolaRiS EEF wrist energy brake lacks live target state"
                    )
                wrist_target_state_invalid = (
                    self._wrist_energy_brake_previous_target_valid
                    & ~(
                        live_joint_pos_target
                        == self._wrist_energy_brake_previous_applied_target
                    ).all(dim=-1)
                )
                status_tensors.append(~wrist_target_state_invalid.any())
            status_values = tuple(
                bool(value)
                for value in torch.stack(status_tensors).detach().cpu().tolist()
            )
            (
                current_finite,
                desired_finite,
                current_quaternion_valid,
                desired_quaternion_valid,
                current_joint_valid,
                current_joint_velocity_valid,
            ) = status_values[:6]
            wrist_target_state_valid = (
                status_values[6] if self._wrist_energy_brake_enabled else True
            )
            if not current_finite:
                # Re-enter the diagnostic helper only on the abort path; the
                # healthy path uses the single combined status synchronization.
                _require_finite(current_state, field="current EEF/joint state")
            self._max_current_joint_soft_limit_violation = torch.maximum(
                self._max_current_joint_soft_limit_violation,
                current_joint_violation.amax(dim=0).to(
                    self._max_current_joint_soft_limit_violation.dtype
                ),
            )
            self._max_current_physx_hard_limit_violation = torch.maximum(
                self._max_current_physx_hard_limit_violation,
                current_hard_limit_violation.amax(dim=0).to(
                    self._max_current_physx_hard_limit_violation.dtype
                ),
            )
            self._max_abs_joint_vel = torch.maximum(
                self._max_abs_joint_vel,
                joint_vel.abs().amax(dim=0).to(self._max_abs_joint_vel.dtype),
            )
            self._minimum_outer_joint_clearance = torch.minimum(
                self._minimum_outer_joint_clearance,
                current_outer_clearance.amin(dim=0).to(
                    self._minimum_outer_joint_clearance.dtype
                ),
            )
            if not current_joint_velocity_valid:
                self._invariant_abort_count += current_joint_velocity_invalid.sum()
                velocity_abort = self._record_current_joint_velocity_abort(
                    joint_vel=joint_vel,
                    exceeded_joint_mask=current_joint_velocity_exceeded,
                )
                self._append_guard_diagnostic(
                    kind="current_joint_velocity_limit_abort",
                    joint_pos=joint_pos,
                    raw_delta=None,
                    raw_target=None,
                    safe_target=None,
                    pose_error=None,
                    jacobian=None,
                )
                raise DifferentialIKInvariantError(
                    format_current_joint_velocity_abort_message(velocity_abort)
                )
            if not desired_finite:
                _require_finite(desired_state, field="desired EEF pose")
            if not current_quaternion_valid:
                self._invariant_abort_count += self.num_envs
                record = self._append_guard_diagnostic(
                    kind="current_eef_quaternion_invariant_abort",
                    joint_pos=joint_pos,
                    raw_delta=None,
                    raw_target=None,
                    safe_target=None,
                    pose_error=None,
                    jacobian=None,
                    eef_quaternion_norm=current_quaternion_norms,
                )
                raise DifferentialIKInvariantError(
                    "PolaRiS EEF IK current quaternion norm violates the named "
                    "unit invariant; aborting before PhysX "
                    f"(norm={record['eef_quaternion_norm']!r}, "
                    f"tolerance={EEF_QUATERNION_UNIT_NORM_TOLERANCE:g})"
                )
            if not desired_quaternion_valid:
                self._invariant_abort_count += self.num_envs
                record = self._append_guard_diagnostic(
                    kind="desired_eef_quaternion_invariant_abort",
                    joint_pos=joint_pos,
                    raw_delta=None,
                    raw_target=None,
                    safe_target=None,
                    pose_error=None,
                    jacobian=None,
                    eef_quaternion_norm=desired_quaternion_norms,
                )
                raise DifferentialIKInvariantError(
                    "PolaRiS EEF IK desired quaternion norm violates the named "
                    "unit invariant; aborting before PhysX "
                    f"(norm={record['eef_quaternion_norm']!r}, "
                    f"tolerance={EEF_QUATERNION_UNIT_NORM_TOLERANCE:g})"
                )
            if not current_joint_valid:
                self._current_joint_limit_abort_count += current_joint_invalid.sum()
                self._append_guard_diagnostic(
                    kind="current_joint_limit_abort",
                    joint_pos=joint_pos,
                    raw_delta=None,
                    raw_target=None,
                    safe_target=None,
                    pose_error=None,
                    jacobian=None,
                )
                raise DifferentialIKInvariantError(
                    "PolaRiS EEF IK current joint position is outside live soft "
                    "limits; aborting before DLS and PhysX"
                )
            if not wrist_target_state_valid:
                if wrist_target_state_invalid is None:
                    raise ValueError(
                        "PolaRiS EEF wrist energy-brake target-state evidence is absent"
                    )
                self._invariant_abort_count += wrist_target_state_invalid.sum()
                self._append_guard_diagnostic(
                    kind="wrist_energy_brake_target_state_abort",
                    joint_pos=joint_pos,
                    raw_delta=None,
                    raw_target=None,
                    safe_target=None,
                    pose_error=None,
                    jacobian=None,
                )
                raise DifferentialIKInvariantError(
                    "PolaRiS EEF wrist energy-brake stored applied target "
                    "drifted from the live articulation mirror; aborting "
                    "before DLS and PhysX"
                )
            jacobian = self._compute_frame_jacobian()
            position_error, axis_angle_error = compute_pose_error(
                ee_pos_curr,
                ee_quat_curr,
                self._ik_controller.ee_pos_des,
                self._ik_controller.ee_quat_des,
                rot_error_type="axis_angle",
            )
            pose_error = torch.cat((position_error, axis_angle_error), dim=1)
            raw_joint_pos_target = self._ik_controller.compute(
                ee_pos_curr, ee_quat_curr, jacobian, joint_pos
            )
        except DifferentialIKInvariantError:
            raise
        except DifferentialIKNumericalError:
            self._record_nonfinite_abort()
            self._append_guard_diagnostic(
                kind="nonfinite_abort",
                joint_pos=joint_pos,
                raw_delta=None,
                raw_target=raw_joint_pos_target,
                safe_target=None,
                pose_error=pose_error,
                jacobian=jacobian,
            )
            raise

        nominal_safe_target, raw_delta, slew_limited, position_limited = (
            _bound_joint_position_target(
                joint_pos,
                raw_joint_pos_target,
                self._nominal_max_delta_joint_pos,
                soft_limits,
                target_guard_band_delta_joint_pos=self._max_delta_joint_pos,
            )
        )
        wrist_energy_brake_result = None
        if self._wrist_energy_brake_enabled:
            wrist_energy_brake_result = _apply_wrist_energy_brake_target(
                joint_pos,
                joint_vel,
                self._wrist_energy_brake_previous_applied_target,
                self._wrist_energy_brake_reversal_detection_armed,
                nominal_safe_target,
                self._max_delta_joint_pos,
                soft_limits,
                self._wrist_energy_brake_latch_remaining,
                self._wrist_energy_brake_joint_indices,
            )
            safe_target = wrist_energy_brake_result.applied_target
        else:
            safe_target = nominal_safe_target
        if close_interlock_transition.active:
            safe_target, _, _, _ = _bound_joint_position_target(
                joint_pos,
                joint_pos,
                self._nominal_max_delta_joint_pos,
                soft_limits,
                target_guard_band_delta_joint_pos=self._max_delta_joint_pos,
            )
        applied_delta = safe_target - joint_pos
        raw_target_violation = torch.maximum(
            torch.clamp(lower - raw_joint_pos_target, min=0.0),
            torch.clamp(raw_joint_pos_target - upper, min=0.0),
        )
        post_clamp_violation = torch.maximum(
            torch.clamp(lower - safe_target, min=0.0),
            torch.clamp(safe_target - upper, min=0.0),
        )
        target_lower = lower + self._max_delta_joint_pos
        target_upper = upper - self._max_delta_joint_pos
        post_clamp_guard_band_violation = torch.maximum(
            torch.clamp(target_lower - safe_target, min=0.0),
            torch.clamp(safe_target - target_upper, min=0.0),
        )
        guard_band_recovery_invalid = (
            post_clamp_guard_band_violation
            > current_joint_violation + JOINT_SLEW_FLOAT32_TOLERANCE_RAD
        )

        self._slew_limit_event_count += slew_limited.any(dim=-1).sum()
        self._slew_limit_joint_count += slew_limited.sum()
        self._position_limit_event_count += position_limited.any(dim=-1).sum()
        self._position_limit_joint_count += position_limited.sum()
        raw_target_nonfinite = ~torch.isfinite(raw_joint_pos_target).all(dim=-1)
        post_clamp_target_invalid = (
            (post_clamp_violation > 0.0)
            | (post_clamp_guard_band_violation > CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD)
            | guard_band_recovery_invalid
        ).any(dim=-1)
        post_clamp_slew_invalid = (
            applied_delta.abs()
            > self._nominal_max_delta_joint_pos + JOINT_SLEW_FLOAT32_TOLERANCE_RAD
        ).any(dim=-1)
        self._post_clamp_target_violation_count += post_clamp_target_invalid.sum()
        self._max_raw_delta_joint_pos = torch.maximum(
            self._max_raw_delta_joint_pos,
            torch.nan_to_num(raw_delta.abs(), nan=0.0, posinf=0.0, neginf=0.0)
            .amax(dim=0)
            .to(self._max_raw_delta_joint_pos.dtype),
        )
        self._max_raw_target_soft_limit_violation = torch.maximum(
            self._max_raw_target_soft_limit_violation,
            torch.nan_to_num(raw_target_violation, nan=0.0, posinf=0.0, neginf=0.0)
            .amax(dim=0)
            .to(self._max_raw_target_soft_limit_violation.dtype),
        )
        self._max_post_clamp_target_soft_limit_violation = torch.maximum(
            self._max_post_clamp_target_soft_limit_violation,
            torch.nan_to_num(post_clamp_violation, nan=0.0, posinf=0.0, neginf=0.0)
            .amax(dim=0)
            .to(self._max_post_clamp_target_soft_limit_violation.dtype),
        )
        self._max_post_clamp_target_guard_band_violation = torch.maximum(
            self._max_post_clamp_target_guard_band_violation,
            torch.nan_to_num(
                post_clamp_guard_band_violation,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            .amax(dim=0)
            .to(self._max_post_clamp_target_guard_band_violation.dtype),
        )
        if pose_error is not None and jacobian is not None:
            self._update_max_raw_delta_diagnostic(
                joint_pos=joint_pos,
                raw_delta=raw_delta,
                raw_target=raw_joint_pos_target,
                safe_target=safe_target,
                pose_error=pose_error,
                jacobian=jacobian,
            )
        if self._ik_controller.fallback_count > fallback_count_before:
            self._append_guard_diagnostic(
                kind="dls_pseudoinverse_fallback",
                joint_pos=joint_pos,
                raw_delta=raw_delta,
                raw_target=raw_joint_pos_target,
                safe_target=safe_target,
                pose_error=pose_error,
                jacobian=jacobian,
            )
        raw_nonfinite, target_invalid, slew_invalid = (
            bool(value)
            for value in torch.stack(
                (
                    raw_target_nonfinite.any(),
                    post_clamp_target_invalid.any(),
                    post_clamp_slew_invalid.any(),
                )
            )
            .detach()
            .cpu()
            .tolist()
        )
        if raw_nonfinite:
            self._record_nonfinite_abort()
            self._append_guard_diagnostic(
                kind="nonfinite_abort",
                joint_pos=joint_pos,
                raw_delta=raw_delta,
                raw_target=raw_joint_pos_target,
                safe_target=safe_target,
                pose_error=pose_error,
                jacobian=jacobian,
            )
            raise DifferentialIKNumericalError(
                "PolaRiS EEF IK safety received a non-finite raw DLS joint "
                "target; aborting before PhysX"
            )
        if target_invalid:
            self._invariant_abort_count += self.num_envs
            self._append_guard_diagnostic(
                kind="post_clamp_position_invariant_abort",
                joint_pos=joint_pos,
                raw_delta=raw_delta,
                raw_target=raw_joint_pos_target,
                safe_target=safe_target,
                pose_error=pose_error,
                jacobian=jacobian,
            )
            raise DifferentialIKNumericalError(
                "PolaRiS EEF IK safety produced a target that violated the "
                "outer-limit or guard-band recovery invariant after clamping"
            )
        if slew_invalid:
            self._invariant_abort_count += self.num_envs
            self._append_guard_diagnostic(
                kind="post_clamp_slew_invariant_abort",
                joint_pos=joint_pos,
                raw_delta=raw_delta,
                raw_target=raw_joint_pos_target,
                safe_target=safe_target,
                pose_error=pose_error,
                jacobian=jacobian,
            )
            raise DifferentialIKNumericalError(
                "PolaRiS EEF IK safety exceeded its live physics-substep "
                "joint slew bound after soft-limit clamping"
            )

        self._max_applied_delta_joint_pos = torch.maximum(
            self._max_applied_delta_joint_pos,
            applied_delta.abs().amax(dim=0).to(self._max_applied_delta_joint_pos.dtype),
        )
        self._asset.set_joint_velocity_target(
            self._zero_joint_velocity_target,
            self._joint_ids,
        )
        self._asset.set_joint_position_target(safe_target, self._joint_ids)
        activation_count_after_apply = (
            self._gripper_close_arm_interlock_activation_count
            + close_interlock_transition.activation_count_delta
        )
        self._gripper_close_arm_interlock_remaining = (
            close_interlock_transition.remaining_after_successful_apply
        )
        self._gripper_close_arm_interlock_observed_endpoint_change_count = (
            close_interlock_transition.observed_endpoint_change_count
        )
        self._gripper_close_arm_interlock_endpoint_observed = (
            close_interlock_transition.endpoint_observed_after_successful_apply
        )
        self._gripper_close_arm_interlock_activation_count += (
            close_interlock_transition.activation_count_delta
        )
        self._gripper_close_arm_interlock_active_apply_count += int(
            close_interlock_transition.active
        )
        if close_interlock_transition.active:
            self._gripper_close_arm_interlock_max_abs_active_delta = torch.maximum(
                self._gripper_close_arm_interlock_max_abs_active_delta,
                applied_delta[0].abs(),
            )
        if (
            self._gripper_close_arm_interlock_enabled
            and activation_count_after_apply > 0
            and not close_interlock_transition.active
        ):
            self._gripper_close_arm_interlock_released_apply_count += 1
            self._gripper_close_arm_interlock_max_abs_released_delta = torch.maximum(
                self._gripper_close_arm_interlock_max_abs_released_delta,
                applied_delta[0].abs(),
            )
        if self._wrist_energy_brake_enabled:
            if wrist_energy_brake_result is None:
                raise ValueError("PolaRiS EEF wrist energy-brake transition is absent")
            self._append_wrist_energy_brake_diagnostics(
                result=wrist_energy_brake_result,
                joint_pos=joint_pos,
                joint_vel=joint_vel,
                previous_applied_target=(
                    self._wrist_energy_brake_previous_applied_target
                ),
                nominal_safe_target=nominal_safe_target,
            )
            self._wrist_energy_brake_trigger_event_count += (
                wrist_energy_brake_result.trigger_joint_mask.any(dim=-1).sum()
            )
            self._wrist_energy_brake_active_substep_count += (
                wrist_energy_brake_result.active_environment_mask.sum()
            )
            self._wrist_energy_brake_attempted_joint_target_count += (
                wrist_energy_brake_result.attempted_joint_mask.sum()
            )
            self._wrist_energy_brake_braked_joint_target_count += (
                wrist_energy_brake_result.braked_joint_mask.sum()
            )
            self._wrist_energy_brake_latch_remaining.copy_(
                wrist_energy_brake_result.next_latch_remaining
            )
            self._wrist_energy_brake_previous_applied_target.copy_(safe_target)
            self._wrist_energy_brake_previous_target_valid.fill_(True)
            self._wrist_energy_brake_reversal_detection_armed.copy_(
                ~wrist_energy_brake_result.active_environment_mask
            )
        if self._failure_substep_trace_enabled:
            if previous_joint_pos_target is None or pose_error is None:
                raise ValueError(
                    "PolaRiS EEF failure substep trace command evidence is absent"
                )
            self._stage_failure_substep_trace(
                joint_pos=joint_pos,
                joint_vel=joint_vel,
                previous_joint_pos_target=previous_joint_pos_target,
                raw_dls_joint_pos_target=raw_joint_pos_target,
                new_joint_pos_target=safe_target,
                new_joint_vel_target=self._zero_joint_velocity_target,
                new_joint_effort_target=self._asset.data.joint_effort_target[
                    :, self._joint_ids
                ],
                current_eef_position=ee_pos_curr,
                current_eef_quaternion=ee_quat_curr,
                desired_eef_position=self._ik_controller.ee_pos_des,
                desired_eef_quaternion=self._ik_controller.ee_quat_des,
                pose_error=pose_error,
            )

    def controller_repair_candidate_report(self) -> dict[str, object]:
        """Return isolated evidence without changing the production report schema."""

        expected_nominal = self._max_delta_joint_pos * (
            ARM_SLEW_HEADROOM_RATIO if self._arm_slew_headroom_enabled else 1.0
        )
        if not torch.equal(expected_nominal, self._nominal_max_delta_joint_pos):
            raise ValueError("PolaRiS EEF nominal arm-slew candidate state drift")
        if not self._gripper_close_arm_interlock_enabled and (
            self._gripper_close_arm_interlock_remaining != 0
            or self._gripper_close_arm_interlock_activation_count != 0
            or self._gripper_close_arm_interlock_active_apply_count != 0
            or self._gripper_close_arm_interlock_released_apply_count != 0
            or self._gripper_close_arm_interlock_endpoint_observed
            or bool(
                self._gripper_close_arm_interlock_max_abs_active_delta.any()
                .detach()
                .cpu()
                .item()
            )
            or bool(
                self._gripper_close_arm_interlock_max_abs_released_delta.any()
                .detach()
                .cpu()
                .item()
            )
        ):
            raise ValueError("Disabled PolaRiS EEF close interlock has evidence")
        return {
            "arm_slew_headroom": {
                "enabled": self._arm_slew_headroom_enabled,
                "profile": ARM_SLEW_HEADROOM_CANDIDATE_PROFILE,
                "ratio": ARM_SLEW_HEADROOM_RATIO,
                "physical_max_delta_joint_pos_rad": self._max_delta_joint_pos[0]
                .detach()
                .cpu()
                .tolist(),
                "nominal_max_delta_joint_pos_rad": (
                    self._nominal_max_delta_joint_pos[0].detach().cpu().tolist()
                ),
            },
            "gripper_close_arm_interlock": {
                "enabled": self._gripper_close_arm_interlock_enabled,
                "profile": GRIPPER_CLOSE_ARM_INTERLOCK_CANDIDATE_PROFILE,
                "configured_substeps": GRIPPER_CLOSE_ARM_INTERLOCK_SUBSTEPS,
                "remaining_substeps": self._gripper_close_arm_interlock_remaining,
                "observed_endpoint_change_count": (
                    self._gripper_close_arm_interlock_observed_endpoint_change_count
                ),
                "endpoint_observed": (
                    self._gripper_close_arm_interlock_endpoint_observed
                ),
                "activation_count": (
                    self._gripper_close_arm_interlock_activation_count
                ),
                "active_apply_count": (
                    self._gripper_close_arm_interlock_active_apply_count
                ),
                "max_abs_active_delta_joint_pos_rad": (
                    self._gripper_close_arm_interlock_max_abs_active_delta.detach()
                    .cpu()
                    .tolist()
                ),
                "released_apply_count": (
                    self._gripper_close_arm_interlock_released_apply_count
                ),
                "max_abs_released_delta_joint_pos_rad": (
                    self._gripper_close_arm_interlock_max_abs_released_delta.detach()
                    .cpu()
                    .tolist()
                ),
            },
        }

    def safety_report(self) -> dict[str, object]:
        """Return JSON-serializable evidence for the active rollout."""

        live_physx_solver_type = int(self._physx_cfg.solver_type)
        if live_physx_solver_type != self._physx_solver_type:
            raise ValueError("PolaRiS EEF live PhysX solver type drifted")
        soft_limits = self._soft_joint_position_limits
        target_limits = self._physx_hard_joint_position_limits
        physx_hard_limit_readback = self._asset.root_physx_view.get_dof_limits().to(
            self._asset.device
        )[:, self._joint_ids, :]
        mirror_hard_limits = self._asset.data.joint_pos_limits[:, self._joint_ids, :]
        derived_soft_limits = self._asset.data.soft_joint_pos_limits[
            :, self._joint_ids, :
        ]
        live_velocity_target = self._asset.data.joint_vel_target[:, self._joint_ids]
        live_joint_velocity_limits = (
            self._asset.root_physx_view.get_dof_max_velocities()
            .to(self._asset.device)[:, self._joint_ids]
            .clone()
        )
        live_joint_effort_limits = (
            self._asset.root_physx_view.get_dof_max_forces()
            .to(self._asset.device)[:, self._joint_ids]
            .clone()
        )
        (
            live_solver_position_iterations,
            live_solver_velocity_iterations,
        ) = _read_articulation_solver_iteration_counts(self._asset)
        if not (
            torch.equal(physx_hard_limit_readback, target_limits)
            and torch.equal(mirror_hard_limits, target_limits)
        ):
            raise ValueError(
                "PolaRiS EEF PhysX hard-limit readback drifted after installation"
            )
        if not torch.equal(
            derived_soft_limits,
            self._physx_derived_soft_joint_position_limits,
        ):
            raise ValueError(
                "PolaRiS EEF PhysX-derived soft-limit readback drifted after "
                "installation"
            )
        if not torch.equal(live_velocity_target, self._zero_joint_velocity_target):
            raise ValueError("PolaRiS EEF live arm velocity target is not exactly zero")
        if not (
            torch.equal(live_joint_velocity_limits, self._joint_velocity_limits)
            and torch.equal(live_joint_effort_limits, self._joint_effort_limits)
        ):
            raise ValueError("PolaRiS EEF live PhysX joint-limit readback drifted")
        if not (
            live_solver_position_iterations == self._solver_position_iteration_counts
            and live_solver_velocity_iterations
            == self._solver_velocity_iteration_counts
        ):
            raise ValueError(
                "PolaRiS EEF composed articulation solver readback drifted"
            )
        if self._wrist_energy_brake_enabled:
            live_position_target = self._asset.data.joint_pos_target[:, self._joint_ids]
            stored_target_mismatch = self._wrist_energy_brake_previous_target_valid & ~(
                live_position_target == self._wrist_energy_brake_previous_applied_target
            ).all(dim=-1)
            if bool(stored_target_mismatch.any().detach().cpu().item()):
                raise ValueError(
                    "PolaRiS EEF wrist energy-brake stored applied target "
                    "drifted before safety reporting"
                )
            _require_finite(
                self._wrist_energy_brake_previous_applied_target,
                field="wrist energy-brake stored applied target",
            )
            if bool(
                (
                    (self._wrist_energy_brake_latch_remaining < 0)
                    | (
                        self._wrist_energy_brake_latch_remaining
                        > WRIST_ENERGY_BRAKE_LATCH_SUBSTEPS
                    )
                )
                .any()
                .detach()
                .cpu()
                .item()
            ):
                raise ValueError("PolaRiS EEF wrist energy-brake latch state drifted")
        soft_limit_bytes = (
            soft_limits[0].detach().cpu().numpy().astype("<f4", copy=False).tobytes()
        )
        target_limit_bytes = (
            target_limits[0].detach().cpu().numpy().astype("<f4", copy=False).tobytes()
        )
        physx_hard_limit_readback_bytes = (
            physx_hard_limit_readback[0]
            .detach()
            .cpu()
            .numpy()
            .astype("<f4", copy=False)
            .tobytes()
        )
        physx_derived_soft_limit_readback_bytes = (
            derived_soft_limits[0]
            .detach()
            .cpu()
            .numpy()
            .astype("<f4", copy=False)
            .tobytes()
        )
        max_diagnostic_index = int(
            self._max_raw_delta_diagnostic_apply_index.detach().cpu().item()
        )
        if max_diagnostic_index < 0:
            max_raw_delta_diagnostic = None
        else:
            max_raw_delta_diagnostic = {
                "kind": "max_raw_delta",
                "episode_index": self._active_episode_index,
                "policy_step": max_diagnostic_index // self._decimation,
                "physics_substep": max_diagnostic_index % self._decimation,
                "joint_pos_rad": self._first_vector(
                    self._max_raw_delta_diagnostic_joint_pos
                ),
                "raw_delta_joint_pos_rad": self._first_vector(
                    self._max_raw_delta_diagnostic_raw_delta
                ),
                "raw_joint_pos_target_rad": self._first_vector(
                    self._max_raw_delta_diagnostic_raw_target
                ),
                "safe_joint_pos_target_rad": self._first_vector(
                    self._max_raw_delta_diagnostic_safe_target
                ),
                "pose_error_norm": float(
                    self._max_raw_delta_diagnostic_pose_error_norm.detach().cpu().item()
                ),
                "jacobian_finite": True,
                "jacobian_max_abs": float(
                    self._max_raw_delta_diagnostic_jacobian_max_abs.detach()
                    .cpu()
                    .item()
                ),
                "eef_quaternion_norm": None,
            }
        report: dict[str, object] = {
            "episode_index": self._active_episode_index,
            "profile": self._safety_profile,
            "apply_actions_cadence": EEF_IK_APPLY_CADENCE,
            "physics_dt": self._physics_dt,
            "control_dt": self._control_dt,
            "decimation": self._decimation,
            "current_joint_soft_limit_tolerance_rad": CURRENT_JOINT_SOFT_LIMIT_TOLERANCE_RAD,
            "target_soft_limit_guard_band_profile": TARGET_SOFT_LIMIT_GUARD_BAND_PROFILE,
            "physx_hard_limit_profile": PHYSX_HARD_LIMIT_PROFILE,
            "physx_derived_soft_limit_profile": PHYSX_DERIVED_SOFT_LIMIT_PROFILE,
            "physx_hard_limit_write_count": self._physx_hard_limit_write_count,
            "arm_velocity_target_profile": ARM_VELOCITY_TARGET_PROFILE,
            "articulation_solver_profile": ARTICULATION_SOLVER_PROFILE,
            "articulation_solver_readback": ARTICULATION_SOLVER_READBACK,
            "physx_solver_type": live_physx_solver_type,
            "solver_position_iteration_count": live_solver_position_iterations[0],
            "solver_velocity_iteration_count": live_solver_velocity_iterations[0],
            "joint_velocity_limit_tolerance_rad_s": JOINT_VELOCITY_LIMIT_TOLERANCE_RAD_S,
            "eef_quaternion_unit_norm_tolerance": EEF_QUATERNION_UNIT_NORM_TOLERANCE,
            "joint_slew_float32_tolerance_rad": JOINT_SLEW_FLOAT32_TOLERANCE_RAD,
            "soft_joint_pos_limit_factor": self._soft_joint_pos_limit_factor,
            "joint_names": list(self._joint_names),
            "joint_velocity_limits_rad_s": live_joint_velocity_limits[0]
            .detach()
            .cpu()
            .tolist(),
            "joint_effort_limits": live_joint_effort_limits[0].detach().cpu().tolist(),
            "max_delta_joint_pos_rad": self._max_delta_joint_pos[0]
            .detach()
            .cpu()
            .tolist(),
            "target_soft_limit_margin_rad": self._max_delta_joint_pos[0]
            .detach()
            .cpu()
            .tolist(),
            "target_joint_pos_limits_rad": target_limits[0].detach().cpu().tolist(),
            "target_joint_pos_limits_float32_sha256": hashlib.sha256(
                target_limit_bytes
            ).hexdigest(),
            "physx_hard_joint_pos_limits_rad": physx_hard_limit_readback[0]
            .detach()
            .cpu()
            .tolist(),
            "physx_hard_joint_pos_limits_float32_sha256": hashlib.sha256(
                physx_hard_limit_readback_bytes
            ).hexdigest(),
            "physx_derived_soft_joint_pos_limits_rad": derived_soft_limits[0]
            .detach()
            .cpu()
            .tolist(),
            "physx_derived_soft_joint_pos_limits_float32_sha256": hashlib.sha256(
                physx_derived_soft_limit_readback_bytes
            ).hexdigest(),
            "arm_velocity_target_rad_s": live_velocity_target[0]
            .detach()
            .cpu()
            .tolist(),
            "soft_joint_pos_limits_rad": soft_limits[0].detach().cpu().tolist(),
            "soft_joint_pos_limits_float32_sha256": hashlib.sha256(
                soft_limit_bytes
            ).hexdigest(),
            "counters": {
                "apply_calls": self._apply_call_count,
                "environment_substeps": self._apply_call_count * self.num_envs,
                "slew_limit_events": int(
                    self._slew_limit_event_count.detach().cpu().item()
                ),
                "slew_limited_joints": int(
                    self._slew_limit_joint_count.detach().cpu().item()
                ),
                "position_limit_events": int(
                    self._position_limit_event_count.detach().cpu().item()
                ),
                "position_limited_joints": int(
                    self._position_limit_joint_count.detach().cpu().item()
                ),
                "post_clamp_target_violations": int(
                    self._post_clamp_target_violation_count.detach().cpu().item()
                ),
                "current_joint_limit_aborts": int(
                    self._current_joint_limit_abort_count.detach().cpu().item()
                ),
                "invariant_aborts": int(
                    self._invariant_abort_count.detach().cpu().item()
                ),
                "nonfinite_aborts": int(
                    self._nonfinite_abort_count.detach().cpu().item()
                ),
                "dls_fallbacks": (
                    self._ik_controller.fallback_count
                    - self._fallback_count_at_episode_start
                ),
                "guard_diagnostics_dropped": self._guard_diagnostics_dropped,
            },
            "maxima": {
                "raw_delta_joint_pos_rad": self._max_raw_delta_joint_pos.detach()
                .cpu()
                .tolist(),
                "applied_delta_joint_pos_rad": self._max_applied_delta_joint_pos.detach()
                .cpu()
                .tolist(),
                "raw_target_soft_limit_violation_rad": self._max_raw_target_soft_limit_violation.detach()
                .cpu()
                .tolist(),
                "post_clamp_target_soft_limit_violation_rad": self._max_post_clamp_target_soft_limit_violation.detach()
                .cpu()
                .tolist(),
                "post_clamp_target_guard_band_violation_rad": self._max_post_clamp_target_guard_band_violation.detach()
                .cpu()
                .tolist(),
                "current_joint_soft_limit_violation_rad": self._max_current_joint_soft_limit_violation.detach()
                .cpu()
                .tolist(),
                "current_physx_hard_limit_violation_rad": self._max_current_physx_hard_limit_violation.detach()
                .cpu()
                .tolist(),
                "abs_joint_vel_rad_s": self._max_abs_joint_vel.detach().cpu().tolist(),
                "minimum_outer_joint_clearance_rad": (
                    self._minimum_outer_joint_clearance
                    if self._apply_call_count > 0
                    else torch.zeros_like(self._minimum_outer_joint_clearance)
                )
                .detach()
                .cpu()
                .tolist(),
            },
            "guard_diagnostics": list(self._guard_diagnostics),
            "max_raw_delta_diagnostic": max_raw_delta_diagnostic,
            "current_joint_velocity_abort": (
                None
                if self._current_joint_velocity_abort is None
                else dict(self._current_joint_velocity_abort)
            ),
        }
        if self._wrist_energy_brake_enabled:
            counters = report["counters"]
            if not isinstance(counters, dict):
                raise ValueError("PolaRiS EEF safety counter object drifted")
            counters.update(
                {
                    "wrist_energy_brake_trigger_events": int(
                        self._wrist_energy_brake_trigger_event_count.detach()
                        .cpu()
                        .item()
                    ),
                    "wrist_energy_brake_active_substeps": int(
                        self._wrist_energy_brake_active_substep_count.detach()
                        .cpu()
                        .item()
                    ),
                    "wrist_energy_brake_attempted_joint_targets": int(
                        self._wrist_energy_brake_attempted_joint_target_count.detach()
                        .cpu()
                        .item()
                    ),
                    "wrist_energy_brake_braked_joint_targets": int(
                        self._wrist_energy_brake_braked_joint_target_count.detach()
                        .cpu()
                        .item()
                    ),
                    "wrist_energy_brake_diagnostics_dropped": (
                        self._wrist_energy_brake_diagnostics_dropped
                    ),
                }
            )
            report.update(
                {
                    "wrist_energy_brake_profile": WRIST_ENERGY_BRAKE_PROFILE,
                    "wrist_energy_brake_joint_names": list(
                        WRIST_ENERGY_BRAKE_JOINT_NAMES
                    ),
                    "wrist_energy_brake_latch_substeps": (
                        WRIST_ENERGY_BRAKE_LATCH_SUBSTEPS
                    ),
                    "wrist_energy_brake_target_shift_fraction": (
                        WRIST_ENERGY_BRAKE_TARGET_SHIFT_FRACTION
                    ),
                    "wrist_energy_brake_target_shift_threshold_rad": (
                        self._wrist_energy_brake_target_shift_threshold[0]
                        .detach()
                        .cpu()
                        .tolist()
                    ),
                    "wrist_energy_brake_latch_remaining_substeps": [
                        int(value)
                        for value in self._wrist_energy_brake_latch_remaining.detach()
                        .cpu()
                        .tolist()
                    ],
                    "wrist_energy_brake_diagnostics": list(
                        self._wrist_energy_brake_diagnostics
                    ),
                }
            )
        if getattr(self, "_gripper_runtime_static", None) is not None:
            report["gripper_runtime_static"] = validate_eef_gripper_static_contract(
                self._gripper_runtime_static
            )
            report["gripper_runtime_dynamic"] = self._gripper_runtime_dynamic_report()
        return report

    def episode_safety_report(self, episode_index: int) -> dict[str, object]:
        """Return the active episode report, rejecting lifecycle drift."""

        if self._active_episode_index != episode_index:
            raise ValueError(
                "EEF IK safety episode lifecycle mismatch: "
                f"active={self._active_episode_index!r}, requested={episode_index!r}"
            )
        return self.safety_report()


@configclass
class RobustDifferentialInverseKinematicsActionCfg(
    DifferentialInverseKinematicsActionCfg
):
    """Configuration for the robust differential-IK action term."""

    enable_failure_substep_trace: bool = False
    """Enable the separate failure-only device trace. Defaults to False."""

    enable_wrist_energy_brake: bool = False
    """Enable the isolated wrist transient diagnostic candidate."""

    enable_arm_slew_headroom: bool = False
    """Use 95% of the physical per-substep arm velocity bound."""

    enable_gripper_close_arm_interlock: bool = False
    """Hold the arm during the bounded EEF gripper-close ramp."""

    class_type = RobustDifferentialInverseKinematicsAction
