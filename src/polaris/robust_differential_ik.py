"""Numerically robust differential-IK action components.

Isaac Lab's damped-least-squares implementation uses a direct float32 matrix
inverse. That is normally appropriate, but a pathological Jacobian after a
dynamics excursion can make the configured damping round away and leave the
normal matrix singular. The classes in this module preserve Isaac Lab's normal
DLS path exactly and use a double-precision pseudo-inverse only after that
direct inverse raises a linear-algebra error.
"""

from __future__ import annotations

import omni.log
import torch

from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.envs.mdp.actions.task_space_actions import (
    DifferentialInverseKinematicsAction,
)
from isaaclab.utils import configclass


class RobustDifferentialIKController(DifferentialIKController):
    """Isaac Lab DLS controller with an exception-only pseudo-inverse fallback."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fallback_count = 0

    def _compute_delta_joint_pos(
        self, delta_pose: torch.Tensor, jacobian: torch.Tensor
    ) -> torch.Tensor:
        if self.cfg.ik_method == "dls" and (
            not torch.isfinite(jacobian).all() or not torch.isfinite(delta_pose).all()
        ):
            return self._compute_dls_pinv_fallback(
                delta_pose,
                jacobian,
                ValueError("non-finite DLS input"),
            )
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
        evaluated in float64 with ``pinv``.  Environments with non-finite inputs,
        or a second linear-algebra failure, hold their current joint targets by
        returning a zero delta.
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
            except torch.linalg.LinAlgError:
                # Holding the current target is safer than propagating an invalid
                # joint command. The throttled warning below records this event.
                pass

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
        self._ik_controller = RobustDifferentialIKController(
            cfg=self.cfg.controller,
            num_envs=self.num_envs,
            device=self.device,
        )


@configclass
class RobustDifferentialInverseKinematicsActionCfg(
    DifferentialInverseKinematicsActionCfg
):
    """Configuration for the robust differential-IK action term."""

    class_type = RobustDifferentialInverseKinematicsAction
