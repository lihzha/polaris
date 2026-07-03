"""EEF-only binary gripper driver target slew.

The public DROID action remains binary: ``0`` requests the exact open joint
target and ``pi/4`` requests the exact closed joint target.  This mixin changes
only how the EEF action term reaches that endpoint.  It anchors once from the
live post-reset driven-finger position and advances the position target by at
most the live configured driver velocity limit times one physics step.

The mixin deliberately has no Isaac Lab import.  Production composes it with
Isaac Lab's pinned ``BinaryJointPositionAction`` implementation, while host
tests compose it with a small behavioral stub of that same interface.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import torch

from polaris.eef_gripper_runtime import DRIVEN_GRIPPER_JOINT_INDEX
from polaris.eef_gripper_runtime import DRIVEN_GRIPPER_JOINT_NAME
from polaris.eef_gripper_runtime import EEF_GRIPPER_TARGET_SLEW_ACTION_CLASS
from polaris.eef_gripper_runtime import EEF_GRIPPER_TARGET_SLEW_PROFILE
from polaris.eef_gripper_runtime import EEF_GRIPPER_TARGET_SLEW_RESET_PROFILE
from polaris.eef_gripper_runtime import GRIPPER_CLOSED_TARGET_FLOAT32
from polaris.eef_gripper_runtime import GRIPPER_DRIVER_VELOCITY_LIMIT_FLOAT32
from polaris.eef_gripper_runtime import GRIPPER_OPEN_TARGET_FLOAT32
from polaris.eef_gripper_runtime import GRIPPER_TARGET_SLEW_FLOAT32_TOLERANCE_RAD
from polaris.eef_gripper_runtime import GRIPPER_TARGET_SLEW_MAX_ANCHOR_FLOAT32
from polaris.eef_gripper_runtime import GRIPPER_TARGET_SLEW_MIN_ANCHOR_FLOAT32
from polaris.eef_gripper_runtime import GRIPPER_TARGET_SLEW_PHYSICS_DT
from polaris.eef_gripper_runtime import GRIPPER_TARGET_SLEW_PHYSICS_HZ
from polaris.eef_gripper_runtime import PINNED_ACTUATOR_DEVICE
from polaris.eef_gripper_runtime import validate_eef_gripper_target_slew_dynamic
from polaris.eef_gripper_runtime import validate_eef_gripper_target_slew_static
from polaris.gripper_semantics import GRIPPER_THRESHOLD_PROFILE


class EefGripperTargetSlewError(RuntimeError):
    """Fail-closed EEF gripper target-slew contract violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EefGripperTargetSlewError(message)


def _same_float32_tensor(value: torch.Tensor, expected: float) -> bool:
    wanted = torch.full_like(value, float(expected))
    return bool(torch.equal(value, wanted))


class EefGripperTargetSlewMixin:
    """Mixin for the EEF-only closed-positive binary position action term."""

    gripper_target_slew_profile = EEF_GRIPPER_TARGET_SLEW_PROFILE
    gripper_target_slew_action_class = EEF_GRIPPER_TARGET_SLEW_ACTION_CLASS

    def __init__(self, cfg: Any, env: Any) -> None:
        super().__init__(cfg, env)
        _require(self.num_envs == 1, "EEF gripper target slew requires one environment")
        _require(
            list(self._joint_names) == [DRIVEN_GRIPPER_JOINT_NAME],
            "EEF gripper target slew driver joint-name drift",
        )
        _require(
            self._raw_actions.dtype == torch.float32
            and self._processed_actions.dtype == torch.float32
            and self._open_command.dtype == torch.float32
            and self._close_command.dtype == torch.float32,
            "EEF gripper target slew action dtype drift",
        )
        _require(
            self._raw_actions.device
            == self._processed_actions.device
            == self._open_command.device
            == self._close_command.device,
            "EEF gripper target slew action device drift",
        )
        _require(
            _same_float32_tensor(self._open_command, GRIPPER_OPEN_TARGET_FLOAT32)
            and _same_float32_tensor(
                self._close_command, GRIPPER_CLOSED_TARGET_FLOAT32
            ),
            "EEF gripper target slew binary endpoint drift",
        )
        physics_dt = getattr(env, "physics_dt", None)
        _require(
            isinstance(physics_dt, (int, float))
            and not isinstance(physics_dt, bool)
            and math.isfinite(float(physics_dt))
            and math.isclose(
                float(physics_dt),
                GRIPPER_TARGET_SLEW_PHYSICS_DT,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "EEF gripper target slew physics cadence drift",
        )
        self._gripper_target_slew_physics_dt = float(physics_dt)
        self._gripper_target_slew_contract: dict[str, Any] | None = None
        self._gripper_target_slew_max_step: torch.Tensor | None = None
        self._reset_gripper_target_slew_state()

    def _require_profile(self) -> None:
        _require(
            self.gripper_target_slew_profile == EEF_GRIPPER_TARGET_SLEW_PROFILE
            and self.gripper_target_slew_action_class
            == EEF_GRIPPER_TARGET_SLEW_ACTION_CLASS,
            "EEF gripper target slew profile drift",
        )
        _require(
            self.num_envs == 1
            and list(self._joint_names) == [DRIVEN_GRIPPER_JOINT_NAME]
            and list(self._joint_ids) == [DRIVEN_GRIPPER_JOINT_INDEX],
            "EEF gripper target slew ownership profile drift",
        )
        _require(
            self._raw_actions.shape == (1, 1)
            and self._processed_actions.shape == (1, 1)
            and self._open_command.shape == (1,)
            and self._close_command.shape == (1,)
            and self._raw_actions.dtype
            == self._processed_actions.dtype
            == self._open_command.dtype
            == self._close_command.dtype
            == torch.float32
            and self._raw_actions.device
            == self._processed_actions.device
            == self._open_command.device
            == self._close_command.device,
            "EEF gripper target slew action tensor profile drift",
        )
        _require(
            _same_float32_tensor(self._open_command, GRIPPER_OPEN_TARGET_FLOAT32)
            and _same_float32_tensor(
                self._close_command, GRIPPER_CLOSED_TARGET_FLOAT32
            ),
            "EEF gripper target slew binary endpoint profile drift",
        )
        _require(
            math.isfinite(self._gripper_target_slew_physics_dt)
            and math.isclose(
                self._gripper_target_slew_physics_dt,
                GRIPPER_TARGET_SLEW_PHYSICS_DT,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "EEF gripper target slew physics cadence profile drift",
        )

    def _live_gripper_driver_limit(self) -> torch.Tensor:
        """Return the exact live float32 ``velocity_limit_sim`` tensor."""

        self._require_profile()
        actuators = getattr(self._asset, "actuators", None)
        _require(isinstance(actuators, Mapping), "missing live actuator mapping")
        actuator = actuators.get("gripper")
        _require(actuator is not None, "missing live gripper actuator")
        cfg = getattr(actuator, "cfg", None)
        _require(cfg is not None, "missing live gripper actuator cfg")
        for field in ("velocity_limit", "velocity_limit_sim"):
            value = getattr(cfg, field, None)
            _require(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and float(value) == GRIPPER_DRIVER_VELOCITY_LIMIT_FLOAT32,
                f"live gripper configured {field} drift",
            )
        legacy = getattr(actuator, "velocity_limit", None)
        simulation = getattr(actuator, "velocity_limit_sim", None)
        for field, value in (
            ("velocity_limit", legacy),
            ("velocity_limit_sim", simulation),
        ):
            _require(
                isinstance(value, torch.Tensor)
                and value.shape == (1, 1)
                and value.dtype == torch.float32
                and str(value.device) == PINNED_ACTUATOR_DEVICE
                and bool(torch.isfinite(value).all().item())
                and _same_float32_tensor(value, GRIPPER_DRIVER_VELOCITY_LIMIT_FLOAT32),
                f"live gripper actuator {field} tensor drift",
            )
        _require(
            torch.equal(legacy, simulation),
            "live gripper legacy/simulation velocity-limit drift",
        )
        return simulation

    def gripper_target_slew_static_contract(self) -> dict[str, Any]:
        """Capture the action-owned static profile from the live driver limit."""

        limit = self._live_gripper_driver_limit()
        physics_dt = torch.full_like(limit, self._gripper_target_slew_physics_dt)
        max_step = torch.mul(limit, physics_dt)
        _require(
            bool(torch.isfinite(max_step).all().item())
            and bool((max_step > 0).all().item()),
            "EEF gripper target slew derived cap is invalid",
        )
        contract = {
            "profile": EEF_GRIPPER_TARGET_SLEW_PROFILE,
            "scope": "eef_pose_only_native_joint_position_unchanged_v1",
            "action_class": EEF_GRIPPER_TARGET_SLEW_ACTION_CLASS,
            "driver_joint_name": DRIVEN_GRIPPER_JOINT_NAME,
            "driver_joint_index": DRIVEN_GRIPPER_JOINT_INDEX,
            "endpoint_semantics_profile": GRIPPER_THRESHOLD_PROFILE,
            "open_target_rad": float(self._open_command[0].detach().cpu().item()),
            "closed_target_rad": float(self._close_command[0].detach().cpu().item()),
            "velocity_limit_source": (
                "live_implicit_actuator_velocity_limit_sim_float32_v1"
            ),
            "velocity_limit_rad_s": float(limit[0, 0].detach().cpu().item()),
            "physics_hz": GRIPPER_TARGET_SLEW_PHYSICS_HZ,
            "physics_dt": self._gripper_target_slew_physics_dt,
            "max_target_step_rad": float(max_step[0, 0].detach().cpu().item()),
            "float32_tolerance_rad": (GRIPPER_TARGET_SLEW_FLOAT32_TOLERANCE_RAD),
            "reset_profile": EEF_GRIPPER_TARGET_SLEW_RESET_PROFILE,
            "tensor_dtype": str(self._processed_actions.dtype),
            "tensor_device": str(self._processed_actions.device),
        }
        return validate_eef_gripper_target_slew_static(contract)

    def install_gripper_target_slew_contract(self, contract: Mapping[str, Any]) -> None:
        """Install once after the live EEF gripper drive is fully validated."""

        _require(
            self._gripper_target_slew_contract is None,
            "EEF gripper target slew contract is already installed",
        )
        validated = validate_eef_gripper_target_slew_static(dict(contract))
        live = self.gripper_target_slew_static_contract()
        _require(validated == live, "EEF gripper target slew install identity drift")
        limit = self._live_gripper_driver_limit()
        physics_dt = torch.full_like(limit, self._gripper_target_slew_physics_dt)
        self._gripper_target_slew_max_step = torch.mul(limit, physics_dt)
        _require(
            float(self._gripper_target_slew_max_step[0, 0].detach().cpu().item())
            == validated["max_target_step_rad"],
            "EEF gripper target slew installed cap drift",
        )
        self._gripper_target_slew_contract = validated

    def _reset_gripper_target_slew_state(self) -> None:
        device = self._processed_actions.device
        dtype = self._processed_actions.dtype
        self._gripper_target_slew_initialized = False
        self._gripper_target_slew_endpoint_seen = False
        self._gripper_target_slew_current = torch.zeros(
            (self.num_envs, 1), dtype=dtype, device=device
        )
        self._gripper_target_slew_endpoint = torch.zeros_like(
            self._gripper_target_slew_current
        )
        self._gripper_target_slew_last_processed_endpoint = torch.zeros_like(
            self._gripper_target_slew_current
        )
        self._gripper_target_slew_initial_anchor = torch.zeros_like(
            self._gripper_target_slew_current
        )
        self._gripper_target_slew_process_calls = 0
        self._gripper_target_slew_apply_calls = 0
        self._gripper_target_slew_initialization_count = 0
        self._gripper_target_slew_endpoint_change_count = 0
        self._gripper_target_slew_repeated_endpoint_process_count = 0
        self._gripper_target_slew_limited_apply_count = 0
        self._gripper_target_slew_endpoint_reached_apply_count = 0
        self._gripper_target_slew_live_limit_validation_count = 0
        self._gripper_target_slew_max_abs_target_step = torch.zeros(
            (), dtype=dtype, device=device
        )
        self._gripper_target_slew_max_abs_endpoint_error_before = torch.zeros(
            (), dtype=dtype, device=device
        )
        self._gripper_target_slew_max_abs_endpoint_error_after = torch.zeros(
            (), dtype=dtype, device=device
        )

    def reset(self, env_ids: Any = None) -> None:
        super().reset(env_ids=env_ids)
        if hasattr(self, "_gripper_target_slew_current"):
            self._reset_gripper_target_slew_state()

    def process_actions(self, actions: torch.Tensor) -> None:
        _require(
            self._gripper_target_slew_contract is not None,
            "EEF gripper target slew contract is not installed",
        )
        self._require_profile()
        _require(
            isinstance(actions, torch.Tensor)
            and actions.shape == self._raw_actions.shape
            and actions.dtype == torch.float32
            and actions.device == self._raw_actions.device
            and bool(torch.isfinite(actions).all().item()),
            "EEF gripper target slew input tensor drift",
        )
        super().process_actions(actions)
        endpoint = self._processed_actions
        _require(
            endpoint.shape == (self.num_envs, 1)
            and endpoint.dtype == torch.float32
            and endpoint.device == self._raw_actions.device
            and bool(torch.isfinite(endpoint).all().item())
            and bool(
                ((endpoint == self._open_command) | (endpoint == self._close_command))
                .all()
                .item()
            ),
            "EEF gripper target slew processed endpoint drift",
        )
        if self._gripper_target_slew_endpoint_seen:
            changed = bool(
                (endpoint != self._gripper_target_slew_last_processed_endpoint)
                .any()
                .item()
            )
            if changed:
                self._gripper_target_slew_endpoint_change_count += 1
            else:
                self._gripper_target_slew_repeated_endpoint_process_count += 1
        self._gripper_target_slew_endpoint.copy_(endpoint)
        self._gripper_target_slew_last_processed_endpoint.copy_(endpoint)
        self._gripper_target_slew_endpoint_seen = True
        self._gripper_target_slew_process_calls += 1

    def _require_live_action_tensor(self, value: Any, *, field: str) -> torch.Tensor:
        _require(
            isinstance(value, torch.Tensor)
            and value.shape == (self.num_envs, 1)
            and value.dtype == self._processed_actions.dtype
            and value.device == self._processed_actions.device
            and bool(torch.isfinite(value).all().item()),
            f"EEF gripper target slew live {field} tensor drift",
        )
        return value

    def apply_actions(self) -> None:
        _require(
            self._gripper_target_slew_contract is not None
            and self._gripper_target_slew_max_step is not None,
            "EEF gripper target slew contract is not installed",
        )
        _require(
            self._gripper_target_slew_endpoint_seen,
            "EEF gripper target slew apply has no binary endpoint",
        )
        self._require_profile()
        live_limit = self._live_gripper_driver_limit()
        live_max_step = torch.mul(
            live_limit,
            torch.full_like(live_limit, self._gripper_target_slew_physics_dt),
        )
        _require(
            torch.equal(live_max_step, self._gripper_target_slew_max_step),
            "EEF gripper target slew live cap drift",
        )

        joint_pos = self._require_live_action_tensor(
            self._asset.data.joint_pos[:, self._joint_ids], field="joint position"
        )
        joint_target = self._require_live_action_tensor(
            self._asset.data.joint_pos_target[:, self._joint_ids],
            field="joint position target",
        )
        initializing = not self._gripper_target_slew_initialized
        if initializing:
            minimum_anchor = torch.full_like(
                joint_pos, GRIPPER_TARGET_SLEW_MIN_ANCHOR_FLOAT32
            )
            maximum_anchor = torch.full_like(
                joint_pos, GRIPPER_TARGET_SLEW_MAX_ANCHOR_FLOAT32
            )
            _require(
                bool(
                    ((joint_pos >= minimum_anchor) & (joint_pos <= maximum_anchor))
                    .all()
                    .item()
                ),
                "EEF gripper target slew initial live anchor outside profile bounds",
            )
            previous_target = joint_pos.clone()
        else:
            _require(
                torch.equal(joint_target, self._gripper_target_slew_current),
                "EEF gripper target slew external target drift",
            )
            previous_target = self._gripper_target_slew_current.clone()

        endpoint = self._gripper_target_slew_endpoint
        delta = endpoint - previous_target
        abs_before = delta.abs()
        limited = abs_before > self._gripper_target_slew_max_step
        step = torch.clamp(
            delta,
            min=-self._gripper_target_slew_max_step,
            max=self._gripper_target_slew_max_step,
        )
        candidate = previous_target + step
        next_target = torch.where(limited, candidate, endpoint)
        applied_step = next_target - previous_target
        # Float32 addition followed by subtraction can make the represented
        # target step one ULP larger than the represented cap. Move the target
        # one representable value back toward the previous target whenever
        # that occurs; the physical write therefore never exceeds the exact
        # live float32 cap even though durable validation retains its named
        # 1e-6 serialization/readback tolerance.
        represented_overshoot = applied_step.abs() > self._gripper_target_slew_max_step
        next_target = torch.where(
            represented_overshoot,
            torch.nextafter(next_target, previous_target),
            next_target,
        )
        applied_step = next_target - previous_target
        abs_after = (endpoint - next_target).abs()
        lower = torch.minimum(self._open_command, self._close_command)
        upper = torch.maximum(self._open_command, self._close_command)
        _require(
            bool(torch.isfinite(next_target).all().item())
            and bool(
                (applied_step.abs() <= self._gripper_target_slew_max_step).all().item()
            )
            and bool(((next_target >= lower) & (next_target <= upper)).all().item())
            and bool((abs_after <= abs_before).all().item()),
            "EEF gripper target slew transition invariant drift",
        )

        self._asset.set_joint_position_target(next_target, joint_ids=self._joint_ids)
        written = self._require_live_action_tensor(
            self._asset.data.joint_pos_target[:, self._joint_ids],
            field="written joint position target",
        )
        _require(
            torch.equal(written, next_target),
            "EEF gripper target slew setter/readback drift",
        )
        if initializing:
            self._gripper_target_slew_initial_anchor.copy_(previous_target)
            self._gripper_target_slew_initialized = True
            self._gripper_target_slew_initialization_count += 1
        self._gripper_target_slew_current.copy_(next_target)
        self._gripper_target_slew_live_limit_validation_count += 1
        self._gripper_target_slew_apply_calls += 1
        if bool(limited.any().item()):
            self._gripper_target_slew_limited_apply_count += 1
        else:
            self._gripper_target_slew_endpoint_reached_apply_count += 1
        self._gripper_target_slew_max_abs_target_step = torch.maximum(
            self._gripper_target_slew_max_abs_target_step,
            applied_step.abs().amax(),
        )
        self._gripper_target_slew_max_abs_endpoint_error_before = torch.maximum(
            self._gripper_target_slew_max_abs_endpoint_error_before,
            abs_before.amax(),
        )
        self._gripper_target_slew_max_abs_endpoint_error_after = torch.maximum(
            self._gripper_target_slew_max_abs_endpoint_error_after,
            abs_after.amax(),
        )

    def gripper_target_slew_dynamic_report(self) -> dict[str, Any]:
        """Return closed per-reset counters and maxima for durable evidence."""

        _require(
            self._gripper_target_slew_contract is not None,
            "EEF gripper target slew contract is not installed",
        )
        self._require_profile()
        self._live_gripper_driver_limit()
        if self._gripper_target_slew_initialized:
            written = self._require_live_action_tensor(
                self._asset.data.joint_pos_target[:, self._joint_ids],
                field="report joint position target",
            )
            _require(
                torch.equal(written, self._gripper_target_slew_current),
                "EEF gripper target slew report target drift",
            )

        def scalar(value: torch.Tensor) -> float:
            return float(value.detach().cpu().item())

        report = {
            "profile": EEF_GRIPPER_TARGET_SLEW_PROFILE,
            "process_action_calls": self._gripper_target_slew_process_calls,
            "apply_calls": self._gripper_target_slew_apply_calls,
            "initialization_count": self._gripper_target_slew_initialization_count,
            "endpoint_change_count": (self._gripper_target_slew_endpoint_change_count),
            "repeated_endpoint_process_count": (
                self._gripper_target_slew_repeated_endpoint_process_count
            ),
            "slew_limited_apply_count": (self._gripper_target_slew_limited_apply_count),
            "endpoint_reached_apply_count": (
                self._gripper_target_slew_endpoint_reached_apply_count
            ),
            "live_limit_validation_count": (
                self._gripper_target_slew_live_limit_validation_count
            ),
            "max_abs_target_step_rad": scalar(
                self._gripper_target_slew_max_abs_target_step
            ),
            "max_abs_endpoint_error_before_step_rad": scalar(
                self._gripper_target_slew_max_abs_endpoint_error_before
            ),
            "max_abs_endpoint_error_after_step_rad": scalar(
                self._gripper_target_slew_max_abs_endpoint_error_after
            ),
            "initial_anchor_rad": (
                scalar(self._gripper_target_slew_initial_anchor[0, 0])
                if self._gripper_target_slew_initialized
                else None
            ),
            "last_requested_endpoint_rad": (
                scalar(self._gripper_target_slew_endpoint[0, 0])
                if self._gripper_target_slew_endpoint_seen
                else None
            ),
            "last_applied_target_rad": (
                scalar(self._gripper_target_slew_current[0, 0])
                if self._gripper_target_slew_initialized
                else None
            ),
        }
        return validate_eef_gripper_target_slew_dynamic(report)


__all__ = ["EefGripperTargetSlewError", "EefGripperTargetSlewMixin"]
