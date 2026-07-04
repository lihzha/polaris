#!/usr/bin/env python3
"""Replay the exact reasoning trace through the production-v4 controller core.

The immutable 294-action fixture is followed by 64 physics substeps that repeat
source action 293, producing exactly 2,416 causal 13-DOF trace entries.  The
production release-ramp implementation is the sole arm-target writer.  A
runtime subclass only observes the committed transaction and proves, bit for
bit, that its target equals an independent helper recomputation, the pending
failure trace, and the live articulation setter readback.

This is a controller replay gate, not a task-success evaluation.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import traceback
from typing import Any, Mapping, Sequence
import zlib

import build_reasoning_fulltrace_replay_fixture as fixture_contract
import smoke_eef_pose_canary_trace_replay as gate0


PROFILE = "reasoning_43075_production_v4_core_release_ramp_replay_v1"
MODULE_IDENTITY = "smoke_eef_pose_reasoning_production_v4_core_replay"
VARIANT = "production_v4_core_ramp16"
SOURCE_TRACE_POLARIS_COMMIT = "0611d384f5f26ef9bd8ff114be273e875c3fe719"
PRODUCTION_BASE_COMMIT = "7fc74d648328432a7f9f06d13c0e82a03f73a0c1"
REPLAY_IMPLEMENTATION_COMMIT = "2ebfe7db5b2a31887481781b214608976e8023db"
REPLAY_PARENT_COMMIT = "e18b8ebbc26fd309d8e45bd58bef9c867948098a"
CONTROLLER_PROFILE = (
    "arm_slew_0p95_gripper_rate0p25_fixed_anchor86_release_ramp16_"
    "mimic100_damping1p2_v4"
)
FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "reasoning_43075_job1098523_fulltrace_actions.json"
)
FIXTURE_SIZE_BYTES = 14478
FIXTURE_SHA256 = "daf2aa682f2296a93170f842a5adb13a4fbc6b2694fa5dca28de7ac7ad83d7cb"
ACTION_ENCODING = {
    "codec": "zlib-9-base64",
    "dtype": "little-endian-float32",
    "action_count": 294,
    "action_width": 8,
    "uncompressed_size_bytes": 9408,
    "uncompressed_sha256": "0e781cd1df2d00f3496c1feb2bf079e9194ad664710ac988cc9f7e8bcde11bce",
    "compressed_size_bytes": 7930,
    "compressed_sha256": "33a2ccb654897a13935358078e4b7366c83152c85741b79e3cebd30e3d489091",
}
ACTION_COUNT = 294
DECIMATION = 8
TAIL_PHYSICS_SUBSTEPS = 64
TAIL_POLICY_STEPS = TAIL_PHYSICS_SUBSTEPS // DECIMATION
TAIL_SOURCE_ACTION_INDEX = ACTION_COUNT - 1
TOTAL_APPLY_COUNT = ACTION_COUNT * DECIMATION + TAIL_PHYSICS_SUBSTEPS
TAIL_ACTION_FLOAT32_SHA256 = (
    "b938c1ae7f29d0d762b48502af53789a7117e364514f3bdf9887ff5e3e36ab50"
)
VIDEO_FPS = 15
VIDEO_HEIGHT = 224
VIDEO_WIDTH = 448
JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "finger_joint",
    "right_outer_knuckle_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
    "left_inner_finger_knuckle_joint",
    "right_inner_finger_knuckle_joint",
]
FOLLOWER_INDICES = [8, 9, 10, 11, 12]
FOLLOWER_GEARS = [-1.0, 1.0, -1.0, 1.0, 1.0]
PRODUCTION_FOLLOWER_LIMIT = 5.0
RELEASE_RAMP_SUBSTEPS = 16
NOMINAL_ARM_SLEW_RATIO = 0.95
RELEASE_RAMP_FRACTIONS = tuple(
    index / (RELEASE_RAMP_SUBSTEPS - 1) for index in range(RELEASE_RAMP_SUBSTEPS)
)
EXPECTED_RAMP_WINDOWS = ((1600, 1615), (2176, 2191), (2334, 2349))
EXPECTED_RAMP_APPLY_INDICES = [
    apply_index
    for first, last in EXPECTED_RAMP_WINDOWS
    for apply_index in range(first, last + 1)
]
EXPECTED_RAMP_INDICES = list(range(RELEASE_RAMP_SUBSTEPS)) * 3
EXPECTED_LIMITED_APPLIES_PER_RAMP = [15, 8, 15]
EXPECTED_LIMITED_JOINTS_PER_RAMP = [81, 35, 105]
EXPECTED_CORE_RAMP_COUNTS = {
    "release_observed_count": 3,
    "ramp_started_count": 3,
    "ramp_completed_count": 3,
    "ramp_cancelled_by_reactivation_count": 0,
    "ramp_target_apply_count": 48,
    "cancelled_ramp_target_apply_count": 0,
    "ramp_limited_target_apply_count": 38,
    "ramp_limited_joint_target_count": 221,
}
ARM_JOINT_IDS = list(range(7))
ARM_JOINT_NAMES = JOINT_NAMES[:7]
ARM_VELOCITY_LIMITS_RAD_S = [
    2.174999952316284,
    2.174999952316284,
    2.174999952316284,
    2.174999952316284,
    2.609999895095825,
    2.609999895095825,
    2.609999895095825,
]
ARM_PHYSICAL_MAX_DELTA_RAD = [
    0.018125001341104507,
    0.018125001341104507,
    0.018125001341104507,
    0.018125001341104507,
    0.02174999937415123,
    0.02174999937415123,
    0.02174999937415123,
]
ARM_NOMINAL_MAX_DELTA_RAD = [
    0.017218751832842827,
    0.017218751832842827,
    0.017218751832842827,
    0.017218751832842827,
    0.020662499591708183,
    0.020662499591708183,
    0.020662499591708183,
]
ARM_SOFT_LIMITS_RAD = [
    [-2.8973000049591064, 2.8973000049591064],
    [-1.7627999782562256, 1.7627999782562256],
    [-2.8973000049591064, 2.8973000049591064],
    [-3.0717999935150146, -0.06979990005493164],
    [-2.8973000049591064, 2.8973000049591064],
    [-0.017499923706054688, 3.752500057220459],
    [-2.8973000049591064, 2.8973000049591064],
]
ARM_POSITION_TOLERANCE_RAD = 1e-5
ARM_VELOCITY_TOLERANCE_RAD_S = 1e-5
ARM_TARGET_SLEW_TOLERANCE_RAD = 1e-6
CORE_RAMP_OBSERVATION_ENTRY_FIELDS = {
    "profile",
    "apply_index",
    "policy_step",
    "physics_substep",
    "arm_joint_ids",
    "arm_joint_names",
    "ramp_index",
    "fraction_float32",
    "nominal_max_delta_joint_pos_rad",
    "current_joint_pos_rad",
    "raw_dls_joint_pos_target_rad",
    "nominal_safe_target_rad",
    "production_core_target_rad",
    "failure_trace_target_rad",
    "live_setter_readback_target_rad",
    "limited_joint_mask",
    "limited_joint_count",
    "max_abs_nominal_to_core_target_change_rad",
    "target_little_endian_float32_sha256",
    "bitwise_contract",
    "endpoint_contract",
    "observer_target_setter_call_count",
    "observer_failure_trace_write_count",
    "observer_release_ramp_state_write_count",
    "observer_gripper_target_or_state_write_count",
}
TRACE_ENTRY_FIELDS = {
    "apply_index",
    "policy_step",
    "physics_substep",
    "raw_action",
    "requested_endpoint_rad",
    "target_after_setter_rad",
    "pre",
    "command_after_setters",
    "post",
}
GRIPPER_SNAPSHOT_FIELDS = {
    "joint_pos_rad",
    "joint_vel_rad_s",
    "joint_acc_rad_s2",
    "joint_pos_target_rad",
    "joint_vel_target_rad_s",
    "joint_effort_target_nm",
}
ALL_JOINT_SNAPSHOT_FIELDS = {
    "all_joint_pos_rad",
    "all_joint_vel_rad_s",
    "all_joint_acc_rad_s2",
    "all_joint_pos_target_rad",
    "all_joint_vel_target_rad_s",
    "all_joint_effort_target_nm",
    "all_computed_torque_nm",
    "all_applied_torque_nm",
}
INTERLOCK_SNAPSHOT_FIELDS = {
    "arm_apply_call_count",
    "configured_substeps",
    "remaining_substeps",
    "anchor_valid",
    "activation_count",
    "active_apply_count",
    "released_apply_count",
    "anchor_joint_pos_rad",
    "release_ramp",
}
RELEASE_RAMP_SNAPSHOT_FIELDS = {
    "enabled",
    "phase",
    "next_index",
    "release_observed_count",
    "started_count",
    "completed_count",
    "cancelled_by_reactivation_count",
    "target_apply_count",
    "cancelled_target_apply_count",
    "limited_target_apply_count",
    "limited_joint_target_count",
    "last_target_apply_index",
    "last_index",
}


class ProductionV4ReplayError(ValueError):
    """A replay input, production identity, or expected outcome drifted."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProductionV4ReplayError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def float32(value: float) -> float:
    """Round one finite number through IEEE little-endian float32."""

    result = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    require(math.isfinite(result), "finite float32 value")
    return result


def float32_equal(left: float, right: float) -> bool:
    return struct.pack("<f", float(left)) == struct.pack("<f", float(right))


def float32_add(left: float, right: float) -> float:
    return float32(float32(left) + float32(right))


def float32_subtract(left: float, right: float) -> float:
    return float32(float32(left) - float32(right))


def float32_multiply(left: float, right: float) -> float:
    return float32(float32(left) * float32(right))


def require_float32_vector_equal(
    actual: Sequence[float], expected: Sequence[float], *, field: str
) -> None:
    require(
        len(actual) == len(expected)
        and all(
            float32_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        ),
        f"{field} float32 identity",
    )


def file_identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    require(path.is_file() and not path.is_symlink(), f"missing/linked file {path}")
    data = path.read_bytes()
    metadata = path.stat()
    return {
        "path": str(path),
        "size_bytes": len(data),
        "sha256": sha256(data),
        "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "nlink": metadata.st_nlink,
    }


def load_actions(path: Path = FIXTURE_PATH) -> tuple[dict[str, Any], list[list[float]]]:
    identity = file_identity(path)
    require(
        identity["size_bytes"] == FIXTURE_SIZE_BYTES
        and identity["sha256"] == FIXTURE_SHA256,
        "full-trace fixture file identity drift",
    )
    payload = gate0.strict_json_loads(path.read_bytes(), field="full-trace fixture")
    require(
        payload.get("schema_version") == 1
        and payload.get("fixture_profile") == fixture_contract.FIXTURE_PROFILE
        and payload.get("polaris_commit") == SOURCE_TRACE_POLARIS_COMMIT,
        "full-trace fixture profile/commit",
    )
    source = payload.get("source")
    require(
        isinstance(source, dict)
        and source.get("job_id") == "1098523"
        and source.get("trace_sha256") == fixture_contract.TRACE_SHA256
        and source.get("event_counts") == fixture_contract.EXPECTED_EVENTS
        and source.get("query_contract") == fixture_contract.QUERY_CONTRACT
        and source.get("expected_failure") == fixture_contract.EXPECTED_FAILURE,
        "full-trace source contract",
    )
    require(payload.get("action_encoding") == ACTION_ENCODING, "action encoding")
    plan = payload.get("action_plan")
    require(
        isinstance(plan, dict)
        and plan.get("profile")
        == "all_recorded_absolute_polaris_actions_exact_float32_v1"
        and plan.get("action_count") == ACTION_COUNT
        and plan.get("action_width") == 8
        and plan.get("gripper_endpoint_changes")
        == fixture_contract.EXPECTED_GRIPPER_CHANGES,
        "action plan contract",
    )
    chunks = payload.get("actions_zlib_base64_chunks")
    require(
        isinstance(chunks, list)
        and chunks
        and all(isinstance(chunk, str) and 0 < len(chunk) <= 120 for chunk in chunks),
        "action base64 chunks",
    )
    try:
        compressed = base64.b64decode("".join(chunks), validate=True)
        raw = zlib.decompress(compressed)
    except (ValueError, zlib.error) as error:
        raise ProductionV4ReplayError("action codec failure") from error
    require(
        len(compressed) == ACTION_ENCODING["compressed_size_bytes"]
        and sha256(compressed) == ACTION_ENCODING["compressed_sha256"]
        and len(raw) == ACTION_ENCODING["uncompressed_size_bytes"]
        and sha256(raw) == ACTION_ENCODING["uncompressed_sha256"],
        "action byte identity",
    )
    actions = [list(values) for values in struct.iter_unpack("<8f", raw)]
    require(len(actions) == ACTION_COUNT, "decoded action count")
    for step, action in enumerate(actions):
        require(
            len(action) == 8 and all(math.isfinite(value) for value in action),
            f"action {step} finite width",
        )
        require(
            abs(math.sqrt(sum(value * value for value in action[3:7])) - 1.0) <= 1e-3,
            f"action {step} quaternion norm",
        )
        require(action[7] in (0.0, 1.0), f"action {step} binary gripper")
    return identity, actions


def effective_actions(
    actions: Sequence[Sequence[float]],
) -> list[list[float]]:
    result = [list(action) for action in actions]
    require(len(result) == ACTION_COUNT, "effective action count")
    require(result == [list(action) for action in actions], "source action mutation")
    return result


def frozen_tail_contract(actions: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Bind the post-source tail to byte-exact repeats of source action 293."""

    require(len(actions) == ACTION_COUNT, "tail source action count")
    final_action = [float(value) for value in actions[TAIL_SOURCE_ACTION_INDEX]]
    require(
        len(final_action) == ACTION_ENCODING["action_width"]
        and all(math.isfinite(value) for value in final_action),
        "tail final action finite width",
    )
    encoded = struct.pack("<8f", *final_action)
    require(
        sha256(encoded) == TAIL_ACTION_FLOAT32_SHA256,
        "frozen tail action byte identity",
    )
    require(
        TAIL_PHYSICS_SUBSTEPS > 0
        and TAIL_PHYSICS_SUBSTEPS % DECIMATION == 0
        and TAIL_POLICY_STEPS == 8,
        "frozen tail cadence constants",
    )
    return {
        "profile": "repeat_final_source_action_for_fixed_physics_tail_v1",
        "source_action_index": TAIL_SOURCE_ACTION_INDEX,
        "action_width": ACTION_ENCODING["action_width"],
        "action_float32_sha256": TAIL_ACTION_FLOAT32_SHA256,
        "action_values_float32": final_action,
        "policy_steps": TAIL_POLICY_STEPS,
        "physics_substeps_per_policy_step": DECIMATION,
        "physics_substeps": TAIL_PHYSICS_SUBSTEPS,
        "command_semantics": "exact_unmodified_final_source_action_repeat_v1",
    }


def vector(tensor: Any, *, length: int, field: str) -> list[float]:
    values = [float(value) for value in tensor.detach().cpu().flatten().tolist()]
    require(
        len(values) == length and all(math.isfinite(value) for value in values),
        f"{field} finite vector",
    )
    return values


def little_endian_float32_sha256(values: Sequence[float], *, field: str) -> str:
    checked = _finite_seven_vector(values, field=field)
    return sha256(struct.pack("<7f", *checked))


def interlock_snapshot(arm_term: Any) -> dict[str, Any]:
    """Read only production interlock/ramp state after a committed arm apply."""

    anchor = getattr(arm_term, "_gripper_close_arm_interlock_anchor", None)
    return {
        "arm_apply_call_count": int(getattr(arm_term, "_apply_call_count")),
        "configured_substeps": int(
            getattr(arm_term, "_gripper_close_arm_interlock_configured_substeps")
        ),
        "remaining_substeps": int(
            getattr(arm_term, "_gripper_close_arm_interlock_remaining")
        ),
        "anchor_valid": bool(
            getattr(arm_term, "_gripper_close_arm_interlock_anchor_valid")
        ),
        "activation_count": int(
            getattr(arm_term, "_gripper_close_arm_interlock_activation_count")
        ),
        "active_apply_count": int(
            getattr(arm_term, "_gripper_close_arm_interlock_active_apply_count")
        ),
        "released_apply_count": int(
            getattr(arm_term, "_gripper_close_arm_interlock_released_apply_count")
        ),
        "anchor_joint_pos_rad": vector(anchor, length=7, field="interlock anchor"),
        "release_ramp": {
            "enabled": bool(getattr(arm_term, "_arm_release_ramp_enabled")),
            "phase": str(getattr(arm_term, "_arm_release_ramp_phase")),
            "next_index": getattr(arm_term, "_arm_release_ramp_next_index"),
            "release_observed_count": int(
                getattr(arm_term, "_arm_release_observed_count")
            ),
            "started_count": int(getattr(arm_term, "_arm_release_ramp_started_count")),
            "completed_count": int(
                getattr(arm_term, "_arm_release_ramp_completed_count")
            ),
            "cancelled_by_reactivation_count": int(
                getattr(arm_term, "_arm_release_ramp_cancelled_by_reactivation_count")
            ),
            "target_apply_count": int(
                getattr(arm_term, "_arm_release_ramp_target_apply_count")
            ),
            "cancelled_target_apply_count": int(
                getattr(arm_term, "_arm_release_ramp_cancelled_target_apply_count")
            ),
            "limited_target_apply_count": int(
                getattr(arm_term, "_arm_release_ramp_limited_target_apply_count")
            ),
            "limited_joint_target_count": int(
                getattr(arm_term, "_arm_release_ramp_limited_joint_target_count")
            ),
            "last_target_apply_index": getattr(
                arm_term, "_arm_release_ramp_last_target_apply_index"
            ),
            "last_index": getattr(arm_term, "_arm_release_ramp_last_index"),
        },
    }


def make_full_trace_arm_class(
    base_class: type,
    *,
    torch_module: Any,
    bound_helper: Any,
    release_helper: Any,
) -> type:
    """Add read-only transaction observation around the production arm class."""

    require(isinstance(base_class, type), "production arm observer base class")
    require(callable(bound_helper), "production nominal-target helper")
    require(callable(release_helper), "production release-ramp helper")

    class ProductionCoreObservingArmAction(base_class):
        def __init__(self, cfg: Any, env: Any) -> None:
            self._full_trace_env = env
            self._production_core_observer_entries: list[dict[str, Any]] = []
            self._production_core_observer_pending_target: dict[str, Any] | None = None
            super().__init__(cfg, env)

        def reset(self, env_ids: Any = None) -> None:
            super().reset(env_ids)
            self._production_core_observer_entries = []
            self._production_core_observer_pending_target = None

        def _arm_joint_ids(self) -> list[int]:
            joint_ids = self._joint_ids
            if hasattr(joint_ids, "detach"):
                joint_ids = joint_ids.detach().cpu().flatten().tolist()
            result = [int(index) for index in joint_ids]
            require(result == ARM_JOINT_IDS, "production observer arm joint IDs")
            return result

        def _set_targets_and_commit_gripper_close_arm_interlock(
            self,
            safe_target: Any,
            staged: Any,
            staged_release_ramp: Any,
            failure_trace: Any,
        ) -> None:
            """Capture the exact core argument, then delegate one transaction."""

            require(
                self._production_core_observer_pending_target is None,
                "production core observer target overlap",
            )
            self._production_core_observer_pending_target = {
                "apply_index": int(getattr(self, "_apply_call_count")) - 1,
                "target": safe_target.detach().clone(),
            }
            try:
                super()._set_targets_and_commit_gripper_close_arm_interlock(
                    safe_target,
                    staged,
                    staged_release_ramp,
                    failure_trace,
                )
            except BaseException:
                self._production_core_observer_pending_target = None
                raise

        def production_core_ramp_observation_report(self) -> dict[str, Any]:
            require(
                self._production_core_observer_pending_target is None,
                "production core observer retained a pending target",
            )
            return {
                "profile": "production_v4_core_release_ramp_bitwise_observer_v1",
                "production_base_commit": PRODUCTION_BASE_COMMIT,
                "controller_profile": CONTROLLER_PROFILE,
                "entry_count": len(self._production_core_observer_entries),
                "entries": list(self._production_core_observer_entries),
                "observer_write_contract": {
                    "target_setter_call_count": 0,
                    "failure_trace_write_count": 0,
                    "release_ramp_state_write_count": 0,
                    "gripper_target_or_state_write_count": 0,
                },
            }

        def apply_actions(self) -> None:
            terms = getattr(self._full_trace_env.action_manager, "_terms", {})
            finger_term = terms.get("finger_joint")
            finalizer = getattr(
                finger_term, "finalize_physics_post_before_next_arm", None
            )
            if callable(finalizer):
                finalizer()

            count_before = int(getattr(self, "_arm_release_ramp_target_apply_count"))
            super().apply_actions()
            count_after = int(getattr(self, "_arm_release_ramp_target_apply_count"))
            require(
                count_after - count_before in (0, 1),
                "production release-ramp target counter cadence",
            )
            captured = self._production_core_observer_pending_target
            require(
                isinstance(captured, dict)
                and captured.get("apply_index")
                == int(getattr(self, "_apply_call_count")) - 1,
                "production core target argument capture",
            )
            self._production_core_observer_pending_target = None
            if count_after == count_before:
                return

            apply_index = int(captured["apply_index"])
            ramp_index = getattr(self, "_arm_release_ramp_last_index")
            require(
                getattr(self, "_arm_release_ramp_last_target_apply_index")
                == apply_index
                and type(ramp_index) is int
                and 0 <= ramp_index < RELEASE_RAMP_SUBSTEPS,
                "production release-ramp last-target identity",
            )
            slot = getattr(self, "_failure_substep_trace_pending_slot")
            pending_apply = getattr(self, "_failure_substep_trace_pending_apply_index")
            buffers = getattr(self, "_failure_substep_trace_buffers")
            require(
                type(slot) is int
                and pending_apply == apply_index
                and isinstance(buffers, dict),
                "production pending failure-trace identity",
            )

            current = buffers["joint_pos_rad"][slot].detach().clone()
            raw_dls = buffers["raw_dls_joint_pos_target_rad"][slot].detach().clone()
            failure_target = buffers["new_joint_pos_target_rad"][slot].detach().clone()
            core_target = captured["target"]
            live_target = (
                self._asset.data.joint_pos_target[:, self._joint_ids].detach().clone()
            )
            nominal_target, _, _, _ = bound_helper(
                current,
                raw_dls,
                self._nominal_max_delta_joint_pos,
                self._soft_joint_position_limits,
                target_guard_band_delta_joint_pos=self._max_delta_joint_pos,
            )
            independent = release_helper(
                current,
                nominal_target,
                self._nominal_max_delta_joint_pos,
                ramp_index=ramp_index,
            )
            require(
                torch_module.equal(core_target, independent.target)
                and torch_module.equal(core_target, failure_target)
                and torch_module.equal(core_target, live_target),
                "production core/helper/failure-trace/live target bit identity",
            )
            endpoint_current_exact = ramp_index != 0 or torch_module.equal(
                core_target, current
            )
            endpoint_nominal_exact = (
                ramp_index != RELEASE_RAMP_SUBSTEPS - 1
                or torch_module.equal(core_target, nominal_target)
            )
            require(
                endpoint_current_exact and endpoint_nominal_exact,
                "production release-ramp endpoint identity",
            )

            current_values = vector(
                current, length=7, field="production ramp current position"
            )
            raw_values = vector(
                raw_dls, length=7, field="production ramp raw DLS target"
            )
            nominal_values = vector(
                nominal_target, length=7, field="production ramp nominal target"
            )
            core_values = vector(
                core_target, length=7, field="production ramp core target"
            )
            failure_values = vector(
                failure_target, length=7, field="production ramp failure target"
            )
            live_values = vector(
                live_target, length=7, field="production ramp live target"
            )
            independent_values = vector(
                independent.target,
                length=7,
                field="production ramp independent target",
            )
            target_hashes = {
                "production_core": little_endian_float32_sha256(
                    core_values, field="production core target hash"
                ),
                "independent_helper": little_endian_float32_sha256(
                    independent_values, field="independent target hash"
                ),
                "failure_trace": little_endian_float32_sha256(
                    failure_values, field="failure target hash"
                ),
                "live_setter_readback": little_endian_float32_sha256(
                    live_values, field="live target hash"
                ),
            }
            require(
                len(set(target_hashes.values())) == 1,
                "production ramp target byte-hash identity",
            )
            limited_mask = [
                bool(item)
                for item in independent.limited_joint_mask[0].detach().cpu().tolist()
            ]
            limited_joint_count = sum(limited_mask)
            maximum_change = float(
                (nominal_target - core_target).abs().amax().detach().cpu().item()
            )
            entry = {
                "profile": "production_core_release_ramp_target_observation_v1",
                "apply_index": apply_index,
                "policy_step": apply_index // DECIMATION,
                "physics_substep": apply_index % DECIMATION,
                "arm_joint_ids": self._arm_joint_ids(),
                "arm_joint_names": list(ARM_JOINT_NAMES),
                "ramp_index": ramp_index,
                "fraction_float32": float(independent.fraction),
                "nominal_max_delta_joint_pos_rad": vector(
                    self._nominal_max_delta_joint_pos,
                    length=7,
                    field="production nominal max delta",
                ),
                "current_joint_pos_rad": current_values,
                "raw_dls_joint_pos_target_rad": raw_values,
                "nominal_safe_target_rad": nominal_values,
                "production_core_target_rad": core_values,
                "failure_trace_target_rad": failure_values,
                "live_setter_readback_target_rad": live_values,
                "limited_joint_mask": limited_mask,
                "limited_joint_count": limited_joint_count,
                "max_abs_nominal_to_core_target_change_rad": maximum_change,
                "target_little_endian_float32_sha256": target_hashes,
                "bitwise_contract": {
                    "core_equals_independent_helper": True,
                    "core_equals_failure_trace": True,
                    "core_equals_live_setter_readback": True,
                },
                "endpoint_contract": {
                    "index0_equals_current": endpoint_current_exact,
                    "index15_equals_nominal": endpoint_nominal_exact,
                },
                "observer_target_setter_call_count": 0,
                "observer_failure_trace_write_count": 0,
                "observer_release_ramp_state_write_count": 0,
                "observer_gripper_target_or_state_write_count": 0,
            }
            require(
                len(self._production_core_observer_entries) == count_before,
                "production observer/core target count binding",
            )
            self._production_core_observer_entries.append(entry)

    ProductionCoreObservingArmAction.__name__ = base_class.__name__
    ProductionCoreObservingArmAction.__qualname__ = base_class.__qualname__
    ProductionCoreObservingArmAction.__module__ = MODULE_IDENTITY
    return ProductionCoreObservingArmAction


def make_full_trace_gripper_class(base_class: type) -> type:
    """Extend the configured v4 tracer with causal, read-only 13-DOF state."""

    production_finalizer = getattr(base_class, "_eef_trace_finalize_pending", None)
    require(
        callable(production_finalizer),
        "configured v4 gripper class lacks the production trace finalizer",
    )
    traced_base = gate0._make_tracing_gripper_class(base_class)  # noqa: SLF001

    class FullTraceEefBinaryJointPositionTargetSlewAction(traced_base):
        def __init__(self, cfg: Any, env: Any) -> None:
            self._full_trace_env = env
            super().__init__(cfg, env)

        def _gate0_snapshot(self) -> dict[str, Any]:
            snapshot = super()._gate0_snapshot()
            data = self._asset.data
            require(list(data.joint_names) == JOINT_NAMES, "live joint ordering drift")
            arm_term = self._full_trace_env.action_manager._terms["arm"]
            snapshot.update(
                {
                    "all_joint_pos_rad": vector(
                        data.joint_pos[0], length=13, field="joint position"
                    ),
                    "all_joint_vel_rad_s": vector(
                        data.joint_vel[0], length=13, field="joint velocity"
                    ),
                    "all_joint_acc_rad_s2": vector(
                        data.joint_acc[0], length=13, field="joint acceleration"
                    ),
                    "all_joint_pos_target_rad": vector(
                        data.joint_pos_target[0], length=13, field="position target"
                    ),
                    "all_joint_vel_target_rad_s": vector(
                        data.joint_vel_target[0], length=13, field="velocity target"
                    ),
                    "all_joint_effort_target_nm": vector(
                        data.joint_effort_target[0], length=13, field="effort target"
                    ),
                    "all_computed_torque_nm": vector(
                        data.computed_torque[0], length=13, field="computed torque"
                    ),
                    "all_applied_torque_nm": vector(
                        data.applied_torque[0], length=13, field="applied torque"
                    ),
                    "interlock": interlock_snapshot(arm_term),
                }
            )
            return snapshot

        def _gate0_finalize_pending(self) -> None:
            # The inherited tracer finalizes at the beginning of the next
            # finger apply, after the next arm command has already been
            # authored. The arm wrapper invokes the explicit original method
            # below before it changes arm targets or interlock state.
            return None

        def finalize_physics_post_before_next_arm(self) -> None:
            traced_base._gate0_finalize_pending(self)  # noqa: SLF001
            production_finalizer(self)

        def apply_actions(self) -> None:
            super().apply_actions()
            require(self._gate0_pending_entry is not None, "missing pending command")
            self._gate0_pending_entry["command_after_setters"] = self._gate0_snapshot()

        def finalize_gate0_failure(self) -> None:
            self.finalize_physics_post_before_next_arm()
            self._gate0_failure_snapshot = self._gate0_snapshot()

        def full_trace(self) -> list[dict[str, Any]]:
            require(self._gate0_pending_entry is None, "pending full-trace entry")
            return list(self._gate0_apply_entries)

        def full_trace_failure_snapshot(self) -> dict[str, Any]:
            require(
                isinstance(self._gate0_failure_snapshot, dict),
                "missing full-trace failure snapshot",
            )
            return dict(self._gate0_failure_snapshot)

    # The production installer intentionally verifies this exact class name.
    FullTraceEefBinaryJointPositionTargetSlewAction.__name__ = (
        "EefBinaryJointPositionTargetSlewAction"
    )
    FullTraceEefBinaryJointPositionTargetSlewAction.__qualname__ = (
        "EefBinaryJointPositionTargetSlewAction"
    )
    FullTraceEefBinaryJointPositionTargetSlewAction.__module__ = MODULE_IDENTITY
    return FullTraceEefBinaryJointPositionTargetSlewAction


def validate_full_trace_snapshot(value: Any, *, field: str) -> dict[str, Any]:
    """Validate one complete synchronized 13-DOF trace phase."""

    require(
        isinstance(value, dict)
        and set(value)
        == GRIPPER_SNAPSHOT_FIELDS | ALL_JOINT_SNAPSHOT_FIELDS | {"interlock"},
        f"{field} full snapshot schema",
    )
    for name in GRIPPER_SNAPSHOT_FIELDS:
        values = value[name]
        require(
            isinstance(values, list)
            and len(values) == 6
            and all(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
                for item in values
            ),
            f"{field}.{name} finite six-vector",
        )
    for name in ALL_JOINT_SNAPSHOT_FIELDS:
        values = value[name]
        require(
            isinstance(values, list)
            and len(values) == len(JOINT_NAMES)
            and all(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
                for item in values
            ),
            f"{field}.{name} finite 13-vector",
        )
    for name in GRIPPER_SNAPSHOT_FIELDS:
        all_name = f"all_{name}"
        require(
            value[name] == value[all_name][7:],
            f"{field}.{name} all-joint binding",
        )
    interlock = value["interlock"]
    require(
        isinstance(interlock, dict) and set(interlock) == INTERLOCK_SNAPSHOT_FIELDS,
        f"{field}.interlock schema",
    )
    for name in (
        "arm_apply_call_count",
        "configured_substeps",
        "remaining_substeps",
        "activation_count",
        "active_apply_count",
        "released_apply_count",
    ):
        require(
            type(interlock[name]) is int and interlock[name] >= 0,
            f"{field}.interlock.{name}",
        )
    require(
        type(interlock["anchor_valid"]) is bool
        and isinstance(interlock["anchor_joint_pos_rad"], list)
        and len(interlock["anchor_joint_pos_rad"]) == 7
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in interlock["anchor_joint_pos_rad"]
        ),
        f"{field}.interlock anchor",
    )
    ramp = interlock["release_ramp"]
    require(
        isinstance(ramp, dict)
        and set(ramp) == RELEASE_RAMP_SNAPSHOT_FIELDS
        and ramp["enabled"] is True
        and ramp["phase"] in ("hold", "ramp", "release")
        and (
            ramp["next_index"] is None
            or (
                type(ramp["next_index"]) is int
                and 0 <= ramp["next_index"] < RELEASE_RAMP_SUBSTEPS
            )
        )
        and all(
            type(ramp[name]) is int and ramp[name] >= 0
            for name in (
                "release_observed_count",
                "started_count",
                "completed_count",
                "cancelled_by_reactivation_count",
                "target_apply_count",
                "cancelled_target_apply_count",
                "limited_target_apply_count",
                "limited_joint_target_count",
            )
        )
        and (
            (ramp["last_target_apply_index"] is None and ramp["last_index"] is None)
            or (
                type(ramp["last_target_apply_index"]) is int
                and ramp["last_target_apply_index"] >= 0
                and type(ramp["last_index"]) is int
                and 0 <= ramp["last_index"] < RELEASE_RAMP_SUBSTEPS
            )
        ),
        f"{field}.interlock release-ramp state",
    )
    return dict(value)


def _finite_seven_vector(value: Any, *, field: str) -> list[float]:
    require(
        isinstance(value, list)
        and len(value) == 7
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in value
        ),
        f"{field} finite seven-vector",
    )
    return [float(item) for item in value]


def _float32_ramp_target(
    *, current: float, nominal: float, maximum_delta: float, index: int
) -> float:
    if index == 0:
        return float32(current)
    if index == RELEASE_RAMP_SUBSTEPS - 1:
        return float32(nominal)
    fraction = float32(index / (RELEASE_RAMP_SUBSTEPS - 1))
    bound = float32_multiply(maximum_delta, fraction)
    nominal_delta = float32_subtract(nominal, current)
    clipped = max(min(nominal_delta, bound), -bound)
    return float32_add(current, clipped)


def validate_core_release_ramp_trace(
    entries: Sequence[Mapping[str, Any]],
    *,
    observation: Mapping[str, Any],
    controller_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Close all 48 production-core ramp transactions against live evidence."""

    require(
        isinstance(observation, Mapping)
        and observation.get("profile")
        == "production_v4_core_release_ramp_bitwise_observer_v1"
        and observation.get("production_base_commit") == PRODUCTION_BASE_COMMIT
        and observation.get("controller_profile") == CONTROLLER_PROFILE,
        "production core ramp observation identity",
    )
    observer_writes = observation.get("observer_write_contract")
    require(
        observer_writes
        == {
            "target_setter_call_count": 0,
            "failure_trace_write_count": 0,
            "release_ramp_state_write_count": 0,
            "gripper_target_or_state_write_count": 0,
        },
        "production observer write contract",
    )
    records = observation.get("entries")
    require(
        isinstance(records, list)
        and observation.get("entry_count") == len(records) == 48,
        "production core ramp observation count",
    )
    require(
        isinstance(controller_report, Mapping),
        "production controller report",
    )
    ramp = controller_report.get("arm_release_ramp")
    interlock = controller_report.get("gripper_close_arm_interlock")
    require(
        isinstance(ramp, Mapping)
        and isinstance(interlock, Mapping)
        and ramp.get("enabled") is True
        and ramp.get("phase") == "release"
        and ramp.get("next_index") is None
        and ramp.get("last_target_apply_index") == 2349
        and ramp.get("last_ramp_index") == 15
        and all(
            ramp.get(field) == expected
            for field, expected in EXPECTED_CORE_RAMP_COUNTS.items()
        )
        and interlock.get("anchor_completion_count") == 1
        and interlock.get("anchor_open_cancel_count") == 2,
        "production core ramp/interlock aggregate contract",
    )

    record_by_apply: dict[int, dict[str, Any]] = {}
    limited_applies_per_ramp = [0, 0, 0]
    limited_joints_per_ramp = [0, 0, 0]
    maximum_change_by_joint = [0.0] * 7
    for ordinal, raw_record in enumerate(records):
        require(
            isinstance(raw_record, dict)
            and set(raw_record) == CORE_RAMP_OBSERVATION_ENTRY_FIELDS,
            f"production ramp record {ordinal} schema",
        )
        record = dict(raw_record)
        apply_index = record.get("apply_index")
        ramp_index = record.get("ramp_index")
        expected_apply = EXPECTED_RAMP_APPLY_INDICES[ordinal]
        expected_ramp_index = EXPECTED_RAMP_INDICES[ordinal]
        require(
            record.get("profile")
            == "production_core_release_ramp_target_observation_v1"
            and apply_index == expected_apply
            and apply_index not in record_by_apply
            and record.get("policy_step") == expected_apply // DECIMATION
            and record.get("physics_substep") == expected_apply % DECIMATION
            and record.get("arm_joint_ids") == ARM_JOINT_IDS
            and record.get("arm_joint_names") == ARM_JOINT_NAMES
            and ramp_index == expected_ramp_index
            and record.get("observer_target_setter_call_count") == 0
            and record.get("observer_failure_trace_write_count") == 0
            and record.get("observer_release_ramp_state_write_count") == 0
            and record.get("observer_gripper_target_or_state_write_count") == 0,
            f"production ramp record {ordinal} identity",
        )
        require(
            float32_equal(
                record.get("fraction_float32"),
                float32(expected_ramp_index / (RELEASE_RAMP_SUBSTEPS - 1)),
            ),
            f"production ramp record {ordinal} fraction",
        )
        maximum_delta = _finite_seven_vector(
            record.get("nominal_max_delta_joint_pos_rad"),
            field=f"production ramp record {ordinal} nominal maximum",
        )
        require_float32_vector_equal(
            maximum_delta,
            ARM_NOMINAL_MAX_DELTA_RAD,
            field=f"production ramp record {ordinal} nominal maximum",
        )
        current = _finite_seven_vector(
            record.get("current_joint_pos_rad"),
            field=f"production ramp record {ordinal} current",
        )
        _finite_seven_vector(
            record.get("raw_dls_joint_pos_target_rad"),
            field=f"production ramp record {ordinal} raw DLS",
        )
        nominal = _finite_seven_vector(
            record.get("nominal_safe_target_rad"),
            field=f"production ramp record {ordinal} nominal",
        )
        core = _finite_seven_vector(
            record.get("production_core_target_rad"),
            field=f"production ramp record {ordinal} core",
        )
        failure = _finite_seven_vector(
            record.get("failure_trace_target_rad"),
            field=f"production ramp record {ordinal} failure trace",
        )
        live = _finite_seven_vector(
            record.get("live_setter_readback_target_rad"),
            field=f"production ramp record {ordinal} live readback",
        )
        recomputed = [
            _float32_ramp_target(
                current=current_value,
                nominal=nominal_value,
                maximum_delta=maximum_value,
                index=expected_ramp_index,
            )
            for current_value, nominal_value, maximum_value in zip(
                current, nominal, maximum_delta, strict=True
            )
        ]
        for name, actual in (
            ("production core", core),
            ("failure trace", failure),
            ("live setter readback", live),
        ):
            require_float32_vector_equal(
                actual,
                recomputed,
                field=f"production ramp record {ordinal} {name}",
            )
        hashes = record.get("target_little_endian_float32_sha256")
        expected_hash = little_endian_float32_sha256(
            recomputed, field=f"production ramp record {ordinal} target hash"
        )
        require(
            hashes
            == {
                "production_core": expected_hash,
                "independent_helper": expected_hash,
                "failure_trace": expected_hash,
                "live_setter_readback": expected_hash,
            }
            and record.get("bitwise_contract")
            == {
                "core_equals_independent_helper": True,
                "core_equals_failure_trace": True,
                "core_equals_live_setter_readback": True,
            }
            and record.get("endpoint_contract")
            == {
                "index0_equals_current": True,
                "index15_equals_nominal": True,
            },
            f"production ramp record {ordinal} bitwise evidence",
        )
        expected_mask = [
            not float32_equal(target, nominal_value)
            for target, nominal_value in zip(core, nominal, strict=True)
        ]
        limited_joint_count = sum(expected_mask)
        require(
            record.get("limited_joint_mask") == expected_mask
            and record.get("limited_joint_count") == limited_joint_count,
            f"production ramp record {ordinal} limited mask",
        )
        maximum_change = max(
            abs(float32_subtract(nominal_value, target))
            for nominal_value, target in zip(nominal, core, strict=True)
        )
        require(
            float32_equal(
                record.get("max_abs_nominal_to_core_target_change_rad"),
                maximum_change,
            ),
            f"production ramp record {ordinal} maximum change",
        )
        ramp_ordinal = ordinal // RELEASE_RAMP_SUBSTEPS
        limited_applies_per_ramp[ramp_ordinal] += int(limited_joint_count > 0)
        limited_joints_per_ramp[ramp_ordinal] += limited_joint_count
        for joint_index, (nominal_value, target) in enumerate(
            zip(nominal, core, strict=True)
        ):
            maximum_change_by_joint[joint_index] = max(
                maximum_change_by_joint[joint_index],
                abs(float32_subtract(nominal_value, target)),
            )

        trace_entry = entries[expected_apply]
        command = trace_entry["command_after_setters"]
        require_float32_vector_equal(
            command["all_joint_pos_rad"][:7],
            current,
            field=f"production ramp record {ordinal} trace current",
        )
        require_float32_vector_equal(
            command["all_joint_pos_target_rad"][:7],
            live,
            field=f"production ramp record {ordinal} trace live target",
        )
        require(
            command["interlock"]["release_ramp"]["target_apply_count"] == ordinal + 1,
            f"production ramp record {ordinal} core counter",
        )
        record_by_apply[expected_apply] = record

    previous_count = 0
    for apply_index, entry in enumerate(entries):
        states = [
            entry[phase]["interlock"]["release_ramp"]
            for phase in ("pre", "command_after_setters", "post")
        ]
        require(
            states[0] == states[1] == states[2] and states[0]["enabled"] is True,
            f"production ramp snapshot identity at apply {apply_index}",
        )
        current_count = states[0]["target_apply_count"]
        require(
            type(current_count) is int
            and current_count - previous_count in (0, 1)
            and (current_count == previous_count + 1)
            is (apply_index in record_by_apply),
            f"production ramp snapshot cadence at apply {apply_index}",
        )
        previous_count = current_count
    require(
        previous_count == 48
        and limited_applies_per_ramp == EXPECTED_LIMITED_APPLIES_PER_RAMP
        and limited_joints_per_ramp == EXPECTED_LIMITED_JOINTS_PER_RAMP
        and sum(limited_applies_per_ramp) == 38
        and sum(limited_joints_per_ramp) == 221,
        "production ramp exact per-window counts",
    )
    report_maximum = _finite_seven_vector(
        ramp.get("max_abs_nominal_to_ramped_target_change_rad"),
        field="production ramp report maximum",
    )
    require_float32_vector_equal(
        report_maximum,
        maximum_change_by_joint,
        field="production ramp record/report maximum",
    )
    encoded = json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return {
        "profile": "production_v4_core_release_ramp_bitwise_gate_v1",
        "controller_profile": CONTROLLER_PROFILE,
        "production_base_commit": PRODUCTION_BASE_COMMIT,
        "entry_count": len(records),
        "canonical_json_sha256": sha256(encoded),
        "apply_windows": [list(window) for window in EXPECTED_RAMP_WINDOWS],
        "ramp_indices": list(EXPECTED_RAMP_INDICES),
        "limited_applies_per_ramp": limited_applies_per_ramp,
        "limited_joints_per_ramp": limited_joints_per_ramp,
        "aggregate_counts": dict(EXPECTED_CORE_RAMP_COUNTS),
        "maximum_change_by_joint_rad": maximum_change_by_joint,
        "observer_write_contract": dict(observer_writes),
        "passed": True,
    }


def validate_arm_trace_safety(
    entries: Sequence[Mapping[str, Any]], *, outcome: Mapping[str, Any]
) -> dict[str, Any]:
    """Close arm q/dq/target safety, including the terminal tail state."""

    classification = outcome.get("classification")
    parsed_failure = outcome.get("parsed_numerical_failure")
    failure_apply_index: int | None = None
    if classification == "production_v4_replay_numerical_failure_observed":
        require(isinstance(parsed_failure, dict), "arm trace parsed failure")
        failure_apply_index = int(parsed_failure["policy_step"]) * DECIMATION + int(
            parsed_failure["physics_substep"]
        )
        require(failure_apply_index == len(entries), "arm trace failure boundary")

    max_abs_velocity = [0.0] * 7
    max_abs_target_delta = [0.0] * 7
    max_target_guard_band_violation = [0.0] * 7
    terminal_failure_over_limit = False
    for apply_index, entry in enumerate(entries):
        for phase in ("pre", "command_after_setters", "post"):
            snapshot = entry[phase]
            positions = snapshot["all_joint_pos_rad"][:7]
            velocities = snapshot["all_joint_vel_rad_s"][:7]
            for joint_index, (position, velocity) in enumerate(
                zip(positions, velocities, strict=True)
            ):
                lower, upper = ARM_SOFT_LIMITS_RAD[joint_index]
                require(
                    lower - ARM_POSITION_TOLERANCE_RAD
                    <= position
                    <= upper + ARM_POSITION_TOLERANCE_RAD,
                    f"arm position safety at apply {apply_index} phase {phase} "
                    f"joint {joint_index}",
                )
                max_abs_velocity[joint_index] = max(
                    max_abs_velocity[joint_index], abs(float(velocity))
                )
                over_limit = (
                    abs(float(velocity))
                    > ARM_VELOCITY_LIMITS_RAD_S[joint_index]
                    + ARM_VELOCITY_TOLERANCE_RAD_S
                )
                allowed_terminal_failure = (
                    failure_apply_index is not None
                    and failure_apply_index > 0
                    and apply_index == failure_apply_index - 1
                    and phase == "post"
                )
                require(
                    not over_limit or allowed_terminal_failure,
                    f"arm velocity safety at apply {apply_index} phase {phase} "
                    f"joint {joint_index}",
                )
                terminal_failure_over_limit |= over_limit and allowed_terminal_failure

        pre = entry["pre"]
        command = entry["command_after_setters"]
        post = entry["post"]
        for field in ("all_joint_pos_rad", "all_joint_vel_rad_s"):
            require_float32_vector_equal(
                pre[field][:7],
                command[field][:7],
                field=f"arm pre/command {field} at apply {apply_index}",
            )
        require_float32_vector_equal(
            pre["all_joint_pos_target_rad"][:7],
            command["all_joint_pos_target_rad"][:7],
            field=f"arm pre/command target at apply {apply_index}",
        )
        require_float32_vector_equal(
            command["all_joint_pos_target_rad"][:7],
            post["all_joint_pos_target_rad"][:7],
            field=f"arm command/post target at apply {apply_index}",
        )
        current = command["all_joint_pos_rad"][:7]
        target = command["all_joint_pos_target_rad"][:7]
        for joint_index, (current_value, target_value) in enumerate(
            zip(current, target, strict=True)
        ):
            lower, upper = ARM_SOFT_LIMITS_RAD[joint_index]
            require(
                lower - ARM_POSITION_TOLERANCE_RAD
                <= target_value
                <= upper + ARM_POSITION_TOLERANCE_RAD,
                f"arm target position safety at apply {apply_index} joint {joint_index}",
            )
            delta = abs(float32_subtract(target_value, current_value))
            max_abs_target_delta[joint_index] = max(
                max_abs_target_delta[joint_index], delta
            )
            require(
                delta
                <= ARM_NOMINAL_MAX_DELTA_RAD[joint_index]
                + ARM_TARGET_SLEW_TOLERANCE_RAD,
                f"arm actual target slew safety at apply {apply_index} "
                f"joint {joint_index}",
            )
            current_outer_violation = max(
                float32_subtract(lower, current_value),
                float32_subtract(current_value, upper),
                0.0,
            )
            target_lower = float32_add(lower, ARM_PHYSICAL_MAX_DELTA_RAD[joint_index])
            target_upper = float32_subtract(
                upper, ARM_PHYSICAL_MAX_DELTA_RAD[joint_index]
            )
            target_guard_violation = max(
                float32_subtract(target_lower, target_value),
                float32_subtract(target_value, target_upper),
                0.0,
            )
            max_target_guard_band_violation[joint_index] = max(
                max_target_guard_band_violation[joint_index],
                target_guard_violation,
            )
            require(
                target_guard_violation
                <= current_outer_violation + ARM_TARGET_SLEW_TOLERANCE_RAD,
                f"arm target guard-band recovery at apply {apply_index} "
                f"joint {joint_index}",
            )
    if failure_apply_index is not None and failure_apply_index > 0:
        require(
            terminal_failure_over_limit,
            "arm trace failure boundary lacks terminal velocity violation",
        )
    if classification == "production_v4_replay_completed_source_and_tail":
        require(not terminal_failure_over_limit, "completed arm trace terminal safety")
    terminal_post = None if not entries else entries[-1]["post"]
    return {
        "profile": "all_completed_and_terminal_arm_state_target_safety_v1",
        "checked_entry_count": len(entries),
        "max_abs_joint_velocity_rad_s": max_abs_velocity,
        "max_abs_actual_target_delta_rad": max_abs_target_delta,
        "max_actual_target_guard_band_violation_rad": (max_target_guard_band_violation),
        "terminal_post_arm_joint_pos_rad": (
            None if terminal_post is None else terminal_post["all_joint_pos_rad"][:7]
        ),
        "terminal_post_arm_joint_vel_rad_s": (
            None if terminal_post is None else terminal_post["all_joint_vel_rad_s"][:7]
        ),
        "terminal_failure_velocity_exception": terminal_failure_over_limit,
        "passed": True,
    }


CONTROLLER_FAILURE_EVIDENCE_FIELDS = {
    "profile",
    "failure_exception",
    "parsed_failure",
    "arm_failure_runtime_evidence",
    "terminal_full_trace_snapshot",
    "full_trace_entry_count_at_failure",
}


def _finite_evidence_values(value: Any, *, field: str) -> list[float]:
    require(
        isinstance(value, dict)
        and set(value) == {"values", "finite_mask", "finite_count"}
        and value.get("finite_mask") == [True] * 7
        and value.get("finite_count") == 7,
        f"{field} finite evidence metadata",
    )
    return _finite_seven_vector(value.get("values"), field=field)


def validate_failed_apply_terminal_snapshot(
    value: Any, *, entries: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Bind the failed attempt's +1 call count without inventing a command."""

    terminal = validate_full_trace_snapshot(
        value, field="controller terminal full-trace snapshot"
    )
    if not entries:
        require(
            terminal["interlock"]["arm_apply_call_count"] == 1,
            "initial controller failure apply count",
        )
        return terminal
    previous = entries[-1]["post"]
    require(isinstance(previous, Mapping), "controller prior terminal trace snapshot")
    for field in GRIPPER_SNAPSHOT_FIELDS | ALL_JOINT_SNAPSHOT_FIELDS:
        require(
            terminal[field] == previous[field],
            f"controller failed-apply terminal {field} identity",
        )
    terminal_interlock = terminal["interlock"]
    previous_interlock = previous["interlock"]
    require(
        terminal_interlock["arm_apply_call_count"]
        == previous_interlock["arm_apply_call_count"] + 1,
        "controller failed-apply call-count delta",
    )
    for field in INTERLOCK_SNAPSHOT_FIELDS - {"arm_apply_call_count"}:
        require(
            terminal_interlock[field] == previous_interlock[field],
            f"controller failed-apply interlock {field} identity",
        )
    return terminal


def validate_controller_failure_evidence(
    value: Any,
    *,
    failure: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Independently bind a caught velocity abort to live controller evidence."""

    require(
        isinstance(value, dict) and set(value) == CONTROLLER_FAILURE_EVIDENCE_FIELDS,
        "controller failure evidence schema",
    )
    require(
        value.get("profile") == "fulltrace_current_velocity_abort_evidence_v1"
        and value.get("failure_exception") == failure
        and value.get("full_trace_entry_count_at_failure") == len(entries),
        "controller failure evidence identity",
    )
    require(
        isinstance(failure, Mapping)
        and set(failure) == {"type", "message", "traceback"}
        and failure.get("type")
        == "polaris.robust_differential_ik.DifferentialIKInvariantError"
        and isinstance(failure.get("message"), str)
        and isinstance(failure.get("traceback"), str),
        "controller failure exception type/schema",
    )
    parsed = gate0.parse_failure_exception(failure["message"])
    require(value.get("parsed_failure") == parsed, "controller parsed failure identity")
    arm_failure = value.get("arm_failure_runtime_evidence")
    gate0._validate_arm_failure_runtime_evidence(  # noqa: SLF001
        arm_failure, expected_failure=parsed
    )
    require(isinstance(arm_failure, dict), "controller arm failure evidence")
    safety = arm_failure.get("ik_safety")
    require(isinstance(safety, dict), "controller failure safety report")
    abort = safety.get("current_joint_velocity_abort")
    require(isinstance(abort, dict), "controller velocity-abort evidence")
    from polaris.eef_ik_safety import (  # noqa: PLC0415
        current_joint_velocity_abort_evidence_sha256,
        format_current_joint_velocity_abort_message,
    )

    require(
        current_joint_velocity_abort_evidence_sha256(abort) == parsed["evidence_sha256"]
        and format_current_joint_velocity_abort_message(abort) == failure["message"]
        and abort.get("policy_step") == parsed["policy_step"]
        and abort.get("physics_substep") == parsed["physics_substep"],
        "controller failure message/evidence digest binding",
    )
    terminal = validate_failed_apply_terminal_snapshot(
        value.get("terminal_full_trace_snapshot"),
        entries=entries,
    )
    for evidence_field, snapshot_field in (
        ("arm_joint_pos_rad", "all_joint_pos_rad"),
        ("arm_joint_vel_rad_s", "all_joint_vel_rad_s"),
        ("arm_joint_target_rad", "all_joint_pos_target_rad"),
        ("arm_joint_velocity_target_rad_s", "all_joint_vel_target_rad_s"),
        ("arm_joint_effort_target_nm", "all_joint_effort_target_nm"),
    ):
        captured = _finite_evidence_values(
            arm_failure.get(evidence_field), field=f"controller {evidence_field}"
        )
        require_float32_vector_equal(
            terminal[snapshot_field][:7],
            captured,
            field=f"controller terminal {snapshot_field}",
        )
    return dict(value)


def summarize_trace(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    max_velocity = [0.0] * len(JOINT_NAMES)
    max_acceleration = [0.0] * len(JOINT_NAMES)
    max_mimic_residual = [0.0] * len(FOLLOWER_INDICES)
    last_active_entries: list[dict[str, int]] = []
    first_released_entries: list[dict[str, int]] = []
    previous_remaining: int | None = None
    for entry in entries:
        for phase in ("pre", "post"):
            snapshot = entry[phase]
            for index, value in enumerate(snapshot["all_joint_vel_rad_s"]):
                max_velocity[index] = max(max_velocity[index], abs(value))
            for index, value in enumerate(snapshot["all_joint_acc_rad_s2"]):
                max_acceleration[index] = max(max_acceleration[index], abs(value))
            positions = snapshot["all_joint_pos_rad"]
            driver = positions[7]
            for follower_offset, (joint_index, gear) in enumerate(
                zip(FOLLOWER_INDICES, FOLLOWER_GEARS, strict=True)
            ):
                residual = positions[joint_index] + gear * driver
                max_mimic_residual[follower_offset] = max(
                    max_mimic_residual[follower_offset], abs(residual)
                )
        remaining = int(entry["pre"]["interlock"]["remaining_substeps"])
        if previous_remaining == 1 and remaining == 0:
            identity = {
                "apply_index": int(entry["apply_index"]),
                "policy_step": int(entry["policy_step"]),
                "physics_substep": int(entry["physics_substep"]),
            }
            last_active_entries.append(identity)
            next_apply = identity["apply_index"] + 1
            if next_apply < len(entries):
                first_released_entries.append(
                    {
                        "apply_index": next_apply,
                        "policy_step": next_apply // DECIMATION,
                        "physics_substep": next_apply % DECIMATION,
                    }
                )
        previous_remaining = remaining
    encoded = json.dumps(
        entries, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return {
        "entry_count": len(entries),
        "first_apply_index": (None if not entries else int(entries[0]["apply_index"])),
        "last_apply_index": None if not entries else int(entries[-1]["apply_index"]),
        "trace_canonical_json_sha256": sha256(encoded),
        "joint_names": list(JOINT_NAMES),
        "max_abs_joint_velocity_rad_s": max_velocity,
        "max_abs_joint_acceleration_rad_s2": max_acceleration,
        "follower_joint_names": [JOINT_NAMES[index] for index in FOLLOWER_INDICES],
        "max_abs_mimic_residual_rad": max_mimic_residual,
        "interlock_last_active_entries": last_active_entries,
        "interlock_first_released_entries": first_released_entries,
    }


def validate_trace_cadence(
    entries: Sequence[Mapping[str, Any]],
    *,
    outcome: Mapping[str, Any],
    ramp_observation: Mapping[str, Any],
    controller_report: Mapping[str, Any],
) -> dict[str, Any]:
    for apply_index, entry in enumerate(entries):
        require(
            isinstance(entry, dict)
            and set(entry) == TRACE_ENTRY_FIELDS
            and entry.get("apply_index") == apply_index
            and entry.get("policy_step") == apply_index // DECIMATION
            and entry.get("physics_substep") == apply_index % DECIMATION,
            f"full-trace cadence drift at apply {apply_index}",
        )
        require(
            entry.get("raw_action") in (0.0, 1.0)
            and all(
                isinstance(entry.get(name), (int, float))
                and not isinstance(entry.get(name), bool)
                and math.isfinite(float(entry[name]))
                for name in ("requested_endpoint_rad", "target_after_setter_rad")
            ),
            f"full-trace command scalar drift at apply {apply_index}",
        )
        for phase in ("pre", "command_after_setters", "post"):
            validate_full_trace_snapshot(
                entry.get(phase), field=f"entry[{apply_index}].{phase}"
            )
    classification = outcome.get("classification")
    completed = outcome.get("source_actions_completed")
    tail_policy_steps_completed = outcome.get("tail_policy_steps_completed")
    tail_physics_substeps_completed = outcome.get("tail_physics_substeps_completed")
    require(
        type(completed) is int and 0 <= completed <= ACTION_COUNT, "completed count"
    )
    require(
        type(tail_policy_steps_completed) is int
        and 0 <= tail_policy_steps_completed <= TAIL_POLICY_STEPS,
        "completed tail policy-step count",
    )
    require(
        type(tail_physics_substeps_completed) is int
        and 0 <= tail_physics_substeps_completed <= TAIL_PHYSICS_SUBSTEPS,
        "completed tail physics-substep count",
    )
    if classification == "production_v4_replay_completed_source_and_tail":
        expected_count = ACTION_COUNT * DECIMATION + TAIL_PHYSICS_SUBSTEPS
        require(
            completed == ACTION_COUNT
            and tail_policy_steps_completed == TAIL_POLICY_STEPS
            and tail_physics_substeps_completed == TAIL_PHYSICS_SUBSTEPS
            and len(entries) == expected_count,
            "completed follow-up full-trace count",
        )
    else:
        parsed_failure = outcome.get("parsed_numerical_failure")
        if isinstance(parsed_failure, dict):
            failure_policy_step = parsed_failure.get("policy_step")
            require(
                type(failure_policy_step) is int
                and failure_policy_step == completed + tail_policy_steps_completed
                and type(parsed_failure.get("physics_substep")) is int,
                "parsed failure/completed-step binding",
            )
            expected_count = (
                failure_policy_step * DECIMATION + parsed_failure["physics_substep"]
            )
            require(
                len(entries) == expected_count,
                "parsed failed-follow-up exact full-trace count",
            )
        else:
            lower = completed * DECIMATION + tail_policy_steps_completed * DECIMATION
            require(
                lower <= len(entries) <= lower + DECIMATION - 1,
                "unparsed failed-follow-up full-trace count/cadence",
            )
            expected_count = None
    source_entry_count = min(len(entries), ACTION_COUNT * DECIMATION)
    tail_entry_count = max(len(entries) - ACTION_COUNT * DECIMATION, 0)
    require(
        tail_entry_count == tail_physics_substeps_completed,
        "tail report/full-trace entry count binding",
    )
    arm_safety_gate = validate_arm_trace_safety(entries, outcome=outcome)
    release_ramp_trace_gate = validate_core_release_ramp_trace(
        entries,
        observation=ramp_observation,
        controller_report=controller_report,
    )
    return {
        "profile": "contiguous_source_actions_then_frozen_tail_phases_v1",
        "controller_profile": CONTROLLER_PROFILE,
        "entry_count": len(entries),
        "expected_entry_count": expected_count,
        "decimation": DECIMATION,
        "source_action_segment": {
            "policy_step_start": 0,
            "policy_steps_requested": ACTION_COUNT,
            "policy_steps_completed": completed,
            "physics_substeps_requested": ACTION_COUNT * DECIMATION,
            "trace_entry_count": source_entry_count,
            "apply_index_start": 0,
            "apply_index_stop_exclusive": source_entry_count,
        },
        "frozen_tail_segment": {
            "policy_step_start": ACTION_COUNT,
            "policy_steps_requested": TAIL_POLICY_STEPS,
            "policy_steps_completed": tail_policy_steps_completed,
            "physics_substeps_requested": TAIL_PHYSICS_SUBSTEPS,
            "physics_substeps_completed": tail_physics_substeps_completed,
            "trace_entry_count": tail_entry_count,
            "apply_index_start": ACTION_COUNT * DECIMATION,
            "apply_index_stop_exclusive": (
                ACTION_COUNT * DECIMATION + tail_entry_count
            ),
        },
        "phase_contract": {
            "pre": (
                "physical_state_before_current_physics_arm_command_after_setter_"
                "gripper_target_before_setter_v1"
            ),
            "command_after_setters": (
                "physical_state_before_current_physics_arm_and_gripper_targets_current_v1"
            ),
            "post": (
                "physical_state_after_current_physics_with_current_targets_captured_"
                "before_next_arm_apply_v1"
            ),
        },
        "arm_safety_gate": arm_safety_gate,
        "release_ramp_trace_gate": release_ramp_trace_gate,
    }


def tensor_evidence(tensor: Any) -> dict[str, Any]:
    return {
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "shape": list(tensor.shape),
        "values": [float(value) for value in tensor.detach().cpu().flatten().tolist()],
    }


def capture_production_v4_runtime_contract(
    *, env: Any, arm_term: Any, torch_module: Any, configured_profile: Any
) -> dict[str, Any]:
    """Verify the configured v4 runtime without mutating any controller limit."""

    robot = env.unwrapped.scene["robot"]
    getter = robot.root_physx_view.get_dof_max_velocities
    before_tensor = getter().clone()
    after_tensor = getter().clone()
    require(
        tuple(before_tensor.shape) == (1, 13)
        and torch_module.equal(before_tensor, after_tensor),
        "production v4 follower velocity-limit identity",
    )
    expected_limits = ARM_VELOCITY_LIMITS_RAD_S + [PRODUCTION_FOLLOWER_LIMIT] * 6
    require_float32_vector_equal(
        vector(before_tensor, length=13, field="production velocity limits"),
        expected_limits,
        field="production velocity limits",
    )
    configured = int(
        getattr(arm_term, "_gripper_close_arm_interlock_configured_substeps")
    )
    require(configured == 86, "production v4 close-anchor duration")
    flags = {
        "failure_substep_trace": bool(
            getattr(arm_term, "_failure_substep_trace_enabled")
        ),
        "arm_slew_headroom": bool(getattr(arm_term, "_arm_slew_headroom_enabled")),
        "gripper_close_arm_interlock": bool(
            getattr(arm_term, "_gripper_close_arm_interlock_enabled")
        ),
        "arm_release_ramp": bool(getattr(arm_term, "_arm_release_ramp_enabled")),
    }
    require(
        configured_profile.profile == CONTROLLER_PROFILE
        and configured_profile.failure_substep_trace_enabled is True
        and configured_profile.all_six_gripper_trace_enabled is True
        and configured_profile.arm_release_ramp_enabled is True
        and all(flags.values()),
        "production v4 profile/runtime flag binding",
    )
    return {
        "profile": "production_v4_core_replay_runtime_contract_v1",
        "variant": VARIANT,
        "controller_profile": CONTROLLER_PROFILE,
        "production_base_commit": PRODUCTION_BASE_COMMIT,
        "configured_profile": {
            "profile": configured_profile.profile,
            "failure_substep_trace_enabled": (
                configured_profile.failure_substep_trace_enabled
            ),
            "all_six_gripper_trace_enabled": (
                configured_profile.all_six_gripper_trace_enabled
            ),
            "arm_slew_headroom_enabled": configured_profile.arm_slew_headroom_enabled,
            "gripper_close_arm_interlock_enabled": (
                configured_profile.gripper_close_arm_interlock_enabled
            ),
            "arm_release_ramp_enabled": configured_profile.arm_release_ramp_enabled,
            "target_slew_rate_0p25_enabled": (
                configured_profile.target_slew_rate_0p25_enabled
            ),
            "target_slew_profile": configured_profile.target_slew_profile,
            "close_interlock_profile": configured_profile.close_interlock_profile,
            "close_interlock_substeps": configured_profile.close_interlock_substeps,
            "fixed_activation_anchor": configured_profile.fixed_activation_anchor,
            "mimic_compliance_profile": configured_profile.mimic_compliance_profile,
        },
        "runtime_flags": flags,
        "passive_follower_velocity_cap_rad_s": PRODUCTION_FOLLOWER_LIMIT,
        "passive_follower_velocity_limit_setter_call_count": 0,
        "velocity_limits_first_read": tensor_evidence(before_tensor),
        "velocity_limits_second_read": tensor_evidence(after_tensor),
        "configured_close_anchor_substeps": configured,
        "observer_target_setter_call_count": 0,
        "observer_failure_trace_write_count": 0,
        "observer_release_ramp_state_write_count": 0,
        "observer_gripper_target_or_state_write_count": 0,
    }


def model_view_frame(observation: Mapping[str, Any]) -> Any:
    import numpy as np  # noqa: PLC0415
    from polaris.policy.lap_eef_pose_client import (  # noqa: PLC0415
        preprocess_lap_wrist_image,
        resize_lap_image,
    )

    external = resize_lap_image(observation["splat"]["external_cam"])
    wrist = preprocess_lap_wrist_image(
        observation["splat"]["wrist_cam"], rotate_180=True
    )
    frame = np.concatenate([external, wrist], axis=1)
    require(
        frame.shape == (VIDEO_HEIGHT, VIDEO_WIDTH, 3) and str(frame.dtype) == "uint8",
        "model-view video frame contract",
    )
    return frame


def terminal_model_view_frame(env: Any) -> Any:
    runtime = env.unwrapped
    runtime.sim.render()
    return model_view_frame({"splat": runtime.custom_render(False)})


def publish_video(path: Path, frames: Sequence[Any]) -> dict[str, Any]:
    import mediapy  # noqa: PLC0415
    from polaris.eval_artifacts import probe_episode_video  # noqa: PLC0415

    path = path.resolve()
    require(path.suffix == ".mp4" and not path.exists(), "video output namespace")
    require(bool(frames), "empty video")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = path.with_name(f".{path.stem}.{os.getpid()}.encoded.tmp.mp4")
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.faststart.tmp.mp4")
    require(not encoded.exists() and not temporary.exists(), "stale video temporary")
    try:
        mediapy.write_video(encoded, frames, fps=VIDEO_FPS)
        subprocess.run(
            [
                "/usr/bin/ffmpeg",
                "-v",
                "error",
                "-xerror",
                "-i",
                str(encoded),
                "-map",
                "0:v:0",
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(temporary),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        probe = probe_episode_video(temporary)
        require(
            probe
            == {
                "frame_count": len(frames),
                "height": VIDEO_HEIGHT,
                "width": VIDEO_WIDTH,
            },
            f"video decode probe drift: {probe!r}",
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
        path.chmod(0o444)
    finally:
        encoded.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
    identity = file_identity(path)
    require(identity["mode"] == "0444" and identity["nlink"] == 1, "video mode/link")
    return {
        **identity,
        "profile": "lap_model_view_external_then_rot180_wrist_224x448_v1",
        "container_profile": "mp4_h264_faststart_v1",
        "fps": VIDEO_FPS,
        "frame_count": len(frames),
        "height": VIDEO_HEIGHT,
        "width": VIDEO_WIDTH,
    }


def classify_outcome(
    failure: Mapping[str, Any] | None,
    source_actions_completed: int,
    tail_policy_steps_completed: int,
    trace_entry_count: int,
) -> dict[str, Any]:
    require(
        type(source_actions_completed) is int
        and 0 <= source_actions_completed <= ACTION_COUNT,
        "follow-up source completion count",
    )
    require(
        type(tail_policy_steps_completed) is int
        and 0 <= tail_policy_steps_completed <= TAIL_POLICY_STEPS,
        "follow-up tail completion count",
    )
    require(type(trace_entry_count) is int and trace_entry_count >= 0, "trace count")
    tail_physics_substeps_completed = max(
        trace_entry_count - ACTION_COUNT * DECIMATION, 0
    )
    require(
        tail_physics_substeps_completed <= TAIL_PHYSICS_SUBSTEPS,
        "follow-up tail trace overflow",
    )
    parsed: dict[str, Any] | None = None
    parse_failure: str | None = None
    failure_segment: str | None = None
    failure_binding_valid = False
    if failure is not None:
        try:
            parsed = gate0.parse_failure_exception(str(failure.get("message", "")))
        except gate0.Gate0ReplayValidationError as error:
            parse_failure = str(error)
        if parsed is not None:
            failure_policy_step = parsed["policy_step"]
            failure_segment = (
                "source_actions"
                if failure_policy_step < ACTION_COUNT
                else "frozen_final_command_tail"
            )
            expected_completed_policy_steps = (
                source_actions_completed + tail_policy_steps_completed
            )
            expected_trace_entries = (
                failure_policy_step * DECIMATION + parsed["physics_substep"]
            )
            failure_binding_valid = (
                failure_policy_step == expected_completed_policy_steps
                and expected_trace_entries == trace_entry_count
                and (
                    (
                        failure_segment == "source_actions"
                        and tail_policy_steps_completed == 0
                    )
                    or (
                        failure_segment == "frozen_final_command_tail"
                        and source_actions_completed == ACTION_COUNT
                    )
                )
            )
    full_completion = (
        failure is None
        and source_actions_completed == ACTION_COUNT
        and tail_policy_steps_completed == TAIL_POLICY_STEPS
        and tail_physics_substeps_completed == TAIL_PHYSICS_SUBSTEPS
        and trace_entry_count == ACTION_COUNT * DECIMATION + TAIL_PHYSICS_SUBSTEPS
    )
    return {
        "replay_completed": full_completion,
        "classification": (
            "production_v4_replay_completed_source_and_tail"
            if full_completion
            else (
                "production_v4_replay_numerical_failure_observed"
                if failure is not None and failure_binding_valid
                else "production_v4_replay_incomplete_or_unbound_failure"
            )
        ),
        "controller_completed_actions": source_actions_completed,
        "source_actions_completed": source_actions_completed,
        "tail_policy_steps_completed": tail_policy_steps_completed,
        "tail_physics_substeps_completed": tail_physics_substeps_completed,
        "failure_segment": failure_segment,
        "parsed_numerical_failure": parsed,
        "numerical_failure_parse_error": parse_failure,
        "failure_binding_valid": failure_binding_valid,
    }


def production_repository_provenance(expected_commit: str) -> dict[str, Any]:
    """Require a clean launch child atop the target-tested production core."""

    repository = gate0._repository_provenance(expected_commit)  # noqa: SLF001
    implementation = subprocess.run(
        ["git", "rev-parse", "HEAD^"],
        cwd=repository["path"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    parent = subprocess.run(
        ["git", "rev-parse", "HEAD^^"],
        cwd=repository["path"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    production_base = subprocess.run(
        ["git", "rev-parse", "HEAD^^^"],
        cwd=repository["path"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(
        implementation == REPLAY_IMPLEMENTATION_COMMIT,
        "production replay implementation parent drift",
    )
    require(parent == REPLAY_PARENT_COMMIT, "production replay test parent drift")
    require(
        production_base == PRODUCTION_BASE_COMMIT,
        "production replay core great-grandparent drift",
    )
    return {
        **repository,
        "replay_implementation_commit": REPLAY_IMPLEMENTATION_COMMIT,
        "replay_implementation_relation": "exact_first_parent_v1",
        "replay_parent_commit": REPLAY_PARENT_COMMIT,
        "replay_parent_relation": "exact_first_grandparent_v1",
        "production_base_commit": PRODUCTION_BASE_COMMIT,
        "production_base_relation": "exact_first_great_grandparent_v1",
        "source_trace_polaris_commit": SOURCE_TRACE_POLARIS_COMMIT,
    }


def run_live(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    import gymnasium as gym  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: PLC0415

    import polaris.environments  # noqa: F401, PLC0415
    from polaris.config import (  # noqa: PLC0415
        EEF_CONTROLLER_RELEASE_RAMP_CANDIDATE_PROFILE,
    )
    from polaris.eef_controller_profile import (  # noqa: PLC0415
        capture_eef_controller_repair_candidate_report,
        configure_eef_controller_profile,
    )
    from polaris.eef_controller_repair import (  # noqa: PLC0415
        apply_arm_release_ramp_target,
        bound_joint_position_target,
    )
    from polaris.eef_gripper_failure_trace import (  # noqa: PLC0415
        validate_eef_all_six_gripper_trace,
    )
    from polaris.eef_gripper_runtime import (  # noqa: PLC0415
        install_eef_gripper_runtime,
        record_eef_gripper_post_policy_step,
    )
    from polaris.eef_runtime_contract import (  # noqa: PLC0415
        begin_eef_safety_episode,
        configure_ego_lap_environment_timeout,
        eef_episode_safety_report,
        validate_eef_runtime_frame,
        validate_ego_lap_runtime_protocol,
    )
    from polaris.environments.droid_cfg import EgoLapEefPoseActionCfg  # noqa: PLC0415
    from polaris.environments.robot_cfg import configure_eef_pose_joint_safety  # noqa: PLC0415
    from polaris.robust_differential_ik import DifferentialIKNumericalError  # noqa: PLC0415
    from polaris.utils import load_eval_initial_conditions  # noqa: PLC0415

    require(
        EEF_CONTROLLER_RELEASE_RAMP_CANDIDATE_PROFILE == CONTROLLER_PROFILE,
        "production v4 controller profile constant",
    )
    repository = production_repository_provenance(args.expected_polaris_commit)
    container = file_identity(Path(args.container_image))
    require(
        container["size_bytes"] == args.expected_container_size_bytes
        and container["sha256"] == args.expected_container_sha256,
        "production replay container identity",
    )
    lifecycle = gate0._slurm_lifecycle(args.launch_id)  # noqa: SLF001
    production_eval = gate0.validate_production_reset_source()
    fixture, source_actions = load_actions()
    actions = effective_actions(source_actions)
    tail_contract = frozen_tail_contract(actions)
    boundary, boundary_identity = gate0._load_boundary_helper()  # noqa: SLF001

    source_root = Path(repository["path"])
    expected_core_sources = {
        "src/polaris/config.py": (
            "ea38e87ab20f204929e39454bd9edf6b321d419cb3cebb61c7a6b9487f12373a"
        ),
        "src/polaris/eef_controller_profile.py": (
            "af5c7d73a0b1bd5bf229c1b54f3c271d46fbdaafc7b81b9a1bd2799133420ec1"
        ),
        "src/polaris/eef_controller_repair.py": (
            "3233945b7a70f1c93612fd1dab13fabf6b79591ea17d610282b6650b2d08f567"
        ),
        "src/polaris/robust_differential_ik.py": (
            "8add3b6bc3f33e2797a2c4cab2aa2ebf4c67c2ab07c197dd9a0cd004bfde49dc"
        ),
    }
    production_core_sources = {
        relative: file_identity(source_root / relative)
        for relative in expected_core_sources
    }
    require(
        all(
            production_core_sources[relative]["sha256"] == expected_sha
            for relative, expected_sha in expected_core_sources.items()
        ),
        "production core source identity",
    )

    env_cfg = parse_env_cfg(
        gate0.ENVIRONMENT, device=args.device, num_envs=1, use_fabric=True
    )
    configure_ego_lap_environment_timeout(env_cfg)
    env_cfg.actions = EgoLapEefPoseActionCfg()
    configure_eef_pose_joint_safety(
        env_cfg.scene.robot,
        physx_cfg=env_cfg.sim.physx,
        enable_gripper_velocity_limit=True,
    )
    configured_profile = configure_eef_controller_profile(
        env_cfg,
        profile=CONTROLLER_PROFILE,
    )
    production_arm_class = env_cfg.actions.arm.class_type
    production_gripper_class = env_cfg.actions.finger_joint.class_type
    env_cfg.actions.arm.class_type = make_full_trace_arm_class(
        production_arm_class,
        torch_module=torch,
        bound_helper=bound_joint_position_target,
        release_helper=apply_arm_release_ramp_target,
    )
    tracing_class = make_full_trace_gripper_class(production_gripper_class)
    env_cfg.actions.finger_joint.class_type = tracing_class
    observer_class_contract = {
        "profile": "post_validated_v4_observational_subclasses_v1",
        "production_arm_module": production_arm_class.__module__,
        "production_arm_qualname": production_arm_class.__qualname__,
        "production_gripper_module": production_gripper_class.__module__,
        "production_gripper_qualname": production_gripper_class.__qualname__,
        "runtime_arm_module": env_cfg.actions.arm.class_type.__module__,
        "runtime_arm_qualname": env_cfg.actions.arm.class_type.__qualname__,
        "runtime_gripper_module": tracing_class.__module__,
        "runtime_gripper_qualname": tracing_class.__qualname__,
        "target_setter_call_count": 0,
        "failure_trace_write_count": 0,
        "release_ramp_state_write_count": 0,
        "gripper_target_or_state_write_count": 0,
    }

    robot_usd_path = Path(env_cfg.scene.robot.spawn.usd_path)
    env = gym.make(gate0.ENVIRONMENT, cfg=env_cfg)
    state["env"] = env
    runtime_protocol = validate_ego_lap_runtime_protocol(env)
    assets = gate0._capture_assets(  # noqa: SLF001
        boundary,
        scene_path=Path(env.unwrapped.usd_file),
        robot_usd_path=robot_usd_path,
    )
    _, initial_conditions = load_eval_initial_conditions(
        usd=env.unwrapped.usd_file, rollouts=1
    )
    require(len(initial_conditions) == 1, "FoodBussing IC0 count")
    observation, _ = env.reset(object_positions=initial_conditions[0])
    production_gripper_contract = install_eef_gripper_runtime(
        env, robot_usd_path=robot_usd_path
    )
    runtime_frame = validate_eef_runtime_frame(env, observation)
    begin_eef_safety_episode(env, 0)
    terms = env.unwrapped.action_manager._terms
    require(list(terms) == ["arm", "finger_joint"], "live action order")
    arm_term = terms["arm"]
    finger_term = terms["finger_joint"]
    require(
        type(arm_term) is env_cfg.actions.arm.class_type
        and type(finger_term) is tracing_class,
        "production replay observer class installation",
    )
    production_runtime = capture_production_v4_runtime_contract(
        env=env,
        arm_term=arm_term,
        torch_module=torch,
        configured_profile=configured_profile,
    )

    frames = [model_view_frame(observation)]
    source_actions_completed = 0
    tail_policy_steps_completed = 0
    numerical_failure: dict[str, Any] | None = None
    controller_failure_evidence: dict[str, Any] | None = None
    production_gripper_trace_finalized = False

    def execute_policy_step(
        *, global_policy_step: int, values: Sequence[float], segment: str
    ) -> tuple[Any | None, dict[str, Any] | None]:
        nonlocal controller_failure_evidence
        nonlocal production_gripper_trace_finalized
        state["policy_step"] = global_policy_step
        state["execution_segment"] = segment
        finger_term.begin_eef_policy_step(
            episode_index=0,
            policy_step=global_policy_step,
        )
        finger_term.begin_gate0_policy_step(global_policy_step)
        action = torch.tensor(values, dtype=torch.float32, device=env.device).reshape(
            1, 8
        )
        try:
            next_observation, _, terminated, truncated, _ = env.step(
                action, expensive=True
            )
        except DifferentialIKNumericalError as error:
            failure = gate0._exception_evidence(error)  # noqa: SLF001
            state["numerical_failure"] = failure
            finger_term.finalize_gate0_failure()
            finger_term.finalize_eef_rollout_trace(numerical_failure=True)
            production_gripper_trace_finalized = True
            failure_entries = finger_term.full_trace()
            controller_failure_evidence = {
                "profile": "production_v4_core_replay_failure_evidence_v1",
                "failure_exception": failure,
                "full_trace_entry_count_at_failure": len(failure_entries),
            }
            state["controller_failure_evidence"] = controller_failure_evidence
            frames.append(terminal_model_view_frame(env))
            return None, failure
        require(
            not bool(terminated[0]),
            f"unexpected termination at global step {global_policy_step}",
        )
        require(
            not bool(truncated[0]),
            f"unexpected truncation at global step {global_policy_step}",
        )
        record_eef_gripper_post_policy_step(env)
        frames.append(model_view_frame(next_observation))
        return next_observation, None

    for step, values in enumerate(actions):
        next_observation, numerical_failure = execute_policy_step(
            global_policy_step=step,
            values=values,
            segment="source_actions",
        )
        if numerical_failure is not None:
            break
        observation = next_observation
        source_actions_completed += 1

    if numerical_failure is None:
        require(
            source_actions_completed == ACTION_COUNT,
            "tail cannot start before all source actions complete",
        )
        final_values = tail_contract["action_values_float32"]
        for tail_step in range(TAIL_POLICY_STEPS):
            next_observation, numerical_failure = execute_policy_step(
                global_policy_step=ACTION_COUNT + tail_step,
                values=final_values,
                segment="frozen_final_command_tail",
            )
            if numerical_failure is not None:
                break
            observation = next_observation
            tail_policy_steps_completed += 1

    if not production_gripper_trace_finalized:
        finger_term.finalize_gate0_failure()
        finger_term.finalize_eef_rollout_trace(
            numerical_failure=numerical_failure is not None
        )
        production_gripper_trace_finalized = True
    entries = finger_term.full_trace()
    outcome = classify_outcome(
        numerical_failure,
        source_actions_completed,
        tail_policy_steps_completed,
        len(entries),
    )
    require(
        outcome["replay_completed"] is True
        and numerical_failure is None
        and controller_failure_evidence is None,
        "production v4 replay did not complete without controller failure",
    )
    all_six_gripper_trace = validate_eef_all_six_gripper_trace(
        finger_term.eef_all_six_gripper_trace(),
        episode_index=0,
        episode_length=ACTION_COUNT + TAIL_POLICY_STEPS,
        numerical_failure=False,
        expected_apply_calls=TOTAL_APPLY_COUNT,
    )
    safety = eef_episode_safety_report(
        env,
        0,
        expected_gripper_target_slew_profile=configured_profile.target_slew_profile,
        expected_eef_controller_profile=CONTROLLER_PROFILE,
    )
    controller_report = capture_eef_controller_repair_candidate_report(
        env,
        safety,
        expected_profile=CONTROLLER_PROFILE,
        expected_target_slew_profile=configured_profile.target_slew_profile,
    )
    observer_reporter = getattr(
        arm_term, "production_core_ramp_observation_report", None
    )
    require(callable(observer_reporter), "production core ramp observer reporter")
    ramp_observation = observer_reporter()
    trace_cadence = validate_trace_cadence(
        entries,
        outcome=outcome,
        ramp_observation=ramp_observation,
        controller_report=controller_report,
    )
    video = publish_video(args.output_video, frames)
    state["policy_step"] = None
    state["execution_segment"] = None
    return {
        "schema_version": 1,
        "profile": PROFILE,
        "passed": True,
        "controller_replay_only": True,
        "variant": VARIANT,
        "controller_profile": CONTROLLER_PROFILE,
        "repository": repository,
        "production_core_sources": production_core_sources,
        "container_image": container,
        "lifecycle": lifecycle,
        "production_eval": production_eval,
        "fixture": fixture,
        "source_trace_polaris_commit": SOURCE_TRACE_POLARIS_COMMIT,
        "source_trace_sha256": fixture_contract.TRACE_SHA256,
        "source_action_float32_sha256": ACTION_ENCODING["uncompressed_sha256"],
        "boundary_helper": boundary_identity,
        "assets": assets,
        "runtime_protocol": runtime_protocol,
        "runtime_frame": runtime_frame,
        "observer_class_contract": observer_class_contract,
        "production_runtime": production_runtime,
        "production_gripper_contract": production_gripper_contract,
        "production_safety": safety,
        "production_controller_report": controller_report,
        "production_all_six_gripper_trace": all_six_gripper_trace,
        "production_core_ramp_observation": ramp_observation,
        "action_count": ACTION_COUNT,
        "actions_completed": source_actions_completed,
        "tail_contract": tail_contract,
        "tail_policy_steps_completed": tail_policy_steps_completed,
        "tail_physics_substeps_completed": outcome["tail_physics_substeps_completed"],
        "total_apply_count": len(entries),
        "numerical_failure": numerical_failure,
        "controller_failure_evidence": controller_failure_evidence,
        "outcome": outcome,
        "full_substep_trace_profile": (
            "all13_after_arm_before_gripper_post_setters_post_physics_"
            "before_next_arm_v3"
        ),
        "full_substep_trace_cadence": trace_cadence,
        "full_substep_trace": entries,
        "full_substep_summary": summarize_trace(entries),
        "video": video,
    }


def parse_args() -> tuple[argparse.Namespace, Any]:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-polaris-commit", required=True)
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--expected-container-size-bytes", type=int, required=True)
    parser.add_argument("--expected-container-sha256", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    require(
        len(args.expected_polaris_commit) == 40
        and all(
            character in "0123456789abcdef"
            for character in args.expected_polaris_commit
        ),
        "expected production replay commit argument",
    )
    args.enable_cameras = True
    args.headless = True
    return args, AppLauncher


def main() -> int:
    args, launcher_type = parse_args()
    require(
        args.output_json.resolve() != args.output_video.resolve(), "output collision"
    )
    require(not args.output_json.exists(), "JSON output already exists")
    require(not args.output_video.exists(), "video output already exists")
    state: dict[str, Any] = {
        "env": None,
        "policy_step": None,
        "execution_segment": None,
        "numerical_failure": None,
        "controller_failure_evidence": None,
        "completed_payload": None,
    }
    simulation_app = None
    try:
        launcher = launcher_type(args)
        simulation_app = launcher.app
        payload = run_live(args, state)
        state["completed_payload"] = payload
        env = state.get("env")
        if env is not None:
            env.close()
            state["env"] = None
        payload["runtime_close"] = {
            "environment_close_completed": True,
            "simulation_app_close_state": (
                "pending_terminal_call_after_raw_publication_v2"
            ),
            "publication_timing": "after_environment_before_simulation_app_close_v2",
            "completion_evidence": (
                "post_kit_validator_requires_zero_simulator_srun_exit_v1"
            ),
        }
        identity = gate0._atomic_write_immutable(args.output_json, payload)  # noqa: SLF001
        print(
            f"POLARIS_PRODUCTION_V4_CORE_REPLAY={identity['path']};"
            f"size={identity['size_bytes']};sha256={identity['sha256']};"
            f"video={payload['video']['path']};video_sha256={payload['video']['sha256']}",
            flush=True,
        )
        # Isaac's SimulationApp.close() is terminal on the pinned Kit runtime:
        # no artifact publication may be placed after this call. The outer
        # wrapper only invokes the post-Kit validator after this process exits
        # zero, and the validator remains responsible for the closed outcome.
        simulation_app.close()
        simulation_app = None
        return 0
    except BaseException as error:
        traceback.print_exception(
            type(error), error, error.__traceback__, file=sys.stderr
        )
        failure_payload = {
            "schema_version": 1,
            "profile": PROFILE,
            "passed": False,
            "controller_replay_only": True,
            "variant": VARIANT,
            "policy_step": state.get("policy_step"),
            "execution_segment": state.get("execution_segment"),
            "numerical_failure": state.get("numerical_failure"),
            "controller_failure_evidence": state.get("controller_failure_evidence"),
            "failure": gate0._exception_evidence(error),  # noqa: SLF001
            "completed_replay_payload": state.get("completed_payload"),
        }
        try:
            if not args.output_json.exists():
                gate0._atomic_write_immutable(args.output_json, failure_payload)  # noqa: SLF001
        except BaseException as persistence_error:
            traceback.print_exception(
                type(persistence_error),
                persistence_error,
                persistence_error.__traceback__,
                file=sys.stderr,
            )
        if simulation_app is not None:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
