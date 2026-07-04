#!/usr/bin/env python3
"""Replay the complete reasoning canary under isolated controller ablations.

This is a diagnostic, not a production evaluation profile.  It replays all
294 byte-pinned absolute actions from job 1098523 and records every completed
120 Hz physics substep.  Four variants differ by exactly one intervention:

* ``baseline``: the exact commit-0611d38 controller candidate;
* ``force_open``: identical requested arm poses with every gripper endpoint
  forced open, disabling both gripper mechanics and close-triggered interlock;
* ``follower_default_limit``: exact actions with passive mimic followers left
  at the source/default 174.5329 rad/s max velocity after the production
  installer has run;
* ``hold_close_anchor``: exact actions with the fixed close anchor extended by
  16 substeps (102 total), so the original step-293/substep-2 failure remains
  inside the hold while all other behavior is unchanged.

All interventions are applied after the normal production runtime installer
and are explicitly recorded.  None is accepted as a production contract by
this script.
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
import smoke_eef_pose_canary_controller_candidate as candidate
import smoke_eef_pose_canary_trace_replay as gate0


PROFILE = "reasoning_43075_fulltrace_gripper_release_ablation_v1"
PRODUCTION_BASE_COMMIT = "0611d384f5f26ef9bd8ff114be273e875c3fe719"
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
VARIANTS = (
    "baseline",
    "force_open",
    "follower_default_limit",
    "hold_close_anchor",
)
ACTION_COUNT = 294
DECIMATION = 8
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
FOLLOWER_DEFAULT_LIMIT = 174.53292846679688
HOLD_CLOSE_ANCHOR_SUBSTEPS = 102


class FullTraceAblationError(ValueError):
    """A diagnostic input, runtime identity, or expected outcome drifted."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FullTraceAblationError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
        and payload.get("polaris_commit") == PRODUCTION_BASE_COMMIT,
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
        raise FullTraceAblationError("action codec failure") from error
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
    actions: Sequence[Sequence[float]], variant: str
) -> list[list[float]]:
    require(variant in VARIANTS, "ablation variant")
    result = [list(action) for action in actions]
    if variant == "force_open":
        for action in result:
            action[7] = 0.0
    require(len(result) == ACTION_COUNT, "effective action count")
    return result


def vector(tensor: Any, *, length: int, field: str) -> list[float]:
    values = [float(value) for value in tensor.detach().cpu().flatten().tolist()]
    require(
        len(values) == length and all(math.isfinite(value) for value in values),
        f"{field} finite vector",
    )
    return values


def interlock_snapshot(arm_term: Any) -> dict[str, Any]:
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
    }


def make_full_trace_arm_class(base_class: type) -> type:
    """Finalize causal post-physics state before the next arm command."""

    class FullTraceArmAction(base_class):
        def __init__(self, cfg: Any, env: Any) -> None:
            self._full_trace_env = env
            super().__init__(cfg, env)

        def apply_actions(self) -> None:
            terms = getattr(self._full_trace_env.action_manager, "_terms", {})
            finger_term = terms.get("finger_joint")
            finalizer = getattr(
                finger_term, "finalize_physics_post_before_next_arm", None
            )
            if callable(finalizer):
                finalizer()
            super().apply_actions()

    FullTraceArmAction.__name__ = base_class.__name__
    FullTraceArmAction.__qualname__ = base_class.__qualname__
    return FullTraceArmAction


def make_full_trace_gripper_class(base_class: type) -> type:
    """Extend the gripper tracer with command-synchronized all-joint state."""

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

    # The production installer intentionally verifies this exact class name.
    FullTraceEefBinaryJointPositionTargetSlewAction.__name__ = (
        "EefBinaryJointPositionTargetSlewAction"
    )
    FullTraceEefBinaryJointPositionTargetSlewAction.__qualname__ = (
        "EefBinaryJointPositionTargetSlewAction"
    )
    return FullTraceEefBinaryJointPositionTargetSlewAction


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
    variant: str,
    outcome: Mapping[str, Any],
) -> dict[str, Any]:
    for apply_index, entry in enumerate(entries):
        require(
            entry.get("apply_index") == apply_index
            and entry.get("policy_step") == apply_index // DECIMATION
            and entry.get("physics_substep") == apply_index % DECIMATION,
            f"full-trace cadence drift at apply {apply_index}",
        )
        require(
            isinstance(entry.get("pre"), dict)
            and isinstance(entry.get("command_after_setters"), dict)
            and isinstance(entry.get("post"), dict),
            f"full-trace phase omission at apply {apply_index}",
        )
    classification = outcome.get("classification")
    completed = outcome.get("controller_completed_actions")
    require(
        type(completed) is int and 0 <= completed <= ACTION_COUNT, "completed count"
    )
    if classification == "baseline_exact_source_abort_reproduced":
        expected_count = 293 * DECIMATION + 2
        require(len(entries) == expected_count, "baseline full-trace count")
    elif classification == "ablation_completed_all_recorded_actions":
        expected_count = ACTION_COUNT * DECIMATION
        require(len(entries) == expected_count, "completed ablation full-trace count")
    else:
        lower = completed * DECIMATION
        parsed_failure = outcome.get("parsed_numerical_failure")
        if isinstance(parsed_failure, dict):
            require(
                parsed_failure.get("policy_step") == completed
                and type(parsed_failure.get("physics_substep")) is int,
                "parsed failure/completed-step binding",
            )
            expected_count = lower + parsed_failure["physics_substep"]
            require(
                len(entries) == expected_count,
                "parsed failed-ablation exact full-trace count",
            )
        else:
            require(
                lower <= len(entries) <= lower + DECIMATION - 1,
                "unparsed failed-ablation full-trace count/cadence",
            )
            expected_count = None
    return {
        "profile": "contiguous_policy_step_physics_substep_phases_v1",
        "variant": variant,
        "entry_count": len(entries),
        "expected_entry_count": expected_count,
        "decimation": DECIMATION,
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
    }


def tensor_evidence(tensor: Any) -> dict[str, Any]:
    return {
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "shape": list(tensor.shape),
        "values": [float(value) for value in tensor.detach().cpu().flatten().tolist()],
    }


def apply_variant_intervention(
    *, variant: str, env: Any, arm_term: Any, torch_module: Any
) -> dict[str, Any]:
    robot = env.unwrapped.scene["robot"]
    getter = robot.root_physx_view.get_dof_max_velocities
    before_tensor = getter().clone()
    require(tuple(before_tensor.shape) == (1, 13), "live PhysX velocity-limit shape")
    setter_call_count = 0
    if variant == "follower_default_limit":
        replacement = before_tensor.clone()
        replacement[:, FOLLOWER_INDICES] = FOLLOWER_DEFAULT_LIMIT
        indices = torch_module.arange(
            replacement.shape[0], dtype=torch_module.int32, device=replacement.device
        )
        robot.root_physx_view.set_dof_max_velocities(replacement, indices)
        setter_call_count = 1
    if variant == "hold_close_anchor":
        arm_term._gripper_close_arm_interlock_configured_substeps = (  # noqa: SLF001
            HOLD_CLOSE_ANCHOR_SUBSTEPS
        )
    after_tensor = getter().clone()
    expected_after = before_tensor.clone()
    if variant == "follower_default_limit":
        expected_after[:, FOLLOWER_INDICES] = FOLLOWER_DEFAULT_LIMIT
    require(
        bool(torch_module.equal(after_tensor, expected_after)),
        "ablation PhysX velocity-limit intervention drift",
    )
    configured = int(arm_term._gripper_close_arm_interlock_configured_substeps)  # noqa: SLF001
    expected_configured = (
        HOLD_CLOSE_ANCHOR_SUBSTEPS
        if variant == "hold_close_anchor"
        else candidate.CANDIDATE_CLOSE_INTERLOCK_SUBSTEPS
    )
    require(configured == expected_configured, "ablation interlock intervention drift")
    return {
        "profile": "post_production_installer_single_variable_diagnostic_only_v1",
        "variant": variant,
        "production_contract_acceptance": False,
        "force_gripper_open": variant == "force_open",
        "passive_follower_velocity_limit_setter_call_count": setter_call_count,
        "velocity_limits_before_intervention": tensor_evidence(before_tensor),
        "velocity_limits_after_intervention": tensor_evidence(after_tensor),
        "configured_close_anchor_substeps": configured,
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
    variant: str, failure: Mapping[str, Any] | None, completed: int
) -> dict[str, Any]:
    if variant == "baseline":
        if failure is None:
            return {
                "diagnostic_completed": False,
                "classification": "baseline_failed_to_reproduce_source_abort",
                "controller_completed_actions": completed,
                "parsed_numerical_failure": None,
                "numerical_failure_parse_error": None,
            }
        try:
            parsed = gate0.parse_failure_exception(str(failure.get("message", "")))
        except gate0.Gate0ReplayValidationError as error:
            return {
                "diagnostic_completed": False,
                "classification": "baseline_abort_unparsed",
                "controller_completed_actions": completed,
                "parsed_numerical_failure": None,
                "numerical_failure_parse_error": str(error),
            }
        exact = parsed == fixture_contract.EXPECTED_FAILURE and completed == 293
        return {
            "diagnostic_completed": exact,
            "classification": (
                "baseline_exact_source_abort_reproduced"
                if exact
                else "baseline_abort_identity_drift"
            ),
            "controller_completed_actions": completed,
            "parsed_numerical_failure": parsed,
            "numerical_failure_parse_error": None,
        }
    parsed: dict[str, Any] | None = None
    parse_failure: str | None = None
    if failure is not None:
        try:
            parsed = gate0.parse_failure_exception(str(failure.get("message", "")))
        except gate0.Gate0ReplayValidationError as error:
            parse_failure = str(error)
    return {
        "diagnostic_completed": True,
        "classification": (
            "ablation_completed_all_recorded_actions"
            if failure is None and completed == ACTION_COUNT
            else "ablation_numerical_failure_observed"
        ),
        "controller_completed_actions": completed,
        "parsed_numerical_failure": parsed,
        "numerical_failure_parse_error": parse_failure,
    }


def diagnostic_repository_provenance(expected_commit: str) -> dict[str, Any]:
    import subprocess

    repository = gate0._repository_provenance(expected_commit)  # noqa: SLF001
    parent = subprocess.run(
        ["git", "rev-parse", "HEAD^"],
        cwd=repository["path"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(parent == PRODUCTION_BASE_COMMIT, "diagnostic production-base parent drift")
    return {
        **repository,
        "production_base_commit": PRODUCTION_BASE_COMMIT,
        "production_base_relation": "exact_first_parent_v1",
    }


def run_live(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    import gymnasium as gym  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: PLC0415

    import polaris.environments  # noqa: F401, PLC0415
    from polaris.eef_gripper_runtime import (  # noqa: PLC0415
        configure_eef_gripper_mimic_compliance_spawn_overlay,
        install_eef_gripper_runtime,
        record_eef_gripper_post_policy_step,
    )
    from polaris.eef_runtime_contract import (  # noqa: PLC0415
        begin_eef_safety_episode,
        configure_ego_lap_environment_timeout,
        validate_eef_runtime_frame,
        validate_ego_lap_runtime_protocol,
    )
    from polaris.environments.droid_cfg import (  # noqa: PLC0415
        EefBinaryJointPositionTargetSlewAction,
        EgoLapEefPoseActionCfg,
    )
    from polaris.environments.robot_cfg import configure_eef_pose_joint_safety  # noqa: PLC0415
    from polaris.robust_differential_ik import DifferentialIKNumericalError  # noqa: PLC0415
    from polaris.utils import load_eval_initial_conditions  # noqa: PLC0415

    repository = diagnostic_repository_provenance(args.expected_polaris_commit)
    container = candidate.validate_container_argument(
        args.container_image,
        size_bytes=args.expected_container_size_bytes,
        sha256=args.expected_container_sha256,
    )
    lifecycle = gate0._slurm_lifecycle(args.launch_id)  # noqa: SLF001
    production_eval = gate0.validate_production_reset_source()
    fixture, source_actions = load_actions()
    actions = effective_actions(source_actions, args.variant)
    boundary, boundary_identity = gate0._load_boundary_helper()  # noqa: SLF001

    env_cfg = parse_env_cfg(
        gate0.ENVIRONMENT, device=args.device, num_envs=1, use_fabric=True
    )
    configure_ego_lap_environment_timeout(env_cfg)
    env_cfg.actions = EgoLapEefPoseActionCfg()
    env_cfg.actions.arm.enable_failure_substep_trace = True
    env_cfg.actions.arm.enable_wrist_energy_brake = False
    env_cfg.actions.arm.enable_arm_slew_headroom = True
    env_cfg.actions.arm.enable_gripper_close_arm_interlock = True
    env_cfg.actions.arm.class_type = make_full_trace_arm_class(
        env_cfg.actions.arm.class_type
    )
    env_cfg.actions.finger_joint.enable_target_slew_rate_0p25_candidate = True
    tracing_class = make_full_trace_gripper_class(
        EefBinaryJointPositionTargetSlewAction
    )
    env_cfg.actions.finger_joint.class_type = tracing_class
    configure_eef_pose_joint_safety(
        env_cfg.scene.robot,
        physx_cfg=env_cfg.sim.physx,
        enable_gripper_velocity_limit=True,
    )
    configure_eef_gripper_mimic_compliance_spawn_overlay(
        env_cfg.scene.robot.spawn,
        target_slew_profile=candidate.CANDIDATE_TARGET_SLEW_PROFILE,
    )
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
    require(type(finger_term) is tracing_class, "full-trace class installation")
    intervention = apply_variant_intervention(
        variant=args.variant, env=env, arm_term=arm_term, torch_module=torch
    )

    frames = [model_view_frame(observation)]
    completed = 0
    numerical_failure: dict[str, Any] | None = None
    for step, values in enumerate(actions):
        state["policy_step"] = step
        finger_term.begin_gate0_policy_step(step)
        action = torch.tensor(values, dtype=torch.float32, device=env.device).reshape(
            1, 8
        )
        try:
            observation, _, terminated, truncated, _ = env.step(action, expensive=True)
        except DifferentialIKNumericalError as error:
            numerical_failure = gate0._exception_evidence(error)  # noqa: SLF001
            state["numerical_failure"] = numerical_failure
            finger_term.finalize_gate0_failure()
            frames.append(terminal_model_view_frame(env))
            break
        require(not bool(terminated[0]), f"unexpected termination at step {step}")
        require(not bool(truncated[0]), f"unexpected truncation at step {step}")
        completed += 1
        record_eef_gripper_post_policy_step(env)
        frames.append(model_view_frame(observation))
    if numerical_failure is None:
        finger_term.finalize_gate0_failure()
    entries = finger_term.full_trace()
    outcome = classify_outcome(args.variant, numerical_failure, completed)
    trace_cadence = validate_trace_cadence(
        entries, variant=args.variant, outcome=outcome
    )
    video = publish_video(args.output_video, frames)
    state["policy_step"] = None
    return {
        "schema_version": 1,
        "profile": PROFILE,
        "passed": outcome["diagnostic_completed"],
        "diagnostic_only": True,
        "variant": args.variant,
        "repository": repository,
        "container_image": container,
        "lifecycle": lifecycle,
        "production_eval": production_eval,
        "fixture": fixture,
        "source_trace_sha256": fixture_contract.TRACE_SHA256,
        "source_action_float32_sha256": ACTION_ENCODING["uncompressed_sha256"],
        "boundary_helper": boundary_identity,
        "assets": assets,
        "runtime_protocol": runtime_protocol,
        "runtime_frame": runtime_frame,
        "production_gripper_contract_before_ablation": production_gripper_contract,
        "intervention": intervention,
        "action_count": ACTION_COUNT,
        "actions_completed": completed,
        "numerical_failure": numerical_failure,
        "outcome": outcome,
        "full_substep_trace_profile": (
            "all13_after_arm_before_gripper_post_setters_post_physics_"
            "before_next_arm_v2"
        ),
        "full_substep_trace_cadence": trace_cadence,
        "full_substep_trace": entries,
        "full_substep_summary": summarize_trace(entries),
        "video": video,
    }


def parse_args() -> tuple[argparse.Namespace, Any]:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
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
        "expected diagnostic commit argument",
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
        "numerical_failure": None,
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
            f"POLARIS_FULLTRACE_ABLATION={identity['path']};"
            f"size={identity['size_bytes']};sha256={identity['sha256']};"
            f"video={payload['video']['path']};video_sha256={payload['video']['sha256']}",
            flush=True,
        )
        # Isaac's SimulationApp.close() is terminal on the pinned Kit runtime:
        # no artifact publication may be placed after this call. The outer
        # wrapper only invokes the post-Kit validator after this process exits
        # zero, and the validator remains responsible for the baseline match.
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
            "diagnostic_only": True,
            "variant": getattr(args, "variant", None),
            "policy_step": state.get("policy_step"),
            "numerical_failure": state.get("numerical_failure"),
            "failure": gate0._exception_evidence(error),  # noqa: SLF001
            "completed_diagnostic_payload": state.get("completed_payload"),
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
