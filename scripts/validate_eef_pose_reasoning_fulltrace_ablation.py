#!/usr/bin/env python3
"""Validate and finalize one post-Kit cap/release full-trace follow-up."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
from typing import Any

import build_reasoning_fulltrace_replay_fixture as fixture_contract
import smoke_eef_pose_reasoning_fulltrace_ablation as diagnostic


PROFILE = "reasoning_43075_fulltrace_cap_release_followup_post_kit_v1"
RESULT_FIELDS = {
    "schema_version",
    "profile",
    "passed",
    "diagnostic_only",
    "variant",
    "repository",
    "container_image",
    "lifecycle",
    "production_eval",
    "fixture",
    "source_trace_sha256",
    "source_action_float32_sha256",
    "boundary_helper",
    "assets",
    "runtime_protocol",
    "runtime_frame",
    "production_gripper_contract_before_ablation",
    "intervention",
    "action_count",
    "actions_completed",
    "tail_contract",
    "tail_policy_steps_completed",
    "tail_physics_substeps_completed",
    "numerical_failure",
    "controller_failure_evidence",
    "outcome",
    "full_substep_trace_profile",
    "full_substep_trace_cadence",
    "full_substep_trace",
    "full_substep_summary",
    "video",
    "runtime_close",
}
EXPECTED_LIMITS_WITH_FOLLOWER_CAP = [
    2.174999952316284,
    2.174999952316284,
    2.174999952316284,
    2.174999952316284,
    2.609999895095825,
    2.609999895095825,
    2.609999895095825,
    5.0,
    5.0,
    5.0,
    5.0,
    5.0,
    5.0,
]


class ValidationError(ValueError):
    """The post-Kit diagnostic artifact is incomplete or inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reject_constant(token: str) -> None:
    raise ValidationError(f"non-standard JSON constant {token!r}")


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def strict_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = file_identity(path)
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"invalid strict JSON {path}: {error}") from error
    require(isinstance(payload, dict), f"JSON object required: {path}")
    return payload, identity


def file_identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    require(path.is_file() and not path.is_symlink(), f"missing/linked file {path}")
    metadata = path.stat()
    data = path.read_bytes()
    return {
        "path": str(path),
        "size_bytes": len(data),
        "sha256": sha256(data),
        "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "nlink": metadata.st_nlink,
    }


def validate_tensor(value: Any, *, field: str) -> list[float]:
    require(
        isinstance(value, dict)
        and set(value) == {"dtype", "device", "shape", "values"}
        and value.get("dtype") == "torch.float32"
        and value.get("device") == "cpu"
        and value.get("shape") == [1, 13],
        f"{field} tensor metadata",
    )
    values = value.get("values")
    require(
        isinstance(values, list)
        and len(values) == 13
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in values
        ),
        f"{field} tensor values",
    )
    return [float(item) for item in values]


def float32_equal(left: float, right: float) -> bool:
    import struct

    return struct.pack("<f", left) == struct.pack("<f", right)


def expected_release_ramp_static(*, enabled: bool) -> dict[str, Any]:
    return {
        "profile": "arm_post_interlock_linear_slew_cap_release_ramp16_v2",
        "enabled": enabled,
        "scope": "arm_position_target_slew_cap_after_interlock_release_v2",
        "substeps": diagnostic.RELEASE_RAMP_SUBSTEPS,
        "fraction_profile": ("inclusive_linear_float32_0_over_15_to_15_over_15_v2"),
        "fractions_float32": [
            diagnostic.float32(item) for item in diagnostic.RELEASE_RAMP_FRACTIONS
        ],
        "nominal_arm_slew_ratio": diagnostic.NOMINAL_ARM_SLEW_RATIO,
        "effective_physical_limit_ratios": [
            diagnostic.float32_multiply(diagnostic.NOMINAL_ARM_SLEW_RATIO, fraction)
            for fraction in diagnostic.RELEASE_RAMP_FRACTIONS
        ],
        "arm_joint_ids": diagnostic.ARM_JOINT_IDS,
        "arm_joint_names": diagnostic.ARM_JOINT_NAMES,
        "nominal_max_delta_joint_pos_rad": diagnostic.ARM_NOMINAL_MAX_DELTA_RAD,
        "formula_profile": (
            "endpoint_exact_else_float32_clamp_nominal_delta_by_scaled_slew_v1"
        ),
        "transaction_profile": (
            "diagnostic_post_super_overlay_commit_after_setter_and_trace_v1"
        ),
        "reset_profile": "clear_all_diagnostic_ramp_state_after_base_reset_v1",
        "target_setter": "Articulation.set_joint_position_target_arm_ids_v1",
        "gripper_target_or_state_write_count": 0,
    }


def validate_release_ramp_runtime(value: Any, *, enabled: bool) -> dict[str, Any]:
    fields = {
        "profile",
        "enabled",
        "release_observed_count",
        "ramp_started_count",
        "ramp_completed_count",
        "ramp_cancelled_by_reactivation_count",
        "ramp_target_apply_count",
        "ramp_limited_target_apply_count",
        "applied_indices",
        "pending_at_report",
        "next_index_at_report",
        "max_abs_nominal_to_ramped_target_change_rad",
        "overlay_entries",
        "gripper_target_or_state_write_count",
    }
    require(
        isinstance(value, dict)
        and set(value) == fields
        and value.get("profile")
        == "arm_post_interlock_linear_slew_cap_release_ramp16_runtime_v2"
        and value.get("enabled") is enabled
        and value.get("gripper_target_or_state_write_count") == 0,
        "release-ramp runtime schema/profile",
    )
    count_fields = fields - {
        "profile",
        "enabled",
        "applied_indices",
        "pending_at_report",
        "next_index_at_report",
        "max_abs_nominal_to_ramped_target_change_rad",
        "overlay_entries",
        "gripper_target_or_state_write_count",
    }
    require(
        all(type(value[field]) is int and value[field] >= 0 for field in count_fields),
        "release-ramp runtime counters",
    )
    applied = value["applied_indices"]
    require(
        isinstance(applied, list)
        and all(
            type(index) is int and 0 <= index < diagnostic.RELEASE_RAMP_SUBSTEPS
            for index in applied
        )
        and value["ramp_target_apply_count"] == len(applied)
        and value["ramp_started_count"] == applied.count(0)
        and value["ramp_completed_count"]
        == applied.count(diagnostic.RELEASE_RAMP_SUBSTEPS - 1)
        and value["ramp_limited_target_apply_count"] <= len(applied),
        "release-ramp applied-index/counter binding",
    )
    for previous, current in zip(applied, applied[1:]):
        require(
            current == previous + 1 or current == 0,
            "release-ramp applied-index sequence",
        )
    pending = value["pending_at_report"]
    next_index = value["next_index_at_report"]
    require(
        type(pending) is bool
        and (
            next_index is None
            or (
                type(next_index) is int
                and 0 <= next_index < diagnostic.RELEASE_RAMP_SUBSTEPS
            )
        )
        and (not pending or next_index == 0),
        "release-ramp terminal state",
    )
    maximum = value["max_abs_nominal_to_ramped_target_change_rad"]
    overlays = value["overlay_entries"]
    require(
        isinstance(maximum, (int, float))
        and not isinstance(maximum, bool)
        and math.isfinite(float(maximum))
        and float(maximum) >= 0.0,
        "release-ramp maximum target change",
    )
    require(
        isinstance(overlays, list)
        and len(overlays) == value["ramp_target_apply_count"],
        "release-ramp overlay count",
    )
    if not enabled:
        require(
            all(value[field] == 0 for field in count_fields)
            and applied == []
            and pending is False
            and next_index is None
            and float(maximum) == 0.0
            and overlays == [],
            "disabled release-ramp retained runtime state",
        )
    else:
        require(
            value["release_observed_count"] >= value["ramp_started_count"]
            and value["ramp_started_count"] >= value["ramp_completed_count"],
            "enabled release-ramp lifecycle counts",
        )
    return dict(value)


def validate_intervention(value: Any, *, variant: str) -> dict[str, Any]:
    require(
        isinstance(value, dict)
        and set(value)
        == {
            "profile",
            "variant",
            "production_contract_acceptance",
            "passive_follower_velocity_cap_rad_s",
            "passive_follower_velocity_limit_setter_call_count",
            "velocity_limits_before_intervention",
            "velocity_limits_after_intervention",
            "configured_close_anchor_substeps",
            "arm_release_mode",
            "arm_release_ramp_static",
            "arm_release_ramp_runtime",
        }
        and value.get("profile") == "post_production_installer_cap_release_followup_v1"
        and value.get("variant") == variant
        and value.get("production_contract_acceptance") is False,
        "intervention schema/profile",
    )
    before = validate_tensor(
        value["velocity_limits_before_intervention"], field="pre-intervention limits"
    )
    after = validate_tensor(
        value["velocity_limits_after_intervention"], field="post-intervention limits"
    )
    require(
        all(
            float32_equal(actual, expected)
            for actual, expected in zip(
                before, EXPECTED_LIMITS_WITH_FOLLOWER_CAP, strict=True
            )
        ),
        "production follower-cap input",
    )
    expected_after = list(EXPECTED_LIMITS_WITH_FOLLOWER_CAP)
    requested_cap = diagnostic.FOLLOWER_CAP_BY_VARIANT[variant]
    expected_after[8:] = [requested_cap] * 5
    require(
        all(
            float32_equal(actual, expected)
            for actual, expected in zip(after, expected_after, strict=True)
        ),
        "variant follower-limit output",
    )
    ramp_enabled = variant == diagnostic.RELEASE_RAMP_VARIANT
    recorded_cap = value.get("passive_follower_velocity_cap_rad_s")
    require(
        isinstance(recorded_cap, (int, float))
        and not isinstance(recorded_cap, bool)
        and math.isfinite(float(recorded_cap))
        and float32_equal(float(recorded_cap), requested_cap)
        and value.get("passive_follower_velocity_limit_setter_call_count")
        == int(requested_cap != diagnostic.PRODUCTION_FOLLOWER_LIMIT)
        and value.get("configured_close_anchor_substeps") == 86
        and value.get("arm_release_mode")
        == ("linear_slew_cap_ramp16" if ramp_enabled else "abrupt")
        and value.get("arm_release_ramp_static")
        == expected_release_ramp_static(enabled=ramp_enabled),
        "pre-registered intervention identity",
    )
    validate_release_ramp_runtime(
        value.get("arm_release_ramp_runtime"), enabled=ramp_enabled
    )
    return dict(value)


def validate_source_scene_controller_provenance(payload: dict[str, Any]) -> None:
    """Close the immutable fixture, scene, and production controller inputs."""

    require(
        payload.get("fixture") == diagnostic.file_identity(diagnostic.FIXTURE_PATH),
        "fixture live/recorded identity",
    )
    production = payload.get("production_eval")
    gate0 = diagnostic.gate0
    require(
        isinstance(production, dict)
        and production.get("size_bytes") == gate0.EXPECTED_PRODUCTION_EVAL_SIZE_BYTES
        and production.get("sha256") == gate0.EXPECTED_PRODUCTION_EVAL_SHA256
        and production.get("reset_profile") == gate0.PRODUCTION_RESET_PROFILE
        and production.get("reset_call") == gate0.PRODUCTION_RESET_CALL
        and production.get("environment_seed") is None
        and production.get("initial_condition_index") == 0
        and production.get("step_render_profile")
        == gate0.PRODUCTION_STEP_RENDER_PROFILE
        and production.get("effective_step_expensive") is True
        and production.get("policy_config_source", {}).get("sha256")
        == gate0.EXPECTED_PRODUCTION_POLICY_CONFIG_SHA256
        and production.get("lap_client_source", {}).get("sha256")
        == gate0.EXPECTED_PRODUCTION_LAP_CLIENT_SHA256,
        "production reset/render/client provenance",
    )
    boundary = payload.get("boundary_helper")
    require(
        isinstance(boundary, dict)
        and boundary.get("size_bytes") == gate0.EXPECTED_BOUNDARY_HELPER_SIZE_BYTES
        and boundary.get("sha256") == gate0.EXPECTED_BOUNDARY_HELPER_SHA256,
        "boundary helper provenance",
    )
    assets = payload.get("assets")
    require(
        isinstance(assets, dict)
        and assets.get("contract") == gate0.EXPECTED_ASSET_CONTRACT
        and assets.get("robot_usd", {}).get("sha256") == gate0.EXPECTED_ROBOT_USD_SHA256
        and assets.get("scene", {}).get("scene", {}).get("sha256")
        == gate0.EXPECTED_ASSET_CONTRACT["scene_sha256"]
        and assets.get("scene", {}).get("initial_conditions", {}).get("sha256")
        == gate0.EXPECTED_ASSET_CONTRACT["initial_conditions_sha256"],
        "scene/robot asset provenance",
    )
    runtime_protocol = payload.get("runtime_protocol")
    require(
        isinstance(runtime_protocol, dict)
        and runtime_protocol.get("physics_hz") == 120.0
        and runtime_protocol.get("physics_dt") == 1.0 / 120.0
        and runtime_protocol.get("decimation") == diagnostic.DECIMATION
        and runtime_protocol.get("policy_hz") == 15.0,
        "runtime physics/policy cadence provenance",
    )
    runtime_frame = payload.get("runtime_frame")
    require(
        isinstance(runtime_frame, dict)
        and runtime_frame.get("action_dim") == 7
        and runtime_frame.get("command_type") == "pose"
        and runtime_frame.get("controlled_body") == "panda_link8"
        and runtime_frame.get("eef_frame") == "panda_link8"
        and runtime_frame.get("reference_frame") == "panda_link0"
        and runtime_frame.get("ik_method") == "dls"
        and runtime_frame.get("dls_damping") == 0.01
        and runtime_frame.get("ik_safety_profile")
        == "panda_velocity_physxlimit_solveriter1_v4"
        and runtime_frame.get("use_relative_mode") is False,
        "runtime EEF controller-frame provenance",
    )
    gripper = payload.get("production_gripper_contract_before_ablation")
    require(
        isinstance(gripper, dict)
        and gripper.get("joint_names") == diagnostic.JOINT_NAMES
        and gripper.get("driver_joint_index") == 7
        and gripper.get("follower_joint_indices") == diagnostic.FOLLOWER_INDICES
        and gripper.get("driver_target_slew", {}).get("profile")
        == (
            "eef_binary_driver_target_slew_rate1p25_from_live_limit5_"
            "per_120hz_substep_candidate_v1"
        )
        and gripper.get("mimic_compliance", {}).get("profile")
        == "robotiq_2f85_live_physx_mimic_frequency100_damping1p2_candidate_v1"
        and gripper.get("measured_velocity_is_hard_bounded_by_limit") is False,
        "production gripper/controller provenance",
    )


def probe_video(path: Path, *, ffprobe: str, ffmpeg: str) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=codec_name,pix_fmt,width,height,field_order,r_frame_rate,"
            "avg_frame_rate,nb_read_frames"
        ),
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    try:
        payload = json.loads(
            result.stdout,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValidationError) as error:
        raise ValidationError("ffprobe returned invalid JSON") from error
    streams = payload.get("streams")
    require(isinstance(streams, list) and len(streams) == 1, "one video stream")
    stream = streams[0]
    require(
        isinstance(stream, dict)
        and stream.get("codec_name") == "h264"
        and stream.get("pix_fmt") == "yuv420p"
        and stream.get("width") == diagnostic.VIDEO_WIDTH
        and stream.get("height") == diagnostic.VIDEO_HEIGHT
        and stream.get("field_order") in (None, "progressive")
        and Fraction(stream.get("r_frame_rate")) == Fraction(diagnostic.VIDEO_FPS, 1)
        and Fraction(stream.get("avg_frame_rate")) == Fraction(diagnostic.VIDEO_FPS, 1),
        f"video codec/layout/cadence: {stream!r}",
    )
    try:
        frame_count = int(stream["nb_read_frames"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValidationError("video decoded frame count") from error
    frame_result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=interlaced_frame,top_field_first",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        frame_payload = json.loads(
            frame_result.stdout,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValidationError) as error:
        raise ValidationError("ffprobe frame scan returned invalid JSON") from error
    frames = frame_payload.get("frames")
    require(
        isinstance(frames, list)
        and len(frames) == frame_count
        and all(
            isinstance(frame, dict)
            and frame.get("interlaced_frame") == 0
            and frame.get("top_field_first") in (None, 0)
            for frame in frames
        ),
        "video progressive frame flags",
    )
    box_layout = mp4_box_layout(path)
    box_types = [box["type"] for box in box_layout]
    require(
        box_types.count("moov") == 1
        and box_types.count("mdat") == 1
        and box_types.index("moov") < box_types.index("mdat"),
        f"MP4 fast-start box order: {box_types!r}",
    )
    subprocess.run(
        [ffmpeg, "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return {
        "profile": ("ffprobe_frame_scan_faststart_and_ffmpeg_full_decode_v2"),
        "codec_name": stream["codec_name"],
        "pixel_format": stream["pix_fmt"],
        "field_order": "progressive",
        "stream_field_order": stream.get("field_order"),
        "progressive_frame_flag_count": len(frames),
        "mp4_top_level_box_types": box_types,
        "width": stream["width"],
        "height": stream["height"],
        "fps": diagnostic.VIDEO_FPS,
        "frame_count": frame_count,
    }


def mp4_box_layout(path: Path) -> list[dict[str, Any]]:
    size = path.stat().st_size
    boxes: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        offset = 0
        while offset < size:
            stream.seek(offset)
            header = stream.read(8)
            require(len(header) == 8, "truncated MP4 box header")
            box_size = int.from_bytes(header[:4], "big")
            try:
                box_type = header[4:8].decode("ascii")
            except UnicodeDecodeError as error:
                raise ValidationError("non-ASCII MP4 box type") from error
            header_size = 8
            if box_size == 1:
                extended = stream.read(8)
                require(len(extended) == 8, "truncated extended MP4 box size")
                box_size = int.from_bytes(extended, "big")
                header_size = 16
            elif box_size == 0:
                box_size = size - offset
            require(
                box_size >= header_size and offset + box_size <= size,
                f"invalid MP4 box {box_type!r} at {offset}",
            )
            boxes.append({"type": box_type, "offset": offset, "size_bytes": box_size})
            offset += box_size
    require(offset == size and bool(boxes), "MP4 top-level box coverage")
    return boxes


def validate_result(
    payload: dict[str, Any],
    *,
    variant: str,
    expected_commit: str,
    expected_job_id: int,
    expected_launch_id: str,
    result_identity: dict[str, Any],
    video_path: Path,
    ffprobe: str,
    ffmpeg: str,
    simulator_srun_exit_code: int,
) -> dict[str, Any]:
    require(set(payload) == RESULT_FIELDS, "result top-level schema")
    require(
        payload.get("schema_version") == 1
        and payload.get("profile") == diagnostic.PROFILE
        and payload.get("passed") is True
        and payload.get("diagnostic_only") is True
        and payload.get("variant") == variant,
        "result profile/variant",
    )
    repository = payload.get("repository")
    require(
        isinstance(repository, dict)
        and repository.get("commit") == expected_commit
        and repository.get("clean_tracked") is True
        and repository.get("diagnostic_base_commit")
        == diagnostic.DIAGNOSTIC_BASE_COMMIT
        and repository.get("diagnostic_base_relation") == "exact_first_parent_v1"
        and repository.get("production_base_commit")
        == diagnostic.PRODUCTION_BASE_COMMIT
        and repository.get("production_base_relation") == "exact_first_grandparent_v1",
        "repository provenance",
    )
    lifecycle = payload.get("lifecycle")
    require(
        isinstance(lifecycle, dict)
        and lifecycle.get("job_id") == expected_job_id
        and lifecycle.get("launch_id") == expected_launch_id
        and lifecycle.get("procid") == lifecycle.get("localid") == 0
        and lifecycle.get("ntasks") == 1,
        "Slurm lifecycle",
    )
    container = payload.get("container_image")
    require(
        isinstance(container, dict)
        and container.get("size_bytes") == 7_183_130_624
        and container.get("sha256")
        == "ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a",
        "container identity",
    )
    require(
        payload.get("source_trace_sha256") == fixture_contract.TRACE_SHA256
        and payload.get("source_action_float32_sha256")
        == diagnostic.ACTION_ENCODING["uncompressed_sha256"]
        and payload.get("action_count") == diagnostic.ACTION_COUNT,
        "source trace/action identity",
    )
    validate_source_scene_controller_provenance(payload)
    _, source_actions = diagnostic.load_actions()
    expected_tail = diagnostic.frozen_tail_contract(source_actions)
    require(payload.get("tail_contract") == expected_tail, "frozen tail contract")
    validate_intervention(payload.get("intervention"), variant=variant)
    entries = payload.get("full_substep_trace")
    require(
        isinstance(entries, list)
        and payload.get("full_substep_trace_profile")
        == (
            "all13_after_arm_before_gripper_post_setters_post_physics_"
            "before_next_arm_v2"
        ),
        "full 13-DOF three-phase substep trace profile",
    )
    numerical_failure = payload.get("numerical_failure")
    require(
        numerical_failure is None or isinstance(numerical_failure, dict),
        "numerical failure schema",
    )
    outcome = payload.get("outcome")
    expected_outcome = diagnostic.classify_outcome(
        variant,
        numerical_failure,
        payload.get("actions_completed"),
        payload.get("tail_policy_steps_completed"),
        len(entries),
    )
    require(
        isinstance(outcome, dict)
        and outcome == expected_outcome
        and outcome.get("diagnostic_completed") is True
        and outcome.get("controller_completed_actions")
        == payload.get("actions_completed")
        and outcome.get("source_actions_completed") == payload.get("actions_completed")
        and outcome.get("tail_policy_steps_completed")
        == payload.get("tail_policy_steps_completed")
        and outcome.get("tail_physics_substeps_completed")
        == payload.get("tail_physics_substeps_completed"),
        "diagnostic outcome",
    )
    classification = outcome.get("classification")
    controller_failure_evidence = payload.get("controller_failure_evidence")
    if classification == "followup_completed_source_and_tail":
        require(
            numerical_failure is None
            and controller_failure_evidence is None
            and payload.get("actions_completed") == diagnostic.ACTION_COUNT
            and payload.get("tail_policy_steps_completed")
            == diagnostic.TAIL_POLICY_STEPS
            and payload.get("tail_physics_substeps_completed")
            == diagnostic.TAIL_PHYSICS_SUBSTEPS
            and outcome.get("failure_segment") is None
            and outcome.get("parsed_numerical_failure") is None
            and outcome.get("numerical_failure_parse_error") is None,
            "complete follow-up outcome",
        )
        if variant == diagnostic.RELEASE_RAMP_VARIANT:
            ramp_runtime = payload["intervention"]["arm_release_ramp_runtime"]
            require(
                ramp_runtime["release_observed_count"] == 3
                and ramp_runtime["ramp_started_count"] == 3
                and ramp_runtime["ramp_completed_count"] == 3
                and ramp_runtime["ramp_cancelled_by_reactivation_count"] == 0
                and ramp_runtime["ramp_target_apply_count"]
                == 3 * diagnostic.RELEASE_RAMP_SUBSTEPS
                and len(ramp_runtime["overlay_entries"])
                == 3 * diagnostic.RELEASE_RAMP_SUBSTEPS
                and ramp_runtime["applied_indices"]
                == list(range(diagnostic.RELEASE_RAMP_SUBSTEPS)) * 3
                and ramp_runtime["pending_at_report"] is False
                and ramp_runtime["next_index_at_report"] is None,
                "complete cap-5 release-ramp lifecycle",
            )
    else:
        require(
            classification == "followup_numerical_failure_observed"
            and isinstance(numerical_failure, dict)
            and isinstance(outcome.get("parsed_numerical_failure"), dict)
            and outcome.get("numerical_failure_parse_error") is None
            and outcome.get("failure_segment")
            in {"source_actions", "frozen_final_command_tail"},
            "follow-up numerical-failure outcome",
        )
        diagnostic.validate_controller_failure_evidence(
            controller_failure_evidence,
            failure=numerical_failure,
            entries=entries,
        )
    require(
        payload.get("tail_physics_substeps_completed")
        == expected_outcome["tail_physics_substeps_completed"],
        "tail physics-substep count recomputation",
    )
    expected_cadence = diagnostic.validate_trace_cadence(
        entries,
        variant=variant,
        outcome=outcome,
        release_ramp_runtime=payload["intervention"]["arm_release_ramp_runtime"],
    )
    require(
        payload.get("full_substep_trace_cadence") == expected_cadence,
        "full-trace cadence report",
    )
    expected_summary = diagnostic.summarize_trace(entries)
    require(
        payload.get("full_substep_summary") == expected_summary,
        "full-trace derived summary",
    )
    close = payload.get("runtime_close")
    require(
        close
        == {
            "environment_close_completed": True,
            "simulation_app_close_state": (
                "pending_terminal_call_after_raw_publication_v2"
            ),
            "publication_timing": "after_environment_before_simulation_app_close_v2",
            "completion_evidence": (
                "post_kit_validator_requires_zero_simulator_srun_exit_v1"
            ),
        },
        "pre-SimulationApp-close result publication",
    )
    require(simulator_srun_exit_code == 0, "simulator srun exit code")
    video_identity = file_identity(video_path)
    recorded_video = payload.get("video")
    require(
        isinstance(recorded_video, dict)
        and all(
            recorded_video.get(field) == video_identity[field]
            for field in ("path", "size_bytes", "sha256", "mode", "nlink")
        )
        and video_identity["mode"] == "0444"
        and video_identity["nlink"] == 1,
        "video recorded/live identity",
    )
    decoded = probe_video(video_path, ffprobe=ffprobe, ffmpeg=ffmpeg)
    expected_frames = (
        payload["actions_completed"]
        + payload["tail_policy_steps_completed"]
        + (2 if numerical_failure is not None else 1)
    )
    require(
        recorded_video.get("frame_count") == expected_frames
        and decoded["frame_count"] == expected_frames,
        "video frame/outcome binding",
    )
    return {
        "schema_version": 1,
        "profile": PROFILE,
        "variant": variant,
        "job_id": expected_job_id,
        "launch_id": expected_launch_id,
        "diagnostic_commit": expected_commit,
        "production_base_commit": diagnostic.PRODUCTION_BASE_COMMIT,
        "result": result_identity,
        "video": video_identity,
        "video_decode": decoded,
        "outcome": outcome,
        "controller_failure_evidence": controller_failure_evidence,
        "full_substep_summary": expected_summary,
        "full_substep_trace_cadence": expected_cadence,
        "tail_contract": expected_tail,
        "intervention": payload["intervention"],
        "runtime_close": close,
        "runtime_exit": {
            "profile": "zero_simulator_srun_after_terminal_simulation_app_close_v1",
            "simulator_srun_exit_code": simulator_srun_exit_code,
            "validated_in_separate_post_kit_process": True,
        },
    }


def atomic_write(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    data = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    path = path.resolve()
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        path.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)
    return file_identity(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=diagnostic.VARIANTS, required=True)
    parser.add_argument("--expected-diagnostic-commit", required=True)
    parser.add_argument("--expected-job-id", type=int, required=True)
    parser.add_argument("--expected-launch-id", required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--ffprobe", default="/usr/bin/ffprobe")
    parser.add_argument("--ffmpeg", default="/usr/bin/ffmpeg")
    parser.add_argument("--simulator-srun-exit-code", type=int, required=True)
    args = parser.parse_args()
    require(
        len(args.expected_diagnostic_commit) == 40
        and all(
            character in "0123456789abcdef"
            for character in args.expected_diagnostic_commit
        ),
        "diagnostic commit grammar",
    )
    require(
        len(args.expected_launch_id) == 64
        and all(
            character in "0123456789abcdef" for character in args.expected_launch_id
        ),
        "launch id grammar",
    )
    payload, result_identity = strict_json(args.result_json)
    require(
        result_identity["mode"] == "0444" and result_identity["nlink"] == 1,
        "result JSON mode/link",
    )
    manifest = validate_result(
        payload,
        variant=args.variant,
        expected_commit=args.expected_diagnostic_commit,
        expected_job_id=args.expected_job_id,
        expected_launch_id=args.expected_launch_id,
        result_identity=result_identity,
        video_path=args.video,
        ffprobe=args.ffprobe,
        ffmpeg=args.ffmpeg,
        simulator_srun_exit_code=args.simulator_srun_exit_code,
    )
    identity = atomic_write(args.output_manifest, manifest)
    print(
        f"POLARIS_FULLTRACE_CAP_RELEASE_VALIDATED={identity['path']};"
        f"size={identity['size_bytes']};sha256={identity['sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
