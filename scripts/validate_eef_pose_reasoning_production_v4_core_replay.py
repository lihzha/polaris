#!/usr/bin/env python3
"""Validate and finalize one exact production-v4 core full-trace replay."""

from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import subprocess
from typing import Any

import build_reasoning_fulltrace_replay_fixture as fixture_contract
import finalize_eef_pose_smoke as safety_post_kit_contract
import numpy as np
from polaris import eef_controller_profile as controller_profile_contract
from polaris import eef_gripper_failure_trace as gripper_trace_contract
from polaris import eef_gripper_runtime as gripper_runtime_contract
from polaris import eef_gripper_target_slew as gripper_target_slew_contract
from polaris import eef_ik_safety as ik_safety_contract
from polaris import eef_runtime_contract as safety_contract
import smoke_eef_pose_reasoning_production_v4_core_replay as replay


PROFILE = "reasoning_43075_production_v4_core_release_ramp_post_kit_v1"
ROOT = Path(__file__).resolve().parents[1]
CONTAINER_SIZE_BYTES = 7_183_130_624
CONTAINER_SHA256 = "ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a"
EXPECTED_TARGET_SLEW_PROFILE = (
    "eef_binary_driver_target_slew_rate1p25_from_live_limit5_"
    "per_120hz_substep_candidate_v1"
)
EXPECTED_GRIPPER_ENDPOINT_CHANGES = 5
EXPECTED_POLICY_STEPS = replay.ACTION_COUNT + replay.TAIL_POLICY_STEPS
POST_KIT_VALIDATOR_SOURCE_SHA256 = {
    "scripts/finalize_eef_pose_smoke.py": (
        "74dccfeb25c9522e5741eb72510f3f7940abd64678be8a357aca102fe2038fc7"
    ),
    "src/polaris/eef_controller_profile.py": (
        "af5c7d73a0b1bd5bf229c1b54f3c271d46fbdaafc7b81b9a1bd2799133420ec1"
    ),
    "src/polaris/eef_gripper_failure_trace.py": (
        "f66af5001f8636333f6db00948a64214909a43b7d6afd1af968397dea33280b0"
    ),
    "src/polaris/eef_gripper_runtime.py": (
        "6a4f8b13ec14c3e11d3be82517bfc71c4b2fc8e4b6c4da119611ed1a4ebe16f9"
    ),
    "src/polaris/eef_gripper_target_slew.py": (
        "1d2a0218d89a72a5692bf79f013fdd350addd2e75507ade5c680d180a93e57a1"
    ),
    "src/polaris/eef_ik_safety.py": (
        "fb3d4f4f813c03de034c9645698b6fe8b8bbd89da0ff8d8dcf9fd8f0c95d7346"
    ),
    "src/polaris/eef_runtime_contract.py": (
        "a5f868f58be6850b9b7a4f9f2c1b719d0c867ad969b2fbec26a843e25db90cef"
    ),
}
POST_KIT_VALIDATOR_MODULES = {
    "scripts/finalize_eef_pose_smoke.py": safety_post_kit_contract,
    "src/polaris/eef_controller_profile.py": controller_profile_contract,
    "src/polaris/eef_gripper_failure_trace.py": gripper_trace_contract,
    "src/polaris/eef_gripper_runtime.py": gripper_runtime_contract,
    "src/polaris/eef_gripper_target_slew.py": gripper_target_slew_contract,
    "src/polaris/eef_ik_safety.py": ik_safety_contract,
    "src/polaris/eef_runtime_contract.py": safety_contract,
}
EXPECTED_CORE_SOURCE_SHA256 = {
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
RESULT_FIELDS = {
    "schema_version",
    "profile",
    "passed",
    "controller_replay_only",
    "variant",
    "controller_profile",
    "repository",
    "production_core_sources",
    "container_image",
    "lifecycle",
    "production_eval",
    "fixture",
    "source_trace_polaris_commit",
    "source_trace_sha256",
    "source_action_float32_sha256",
    "boundary_helper",
    "assets",
    "runtime_protocol",
    "runtime_frame",
    "observer_class_contract",
    "production_runtime",
    "production_gripper_contract",
    "production_safety",
    "production_controller_report",
    "production_all_six_gripper_trace",
    "production_core_ramp_observation",
    "action_count",
    "actions_completed",
    "tail_contract",
    "tail_policy_steps_completed",
    "tail_physics_substeps_completed",
    "total_apply_count",
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
EXPECTED_LIMITS = replay.ARM_VELOCITY_LIMITS_RAD_S + [5.0] * 6


class ValidationError(ValueError):
    """The post-Kit production replay artifact is incomplete or inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reject_constant(token: str) -> None:
    raise ValidationError(f"non-standard JSON constant {token!r}")


def checked_json_float(token: str) -> float:
    value = float(token)
    require(math.isfinite(value), f"non-finite JSON float {token!r}")
    return value


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def audit_json_numbers(value: Any, *, path: str = "$") -> None:
    """Reject non-JSON or non-finite numeric values at every nesting depth."""

    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        require(math.isfinite(value), f"non-finite JSON number at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            audit_json_numbers(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            require(type(key) is str, f"non-string JSON key at {path}")
            audit_json_numbers(item, path=f"{path}.{key}")
        return
    raise ValidationError(f"non-JSON value at {path}: {type(value).__name__}")


def strict_json_loads(text: str, *, field: str) -> Any:
    try:
        payload = json.loads(
            text,
            parse_float=checked_json_float,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, OverflowError) as error:
        raise ValidationError(f"invalid strict JSON {field}: {error}") from error
    audit_json_numbers(payload)
    return payload


def strict_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    data, identity = read_file_identity(path)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValidationError(f"invalid UTF-8 JSON {path}: {error}") from error
    payload = strict_json_loads(text, field=str(path))
    require(isinstance(payload, dict), f"JSON object required: {path}")
    return payload, identity


def read_file_identity(path: Path) -> tuple[bytes, dict[str, Any]]:
    original = Path(path)
    try:
        lexical_metadata = original.lstat()
    except FileNotFoundError as error:
        raise ValidationError(f"missing file {original}") from error
    require(
        not stat.S_ISLNK(lexical_metadata.st_mode)
        and stat.S_ISREG(lexical_metadata.st_mode),
        f"missing/linked file {original}",
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(original, flags)
    except OSError as error:
        raise ValidationError(
            f"cannot open nonlinked file {original}: {error}"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        require(
            stat.S_ISREG(metadata.st_mode)
            and (metadata.st_dev, metadata.st_ino)
            == (lexical_metadata.st_dev, lexical_metadata.st_ino),
            f"file identity changed before open: {original}",
        )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read()
        final_metadata = os.fstat(descriptor)
        require(
            (final_metadata.st_dev, final_metadata.st_ino, final_metadata.st_size)
            == (metadata.st_dev, metadata.st_ino, len(data)),
            f"file identity changed during read: {original}",
        )
    finally:
        os.close(descriptor)
    resolved = original.resolve(strict=True)
    resolved_metadata = resolved.stat()
    require(
        (resolved_metadata.st_dev, resolved_metadata.st_ino)
        == (metadata.st_dev, metadata.st_ino),
        f"resolved file identity drift: {original}",
    )
    identity = {
        "path": str(resolved),
        "size_bytes": len(data),
        "sha256": sha256(data),
        "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "nlink": metadata.st_nlink,
    }
    return data, identity


def file_identity(path: Path) -> dict[str, Any]:
    _data, identity = read_file_identity(path)
    return identity


def canonical_json_bytes(value: Any) -> bytes:
    audit_json_numbers(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def typed_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(right, dict):
        return set(left) == set(right) and all(
            typed_equal(left[name], expected) for name, expected in right.items()
        )
    if isinstance(right, list):
        return len(left) == len(right) and all(
            typed_equal(actual, expected)
            for actual, expected in zip(left, right, strict=True)
        )
    return bool(left == right)


def validate_post_kit_validator_sources() -> dict[str, dict[str, Any]]:
    identities: dict[str, dict[str, Any]] = {}
    for relative_path, expected_sha256 in POST_KIT_VALIDATOR_SOURCE_SHA256.items():
        identity = file_identity(ROOT / relative_path)
        require(
            identity["sha256"] == expected_sha256,
            f"post-Kit validator source hash: {relative_path}",
        )
        module_file = getattr(
            POST_KIT_VALIDATOR_MODULES[relative_path], "__file__", None
        )
        require(
            isinstance(module_file, str)
            and os.path.samefile(module_file, identity["path"]),
            f"post-Kit imported source identity: {relative_path}",
        )
        identities[relative_path] = identity
    return identities


def float32_equal(left: float, right: float) -> bool:
    try:
        return struct.pack("<f", float(left)) == struct.pack("<f", float(right))
    except (OverflowError, TypeError, ValueError, struct.error):
        return False


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


def validate_source_scene_controller_provenance(payload: dict[str, Any]) -> None:
    """Close immutable source, scene, profile, and observer inputs."""

    require(
        payload.get("fixture") == replay.file_identity(replay.FIXTURE_PATH),
        "fixture live/recorded identity",
    )
    gate0 = replay.gate0
    production = payload.get("production_eval")
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
        and production.get("effective_step_expensive") is True,
        "production reset/render provenance",
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
    protocol = payload.get("runtime_protocol")
    require(
        isinstance(protocol, dict)
        and protocol.get("physics_hz") == 120.0
        and protocol.get("physics_dt") == 1.0 / 120.0
        and protocol.get("decimation") == replay.DECIMATION
        and protocol.get("policy_hz") == 15.0,
        "runtime cadence provenance",
    )
    frame = payload.get("runtime_frame")
    require(
        isinstance(frame, dict)
        and frame.get("action_dim") == 7
        and frame.get("command_type") == "pose"
        and frame.get("controlled_body") == "panda_link8"
        and frame.get("eef_frame") == "panda_link8"
        and frame.get("reference_frame") == "panda_link0"
        and frame.get("ik_method") == "dls"
        and frame.get("dls_damping") == 0.01
        and frame.get("ik_safety_profile") == "panda_velocity_physxlimit_solveriter1_v4"
        and frame.get("use_relative_mode") is False,
        "runtime EEF frame provenance",
    )

    core_sources = payload.get("production_core_sources")
    require(
        isinstance(core_sources, dict)
        and set(core_sources) == set(EXPECTED_CORE_SOURCE_SHA256)
        and all(
            core_sources[path].get("sha256") == expected
            for path, expected in EXPECTED_CORE_SOURCE_SHA256.items()
        ),
        "production core source hashes",
    )
    observer = payload.get("observer_class_contract")
    require(
        isinstance(observer, dict)
        and observer.get("profile") == "post_validated_v4_observational_subclasses_v1"
        and observer.get("production_arm_module") == "polaris.robust_differential_ik"
        and observer.get("production_gripper_module")
        == "polaris.eef_gripper_failure_trace"
        and observer.get("runtime_arm_module")
        == "smoke_eef_pose_reasoning_production_v4_core_replay"
        and observer.get("runtime_gripper_module")
        == "smoke_eef_pose_reasoning_production_v4_core_replay"
        and all(
            observer.get(field) == 0
            for field in (
                "target_setter_call_count",
                "failure_trace_write_count",
                "release_ramp_state_write_count",
                "gripper_target_or_state_write_count",
            )
        ),
        "observational subclass contract",
    )
    runtime = payload.get("production_runtime")
    require(
        isinstance(runtime, dict)
        and runtime.get("profile") == "production_v4_core_replay_runtime_contract_v1"
        and runtime.get("variant") == replay.VARIANT
        and runtime.get("controller_profile") == replay.CONTROLLER_PROFILE
        and runtime.get("production_base_commit") == replay.PRODUCTION_BASE_COMMIT
        and runtime.get("configured_close_anchor_substeps") == 86
        and runtime.get("passive_follower_velocity_cap_rad_s") == 5.0
        and runtime.get("passive_follower_velocity_limit_setter_call_count") == 0
        and runtime.get("runtime_flags")
        == {
            "failure_substep_trace": True,
            "arm_slew_headroom": True,
            "gripper_close_arm_interlock": True,
            "arm_release_ramp": True,
        }
        and runtime.get("configured_profile", {}).get("profile")
        == replay.CONTROLLER_PROFILE
        and runtime.get("configured_profile", {}).get("all_six_gripper_trace_enabled")
        is True
        and runtime.get("configured_profile", {}).get("arm_release_ramp_enabled")
        is True
        and all(
            runtime.get(field) == 0
            for field in (
                "observer_target_setter_call_count",
                "observer_failure_trace_write_count",
                "observer_release_ramp_state_write_count",
                "observer_gripper_target_or_state_write_count",
            )
        ),
        "production v4 runtime contract",
    )
    first_limits = validate_tensor(
        runtime.get("velocity_limits_first_read"),
        field="first production velocity-limit read",
    )
    second_limits = validate_tensor(
        runtime.get("velocity_limits_second_read"),
        field="second production velocity-limit read",
    )
    require(
        all(
            float32_equal(actual, expected)
            for actual, expected in zip(first_limits, EXPECTED_LIMITS, strict=True)
        )
        and first_limits == second_limits,
        "production velocity-limit read identity",
    )
    gripper = payload.get("production_gripper_contract")
    require(
        isinstance(gripper, dict)
        and gripper.get("joint_names") == replay.JOINT_NAMES
        and gripper.get("driver_joint_index") == 7
        and gripper.get("follower_joint_indices") == replay.FOLLOWER_INDICES
        and gripper.get("driver_target_slew", {}).get("profile")
        == (
            "eef_binary_driver_target_slew_rate1p25_from_live_limit5_"
            "per_120hz_substep_candidate_v1"
        )
        and gripper.get("mimic_compliance", {}).get("profile")
        == "robotiq_2f85_live_physx_mimic_frequency100_damping1p2_candidate_v1"
        and gripper.get("measured_velocity_is_hard_bounded_by_limit") is False,
        "production v4 gripper contract",
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
        payload = strict_json_loads(result.stdout, field="ffprobe stream scan")
    except ValidationError as error:
        raise ValidationError("ffprobe returned invalid JSON") from error
    streams = payload.get("streams")
    require(isinstance(streams, list) and len(streams) == 1, "one video stream")
    stream = streams[0]
    require(
        isinstance(stream, dict)
        and stream.get("codec_name") == "h264"
        and stream.get("pix_fmt") == "yuv420p"
        and stream.get("width") == replay.VIDEO_WIDTH
        and stream.get("height") == replay.VIDEO_HEIGHT
        and stream.get("field_order") in (None, "progressive")
        and Fraction(stream.get("r_frame_rate")) == Fraction(replay.VIDEO_FPS, 1)
        and Fraction(stream.get("avg_frame_rate")) == Fraction(replay.VIDEO_FPS, 1),
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
        frame_payload = strict_json_loads(
            frame_result.stdout, field="ffprobe frame scan"
        )
    except ValidationError as error:
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
        "fps": replay.VIDEO_FPS,
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


def gripper_snapshot_from_full_trace(value: Any, *, field: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{field} snapshot")
    mapping = {
        "joint_pos_rad": "all_joint_pos_rad",
        "joint_vel_rad_s": "all_joint_vel_rad_s",
        "joint_acc_rad_s2": "all_joint_acc_rad_s2",
        "joint_pos_target_rad": "all_joint_pos_target_rad",
        "joint_vel_target_rad_s": "all_joint_vel_target_rad_s",
        "joint_effort_target_nm": "all_joint_effort_target_nm",
    }
    result: dict[str, Any] = {}
    for gripper_field, full_field in mapping.items():
        vector = value.get(full_field)
        require(
            isinstance(vector, list) and len(vector) == 13,
            f"{field}.{full_field}",
        )
        result[gripper_field] = vector[7:]
    return result


def validate_production_all_six_gripper_trace(
    value: Any,
    *,
    full_substep_trace: Any,
) -> dict[str, Any]:
    """Independently close and retain the exact production all-six evidence."""

    original = copy.deepcopy(value)
    try:
        validated = gripper_trace_contract.validate_eef_all_six_gripper_trace(
            value,
            episode_index=0,
            episode_length=EXPECTED_POLICY_STEPS,
            numerical_failure=False,
            expected_apply_calls=replay.TOTAL_APPLY_COUNT,
        )
    except (TypeError, ValueError) as error:
        raise ValidationError(f"production all-six gripper trace: {error}") from error
    require(
        typed_equal(validated, original) and typed_equal(value, original),
        "production all-six gripper canonical equality",
    )
    require(
        isinstance(full_substep_trace, list)
        and len(full_substep_trace) == replay.TOTAL_APPLY_COUNT,
        "all-six/full-trace apply count",
    )
    entries = validated["entries"]
    first_retained_apply = replay.TOTAL_APPLY_COUNT - len(entries)
    require(
        first_retained_apply == 2_352
        and validated["dropped_entries"] == 2_352
        and [entry["apply_index"] for entry in entries] == list(range(2_352, 2_416))
        and [entry["policy_step"] for entry in entries]
        == [policy for policy in range(294, 302) for _ in range(replay.DECIMATION)],
        "production all-six exact retained tail",
    )
    _, source_actions = replay.load_actions()
    policy_actions = [
        *source_actions,
        *([source_actions[-1]] * replay.TAIL_POLICY_STEPS),
    ]
    closed = gripper_runtime_contract.GRIPPER_CLOSED_TARGET_FLOAT32
    opened = gripper_runtime_contract.GRIPPER_OPEN_TARGET_FLOAT32
    for apply_index, full_entry in enumerate(full_substep_trace):
        require(isinstance(full_entry, dict), f"full trace action {apply_index}")
        policy_step = apply_index // replay.DECIMATION
        expected_raw = policy_actions[policy_step][7]
        if float32_equal(expected_raw, 0.0):
            expected_endpoint = opened
        elif float32_equal(expected_raw, 1.0):
            expected_endpoint = closed
        else:
            raise ValidationError(
                f"non-binary fixture gripper action at policy step {policy_step}"
            )
        require(
            full_entry.get("apply_index") == apply_index
            and full_entry.get("policy_step") == policy_step
            and full_entry.get("physics_substep") == apply_index % replay.DECIMATION
            and isinstance(full_entry.get("raw_action"), (int, float))
            and not isinstance(full_entry.get("raw_action"), bool)
            and float32_equal(full_entry["raw_action"], expected_raw)
            and isinstance(full_entry.get("requested_endpoint_rad"), (int, float))
            and not isinstance(full_entry.get("requested_endpoint_rad"), bool)
            and float32_equal(full_entry["requested_endpoint_rad"], expected_endpoint),
            f"full trace exact fixture gripper action at apply {apply_index}",
        )
    for entry in entries:
        apply_index = entry["apply_index"]
        full_entry = full_substep_trace[apply_index]
        require(
            isinstance(full_entry, dict)
            and full_entry.get("apply_index") == apply_index
            and full_entry.get("policy_step") == entry["policy_step"]
            and full_entry.get("physics_substep") == entry["physics_substep"],
            f"all-six/full-trace identity at apply {apply_index}",
        )
        command = full_entry.get("command_after_setters")
        require(
            isinstance(command, dict)
            and isinstance(command.get("all_joint_pos_target_rad"), list)
            and len(command["all_joint_pos_target_rad"]) == 13,
            f"all-six/full-trace command at apply {apply_index}",
        )
        expected_raw = policy_actions[entry["policy_step"]][7]
        require(
            float32_equal(entry["raw_action"], expected_raw)
            and float32_equal(full_entry.get("raw_action"), expected_raw)
            and float32_equal(entry["requested_endpoint_rad"], closed)
            and float32_equal(full_entry.get("requested_endpoint_rad"), closed)
            and float32_equal(entry["target_after_setter_rad"], closed)
            and float32_equal(full_entry.get("target_after_setter_rad"), closed)
            and typed_equal(
                entry["pre"],
                gripper_snapshot_from_full_trace(
                    full_entry.get("pre"), field=f"full trace pre {apply_index}"
                ),
            )
            and typed_equal(
                entry["post"],
                gripper_snapshot_from_full_trace(
                    full_entry.get("post"), field=f"full trace post {apply_index}"
                ),
            )
            and float32_equal(
                entry["target_after_setter_rad"],
                command["all_joint_pos_target_rad"][7],
            ),
            f"all-six/full-trace causal binding at apply {apply_index}",
        )
    require(
        typed_equal(
            validated["initial_snapshot"],
            gripper_snapshot_from_full_trace(
                full_substep_trace[0].get("pre"), field="full trace initial pre"
            ),
        )
        and typed_equal(
            validated["terminal_snapshot"],
            gripper_snapshot_from_full_trace(
                full_substep_trace[-1].get("post"), field="full trace terminal post"
            ),
        )
        and float32_equal(
            validated["terminal_snapshot"]["joint_pos_target_rad"][0], closed
        ),
        "production all-six initial/terminal full-trace binding",
    )
    encoded = canonical_json_bytes(validated)
    return {
        "profile": "post_kit_exact_all_six_gripper_trace_validation_v1",
        "episode_index": 0,
        "episode_length": EXPECTED_POLICY_STEPS,
        "expected_apply_calls": replay.TOTAL_APPLY_COUNT,
        "numerical_failure": False,
        "canonical_json_sha256": sha256(encoded),
        "canonical_json_size_bytes": len(encoded),
        "validated_trace": copy.deepcopy(validated),
    }


def derive_gripper_target_slew_evidence(
    all_six_gripper_trace: dict[str, Any],
) -> dict[str, Any]:
    """Replay production float32 target slew from the retained initial anchor."""

    initial = all_six_gripper_trace.get("initial_snapshot")
    require(isinstance(initial, dict), "gripper target-slew initial snapshot")
    initial_positions = initial.get("joint_pos_rad")
    require(
        isinstance(initial_positions, list) and len(initial_positions) == 6,
        "gripper target-slew initial position vector",
    )
    initial_anchor = initial_positions[0]
    require(
        isinstance(initial_anchor, (int, float))
        and not isinstance(initial_anchor, bool)
        and math.isfinite(float(initial_anchor)),
        "gripper target-slew initial anchor",
    )
    profile = gripper_runtime_contract.eef_gripper_target_slew_profile(
        EXPECTED_TARGET_SLEW_PROFILE
    )
    previous = np.float32(initial_anchor)
    maximum_step = np.float32(profile.max_target_step_rad_float32)
    minimum_anchor = np.float32(
        gripper_runtime_contract.GRIPPER_TARGET_SLEW_MIN_ANCHOR_FLOAT32
    )
    maximum_anchor = np.float32(
        gripper_runtime_contract.GRIPPER_TARGET_SLEW_MAX_ANCHOR_FLOAT32
    )
    require(
        bool(minimum_anchor <= previous <= maximum_anchor)
        and float32_equal(float(previous), 0.0),
        "gripper target-slew initial anchor bounds",
    )
    _, source_actions = replay.load_actions()
    require(
        len(source_actions) == replay.ACTION_COUNT
        and all(
            isinstance(action, list) and len(action) == 8 for action in source_actions
        ),
        "gripper target-slew source action shape",
    )
    policy_actions = [
        *source_actions,
        *([source_actions[-1]] * replay.TAIL_POLICY_STEPS),
    ]
    require(
        len(policy_actions) == EXPECTED_POLICY_STEPS,
        "gripper target-slew policy count",
    )
    previous_endpoint: np.float32 | None = None
    endpoint_changes = 0
    endpoint_change_policy_steps: list[int] = []
    endpoint_change_apply_indices: list[int] = []
    limited_apply_count = 0
    apply_count = 0
    maximum_target_step = np.float32(0.0)
    maximum_error_before = np.float32(0.0)
    maximum_error_after = np.float32(0.0)
    endpoint = np.float32(0.0)
    for policy_index, action in enumerate(policy_actions):
        raw = float(action[7])
        if float32_equal(raw, 0.0):
            endpoint = np.float32(gripper_runtime_contract.GRIPPER_OPEN_TARGET_FLOAT32)
        elif float32_equal(raw, 1.0):
            endpoint = np.float32(
                gripper_runtime_contract.GRIPPER_CLOSED_TARGET_FLOAT32
            )
        else:
            raise ValidationError(
                f"non-binary fixture gripper action at policy step {policy_index}"
            )
        if previous_endpoint is not None and endpoint != previous_endpoint:
            endpoint_changes += 1
            endpoint_change_policy_steps.append(policy_index)
            endpoint_change_apply_indices.append(policy_index * replay.DECIMATION)
        previous_endpoint = endpoint
        for _ in range(replay.DECIMATION):
            delta = np.subtract(endpoint, previous, dtype=np.float32)
            error_before = np.abs(delta)
            limited = bool(error_before > maximum_step)
            step = np.clip(delta, -maximum_step, maximum_step).astype(np.float32)
            candidate = np.add(previous, step, dtype=np.float32)
            next_target = candidate if limited else endpoint
            applied_step = np.subtract(next_target, previous, dtype=np.float32)
            if np.abs(applied_step) > maximum_step:
                next_target = np.nextafter(next_target, previous, dtype=np.float32)
                applied_step = np.subtract(next_target, previous, dtype=np.float32)
            error_after = np.abs(np.subtract(endpoint, next_target, dtype=np.float32))
            require(
                bool(np.isfinite(next_target))
                and bool(np.isfinite(applied_step))
                and bool(np.abs(applied_step) <= maximum_step)
                and bool(minimum_anchor <= next_target <= maximum_anchor)
                and bool(error_after <= error_before),
                "gripper target-slew independent simulation invariant",
            )
            maximum_target_step = np.maximum(maximum_target_step, np.abs(applied_step))
            maximum_error_before = np.maximum(maximum_error_before, error_before)
            maximum_error_after = np.maximum(maximum_error_after, error_after)
            previous = np.float32(next_target)
            limited_apply_count += int(limited)
            apply_count += 1
    require(
        apply_count == replay.TOTAL_APPLY_COUNT
        and endpoint_changes == EXPECTED_GRIPPER_ENDPOINT_CHANGES,
        "gripper target-slew independent fixture cadence",
    )
    require(
        endpoint_change_policy_steps == [198, 200, 265, 272, 281]
        and endpoint_change_apply_indices == [1_584, 1_600, 2_120, 2_176, 2_248]
        and limited_apply_count == 217
        and apply_count - limited_apply_count == 2_199
        and float32_equal(
            float(previous), gripper_runtime_contract.GRIPPER_CLOSED_TARGET_FLOAT32
        ),
        "gripper target-slew exact endpoint transition indices",
    )
    return {
        "profile": "exact_fixture_float32_gripper_target_slew_replay_v1",
        "target_slew_profile": profile.profile,
        "process_action_calls": len(policy_actions),
        "apply_calls": apply_count,
        "endpoint_change_count": endpoint_changes,
        "endpoint_change_policy_steps": endpoint_change_policy_steps,
        "endpoint_change_apply_indices": endpoint_change_apply_indices,
        "repeated_endpoint_process_count": (len(policy_actions) - 1 - endpoint_changes),
        "slew_limited_apply_count": limited_apply_count,
        "endpoint_reached_apply_count": apply_count - limited_apply_count,
        "initial_anchor_rad": float(np.float32(initial_anchor)),
        "last_requested_endpoint_rad": float(endpoint),
        "last_applied_target_rad": float(previous),
        "max_abs_target_step_rad": float(maximum_target_step),
        "max_abs_endpoint_error_before_step_rad": float(maximum_error_before),
        "max_abs_endpoint_error_after_step_rad": float(maximum_error_after),
    }


def validate_production_safety(
    value: Any,
    *,
    all_six_gripper_trace: dict[str, Any],
    controller_report: Any,
) -> dict[str, Any]:
    """Apply the official closed safety/profile validators after Kit exits."""

    require(isinstance(value, dict), "production safety object")
    original = copy.deepcopy(value)
    original_controller_report = copy.deepcopy(controller_report)
    expected_target_slew = derive_gripper_target_slew_evidence(all_six_gripper_trace)
    episode_result = {
        "episode": 0,
        "episode_length": EXPECTED_POLICY_STEPS,
        "success": False,
        "progress": 0.0,
        "numerical_failure": False,
        "numerical_failure_reason": "",
    }
    try:
        cadence = safety_contract.validate_episode_safety_cadence(
            safety=value,
            episode_result=episode_result,
            expected_gripper_target_slew_profile=EXPECTED_TARGET_SLEW_PROFILE,
        )
        safety_post_kit_contract._validate_safety_report(  # noqa: SLF001
            value,
            field="production safety",
            episode_index=0,
            apply_calls=replay.TOTAL_APPLY_COUNT,
            expect_closed_target=True,
            expected_endpoint_change_count=EXPECTED_GRIPPER_ENDPOINT_CHANGES,
            expected_gripper_target_slew_profile=EXPECTED_TARGET_SLEW_PROFILE,
            expected_slew_limited_apply_count=expected_target_slew[
                "slew_limited_apply_count"
            ],
        )
        specification = (
            controller_profile_contract.validate_eef_controller_safety_evidence(
                value,
                expected_profile=replay.CONTROLLER_PROFILE,
                expected_target_slew_profile=EXPECTED_TARGET_SLEW_PROFILE,
            )
        )
        apply_calls, committed_apply_calls = (
            controller_profile_contract.eef_controller_apply_counts_from_safety(value)
        )
        validated_controller_report = (
            controller_profile_contract.validate_eef_controller_repair_candidate_report(
                controller_report,
                expected_profile=replay.CONTROLLER_PROFILE,
                expected_target_slew_profile=EXPECTED_TARGET_SLEW_PROFILE,
                expected_physical_max_delta_joint_pos_rad=value.get(
                    "max_delta_joint_pos_rad"
                ),
                apply_calls=apply_calls,
                committed_apply_calls=committed_apply_calls,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValidationError(
            f"production safety closed-schema validation: {error}"
        ) from error
    require(
        typed_equal(value, original)
        and typed_equal(validated_controller_report, original_controller_report)
        and typed_equal(controller_report, original_controller_report),
        "production safety canonical equality",
    )
    require(
        cadence
        == {
            "apply_calls": replay.TOTAL_APPLY_COUNT,
            "expected_decimation": replay.DECIMATION,
            "failed_policy_step": None,
            "failed_physics_substep": None,
            "abort_count": 0,
        }
        and apply_calls == replay.TOTAL_APPLY_COUNT
        and committed_apply_calls == replay.TOTAL_APPLY_COUNT
        and validated_controller_report == controller_report
        and validated_controller_report.get("gripper_close_arm_interlock", {}).get(
            "observed_endpoint_change_count"
        )
        == EXPECTED_GRIPPER_ENDPOINT_CHANGES
        and specification.profile == replay.CONTROLLER_PROFILE
        and specification.target_slew_profile == EXPECTED_TARGET_SLEW_PROFILE
        and specification.all_six_gripper_trace_enabled is True
        and specification.arm_release_ramp_enabled is True,
        "production safety v4 profile binding",
    )
    counters = value.get("counters")
    require(
        isinstance(counters, dict)
        and counters.get("apply_calls") == replay.TOTAL_APPLY_COUNT
        and counters.get("environment_substeps") == replay.TOTAL_APPLY_COUNT
        and all(
            counters.get(field) == 0
            for field in (
                "current_joint_limit_aborts",
                "invariant_aborts",
                "nonfinite_aborts",
                "dls_fallbacks",
                "post_clamp_target_violations",
                "guard_diagnostics_dropped",
            )
        )
        and value.get("episode_index") == 0
        and value.get("current_joint_velocity_abort") is None,
        "production safety exact success cadence",
    )
    dynamic = value.get("gripper_runtime_dynamic")
    require(isinstance(dynamic, dict), "production safety gripper dynamic")
    target_slew = dynamic.get("driver_target_slew")
    require(isinstance(target_slew, dict), "production safety target-slew dynamic")
    exact_integer_fields = (
        "process_action_calls",
        "apply_calls",
        "endpoint_change_count",
        "repeated_endpoint_process_count",
        "slew_limited_apply_count",
        "endpoint_reached_apply_count",
    )
    exact_float32_fields = (
        "initial_anchor_rad",
        "last_requested_endpoint_rad",
        "last_applied_target_rad",
        "max_abs_target_step_rad",
        "max_abs_endpoint_error_before_step_rad",
        "max_abs_endpoint_error_after_step_rad",
    )
    require(
        dynamic.get("apply_entry_samples") == replay.TOTAL_APPLY_COUNT
        and dynamic.get("post_policy_step_samples") == EXPECTED_POLICY_STEPS
        and dynamic.get("nonfinite_samples") == 0
        and dynamic.get("dropped_diagnostics") == 0
        and target_slew.get("profile") == EXPECTED_TARGET_SLEW_PROFILE
        and target_slew.get("initialization_count") == 1
        and target_slew.get("live_limit_validation_count") == replay.TOTAL_APPLY_COUNT
        and all(
            target_slew.get(field) == expected_target_slew[field]
            for field in exact_integer_fields
        )
        and all(
            isinstance(target_slew.get(field), (int, float))
            and not isinstance(target_slew.get(field), bool)
            and float32_equal(target_slew[field], expected_target_slew[field])
            for field in exact_float32_fields
        ),
        "production safety exact fixture target-slew binding",
    )
    terminal = dynamic.get("terminal_state")
    closed = gripper_runtime_contract.GRIPPER_CLOSED_TARGET_FLOAT32
    require(
        isinstance(terminal, dict)
        and terminal.get("sample_index") == 2_717
        and isinstance(terminal.get("joint_position_target_rad"), list)
        and len(terminal["joint_position_target_rad"]) == 6
        and float32_equal(terminal["joint_position_target_rad"][0], closed)
        and float32_equal(target_slew.get("last_requested_endpoint_rad"), closed)
        and float32_equal(target_slew.get("last_applied_target_rad"), closed)
        and float32_equal(
            all_six_gripper_trace["terminal_snapshot"]["joint_pos_target_rad"][0],
            closed,
        ),
        "production safety terminal all-six binding",
    )
    encoded = canonical_json_bytes(value)
    controller_encoded = canonical_json_bytes(validated_controller_report)
    return {
        "profile": "post_kit_closed_production_safety_validation_v1",
        "episode_index": 0,
        "policy_steps": EXPECTED_POLICY_STEPS,
        "apply_calls": replay.TOTAL_APPLY_COUNT,
        "controller_profile": replay.CONTROLLER_PROFILE,
        "target_slew_profile": EXPECTED_TARGET_SLEW_PROFILE,
        "canonical_json_sha256": sha256(encoded),
        "canonical_json_size_bytes": len(encoded),
        "controller_report_canonical_json_sha256": sha256(controller_encoded),
        "controller_report_canonical_json_size_bytes": len(controller_encoded),
        "episode_safety_cadence": cadence,
        "independent_target_slew_replay": expected_target_slew,
        "validated_safety": copy.deepcopy(value),
        "validated_controller_report": copy.deepcopy(validated_controller_report),
    }


def validate_result(
    payload: dict[str, Any],
    *,
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
        and payload.get("profile") == replay.PROFILE
        and payload.get("passed") is True
        and payload.get("controller_replay_only") is True
        and payload.get("variant") == replay.VARIANT
        and payload.get("controller_profile") == replay.CONTROLLER_PROFILE,
        "result profile",
    )
    validator_sources = validate_post_kit_validator_sources()
    gripper_validation = validate_production_all_six_gripper_trace(
        payload.get("production_all_six_gripper_trace"),
        full_substep_trace=payload.get("full_substep_trace"),
    )
    safety_validation = validate_production_safety(
        payload.get("production_safety"),
        all_six_gripper_trace=gripper_validation["validated_trace"],
        controller_report=payload.get("production_controller_report"),
    )
    repository = payload.get("repository")
    require(
        isinstance(repository, dict)
        and repository.get("commit") == expected_commit
        and repository.get("clean_tracked") is True
        and repository.get("replay_publication_fix_commit")
        == replay.REPLAY_PUBLICATION_FIX_COMMIT
        and repository.get("replay_publication_fix_relation") == "exact_first_parent_v1"
        and repository.get("replay_validation_fix_commit")
        == replay.REPLAY_VALIDATION_FIX_COMMIT
        and repository.get("replay_validation_fix_relation")
        == "exact_first_grandparent_v1"
        and repository.get("replay_implementation_commit")
        == replay.REPLAY_IMPLEMENTATION_COMMIT
        and repository.get("replay_implementation_relation")
        == "exact_first_great_grandparent_v1"
        and repository.get("replay_parent_commit") == replay.REPLAY_PARENT_COMMIT
        and repository.get("replay_parent_relation")
        == "exact_first_great_great_grandparent_v1"
        and repository.get("production_base_commit") == replay.PRODUCTION_BASE_COMMIT
        and repository.get("production_base_relation")
        == "exact_first_great_great_great_grandparent_v1"
        and repository.get("source_trace_polaris_commit")
        == replay.SOURCE_TRACE_POLARIS_COMMIT,
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
        and container.get("size_bytes") == CONTAINER_SIZE_BYTES
        and container.get("sha256") == CONTAINER_SHA256,
        "container identity",
    )
    require(
        payload.get("source_trace_polaris_commit") == replay.SOURCE_TRACE_POLARIS_COMMIT
        and payload.get("source_trace_sha256") == fixture_contract.TRACE_SHA256
        and payload.get("source_action_float32_sha256")
        == replay.ACTION_ENCODING["uncompressed_sha256"]
        and payload.get("action_count") == replay.ACTION_COUNT
        and payload.get("actions_completed") == replay.ACTION_COUNT
        and payload.get("tail_policy_steps_completed") == replay.TAIL_POLICY_STEPS
        and payload.get("tail_physics_substeps_completed")
        == replay.TAIL_PHYSICS_SUBSTEPS
        and payload.get("total_apply_count") == replay.TOTAL_APPLY_COUNT
        and payload.get("numerical_failure") is None
        and payload.get("controller_failure_evidence") is None,
        "exact action/tail completion",
    )
    validate_source_scene_controller_provenance(payload)
    _, source_actions = replay.load_actions()
    expected_tail = replay.frozen_tail_contract(source_actions)
    require(payload.get("tail_contract") == expected_tail, "frozen tail contract")

    entries = payload.get("full_substep_trace")
    require(
        isinstance(entries, list)
        and len(entries) == replay.TOTAL_APPLY_COUNT
        and payload.get("full_substep_trace_profile")
        == (
            "all13_after_arm_before_gripper_post_setters_post_physics_"
            "before_next_arm_v3"
        ),
        "full 13-DOF trace profile/count",
    )
    outcome = payload.get("outcome")
    expected_outcome = replay.classify_outcome(
        None,
        replay.ACTION_COUNT,
        replay.TAIL_POLICY_STEPS,
        len(entries),
    )
    require(
        outcome == expected_outcome
        and outcome.get("replay_completed") is True
        and outcome.get("classification")
        == "production_v4_replay_completed_source_and_tail"
        and outcome.get("failure_binding_valid") is False,
        "zero-failure production replay outcome",
    )
    controller_report = payload.get("production_controller_report")
    ramp_observation = payload.get("production_core_ramp_observation")
    expected_cadence = replay.validate_trace_cadence(
        entries,
        outcome=outcome,
        ramp_observation=ramp_observation,
        controller_report=controller_report,
    )
    require(
        payload.get("full_substep_trace_cadence") == expected_cadence,
        "full-trace cadence report",
    )
    ramp_gate = expected_cadence["release_ramp_trace_gate"]
    require(
        ramp_gate.get("passed") is True
        and ramp_gate.get("entry_count") == 48
        and ramp_gate.get("apply_windows")
        == [list(window) for window in replay.EXPECTED_RAMP_WINDOWS]
        and ramp_gate.get("ramp_indices") == replay.EXPECTED_RAMP_INDICES
        and ramp_gate.get("limited_applies_per_ramp")
        == replay.EXPECTED_LIMITED_APPLIES_PER_RAMP
        and ramp_gate.get("limited_joints_per_ramp")
        == replay.EXPECTED_LIMITED_JOINTS_PER_RAMP
        and ramp_gate.get("aggregate_counts") == replay.EXPECTED_CORE_RAMP_COUNTS
        and ramp_gate.get("observer_write_contract")
        == {
            "target_setter_call_count": 0,
            "failure_trace_write_count": 0,
            "release_ramp_state_write_count": 0,
            "gripper_target_or_state_write_count": 0,
        },
        "exact production core release-ramp gate",
    )
    expected_summary = replay.summarize_trace(entries)
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
    expected_frames = replay.ACTION_COUNT + replay.TAIL_POLICY_STEPS + 1
    require(
        recorded_video.get("frame_count") == expected_frames
        and decoded["frame_count"] == expected_frames,
        "video frame/outcome binding",
    )
    return {
        "schema_version": 1,
        "profile": PROFILE,
        "variant": replay.VARIANT,
        "controller_profile": replay.CONTROLLER_PROFILE,
        "job_id": expected_job_id,
        "launch_id": expected_launch_id,
        "replay_commit": expected_commit,
        "replay_publication_fix_commit": replay.REPLAY_PUBLICATION_FIX_COMMIT,
        "replay_validation_fix_commit": replay.REPLAY_VALIDATION_FIX_COMMIT,
        "replay_implementation_commit": replay.REPLAY_IMPLEMENTATION_COMMIT,
        "replay_parent_commit": replay.REPLAY_PARENT_COMMIT,
        "production_base_commit": replay.PRODUCTION_BASE_COMMIT,
        "source_trace_polaris_commit": replay.SOURCE_TRACE_POLARIS_COMMIT,
        "result": result_identity,
        "video": video_identity,
        "video_decode": decoded,
        "outcome": outcome,
        "production_controller_report": controller_report,
        "production_core_ramp_gate": ramp_gate,
        "full_substep_summary": expected_summary,
        "full_substep_trace_cadence": expected_cadence,
        "tail_contract": expected_tail,
        "runtime_close": close,
        "post_kit_validator_sources": validator_sources,
        "production_all_six_gripper_trace_validation": gripper_validation,
        "production_safety_validation": safety_validation,
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
    original = Path(path)
    require(
        not original.exists() and not original.is_symlink(),
        f"refusing to overwrite {original}",
    )
    original.parent.mkdir(parents=True, exist_ok=True)
    parent = original.parent.resolve(strict=True)
    require(
        parent.is_dir() and not parent.is_symlink(), f"invalid output parent {parent}"
    )
    path = parent / original.name
    require(
        not path.exists() and not path.is_symlink(), f"refusing to overwrite {path}"
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    directory_descriptor = os.open(
        parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    identity = file_identity(path)
    require(
        identity["mode"] == "0444" and identity["nlink"] == 1,
        "published manifest mode/link",
    )
    return identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-replay-commit", required=True)
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
        len(args.expected_replay_commit) == 40
        and all(
            character in "0123456789abcdef" for character in args.expected_replay_commit
        ),
        "replay commit grammar",
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
        expected_commit=args.expected_replay_commit,
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
        f"POLARIS_PRODUCTION_V4_CORE_REPLAY_VALIDATED={identity['path']};"
        f"size={identity['size_bytes']};sha256={identity['sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
