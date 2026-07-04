#!/usr/bin/env python3
"""Validate and finalize one exact production-v4 core full-trace replay."""

from __future__ import annotations

import argparse
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
import smoke_eef_pose_reasoning_production_v4_core_replay as replay


PROFILE = "reasoning_43075_production_v4_core_release_ramp_post_kit_v1"
CONTAINER_SIZE_BYTES = 7_183_130_624
CONTAINER_SHA256 = "ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a"
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


def float32_equal(left: float, right: float) -> bool:
    return struct.pack("<f", float(left)) == struct.pack("<f", float(right))


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
    repository = payload.get("repository")
    require(
        isinstance(repository, dict)
        and repository.get("commit") == expected_commit
        and repository.get("clean_tracked") is True
        and repository.get("replay_parent_commit") == replay.REPLAY_PARENT_COMMIT
        and repository.get("replay_parent_relation") == "exact_first_parent_v1"
        and repository.get("production_base_commit") == replay.PRODUCTION_BASE_COMMIT
        and repository.get("production_base_relation") == "exact_first_grandparent_v1"
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
    safety = payload.get("production_safety")
    require(
        isinstance(safety, dict)
        and safety.get("counters", {}).get("apply_calls") == replay.TOTAL_APPLY_COUNT
        and safety.get("counters", {}).get("current_joint_limit_aborts") == 0
        and safety.get("counters", {}).get("invariant_aborts") == 0
        and safety.get("counters", {}).get("nonfinite_aborts") == 0
        and safety.get("counters", {}).get("dls_fallbacks") == 0
        and safety.get("counters", {}).get("post_clamp_target_violations") == 0
        and safety.get("counters", {}).get("guard_diagnostics_dropped") == 0,
        "production safety zero-failure counters",
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
