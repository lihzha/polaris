#!/usr/bin/env python3
"""Validate and finalize one post-Kit reasoning full-trace ablation."""

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


PROFILE = "reasoning_43075_fulltrace_ablation_post_kit_validation_v3"
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
    "numerical_failure",
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


def validate_intervention(value: Any, *, variant: str) -> dict[str, Any]:
    require(
        isinstance(value, dict)
        and set(value)
        == {
            "profile",
            "variant",
            "production_contract_acceptance",
            "force_gripper_open",
            "passive_follower_velocity_limit_setter_call_count",
            "velocity_limits_before_intervention",
            "velocity_limits_after_intervention",
            "configured_close_anchor_substeps",
        }
        and value.get("profile")
        == "post_production_installer_single_variable_diagnostic_only_v1"
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
    if variant == "follower_default_limit":
        expected_after[8:] = [diagnostic.FOLLOWER_DEFAULT_LIMIT] * 5
    require(
        all(
            float32_equal(actual, expected)
            for actual, expected in zip(after, expected_after, strict=True)
        ),
        "variant follower-limit output",
    )
    require(
        value.get("force_gripper_open") is (variant == "force_open")
        and value.get("passive_follower_velocity_limit_setter_call_count")
        == int(variant == "follower_default_limit")
        and value.get("configured_close_anchor_substeps")
        == (
            diagnostic.HOLD_CLOSE_ANCHOR_SUBSTEPS
            if variant == "hold_close_anchor"
            else 86
        ),
        "single-variable intervention identity",
    )
    return dict(value)


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
        and repository.get("production_base_commit")
        == diagnostic.PRODUCTION_BASE_COMMIT
        and repository.get("production_base_relation") == "exact_first_parent_v1",
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
    validate_intervention(payload.get("intervention"), variant=variant)
    outcome = payload.get("outcome")
    require(
        isinstance(outcome, dict)
        and outcome.get("diagnostic_completed") is True
        and outcome.get("controller_completed_actions")
        == payload.get("actions_completed"),
        "diagnostic outcome",
    )
    if variant == "baseline":
        require(
            outcome.get("classification") == "baseline_exact_source_abort_reproduced"
            and outcome.get("parsed_numerical_failure")
            == fixture_contract.EXPECTED_FAILURE
            and payload.get("actions_completed") == 293,
            "baseline reproduction outcome",
        )
    else:
        require(
            outcome.get("classification")
            in {
                "ablation_completed_all_recorded_actions",
                "ablation_numerical_failure_observed",
            },
            "novel ablation outcome classification",
        )
    entries = payload.get("full_substep_trace")
    require(isinstance(entries, list), "full substep trace")
    expected_cadence = diagnostic.validate_trace_cadence(
        entries, variant=variant, outcome=outcome
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
    failure = payload.get("numerical_failure")
    expected_frames = payload["actions_completed"] + (2 if failure is not None else 1)
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
        "full_substep_summary": expected_summary,
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
        f"POLARIS_FULLTRACE_VALIDATED={identity['path']};"
        f"size={identity['size_bytes']};sha256={identity['sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
