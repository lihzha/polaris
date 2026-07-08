"""Pinned full-decode evidence for official pi0.5 joint-position videos.

The PolaRiS host does not provide media tools.  This module is executed in the
same immutable Pyxis image as the evaluator, where it verifies the exact
``ffprobe`` and ``ffmpeg`` binaries, probes every rollout video, and performs a
complete error-fatal decode.  The host close transaction later seals the videos
and requires their identities to equal this immutable report.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
from typing import Any

from polaris.pi05_droid_jointpos_immutable import (
    file_sha256,
    publish_immutable_json,
    validate_immutable_json,
)


PI05_DROID_JOINTPOS_VIDEO_PROFILE = "openpi_pi05_droid_jointpos_polaris_full_decode_v2"
PI05_DROID_JOINTPOS_VIDEO_FILENAME = "pi05_droid_jointpos_video_validation.json"
PI05_DROID_JOINTPOS_PYXIS_SHA256 = (
    "ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a"
)
PI05_DROID_JOINTPOS_VIDEO_CONTRACT = {
    "codec": "h264",
    "pixel_format": "yuv420p",
    "field_order": "progressive",
    "width": 448,
    "height": 224,
    "frame_count": 450,
    "fps_numerator": 15,
    "fps_denominator": 1,
    "duration_seconds": 30.0,
}
PI05_DROID_JOINTPOS_TERMINAL_IMAGE_CONTRACT = {
    "codec": "png",
    "pixel_format": "rgb24",
    "width": 448,
    "height": 224,
    "decoded_frame_count": 1,
    "decoded_rgb24_bytes": 448 * 224 * 3,
    "source": "post_action450_returned_expensive_splat_observation",
}
PI05_DROID_JOINTPOS_MEDIA_TOOLS = {
    "ffprobe": {
        "container_path": "/usr/bin/ffprobe",
        "size": 178_832,
        "sha256": ("d4f3ef9c12be756793cad83dd2004d89f49c1c4094053bfbbe7e28925c8fa4fd"),
    },
    "ffmpeg": {
        "container_path": "/usr/bin/ffmpeg",
        "size": 301_544,
        "sha256": ("36d94a605d612e4090d1b8aec889d0c0801c6eafb1593c90f5c0dfd2e2966a45"),
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _strict_json_bytes(payload: bytes, field: str) -> Any:
    try:
        return json.loads(
            payload,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{field} contains non-finite JSON: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} is not strict JSON") from error


def _lower_sha256(value: Any, field: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{field} must be one lowercase SHA-256",
    )
    return value


def _regular_single_link(path: Path, field: str) -> os.stat_result:
    if path.is_symlink():
        raise ValueError(f"{field} must not be a symlink")
    try:
        result = path.stat()
    except OSError as error:
        raise ValueError(f"{field} is unavailable") from error
    if not stat.S_ISREG(result.st_mode) or result.st_nlink != 1:
        raise ValueError(f"{field} must be one regular link")
    if result.st_size <= 0:
        raise ValueError(f"{field} must be nonempty")
    return result


def _stable_identity(result: os.stat_result) -> tuple[int, ...]:
    return (
        result.st_dev,
        result.st_ino,
        result.st_mode,
        result.st_nlink,
        result.st_size,
        result.st_mtime_ns,
        result.st_ctime_ns,
    )


def _subprocess_environment() -> dict[str, str]:
    return {**os.environ, "LANG": "C", "LC_ALL": "C"}


def attest_media_tools() -> dict[str, Any]:
    """Require exact media binaries from the pinned Pyxis root filesystem."""

    tools: dict[str, Any] = {}
    for name in ("ffprobe", "ffmpeg"):
        expected = PI05_DROID_JOINTPOS_MEDIA_TOOLS[name]
        path = Path(expected["container_path"])
        if path.resolve(strict=True) != path:
            raise ValueError(f"Pinned {name} path must not traverse a symlink")
        before = _regular_single_link(path, f"pinned {name}")
        if (
            stat.S_IMODE(before.st_mode) != 0o755
            or before.st_size != expected["size"]
            or not os.access(path, os.X_OK)
        ):
            raise ValueError(f"Pinned {name} mode/size mismatch")
        digest = file_sha256(path)
        if digest != expected["sha256"]:
            raise ValueError(f"Pinned {name} SHA-256 mismatch")
        try:
            completed = subprocess.run(
                [str(path), "-version"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
                env=_subprocess_environment(),
            )
        except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as error:
            raise ValueError(f"Pinned {name} version probe failed") from error
        version_line = completed.stdout.splitlines()[0] if completed.stdout else ""
        if not version_line.startswith(f"{name} version "):
            raise ValueError(f"Pinned {name} version line mismatch")
        after = _regular_single_link(path, f"pinned {name}")
        if (
            _stable_identity(after) != _stable_identity(before)
            or file_sha256(path) != digest
        ):
            raise ValueError(f"Pinned {name} changed during attestation")
        tools[name] = {
            **expected,
            "mode": "0755",
            "nlink": 1,
            "version_line": version_line,
        }
    return {
        "profile": "pinned_polaris_pyxis_image_media_tools_v1",
        "pyxis_image_sha256": PI05_DROID_JOINTPOS_PYXIS_SHA256,
        "root_filesystem_source": "immutable_enroot_squashfs_plus_ephemeral_overlay",
        "tools": tools,
    }


def _run_ffprobe(video_path: Path) -> dict[str, Any]:
    ffprobe = PI05_DROID_JOINTPOS_MEDIA_TOOLS["ffprobe"]["container_path"]
    command = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        (
            "stream=codec_type,codec_name,pix_fmt,field_order,width,height,"
            "avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration:"
            "format=format_name,duration"
        ),
        "-of",
        "json",
        str(video_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            timeout=600,
            env=_subprocess_environment(),
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"ffprobe failed for {video_path}") from error
    if completed.stderr.strip():
        raise ValueError(f"ffprobe emitted an error for {video_path}")
    value = _strict_json_bytes(completed.stdout, f"ffprobe output for {video_path}")
    if (
        not isinstance(value, dict)
        or not {"streams", "format"}.issubset(value)
        or not set(value).issubset({"streams", "format", "programs", "stream_groups"})
        or any(
            value.get(name) != []
            for name in ("programs", "stream_groups")
            if name in value
        )
    ):
        raise ValueError(f"ffprobe schema mismatch for {video_path}")
    streams = value["streams"]
    if not isinstance(streams, list) or len(streams) != 1:
        raise ValueError(f"Expected exactly one video stream in {video_path}")
    stream = streams[0]
    format_value = value["format"]
    required_stream = {
        "codec_type",
        "codec_name",
        "pix_fmt",
        "field_order",
        "width",
        "height",
        "avg_frame_rate",
        "r_frame_rate",
        "nb_frames",
        "nb_read_frames",
        "duration",
    }
    if (
        not isinstance(stream, dict)
        or set(stream) != required_stream
        or not isinstance(format_value, dict)
        or set(format_value) != {"format_name", "duration"}
    ):
        raise ValueError(f"ffprobe video fields are incomplete for {video_path}")
    try:
        average_rate = Fraction(stream["avg_frame_rate"])
        real_rate = Fraction(stream["r_frame_rate"])
        frame_count = int(stream["nb_frames"])
        decoded_frame_count = int(stream["nb_read_frames"])
        stream_duration = float(stream["duration"])
        format_duration = float(format_value["duration"])
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(
            f"ffprobe numeric fields are invalid for {video_path}"
        ) from error
    expected = PI05_DROID_JOINTPOS_VIDEO_CONTRACT
    if (
        stream["codec_type"] != "video"
        or stream["codec_name"] != expected["codec"]
        or stream["pix_fmt"] != expected["pixel_format"]
        or stream["field_order"] != expected["field_order"]
        or stream["width"] != expected["width"]
        or stream["height"] != expected["height"]
        or average_rate
        != Fraction(expected["fps_numerator"], expected["fps_denominator"])
        or real_rate != average_rate
        or frame_count != expected["frame_count"]
        or decoded_frame_count != expected["frame_count"]
        or not math.isclose(
            stream_duration, expected["duration_seconds"], rel_tol=0.0, abs_tol=1e-6
        )
        or not math.isclose(
            format_duration, expected["duration_seconds"], rel_tol=0.0, abs_tol=1e-6
        )
        or "mp4" not in format_value["format_name"].split(",")
    ):
        raise ValueError(f"Video probe contract mismatch for {video_path}")
    return {
        "format_name": format_value["format_name"],
        "codec": stream["codec_name"],
        "pixel_format": stream["pix_fmt"],
        "field_order": stream["field_order"],
        "width": stream["width"],
        "height": stream["height"],
        "average_frame_rate": stream["avg_frame_rate"],
        "real_frame_rate": stream["r_frame_rate"],
        "container_frame_count": frame_count,
        "decoded_probe_frame_count": decoded_frame_count,
        "stream_duration_seconds": stream_duration,
        "format_duration_seconds": format_duration,
    }


def _run_full_decode(video_path: Path) -> dict[str, Any]:
    ffmpeg = PI05_DROID_JOINTPOS_MEDIA_TOOLS["ffmpeg"]["container_path"]
    command = [
        ffmpeg,
        "-v",
        "error",
        "-xerror",
        "-nostdin",
        "-progress",
        "pipe:1",
        "-nostats",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vsync",
        "0",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
            env=_subprocess_environment(),
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"Full ffmpeg decode failed for {video_path}") from error
    if completed.stderr.strip():
        raise ValueError(f"Full ffmpeg decode emitted an error for {video_path}")
    frames = []
    progress = []
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if not separator:
            continue
        if key == "frame":
            try:
                frames.append(int(value))
            except ValueError as error:
                raise ValueError(
                    f"Invalid ffmpeg frame count for {video_path}"
                ) from error
        elif key == "progress":
            progress.append(value)
    if (
        not frames
        or frames[-1] != PI05_DROID_JOINTPOS_VIDEO_CONTRACT["frame_count"]
        or not progress
        or progress[-1] != "end"
    ):
        raise ValueError(f"Full ffmpeg decode did not consume 450 frames: {video_path}")
    return {
        "profile": "ffmpeg_error_fatal_full_video_decode_to_null_v1",
        "status": "pass",
        "decoded_frame_count": frames[-1],
        "progress_terminal": progress[-1],
    }


def inspect_video(video_path: Path, episode_index: int) -> dict[str, Any]:
    video_path = Path(video_path)
    if video_path.is_symlink():
        raise ValueError(f"Episode {episode_index} video must not be a symlink")
    video_path = video_path.resolve(strict=True)
    before = _regular_single_link(video_path, f"episode {episode_index} video")
    digest_before = file_sha256(video_path)
    probe = _run_ffprobe(video_path)
    full_decode = _run_full_decode(video_path)
    after = _regular_single_link(video_path, f"episode {episode_index} video")
    digest_after = file_sha256(video_path)
    if (
        _stable_identity(after) != _stable_identity(before)
        or digest_after != digest_before
    ):
        raise ValueError(f"Episode {episode_index} video changed during validation")
    return {
        "episode_index": episode_index,
        "artifact": {
            "path": str(video_path),
            "size": after.st_size,
            "sha256": digest_after,
        },
        "probe": probe,
        "full_decode": full_decode,
    }


def _inspect_terminal_image(image_path: Path, episode_index: int) -> dict[str, Any]:
    """Probe and fully decode one trace-bound post-action-450 PNG."""

    image_path = Path(image_path)
    if image_path.is_symlink():
        raise ValueError(
            f"Episode {episode_index} terminal image must not be a symlink"
        )
    image_path = image_path.resolve(strict=True)
    before = _regular_single_link(image_path, f"episode {episode_index} terminal image")
    digest_before = file_sha256(image_path)
    ffprobe = PI05_DROID_JOINTPOS_MEDIA_TOOLS["ffprobe"]["container_path"]
    probe_command = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        (
            "stream=codec_type,codec_name,pix_fmt,width,height,nb_read_frames:"
            "format=format_name"
        ),
        "-of",
        "json",
        str(image_path),
    ]
    try:
        probed = subprocess.run(
            probe_command,
            check=True,
            capture_output=True,
            timeout=120,
            env=_subprocess_environment(),
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"ffprobe failed for terminal image {image_path}") from error
    if probed.stderr.strip():
        raise ValueError(f"ffprobe emitted an error for terminal image {image_path}")
    probe_value = _strict_json_bytes(
        probed.stdout, f"ffprobe output for terminal image {image_path}"
    )
    if (
        not isinstance(probe_value, dict)
        or not {"streams", "format"}.issubset(probe_value)
        or not set(probe_value).issubset(
            {"streams", "format", "programs", "stream_groups"}
        )
        or any(
            probe_value.get(name) != []
            for name in ("programs", "stream_groups")
            if name in probe_value
        )
        or not isinstance(probe_value["streams"], list)
        or len(probe_value["streams"]) != 1
    ):
        raise ValueError(f"Terminal PNG ffprobe schema mismatch: {image_path}")
    stream = probe_value["streams"][0]
    format_value = probe_value["format"]
    expected = PI05_DROID_JOINTPOS_TERMINAL_IMAGE_CONTRACT
    if (
        not isinstance(stream, dict)
        or set(stream)
        != {
            "codec_type",
            "codec_name",
            "pix_fmt",
            "width",
            "height",
            "nb_read_frames",
        }
        or stream
        != {
            "codec_type": "video",
            "codec_name": expected["codec"],
            "pix_fmt": expected["pixel_format"],
            "width": expected["width"],
            "height": expected["height"],
            "nb_read_frames": str(expected["decoded_frame_count"]),
        }
        or not isinstance(format_value, dict)
        or set(format_value) != {"format_name"}
        or format_value["format_name"] != "png_pipe"
    ):
        raise ValueError(f"Terminal PNG probe contract mismatch: {image_path}")

    ffmpeg = PI05_DROID_JOINTPOS_MEDIA_TOOLS["ffmpeg"]["container_path"]
    decode_command = [
        ffmpeg,
        "-v",
        "error",
        "-xerror",
        "-nostdin",
        "-i",
        str(image_path),
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    try:
        decoded = subprocess.run(
            decode_command,
            check=True,
            capture_output=True,
            timeout=120,
            env=_subprocess_environment(),
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError(
            f"Full decode failed for terminal image {image_path}"
        ) from error
    if decoded.stderr.strip() or len(decoded.stdout) != expected["decoded_rgb24_bytes"]:
        raise ValueError(f"Terminal PNG decoded RGB24 contract mismatch: {image_path}")

    after = _regular_single_link(image_path, f"episode {episode_index} terminal image")
    digest_after = file_sha256(image_path)
    if (
        _stable_identity(after) != _stable_identity(before)
        or digest_after != digest_before
    ):
        raise ValueError(
            f"Episode {episode_index} terminal image changed during decode"
        )
    return {
        "episode_index": episode_index,
        "artifact": {
            "path": str(image_path),
            "size": after.st_size,
            "sha256": digest_after,
        },
        "probe": {
            "format_name": format_value["format_name"],
            "codec": stream["codec_name"],
            "pixel_format": stream["pix_fmt"],
            "width": stream["width"],
            "height": stream["height"],
            "decoded_frame_count": int(stream["nb_read_frames"]),
        },
        "full_decode": {
            "profile": "ffmpeg_error_fatal_png_to_rgb24_v1",
            "status": "pass",
            "decoded_rgb24_bytes": len(decoded.stdout),
            "decoded_rgb24_sha256": hashlib.sha256(decoded.stdout).hexdigest(),
        },
    }


def validate_video_report(
    value: Any,
    *,
    expected_rollouts: int,
    expected_video_identities: list[dict[str, Any]] | None = None,
    expected_terminal_image_identities: list[dict[str, Any]] | None = None,
    expected_terminal_pixel_sha256: list[str] | None = None,
) -> dict[str, Any]:
    """Validate a closed media report and optionally join it to sealed videos."""

    if type(expected_rollouts) is not int or expected_rollouts <= 0:
        raise ValueError("Expected video rollouts must be one positive integer")
    required = {
        "schema_version",
        "profile",
        "status",
        "execution_environment",
        "video_contract",
        "terminal_image_contract",
        "expected_rollouts",
        "videos",
        "terminal_images",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Joint-position video report schema mismatch")
    if (
        value["schema_version"] != 1
        or value["profile"] != PI05_DROID_JOINTPOS_VIDEO_PROFILE
        or value["status"] != "pass"
        or value["video_contract"] != PI05_DROID_JOINTPOS_VIDEO_CONTRACT
        or value["terminal_image_contract"]
        != PI05_DROID_JOINTPOS_TERMINAL_IMAGE_CONTRACT
        or value["expected_rollouts"] != expected_rollouts
    ):
        raise ValueError("Joint-position video report identity mismatch")
    execution = value["execution_environment"]
    if (
        not isinstance(execution, dict)
        or set(execution)
        != {
            "profile",
            "pyxis_image_sha256",
            "root_filesystem_source",
            "tools",
        }
        or execution["profile"] != "pinned_polaris_pyxis_image_media_tools_v1"
        or execution["pyxis_image_sha256"] != PI05_DROID_JOINTPOS_PYXIS_SHA256
        or execution["root_filesystem_source"]
        != "immutable_enroot_squashfs_plus_ephemeral_overlay"
    ):
        raise ValueError("Joint-position video execution environment mismatch")
    tools = execution["tools"]
    if not isinstance(tools, dict) or set(tools) != {"ffprobe", "ffmpeg"}:
        raise ValueError("Joint-position video media-tool inventory mismatch")
    for name, expected in PI05_DROID_JOINTPOS_MEDIA_TOOLS.items():
        tool = tools[name]
        if (
            not isinstance(tool, dict)
            or set(tool)
            != {"container_path", "size", "sha256", "mode", "nlink", "version_line"}
            or {key: tool[key] for key in expected} != expected
            or tool["mode"] != "0755"
            or tool["nlink"] != 1
            or not isinstance(tool["version_line"], str)
            or not tool["version_line"].startswith(f"{name} version ")
        ):
            raise ValueError(f"Joint-position pinned {name} identity mismatch")
    videos = value["videos"]
    if not isinstance(videos, list) or len(videos) != expected_rollouts:
        raise ValueError("Joint-position video report count mismatch")
    if (
        expected_video_identities is not None
        and len(expected_video_identities) != expected_rollouts
    ):
        raise ValueError("Sealed video identity count mismatch")
    if (
        expected_terminal_image_identities is not None
        and len(expected_terminal_image_identities) != expected_rollouts
    ):
        raise ValueError("Sealed terminal-image identity count mismatch")
    if (
        expected_terminal_pixel_sha256 is not None
        and len(expected_terminal_pixel_sha256) != expected_rollouts
    ):
        raise ValueError("Trace terminal-image digest count mismatch")
    for index, video in enumerate(videos):
        if not isinstance(video, dict) or set(video) != {
            "episode_index",
            "artifact",
            "probe",
            "full_decode",
        }:
            raise ValueError(f"Episode {index} video record schema mismatch")
        artifact = video["artifact"]
        if (
            video["episode_index"] != index
            or not isinstance(artifact, dict)
            or set(artifact) != {"path", "size", "sha256"}
            or not isinstance(artifact["path"], str)
            or Path(artifact["path"]).name != f"episode_{index}.mp4"
            or type(artifact["size"]) is not int
            or artifact["size"] <= 0
        ):
            raise ValueError(f"Episode {index} video artifact mismatch")
        _lower_sha256(artifact["sha256"], f"episode {index} video")
        expected_probe = {
            "codec": "h264",
            "pixel_format": "yuv420p",
            "field_order": "progressive",
            "width": 448,
            "height": 224,
            "average_frame_rate": "15/1",
            "real_frame_rate": "15/1",
            "container_frame_count": 450,
            "decoded_probe_frame_count": 450,
            "stream_duration_seconds": 30.0,
            "format_duration_seconds": 30.0,
        }
        probe = video["probe"]
        if (
            not isinstance(probe, dict)
            or set(probe) != {"format_name", *expected_probe}
            or {key: probe[key] for key in expected_probe} != expected_probe
            or not isinstance(probe["format_name"], str)
            or "mp4" not in probe["format_name"].split(",")
        ):
            raise ValueError(f"Episode {index} video probe mismatch")
        if video["full_decode"] != {
            "profile": "ffmpeg_error_fatal_full_video_decode_to_null_v1",
            "status": "pass",
            "decoded_frame_count": 450,
            "progress_terminal": "end",
        }:
            raise ValueError(f"Episode {index} full-decode evidence mismatch")
        if expected_video_identities is not None:
            sealed = expected_video_identities[index]
            if not isinstance(sealed, dict) or any(
                artifact[key] != sealed.get(key) for key in ("path", "size", "sha256")
            ):
                raise ValueError(
                    f"Episode {index} decode identity differs from sealed video"
                )
    terminal_images = value["terminal_images"]
    if (
        not isinstance(terminal_images, list)
        or len(terminal_images) != expected_rollouts
    ):
        raise ValueError("Joint-position terminal-image report count mismatch")
    terminal_contract = PI05_DROID_JOINTPOS_TERMINAL_IMAGE_CONTRACT
    for index, terminal in enumerate(terminal_images):
        if not isinstance(terminal, dict) or set(terminal) != {
            "episode_index",
            "artifact",
            "probe",
            "full_decode",
        }:
            raise ValueError(f"Episode {index} terminal-image record schema mismatch")
        artifact = terminal["artifact"]
        if (
            terminal["episode_index"] != index
            or not isinstance(artifact, dict)
            or set(artifact) != {"path", "size", "sha256"}
            or not isinstance(artifact["path"], str)
            or Path(artifact["path"]).name != f"episode_{index}_terminal.png"
            or type(artifact["size"]) is not int
            or artifact["size"] <= 0
        ):
            raise ValueError(f"Episode {index} terminal-image artifact mismatch")
        _lower_sha256(artifact["sha256"], f"episode {index} terminal image")
        if terminal["probe"] != {
            "format_name": "png_pipe",
            "codec": terminal_contract["codec"],
            "pixel_format": terminal_contract["pixel_format"],
            "width": terminal_contract["width"],
            "height": terminal_contract["height"],
            "decoded_frame_count": terminal_contract["decoded_frame_count"],
        }:
            raise ValueError(f"Episode {index} terminal-image probe mismatch")
        full_decode = terminal["full_decode"]
        if (
            not isinstance(full_decode, dict)
            or set(full_decode)
            != {
                "profile",
                "status",
                "decoded_rgb24_bytes",
                "decoded_rgb24_sha256",
            }
            or full_decode["profile"] != "ffmpeg_error_fatal_png_to_rgb24_v1"
            or full_decode["status"] != "pass"
            or full_decode["decoded_rgb24_bytes"]
            != terminal_contract["decoded_rgb24_bytes"]
        ):
            raise ValueError(f"Episode {index} terminal-image decode mismatch")
        _lower_sha256(
            full_decode["decoded_rgb24_sha256"],
            f"episode {index} terminal decoded pixels",
        )
        if expected_terminal_image_identities is not None:
            sealed = expected_terminal_image_identities[index]
            if not isinstance(sealed, dict) or any(
                artifact[key] != sealed.get(key) for key in ("path", "size", "sha256")
            ):
                raise ValueError(
                    f"Episode {index} decoded terminal image differs from sealed file"
                )
        if (
            expected_terminal_pixel_sha256 is not None
            and full_decode["decoded_rgb24_sha256"]
            != expected_terminal_pixel_sha256[index]
        ):
            raise ValueError(
                f"Episode {index} terminal image differs from action-450 trace"
            )
    return json.loads(json.dumps(value, allow_nan=False))


def build_video_report(
    task_dir: Path,
    *,
    expected_rollouts: int,
    container_image_sha256: str,
) -> dict[str, Any]:
    if type(expected_rollouts) is not int or expected_rollouts <= 0:
        raise ValueError("Expected video rollouts must be one positive integer")
    if container_image_sha256 != PI05_DROID_JOINTPOS_PYXIS_SHA256:
        raise ValueError("Video validator Pyxis image SHA-256 mismatch")
    task_dir = Path(task_dir)
    if task_dir.is_symlink():
        raise ValueError("Video task directory is invalid")
    task_dir = task_dir.resolve(strict=True)
    if not task_dir.is_dir():
        raise ValueError("Video task directory is invalid")
    expected_paths = [
        task_dir / f"episode_{index}.mp4" for index in range(expected_rollouts)
    ]
    expected_terminal_paths = [
        task_dir / f"episode_{index}_terminal.png" for index in range(expected_rollouts)
    ]
    if set(task_dir.glob("episode_*.mp4")) != set(expected_paths):
        raise ValueError("Video filename/count mismatch")
    if set(task_dir.glob("episode_*_terminal.png")) != set(expected_terminal_paths):
        raise ValueError("Terminal-image filename/count mismatch")
    report = {
        "schema_version": 1,
        "profile": PI05_DROID_JOINTPOS_VIDEO_PROFILE,
        "status": "pass",
        "execution_environment": attest_media_tools(),
        "video_contract": dict(PI05_DROID_JOINTPOS_VIDEO_CONTRACT),
        "terminal_image_contract": dict(PI05_DROID_JOINTPOS_TERMINAL_IMAGE_CONTRACT),
        "expected_rollouts": expected_rollouts,
        "videos": [
            inspect_video(path, index) for index, path in enumerate(expected_paths)
        ],
        "terminal_images": [
            _inspect_terminal_image(path, index)
            for index, path in enumerate(expected_terminal_paths)
        ],
    }
    return validate_video_report(report, expected_rollouts=expected_rollouts)


def publish_video_report(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    if Path(path).name != PI05_DROID_JOINTPOS_VIDEO_FILENAME:
        raise ValueError("Unexpected joint-position video report filename")
    if not isinstance(value, dict):
        raise ValueError("Joint-position video report must be one object")
    canonical = validate_video_report(
        value, expected_rollouts=value.get("expected_rollouts", 0)
    )
    return publish_immutable_json(Path(path), canonical)


def validate_persisted_video_report(
    path: Path,
    *,
    expected_rollouts: int,
    expected_video_identities: list[dict[str, Any]] | None = None,
    expected_terminal_image_identities: list[dict[str, Any]] | None = None,
    expected_terminal_pixel_sha256: list[str] | None = None,
) -> dict[str, Any]:
    if Path(path).name != PI05_DROID_JOINTPOS_VIDEO_FILENAME:
        raise ValueError("Unexpected joint-position video report filename")
    artifact = validate_immutable_json(Path(path))
    artifact["value"] = validate_video_report(
        artifact["value"],
        expected_rollouts=expected_rollouts,
        expected_video_identities=expected_video_identities,
        expected_terminal_image_identities=expected_terminal_image_identities,
        expected_terminal_pixel_sha256=expected_terminal_pixel_sha256,
    )
    return artifact


__all__ = [
    "PI05_DROID_JOINTPOS_MEDIA_TOOLS",
    "PI05_DROID_JOINTPOS_PYXIS_SHA256",
    "PI05_DROID_JOINTPOS_TERMINAL_IMAGE_CONTRACT",
    "PI05_DROID_JOINTPOS_VIDEO_CONTRACT",
    "PI05_DROID_JOINTPOS_VIDEO_FILENAME",
    "PI05_DROID_JOINTPOS_VIDEO_PROFILE",
    "attest_media_tools",
    "build_video_report",
    "inspect_video",
    "publish_video_report",
    "validate_persisted_video_report",
    "validate_video_report",
]
