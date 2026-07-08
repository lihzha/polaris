import copy
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

import polaris.pi05_droid_jointpos_video as video


def _valid_report(task_dir: Path, rollouts: int = 1):
    tools = {}
    for name, identity in video.PI05_DROID_JOINTPOS_MEDIA_TOOLS.items():
        tools[name] = {
            **identity,
            "mode": "0755",
            "nlink": 1,
            "version_line": f"{name} version pinned-test",
        }
    records = []
    identities = []
    terminal_records = []
    terminal_identities = []
    terminal_pixel_hashes = []
    for index in range(rollouts):
        path = (task_dir / f"episode_{index}.mp4").resolve()
        digest = hashlib.sha256(f"video-{index}".encode()).hexdigest()
        artifact = {"path": str(path), "size": 100 + index, "sha256": digest}
        identities.append({**artifact, "mode": "0444", "nlink": 1})
        records.append(
            {
                "episode_index": index,
                "artifact": artifact,
                "probe": {
                    "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
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
                },
                "full_decode": {
                    "profile": "ffmpeg_error_fatal_full_video_decode_to_null_v1",
                    "status": "pass",
                    "decoded_frame_count": 450,
                    "progress_terminal": "end",
                },
            }
        )
        terminal_path = (task_dir / f"episode_{index}_terminal.png").resolve()
        terminal_digest = hashlib.sha256(f"terminal-{index}".encode()).hexdigest()
        terminal_pixel_digest = hashlib.sha256(
            f"terminal-pixels-{index}".encode()
        ).hexdigest()
        terminal_artifact = {
            "path": str(terminal_path),
            "size": 200 + index,
            "sha256": terminal_digest,
        }
        terminal_identities.append({**terminal_artifact, "mode": "0444", "nlink": 1})
        terminal_pixel_hashes.append(terminal_pixel_digest)
        terminal_records.append(
            {
                "episode_index": index,
                "artifact": terminal_artifact,
                "probe": {
                    "format_name": "png_pipe",
                    "codec": "png",
                    "pixel_format": "rgb24",
                    "width": 448,
                    "height": 224,
                    "decoded_frame_count": 1,
                },
                "full_decode": {
                    "profile": "ffmpeg_error_fatal_png_to_rgb24_v1",
                    "status": "pass",
                    "decoded_rgb24_bytes": 448 * 224 * 3,
                    "decoded_rgb24_sha256": terminal_pixel_digest,
                },
            }
        )
    return (
        {
            "schema_version": 1,
            "profile": video.PI05_DROID_JOINTPOS_VIDEO_PROFILE,
            "status": "pass",
            "execution_environment": {
                "profile": "pinned_polaris_pyxis_image_media_tools_v1",
                "pyxis_image_sha256": video.PI05_DROID_JOINTPOS_PYXIS_SHA256,
                "root_filesystem_source": (
                    "immutable_enroot_squashfs_plus_ephemeral_overlay"
                ),
                "tools": tools,
            },
            "video_contract": dict(video.PI05_DROID_JOINTPOS_VIDEO_CONTRACT),
            "terminal_image_contract": dict(
                video.PI05_DROID_JOINTPOS_TERMINAL_IMAGE_CONTRACT
            ),
            "expected_rollouts": rollouts,
            "videos": records,
            "terminal_images": terminal_records,
        },
        identities,
        terminal_identities,
        terminal_pixel_hashes,
    )


def test_video_report_binds_pinned_image_tools_and_sealed_video_identity(tmp_path):
    report, identities, terminal_identities, terminal_pixel_hashes = _valid_report(
        tmp_path
    )
    assert (
        video.validate_video_report(
            report,
            expected_rollouts=1,
            expected_video_identities=identities,
            expected_terminal_image_identities=terminal_identities,
            expected_terminal_pixel_sha256=terminal_pixel_hashes,
        )
        == report
    )
    output = tmp_path / video.PI05_DROID_JOINTPOS_VIDEO_FILENAME
    video.publish_video_report(output, report)
    artifact = video.validate_persisted_video_report(
        output,
        expected_rollouts=1,
        expected_video_identities=identities,
        expected_terminal_image_identities=terminal_identities,
        expected_terminal_pixel_sha256=terminal_pixel_hashes,
    )
    assert artifact["mode"] == "0444"
    assert artifact["value"] == report


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("videos", 0, "probe", "codec"), "hevc", "video probe mismatch"),
        (
            ("videos", 0, "full_decode", "decoded_frame_count"),
            449,
            "full-decode evidence mismatch",
        ),
        (
            ("execution_environment", "pyxis_image_sha256"),
            "0" * 64,
            "execution environment mismatch",
        ),
    ],
)
def test_video_report_rejects_media_contract_drift(tmp_path, path, value, match):
    report, identities, terminal_identities, terminal_pixel_hashes = _valid_report(
        tmp_path
    )
    mutated = copy.deepcopy(report)
    target = mutated
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        video.validate_video_report(
            mutated,
            expected_rollouts=1,
            expected_video_identities=identities,
            expected_terminal_image_identities=terminal_identities,
            expected_terminal_pixel_sha256=terminal_pixel_hashes,
        )


def test_video_report_rejects_decode_identity_different_from_sealed_video(tmp_path):
    report, identities, terminal_identities, terminal_pixel_hashes = _valid_report(
        tmp_path
    )
    identities[0]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="differs from sealed video"):
        video.validate_video_report(
            report,
            expected_rollouts=1,
            expected_video_identities=identities,
            expected_terminal_image_identities=terminal_identities,
            expected_terminal_pixel_sha256=terminal_pixel_hashes,
        )


def test_video_report_rejects_terminal_pixels_different_from_action450_trace(tmp_path):
    report, identities, terminal_identities, terminal_pixel_hashes = _valid_report(
        tmp_path
    )
    terminal_pixel_hashes[0] = "f" * 64
    with pytest.raises(ValueError, match="differs from action-450 trace"):
        video.validate_video_report(
            report,
            expected_rollouts=1,
            expected_video_identities=identities,
            expected_terminal_image_identities=terminal_identities,
            expected_terminal_pixel_sha256=terminal_pixel_hashes,
        )


def test_terminal_png_is_fully_decoded_to_trace_hash(tmp_path, monkeypatch):
    path = tmp_path / "episode_0_terminal.png"
    path.write_bytes(b"synthetic png payload")
    decoded = bytes(index % 251 for index in range(448 * 224 * 3))
    probe = {
        "programs": [],
        "stream_groups": [],
        "streams": [
            {
                "codec_name": "png",
                "codec_type": "video",
                "width": 448,
                "height": 224,
                "pix_fmt": "rgb24",
                "nb_read_frames": "1",
            }
        ],
        "format": {"format_name": "png_pipe"},
    }

    def fake_run(command, **kwargs):
        assert kwargs["timeout"] == 120
        if command[0] == "/usr/bin/ffprobe":
            return SimpleNamespace(stdout=json.dumps(probe).encode(), stderr=b"")
        assert command[0] == "/usr/bin/ffmpeg"
        assert command[-4:] == ["rawvideo", "-pix_fmt", "rgb24", "-"]
        return SimpleNamespace(stdout=decoded, stderr=b"")

    monkeypatch.setattr(video.subprocess, "run", fake_run)
    result = video._inspect_terminal_image(path, 0)
    assert result["probe"]["decoded_frame_count"] == 1
    assert (
        result["full_decode"]["decoded_rgb24_sha256"]
        == hashlib.sha256(decoded).hexdigest()
    )


def test_ffprobe_requires_exact_450_frame_h264_yuv420p_contract(tmp_path, monkeypatch):
    probe = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "field_order": "progressive",
                "width": 448,
                "height": 224,
                "avg_frame_rate": "15/1",
                "r_frame_rate": "15/1",
                "nb_frames": "450",
                "nb_read_frames": "450",
                "duration": "30.000000",
            }
        ],
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": "30.000000",
        },
    }

    def fake_run(command, **kwargs):
        assert command[0] == "/usr/bin/ffprobe"
        assert "-count_frames" in command
        assert kwargs["timeout"] == 600
        return SimpleNamespace(stdout=json.dumps(probe).encode(), stderr=b"")

    monkeypatch.setattr(video.subprocess, "run", fake_run)
    result = video._run_ffprobe(tmp_path / "episode_0.mp4")
    assert result["decoded_probe_frame_count"] == 450
    assert result["codec"] == "h264"


def test_ffmpeg_full_decode_is_error_fatal_and_counts_every_frame(
    tmp_path, monkeypatch
):
    def fake_run(command, **kwargs):
        assert command[0] == "/usr/bin/ffmpeg"
        assert "-xerror" in command
        assert command[-2:] == ["null", "-"]
        assert kwargs["timeout"] == 600
        return SimpleNamespace(
            stdout="frame=1\nprogress=continue\nframe=450\nprogress=end\n",
            stderr="",
        )

    monkeypatch.setattr(video.subprocess, "run", fake_run)
    result = video._run_full_decode(tmp_path / "episode_0.mp4")
    assert result["status"] == "pass"
    assert result["decoded_frame_count"] == 450


@pytest.mark.parametrize("rollouts", [0, -1, True])
def test_video_builder_rejects_invalid_rollout_count_before_tool_attestation(
    tmp_path, monkeypatch, rollouts
):
    def unexpected_attestation():
        raise AssertionError("tool attestation must not run for an invalid contract")

    monkeypatch.setattr(video, "attest_media_tools", unexpected_attestation)
    with pytest.raises(ValueError, match="positive integer"):
        video.build_video_report(
            tmp_path,
            expected_rollouts=rollouts,
            container_image_sha256=video.PI05_DROID_JOINTPOS_PYXIS_SHA256,
        )


def test_video_inspection_rejects_symlink_before_probe(tmp_path, monkeypatch):
    target = tmp_path / "real" / "episode_0.mp4"
    target.parent.mkdir()
    target.write_bytes(b"video")
    alias = tmp_path / "episode_0.mp4"
    alias.symlink_to(target)

    def unexpected_probe(_path):
        raise AssertionError("a symlink must be rejected before media execution")

    monkeypatch.setattr(video, "_run_ffprobe", unexpected_probe)
    with pytest.raises(ValueError, match="must not be a symlink"):
        video.inspect_video(alias, 0)


def _terminal_marker_source(wrapper: str) -> str:
    anchor = '"${OPENPI_DIR}/.venv/bin/python" - "${destination}" "$@" <<\'PY\'\n'
    start = wrapper.index(anchor) + len(anchor)
    end = wrapper.index("\nPY\n}", start)
    return wrapper[start:end]


def test_terminal_marker_publisher_is_immutable_and_non_overwriting(tmp_path):
    wrapper = (
        Path(__file__).parents[1]
        / "scripts/polaris/eval_pi05_droid_jointpos_polaris.sh"
    ).read_text()
    publisher = _terminal_marker_source(wrapper)
    marker = tmp_path / "SUCCESS"
    command = [
        sys.executable,
        "-",
        str(marker),
        "status=success",
        "evidence_manifest_sha256=" + "1" * 64,
    ]
    completed = subprocess.run(
        command,
        input=publisher,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert marker.read_text() == (
        "status=success\n" + "evidence_manifest_sha256=" + "1" * 64 + "\n"
    )
    assert stat.S_IMODE(marker.stat().st_mode) == 0o444
    assert marker.stat().st_nlink == 1

    collision = subprocess.run(
        [sys.executable, "-", str(marker), "status=replaced"],
        input=publisher,
        text=True,
        capture_output=True,
        check=False,
    )
    assert collision.returncode != 0
    assert marker.read_text().startswith("status=success\n")
    assert not list(tmp_path.glob(".SUCCESS.partial-*"))


def test_wrapper_runs_pinned_decode_before_sealed_evidence_transaction():
    root = Path(__file__).parents[1]
    wrapper = (root / "scripts/polaris/eval_pi05_droid_jointpos_polaris.sh").read_text()
    terminal_count = wrapper.index("terminal_image_count=")
    rng = wrapper.index("verify_pi05_droid_jointpos_rng_stream.py")
    decode = wrapper.index('"${video_validation_command[@]}"', rng)
    evidence = wrapper.index("-m polaris.pi05_droid_jointpos_evidence", decode)
    assert terminal_count < rng < decode < evidence
    assert '--container-image-sha256 "${PYXIS_IMAGE_SHA256}"' in wrapper
    assert "pi05_droid_jointpos_video_validation.json" in wrapper

    evaluator = (root / "scripts/eval.py").read_text()
    record = evaluator.index(
        "policy_client.record_execution(", evaluator.index("if audited_jointpos:")
    )
    terminal = evaluator.index("policy_client.final_terminal_visualization()", record)
    png = evaluator.index("Image.fromarray(terminal_visualization).save", terminal)
    mp4 = evaluator.index("mediapy.write_video", png)
    assert record < terminal < png < mp4
