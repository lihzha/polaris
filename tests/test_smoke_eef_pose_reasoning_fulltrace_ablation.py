from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_module(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fixture_builder = load_module(
    "reasoning_fulltrace_fixture_builder",
    "build_reasoning_fulltrace_replay_fixture.py",
)
diagnostic = load_module(
    "reasoning_fulltrace_ablation",
    "smoke_eef_pose_reasoning_fulltrace_ablation.py",
)
validator = load_module(
    "reasoning_fulltrace_ablation_validator",
    "validate_eef_pose_reasoning_fulltrace_ablation.py",
)


def mp4_box(box_type: str, payload: bytes = b"") -> bytes:
    encoded_type = box_type.encode("ascii")
    assert len(encoded_type) == 4
    size = 8 + len(payload)
    return size.to_bytes(4, "big") + encoded_type + payload


def mock_video_probe_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stream_field_order: str | None,
    interlaced_flags: list[int],
) -> None:
    stream = {
        "codec_name": "h264",
        "width": diagnostic.VIDEO_WIDTH,
        "height": diagnostic.VIDEO_HEIGHT,
        "pix_fmt": "yuv420p",
        "r_frame_rate": f"{diagnostic.VIDEO_FPS}/1",
        "avg_frame_rate": f"{diagnostic.VIDEO_FPS}/1",
        "nb_read_frames": str(len(interlaced_flags)),
    }
    if stream_field_order is not None:
        stream["field_order"] = stream_field_order

    def fake_run(
        command: list[str], **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        assert _kwargs.get("check") is True
        if command[0] == "/synthetic/ffprobe" and "-count_frames" in command:
            assert (
                "stream=codec_name,pix_fmt,width,height,field_order,"
                in command[command.index("-show_entries") + 1]
            )
            assert _kwargs.get("capture_output") is True
            assert _kwargs.get("text") is True
            output = json.dumps({"streams": [stream]})
        elif command[0] == "/synthetic/ffprobe":
            assert command[command.index("-show_entries") + 1] == (
                "frame=interlaced_frame,top_field_first"
            )
            assert _kwargs.get("capture_output") is True
            assert _kwargs.get("text") is True
            output = json.dumps(
                {
                    "frames": [
                        {"interlaced_frame": flag, "top_field_first": 0}
                        for flag in interlaced_flags
                    ]
                }
            )
        elif command[0] == "/synthetic/ffmpeg":
            assert "-xerror" in command
            output = ""
        else:  # pragma: no cover - gives a useful failure for command drift
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(validator.subprocess, "run", fake_run)


def test_fixture_identity_and_all_recorded_actions_decode() -> None:
    identity, actions = diagnostic.load_actions()
    assert identity["size_bytes"] == diagnostic.FIXTURE_SIZE_BYTES
    assert identity["sha256"] == diagnostic.FIXTURE_SHA256
    assert len(actions) == 294
    assert actions[0][7] == 0.0
    assert actions[-1][7] == 1.0
    changes = []
    previous = actions[0][7]
    for step, action in enumerate(actions[1:], start=1):
        if action[7] != previous:
            changes.append({"step": step, "from": previous, "to": action[7]})
        previous = action[7]
    assert changes == fixture_builder.EXPECTED_GRIPPER_CHANGES


def test_fixture_tamper_is_rejected(tmp_path: Path) -> None:
    tampered = tmp_path / "fixture.json"
    data = diagnostic.FIXTURE_PATH.read_bytes()
    tampered.write_bytes(data[:-2] + b" \n")
    with pytest.raises(diagnostic.FullTraceAblationError, match="identity drift"):
        diagnostic.load_actions(tampered)


def test_video_probe_accepts_missing_stream_field_order_with_frame_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "faststart.mp4"
    video.write_bytes(mp4_box("ftyp") + mp4_box("moov") + mp4_box("mdat"))
    mock_video_probe_subprocess(
        monkeypatch,
        stream_field_order=None,
        interlaced_flags=[0, 0],
    )
    decoded = validator.probe_video(
        video,
        ffprobe="/synthetic/ffprobe",
        ffmpeg="/synthetic/ffmpeg",
    )
    assert decoded["profile"] == (
        "ffprobe_frame_scan_faststart_and_ffmpeg_full_decode_v2"
    )
    assert decoded["field_order"] == "progressive"
    assert decoded["stream_field_order"] is None
    assert decoded["progressive_frame_flag_count"] == 2
    assert decoded["mp4_top_level_box_types"] == ["ftyp", "moov", "mdat"]


def test_video_probe_rejects_interlaced_frame_or_nonfaststart_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(mp4_box("ftyp") + mp4_box("moov") + mp4_box("mdat"))
    mock_video_probe_subprocess(
        monkeypatch,
        stream_field_order=None,
        interlaced_flags=[0, 1],
    )
    with pytest.raises(validator.ValidationError, match="progressive frame flags"):
        validator.probe_video(
            video,
            ffprobe="/synthetic/ffprobe",
            ffmpeg="/synthetic/ffmpeg",
        )

    video.write_bytes(mp4_box("ftyp") + mp4_box("mdat") + mp4_box("moov"))
    mock_video_probe_subprocess(
        monkeypatch,
        stream_field_order="progressive",
        interlaced_flags=[0, 0],
    )
    with pytest.raises(validator.ValidationError, match="fast-start box order"):
        validator.probe_video(
            video,
            ffprobe="/synthetic/ffprobe",
            ffmpeg="/synthetic/ffmpeg",
        )


def test_video_publication_faststarts_before_immutable_link() -> None:
    source = (SCRIPTS / "smoke_eef_pose_reasoning_fulltrace_ablation.py").read_text()
    publish = source[
        source.index("def publish_video(") : source.index("def classify_outcome(")
    ]
    encoded = publish.index("mediapy.write_video(encoded")
    faststart = publish.index('"+faststart"')
    immutable_link = publish.index("os.link(temporary, path)")
    assert encoded < faststart < immutable_link


def test_mp4_box_layout_rejects_truncated_or_out_of_bounds_box(
    tmp_path: Path,
) -> None:
    video = tmp_path / "malformed.mp4"
    video.write_bytes((1).to_bytes(4, "big") + b"moov" + b"short")
    with pytest.raises(validator.ValidationError, match="extended MP4 box size"):
        validator.mp4_box_layout(video)

    video.write_bytes((64).to_bytes(4, "big") + b"mdat" + b"short")
    with pytest.raises(validator.ValidationError, match="invalid MP4 box"):
        validator.mp4_box_layout(video)


def test_force_open_changes_only_the_gripper_dimension() -> None:
    _, source = diagnostic.load_actions()
    opened = diagnostic.effective_actions(source, "force_open")
    assert all(action[7] == 0.0 for action in opened)
    assert all(
        effective[:7] == original[:7]
        for effective, original in zip(opened, source, strict=True)
    )
    for variant in diagnostic.VARIANTS:
        effective = diagnostic.effective_actions(source, variant)
        if variant != "force_open":
            assert effective == source


def snapshot(
    positions: list[float],
    velocities: list[float],
    accelerations: list[float],
    *,
    remaining: int,
) -> dict[str, Any]:
    zeros = [0.0] * 13
    return {
        "joint_pos_rad": positions[7:],
        "joint_vel_rad_s": velocities[7:],
        "joint_acc_rad_s2": accelerations[7:],
        "joint_pos_target_rad": positions[7:],
        "joint_vel_target_rad_s": zeros[7:],
        "joint_effort_target_nm": zeros[7:],
        "all_joint_pos_rad": positions,
        "all_joint_vel_rad_s": velocities,
        "all_joint_acc_rad_s2": accelerations,
        "all_joint_pos_target_rad": positions,
        "all_joint_vel_target_rad_s": zeros,
        "all_joint_effort_target_nm": zeros,
        "all_computed_torque_nm": zeros,
        "all_applied_torque_nm": zeros,
        "interlock": {
            "arm_apply_call_count": 1,
            "configured_substeps": 86,
            "remaining_substeps": remaining,
            "anchor_valid": remaining > 0,
            "activation_count": 1,
            "active_apply_count": 1,
            "released_apply_count": int(remaining == 0),
            "anchor_joint_pos_rad": [0.0] * 7,
        },
    }


def test_trace_summary_uses_physx_mimic_equation_and_marks_release() -> None:
    # q_follower + G*q_driver == 0 for G=[-1,+1,-1,+1,+1].
    driver = 0.2
    positions = [0.0] * 7 + [driver, driver, -driver, driver, -driver, -driver]
    velocities = [0.0] * 13
    velocities[6] = -2.5
    velocities[10] = 8.0
    accelerations = [0.0] * 13
    accelerations[10] = 900.0
    entries = [
        {
            "apply_index": 0,
            "policy_step": 0,
            "physics_substep": 0,
            "raw_action": 1.0,
            "requested_endpoint_rad": 0.7853981852531433,
            "target_after_setter_rad": 0.1,
            "pre": snapshot(positions, velocities, accelerations, remaining=1),
            "command_after_setters": snapshot(
                positions, velocities, accelerations, remaining=1
            ),
            "post": snapshot(positions, velocities, accelerations, remaining=1),
        },
        {
            "apply_index": 1,
            "policy_step": 0,
            "physics_substep": 1,
            "raw_action": 1.0,
            "requested_endpoint_rad": 0.7853981852531433,
            "target_after_setter_rad": 0.2,
            "pre": snapshot(positions, velocities, accelerations, remaining=0),
            "command_after_setters": snapshot(
                positions, velocities, accelerations, remaining=0
            ),
            "post": snapshot(positions, velocities, accelerations, remaining=0),
        },
    ]
    summary = diagnostic.summarize_trace(entries)
    assert summary["entry_count"] == 2
    assert summary["max_abs_joint_velocity_rad_s"][6] == 2.5
    assert summary["max_abs_joint_velocity_rad_s"][10] == 8.0
    assert summary["max_abs_joint_acceleration_rad_s2"][10] == 900.0
    assert summary["max_abs_mimic_residual_rad"] == [0.0] * 5
    assert summary["interlock_last_active_entries"] == [
        {"apply_index": 1, "policy_step": 0, "physics_substep": 1}
    ]
    assert summary["interlock_first_released_entries"] == []
    cadence = diagnostic.validate_trace_cadence(
        entries,
        variant="force_open",
        outcome={
            "classification": "ablation_numerical_failure_observed",
            "controller_completed_actions": 0,
        },
    )
    assert cadence["entry_count"] == 2
    assert cadence["expected_entry_count"] is None
    entries[1]["physics_substep"] = 7
    with pytest.raises(diagnostic.FullTraceAblationError, match="cadence drift"):
        diagnostic.validate_trace_cadence(
            entries,
            variant="force_open",
            outcome={
                "classification": "ablation_numerical_failure_observed",
                "controller_completed_actions": 0,
            },
        )


def test_expected_outcomes_are_closed() -> None:
    failure = {
        "message": (
            "PolaRiS EEF IK current joint velocity exceeds the live simulation limit; "
            "aborting before DLS and PhysX (joint='panda_joint7', "
            "velocity_rad_s=2.923665761947632, limit_rad_s=2.609999895095825, "
            "excess_rad_s=0.31366586685180664, policy_step=293, "
            "physics_substep=2, evidence_sha256="
            "81ab9fb0cf1b74d67abbafb75ecc2ded5e606547fb46eec3e2b5a06acadd2959)"
        )
    }
    baseline = diagnostic.classify_outcome("baseline", failure, 293)
    assert baseline["diagnostic_completed"] is True
    assert baseline["classification"] == "baseline_exact_source_abort_reproduced"
    assert (
        diagnostic.classify_outcome("baseline", None, 294)["diagnostic_completed"]
        is False
    )
    unparsed = diagnostic.classify_outcome(
        "baseline", {"message": "different numerical failure"}, 17
    )
    assert unparsed["diagnostic_completed"] is False
    assert unparsed["classification"] == "baseline_abort_unparsed"
    assert unparsed["numerical_failure_parse_error"]
    for variant in diagnostic.VARIANTS[1:]:
        completed = diagnostic.classify_outcome(variant, None, 294)
        assert completed["diagnostic_completed"] is True
        assert completed["classification"] == "ablation_completed_all_recorded_actions"
        failed = diagnostic.classify_outcome(variant, failure, 293)
        assert failed["diagnostic_completed"] is True
        assert failed["classification"] == "ablation_numerical_failure_observed"


def test_interventions_are_pre_registered_and_one_variable() -> None:
    assert diagnostic.HOLD_CLOSE_ANCHOR_SUBSTEPS == 86 + 16
    assert diagnostic.FOLLOWER_DEFAULT_LIMIT == 174.53292846679688
    source = (SCRIPTS / "smoke_eef_pose_reasoning_fulltrace_ablation.py").read_text()
    assert source.count("set_dof_max_velocities(replacement, indices)") == 1
    assert source.count("action[7] = 0.0") == 1
    assert source.count("_gripper_close_arm_interlock_configured_substeps =") == 1


def test_arm_wrapper_finalizes_post_physics_before_next_arm_command() -> None:
    events: list[str] = []

    class Finger:
        def finalize_physics_post_before_next_arm(self) -> None:
            events.append("post_physics")

    class Manager:
        _terms = {"finger_joint": Finger()}

    class Env:
        action_manager = Manager()

    class Arm:
        def __init__(self, _cfg: Any, _env: Any) -> None:
            pass

        def apply_actions(self) -> None:
            events.append("arm_command")

    wrapped = diagnostic.make_full_trace_arm_class(Arm)(object(), Env())
    wrapped.apply_actions()
    assert events == ["post_physics", "arm_command"]


def test_validator_closes_each_single_variable_intervention() -> None:
    before = list(validator.EXPECTED_LIMITS_WITH_FOLLOWER_CAP)

    def tensor(values: list[float]) -> dict[str, Any]:
        return {
            "dtype": "torch.float32",
            "device": "cpu",
            "shape": [1, 13],
            "values": values,
        }

    for variant in diagnostic.VARIANTS:
        after = list(before)
        if variant == "follower_default_limit":
            after[8:] = [diagnostic.FOLLOWER_DEFAULT_LIMIT] * 5
        intervention = {
            "profile": "post_production_installer_single_variable_diagnostic_only_v1",
            "variant": variant,
            "production_contract_acceptance": False,
            "force_gripper_open": variant == "force_open",
            "passive_follower_velocity_limit_setter_call_count": int(
                variant == "follower_default_limit"
            ),
            "velocity_limits_before_intervention": tensor(before),
            "velocity_limits_after_intervention": tensor(after),
            "configured_close_anchor_substeps": (
                diagnostic.HOLD_CLOSE_ANCHOR_SUBSTEPS
                if variant == "hold_close_anchor"
                else 86
            ),
        }
        assert (
            validator.validate_intervention(intervention, variant=variant)
            == intervention
        )


def minimal_post_kit_payload(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"synthetic-video")
    video.chmod(0o444)
    video_identity = validator.file_identity(video)
    before = list(validator.EXPECTED_LIMITS_WITH_FOLLOWER_CAP)

    def tensor(values: list[float]) -> dict[str, Any]:
        return {
            "dtype": "torch.float32",
            "device": "cpu",
            "shape": [1, 13],
            "values": values,
        }

    failure = {
        "type": "synthetic.NumericalFailure",
        "message": "different numerical failure",
        "traceback": "synthetic traceback",
    }
    outcome = diagnostic.classify_outcome("force_open", failure, 0)
    entries: list[dict[str, Any]] = []
    cadence = diagnostic.validate_trace_cadence(
        entries, variant="force_open", outcome=outcome
    )
    summary = diagnostic.summarize_trace(entries)
    intervention = {
        "profile": "post_production_installer_single_variable_diagnostic_only_v1",
        "variant": "force_open",
        "production_contract_acceptance": False,
        "force_gripper_open": True,
        "passive_follower_velocity_limit_setter_call_count": 0,
        "velocity_limits_before_intervention": tensor(before),
        "velocity_limits_after_intervention": tensor(before),
        "configured_close_anchor_substeps": 86,
    }
    payload = {
        "schema_version": 1,
        "profile": diagnostic.PROFILE,
        "passed": True,
        "diagnostic_only": True,
        "variant": "force_open",
        "repository": {
            "path": "/synthetic/repo",
            "commit": "a" * 40,
            "clean_tracked": True,
            "production_base_commit": diagnostic.PRODUCTION_BASE_COMMIT,
            "production_base_relation": "exact_first_parent_v1",
        },
        "container_image": {
            "path": "/synthetic/image.sqsh",
            "size_bytes": 7_183_130_624,
            "sha256": (
                "ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a"
            ),
        },
        "lifecycle": {
            "job_id": 123,
            "launch_id": "b" * 64,
            "procid": 0,
            "localid": 0,
            "ntasks": 1,
        },
        "production_eval": {},
        "fixture": {},
        "source_trace_sha256": fixture_builder.TRACE_SHA256,
        "source_action_float32_sha256": diagnostic.ACTION_ENCODING[
            "uncompressed_sha256"
        ],
        "boundary_helper": {},
        "assets": {},
        "runtime_protocol": {},
        "runtime_frame": {},
        "production_gripper_contract_before_ablation": {},
        "intervention": intervention,
        "action_count": diagnostic.ACTION_COUNT,
        "actions_completed": 0,
        "numerical_failure": failure,
        "outcome": outcome,
        "full_substep_trace_profile": (
            "all13_after_arm_before_gripper_post_setters_post_physics_"
            "before_next_arm_v2"
        ),
        "full_substep_trace_cadence": cadence,
        "full_substep_trace": entries,
        "full_substep_summary": summary,
        "video": {
            **video_identity,
            "profile": "synthetic",
            "fps": 15,
            "frame_count": 2,
            "height": 224,
            "width": 448,
        },
        "runtime_close": {
            "environment_close_completed": True,
            "simulation_app_close_state": (
                "pending_terminal_call_after_raw_publication_v2"
            ),
            "publication_timing": "after_environment_before_simulation_app_close_v2",
            "completion_evidence": (
                "post_kit_validator_requires_zero_simulator_srun_exit_v1"
            ),
        },
    }
    assert set(payload) == validator.RESULT_FIELDS
    return payload, video


def test_post_kit_validator_promotes_only_zero_simulator_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, video = minimal_post_kit_payload(tmp_path)
    monkeypatch.setattr(
        validator,
        "probe_video",
        lambda *_args, **_kwargs: {
            "profile": "synthetic-full-decode",
            "codec_name": "h264",
            "pixel_format": "yuv420p",
            "field_order": "progressive",
            "width": 448,
            "height": 224,
            "fps": 15,
            "frame_count": 2,
        },
    )
    kwargs = {
        "variant": "force_open",
        "expected_commit": "a" * 40,
        "expected_job_id": 123,
        "expected_launch_id": "b" * 64,
        "result_identity": {
            "path": "/synthetic/result.json",
            "size_bytes": 1,
            "sha256": "c" * 64,
            "mode": "0444",
            "nlink": 1,
        },
        "video_path": video,
        "ffprobe": "/synthetic/ffprobe",
        "ffmpeg": "/synthetic/ffmpeg",
    }
    manifest = validator.validate_result(payload, simulator_srun_exit_code=0, **kwargs)
    assert manifest["profile"].endswith("_v3")
    assert manifest["runtime_exit"] == {
        "profile": "zero_simulator_srun_after_terminal_simulation_app_close_v1",
        "simulator_srun_exit_code": 0,
        "validated_in_separate_post_kit_process": True,
    }
    with pytest.raises(validator.ValidationError, match="simulator srun exit code"):
        validator.validate_result(payload, simulator_srun_exit_code=1, **kwargs)


def test_post_kit_validator_rejects_old_close_schema_and_baseline_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, video = minimal_post_kit_payload(tmp_path)
    monkeypatch.setattr(validator, "probe_video", lambda *_args, **_kwargs: {})
    kwargs = {
        "expected_commit": "a" * 40,
        "expected_job_id": 123,
        "expected_launch_id": "b" * 64,
        "result_identity": {},
        "video_path": video,
        "ffprobe": "/synthetic/ffprobe",
        "ffmpeg": "/synthetic/ffmpeg",
        "simulator_srun_exit_code": 0,
    }
    old = copy.deepcopy(payload)
    old["runtime_close"] = {
        "environment_close_completed": True,
        "simulation_app_close_invoked_after_publication": True,
        "publication_timing": "after_environment_before_simulation_app_close_v2",
        "completion_evidence": (
            "post_kit_validator_requires_zero_simulator_srun_exit_v1"
        ),
    }
    with pytest.raises(
        validator.ValidationError, match="pre-SimulationApp-close result publication"
    ):
        validator.validate_result(old, variant="force_open", **kwargs)

    mismatch = copy.deepcopy(payload)
    mismatch["variant"] = "baseline"
    mismatch["intervention"]["variant"] = "baseline"
    mismatch["intervention"]["force_gripper_open"] = False
    mismatch["actions_completed"] = 294
    mismatch["numerical_failure"] = None
    mismatch["outcome"] = diagnostic.classify_outcome("baseline", None, 294)
    with pytest.raises(
        validator.ValidationError,
        match="result profile/variant|diagnostic outcome|baseline",
    ):
        validator.validate_result(mismatch, variant="baseline", **kwargs)


def test_wrapper_orders_post_kit_validation_after_simulator_exit_gate() -> None:
    runner_source = (
        SCRIPTS / "smoke_eef_pose_reasoning_fulltrace_ablation.py"
    ).read_text()
    main_source = runner_source[runner_source.index("def main()") :]
    assert main_source.index("_atomic_write_immutable") < main_source.index(
        "simulation_app.close()"
    )
    wrapper_source = (
        SCRIPTS / "run_eef_pose_reasoning_fulltrace_ablation_srun.sh"
    ).read_text()
    assert wrapper_source.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    first_runner = wrapper_source.index('/.venv/bin/python "${runner}"')
    validator_call = wrapper_source.index('/.venv/bin/python "${validator}"')
    success_write = wrapper_source.index("success_temporary=")
    assert first_runner < validator_call < success_write
    assert "--simulator-srun-exit-code 0" in wrapper_source
