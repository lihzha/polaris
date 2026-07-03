from __future__ import annotations

import ast
import copy
import inspect
import json
import math
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from scripts import smoke_eef_pose_gripper_impulse_diagnostic as diagnostic
from scripts import smoke_eef_pose_boundary_replay as boundary


def _actions():
    _, actions = boundary.load_replay_fixture()
    return actions


def test_exact_action_plan_preserves_fixture_prefix_bitwise():
    actions = _actions()
    plan, effective = diagnostic.build_action_plan(actions, mode="exact")
    assert plan["overrides"] == []
    assert plan["source_gripper_transitions"] == [
        {"policy_step": 115, "from_float32": 0.0, "to_float32": 1.0}
    ]
    assert plan["effective_gripper_transitions"] == plan["source_gripper_transitions"]
    assert effective == actions[:118]


def test_delay_action_plan_changes_only_action_115_gripper_bit():
    actions = _actions()
    plan, effective = diagnostic.build_action_plan(
        actions, mode="delay_first_close_one_step"
    )
    changed = []
    for step, (original, actual) in enumerate(
        zip(actions[:118], effective, strict=True)
    ):
        for index, (left, right) in enumerate(zip(original, actual, strict=True)):
            if not diagnostic._same_float32(left, right):  # noqa: SLF001
                changed.append((step, index, left, right))
    assert changed == [(115, 7, 1.0, 0.0)]
    assert effective[116][7] == 1.0
    assert plan["effective_gripper_transitions"] == [
        {"policy_step": 116, "from_float32": 0.0, "to_float32": 1.0}
    ]
    assert (
        plan["arm_prefix_float32_sha256"]
        == diagnostic._float32_arm_sha256(  # noqa: SLF001
            actions[:118]
        )
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda actions: actions.__setitem__(100, [*actions[100][:7], 1.0]),
            "exactly one",
        ),
        (lambda actions: actions[115].__setitem__(7, 0.0), "exactly one"),
    ],
)
def test_action_plan_rejects_fixture_transition_drift(mutation, message):
    actions = copy.deepcopy(_actions())
    mutation(actions)
    with pytest.raises(diagnostic.GripperImpulseDiagnosticError, match=message):
        diagnostic.build_action_plan(actions, mode="exact")


def test_tensor_evidence_preserves_finite_values_and_marks_specials():
    evidence = diagnostic.tensor_evidence([[1.0, float("inf")], [float("nan"), -2.0]])
    assert evidence == {
        "shape": [4],
        "dtype": "python_float64",
        "device": "host",
        "values": [1.0, 0.0, 0.0, -2.0],
        "finite_mask": [True, False, False, True],
        "finite_count": 2,
        "nonfinite": [
            {"flat_index": 1, "kind": "positive_infinity"},
            {"flat_index": 2, "kind": "nan"},
        ],
    }
    diagnostic.validate_tensor_evidence(evidence, field="probe")


@pytest.mark.parametrize(
    ("field", "value"),
    [("values", [True]), ("finite_count", True)],
)
def test_tensor_evidence_rejects_boolean_numeric_impersonation(field, value):
    evidence = _tensor([1])
    evidence[field] = value
    with pytest.raises(diagnostic.GripperImpulseDiagnosticError):
        diagnostic.validate_tensor_evidence(evidence, field="probe")


def test_immutable_json_is_mode_0444_and_non_overwriting(tmp_path: Path):
    path = tmp_path / "capture.json"
    identity = diagnostic.publish_immutable_json(path, {"value": 1})
    assert identity["mode"] == "0444"
    assert identity["nlink"] == 1
    assert json.loads(path.read_text()) == {"value": 1}
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    with pytest.raises(FileExistsError):
        diagnostic.publish_immutable_json(path, {"value": 2})


def test_immutable_video_uses_exclusive_publication(tmp_path: Path):
    path = tmp_path / "capture.mp4"

    def writer(temporary, frames, *, fps):
        assert fps == 15
        Path(temporary).write_bytes(b"fake-mp4")

    def probe(_path):
        return {"frame_count": 2, "height": 224, "width": 448}

    identity = diagnostic.publish_immutable_video(
        path, [object(), object()], writer=writer, probe=probe
    )
    assert identity["mode"] == "0444"
    assert identity["nlink"] == 1
    assert identity["frame_count"] == 2
    with pytest.raises(diagnostic.GripperImpulseDiagnosticError, match="overwrite"):
        diagnostic.publish_immutable_video(
            path, [object(), object()], writer=writer, probe=probe
        )


def test_immutable_video_rejects_boolean_frame_count_impostor(tmp_path: Path):
    def writer(temporary, _frames, *, fps):
        assert fps == 15
        Path(temporary).write_bytes(b"fake-mp4")

    with pytest.raises(diagnostic.GripperImpulseDiagnosticError, match="shape drift"):
        diagnostic.publish_immutable_video(
            tmp_path / "capture.mp4",
            [object()],
            writer=writer,
            probe=lambda _path: {
                "frame_count": True,
                "height": 224,
                "width": 448,
            },
        )


def test_stdlib_video_probe_binds_single_stream_fps_duration_and_full_decode(
    monkeypatch, tmp_path: Path
):
    path = tmp_path / "capture.mp4"
    path.write_bytes(b"mp4")
    calls = []
    ffprobe_payload = {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "width": 448,
                "height": 224,
                "nb_read_frames": "117",
                "avg_frame_rate": "15/1",
                "r_frame_rate": "15/1",
                "duration": "7.800000",
                "duration_ts": 119808,
                "time_base": "1/15360",
            }
        ],
        "format": {"duration": "7.800000"},
    }

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[0] == "ffprobe":
            return SimpleNamespace(stdout=json.dumps(ffprobe_payload), stderr="")
        assert argv[0] == "ffmpeg"
        return SimpleNamespace(stdout=b"", stderr=b"")

    monkeypatch.setattr(diagnostic.subprocess, "run", run)
    assert diagnostic._probe_video_stdlib(path) == {  # noqa: SLF001
        "frame_count": 117,
        "height": 224,
        "width": 448,
    }
    assert [call[0][0] for call in calls] == ["ffprobe", "ffmpeg"]
    assert "-select_streams" not in calls[0][0]
    show_entries = calls[0][0][calls[0][0].index("-show_entries") + 1]
    assert "duration,duration_ts,time_base" in show_entries
    assert calls[1][0][-2:] == ["null", "-"]


@pytest.mark.parametrize("format_duration", ["7.933333", "7.934000"])
def test_stdlib_video_probe_accepts_119_frames_with_canonical_container_rounding(
    monkeypatch, tmp_path: Path, format_duration: str
):
    payload = {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "width": 448,
                "height": 224,
                "nb_read_frames": "119",
                "avg_frame_rate": "15/1",
                "r_frame_rate": "15/1",
                "duration": "7.933333",
                "duration_ts": 121856,
                "time_base": "1/15360",
            }
        ],
        "format": {"duration": format_duration},
    }

    def run(argv, **_kwargs):
        return SimpleNamespace(
            stdout=json.dumps(payload) if argv[0] == "ffprobe" else b"",
            stderr="" if argv[0] == "ffprobe" else b"",
        )

    monkeypatch.setattr(diagnostic.subprocess, "run", run)
    assert diagnostic._probe_video_stdlib(  # noqa: SLF001
        tmp_path / "capture.mp4"
    ) == {"frame_count": 119, "height": 224, "width": 448}


def test_stdlib_video_probe_rejects_genuine_119_frame_tick_cadence_mismatch(
    monkeypatch, tmp_path: Path
):
    payload = {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "width": 448,
                "height": 224,
                "nb_read_frames": "119",
                "avg_frame_rate": "15/1",
                "r_frame_rate": "15/1",
                "duration": "7.933333",
                # 118 frames at the declared time base, despite decoding 119.
                "duration_ts": 120832,
                "time_base": "1/15360",
            }
        ],
        "format": {"duration": "7.934000"},
    }
    monkeypatch.setattr(
        diagnostic.subprocess,
        "run",
        lambda _argv, **_kwargs: SimpleNamespace(stdout=json.dumps(payload), stderr=""),
    )
    with pytest.raises(
        diagnostic.GripperImpulseDiagnosticError,
        match="duration_ts/time_base cadence",
    ):
        diagnostic._probe_video_stdlib(tmp_path / "capture.mp4")  # noqa: SLF001


@pytest.mark.parametrize("format_duration", ["7.933000", "7.935000"])
def test_stdlib_video_probe_rejects_noncanonical_119_frame_container_rounding(
    monkeypatch, tmp_path: Path, format_duration: str
):
    payload = {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "width": 448,
                "height": 224,
                "nb_read_frames": "119",
                "avg_frame_rate": "15/1",
                "r_frame_rate": "15/1",
                "duration": "7.933333",
                "duration_ts": 121856,
                "time_base": "1/15360",
            }
        ],
        "format": {"duration": format_duration},
    }
    monkeypatch.setattr(
        diagnostic.subprocess,
        "run",
        lambda _argv, **_kwargs: SimpleNamespace(stdout=json.dumps(payload), stderr=""),
    )
    with pytest.raises(
        diagnostic.GripperImpulseDiagnosticError,
        match="stream-or-millisecond-ceiling",
    ):
        diagnostic._probe_video_stdlib(tmp_path / "capture.mp4")  # noqa: SLF001


@pytest.mark.parametrize(
    "case",
    [
        "second_stream",
        "wrong_fps",
        "wrong_stream_duration",
        "wrong_duration_ts",
        "wrong_time_base",
        "wrong_format_duration",
        "missing_time_base",
        "extra_stream_field",
    ],
)
def test_stdlib_video_probe_fails_closed_on_stream_cadence_drift(
    monkeypatch, tmp_path: Path, case: str
):
    stream = {
        "index": 0,
        "codec_type": "video",
        "width": 448,
        "height": 224,
        "nb_read_frames": "117",
        "avg_frame_rate": "15/1",
        "r_frame_rate": "15/1",
        "duration": "7.800000",
        "duration_ts": 119808,
        "time_base": "1/15360",
    }
    payload = {"streams": [stream], "format": {"duration": "7.800000"}}
    if case == "second_stream":
        payload["streams"].append({"index": 1, "codec_type": "audio"})
    elif case == "wrong_fps":
        stream["avg_frame_rate"] = "30/1"
    elif case == "wrong_stream_duration":
        stream["duration"] = "7.799000"
    elif case == "wrong_duration_ts":
        stream["duration_ts"] = 120832
    elif case == "wrong_time_base":
        stream["time_base"] = "1/1000"
    elif case == "wrong_format_duration":
        payload["format"]["duration"] = "7.801000"
    elif case == "missing_time_base":
        del stream["time_base"]
    elif case == "extra_stream_field":
        stream["start_time"] = "0.000000"
    monkeypatch.setattr(
        diagnostic.subprocess,
        "run",
        lambda _argv, **_kwargs: SimpleNamespace(stdout=json.dumps(payload), stderr=""),
    )
    with pytest.raises(diagnostic.GripperImpulseDiagnosticError):
        diagnostic._probe_video_stdlib(tmp_path / "capture.mp4")  # noqa: SLF001


def test_host_parsers_cover_runtime_and_post_kit_modes(tmp_path: Path):
    runtime = diagnostic.build_runtime_parser().parse_args(
        [
            "--output-json",
            str(tmp_path / "capture.json"),
            "--output-video",
            str(tmp_path / "capture.mp4"),
            "--output-ready-marker",
            str(tmp_path / "ready.json"),
            "--runtime-exit",
            str(tmp_path / "runtime.exit"),
            "--mode",
            "exact",
            "--expected-source-sha256",
            "0" * 64,
            "--expected-source-size-bytes",
            "1",
        ]
    )
    assert runtime.enable_gripper_velocity_limit_candidate is False
    diagnostic._validate_runtime_output_paths(runtime)  # noqa: SLF001
    candidate_runtime = diagnostic.build_runtime_parser().parse_args(
        [
            "--output-json",
            str(tmp_path / "candidate.json"),
            "--output-video",
            str(tmp_path / "candidate.mp4"),
            "--output-ready-marker",
            str(tmp_path / "candidate-ready.json"),
            "--runtime-exit",
            str(tmp_path / "candidate-runtime.exit"),
            "--mode",
            "exact",
            "--enable-gripper-velocity-limit-candidate",
            "--expected-source-sha256",
            "0" * 64,
            "--expected-source-size-bytes",
            "1",
        ]
    )
    assert candidate_runtime.enable_gripper_velocity_limit_candidate is True
    validation = diagnostic.build_validation_parser().parse_args(
        [
            "--action",
            "validate",
            "--validate-capture",
            str(tmp_path / "capture.json"),
            "--video",
            str(tmp_path / "capture.mp4"),
            "--ready-marker",
            str(tmp_path / "ready.json"),
            "--runtime-exit",
            str(tmp_path / "runtime.exit"),
            "--outer-srun-status",
            str(tmp_path / "srun.exit"),
            "--intended-attestation-path",
            str(tmp_path / "final.attestation.json"),
            "--expected-mode",
            "delay_first_close_one_step",
            "--expected-gripper-drive-profile",
            diagnostic.GRIPPER_VELOCITY_LIMIT_CANDIDATE_DRIVE_PROFILE,
            "--polaris-repo",
            str(tmp_path / "repo"),
            "--expected-polaris-commit",
            "0" * 40,
            "--expected-diagnostic-sha256",
            "0" * 64,
            "--expected-finalizer-sha256",
            "0" * 64,
            "--expected-boundary-sha256",
            "0" * 64,
            "--expected-fixture-sha256",
            "0" * 64,
            "--container-image",
            str(tmp_path / "image.sqsh"),
            "--expected-container-image-sha256",
            "0" * 64,
            "--submitted-saved-wrapper",
            str(tmp_path / "submitted.sh"),
            "--expected-submitted-saved-wrapper-sha256",
            "0" * 64,
            "--runtime-dollar-zero-snapshot",
            str(tmp_path / "runtime-dollar-zero.sh"),
            "--expected-runtime-dollar-zero-sha256",
            "0" * 64,
            "--scontrol-batch-script-snapshot",
            str(tmp_path / "scontrol-batch-script.sh"),
            "--expected-scontrol-batch-script-sha256",
            "0" * 64,
            "--slurm-job-oneliner-snapshot",
            str(tmp_path / "slurm-job.oneliner"),
            "--expected-slurm-job-id",
            "123",
            "--expected-slurm-node-list",
            "l401",
            "--expected-slurm-node-name",
            "l401",
            "--expected-slurm-account",
            "ailab",
            "--expected-slurm-partition",
            "gpu",
            "--expected-slurm-job-name",
            "impulse",
            "--expected-slurm-num-nodes",
            "1",
            "--expected-slurm-num-cpus",
            "8",
            "--expected-slurm-num-tasks",
            "1",
            "--expected-slurm-req-tres",
            "cpu=8,mem=64G,gres/gpu=1,node=1,billing=8",
            "--expected-slurm-alloc-tres",
            "cpu=8,mem=64G,gres/gpu=1,node=1,billing=8",
            "--expected-slurm-tres-per-node",
            "gres/gpu:l40s:1",
            "--expected-slurm-gpus-on-node",
            "1",
            "--expected-slurm-cpus-per-task",
            "8",
            "--expected-slurm-output",
            str(tmp_path / "slurm.out"),
            "--expected-slurm-command",
            str(tmp_path / "submitted.sh"),
            "--expected-slurm-work-dir",
            str(tmp_path),
        ]
    )
    assert validation.expected_mode == "delay_first_close_one_step"
    assert (
        validation.expected_gripper_drive_profile
        == diagnostic.GRIPPER_VELOCITY_LIMIT_CANDIDATE_DRIVE_PROFILE
    )


def test_parent_revalidation_independently_derives_drive_profile_from_runtime_flag():
    source = inspect.getsource(
        diagnostic._host_revalidate_runtime_artifacts  # noqa: SLF001
    )

    assert "expected_gripper_drive_profile=_selected_gripper_drive_profile(" in source
    assert "args_cli.enable_gripper_velocity_limit_candidate" in source


def test_diagnostic_source_is_launch_bound_and_live_rehashed():
    identity = diagnostic._file_identity(Path(diagnostic.__file__))  # noqa: SLF001
    source = diagnostic.capture_diagnostic_source(
        expected_sha256=identity["sha256"],
        expected_size_bytes=identity["size_bytes"],
    )
    assert source["actual"] == identity
    for field, value in (
        ("sha256", "0" * 64),
        ("size_bytes", identity["size_bytes"] + 1),
    ):
        tampered = copy.deepcopy(source)
        tampered["launch_expected"][field] = value
        with pytest.raises(
            diagnostic.GripperImpulseDiagnosticError,
            match="launch-provided identity",
        ):
            diagnostic._validate_diagnostic_source(tampered)  # noqa: SLF001


def _diagnostic_outcome(
    mode: str,
    failure_step: int | None,
    failure_substep: int | None = None,
):
    failure = failure_step is not None
    if failure and failure_substep is None:
        failure_substep = diagnostic.REFERENCE_EXACT_FAILURE_PHYSICS_SUBSTEP
    failure_apply_index = (
        failure_step * diagnostic.DECIMATION + failure_substep if failure else None
    )
    return {
        "kind": (
            "allowed_velocity_guard_failure"
            if failure
            else "diagnostic_horizon_reached"
        ),
        "mode": mode,
        "reference_exact_failure_policy_step": 115,
        "reference_exact_failure_physics_substep": 2,
        "allowed_failure_policy_steps": diagnostic.ALLOWED_FAILURE_POLICY_STEPS[mode],
        "failure_policy_step": failure_step,
        "failure_physics_substep": failure_substep,
        "failure_apply_index": failure_apply_index,
        "last_attempted_policy_step": (
            failure_step if failure else diagnostic.HORIZON_POLICY_STEP
        ),
        "completed_horizon_policy_step": (
            None if failure else diagnostic.HORIZON_POLICY_STEP
        ),
        "controller_failure": (
            {
                "type": "polaris.robust_differential_ik.DifferentialIKInvariantError",
                "message": "joint velocity exceeds configured limit",
                "traceback": "trace",
            }
            if failure
            else None
        ),
        "causal_interpretation": diagnostic._causal_interpretation(  # noqa: SLF001
            mode, failure_step, failure_substep
        ),
        "timing_classification": diagnostic._timing_classification(  # noqa: SLF001
            mode, failure_step, failure_substep
        ),
    }


@pytest.mark.parametrize(
    ("mode", "failure_step", "expected_frames", "expected_phase"),
    [
        ("exact", 115, 117, "post_guard_bound_state_no_physics_advance"),
        (
            "delay_first_close_one_step",
            115,
            117,
            "post_guard_bound_state_no_physics_advance",
        ),
        (
            "delay_first_close_one_step",
            116,
            118,
            "post_guard_bound_state_no_physics_advance",
        ),
        ("exact", None, 119, "post_horizon_bound_state_no_physics_advance"),
        (
            "delay_first_close_one_step",
            None,
            119,
            "post_horizon_bound_state_no_physics_advance",
        ),
    ],
)
def test_video_phase_contract_includes_one_terminal_no_advance_frame(
    mode, failure_step, expected_frames, expected_phase
):
    outcome = _diagnostic_outcome(mode, failure_step)
    diagnostic._validate_outcome(outcome, mode=mode)  # noqa: SLF001
    phase = diagnostic.build_video_phase_contract(outcome)
    assert phase["total_frame_count"] == expected_frames
    assert phase["terminal_frame_count"] == 1
    assert phase["terminal_frame_index"] == expected_frames - 1
    assert phase["terminal_frame_phase"] == expected_phase
    assert phase["physics_advanced_for_terminal_frame"] is False
    diagnostic.validate_video_phase_contract(phase, outcome=outcome)
    for field, value in (
        ("terminal_frame_count", 0),
        ("terminal_frame_index", expected_frames - 2),
        ("physics_advanced_for_terminal_frame", True),
    ):
        tampered = copy.deepcopy(phase)
        tampered[field] = value
        with pytest.raises(diagnostic.GripperImpulseDiagnosticError):
            diagnostic.validate_video_phase_contract(tampered, outcome=outcome)


def _runtime_identity_fixture():
    protocol = {
        "episode_steps": 450,
        "policy_hz": 15.0,
        "step_dt": 1.0 / 15.0,
        "physics_hz": 120.0,
        "physics_dt": 1.0 / 120.0,
        "decimation": 8,
        "reset_seed": 0,
        "initial_condition_index": 0,
    }
    frame = {
        "eef_frame": "panda_link8",
        "reference_frame": "panda_link0",
        "position_error_m": 0.0,
        "rotation_error_rad": 0.0,
        "controlled_body": "panda_link8",
        "body_offset": "identity",
        "command_type": "pose",
        "use_relative_mode": False,
        "ik_method": "dls",
        "dls_damping": 0.01,
        "arm_scale": 1.0,
        "arm_joint_names": [f"panda_joint{index}" for index in range(1, 8)],
        "gripper_threshold_profile": ("closed_positive_ge_0p5_inverse_open_gt_0p5_v1"),
        "ik_safety_profile": "safety-profile",
        "action_dim": 7,
    }
    return protocol, frame, {"profile": "safety-profile"}


def test_runtime_identity_explicitly_binds_reset_and_initial_condition():
    protocol, frame, safety = _runtime_identity_fixture()
    diagnostic._validate_runtime_identity(  # noqa: SLF001
        protocol, frame, arm_safety=safety
    )
    for field, value in (
        ("reset_seed", 1),
        ("initial_condition_index", 1),
        ("reset_seed", False),
        ("initial_condition_index", False),
    ):
        tampered = copy.deepcopy(protocol)
        tampered[field] = value
        with pytest.raises(diagnostic.GripperImpulseDiagnosticError):
            diagnostic._validate_runtime_identity(  # noqa: SLF001
                tampered, frame, arm_safety=safety
            )


def test_ready_marker_rejects_preclose_and_typed_identity_tamper(tmp_path: Path):
    raw = diagnostic.publish_immutable_json(tmp_path / "raw.json", {"raw": True})
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    video_path.chmod(0o444)
    video = diagnostic._file_identity(video_path)  # noqa: SLF001
    source_identity = diagnostic._file_identity(  # noqa: SLF001
        Path(diagnostic.__file__)
    )
    source = diagnostic.capture_diagnostic_source(
        expected_sha256=source_identity["sha256"],
        expected_size_bytes=source_identity["size_bytes"],
    )
    marker = {
        "schema_version": 1,
        "profile": diagnostic.READY_MARKER_PROFILE,
        "stage": "simulation_app_close_pending",
        "mode": "exact",
        "raw_result": raw,
        "video": video,
        "diagnostic_source": source,
        "runtime_exit_contract": diagnostic.build_runtime_exit_contract(
            tmp_path / "runtime.exit"
        ),
    }
    diagnostic.validate_ready_marker(
        marker,
        mode="exact",
        raw_identity=raw,
        video_identity=video,
        diagnostic_source=source,
        runtime_exit_contract=marker["runtime_exit_contract"],
    )
    for field, value in (
        ("schema_version", True),
        ("stage", "close_environment"),
        ("raw_result", video),
        ("diagnostic_source", {"tampered": True}),
        ("runtime_exit_contract", {"tampered": True}),
    ):
        tampered = copy.deepcopy(marker)
        tampered[field] = value
        with pytest.raises(diagnostic.GripperImpulseDiagnosticError):
            diagnostic.validate_ready_marker(
                tampered,
                mode="exact",
                raw_identity=raw,
                video_identity=video,
                diagnostic_source=source,
                runtime_exit_contract=marker["runtime_exit_contract"],
            )


@pytest.mark.parametrize(
    ("payload", "process_return_code", "expected"),
    [
        (b"\x00", 0, 0),
        (b"\x01", 1, 1),
        (b"\x00", 1, 1),
        (b"\x00", -9, 1),
        (b"\x01", 0, 1),
        (b"\x01", -11, 1),
        (b"", 0, 1),
        (b"\x00\x00", 0, 1),
        (b"\x02", 2, 1),
        (b"\x00", False, 1),
        (b"\x01", True, 1),
    ],
)
def test_parent_reconciles_child_pipe_and_wait_status_fail_closed(
    payload, process_return_code, expected
):
    assert (
        diagnostic._resolve_child_result(  # noqa: SLF001
            payload, process_return_code
        )
        == expected
    )


def test_child_result_pipe_is_write_only_noninheritable_and_exact(monkeypatch):
    read_descriptor, write_descriptor = os.pipe()
    monkeypatch.setenv(diagnostic.CHILD_RESULT_FD_ENV, str(write_descriptor))
    prepared = diagnostic._prepare_child_result_descriptor()  # noqa: SLF001
    assert prepared == write_descriptor
    assert not os.get_inheritable(prepared)
    assert diagnostic.CHILD_RESULT_FD_ENV not in os.environ
    diagnostic._write_child_result_byte(0, prepared)  # noqa: SLF001
    assert os.read(read_descriptor, 2) == b"\x00"
    assert os.read(read_descriptor, 2) == b""
    with pytest.raises(OSError):
        os.fstat(write_descriptor)
    os.close(read_descriptor)


@pytest.mark.parametrize("payload", [b"\x00", b"\x01"])
def test_parent_nonblocking_pipe_reader_requires_one_byte_and_eof(payload):
    read_descriptor, write_descriptor = os.pipe()
    os.write(write_descriptor, payload)
    os.close(write_descriptor)
    try:
        assert (
            diagnostic._read_exact_child_result_byte_and_eof(  # noqa: SLF001
                read_descriptor
            )
            == payload
        )
        assert os.get_blocking(read_descriptor) is False
    finally:
        os.close(read_descriptor)


@pytest.mark.parametrize("payload", [b"", b"\x00\x01"])
def test_parent_nonblocking_pipe_reader_rejects_missing_or_extra_bytes(payload):
    read_descriptor, write_descriptor = os.pipe()
    if payload:
        os.write(write_descriptor, payload)
    os.close(write_descriptor)
    try:
        with pytest.raises(diagnostic.GripperImpulseDiagnosticError):
            diagnostic._read_exact_child_result_byte_and_eof(  # noqa: SLF001
                read_descriptor
            )
    finally:
        os.close(read_descriptor)


def test_parent_rejects_and_kills_surviving_child_process_group(monkeypatch):
    killed = []
    monkeypatch.setattr(diagnostic, "_process_group_exists", lambda _pgid: True)
    monkeypatch.setattr(
        diagnostic,
        "_kill_process_group",
        lambda pgid, signum: killed.append((pgid, signum)),
    )
    monkeypatch.setattr(
        diagnostic, "_wait_for_process_group_exit", lambda _pgid, timeout: True
    )
    with pytest.raises(
        diagnostic.GripperImpulseDiagnosticError, match="surviving process-group"
    ):
        diagnostic._reject_and_kill_surviving_process_group(1234)  # noqa: SLF001
    assert killed == [(1234, diagnostic.signal.SIGKILL)]


def test_timeout_cleanup_drains_group_even_after_leader_already_exited(monkeypatch):
    class Process:
        pid = 1234

        def poll(self):
            return 0

        def wait(self, timeout=None):
            assert timeout is None
            return 0

    existence = iter([True, False])
    killed = []
    monkeypatch.setattr(
        diagnostic, "_process_group_exists", lambda _pgid: next(existence)
    )
    monkeypatch.setattr(
        diagnostic,
        "_kill_process_group",
        lambda pgid, signum: killed.append((pgid, signum)),
    )
    diagnostic._terminate_and_reap_kit_child(Process())  # noqa: SLF001
    assert killed == [
        (1234, diagnostic.signal.SIGTERM),
        (1234, diagnostic.signal.SIGKILL),
    ]


def test_child_entry_unblocks_parent_cleanup_signals_before_other_work():
    cleanup_signals = set(diagnostic.PARENT_CLEANUP_SIGNALS)
    previous_mask = diagnostic.signal.pthread_sigmask(
        diagnostic.signal.SIG_BLOCK, cleanup_signals
    )
    try:
        diagnostic._unblock_child_cleanup_signals()  # noqa: SLF001
        current_mask = diagnostic.signal.pthread_sigmask(
            diagnostic.signal.SIG_BLOCK, ()
        )
        assert cleanup_signals.isdisjoint(current_mask)
    finally:
        diagnostic.signal.pthread_sigmask(diagnostic.signal.SIG_SETMASK, previous_mask)
    tree = ast.parse(inspect.getsource(diagnostic._child_runtime_main))  # noqa: SLF001
    first_statement = tree.body[0].body[0]
    assert isinstance(first_statement, ast.Expr)
    assert isinstance(first_statement.value, ast.Call)
    assert isinstance(first_statement.value.func, ast.Name)
    assert first_statement.value.func.id == "_unblock_child_cleanup_signals"


def test_child_close_commit_has_no_post_close_python_io():
    source = inspect.getsource(diagnostic._child_runtime_main)  # noqa: SLF001
    success_sequence = (
        "try:\n"
        "            # The byte is only a pre-close intent.  The stdlib parent accepts\n"
        "            # success only after normal wait status, process-group drain, pipe\n"
        "            # EOF, immutable ready-marker validation, and artifact revalidation.\n"
        "            _write_child_result_byte(0, result_descriptor)\n"
        "            publish_immutable_json(args_cli.output_ready_marker, ready_payload)\n"
        "            simulation_app.close()\n"
        "        except BaseException:\n"
        "            os._exit(1)\n"
        "        os._exit(0)"
    )
    assert success_sequence in source
    assert "publish_immutable_exit_status" not in source
    tree = ast.parse(source)
    close_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "close"
    ]
    assert close_calls


def test_parent_validator_error_publishes_nonzero_status(monkeypatch, tmp_path):
    class ExitCalled(BaseException):
        pass

    args = type("Args", (), {"runtime_exit": tmp_path / "runtime.exit"})()
    published = []
    monkeypatch.setattr(diagnostic, "_parse_parent_runtime_args", lambda _argv: args)
    monkeypatch.setattr(diagnostic, "_run_kit_child", lambda _argv: (b"\x00", 0))
    monkeypatch.setattr(
        diagnostic,
        "_host_revalidate_runtime_artifacts",
        lambda _args: (_ for _ in ()).throw(ValueError("invalid artifacts")),
    )
    monkeypatch.setattr(
        diagnostic,
        "_publish_parent_exit_and_terminate",
        lambda path, code: published.append((path, code)),
    )
    monkeypatch.setattr(
        diagnostic.os, "_exit", lambda _code: (_ for _ in ()).throw(ExitCalled())
    )
    with pytest.raises(ExitCalled):
        diagnostic._parent_runtime_main([])  # noqa: SLF001
    assert published == [(args.runtime_exit, 1)]


@pytest.mark.parametrize("exit_code", [0, 1])
def test_parent_exit_transport_is_atomic_exact_and_forces_exit(
    monkeypatch, tmp_path, exit_code
):
    class ExitCalled(BaseException):
        pass

    observed = []
    monkeypatch.setattr(
        diagnostic.os,
        "_exit",
        lambda code: observed.append(code) or (_ for _ in ()).throw(ExitCalled()),
    )
    path = tmp_path / f"runtime-{exit_code}.exit"
    with pytest.raises(ExitCalled):
        diagnostic._publish_parent_exit_and_terminate(  # noqa: SLF001
            path, exit_code
        )
    assert path.read_bytes() == f"{exit_code}\n".encode()
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert path.stat().st_nlink == 1
    assert observed == [exit_code]


def _tensor(
    shape,
    *,
    device=diagnostic.PINNED_CACHED_DEVICE,
    dtype=diagnostic.PINNED_TENSOR_DTYPE,
):
    size = math.prod(shape)
    return {
        "shape": shape,
        "dtype": dtype,
        "device": device,
        "values": [0.0] * size,
        "finite_mask": [True] * size,
        "finite_count": size,
        "nonfinite": [],
    }


def _trace_vector(values):
    values = list(values)
    return {
        "values": values,
        "finite_mask": [True] * len(values),
        "finite_count": len(values),
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("finite_count", 7.0),
        ("finite_mask", [1] * 7),
    ],
)
def test_arm_trace_vector_rejects_equality_only_type_impostors(field, value):
    vector = _trace_vector([0.0] * 7)
    vector[field] = value
    with pytest.raises(diagnostic.GripperImpulseDiagnosticError):
        diagnostic._validate_arm_trace_vector(  # noqa: SLF001
            vector, field="arm vector", width=7
        )


def _snapshot(*, timestamp: float = 1.0, gripper_target: float = 0.0):
    joint_count = len(diagnostic.EXPECTED_DROID_JOINT_NAMES)
    body_count = len(diagnostic.EXPECTED_DROID_BODY_NAMES)
    joint = _tensor([joint_count])
    body = _tensor([body_count, 6])
    snapshot = {
        "articulation_data_sim_timestamp": timestamp,
        "joint_names": list(diagnostic.EXPECTED_DROID_JOINT_NAMES),
        "joint_position_rad": copy.deepcopy(joint),
        "joint_velocity_rad_s": copy.deepcopy(joint),
        "joint_acceleration_rad_s2": copy.deepcopy(joint),
        "joint_position_target_rad": copy.deepcopy(joint),
        "joint_velocity_target_rad_s": copy.deepcopy(joint),
        "joint_effort_target_nm": copy.deepcopy(joint),
        "approximate_pd_computed_torque_nm": copy.deepcopy(joint),
        "approximate_pd_applied_torque_nm": copy.deepcopy(joint),
        "physx_joint_position_rad": copy.deepcopy(joint),
        "physx_joint_velocity_rad_s": copy.deepcopy(joint),
        "physx_projected_joint_force_nm": copy.deepcopy(joint),
        "physx_joint_velocity_limit_rad_s": copy.deepcopy(joint),
        "physx_joint_effort_limit_nm": copy.deepcopy(joint),
        "physx_joint_stiffness_nm_per_rad": copy.deepcopy(joint),
        "physx_joint_damping_nm_s_per_rad": copy.deepcopy(joint),
        "body_names": list(diagnostic.EXPECTED_DROID_BODY_NAMES),
        "body_com_velocity_world": copy.deepcopy(body),
        "body_com_acceleration_world": copy.deepcopy(body),
        "physx_link_incoming_joint_wrench_child_joint_frame": copy.deepcopy(body),
        "incoming_joint_wrench_semantics": (
            "physx_link_incoming_joint_total_6d_wrench_child_joint_frame_v1"
        ),
    }
    gripper_index = diagnostic.EXPECTED_DROID_JOINT_NAMES.index("finger_joint")
    snapshot["joint_position_target_rad"]["values"][gripper_index] = gripper_target
    computed_torque = boundary._float32_multiply(  # noqa: SLF001
        diagnostic.PROBED_GRIPPER_DRIVE_FLOAT32_VALUES["stiffness_nm_per_rad"],
        gripper_target,
    )
    effort_limit = diagnostic.PROBED_GRIPPER_DRIVE_FLOAT32_VALUES["effort_limit_nm"]
    applied_torque = min(max(computed_torque, -effort_limit), effort_limit)
    snapshot["approximate_pd_computed_torque_nm"]["values"][gripper_index] = (
        computed_torque
    )
    snapshot["approximate_pd_applied_torque_nm"]["values"][gripper_index] = (
        applied_torque
    )
    for field, value in (
        (
            "physx_joint_velocity_limit_rad_s",
            diagnostic.PROBED_GRIPPER_DRIVE_FLOAT32_VALUES["velocity_limit_rad_s"],
        ),
        (
            "physx_joint_effort_limit_nm",
            diagnostic.PROBED_GRIPPER_DRIVE_FLOAT32_VALUES["effort_limit_nm"],
        ),
        (
            "physx_joint_stiffness_nm_per_rad",
            diagnostic.PROBED_GRIPPER_DRIVE_FLOAT32_VALUES["stiffness_nm_per_rad"],
        ),
        (
            "physx_joint_damping_nm_s_per_rad",
            diagnostic.PROBED_GRIPPER_DRIVE_FLOAT32_VALUES["damping_nm_s_per_rad"],
        ),
    ):
        snapshot[field]["device"] = diagnostic.PINNED_STATIC_PHYSX_DEVICE
        snapshot[field]["values"][gripper_index] = value
    return snapshot


def _failure_trace(*, mode: str, failure_step: int, failure_substep: int = 2):
    entries = []
    failure_apply_index = failure_step * 8 + failure_substep
    last_apply = failure_apply_index - 1
    first_apply = diagnostic.RELEVANT_POLICY_STEP_START * diagnostic.DECIMATION
    for apply_index in range(first_apply, last_apply + 1):
        policy_step, substep = divmod(apply_index, 8)
        original = diagnostic._expected_gripper_value(  # noqa: SLF001
            mode, policy_step, original=True
        )
        effective = diagnostic._expected_gripper_value(  # noqa: SLF001
            mode, policy_step, original=False
        )
        target = diagnostic.GRIPPER_CLOSED_TARGET_RAD if effective else 0.0
        previous_policy_step = (apply_index - 1) // diagnostic.DECIMATION
        previous_effective = diagnostic._expected_gripper_value(  # noqa: SLF001
            mode, previous_policy_step, original=False
        )
        previous_target = (
            diagnostic.GRIPPER_CLOSED_TARGET_RAD if previous_effective else 0.0
        )
        offset = apply_index - first_apply
        pre = _snapshot(
            timestamp=1.0 + offset * diagnostic.TIMESTAMP_DT_SECONDS,
            gripper_target=previous_target,
        )
        post = _snapshot(
            timestamp=1.0 + (offset + 1) * diagnostic.TIMESTAMP_DT_SECONDS,
            gripper_target=target,
        )
        entries.append(
            {
                "apply_index": apply_index,
                "policy_step": policy_step,
                "physics_substep": substep,
                "original_gripper_closed_action": original,
                "effective_gripper_closed_action": effective,
                "raw_action_at_stage": effective,
                "processed_target_at_stage_rad": target,
                "pre_apply": pre,
                "target_after_setter_rad": {
                    "shape": [1],
                    "dtype": diagnostic.PINNED_TENSOR_DTYPE,
                    "device": diagnostic.PINNED_CACHED_DEVICE,
                    "values": [target],
                    "finite_mask": [True],
                    "finite_count": 1,
                    "nonfinite": [],
                },
                "post_physics": post,
                "finalization_reason": (
                    "arm_guard_exception"
                    if apply_index == last_apply
                    else "next_gripper_apply"
                ),
            }
        )
    return {
        "schema_version": 1,
        "profile": diagnostic.FINGER_TRACE_PROFILE,
        "capacity": 48,
        "relevant_policy_step_start": diagnostic.RELEVANT_POLICY_STEP_START,
        "relevant_policy_step_end": 117,
        "total_staged_apply_count": failure_apply_index,
        "total_finalized_apply_count": failure_apply_index,
        "pending_apply_count": 0,
        "dropped_relevant_entry_count": 0,
        "tensor_capture_contract": diagnostic._expected_tensor_capture_contract(),  # noqa: SLF001
        "timestamp_contract": copy.deepcopy(diagnostic.TIMESTAMP_CONTRACT),
        "entries": entries,
    }


def test_finger_trace_closed_schema_and_terminal_count_contract():
    plan, _ = diagnostic.build_action_plan(_actions(), mode="exact")
    outcome = _diagnostic_outcome("exact", 115)
    trace = _failure_trace(mode="exact", failure_step=115)
    arm_safety = {"counters": {"apply_calls": 923}}
    diagnostic.validate_finger_trace(
        trace,
        action_plan=plan,
        outcome=outcome,
        arm_safety=arm_safety,
        gripper_drive=_gripper_contract(),
    )
    tampered = copy.deepcopy(trace)
    tampered["entries"][-1]["unexpected"] = True
    with pytest.raises(diagnostic.GripperImpulseDiagnosticError, match="schema"):
        diagnostic.validate_finger_trace(
            tampered,
            action_plan=plan,
            outcome=outcome,
            arm_safety=arm_safety,
            gripper_drive=_gripper_contract(),
        )


@pytest.mark.parametrize(
    ("failure_step", "classification"),
    [
        (115, "delayed_close_did_not_shift_reference"),
        (116, "delayed_close_shifted_reference_one_policy_step"),
    ],
)
def test_delay_mode_accepts_no_shift_and_one_step_shift(
    failure_step: int, classification: str
):
    mode = "delay_first_close_one_step"
    plan, _ = diagnostic.build_action_plan(_actions(), mode=mode)
    outcome = _diagnostic_outcome(mode, failure_step)
    assert outcome["timing_classification"] == classification
    diagnostic._validate_outcome(outcome, mode=mode)  # noqa: SLF001
    trace = _failure_trace(mode=mode, failure_step=failure_step)
    arm_calls = failure_step * 8 + 3
    diagnostic.validate_finger_trace(
        trace,
        action_plan=plan,
        outcome=outcome,
        arm_safety={"counters": {"apply_calls": arm_calls}},
        gripper_drive=_gripper_contract(),
    )
    assert trace["total_staged_apply_count"] == arm_calls - 1
    assert trace["entries"][-1]["apply_index"] == failure_step * 8 + 1


def test_unexpected_relevant_window_failure_is_complete_but_inconclusive():
    mode = "exact"
    failure_step = diagnostic.FAILURE_POLICY_STEP_START
    failure_substep = 0
    plan, _ = diagnostic.build_action_plan(_actions(), mode=mode)
    outcome = _diagnostic_outcome(mode, failure_step, failure_substep)
    assert (
        outcome["timing_classification"]
        == "unexpected_complete_failure_timing_inconclusive"
    )
    diagnostic._validate_outcome(outcome, mode=mode)  # noqa: SLF001
    trace = _failure_trace(
        mode=mode,
        failure_step=failure_step,
        failure_substep=failure_substep,
    )
    diagnostic.validate_finger_trace(
        trace,
        action_plan=plan,
        outcome=outcome,
        arm_safety={"counters": {"apply_calls": outcome["failure_apply_index"] + 1}},
        gripper_drive=_gripper_contract(),
    )
    assert trace["entries"][-1]["apply_index"] == outcome["failure_apply_index"] - 1


@pytest.mark.parametrize(
    "case",
    [
        "joint_order",
        "body_order",
        "nonfinite",
        "cached_direct",
        "cached_device",
        "dynamic_device",
        "static_device",
        "wrong_dtype",
        "swapped_contract_devices",
        "wrong_field_classification",
        "probe_identity",
        "timestamp_type",
        "timestamp_cadence",
        "setter_post_target",
        "drive_binding",
        "torque_binding",
        "timestamp_contract",
    ],
)
def test_finger_trace_rejects_every_root_causal_binding_drift(case: str):
    plan, _ = diagnostic.build_action_plan(_actions(), mode="exact")
    outcome = _diagnostic_outcome("exact", 115)
    trace = _failure_trace(mode="exact", failure_step=115)
    snapshot = trace["entries"][0]["pre_apply"]
    gripper_index = diagnostic.EXPECTED_DROID_JOINT_NAMES.index("finger_joint")
    if case == "joint_order":
        snapshot["joint_names"][0], snapshot["joint_names"][1] = (
            snapshot["joint_names"][1],
            snapshot["joint_names"][0],
        )
    elif case == "body_order":
        snapshot["body_names"][0], snapshot["body_names"][1] = (
            snapshot["body_names"][1],
            snapshot["body_names"][0],
        )
    elif case == "nonfinite":
        tensor = snapshot["physx_projected_joint_force_nm"]
        tensor["finite_mask"][0] = False
        tensor["finite_count"] -= 1
        tensor["nonfinite"] = [{"flat_index": 0, "kind": "nan"}]
    elif case == "cached_direct":
        snapshot["physx_joint_position_rad"]["values"][0] = 1.0
    elif case == "cached_device":
        snapshot["joint_acceleration_rad_s2"]["device"] = "cpu"
    elif case == "dynamic_device":
        snapshot["physx_projected_joint_force_nm"]["device"] = "cuda:1"
    elif case == "static_device":
        snapshot["physx_joint_velocity_limit_rad_s"]["device"] = "cuda:0"
    elif case == "wrong_dtype":
        snapshot["physx_joint_stiffness_nm_per_rad"]["dtype"] = "torch.float64"
    elif case == "swapped_contract_devices":
        contract = trace["tensor_capture_contract"]
        contract["cached_articulation_device"] = "cpu"
        contract["static_physx_device"] = "cuda:0"
    elif case == "wrong_field_classification":
        contract = trace["tensor_capture_contract"]
        contract["cached_articulation_fields"][0] = "physx_joint_velocity_limit_rad_s"
    elif case == "probe_identity":
        trace["tensor_capture_contract"]["authoritative_device_probe"][
            "result_sha256"
        ] = "0" * 64
    elif case == "timestamp_type":
        snapshot["articulation_data_sim_timestamp"] = True
    elif case == "timestamp_cadence":
        trace["entries"][0]["post_physics"]["articulation_data_sim_timestamp"] += 1e-3
    elif case == "setter_post_target":
        trace["entries"][0]["target_after_setter_rad"]["values"][0] = 1.0
    elif case == "drive_binding":
        snapshot["physx_joint_effort_limit_nm"]["values"][gripper_index] = 199.0
    elif case == "torque_binding":
        trace["entries"][0]["post_physics"]["approximate_pd_computed_torque_nm"][
            "values"
        ][gripper_index] = 1.0
    else:
        trace["timestamp_contract"]["absolute_tolerance_seconds"] = 1.0
    with pytest.raises(diagnostic.GripperImpulseDiagnosticError):
        diagnostic.validate_finger_trace(
            trace,
            action_plan=plan,
            outcome=outcome,
            arm_safety={"counters": {"apply_calls": 923}},
            gripper_drive=_gripper_contract(),
        )


@pytest.mark.parametrize(
    "field",
    [
        "reference_exact_failure_policy_step",
        "reference_exact_failure_physics_substep",
        "failure_policy_step",
        "failure_physics_substep",
        "failure_apply_index",
        "last_attempted_policy_step",
    ],
)
def test_outcome_rejects_boolean_integer_impersonation(field: str):
    outcome = _diagnostic_outcome("exact", 115)
    outcome[field] = True
    with pytest.raises(diagnostic.GripperImpulseDiagnosticError):
        diagnostic._validate_outcome(outcome, mode="exact")  # noqa: SLF001


@pytest.mark.parametrize("field", ["schema_version", "fixture_action_count"])
def test_action_plan_rejects_boolean_integer_impersonation(field: str):
    plan, _ = diagnostic.build_action_plan(_actions(), mode="exact")
    plan[field] = True
    with pytest.raises(diagnostic.GripperImpulseDiagnosticError):
        diagnostic.validate_action_plan(plan)


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "capacity",
        "relevant_policy_step_start",
        "relevant_policy_step_end",
        "total_staged_apply_count",
        "total_finalized_apply_count",
        "pending_apply_count",
        "dropped_relevant_entry_count",
    ],
)
def test_finger_trace_rejects_boolean_count_impersonation(field: str):
    plan, _ = diagnostic.build_action_plan(_actions(), mode="exact")
    outcome = _diagnostic_outcome("exact", 115)
    trace = _failure_trace(mode="exact", failure_step=115)
    trace[field] = True
    with pytest.raises(diagnostic.GripperImpulseDiagnosticError):
        diagnostic.validate_finger_trace(
            trace,
            action_plan=plan,
            outcome=outcome,
            arm_safety={"counters": {"apply_calls": 923}},
            gripper_drive=_gripper_contract(),
        )


@pytest.mark.parametrize("field", ["apply_index", "policy_step", "physics_substep"])
def test_finger_entry_rejects_boolean_index_impersonation(field: str):
    plan, _ = diagnostic.build_action_plan(_actions(), mode="exact")
    outcome = _diagnostic_outcome("exact", 115)
    trace = _failure_trace(mode="exact", failure_step=115)
    trace["entries"][0][field] = True
    with pytest.raises(diagnostic.GripperImpulseDiagnosticError):
        diagnostic.validate_finger_trace(
            trace,
            action_plan=plan,
            outcome=outcome,
            arm_safety={"counters": {"apply_calls": 923}},
            gripper_drive=_gripper_contract(),
        )


@pytest.mark.parametrize(
    "field", sorted(diagnostic.SOLVER_CONTRACT_FIELDS - {"profile"})
)
def test_solver_contract_rejects_boolean_integer_impersonation(field: str):
    solver = {
        "profile": diagnostic.SOLVER_CHANGE_PROFILE,
        "configured_solver_velocity_iterations_before_eef_setup": 0,
        "configured_solver_velocity_iterations_after_eef_setup": 1,
        "live_solver_velocity_iterations": 1,
        "live_solver_position_iterations": 64,
        "live_physx_solver_type": 1,
    }
    solver[field] = True
    with pytest.raises(diagnostic.GripperImpulseDiagnosticError):
        diagnostic._validate_solver_contract(solver)  # noqa: SLF001


def _scalar_evidence(value, *, device):
    tensor = _tensor([1, 1], device=device)
    tensor["values"] = [value]
    return tensor


def _gripper_contract(*, candidate_enabled=False):
    probed = (
        diagnostic.GRIPPER_VELOCITY_LIMIT_CANDIDATE_FLOAT32_VALUES
        if candidate_enabled
        else diagnostic.PROBED_GRIPPER_DRIVE_FLOAT32_VALUES
    )
    profile = (
        diagnostic.GRIPPER_VELOCITY_LIMIT_CANDIDATE_DRIVE_PROFILE
        if candidate_enabled
        else diagnostic.GRIPPER_DRIVE_PROFILE
    )
    return {
        "profile": profile,
        "actuator_name": "gripper",
        "joint_names": ["finger_joint"],
        "joint_indices": [7],
        "action_term_joint_names": ["finger_joint"],
        "action_term_joint_indices": [7],
        "actuator_joint_names": ["finger_joint"],
        "actuator_joint_indices": [7],
        "authoritative_device_probe": copy.deepcopy(diagnostic.DEVICE_PROBE_EVIDENCE),
        "configured_before_articulation_build": {
            "legacy_velocity_limit_rad_s": 5.0,
            "velocity_limit_sim_rad_s": 5.0 if candidate_enabled else None,
            "legacy_effort_limit_nm": 200.0,
            "effort_limit_sim_nm": 200.0 if candidate_enabled else None,
            "stiffness": None,
            "damping": None,
        },
        "live_actuator": {
            "cfg_velocity_limit": 5.0 if candidate_enabled else None,
            "cfg_velocity_limit_sim": 5.0 if candidate_enabled else None,
            "cfg_effort_limit": 200.0,
            "cfg_effort_limit_sim": 200.0,
            "cfg_stiffness": None,
            "cfg_damping": None,
            "resolved_velocity_limit_rad_s": _scalar_evidence(
                probed["velocity_limit_rad_s"],
                device=diagnostic.PINNED_ACTUATOR_DEVICE,
            ),
            "resolved_velocity_limit_sim_rad_s": _scalar_evidence(
                probed["velocity_limit_rad_s"],
                device=diagnostic.PINNED_ACTUATOR_DEVICE,
            ),
            "resolved_effort_limit_nm": _scalar_evidence(
                probed["effort_limit_nm"], device=diagnostic.PINNED_ACTUATOR_DEVICE
            ),
            "resolved_effort_limit_sim_nm": _scalar_evidence(
                probed["effort_limit_nm"], device=diagnostic.PINNED_ACTUATOR_DEVICE
            ),
            "resolved_stiffness_nm_per_rad": _scalar_evidence(
                probed["stiffness_nm_per_rad"],
                device=diagnostic.PINNED_ACTUATOR_DEVICE,
            ),
            "resolved_damping_nm_s_per_rad": _scalar_evidence(
                probed["damping_nm_s_per_rad"],
                device=diagnostic.PINNED_ACTUATOR_DEVICE,
            ),
        },
        "live_physx_readback": {
            "velocity_limit_rad_s": _scalar_evidence(
                probed["velocity_limit_rad_s"],
                device=diagnostic.PINNED_STATIC_PHYSX_DEVICE,
            ),
            "effort_limit_nm": _scalar_evidence(
                probed["effort_limit_nm"],
                device=diagnostic.PINNED_STATIC_PHYSX_DEVICE,
            ),
            "stiffness_nm_per_rad": _scalar_evidence(
                probed["stiffness_nm_per_rad"],
                device=diagnostic.PINNED_STATIC_PHYSX_DEVICE,
            ),
            "damping_nm_s_per_rad": _scalar_evidence(
                probed["damping_nm_s_per_rad"],
                device=diagnostic.PINNED_STATIC_PHYSX_DEVICE,
            ),
        },
        "legacy_velocity_limit_behavior": (
            (
                "isaaclab_2p3_explicit_velocity_limit_sim_5_enforced_"
                "eef_diagnostic_only_v1"
            )
            if candidate_enabled
            else (
                "isaaclab_2p3_implicit_legacy_velocity_limit_5_ignored_"
                "velocity_limit_sim_unset_v1"
            )
        ),
        "effort_limit_behavior": (
            "implicit_equal_legacy_and_sim_effort_limit_200_enforced_v1"
            if candidate_enabled
            else (
                "implicit_legacy_effort_limit_200_promoted_to_effort_limit_sim_"
                "and_enforced_v1"
            )
        ),
        "incoming_joint_wrench_semantics": (
            "physx_total_incoming_joint_wrench_not_contact_force_child_joint_frame_v1"
        ),
        "computed_applied_torque_semantics": (
            "isaaclab_implicit_actuator_approximate_pd_preclip_and_"
            "effortlimit_clipped_v1"
        ),
    }


def test_gripper_contract_rejects_legacy_five_as_effective_physx_velocity():
    contract = _gripper_contract()
    live = contract["live_actuator"]
    live["resolved_velocity_limit_rad_s"]["values"] = [5.0]
    live["resolved_velocity_limit_sim_rad_s"]["values"] = [5.0]
    contract["live_physx_readback"]["velocity_limit_rad_s"]["values"] = [5.0]
    with pytest.raises(diagnostic.GripperImpulseDiagnosticError, match="job1098162"):
        diagnostic._validate_gripper_drive_contract(contract)  # noqa: SLF001


def test_gripper_contract_accepts_job1098162_resolved_values_not_legacy_five():
    assert diagnostic.PROBED_GRIPPER_DRIVE_FLOAT32_VALUES == {
        "velocity_limit_rad_s": 8.726646423339844,
        "effort_limit_nm": 200.0,
        "stiffness_nm_per_rad": 5729.578125,
        "damping_nm_s_per_rad": 0.011459155939519405,
    }
    contract = _gripper_contract()
    physx = contract["live_physx_readback"]
    validated = diagnostic._validate_gripper_drive_contract(contract)  # noqa: SLF001
    assert (
        validated["configured_before_articulation_build"]["legacy_velocity_limit_rad_s"]
        == 5.0
    )
    assert physx["velocity_limit_rad_s"]["values"] != [5.0]


def test_gripper_candidate_contract_requires_exact_split_device_five_rad_s_cap():
    contract = _gripper_contract(candidate_enabled=True)

    validated = diagnostic._validate_gripper_drive_contract(  # noqa: SLF001
        contract,
        expected_profile=diagnostic.GRIPPER_VELOCITY_LIMIT_CANDIDATE_DRIVE_PROFILE,
    )

    assert (
        validated["profile"]
        == diagnostic.GRIPPER_VELOCITY_LIMIT_CANDIDATE_DRIVE_PROFILE
    )
    assert validated["configured_before_articulation_build"] == {
        "legacy_velocity_limit_rad_s": 5.0,
        "velocity_limit_sim_rad_s": 5.0,
        "legacy_effort_limit_nm": 200.0,
        "effort_limit_sim_nm": 200.0,
        "stiffness": None,
        "damping": None,
    }
    actuator = validated["live_actuator"]["resolved_velocity_limit_rad_s"]
    physx = validated["live_physx_readback"]["velocity_limit_rad_s"]
    assert validated["live_actuator"]["cfg_velocity_limit"] == 5.0
    assert validated["live_actuator"]["cfg_velocity_limit_sim"] == 5.0
    assert actuator["device"] == diagnostic.PINNED_ACTUATOR_DEVICE
    assert physx["device"] == diagnostic.PINNED_STATIC_PHYSX_DEVICE
    assert actuator["values"] == physx["values"] == [5.0]


@pytest.mark.parametrize("candidate_enabled", [False, True])
def test_gripper_drive_contract_rejects_independent_profile_swap(candidate_enabled):
    expected_profile = (
        diagnostic.GRIPPER_DRIVE_PROFILE
        if candidate_enabled
        else diagnostic.GRIPPER_VELOCITY_LIMIT_CANDIDATE_DRIVE_PROFILE
    )
    with pytest.raises(
        diagnostic.GripperImpulseDiagnosticError,
        match="independent expectation",
    ):
        diagnostic._validate_gripper_drive_contract(  # noqa: SLF001
            _gripper_contract(candidate_enabled=candidate_enabled),
            expected_profile=expected_profile,
        )


def test_gripper_drive_contract_rejects_open_ended_expected_profile():
    with pytest.raises(
        diagnostic.GripperImpulseDiagnosticError,
        match="expected gripper drive profile",
    ):
        diagnostic._validate_gripper_drive_contract(  # noqa: SLF001
            _gripper_contract(candidate_enabled=True),
            expected_profile="candidate",
        )


def test_omitted_runtime_candidate_flag_cannot_satisfy_candidate_finalizer_intent(
    tmp_path,
):
    runtime = diagnostic.build_runtime_parser().parse_args(
        [
            "--output-json",
            str(tmp_path / "capture.json"),
            "--output-video",
            str(tmp_path / "capture.mp4"),
            "--output-ready-marker",
            str(tmp_path / "ready.json"),
            "--runtime-exit",
            str(tmp_path / "runtime.exit"),
            "--mode",
            "exact",
            "--expected-source-sha256",
            "0" * 64,
            "--expected-source-size-bytes",
            "1",
        ]
    )
    runtime_profile = diagnostic._selected_gripper_drive_profile(  # noqa: SLF001
        runtime.enable_gripper_velocity_limit_candidate
    )

    assert runtime_profile == diagnostic.GRIPPER_DRIVE_PROFILE
    with pytest.raises(
        diagnostic.GripperImpulseDiagnosticError,
        match="independent expectation",
    ):
        diagnostic._validate_gripper_drive_contract(  # noqa: SLF001
            _gripper_contract(candidate_enabled=False),
            expected_profile=(
                diagnostic.GRIPPER_VELOCITY_LIMIT_CANDIDATE_DRIVE_PROFILE
            ),
        )


def test_gripper_candidate_contract_cross_binds_full_static_physx_snapshot():
    contract = diagnostic._validate_gripper_drive_contract(  # noqa: SLF001
        _gripper_contract(candidate_enabled=True)
    )
    snapshot = {"joint_names": list(diagnostic.EXPECTED_DROID_JOINT_NAMES)}
    joint_index = diagnostic.EXPECTED_DROID_JOINT_NAMES.index("finger_joint")
    for snapshot_field, drive_field in (
        ("physx_joint_velocity_limit_rad_s", "velocity_limit_rad_s"),
        ("physx_joint_effort_limit_nm", "effort_limit_nm"),
        ("physx_joint_stiffness_nm_per_rad", "stiffness_nm_per_rad"),
        ("physx_joint_damping_nm_s_per_rad", "damping_nm_s_per_rad"),
    ):
        evidence = _tensor([len(diagnostic.EXPECTED_DROID_JOINT_NAMES)])
        evidence["device"] = diagnostic.PINNED_STATIC_PHYSX_DEVICE
        evidence["values"][joint_index] = contract["live_physx_readback"][drive_field][
            "values"
        ][0]
        snapshot[snapshot_field] = evidence

    diagnostic._validate_snapshot_gripper_binding(  # noqa: SLF001
        snapshot,
        gripper_drive=contract,
        field="candidate snapshot",
    )

    snapshot["physx_joint_velocity_limit_rad_s"]["values"][joint_index] = (
        diagnostic.PROBED_GRIPPER_DRIVE_FLOAT32_VALUES["velocity_limit_rad_s"]
    )
    with pytest.raises(
        diagnostic.GripperImpulseDiagnosticError,
        match="live-drive identity",
    ):
        diagnostic._validate_snapshot_gripper_binding(  # noqa: SLF001
            snapshot,
            gripper_drive=contract,
            field="candidate snapshot",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "legacy_profile",
        "legacy_value",
        "swapped_devices",
        "unset_legacy_cfg",
        "unset_sim_cfg",
    ],
)
def test_gripper_candidate_contract_rejects_profile_value_device_or_cfg_drift(
    mutation,
):
    contract = _gripper_contract(candidate_enabled=True)
    if mutation == "legacy_profile":
        contract["profile"] = diagnostic.GRIPPER_DRIVE_PROFILE
    elif mutation == "legacy_value":
        for field in (
            "resolved_velocity_limit_rad_s",
            "resolved_velocity_limit_sim_rad_s",
        ):
            contract["live_actuator"][field]["values"] = [8.726646423339844]
        contract["live_physx_readback"]["velocity_limit_rad_s"]["values"] = [
            8.726646423339844
        ]
    elif mutation == "swapped_devices":
        contract["live_actuator"]["resolved_velocity_limit_rad_s"]["device"] = "cpu"
        contract["live_physx_readback"]["velocity_limit_rad_s"]["device"] = "cuda:0"
    elif mutation == "unset_legacy_cfg":
        contract["live_actuator"]["cfg_velocity_limit"] = None
    else:
        contract["live_actuator"]["cfg_velocity_limit_sim"] = None
    with pytest.raises(diagnostic.GripperImpulseDiagnosticError):
        diagnostic._validate_gripper_drive_contract(contract)  # noqa: SLF001


@pytest.mark.parametrize(
    ("live_fields", "physx_field"),
    [
        (
            [
                "resolved_velocity_limit_rad_s",
                "resolved_velocity_limit_sim_rad_s",
            ],
            "velocity_limit_rad_s",
        ),
        (
            ["resolved_effort_limit_nm", "resolved_effort_limit_sim_nm"],
            "effort_limit_nm",
        ),
        (["resolved_stiffness_nm_per_rad"], "stiffness_nm_per_rad"),
        (["resolved_damping_nm_s_per_rad"], "damping_nm_s_per_rad"),
    ],
)
def test_gripper_contract_rejects_mirrored_but_unprobed_drive_values(
    live_fields, physx_field
):
    contract = _gripper_contract()
    wrong_value = diagnostic.PROBED_GRIPPER_DRIVE_FLOAT32_VALUES[physx_field] + 1.0
    for live_field in live_fields:
        contract["live_actuator"][live_field]["values"] = [wrong_value]
    contract["live_physx_readback"][physx_field]["values"] = [wrong_value]
    with pytest.raises(diagnostic.GripperImpulseDiagnosticError, match="job1098162"):
        diagnostic._validate_gripper_drive_contract(contract)  # noqa: SLF001


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("joint_names", ["panda_joint7"]),
        ("joint_indices", [6]),
        ("action_term_joint_names", ["panda_joint7"]),
        ("action_term_joint_indices", [6]),
        ("actuator_joint_names", ["panda_joint7"]),
        ("actuator_joint_indices", [6]),
    ],
)
def test_gripper_contract_rejects_action_term_or_actuator_identity_drift(field, value):
    contract = _gripper_contract()
    contract[field] = value
    with pytest.raises(diagnostic.GripperImpulseDiagnosticError, match="cross-binding"):
        diagnostic._validate_gripper_drive_contract(contract)  # noqa: SLF001


@pytest.mark.parametrize("field", ["cfg_stiffness", "cfg_damping"])
def test_gripper_contract_requires_probed_null_live_gain_cfg(field):
    contract = _gripper_contract()
    contract["live_actuator"][field] = 1.0
    with pytest.raises(
        diagnostic.GripperImpulseDiagnosticError,
        match="implicit-actuator cfg behavior",
    ):
        diagnostic._validate_gripper_drive_contract(contract)  # noqa: SLF001


def test_gripper_mirror_excludes_only_device_and_still_rejects_value_mismatch():
    contract = _gripper_contract()
    actuator = contract["live_actuator"]["resolved_velocity_limit_rad_s"]
    static_physx = contract["live_physx_readback"]["velocity_limit_rad_s"]
    assert actuator["device"] == diagnostic.PINNED_ACTUATOR_DEVICE
    assert static_physx["device"] == diagnostic.PINNED_STATIC_PHYSX_DEVICE
    assert diagnostic._tensor_evidence_equal_excluding_device(  # noqa: SLF001
        actuator, static_physx
    )
    actuator["values"][0] = 4.0
    assert not diagnostic._tensor_evidence_equal_excluding_device(  # noqa: SLF001
        actuator, static_physx
    )
    with pytest.raises(diagnostic.GripperImpulseDiagnosticError, match="mirror drift"):
        diagnostic._validate_gripper_drive_contract(contract)  # noqa: SLF001


class _FakeDirectTensor:
    def __init__(self, *, device, shape, dtype=diagnostic.PINNED_TENSOR_DTYPE):
        self.device = device
        self.dtype = dtype
        self.shape = shape

    def clone(self):
        return self


def _fake_direct_robot(getter_name, tensor):
    view = SimpleNamespace(**{getter_name: lambda: tensor})
    return SimpleNamespace(root_physx_view=view, device="cuda:0")


def test_every_direct_physx_getter_has_closed_device_dtype_shape_contract():
    assert (
        set(diagnostic.DIRECT_PHYSX_GETTER_CONTRACT)
        == diagnostic.EXPECTED_DIRECT_PHYSX_GETTER_NAMES
    )
    assert {
        contract["snapshot_field"]
        for contract in diagnostic.DIRECT_PHYSX_GETTER_CONTRACT.values()
    } == set(diagnostic.DYNAMIC_PHYSX_TENSOR_FIELDS) | set(
        diagnostic.STATIC_PHYSX_TENSOR_FIELDS
    )
    for getter_name, contract in diagnostic.DIRECT_PHYSX_GETTER_CONTRACT.items():
        tensor = _FakeDirectTensor(device=contract["device"], shape=contract["shape"])
        assert (
            diagnostic._direct_physx_tensor(  # noqa: SLF001
                _fake_direct_robot(getter_name, tensor), getter_name
            )
            is tensor
        )


@pytest.mark.parametrize(
    ("getter_name", "mutation"),
    [
        ("get_dof_positions", {"device": "cpu"}),
        ("get_dof_max_velocities", {"device": "cuda:0"}),
        ("get_dof_stiffnesses", {"device": "cuda:1"}),
        ("get_dof_dampings", {"dtype": "torch.float64"}),
        ("get_link_velocities", {"shape": [1, 18, 7]}),
    ],
)
def test_direct_physx_getter_rejects_swapped_arbitrary_device_dtype_shape(
    getter_name, mutation
):
    contract = diagnostic.DIRECT_PHYSX_GETTER_CONTRACT[getter_name]
    values = {
        "device": contract["device"],
        "shape": contract["shape"],
        "dtype": diagnostic.PINNED_TENSOR_DTYPE,
        **mutation,
    }
    tensor = _FakeDirectTensor(**values)
    with pytest.raises(diagnostic.GripperImpulseDiagnosticError):
        diagnostic._direct_physx_tensor(  # noqa: SLF001
            _fake_direct_robot(getter_name, tensor), getter_name
        )


def test_direct_physx_getter_rejects_unclassified_getter():
    tensor = _FakeDirectTensor(device="cuda:0", shape=[1, 13])
    with pytest.raises(diagnostic.GripperImpulseDiagnosticError, match="unclassified"):
        diagnostic._direct_physx_tensor(  # noqa: SLF001
            _fake_direct_robot("get_unknown", tensor), "get_unknown"
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("nonfinite", "finite scalar"),
        ("negative_limit", "must be positive"),
        ("negative_gain", "must be nonnegative"),
        ("zero_physx_limit", "invalid sign"),
        ("mirror", "mirror drift"),
        ("dtype", "CUDA device/dtype"),
        ("device", "CUDA device/dtype"),
        ("physx_device", "CPU device/dtype"),
        ("swapped_devices", "CUDA device/dtype"),
        ("probe", "authoritative device probe"),
        ("shape", "finite scalar"),
    ],
)
def test_gripper_contract_rejects_invalid_live_drive_evidence(case, message):
    contract = _gripper_contract()
    live = contract["live_actuator"]
    physx = contract["live_physx_readback"]
    if case == "nonfinite":
        tensor = live["resolved_velocity_limit_rad_s"]
        tensor.update(
            {
                "values": [0.0],
                "finite_mask": [False],
                "finite_count": 0,
                "nonfinite": [{"flat_index": 0, "kind": "nan"}],
            }
        )
    elif case == "negative_limit":
        live["resolved_velocity_limit_rad_s"]["values"] = [-1.0]
    elif case == "negative_gain":
        live["resolved_stiffness_nm_per_rad"]["values"] = [-1.0]
    elif case == "zero_physx_limit":
        physx["velocity_limit_rad_s"]["values"] = [0.0]
    elif case == "mirror":
        live["resolved_velocity_limit_rad_s"]["values"] = [4.0]
    elif case == "dtype":
        live["resolved_velocity_limit_rad_s"]["dtype"] = "torch.float64"
    elif case == "device":
        live["resolved_velocity_limit_rad_s"]["device"] = "cpu"
    elif case == "physx_device":
        physx["velocity_limit_rad_s"]["device"] = "cuda:0"
    elif case == "swapped_devices":
        live["resolved_velocity_limit_rad_s"]["device"] = "cpu"
        physx["velocity_limit_rad_s"]["device"] = "cuda:0"
    elif case == "probe":
        contract["authoritative_device_probe"]["result_sha256"] = "0" * 64
    elif case == "shape":
        tensor = live["resolved_velocity_limit_rad_s"]
        tensor.update(
            {
                "shape": [2],
                "values": [5.0, 5.0],
                "finite_mask": [True, True],
                "finite_count": 2,
            }
        )
    with pytest.raises(diagnostic.GripperImpulseDiagnosticError, match=message):
        diagnostic._validate_gripper_drive_contract(contract)  # noqa: SLF001


def _terminal_crossbind_fixture(*, failure: bool):
    arm_names = [f"panda_joint{index}" for index in range(1, 8)]
    joint_names = list(diagnostic.EXPECTED_DROID_JOINT_NAMES)
    values = [0.0] * len(joint_names)

    def evidence(all_values):
        tensor = _tensor([len(all_values)])
        tensor["values"] = list(all_values)
        return tensor

    post = _snapshot()
    post["joint_names"] = joint_names
    for field in (
        "joint_position_rad",
        "joint_velocity_rad_s",
        "joint_acceleration_rad_s2",
        "joint_position_target_rad",
        "joint_velocity_target_rad_s",
        "joint_effort_target_nm",
        "approximate_pd_computed_torque_nm",
        "approximate_pd_applied_torque_nm",
        "physx_joint_position_rad",
        "physx_joint_velocity_rad_s",
        "physx_projected_joint_force_nm",
        "physx_joint_velocity_limit_rad_s",
        "physx_joint_effort_limit_nm",
        "physx_joint_stiffness_nm_per_rad",
        "physx_joint_damping_nm_s_per_rad",
    ):
        post[field] = evidence(values)
        if field in diagnostic.STATIC_PHYSX_TENSOR_FIELDS:
            post[field]["device"] = diagnostic.PINNED_STATIC_PHYSX_DEVICE
    for field, arm_values, finger_value in (
        (
            "physx_joint_velocity_limit_rad_s",
            boundary.EXPECTED_VELOCITY_LIMITS_RAD_S,
            diagnostic.PROBED_GRIPPER_DRIVE_FLOAT32_VALUES["velocity_limit_rad_s"],
        ),
        (
            "physx_joint_effort_limit_nm",
            boundary.EXPECTED_EFFORT_LIMITS,
            diagnostic.PROBED_GRIPPER_DRIVE_FLOAT32_VALUES["effort_limit_nm"],
        ),
        (
            "physx_joint_stiffness_nm_per_rad",
            boundary.EXPECTED_JOINT_DRIVE_STIFFNESS,
            diagnostic.PROBED_GRIPPER_DRIVE_FLOAT32_VALUES["stiffness_nm_per_rad"],
        ),
        (
            "physx_joint_damping_nm_s_per_rad",
            boundary.EXPECTED_JOINT_DRIVE_DAMPING,
            diagnostic.PROBED_GRIPPER_DRIVE_FLOAT32_VALUES["damping_nm_s_per_rad"],
        ),
    ):
        post[field]["values"] = [
            *arm_values,
            finger_value,
            *([0.0] * (len(joint_names) - len(arm_values) - 1)),
        ]
    arm_vector = _trace_vector([0.0] * 7)
    total_completed = 922 if failure else 944
    entries = []
    for apply_index in range(total_completed - 64, total_completed):
        vectors = {
            field: _trace_vector([0.0] * width)
            for field, width in boundary.FAILURE_SUBSTEP_TRACE_VECTOR_WIDTHS.items()
        }
        vectors["current_eef_quaternion_wxyz"] = _trace_vector([1.0, 0.0, 0.0, 0.0])
        vectors["desired_eef_quaternion_wxyz"] = _trace_vector([1.0, 0.0, 0.0, 0.0])
        entries.append(
            {
                "apply_index": apply_index,
                "policy_step": apply_index // 8,
                "physics_substep": apply_index % 8,
                **vectors,
            }
        )
    trace = {
        "schema_version": 1,
        "profile": boundary.FAILURE_SUBSTEP_TRACE_PROFILE,
        "episode_index": 0,
        "capacity": boundary.FAILURE_SUBSTEP_TRACE_CAPACITY,
        "policy_step_capacity": boundary.FAILURE_SUBSTEP_TRACE_CAPACITY // 8,
        "decimation": 8,
        "joint_names": arm_names,
        "joint_drive_stiffness": list(boundary.EXPECTED_JOINT_DRIVE_STIFFNESS),
        "joint_drive_damping": list(boundary.EXPECTED_JOINT_DRIVE_DAMPING),
        "joint_effort_limits": list(boundary.EXPECTED_EFFORT_LIMITS),
        "effort_semantics": boundary.FAILURE_SUBSTEP_TRACE_EFFORT_SEMANTICS,
        "phase_contract": boundary.FAILURE_SUBSTEP_TRACE_PHASE_CONTRACT,
        "completed_entry_count": 64,
        "total_completed_entry_count": total_completed,
        "dropped_prefix_entry_count": total_completed - 64,
        "pending_entry_count": 0,
        "pending_apply_index": None,
        "entries": entries,
    }
    finger_trace = {
        "entries": [
            {
                "apply_index": apply_index,
                "pre_apply": copy.deepcopy(post),
                "post_physics": copy.deepcopy(post),
            }
            for apply_index in range(904, total_completed)
        ]
    }
    outcome = {
        "kind": (
            "allowed_velocity_guard_failure"
            if failure
            else "diagnostic_horizon_reached"
        ),
        "failure_policy_step": 115 if failure else None,
        "failure_apply_index": 922 if failure else None,
    }
    failure_evidence = (
        {
            "physx_arm_joint_pos_rad": copy.deepcopy(arm_vector),
            "physx_arm_joint_vel_rad_s": copy.deepcopy(arm_vector),
        }
        if failure
        else None
    )
    arm_safety = {"counters": {"apply_calls": total_completed + int(failure)}}
    return trace, finger_trace, outcome, arm_safety, failure_evidence


@pytest.mark.parametrize("failure", [False, True])
def test_arm_and_finger_terminal_state_are_cross_bound(failure: bool):
    trace, finger_trace, outcome, arm_safety, failure_evidence = (
        _terminal_crossbind_fixture(failure=failure)
    )
    diagnostic._validate_arm_substep_trace_terminal(  # noqa: SLF001
        trace,
        outcome=outcome,
        arm_safety=arm_safety,
        finger_trace=finger_trace,
        failure_evidence=failure_evidence,
    )


@pytest.mark.parametrize(
    "snapshot_field",
    ["joint_velocity_target_rad_s", "joint_effort_target_nm"],
)
def test_every_overlapping_arm_command_cross_bind_rejects_tamper(
    snapshot_field: str,
):
    trace, finger_trace, outcome, arm_safety, failure_evidence = (
        _terminal_crossbind_fixture(failure=True)
    )
    finger_trace["entries"][0]["pre_apply"][snapshot_field]["values"][0] = -1.0
    with pytest.raises(
        diagnostic.GripperImpulseDiagnosticError, match="overlapping entry"
    ):
        diagnostic._validate_arm_substep_trace_terminal(  # noqa: SLF001
            trace,
            outcome=outcome,
            arm_safety=arm_safety,
            finger_trace=finger_trace,
            failure_evidence=failure_evidence,
        )


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize(
    "direct_field",
    ["physx_joint_position_rad", "physx_joint_velocity_rad_s"],
)
def test_terminal_cached_direct_physx_cross_bind_rejects_tamper(
    failure: bool, direct_field: str
):
    trace, finger_trace, outcome, arm_safety, failure_evidence = (
        _terminal_crossbind_fixture(failure=failure)
    )
    finger_trace["entries"][-1]["post_physics"][direct_field]["values"][0] = -1.0
    with pytest.raises(
        diagnostic.GripperImpulseDiagnosticError, match="cached/direct PhysX"
    ):
        diagnostic._validate_arm_substep_trace_terminal(  # noqa: SLF001
            trace,
            outcome=outcome,
            arm_safety=arm_safety,
            finger_trace=finger_trace,
            failure_evidence=failure_evidence,
        )


def test_horizon_arm_trace_rejects_one_entry_null_header_shortcut():
    trace, _, _, _, _ = _terminal_crossbind_fixture(failure=False)
    malformed = {field: None for field in boundary.FAILURE_SUBSTEP_TRACE_FIELDS}
    malformed.update(
        {
            "total_completed_entry_count": 944,
            "completed_entry_count": 1,
            "dropped_prefix_entry_count": 943,
            "pending_entry_count": 0,
            "pending_apply_index": None,
            "entries": [copy.deepcopy(trace["entries"][-1])],
        }
    )
    with pytest.raises(
        diagnostic.GripperImpulseDiagnosticError, match="arm trace header"
    ):
        diagnostic._validate_closed_arm_substep_trace(  # noqa: SLF001
            malformed, expected_total_completed=944
        )


@pytest.mark.parametrize(
    "failure_field", ["physx_arm_joint_pos_rad", "physx_arm_joint_vel_rad_s"]
)
def test_terminal_failure_physx_cross_bind_rejects_tamper(failure_field: str):
    trace, finger_trace, outcome, arm_safety, failure_evidence = (
        _terminal_crossbind_fixture(failure=True)
    )
    failure_evidence[failure_field]["values"][0] = -1.0
    with pytest.raises(
        diagnostic.GripperImpulseDiagnosticError, match="failure terminal PhysX"
    ):
        diagnostic._validate_arm_substep_trace_terminal(  # noqa: SLF001
            trace,
            outcome=outcome,
            arm_safety=arm_safety,
            finger_trace=finger_trace,
            failure_evidence=failure_evidence,
        )


def _complete_failure_evidence(trace, finger_trace, arm_safety):
    snapshot = finger_trace["entries"][-1]["post_physics"]
    arm_indices = list(range(7))

    def vector(snapshot_field):
        return _trace_vector(
            snapshot[snapshot_field]["values"][index] for index in arm_indices
        )

    def tensor(snapshot_field):
        source = snapshot[snapshot_field]
        return {
            "shape": [7],
            "dtype": source["dtype"],
            "device": source["device"],
            "values": [source["values"][index] for index in arm_indices],
            "finite_mask": [True] * 7,
            "finite_count": 7,
            "nonfinite": [],
        }

    return {
        "policy_step": 115,
        "arm_joint_names": [f"panda_joint{index}" for index in range(1, 8)],
        "articulation_data_sim_timestamp": snapshot["articulation_data_sim_timestamp"],
        "arm_joint_pos_rad": vector("joint_position_rad"),
        "arm_joint_vel_rad_s": vector("joint_velocity_rad_s"),
        "arm_joint_target_rad": vector("joint_position_target_rad"),
        "arm_joint_velocity_target_rad_s": vector("joint_velocity_target_rad_s"),
        "arm_joint_effort_target_nm": vector("joint_effort_target_nm"),
        "physx_arm_joint_pos_rad": vector("physx_joint_position_rad"),
        "physx_arm_joint_vel_rad_s": vector("physx_joint_velocity_rad_s"),
        "cached_minus_physx_arm_joint_pos_rad": _trace_vector([0.0] * 7),
        "cached_minus_physx_arm_joint_vel_rad_s": _trace_vector([0.0] * 7),
        "physx_arm_velocity_limits_rad_s": _trace_vector(
            boundary.EXPECTED_VELOCITY_LIMITS_RAD_S
        ),
        "physx_arm_effort_limits": _trace_vector(boundary.EXPECTED_EFFORT_LIMITS),
        "physx_arm_projected_joint_force_generalized_si": tensor(
            "physx_projected_joint_force_nm"
        ),
        "physx_arm_stiffness_nm_per_rad": tensor("physx_joint_stiffness_nm_per_rad"),
        "physx_arm_damping_nm_s_per_rad": tensor("physx_joint_damping_nm_s_per_rad"),
        "arm_computed_torque": vector("approximate_pd_computed_torque_nm"),
        "arm_applied_torque": vector("approximate_pd_applied_torque_nm"),
        "ik_safety": copy.deepcopy(arm_safety),
        "controller_substep_trace": copy.deepcopy(trace),
        "controller_substep_trace_error": None,
    }


def _tamper_failure_evidence_field(evidence, field):
    vector_fields = {
        "arm_joint_pos_rad",
        "arm_joint_vel_rad_s",
        "arm_joint_target_rad",
        "arm_joint_velocity_target_rad_s",
        "arm_joint_effort_target_nm",
        "physx_arm_joint_pos_rad",
        "physx_arm_joint_vel_rad_s",
        "cached_minus_physx_arm_joint_pos_rad",
        "cached_minus_physx_arm_joint_vel_rad_s",
        "physx_arm_velocity_limits_rad_s",
        "physx_arm_effort_limits",
        "arm_computed_torque",
        "arm_applied_torque",
    }
    tensor_fields = {
        "physx_arm_projected_joint_force_generalized_si",
        "physx_arm_stiffness_nm_per_rad",
        "physx_arm_damping_nm_s_per_rad",
    }
    if field == "policy_step":
        evidence[field] = 116
    elif field == "arm_joint_names":
        evidence[field][0] = "wrong_joint"
    elif field == "articulation_data_sim_timestamp":
        evidence[field] = 2.0
    elif field in vector_fields | tensor_fields:
        evidence[field]["values"][0] += 1.0
    elif field == "ik_safety":
        evidence[field]["counters"]["apply_calls"] += 1
    elif field == "controller_substep_trace":
        evidence[field]["profile"] = "tampered"
    elif field == "controller_substep_trace_error":
        evidence[field] = "trace unavailable"
    else:  # pragma: no cover - the closed field set makes this unreachable.
        raise AssertionError(field)


@pytest.mark.parametrize("field", sorted(diagnostic.FAILURE_RUNTIME_EVIDENCE_FIELDS))
def test_every_failure_runtime_evidence_field_is_adversarially_bound(field):
    trace, finger_trace, outcome, arm_safety, _ = _terminal_crossbind_fixture(
        failure=True
    )
    evidence = _complete_failure_evidence(trace, finger_trace, arm_safety)
    diagnostic._validate_failure_runtime_evidence(  # noqa: SLF001
        evidence,
        outcome=outcome,
        arm_safety=arm_safety,
        arm_substep_trace=trace,
        finger_trace=finger_trace,
    )
    _tamper_failure_evidence_field(evidence, field)
    with pytest.raises(diagnostic.GripperImpulseDiagnosticError):
        diagnostic._validate_failure_runtime_evidence(  # noqa: SLF001
            evidence,
            outcome=outcome,
            arm_safety=arm_safety,
            arm_substep_trace=trace,
            finger_trace=finger_trace,
        )


@pytest.mark.parametrize("duplicate", ["ik_safety", "controller_substep_trace"])
def test_failure_runtime_duplicate_identity_rejects_numeric_type_drift(duplicate):
    trace, finger_trace, outcome, arm_safety, _ = _terminal_crossbind_fixture(
        failure=True
    )
    evidence = _complete_failure_evidence(trace, finger_trace, arm_safety)
    if duplicate == "ik_safety":
        evidence[duplicate]["counters"]["apply_calls"] = 923.0
    else:
        evidence[duplicate]["schema_version"] = 1.0
    with pytest.raises(diagnostic.GripperImpulseDiagnosticError):
        diagnostic._validate_failure_runtime_evidence(  # noqa: SLF001
            evidence,
            outcome=outcome,
            arm_safety=arm_safety,
            arm_substep_trace=trace,
            finger_trace=finger_trace,
        )
