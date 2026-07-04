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


def test_all_followup_variants_preserve_source_actions_and_pin_frozen_tail() -> None:
    _, source = diagnostic.load_actions()
    for variant in diagnostic.VARIANTS:
        effective = diagnostic.effective_actions(source, variant)
        assert effective == source
        assert effective is not source
    tail = diagnostic.frozen_tail_contract(source)
    assert tail == {
        "profile": "repeat_final_source_action_for_fixed_physics_tail_v1",
        "source_action_index": 293,
        "action_width": 8,
        "action_float32_sha256": diagnostic.TAIL_ACTION_FLOAT32_SHA256,
        "action_values_float32": source[-1],
        "policy_steps": 8,
        "physics_substeps_per_policy_step": 8,
        "physics_substeps": 64,
        "command_semantics": "exact_unmodified_final_source_action_repeat_v1",
    }


def snapshot(
    positions: list[float],
    velocities: list[float],
    accelerations: list[float],
    *,
    remaining: int,
) -> dict[str, Any]:
    zeros = [0.0] * 13
    return {
        "joint_pos_rad": list(positions[7:]),
        "joint_vel_rad_s": list(velocities[7:]),
        "joint_acc_rad_s2": list(accelerations[7:]),
        "joint_pos_target_rad": list(positions[7:]),
        "joint_vel_target_rad_s": zeros[7:],
        "joint_effort_target_nm": zeros[7:],
        "all_joint_pos_rad": list(positions),
        "all_joint_vel_rad_s": list(velocities),
        "all_joint_acc_rad_s2": list(accelerations),
        "all_joint_pos_target_rad": list(positions),
        "all_joint_vel_target_rad_s": list(zeros),
        "all_joint_effort_target_nm": list(zeros),
        "all_computed_torque_nm": list(zeros),
        "all_applied_torque_nm": list(zeros),
        "interlock": {
            "arm_apply_call_count": 1,
            "configured_substeps": 86,
            "remaining_substeps": remaining,
            "anchor_valid": remaining > 0,
            "activation_count": 1,
            "active_apply_count": 1,
            "released_apply_count": int(remaining == 0),
            "anchor_joint_pos_rad": [0.0] * 7,
            "release_ramp": {
                "enabled": False,
                "pending": False,
                "next_index": None,
                "target_apply_count": 0,
            },
        },
    }


def disabled_release_ramp_runtime() -> dict[str, Any]:
    return {
        "profile": "arm_post_interlock_linear_slew_cap_release_ramp16_runtime_v2",
        "enabled": False,
        "release_observed_count": 0,
        "ramp_started_count": 0,
        "ramp_completed_count": 0,
        "ramp_cancelled_by_reactivation_count": 0,
        "ramp_target_apply_count": 0,
        "ramp_limited_target_apply_count": 0,
        "applied_indices": [],
        "pending_at_report": False,
        "next_index_at_report": None,
        "max_abs_nominal_to_ramped_target_change_rad": 0.0,
        "overlay_entries": [],
        "gripper_target_or_state_write_count": 0,
    }


def test_trace_summary_uses_physx_mimic_equation_and_marks_release() -> None:
    # q_follower + G*q_driver == 0 for G=[-1,+1,-1,+1,+1].
    driver = 0.2
    positions = [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0] + [
        driver,
        driver,
        -driver,
        driver,
        -driver,
        -driver,
    ]
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
        variant="cap8_abrupt_release",
        outcome={
            "classification": "followup_incomplete_or_unbound_failure",
            "source_actions_completed": 0,
            "tail_policy_steps_completed": 0,
            "tail_physics_substeps_completed": 0,
            "parsed_numerical_failure": None,
        },
        release_ramp_runtime=disabled_release_ramp_runtime(),
    )
    assert cadence["entry_count"] == 2
    assert cadence["expected_entry_count"] is None
    assert cadence["source_action_segment"]["trace_entry_count"] == 2
    assert cadence["frozen_tail_segment"]["trace_entry_count"] == 0
    entries[1]["physics_substep"] = 7
    with pytest.raises(diagnostic.FullTraceAblationError, match="cadence drift"):
        diagnostic.validate_trace_cadence(
            entries,
            variant="cap8_abrupt_release",
            outcome={
                "classification": "followup_incomplete_or_unbound_failure",
                "source_actions_completed": 0,
                "tail_policy_steps_completed": 0,
                "tail_physics_substeps_completed": 0,
                "parsed_numerical_failure": None,
            },
            release_ramp_runtime=disabled_release_ramp_runtime(),
        )


def test_complete_trace_cadence_separates_294_actions_from_exact_64_tick_tail() -> None:
    positions = [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0] + [0.0] * 6
    velocities = [0.0] * 13
    accelerations = [0.0] * 13
    phase = snapshot(positions, velocities, accelerations, remaining=0)
    entry_count = diagnostic.ACTION_COUNT * diagnostic.DECIMATION + 64
    entries = [
        {
            "apply_index": apply_index,
            "policy_step": apply_index // diagnostic.DECIMATION,
            "physics_substep": apply_index % diagnostic.DECIMATION,
            "raw_action": 1.0,
            "requested_endpoint_rad": 0.7853981852531433,
            "target_after_setter_rad": 0.7853981852531433,
            "pre": phase,
            "command_after_setters": phase,
            "post": phase,
        }
        for apply_index in range(entry_count)
    ]
    outcome = diagnostic.classify_outcome(
        "cap8_abrupt_release",
        None,
        diagnostic.ACTION_COUNT,
        diagnostic.TAIL_POLICY_STEPS,
        len(entries),
    )
    cadence = diagnostic.validate_trace_cadence(
        entries,
        variant="cap8_abrupt_release",
        outcome=outcome,
        release_ramp_runtime=disabled_release_ramp_runtime(),
    )
    assert cadence["source_action_segment"]["trace_entry_count"] == 294 * 8
    assert cadence["frozen_tail_segment"] == {
        "policy_step_start": 294,
        "policy_steps_requested": 8,
        "policy_steps_completed": 8,
        "physics_substeps_requested": 64,
        "physics_substeps_completed": 64,
        "trace_entry_count": 64,
        "apply_index_start": 2352,
        "apply_index_stop_exclusive": 2416,
    }
    assert entries[-1]["policy_step"] == 301
    assert entries[-1]["physics_substep"] == 7


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
    for variant in diagnostic.VARIANTS:
        completed = diagnostic.classify_outcome(
            variant,
            None,
            294,
            8,
            294 * 8 + 64,
        )
        assert completed["diagnostic_completed"] is True
        assert completed["classification"] == "followup_completed_source_and_tail"
        assert completed["tail_physics_substeps_completed"] == 64
        failed = diagnostic.classify_outcome(
            variant,
            failure,
            293,
            0,
            293 * 8 + 2,
        )
        assert failed["diagnostic_completed"] is True
        assert failed["classification"] == "followup_numerical_failure_observed"
        assert failed["failure_segment"] == "source_actions"
        unparsed = diagnostic.classify_outcome(
            variant,
            {"message": "different numerical failure"},
            17,
            0,
            17 * 8,
        )
        assert unparsed["diagnostic_completed"] is False
        assert unparsed["classification"] == "followup_incomplete_or_unbound_failure"
        assert unparsed["numerical_failure_parse_error"]

    tail_failure = copy.deepcopy(failure)
    tail_failure["message"] = tail_failure["message"].replace(
        "policy_step=293, physics_substep=2",
        "policy_step=300, physics_substep=3",
    )
    tail = diagnostic.classify_outcome(
        "cap5_release_ramp16",
        tail_failure,
        294,
        6,
        300 * 8 + 3,
    )
    assert tail["diagnostic_completed"] is True
    assert tail["failure_segment"] == "frozen_final_command_tail"
    assert tail["tail_physics_substeps_completed"] == 51


def test_interventions_are_pre_registered_and_one_variable() -> None:
    assert diagnostic.VARIANTS == (
        "cap8_abrupt_release",
        "cap24_abrupt_release",
        "cap5_release_ramp16",
    )
    assert diagnostic.FOLLOWER_CAP_BY_VARIANT == {
        "cap8_abrupt_release": 8.0,
        "cap24_abrupt_release": 24.0,
        "cap5_release_ramp16": 5.0,
    }
    assert diagnostic.RELEASE_RAMP_SUBSTEPS == 16
    assert diagnostic.RELEASE_RAMP_FRACTIONS[0] == 0.0
    assert diagnostic.RELEASE_RAMP_FRACTIONS[-1] == 1.0
    assert diagnostic.TAIL_PHYSICS_SUBSTEPS == 64
    source = (SCRIPTS / "smoke_eef_pose_reasoning_fulltrace_ablation.py").read_text()
    assert source.count("set_dof_max_velocities(replacement, indices)") == 1
    assert "action[7] = 0.0" not in source
    assert "HOLD_CLOSE_ANCHOR_SUBSTEPS" not in source


def test_release_ramp_lifecycle_is_exactly_16_inclusive_linear_substeps() -> None:
    transition = diagnostic.advance_release_ramp(
        enabled=True,
        remaining_before_apply=1,
        remaining_after_apply=0,
        current_apply_was_interlock_active=True,
        pending_before_apply=False,
        next_index_before_apply=None,
    )
    assert transition == diagnostic.ReleaseRampTransition(
        applied_index=None,
        next_index=0,
        pending_after_apply=True,
        release_observed=True,
        ramp_started=False,
        cancelled_by_reactivation=False,
    )
    applied: list[int] = []
    pending = transition.pending_after_apply
    next_index = transition.next_index
    for _ in range(diagnostic.RELEASE_RAMP_SUBSTEPS):
        transition = diagnostic.advance_release_ramp(
            enabled=True,
            remaining_before_apply=0,
            remaining_after_apply=0,
            current_apply_was_interlock_active=False,
            pending_before_apply=pending,
            next_index_before_apply=next_index,
        )
        assert transition.applied_index is not None
        applied.append(transition.applied_index)
        pending = transition.pending_after_apply
        next_index = transition.next_index
    assert applied == list(range(16))
    assert pending is False
    assert next_index is None
    assert [diagnostic.release_ramp_fraction(index) for index in applied] == list(
        diagnostic.RELEASE_RAMP_FRACTIONS
    )
    assert diagnostic.release_ramp_fraction(0) == 0.0
    assert diagnostic.release_ramp_fraction(15) == 1.0


def test_release_ramp_starts_on_same_apply_for_open_cancellation() -> None:
    transition = diagnostic.advance_release_ramp(
        enabled=True,
        remaining_before_apply=12,
        remaining_after_apply=0,
        current_apply_was_interlock_active=False,
        pending_before_apply=False,
        next_index_before_apply=None,
    )
    assert transition.applied_index == 0
    assert transition.next_index == 1
    assert transition.release_observed is True
    assert transition.ramp_started is True


def test_release_ramp_wrapper_writes_only_seven_arm_position_targets() -> None:
    import numpy as np

    class Tensor:
        def __init__(self, values: Any) -> None:
            self.values = np.asarray(values)

        def __getitem__(self, key: Any) -> "Tensor":
            return Tensor(self.values[key])

        def __add__(self, other: Any) -> "Tensor":
            values = other.values if isinstance(other, Tensor) else other
            return Tensor(self.values + values)

        def __sub__(self, other: Any) -> "Tensor":
            values = other.values if isinstance(other, Tensor) else other
            return Tensor(self.values - values)

        def __mul__(self, other: Any) -> "Tensor":
            values = other.values if isinstance(other, Tensor) else other
            return Tensor(self.values * values)

        def __neg__(self) -> "Tensor":
            return Tensor(-self.values)

        def __gt__(self, other: Any) -> "Tensor":
            values = other.values if isinstance(other, Tensor) else other
            return Tensor(self.values > values)

        def abs(self) -> "Tensor":
            return Tensor(np.abs(self.values))

        def all(self) -> "Tensor":
            return Tensor(np.asarray(self.values.all()))

        def any(self) -> "Tensor":
            return Tensor(np.asarray(self.values.any()))

        def amax(self) -> "Tensor":
            return Tensor(np.asarray(self.values.max()))

        def item(self) -> Any:
            return self.values.item()

        def detach(self) -> "Tensor":
            return self

        def cpu(self) -> "Tensor":
            return self

        def clone(self) -> "Tensor":
            return Tensor(self.values.copy())

        def flatten(self) -> "Tensor":
            return Tensor(self.values.flatten())

        def tolist(self) -> Any:
            return self.values.tolist()

    class FakeTorch:
        @staticmethod
        def minimum(left: Tensor, right: Tensor) -> Tensor:
            return Tensor(np.minimum(left.values, right.values))

        @staticmethod
        def maximum(left: Tensor, right: Tensor) -> Tensor:
            return Tensor(np.maximum(left.values, right.values))

        @staticmethod
        def isfinite(value: Tensor) -> Tensor:
            return Tensor(np.isfinite(value.values))

    class Data:
        joint_pos = Tensor([[0.0] * 7])
        joint_pos_target = Tensor([[0.0] * 7])

    class Asset:
        data = Data()

        def __init__(self) -> None:
            self.position_target_writes: list[tuple[list[int], list[list[float]]]] = []
            self.fail_next_position_target_write = False

        def set_joint_position_target(
            self, target: Tensor, joint_ids: list[int]
        ) -> None:
            if self.fail_next_position_target_write:
                self.fail_next_position_target_write = False
                raise RuntimeError("synthetic target setter failure")
            self.position_target_writes.append(
                (list(joint_ids), target.values.tolist())
            )
            self.data.joint_pos_target = Tensor(target.values.copy())

    class Finger:
        def finalize_physics_post_before_next_arm(self) -> None:
            return None

    class Manager:
        _terms = {"finger_joint": Finger()}

    class Env:
        action_manager = Manager()

    class Arm:
        def __init__(self, _cfg: Any, _env: Any) -> None:
            self._asset = Asset()
            self._joint_ids = list(range(7))
            self._nominal_max_delta_joint_pos = Tensor([0.1] * 7)
            self._gripper_close_arm_interlock_remaining = 1
            self._gripper_close_arm_interlock_active_apply_count = 0
            self._failure_substep_trace_enabled = True
            self._failure_substep_trace_pending_slot = 0
            self._failure_substep_trace_pending_apply_index = None
            self._apply_call_count = 0
            self.failure_trace_targets: list[list[list[float]]] = []

        def apply_actions(self) -> None:
            self._apply_call_count += 1
            self._failure_substep_trace_pending_apply_index = self._apply_call_count - 1
            self._asset.data.joint_pos_target = Tensor([[0.1] * 7])
            # Model an open cancellation: this same apply is already released.
            self._gripper_close_arm_interlock_remaining = 0

        def _copy_failure_substep_trace_value(
            self, *, field: str, slot: int, value: Tensor
        ) -> None:
            assert field == "new_joint_pos_target_rad"
            assert slot == 0
            self.failure_trace_targets.append(value.values.tolist())

        def reset(self, _env_ids: Any = None) -> None:
            return None

    wrapped = diagnostic.make_full_trace_arm_class(
        Arm,
        enable_release_ramp=True,
        torch_module=FakeTorch,
    )(object(), Env())
    wrapped.apply_actions()
    assert wrapped._asset.position_target_writes == [(list(range(7)), [[0.0] * 7])]
    wrapped.apply_actions()
    second = wrapped._asset.position_target_writes[-1]
    assert second[0] == list(range(7))
    assert second[1][0] == pytest.approx([0.1 / 15.0] * 7)
    report = wrapped.diagnostic_release_ramp_runtime_report()
    assert report["applied_indices"] == [0, 1]
    assert len(report["overlay_entries"]) == 2
    assert report["overlay_entries"][0]["failure_trace_target_rewrite_completed"]
    assert wrapped.failure_trace_targets[0] == [[0.0] * 7]
    assert wrapped.failure_trace_targets[1][0] == pytest.approx([0.1 / 15.0] * 7)
    assert report["gripper_target_or_state_write_count"] == 0

    wrapped._asset.fail_next_position_target_write = True
    before_failure = copy.deepcopy(report)
    with pytest.raises(RuntimeError, match="synthetic target setter failure"):
        wrapped.apply_actions()
    assert wrapped.diagnostic_release_ramp_runtime_report() == before_failure

    wrapped.reset()
    reset_report = wrapped.diagnostic_release_ramp_runtime_report()
    assert reset_report["ramp_target_apply_count"] == 0
    assert reset_report["applied_indices"] == []
    assert reset_report["overlay_entries"] == []


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


def test_release_ramp_trace_recomputes_actual_float32_target() -> None:
    current = [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0] + [0.0] * 6
    nominal = list(current[:7])
    nominal[0] = diagnostic.float32(0.01)
    zeros = [0.0] * 13
    phase = snapshot(current, zeros, zeros, remaining=0)
    phase["interlock"]["release_ramp"] = {
        "enabled": True,
        "pending": False,
        "next_index": 1,
        "target_apply_count": 1,
    }
    entry = {
        "apply_index": 0,
        "policy_step": 0,
        "physics_substep": 0,
        "raw_action": 1.0,
        "requested_endpoint_rad": 0.7853981852531433,
        "target_after_setter_rad": 0.0,
        "pre": copy.deepcopy(phase),
        "command_after_setters": copy.deepcopy(phase),
        "post": copy.deepcopy(phase),
    }
    maximum_change = abs(diagnostic.float32_subtract(nominal[0], current[0]))
    overlay = {
        "profile": "float32_arm_slew_cap_overlay_apply_v1",
        "apply_index": 0,
        "policy_step": 0,
        "physics_substep": 0,
        "arm_joint_ids": diagnostic.ARM_JOINT_IDS,
        "arm_joint_names": diagnostic.ARM_JOINT_NAMES,
        "ramp_index": 0,
        "fraction_float32": 0.0,
        "nominal_max_delta_joint_pos_rad": diagnostic.ARM_NOMINAL_MAX_DELTA_RAD,
        "current_joint_pos_rad": current[:7],
        "nominal_pre_overlay_target_rad": nominal,
        "final_target_after_setter_rad": current[:7],
        "formula_profile": (
            "endpoint_exact_else_float32_clamp_nominal_delta_by_scaled_slew_v1"
        ),
        "target_setter_call_count": 1,
        "failure_trace_target_rewrite_completed": True,
        "gripper_target_or_state_write_count": 0,
    }
    runtime = {
        "profile": "arm_post_interlock_linear_slew_cap_release_ramp16_runtime_v2",
        "enabled": True,
        "release_observed_count": 1,
        "ramp_started_count": 1,
        "ramp_completed_count": 0,
        "ramp_cancelled_by_reactivation_count": 0,
        "ramp_target_apply_count": 1,
        "ramp_limited_target_apply_count": 1,
        "applied_indices": [0],
        "pending_at_report": False,
        "next_index_at_report": 1,
        "max_abs_nominal_to_ramped_target_change_rad": maximum_change,
        "overlay_entries": [overlay],
        "gripper_target_or_state_write_count": 0,
    }
    outcome = {
        "classification": "followup_incomplete_or_unbound_failure",
        "source_actions_completed": 0,
        "tail_policy_steps_completed": 0,
        "tail_physics_substeps_completed": 0,
        "parsed_numerical_failure": None,
    }
    cadence = diagnostic.validate_trace_cadence(
        [entry],
        variant=diagnostic.RELEASE_RAMP_VARIANT,
        outcome=outcome,
        release_ramp_runtime=runtime,
    )
    assert cadence["release_ramp_trace_gate"]["overlay_entry_count"] == 1

    no_op = copy.deepcopy(runtime)
    no_op["overlay_entries"][0]["final_target_after_setter_rad"] = nominal
    with pytest.raises(
        diagnostic.FullTraceAblationError,
        match="recomputed final target",
    ):
        diagnostic.validate_release_ramp_trace(
            [entry], variant=diagnostic.RELEASE_RAMP_VARIANT, runtime=no_op
        )


def test_arm_trace_safety_rejects_final_tail_velocity_and_bad_actual_target() -> None:
    positions = [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0] + [0.0] * 6
    zeros = [0.0] * 13
    pre = snapshot(positions, zeros, zeros, remaining=0)
    command = copy.deepcopy(pre)
    post = copy.deepcopy(pre)
    post["all_joint_vel_rad_s"][6] = 2.7
    entry = {
        "apply_index": 0,
        "policy_step": 0,
        "physics_substep": 0,
        "raw_action": 1.0,
        "requested_endpoint_rad": 0.7853981852531433,
        "target_after_setter_rad": 0.0,
        "pre": pre,
        "command_after_setters": command,
        "post": post,
    }
    completed = {
        "classification": "followup_completed_source_and_tail",
        "parsed_numerical_failure": None,
    }
    with pytest.raises(diagnostic.FullTraceAblationError, match="arm velocity safety"):
        diagnostic.validate_arm_trace_safety([entry], outcome=completed)

    post["all_joint_vel_rad_s"][6] = 0.0
    bad_target = diagnostic.float32(positions[0] + 0.1)
    for phase in (pre, command, post):
        phase["all_joint_pos_target_rad"][0] = bad_target
    with pytest.raises(
        diagnostic.FullTraceAblationError, match="actual target slew safety"
    ):
        diagnostic.validate_arm_trace_safety([entry], outcome=completed)

    near_limit = list(positions)
    near_limit[0] = diagnostic.float32(diagnostic.ARM_SOFT_LIMITS_RAD[0][1] - 0.001)
    guard_phase = snapshot(near_limit, zeros, zeros, remaining=0)
    guard_entry = {
        **entry,
        "pre": copy.deepcopy(guard_phase),
        "command_after_setters": copy.deepcopy(guard_phase),
        "post": copy.deepcopy(guard_phase),
    }
    with pytest.raises(
        diagnostic.FullTraceAblationError, match="target guard-band recovery"
    ):
        diagnostic.validate_arm_trace_safety([guard_entry], outcome=completed)


def test_failed_apply_terminal_snapshot_allows_only_one_apply_count_delta() -> None:
    positions = [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0] + [0.0] * 6
    zeros = [0.0] * 13
    prior = snapshot(positions, zeros, zeros, remaining=0)
    entry = {"post": prior}
    terminal = copy.deepcopy(prior)
    terminal["interlock"]["arm_apply_call_count"] += 1
    assert (
        diagnostic.validate_failed_apply_terminal_snapshot(terminal, entries=[entry])
        == terminal
    )

    missing_delta = copy.deepcopy(terminal)
    missing_delta["interlock"]["arm_apply_call_count"] -= 1
    with pytest.raises(
        diagnostic.FullTraceAblationError, match="failed-apply call-count delta"
    ):
        diagnostic.validate_failed_apply_terminal_snapshot(
            missing_delta, entries=[entry]
        )

    changed_ramp = copy.deepcopy(terminal)
    changed_ramp["interlock"]["release_ramp"]["pending"] = True
    with pytest.raises(
        diagnostic.FullTraceAblationError,
        match="failed-apply interlock release_ramp identity",
    ):
        diagnostic.validate_failed_apply_terminal_snapshot(
            changed_ramp, entries=[entry]
        )


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
        cap = diagnostic.FOLLOWER_CAP_BY_VARIANT[variant]
        after[8:] = [cap] * 5
        ramp_enabled = variant == diagnostic.RELEASE_RAMP_VARIANT
        ramp_runtime = disabled_release_ramp_runtime()
        ramp_runtime["enabled"] = ramp_enabled
        intervention = {
            "profile": "post_production_installer_cap_release_followup_v1",
            "variant": variant,
            "production_contract_acceptance": False,
            "passive_follower_velocity_cap_rad_s": cap,
            "passive_follower_velocity_limit_setter_call_count": int(
                cap != diagnostic.PRODUCTION_FOLLOWER_LIMIT
            ),
            "velocity_limits_before_intervention": tensor(before),
            "velocity_limits_after_intervention": tensor(after),
            "configured_close_anchor_substeps": 86,
            "arm_release_mode": (
                "linear_slew_cap_ramp16" if ramp_enabled else "abrupt"
            ),
            "arm_release_ramp_static": validator.expected_release_ramp_static(
                enabled=ramp_enabled
            ),
            "arm_release_ramp_runtime": ramp_runtime,
        }
        assert (
            validator.validate_intervention(intervention, variant=variant)
            == intervention
        )


def test_validator_accepts_only_ordered_complete_release_ramp_indices() -> None:
    applied = list(range(diagnostic.RELEASE_RAMP_SUBSTEPS)) * 3
    runtime = {
        "profile": "arm_post_interlock_linear_slew_cap_release_ramp16_runtime_v2",
        "enabled": True,
        "release_observed_count": 3,
        "ramp_started_count": 3,
        "ramp_completed_count": 3,
        "ramp_cancelled_by_reactivation_count": 0,
        "ramp_target_apply_count": len(applied),
        "ramp_limited_target_apply_count": 45,
        "applied_indices": applied,
        "pending_at_report": False,
        "next_index_at_report": None,
        "max_abs_nominal_to_ramped_target_change_rad": 0.02,
        "overlay_entries": [{} for _ in applied],
        "gripper_target_or_state_write_count": 0,
    }
    assert validator.validate_release_ramp_runtime(runtime, enabled=True) == runtime
    invalid = copy.deepcopy(runtime)
    invalid["applied_indices"][17] = 7
    with pytest.raises(validator.ValidationError, match="applied-index sequence"):
        validator.validate_release_ramp_runtime(invalid, enabled=True)


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
        "message": (
            "PolaRiS EEF IK current joint velocity exceeds the live simulation limit; "
            "aborting before DLS and PhysX (joint='panda_joint7', "
            "velocity_rad_s=2.7, limit_rad_s=2.609999895095825, "
            "excess_rad_s=0.090000104904175, policy_step=0, "
            "physics_substep=0, evidence_sha256=" + "1" * 64 + ")"
        ),
        "traceback": "synthetic traceback",
    }
    variant = "cap8_abrupt_release"
    outcome = diagnostic.classify_outcome(variant, failure, 0, 0, 0)
    entries: list[dict[str, Any]] = []
    cadence = diagnostic.validate_trace_cadence(
        entries,
        variant=variant,
        outcome=outcome,
        release_ramp_runtime=disabled_release_ramp_runtime(),
    )
    summary = diagnostic.summarize_trace(entries)
    ramp_enabled = False
    intervention = {
        "profile": "post_production_installer_cap_release_followup_v1",
        "variant": variant,
        "production_contract_acceptance": False,
        "passive_follower_velocity_cap_rad_s": 8.0,
        "passive_follower_velocity_limit_setter_call_count": 1,
        "velocity_limits_before_intervention": tensor(before),
        "velocity_limits_after_intervention": tensor(before[:8] + [8.0] * 5),
        "configured_close_anchor_substeps": 86,
        "arm_release_mode": "abrupt",
        "arm_release_ramp_static": validator.expected_release_ramp_static(
            enabled=ramp_enabled
        ),
        "arm_release_ramp_runtime": disabled_release_ramp_runtime(),
    }
    _, source_actions = diagnostic.load_actions()
    payload = {
        "schema_version": 1,
        "profile": diagnostic.PROFILE,
        "passed": True,
        "diagnostic_only": True,
        "variant": variant,
        "repository": {
            "path": "/synthetic/repo",
            "commit": "a" * 40,
            "clean_tracked": True,
            "diagnostic_base_commit": diagnostic.DIAGNOSTIC_BASE_COMMIT,
            "diagnostic_base_relation": "exact_first_parent_v1",
            "production_base_commit": diagnostic.PRODUCTION_BASE_COMMIT,
            "production_base_relation": "exact_first_grandparent_v1",
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
        "production_eval": {
            "size_bytes": diagnostic.gate0.EXPECTED_PRODUCTION_EVAL_SIZE_BYTES,
            "sha256": diagnostic.gate0.EXPECTED_PRODUCTION_EVAL_SHA256,
            "reset_profile": diagnostic.gate0.PRODUCTION_RESET_PROFILE,
            "reset_call": diagnostic.gate0.PRODUCTION_RESET_CALL,
            "environment_seed": None,
            "initial_condition_index": 0,
            "step_render_profile": diagnostic.gate0.PRODUCTION_STEP_RENDER_PROFILE,
            "effective_step_expensive": True,
            "policy_config_source": {
                "sha256": diagnostic.gate0.EXPECTED_PRODUCTION_POLICY_CONFIG_SHA256
            },
            "lap_client_source": {
                "sha256": diagnostic.gate0.EXPECTED_PRODUCTION_LAP_CLIENT_SHA256
            },
        },
        "fixture": diagnostic.file_identity(diagnostic.FIXTURE_PATH),
        "source_trace_sha256": fixture_builder.TRACE_SHA256,
        "source_action_float32_sha256": diagnostic.ACTION_ENCODING[
            "uncompressed_sha256"
        ],
        "boundary_helper": {
            "size_bytes": diagnostic.gate0.EXPECTED_BOUNDARY_HELPER_SIZE_BYTES,
            "sha256": diagnostic.gate0.EXPECTED_BOUNDARY_HELPER_SHA256,
        },
        "assets": {
            "contract": diagnostic.gate0.EXPECTED_ASSET_CONTRACT,
            "robot_usd": {"sha256": diagnostic.gate0.EXPECTED_ROBOT_USD_SHA256},
            "scene": {
                "scene": {
                    "sha256": diagnostic.gate0.EXPECTED_ASSET_CONTRACT["scene_sha256"]
                },
                "initial_conditions": {
                    "sha256": diagnostic.gate0.EXPECTED_ASSET_CONTRACT[
                        "initial_conditions_sha256"
                    ]
                },
            },
        },
        "runtime_protocol": {
            "physics_hz": 120.0,
            "physics_dt": 1.0 / 120.0,
            "decimation": 8,
            "policy_hz": 15.0,
        },
        "runtime_frame": {
            "action_dim": 7,
            "command_type": "pose",
            "controlled_body": "panda_link8",
            "eef_frame": "panda_link8",
            "reference_frame": "panda_link0",
            "ik_method": "dls",
            "dls_damping": 0.01,
            "ik_safety_profile": "panda_velocity_physxlimit_solveriter1_v4",
            "use_relative_mode": False,
        },
        "production_gripper_contract_before_ablation": {
            "joint_names": diagnostic.JOINT_NAMES,
            "driver_joint_index": 7,
            "follower_joint_indices": diagnostic.FOLLOWER_INDICES,
            "driver_target_slew": {
                "profile": (
                    "eef_binary_driver_target_slew_rate1p25_from_live_limit5_"
                    "per_120hz_substep_candidate_v1"
                )
            },
            "mimic_compliance": {
                "profile": (
                    "robotiq_2f85_live_physx_mimic_frequency100_damping1p2_candidate_v1"
                )
            },
            "measured_velocity_is_hard_bounded_by_limit": False,
        },
        "intervention": intervention,
        "action_count": diagnostic.ACTION_COUNT,
        "actions_completed": 0,
        "tail_contract": diagnostic.frozen_tail_contract(source_actions),
        "tail_policy_steps_completed": 0,
        "tail_physics_substeps_completed": 0,
        "numerical_failure": failure,
        "controller_failure_evidence": {"synthetic": True},
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
        validator.diagnostic,
        "validate_controller_failure_evidence",
        lambda value, **_kwargs: value,
    )
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
        "variant": "cap8_abrupt_release",
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
    assert manifest["profile"].endswith("_v1")
    assert manifest["runtime_exit"] == {
        "profile": "zero_simulator_srun_after_terminal_simulation_app_close_v1",
        "simulator_srun_exit_code": 0,
        "validated_in_separate_post_kit_process": True,
    }
    with pytest.raises(validator.ValidationError, match="simulator srun exit code"):
        validator.validate_result(payload, simulator_srun_exit_code=1, **kwargs)


def test_post_kit_validator_rejects_old_close_schema_and_variant_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, video = minimal_post_kit_payload(tmp_path)
    monkeypatch.setattr(
        validator.diagnostic,
        "validate_controller_failure_evidence",
        lambda value, **_kwargs: value,
    )
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
        validator.validate_result(old, variant="cap8_abrupt_release", **kwargs)

    mismatch = copy.deepcopy(payload)
    mismatch["variant"] = "cap24_abrupt_release"
    with pytest.raises(
        validator.ValidationError,
        match="result profile/variant|intervention",
    ):
        validator.validate_result(mismatch, variant="cap24_abrupt_release", **kwargs)


def test_post_kit_validator_rejects_synthetic_numerical_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, video = minimal_post_kit_payload(tmp_path)
    monkeypatch.setattr(validator, "probe_video", lambda *_args, **_kwargs: {})
    with pytest.raises(
        validator.diagnostic.FullTraceAblationError,
        match="controller failure evidence schema",
    ):
        validator.validate_result(
            payload,
            variant="cap8_abrupt_release",
            expected_commit="a" * 40,
            expected_job_id=123,
            expected_launch_id="b" * 64,
            result_identity={},
            video_path=video,
            ffprobe="/synthetic/ffprobe",
            ffmpeg="/synthetic/ffmpeg",
            simulator_srun_exit_code=0,
        )


def test_post_kit_validator_recomputes_outcome_from_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, video = minimal_post_kit_payload(tmp_path)
    payload["outcome"]["failure_segment"] = "frozen_final_command_tail"
    monkeypatch.setattr(validator, "probe_video", lambda *_args, **_kwargs: {})
    with pytest.raises(validator.ValidationError, match="diagnostic outcome"):
        validator.validate_result(
            payload,
            variant="cap8_abrupt_release",
            expected_commit="a" * 40,
            expected_job_id=123,
            expected_launch_id="b" * 64,
            result_identity={},
            video_path=video,
            ffprobe="/synthetic/ffprobe",
            ffmpeg="/synthetic/ffmpeg",
            simulator_srun_exit_code=0,
        )


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
    assert (
        "cap8_abrupt_release|cap24_abrupt_release|cap5_release_ramp16" in wrapper_source
    )
    assert "baseline|force_open|follower_default_limit|hold_close_anchor" not in (
        wrapper_source
    )
    assert 'rev-parse HEAD^)" == "${diagnostic_base_commit}"' in wrapper_source
    assert 'rev-parse HEAD^^)" == "${base_commit}"' in wrapper_source
