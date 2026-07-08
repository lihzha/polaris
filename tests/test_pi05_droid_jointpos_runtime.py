import copy
import json
from pathlib import Path

import numpy as np
import pytest

import polaris.pi05_droid_jointpos_runtime as runtime


ROOT = Path(__file__).resolve().parents[1]


def _runtime_report():
    report = {
        "schema_version": 1,
        "profile": runtime.PI05_DROID_JOINTPOS_PROFILE,
        "status": "pass",
        "boundary": {
            "profile": runtime.PI05_DROID_JOINTPOS_BOUNDARY_PROFILE,
            "outer_steps": 450,
            "internal_max_episode_steps": 451,
            "returned_terminal_flags": "all_false",
            "terminal_rubric_source": "post_action_450_pre_autoreset_info",
        },
        "timing": {
            "physics_dt_seconds": 1 / 120,
            "physics_frequency_hz": 120,
            "decimation": 8,
            "policy_frequency_hz": 15,
        },
        "joint_names": list(runtime.PANDA_ARM_JOINT_NAMES),
        "action": {
            "term_class": runtime._ACTION_TERM_CLASS,
            "cfg_class": runtime._ACTION_CFG_CLASS,
            "base_class": runtime._ACTION_BASE_CLASS,
            "cfg_base_class": runtime._ACTION_CFG_BASE_CLASS,
            "preserve_order": True,
            "scale": 1.0,
            "offset": 0.0,
            "use_default_offset": False,
            "clip": None,
            "semantic": "absolute_joint_position_observation_only_no_guard",
            "setter_calls_per_outer_step": 8,
        },
        "observation": {
            "term_order": [
                "arm_joint_pos",
                "gripper_pos",
                "eef_pos",
                "eef_quat",
            ],
            "enable_corruption": False,
            "concatenate_terms": False,
            "state_layout": {
                "arm_joint_indices": list(range(7)),
                "gripper_joint_index": 7,
                "historical_filter_order_equivalent": True,
            },
            "terms": {
                "arm_joint_pos": {
                    "function": (
                        "polaris.environments.pi05_droid_jointpos_cfg."
                        "ordered_arm_joint_position"
                    ),
                    "noise": None,
                    "clip": None,
                },
                "gripper_pos": {
                    "function": (
                        "polaris.environments.pi05_droid_jointpos_cfg."
                        "closed_positive_gripper_position"
                    ),
                    "noise": {
                        "class": ("isaaclab.utils.noise.noise_cfg.GaussianNoiseCfg"),
                        "mean": 0.0,
                        "std": 0.05,
                        "active": False,
                    },
                    "clip": [0.0, 1.0],
                },
                "eef_pos": {
                    "function": "polaris.environments.droid_cfg.eef_pos",
                    "noise": None,
                    "clip": None,
                },
                "eef_quat": {
                    "function": "polaris.environments.droid_cfg.eef_quat",
                    "noise": None,
                    "clip": None,
                },
            },
        },
        "configured_actuators": {
            "panda_shoulder": {
                "joint_names_expr": ["panda_joint[1-4]"],
                "stiffness": 400.0,
                "damping": 80.0,
                "effort_limit": 87.0,
                "velocity_limit": 2.175,
            },
            "panda_forearm": {
                "joint_names_expr": ["panda_joint[5-7]"],
                "stiffness": 400.0,
                "damping": 80.0,
                "effort_limit": 12.0,
                "velocity_limit": 2.61,
            },
        },
        "live_actuator_and_limits": {
            "joint_stiffness": runtime._EXPECTED_STIFFNESS.tolist(),
            "joint_damping": runtime._EXPECTED_DAMPING.tolist(),
            "joint_effort_limits": runtime._EXPECTED_EFFORT.tolist(),
            "joint_velocity_limits": runtime._EXPECTED_VELOCITY.tolist(),
            "hard_joint_position_limits": runtime._EXPECTED_HARD_LIMITS.tolist(),
            "soft_joint_position_limits": runtime._EXPECTED_SOFT_LIMITS.tolist(),
        },
        "direct_physx_actuator_and_limits": {
            "joint_stiffness": runtime._EXPECTED_STIFFNESS.tolist(),
            "joint_damping": runtime._EXPECTED_DAMPING.tolist(),
            "joint_effort_limits": runtime._EXPECTED_EFFORT.tolist(),
            "joint_velocity_limits": runtime._EXPECTED_VELOCITY.tolist(),
            "hard_joint_position_limits": runtime._EXPECTED_HARD_LIMITS.tolist(),
        },
        "cameras": {
            name: {"shape": [720, 1280, 3], "dtype": "uint8"}
            for name in runtime.PI05_DROID_JOINTPOS_SENSOR_NAMES
        },
        "gripper": {
            "action_class": runtime._GRIPPER_ACTION_CLASS,
            "joint_name": "finger_joint",
            "threshold": "closed_if_gt_0p5_else_open",
            "open_target_rad": 0.0,
            "closed_target_rad": float(np.float32(np.pi / 4)),
            "observation": (
                "finger_joint_position_divided_by_pi_over_4_closed_positive"
            ),
        },
    }
    report["runtime_sha256"] = runtime.canonical_sha256(report)
    return report


def _rehash(report):
    report = copy.deepcopy(report)
    report.pop("runtime_sha256", None)
    report["runtime_sha256"] = runtime.canonical_sha256(report)
    return report


def test_execution_recorder_observes_exact_upstream_eight_hold_path():
    recorder = runtime.JointPositionExecutionRecorder()
    target = np.arange(7, dtype=np.float32)[None]
    recorder.begin_policy_step(target, target.copy())
    for _ in range(8):
        recorder.record_apply_target(target.copy())
    report = recorder.finish_policy_step(target.copy())
    assert report["apply_target_hold_count"] == 8
    assert report["processing"].endswith("no_clip")
    assert report["post_step_articulation_target"] == target[0].tolist()


def test_execution_recorder_rejects_processing_or_hold_drift():
    recorder = runtime.JointPositionExecutionRecorder()
    target = np.zeros((1, 7), dtype=np.float32)
    recorder.begin_policy_step(target, target + np.float32(0.1))
    for _ in range(8):
        recorder.record_apply_target(target)
    with pytest.raises(ValueError, match="processing changed"):
        recorder.finish_policy_step(target)

    recorder = runtime.JointPositionExecutionRecorder()
    recorder.begin_policy_step(target, target)
    for _ in range(7):
        recorder.record_apply_target(target)
    with pytest.raises(ValueError, match="exactly eight"):
        recorder.finish_policy_step(target)


def test_runtime_validator_is_closed_over_every_live_contract_surface():
    report = _runtime_report()
    assert runtime.validate_jointpos_runtime_report(report) == report
    for mutate, message in (
        (
            lambda value: value["observation"]["terms"]["gripper_pos"].update(
                {"clip": None}
            ),
            "observation",
        ),
        (
            lambda value: value["configured_actuators"]["panda_shoulder"].update(
                {"stiffness": 399.0}
            ),
            "configured actuator",
        ),
        (
            lambda value: value["direct_physx_actuator_and_limits"].update(
                {"joint_stiffness": [[399.0] * 7]}
            ),
            "direct PhysX",
        ),
        (
            lambda value: value["gripper"].update({"threshold": "open_positive"}),
            "gripper",
        ),
    ):
        tampered = copy.deepcopy(report)
        mutate(tampered)
        tampered = _rehash(tampered)
        with pytest.raises(ValueError, match=message):
            runtime.validate_jointpos_runtime_report(tampered)


def test_runtime_artifact_publication_is_immutable_and_no_replace(tmp_path: Path):
    destination = tmp_path / "runtime.json"
    report = _runtime_report()
    artifact = runtime.publish_jointpos_runtime(destination, report)
    assert artifact["mode"] == "0444"
    assert destination.stat().st_nlink == 1
    assert (
        json.loads(destination.read_text())["runtime_sha256"]
        == report["runtime_sha256"]
    )
    with pytest.raises(FileExistsError):
        runtime.publish_jointpos_runtime(destination, report)


def test_timeout_configuration_keeps_internal_step_beyond_outer_horizon():
    cfg = type("Cfg", (), {"episode_length_s": 0.0})()
    seconds = runtime.configure_jointpos_timeout(cfg)
    assert seconds == 451 / 15
    assert cfg.episode_length_s == seconds


def test_jointpos_observation_cfg_preserves_historical_clip_noise_and_eef_terms():
    source = (ROOT / "src/polaris/environments/pi05_droid_jointpos_cfg.py").read_text(
        encoding="utf-8"
    )
    assert "noise=noise.GaussianNoiseCfg(std=0.05)" in source
    assert "clip=(0.0, 1.0)" in source
    assert "eef_pos = ObsTerm(func=eef_pos)" in source
    assert "eef_quat = ObsTerm(func=eef_quat)" in source
    assert "self.enable_corruption = False" in source
    assert "self.concatenate_terms = False" in source
