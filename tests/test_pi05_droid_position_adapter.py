import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from conftest import make_joint_velocity_runtime_report
from polaris.config import PolicyArgs
from polaris.pi05_droid_jointvelocity_contract import (
    expected_pi05_droid_jointvelocity_contract,
)
from polaris.pi05_droid_native_eval_contract import (
    PI05_DROID_NATIVE_CONFIGURED_EPISODE_LENGTH_SECONDS,
    PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS,
    make_environment_runtime_contract,
    should_render_expensive,
)
from polaris.pi05_droid_position_adapter import (
    OFFICIAL_DROID_COMMIT,
    OFFICIAL_DROID_CONTROL_SOURCES,
    PI05_DROID_ISAACLAB_SOFT_LIMITS_LE_F32_SHA256,
    PI05_DROID_ISAACLAB_SOFT_LIMITS_RAD,
    PI05_DROID_PHYSX_HARD_LIMITS_LE_F32_SHA256,
    PI05_DROID_PHYSX_HARD_LIMITS_RAD,
    PI05_DROID_TARGET_GUARD_LIMITS_LE_F32_SHA256,
    PI05_DROID_TARGET_GUARD_LIMITS_RAD,
    PositionTargetHoldRecorder,
    adapt_official_droid_action,
    derive_isaaclab_factor1_soft_limits,
    evaluate_position_target_guard,
    expected_position_limit_contract,
    validate_position_adapter_evidence,
    validate_position_limit_contract,
)
from polaris.native_gripper_runtime import (
    EXPECTED_DROID_JOINT_NAMES,
    NATIVE_GRIPPER_DYNAMIC_PROFILE,
)
from polaris.pi05_droid_position_contract import (
    NATIVE_GRIPPER_DRIVE_PROFILE,
    PANDA_ARM_EFFORT_LIMITS,
    PANDA_ARM_VELOCITY_LIMITS,
    PI05_DROID_CONTRACT_FILENAME,
    PI05_DROID_ISAACLAB_SOURCE_SHA256,
    PI05_DROID_POSITION_ADAPTER_PROFILE,
    PI05_DROID_POSITION_DRIVE_DAMPING,
    PI05_DROID_POSITION_DRIVE_STIFFNESS,
    PI05_DROID_POSITION_MODEL_EVAL_CONTRACT,
    expected_pi05_droid_position_contract,
    expected_pi05_droid_position_server_metadata,
    publish_immutable_position_serving_contract,
)
from polaris.policy.droid_delta_position_client import (
    DroidDeltaJointPositionClient,
    PositionTargetLimitError,
    validate_position_trace_record,
)
import polaris.pi05_droid_position_runtime as position_runtime
from polaris.pi05_droid_position_runtime import (
    make_position_failure_sidecar,
    validate_position_adapter_runtime_report,
)
from polaris.pi05_droid_position_smoke import validate_position_limit_guard_probe


ROOT = Path(__file__).parents[1]
VALID_PANDA_Q = np.asarray([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0], dtype=np.float32)


def _array_report(values, *, device="cuda:0"):
    array = np.asarray(values, dtype=np.float32)
    return {
        "shape": list(array.shape),
        "dtype": "torch.float32",
        "device": device,
        "values": array.tolist(),
    }


def _reference_position_runtime_report():
    inherited = make_joint_velocity_runtime_report()
    hard_limits = np.asarray([PI05_DROID_PHYSX_HARD_LIMITS_RAD], dtype=np.float32)
    soft_limits = np.asarray([PI05_DROID_ISAACLAB_SOFT_LIMITS_RAD], dtype=np.float32)
    stiffness = np.full((1, 7), PI05_DROID_POSITION_DRIVE_STIFFNESS, np.float32)
    damping = np.full((1, 7), PI05_DROID_POSITION_DRIVE_DAMPING, np.float32)
    effort = np.asarray([PANDA_ARM_EFFORT_LIMITS], dtype=np.float32)
    velocity = np.asarray(
        inherited["velocity_drive"]["buffered"]["velocity_limit"]["values"],
        dtype=np.float32,
    )
    live_arrays = {
        "joint_stiffness": stiffness.tolist(),
        "joint_damping": damping.tolist(),
        "joint_effort_limits": effort.tolist(),
        "joint_velocity_limits": velocity.tolist(),
        "soft_joint_position_limits": soft_limits.tolist(),
    }
    direct_arrays = {
        field: _array_report(values, device="cpu")
        for field, values in live_arrays.items()
        if field != "soft_joint_position_limits"
    }
    gripper_drive = inherited["gripper"]["drive"]
    report = {
        "schema_version": 1,
        "profile": PI05_DROID_POSITION_ADAPTER_PROFILE,
        "status": "pass",
        "isaaclab_version": "2.3.0",
        "isaaclab_source_sha256": dict(PI05_DROID_ISAACLAB_SOURCE_SHA256),
        "polaris_runtime_source_sha256": {
            "droid_cfg.py": "e" * 64,
            "pi05_droid_position_cfg.py": "a" * 64,
            "pi05_droid_position_robot_cfg.py": "b" * 64,
            "native_gripper_runtime.py": "c" * 64,
            "manager_based_rl_splat_environment.py": "d" * 64,
        },
        "policy_frequency_hz": 15,
        "physics_frequency_hz": 120,
        "decimation": 8,
        "gripper_observation": copy.deepcopy(inherited["gripper_observation"]),
        "policy_observation": {
            "group_class": (
                "polaris.environments.droid_cfg."
                "DroidJointVelocityObservationCfg.PolicyCfg"
            ),
            "term_order": ["arm_joint_pos", "arm_joint_vel", "gripper_pos"],
            "enable_corruption": False,
            "concatenate_terms": False,
            "terms": {
                "arm_joint_pos": {
                    "function": (
                        "polaris.environments.droid_cfg.ordered_arm_joint_pos"
                    ),
                    "noise": None,
                    "clip": None,
                },
                "arm_joint_vel": {
                    "function": (
                        "polaris.environments.droid_cfg.ordered_arm_joint_vel"
                    ),
                    "noise": None,
                    "clip": None,
                },
                "gripper_pos": {
                    "function": "polaris.environments.droid_cfg.gripper_pos",
                    "noise": None,
                    "clip": None,
                },
            },
        },
        "reset_event_order": ["reset_all", "cap_gripper_followers"],
        "action_term_order": ["arm", "finger_joint"],
        "joint_names": list(inherited["joint_names"]),
        "joint_indices": list(range(7)),
        "action_term_class": (
            "polaris.environments.pi05_droid_position_cfg."
            "AuditedDroidDeltaJointPositionAction"
        ),
        "action_cfg_class": (
            "polaris.environments.pi05_droid_position_cfg."
            "AuditedDroidDeltaJointPositionActionCfg"
        ),
        "action_cfg_base_class": (
            "isaaclab.envs.mdp.actions.actions_cfg.JointPositionActionCfg"
        ),
        "scale": 1.0,
        "offset": 0.0,
        "use_default_offset": False,
        "preserve_order": True,
        "action_manager_clip": None,
        "action_buffers": copy.deepcopy(inherited["action_buffers"]),
        "target_mode": "absolute_joint_position",
        "target_hold": {
            "apply_calls_per_policy_step": 8,
            "recorder_method": "consume_position_target_hold_report",
            "setter": "Articulation.set_joint_position_target",
        },
        "position_drive": {
            "semantic_role": (
                "existing_polaris_NVIDIA_DROID_simulator_analogue_of_"
                "hardware_cartesian_impedance_update_desired_joint_positions"
            ),
            "claims_exact_hardware_controller_gains": False,
            "shoulder_joint_names_expr": ["panda_joint[1-4]"],
            "forearm_joint_names_expr": ["panda_joint[5-7]"],
            "position_stiffness": [400.0, 400.0],
            "velocity_damping": [80.0, 80.0],
            "effort_limit_sim": [
                PANDA_ARM_EFFORT_LIMITS[0],
                PANDA_ARM_EFFORT_LIMITS[4],
            ],
            "velocity_limit_sim": [
                PANDA_ARM_VELOCITY_LIMITS[0],
                PANDA_ARM_VELOCITY_LIMITS[4],
            ],
        },
        "live_position_drive": live_arrays,
        "direct_position_drive": direct_arrays,
        "position_limit_readback": {
            "contract": expected_position_limit_contract(),
            "configured_soft_joint_pos_limit_factor": 1.0,
            "buffered_hard": _array_report(hard_limits),
            "buffered_isaaclab_soft": _array_report(soft_limits),
            "direct_physx_hard": _array_report(hard_limits, device="cpu"),
        },
        "gripper": {
            "action_term_class": (
                "polaris.environments.droid_cfg.BinaryJointPositionZeroToOneAction"
            ),
            "semantics": "absolute_closed_positive_gt_0p5",
            "open_target_rad": 0.0,
            "closed_target_rad": float(np.float32(np.pi / 4.0)),
            "drive_profile": NATIVE_GRIPPER_DRIVE_PROFILE,
            "joint_names_expr": ["finger_joint"],
            "actuator": copy.deepcopy(gripper_drive["actuator"]),
            "direct_physx": copy.deepcopy(gripper_drive["direct_physx"]),
            "action_buffers": {
                field: copy.deepcopy(inherited["gripper"][field])
                for field in (
                    "open_command",
                    "closed_command",
                    "raw_action",
                    "processed_action",
                )
            },
        },
        "all_six_gripper": copy.deepcopy(inherited["all_six_gripper"]),
    }
    report["runtime_sha256"] = position_runtime._runtime_sha256(report)
    return report


def test_official_droid_math_clips_binarizes_and_anchors_fresh_each_step():
    raw = np.asarray([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 0.5])
    first_q = np.asarray([0.0, -0.5, 0.1, -2.0, 0.2, 1.5, -0.1], dtype=np.float32)
    first = adapt_official_droid_action(raw, first_q)
    np.testing.assert_array_equal(
        first["clipped_action"], [-1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0, 0.0]
    )
    np.testing.assert_allclose(
        first["absolute_joint_position_target_rad"],
        first_q.astype(np.float64) + 0.2 * np.asarray([-1, -1, -0.5, 0, 0.5, 1, 1]),
        rtol=0,
        atol=0,
    )

    # The second measurement deliberately differs from both the first
    # measurement and the first target. The second target must use it directly.
    second_q = first_q + np.asarray([0.03, 0.01, -0.02, 0.04, 0.0, -0.03, 0.02])
    second = adapt_official_droid_action(np.ones(8), second_q)
    np.testing.assert_allclose(
        second["absolute_joint_position_target_rad"],
        second_q.astype(np.float64) + 0.2,
        rtol=0,
        atol=0,
    )
    assert (
        second["absolute_joint_position_target_rad"]
        != first["absolute_joint_position_target_rad"]
    )


def test_adapter_evidence_and_eight_substep_hold_fail_closed():
    evidence = adapt_official_droid_action(np.zeros(8), np.arange(7, dtype=np.float32))
    tampered = copy.deepcopy(evidence)
    tampered["absolute_joint_position_target_rad"][0] += 0.01
    with pytest.raises(ValueError, match="math mismatch"):
        validate_position_adapter_evidence(tampered)

    target = np.arange(7, dtype=np.float32).reshape(1, 7)
    recorder = PositionTargetHoldRecorder()
    recorder.begin_policy_step(target)
    for _ in range(8):
        recorder.record_physics_substep(target)
    report = recorder.finish_policy_step()
    assert report["apply_calls"] == 8
    assert report["unique_applied_target_count"] == 1

    recorder.begin_policy_step(target)
    for _ in range(7):
        recorder.record_physics_substep(target)
    with pytest.raises(ValueError, match="exactly eight"):
        recorder.finish_policy_step()


def test_position_limits_separate_raw_factor1_rounding_and_intersection_guard():
    hard = np.asarray([PI05_DROID_PHYSX_HARD_LIMITS_RAD], dtype=np.float32)
    expected_soft = np.asarray([PI05_DROID_ISAACLAB_SOFT_LIMITS_RAD], dtype=np.float32)
    derived_soft = derive_isaaclab_factor1_soft_limits(hard)
    np.testing.assert_array_equal(derived_soft, expected_soft)
    assert derived_soft[0, 3, 1] > hard[0, 3, 1]
    assert derived_soft[0, 5, 0] > hard[0, 5, 0]

    contract = validate_position_limit_contract(expected_position_limit_contract())
    assert (
        contract["hard_limits"]["little_endian_float32_sha256"]
        == PI05_DROID_PHYSX_HARD_LIMITS_LE_F32_SHA256
    )
    assert (
        contract["isaaclab_soft_limits"]["little_endian_float32_sha256"]
        == PI05_DROID_ISAACLAB_SOFT_LIMITS_LE_F32_SHA256
    )
    assert (
        contract["target_guard"]["little_endian_float32_sha256"]
        == PI05_DROID_TARGET_GUARD_LIMITS_LE_F32_SHA256
    )
    guard = np.asarray([PI05_DROID_TARGET_GUARD_LIMITS_RAD], dtype=np.float32)
    assert guard[0, 3, 1] == hard[0, 3, 1]
    assert guard[0, 5, 0] == derived_soft[0, 5, 0]
    assert contract["target_guard"]["guard_inset_rad"] == 0.0

    target = np.mean(guard, axis=-1, dtype=np.float32).astype(np.float64)
    target[0, 3] = np.nextafter(np.float64(guard[0, 3, 1]), np.inf)
    cast_target, recomputed_guard, violation = evaluate_position_target_guard(
        target, hard, derived_soft
    )
    assert cast_target[0, 3] == guard[0, 3, 1]
    assert not violation.any()
    np.testing.assert_array_equal(recomputed_guard, guard)

    target[0, 3] = np.nextafter(guard[0, 3, 1], np.float32(np.inf), dtype=np.float32)
    cast_target, _, violation = evaluate_position_target_guard(
        target, hard, derived_soft
    )
    assert cast_target[0, 3] > guard[0, 3, 1]
    assert np.flatnonzero(violation[0]).tolist() == [3]

    target = np.mean(guard, axis=-1, dtype=np.float32).astype(np.float64)
    target[0, 5] = guard[0, 5, 0]
    _, _, violation = evaluate_position_target_guard(target, hard, derived_soft)
    assert not violation.any()
    target[0, 5] = np.nextafter(guard[0, 5, 0], np.float32(-np.inf), dtype=np.float32)
    _, _, violation = evaluate_position_target_guard(target, hard, derived_soft)
    assert np.flatnonzero(violation[0]).tolist() == [5]


def test_position_smoke_guard_probe_is_one_ulp_outside_hard_but_inside_soft():
    contract = expected_position_limit_contract()
    hard = np.asarray(contract["hard_limits"]["values_rad"], dtype=np.float32)
    soft = np.asarray(contract["isaaclab_soft_limits"]["values_rad"], dtype=np.float32)
    guard = np.asarray(contract["target_guard"]["values_rad"], dtype=np.float32)
    joint_index = 3
    adversarial = np.nextafter(
        guard[0, joint_index, 1], np.float32(np.inf), dtype=np.float32
    )
    probe = {
        "position_limit_contract": contract,
        "joint_index": joint_index,
        "controlling_bound_source": "live_joint_pos_limits",
        "hard_upper_limit_rad": float(hard[0, joint_index, 1]),
        "soft_upper_limit_rad": float(soft[0, joint_index, 1]),
        "intersection_guard_upper_limit_rad": float(guard[0, joint_index, 1]),
        "adversarial_target_rad": float(adversarial),
        "adversarial_is_one_float32_step_above_guard": True,
        "adversarial_inside_soft_limit": True,
        "adversarial_outside_hard_limit": True,
        "articulation_target_before": VALID_PANDA_Q.tolist(),
        "articulation_target_after": VALID_PANDA_Q.tolist(),
        "exception_type": "PositionActionTargetLimitError",
        "exception_message": "intersection guard rejected target before setter",
        "setter_unchanged": True,
    }
    assert validate_position_limit_guard_probe(probe) == probe
    tampered = copy.deepcopy(probe)
    tampered["adversarial_target_rad"] = float(
        np.nextafter(adversarial, np.float32(np.inf), dtype=np.float32)
    )
    with pytest.raises(ValueError, match="did not fail before setter"):
        validate_position_limit_guard_probe(tampered)


def test_position_contract_preserves_model_inputs_but_removes_velocity_actuation():
    old = expected_pi05_droid_jointvelocity_contract()
    new = expected_pi05_droid_position_contract()
    for field in ("checkpoint", "normalization", "openpi"):
        assert new[field] == old[field]
    inherited_policy_input = copy.deepcopy(new["policy_input"])
    native_images = inherited_policy_input.pop("native_images")
    assert inherited_policy_input == old["policy_input"]
    assert native_images == [
        {"source": "external_cam", "shape": [720, 1280, 3], "dtype": "uint8"},
        {"source": "wrist_cam", "shape": [720, 1280, 3], "dtype": "uint8"},
    ]
    assert new["normalization"]["scope"] == "checkpoint_global_droid"
    assert new["policy_input"]["wrist_rotation_degrees"] == 0
    assert new["policy_output"]["response_shape"] == [15, 8]
    assert new["policy_output"]["execute_first"] == 8
    assert new["control"]["target_mode"] == "absolute_joint_position"
    assert new["control"]["command_anchor"].startswith("fresh_measured")
    assert (
        new["control"]["position_drive"]["claims_exact_hardware_controller_gains"]
        is False
    )
    assert new["control"]["position_limits"] == expected_position_limit_contract()
    assert new["official_droid"]["revision"] == OFFICIAL_DROID_COMMIT
    assert new["official_droid"]["control_sources"] == OFFICIAL_DROID_CONTROL_SOURCES

    model_json = json.dumps(PI05_DROID_POSITION_MODEL_EVAL_CONTRACT, sort_keys=True)
    assert "panda_joint1_through_7_velocity_radians_per_second" not in model_json
    assert "joint_position_action_interpretation" not in model_json
    assert "fresh_live_measured_panda_q" in model_json


def test_position_runtime_round_trip_closes_direct_drive_and_gripper_fields():
    report = _reference_position_runtime_report()
    validated = validate_position_adapter_runtime_report(report)
    assert validated == report

    bad_direct = copy.deepcopy(report)
    bad_direct["direct_position_drive"]["joint_stiffness"]["values"][0][0] = 399.0
    bad_direct["runtime_sha256"] = position_runtime._runtime_sha256(bad_direct)
    with pytest.raises(ValueError, match="direct joint_stiffness"):
        validate_position_adapter_runtime_report(bad_direct)

    bad_gripper = copy.deepcopy(report)
    bad_gripper["gripper"]["direct_physx"]["velocity_limit"]["values"][0][0] = 4.0
    bad_gripper["runtime_sha256"] = position_runtime._runtime_sha256(bad_gripper)
    with pytest.raises(ValueError, match="gripper direct_physx velocity_limit"):
        validate_position_adapter_runtime_report(bad_gripper)

    bad_observation = copy.deepcopy(report)
    bad_observation["policy_observation"]["terms"]["arm_joint_pos"]["clip"] = [
        -1.0,
        1.0,
    ]
    bad_observation["runtime_sha256"] = position_runtime._runtime_sha256(
        bad_observation
    )
    with pytest.raises(ValueError, match="policy_observation"):
        validate_position_adapter_runtime_report(bad_observation)

    bad_hard = copy.deepcopy(report)
    bad_hard["position_limit_readback"]["buffered_hard"]["values"][0][3][1] = (
        PI05_DROID_ISAACLAB_SOFT_LIMITS_RAD[3][1]
    )
    bad_hard["runtime_sha256"] = position_runtime._runtime_sha256(bad_hard)
    with pytest.raises(ValueError, match="buffered PhysX hard limits"):
        validate_position_adapter_runtime_report(bad_hard)

    bad_serialization = copy.deepcopy(report)
    bad_serialization["position_limit_readback"]["buffered_hard"]["values"][0][0][
        0
    ] += 1e-12
    bad_serialization["runtime_sha256"] = position_runtime._runtime_sha256(
        bad_serialization
    )
    with pytest.raises(ValueError, match="exact serialized float32"):
        validate_position_adapter_runtime_report(bad_serialization)


def test_position_serving_contract_is_immutable_and_exact(tmp_path):
    metadata = expected_pi05_droid_position_server_metadata()
    path = tmp_path / PI05_DROID_CONTRACT_FILENAME
    report = publish_immutable_position_serving_contract(path, metadata)
    assert report["mode"] == "0444"
    assert report["nlink"] == 1
    with pytest.raises(FileExistsError):
        publish_immutable_position_serving_contract(path, metadata)


class _FakeServer:
    def __init__(self, actions):
        self.actions = actions
        self.requests = []
        self.metadata = expected_pi05_droid_position_server_metadata()

    def get_server_metadata(self):
        return self.metadata

    def infer(self, request):
        self.requests.append(request)
        return {"actions": self.actions}


class _Sensor:
    def __init__(self):
        self.frame = np.asarray([1], dtype=np.int64)


class _Robot:
    def __init__(self):
        hard_limits = np.broadcast_to(
            np.asarray([-3.0, 3.0], dtype=np.float32), (1, 13, 2)
        ).copy()
        hard_limits[:, :7] = np.asarray(
            [PI05_DROID_PHYSX_HARD_LIMITS_RAD], dtype=np.float32
        )
        soft_limits = hard_limits.copy()
        soft_limits[:, :7] = np.asarray(
            [PI05_DROID_ISAACLAB_SOFT_LIMITS_RAD], dtype=np.float32
        )
        joint_pos = np.zeros((1, 13), dtype=np.float32)
        joint_pos[:, :7] = VALID_PANDA_Q
        self.data = SimpleNamespace(
            joint_pos=joint_pos,
            joint_vel=np.zeros((1, 13), dtype=np.float32),
            joint_pos_target=np.zeros((1, 13), dtype=np.float32),
            joint_pos_limits=hard_limits,
            soft_joint_pos_limits=soft_limits,
        )

    def find_joints(self, names, preserve_order=False):
        assert preserve_order is True
        if names == ["finger_joint"]:
            return [7], ["finger_joint"]
        return list(range(7)), list(names)


class _Arm:
    def __init__(self):
        self.processed_actions = np.zeros((1, 7), dtype=np.float32)
        self.hold = PositionTargetHoldRecorder()

    def set_target(self, target):
        target = np.asarray(target, dtype=np.float32).reshape(1, 7)
        self.processed_actions[:] = target
        self.hold.begin_policy_step(target)
        for _ in range(8):
            self.hold.record_physics_substep(target)

    def consume_position_target_hold_report(self):
        return self.hold.finish_policy_step()


class _Env:
    def __init__(self):
        self.robot = _Robot()
        self.arm = _Arm()
        self.finger = SimpleNamespace(
            processed_actions=np.zeros((1, 1), dtype=np.float32)
        )
        sensors = {"external_cam": _Sensor(), "wrist_cam": _Sensor()}
        self.scene = {"robot": self.robot, **sensors}
        self.scene = type("Scene", (dict,), {})(self.scene)
        self.scene.sensors = sensors
        self.action_manager = SimpleNamespace(
            _terms={"arm": self.arm, "finger_joint": self.finger}
        )
        self.max_episode_length = PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS
        self.episode_length_buf = np.asarray([0], dtype=np.int64)
        self._sim_step_counter = 0
        self.common_step_counter = 0

    @property
    def unwrapped(self):
        return self

    def apply(self, action, q_after):
        self.arm.set_target(action[:7])
        self.robot.data.joint_pos_target[:, :7] = np.asarray(
            action[:7], dtype=np.float32
        )
        finger = np.float32(np.pi / 4.0 if action[7] == 1.0 else 0.0)
        self.finger.processed_actions[:] = finger
        self.robot.data.joint_pos_target[:, 7] = finger
        self.robot.data.joint_pos[:, :7] = np.asarray(q_after, dtype=np.float32)
        self.episode_length_buf += 1
        self._sim_step_counter += 8
        self.common_step_counter += 1
        for sensor in self.scene.sensors.values():
            sensor.frame += 1


def _observation(q):
    return {
        "splat": {
            "external_cam": np.full((720, 1280, 3), 7, dtype=np.uint8),
            "wrist_cam": np.full((720, 1280, 3), 11, dtype=np.uint8),
        },
        "policy": {
            "arm_joint_pos": np.asarray([q], dtype=np.float32),
            "arm_joint_vel": np.zeros((1, 7), dtype=np.float32),
            "gripper_pos": np.zeros((1, 1), dtype=np.float32),
        },
    }


class _CudaLikeExactBool:
    """Exercise the CUDA tensor path without requiring CUDA in unit tests."""

    def __init__(self, value):
        self.value = value

    def __array__(self, *_args, **_kwargs):
        raise TypeError("cannot convert a CUDA tensor directly to NumPy")

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return [self.value]


def _args(tmp_path):
    path = tmp_path / PI05_DROID_CONTRACT_FILENAME
    publish_immutable_position_serving_contract(
        path, expected_pi05_droid_position_server_metadata()
    )
    return PolicyArgs(
        client="DroidDeltaJointPosition",
        open_loop_horizon=8,
        expected_action_horizon=15,
        expected_action_dim=8,
        state_type="joint_position",
        rotate_wrist_180=False,
        render_every_step=True,
        trace_path=str(tmp_path / "trace.jsonl"),
        policy_profile=PI05_DROID_POSITION_ADAPTER_PROFILE,
        serving_contract_path=str(path),
        openpi_dir=str(ROOT / "third_party/openpi"),
    )


def _bind(client, env):
    runtime = make_environment_runtime_contract(
        configured_episode_length_seconds=PI05_DROID_NATIVE_CONFIGURED_EPISODE_LENGTH_SECONDS,
        live_max_episode_length=PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS,
    )
    client.bind_evaluation_runtime(runtime)
    client.reset()
    client.begin_rollout(env)
    return runtime


def test_client_reanchors_second_open_loop_action_to_fresh_live_q(tmp_path):
    actions = np.zeros((15, 8), dtype=np.float64)
    actions[:2, 0] = 1.0
    server = _FakeServer(actions)
    env = _Env()
    with mock.patch(
        "polaris.policy.droid_delta_position_client.websocket_client_policy.WebsocketClientPolicy",
        return_value=server,
    ):
        client = DroidDeltaJointPositionClient(_args(tmp_path))
    _bind(client, env)

    first, _ = client.infer(_observation(VALID_PANDA_Q), "move")
    assert first[0] == pytest.approx(0.2)
    actual_after = VALID_PANDA_Q.copy()
    actual_after[0] = np.float32(0.04)
    env.apply(first, actual_after)
    client.record_execution(
        _observation(actual_after),
        env,
        terminated=_CudaLikeExactBool(False),
        truncated=_CudaLikeExactBool(False),
    )

    second, _ = client.infer(_observation(actual_after), "move")
    assert second[0] == pytest.approx(0.24)
    assert second[0] != pytest.approx(0.4)
    records = [
        json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()
    ]
    for record in records:
        validate_position_trace_record(record)
    actions_in_trace = [
        record
        for record in records
        if record["record_type"] == "openpi_droid_position_action"
    ]
    query = next(
        record
        for record in records
        if record["record_type"] == "openpi_droid_position_query"
    )
    assert query["images"]["native_external"]["shape"] == [720, 1280, 3]
    assert query["images"]["native_wrist"]["shape"] == [720, 1280, 3]
    assert query["images"]["external"]["shape"] == [224, 224, 3]
    assert query["images"]["wrist"]["shape"] == [224, 224, 3]
    assert len(query["images"]["native_external"]["sha256"]) == 64
    assert actions_in_trace[1]["live_pre_step_joint_position"][0] == pytest.approx(0.04)
    assert actions_in_trace[1]["adapter"]["absolute_joint_position_target_rad"][
        0
    ] == pytest.approx(0.24)
    tampered_action = copy.deepcopy(actions_in_trace[0])
    tampered_action["guarded_float32_joint_position_target"][0] += 1e-12
    with pytest.raises(ValueError, match="exact serialized float32"):
        validate_position_trace_record(tampered_action)


def test_client_rejects_camera_resolution_dtype_and_state_dtype_drift():
    wrong_shape = _observation(np.zeros(7, dtype=np.float32))
    wrong_shape["splat"]["external_cam"] = np.zeros((719, 1280, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"shape \[720,1280,3\]"):
        DroidDeltaJointPositionClient._extract_observation(wrong_shape)

    wrong_dtype = _observation(np.zeros(7, dtype=np.float32))
    wrong_dtype["splat"]["wrist_cam"] = wrong_dtype["splat"]["wrist_cam"].astype(
        np.float32
    )
    with pytest.raises(ValueError, match="native uint8 RGB"):
        DroidDeltaJointPositionClient._extract_observation(wrong_dtype)

    wrong_state_dtype = _observation(np.zeros(7, dtype=np.float32))
    wrong_state_dtype["policy"]["arm_joint_pos"] = wrong_state_dtype["policy"][
        "arm_joint_pos"
    ].astype(np.float64)
    with pytest.raises(ValueError, match="seven ordered Panda joints"):
        DroidDeltaJointPositionClient._extract_observation(wrong_state_dtype)


def test_client_rejects_target_beyond_live_limit_before_setter(tmp_path):
    actions = np.zeros((15, 8), dtype=np.float64)
    actions[0, 0] = 1.0
    server = _FakeServer(actions)
    env = _Env()
    q = VALID_PANDA_Q.copy()
    q[0] = np.float32(2.95)
    env.robot.data.joint_pos[:, :7] = q
    with mock.patch(
        "polaris.policy.droid_delta_position_client.websocket_client_policy.WebsocketClientPolicy",
        return_value=server,
    ):
        client = DroidDeltaJointPositionClient(_args(tmp_path))
    runtime = _bind(client, env)
    with pytest.raises(PositionTargetLimitError, match="before setter") as caught:
        client.infer(_observation(q), "move")
    assert env.robot.data.joint_pos_target[0, 0] == 0.0
    dynamic = {
        "schema_version": 3,
        "profile": NATIVE_GRIPPER_DYNAMIC_PROFILE,
        "joint_names": list(EXPECTED_DROID_JOINT_NAMES),
        "joint_indices": list(range(13)),
        "apply_calls": 0,
        "post_policy_step_samples": 0,
        "sample_count": 0,
        "max_abs_joint_velocity_rad_s": [0.0] * 13,
        "max_abs_joint_acceleration_rad_s2": [0.0] * 13,
        "terminal_velocity_failure": None,
        "samples": None,
    }
    terminal = client.record_target_limit_failure(caught.value, env, dynamic)
    trace = client.finalized_trace_artifact
    assert trace["summary"]["status"] == "numerical_failure"
    assert terminal["incident"]["setter_calls_for_rejected_target"] == 0
    identity = {
        "path": "/tmp/fake.mp4",
        "size": 1,
        "sha256": "0" * 64,
        "mode": "0444",
        "nlink": 1,
    }
    sidecar = make_position_failure_sidecar(
        episode_result={
            "episode": 0,
            "episode_length": 1,
            "success": False,
            "progress": 0.0,
            "numerical_failure": True,
            "numerical_failure_reason": terminal["reason"],
        },
        environment_runtime_contract=runtime,
        terminal_failure=terminal,
        dynamic_report=dynamic,
        trace_artifact=trace,
        video_artifact=identity,
        incident_artifact=caught.value.incident_artifact,
    )
    assert sidecar["profile"].endswith("failure_sidecar_v1")


def test_position_profile_forces_expensive_render_every_step():
    assert should_render_expensive(
        policy_client_name="DroidDeltaJointPosition",
        render_every_step=True,
        needs_next_policy_render=False,
    )
