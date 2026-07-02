import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from polaris.eef_runtime_contract import atomic_write_runtime_contract
from polaris.eef_runtime_contract import validate_eef_runtime_frame
from polaris.eef_runtime_contract import validate_ego_lap_runtime_protocol
from polaris.gripper_semantics import GRIPPER_THRESHOLD_PROFILE


def _wxyz(rotation: Rotation) -> np.ndarray:
    return rotation.as_quat()[[3, 0, 1, 2]]


def _runtime_fixture():
    link0_rotation = Rotation.from_euler("z", 20, degrees=True)
    relative_rotation = Rotation.from_euler("xyz", [10, -5, 30], degrees=True)
    link8_rotation = link0_rotation * relative_rotation
    link0_position = np.array([0.1, -0.2, 0.3])
    relative_position = np.array([0.4, 0.05, 0.2])
    link8_position = link0_position + link0_rotation.apply(relative_position)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_names=["panda_link0", "panda_link8"],
            body_pos_w=np.array([[link0_position, link8_position]]),
            body_quat_w=np.array([[_wxyz(link0_rotation), _wxyz(link8_rotation)]]),
        )
    )
    offset = SimpleNamespace(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0))
    controller = SimpleNamespace(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.01},
    )
    arm_term = SimpleNamespace(
        cfg=SimpleNamespace(
            body_name="panda_link8",
            body_offset=offset,
            controller=controller,
            scale=1.0,
        ),
        action_dim=7,
        _body_idx=1,
        _joint_names=[f"panda_joint{index}" for index in range(1, 8)],
    )
    finger_term = SimpleNamespace(gripper_threshold_profile=GRIPPER_THRESHOLD_PROFILE)
    runtime = SimpleNamespace(
        max_episode_length=450,
        step_dt=1.0 / 15.0,
        scene={"robot": robot},
        action_manager=SimpleNamespace(
            _terms={"arm": arm_term, "finger_joint": finger_term}
        ),
    )
    env = SimpleNamespace(unwrapped=runtime, max_episode_length=450)
    observation = {
        "policy": {
            "eef_pos": relative_position[None, :],
            "eef_quat": _wxyz(relative_rotation)[None, :],
        }
    }
    return env, observation


def test_runtime_protocol_requires_exact_450_steps_at_15hz():
    env, _ = _runtime_fixture()
    resolved = validate_ego_lap_runtime_protocol(env)
    assert resolved["episode_steps"] == 450
    assert resolved["policy_hz"] == 15.0

    env.max_episode_length = 449
    with pytest.raises(ValueError, match="450"):
        validate_ego_lap_runtime_protocol(env)
    env.max_episode_length = 450
    env.unwrapped.step_dt = 1.0 / 10.0
    with pytest.raises(ValueError, match="15 Hz"):
        validate_ego_lap_runtime_protocol(env)


def test_runtime_frame_matches_direct_link8_and_absolute_action_term():
    env, observation = _runtime_fixture()
    result = validate_eef_runtime_frame(env, observation)
    assert result["eef_frame"] == "panda_link8"
    assert result["position_error_m"] < 1e-12
    assert result["rotation_error_rad"] < 1e-12
    assert result["reference_frame"] == "panda_link0"
    assert result["controlled_body"] == "panda_link8"
    assert result["body_offset"] == "identity"
    assert result["command_type"] == "pose"
    assert result["use_relative_mode"] is False
    assert result["ik_method"] == "dls"
    assert result["dls_damping"] == 0.01
    assert result["arm_scale"] == 1.0
    assert result["arm_joint_names"] == [f"panda_joint{index}" for index in range(1, 8)]
    assert result["gripper_threshold_profile"] == GRIPPER_THRESHOLD_PROFILE
    assert result["action_dim"] == 7


def test_runtime_contract_is_atomic_and_has_exact_evidence_schema():
    env, observation = _runtime_fixture()
    protocol = validate_ego_lap_runtime_protocol(env)
    frame = validate_eef_runtime_frame(env, observation)

    with tempfile.TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / "nested" / "runtime.json"
        path.parent.mkdir(parents=True)
        path.write_text('{"stale": true}\n', encoding="utf-8")
        atomic_write_runtime_contract(path, protocol=protocol, frame=frame)
        payload = json.loads(path.read_text(encoding="utf-8"))

        assert payload == {
            "schema_version": 1,
            "protocol": {
                "episode_steps": 450,
                "policy_hz": 15.0,
                "step_dt": 1.0 / 15.0,
            },
            "frame": frame,
        }
        assert not list(path.parent.glob(".*.tmp"))


def test_runtime_frame_rejects_observation_and_controller_drift():
    env, observation = _runtime_fixture()
    observation["policy"]["eef_pos"] = observation["policy"]["eef_pos"].copy()
    observation["policy"]["eef_pos"][0, 0] += 0.01
    with pytest.raises(ValueError, match="direct panda_link0->panda_link8"):
        validate_eef_runtime_frame(env, observation)

    env, observation = _runtime_fixture()
    env.unwrapped.action_manager._terms["arm"].cfg.body_name = "base_link"
    with pytest.raises(ValueError, match="does not control physical panda_link8"):
        validate_eef_runtime_frame(env, observation)


def test_runtime_frame_rejects_nonidentity_offset_and_relative_mode():
    env, observation = _runtime_fixture()
    env.unwrapped.action_manager._terms["arm"].cfg.body_offset.pos = (0.0, 0.0, 0.01)
    with pytest.raises(ValueError, match="offset is not identity"):
        validate_eef_runtime_frame(env, observation)

    env, observation = _runtime_fixture()
    env.unwrapped.action_manager._terms["arm"].cfg.controller.use_relative_mode = True
    with pytest.raises(ValueError, match="not absolute pose"):
        validate_eef_runtime_frame(env, observation)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("ik_method", "pinv", "damped least-squares"),
        ("damping", 0.1, "DLS damping"),
        ("scale", 0.5, "action scale"),
        (
            "joint_names",
            list(reversed([f"panda_joint{i}" for i in range(1, 8)])),
            "joint order",
        ),
        ("gripper_profile", "closed_positive_gt_0p5", "gripper threshold semantics"),
    ],
)
def test_runtime_frame_rejects_controller_semantics_drift(field, value, match):
    env, observation = _runtime_fixture()
    arm = env.unwrapped.action_manager._terms["arm"]
    if field == "ik_method":
        arm.cfg.controller.ik_method = value
    elif field == "damping":
        arm.cfg.controller.ik_params["lambda_val"] = value
    elif field == "scale":
        arm.cfg.scale = value
    elif field == "joint_names":
        arm._joint_names = value
    else:
        env.unwrapped.action_manager._terms[
            "finger_joint"
        ].gripper_threshold_profile = value

    with pytest.raises(ValueError, match=match):
        validate_eef_runtime_frame(env, observation)
