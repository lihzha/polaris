from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from polaris.eef_runtime_contract import validate_eef_runtime_frame
from polaris.eef_runtime_contract import validate_ego_lap_runtime_protocol


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
    controller = SimpleNamespace(command_type="pose", use_relative_mode=False)
    arm_term = SimpleNamespace(
        cfg=SimpleNamespace(
            body_name="panda_link8",
            body_offset=offset,
            controller=controller,
        ),
        action_dim=7,
        _body_idx=1,
    )
    runtime = SimpleNamespace(
        max_episode_length=450,
        step_dt=1.0 / 15.0,
        scene={"robot": robot},
        action_manager=SimpleNamespace(_terms={"arm": arm_term}),
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
