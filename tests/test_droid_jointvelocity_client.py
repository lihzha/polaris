import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from polaris.config import PolicyArgs
from polaris.pi05_droid_jointvelocity_contract import (
    PI05_DROID_CONTRACT_FILENAME,
    PI05_DROID_CONTRACT_METADATA_KEY,
    PI05_DROID_JOINTVELOCITY_PROFILE,
    expected_pi05_droid_server_metadata,
    make_openpi_runtime_attestation,
    publish_immutable_serving_contract,
    reference_openpi_runtime_attestation,
)
from polaris.pi05_droid_native_eval_contract import (
    PI05_DROID_NATIVE_CONFIGURED_EPISODE_LENGTH_SECONDS,
    PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS,
    make_environment_runtime_contract,
)
from polaris.policy.droid_jointvelocity_client import (
    DroidJointVelocityClient,
    JointVelocityObservationNumericalError,
    process_native_jointvelocity_action,
)


ROOT = Path(__file__).parents[1]


class _FakePolicyServer:
    def __init__(self, actions, metadata=None):
        self.actions = actions
        self.metadata = (
            expected_pi05_droid_server_metadata() if metadata is None else metadata
        )
        self.requests = []

    def get_server_metadata(self):
        return self.metadata

    def infer(self, request):
        self.requests.append(request)
        return {"actions": self.actions}


class _FakeRobot:
    def __init__(self):
        self.data = SimpleNamespace(
            joint_vel_target=np.zeros((1, 7), dtype=np.float32),
            joint_pos_target=np.zeros((1, 8), dtype=np.float32),
        )

    def find_joints(self, names, preserve_order=False):
        assert preserve_order is True
        if names == ["finger_joint"]:
            return [7], list(names)
        return list(range(7)), list(names)


class _FakeSensor:
    def __init__(self):
        self.frame = np.asarray([1], dtype=np.int64)


class _FakeScene(dict):
    def __init__(self, robot):
        self.sensors = {
            "external_cam": _FakeSensor(),
            "wrist_cam": _FakeSensor(),
        }
        super().__init__({"robot": robot, **self.sensors})


class _FakeEnv:
    def __init__(self):
        self.arm_term = SimpleNamespace(
            processed_actions=np.zeros((1, 7), dtype=np.float32)
        )
        self.finger_term = SimpleNamespace(
            processed_actions=np.zeros((1, 1), dtype=np.float32)
        )
        self.robot = _FakeRobot()
        self.action_manager = SimpleNamespace(
            _terms={"arm": self.arm_term, "finger_joint": self.finger_term}
        )
        self.scene = _FakeScene(self.robot)
        self.max_episode_length = PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS
        self.episode_length_buf = np.asarray([0], dtype=np.int64)
        self._sim_step_counter = 0
        self.common_step_counter = 0

    @property
    def unwrapped(self):
        return self

    def advance(self, *, refresh_sensors=True):
        self.episode_length_buf += 1
        self._sim_step_counter += 8
        self.common_step_counter += 1
        if refresh_sensors:
            for sensor in self.scene.sensors.values():
                sensor.frame += 1

    def simulate_auto_reset(self):
        self.episode_length_buf[:] = 0
        self._sim_step_counter += 8
        self.common_step_counter += 1
        for sensor in self.scene.sensors.values():
            sensor.frame[:] = 1


def _args(tmp_path, trace_path=None, metadata=None):
    contract_path = tmp_path / PI05_DROID_CONTRACT_FILENAME
    if not contract_path.exists():
        publish_immutable_serving_contract(
            contract_path,
            expected_pi05_droid_server_metadata() if metadata is None else metadata,
        )
    return PolicyArgs(
        client="DroidJointVelocity",
        open_loop_horizon=8,
        expected_action_horizon=15,
        expected_action_dim=8,
        state_type="joint_position",
        rotate_wrist_180=False,
        render_every_step=True,
        trace_path=trace_path,
        policy_profile=PI05_DROID_JOINTVELOCITY_PROFILE,
        serving_contract_path=str(contract_path),
        openpi_dir=str(ROOT / "third_party/openpi"),
    )


def _set_live_targets(env, action):
    env.arm_term.processed_actions[:] = action[:7]
    env.robot.data.joint_vel_target[:] = action[:7]
    expected_finger = np.float32(np.pi / 4.0 if action[7] == 1.0 else 0.0)
    env.finger_term.processed_actions[:] = expected_finger
    env.robot.data.joint_pos_target[:, 7] = expected_finger


def _bind_and_begin(client, env):
    runtime = make_environment_runtime_contract(
        configured_episode_length_seconds=(
            PI05_DROID_NATIVE_CONFIGURED_EPISODE_LENGTH_SECONDS
        ),
        live_max_episode_length=PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS,
    )
    client.bind_evaluation_runtime(runtime)
    client.reset()
    client.begin_rollout(env)
    return runtime


def _observation(q=None, dq=None):
    return {
        "splat": {
            "external_cam": np.full((240, 320, 3), 7, dtype=np.uint8),
            "wrist_cam": np.full((180, 240, 3), 11, dtype=np.uint8),
        },
        "policy": {
            "arm_joint_pos": np.asarray([q or [0.0] * 7], dtype=np.float32),
            "arm_joint_vel": np.asarray([dq or [0.0] * 7], dtype=np.float32),
            "gripper_pos": np.asarray([[0.25]], dtype=np.float32),
        },
    }


def test_upstream_processing_binarizes_then_clips_every_dimension():
    raw = np.asarray([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 0.5], dtype=np.float32)
    binary, clipped = process_native_jointvelocity_action(raw)
    assert binary.dtype == np.float64
    np.testing.assert_array_equal(binary, [-2, -1, -0.5, 0, 0.5, 1, 2, 0])
    np.testing.assert_array_equal(clipped, [-1, -1, -0.5, 0, 0.5, 1, 1, 0])

    raw[-1] = np.nextafter(np.float32(0.5), np.float32(1.0))
    _, clipped = process_native_jointvelocity_action(raw)
    assert clipped[-1] == 1.0


def test_exact_request_execute8_trace_and_live_target_contract(tmp_path):
    actions = np.arange(15 * 8, dtype=np.float64).reshape(15, 8) / 10.0 - 2.0
    actions[:, -1] = np.linspace(0.0, 1.0, 15)
    server = _FakePolicyServer(actions)
    env = _FakeEnv()

    trace_path = tmp_path / "trace.jsonl"
    with mock.patch(
        "polaris.policy.droid_jointvelocity_client.websocket_client_policy.WebsocketClientPolicy",
        return_value=server,
    ):
        client = DroidJointVelocityClient(_args(tmp_path, str(trace_path)))
    _bind_and_begin(client, env)
    returned = []
    for action_index in range(8):
        action, visualization = client.infer(
            _observation(dq=[0.01 * action_index] * 7),
            "put all foods in the bowl",
            return_viz=True,
        )
        _set_live_targets(env, action)
        env.advance()
        client.record_execution(
            _observation(
                q=[0.001 * (action_index + 1)] * 7,
                dq=[0.02 * (action_index + 1)] * 7,
            ),
            env,
            terminated=[False],
            truncated=[False],
        )
        returned.append((action, visualization))

    records = [json.loads(line) for line in trace_path.read_text().splitlines()]

    assert len(server.requests) == 1
    request = server.requests[0]
    assert request["observation/exterior_image_1_left"].shape == (224, 224, 3)
    assert request["observation/wrist_image_left"].shape == (224, 224, 3)
    np.testing.assert_array_equal(
        request["observation/joint_position"], np.zeros(7, dtype=np.float32)
    )
    assert len(returned) == 8
    assert all(action.dtype == np.float64 for action, _ in returned)
    assert all(visualization.shape == (224, 448, 3) for _, visualization in returned)
    assert len(records) == 18
    assert records[0]["record_type"] == "openpi_joint_velocity_rollout_start"
    assert records[1]["record_type"] == "openpi_joint_velocity_query"
    assert records[1]["response_action_shape"] == [15, 8]
    assert records[1]["execution_horizon"] == 8
    contract = server.metadata[PI05_DROID_CONTRACT_METADATA_KEY]
    assert records[1]["serving_contract_sha256"] == contract["contract_sha256"]
    assert len(records[1]["serving_contract_artifact_sha256"]) == 64
    assert records[1]["serving_contract_artifact_size"] > 0
    assert len(records[1]["client_runtime_attestation_sha256"]) == 64
    assert records[1]["internal_max_episode_length"] == 451
    assert records[1]["images"]["model_order"] == [
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb_masked",
    ]
    for action_index in range(8):
        action_record = records[2 + action_index * 2]
        execution_record = records[3 + action_index * 2]
        assert action_record["record_type"] == "openpi_joint_velocity_action"
        assert execution_record["record_type"] == "openpi_joint_velocity_execution"
        assert (
            action_record["serving_contract_artifact_sha256"]
            == execution_record["serving_contract_artifact_sha256"]
            == records[1]["serving_contract_artifact_sha256"]
        )
        assert action_record["chunk_action_index"] == action_index
        np.testing.assert_array_equal(
            action_record["emitted_joint_velocity"], returned[action_index][0][:7]
        )
        np.testing.assert_array_equal(
            execution_record["articulation_joint_velocity_target"],
            np.asarray(returned[action_index][0][:7], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            action_record["measured_normalized_gripper_position_before"], [0.25]
        )
        np.testing.assert_array_equal(
            execution_record["measured_normalized_gripper_position_after"], [0.25]
        )
        assert execution_record["outer_step_index"] == action_index
        assert execution_record["terminated"] is False
        assert execution_record["truncated"] is False
        assert (
            execution_record["environment_after"]["episode_length"] == action_index + 1
        )


def test_contract_observation_and_execution_fail_closed(tmp_path):
    actions = np.zeros((15, 8), dtype=np.float64)
    bad_metadata = expected_pi05_droid_server_metadata()
    bad_metadata[PI05_DROID_CONTRACT_METADATA_KEY]["openpi"][
        "inference_compatibility_commit"
    ] = "0" * 40
    with mock.patch(
        "polaris.policy.droid_jointvelocity_client.websocket_client_policy.WebsocketClientPolicy",
        return_value=_FakePolicyServer(actions, bad_metadata),
    ):
        with pytest.raises(ValueError, match="SHA-256 is invalid"):
            DroidJointVelocityClient(_args(tmp_path))

    with mock.patch(
        "polaris.policy.droid_jointvelocity_client.websocket_client_policy.WebsocketClientPolicy",
        return_value=_FakePolicyServer(actions),
    ):
        client = DroidJointVelocityClient(_args(tmp_path))
    env = _FakeEnv()
    _bind_and_begin(client, env)
    observation = _observation()
    observation["policy"]["arm_joint_vel"][0, 3] = np.nan
    with pytest.raises(JointVelocityObservationNumericalError, match="non-finite"):
        client.infer(observation, "test")

    action, _ = client.infer(_observation(), "test")
    _set_live_targets(env, action)
    env.advance()
    env.robot.data.joint_vel_target[0, 0] = 0.25
    with pytest.raises(ValueError, match="target differs"):
        client.record_execution(
            _observation(), env, terminated=[False], truncated=[False]
        )


def test_response_and_proprioception_dtypes_are_exact(tmp_path):
    with mock.patch(
        "polaris.policy.droid_jointvelocity_client.websocket_client_policy.WebsocketClientPolicy",
        return_value=_FakePolicyServer(np.zeros((15, 8), dtype=np.float32)),
    ):
        client = DroidJointVelocityClient(_args(tmp_path))
    _bind_and_begin(client, _FakeEnv())
    with pytest.raises(ValueError, match="response must be float64"):
        client.infer(_observation(), "test")

    with mock.patch(
        "polaris.policy.droid_jointvelocity_client.websocket_client_policy.WebsocketClientPolicy",
        return_value=_FakePolicyServer(np.zeros((15, 8), dtype=np.float64)),
    ):
        client = DroidJointVelocityClient(_args(tmp_path))
    _bind_and_begin(client, _FakeEnv())
    observation = _observation()
    observation["policy"]["arm_joint_pos"] = observation["policy"][
        "arm_joint_pos"
    ].astype(np.float64)
    with pytest.raises(ValueError, match="joint position must be float32"):
        client.infer(observation, "test")


def test_live_gripper_target_and_persisted_handshake_fail_closed(tmp_path):
    actions = np.zeros((15, 8), dtype=np.float64)
    actions[0, 7] = 0.5
    with mock.patch(
        "polaris.policy.droid_jointvelocity_client.websocket_client_policy.WebsocketClientPolicy",
        return_value=_FakePolicyServer(actions),
    ):
        client = DroidJointVelocityClient(_args(tmp_path))
    env = _FakeEnv()
    _bind_and_begin(client, env)
    action, _ = client.infer(_observation(), "test exact threshold")
    assert action[7] == 0.0
    _set_live_targets(env, action)
    env.advance()
    env.finger_term.processed_actions[0, 0] = np.float32(np.pi / 4.0)
    with pytest.raises(ValueError, match="binary gripper target"):
        client.record_execution(
            _observation(), env, terminated=[False], truncated=[False]
        )

    alternate_records = [
        record
        for record in reference_openpi_runtime_attestation()["imported_modules"]
        if record["relative_path"]
        in {
            "src/openpi/models/model.py",
            "src/openpi/models/tokenizer.py",
            "src/openpi/models/pi0.py",
            "src/openpi/policies/policy.py",
            "src/openpi/policies/policy_config.py",
            "src/openpi/serving/websocket_policy_server.py",
            "src/openpi/transforms.py",
        }
    ]
    alternate_metadata = expected_pi05_droid_server_metadata(
        make_openpi_runtime_attestation(alternate_records)
    )
    with mock.patch(
        "polaris.policy.droid_jointvelocity_client.websocket_client_policy.WebsocketClientPolicy",
        return_value=_FakePolicyServer(actions, alternate_metadata),
    ):
        with pytest.raises(ValueError, match="differs from live handshake"):
            DroidJointVelocityClient(_args(tmp_path))


def test_step_450_auto_reset_is_rejected_before_reset_observation_is_recorded(tmp_path):
    actions = np.zeros((15, 8), dtype=np.float64)
    with mock.patch(
        "polaris.policy.droid_jointvelocity_client.websocket_client_policy.WebsocketClientPolicy",
        return_value=_FakePolicyServer(actions),
    ):
        client = DroidJointVelocityClient(_args(tmp_path))
    env = _FakeEnv()
    _bind_and_begin(client, env)
    client._execution_step_index = 449  # noqa: SLF001
    client._terminated_false_count = 449  # noqa: SLF001
    client._truncated_false_count = 449  # noqa: SLF001
    env.episode_length_buf[:] = 449
    env._sim_step_counter = 449 * 8  # noqa: SLF001
    env.common_step_counter = 449
    for sensor in env.scene.sensors.values():
        sensor.frame[:] = 450

    action, _ = client.infer(_observation(), "terminal timeout probe")
    _set_live_targets(env, action)
    env.simulate_auto_reset()
    with pytest.raises(ValueError, match="auto-reset boundary"):
        client.record_execution(
            _observation(q=[9.0] * 7),
            env,
            terminated=[False],
            truncated=[True],
        )


def test_sensor_frame_counter_not_image_hash_variation_is_liveness_gate(tmp_path):
    actions = np.zeros((15, 8), dtype=np.float64)
    with mock.patch(
        "polaris.policy.droid_jointvelocity_client.websocket_client_policy.WebsocketClientPolicy",
        return_value=_FakePolicyServer(actions),
    ):
        client = DroidJointVelocityClient(_args(tmp_path))
    env = _FakeEnv()
    _bind_and_begin(client, env)
    action, _ = client.infer(_observation(), "stale sensor probe")
    _set_live_targets(env, action)
    env.advance(refresh_sensors=False)
    with pytest.raises(ValueError, match="camera frame counter"):
        client.record_execution(
            _observation(), env, terminated=[False], truncated=[False]
        )
