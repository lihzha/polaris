import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from polaris.config import PolicyArgs
from polaris.pi05_droid_jointvelocity_contract import (
    PI05_DROID_JOINTVELOCITY_PROFILE,
    expected_pi05_droid_server_metadata,
)
from polaris.policy.droid_jointvelocity_client import (
    DroidJointVelocityClient,
    JointVelocityObservationNumericalError,
    process_native_jointvelocity_action,
)


class _FakePolicyServer:
    def __init__(self, actions, metadata=None):
        self.actions = actions
        self.metadata = metadata or expected_pi05_droid_server_metadata()
        self.requests = []

    def get_server_metadata(self):
        return self.metadata

    def infer(self, request):
        self.requests.append(request)
        return {"actions": self.actions}


class _FakeRobot:
    def __init__(self):
        self.data = SimpleNamespace(joint_vel_target=np.zeros((1, 7), dtype=np.float32))

    def find_joints(self, names, preserve_order=False):
        assert preserve_order is True
        return list(range(7)), list(names)


class _FakeEnv:
    def __init__(self):
        self.arm_term = SimpleNamespace(
            processed_actions=np.zeros((1, 7), dtype=np.float32)
        )
        self.robot = _FakeRobot()
        self.action_manager = SimpleNamespace(_terms={"arm": self.arm_term})
        self.scene = {"robot": self.robot}

    @property
    def unwrapped(self):
        return self


def _args(trace_path=None):
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
    )


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


def test_exact_request_execute8_trace_and_live_target_contract():
    actions = np.arange(15 * 8, dtype=np.float64).reshape(15, 8) / 10.0 - 2.0
    actions[:, -1] = np.linspace(0.0, 1.0, 15)
    server = _FakePolicyServer(actions)
    env = _FakeEnv()

    with tempfile.TemporaryDirectory() as temporary_directory:
        trace_path = Path(temporary_directory) / "trace.jsonl"
        with mock.patch(
            "polaris.policy.droid_jointvelocity_client.websocket_client_policy.WebsocketClientPolicy",
            return_value=server,
        ):
            client = DroidJointVelocityClient(_args(str(trace_path)))
        client.reset()
        returned = []
        for action_index in range(8):
            action, visualization = client.infer(
                _observation(dq=[0.01 * action_index] * 7),
                "put all foods in the bowl",
                return_viz=True,
            )
            env.arm_term.processed_actions[:] = action[:7]
            env.robot.data.joint_vel_target[:] = action[:7]
            client.record_execution(
                _observation(
                    q=[0.001 * (action_index + 1)] * 7,
                    dq=[0.02 * (action_index + 1)] * 7,
                ),
                env,
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
    assert len(records) == 17
    assert records[0]["record_type"] == "openpi_joint_velocity_query"
    assert records[0]["response_action_shape"] == [15, 8]
    assert records[0]["execution_horizon"] == 8
    assert records[0]["images"]["model_order"] == [
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb_masked",
    ]
    for action_index in range(8):
        action_record = records[1 + action_index * 2]
        execution_record = records[2 + action_index * 2]
        assert action_record["record_type"] == "openpi_joint_velocity_action"
        assert execution_record["record_type"] == "openpi_joint_velocity_execution"
        assert action_record["chunk_action_index"] == action_index
        np.testing.assert_array_equal(
            action_record["emitted_joint_velocity"], returned[action_index][0][:7]
        )
        np.testing.assert_array_equal(
            execution_record["articulation_joint_velocity_target"],
            np.asarray(returned[action_index][0][:7], dtype=np.float32),
        )


def test_contract_observation_and_execution_fail_closed():
    actions = np.zeros((15, 8), dtype=np.float64)
    bad_metadata = expected_pi05_droid_server_metadata()
    bad_metadata["polaris_pi05_droid_contract"]["openpi"]["commit"] = "0" * 40
    with mock.patch(
        "polaris.policy.droid_jointvelocity_client.websocket_client_policy.WebsocketClientPolicy",
        return_value=_FakePolicyServer(actions, bad_metadata),
    ):
        with pytest.raises(ValueError, match="contract mismatch"):
            DroidJointVelocityClient(_args())

    with mock.patch(
        "polaris.policy.droid_jointvelocity_client.websocket_client_policy.WebsocketClientPolicy",
        return_value=_FakePolicyServer(actions),
    ):
        client = DroidJointVelocityClient(_args())
    client.reset()
    observation = _observation()
    observation["policy"]["arm_joint_vel"][0, 3] = np.nan
    with pytest.raises(JointVelocityObservationNumericalError, match="non-finite"):
        client.infer(observation, "test")

    action, _ = client.infer(_observation(), "test")
    env = _FakeEnv()
    env.arm_term.processed_actions[:] = action[:7]
    env.robot.data.joint_vel_target[:] = action[:7]
    env.robot.data.joint_vel_target[0, 0] = 0.25
    with pytest.raises(ValueError, match="target differs"):
        client.record_execution(_observation(), env)


def test_response_and_proprioception_dtypes_are_exact():
    with mock.patch(
        "polaris.policy.droid_jointvelocity_client.websocket_client_policy.WebsocketClientPolicy",
        return_value=_FakePolicyServer(np.zeros((15, 8), dtype=np.float32)),
    ):
        client = DroidJointVelocityClient(_args())
    client.reset()
    with pytest.raises(ValueError, match="response must be float64"):
        client.infer(_observation(), "test")

    with mock.patch(
        "polaris.policy.droid_jointvelocity_client.websocket_client_policy.WebsocketClientPolicy",
        return_value=_FakePolicyServer(np.zeros((15, 8), dtype=np.float64)),
    ):
        client = DroidJointVelocityClient(_args())
    client.reset()
    observation = _observation()
    observation["policy"]["arm_joint_pos"] = observation["policy"][
        "arm_joint_pos"
    ].astype(np.float64)
    with pytest.raises(ValueError, match="joint position must be float32"):
        client.infer(observation, "test")
