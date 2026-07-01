import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from polaris.config import PolicyArgs
from polaris.policy.droid_jointpos_client import (
    DroidJointPosClient,
    JointPositionObservationNumericalError,
    PI05_DROID_CONTRACT_MARKER,
    validate_joint_action_chunk,
)


class _FakePolicyServer:
    def __init__(self, actions):
        self.actions = actions
        self.requests = []

    def get_server_metadata(self):
        return {"test_server": True}

    def infer(self, request):
        self.requests.append(request)
        return {"actions": self.actions}


class JointActionValidationTest(unittest.TestCase):
    def test_expected_shape_and_finiteness_are_enforced(self):
        valid = np.zeros((15, 8), dtype=np.float32)
        actual = validate_joint_action_chunk(
            {"actions": valid},
            open_loop_horizon=8,
            expected_action_horizon=15,
            expected_action_dim=8,
        )
        self.assertIs(actual, valid)

        with self.assertRaisesRegex(ValueError, "horizon mismatch"):
            validate_joint_action_chunk(
                {"actions": np.zeros((14, 8))},
                open_loop_horizon=8,
                expected_action_horizon=15,
                expected_action_dim=8,
            )
        with self.assertRaisesRegex(ValueError, "width mismatch"):
            validate_joint_action_chunk(
                {"actions": np.zeros((15, 7))},
                open_loop_horizon=8,
                expected_action_horizon=15,
                expected_action_dim=8,
            )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_joint_action_chunk(
                {"actions": np.full((15, 8), np.nan)},
                open_loop_horizon=8,
            )


class DroidJointPosClientContractTest(unittest.TestCase):
    def test_nonfinite_joint_observation_is_a_numerical_failure(self):
        observation = {
            "splat": {
                "external_cam": np.zeros((224, 224, 3), dtype=np.uint8),
                "wrist_cam": np.zeros((224, 224, 3), dtype=np.uint8),
            },
            "policy": {
                "arm_joint_pos": torch.tensor(
                    [[0.0, 0.0, 0.0, float("nan"), 0.0, 0.0, 0.0]]
                ),
                "gripper_pos": torch.tensor([[0.25]]),
            },
        }
        args = PolicyArgs(
            client="DroidJointPos",
            open_loop_horizon=8,
            expected_action_horizon=15,
            expected_action_dim=8,
            state_type="joint_position",
        )
        with mock.patch(
            "polaris.policy.droid_jointpos_client.websocket_client_policy.WebsocketClientPolicy",
            return_value=_FakePolicyServer(np.zeros((15, 8))),
        ):
            client = DroidJointPosClient(args)

        with self.assertRaisesRegex(
            JointPositionObservationNumericalError, "non-finite"
        ):
            client.infer(observation, "test instruction", return_viz=True)

    def test_existing_trace_advances_the_next_reset_index(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace_path = Path(temporary_directory) / "trace.jsonl"
            trace_path.write_text(json.dumps({"reset_index": 15}) + "\n")
            args = PolicyArgs(
                client="DroidJointPos",
                open_loop_horizon=8,
                trace_path=str(trace_path),
                expected_action_horizon=15,
                expected_action_dim=8,
                state_type="joint_position",
            )
            with mock.patch(
                "polaris.policy.droid_jointpos_client.websocket_client_policy.WebsocketClientPolicy",
                return_value=_FakePolicyServer(np.zeros((15, 8))),
            ):
                client = DroidJointPosClient(args)

            self.assertEqual(client.reset_index, 15)
            client.reset()
            self.assertEqual(client.reset_index, 16)

    def test_official_pi05_request_execution_and_trace_contract(self):
        actions = np.arange(15 * 8, dtype=np.float32).reshape(15, 8) / 100.0
        actions[:, -1] = np.linspace(0.0, 1.0, 15)
        fake_server = _FakePolicyServer(actions)
        external = np.zeros((224, 224, 3), dtype=np.uint8)
        external[..., 0] = 7
        wrist = np.zeros((224, 224, 3), dtype=np.uint8)
        wrist[0, 0] = [1, 2, 3]
        observation = {
            "splat": {"external_cam": external, "wrist_cam": wrist},
            "policy": {
                "arm_joint_pos": torch.arange(7, dtype=torch.float32)[None],
                "gripper_pos": torch.tensor([[0.25]], dtype=torch.float32),
            },
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            trace_path = Path(temporary_directory) / "trace.jsonl"
            args = PolicyArgs(
                client="DroidJointPos",
                open_loop_horizon=8,
                trace_path=str(trace_path),
                expected_action_horizon=15,
                expected_action_dim=8,
                state_type="joint_position",
            )
            with (
                mock.patch(
                    "polaris.policy.droid_jointpos_client.websocket_client_policy.WebsocketClientPolicy",
                    return_value=fake_server,
                ),
                mock.patch("builtins.print") as print_mock,
            ):
                client = DroidJointPosClient(args)
                client.reset()
                returned = [
                    client.infer(
                        observation, "put all foods in the bowl", return_viz=True
                    )
                    for _ in range(8)
                ]

            records = [json.loads(line) for line in trace_path.read_text().splitlines()]

        self.assertEqual(len(fake_server.requests), 1)
        request = fake_server.requests[0]
        np.testing.assert_array_equal(
            request["observation/exterior_image_1_left"], external
        )
        np.testing.assert_array_equal(request["observation/wrist_image_left"], wrist)
        np.testing.assert_array_equal(
            request["observation/joint_position"], np.arange(7, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            request["observation/gripper_position"], np.array([0.25], dtype=np.float32)
        )
        self.assertEqual(request["prompt"], "put all foods in the bowl")
        for index, (action, visualization) in enumerate(returned):
            np.testing.assert_array_equal(action[:-1], actions[index, :-1])
            self.assertEqual(action[-1], float(actions[index, -1] > 0.5))
            self.assertEqual(action.dtype, np.float64)
            self.assertEqual(visualization.shape, (224, 448, 3))

        self.assertEqual(len(records), 9)
        trace = records[0]
        self.assertEqual(trace["response_action_shape"], [15, 8])
        self.assertEqual(trace["execution_horizon"], 8)
        self.assertEqual(len(trace["planned_action_chunk"]), 8)
        self.assertEqual(trace["images"]["external"]["shape"], [224, 224, 3])
        self.assertEqual(trace["images"]["wrist"]["shape"], [224, 224, 3])
        self.assertEqual(trace["images"]["wrist_rotation_degrees"], 0)
        self.assertTrue(
            print_mock.call_args_list[0].args[0].startswith(PI05_DROID_CONTRACT_MARKER)
        )
        for action_index, action_record in enumerate(records[1:]):
            self.assertEqual(
                action_record["record_type"], "openpi_joint_position_action"
            )
            self.assertEqual(action_record["chunk_action_index"], action_index)
            np.testing.assert_array_equal(
                action_record["raw_action"], actions[action_index]
            )
            np.testing.assert_array_equal(
                action_record["emitted_action"], returned[action_index][0]
            )


if __name__ == "__main__":
    unittest.main()
