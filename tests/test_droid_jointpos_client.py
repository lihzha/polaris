import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from polaris.config import PolicyArgs
from polaris.evaluation_seed import make_live_environment_seed_contract
from polaris.policy.droid_jointpos_client import (
    DroidJointPosClient,
    JointPositionObservationNumericalError,
    PI05_DROID_CONTRACT_MARKER,
    validate_joint_action_chunk,
)
from polaris.pi05_droid_jointpos_serving_contract import (
    PI05_DROID_JOINTPOS_SERVER_MODEL_RESIZE,
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


def _live_seed_contract(seed=0):
    live_env = type(
        "LiveEnv",
        (),
        {
            "cfg": type(
                "Cfg",
                (),
                {
                    "seed": seed,
                    "sim": type(
                        "Sim",
                        (),
                        {
                            "physx": type(
                                "Physx",
                                (),
                                {"enable_enhanced_determinism": False},
                            )()
                        },
                    )(),
                },
            )()
        },
    )()
    return make_live_environment_seed_contract(live_env, seed)


def _policy_args(trace_path, **overrides):
    values = {
        "client": "DroidJointPos",
        "open_loop_horizon": 8,
        "trace_path": str(trace_path),
        "expected_action_horizon": 15,
        "expected_action_dim": 8,
        "state_type": "joint_position",
        "rotate_wrist_180": False,
    }
    values.update(overrides)
    return PolicyArgs(**values)


def _construct_client(args, server):
    with (
        mock.patch(
            "polaris.policy.droid_jointpos_client.websocket_client_policy."
            "WebsocketClientPolicy",
            return_value=server,
        ),
        mock.patch(
            "polaris.policy.droid_jointpos_client."
            "validate_pi05_droid_jointpos_server_metadata",
            return_value={"contract_sha256": "a" * 64},
        ),
        mock.patch(
            "polaris.policy.droid_jointpos_client."
            "pi05_droid_jointpos_server_contract_sha256",
            return_value="a" * 64,
        ),
    ):
        return DroidJointPosClient(args)


def _bind_test_runtime(client):
    client.runtime_contract = {"runtime_sha256": "b" * 64}
    client.runtime_contract_sha256 = "b" * 64


def _environment_boundary(step=0):
    return {
        "boundary_profile": "outer450_internal451_no_autoreset",
        "live_max_episode_length": 451,
        "episode_length": step,
        "sim_step_counter": step * 8,
        "common_step_counter": step,
        "sensor_frame_counters": {
            "external_cam": step,
            "wrist_cam": step,
        },
    }


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
                "external_cam": np.zeros((720, 1280, 3), dtype=np.uint8),
                "wrist_cam": np.zeros((720, 1280, 3), dtype=np.uint8),
            },
            "policy": {
                "arm_joint_pos": torch.tensor(
                    [[0.0, 0.0, 0.0, float("nan"), 0.0, 0.0, 0.0]]
                ),
                "gripper_pos": torch.tensor([[0.25]]),
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            args = _policy_args(Path(temporary_directory) / "trace.jsonl")
            client = _construct_client(
                args, _FakePolicyServer(np.zeros((15, 8), dtype=np.float64))
            )
            client._last_environment_after = _environment_boundary()
            with self.assertRaisesRegex(
                JointPositionObservationNumericalError, "non-finite"
            ):
                client.infer(observation, "test instruction", return_viz=True)

    def test_historical_gripper_observation_clip_is_required_live(self):
        observation = {
            "splat": {
                "external_cam": np.zeros((720, 1280, 3), dtype=np.uint8),
                "wrist_cam": np.zeros((720, 1280, 3), dtype=np.uint8),
            },
            "policy": {
                "arm_joint_pos": torch.zeros((1, 7), dtype=torch.float32),
                "gripper_pos": torch.tensor([[1.01]], dtype=torch.float32),
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            args = _policy_args(Path(temporary_directory) / "trace.jsonl")
            client = _construct_client(
                args, _FakePolicyServer(np.zeros((15, 8), dtype=np.float64))
            )
            client._last_environment_after = _environment_boundary()
            with self.assertRaisesRegex(ValueError, "clipped to"):
                client.infer(observation, "test instruction", return_viz=True)

    def test_existing_trace_advances_the_next_reset_index(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace_path = Path(temporary_directory) / "trace.jsonl"
            trace_path.write_text(json.dumps({"reset_index": 15}) + "\n")
            args = _policy_args(trace_path)
            client = _construct_client(
                args, _FakePolicyServer(np.zeros((15, 8), dtype=np.float64))
            )

            self.assertEqual(client.reset_index, 15)
            client.bind_environment_seed_contract(_live_seed_contract())
            _bind_test_runtime(client)
            client.reset(episode_index=16, episode_seed=16)
            self.assertEqual(client.reset_index, 16)

    def test_official_pi05_request_execution_and_trace_contract(self):
        actions = np.arange(15 * 8, dtype=np.float64).reshape(15, 8) / 100.0
        actions[:, -1] = np.linspace(0.0, 1.0, 15)
        fake_server = _FakePolicyServer(actions)
        row = np.arange(720, dtype=np.uint16)[:, None]
        column = np.arange(1280, dtype=np.uint16)[None, :]
        external = np.stack(
            [
                np.broadcast_to(row % 251, (720, 1280)),
                np.broadcast_to(column % 253, (720, 1280)),
                (row + column) % 255,
            ],
            axis=-1,
        ).astype(np.uint8)
        wrist = np.flip(external, axis=(0, 1)).copy()
        observation = {
            "splat": {"external_cam": external, "wrist_cam": wrist},
            "policy": {
                "arm_joint_pos": torch.arange(7, dtype=torch.float32)[None],
                "gripper_pos": torch.tensor([[0.25]], dtype=torch.float32),
            },
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            trace_path = Path(temporary_directory) / "trace.jsonl"
            args = _policy_args(trace_path)
            with (
                mock.patch(
                    "polaris.policy.droid_jointpos_client.websocket_client_policy."
                    "WebsocketClientPolicy",
                    return_value=fake_server,
                ),
                mock.patch(
                    "polaris.policy.droid_jointpos_client."
                    "validate_pi05_droid_jointpos_server_metadata",
                    return_value={"contract_sha256": "a" * 64},
                ),
                mock.patch(
                    "polaris.policy.droid_jointpos_client."
                    "pi05_droid_jointpos_server_contract_sha256",
                    return_value="a" * 64,
                ),
                mock.patch("builtins.print") as print_mock,
            ):
                client = DroidJointPosClient(args)
                client.bind_environment_seed_contract(_live_seed_contract())
                _bind_test_runtime(client)
                client.reset(episode_index=0, episode_seed=0)
                client._last_environment_after = _environment_boundary()
                returned = []
                for _ in range(8):
                    returned.append(
                        client.infer(
                            observation,
                            "put all foods in the bowl",
                            return_viz=True,
                        )
                    )
                    # This focused test covers query/emission serialization;
                    # live execution evidence has a separate fake-runtime test.
                    client._pending_execution = None

            records = [json.loads(line) for line in trace_path.read_text().splitlines()]

        self.assertEqual(len(fake_server.requests), 1)
        request = fake_server.requests[0]
        self.assertEqual(
            request["observation/exterior_image_1_left"].shape, (720, 1280, 3)
        )
        self.assertEqual(request["observation/wrist_image_left"].shape, (720, 1280, 3))
        self.assertEqual(request["observation/exterior_image_1_left"].dtype, np.uint8)
        self.assertEqual(request["observation/wrist_image_left"].dtype, np.uint8)
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
        self.assertEqual(trace["schema_version"], 4)
        self.assertEqual(trace["environment_rng"]["base_seed"], 0)
        self.assertEqual(trace["environment_rng"]["episode_index"], 0)
        self.assertEqual(trace["environment_rng"]["episode_seed"], 0)
        self.assertEqual(
            trace["environment_rng"]["determinism_claim"],
            "rng_bound_not_bitwise",
        )
        self.assertEqual(trace["response_action_shape"], [15, 8])
        self.assertEqual(trace["execution_horizon"], 8)
        self.assertEqual(len(trace["planned_action_chunk"]), 8)
        self.assertEqual(trace["images"]["request_external"]["shape"], [720, 1280, 3])
        self.assertEqual(trace["images"]["request_wrist"]["shape"], [720, 1280, 3])
        self.assertEqual(
            trace["images"]["request_external"]["sha256"],
            trace["images"]["native_external"]["sha256"],
        )
        self.assertEqual(
            trace["images"]["request_wrist"]["sha256"],
            trace["images"]["native_wrist"]["sha256"],
        )
        self.assertEqual(
            trace["images"]["visualization_external"]["shape"], [224, 224, 3]
        )
        self.assertEqual(trace["images"]["visualization_wrist"]["shape"], [224, 224, 3])
        self.assertIsNone(trace["images"]["client_model_spatial_transform"])
        self.assertEqual(
            trace["images"]["server_model_resize"],
            PI05_DROID_JOINTPOS_SERVER_MODEL_RESIZE,
        )
        self.assertIn("non_model", trace["images"]["visualization_spatial_transform"])
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

    def test_environment_seed_binding_and_episode_reset_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            args = _policy_args(Path(temporary_directory) / "trace.jsonl")
            client = _construct_client(
                args, _FakePolicyServer(np.zeros((15, 8), dtype=np.float64))
            )
            contract = _live_seed_contract(5)
            client.bind_environment_seed_contract(contract)
            _bind_test_runtime(client)
            with self.assertRaisesRegex(ValueError, "Episode seed"):
                client.reset(episode_index=0, episode_seed=6)
            client.reset(episode_index=0, episode_seed=5)
            with self.assertRaisesRegex(RuntimeError, "more than once"):
                client.bind_environment_seed_contract(contract)

    def test_live_execution_record_cross_binds_target_counters_and_post_state(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace_path = Path(temporary_directory) / "trace.jsonl"
            actions = np.zeros((15, 8), dtype=np.float64)
            actions[:, :7] = np.arange(7, dtype=np.float64) / 10
            client = _construct_client(
                _policy_args(trace_path), _FakePolicyServer(actions)
            )
            client.bind_environment_seed_contract(_live_seed_contract())
            _bind_test_runtime(client)
            client.reset(episode_index=0, episode_seed=0)
            before = _environment_boundary()
            client._rollout_environment_before = before
            client._last_environment_after = _environment_boundary(449)
            client.outer_step_index = 449
            observation = {
                "splat": {
                    "external_cam": np.zeros((720, 1280, 3), dtype=np.uint8),
                    "wrist_cam": np.zeros((720, 1280, 3), dtype=np.uint8),
                },
                "policy": {
                    "arm_joint_pos": torch.zeros((1, 7), dtype=torch.float32),
                    "gripper_pos": torch.zeros((1, 1), dtype=torch.float32),
                },
            }
            emitted, _ = client.infer(observation, "test execution", return_viz=True)
            target = np.asarray(emitted[:7], dtype=np.float32)

            class Arm:
                def consume_joint_position_execution_report(self):
                    values = target.tolist()
                    return {
                        "schema_version": 1,
                        "processing": (
                            "upstream_joint_position_action_scale1_offset0_no_clip"
                        ),
                        "raw_action_buffer": values,
                        "processed_action_buffer": values,
                        "apply_target_holds": [values] * 8,
                        "apply_target_hold_count": 8,
                        "post_step_articulation_target": values,
                    }

            class Robot:
                def __init__(self):
                    self.data = type(
                        "Data",
                        (),
                        {"joint_pos_target": torch.zeros((1, 8), dtype=torch.float32)},
                    )()

                def find_joints(self, names, preserve_order=True):
                    del preserve_order
                    if names == ["finger_joint"]:
                        return [7], ["finger_joint"]
                    return list(range(7)), [f"panda_joint{i}" for i in range(1, 8)]

            sensors = {
                name: type(
                    "Sensor",
                    (),
                    {"frame": torch.tensor([450], dtype=torch.int64)},
                )()
                for name in ("external_cam", "wrist_cam")
            }

            class Scene(dict):
                pass

            scene = Scene(robot=Robot())
            scene.sensors = sensors
            root = type(
                "Root",
                (),
                {
                    "max_episode_length": 451,
                    "_sim_step_counter": 3600,
                    "common_step_counter": 450,
                    "episode_length_buf": torch.tensor([450], dtype=torch.int64),
                    "scene": scene,
                    "action_manager": type(
                        "ActionManager",
                        (),
                        {
                            "_terms": {
                                "arm": Arm(),
                                "finger_joint": type(
                                    "Finger",
                                    (),
                                    {
                                        "processed_actions": torch.zeros(
                                            (1, 1), dtype=torch.float32
                                        )
                                    },
                                )(),
                            }
                        },
                    )(),
                },
            )()
            record = client.record_execution(
                observation,
                root,
                terminated=torch.tensor([False]),
                truncated=torch.tensor([False]),
                terminal_rubric={"success": True, "progress": 1.0, "metrics": {}},
            )
            self.assertEqual(record["environment_after"]["episode_length"], 450)
            self.assertEqual(record["action_execution"]["apply_target_hold_count"], 8)
            self.assertEqual(record["measured_joint_position_after"], [0.0] * 7)
            self.assertEqual(
                record["terminal_visualization"]["source"],
                "post_action450_returned_expensive_splat_observation",
            )
            terminal = client.final_terminal_visualization()
            self.assertEqual(terminal.shape, (224, 448, 3))
            self.assertEqual(terminal.dtype, np.uint8)
            records = [json.loads(line) for line in trace_path.read_text().splitlines()]
            self.assertEqual(
                [entry["record_type"] for entry in records],
                [
                    "openpi_joint_position_query",
                    "openpi_joint_position_action",
                    "openpi_joint_position_execution",
                ],
            )


if __name__ == "__main__":
    unittest.main()
