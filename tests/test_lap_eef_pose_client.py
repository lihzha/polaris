import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

from polaris.config import LAP_EEF_FRAME, PolicyArgs
from polaris.policy.abstract_client import InferenceClient
from polaris.policy.lap_eef_pose_client import (
    EgoLAPEefPoseClient,
    LAP_EEF_FRAME_MARKER,
    LAP_IMAGE_PREPROCESSOR_MARKER,
    anchor_action_chunk,
    build_lap_state,
    egocentric_action_chunk_to_base,
    quaternion_wxyz_to_rot6d,
    resolve_action_frame,
    rotate_image_180,
    validate_action_chunk,
)


def _xyzw_to_wxyz(quaternion):
    return np.asarray(quaternion)[[3, 0, 1, 2]]


class PoseConversionTest(unittest.TestCase):
    def test_rot6d_uses_first_two_rotation_matrix_columns(self):
        quaternion = _xyzw_to_wxyz(Rotation.from_euler("z", 90, degrees=True).as_quat())
        actual = quaternion_wxyz_to_rot6d(quaternion)
        np.testing.assert_allclose(
            actual,
            np.array([0.0, 1.0, 0.0, -1.0, 0.0, 0.0]),
            atol=1e-7,
        )

    def test_state_is_xyz_rot6d_and_open_positive_gripper(self):
        state = build_lap_state(
            np.array([0.4, -0.2, 0.3]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.25]),
        )
        np.testing.assert_allclose(
            state,
            np.array([0.4, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]),
        )

    def test_state_gripper_matches_official_half_threshold(self):
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        open_state = build_lap_state(np.zeros(3), identity, np.array([0.49]))
        closed_state = build_lap_state(np.zeros(3), identity, np.array([0.5]))
        self.assertEqual(open_state[-1], 1.0)
        self.assertEqual(closed_state[-1], 0.0)

    def test_entire_chunk_uses_one_anchor_and_right_relative_rotations(self):
        anchor_position = np.array([0.5, -0.1, 0.2])
        anchor_rotation = Rotation.from_euler("z", 90, degrees=True)
        anchor_wxyz = _xyzw_to_wxyz(anchor_rotation.as_quat())
        deltas = np.array(
            [
                [0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 1.0],
                [0.0, 0.2, 0.0, 0.0, -0.3, 0.0, 0.8],
            ]
        )

        actions = anchor_action_chunk(deltas, anchor_position, anchor_wxyz)

        np.testing.assert_allclose(
            actions[:, :3], anchor_position[None, :] + deltas[:, :3]
        )
        expected_rotation = anchor_rotation * Rotation.from_euler("xyz", deltas[:, 3:6])
        actual_rotation = Rotation.from_quat(actions[:, [4, 5, 6, 3]])
        np.testing.assert_allclose(
            actual_rotation.as_matrix(), expected_rotation.as_matrix(), atol=1e-6
        )
        np.testing.assert_allclose(actions[:, 7], np.array([0.0, 0.2]), atol=1e-7)

    def test_egocentric_droid_chunk_is_inverted_before_anchoring(self):
        anchor_position = np.array([0.5, -0.1, 0.2])
        anchor_rotation = Rotation.from_euler("z", 90, degrees=True)
        anchor_wxyz = _xyzw_to_wxyz(anchor_rotation.as_quat())
        semantic_eef_actions = np.array([[0.1, 0.2, -0.3, 0.1, -0.2, 0.3, 0.75]])

        base_deltas = egocentric_action_chunk_to_base(
            semantic_eef_actions,
            anchor_wxyz,
            dataset_name="droid",
            rotation_applied=True,
        )
        expected_geometric_eef_position = np.array([0.1, -0.2, 0.3])
        np.testing.assert_allclose(
            base_deltas[0, :3],
            anchor_rotation.apply(expected_geometric_eef_position),
            atol=1e-7,
        )

        actions = anchor_action_chunk(
            semantic_eef_actions,
            anchor_position,
            anchor_wxyz,
            action_frame="egocentric",
            dataset_name="droid",
            rotation_applied=True,
        )
        np.testing.assert_allclose(
            actions[0, :3], anchor_position + base_deltas[0, :3], atol=1e-7
        )
        expected_target = anchor_rotation * Rotation.from_euler(
            "xyz", base_deltas[0, 3:6]
        )
        actual_target = Rotation.from_quat(actions[0, [4, 5, 6, 3]])
        np.testing.assert_allclose(
            actual_target.as_matrix(), expected_target.as_matrix(), atol=1e-6
        )

    def test_frame_description_resolution_is_strict(self):
        self.assertEqual(resolve_action_frame("robot_base"), "robot_base")
        self.assertEqual(resolve_action_frame("egocentric"), "egocentric")
        with self.assertRaisesRegex(ValueError, "Unsupported action_frame"):
            resolve_action_frame("world frame")

    def test_action_validation_is_strict(self):
        with self.assertRaisesRegex(ValueError, r"\(T, 7\)"):
            validate_action_chunk({"actions": np.zeros(7)})
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_action_chunk({"actions": np.full((2, 7), np.nan)})
        with self.assertRaisesRegex(KeyError, "actions"):
            validate_action_chunk({})

    def test_wrist_rotation_is_exact(self):
        image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        np.testing.assert_array_equal(rotate_image_180(image), image[::-1, ::-1])


class _FakePolicyServer:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def infer(self, request):
        self.requests.append(request)
        return self.response


class ClientContractTest(unittest.TestCase):
    def test_registered_client_builds_request_and_reuses_anchored_chunk(self):
        fake_server = _FakePolicyServer(
            {
                "actions": np.array(
                    [
                        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                )
            }
        )
        external = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        wrist = np.arange(12, 24, dtype=np.uint8).reshape(2, 2, 3)
        observation = {
            "splat": {"external_cam": external, "wrist_cam": wrist},
            "policy": {
                "eef_pos": np.array([[0.4, 0.0, 0.2]]),
                "eef_quat": np.array([[1.0, 0.0, 0.0, 0.0]]),
                "gripper_pos": np.array([[0.0]]),
            },
        }
        args = PolicyArgs(
            client="EgoLAPEefPose",
            open_loop_horizon=2,
            frame_description="robot base frame",
        )

        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            mock.patch(
                "polaris.policy.lap_eef_pose_client.websocket_client_policy.WebsocketClientPolicy",
                return_value=fake_server,
            ),
            mock.patch(
                "polaris.policy.lap_eef_pose_client.resize_lap_image",
                side_effect=lambda image, *_: np.asarray(image),
            ),
            mock.patch("builtins.print") as print_mock,
        ):
            args.trace_path = str(Path(temporary_directory) / "trace.jsonl")
            client = EgoLAPEefPoseClient(args)
            client.reset()
            first_action, first_viz = client.infer(
                observation, "pick up the cup", return_viz=True
            )
            self.assertTrue(client.rerender)
            client.args.render_every_step = False
            self.assertFalse(client.rerender)
            client.args.render_every_step = True
            second_action, second_viz = client.infer(
                observation, "pick up the cup", return_viz=True
            )
            trace_records = [
                json.loads(line)
                for line in Path(args.trace_path).read_text().splitlines()
            ]

        self.assertEqual(
            print_mock.call_args_list[0],
            mock.call(LAP_IMAGE_PREPROCESSOR_MARKER, flush=True),
        )
        self.assertEqual(
            print_mock.call_args_list[1],
            mock.call(LAP_EEF_FRAME_MARKER, flush=True),
        )
        image_io_marker = print_mock.call_args_list[2].args[0]
        self.assertTrue(image_io_marker.startswith("POLARIS_LAP_IMAGE_IO="))
        image_io = json.loads(image_io_marker.split("=", maxsplit=1)[1])
        self.assertEqual(
            image_io,
            {
                "external_input": {"shape": [2, 2, 3], "dtype": "uint8"},
                "wrist_input": {"shape": [2, 2, 3], "dtype": "uint8"},
                "external_output": {"shape": [2, 2, 3], "dtype": "uint8"},
                "wrist_output": {"shape": [2, 2, 3], "dtype": "uint8"},
            },
        )
        self.assertEqual(len(print_mock.call_args_list), 3)
        self.assertIs(
            InferenceClient.REGISTERED_CLIENTS["EgoLAPEefPose"], EgoLAPEefPoseClient
        )
        self.assertEqual(len(fake_server.requests), 1)
        request = fake_server.requests[0]
        np.testing.assert_array_equal(request["observation"]["base_0_rgb"], external)
        np.testing.assert_array_equal(
            request["observation"]["left_wrist_0_rgb"], wrist[::-1, ::-1]
        )
        self.assertEqual(request["observation"]["state"].shape, (10,))
        self.assertEqual(request["frame_description"], "robot base frame")
        self.assertEqual(request["eef_frame"], LAP_EEF_FRAME)
        np.testing.assert_allclose(first_action[:3], np.array([0.5, 0.0, 0.2]))
        np.testing.assert_allclose(second_action[:3], np.array([0.6, 0.0, 0.2]))
        self.assertEqual(first_action[7], 0.0)
        self.assertEqual(second_action[7], 1.0)
        self.assertEqual(first_viz.shape, (2, 4, 3))
        self.assertEqual(second_viz.shape, (2, 4, 3))
        self.assertTrue(client.rerender)
        self.assertEqual(
            [record["event"] for record in trace_records],
            ["reset", "query", "action", "action"],
        )
        self.assertEqual(trace_records[1]["eef_frame"], LAP_EEF_FRAME)

    def test_client_rejects_non_droid_eef_frame(self):
        args = PolicyArgs(client="EgoLAPEefPose")
        args.eef_frame = "base_link"  # type: ignore[assignment]
        with self.assertRaisesRegex(ValueError, "panda_link8"):
            EgoLAPEefPoseClient(args)


if __name__ == "__main__":
    unittest.main()
