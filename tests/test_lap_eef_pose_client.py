import copy
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
    interpolate_ar_endpoint,
    quaternion_wxyz_to_rot6d,
    resolve_action_frame,
    rotate_image_180,
    validate_action_chunk,
)
from polaris.policy.ego_lap_contract import (
    ego_lap_contract_digest,
    persist_ego_lap_contract,
    validate_ego_lap_server_metadata,
)


def _serving_metadata(*, policy_type="flow", frame_description="robot base frame"):
    response_horizon = 16 if policy_type == "flow" else 1
    response_semantics = (
        "cumulative_delta_targets" if policy_type == "flow" else "total_delta_endpoint"
    )
    contract = {
        "schema_version": 2,
        "checkpoint_profile": "original_lap_public_3b_v1",
        "checkpoint_path": "/checkpoints/LAP-3B",
        "checkpoint_manifest_validated": False,
        "policy_type": policy_type,
        "model": {
            "action_dim": 7,
            "action_horizon": 16,
            "state_dim": 10,
            "image_resolution": [224, 224],
            "model_image_keys": ["base_0_rgb", "left_wrist_0_rgb"],
            "prompt_format": "lap",
            "enable_langact_training": True,
        },
        "policy_input": {
            "primary_image_key": "base_0_rgb",
            "wrist_image_key": "left_wrist_0_rgb",
            "image_color_space": "RGB",
            "image_dtype": "uint8",
            "image_resolution": [224, 224],
            "model_image_order": ["external", "wrist"],
            "wrist_rotation_degrees": 180,
            "dataset_name": "droid",
            "request_state_type": "eef_pose",
            "is_bimanual": False,
            "state_encoding": "EEF_R6",
            "state_layout": "xyz+r6_first_two_columns+gripper_open",
            "gripper_open_value": 1.0,
            "gripper_closed_value": 0.0,
        },
        "policy_output": {
            "action_encoding": "EEF_POS",
            "action_dim": 7,
            "model_action_horizon": 16,
            "response_horizon": response_horizon,
            "response_semantics": response_semantics,
            "action_layout": "delta_xyz+delta_extrinsic_xyz_euler+gripper_open",
            "translation_frame": "robot_base",
            "translation_units": "meters",
            "rotation_representation": "extrinsic_xyz_euler_delta",
            "rotation_composition": "right_multiply_current_by_delta",
            "rotation_units": "radians",
            "gripper_open_value": 1.0,
            "gripper_closed_value": 0.0,
        },
        "normalization": {
            "source": "checkpoint_assets",
            "type": "bounds_q99",
            "scope": "global",
            "policy_category": "single_arm",
            "selected_stats_sha256": "a" * 64,
            "selected_stats_keys": ["actions", "state"],
            "formula_schema_version": 1,
            "formula_profile": "q99_train_matched_v1",
            "input_formula_id": "q99_input_eps1e-8_clip_zero0_v1",
            "output_formula_id": "q99_output_eps1e-8_zeroq01_extrapolate_v1",
            "formula_probe_sha256": "b" * 64,
        },
        "language_action": {
            "format": "verbose_eef_with_rotation",
            "frame_description": frame_description,
        },
        "execution": {
            "schema_version": 2,
            "live_pipeline_validated": True,
            "inference_data_config": {
                "wrist_image_dropout_prob": 0.0,
                "mask_zero_img_prob": 0.0,
                "state_dropout": 0.0,
            },
            "image_routing": {"model_image_order": ["external", "wrist"]},
            "droid_image_preprocessing": {
                "dataset_name": "droid",
                "registry_requires_wrist_rotation": True,
                "not_rotate_wrist_prob": 0.0,
                "resize_resolution": [224, 224],
                "rotation_applied": True,
                "wrist_rotation_degrees": 180,
            },
            "ar_training_roundtrip_probe": {
                "dataset_name": "droid",
                "matches": True,
            },
            "normalization_formula": {
                "schema_version": 1,
                "profile": "q99_train_matched_v1",
                "input_formula_id": "q99_input_eps1e-8_clip_zero0_v1",
                "output_formula_id": "q99_output_eps1e-8_zeroq01_extrapolate_v1",
                "sha256": "b" * 64,
                "training_policy_input_probe": {"matches": True},
                "training_policy_roundtrip_probe": {"matches": True},
                "output_extrapolation_probe": {
                    "extrapolates_beyond_q01_q99": True,
                    "zero_range_is_exact_q01": True,
                },
            },
        },
        "polaris": {
            "profile": "panda_link8_eef_pose_single_arm_v1",
            "compatible": True,
            "incompatibilities": [],
            "eef_frame": "panda_link8",
            "normalization_scope": "global",
            "normalization_category": None,
            "q99_formula_profile": "q99_train_matched_v1",
            "numeric_action_frame": "robot_base",
        },
    }
    contract["sha256"] = ego_lap_contract_digest(contract)
    return {"ego_lap_serving_contract": contract}


def _xyzw_to_wxyz(quaternion):
    return np.asarray(quaternion)[[3, 0, 1, 2]]


def _validate_metadata(metadata, **overrides):
    arguments = {
        "expected_checkpoint_profile": None,
        "expected_checkpoint_path": None,
        "expected_policy_type": None,
        "expected_normalization_scope": None,
        "expected_normalization_stats_sha256": None,
        "expected_normalization_profile": None,
        "expected_normalization_input_formula": None,
        "expected_normalization_output_formula": None,
        "expected_frame_description": None,
        "expected_action_frame": None,
        "expected_dataset_name": "droid",
        "expected_state_type": "eef_pose",
        "expected_open_loop_horizon": 8,
        "ar_interpolation_steps": 16,
    }
    arguments.update(overrides)
    return validate_ego_lap_server_metadata(metadata, **arguments)


class ServingMetadataContractTest(unittest.TestCase):
    def test_global_scope_does_not_select_category(self):
        metadata = _serving_metadata()

        contract = _validate_metadata(
            metadata,
            expected_normalization_scope="global",
        )

        self.assertEqual(contract.normalization_scope, "global")
        self.assertEqual(
            metadata["ego_lap_serving_contract"]["polaris"]["normalization_scope"],
            "global",
        )

    def test_category_scope_requires_single_arm(self):
        metadata = _serving_metadata()
        normalization = metadata["ego_lap_serving_contract"]["normalization"]
        normalization["scope"] = "category"
        normalization["policy_category"] = "bimanual"

        with self.assertRaisesRegex(ValueError, "policy_category"):
            _validate_metadata(
                metadata,
                expected_normalization_scope="category",
            )

        normalization["policy_category"] = "single_arm"
        metadata["ego_lap_serving_contract"]["polaris"]["normalization_scope"] = (
            "category"
        )
        metadata["ego_lap_serving_contract"]["polaris"]["normalization_category"] = (
            "single_arm"
        )
        metadata["ego_lap_serving_contract"]["sha256"] = ego_lap_contract_digest(
            metadata["ego_lap_serving_contract"]
        )
        contract = _validate_metadata(
            metadata,
            expected_normalization_scope="category",
        )
        self.assertEqual(contract.normalization_scope, "category")

    def test_modern_manifest_image_routing_is_accepted(self):
        metadata = _serving_metadata(frame_description="egocentric frame")
        document = metadata["ego_lap_serving_contract"]
        document["checkpoint_profile"] = "manifest_execution_v2"
        document["checkpoint_manifest_validated"] = True
        document["model"]["model_image_keys"] = [
            "camera_0_rgb",
            "camera_1_rgb",
            "camera_2_rgb",
        ]
        document["policy_input"]["model_image_order"] = [
            "wrist",
            "external",
            "blank",
        ]
        document["execution"]["image_routing"]["model_image_order"] = [
            "wrist",
            "external",
            "blank",
        ]
        document["sha256"] = ego_lap_contract_digest(document)

        contract = _validate_metadata(
            metadata,
            expected_checkpoint_profile="manifest_execution_v2",
            expected_frame_description="egocentric frame",
        )

        self.assertEqual(contract.frame_description, "egocentric frame")

    def test_contract_identity_rejects_tampering(self):
        metadata = _serving_metadata()
        document = metadata["ego_lap_serving_contract"]
        self.assertEqual(document["sha256"], ego_lap_contract_digest(document))

        document["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "sha256"):
            _validate_metadata(metadata)

        document["sha256"] = ego_lap_contract_digest(document).upper()
        with self.assertRaisesRegex(ValueError, "lowercase"):
            _validate_metadata(metadata)

    def test_contract_rejects_dropout_and_checkpoint_drift(self):
        metadata = _serving_metadata()
        drifted = copy.deepcopy(metadata)
        drifted["ego_lap_serving_contract"]["execution"]["inference_data_config"][
            "state_dropout"
        ] = 0.1
        with self.assertRaisesRegex(ValueError, "state_dropout"):
            _validate_metadata(drifted)

        with self.assertRaisesRegex(ValueError, "checkpoint_path"):
            _validate_metadata(
                metadata,
                expected_checkpoint_path="/checkpoints/different",
            )

        incompatible = copy.deepcopy(metadata)
        incompatible["ego_lap_serving_contract"]["polaris"].update(
            {
                "compatible": False,
                "incompatibilities": ["wrong EEF frame"],
                "eef_frame": "base_link",
            }
        )
        with self.assertRaisesRegex(ValueError, "polaris.compatible"):
            _validate_metadata(incompatible)

    def test_normalization_formula_profile_is_fail_closed(self):
        metadata = _serving_metadata()

        contract = _validate_metadata(
            metadata,
            expected_normalization_profile="q99_train_matched_v1",
            expected_normalization_input_formula="q99_input_eps1e-8_clip_zero0_v1",
            expected_normalization_output_formula="q99_output_eps1e-8_zeroq01_extrapolate_v1",
        )

        self.assertEqual(contract.normalization_profile, "q99_train_matched_v1")
        metadata["ego_lap_serving_contract"]["execution"]["normalization_formula"][
            "training_policy_roundtrip_probe"
        ]["matches"] = False
        with self.assertRaisesRegex(ValueError, "training_policy_roundtrip_probe"):
            _validate_metadata(metadata)

    def test_ar_contract_derives_one_to_sixteen_to_four_protocol(self):
        contract = _validate_metadata(
            _serving_metadata(policy_type="ar"),
            expected_policy_type="ar",
            expected_open_loop_horizon=4,
        )

        self.assertEqual(contract.response_horizon, 1)
        self.assertEqual(contract.interpolation_steps, 16)
        self.assertEqual(contract.execution_horizon, 4)

    def test_persisted_contract_refuses_resume_drift(self):
        document = _serving_metadata()["ego_lap_serving_contract"]
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "contract.json"
            persist_ego_lap_contract(document, path)
            persist_ego_lap_contract(copy.deepcopy(document), path)
            drifted = copy.deepcopy(document)
            drifted["checkpoint_path"] = "/checkpoints/different"

            with self.assertRaisesRegex(ValueError, "different"):
                persist_ego_lap_contract(drifted, path)


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
        np.testing.assert_array_equal(
            validate_action_chunk(
                {"actions": np.zeros(7)},
                expected_horizon=1,
            ),
            np.zeros((1, 7)),
        )

    def test_wrist_rotation_is_exact(self):
        image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        np.testing.assert_array_equal(rotate_image_180(image), image[::-1, ::-1])

    def test_ar_endpoint_interpolation_scales_motion_and_holds_gripper(self):
        endpoint = np.array([[0.16, -0.32, 0.48, 0.8, -1.6, 0.4, 1.0]])

        chunk = interpolate_ar_endpoint(endpoint, steps=16)

        self.assertEqual(chunk.shape, (16, 7))
        np.testing.assert_allclose(chunk[0, :6], endpoint[0, :6] / 16)
        np.testing.assert_allclose(chunk[-1, :6], endpoint[0, :6])
        np.testing.assert_array_equal(chunk[:, 6], np.ones(16))


class _FakePolicyServer:
    def __init__(self, response, metadata=None):
        self.response = response
        self.metadata = metadata or _serving_metadata()
        self.requests = []

    def get_server_metadata(self):
        return self.metadata

    def infer(self, request):
        self.requests.append(request)
        return self.response


class ClientContractTest(unittest.TestCase):
    def test_registered_client_builds_request_and_reuses_anchored_chunk(self):
        fake_server = _FakePolicyServer(
            {
                "actions": np.vstack(
                    [
                        np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                        np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                        np.zeros((14, 7)),
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
            open_loop_horizon=8,
            frame_description="robot base frame",
            action_frame="robot_base",
            checkpoint_profile="original_lap_public_3b_v1",
            checkpoint_path="/checkpoints/LAP-3B",
            policy_type="flow",
            normalization_scope="global",
            normalization_stats_sha256="a" * 64,
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
            args.contract_output = str(
                Path(temporary_directory) / "serving_contract.json"
            )
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
            persisted_contract = json.loads(Path(args.contract_output).read_text())

        self.assertEqual(
            print_mock.call_args_list[0],
            mock.call(LAP_IMAGE_PREPROCESSOR_MARKER, flush=True),
        )
        self.assertEqual(
            print_mock.call_args_list[1],
            mock.call(LAP_EEF_FRAME_MARKER, flush=True),
        )
        self.assertTrue(
            print_mock.call_args_list[2]
            .args[0]
            .startswith("POLARIS_LAP_SERVING_CONTRACT=")
        )
        image_io_marker = print_mock.call_args_list[3].args[0]
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
        self.assertEqual(len(print_mock.call_args_list), 4)
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
        self.assertEqual(trace_records[1]["policy_type"], "flow")
        self.assertEqual(len(trace_records[1]["server_delta_chunk"]), 16)
        self.assertEqual(
            persisted_contract,
            fake_server.metadata["ego_lap_serving_contract"],
        )

    def test_ar_client_interpolates_endpoint_and_executes_first_four(self):
        fake_server = _FakePolicyServer(
            {"actions": np.array([[0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])},
            metadata=_serving_metadata(policy_type="ar"),
        )
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        observation = {
            "splat": {"external_cam": image, "wrist_cam": image},
            "policy": {
                "eef_pos": np.array([[0.4, 0.0, 0.2]]),
                "eef_quat": np.array([[1.0, 0.0, 0.0, 0.0]]),
                "gripper_pos": np.array([[0.0]]),
            },
        }
        args = PolicyArgs(
            client="EgoLAPEefPose",
            open_loop_horizon=4,
            policy_type="ar",
            frame_description="robot base frame",
            action_frame="robot_base",
        )

        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            mock.patch(
                "polaris.policy.lap_eef_pose_client.websocket_client_policy.WebsocketClientPolicy",
                return_value=fake_server,
            ),
            mock.patch(
                "polaris.policy.lap_eef_pose_client.resize_lap_image",
                side_effect=lambda value, *_: np.asarray(value),
            ),
            mock.patch("builtins.print"),
        ):
            args.trace_path = str(Path(temporary_directory) / "trace.jsonl")
            args.contract_output = str(Path(temporary_directory) / "contract.json")
            client = EgoLAPEefPoseClient(args)
            client.reset()
            actions = [client.infer(observation, "move")[0] for _ in range(4)]
            records = [
                json.loads(line)
                for line in Path(args.trace_path).read_text().splitlines()
            ]

        self.assertEqual(len(fake_server.requests), 1)
        np.testing.assert_allclose(
            np.asarray(actions)[:, 0],
            np.array([0.41, 0.42, 0.43, 0.44]),
            atol=1e-7,
        )
        np.testing.assert_array_equal(np.asarray(actions)[:, 7], np.zeros(4))
        query = records[1]
        self.assertEqual(query["response_semantics"], "total_delta_endpoint")
        self.assertEqual(len(query["server_delta_chunk"]), 1)
        self.assertEqual(len(query["raw_delta_chunk"]), 16)
        self.assertEqual(query["execution_horizon"], 4)

    def test_client_rejects_non_droid_eef_frame(self):
        args = PolicyArgs(client="EgoLAPEefPose")
        args.eef_frame = "base_link"  # type: ignore[assignment]
        with self.assertRaisesRegex(ValueError, "panda_link8"):
            EgoLAPEefPoseClient(args)


if __name__ == "__main__":
    unittest.main()
