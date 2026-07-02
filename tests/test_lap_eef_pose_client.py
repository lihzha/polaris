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
    FRANKA_DROID_SINGLE_ARM_PROFILE,
    MANIFEST_V1_PROFILE,
    MANIFEST_TRAIN_MATCHED_R6_MODE,
    ORIGINAL_LAP_PROFILE,
    PUBLIC_LAP_TRAIN_MATCHED_R6_MODE,
    R6_COLUMNS_STATE_LAYOUT,
    R6_ROWS_STATE_LAYOUT,
    ego_lap_contract_digest,
    ego_lap_nested_digest,
    persist_ego_lap_contract,
    validate_ego_lap_server_metadata,
)


def _serving_metadata(
    *,
    policy_type="flow",
    frame_description="robot base frame",
    checkpoint_profile=ORIGINAL_LAP_PROFILE,
    mixed_model=False,
):
    response_horizon = 16 if policy_type == "flow" else 1
    response_semantics = (
        "cumulative_delta_targets" if policy_type == "flow" else "total_delta_endpoint"
    )
    is_public_lap = checkpoint_profile == ORIGINAL_LAP_PROFILE
    model_image_keys = (
        ["base_0_rgb", "left_wrist_0_rgb"]
        if is_public_lap
        else ["camera_0_rgb", "camera_1_rgb", "camera_2_rgb"]
    )
    model_image_order = (
        ["external", "wrist"] if is_public_lap else ["wrist", "external", "blank"]
    )
    state_layout = R6_ROWS_STATE_LAYOUT if is_public_lap else R6_COLUMNS_STATE_LAYOUT
    state_layout_mode = (
        PUBLIC_LAP_TRAIN_MATCHED_R6_MODE
        if is_public_lap
        else MANIFEST_TRAIN_MATCHED_R6_MODE
    )
    native_action_dim = 14 if mixed_model else 7
    native_state_dim = 20 if mixed_model else 10
    normalized_state_dim = 14 if mixed_model else 10
    state_stats_dim = 20 if mixed_model else 10
    action_stats_dim = 14 if mixed_model else 7
    state_adapter = (
        {
            "type": "normalize_then_zero_pad",
            "request_state_dim": 10,
            "model_state_dim": 14,
            "checkpoint_stats_dim": 20,
        }
        if mixed_model
        else None
    )
    action_projection = (
        {
            "type": "leading_dimensions",
            "source_action_dim": 14,
            "output_action_dim": 7,
            "indices": list(range(7)),
            "applied_after": "checkpoint_unnormalization",
        }
        if mixed_model
        else None
    )
    contract = {
        "schema_version": 2,
        "checkpoint_profile": checkpoint_profile,
        "checkpoint_path": "/checkpoints/LAP-3B",
        "checkpoint_manifest_validated": not is_public_lap,
        "serving_profile": FRANKA_DROID_SINGLE_ARM_PROFILE,
        "policy_type": policy_type,
        "action_sampling": {
            "profile": (
                "flow_explicit_euler_t1_to_t0_v1"
                if policy_type == "flow"
                else "autoregressive_max500_temp0_eos_v1"
            ),
            "flow_num_steps": 10 if policy_type == "flow" else None,
            "initial_rng_seed": 0,
            "ar_max_decoding_steps": 500 if policy_type == "ar" else None,
            "ar_temperature": 0.0 if policy_type == "ar" else None,
            "ar_stop_at_eos": True if policy_type == "ar" else None,
        },
        "model": {
            "action_objective": "flow",
            "stop_action_to_vlm_grad": not is_public_lap,
            "dtype": "bfloat16",
            "legacy_image_order": is_public_lap,
            "action_dim": native_action_dim,
            "action_horizon": 16,
            "state_dim": native_state_dim,
            "image_resolution": [224, 224],
            "model_image_keys": model_image_keys,
            "prompt_format": "lap",
            "enable_langact_training": True,
        },
        "policy_input": {
            "primary_image_key": "base_0_rgb",
            "wrist_image_key": "left_wrist_0_rgb",
            "image_color_space": "RGB",
            "image_dtype": "uint8",
            "image_resolution": [224, 224],
            "model_image_order": model_image_order,
            "wrist_rotation_degrees": 180,
            "image_preprocessing": {
                "resize_profile": (
                    "tf_bilinear_half_pixel_antialias_false_uint8_round_symmetric_zero_pad_v1"
                ),
                "wrist_operation_order": ["resize_with_pad", "rotate_180"],
            },
            "dataset_name": "droid",
            "request_state_type": "eef_pose",
            "request_state_dim": 10,
            "state_dtype": "float32",
            "is_bimanual": False,
            "state_encoding": "EEF_R6",
            "state_layout": state_layout,
            "state_layout_mode": state_layout_mode,
            "gripper_open_value": 1.0,
            "gripper_closed_value": 0.0,
            "state_adapter": state_adapter,
        },
        "policy_output": {
            "action_encoding": "EEF_POS",
            "action_dim": 7,
            "native_action_dim": native_action_dim,
            "action_projection": action_projection,
            "model_action_horizon": 16,
            "response_horizon": response_horizon,
            "response_semantics": response_semantics,
            "execution_horizon": 8,
            "ar_endpoint_interpolation_profile": (
                "so3_right_multiply_slerp_identity_to_delta_inclusive_0_1_8_v1"
                if policy_type == "ar"
                else None
            ),
            "ar_endpoint_interpolation_steps": 8 if policy_type == "ar" else None,
            "action_layout": "delta_xyz+delta_extrinsic_xyz_euler+gripper_open",
            "translation_frame": "robot_base",
            "translation_units": "meters",
            "rotation_representation": "extrinsic_xyz_euler_delta",
            "rotation_composition": "right_multiply_current_by_delta",
            "rotation_units": "radians",
            "gripper_open_value": 1.0,
            "gripper_closed_value": 0.0,
            "gripper_execution_profile": "binary_model_open_gt_0p5_else_closed_v1",
            "gripper_threshold": 0.5,
        },
        "normalization": {
            "source": "checkpoint_assets",
            "compute_dtype": "float32",
            "type": "bounds_q99",
            "scope": "global",
            "policy_category": "single_arm",
            "selected_stats_sha256": "",
            "selected_stats_keys": ["actions", "state"],
            "formula_schema_version": 1,
            "formula_profile": "q99_train_matched_v1",
            "input_formula_id": "q99_input_eps1e-8_clip_zero0_v1",
            "output_formula_id": "q99_output_eps1e-8_zeroq01_extrapolate_v1",
            "formula_probe_sha256": "",
        },
        "language_action": {
            "format": "verbose_eef_with_rotation",
            "frame_description": frame_description,
        },
        "execution": {
            "schema_version": 2,
            "live_pipeline_validated": True,
            "normalization_compute_dtype": "float32",
            "inference_data_config": {
                "wrist_image_dropout_prob": 0.0,
                "mask_zero_img_prob": 0.0,
                "state_dropout": 0.0,
            },
            "image_routing": {"model_image_order": model_image_order},
            "droid_image_preprocessing": {
                "dataset_name": "droid",
                "registry_requires_wrist_rotation": True,
                "not_rotate_wrist_prob": 0.0,
                "resize_resolution": [224, 224],
                "rotation_applied": True,
                "wrist_rotation_degrees": 180,
                "operation_order_probe": {
                    "wrist_operation_order": ["resize_with_pad", "rotate_180"],
                    "distinguished": True,
                    "input_wrist_shape": [17, 29, 3],
                },
            },
            "ar_training_roundtrip_probe": {
                "dataset_name": "droid",
                "matches": True,
            },
            "normalized_state_probe": {"shape": [normalized_state_dim]},
            "flow_zero_action_probe": {"shape": [16, 7]},
            "output_transform_types": [
                "lap.transforms.Unnormalize",
                "lap.policies.transforms.output_transforms.CoTOutputs",
                *(
                    ["lap.policies.policy_config_adapter.ProjectSingleArmActions"]
                    if mixed_model
                    else []
                ),
            ],
            "normalization_formula": {
                "schema_version": 1,
                "profile": "q99_train_matched_v1",
                "input_formula_id": "q99_input_eps1e-8_clip_zero0_v1",
                "input_epsilon": 1e-8,
                "input_clip": [-1.0, 1.0],
                "input_zero_range_value": 0.0,
                "output_formula_id": "q99_output_eps1e-8_zeroq01_extrapolate_v1",
                "output_epsilon": 1e-8,
                "output_clip": None,
                "output_zero_range_value": "q01",
                "output_extrapolates_beyond_unit_interval": True,
                "applicable": True,
                "transform_strategy": "standard",
                "training_input_formula_id": "q99_input_eps1e-8_clip_zero0_v1",
                "training_policy_input_probe": {"matches": True},
                "training_policy_roundtrip_probe": {"matches": True},
                "output_extrapolation_probe": {
                    "extrapolates_beyond_q01_q99": True,
                    "zero_range_is_exact_q01": True,
                },
                "actual_stats_probe": {
                    "compute_dtype": "float32",
                    "stats_dtypes": {
                        "state": {
                            "mean": "float32",
                            "std": "float32",
                            "q01": "float32",
                            "q99": "float32",
                        },
                        "actions": {
                            "mean": "float32",
                            "std": "float32",
                            "q01": "float32",
                            "q99": "float32",
                        },
                    },
                    "training_policy_input_exact": True,
                    "policy_input_reference_exact": True,
                    "policy_output_reference_exact": True,
                    "state_probe_sha256": "0" * 64,
                    "action_probe_sha256": "1" * 64,
                },
            },
            "normalization_stats": {
                "keys": ["actions", "state"],
                "arrays": {
                    "actions": {
                        "q01": {"shape": [action_stats_dim]},
                        "q99": {"shape": [action_stats_dim]},
                    },
                    "state": {
                        "q01": {"shape": [state_stats_dim]},
                        "q99": {"shape": [state_stats_dim]},
                    },
                },
            },
        },
        "polaris": {
            "profile": "panda_link8_eef_pose_single_arm_v1",
            "serving_profile": FRANKA_DROID_SINGLE_ARM_PROFILE,
            "compatible": True,
            "incompatibilities": [],
            "eef_frame": "panda_link8",
            "normalization_scope": "global",
            "normalization_category": None,
            "q99_formula_profile": "q99_train_matched_v1",
            "numeric_action_frame": "robot_base",
            "state_layout": state_layout,
            "state_layout_mode": state_layout_mode,
        },
    }
    formula = contract["execution"]["normalization_formula"]
    formula["sha256"] = ego_lap_nested_digest(formula)
    stats = contract["execution"]["normalization_stats"]
    stats["sha256"] = ego_lap_nested_digest(stats)
    contract["normalization"]["formula_probe_sha256"] = formula["sha256"]
    contract["normalization"]["selected_stats_sha256"] = stats["sha256"]
    contract["execution"]["sha256"] = ego_lap_nested_digest(contract["execution"])
    contract["sha256"] = ego_lap_contract_digest(contract)
    return {"ego_lap_serving_contract": contract}


def _rehash_contract(metadata):
    contract = metadata["ego_lap_serving_contract"]
    formula = contract["execution"]["normalization_formula"]
    formula["sha256"] = ego_lap_nested_digest(formula)
    stats = contract["execution"]["normalization_stats"]
    stats["sha256"] = ego_lap_nested_digest(stats)
    contract["normalization"]["formula_probe_sha256"] = formula["sha256"]
    contract["normalization"]["selected_stats_sha256"] = stats["sha256"]
    contract["execution"]["sha256"] = ego_lap_nested_digest(contract["execution"])
    contract["sha256"] = ego_lap_contract_digest(contract)


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
        "ar_interpolation_steps": 8,
    }
    arguments.update(overrides)
    return validate_ego_lap_server_metadata(metadata, **arguments)


class ServingMetadataContractTest(unittest.TestCase):
    def test_native_and_mixed_models_share_exact_ten_in_seven_out_contract(self):
        native = _validate_metadata(_serving_metadata())
        self.assertEqual(native.serving_profile, FRANKA_DROID_SINGLE_ARM_PROFILE)
        self.assertEqual((native.native_action_dim, native.native_state_dim), (7, 10))
        self.assertEqual((native.request_state_dim, native.served_action_dim), (10, 7))
        self.assertIsNone(native.interpolation_steps)
        self.assertIsNone(native.ar_endpoint_interpolation_profile)
        self.assertIsNone(native.state_adapter)
        self.assertIsNone(native.action_projection)

        mixed = _validate_metadata(
            _serving_metadata(
                checkpoint_profile=MANIFEST_V1_PROFILE,
                mixed_model=True,
            ),
            expected_checkpoint_profile=MANIFEST_V1_PROFILE,
        )
        self.assertEqual((mixed.native_action_dim, mixed.native_state_dim), (14, 20))
        self.assertEqual((mixed.request_state_dim, mixed.served_action_dim), (10, 7))
        self.assertEqual(
            mixed.state_adapter,
            {
                "type": "normalize_then_zero_pad",
                "request_state_dim": 10,
                "model_state_dim": 14,
                "checkpoint_stats_dim": 20,
            },
        )
        self.assertEqual(
            mixed.action_projection,
            {
                "type": "leading_dimensions",
                "source_action_dim": 14,
                "output_action_dim": 7,
                "indices": list(range(7)),
                "applied_after": "checkpoint_unnormalization",
            },
        )
        self.assertEqual(mixed.state_layout, R6_COLUMNS_STATE_LAYOUT)
        self.assertEqual(mixed.state_layout_mode, MANIFEST_TRAIN_MATCHED_R6_MODE)
        self.assertEqual(mixed.normalization_profile, "q99_train_matched_v1")

    def test_mixed_model_contract_rejects_profile_adapter_and_width_tampering(self):
        cases = (
            (("serving_profile",), "native", "serving_profile"),
            (("polaris", "serving_profile"), "native", "polaris.serving_profile"),
            (
                ("checkpoint_profile",),
                "manifest_execution_v2",
                "checkpoint_profile for mixed",
            ),
            (("policy_input", "request_state_dim"), 14, "request_state_dim"),
            (
                ("policy_input", "state_adapter", "model_state_dim"),
                20,
                "state_adapter",
            ),
            (("policy_output", "action_dim"), 14, "policy_output.action_dim"),
            (
                ("policy_output", "native_action_dim"),
                7,
                "native_action_dim",
            ),
            (
                ("policy_output", "action_projection", "indices"),
                list(reversed(range(7))),
                "action_projection",
            ),
            (
                ("policy_output", "action_projection", "applied_after"),
                "model_output",
                "action_projection",
            ),
            (
                ("execution", "normalized_state_probe", "shape"),
                [10],
                "normalized_state_probe.shape",
            ),
            (
                ("execution", "flow_zero_action_probe", "shape"),
                [16, 14],
                "flow_zero_action_probe.shape",
            ),
            (
                ("execution", "normalization_stats", "arrays", "state", "q01", "shape"),
                [19],
                "state.q01.shape",
            ),
            (
                (
                    "execution",
                    "normalization_stats",
                    "arrays",
                    "actions",
                    "q99",
                    "shape",
                ),
                [7],
                "actions.q99.shape",
            ),
        )

        for path, value, error in cases:
            with self.subTest(path=".".join(path)):
                metadata = _serving_metadata(
                    checkpoint_profile=MANIFEST_V1_PROFILE,
                    mixed_model=True,
                )
                node = metadata["ego_lap_serving_contract"]
                for component in path[:-1]:
                    node = node[component]
                node[path[-1]] = value
                _rehash_contract(metadata)
                with self.assertRaisesRegex(ValueError, error):
                    _validate_metadata(metadata)

    def test_mixed_model_requires_unnormalize_before_exactly_one_projection(self):
        for transform_types, error in (
            (
                [
                    "lap.policies.policy_config_adapter.ProjectSingleArmActions",
                    "lap.transforms.Unnormalize",
                ],
                "unnormalization before",
            ),
            (["lap.transforms.Unnormalize"], "exactly one"),
            (
                [
                    "lap.transforms.Unnormalize",
                    "lap.policies.policy_config_adapter.ProjectSingleArmActions",
                    "lap.policies.policy_config_adapter.ProjectSingleArmActions",
                ],
                "exactly one",
            ),
        ):
            with self.subTest(transform_types=transform_types):
                metadata = _serving_metadata(
                    checkpoint_profile=MANIFEST_V1_PROFILE,
                    mixed_model=True,
                )
                metadata["ego_lap_serving_contract"]["execution"][
                    "output_transform_types"
                ] = transform_types
                _rehash_contract(metadata)
                with self.assertRaisesRegex(ValueError, error):
                    _validate_metadata(metadata)

    def test_native_model_rejects_spurious_single_arm_projection(self):
        metadata = _serving_metadata()
        document = metadata["ego_lap_serving_contract"]
        document["policy_output"]["action_projection"] = {
            "type": "leading_dimensions",
            "source_action_dim": 14,
            "output_action_dim": 7,
            "indices": list(range(7)),
            "applied_after": "checkpoint_unnormalization",
        }
        document["execution"]["output_transform_types"].append(
            "lap.policies.policy_config_adapter.ProjectSingleArmActions"
        )
        _rehash_contract(metadata)

        with self.assertRaisesRegex(ValueError, "action_projection"):
            _validate_metadata(metadata)

    def test_global_scope_does_not_select_category(self):
        metadata = _serving_metadata()

        contract = _validate_metadata(
            metadata,
            expected_normalization_scope="global",
        )

        self.assertEqual(contract.normalization_scope, "global")
        self.assertEqual(
            metadata["ego_lap_serving_contract"]["normalization"]["policy_category"],
            "single_arm",
        )
        self.assertEqual(
            metadata["ego_lap_serving_contract"]["polaris"]["normalization_scope"],
            "global",
        )
        self.assertIsNone(
            metadata["ego_lap_serving_contract"]["polaris"]["normalization_category"]
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
        _rehash_contract(metadata)
        contract = _validate_metadata(
            metadata,
            expected_normalization_scope="category",
        )
        self.assertEqual(contract.normalization_scope, "category")

    def test_modern_manifest_image_routing_is_accepted(self):
        metadata = _serving_metadata(
            frame_description="egocentric frame",
            checkpoint_profile="manifest_execution_v2",
        )

        contract = _validate_metadata(
            metadata,
            expected_checkpoint_profile="manifest_execution_v2",
            expected_frame_description="egocentric frame",
        )

        self.assertEqual(contract.frame_description, "egocentric frame")
        self.assertEqual(contract.state_layout, R6_COLUMNS_STATE_LAYOUT)
        self.assertEqual(
            contract.state_layout_mode,
            MANIFEST_TRAIN_MATCHED_R6_MODE,
        )

    def test_state_layout_is_checkpoint_specific_and_redundantly_bound(self):
        public_contract = _validate_metadata(_serving_metadata())
        self.assertEqual(public_contract.state_layout, R6_ROWS_STATE_LAYOUT)
        self.assertEqual(
            public_contract.state_layout_mode,
            PUBLIC_LAP_TRAIN_MATCHED_R6_MODE,
        )

        for field in ("policy_input", "polaris"):
            with self.subTest(field=field):
                metadata = _serving_metadata()
                document = metadata["ego_lap_serving_contract"]
                document[field]["state_layout"] = R6_COLUMNS_STATE_LAYOUT
                document["sha256"] = ego_lap_contract_digest(document)
                with self.assertRaisesRegex(ValueError, rf"{field}\.state_layout"):
                    _validate_metadata(metadata)

        metadata = _serving_metadata(checkpoint_profile="manifest_execution_v2")
        document = metadata["ego_lap_serving_contract"]
        document["polaris"]["state_layout_mode"] = PUBLIC_LAP_TRAIN_MATCHED_R6_MODE
        document["sha256"] = ego_lap_contract_digest(document)
        with self.assertRaisesRegex(ValueError, "polaris.state_layout_mode"):
            _validate_metadata(metadata)

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
        _rehash_contract(drifted)
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

    def test_contract_rejects_gripper_execution_drift(self):
        for field, value in (
            ("gripper_execution_profile", "continuous_v1"),
            ("gripper_threshold", 0.49),
        ):
            with self.subTest(field=field):
                metadata = _serving_metadata()
                document = metadata["ego_lap_serving_contract"]
                document["policy_output"][field] = value
                document["sha256"] = ego_lap_contract_digest(document)
                with self.assertRaisesRegex(ValueError, field):
                    _validate_metadata(metadata)

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
        _rehash_contract(metadata)
        with self.assertRaisesRegex(ValueError, "training_policy_roundtrip_probe"):
            _validate_metadata(metadata)

    def test_image_preprocessing_order_is_fail_closed(self):
        metadata = _serving_metadata()
        document = metadata["ego_lap_serving_contract"]
        document["policy_input"]["image_preprocessing"]["wrist_operation_order"] = [
            "rotate_180",
            "resize_with_pad",
        ]
        _rehash_contract(metadata)

        with self.assertRaisesRegex(ValueError, "wrist_operation_order"):
            _validate_metadata(metadata)

    def test_action_sampler_is_fail_closed(self):
        metadata = _serving_metadata()
        metadata["ego_lap_serving_contract"]["action_sampling"]["flow_num_steps"] = 9
        _rehash_contract(metadata)

        with self.assertRaisesRegex(ValueError, "action_sampling.flow_num_steps"):
            _validate_metadata(metadata)

        metadata = _serving_metadata(policy_type="ar")
        metadata["ego_lap_serving_contract"]["action_sampling"][
            "ar_max_decoding_steps"
        ] = 499
        _rehash_contract(metadata)
        with self.assertRaisesRegex(
            ValueError, "action_sampling.ar_max_decoding_steps"
        ):
            _validate_metadata(metadata)

    def test_public_model_path_is_exact_and_modern_value_is_manifest_bound(self):
        metadata = _serving_metadata()
        metadata["ego_lap_serving_contract"]["model"]["stop_action_to_vlm_grad"] = True
        _rehash_contract(metadata)
        with self.assertRaisesRegex(ValueError, "model.stop_action_to_vlm_grad"):
            _validate_metadata(metadata)

        metadata = _serving_metadata(checkpoint_profile="manifest_execution_v2")
        metadata["ego_lap_serving_contract"]["model"]["stop_action_to_vlm_grad"] = False
        _rehash_contract(metadata)
        _validate_metadata(metadata)

    def test_execution_image_operation_order_probe_is_fail_closed(self):
        metadata = _serving_metadata()
        metadata["ego_lap_serving_contract"]["execution"]["droid_image_preprocessing"][
            "operation_order_probe"
        ]["wrist_operation_order"] = ["rotate_180", "resize_with_pad"]
        _rehash_contract(metadata)
        with self.assertRaisesRegex(
            ValueError, "operation_order_probe.wrist_operation_order"
        ):
            _validate_metadata(metadata)

    def test_normalization_compute_dtype_is_fail_closed(self):
        metadata = _serving_metadata()
        document = metadata["ego_lap_serving_contract"]
        document["normalization"]["compute_dtype"] = "float64"
        _rehash_contract(metadata)
        with self.assertRaisesRegex(ValueError, "normalization.compute_dtype"):
            _validate_metadata(metadata)

        metadata = _serving_metadata()
        document = metadata["ego_lap_serving_contract"]
        document["execution"]["normalization_formula"]["actual_stats_probe"][
            "stats_dtypes"
        ]["state"]["q01"] = "float64"
        _rehash_contract(metadata)
        with self.assertRaisesRegex(ValueError, "stats_dtypes.state.q01"):
            _validate_metadata(metadata)

    def test_numeric_action_frame_is_unconditionally_robot_base(self):
        metadata = _serving_metadata(frame_description="egocentric frame")
        document = metadata["ego_lap_serving_contract"]
        document["policy_output"]["translation_frame"] = "egocentric"
        document["polaris"]["numeric_action_frame"] = "egocentric"
        _rehash_contract(metadata)

        with self.assertRaisesRegex(ValueError, "translation_frame"):
            _validate_metadata(metadata)

    def test_nested_formula_hash_rejects_unchecked_probe_tampering(self):
        metadata = _serving_metadata()
        document = metadata["ego_lap_serving_contract"]
        document["execution"]["normalization_formula"]["training_policy_input_probe"][
            "unvalidated_value"
        ] = 17
        # Keep the stale formula identity while making both enclosing identities
        # self-consistent. The nested formula check must catch the mutation.
        document["execution"]["sha256"] = ego_lap_nested_digest(document["execution"])
        document["sha256"] = ego_lap_contract_digest(document)

        with self.assertRaisesRegex(ValueError, "normalization_formula.sha256"):
            _validate_metadata(metadata)

    def test_nested_stats_hash_and_top_level_link_are_fail_closed(self):
        metadata = _serving_metadata()
        document = metadata["ego_lap_serving_contract"]
        document["execution"]["normalization_stats"]["arrays"]["state"] = {
            "q01": {"sha256": "f" * 64}
        }
        document["execution"]["sha256"] = ego_lap_nested_digest(document["execution"])
        document["sha256"] = ego_lap_contract_digest(document)
        with self.assertRaisesRegex(ValueError, "normalization_stats.sha256"):
            _validate_metadata(metadata)

        metadata = _serving_metadata()
        document = metadata["ego_lap_serving_contract"]
        document["normalization"]["selected_stats_sha256"] = "f" * 64
        document["sha256"] = ego_lap_contract_digest(document)
        with self.assertRaisesRegex(ValueError, "selected_stats_sha256"):
            _validate_metadata(metadata)

    def test_execution_hash_rejects_unchecked_field_tampering(self):
        metadata = _serving_metadata()
        document = metadata["ego_lap_serving_contract"]
        document["execution"]["unchecked_field"] = {"drift": True}
        document["sha256"] = ego_lap_contract_digest(document)

        with self.assertRaisesRegex(ValueError, "execution.sha256"):
            _validate_metadata(metadata)

    def test_formula_profile_constants_and_polaris_profile_are_exact(self):
        metadata = _serving_metadata()
        document = metadata["ego_lap_serving_contract"]
        document["execution"]["normalization_formula"]["input_epsilon"] = 1e-6
        _rehash_contract(metadata)
        with self.assertRaisesRegex(ValueError, "input_epsilon"):
            _validate_metadata(metadata)

        metadata = _serving_metadata()
        document = metadata["ego_lap_serving_contract"]
        document["polaris"]["profile"] = "another_link8_profile"
        document["sha256"] = ego_lap_contract_digest(document)
        with self.assertRaisesRegex(ValueError, "polaris.profile"):
            _validate_metadata(metadata)

    def test_legacy_formula_profile_is_exact_but_supported_for_public_lap(self):
        metadata = _serving_metadata()
        document = metadata["ego_lap_serving_contract"]
        normalization = document["normalization"]
        normalization.update(
            {
                "compute_dtype": "float64",
                "formula_profile": "q99_legacy_upstream_v1",
                "input_formula_id": "q99_input_eps1e-6_no_clip_zero0_v1",
                "output_formula_id": "q99_output_eps1e-6_no_zero_override_extrapolate_v1",
            }
        )
        formula = document["execution"]["normalization_formula"]
        formula.update(
            {
                "profile": "q99_legacy_upstream_v1",
                "input_formula_id": "q99_input_eps1e-6_no_clip_zero0_v1",
                "input_epsilon": 1e-6,
                "input_clip": None,
                "input_zero_range_value": 0.0,
                "output_formula_id": "q99_output_eps1e-6_no_zero_override_extrapolate_v1",
                "output_epsilon": 1e-6,
                "output_clip": None,
                "output_zero_range_value": "formula_result",
                "output_extrapolates_beyond_unit_interval": True,
            }
        )
        formula["training_policy_input_probe"]["matches"] = False
        formula["training_policy_roundtrip_probe"]["matches"] = False
        formula["output_extrapolation_probe"]["zero_range_is_exact_q01"] = False
        actual_stats_probe = formula["actual_stats_probe"]
        actual_stats_probe["compute_dtype"] = "float64"
        actual_stats_probe["training_policy_input_exact"] = False
        for group_dtypes in actual_stats_probe["stats_dtypes"].values():
            for field_name in group_dtypes:
                group_dtypes[field_name] = "float64"
        document["execution"]["normalization_compute_dtype"] = "float64"
        document["polaris"]["q99_formula_profile"] = "q99_legacy_upstream_v1"
        _rehash_contract(metadata)

        contract = _validate_metadata(
            metadata,
            expected_normalization_profile="q99_legacy_upstream_v1",
        )
        self.assertEqual(contract.normalization_profile, "q99_legacy_upstream_v1")

        mixed_metadata = _serving_metadata(
            checkpoint_profile=MANIFEST_V1_PROFILE,
            mixed_model=True,
        )
        mixed_document = mixed_metadata["ego_lap_serving_contract"]
        mixed_document["normalization"].update(
            {
                key: copy.deepcopy(normalization[key])
                for key in (
                    "compute_dtype",
                    "formula_profile",
                    "input_formula_id",
                    "output_formula_id",
                )
            }
        )
        mixed_document["execution"]["normalization_compute_dtype"] = "float64"
        mixed_document["execution"]["normalization_formula"] = copy.deepcopy(formula)
        mixed_document["polaris"]["q99_formula_profile"] = "q99_legacy_upstream_v1"
        _rehash_contract(mixed_metadata)
        with self.assertRaisesRegex(ValueError, "requires.*q99_train_matched_v1"):
            _validate_metadata(mixed_metadata)

    def test_ar_contract_derives_one_to_eight_to_eight_protocol(self):
        contract = _validate_metadata(
            _serving_metadata(policy_type="ar"),
            expected_policy_type="ar",
            expected_open_loop_horizon=8,
        )

        self.assertEqual(contract.response_horizon, 1)
        self.assertEqual(contract.interpolation_steps, 8)
        self.assertEqual(contract.execution_horizon, 8)

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
    def test_rot6d_row_and_column_layouts_have_exact_asymmetric_order(self):
        # This quaternion produces the asymmetric cyclic permutation matrix
        # [[0, 0, 1], [1, 0, 0], [0, 1, 0]], so rows and columns cannot pass
        # accidentally through symmetry.
        quaternion = np.array([0.5, 0.5, 0.5, 0.5])
        rows = quaternion_wxyz_to_rot6d(
            quaternion,
            state_layout=R6_ROWS_STATE_LAYOUT,
        )
        columns = quaternion_wxyz_to_rot6d(
            quaternion,
            state_layout=R6_COLUMNS_STATE_LAYOUT,
        )
        np.testing.assert_array_equal(
            rows,
            np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
        )
        np.testing.assert_array_equal(
            columns,
            np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        )

        with self.assertRaisesRegex(ValueError, "Unsupported Ego-LAP R6"):
            quaternion_wxyz_to_rot6d(quaternion, state_layout="ambiguous_r6")

    def test_state_is_xyz_rot6d_and_open_positive_gripper(self):
        state = build_lap_state(
            np.array([0.4, -0.2, 0.3]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.25]),
            state_layout=R6_COLUMNS_STATE_LAYOUT,
        )
        np.testing.assert_allclose(
            state,
            np.array([0.4, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]),
        )

    def test_state_gripper_matches_official_half_threshold(self):
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        open_state = build_lap_state(
            np.zeros(3),
            identity,
            np.array([0.49]),
            state_layout=R6_ROWS_STATE_LAYOUT,
        )
        closed_state = build_lap_state(
            np.zeros(3),
            identity,
            np.array([0.5]),
            state_layout=R6_ROWS_STATE_LAYOUT,
        )
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
        np.testing.assert_array_equal(actions[:, 7], np.array([0.0, 0.0]))

    def test_anchored_gripper_is_binary_at_training_boundary(self):
        deltas = np.zeros((3, 7), dtype=float)
        deltas[:, 6] = [0.49, 0.5, 0.51]

        actions = anchor_action_chunk(
            deltas,
            np.zeros(3),
            np.array([1.0, 0.0, 0.0, 0.0]),
        )

        np.testing.assert_array_equal(actions[:, 7], np.array([1.0, 1.0, 0.0]))

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

        chunk = interpolate_ar_endpoint(endpoint, steps=8)

        self.assertEqual(chunk.shape, (8, 7))
        np.testing.assert_array_equal(chunk[0, :6], np.zeros(6))
        np.testing.assert_allclose(chunk[1, :3], endpoint[0, :3] / 7)
        np.testing.assert_allclose(chunk[3, :3], endpoint[0, :3] * (3 / 7))
        np.testing.assert_allclose(chunk[-1, :3], endpoint[0, :3])
        endpoint_rotvec = Rotation.from_euler("xyz", endpoint[0, 3:6]).as_rotvec()
        expected_rotations = Rotation.from_rotvec(
            np.linspace(0.0, 1.0, 8)[:, None] * endpoint_rotvec
        )
        actual_rotations = Rotation.from_euler("xyz", chunk[:, 3:6])
        np.testing.assert_allclose(
            actual_rotations.as_matrix(),
            expected_rotations.as_matrix(),
            atol=1e-10,
        )
        self.assertFalse(np.allclose(chunk[3, 3:6], endpoint[0, 3:6] * (3 / 7)))
        np.testing.assert_allclose(
            actual_rotations[-1].as_matrix(),
            Rotation.from_euler("xyz", endpoint[0, 3:6]).as_matrix(),
            atol=1e-10,
        )
        np.testing.assert_array_equal(chunk[:, 6], np.ones(8))


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
        expected_stats_sha256 = fake_server.metadata["ego_lap_serving_contract"][
            "normalization"
        ]["selected_stats_sha256"]
        external = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        wrist = np.arange(12, 24, dtype=np.uint8).reshape(2, 2, 3)
        observation = {
            "splat": {"external_cam": external, "wrist_cam": wrist},
            "policy": {
                "eef_pos": np.array([[0.4, 0.0, 0.2]]),
                "eef_quat": np.array([[0.5, 0.5, 0.5, 0.5]]),
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
            normalization_stats_sha256=expected_stats_sha256,
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
        self.assertEqual(request["observation"]["state"].dtype, np.float32)
        self.assertEqual(
            request["observation"]["cartesian_position"].dtype,
            np.float32,
        )
        self.assertEqual(
            request["observation"]["gripper_position"].dtype,
            np.float32,
        )
        np.testing.assert_array_equal(
            request["observation"]["state"][3:9],
            np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        )
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
        self.assertEqual(trace_records[1]["state_layout"], R6_ROWS_STATE_LAYOUT)
        self.assertEqual(
            trace_records[1]["state_layout_mode"],
            PUBLIC_LAP_TRAIN_MATCHED_R6_MODE,
        )
        self.assertEqual(len(trace_records[1]["server_delta_chunk"]), 16)
        self.assertEqual(
            persisted_contract,
            fake_server.metadata["ego_lap_serving_contract"],
        )

    def test_manifest_client_uses_contracted_column_r6_layout(self):
        fake_server = _FakePolicyServer(
            {"actions": np.zeros((16, 7))},
            metadata=_serving_metadata(checkpoint_profile="manifest_execution_v2"),
        )
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        observation = {
            "splat": {"external_cam": image, "wrist_cam": image},
            "policy": {
                "eef_pos": np.array([[0.4, 0.0, 0.2]]),
                "eef_quat": np.array([[0.5, 0.5, 0.5, 0.5]]),
                "gripper_pos": np.array([[0.0]]),
            },
        }
        args = PolicyArgs(
            client="EgoLAPEefPose",
            checkpoint_profile="manifest_execution_v2",
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
            client.infer(observation, "move")
            records = [
                json.loads(line)
                for line in Path(args.trace_path).read_text().splitlines()
            ]

        request_state = fake_server.requests[0]["observation"]["state"]
        np.testing.assert_array_equal(
            request_state[3:9],
            np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
        self.assertEqual(records[1]["state_layout"], R6_COLUMNS_STATE_LAYOUT)
        self.assertEqual(
            records[1]["state_layout_mode"],
            MANIFEST_TRAIN_MATCHED_R6_MODE,
        )

    def test_mixed_manifest_client_keeps_ten_dimensional_request_and_seven_dimensional_response(
        self,
    ):
        fake_server = _FakePolicyServer(
            {"actions": np.zeros((16, 7))},
            metadata=_serving_metadata(
                checkpoint_profile=MANIFEST_V1_PROFILE,
                mixed_model=True,
            ),
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
            checkpoint_profile=MANIFEST_V1_PROFILE,
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
            action, _ = client.infer(observation, "move")

        self.assertEqual(fake_server.requests[0]["observation"]["state"].shape, (10,))
        self.assertEqual(action.shape, (8,))
        self.assertEqual(client.contract.native_action_dim, 14)
        self.assertEqual(client.contract.native_state_dim, 20)
        self.assertEqual(client.contract.served_action_dim, 7)

    def test_ar_client_interpolates_endpoint_and_executes_all_eight(self):
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
            open_loop_horizon=8,
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
            actions = [client.infer(observation, "move")[0] for _ in range(8)]
            records = [
                json.loads(line)
                for line in Path(args.trace_path).read_text().splitlines()
            ]

        self.assertEqual(len(fake_server.requests), 1)
        np.testing.assert_allclose(
            np.asarray(actions)[:, 0],
            0.4 + np.linspace(0.0, 0.16, 8),
            atol=1e-7,
        )
        np.testing.assert_array_equal(np.asarray(actions)[:, 7], np.zeros(8))
        query = records[1]
        self.assertEqual(query["response_semantics"], "total_delta_endpoint")
        self.assertEqual(len(query["server_delta_chunk"]), 1)
        self.assertEqual(len(query["raw_delta_chunk"]), 8)
        np.testing.assert_array_equal(
            query["raw_delta_chunk"][0], np.array([0.0] * 6 + [1.0])
        )
        self.assertEqual(query["raw_delta_chunk"][-1][0], 0.16)
        self.assertEqual(query["execution_horizon"], 8)

    def test_per_episode_trace_uses_global_id_and_reconciles_partial_retry(self):
        fake_server = _FakePolicyServer(
            {"actions": np.zeros((16, 7))},
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
        args = PolicyArgs(client="EgoLAPEefPose", open_loop_horizon=8)

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
            trace_dir = Path(temporary_directory) / "policy_traces"
            args.trace_dir = str(trace_dir)
            args.contract_output = str(Path(temporary_directory) / "contract.json")
            client = EgoLAPEefPoseClient(args)

            # Simulate a preempted attempt, then reset the same global episode.
            client.reset(episode_index=7)
            client.infer(observation, "move")
            client.reset(episode_index=7)
            client.infer(observation, "move")
            finalized = client.finalize_episode(
                episode_length=1,
                success=False,
                progress=0.25,
            )
            records = [json.loads(line) for line in finalized.read_text().splitlines()]

        self.assertEqual(finalized.name, "episode_000007.jsonl")
        self.assertEqual(records[0]["event"], "reset")
        self.assertEqual(records[-1]["event"], "episode_complete")
        self.assertTrue(all(record["episode"] == 7 for record in records))
        self.assertEqual(sum(record["event"] == "action" for record in records), 1)
        self.assertEqual(records[-1]["episode_length"], 1)
        self.assertEqual(records[-1]["status"], "completed")

    def test_client_rejects_non_droid_eef_frame(self):
        args = PolicyArgs(client="EgoLAPEefPose")
        args.eef_frame = "base_link"  # type: ignore[assignment]
        with self.assertRaisesRegex(ValueError, "panda_link8"):
            EgoLAPEefPoseClient(args)


if __name__ == "__main__":
    unittest.main()
