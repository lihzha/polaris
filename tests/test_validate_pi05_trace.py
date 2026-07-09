import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from polaris.evaluation_seed import environment_seed_contract_sha256
from polaris.pi05_droid_jointpos_image_contract import (
    CLIENT_RESIZE_PROFILE,
    FILTER_PROBE,
    IMAGE_PROFILE,
    static_image_contract,
)


SCRIPT_PATH = Path(__file__).parents[1] / "scripts/polaris/validate_pi05_trace.py"
SPEC = importlib.util.spec_from_file_location("validate_pi05_trace", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _records(
    action_count: int = 8,
    *,
    seeded: bool = False,
    reset_index: int = 0,
    base_seed: int = 0,
) -> list[dict]:
    response = [[float(row + column) for column in range(8)] for row in range(15)]
    planned = [row.copy() for row in response[:8]]
    for action_index, action in enumerate(planned):
        action[-1] = float(response[action_index][-1] > 0.5)
    query = {
        "schema_version": 1,
        "record_type": "openpi_joint_position_query",
        "reset_index": reset_index,
        "query_index": 0,
        "prompt": "put all foods in the bowl",
        "state": {
            "joint_position": [0.0] * 7,
            "gripper_position": [0.0],
        },
        "images": {
            "external": {"shape": [224, 224, 3], "dtype": "uint8", "sha256": "a" * 64},
            "wrist": {"shape": [224, 224, 3], "dtype": "uint8", "sha256": "b" * 64},
            "wrist_rotation_degrees": 0,
        },
        "response_action_shape": [15, 8],
        "response_action_chunk": response,
        "execution_horizon": 8,
        "planned_action_chunk": planned,
    }
    if seeded:
        query["schema_version"] = 2
        query["environment_rng"] = {
            "schema_version": 1,
            "profile": "isaaclab_env_seed_base_plus_episode_v1",
            "base_seed": base_seed,
            "scheme": "base_plus_episode_index_v1",
            "episode_index": reset_index,
            "episode_seed": base_seed + reset_index,
            "live_cfg_seed": base_seed,
            "physx_enhanced_determinism": False,
            "determinism_claim": "rng_bound_not_bitwise",
        }
    actions = []
    for action_index in range(action_count):
        actions.append(
            {
                "schema_version": 1,
                "record_type": "openpi_joint_position_action",
                "reset_index": reset_index,
                "query_index": 0,
                "chunk_action_index": action_index,
                "raw_action": response[action_index],
                "emitted_action": planned[action_index],
            }
        )
        if seeded:
            actions[-1]["schema_version"] = 2
    return [query, *actions]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _image_id(shape, dtype, digest):
    return {"shape": list(shape), "dtype": dtype, "sha256": digest}


def _camera_evidence(name, final_digest):
    post = _image_id((720, 1280, 3), "uint8", "4" * 64)
    return {
        "schema_version": 1,
        "profile": IMAGE_PROFILE,
        "contract_sha256": static_image_contract()["contract_sha256"],
        "camera_name": name,
        "background_source": "filtered_splat",
        "renderer_stages": {
            "source": "splat_renderer_output",
            "cv2_runtime": {
                "provider": "opencv-python-headless",
                "package_version": "4.11.0.86",
                "opencv_version": "4.11.0",
                "module_path": "/.venv/lib/python3.11/site-packages/cv2/__init__.py",
                "interpolation": "cv2.INTER_LINEAR",
                "interpolation_value": 1,
                "probe": dict(FILTER_PROBE),
            },
            "renderer_float": _image_id((720, 1280, 3), "float32", "1" * 64),
            "pre_filter_uint8": _image_id((720, 1280, 3), "uint8", "2" * 64),
            "half_resolution_uint8": _image_id((360, 640, 3), "uint8", "3" * 64),
            "post_filter_splat_uint8": post,
        },
        "composite_background_uint8": post,
        "composite_mask_int64": _image_id((720, 1280, 1), "int64", "5" * 64),
        "composite_mask_coverage": {
            "true_pixel_count": 100,
            "total_pixel_count": 720 * 1280,
            "true_fraction": 100 / (720 * 1280),
        },
        "sim_rgb_layer_uint8": _image_id((720, 1280, 3), "uint8", "6" * 64),
        "final_composite_uint8": _image_id((720, 1280, 3), "uint8", final_digest),
    }


def _resize_evidence(final_digest, wire_digest):
    return {
        "profile": CLIENT_RESIZE_PROFILE,
        "runtime": {
            "profile": CLIENT_RESIZE_PROFILE,
            "implementation": "openpi_client.image_tools.resize_with_pad",
            "backend": "PIL.Image.resize",
            "method": "PIL.Image.Resampling.BILINEAR",
            "padding": "symmetric_zero",
            "source": {
                "path": "/openpi/image_tools.py",
                "size": 1,
                "sha256": (
                    "d48b4bd7f44e79fe6db8a8e07c9161144fa250be686e1245014a8b47e6171977"
                ),
            },
            "probe_output_sha256": (
                "4485703601c6d6fa2d256374d7b7e2fb9c60d585e8278aa4251983a96ec74cc5"
            ),
            "server_224_to_224": "early_return_same_array_no_pixel_change",
            "pillow_version": "11.3.0",
            "pillow_module": "PIL.Image",
        },
        "input_final_composite": _image_id((720, 1280, 3), "uint8", final_digest),
        "wire_request": _image_id((224, 224, 3), "uint8", wire_digest),
    }


def _attested_records(base_seed: int = 7) -> list[dict]:
    seed_contract = {
        "schema_version": 1,
        "profile": "isaaclab_env_seed_base_plus_episode_v1",
        "base_seed": base_seed,
        "scheme": "base_plus_episode_index_v1",
        "live_cfg_seed": base_seed,
        "physx_enhanced_determinism": False,
        "determinism_claim": "rng_bound_not_bitwise",
        "binding": "env_cfg_seed_before_gym_make_and_reset_seed_per_episode",
    }
    seed_hash = environment_seed_contract_sha256(seed_contract)
    identity = {
        "schema_version": 5,
        "profile": "openpi_pi05_droid_native_joint_position_v2",
        "reset_index": 0,
        "server_contract_sha256": "c" * 64,
        "environment_seed_contract_sha256": seed_hash,
        "runtime_contract_sha256": "d" * 64,
        "physx_enhanced_determinism": False,
    }
    rng = {
        "schema_version": 2,
        "profile": seed_contract["profile"],
        "base_seed": base_seed,
        "scheme": seed_contract["scheme"],
        "episode_index": 0,
        "episode_seed": base_seed,
        "live_cfg_seed": base_seed,
        "physx_enhanced_determinism": False,
        "determinism_claim": "rng_bound_not_bitwise",
        "environment_seed_contract_sha256": seed_hash,
    }
    response = [[0.1] * 7 + [0.0] for _ in range(15)]
    processed = np.asarray([0.1] * 7, dtype=np.float32).tolist()
    records = []
    sim_base = 100
    sensor_base = 5
    for outer_step in range(450):
        query_index = outer_step // 8
        chunk_index = outer_step % 8
        global_query_index = query_index
        if chunk_index == 0:
            records.append(
                {
                    **identity,
                    "record_type": "openpi_joint_position_query",
                    "query_index": query_index,
                    "global_query_index": global_query_index,
                    "environment_rng": rng,
                    "sensor_frame_counters": {
                        "external_cam": sensor_base + outer_step,
                        "wrist_cam": sensor_base + outer_step,
                    },
                    "prompt": "put all foods in the bowl",
                    "state": {
                        "joint_position": [0.0] * 7,
                        "gripper_position": [0.0],
                    },
                    "images": {
                        "environment_image_contract": static_image_contract(),
                        "external_camera_pipeline": _camera_evidence(
                            "external_cam", "a" * 64
                        ),
                        "wrist_camera_pipeline": _camera_evidence(
                            "wrist_cam", "b" * 64
                        ),
                        "final_composite_external": _image_id(
                            (720, 1280, 3), "uint8", "a" * 64
                        ),
                        "final_composite_wrist": _image_id(
                            (720, 1280, 3), "uint8", "b" * 64
                        ),
                        "client_resize_external": _resize_evidence("a" * 64, "e" * 64),
                        "client_resize_wrist": _resize_evidence("b" * 64, "f" * 64),
                        "request_external": _image_id((224, 224, 3), "uint8", "e" * 64),
                        "request_wrist": _image_id((224, 224, 3), "uint8", "f" * 64),
                        "server224_external_idempotent": _image_id(
                            (224, 224, 3), "uint8", "e" * 64
                        ),
                        "server224_wrist_idempotent": _image_id(
                            (224, 224, 3), "uint8", "f" * 64
                        ),
                        "query_visualization_external": _image_id(
                            (224, 224, 3), "uint8", "e" * 64
                        ),
                        "query_visualization_wrist": _image_id(
                            (224, 224, 3), "uint8", "f" * 64
                        ),
                        "model_order": [
                            "base_0_rgb",
                            "left_wrist_0_rgb",
                            "right_wrist_0_rgb_masked",
                        ],
                        "client_model_spatial_transform": CLIENT_RESIZE_PROFILE,
                        "server_model_resize": (
                            MODULE.PI05_DROID_JOINTPOS_SERVER_MODEL_RESIZE
                        ),
                        "masked_third_slot": (
                            "server_DroidInputs_zeros_like_base_mask_false"
                        ),
                        "query_visualization_source": (
                            "byte_identical_client224_wire_model_input"
                        ),
                        "wrist_rotation_degrees": 0,
                    },
                    "response_action_shape": [15, 8],
                    "response_action_dtype": "float64",
                    "response_action_chunk": response,
                    "execution_horizon": 8,
                    "planned_action_chunk": response[:8],
                }
            )
        action_identity = {
            **identity,
            "query_index": query_index,
            "global_query_index": global_query_index,
            "chunk_action_index": chunk_index,
        }
        emitted = response[chunk_index]
        records.append(
            {
                **action_identity,
                "record_type": "openpi_joint_position_action",
                "raw_action": emitted,
                "emitted_action": emitted,
            }
        )
        before = {
            "boundary_profile": "outer450_internal451_no_autoreset",
            "live_max_episode_length": 451,
            "episode_length": outer_step,
            "sim_step_counter": sim_base + outer_step * 8,
            "common_step_counter": outer_step,
            "sensor_frame_counters": {
                "external_cam": sensor_base + outer_step,
                "wrist_cam": sensor_base + outer_step,
            },
        }
        after = {
            **before,
            "episode_length": outer_step + 1,
            "sim_step_counter": sim_base + (outer_step + 1) * 8,
            "common_step_counter": outer_step + 1,
            "sensor_frame_counters": {
                "external_cam": sensor_base + outer_step + 1,
                "wrist_cam": sensor_base + outer_step + 1,
            },
        }
        records.append(
            {
                **action_identity,
                "record_type": "openpi_joint_position_execution",
                "outer_step_index": outer_step,
                "emitted_action": emitted,
                "action_execution": {
                    "schema_version": 1,
                    "processing": (
                        "upstream_joint_position_action_scale1_offset0_no_clip"
                    ),
                    "raw_action_buffer": processed,
                    "processed_action_buffer": processed,
                    "apply_target_holds": [processed] * 8,
                    "apply_target_hold_count": 8,
                    "post_step_articulation_target": processed,
                },
                "processed_finger_position_target": [0.0],
                "articulation_finger_position_target": [0.0],
                "measured_joint_position_after": [0.0] * 7,
                "measured_closed_positive_gripper_after": [0.0],
                "environment_before": before,
                "environment_after": after,
                "terminated": False,
                "truncated": False,
                "terminal_rubric": (
                    {"success": True, "progress": 0.5, "metrics": {}}
                    if outer_step == 449
                    else None
                ),
                "terminal_visualization": (
                    {
                        "shape": [224, 448, 3],
                        "dtype": "uint8",
                        "sha256": "7" * 64,
                        "source": (
                            "post_action450_returned_nonexpensive_sim_camera_observation"
                        ),
                    }
                    if outer_step == 449
                    else None
                ),
            }
        )
    return records


class TraceAuditTest(unittest.TestCase):
    def test_valid_seeded_trace_binds_expected_episode_rng(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace_path = Path(temporary_directory) / "trace.jsonl"
            metrics_path = Path(temporary_directory) / "eval_results.csv"
            _write_jsonl(trace_path, _attested_records(base_seed=7))
            metrics_path.write_text(
                "episode,episode_length,success,progress,numerical_failure,"
                "numerical_failure_reason\n0,450,True,0.5,False,\n"
            )
            expected_metrics_sha256 = hashlib.sha256(
                metrics_path.read_bytes()
            ).hexdigest()
            with mock.patch(
                "polaris.pi05_droid_jointpos_runtime."
                "validate_jointpos_runtime_artifact",
                return_value={"runtime_sha256": "d" * 64},
            ):
                summary = MODULE.audit_trace(
                    trace_path,
                    metrics_csv=metrics_path,
                    expected_environment_seed=7,
                    expected_server_contract_sha256="c" * 64,
                    runtime_contract_path=Path(temporary_directory) / "runtime.json",
                )

        self.assertEqual(summary["environment_base_seed"], 7)
        self.assertEqual(summary["metrics_sha256"], expected_metrics_sha256)
        self.assertEqual(
            summary["environment_seed_scheme"], "base_plus_episode_index_v1"
        )
        self.assertEqual(summary["environment_episode_seeds"], [7])
        self.assertEqual(summary["episode_query_counts"], [57])
        self.assertEqual(summary["cumulative_query_counts"], [57])
        self.assertEqual(summary["request_image_shape"], [224, 224, 3])
        self.assertEqual(summary["request_image_dtype"], "uint8")
        self.assertEqual(
            summary["client_model_spatial_transform"], CLIENT_RESIZE_PROFILE
        )
        self.assertEqual(
            summary["server_model_resize"],
            MODULE.PI05_DROID_JOINTPOS_SERVER_MODEL_RESIZE,
        )
        self.assertTrue(summary["query_visualization_is_model_input"])
        self.assertFalse(summary["interquery_visualization_is_model_input"])
        self.assertEqual(summary["terminal_visualization_shape"], [224, 448, 3])
        self.assertEqual(summary["terminal_visualization_sha256"], ["7" * 64])
        self.assertFalse(summary["environment_physx_enhanced_determinism"])
        self.assertEqual(
            summary["environment_determinism_claim"], "rng_bound_not_bitwise"
        )

    def test_valid_official_pi05_trace_matches_metrics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace_path = Path(temporary_directory) / "trace.jsonl"
            metrics_path = Path(temporary_directory) / "eval_results.csv"
            _write_jsonl(trace_path, _records())
            metrics_path.write_text("episode,episode_length\n0,8\n")
            summary = MODULE.audit_trace(
                trace_path,
                expected_prompt="put all foods in the bowl",
                metrics_csv=metrics_path,
            )

        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["query_records"], 1)
        self.assertEqual(summary["emitted_action_records"], 8)
        self.assertEqual(summary["episode_lengths"], [8])
        self.assertEqual(summary["response_action_shape"], [15, 8])
        self.assertEqual(summary["wrist_rotation_degrees"], 0)

    def test_rotated_wrist_contract_fails(self):
        records = _records()
        records[0]["images"]["wrist_rotation_degrees"] = 180
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace_path = Path(temporary_directory) / "trace.jsonl"
            _write_jsonl(trace_path, records)
            with self.assertRaisesRegex(ValueError, "wrist must be unrotated"):
                MODULE.audit_trace(trace_path)

    def test_duplicate_query_contract_fails(self):
        records = _records()
        records.append(records[0])
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace_path = Path(temporary_directory) / "trace.jsonl"
            _write_jsonl(trace_path, records)
            with self.assertRaisesRegex(ValueError, "duplicate query"):
                MODULE.audit_trace(trace_path)

    def test_action_before_query_fails(self):
        records = _records()
        records[0], records[1] = records[1], records[0]
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace_path = Path(temporary_directory) / "trace.jsonl"
            _write_jsonl(trace_path, records)
            with self.assertRaisesRegex(ValueError, "appears before its query"):
                MODULE.audit_trace(trace_path)

    def test_metrics_action_count_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace_path = Path(temporary_directory) / "trace.jsonl"
            metrics_path = Path(temporary_directory) / "eval_results.csv"
            _write_jsonl(trace_path, _records(action_count=7))
            metrics_path.write_text("episode,episode_length\n0,8\n")
            with self.assertRaisesRegex(ValueError, "emitted 7 actions"):
                MODULE.audit_trace(trace_path, metrics_csv=metrics_path)

    def test_seeded_trace_rejects_missing_or_wrong_rng_provenance(self):
        records = _attested_records(base_seed=7)
        del records[0]["environment_rng"]
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace_path = Path(temporary_directory) / "trace.jsonl"
            _write_jsonl(trace_path, records)
            with self.assertRaisesRegex(ValueError, "query schema"):
                MODULE.audit_trace(
                    trace_path,
                    expected_environment_seed=7,
                    expected_server_contract_sha256="c" * 64,
                    runtime_contract_path=Path(temporary_directory) / "runtime.json",
                )

        records = _attested_records(base_seed=7)
        records[0]["environment_rng"]["episode_seed"] = 8
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace_path = Path(temporary_directory) / "trace.jsonl"
            _write_jsonl(trace_path, records)
            with self.assertRaisesRegex(ValueError, "derived episode seed"):
                MODULE.audit_trace(
                    trace_path,
                    expected_environment_seed=7,
                    expected_server_contract_sha256="c" * 64,
                    runtime_contract_path=Path(temporary_directory) / "runtime.json",
                )

        records = _attested_records(base_seed=7)
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace_path = Path(temporary_directory) / "trace.jsonl"
            _write_jsonl(trace_path, records)
            with self.assertRaisesRegex(ValueError, "expected environment base seed"):
                MODULE.audit_trace(
                    trace_path,
                    expected_environment_seed=8,
                    expected_server_contract_sha256="c" * 64,
                    runtime_contract_path=Path(temporary_directory) / "runtime.json",
                )

    def test_attested_numeric_vectors_reject_boolean_values(self):
        records = _attested_records()
        records[0]["state"]["joint_position"][0] = True
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace_path = Path(temporary_directory) / "trace.jsonl"
            _write_jsonl(trace_path, records)
            with self.assertRaisesRegex(ValueError, "non-finite or non-numeric"):
                MODULE.audit_trace(
                    trace_path,
                    expected_environment_seed=7,
                    expected_server_contract_sha256="c" * 64,
                    runtime_contract_path=Path(temporary_directory) / "runtime.json",
                )

    def test_attested_trace_rejects_non_wire_shape_model_request(self):
        records = _attested_records()
        records[0]["images"]["request_external"]["shape"] = [720, 1280, 3]
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace_path = Path(temporary_directory) / "trace.jsonl"
            _write_jsonl(trace_path, records)
            with self.assertRaisesRegex(ValueError, "image shape/dtype mismatch"):
                MODULE.audit_trace(
                    trace_path,
                    expected_environment_seed=7,
                    expected_server_contract_sha256="c" * 64,
                    runtime_contract_path=Path(temporary_directory) / "runtime.json",
                )

    def test_attested_trace_requires_byte_identical_wire_model_request(self):
        records = _attested_records()
        records[0]["images"]["request_wrist"]["sha256"] = "9" * 64
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace_path = Path(temporary_directory) / "trace.jsonl"
            _write_jsonl(trace_path, records)
            with self.assertRaisesRegex(
                ValueError, "final720/client224/wire/model/viz identity mismatch"
            ):
                MODULE.audit_trace(
                    trace_path,
                    expected_environment_seed=7,
                    expected_server_contract_sha256="c" * 64,
                    runtime_contract_path=Path(temporary_directory) / "runtime.json",
                )

    def test_attested_trace_requires_post_action450_terminal_visualization(self):
        records = _attested_records()
        records[-1]["terminal_visualization"] = None
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace_path = Path(temporary_directory) / "trace.jsonl"
            _write_jsonl(trace_path, records)
            with self.assertRaisesRegex(ValueError, "terminal visualization schema"):
                MODULE.audit_trace(
                    trace_path,
                    expected_environment_seed=7,
                    expected_server_contract_sha256="c" * 64,
                    runtime_contract_path=Path(temporary_directory) / "runtime.json",
                )


if __name__ == "__main__":
    unittest.main()
