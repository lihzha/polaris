import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


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


class TraceAuditTest(unittest.TestCase):
    def test_valid_seeded_trace_binds_expected_episode_rng(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace_path = Path(temporary_directory) / "trace.jsonl"
            metrics_path = Path(temporary_directory) / "eval_results.csv"
            _write_jsonl(trace_path, _records(seeded=True, base_seed=7))
            metrics_path.write_text("episode,episode_length\n0,8\n")
            summary = MODULE.audit_trace(
                trace_path,
                metrics_csv=metrics_path,
                expected_environment_seed=7,
            )

        self.assertEqual(summary["environment_base_seed"], 7)
        self.assertEqual(
            summary["environment_seed_scheme"], "base_plus_episode_index_v1"
        )
        self.assertEqual(summary["environment_episode_seeds"], [7])
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
        records = _records(seeded=True, base_seed=7)
        del records[0]["environment_rng"]
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace_path = Path(temporary_directory) / "trace.jsonl"
            _write_jsonl(trace_path, records)
            with self.assertRaisesRegex(ValueError, "environment_rng fields"):
                MODULE.audit_trace(trace_path, expected_environment_seed=7)

        records = _records(seeded=True, base_seed=7)
        records[0]["environment_rng"]["episode_seed"] = 8
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace_path = Path(temporary_directory) / "trace.jsonl"
            _write_jsonl(trace_path, records)
            with self.assertRaisesRegex(ValueError, "derived episode seed"):
                MODULE.audit_trace(trace_path, expected_environment_seed=7)

        records = _records(seeded=True, base_seed=7)
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace_path = Path(temporary_directory) / "trace.jsonl"
            _write_jsonl(trace_path, records)
            with self.assertRaisesRegex(ValueError, "expected environment base seed"):
                MODULE.audit_trace(trace_path, expected_environment_seed=8)


if __name__ == "__main__":
    unittest.main()
