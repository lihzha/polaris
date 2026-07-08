import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts/polaris/audit_pi05_joint_bounds.py"
SPEC = importlib.util.spec_from_file_location("audit_pi05_joint_bounds", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _query(reset_index, query_index, state, response_rows):
    return {
        "record_type": "openpi_joint_position_query",
        "reset_index": reset_index,
        "query_index": query_index,
        "state": {"joint_position": state},
        "response_action_chunk": [row + [0.0] for row in response_rows],
    }


def _action(reset_index, query_index, action, chunk_action_index=0):
    return {
        "record_type": "openpi_joint_position_action",
        "reset_index": reset_index,
        "query_index": query_index,
        "chunk_action_index": chunk_action_index,
        "emitted_action": action + [0.0],
    }


def _execution(
    reset_index,
    query_index,
    chunk_action_index,
    outer_step_index,
    action,
    state_after,
):
    return {
        "record_type": "openpi_joint_position_execution",
        "reset_index": reset_index,
        "query_index": query_index,
        "chunk_action_index": chunk_action_index,
        "outer_step_index": outer_step_index,
        "emitted_action": action + [0.0],
        "measured_joint_position_after": state_after,
    }


class JointBoundAuditTest(unittest.TestCase):
    def test_state_and_target_excursions_are_separated(self):
        valid = [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        state_oob = [0.0, 0.0, 0.0, -1.0, 0.0, 4.0, 0.0]
        target_oob = [0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0]
        records = [
            _query(0, 0, valid, [valid] * 14 + [target_oob]),
            _action(0, 0, valid),
            _query(1, 0, valid, [valid] * 15),
            _action(1, 0, target_oob),
            _query(2, 0, valid, [valid] * 15),
            _action(2, 0, valid),
            _query(2, 1, state_oob, [state_oob] * 15),
            _action(2, 1, state_oob),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trace = root / "trace.jsonl"
            metrics = root / "metrics.csv"
            trace.write_text("".join(json.dumps(record) + "\n" for record in records))
            metrics.write_text(
                "episode,episode_length,success,progress,numerical_failure\n"
                "0,1,False,0.0,False\n"
                "1,1,True,1.0,False\n"
                "2,2,True,1.0,True\n"
            )

            summary = MODULE.audit_joint_bounds(trace, metrics)

        self.assertEqual(summary["state_oob_episodes"], [2])
        self.assertEqual(summary["executed_target_only_oob_episodes"], [1])
        self.assertEqual(summary["unexecuted_response_only_oob_episodes"], [0])
        self.assertEqual(summary["state_oob_success_episodes"], [2])
        self.assertEqual(
            summary["state_valid_success_count_counting_invalid_as_failures"], 1
        )
        self.assertTrue(
            summary["first_state_violation"]["2"]["preceding_emitted_targets_in_bounds"]
        )
        self.assertTrue(summary["state_audit_is_lower_bound"])
        self.assertEqual(summary["state_observation_coverage"], "policy_queries_only")

    def test_schema4_execution_states_provide_every_step_coverage(self):
        valid = [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        transient_state_oob = [0.0, 0.0, 0.0, -1.0, 0.0, 4.0, 0.0]
        records = [
            _query(0, 0, valid, [valid] * 15),
            {
                **_action(0, 0, valid),
                "chunk_action_index": 0,
            },
            _execution(0, 0, 0, 0, valid, transient_state_oob),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trace = root / "trace.jsonl"
            metrics = root / "metrics.csv"
            trace.write_text("".join(json.dumps(record) + "\n" for record in records))
            metrics.write_text(
                "episode,episode_length,success,progress,numerical_failure\n"
                "0,1,True,1.0,False\n"
            )

            summary = MODULE.audit_joint_bounds(trace, metrics)

        self.assertFalse(summary["state_audit_is_lower_bound"])
        self.assertEqual(
            summary["state_observation_coverage"],
            "initial_query_plus_post_action_every_step",
        )
        self.assertEqual(summary["execution_record_count"], 1)
        self.assertEqual(summary["state_oob_episodes"], [0])
        self.assertEqual(summary["state_oob_success_episodes"], [0])
        self.assertEqual(
            summary["state_valid_success_count_counting_invalid_as_failures"], 0
        )
        violation = summary["first_state_violation"]["0"]
        self.assertEqual(violation["record_type"], "openpi_joint_position_execution")
        self.assertEqual(violation["outer_step_index"], 0)
        self.assertTrue(violation["preceding_emitted_targets_in_bounds"])

    def test_schema4_action_execution_target_mismatch_is_rejected(self):
        valid = [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        other = [0.1, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        records = [
            _query(0, 0, valid, [valid] * 15),
            {**_action(0, 0, valid), "chunk_action_index": 0},
            _execution(0, 0, 0, 0, other, valid),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trace = root / "trace.jsonl"
            metrics = root / "metrics.csv"
            trace.write_text("".join(json.dumps(record) + "\n" for record in records))
            metrics.write_text(
                "episode,episode_length,success,progress,numerical_failure\n"
                "0,1,False,0.0,False\n"
            )
            with self.assertRaisesRegex(ValueError, "differs from emitted action"):
                MODULE.audit_joint_bounds(trace, metrics)


if __name__ == "__main__":
    unittest.main()
