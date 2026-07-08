import importlib.util
import copy
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts/polaris/audit_pi05_joint_bounds.py"
SPEC = importlib.util.spec_from_file_location("audit_pi05_joint_bounds", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _query(reset_index, query_index, state, response_rows, schema_version=1):
    return {
        "schema_version": schema_version,
        "record_type": "openpi_joint_position_query",
        "reset_index": reset_index,
        "query_index": query_index,
        "state": {"joint_position": state},
        "response_action_chunk": [row + [0.0] for row in response_rows],
    }


def _action(
    reset_index,
    query_index,
    action,
    chunk_action_index=0,
    schema_version=1,
):
    return {
        "schema_version": schema_version,
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
        "schema_version": 4,
        "record_type": "openpi_joint_position_execution",
        "reset_index": reset_index,
        "query_index": query_index,
        "chunk_action_index": chunk_action_index,
        "outer_step_index": outer_step_index,
        "emitted_action": action + [0.0],
        "measured_joint_position_after": state_after,
    }


def _schema4_records(
    episode_length,
    *,
    initial_state=None,
    final_state=None,
):
    valid = [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
    records = []
    for query_index in range((episode_length + 7) // 8):
        state = initial_state if query_index == 0 and initial_state else valid
        records.append(
            _query(
                0,
                query_index,
                state,
                [valid] * 15,
                schema_version=4,
            )
        )
        action_count = min(8, episode_length - query_index * 8)
        for chunk_action_index in range(action_count):
            outer_step_index = query_index * 8 + chunk_action_index
            state_after = (
                final_state
                if outer_step_index == episode_length - 1 and final_state
                else valid
            )
            records.append(
                _action(
                    0,
                    query_index,
                    valid,
                    chunk_action_index,
                    schema_version=4,
                )
            )
            records.append(
                _execution(
                    0,
                    query_index,
                    chunk_action_index,
                    outer_step_index,
                    valid,
                    state_after,
                )
            )
    return records


def _run_audit(records, episode_length=1, success=False, tolerance=1e-3):
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        trace = root / "trace.jsonl"
        metrics = root / "metrics.csv"
        trace.write_text("".join(json.dumps(record) + "\n" for record in records))
        metrics.write_text(
            "episode,episode_length,success,progress,numerical_failure\n"
            f"0,{episode_length},{success},{1.0 if success else 0.0},False\n"
        )
        return MODULE.audit_joint_bounds(trace, metrics, tolerance=tolerance)


class JointBoundAuditTest(unittest.TestCase):
    def test_state_and_target_excursions_are_separated(self):
        valid = [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        state_oob = [0.0, 0.0, 0.0, -1.0, 0.0, 4.0, 0.0]
        target_oob = [0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0]
        records = [
            _query(0, 0, valid, [valid] * 14 + [target_oob]),
            _action(0, 0, valid),
            _query(1, 0, valid, [target_oob] + [valid] * 14),
            _action(1, 0, target_oob),
            _query(2, 0, valid, [valid] * 15),
            *[_action(2, 0, valid, chunk) for chunk in range(8)],
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
                "2,9,True,1.0,True\n"
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
            _query(0, 0, valid, [valid] * 15, schema_version=4),
            {
                **_action(0, 0, valid, schema_version=4),
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
            _query(0, 0, valid, [valid] * 15, schema_version=4),
            {
                **_action(0, 0, valid, schema_version=4),
                "chunk_action_index": 0,
            },
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

    def test_full_450_step_trace_binds_terminal_state_and_hashes(self):
        terminal_oob = [0.0, 0.0, 0.0, -1.0, 0.0, 4.0, 0.0]
        summary = _run_audit(
            _schema4_records(450, final_state=terminal_oob),
            episode_length=450,
            success=True,
        )

        self.assertEqual(summary["trace_schema_version"], 4)
        self.assertEqual(summary["query_record_count"], 57)
        self.assertEqual(summary["action_record_count"], 450)
        self.assertEqual(summary["execution_record_count"], 450)
        self.assertEqual(summary["state_oob_episodes"], [0])
        violation = summary["first_state_violation"]["0"]
        self.assertEqual(violation["outer_step_index"], 449)
        self.assertEqual(violation["query_index"], 56)
        self.assertEqual(violation["chunk_action_index"], 1)
        self.assertEqual(violation["max_abs_query_state"], 1.0)
        self.assertEqual(violation["max_abs_recorded_state"], 4.0)
        self.assertEqual(len(summary["trace_sha256"]), 64)
        self.assertEqual(len(summary["metrics_sha256"]), 64)

    def test_schema4_cannot_downgrade_when_all_executions_are_removed(self):
        records = [
            record
            for record in _schema4_records(1)
            if record["record_type"] != "openpi_joint_position_execution"
        ]
        with self.assertRaisesRegex(ValueError, "requires per-action execution"):
            _run_audit(records)

    def test_actions_cannot_reference_a_removed_query(self):
        records = [
            record
            for record in _schema4_records(9)
            if not (
                record["record_type"] == "openpi_joint_position_query"
                and record["query_index"] == 1
            )
        ]
        with self.assertRaisesRegex(ValueError, "no preceding query"):
            _run_audit(records, episode_length=9)

    def test_execution_steps_cannot_be_swapped(self):
        records = copy.deepcopy(_schema4_records(2))
        executions = [
            record
            for record in records
            if record["record_type"] == "openpi_joint_position_execution"
        ]
        executions[0]["outer_step_index"], executions[1]["outer_step_index"] = (
            executions[1]["outer_step_index"],
            executions[0]["outer_step_index"],
        )
        with self.assertRaisesRegex(ValueError, "Execution step is not contiguous"):
            _run_audit(records, episode_length=2)

    def test_legacy_trace_cannot_omit_actions(self):
        valid = [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        records = [_query(0, 0, valid, [valid] * 15)]
        with self.assertRaisesRegex(ValueError, "Action records"):
            _run_audit(records)

    def test_noncanonical_tolerance_is_rejected(self):
        for tolerance in (-1.0, float("inf"), 100.0):
            with self.subTest(tolerance=tolerance):
                with self.assertRaisesRegex(ValueError, "tolerance exactly"):
                    _run_audit(_schema4_records(1), tolerance=tolerance)

    def test_execution_must_immediately_follow_its_action(self):
        records = _schema4_records(1)
        records[1], records[2] = records[2], records[1]
        with self.assertRaisesRegex(ValueError, "does not follow its action"):
            _run_audit(records)

    def test_query_state_must_equal_preceding_post_action_state(self):
        records = _schema4_records(9)
        query_one = next(
            record
            for record in records
            if record["record_type"] == "openpi_joint_position_query"
            and record["query_index"] == 1
        )
        query_one["state"]["joint_position"] = [0.1, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        with self.assertRaisesRegex(ValueError, "preceding post-action state"):
            _run_audit(records, episode_length=9)

    def test_complete_action_execution_pairs_cannot_be_reordered(self):
        records = _schema4_records(2)
        records[1:5] = records[3:5] + records[1:3]
        with self.assertRaisesRegex(ValueError, "Execution step is not contiguous"):
            _run_audit(records, episode_length=2)

    def test_action_execution_pair_cannot_precede_its_query(self):
        query, action, execution = _schema4_records(1)
        with self.assertRaisesRegex(ValueError, "no preceding query"):
            _run_audit([action, execution, query])


if __name__ == "__main__":
    unittest.main()
