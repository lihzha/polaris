import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts/polaris/summarize_pi05_eval.py"
SPEC = importlib.util.spec_from_file_location("summarize_pi05_eval", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_task(root, task_name, *, success=True, numerical=False, state_oob=False):
    task_dir = root / task_name
    task_dir.mkdir(parents=True)
    metrics = task_dir / "eval_results.csv"
    trace = task_dir / "policy_traces.jsonl"
    metrics.write_text(
        "episode,episode_length,success,progress,numerical_failure\n"
        f"0,1,{success},{1.0 if success else 0.0},{numerical}\n"
    )
    trace.write_text(json.dumps({"task": task_name}) + "\n")
    trace_sha256 = _sha256(trace)
    metrics_sha256 = _sha256(metrics)
    (task_dir / "policy_trace_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "status": "pass",
                "trace_sha256": trace_sha256,
                "metrics_sha256": metrics_sha256,
                "reset_count": 1,
                "episode_lengths": [1],
                "query_records": 1,
                "emitted_action_records": 1,
                "execution_records": 1,
            }
        )
        + "\n"
    )
    success_episodes = [0] if success else []
    numerical_episodes = [0] if numerical else []
    state_oob_episodes = [0] if state_oob else []
    state_oob_successes = [0] if state_oob and success else []
    (task_dir / "joint_bound_audit.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "pass",
                "episode_count": 1,
                "tolerance_radians": MODULE.PANDA_BOUND_TOLERANCE_RADIANS,
                "panda_joint_limits_radians": MODULE.PANDA_JOINT_LIMITS,
                "trace_schema_version": 4,
                "trace_sha256": trace_sha256,
                "metrics_sha256": metrics_sha256,
                "query_record_count": 1,
                "action_record_count": 1,
                "execution_record_count": 1,
                "official_success_episodes": success_episodes,
                "official_success_count": len(success_episodes),
                "recorded_numerical_failure_episodes": numerical_episodes,
                "state_audit_is_lower_bound": False,
                "state_observation_coverage": (
                    "initial_query_plus_post_action_every_step"
                ),
                "state_oob_episodes": state_oob_episodes,
                "state_oob_episode_count": len(state_oob_episodes),
                "state_oob_success_episodes": state_oob_successes,
                "state_valid_success_count_counting_invalid_as_failures": len(
                    set(success_episodes) - set(state_oob_episodes)
                ),
                "executed_target_oob_episodes": [],
                "executed_target_only_oob_episodes": [],
                "full_response_oob_episodes": [],
                "unexecuted_response_only_oob_episodes": [],
            }
        )
        + "\n"
    )
    return task_dir


def _complete_root(root, *, first_numerical=False, first_state_oob=False):
    for index, task_name in enumerate(MODULE.TASK_ORDER):
        _write_task(
            root,
            task_name,
            numerical=index == 0 and first_numerical,
            state_oob=index == 0 and first_state_oob,
        )


class SummarizePi05EvalTest(unittest.TestCase):
    def test_valid_schema4_inputs_are_hash_and_count_bound(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _complete_root(root, first_numerical=True, first_state_oob=True)
            summary = MODULE.summarize(root)

        self.assertEqual(summary["schema_version"], 2)
        self.assertEqual(summary["episode_count"], 6)
        self.assertEqual(summary["official_success_count"], 6)
        self.assertEqual(
            summary["state_valid_success_count_counting_invalid_as_failures"], 5
        )
        self.assertEqual(
            summary["state_oob_but_not_recorded_numerical_count_lower_bound"], 0
        )
        self.assertFalse(summary["state_audit_is_lower_bound"])
        self.assertTrue(summary["all_trace_validators_passed"])

    def test_failed_joint_audit_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _complete_root(root)
            path = root / MODULE.TASK_ORDER[0] / "joint_bound_audit.json"
            value = json.loads(path.read_text())
            value["status"] = "failed"
            path.write_text(json.dumps(value) + "\n")
            with self.assertRaisesRegex(ValueError, "Joint audit did not pass"):
                MODULE.summarize(root)

    def test_stale_joint_audit_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _complete_root(root)
            path = root / MODULE.TASK_ORDER[0] / "joint_bound_audit.json"
            value = json.loads(path.read_text())
            value["metrics_sha256"] = "0" * 64
            path.write_text(json.dumps(value) + "\n")
            with self.assertRaisesRegex(ValueError, "hashes do not match"):
                MODULE.summarize(root)

    def test_impossible_state_valid_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _complete_root(root)
            path = root / MODULE.TASK_ORDER[0] / "joint_bound_audit.json"
            value = json.loads(path.read_text())
            value["state_valid_success_count_counting_invalid_as_failures"] = 2
            path.write_text(json.dumps(value) + "\n")
            with self.assertRaisesRegex(ValueError, "state-valid counts"):
                MODULE.summarize(root)

    def test_failed_or_stale_trace_summary_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _complete_root(root)
            path = root / MODULE.TASK_ORDER[0] / "policy_trace_summary.json"
            value = json.loads(path.read_text())
            value["trace_sha256"] = "f" * 64
            path.write_text(json.dumps(value) + "\n")
            with self.assertRaisesRegex(ValueError, "does not match the trace"):
                MODULE.summarize(root)

    def test_schema4_audit_cannot_claim_legacy_coverage(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _complete_root(root)
            path = root / MODULE.TASK_ORDER[0] / "joint_bound_audit.json"
            value = json.loads(path.read_text())
            value["state_audit_is_lower_bound"] = True
            value["state_observation_coverage"] = "policy_queries_only"
            value["execution_record_count"] = 0
            path.write_text(json.dumps(value) + "\n")
            with self.assertRaisesRegex(ValueError, "cannot be summarized"):
                MODULE.summarize(root)

    def test_noncanonical_tolerance_or_limits_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _complete_root(root)
            path = root / MODULE.TASK_ORDER[0] / "joint_bound_audit.json"
            value = json.loads(path.read_text())
            value["tolerance_radians"] = 100.0
            path.write_text(json.dumps(value) + "\n")
            with self.assertRaisesRegex(ValueError, "not canonical"):
                MODULE.summarize(root)

    def test_csv_change_cannot_bypass_prior_trace_validation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _complete_root(root)
            task_dir = root / MODULE.TASK_ORDER[0]
            metrics = task_dir / "eval_results.csv"
            metrics.write_text(
                "episode,episode_length,success,progress,numerical_failure\n"
                "0,1,False,0.0,False\n"
            )
            audit_path = task_dir / "joint_bound_audit.json"
            audit = json.loads(audit_path.read_text())
            audit["metrics_sha256"] = _sha256(metrics)
            audit["official_success_episodes"] = []
            audit["official_success_count"] = 0
            audit["state_oob_success_episodes"] = []
            audit["state_valid_success_count_counting_invalid_as_failures"] = 0
            audit_path.write_text(json.dumps(audit) + "\n")
            with self.assertRaisesRegex(ValueError, "does not match metrics"):
                MODULE.summarize(root)


if __name__ == "__main__":
    unittest.main()
