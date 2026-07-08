#!/usr/bin/env python3
"""Aggregate final per-task pi0.5 PolaRiS metrics and physical audits."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


TASK_ORDER = (
    "BlockStack",
    "FoodBussing",
    "PanClean",
    "MoveLatteCup",
    "OrganizeTools",
    "TapeIntoContainer",
)
PANDA_JOINT_LIMITS = [
    [-2.8973, 2.8973],
    [-1.7628, 1.7628],
    [-2.8973, 2.8973],
    [-3.0718, -0.0698],
    [-2.8973, 2.8973],
    [-0.0175, 3.7525],
    [-2.8973, 2.8973],
]
PANDA_BOUND_TOLERANCE_RADIANS = 1e-3


def _distribution(values: list[float | int]) -> dict[str, int]:
    counts = Counter(values)
    return {f"{value:.12g}": counts[value] for value in sorted(counts, key=float)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_set(value, episode_count: int, field: str) -> set[int]:
    _require(
        isinstance(value, list)
        and all(type(index) is int for index in value)
        and value == sorted(set(value))
        and all(0 <= index < episode_count for index in value),
        f"Invalid {field}",
    )
    return set(value)


def summarize(root: Path) -> dict:
    tasks = []
    for task_name in TASK_ORDER:
        task_dir = root / task_name
        metrics_path = task_dir / "eval_results.csv"
        trace_path = task_dir / "policy_traces.jsonl"
        trace_summary_path = task_dir / "policy_trace_summary.json"
        joint_audit_path = task_dir / "joint_bound_audit.json"
        with metrics_path.open(newline="", encoding="utf-8") as metrics_file:
            rows = list(csv.DictReader(metrics_file))
        if not rows:
            raise ValueError(f"No metrics for {task_name}")
        for expected_episode, row in enumerate(rows):
            if int(float(row["episode"])) != expected_episode:
                raise ValueError(
                    f"Non-contiguous metrics for {task_name} at {expected_episode}"
                )
        trace_summary = json.loads(trace_summary_path.read_text())
        joint_audit = json.loads(joint_audit_path.read_text())
        successes = [
            index for index, row in enumerate(rows) if row["success"] == "True"
        ]
        progress = [float(row["progress"]) for row in rows]
        lengths = [int(float(row["episode_length"])) for row in rows]
        numerical_failures = [
            index
            for index, row in enumerate(rows)
            if row.get("numerical_failure") == "True"
        ]
        episode_count = len(rows)
        success_set = set(successes)
        numerical_failure_set = set(numerical_failures)
        actual_trace_sha256 = _sha256_file(trace_path)
        actual_metrics_sha256 = _sha256_file(metrics_path)
        expected_query_records = sum(math.ceil(length / 8) for length in lengths)
        expected_action_records = sum(lengths)

        _require(trace_summary.get("status") == "pass", "Trace summary did not pass")
        _require(
            trace_summary.get("trace_sha256") == actual_trace_sha256,
            "Trace summary SHA-256 does not match the trace",
        )
        _require(
            trace_summary.get("metrics_sha256") == actual_metrics_sha256,
            "Trace summary SHA-256 does not match metrics",
        )
        _require(
            trace_summary.get("reset_count") == episode_count
            and trace_summary.get("episode_lengths") == lengths
            and trace_summary.get("query_records") == expected_query_records
            and trace_summary.get("emitted_action_records") == expected_action_records,
            "Trace summary counts do not match metrics",
        )

        _require(joint_audit.get("status") == "pass", "Joint audit did not pass")
        _require(
            joint_audit.get("schema_version") == 2
            and joint_audit.get("episode_count") == episode_count,
            "Joint audit schema/count does not match metrics",
        )
        _require(
            joint_audit.get("tolerance_radians") == PANDA_BOUND_TOLERANCE_RADIANS
            and joint_audit.get("panda_joint_limits_radians") == PANDA_JOINT_LIMITS,
            "Joint audit tolerance/limit profile is not canonical",
        )
        _require(
            joint_audit.get("trace_sha256") == actual_trace_sha256
            and joint_audit.get("metrics_sha256") == actual_metrics_sha256,
            "Joint audit hashes do not match trace/metrics",
        )
        _require(
            joint_audit.get("query_record_count") == expected_query_records
            and joint_audit.get("action_record_count") == expected_action_records,
            "Joint audit record counts do not match metrics",
        )
        audit_successes = _episode_set(
            joint_audit.get("official_success_episodes"),
            episode_count,
            "official_success_episodes",
        )
        audit_numerical_failures = _episode_set(
            joint_audit.get("recorded_numerical_failure_episodes"),
            episode_count,
            "recorded_numerical_failure_episodes",
        )
        state_oob = _episode_set(
            joint_audit.get("state_oob_episodes"),
            episode_count,
            "state_oob_episodes",
        )
        state_oob_successes = _episode_set(
            joint_audit.get("state_oob_success_episodes"),
            episode_count,
            "state_oob_success_episodes",
        )
        executed_target_oob = _episode_set(
            joint_audit.get("executed_target_oob_episodes"),
            episode_count,
            "executed_target_oob_episodes",
        )
        executed_target_only_oob = _episode_set(
            joint_audit.get("executed_target_only_oob_episodes"),
            episode_count,
            "executed_target_only_oob_episodes",
        )
        full_response_oob = _episode_set(
            joint_audit.get("full_response_oob_episodes"),
            episode_count,
            "full_response_oob_episodes",
        )
        unexecuted_response_only_oob = _episode_set(
            joint_audit.get("unexecuted_response_only_oob_episodes"),
            episode_count,
            "unexecuted_response_only_oob_episodes",
        )
        _require(
            audit_successes == success_set
            and joint_audit.get("official_success_count") == len(success_set)
            and audit_numerical_failures == numerical_failure_set,
            "Joint audit metric episode sets do not match CSV",
        )
        _require(
            joint_audit.get("state_oob_episode_count") == len(state_oob)
            and state_oob_successes == state_oob & success_set
            and joint_audit.get(
                "state_valid_success_count_counting_invalid_as_failures"
            )
            == len(success_set - state_oob),
            "Joint audit state-valid counts are inconsistent",
        )
        _require(
            executed_target_oob <= full_response_oob
            and executed_target_only_oob == executed_target_oob - state_oob
            and unexecuted_response_only_oob == full_response_oob - executed_target_oob,
            "Joint audit target episode sets are inconsistent",
        )
        state_audit_is_lower_bound = joint_audit.get("state_audit_is_lower_bound")
        _require(
            type(state_audit_is_lower_bound) is bool,
            "Joint audit lower-bound flag is invalid",
        )
        if state_audit_is_lower_bound:
            _require(
                joint_audit.get("state_observation_coverage") == "policy_queries_only"
                and joint_audit.get("execution_record_count") == 0,
                "Legacy joint audit coverage is inconsistent",
            )
            _require(
                joint_audit.get("trace_schema_version") in {1, 2, 3}
                and trace_summary.get("schema_version") != 4,
                "Schema-4 evidence cannot be summarized as a legacy lower bound",
            )
        else:
            _require(
                joint_audit.get("trace_schema_version") == 4
                and trace_summary.get("schema_version") == 4
                and joint_audit.get("state_observation_coverage")
                == "initial_query_plus_post_action_every_step"
                and joint_audit.get("execution_record_count") == expected_action_records
                and trace_summary.get("execution_records") == expected_action_records,
                "Schema-4 joint audit coverage is incomplete",
            )
        tasks.append(
            {
                "task": task_name,
                "episodes": len(rows),
                "success_count": len(successes),
                "success_rate": len(successes) / len(rows),
                "success_episodes": successes,
                "mean_progress": sum(progress) / len(progress),
                "progress_distribution": _distribution(progress),
                "episode_length_distribution": _distribution(lengths),
                "recorded_numerical_failure_count": len(numerical_failures),
                "recorded_numerical_failure_episodes": numerical_failures,
                "trace_status": trace_summary["status"],
                "trace_reset_count": trace_summary["reset_count"],
                "trace_query_records": trace_summary["query_records"],
                "trace_emitted_action_records": trace_summary["emitted_action_records"],
                "trace_sha256": trace_summary["trace_sha256"],
                "metrics_sha256": actual_metrics_sha256,
                "joint_audit_sha256": _sha256_file(joint_audit_path),
                "state_audit_is_lower_bound": state_audit_is_lower_bound,
                "state_observation_coverage": joint_audit["state_observation_coverage"],
                "state_oob_episode_count_lower_bound": joint_audit[
                    "state_oob_episode_count"
                ],
                "state_oob_episodes_lower_bound": joint_audit["state_oob_episodes"],
                "state_oob_success_episodes": joint_audit["state_oob_success_episodes"],
                "state_valid_success_count_counting_invalid_as_failures": joint_audit[
                    "state_valid_success_count_counting_invalid_as_failures"
                ],
                "executed_target_only_oob_episode_count": len(
                    joint_audit["executed_target_only_oob_episodes"]
                ),
                "state_oob_but_not_recorded_numerical_count": len(
                    state_oob - numerical_failure_set
                ),
            }
        )

    episode_count = sum(task["episodes"] for task in tasks)
    official_successes = sum(task["success_count"] for task in tasks)
    state_valid_successes = sum(
        task["state_valid_success_count_counting_invalid_as_failures"] for task in tasks
    )
    state_oob_count = sum(task["state_oob_episode_count_lower_bound"] for task in tasks)
    numerical_failure_count = sum(
        task["recorded_numerical_failure_count"] for task in tasks
    )
    weighted_progress = sum(task["mean_progress"] * task["episodes"] for task in tasks)
    state_audit_is_lower_bound = any(
        task["state_audit_is_lower_bound"] for task in tasks
    )
    state_observation_coverage = sorted(
        {task["state_observation_coverage"] for task in tasks}
    )
    return {
        "schema_version": 2,
        "task_count": len(tasks),
        "episode_count": episode_count,
        "official_success_count": official_successes,
        "official_success_rate": official_successes / episode_count,
        "official_mean_progress": weighted_progress / episode_count,
        "recorded_numerical_failure_count": numerical_failure_count,
        "state_audit_is_lower_bound": state_audit_is_lower_bound,
        "state_observation_coverage": state_observation_coverage,
        "state_oob_episode_count_lower_bound": state_oob_count,
        "state_oob_but_not_recorded_numerical_count_lower_bound": (
            sum(task["state_oob_but_not_recorded_numerical_count"] for task in tasks)
        ),
        "state_valid_success_count_counting_invalid_as_failures": state_valid_successes,
        "state_valid_success_rate_counting_invalid_as_failures": (
            state_valid_successes / episode_count
        ),
        "all_trace_validators_passed": all(
            task["trace_status"] == "pass" for task in tasks
        ),
        "tasks": tasks,
        "physical_audit_note": (
            "At least one task only has policy-query state samples, so aggregate "
            "State-OOB counts remain lower bounds. Target-only excursions are reported "
            "but do not reduce the state-valid success numerator."
            if state_audit_is_lower_bound
            else "Schema-4 execution records cover the initial state and every "
            "post-action state, so State-OOB counts are exact over recorded rollout "
            "states. Target-only excursions are reported but do not reduce the "
            "state-valid success numerator."
        ),
    }


def _write_tsv(path: Path, summary: dict) -> None:
    fields = (
        "task",
        "episodes",
        "success_count",
        "success_rate",
        "mean_progress",
        "recorded_numerical_failure_count",
        "state_audit_is_lower_bound",
        "state_observation_coverage",
        "state_oob_episode_count_lower_bound",
        "state_valid_success_count_counting_invalid_as_failures",
        "trace_sha256",
    )
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for task in summary["tasks"]:
            writer.writerow({field: task[field] for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tsv", type=Path)
    args = parser.parse_args()
    summary = summarize(args.root)
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    if args.tsv is not None:
        args.tsv.parent.mkdir(parents=True, exist_ok=True)
        _write_tsv(args.tsv, summary)
    print(rendered, end="")


if __name__ == "__main__":
    main()
