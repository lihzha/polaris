#!/usr/bin/env python3
"""Aggregate final per-task pi0.5 PolaRiS metrics and physical audits."""

import argparse
import csv
import json
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


def _distribution(values: list[float | int]) -> dict[str, int]:
    counts = Counter(values)
    return {
        f"{value:.12g}": counts[value]
        for value in sorted(counts, key=float)
    }


def summarize(root: Path) -> dict:
    tasks = []
    for task_name in TASK_ORDER:
        task_dir = root / task_name
        with (task_dir / "eval_results.csv").open(
            newline="", encoding="utf-8"
        ) as metrics_file:
            rows = list(csv.DictReader(metrics_file))
        if not rows:
            raise ValueError(f"No metrics for {task_name}")
        for expected_episode, row in enumerate(rows):
            if int(float(row["episode"])) != expected_episode:
                raise ValueError(
                    f"Non-contiguous metrics for {task_name} at {expected_episode}"
                )
        trace_summary = json.loads(
            (task_dir / "policy_trace_summary.json").read_text()
        )
        joint_audit = json.loads((task_dir / "joint_bound_audit.json").read_text())
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
                "trace_emitted_action_records": trace_summary[
                    "emitted_action_records"
                ],
                "trace_sha256": trace_summary["trace_sha256"],
                "state_oob_episode_count_lower_bound": joint_audit[
                    "state_oob_episode_count"
                ],
                "state_oob_episodes_lower_bound": joint_audit[
                    "state_oob_episodes"
                ],
                "state_oob_success_episodes": joint_audit[
                    "state_oob_success_episodes"
                ],
                "state_valid_success_count_counting_invalid_as_failures": joint_audit[
                    "state_valid_success_count_counting_invalid_as_failures"
                ],
                "executed_target_only_oob_episode_count": len(
                    joint_audit["executed_target_only_oob_episodes"]
                ),
            }
        )

    episode_count = sum(task["episodes"] for task in tasks)
    official_successes = sum(task["success_count"] for task in tasks)
    state_valid_successes = sum(
        task["state_valid_success_count_counting_invalid_as_failures"]
        for task in tasks
    )
    state_oob_count = sum(
        task["state_oob_episode_count_lower_bound"] for task in tasks
    )
    numerical_failure_count = sum(
        task["recorded_numerical_failure_count"] for task in tasks
    )
    weighted_progress = sum(
        task["mean_progress"] * task["episodes"] for task in tasks
    )
    return {
        "schema_version": 1,
        "task_count": len(tasks),
        "episode_count": episode_count,
        "official_success_count": official_successes,
        "official_success_rate": official_successes / episode_count,
        "official_mean_progress": weighted_progress / episode_count,
        "recorded_numerical_failure_count": numerical_failure_count,
        "state_oob_episode_count_lower_bound": state_oob_count,
        "state_oob_but_not_recorded_numerical_count_lower_bound": (
            state_oob_count - numerical_failure_count
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
            "State-OOB counts are lower bounds because state is sampled at policy "
            "queries every eight actions. Target-only excursions are reported but do "
            "not reduce the state-valid success numerator."
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
