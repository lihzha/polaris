#!/usr/bin/env python3
"""Audit pi0.5 traces against standard Panda joint-position bounds."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


PANDA_JOINT_LIMITS = (
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
)


def _violating_joints(values: list[float], tolerance: float) -> list[int]:
    if len(values) != len(PANDA_JOINT_LIMITS):
        raise ValueError(f"Expected seven joints, got {len(values)}")
    violating = []
    for index, (value, (lower, upper)) in enumerate(
        zip(values, PANDA_JOINT_LIMITS, strict=True)
    ):
        if not math.isfinite(value):
            raise ValueError("Joint-bound audit requires finite trace values")
        if value < lower - tolerance or value > upper + tolerance:
            violating.append(index)
    return violating


def _load_metrics(metrics_csv: Path) -> list[dict]:
    with metrics_csv.open(newline="", encoding="utf-8") as metrics_file:
        rows = list(csv.DictReader(metrics_file))
    for expected_episode, row in enumerate(rows):
        if int(float(row["episode"])) != expected_episode:
            raise ValueError(f"Metrics are not contiguous at episode {expected_episode}")
    return rows


def audit_joint_bounds(
    trace_path: Path,
    metrics_csv: Path,
    tolerance: float = 1e-3,
) -> dict:
    metrics = _load_metrics(metrics_csv)
    query_states = {}
    response_rows_by_episode = defaultdict(list)
    emitted_rows_by_query = defaultdict(list)
    emitted_rows_by_episode = defaultdict(list)

    with trace_path.open(encoding="utf-8") as trace_file:
        for line_number, line in enumerate(trace_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at line {line_number}: {error}") from error
            reset_index = record.get("reset_index")
            query_index = record.get("query_index")
            if not isinstance(reset_index, int) or not isinstance(query_index, int):
                raise ValueError(f"Invalid trace identity at line {line_number}")
            key = (reset_index, query_index)
            record_type = record.get("record_type")
            if record_type == "openpi_joint_position_query":
                state = [float(value) for value in record["state"]["joint_position"]]
                query_states[key] = state
                for action in record["response_action_chunk"]:
                    response_rows_by_episode[reset_index].append(
                        [float(value) for value in action[:7]]
                    )
            elif record_type == "openpi_joint_position_action":
                action = [float(value) for value in record["emitted_action"][:7]]
                emitted_rows_by_query[key].append(action)
                emitted_rows_by_episode[reset_index].append(action)
            else:
                raise ValueError(f"Unexpected record type at line {line_number}")

    state_oob = set()
    executed_target_oob = set()
    full_response_oob = set()
    first_state_violation = {}
    episode_max_state_abs = defaultdict(float)
    episode_max_executed_target_abs = defaultdict(float)

    for (episode, query_index), state in sorted(query_states.items()):
        episode_max_state_abs[episode] = max(
            episode_max_state_abs[episode], max(abs(value) for value in state)
        )
        violating = _violating_joints(state, tolerance)
        if violating:
            state_oob.add(episode)
            first_state_violation.setdefault(
                episode,
                {
                    "query_index": query_index,
                    "violating_joint_indices": [index + 1 for index in violating],
                },
            )

    for episode, actions in emitted_rows_by_episode.items():
        for action in actions:
            episode_max_executed_target_abs[episode] = max(
                episode_max_executed_target_abs[episode],
                max(abs(value) for value in action),
            )
            if _violating_joints(action, tolerance):
                executed_target_oob.add(episode)

    for episode, actions in response_rows_by_episode.items():
        if any(_violating_joints(action, tolerance) for action in actions):
            full_response_oob.add(episode)

    for episode, detail in first_state_violation.items():
        query_index = detail["query_index"]
        if query_index == 0:
            detail["preceding_emitted_targets_in_bounds"] = None
        else:
            preceding = emitted_rows_by_query[(episode, query_index - 1)]
            detail["preceding_emitted_targets_in_bounds"] = bool(preceding) and all(
                not _violating_joints(action, tolerance) for action in preceding
            )
        detail["max_abs_query_state"] = episode_max_state_abs[episode]
        detail["max_abs_executed_target"] = episode_max_executed_target_abs[episode]
        if episode < len(metrics):
            row = metrics[episode]
            detail["success"] = row["success"] == "True"
            detail["progress"] = float(row["progress"])
            detail["numerical_failure"] = row.get("numerical_failure") == "True"
            detail["episode_length"] = int(float(row["episode_length"]))

    success_episodes = {
        index for index, row in enumerate(metrics) if row["success"] == "True"
    }
    numerical_failure_episodes = {
        index
        for index, row in enumerate(metrics)
        if row.get("numerical_failure") == "True"
    }
    state_invalid_successes = success_episodes & state_oob

    return {
        "schema_version": 1,
        "trace_path": str(trace_path.resolve()),
        "metrics_csv": str(metrics_csv.resolve()),
        "tolerance_radians": tolerance,
        "panda_joint_limits_radians": [list(bounds) for bounds in PANDA_JOINT_LIMITS],
        "episode_count": len(metrics),
        "official_success_episodes": sorted(success_episodes),
        "official_success_count": len(success_episodes),
        "recorded_numerical_failure_episodes": sorted(numerical_failure_episodes),
        "state_oob_episodes": sorted(state_oob),
        "state_oob_episode_count": len(state_oob),
        "state_oob_success_episodes": sorted(state_invalid_successes),
        "state_valid_success_count_counting_invalid_as_failures": len(
            success_episodes - state_oob
        ),
        "executed_target_oob_episodes": sorted(executed_target_oob),
        "executed_target_only_oob_episodes": sorted(executed_target_oob - state_oob),
        "full_response_oob_episodes": sorted(full_response_oob),
        "unexecuted_response_only_oob_episodes": sorted(
            full_response_oob - executed_target_oob
        ),
        "first_state_violation": {
            str(episode): detail
            for episode, detail in sorted(first_state_violation.items())
        },
        "interpretation": (
            "State-OOB episodes are a lower bound because proprioception is recorded "
            "only at policy queries (every eight actions). Target-only excursions are "
            "reported separately and are not automatically classified as invalid states."
        ),
        "status": "pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = audit_joint_bounds(args.trace, args.metrics_csv, args.tolerance)
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
