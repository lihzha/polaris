#!/usr/bin/env python3
"""Audit pi0.5 traces against standard Panda joint-position bounds.

Schema-4 traces contain a measured post-action state for every simulator step.
Older traces only contain policy-query states, so their state audit remains a
lower bound.  The output says explicitly which coverage was available.
"""

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
            raise ValueError(
                f"Metrics are not contiguous at episode {expected_episode}"
            )
    return rows


def audit_joint_bounds(
    trace_path: Path,
    metrics_csv: Path,
    tolerance: float = 1e-3,
) -> dict:
    metrics = _load_metrics(metrics_csv)
    query_states = {}
    execution_states = {}
    response_rows_by_episode = defaultdict(list)
    action_rows = {}
    execution_rows = {}

    with trace_path.open(encoding="utf-8") as trace_file:
        for line_number, line in enumerate(trace_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {error}"
                ) from error
            reset_index = record.get("reset_index")
            query_index = record.get("query_index")
            if type(reset_index) is not int or type(query_index) is not int:
                raise ValueError(f"Invalid trace identity at line {line_number}")
            query_key = (reset_index, query_index)
            record_type = record.get("record_type")
            if record_type == "openpi_joint_position_query":
                if query_key in query_states:
                    raise ValueError(f"Duplicate query record at line {line_number}")
                state = [float(value) for value in record["state"]["joint_position"]]
                _violating_joints(state, tolerance)
                query_states[query_key] = state
                for action in record["response_action_chunk"]:
                    response_rows_by_episode[reset_index].append(
                        [float(value) for value in action[:7]]
                    )
                continue

            if record_type not in {
                "openpi_joint_position_action",
                "openpi_joint_position_execution",
            }:
                raise ValueError(f"Unexpected record type at line {line_number}")
            chunk_action_index = record.get("chunk_action_index")
            if type(chunk_action_index) is not int or not 0 <= chunk_action_index < 8:
                raise ValueError(f"Invalid action identity at line {line_number}")
            action_key = (reset_index, query_index, chunk_action_index)
            action = [float(value) for value in record["emitted_action"][:7]]
            _violating_joints(action, tolerance)
            if record_type == "openpi_joint_position_action":
                if action_key in action_rows:
                    raise ValueError(f"Duplicate action record at line {line_number}")
                action_rows[action_key] = action
            else:
                if action_key in execution_rows:
                    raise ValueError(
                        f"Duplicate action-execution record at line {line_number}"
                    )
                outer_step_index = record.get("outer_step_index")
                if type(outer_step_index) is not int or outer_step_index < 0:
                    raise ValueError(
                        f"Invalid execution step identity at line {line_number}"
                    )
                execution_key = (reset_index, outer_step_index)
                if execution_key in execution_states:
                    raise ValueError(f"Duplicate execution state at line {line_number}")
                state = [
                    float(value) for value in record["measured_joint_position_after"]
                ]
                _violating_joints(state, tolerance)
                execution_rows[action_key] = action
                execution_states[execution_key] = {
                    "state": state,
                    "action_key": action_key,
                    "query_index": query_index,
                    "chunk_action_index": chunk_action_index,
                }

    expected_episodes = set(range(len(metrics)))
    actual_query_episodes = {reset_index for reset_index, _ in query_states}
    if actual_query_episodes != expected_episodes or any(
        (episode, 0) not in query_states for episode in expected_episodes
    ):
        raise ValueError("Trace query episodes do not match metrics episodes")

    if execution_rows:
        if set(action_rows) != set(execution_rows):
            raise ValueError(
                "Schema-4 action and execution records do not have identical keys"
            )
        for key in sorted(action_rows):
            if action_rows[key] != execution_rows[key]:
                raise ValueError(
                    f"Execution target differs from emitted action at {key}"
                )
        actual_execution_episodes = {reset_index for reset_index, _ in execution_states}
        if actual_execution_episodes != expected_episodes:
            raise ValueError(
                "Schema-4 execution episodes do not match metrics episodes"
            )
        for episode, row in enumerate(metrics):
            episode_length = int(float(row["episode_length"]))
            actual_steps = sorted(
                step for reset_index, step in execution_states if reset_index == episode
            )
            if actual_steps != list(range(episode_length)):
                raise ValueError(
                    f"Episode {episode} execution steps are not exactly "
                    f"0..{episode_length - 1}"
                )
        emitted_rows = execution_rows
        state_observation_coverage = "initial_query_plus_post_action_every_step"
        state_audit_is_lower_bound = False
    else:
        emitted_rows = action_rows
        state_observation_coverage = "policy_queries_only"
        state_audit_is_lower_bound = True

    emitted_rows_by_episode = defaultdict(list)
    emitted_rows_by_query = defaultdict(list)
    for (episode, query_index, chunk_action_index), action in emitted_rows.items():
        emitted_rows_by_episode[episode].append(action)
        emitted_rows_by_query[(episode, query_index)].append(
            (chunk_action_index, action)
        )

    state_oob = set()
    executed_target_oob = set()
    full_response_oob = set()
    first_state_violation = {}
    episode_max_state_abs = defaultdict(float)
    episode_max_executed_target_abs = defaultdict(float)

    state_samples = []
    for (episode, query_index), state in query_states.items():
        state_samples.append(
            {
                "episode": episode,
                "time_index": query_index * 8,
                "sample_priority": 1,
                "record_type": "openpi_joint_position_query",
                "query_index": query_index,
                "outer_step_index": None,
                "chunk_action_index": None,
                "state": state,
                "preceding_action_key": None,
            }
        )
    for (episode, outer_step_index), detail in execution_states.items():
        state_samples.append(
            {
                "episode": episode,
                "time_index": outer_step_index + 1,
                "sample_priority": 0,
                "record_type": "openpi_joint_position_execution",
                "query_index": detail["query_index"],
                "outer_step_index": outer_step_index,
                "chunk_action_index": detail["chunk_action_index"],
                "state": detail["state"],
                "preceding_action_key": detail["action_key"],
            }
        )

    for sample in sorted(
        state_samples,
        key=lambda value: (
            value["episode"],
            value["time_index"],
            value["sample_priority"],
        ),
    ):
        episode = sample["episode"]
        state = sample["state"]
        episode_max_state_abs[episode] = max(
            episode_max_state_abs[episode], max(abs(value) for value in state)
        )
        violating = _violating_joints(state, tolerance)
        if violating:
            state_oob.add(episode)
            first_state_violation.setdefault(
                episode,
                {
                    "record_type": sample["record_type"],
                    "query_index": sample["query_index"],
                    "outer_step_index": sample["outer_step_index"],
                    "chunk_action_index": sample["chunk_action_index"],
                    "violating_joint_indices": [index + 1 for index in violating],
                    "preceding_action_key": sample["preceding_action_key"],
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
        preceding_action_key = detail.pop("preceding_action_key")
        if preceding_action_key is not None:
            detail["preceding_emitted_targets_in_bounds"] = not _violating_joints(
                emitted_rows[preceding_action_key], tolerance
            )
        elif query_index == 0:
            detail["preceding_emitted_targets_in_bounds"] = None
        else:
            preceding = [
                action
                for _, action in sorted(
                    emitted_rows_by_query[(episode, query_index - 1)]
                )
            ]
            detail["preceding_emitted_targets_in_bounds"] = bool(preceding) and all(
                not _violating_joints(action, tolerance) for action in preceding
            )
        detail["max_abs_recorded_state"] = episode_max_state_abs[episode]
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
        "state_observation_coverage": state_observation_coverage,
        "state_audit_is_lower_bound": state_audit_is_lower_bound,
        "execution_record_count": len(execution_rows),
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
            "State-OOB episodes are exact over the recorded initial and post-action "
            "states because schema-4 execution records cover every action. Target-only "
            "excursions are reported separately and are not automatically classified "
            "as invalid states."
            if not state_audit_is_lower_bound
            else "State-OOB episodes are a lower bound because this legacy trace records "
            "proprioception only at policy queries. Target-only excursions are reported "
            "separately and are not automatically classified as invalid states."
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
