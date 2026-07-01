#!/usr/bin/env python3
"""Fail-closed audit for official PolaRiS pi0.5 joint-position traces."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_vector(value, length: int, name: str) -> list[float]:
    _require(
        isinstance(value, list) and len(value) == length,
        f"{name} must have {length} values",
    )
    _require(
        all(isinstance(item, int | float) and math.isfinite(item) for item in value),
        f"{name} contains a non-finite or non-numeric value",
    )
    return value


def _finite_matrix(value, rows: int, columns: int, name: str) -> list[list[float]]:
    _require(
        isinstance(value, list) and len(value) == rows,
        f"{name} must have {rows} rows",
    )
    for row_index, row in enumerate(value):
        _finite_vector(row, columns, f"{name}[{row_index}]")
    return value


def _episode_lengths(metrics_csv: Path) -> list[int]:
    with metrics_csv.open(newline="") as metrics_file:
        rows = list(csv.DictReader(metrics_file))
    _require(rows, f"Metrics contain no episodes: {metrics_csv}")
    lengths = []
    for expected_episode, row in enumerate(rows):
        _require(
            "episode" in row and "episode_length" in row,
            "Metrics are missing episode fields",
        )
        episode = int(float(row["episode"]))
        episode_length = int(float(row["episode_length"]))
        _require(
            episode == expected_episode,
            f"Metrics episode order mismatch at row {expected_episode}",
        )
        _require(
            episode_length > 0,
            f"Episode {episode} has non-positive length {episode_length}",
        )
        lengths.append(episode_length)
    return lengths


def audit_trace(
    trace_path: Path,
    expected_prompt: str | None = None,
    metrics_csv: Path | None = None,
) -> dict:
    records = []
    digest = hashlib.sha256()
    with trace_path.open("rb") as trace_file:
        for line_number, raw_line in enumerate(trace_file, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                records.append(json.loads(raw_line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {error}"
                ) from error

    _require(records, f"Trace contains no records: {trace_path}")
    queries = {}
    emissions = {}
    prompts = set()
    external_hashes = set()
    wrist_hashes = set()
    raw_joint_min = [math.inf] * 7
    raw_joint_max = [-math.inf] * 7

    for record_index, record in enumerate(records):
        prefix = f"record {record_index}"
        _require(
            record.get("schema_version") == 1, f"{prefix}: schema_version must be 1"
        )
        reset_index = record.get("reset_index")
        query_index = record.get("query_index")
        _require(
            isinstance(reset_index, int) and reset_index >= 0,
            f"{prefix}: reset_index must be a non-negative integer",
        )
        _require(
            isinstance(query_index, int) and query_index >= 0,
            f"{prefix}: query_index must be a non-negative integer",
        )
        query_key = (reset_index, query_index)
        record_type = record.get("record_type")

        if record_type == "openpi_joint_position_action":
            _require(
                query_key in queries,
                f"{prefix}: emitted action appears before its query {query_key}",
            )
            action_index = record.get("chunk_action_index")
            _require(
                isinstance(action_index, int) and 0 <= action_index < 8,
                f"{prefix}: chunk_action_index must be in [0, 7]",
            )
            emission_key = (*query_key, action_index)
            _require(
                emission_key not in emissions,
                f"{prefix}: duplicate emitted action {emission_key}",
            )
            raw_action = _finite_vector(
                record.get("raw_action"), 8, f"{prefix} raw action"
            )
            emitted_action = _finite_vector(
                record.get("emitted_action"), 8, f"{prefix} emitted action"
            )
            _require(
                emitted_action[:7] == raw_action[:7],
                f"{prefix}: emitted joint target differs from the server response",
            )
            _require(
                emitted_action[7] in (0, 1), f"{prefix}: emitted gripper is not binary"
            )
            _require(
                emitted_action[7] == int(raw_action[7] > 0.5),
                f"{prefix}: emitted gripper threshold mismatch",
            )
            emissions[emission_key] = record
            continue

        _require(
            record_type == "openpi_joint_position_query",
            f"{prefix}: unexpected record_type {record_type!r}",
        )
        _require(query_key not in queries, f"{prefix}: duplicate query {query_key}")
        _require(
            record.get("response_action_shape") == [15, 8],
            f"{prefix}: expected 15x8 response",
        )
        _require(
            record.get("execution_horizon") == 8,
            f"{prefix}: execution horizon must be 8",
        )

        state = record.get("state", {})
        _finite_vector(state.get("joint_position"), 7, f"{prefix} joint state")
        _finite_vector(state.get("gripper_position"), 1, f"{prefix} gripper state")

        images = record.get("images", {})
        _require(
            images.get("wrist_rotation_degrees") == 0,
            f"{prefix}: wrist must be unrotated",
        )
        for image_name, hashes in (
            ("external", external_hashes),
            ("wrist", wrist_hashes),
        ):
            image = images.get(image_name, {})
            _require(
                image.get("shape") == [224, 224, 3],
                f"{prefix}: {image_name} must be 224x224 RGB",
            )
            _require(
                image.get("dtype") == "uint8", f"{prefix}: {image_name} must be uint8"
            )
            image_hash = image.get("sha256")
            _require(
                isinstance(image_hash, str)
                and len(image_hash) == 64
                and all(character in "0123456789abcdef" for character in image_hash),
                f"{prefix}: invalid {image_name} SHA-256",
            )
            hashes.add(image_hash)

        raw_actions = _finite_matrix(
            record.get("response_action_chunk"), 15, 8, f"{prefix} response actions"
        )
        planned_actions = _finite_matrix(
            record.get("planned_action_chunk"), 8, 8, f"{prefix} planned actions"
        )
        for action_index, planned_action in enumerate(planned_actions):
            raw_action = raw_actions[action_index]
            _require(
                planned_action[:7] == raw_action[:7],
                f"{prefix}: planned joint target {action_index} differs from server response",
            )
            _require(
                planned_action[7] in (0, 1), f"{prefix}: planned gripper is not binary"
            )
            _require(
                planned_action[7] == int(raw_action[7] > 0.5),
                f"{prefix}: planned gripper threshold mismatch",
            )
        for action in raw_actions:
            for dimension in range(7):
                raw_joint_min[dimension] = min(
                    raw_joint_min[dimension], action[dimension]
                )
                raw_joint_max[dimension] = max(
                    raw_joint_max[dimension], action[dimension]
                )

        prompt = record.get("prompt")
        _require(
            isinstance(prompt, str) and prompt,
            f"{prefix}: prompt must be a nonempty string",
        )
        prompts.add(prompt)
        queries[query_key] = record

    _require(queries, "Trace contains no query records")
    if expected_prompt is not None:
        _require(
            prompts == {expected_prompt},
            f"Prompt mismatch: expected {expected_prompt!r}, got {sorted(prompts)!r}",
        )

    queries_by_reset = defaultdict(list)
    emissions_by_query = defaultdict(list)
    emissions_by_reset = defaultdict(int)
    for reset_index, query_index in queries:
        queries_by_reset[reset_index].append(query_index)
    for reset_index, query_index, action_index in emissions:
        _require(
            (reset_index, query_index) in queries,
            f"Emitted action {(reset_index, query_index, action_index)} has no query",
        )
        query = queries[(reset_index, query_index)]
        raw_action = query["response_action_chunk"][action_index]
        emission = emissions[(reset_index, query_index, action_index)]
        _require(
            emission["raw_action"] == raw_action,
            "Emitted raw action does not match its query",
        )
        emissions_by_query[(reset_index, query_index)].append(action_index)
        emissions_by_reset[reset_index] += 1

    for reset_index, query_indices in queries_by_reset.items():
        query_indices.sort()
        _require(
            query_indices == list(range(len(query_indices))),
            f"Reset {reset_index} query indices are not contiguous from zero",
        )
        for query_index in query_indices:
            action_indices = sorted(emissions_by_query[(reset_index, query_index)])
            _require(
                action_indices, f"Query {(reset_index, query_index)} emitted no actions"
            )
            _require(
                action_indices == list(range(len(action_indices))),
                f"Query {(reset_index, query_index)} action indices are not contiguous from zero",
            )
            if query_index != query_indices[-1]:
                _require(
                    len(action_indices) == 8,
                    f"Non-final query {(reset_index, query_index)} did not emit all 8 actions",
                )

    episode_lengths = None
    if metrics_csv is not None:
        episode_lengths = _episode_lengths(metrics_csv)
        expected_resets = set(range(len(episode_lengths)))
        _require(
            set(queries_by_reset) == expected_resets,
            f"Trace resets {sorted(queries_by_reset)} do not match metrics episodes {sorted(expected_resets)}",
        )
        for reset_index, episode_length in enumerate(episode_lengths):
            _require(
                emissions_by_reset[reset_index] == episode_length,
                f"Episode {reset_index} emitted {emissions_by_reset[reset_index]} actions but metrics report {episode_length}",
            )
            expected_queries = math.ceil(episode_length / 8)
            _require(
                len(queries_by_reset[reset_index]) == expected_queries,
                f"Episode {reset_index} has {len(queries_by_reset[reset_index])} queries; expected {expected_queries}",
            )

    return {
        "schema_version": 2,
        "trace_path": str(trace_path.resolve()),
        "trace_sha256": digest.hexdigest(),
        "query_records": len(queries),
        "emitted_action_records": len(emissions),
        "reset_count": len(queries_by_reset),
        "episode_lengths": episode_lengths,
        "prompts": sorted(prompts),
        "response_action_shape": [15, 8],
        "execution_horizon": 8,
        "image_shape": [224, 224, 3],
        "wrist_rotation_degrees": 0,
        "distinct_external_frames": len(external_hashes),
        "distinct_wrist_frames": len(wrist_hashes),
        "raw_joint_target_min": raw_joint_min,
        "raw_joint_target_max": raw_joint_max,
        "status": "pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--expected-prompt")
    parser.add_argument("--metrics-csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = audit_trace(
        args.trace,
        expected_prompt=args.expected_prompt,
        metrics_csv=args.metrics_csv,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
