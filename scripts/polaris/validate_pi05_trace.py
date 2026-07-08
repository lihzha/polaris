#!/usr/bin/env python3
"""Fail-closed audit for official PolaRiS pi0.5 joint-position traces."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


ENVIRONMENT_SEED_PROFILE = "isaaclab_env_seed_base_plus_episode_v1"
ENVIRONMENT_SEED_SCHEME = "base_plus_episode_index_v1"
ENVIRONMENT_DETERMINISM_CLAIM = "rng_bound_not_bitwise"
MAX_ENVIRONMENT_SEED = 2**32 - 1


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


def _uint32(value, name: str) -> int:
    _require(
        type(value) is int and 0 <= value <= MAX_ENVIRONMENT_SEED,
        f"{name} must be a uint32 integer",
    )
    return value


def _environment_rng(
    value, *, reset_index: int, expected_base_seed: int | None, prefix: str
) -> dict:
    expected_keys = {
        "schema_version",
        "profile",
        "base_seed",
        "scheme",
        "episode_index",
        "episode_seed",
        "live_cfg_seed",
        "physx_enhanced_determinism",
        "determinism_claim",
    }
    _require(
        isinstance(value, dict) and set(value) == expected_keys,
        f"{prefix}: environment_rng fields are not canonical",
    )
    base_seed = _uint32(value.get("base_seed"), f"{prefix} base seed")
    episode_index = _uint32(
        value.get("episode_index"), f"{prefix} episode index"
    )
    episode_seed = _uint32(value.get("episode_seed"), f"{prefix} episode seed")
    live_cfg_seed = _uint32(value.get("live_cfg_seed"), f"{prefix} live cfg seed")
    _require(value.get("schema_version") == 1, f"{prefix}: RNG schema mismatch")
    _require(
        value.get("profile") == ENVIRONMENT_SEED_PROFILE,
        f"{prefix}: environment seed profile mismatch",
    )
    _require(
        value.get("scheme") == ENVIRONMENT_SEED_SCHEME,
        f"{prefix}: environment seed scheme mismatch",
    )
    _require(
        value.get("determinism_claim") == ENVIRONMENT_DETERMINISM_CLAIM,
        f"{prefix}: environment determinism claim mismatch",
    )
    _require(
        type(value.get("physx_enhanced_determinism")) is bool,
        f"{prefix}: PhysX determinism flag must be boolean",
    )
    _require(
        episode_index == reset_index,
        f"{prefix}: episode index does not match reset index",
    )
    _require(live_cfg_seed == base_seed, f"{prefix}: live cfg seed mismatch")
    _require(
        base_seed + episode_index <= MAX_ENVIRONMENT_SEED
        and episode_seed == base_seed + episode_index,
        f"{prefix}: derived episode seed mismatch",
    )
    if expected_base_seed is not None:
        _require(
            base_seed == expected_base_seed,
            f"{prefix}: expected environment base seed {expected_base_seed}, got {base_seed}",
        )
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
    expected_environment_seed: int | None = None,
) -> dict:
    if expected_environment_seed is not None:
        _uint32(expected_environment_seed, "expected environment base seed")
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
    environment_base_seeds = set()
    environment_seed_schemes = set()
    environment_live_cfg_seeds = set()
    environment_physx_flags = set()
    environment_determinism_claims = set()
    episode_seeds = {}

    for record_index, record in enumerate(records):
        prefix = f"record {record_index}"
        schema_version = record.get("schema_version")
        if expected_environment_seed is None:
            _require(
                schema_version in {1, 2},
                f"{prefix}: schema_version must be 1 or 2",
            )
        else:
            _require(
                schema_version == 2,
                f"{prefix}: seeded trace schema_version must be 2",
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

        if schema_version == 2 or expected_environment_seed is not None:
            rng = _environment_rng(
                record.get("environment_rng"),
                reset_index=reset_index,
                expected_base_seed=expected_environment_seed,
                prefix=prefix,
            )
            environment_base_seeds.add(rng["base_seed"])
            environment_seed_schemes.add(rng["scheme"])
            environment_live_cfg_seeds.add(rng["live_cfg_seed"])
            environment_physx_flags.add(rng["physx_enhanced_determinism"])
            environment_determinism_claims.add(rng["determinism_claim"])
            previous_episode_seed = episode_seeds.setdefault(
                reset_index, rng["episode_seed"]
            )
            _require(
                previous_episode_seed == rng["episode_seed"],
                f"{prefix}: mixed episode seeds within one reset",
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
    if expected_environment_seed is not None:
        _require(
            environment_base_seeds == {expected_environment_seed},
            "Seeded trace did not bind the expected environment base seed",
        )
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
        "environment_base_seed": (
            next(iter(environment_base_seeds))
            if len(environment_base_seeds) == 1
            else None
        ),
        "environment_seed_scheme": (
            next(iter(environment_seed_schemes))
            if len(environment_seed_schemes) == 1
            else None
        ),
        "environment_live_cfg_seed": (
            next(iter(environment_live_cfg_seeds))
            if len(environment_live_cfg_seeds) == 1
            else None
        ),
        "environment_physx_enhanced_determinism": (
            next(iter(environment_physx_flags))
            if len(environment_physx_flags) == 1
            else None
        ),
        "environment_determinism_claim": (
            next(iter(environment_determinism_claims))
            if len(environment_determinism_claims) == 1
            else None
        ),
        "environment_episode_seeds": [
            episode_seeds[index] for index in sorted(episode_seeds)
        ],
        "status": "pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--expected-prompt")
    parser.add_argument("--metrics-csv", type=Path)
    parser.add_argument("--expected-environment-seed", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = audit_trace(
        args.trace,
        expected_prompt=args.expected_prompt,
        metrics_csv=args.metrics_csv,
        expected_environment_seed=args.expected_environment_seed,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
