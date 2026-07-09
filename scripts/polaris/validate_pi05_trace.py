#!/usr/bin/env python3
"""Fail-closed audit for official PolaRiS pi0.5 joint-position traces."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from polaris.pi05_droid_jointpos_serving_contract import (
    PI05_DROID_JOINTPOS_SERVER_MODEL_RESIZE,
)
from polaris.pi05_droid_jointpos_runtime import (
    PI05_DROID_JOINTPOS_PROFILE,
    PI05_DROID_JOINTPOS_TRACE_SCHEMA_VERSION,
)


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
        all(
            isinstance(item, int | float)
            and not isinstance(item, bool)
            and math.isfinite(item)
            for item in value
        ),
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
    value,
    *,
    reset_index: int,
    expected_base_seed: int | None,
    prefix: str,
    require_contract_hash: bool = False,
) -> dict:
    legacy_keys = {
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
    expected_keys = set(legacy_keys)
    if require_contract_hash:
        expected_keys.add("environment_seed_contract_sha256")
    _require(
        isinstance(value, dict) and set(value) == expected_keys,
        f"{prefix}: environment_rng fields are not canonical",
    )
    base_seed = _uint32(value.get("base_seed"), f"{prefix} base seed")
    episode_index = _uint32(value.get("episode_index"), f"{prefix} episode index")
    episode_seed = _uint32(value.get("episode_seed"), f"{prefix} episode seed")
    live_cfg_seed = _uint32(value.get("live_cfg_seed"), f"{prefix} live cfg seed")
    _require(
        value.get("schema_version") == (2 if require_contract_hash else 1),
        f"{prefix}: RNG schema mismatch",
    )
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
    if require_contract_hash:
        seed_contract = {
            "schema_version": 1,
            "profile": value["profile"],
            "base_seed": base_seed,
            "scheme": value["scheme"],
            "live_cfg_seed": live_cfg_seed,
            "physx_enhanced_determinism": value["physx_enhanced_determinism"],
            "determinism_claim": value["determinism_claim"],
            "binding": "env_cfg_seed_before_gym_make_and_reset_seed_per_episode",
        }
        expected_digest = hashlib.sha256(
            json.dumps(
                seed_contract,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()
        _require(
            value["environment_seed_contract_sha256"] == expected_digest,
            f"{prefix}: environment seed contract SHA-256 mismatch",
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_legacy_trace(
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
        "metrics_sha256": (
            _sha256_file(metrics_csv) if metrics_csv is not None else None
        ),
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


def _hex_digest(value, name: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be a lowercase SHA-256",
    )
    return value


def _sensor_counters(value, prefix: str) -> dict[str, int]:
    _require(
        isinstance(value, dict)
        and set(value) == {"external_cam", "wrist_cam"}
        and all(type(counter) is int and counter >= 0 for counter in value.values()),
        f"{prefix}: sensor frame counters are invalid",
    )
    return value


def _environment_state(value, prefix: str) -> dict:
    expected_keys = {
        "boundary_profile",
        "live_max_episode_length",
        "episode_length",
        "sim_step_counter",
        "common_step_counter",
        "sensor_frame_counters",
    }
    _require(
        isinstance(value, dict) and set(value) == expected_keys,
        f"{prefix}: environment state schema mismatch",
    )
    _require(
        value["boundary_profile"] == "outer450_internal451_no_autoreset",
        f"{prefix}: environment boundary profile mismatch",
    )
    _require(
        value["live_max_episode_length"] == 451,
        f"{prefix}: live max episode length must be 451",
    )
    for field in ("episode_length", "sim_step_counter", "common_step_counter"):
        _require(
            type(value[field]) is int and value[field] >= 0,
            f"{prefix}: {field} must be a nonnegative integer",
        )
    _sensor_counters(value["sensor_frame_counters"], prefix)
    return value


def _image_identity(value, shape: list[int], prefix: str) -> dict:
    _require(
        isinstance(value, dict) and set(value) == {"shape", "dtype", "sha256"},
        f"{prefix}: image identity schema mismatch",
    )
    _require(
        value["shape"] == shape and value["dtype"] == "uint8",
        f"{prefix}: image shape/dtype mismatch",
    )
    _hex_digest(value["sha256"], f"{prefix} image")
    return value


def _terminal_image_identity(value, prefix: str) -> dict:
    _require(
        isinstance(value, dict)
        and set(value) == {"shape", "dtype", "sha256", "source"},
        f"{prefix}: terminal visualization schema mismatch",
    )
    _require(
        value["shape"] == [224, 448, 3]
        and value["dtype"] == "uint8"
        and value["source"] == "post_action450_returned_expensive_splat_observation",
        f"{prefix}: terminal visualization identity mismatch",
    )
    _hex_digest(value["sha256"], f"{prefix} terminal visualization")
    return value


def _trace_identity(record: dict, prefix: str) -> tuple[str, str, str, bool]:
    _require(
        record.get("schema_version") == PI05_DROID_JOINTPOS_TRACE_SCHEMA_VERSION,
        f"{prefix}: attested trace schema_version must be "
        f"{PI05_DROID_JOINTPOS_TRACE_SCHEMA_VERSION}",
    )
    _require(
        record.get("profile") == PI05_DROID_JOINTPOS_PROFILE,
        f"{prefix}: joint-position profile mismatch",
    )
    server = _hex_digest(
        record.get("server_contract_sha256"), f"{prefix} server contract"
    )
    environment = _hex_digest(
        record.get("environment_seed_contract_sha256"),
        f"{prefix} environment seed contract",
    )
    runtime = _hex_digest(
        record.get("runtime_contract_sha256"), f"{prefix} runtime contract"
    )
    physx = record.get("physx_enhanced_determinism")
    _require(type(physx) is bool, f"{prefix}: PhysX flag must be boolean")
    return server, environment, runtime, physx


def _validate_execution_report(value, emitted_action, prefix: str) -> dict:
    expected_keys = {
        "schema_version",
        "processing",
        "raw_action_buffer",
        "processed_action_buffer",
        "apply_target_holds",
        "apply_target_hold_count",
        "post_step_articulation_target",
    }
    _require(
        isinstance(value, dict) and set(value) == expected_keys,
        f"{prefix}: action execution schema mismatch",
    )
    _require(value["schema_version"] == 1, f"{prefix}: execution version mismatch")
    _require(
        value["processing"] == "upstream_joint_position_action_scale1_offset0_no_clip",
        f"{prefix}: upstream processing identity mismatch",
    )
    expected = np.asarray(emitted_action[:7], dtype=np.float32)
    raw = np.asarray(
        _finite_vector(value["raw_action_buffer"], 7, f"{prefix} raw buffer"),
        dtype=np.float32,
    )
    processed = np.asarray(
        _finite_vector(
            value["processed_action_buffer"], 7, f"{prefix} processed buffer"
        ),
        dtype=np.float32,
    )
    post = np.asarray(
        _finite_vector(
            value["post_step_articulation_target"], 7, f"{prefix} post target"
        ),
        dtype=np.float32,
    )
    holds = _finite_matrix(value["apply_target_holds"], 8, 7, f"{prefix} target holds")
    _require(
        value["apply_target_hold_count"] == 8,
        f"{prefix}: target hold count must be eight",
    )
    _require(
        np.array_equal(raw, expected)
        and np.array_equal(processed, expected)
        and np.array_equal(post, expected)
        and all(
            np.array_equal(np.asarray(hold, dtype=np.float32), expected)
            for hold in holds
        ),
        f"{prefix}: live target evidence differs from emitted action",
    )
    return value


def _audit_attested_trace(
    records: list[dict],
    *,
    trace_path: Path,
    digest: str,
    expected_prompt: str | None,
    metrics_csv: Path | None,
    expected_environment_seed: int,
    expected_server_contract_sha256: str | None,
    runtime_contract_path: Path | None,
) -> dict:
    _uint32(expected_environment_seed, "expected environment base seed")
    common = {
        "schema_version",
        "profile",
        "record_type",
        "reset_index",
        "query_index",
        "global_query_index",
        "server_contract_sha256",
        "environment_seed_contract_sha256",
        "runtime_contract_sha256",
        "physx_enhanced_determinism",
    }
    query_fields = common | {
        "environment_rng",
        "sensor_frame_counters",
        "prompt",
        "state",
        "images",
        "response_action_shape",
        "response_action_dtype",
        "response_action_chunk",
        "execution_horizon",
        "planned_action_chunk",
    }
    action_fields = common | {"chunk_action_index", "raw_action", "emitted_action"}
    execution_fields = common | {
        "chunk_action_index",
        "outer_step_index",
        "emitted_action",
        "action_execution",
        "processed_finger_position_target",
        "articulation_finger_position_target",
        "measured_joint_position_after",
        "measured_closed_positive_gripper_after",
        "environment_before",
        "environment_after",
        "terminated",
        "truncated",
        "terminal_rubric",
        "terminal_visualization",
    }
    identities = set()
    queries = {}
    actions = {}
    executions = {}
    prompts = set()
    episode_seeds = {}
    global_queries = []
    external_hashes = set()
    wrist_hashes = set()
    raw_joint_min = [math.inf] * 7
    raw_joint_max = [-math.inf] * 7
    previous_environment_by_reset = {}
    expected_outer_step = defaultdict(int)
    terminal_visualizations = {}
    pending_action = None

    for record_index, record in enumerate(records):
        prefix = f"record {record_index}"
        _require(isinstance(record, dict), f"{prefix}: trace record must be an object")
        identity = _trace_identity(record, prefix)
        identities.add(identity)
        reset_index = record.get("reset_index")
        query_index = record.get("query_index")
        global_query_index = record.get("global_query_index")
        _require(
            type(reset_index) is int and reset_index >= 0,
            f"{prefix}: reset index is invalid",
        )
        _require(
            type(query_index) is int and query_index >= 0,
            f"{prefix}: query index is invalid",
        )
        _require(
            type(global_query_index) is int and global_query_index >= 0,
            f"{prefix}: global query index is invalid",
        )
        record_type = record.get("record_type")
        query_key = (reset_index, query_index)
        if record_type == "openpi_joint_position_query":
            _require(set(record) == query_fields, f"{prefix}: query schema mismatch")
            _require(pending_action is None, f"{prefix}: query interrupts an action")
            _require(query_key not in queries, f"{prefix}: duplicate query {query_key}")
            rng = _environment_rng(
                record["environment_rng"],
                reset_index=reset_index,
                expected_base_seed=expected_environment_seed,
                prefix=prefix,
                require_contract_hash=True,
            )
            _require(
                rng["environment_seed_contract_sha256"] == identity[1]
                and rng["physx_enhanced_determinism"] == identity[3],
                f"{prefix}: RNG/live trace identity mismatch",
            )
            episode_seeds.setdefault(reset_index, rng["episode_seed"])
            _require(
                episode_seeds[reset_index] == rng["episode_seed"],
                f"{prefix}: mixed episode seed",
            )
            expected_local = sum(1 for key in queries if key[0] == reset_index)
            _require(
                query_index == expected_local,
                f"{prefix}: local query index is not contiguous",
            )
            global_queries.append(global_query_index)
            state = record["state"]
            _require(
                isinstance(state, dict)
                and set(state) == {"joint_position", "gripper_position"},
                f"{prefix}: state schema mismatch",
            )
            _finite_vector(state["joint_position"], 7, f"{prefix} joint state")
            _finite_vector(state["gripper_position"], 1, f"{prefix} gripper state")
            images = record["images"]
            _require(
                isinstance(images, dict)
                and set(images)
                == {
                    "native_external",
                    "native_wrist",
                    "request_external",
                    "request_wrist",
                    "visualization_external",
                    "visualization_wrist",
                    "model_order",
                    "client_model_spatial_transform",
                    "server_model_resize",
                    "masked_third_slot",
                    "visualization_spatial_transform",
                    "wrist_rotation_degrees",
                },
                f"{prefix}: image schema mismatch",
            )
            native_external = _image_identity(
                images["native_external"], [720, 1280, 3], f"{prefix} native external"
            )
            native_wrist = _image_identity(
                images["native_wrist"], [720, 1280, 3], f"{prefix} native wrist"
            )
            request_external = _image_identity(
                images["request_external"],
                [720, 1280, 3],
                f"{prefix} native external request",
            )
            request_wrist = _image_identity(
                images["request_wrist"],
                [720, 1280, 3],
                f"{prefix} native wrist request",
            )
            visualization_external = _image_identity(
                images["visualization_external"],
                [224, 224, 3],
                f"{prefix} non-model external visualization",
            )
            visualization_wrist = _image_identity(
                images["visualization_wrist"],
                [224, 224, 3],
                f"{prefix} non-model wrist visualization",
            )
            _require(
                request_external["sha256"] == native_external["sha256"]
                and request_wrist["sha256"] == native_wrist["sha256"],
                f"{prefix}: model request is not byte-identical native imagery",
            )
            _require(
                images["model_order"]
                == ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb_masked"]
                and images["client_model_spatial_transform"] is None
                and images["server_model_resize"]
                == PI05_DROID_JOINTPOS_SERVER_MODEL_RESIZE
                and images["masked_third_slot"]
                == "server_DroidInputs_zeros_like_base_mask_false"
                and images["visualization_spatial_transform"]
                == ("openpi_client.image_tools.resize_with_pad_PIL_bilinear_non_model")
                and images["wrist_rotation_degrees"] == 0,
                f"{prefix}: model image preprocessing/order/rotation mismatch",
            )
            external_hashes.add(request_external["sha256"])
            wrist_hashes.add(request_wrist["sha256"])
            _require(
                visualization_external["shape"] == [224, 224, 3]
                and visualization_wrist["shape"] == [224, 224, 3],
                f"{prefix}: visualization resize escaped its non-model boundary",
            )
            _require(
                record["response_action_shape"] == [15, 8]
                and record["response_action_dtype"] == "float64"
                and record["execution_horizon"] == 8,
                f"{prefix}: response/execution contract mismatch",
            )
            raw = _finite_matrix(
                record["response_action_chunk"], 15, 8, f"{prefix} response"
            )
            planned = _finite_matrix(
                record["planned_action_chunk"], 8, 8, f"{prefix} planned"
            )
            for action_index, planned_action in enumerate(planned):
                _require(
                    planned_action[:7] == raw[action_index][:7]
                    and planned_action[7] == float(raw[action_index][7] > 0.5),
                    f"{prefix}: planned action {action_index} mismatch",
                )
            for action in raw:
                for dimension in range(7):
                    raw_joint_min[dimension] = min(
                        raw_joint_min[dimension], action[dimension]
                    )
                    raw_joint_max[dimension] = max(
                        raw_joint_max[dimension], action[dimension]
                    )
            _sensor_counters(record["sensor_frame_counters"], prefix)
            current_environment = previous_environment_by_reset.get(reset_index)
            if current_environment is not None:
                _require(
                    record["sensor_frame_counters"]
                    == current_environment["sensor_frame_counters"],
                    f"{prefix}: query sensor counters are not the live action boundary",
                )
            prompt = record["prompt"]
            _require(isinstance(prompt, str) and prompt, f"{prefix}: prompt is invalid")
            prompts.add(prompt)
            queries[query_key] = record
            continue

        if record_type == "openpi_joint_position_action":
            _require(set(record) == action_fields, f"{prefix}: action schema mismatch")
            _require(query_key in queries, f"{prefix}: action has no query")
            action_index = record["chunk_action_index"]
            _require(
                type(action_index) is int and 0 <= action_index < 8,
                f"{prefix}: action chunk index mismatch",
            )
            key = (*query_key, action_index)
            _require(key not in actions, f"{prefix}: duplicate action {key}")
            _require(pending_action is None, f"{prefix}: prior action has no execution")
            _require(
                global_query_index == queries[query_key]["global_query_index"],
                f"{prefix}: action global query identity mismatch",
            )
            raw = _finite_vector(record["raw_action"], 8, f"{prefix} raw action")
            emitted = _finite_vector(
                record["emitted_action"], 8, f"{prefix} emitted action"
            )
            _require(
                raw == queries[query_key]["response_action_chunk"][action_index]
                and emitted[:7] == raw[:7]
                and emitted[7] == float(raw[7] > 0.5),
                f"{prefix}: emitted action differs from server response",
            )
            actions[key] = record
            pending_action = key
            continue

        _require(
            record_type == "openpi_joint_position_execution",
            f"{prefix}: unexpected record type {record_type!r}",
        )
        _require(
            set(record) == execution_fields, f"{prefix}: execution schema mismatch"
        )
        action_index = record["chunk_action_index"]
        key = (*query_key, action_index)
        _require(
            pending_action == key and key in actions,
            f"{prefix}: execution/action ordering mismatch",
        )
        _require(key not in executions, f"{prefix}: duplicate execution {key}")
        action = actions[key]
        _require(
            global_query_index == action["global_query_index"]
            and record["emitted_action"] == action["emitted_action"],
            f"{prefix}: execution emitted-action identity mismatch",
        )
        outer_step = expected_outer_step[reset_index]
        _require(
            record["outer_step_index"] == outer_step,
            f"{prefix}: outer step index is not contiguous",
        )
        _validate_execution_report(
            record["action_execution"], record["emitted_action"], prefix
        )
        expected_finger = (
            float(np.float32(np.pi / 4.0))
            if record["emitted_action"][7] == 1.0
            else 0.0
        )
        processed_finger = _finite_vector(
            record["processed_finger_position_target"], 1, f"{prefix} processed finger"
        )
        articulation_finger = _finite_vector(
            record["articulation_finger_position_target"],
            1,
            f"{prefix} articulation finger",
        )
        _require(
            processed_finger == [expected_finger]
            and articulation_finger == [expected_finger],
            f"{prefix}: closed-positive gripper target mismatch",
        )
        _finite_vector(
            record["measured_joint_position_after"], 7, f"{prefix} measured joint state"
        )
        _finite_vector(
            record["measured_closed_positive_gripper_after"],
            1,
            f"{prefix} measured gripper state",
        )
        environment_before = _environment_state(record["environment_before"], prefix)
        environment_after = _environment_state(record["environment_after"], prefix)
        if action_index == 0:
            _require(
                queries[query_key]["sensor_frame_counters"]
                == environment_before["sensor_frame_counters"],
                f"{prefix}: query camera counters differ from execution boundary",
            )
        previous = previous_environment_by_reset.get(reset_index)
        if previous is not None:
            _require(
                environment_before == previous, f"{prefix}: environment boundary gap"
            )
        _require(
            environment_after["episode_length"] == outer_step + 1
            and environment_after["sim_step_counter"]
            == environment_before["sim_step_counter"] + 8
            and environment_after["common_step_counter"]
            == environment_before["common_step_counter"] + 1
            and all(
                environment_after["sensor_frame_counters"][name]
                == environment_before["sensor_frame_counters"][name] + 1
                for name in ("external_cam", "wrist_cam")
            ),
            f"{prefix}: execution simulator/camera cadence mismatch",
        )
        _require(
            record["terminated"] is False and record["truncated"] is False,
            f"{prefix}: 450/451 boundary returned a terminal flag",
        )
        terminal_rubric = record["terminal_rubric"]
        terminal_visualization = record["terminal_visualization"]
        if outer_step == 449:
            _require(
                isinstance(terminal_rubric, dict)
                and set(terminal_rubric) == {"success", "progress", "metrics"}
                and type(terminal_rubric["success"]) is bool
                and isinstance(terminal_rubric["metrics"], dict)
                and isinstance(terminal_rubric["progress"], int | float)
                and not isinstance(terminal_rubric["progress"], bool)
                and math.isfinite(terminal_rubric["progress"]),
                f"{prefix}: action-450 live terminal rubric mismatch",
            )
            terminal_visualizations[reset_index] = _terminal_image_identity(
                terminal_visualization, prefix
            )["sha256"]
        else:
            _require(terminal_rubric is None, f"{prefix}: premature terminal rubric")
            _require(
                terminal_visualization is None,
                f"{prefix}: premature terminal visualization",
            )
        previous_environment_by_reset[reset_index] = environment_after
        expected_outer_step[reset_index] += 1
        executions[key] = record
        pending_action = None

    _require(pending_action is None, "trace ends with an unexecuted action")
    _require(queries and executions, "attested trace has no queries/executions")
    _require(len(identities) == 1, "trace contract identity or PhysX flag drifted")
    server_hash, seed_hash, runtime_hash, physx_flag = next(iter(identities))
    _require(
        physx_flag is False,
        "pinned joint-position protocol requires PhysX enhanced determinism false",
    )
    _require(
        global_queries == list(range(len(global_queries))),
        "global policy request indices are not contiguous from zero",
    )
    if expected_server_contract_sha256 is not None:
        _require(
            server_hash == expected_server_contract_sha256,
            "trace server contract differs from expected contract",
        )
    if expected_prompt is not None:
        _require(prompts == {expected_prompt}, "trace prompt mismatch")

    queries_by_reset = defaultdict(list)
    actions_by_query = defaultdict(list)
    for reset_index, query_index in queries:
        queries_by_reset[reset_index].append(query_index)
    for reset_index, query_index, action_index in executions:
        actions_by_query[(reset_index, query_index)].append(action_index)
    for reset_index, query_indices in queries_by_reset.items():
        query_indices.sort()
        _require(
            query_indices == list(range(len(query_indices))),
            f"reset {reset_index} query indices are not contiguous",
        )
        for query_index in query_indices:
            indices = sorted(actions_by_query[(reset_index, query_index)])
            _require(
                indices == list(range(len(indices))) and indices,
                f"query {(reset_index, query_index)} execution indices are not contiguous",
            )
            if query_index != query_indices[-1]:
                _require(
                    len(indices) == 8,
                    f"non-final query {(reset_index, query_index)} is partial",
                )
        _require(
            expected_outer_step[reset_index] == 450,
            f"reset {reset_index} did not complete exactly 450 executions",
        )
        _require(
            len(query_indices) == 57,
            f"reset {reset_index} did not issue exactly 57 execute-eight queries",
        )
        final_execution = next(
            record
            for key, record in executions.items()
            if key[0] == reset_index and record["outer_step_index"] == 449
        )
        _require(
            final_execution["environment_after"]["episode_length"] == 450
            and final_execution["terminal_rubric"] is not None,
            f"reset {reset_index} lacks the action-450 live terminal boundary",
        )
        _require(
            reset_index in terminal_visualizations,
            f"reset {reset_index} lacks post-action-450 visual evidence",
        )
    _require(
        set(queries_by_reset) == set(range(len(queries_by_reset))),
        "trace reset indices are not contiguous from zero",
    )

    episode_lengths = None
    if metrics_csv is not None:
        episode_lengths = _episode_lengths(metrics_csv)
        with metrics_csv.open(newline="") as metrics_file:
            metric_rows = list(csv.DictReader(metrics_file))
        _require(
            set(queries_by_reset) == set(range(len(episode_lengths))),
            "trace resets do not match metrics episodes",
        )
        for reset_index, episode_length in enumerate(episode_lengths):
            _require(
                episode_length == 450,
                f"episode {reset_index} metrics length must be exactly 450",
            )
            execution_count = sum(1 for key in executions if key[0] == reset_index)
            _require(
                execution_count == episode_length,
                f"episode {reset_index} execution count differs from metrics",
            )
            _require(
                len(queries_by_reset[reset_index]) == math.ceil(episode_length / 8),
                f"episode {reset_index} query count differs from execute-eight",
            )
            terminal = next(
                record["terminal_rubric"]
                for key, record in executions.items()
                if key[0] == reset_index and record["outer_step_index"] == 449
            )
            metric = metric_rows[reset_index]
            _require(
                metric.get("success", "").strip().lower()
                == str(terminal["success"]).lower(),
                f"episode {reset_index} CSV success differs from live terminal rubric",
            )
            metric_progress = float(metric.get("progress", "nan"))
            _require(
                math.isfinite(metric_progress)
                and metric_progress == float(terminal["progress"]),
                f"episode {reset_index} CSV progress differs from live terminal rubric",
            )
            if "numerical_failure" in metric:
                _require(
                    metric["numerical_failure"].strip().lower() == "false",
                    f"episode {reset_index} CSV reports a numerical failure",
                )
            if "numerical_failure_reason" in metric:
                _require(
                    not metric["numerical_failure_reason"].strip(),
                    f"episode {reset_index} CSV has a failure reason",
                )

    if runtime_contract_path is not None:
        from polaris.pi05_droid_jointpos_runtime import (
            validate_jointpos_runtime_artifact,
        )

        artifact = validate_jointpos_runtime_artifact(
            runtime_contract_path, expected_runtime_sha256=runtime_hash
        )
    else:
        artifact = None
    episode_query_counts = [
        len(queries_by_reset[index]) for index in sorted(queries_by_reset)
    ]
    cumulative = []
    total = 0
    for count in episode_query_counts:
        total += count
        cumulative.append(total)
    return {
        "schema_version": PI05_DROID_JOINTPOS_TRACE_SCHEMA_VERSION,
        "trace_path": str(trace_path.resolve()),
        "trace_sha256": digest,
        "metrics_sha256": (
            _sha256_file(metrics_csv) if metrics_csv is not None else None
        ),
        "query_records": len(queries),
        "emitted_action_records": len(actions),
        "execution_records": len(executions),
        "reset_count": len(queries_by_reset),
        "episode_lengths": episode_lengths,
        "episode_query_counts": episode_query_counts,
        "cumulative_query_counts": cumulative,
        "global_query_indices_contiguous": True,
        "prompts": sorted(prompts),
        "response_action_shape": [15, 8],
        "execution_horizon": 8,
        "native_image_shape": [720, 1280, 3],
        "request_image_shape": [720, 1280, 3],
        "request_image_dtype": "uint8",
        "client_model_spatial_transform": None,
        "server_model_resize": PI05_DROID_JOINTPOS_SERVER_MODEL_RESIZE,
        "model_image_shape": [224, 224, 3],
        "visualization_image_shape": [224, 224, 3],
        "visualization_is_model_input": False,
        "terminal_visualization_shape": [224, 448, 3],
        "terminal_visualization_dtype": "uint8",
        "terminal_visualization_source": (
            "post_action450_returned_expensive_splat_observation"
        ),
        "terminal_visualization_sha256": [
            terminal_visualizations[index] for index in sorted(terminal_visualizations)
        ],
        "wrist_rotation_degrees": 0,
        "distinct_external_frames": len(external_hashes),
        "distinct_wrist_frames": len(wrist_hashes),
        "raw_joint_target_min": raw_joint_min,
        "raw_joint_target_max": raw_joint_max,
        "environment_base_seed": expected_environment_seed,
        "environment_seed_scheme": "base_plus_episode_index_v1",
        "environment_seed_contract_sha256": seed_hash,
        "environment_episode_seeds": [
            episode_seeds[index] for index in sorted(episode_seeds)
        ],
        "environment_physx_enhanced_determinism": physx_flag,
        "environment_determinism_claim": "rng_bound_not_bitwise",
        "server_contract_sha256": server_hash,
        "runtime_contract_sha256": runtime_hash,
        "runtime_artifact": artifact,
        "boundary_profile": "outer450_internal451_no_autoreset",
        "status": "pass",
    }


def audit_trace(
    trace_path: Path,
    expected_prompt: str | None = None,
    metrics_csv: Path | None = None,
    expected_environment_seed: int | None = None,
    expected_server_contract_sha256: str | None = None,
    runtime_contract_path: Path | None = None,
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
    versions = {record.get("schema_version") for record in records}
    if versions == {PI05_DROID_JOINTPOS_TRACE_SCHEMA_VERSION}:
        _require(
            expected_environment_seed is not None,
            "attested trace requires --expected-environment-seed",
        )
        _require(
            expected_server_contract_sha256 is not None,
            "attested trace requires --expected-server-contract-sha256",
        )
        _require(
            runtime_contract_path is not None,
            "attested trace requires --runtime-contract",
        )
        return _audit_attested_trace(
            records,
            trace_path=trace_path,
            digest=digest.hexdigest(),
            expected_prompt=expected_prompt,
            metrics_csv=metrics_csv,
            expected_environment_seed=expected_environment_seed,
            expected_server_contract_sha256=expected_server_contract_sha256,
            runtime_contract_path=runtime_contract_path,
        )
    _require(
        expected_environment_seed is None,
        "seeded validation requires the current attested trace schema",
    )
    _require(
        expected_server_contract_sha256 is None and runtime_contract_path is None,
        "legacy traces cannot satisfy server/runtime attestation",
    )
    return _audit_legacy_trace(
        trace_path,
        expected_prompt=expected_prompt,
        metrics_csv=metrics_csv,
        expected_environment_seed=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--expected-prompt")
    parser.add_argument("--metrics-csv", type=Path)
    parser.add_argument("--expected-environment-seed", type=int)
    parser.add_argument("--expected-server-contract-sha256")
    parser.add_argument("--runtime-contract", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = audit_trace(
        args.trace,
        expected_prompt=args.expected_prompt,
        metrics_csv=args.metrics_csv,
        expected_environment_seed=args.expected_environment_seed,
        expected_server_contract_sha256=args.expected_server_contract_sha256,
        runtime_contract_path=args.runtime_contract,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
