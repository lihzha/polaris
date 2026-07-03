#!/usr/bin/env python3
"""Generate immutable Gate-0 replay fixtures from two mirrored policy traces.

The generator is intentionally host-only and standard-library-only.  It
accepts only the two byte-pinned postrun traces from jobs 1098292 and 1098294,
checks their closed query/action/failure contracts, and emits a deterministic
120-action model-free replay plan for each trace.  Actions after the observed
failure are copied only from the remainder of query 14's recorded eight-action
execute window.  The other eight planned query actions remain provenance and
are never presented as executable baseline actions.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any
import zlib


SCHEMA_VERSION = 1
FIXTURE_PROFILE = "polaris_eef_canary_gate0_trace_replay_v1"
TRACE_PROFILE = "ego_lap_eef_pose_runtime_trace_v2"
ENVIRONMENT = "DROID-FoodBussing"
INITIAL_CONDITION_INDEX = 0
QUERY_INDEX = 14
QUERY_START_STEP = 112
QUERY_PLAN_ACTION_COUNT = 16
QUERY_EXECUTION_HORIZON = 8
REPLAY_ACTION_COUNT = 120
ACTION_WIDTH = 8
BASE_COMMIT = "712240cbb215ecb31830cdb2ee65e91704160372"

DEFAULT_TRACE_ROOT = Path(
    "/home/lzha/code/ego-lap/.codex_artifacts/polaris-canaries-9129504/postrun"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "fixtures"

ASSET_CONTRACT = {
    "environment": ENVIRONMENT,
    "initial_condition_index": INITIAL_CONDITION_INDEX,
    "initial_conditions_sha256": (
        "40091faee14f692350220871d30705294f21f17ae3d2974cd3c09a34d560f5de"
    ),
    "polaris_hub_revision": "8c7e4103e266ef83d8b1ad2e9a63116edd5f155b",
    "robot_usd_sha256": (
        "d8379925b103963dbf3e7c85bcc4ae101b81b7c1d7dabe7d2e964f41d069ec44"
    ),
    "scene_sha256": (
        "82cd641e422935b394ce7ea7b6be55214c9952a2544000222921e544c409b489"
    ),
}

QUERY_CONTRACT_FIELDS = (
    "checkpoint_profile",
    "checkpoint_path",
    "contract_sha256",
    "policy_type",
    "response_semantics",
    "execution_horizon",
    "ar_endpoint_interpolation_profile",
    "ar_endpoint_interpolation_steps",
    "gripper_execution_profile",
    "gripper_threshold",
    "action_sampler_profile",
    "flow_num_steps",
    "initial_rng_seed",
    "ar_max_decoding_steps",
    "ar_temperature",
    "ar_stop_at_eos",
    "frame_description",
    "eef_frame",
    "numeric_action_frame",
    "normalization_scope",
    "normalization_stats_sha256",
    "normalization_profile",
    "normalization_compute_dtype",
    "normalization_input_formula",
    "normalization_output_formula",
    "normalization_formula_probe_sha256",
    "state_layout",
    "state_layout_mode",
    "polaris_profile",
)

VARIANTS: dict[str, dict[str, Any]] = {
    "official_lap3b": {
        "job_id": "1098292",
        "filename": "official_lap3b_job1098292_gate0_actions.json",
        "trace_size_bytes": 338377,
        "trace_sha256": (
            "490ba92f39abb1fd83c8382dd1b7a16f4e1a12e86df29cbfb484b6395474789c"
        ),
        "event_counts": {
            "action": 118,
            "episode_complete": 1,
            "execution": 117,
            "execution_failure": 1,
            "query": 15,
            "reset": 1,
        },
        "failure": {
            "action_event_count": 118,
            "evidence_sha256": (
                "63c061ec5a47a8bc085547f2abd8dcbc266c9616664d252e29c39ef53864a5f3"
            ),
            "execution_event_count": 117,
            "joint_name": "panda_joint5",
            "physics_substep": 6,
            "policy_step": 117,
            "query": 14,
            "query_chunk_index": 5,
        },
        "query_contract": {
            "checkpoint_profile": "original_lap_public_3b_v1",
            "checkpoint_path": (
                "/lustre/fs11/portfolios/nvr/projects/nvr_lpr_rvp/users/lzha/"
                "checkpoints/LAP-3B-601db9c1"
            ),
            "contract_sha256": (
                "fe47ce07e6da2c1acc4a2b3b388f5a3fe774e4f22432bef091f9ed0e9df2b87e"
            ),
            "policy_type": "flow",
            "response_semantics": "cumulative_delta_targets",
            "execution_horizon": 8,
            "ar_endpoint_interpolation_profile": None,
            "ar_endpoint_interpolation_steps": None,
            "gripper_execution_profile": "binary_model_open_gt_0p5_else_closed_v1",
            "gripper_threshold": 0.5,
            "action_sampler_profile": "flow_explicit_euler_t1_to_t0_v1",
            "flow_num_steps": 10,
            "initial_rng_seed": 0,
            "ar_max_decoding_steps": None,
            "ar_temperature": None,
            "ar_stop_at_eos": None,
            "frame_description": "robot base frame",
            "eef_frame": "panda_link8",
            "numeric_action_frame": "robot_base",
            "normalization_scope": "global",
            "normalization_stats_sha256": (
                "1b8102a7ab33e5b3b97c6b6f629a1b3dff3100f9f59e25cd7cab0beda7ce9eb8"
            ),
            "normalization_profile": "q99_train_matched_v1",
            "normalization_compute_dtype": "float32",
            "normalization_input_formula": "q99_input_eps1e-8_clip_zero0_v1",
            "normalization_output_formula": (
                "q99_output_eps1e-8_zeroq01_extrapolate_v1"
            ),
            "normalization_formula_probe_sha256": (
                "b30e80077b5d2ba2fd9b543315591379d8be3c19cd8d8cd1360f76868b13a7e0"
            ),
            "state_layout": "xyz+r6_first_two_rows+gripper_open",
            "state_layout_mode": "public_lap_train_matched_rows_v1",
            "polaris_profile": "panda_link8_eef_pose_single_arm_v1",
        },
    },
    "reasoning_43075": {
        "job_id": "1098294",
        "filename": "reasoning_43075_job1098294_gate0_actions.json",
        "trace_size_bytes": 331875,
        "trace_sha256": (
            "a48107f2268a38f507bae5848f194f2680ccc52eedd042039cea5fd3cbebd948"
        ),
        "event_counts": {
            "action": 113,
            "episode_complete": 1,
            "execution": 112,
            "execution_failure": 1,
            "query": 15,
            "reset": 1,
        },
        "failure": {
            "action_event_count": 113,
            "evidence_sha256": (
                "3c6242a645b40fe29f7223dc4a146cdb7ee04fe661b136098929bb9b973580b8"
            ),
            "execution_event_count": 112,
            "joint_name": "panda_joint7",
            "physics_substep": 2,
            "policy_step": 112,
            "query": 14,
            "query_chunk_index": 0,
        },
        "query_contract": {
            "checkpoint_profile": "manifest_v1_canonical",
            "checkpoint_path": (
                "gs://v6_east1d/checkpoints/lap_oxe_magic_soup_reasoning_full/"
                "oxe_magic_soup_reasoning_full_v2_flow_pred0_cf0_ckpt25_v6_32_"
                "b512_s42_20260630/43075"
            ),
            "contract_sha256": (
                "549f67a7d3eb175696a52fc0da760e94f8ba9df5599c16a5be83b4fb2897b605"
            ),
            "policy_type": "flow",
            "response_semantics": "cumulative_delta_targets",
            "execution_horizon": 8,
            "ar_endpoint_interpolation_profile": None,
            "ar_endpoint_interpolation_steps": None,
            "gripper_execution_profile": "binary_model_open_gt_0p5_else_closed_v1",
            "gripper_threshold": 0.5,
            "action_sampler_profile": "flow_explicit_euler_t1_to_t0_v1",
            "flow_num_steps": 10,
            "initial_rng_seed": 0,
            "ar_max_decoding_steps": None,
            "ar_temperature": None,
            "ar_stop_at_eos": None,
            "frame_description": "egocentric frame",
            "eef_frame": "panda_link8",
            "numeric_action_frame": "robot_base",
            "normalization_scope": "global",
            "normalization_stats_sha256": (
                "f44d886688ac9a7ca51df870fd6f75f65b24e8d56aa4de3016694300769f1f5b"
            ),
            "normalization_profile": "q99_train_matched_v1",
            "normalization_compute_dtype": "float32",
            "normalization_input_formula": "q99_input_eps1e-8_clip_zero0_v1",
            "normalization_output_formula": (
                "q99_output_eps1e-8_zeroq01_extrapolate_v1"
            ),
            "normalization_formula_probe_sha256": (
                "b30e80077b5d2ba2fd9b543315591379d8be3c19cd8d8cd1360f76868b13a7e0"
            ),
            "state_layout": "xyz+r6_first_two_columns+gripper_open",
            "state_layout_mode": "manifest_train_matched_columns_v1",
            "polaris_profile": "panda_link8_eef_pose_single_arm_v1",
        },
    },
}


class FixtureGenerationError(ValueError):
    """A mirrored trace or generated fixture violated the closed contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FixtureGenerationError(message)


def _reject_constant(token: str) -> None:
    raise FixtureGenerationError(f"non-standard JSON constant {token!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FixtureGenerationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def strict_json_loads(data: bytes, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FixtureGenerationError(f"{field} is not strict JSON: {error}") from error
    _require(isinstance(value, dict), f"{field} must be an object")
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _float32_bytes(actions: list[list[float]]) -> bytes:
    return b"".join(struct.pack("<8f", *action) for action in actions)


def _float32_equal(left: Any, right: Any) -> bool:
    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
        and math.isfinite(float(left))
        and math.isfinite(float(right))
        and struct.pack("<f", float(left)) == struct.pack("<f", float(right))
    )


def _validate_action(value: Any, *, field: str) -> list[float]:
    _require(isinstance(value, list) and len(value) == ACTION_WIDTH, f"{field} shape")
    result: list[float] = []
    for index, item in enumerate(value):
        _require(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item)),
            f"{field}[{index}] must be finite",
        )
        result.append(float(item))
    norm = math.sqrt(sum(component * component for component in result[3:7]))
    _require(abs(norm - 1.0) <= 1e-3, f"{field} quaternion norm")
    _require(result[7] in (0.0, 1.0), f"{field} gripper endpoint")
    return result


def _parse_trace(data: bytes) -> list[dict[str, Any]]:
    _require(data.endswith(b"\n"), "trace must end with a newline")
    lines = data.splitlines()
    _require(lines and all(lines), "trace must contain only non-empty JSONL records")
    return [
        strict_json_loads(line, field=f"trace line {index}")
        for index, line in enumerate(lines, start=1)
    ]


def _failure_digest(reason: Any) -> str:
    _require(isinstance(reason, str), "failure reason must be a string")
    marker = "evidence_sha256="
    _require(reason.count(marker) == 1, "failure reason evidence digest marker")
    digest = reason.split(marker, 1)[1].split(")", 1)[0]
    _require(
        len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
        "failure reason evidence digest",
    )
    return digest


def _trace_path(trace_root: Path, config: dict[str, Any]) -> Path:
    return (
        trace_root
        / f"job_{config['job_id']}"
        / "results"
        / "policy_traces"
        / "episode_000000.jsonl"
    )


def build_fixture(variant: str, trace_path: Path) -> dict[str, Any]:
    """Validate one exact source trace and build its deterministic fixture."""

    _require(variant in VARIANTS, f"unknown variant {variant!r}")
    config = VARIANTS[variant]
    _require(trace_path.is_file(), f"missing source trace: {trace_path}")
    trace_bytes = trace_path.read_bytes()
    _require(len(trace_bytes) == config["trace_size_bytes"], "source trace size drift")
    _require(
        _sha256(trace_bytes) == config["trace_sha256"], "source trace digest drift"
    )
    events = _parse_trace(trace_bytes)
    _require(
        all(
            event.get("schema_version") == 2
            and event.get("trace_profile") == TRACE_PROFILE
            for event in events
        ),
        "trace schema/profile drift",
    )
    event_counts = dict(sorted(Counter(event.get("event") for event in events).items()))
    _require(event_counts == config["event_counts"], "trace event counts drift")

    reset = [event for event in events if event["event"] == "reset"]
    complete = [event for event in events if event["event"] == "episode_complete"]
    failures = [event for event in events if event["event"] == "execution_failure"]
    queries = [event for event in events if event["event"] == "query"]
    action_events = [event for event in events if event["event"] == "action"]
    executions = [event for event in events if event["event"] == "execution"]
    _require(
        len(reset) == len(complete) == len(failures) == 1, "terminal event cardinality"
    )
    _require(reset[0].get("episode") == 0, "reset episode drift")
    _require(
        complete[0].get("episode") == 0
        and complete[0].get("status") == "numerical_failure"
        and complete[0].get("numerical_failure") is True,
        "episode completion drift",
    )
    _require(len(queries) == 15, "query cardinality drift")
    _require(
        [query.get("query") for query in queries] == list(range(15))
        and [query.get("step") for query in queries] == list(range(0, 120, 8)),
        "query cadence drift",
    )
    expected_query_contract = config["query_contract"]
    for index, query in enumerate(queries):
        actual = {field: query.get(field) for field in QUERY_CONTRACT_FIELDS}
        _require(actual == expected_query_contract, f"query {index} contract drift")
        _require(
            isinstance(query.get("anchored_action_chunk"), list)
            and len(query["anchored_action_chunk"]) == QUERY_PLAN_ACTION_COUNT,
            f"query {index} planned action count",
        )

    failure = failures[0]
    expected_failure = config["failure"]
    for field in ("step", "query", "chunk_index"):
        fixture_field = {
            "step": "policy_step",
            "query": "query",
            "chunk_index": "query_chunk_index",
        }[field]
        _require(
            failure.get(field) == expected_failure[fixture_field],
            f"failure {field} drift",
        )
    reason = failure.get("numerical_failure_reason")
    _require(
        _failure_digest(reason) == expected_failure["evidence_sha256"],
        "failure evidence digest drift",
    )
    _require(
        f"joint='{expected_failure['joint_name']}'" in reason
        and f"policy_step={expected_failure['policy_step']}" in reason
        and f"physics_substep={expected_failure['physics_substep']}" in reason,
        "failure cadence/joint drift",
    )
    _require(
        len(action_events) == expected_failure["action_event_count"]
        and len(executions) == expected_failure["execution_event_count"],
        "action/execution failure cadence drift",
    )
    _require(
        [event.get("step") for event in action_events]
        == list(range(len(action_events))),
        "action step continuity drift",
    )
    _require(
        [event.get("step") for event in executions] == list(range(len(executions))),
        "execution step continuity drift",
    )

    query_by_index = {query["query"]: query for query in queries}
    observed_actions: list[list[float]] = []
    for event in action_events:
        step = event.get("step")
        query_index = event.get("query")
        chunk_index = event.get("chunk_index")
        _require(
            type(step) is int
            and type(query_index) is int
            and type(chunk_index) is int
            and query_index == step // 8
            and chunk_index == step % 8,
            f"action step {step!r} query/chunk cadence drift",
        )
        action = _validate_action(event.get("polaris_action"), field=f"action {step}")
        planned = _validate_action(
            query_by_index[query_index]["anchored_action_chunk"][chunk_index],
            field=f"query {query_index} planned action {chunk_index}",
        )
        _require(
            all(_float32_equal(left, right) for left, right in zip(action, planned)),
            f"action {step} differs from its planned query action",
        )
        observed_actions.append(action)

    query14 = queries[QUERY_INDEX]
    _require(
        query14.get("query") == QUERY_INDEX and query14.get("step") == QUERY_START_STEP,
        "query-14 identity drift",
    )
    query14_actions = [
        _validate_action(action, field=f"query-14 planned action {index}")
        for index, action in enumerate(query14["anchored_action_chunk"])
    ]
    observed_query14_count = len(observed_actions) - QUERY_START_STEP
    _require(
        observed_query14_count == expected_failure["query_chunk_index"] + 1,
        "observed query-14 prefix drift",
    )
    for index in range(observed_query14_count):
        _require(
            all(
                _float32_equal(left, right)
                for left, right in zip(
                    observed_actions[QUERY_START_STEP + index], query14_actions[index]
                )
            ),
            f"observed query-14 action {index} drift",
        )
    continuation = query14_actions[observed_query14_count:QUERY_EXECUTION_HORIZON]
    replay_actions = observed_actions + continuation
    _require(len(replay_actions) == REPLAY_ACTION_COUNT, "replay action count drift")
    _require(
        replay_actions[QUERY_START_STEP:] == query14_actions[:QUERY_EXECUTION_HORIZON],
        "replay query-14 execute window is not complete",
    )
    if variant == "official_lap3b":
        endpoints = [action[7] for action in query14_actions]
        _require(
            endpoints == [0.0, 0.0, 0.0] + [1.0] * 13,
            "official query-14 close endpoint plan drift",
        )
        _require(
            len(continuation) == 2 and all(action[7] == 1.0 for action in continuation),
            "official continuation does not include endpoint/post-close tail",
        )

    observed_raw = _float32_bytes(observed_actions)
    query14_raw = _float32_bytes(query14_actions)
    replay_raw = _float32_bytes(replay_actions)
    compressed = zlib.compress(replay_raw, level=9)
    encoded = base64.b64encode(compressed).decode("ascii")
    chunks = [encoded[index : index + 120] for index in range(0, len(encoded), 120)]
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_profile": FIXTURE_PROFILE,
        "variant": variant,
        "polaris_base_commit": BASE_COMMIT,
        "source": {
            "job_id": config["job_id"],
            "relative_trace_path": (
                f"job_{config['job_id']}/results/policy_traces/episode_000000.jsonl"
            ),
            "trace_profile": TRACE_PROFILE,
            "trace_size_bytes": len(trace_bytes),
            "trace_sha256": _sha256(trace_bytes),
            "event_counts": event_counts,
            "query_contract": expected_query_contract,
            "expected_failure": expected_failure,
        },
        "asset_contract": ASSET_CONTRACT,
        "action_plan": {
            "profile": "observed_prefix_plus_recorded_query14_continuation_v1",
            "observed_action_count": len(observed_actions),
            "observed_action_float32_sha256": _sha256(observed_raw),
            "planned_continuation_action_count": len(continuation),
            "query14_start_step": QUERY_START_STEP,
            "query14_observed_action_count": observed_query14_count,
            "query14_executable_action_count": QUERY_EXECUTION_HORIZON,
            "query14_planned_action_count": QUERY_PLAN_ACTION_COUNT,
            "query14_action_float32_sha256": _sha256(query14_raw),
            "replay_action_count": len(replay_actions),
            "continuation_semantics": (
                "recorded_query_execute_window_diagnostic_only_not_source_execution"
            ),
        },
        "action_encoding": {
            "action_count": len(replay_actions),
            "action_width": ACTION_WIDTH,
            "codec": "zlib-9-base64",
            "dtype": "little-endian-float32",
            "uncompressed_size_bytes": len(replay_raw),
            "uncompressed_sha256": _sha256(replay_raw),
            "compressed_size_bytes": len(compressed),
            "compressed_sha256": _sha256(compressed),
        },
        "actions_zlib_base64_chunks": chunks,
    }


def fixture_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def generate_all(trace_root: Path, output_dir: Path, *, check: bool) -> None:
    for variant, config in VARIANTS.items():
        payload = build_fixture(variant, _trace_path(trace_root, config))
        expected = fixture_bytes(payload)
        output = output_dir / config["filename"]
        if check:
            _require(output.is_file(), f"missing generated fixture: {output}")
            _require(
                output.read_bytes() == expected, f"generated fixture drift: {output}"
            )
            print(f"verified {variant}: {output} sha256={_sha256(expected)}")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(expected)
            print(f"wrote {variant}: {output} sha256={_sha256(expected)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    generate_all(args.trace_root.resolve(), args.output_dir.resolve(), check=args.check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
