#!/usr/bin/env python3
"""Build the immutable job-1098523 reasoning replay fixture.

The input is the completed Ego-LAP policy trace from the controller-candidate
canary.  This host-only tool accepts exactly that trace, validates its serving
and train/eval contracts, and serializes every 8-wide absolute PolaRiS action
as little-endian float32 bytes.  The resulting fixture is model-free: replay
does not need a checkpoint or policy server.
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


TRACE_SHA256 = "db2436639cd2ddf2c9252346c837b9a081a6563048d8005d1d8b7cf2957aea80"
TRACE_PROFILE = "ego_lap_eef_pose_runtime_trace_v2"
FIXTURE_PROFILE = "reasoning_43075_job1098523_fulltrace_actions_v1"
POLARIS_COMMIT = "0611d384f5f26ef9bd8ff114be273e875c3fe719"
EXPECTED_EVENTS = {
    "reset": 1,
    "query": 37,
    "action": 294,
    "execution": 293,
    "execution_failure": 1,
    "episode_complete": 1,
}
QUERY_CONTRACT = {
    "checkpoint_path": (
        "gs://v6_east1d/checkpoints/lap_oxe_magic_soup_reasoning_full/"
        "oxe_magic_soup_reasoning_full_v2_flow_pred0_cf0_ckpt25_v6_32_b512_"
        "s42_20260630/43075"
    ),
    "contract_sha256": "549f67a7d3eb175696a52fc0da760e94f8ba9df5599c16a5be83b4fb2897b605",
    "policy_type": "flow",
    "response_semantics": "cumulative_delta_targets",
    "execution_horizon": 8,
    "gripper_execution_profile": "binary_model_open_gt_0p5_else_closed_v1",
    "gripper_threshold": 0.5,
    "action_sampler_profile": "flow_explicit_euler_t1_to_t0_v1",
    "flow_num_steps": 10,
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
    "normalization_output_formula": "q99_output_eps1e-8_zeroq01_extrapolate_v1",
    "normalization_formula_probe_sha256": (
        "b30e80077b5d2ba2fd9b543315591379d8be3c19cd8d8cd1360f76868b13a7e0"
    ),
    "state_layout": "xyz+r6_first_two_columns+gripper_open",
    "state_layout_mode": "manifest_train_matched_columns_v1",
    "polaris_profile": "panda_link8_eef_pose_single_arm_v1",
}
EXPECTED_GRIPPER_CHANGES = [
    {"step": 198, "from": 0.0, "to": 1.0},
    {"step": 200, "from": 1.0, "to": 0.0},
    {"step": 265, "from": 0.0, "to": 1.0},
    {"step": 272, "from": 1.0, "to": 0.0},
    {"step": 281, "from": 0.0, "to": 1.0},
]
EXPECTED_FAILURE = {
    "policy_step": 293,
    "physics_substep": 2,
    "joint_name": "panda_joint7",
    "evidence_sha256": "81ab9fb0cf1b74d67abbafb75ecc2ded5e606547fb46eec3e2b5a06acadd2959",
}


class FixtureBuildError(ValueError):
    """The input trace does not match the closed source contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FixtureBuildError(message)


def reject_constant(token: str) -> None:
    raise FixtureBuildError(f"non-standard JSON constant {token!r}")


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_trace(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    raw = path.read_bytes()
    require(sha256(raw) == TRACE_SHA256, "source trace SHA-256 drift")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            row = json.loads(
                line.decode("utf-8"),
                parse_constant=reject_constant,
                object_pairs_hook=reject_duplicate_keys,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FixtureBuildError(
                f"invalid JSONL line {line_number}: {error}"
            ) from error
        require(isinstance(row, dict), f"line {line_number} is not an object")
        require(row.get("schema_version") == 2, f"line {line_number} schema")
        if row.get("event") != "episode_complete":
            require(
                row.get("trace_profile") == TRACE_PROFILE,
                f"line {line_number} trace profile",
            )
        rows.append(row)
    require(
        Counter(row.get("event") for row in rows) == EXPECTED_EVENTS, "event counts"
    )
    return raw, rows


def finite_action(value: Any, *, step: int) -> list[float]:
    require(isinstance(value, list) and len(value) == 8, f"action {step} width")
    result: list[float] = []
    for index, item in enumerate(value):
        require(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item)),
            f"action {step}[{index}] finite scalar",
        )
        result.append(float(item))
    norm = math.sqrt(sum(component * component for component in result[3:7]))
    require(abs(norm - 1.0) <= 1e-3, f"action {step} quaternion norm")
    require(result[7] in (0.0, 1.0), f"action {step} binary gripper")
    return result


def build_payload(path: Path) -> dict[str, Any]:
    _, rows = load_trace(path)
    queries = [row for row in rows if row["event"] == "query"]
    actions = [row for row in rows if row["event"] == "action"]
    executions = [row for row in rows if row["event"] == "execution"]
    failures = [row for row in rows if row["event"] == "execution_failure"]
    require(
        all(
            all(query.get(key) == value for key, value in QUERY_CONTRACT.items())
            for query in queries
        ),
        "query serving/train-eval contract drift",
    )
    require(
        [query.get("query") for query in queries] == list(range(37)), "query indices"
    )
    require([row.get("step") for row in actions] == list(range(294)), "action steps")
    require(
        [row.get("step") for row in executions] == list(range(293)), "execution steps"
    )
    require(
        all(
            row.get("query") == row["step"] // 8
            and row.get("chunk_index") == row["step"] % 8
            for row in actions
        ),
        "action query/chunk cadence",
    )
    vectors = [
        finite_action(row.get("polaris_action"), step=index)
        for index, row in enumerate(actions)
    ]
    changes: list[dict[str, float | int]] = []
    previous = vectors[0][7]
    for step, action in enumerate(vectors[1:], start=1):
        current = action[7]
        if current != previous:
            changes.append({"step": step, "from": previous, "to": current})
        previous = current
    require(changes == EXPECTED_GRIPPER_CHANGES, "gripper endpoint schedule")
    failure = failures[0]
    reason = failure.get("numerical_failure_reason", "")
    require(
        failure.get("step") == EXPECTED_FAILURE["policy_step"]
        and "joint='panda_joint7'" in reason
        and "physics_substep=2" in reason
        and EXPECTED_FAILURE["evidence_sha256"] in reason,
        "terminal numerical failure identity",
    )
    packed = b"".join(struct.pack("<8f", *action) for action in vectors)
    compressed = zlib.compress(packed, level=9)
    encoded = base64.b64encode(compressed).decode("ascii")
    chunks = [encoded[index : index + 120] for index in range(0, len(encoded), 120)]
    return {
        "schema_version": 1,
        "fixture_profile": FIXTURE_PROFILE,
        "polaris_commit": POLARIS_COMMIT,
        "source": {
            "job_id": "1098523",
            "trace_sha256": TRACE_SHA256,
            "trace_profile": TRACE_PROFILE,
            "event_counts": EXPECTED_EVENTS,
            "query_contract": QUERY_CONTRACT,
            "expected_failure": EXPECTED_FAILURE,
        },
        "action_plan": {
            "profile": "all_recorded_absolute_polaris_actions_exact_float32_v1",
            "action_count": len(vectors),
            "action_width": 8,
            "gripper_endpoint_changes": changes,
        },
        "action_encoding": {
            "codec": "zlib-9-base64",
            "dtype": "little-endian-float32",
            "action_count": len(vectors),
            "action_width": 8,
            "uncompressed_size_bytes": len(packed),
            "uncompressed_sha256": sha256(packed),
            "compressed_size_bytes": len(compressed),
            "compressed_sha256": sha256(compressed),
        },
        "actions_zlib_base64_chunks": chunks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), f"refusing to overwrite {args.output}")
    payload = build_payload(args.trace)
    data = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    print(f"fixture={args.output.resolve()};size={len(data)};sha256={sha256(data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
