#!/usr/bin/env python3
"""Join the official pi0.5 trace to its final JAX RNG-stream proof."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from polaris.pi05_droid_jointpos_serving_contract import (
    PI05_DROID_JOINTPOS_JAX_CONFIG,
    PI05_DROID_JOINTPOS_METADATA_KEY,
    validate_persisted_pi05_droid_jointpos_model_runtime,
    validate_persisted_pi05_droid_jointpos_rng_stream,
    validate_persisted_pi05_droid_jointpos_serving_contract,
)
from polaris.pi05_droid_jointpos_immutable import publish_immutable_json
from polaris.pi05_droid_jointpos_runtime import (
    PI05_DROID_JOINTPOS_TRACE_SCHEMA_VERSION,
)


QUERIES_PER_EPISODE = 57


def _strict_json(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_bytes(),
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"{field} contains non-finite JSON: {item}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} is not readable strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be one JSON object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_rng_key_data(request_count: int) -> list[int]:
    """Recompute the official Policy.infer stored-key recurrence independently."""

    import jax

    if jax.__version__ != "0.5.3":
        raise ValueError(f"RNG verifier requires JAX 0.5.3, got {jax.__version__}")
    live_config = {
        "default_prng_impl": jax.config.jax_default_prng_impl,
        "legacy_prng_key": jax.config.jax_legacy_prng_key,
        "threefry_partitionable": jax.config.jax_threefry_partitionable,
        "random_seed_offset": jax.config.jax_random_seed_offset,
        "default_matmul_precision": jax.config.jax_default_matmul_precision,
        "disable_jit": jax.config.jax_disable_jit,
        "enable_x64": jax.config.jax_enable_x64,
    }
    if live_config != PI05_DROID_JOINTPOS_JAX_CONFIG:
        raise ValueError(f"RNG verifier JAX config mismatch: {live_config}")

    @jax.jit
    def advance_rng(key):
        return jax.lax.fori_loop(
            0,
            request_count,
            lambda _index, current: jax.random.split(current)[0],
            key,
        )

    expected = advance_rng(jax.random.key(0))
    expected.block_until_ready()
    return jax.random.key_data(expected).tolist()


def verify_rng_stream(
    *,
    rng_stream_path: Path,
    trace_summary_path: Path,
    serving_contract_path: Path,
    model_runtime_path: Path,
    expected_rollouts: int,
    expected_server_pid: int,
) -> dict[str, Any]:
    """Prove trace requests exhaust the server's complete RNG stream exactly."""

    if type(expected_rollouts) is not int or expected_rollouts <= 0:
        raise ValueError("Expected rollouts must be one positive integer")
    if type(expected_server_pid) is not int or expected_server_pid <= 0:
        raise ValueError("Expected server PID must be one positive integer")
    serving = validate_persisted_pi05_droid_jointpos_serving_contract(
        serving_contract_path
    )
    model_runtime = validate_persisted_pi05_droid_jointpos_model_runtime(
        model_runtime_path, serving["value"]
    )
    server_contract = serving["value"][PI05_DROID_JOINTPOS_METADATA_KEY]
    contract_sha256 = server_contract["contract_sha256"]
    trace = _strict_json(trace_summary_path, "trace summary")
    expected_episode_counts = [QUERIES_PER_EPISODE] * expected_rollouts
    expected_cumulative = [
        QUERIES_PER_EPISODE * (index + 1) for index in range(expected_rollouts)
    ]
    expected_request_count = QUERIES_PER_EPISODE * expected_rollouts
    if (
        trace.get("schema_version") != PI05_DROID_JOINTPOS_TRACE_SCHEMA_VERSION
        or trace.get("status") != "pass"
        or trace.get("reset_count") != expected_rollouts
        or trace.get("episode_lengths") != [450] * expected_rollouts
        or trace.get("episode_query_counts") != expected_episode_counts
        or trace.get("cumulative_query_counts") != expected_cumulative
        or trace.get("query_records") != expected_request_count
        or trace.get("global_query_indices_contiguous") is not True
        or trace.get("server_contract_sha256") != contract_sha256
    ):
        raise ValueError("Trace summary does not prove the exact evaluator requests")
    model_server = model_runtime["value"]["server"]
    if model_server["expected_request_count"] != expected_request_count:
        raise ValueError("Model runtime request count differs from the trace")
    rng = validate_persisted_pi05_droid_jointpos_rng_stream(
        rng_stream_path,
        expected_request_count=expected_request_count,
        expected_contract_sha256=contract_sha256,
    )
    report = rng["value"]
    if report["server_pid"] != expected_server_pid:
        raise ValueError("RNG stream came from a different policy-server process")
    independently_expected_key = _expected_rng_key_data(expected_request_count)
    if (
        report["initial_key_data"] != [0, 0]
        or report["final_key_data"] != independently_expected_key
        or report["expected_final_key_data"] != independently_expected_key
        or report["observed_request_count"] != expected_request_count
    ):
        raise ValueError("Server RNG stream differs from the independent recurrence")
    return {
        "schema_version": 1,
        "profile": "openpi_pi05_droid_jointpos_request_identity_v1",
        "status": "pass",
        "proof": "trace_requests_equal_complete_official_policy_rng_stream",
        "server_pid": expected_server_pid,
        "expected_rollouts": expected_rollouts,
        "queries_per_episode": QUERIES_PER_EPISODE,
        "request_count": expected_request_count,
        "initial_key_data": [0, 0],
        "final_key_data": independently_expected_key,
        "server_contract_sha256": contract_sha256,
        "trace_summary_sha256": _file_sha256(trace_summary_path),
        "rng_stream_artifact_sha256": rng["sha256"],
        "serving_contract_artifact_sha256": serving["sha256"],
        "model_runtime_artifact_sha256": model_runtime["sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rng-stream", type=Path, required=True)
    parser.add_argument("--trace-summary", type=Path, required=True)
    parser.add_argument("--serving-contract", type=Path, required=True)
    parser.add_argument("--model-runtime", type=Path, required=True)
    parser.add_argument("--expected-rollouts", type=int, required=True)
    parser.add_argument("--expected-server-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_rng_stream(
        rng_stream_path=args.rng_stream,
        trace_summary_path=args.trace_summary,
        serving_contract_path=args.serving_contract,
        model_runtime_path=args.model_runtime,
        expected_rollouts=args.expected_rollouts,
        expected_server_pid=args.expected_server_pid,
    )
    artifact = publish_immutable_json(args.output, result)
    print(json.dumps(artifact, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
