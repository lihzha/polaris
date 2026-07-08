#!/usr/bin/env python3
"""Serve the released PolaRiS pi0.5 checkpoint with a closed live contract."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import os
from pathlib import Path
import signal
import sys

from polaris.pi05_droid_jointpos_serving_contract import (
    PI05_DROID_JOINTPOS_BIND_HOST,
    PI05_DROID_JOINTPOS_METADATA_KEY,
    PI05_DROID_JOINTPOS_REQUIRED_RUNTIME_ENVIRONMENT,
    attest_imported_openpi_modules,
    capture_openpi_host_runtime,
    expected_pi05_droid_jointpos_server_metadata,
    make_pi05_droid_jointpos_rng_stream_report,
    make_pi05_droid_jointpos_model_runtime,
    publish_pi05_droid_jointpos_model_runtime,
    publish_pi05_droid_jointpos_rng_stream,
    publish_pi05_droid_jointpos_serving_contract,
    validate_official_data_config,
    validate_official_policy_runtime,
    validate_official_train_config,
    verify_paligemma_tokenizer_artifact,
    verify_openpi_git_checkout,
    verify_openpi_package_environment,
    verify_pi05_droid_jointpos_checkpoint,
)


def _install_controlled_openpi_path(openpi_dir: Path) -> None:
    if any(
        name == "openpi"
        or name.startswith("openpi.")
        or name == "openpi_client"
        or name.startswith("openpi_client.")
        for name in sys.modules
    ):
        raise RuntimeError("OpenPI was imported before --openpi-dir was bound")
    raw_roots = [
        openpi_dir / "src",
        openpi_dir / "packages/openpi-client/src",
    ]
    if any(root.is_symlink() or not root.is_dir() for root in raw_roots):
        raise ValueError("Controlled OpenPI source roots are missing")
    roots = [root.resolve() for root in raw_roots]
    root_strings = [str(root) for root in roots]
    sys.path[:] = [entry for entry in sys.path if entry not in root_strings]
    sys.path[0:0] = root_strings
    importlib.invalidate_caches()


async def _run_official_server_until_quiesced(
    server: object, shutdown_requested: asyncio.Event
) -> None:
    """Cancel official ``run`` and await its listener/handler drain barrier."""

    server_task = asyncio.create_task(server.run(), name="official-openpi-server")
    shutdown_task = asyncio.create_task(
        shutdown_requested.wait(), name="pi05-rng-shutdown-signal"
    )
    try:
        done, _pending = await asyncio.wait(
            (server_task, shutdown_task), return_when=asyncio.FIRST_COMPLETED
        )
        if server_task in done:
            await server_task
            raise RuntimeError("Official OpenPI server stopped before SIGUSR1")
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
        if not server_task.done():
            raise RuntimeError("Official OpenPI server did not quiesce")
    finally:
        shutdown_task.cancel()
        try:
            await shutdown_task
        except asyncio.CancelledError:
            pass
        if not server_task.done():
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass


async def _serve_then_publish_final_rng(
    *, server: object, publish_final_rng_snapshot: object
) -> None:
    shutdown_requested = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGUSR1, shutdown_requested.set)
    try:
        await _run_official_server_until_quiesced(server, shutdown_requested)
        # WebSocket Server.__aexit__ has now closed the listener and waited for
        # every connection handler; no later Policy.infer call can mutate _rng.
        publish_final_rng_snapshot()
    finally:
        loop.remove_signal_handler(signal.SIGUSR1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--openpi-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--serving-contract-output", type=Path, required=True)
    parser.add_argument("--model-runtime-contract-output", type=Path, required=True)
    parser.add_argument("--rng-stream-output", type=Path, required=True)
    parser.add_argument("--expected-request-count", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("WebSocket port must be in [1, 65535]")
    if args.expected_request_count <= 0:
        raise ValueError("Expected policy request count must be positive")
    required_runtime_environment = {
        key: os.environ.get(key)
        for key in PI05_DROID_JOINTPOS_REQUIRED_RUNTIME_ENVIRONMENT
    }
    if required_runtime_environment != PI05_DROID_JOINTPOS_REQUIRED_RUNTIME_ENVIRONMENT:
        raise ValueError("Required OpenPI runtime environment differs before imports")

    checkout = verify_openpi_git_checkout(args.openpi_dir)
    openpi_dir = Path(checkout["root"])
    _install_controlled_openpi_path(openpi_dir)
    preimport_package_environment = verify_openpi_package_environment(openpi_dir)

    # These imports must stay after the controlled-path gate above.
    import jax
    from openpi_client import image_tools as openpi_client_image_tools
    from openpi.models import model as openpi_model
    from openpi.models import pi0 as openpi_pi0
    from openpi.models import pi0_config as openpi_pi0_config
    from openpi.models import tokenizer as openpi_tokenizer
    from openpi.policies import droid_policy as openpi_droid_policy
    from openpi.policies import policy as openpi_policy
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.shared import image_tools as openpi_shared_image_tools
    from openpi.shared import normalize as openpi_normalize
    from openpi.training import checkpoints as openpi_checkpoints
    from openpi.training import config
    from openpi.training.misc import polaris_config as openpi_polaris_config
    import openpi.transforms as openpi_transforms

    if jax.config.x64_enabled:
        raise ValueError("Attested pi0.5 serving requires JAX x64 disabled")
    checkpoint_report = verify_pi05_droid_jointpos_checkpoint(
        args.checkpoint_dir, args.manifest
    )
    train_config = config.get_config("pi05_droid_jointpos_polaris")
    train_report = validate_official_train_config(train_config)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    data_report = validate_official_data_config(data_config)

    # No overrides: this is the exact official checkpoint/config construction.
    policy = policy_config.create_trained_policy(
        train_config, args.checkpoint_dir.resolve()
    )
    policy_report = validate_official_policy_runtime(
        policy=policy,
        jax_module=jax,
        expected_norm_values_sha256=checkpoint_report["normalization"]["values_sha256"],
    )
    tokenizer_artifact = verify_paligemma_tokenizer_artifact(openpi_tokenizer.download)

    required_modules = (
        openpi_model,
        openpi_pi0,
        openpi_pi0_config,
        openpi_tokenizer,
        openpi_droid_policy,
        openpi_policy,
        openpi_client_image_tools,
        openpi_shared_image_tools,
        openpi_normalize,
        openpi_checkpoints,
        openpi_polaris_config,
        openpi_transforms,
        websocket_policy_server,
    )
    if any(
        not module.__name__.startswith(("openpi.", "openpi_client."))
        for module in required_modules
    ):
        raise RuntimeError("Unexpected OpenPI module namespace")
    verify_openpi_git_checkout(openpi_dir)
    runtime_attestation = attest_imported_openpi_modules(openpi_dir)
    host_runtime = capture_openpi_host_runtime(
        openpi_dir,
        jax,
        preimport_package_environment=preimport_package_environment,
    )
    metadata = expected_pi05_droid_jointpos_server_metadata(runtime_attestation)
    if not metadata:
        raise RuntimeError("Attested WebSocket metadata must be nonempty")
    model_runtime = make_pi05_droid_jointpos_model_runtime(
        checkpoint=checkpoint_report,
        train_config=train_report,
        data_config=data_report,
        policy_runtime=policy_report,
        openpi_checkout=checkout,
        openpi_runtime_attestation=runtime_attestation,
        host_runtime=host_runtime,
        tokenizer_artifact=tokenizer_artifact,
        expected_request_count=args.expected_request_count,
        serving_metadata=metadata,
    )
    serving_artifact = publish_pi05_droid_jointpos_serving_contract(
        args.serving_contract_output, metadata
    )
    runtime_artifact = publish_pi05_droid_jointpos_model_runtime(
        args.model_runtime_contract_output, model_runtime, metadata
    )
    logging.info("Immutable serving contract: %s", serving_artifact)
    logging.info("Immutable model-runtime contract: %s", runtime_artifact)

    def publish_final_rng_snapshot() -> None:
        final_key_data = jax.random.key_data(policy._rng).tolist()

        @jax.jit
        def advance_rng(key):
            return jax.lax.fori_loop(
                0,
                args.expected_request_count,
                lambda _index, current: jax.random.split(current)[0],
                key,
            )

        expected_key = advance_rng(jax.random.key(0))
        expected_key.block_until_ready()
        expected_key_data = jax.random.key_data(expected_key).tolist()
        if final_key_data != expected_key_data:
            raise ValueError(
                "Final policy RNG key does not match the exact expected "
                f"request count {args.expected_request_count}: "
                f"actual={final_key_data}, expected={expected_key_data}"
            )
        rng_report = make_pi05_droid_jointpos_rng_stream_report(
            server_pid=os.getpid(),
            initial_key_data=policy_report["initial_jax_key_data"],
            final_key_data=final_key_data,
            expected_final_key_data=expected_key_data,
            expected_request_count=args.expected_request_count,
            metadata_contract_sha256=metadata[PI05_DROID_JOINTPOS_METADATA_KEY][
                "contract_sha256"
            ],
        )
        artifact = publish_pi05_droid_jointpos_rng_stream(
            args.rng_stream_output, rng_report
        )
        logging.info("Immutable final RNG stream: %s", artifact)

    # Use the unmodified official server and unwrapped official policy.
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=PI05_DROID_JOINTPOS_BIND_HOST,
        port=args.port,
        metadata=metadata,
    )
    try:
        asyncio.run(
            _serve_then_publish_final_rng(
                server=server,
                publish_final_rng_snapshot=publish_final_rng_snapshot,
            )
        )
    except BaseException:
        logging.exception("Final RNG-stream attestation or official serving failed")
        sys.stdout.flush()
        sys.stderr.flush()
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
