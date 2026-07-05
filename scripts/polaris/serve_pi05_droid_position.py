#!/usr/bin/env python3
"""Serve official ``pi05_droid`` with the position-adapter handshake."""

from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path
import sys

from polaris.pi05_droid_bound_port import BoundPortWebsocketPolicyServer
from polaris.pi05_droid_native_eval_contract import (
    publish_immutable_json,
)
from polaris.pi05_droid_position_adapter import PI05_DROID_POSITION_ADAPTER_PROFILE
from polaris.pi05_droid_position_contract import (
    attest_imported_openpi_modules,
    expected_pi05_droid_position_server_metadata,
    PI05_DROID_POSITION_MODEL_EVAL_CONTRACT,
    PI05_DROID_POSITION_TRANSFORM_RUNTIME_CONTRACT,
    publish_immutable_position_serving_contract,
    verify_official_droid_git_checkout,
    verify_openpi_git_checkout,
    verify_pi05_droid_checkpoint,
    verify_profile_source_files,
)


MODEL_RUNTIME_PROFILE = "openpi_pi05_droid_position_model_runtime_v1"


def _load_sealed_model_helpers():
    # This script sits beside the previously reviewed model/checkpoint server.
    # Reuse only its model-side validators; the metadata and actuator profile
    # are built and published by this distinct position path.
    import serve_pi05_droid_native_jointvelocity as sealed

    return sealed


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
        raise ValueError("controlled OpenPI source roots are missing")
    roots = [root.resolve() for root in raw_roots]
    root_strings = [str(root) for root in roots]
    sys.path[:] = [entry for entry in sys.path if entry not in root_strings]
    sys.path[0:0] = root_strings
    importlib.invalidate_caches()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--openpi-dir", type=Path, required=True)
    parser.add_argument("--droid-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--serving-contract-output", type=Path, required=True)
    parser.add_argument("--model-runtime-contract-output", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--bound-port-output", type=Path, required=True)
    parser.add_argument("--bound-port-token", required=True)
    args = parser.parse_args()

    checkout = verify_openpi_git_checkout(args.openpi_dir)
    openpi_dir = Path(checkout["root"])
    verify_profile_source_files(openpi_dir)
    droid_source_report = verify_official_droid_git_checkout(args.droid_dir)
    _install_controlled_openpi_path(openpi_dir)

    import jax
    from openpi.models import model as openpi_model
    from openpi.models import pi0 as openpi_pi0
    from openpi.models import tokenizer as openpi_tokenizer
    from openpi.policies import policy as openpi_policy
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config
    import openpi.transforms as openpi_transforms

    sealed = _load_sealed_model_helpers()
    if jax.config.x64_enabled:
        raise ValueError("pi05_droid serving requires JAX 64-bit mode disabled")
    checkpoint_report = verify_pi05_droid_checkpoint(
        args.checkpoint_dir, args.manifest, full_md5=True
    )
    train_config = config.get_config("pi05_droid")
    static_contract = sealed.validate_official_pi05_train_config(train_config)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    transform_contract = sealed.validate_official_pi05_data_config(data_config)
    if transform_contract != PI05_DROID_POSITION_TRANSFORM_RUNTIME_CONTRACT:
        raise ValueError("position model transform contract mismatch")
    policy = policy_config.create_trained_policy(
        train_config, args.checkpoint_dir.resolve()
    )
    policy_runtime = sealed.validate_official_pi05_policy_runtime(
        metadata=policy.metadata,
        sample_kwargs=policy._sample_kwargs,
        rng_key_data=jax.random.key_data(policy._rng).tolist(),
    )
    modules = (
        openpi_model,
        openpi_pi0,
        openpi_tokenizer,
        openpi_policy,
        openpi_transforms,
    )
    if any(not module.__name__.startswith("openpi.") for module in modules):
        raise RuntimeError("unexpected OpenPI module origin")
    verify_openpi_git_checkout(openpi_dir)
    runtime_attestation = attest_imported_openpi_modules(openpi_dir)
    metadata = expected_pi05_droid_position_server_metadata(runtime_attestation)

    def publish_listener_artifacts(actual_port: int) -> None:
        logging.info("position server owns port %d", actual_port)
        publish_immutable_position_serving_contract(
            args.serving_contract_output, metadata
        )
        publish_immutable_json(
            args.model_runtime_contract_output,
            {
                "schema_version": 1,
                "profile": MODEL_RUNTIME_PROFILE,
                "position_execution_profile": PI05_DROID_POSITION_ADAPTER_PROFILE,
                "checkpoint": checkpoint_report,
                "train_config": static_contract,
                "transform_runtime": transform_contract,
                "policy": policy_runtime,
                "official_model_eval_contract": PI05_DROID_POSITION_MODEL_EVAL_CONTRACT,
                "transform_contract_reference": PI05_DROID_POSITION_TRANSFORM_RUNTIME_CONTRACT,
                "official_droid_source": droid_source_report,
                "openpi_runtime_attestation": runtime_attestation,
            },
        )

    server = BoundPortWebsocketPolicyServer(
        websocket_policy_server=websocket_policy_server,
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
        bound_port_output=args.bound_port_output,
        launch_token=args.bound_port_token,
        publish_listener_artifacts=publish_listener_artifacts,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
