#!/usr/bin/env python3
"""Serve pinned ``pi05_droid`` inference with an immutable full handshake."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path


def _install_controlled_openpi_path(openpi_dir: Path) -> None:
    if any(
        name == "openpi"
        or name.startswith("openpi.")
        or name == "openpi_client"
        or name.startswith("openpi_client.")
        for name in sys.modules
    ):
        raise RuntimeError("OpenPI was imported before --openpi-dir was bound")
    requested_roots = [
        openpi_dir / "src",
        openpi_dir / "packages/openpi-client/src",
    ]
    for root in requested_roots:
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"Missing regular OpenPI import root: {root}")
    roots = [root.resolve() for root in requested_roots]
    root_strings = [str(root) for root in roots]
    sys.path[:] = [entry for entry in sys.path if entry not in root_strings]
    sys.path[0:0] = root_strings
    importlib.invalidate_caches()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--openpi-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--serving-contract-output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    from polaris.pi05_droid_jointvelocity_contract import (
        attest_imported_openpi_modules,
        expected_pi05_droid_server_metadata,
        publish_immutable_serving_contract,
        verify_openpi_git_checkout,
        verify_pi05_droid_checkpoint,
        verify_profile_source_files,
    )

    checkout = verify_openpi_git_checkout(args.openpi_dir)
    openpi_dir = Path(checkout["root"])
    verify_profile_source_files(openpi_dir)
    _install_controlled_openpi_path(openpi_dir)

    # Imports are intentionally after path installation and the first clean-HEAD gate.
    import jax
    from openpi.models import model as openpi_model
    from openpi.models import pi0 as openpi_pi0
    from openpi.models import tokenizer as openpi_tokenizer
    from openpi.policies import policy as openpi_policy
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config
    import openpi.transforms as openpi_transforms

    if jax.config.x64_enabled:
        raise ValueError("pi05_droid serving requires JAX 64-bit mode to be disabled")
    checkpoint_report = verify_pi05_droid_checkpoint(
        args.checkpoint_dir, args.manifest, full_md5=True
    )

    train_config = config.get_config("pi05_droid")
    static_contract = {
        "name": train_config.name,
        "model_type": train_config.model.model_type.value,
        "pi05": train_config.model.pi05,
        "dtype": train_config.model.dtype,
        "action_horizon": train_config.model.action_horizon,
        "action_dim": train_config.model.action_dim,
        "asset_id": train_config.data.assets.asset_id,
        "policy_metadata": train_config.policy_metadata,
    }
    expected_static_contract = {
        "name": "pi05_droid",
        "model_type": "pi05",
        "pi05": True,
        "dtype": "bfloat16",
        "action_horizon": 15,
        "action_dim": 32,
        "asset_id": "droid",
        "policy_metadata": None,
    }
    if static_contract != expected_static_contract:
        raise ValueError(
            "Resolved OpenPI pi05_droid config mismatch: "
            f"expected {expected_static_contract}, got {static_contract}"
        )

    logging.info("Verified checkpoint: %s", checkpoint_report)
    policy = policy_config.create_trained_policy(
        train_config, args.checkpoint_dir.resolve()
    )
    if policy.metadata:
        raise ValueError(
            f"Official pi05_droid unexpectedly supplied metadata: {policy.metadata}"
        )
    if policy._sample_kwargs != {}:
        raise ValueError(
            f"Official pi05_droid must use default 10-step sampling: {policy._sample_kwargs}"
        )
    if jax.random.key_data(policy._rng).tolist() != [0, 0]:
        raise ValueError("Official pi05_droid policy RNG did not initialize from key 0")

    # Keep explicit references alive and prove these exact modules were imported.
    if openpi_model.__name__ != "openpi.models.model":
        raise RuntimeError("Unexpected OpenPI model module")
    if openpi_tokenizer.__name__ != "openpi.models.tokenizer":
        raise RuntimeError("Unexpected OpenPI tokenizer module")
    if openpi_pi0.__name__ != "openpi.models.pi0":
        raise RuntimeError("Unexpected OpenPI pi0 module")
    if openpi_policy.__name__ != "openpi.policies.policy":
        raise RuntimeError("Unexpected OpenPI policy module")
    if openpi_transforms.__name__ != "openpi.transforms":
        raise RuntimeError("Unexpected OpenPI transforms module")
    verify_openpi_git_checkout(openpi_dir)
    runtime_attestation = attest_imported_openpi_modules(openpi_dir)
    metadata = expected_pi05_droid_server_metadata(runtime_attestation)
    contract_artifact = publish_immutable_serving_contract(
        args.serving_contract_output, metadata
    )
    logging.info("Immutable serving contract: %s", contract_artifact)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
