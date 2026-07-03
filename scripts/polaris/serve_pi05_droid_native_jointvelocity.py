#!/usr/bin/env python3
"""Serve the pinned official ``pi05_droid`` policy with exact metadata."""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path


def _git_head(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--openpi-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config
    import jax

    from polaris.pi05_droid_jointvelocity_contract import (
        PI05_DROID_OPENPI_COMMIT,
        expected_pi05_droid_server_metadata,
        verify_pi05_droid_checkpoint,
        verify_profile_source_files,
    )

    openpi_dir = args.openpi_dir.resolve()
    if _git_head(openpi_dir) != PI05_DROID_OPENPI_COMMIT:
        raise ValueError(
            f"OpenPI must be exactly {PI05_DROID_OPENPI_COMMIT}; "
            f"got {_git_head(openpi_dir)}"
        )
    verify_profile_source_files(openpi_dir)
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
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=expected_pi05_droid_server_metadata(),
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
