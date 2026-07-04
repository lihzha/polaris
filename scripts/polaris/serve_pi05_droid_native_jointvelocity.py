#!/usr/bin/env python3
"""Serve pinned ``pi05_droid`` inference with an immutable full handshake."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

from polaris.pi05_droid_native_eval_contract import (
    PI05_DROID_NATIVE_CANARY_PROFILE,
    PI05_DROID_NATIVE_MODEL_EVAL_CONTRACT,
    PI05_DROID_NATIVE_TRANSFORM_RUNTIME_CONTRACT,
    canonical_json_bytes,
    publish_immutable_json,
)
from polaris.pi05_droid_bound_port import BoundPortWebsocketPolicyServer


def _class_path(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _require_exact_json(value: object, expected: object, message: str) -> None:
    if canonical_json_bytes(value) != canonical_json_bytes(expected):
        raise ValueError(message)


def validate_official_pi05_train_config(train_config: object) -> dict[str, object]:
    """Bind the static checkpoint architecture with JSON type identity."""

    try:
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
    except AttributeError as error:
        raise ValueError("Resolved OpenPI pi05_droid config is incomplete") from error
    expected = {
        "name": "pi05_droid",
        "model_type": "pi05",
        "pi05": True,
        "dtype": "bfloat16",
        "action_horizon": 15,
        "action_dim": 32,
        "asset_id": "droid",
        "policy_metadata": None,
    }
    _require_exact_json(
        static_contract,
        expected,
        f"Resolved OpenPI pi05_droid config mismatch: expected {expected}, got {static_contract}",
    )
    return static_contract


def validate_official_pi05_policy_runtime(
    *, metadata: object, sample_kwargs: object, rng_key_data: object
) -> dict[str, object]:
    """Bind empty metadata/default sampler and the exact integer JAX key 0."""

    observed = {
        "metadata": metadata,
        "sample_kwargs": sample_kwargs,
        "rng_key_data": rng_key_data,
    }
    expected = {"metadata": {}, "sample_kwargs": {}, "rng_key_data": [0, 0]}
    _require_exact_json(
        observed,
        expected,
        f"Official pi05_droid policy runtime mismatch: {observed}",
    )
    return observed


def validate_official_pi05_data_config(data_config: object) -> dict[str, object]:
    """Reject delta/absolute or joint-position transforms before weight load."""

    data = getattr(data_config, "data_transforms", None)
    model = getattr(data_config, "model_transforms", None)
    repack = getattr(data_config, "repack_transforms", None)
    if any(group is None for group in (data, model, repack)):
        raise ValueError("pi05_droid transform groups are missing")

    def paths(group: object, field: str) -> list[str]:
        values = getattr(group, field, None)
        if type(values) not in (list, tuple):
            raise ValueError(f"pi05_droid transform group {field} is not a sequence")
        return [_class_path(value) for value in values]

    observed = {
        "asset_id": getattr(data_config, "asset_id", None),
        "use_quantile_norm": getattr(data_config, "use_quantile_norm", None),
        "repack_inputs": paths(repack, "inputs"),
        "repack_outputs": paths(repack, "outputs"),
        "data_inputs": paths(data, "inputs"),
        "data_outputs": paths(data, "outputs"),
        "model_inputs": paths(model, "inputs"),
        "model_outputs": paths(model, "outputs"),
        "sequence_types": {
            "repack_inputs": type(repack.inputs).__name__,
            "repack_outputs": type(repack.outputs).__name__,
            "data_inputs": type(data.inputs).__name__,
            "data_outputs": type(data.outputs).__name__,
            "model_inputs": type(model.inputs).__name__,
            "model_outputs": type(model.outputs).__name__,
        },
    }
    expected = {
        key: PI05_DROID_NATIVE_TRANSFORM_RUNTIME_CONTRACT[key] for key in observed
    }
    _require_exact_json(
        observed,
        expected,
        f"Official pi05_droid transform pipeline mismatch: {observed}",
    )
    data_input = data.inputs[0]
    resize = model.inputs[1]
    tokenizer = model.inputs[2]
    padding = model.inputs[3]
    if (
        getattr(getattr(data_input, "model_type", None), "value", None) != "pi05"
        or getattr(model.inputs[0], "prompt", "missing") is not None
        or _class_path(getattr(tokenizer, "tokenizer", None))
        != "openpi.models.tokenizer.PaligemmaTokenizer"
        or getattr(tokenizer, "discrete_state_input", None) is not True
    ):
        raise ValueError("Official pi05_droid transform parameters mismatch")
    _require_exact_json(
        [getattr(resize, "height", None), getattr(resize, "width", None)],
        [224, 224],
        "Official pi05_droid transform parameters mismatch",
    )
    _require_exact_json(
        getattr(padding, "model_action_dim", None),
        32,
        "Official pi05_droid transform parameters mismatch",
    )
    report = {
        **observed,
        "droid_input_model_type": "pi05",
        "resize": [224, 224],
        "tokenizer": "openpi.models.tokenizer.PaligemmaTokenizer",
        "discrete_state_input": True,
        "model_action_dim": 32,
        "forbidden_transforms_absent": [
            "openpi.transforms.DeltaActions",
            "openpi.transforms.AbsoluteActions",
        ],
        "output_projection": "DroidOutputs_leading8",
    }
    _require_exact_json(
        report,
        PI05_DROID_NATIVE_TRANSFORM_RUNTIME_CONTRACT,
        "Official pi05_droid full transform contract mismatch",
    )
    return report


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
    parser.add_argument("--model-runtime-contract-output", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--bound-port-output", type=Path, required=True)
    parser.add_argument("--bound-port-token", required=True)
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
    static_contract = validate_official_pi05_train_config(train_config)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    transform_contract = validate_official_pi05_data_config(data_config)
    logging.info("Verified pi05_droid transform contract: %s", transform_contract)

    logging.info("Verified checkpoint: %s", checkpoint_report)
    policy = policy_config.create_trained_policy(
        train_config, args.checkpoint_dir.resolve()
    )
    policy_runtime = validate_official_pi05_policy_runtime(
        metadata=policy.metadata,
        sample_kwargs=policy._sample_kwargs,
        rng_key_data=jax.random.key_data(policy._rng).tolist(),
    )

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

    def publish_listener_artifacts(actual_port: int) -> None:
        logging.info(
            "WebSocket listener owns OS-assigned port %d; publishing contracts",
            actual_port,
        )
        contract_artifact = publish_immutable_serving_contract(
            args.serving_contract_output, metadata
        )
        logging.info("Immutable serving contract: %s", contract_artifact)
        model_runtime_artifact = publish_immutable_json(
            args.model_runtime_contract_output,
            {
                "schema_version": 1,
                "profile": PI05_DROID_NATIVE_CANARY_PROFILE,
                "status": "pass",
                "checkpoint": checkpoint_report,
                "train_config": static_contract,
                "transform_runtime": transform_contract,
                "policy": policy_runtime,
                "official_model_eval_contract": PI05_DROID_NATIVE_MODEL_EVAL_CONTRACT,
                "openpi_runtime_attestation": runtime_attestation,
            },
        )
        logging.info("Immutable model runtime contract: %s", model_runtime_artifact)

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
