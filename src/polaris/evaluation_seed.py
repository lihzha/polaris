"""Explicit environment-seed binding for reproducible PolaRiS evaluation."""

from __future__ import annotations

import json
from typing import Any


ENVIRONMENT_SEED_CONTRACT_MARKER = "POLARIS_ENVIRONMENT_SEED_CONTRACT="
ENVIRONMENT_SEED_PROFILE = "isaaclab_env_cfg_seed_v1"
MAX_ENVIRONMENT_SEED = 2**32 - 1


def make_environment_seed_contract(seed: int) -> dict[str, Any]:
    """Validate one explicit seed and return its closed provenance contract."""

    if type(seed) is not int or not 0 <= seed <= MAX_ENVIRONMENT_SEED:
        raise ValueError(
            "environment_seed must be an integer in "
            f"[0, {MAX_ENVIRONMENT_SEED}]"
        )
    return {
        "schema_version": 1,
        "profile": ENVIRONMENT_SEED_PROFILE,
        "seed": seed,
        "binding": "env_cfg.seed_before_gym_make",
    }


def bind_environment_seed(env_cfg: Any, seed: int) -> dict[str, Any]:
    """Bind a validated seed to Isaac Lab before environment construction."""

    contract = make_environment_seed_contract(seed)
    if not hasattr(env_cfg, "seed"):
        raise ValueError("Isaac Lab environment config has no seed field")
    env_cfg.seed = seed
    if env_cfg.seed != seed:
        raise ValueError("Isaac Lab environment config did not retain the seed")
    return contract


def format_environment_seed_contract(contract: dict[str, Any]) -> str:
    """Format the exact log marker consumed by launch completion checks."""

    expected = make_environment_seed_contract(contract.get("seed"))
    if contract != expected:
        raise ValueError("Environment seed contract is not canonical")
    return ENVIRONMENT_SEED_CONTRACT_MARKER + json.dumps(
        contract, sort_keys=True, separators=(",", ":")
    )
