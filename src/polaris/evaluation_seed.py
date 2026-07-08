"""Explicit Isaac Lab seed control and provenance for PolaRiS evaluation."""

from __future__ import annotations

import hashlib
import json
from typing import Any


ENVIRONMENT_SEED_CONTRACT_MARKER = "POLARIS_PI05_DROID_ENVIRONMENT_CONTRACT="
ENVIRONMENT_SEED_PROFILE = "isaaclab_env_seed_base_plus_episode_v1"
ENVIRONMENT_SEED_SCHEME = "base_plus_episode_index_v1"
ENVIRONMENT_DETERMINISM_CLAIM = "rng_bound_not_bitwise"
MAX_ENVIRONMENT_SEED = 2**32 - 1


def _uint32(value: Any, name: str) -> int:
    if type(value) is not int or not 0 <= value <= MAX_ENVIRONMENT_SEED:
        raise ValueError(f"{name} must be an integer in [0, {MAX_ENVIRONMENT_SEED}]")
    return value


def episode_environment_seed(base_seed: int, episode_index: int) -> int:
    """Derive a resume-invariant logical RNG seed for one episode."""

    base_seed = _uint32(base_seed, "environment base seed")
    episode_index = _uint32(episode_index, "episode index")
    episode_seed = base_seed + episode_index
    if episode_seed > MAX_ENVIRONMENT_SEED:
        raise ValueError("derived episode seed exceeds the uint32 range")
    return episode_seed


def validate_episode_seed_range(base_seed: int, episode_count: int) -> int:
    """Prevalidate every episode-derived seed before simulator construction."""

    base_seed = _uint32(base_seed, "environment base seed")
    if type(episode_count) is not int or episode_count <= 0:
        raise ValueError("episode count must be a positive integer")
    return episode_environment_seed(base_seed, episode_count - 1)


def bind_environment_seed(env_cfg: Any, base_seed: int) -> None:
    """Bind a validated base seed before Isaac Lab constructs the environment."""

    base_seed = _uint32(base_seed, "environment base seed")
    if not hasattr(env_cfg, "seed"):
        raise ValueError("Isaac Lab environment config has no seed field")
    env_cfg.seed = base_seed
    if env_cfg.seed != base_seed:
        raise ValueError("Isaac Lab environment config did not retain the seed")


def _canonical_live_contract(contract: dict[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "profile",
        "base_seed",
        "scheme",
        "live_cfg_seed",
        "physx_enhanced_determinism",
        "determinism_claim",
        "binding",
    }
    if not isinstance(contract, dict) or set(contract) != expected_keys:
        raise ValueError("Environment seed contract fields are not canonical")
    base_seed = _uint32(contract.get("base_seed"), "environment base seed")
    live_cfg_seed = _uint32(contract.get("live_cfg_seed"), "live cfg seed")
    if live_cfg_seed != base_seed:
        raise ValueError("Live Isaac Lab seed differs from the configured base seed")
    if contract.get("schema_version") != 1:
        raise ValueError("Environment seed contract schema mismatch")
    if contract.get("profile") != ENVIRONMENT_SEED_PROFILE:
        raise ValueError("Environment seed profile mismatch")
    if contract.get("scheme") != ENVIRONMENT_SEED_SCHEME:
        raise ValueError("Environment seed scheme mismatch")
    if contract.get("determinism_claim") != ENVIRONMENT_DETERMINISM_CLAIM:
        raise ValueError("Environment determinism claim mismatch")
    if contract.get("binding") != (
        "env_cfg_seed_before_gym_make_and_reset_seed_per_episode"
    ):
        raise ValueError("Environment seed binding mismatch")
    if type(contract.get("physx_enhanced_determinism")) is not bool:
        raise ValueError("PhysX enhanced-determinism flag must be boolean")
    return dict(contract)


def make_live_environment_seed_contract(env: Any, base_seed: int) -> dict[str, Any]:
    """Read back the constructed environment and bind its live seed settings."""

    base_seed = _uint32(base_seed, "environment base seed")
    live_env = getattr(env, "unwrapped", env)
    cfg = getattr(live_env, "cfg", None)
    if cfg is None:
        raise ValueError("Live Isaac Lab environment has no cfg")
    live_cfg_seed = getattr(cfg, "seed", None)
    physx = getattr(getattr(cfg, "sim", None), "physx", None)
    if physx is None or not hasattr(physx, "enable_enhanced_determinism"):
        raise ValueError("Live Isaac Lab environment has no PhysX determinism flag")
    return _canonical_live_contract(
        {
            "schema_version": 1,
            "profile": ENVIRONMENT_SEED_PROFILE,
            "base_seed": base_seed,
            "scheme": ENVIRONMENT_SEED_SCHEME,
            "live_cfg_seed": live_cfg_seed,
            "physx_enhanced_determinism": physx.enable_enhanced_determinism,
            "determinism_claim": ENVIRONMENT_DETERMINISM_CLAIM,
            "binding": "env_cfg_seed_before_gym_make_and_reset_seed_per_episode",
        }
    )


def validate_live_environment_seed_contract(
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Return a defensive copy of one exact live environment contract."""

    return _canonical_live_contract(contract)


def environment_seed_contract_sha256(contract: dict[str, Any]) -> str:
    """Hash one canonical live seed contract for trace cross-binding."""

    canonical = validate_live_environment_seed_contract(contract)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def make_episode_environment_rng(
    live_contract: dict[str, Any], episode_index: int
) -> dict[str, Any]:
    """Create query provenance for one episode under the bound seed scheme."""

    live = validate_live_environment_seed_contract(live_contract)
    episode_index = _uint32(episode_index, "episode index")
    return {
        "schema_version": 2,
        "profile": live["profile"],
        "base_seed": live["base_seed"],
        "scheme": live["scheme"],
        "episode_index": episode_index,
        "episode_seed": episode_environment_seed(live["base_seed"], episode_index),
        "live_cfg_seed": live["live_cfg_seed"],
        "physx_enhanced_determinism": live["physx_enhanced_determinism"],
        "determinism_claim": live["determinism_claim"],
        "environment_seed_contract_sha256": environment_seed_contract_sha256(live),
    }


def validate_episode_environment_rng(
    value: dict[str, Any],
    live_contract: dict[str, Any],
    episode_index: int,
) -> dict[str, Any]:
    """Require exact per-episode provenance and return a defensive copy."""

    expected = make_episode_environment_rng(live_contract, episode_index)
    if value != expected:
        raise ValueError("Episode environment RNG provenance mismatch")
    return dict(value)


def format_environment_seed_contract(contract: dict[str, Any]) -> str:
    """Format the exact live marker consumed by completion checks."""

    canonical = validate_live_environment_seed_contract(contract)
    return ENVIRONMENT_SEED_CONTRACT_MARKER + json.dumps(
        canonical, sort_keys=True, separators=(",", ":")
    )
