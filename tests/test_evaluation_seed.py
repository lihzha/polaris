from pathlib import Path
from types import SimpleNamespace

import pytest

from polaris.evaluation_seed import (
    ENVIRONMENT_DETERMINISM_CLAIM,
    ENVIRONMENT_SEED_CONTRACT_MARKER,
    ENVIRONMENT_SEED_PROFILE,
    ENVIRONMENT_SEED_SCHEME,
    MAX_ENVIRONMENT_SEED,
    bind_environment_seed,
    episode_environment_seed,
    format_environment_seed_contract,
    make_episode_environment_rng,
    make_live_environment_seed_contract,
    validate_episode_environment_rng,
)


ROOT = Path(__file__).resolve().parents[1]


def _env(seed=0, enhanced_determinism=False):
    cfg = SimpleNamespace(
        seed=seed,
        sim=SimpleNamespace(
            physx=SimpleNamespace(
                enable_enhanced_determinism=enhanced_determinism
            )
        ),
    )
    return SimpleNamespace(unwrapped=SimpleNamespace(cfg=cfg))


def test_seed_contract_binds_and_reads_back_live_isaac_config():
    env_cfg = SimpleNamespace(seed=None)
    bind_environment_seed(env_cfg, 0)
    assert env_cfg.seed == 0
    contract = make_live_environment_seed_contract(_env(), 0)
    assert contract == {
        "schema_version": 1,
        "profile": ENVIRONMENT_SEED_PROFILE,
        "base_seed": 0,
        "scheme": ENVIRONMENT_SEED_SCHEME,
        "live_cfg_seed": 0,
        "physx_enhanced_determinism": False,
        "determinism_claim": ENVIRONMENT_DETERMINISM_CLAIM,
        "binding": "env_cfg_seed_before_gym_make_and_reset_seed_per_episode",
    }
    marker = format_environment_seed_contract(contract)
    assert marker.startswith(ENVIRONMENT_SEED_CONTRACT_MARKER)
    assert '"base_seed":0' in marker
    assert '"determinism_claim":"rng_bound_not_bitwise"' in marker


def test_episode_seed_is_resume_invariant_and_bound_in_query_provenance():
    contract = make_live_environment_seed_contract(_env(seed=11), 11)
    assert episode_environment_seed(11, 0) == 11
    assert episode_environment_seed(11, 7) == 18
    episode_rng = make_episode_environment_rng(contract, 7)
    assert episode_rng["episode_seed"] == 18
    assert episode_rng["episode_index"] == 7
    assert validate_episode_environment_rng(episode_rng, contract, 7) == episode_rng


@pytest.mark.parametrize(
    "base_seed,episode_index",
    [
        (None, 0),
        (True, 0),
        (-1, 0),
        (MAX_ENVIRONMENT_SEED + 1, 0),
        (0.0, 0),
        ("0", 0),
        (0, -1),
        (0, False),
        (MAX_ENVIRONMENT_SEED, 1),
    ],
)
def test_seed_contract_rejects_implicit_or_out_of_range_values(
    base_seed, episode_index
):
    with pytest.raises(ValueError):
        episode_environment_seed(base_seed, episode_index)


def test_live_seed_readback_and_episode_provenance_fail_closed():
    with pytest.raises(ValueError, match="differs"):
        make_live_environment_seed_contract(_env(seed=1), 0)
    with pytest.raises(ValueError, match="no seed field"):
        bind_environment_seed(SimpleNamespace(), 0)
    contract = make_live_environment_seed_contract(_env(), 0)
    wrong = make_episode_environment_rng(contract, 0)
    wrong["episode_seed"] = 1
    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_episode_environment_rng(wrong, contract, 0)


def test_native_eval_binds_seed_before_environment_construction_and_reads_live():
    source = (ROOT / "scripts/eval.py").read_text(encoding="utf-8")
    bind_call = "bind_environment_seed(env_cfg, eval_args.environment_seed)"
    gym_call = "env: ManagerBasedRLSplatEnv = gym.make("
    live_call = "environment_seed_contract = make_live_environment_seed_contract("
    assert source.index(bind_call) < source.index(gym_call) < source.index(live_call)
    assert 'DroidJointPos requires --environment-seed' in source
    assert "seed=episode_seed" in source


def test_native_launcher_persists_and_completion_gates_seed_contract():
    worker = (
        ROOT / "scripts/polaris/eval_pi05_droid_jointpos_polaris.sh"
    ).read_text(encoding="utf-8")
    submitter = (
        ROOT / "scripts/polaris/submit_pi05_droid_jointpos_polaris.sh"
    ).read_text(encoding="utf-8")
    sbatch = (ROOT / "scripts/polaris/l40s_pi05_eval_job.sbatch").read_text(
        encoding="utf-8"
    )
    for source in (worker, submitter, sbatch):
        assert 'ENVIRONMENT_SEED="${ENVIRONMENT_SEED:-0}"' in source
    assert "--environment-seed" in worker
    assert "POLARIS_PI05_DROID_ENVIRONMENT_CONTRACT=" in worker
    assert "--expected-environment-seed" in worker
    assert "Seed not set for the environment" in worker
    assert "Seed-bound native evaluation forbids resume" in worker
    assert "ENVIRONMENT_SEED_SCHEME=%q" in worker
    assert "PHYSX_ENHANCED_DETERMINISM=false" in worker
    assert "foodbussing50" in submitter


def test_same_gpu_repeat_submitter_is_two_fresh_sequential_runs():
    repeat_job = (
        ROOT / "scripts/polaris/l40s_pi05_jointpos_seed_repeat.sbatch"
    ).read_text(encoding="utf-8")
    repeat_submitter = (
        ROOT / "scripts/polaris/submit_pi05_jointpos_seed_repeat.sh"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --gpus-per-node=1" in repeat_job
    assert "for repeat_id in a b" in repeat_job
    assert "repeat${repeat_id}" in repeat_job
    assert "eval_pi05_droid_jointpos_polaris.sh" in repeat_job
    assert "--array" not in repeat_job + repeat_submitter
    assert "sbatch --parsable" in repeat_submitter
