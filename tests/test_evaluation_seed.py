from pathlib import Path
from types import SimpleNamespace

import pytest

from polaris.evaluation_seed import (
    ENVIRONMENT_SEED_CONTRACT_MARKER,
    ENVIRONMENT_SEED_PROFILE,
    MAX_ENVIRONMENT_SEED,
    bind_environment_seed,
    format_environment_seed_contract,
    make_environment_seed_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def test_seed_contract_binds_exact_isaac_config_field():
    env_cfg = SimpleNamespace(seed=None)
    contract = bind_environment_seed(env_cfg, 0)
    assert env_cfg.seed == 0
    assert contract == {
        "schema_version": 1,
        "profile": ENVIRONMENT_SEED_PROFILE,
        "seed": 0,
        "binding": "env_cfg.seed_before_gym_make",
    }
    assert format_environment_seed_contract(contract) == (
        ENVIRONMENT_SEED_CONTRACT_MARKER
        + '{"binding":"env_cfg.seed_before_gym_make",'
        '"profile":"isaaclab_env_cfg_seed_v1","schema_version":1,"seed":0}'
    )


@pytest.mark.parametrize(
    "seed",
    [None, True, False, -1, MAX_ENVIRONMENT_SEED + 1, 0.0, "0"],
)
def test_seed_contract_rejects_implicit_or_noncanonical_values(seed):
    with pytest.raises(ValueError, match="environment_seed"):
        make_environment_seed_contract(seed)


def test_seed_binding_rejects_config_without_seed_field():
    with pytest.raises(ValueError, match="no seed field"):
        bind_environment_seed(SimpleNamespace(), 0)


def test_native_eval_binds_seed_before_environment_construction():
    source = (ROOT / "scripts/eval.py").read_text(encoding="utf-8")
    bind_call = "environment_seed_contract = bind_environment_seed("
    gym_call = "env: ManagerBasedRLSplatEnv = gym.make("
    assert source.index(bind_call) < source.index(gym_call)
    assert 'DroidJointPos requires --environment-seed' in source


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
    assert "POLARIS_ENVIRONMENT_SEED_CONTRACT=" in worker
    assert "ENVIRONMENT_SEED=%q" in worker
    assert "ENVIRONMENT_SEED_PROFILE=isaaclab_env_cfg_seed_v1" in worker
    assert "environment_seed" in submitter
