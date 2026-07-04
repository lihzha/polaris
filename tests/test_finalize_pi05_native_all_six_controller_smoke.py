import hashlib
from pathlib import Path
import subprocess

import pytest

from scripts.polaris import (
    finalize_pi05_droid_native_jointvelocity_eval as canary_finalizer,
)
from scripts.polaris import (
    finalize_pi05_native_all_six_controller_smoke as finalizer,
)
from polaris import pi05_droid_native_eval_contract as native_contract
from polaris.pi05_droid_native_eval_contract import (
    PI05_DROID_ALL_SIX_CONTROLLER_SOURCE_COMMIT,
)


ROOT = Path(__file__).parents[1]


def _git_show(revision: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "-C", ROOT, "show", f"{revision}:{relative}"],
        check=True,
        capture_output=True,
    ).stdout


def test_official_manifest_is_byte_unchanged_and_serve_validation_is_exactly_reviewed():
    for relative in finalizer.UNCHANGED_MODEL_IO_PATHS:
        current = (ROOT / relative).read_bytes()
        accepted = _git_show(PI05_DROID_ALL_SIX_CONTROLLER_SOURCE_COMMIT, relative)
        base = _git_show(finalizer.BASE_COMMIT, relative)
        assert accepted == base
        assert hashlib.sha256(accepted).hexdigest() == hashlib.sha256(base).hexdigest()
        assert current == accepted
    serve_relative = finalizer.REVIEWED_ADDITIVE_MODEL_VALIDATION_PATH
    current_serve = (ROOT / serve_relative).read_bytes()
    accepted_serve = _git_show(
        PI05_DROID_ALL_SIX_CONTROLLER_SOURCE_COMMIT, serve_relative
    )
    base_serve = _git_show(finalizer.BASE_COMMIT, serve_relative)
    assert accepted_serve != current_serve
    assert current_serve != base_serve
    reviewed = finalizer._reviewed_serve_validation(accepted_serve, base_serve)
    assert reviewed["sha256"] == (
        finalizer.REVIEWED_ADDITIVE_MODEL_VALIDATION_SOURCE_SHA256
    )
    assert reviewed["validation_profile"] == (
        finalizer.REVIEWED_ADDITIVE_MODEL_VALIDATION_PROFILE
    )
    assert reviewed["model_semantics_sha256"] == reviewed["base_model_semantics_sha256"]
    descendant = canary_finalizer._validate_atomic_port_server_descendant(
        ROOT,
        serve_relative,
        attested_record=reviewed,
    )
    assert descendant["profile"] == canary_finalizer.ATOMIC_PORT_SERVER_PROFILE
    assert descendant["base_sha256"] == reviewed["sha256"]
    assert (
        subprocess.run(
            ["git", "-C", ROOT, "ls-tree", "HEAD", "third_party/openpi"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()[2]
        == finalizer.OPENPI_COMMIT
    )


def test_reviewed_serve_gate_rejects_validation_or_model_semantic_drift(monkeypatch):
    relative = finalizer.REVIEWED_ADDITIVE_MODEL_VALIDATION_PATH
    current = _git_show(PI05_DROID_ALL_SIX_CONTROLLER_SOURCE_COMMIT, relative)
    base = _git_show(finalizer.BASE_COMMIT, relative)

    validation_drift = current.replace(
        b'"action_horizon": 15', b'"action_horizon": 16', 1
    )
    assert validation_drift != current
    assert finalizer._serve_model_semantic_symbols(
        validation_drift
    ) == finalizer._serve_model_semantic_symbols(base)
    with pytest.raises(ValueError, match="reviewed validation source drift"):
        finalizer._reviewed_serve_validation(validation_drift, base)

    model_drift = current.replace(
        b"train_config, args.checkpoint_dir.resolve()",
        b"train_config, args.checkpoint_dir",
        1,
    )
    assert model_drift != current
    monkeypatch.setattr(
        finalizer,
        "REVIEWED_ADDITIVE_MODEL_VALIDATION_SOURCE_SHA256",
        hashlib.sha256(model_drift).hexdigest(),
    )
    with pytest.raises(ValueError, match="model/checkpoint/server semantics changed"):
        finalizer._reviewed_serve_validation(model_drift, base)


def test_recovery_keeps_policy_input_output_semantic_symbols_identical_to_base():
    relative = "src/polaris/policy/droid_jointvelocity_client.py"
    current = (ROOT / relative).read_bytes()
    base = _git_show(finalizer.BASE_COMMIT, relative)
    assert finalizer._policy_semantic_symbols(
        current, require_gripper_observation_guard=True
    ) == finalizer._policy_semantic_symbols(
        base, require_gripper_observation_guard=False
    )

    transformed = current.replace(b"1.0 + tolerance", b"1.0 + 2 * tolerance", 1)
    with pytest.raises(ValueError, match="raw-gripper observation guard drift"):
        finalizer._policy_semantic_symbols(
            transformed, require_gripper_observation_guard=True
        )


def test_controller_smoke_surface_has_no_model_checkpoint_or_network_path():
    paths = (
        "scripts/smoke_pi05_native_all_six_controller.py",
        "src/polaris/native_all_six_smoke.py",
        "src/polaris/native_gripper_runtime.py",
        "scripts/polaris/l40s_pi05_native_all_six_controller_smoke.sbatch",
    )
    forbidden = (
        "maybe_download",
        "WebsocketClientPolicy",
        "serve_policy",
        "gs://",
        "gcsfs",
    )
    for relative in paths:
        text = (ROOT / relative).read_text()
        assert not any(token in text for token in forbidden)
    wrapper = (ROOT / paths[-1]).read_text()
    assert "#SBATCH --gpus-per-node=1" in wrapper
    assert "#SBATCH --no-requeue" in wrapper
    assert "--environment DROID-FoodBussing" in wrapper
    assert "scope=controller_only_no_model_no_checkpoint" not in wrapper


def test_finalizer_source_allowlist_binds_every_new_runtime_file():
    required = {
        "scripts/smoke_pi05_native_all_six_controller.py",
        "scripts/polaris/finalize_pi05_native_all_six_controller_smoke.py",
        "scripts/polaris/l40s_pi05_native_all_six_controller_smoke.sbatch",
        "scripts/polaris/submit_pi05_native_all_six_controller_smoke.sh",
        "src/polaris/environments/droid_cfg.py",
        "src/polaris/environments/manager_based_rl_splat_environment.py",
        "src/polaris/joint_velocity_runtime.py",
        "src/polaris/native_all_six_smoke.py",
        "src/polaris/native_gripper_runtime.py",
        "src/polaris/pi05_droid_jointvelocity_contract.py",
        "src/polaris/pi05_droid_native_lifecycle.py",
        "src/polaris/policy/droid_jointvelocity_client.py",
    }
    assert required <= set(finalizer.SOURCE_PATHS)
    assert finalizer.UNCHANGED_MODEL_IO_PATHS == (
        "scripts/polaris/pi05_droid_native_gcs_manifest.tsv",
    )
    assert finalizer.REVIEWED_ADDITIVE_MODEL_VALIDATION_PATH not in (
        finalizer.UNCHANGED_MODEL_IO_PATHS
    )


def test_controller_gate_and_canary_consumer_share_exact_validation_profile():
    assert native_contract.PI05_DROID_ALL_SIX_UNCHANGED_POLICY_IO_PATHS == (
        finalizer.UNCHANGED_MODEL_IO_PATHS
    )
    assert native_contract.PI05_DROID_ALL_SIX_REVIEWED_MODEL_VALIDATION_PATHS == (
        finalizer.REVIEWED_ADDITIVE_MODEL_VALIDATION_PATH,
    )
    assert native_contract.PI05_DROID_ALL_SIX_MODEL_VALIDATION_BASE_COMMIT == (
        finalizer.BASE_COMMIT
    )
    assert native_contract.PI05_DROID_ALL_SIX_MODEL_VALIDATION_PROFILE == (
        finalizer.REVIEWED_ADDITIVE_MODEL_VALIDATION_PROFILE
    )
    assert native_contract.PI05_DROID_ALL_SIX_MODEL_VALIDATION_SOURCE_SHA256 == (
        finalizer.REVIEWED_ADDITIVE_MODEL_VALIDATION_SOURCE_SHA256
    )
    assert native_contract.PI05_DROID_ALL_SIX_MODEL_VALIDATION_BASE_SHA256 == (
        finalizer.REVIEWED_ADDITIVE_MODEL_VALIDATION_BASE_SHA256
    )


def test_wrapper_and_finalizer_pin_hub_revision_metadata_for_every_asset():
    wrapper = (
        ROOT / "scripts/polaris/l40s_pi05_native_all_six_controller_smoke.sbatch"
    ).read_text()
    assert finalizer.HUB_REVISION in wrapper
    for relative, expected in finalizer.ASSETS.items():
        assert relative in wrapper
        assert expected["sha256"] in wrapper
        assert expected["metadata_sha256"] in wrapper
