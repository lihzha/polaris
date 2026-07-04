import copy
import hashlib
import json
from pathlib import Path

import pytest

from polaris.pi05_droid_jointvelocity_contract import (
    NATIVE_GRIPPER_DRIVE_PROFILE,
    NATIVE_GRIPPER_EFFORT_LIMIT,
    NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S,
    PI05_DROID_CHECKPOINT_MANIFEST_SHA256,
    PI05_DROID_CONTRACT_FILENAME,
    PI05_DROID_CONTRACT_METADATA_KEY,
    PI05_DROID_GRIPPER_OBSERVATION_BOUND_TOLERANCE,
    PI05_DROID_GRIPPER_OBSERVATION_CONTRACT,
    PI05_DROID_JOINTVELOCITY_PROFILE,
    PI05_DROID_OPENPI_INFERENCE_COMPATIBILITY_COMMIT,
    PI05_DROID_POLARIS_RUNTIME_SOURCE_SHA256,
    attest_imported_openpi_modules,
    contract_sha256,
    expected_pi05_droid_jointvelocity_contract,
    expected_pi05_droid_server_metadata,
    publish_immutable_serving_contract,
    reference_openpi_runtime_attestation,
    serving_contract_bytes,
    validate_openpi_runtime_attestation,
    validate_persisted_serving_contract,
    validate_pi05_droid_server_metadata,
    verify_openpi_git_checkout,
    verify_profile_manifest,
    verify_profile_source_files,
)


ROOT = Path(__file__).parents[1]


def test_exact_contract_binds_checkpoint_norm_openpi_action_and_control():
    contract = expected_pi05_droid_jointvelocity_contract()

    assert contract["profile"] == PI05_DROID_JOINTVELOCITY_PROFILE
    assert contract["checkpoint"] == {
        "uri": "gs://openpi-assets/checkpoints/pi05_droid",
        "content_manifest_profile": "gcs_path_size_md5_v1",
        "content_manifest_sha256": PI05_DROID_CHECKPOINT_MANIFEST_SHA256,
        "object_count": 20,
        "total_bytes": 12_429_488_598,
    }
    assert contract["normalization"]["sha256"] == (
        "403b3a22f897e9ae5dd617966a3c8f7d1835ac79dfd5a8993179514be26a3b8b"
    )
    assert contract["normalization"]["scope"] == "checkpoint_global_droid"
    assert contract["openpi"]["inference_compatibility_commit"] == (
        PI05_DROID_OPENPI_INFERENCE_COMPATIBILITY_COMMIT
    )
    assert (
        contract["openpi"]["training_revision_provenance"]
        == "unavailable_in_released_checkpoint"
    )
    assert (
        contract["openpi"]["runtime_attestation"]["compatibility_role"]
        == "inference_only_not_training_provenance"
    )
    assert contract["openpi"]["model_action_horizon"] == 15
    assert contract["policy_output"]["response_shape"] == [15, 8]
    assert contract["policy_output"]["execute_first"] == 8
    assert contract["policy_input"]["gripper_observation"] == (
        PI05_DROID_GRIPPER_OBSERVATION_CONTRACT
    )
    assert PI05_DROID_GRIPPER_OBSERVATION_BOUND_TOLERANCE == 8 * 2.0**-23
    assert PI05_DROID_GRIPPER_OBSERVATION_BOUND_TOLERANCE <= 1e-6
    assert contract["policy_input"]["gripper_observation"]["raw_value_preserved"]
    assert (
        contract["policy_input"]["gripper_observation"][
            "server_pre_normalization_transform"
        ]
        == "none"
    )
    assert contract["control"]["position_integration"] == "forbidden"
    assert contract["control"]["action_cfg"] == (
        "polaris_AuditedDroidJointVelocityActionCfg"
    )
    assert contract["control"]["action_cfg_base"] == ("isaaclab_JointVelocityActionCfg")
    assert contract["control"]["velocity_drive"]["position_stiffness"] == 0.0
    assert contract["control"]["gripper_drive"]["profile"] == (
        NATIVE_GRIPPER_DRIVE_PROFILE
    )
    assert (
        contract["control"]["gripper_drive"]["configured"]["velocity_limit_sim"]
        == NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S
    )
    assert (
        contract["control"]["gripper_drive"]["configured"]["effort_limit_sim"]
        == NATIVE_GRIPPER_EFFORT_LIMIT
    )
    assert contract["control"]["gripper_drive"]["live"] == {
        "actuator_cuda": {
            "device": "cuda:0",
            "dtype": "torch.float32",
            "shape": [1, 1],
            "stiffness": 5729.578125,
            "damping": 0.011459155939519405,
            "effort_limit": NATIVE_GRIPPER_EFFORT_LIMIT,
            "effort_limit_sim": NATIVE_GRIPPER_EFFORT_LIMIT,
            "velocity_limit": NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S,
            "velocity_limit_sim": NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S,
        },
        "direct_physx_cpu": {
            "device": "cpu",
            "dtype": "torch.float32",
            "shape": [1, 1],
            "stiffness": 5729.578125,
            "damping": 0.011459155939519405,
            "effort_limit": NATIVE_GRIPPER_EFFORT_LIMIT,
            "velocity_limit": NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S,
        },
    }
    assert contract["artifact"]["filename"] == PI05_DROID_CONTRACT_FILENAME
    assert contract["contract_sha256"] == contract_sha256(contract)


def test_eval_profile_intent_is_required_only_for_native_jointvelocity():
    from polaris.config import EvalArgs, PolicyArgs

    joint_position = EvalArgs(
        policy=PolicyArgs(), environment="DROID-FoodBussing", run_folder="unused"
    )
    assert joint_position.control_mode == "joint-position"
    assert joint_position.expected_gripper_drive_profile is None

    eval_source = (ROOT / "scripts/eval.py").read_text(encoding="utf-8")
    assert "joint-velocity expected gripper drive profile mismatch" in eval_source
    assert (
        "expected gripper drive profile is valid only for joint-velocity control"
        in eval_source
    )


def test_polaris_runtime_source_manifest_matches_the_exact_local_sources():
    source_paths = {
        "droid_cfg.py": ROOT / "src/polaris/environments/droid_cfg.py",
        "robot_cfg.py": ROOT / "src/polaris/environments/robot_cfg.py",
        "native_gripper_runtime.py": ROOT / "src/polaris/native_gripper_runtime.py",
        "manager_based_rl_splat_environment.py": (
            ROOT / "src/polaris/environments/manager_based_rl_splat_environment.py"
        ),
    }
    assert set(source_paths) == set(PI05_DROID_POLARIS_RUNTIME_SOURCE_SHA256)
    actual = {
        filename: hashlib.sha256(source_paths[filename].read_bytes()).hexdigest()
        for filename in PI05_DROID_POLARIS_RUNTIME_SOURCE_SHA256
    }
    assert actual == PI05_DROID_POLARIS_RUNTIME_SOURCE_SHA256


def test_server_metadata_rejects_contract_attestation_and_top_level_tampering():
    metadata = expected_pi05_droid_server_metadata()
    validated = validate_pi05_droid_server_metadata(metadata)
    assert validated == metadata[PI05_DROID_CONTRACT_METADATA_KEY]
    assert validated is not metadata[PI05_DROID_CONTRACT_METADATA_KEY]

    tampered = copy.deepcopy(metadata)
    tampered[PI05_DROID_CONTRACT_METADATA_KEY]["checkpoint"]["uri"] += "_wrong"
    with pytest.raises(ValueError, match="SHA-256 is invalid"):
        validate_pi05_droid_server_metadata(tampered)

    extra = copy.deepcopy(metadata)
    extra["unbound"] = True
    with pytest.raises(ValueError, match="handshake schema mismatch"):
        validate_pi05_droid_server_metadata(extra)

    attestation = reference_openpi_runtime_attestation()
    model_record = next(
        record
        for record in attestation["imported_modules"]
        if record["relative_path"] == "src/openpi/models/model.py"
    )
    model_record["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="critical imported source digest mismatch"):
        validate_openpi_runtime_attestation(attestation)


def test_full_handshake_is_persisted_once_as_exact_immutable_bytes(tmp_path):
    metadata = expected_pi05_droid_server_metadata()
    path = tmp_path / PI05_DROID_CONTRACT_FILENAME
    identity = publish_immutable_serving_contract(path, metadata)

    assert path.read_bytes() == serving_contract_bytes(metadata)
    assert (
        identity["contract_sha256"]
        == metadata[PI05_DROID_CONTRACT_METADATA_KEY]["contract_sha256"]
    )
    assert identity["mode"] == "0444"
    assert validate_persisted_serving_contract(path, metadata) == identity
    with pytest.raises(FileExistsError):
        publish_immutable_serving_contract(path, metadata)

    path.chmod(0o644)
    path.write_text(json.dumps(metadata), encoding="ascii")
    path.chmod(0o444)
    with pytest.raises(ValueError, match="canonical JSON"):
        validate_persisted_serving_contract(path, metadata)


def test_manifest_pinned_checkout_and_openpi_sources_validate():
    report = verify_profile_manifest(
        ROOT / "scripts/polaris/pi05_droid_native_gcs_manifest.tsv"
    )
    assert report == {
        "sha256": PI05_DROID_CHECKPOINT_MANIFEST_SHA256,
        "object_count": 20,
        "total_bytes": 12_429_488_598,
    }
    openpi_dir = ROOT / "third_party/openpi"
    checkout = verify_openpi_git_checkout(openpi_dir)
    assert checkout["git_head"] == PI05_DROID_OPENPI_INFERENCE_COMPATIBILITY_COMMIT
    assert checkout["git_tracked_and_untracked_clean"] is True
    source_report = verify_profile_source_files(openpi_dir)
    assert "examples/droid/main.py" in source_report
    assert "src/openpi/models/model.py" in source_report
    assert "src/openpi/models/tokenizer.py" in source_report
    assert "src/openpi/serving/websocket_policy_server.py" in source_report


def test_real_server_imports_attest_leaf_sources_and_namespace_origins():
    from openpi.models import model, pi0, tokenizer
    from openpi.policies import policy, policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config
    import openpi.transforms as transforms

    for module in (
        model,
        pi0,
        tokenizer,
        policy,
        policy_config,
        websocket_policy_server,
        config,
        transforms,
    ):
        assert module.__file__
    attestation = attest_imported_openpi_modules(ROOT / "third_party/openpi")
    imported_paths = {
        record["relative_path"] for record in attestation["imported_modules"]
    }
    assert "src/openpi/models/model.py" in imported_paths
    assert "src/openpi/models/tokenizer.py" in imported_paths
    assert "src/openpi/serving/websocket_policy_server.py" in imported_paths
    namespace_modules = {
        record["module"] for record in attestation["namespace_packages"]
    }
    assert {"openpi.policies", "openpi.serving", "openpi.training"}.issubset(
        namespace_modules
    )
