import copy
from pathlib import Path

import pytest

from polaris.pi05_droid_jointvelocity_contract import (
    PI05_DROID_CHECKPOINT_MANIFEST_SHA256,
    PI05_DROID_JOINTVELOCITY_PROFILE,
    contract_sha256,
    expected_pi05_droid_jointvelocity_contract,
    expected_pi05_droid_server_metadata,
    validate_pi05_droid_server_metadata,
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
    assert contract["openpi"]["commit"] == ("bd70b8f4011e85b3f3b0f039f12113f78718e7bf")
    assert contract["openpi"]["model_action_horizon"] == 15
    assert contract["policy_output"]["response_shape"] == [15, 8]
    assert contract["policy_output"]["execute_first"] == 8
    assert contract["control"]["position_integration"] == "forbidden"
    assert contract["control"]["velocity_drive"]["position_stiffness"] == 0.0
    assert contract["contract_sha256"] == contract_sha256(contract)


def test_server_metadata_rejects_any_contract_or_top_level_tampering():
    metadata = expected_pi05_droid_server_metadata()
    validated = validate_pi05_droid_server_metadata(metadata)
    assert validated == metadata["polaris_pi05_droid_contract"]
    assert validated is not metadata["polaris_pi05_droid_contract"]

    tampered = copy.deepcopy(metadata)
    tampered["polaris_pi05_droid_contract"]["checkpoint"]["uri"] += "_wrong"
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_pi05_droid_server_metadata(tampered)

    extra = copy.deepcopy(metadata)
    extra["unbound"] = True
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_pi05_droid_server_metadata(extra)


def test_manifest_and_pinned_openpi_sources_validate():
    report = verify_profile_manifest(
        ROOT / "scripts/polaris/pi05_droid_native_gcs_manifest.tsv"
    )
    assert report == {
        "sha256": PI05_DROID_CHECKPOINT_MANIFEST_SHA256,
        "object_count": 20,
        "total_bytes": 12_429_488_598,
    }
    source_report = verify_profile_source_files(ROOT / "third_party/openpi")
    assert "examples/droid/main.py" in source_report
    assert "src/openpi/transforms.py" in source_report
