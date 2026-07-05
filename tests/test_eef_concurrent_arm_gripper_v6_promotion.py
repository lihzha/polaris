from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from polaris import eef_concurrent_arm_gripper_v6_promotion as promotion


REPO_ROOT = Path(__file__).parents[1]
IMAGE_ATTESTATION_ARTIFACT = Path(
    os.environ.get(
        "POLARIS_V6_IMAGE_ATTESTATION",
        (
            "/home/lzha/code/ego-lap/.codex_artifacts/"
            "polaris_image_evidence_5c9a2c5_job1098982/"
            "smoke-1098982.image-evidence-attestation.json"
        ),
    )
)
OLD_CONTROLLER_JOB = "1098922"
OLD_CONTROLLER_PRODUCER = "6e4b7c5be5ff6db670970774be3250c5d5ffa4d2"
OLD_CONTROLLER_ATTESTATION_SHA256 = (
    "c359e978bf4aede7555fd3d6118a2abf5f7f4c2e5cf058326d7c3304bda2305a"
)
OLD_PROMOTION_EVIDENCE_SHA256 = (
    "714b22a185ff06135cdc84d03a17347943c405b3d782f3a0141455f0194eb937"
)


def _set_nested(
    value: dict[str, object], path: tuple[str, ...], replacement: object
) -> None:
    cursor: object = value
    for part in path[:-1]:
        assert type(cursor) is dict
        cursor = cursor[part]
    assert type(cursor) is dict
    cursor[path[-1]] = replacement


def _write_controller_attestation(root: Path) -> Path:
    path = root / "smoke-1098975.promotion-attestation.json"
    path.write_bytes(promotion.canonical_controller_smoke_attestation_bytes())
    path.chmod(0o444)
    return path


def _write_image_attestation(root: Path) -> Path:
    if not IMAGE_ATTESTATION_ARTIFACT.is_file():
        pytest.skip(
            "Set POLARIS_V6_IMAGE_ATTESTATION to the exact job-1098982 attestation"
        )
    path = root / "smoke-1098982.image-evidence-attestation.json"
    path.write_bytes(IMAGE_ATTESTATION_ARTIFACT.read_bytes())
    path.chmod(0o444)
    return path


def _all_inherited_source_specs() -> dict[str, str]:
    return {
        **dict(promotion.CONTROLLER_IMPLEMENTATION_SOURCE_SHA256),
        **dict(promotion.IMAGE_IMPLEMENTATION_SOURCE_SHA256),
        **dict(promotion.IMAGE_SMOKE_PRODUCER_SOURCE_SHA256),
        **dict(promotion.PRESERVED_V5_SOURCE_SHA256),
        promotion.CONTROLLER_FINALIZER_PATH: promotion.CONTROLLER_FINALIZER_SHA256,
    }


def _materialize_inherited_sources(root: Path) -> None:
    for relative in _all_inherited_source_specs():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPO_ROOT / relative).read_bytes())


def test_finalized_manifest_is_closed_and_authorizes_only_canaries() -> None:
    evidence = promotion.canonical_eef_concurrent_arm_gripper_v6_promotion_evidence()

    assert promotion.PROMOTION_EVIDENCE_SHA256 == (
        "5a03e09fc5dd8d3c3d196909695d58e7ae589a99d3db9fe730877481858ab301"
    )
    assert set(evidence) == {
        "schema_version",
        "profile",
        "status",
        "lineage",
        "controller_smoke",
        "image_smoke",
        "paired_checkpoint_contract",
        "authorization",
    }
    assert evidence["schema_version"] == 2
    assert evidence["status"] == promotion.FINAL_PROMOTION_STATUS
    assert promotion.image_evidence_finalized() is True
    authorization = evidence["authorization"]
    assert authorization["evidence_finalized"] is True
    assert authorization["canary_authorized"] is True
    assert authorization["allowed_eval_scales"] == ["canary"]
    assert authorization["allowed_tasks"] == ["DROID-FoodBussing"]
    assert authorization["allowed_checkpoint_roles"] == [
        "official_lap3b",
        "reasoning_43075",
    ]
    assert authorization["smoke_suite_authorized"] is False
    assert authorization["standard_authorized"] is False
    assert authorization["native_joint_position_or_pi05_authorized"] is False
    assert authorization["pending_replacement_fields"] == []
    assert (
        promotion.validate_eef_concurrent_arm_gripper_v6_promotion_evidence(evidence)
        == evidence
    )


def test_c5_and_job1098982_identity_is_exact_and_has_no_sentinels() -> None:
    identity = promotion._image_evidence_identity_fields()

    assert set(identity) == {
        "image_evidence_commit",
        "image_evidence_tree",
        "image_evidence_parent",
        "image_evidence_finalizer_sha256",
        "image_evidence_finalizer_size_bytes",
        "image_attestation_path",
        "image_attestation_sha256",
        "image_attestation_size_bytes",
    }
    assert identity["image_evidence_commit"] == (
        "5c9a2c50f564fb58d58777fbe34fb831ba362ec3"
    )
    assert identity["image_evidence_parent"] == promotion.IMAGE_SMOKE_PRODUCER_COMMIT
    assert identity["image_attestation_size_bytes"] == 28_839
    source = Path(promotion.__file__).read_text()
    assert "PENDING_" not in source


def test_no_superseded_controller_authority_survives() -> None:
    source = Path(promotion.__file__).read_text()
    evidence = promotion.canonical_eef_concurrent_arm_gripper_v6_promotion_evidence()

    assert OLD_CONTROLLER_JOB not in source
    assert OLD_CONTROLLER_PRODUCER not in source
    assert OLD_CONTROLLER_ATTESTATION_SHA256 not in source
    assert OLD_PROMOTION_EVIDENCE_SHA256 not in source
    assert evidence["lineage"]["controller_implementation"]["commit"] == (
        promotion.CONTROLLER_IMPLEMENTATION_COMMIT
    )
    assert evidence["lineage"]["controller_implementation"]["parent"] == (
        promotion.CONTROLLER_IMPLEMENTATION_PARENT
    )
    assert (
        evidence["lineage"]["controller_implementation"][
            "historical_parent_is_not_authorized"
        ]
        is True
    )


def test_controller_attestation_is_exact_job1098975_evidence() -> None:
    data = promotion.canonical_controller_smoke_attestation_bytes()
    value = json.loads(data)

    assert len(data) == 11_481
    assert hashlib.sha256(data).hexdigest() == (
        "4b5f53524590711874a06fa3d2f47b1b430df7ff7b82b445d14e44db2c4e1e90"
    )
    promotion._validate_controller_attestation_semantics(value)
    assert value["producer"]["polaris_commit"] == (
        promotion.CONTROLLER_IMPLEMENTATION_COMMIT
    )
    assert value["reviewer"]["evidence_commit"] == (
        promotion.CONTROLLER_EVIDENCE_COMMIT
    )
    assert value["validation_summary"]["total_controller_apply_calls"] == 5856
    assert value["coverage_limits"]["camera_image_contract_validated"] is False


def test_lineage_binds_c1_c2_image_implementation_integration_c4_and_c5() -> None:
    lineage = promotion.canonical_eef_concurrent_arm_gripper_v6_promotion_evidence()[
        "lineage"
    ]

    assert lineage["controller_implementation"] == {
        "commit": "39418400493cdcf8cd8272608980a798f7929a20",
        "tree": "7fc1ff24053e3aeab5ed3e06068089b5aa596bc6",
        "parent": "ee6d09351bed75e32db93ecf59c039a8e99fac9f",
        "historical_parent_is_not_authorized": True,
        "source_sha256": dict(promotion.CONTROLLER_IMPLEMENTATION_SOURCE_SHA256),
    }
    assert lineage["controller_evidence"]["commit"] == (
        "be2f608fd72d1441a777264cd6842f00fd5bf6e8"
    )
    assert lineage["controller_evidence"]["parent"] == (
        promotion.CONTROLLER_IMPLEMENTATION_COMMIT
    )
    assert lineage["image_implementation"]["commit"] == (
        "f1d32a3ca73ef8613b1c3e38f31f70fd06637857"
    )
    assert lineage["image_implementation"]["parent"] == (
        promotion.CONTROLLER_EVIDENCE_COMMIT
    )
    assert lineage["image_integration_tip"]["commit"] == (
        "42e266353df71d5906e98975165f8aa021020dad"
    )
    assert lineage["image_integration_tip"]["runtime_semantics_changed"] is False
    assert lineage["image_smoke_producer"]["commit"] == (
        "9d296361bb323b2e309a3b92a204c102908c61a6"
    )
    assert lineage["image_smoke_producer"]["parent"] == (
        promotion.IMAGE_INTEGRATION_COMMIT
    )
    assert lineage["image_evidence"] == {
        "commit": "5c9a2c50f564fb58d58777fbe34fb831ba362ec3",
        "tree": "707a5d7b659e7c4dfc13d19ede9ce8a8077aeec7",
        "parent": promotion.IMAGE_SMOKE_PRODUCER_COMMIT,
        "expected_parent": promotion.IMAGE_SMOKE_PRODUCER_COMMIT,
        "changed_paths": list(promotion.IMAGE_EVIDENCE_CHANGED_PATHS),
        "finalizer_path": promotion.IMAGE_EVIDENCE_FINALIZER_PATH,
        "finalizer_sha256": promotion.IMAGE_EVIDENCE_FINALIZER_SHA256,
        "finalizer_size_bytes": 81_744,
        "finalized": True,
    }
    assert lineage["promotion_changes_runtime_behavior"] is False


def test_image_contract_binds_real_foodbussing_pixels_and_both_model_routes() -> None:
    contract = promotion.canonical_image_contract_expectation()

    assert contract["environment"] == "DROID-FoodBussing"
    assert contract["camera_keys"] == ["external_cam", "wrist_cam"]
    assert contract["native_shape"] == [720, 1280, 3]
    assert contract["preprocessed_shape"] == [224, 224, 3]
    assert contract["wrist_operation_order"] == [
        "resize_with_pad",
        "rotate_180",
    ]
    assert contract["operation_order_probe_noncommuting"] is True
    assert contract["official_lap3b_route"] == {
        "model_image_keys": ["base_0_rgb", "left_wrist_0_rgb"],
        "model_image_order": ["external", "wrist"],
        "model_image_resolution": [224, 224],
    }
    assert contract["reasoning_43075_route"]["model_image_order"] == [
        "wrist",
        "external",
        "blank",
    ]
    assert contract["reasoning_43075_route"]["blank_image"] == {
        "shape": [224, 224, 3],
        "dtype": "uint8",
        "value": 0,
    }


def test_paired_request_is_exact_flow_global_q99_and_not_pi05() -> None:
    request = promotion.canonical_paired_checkpoint_canary_request()

    assert promotion.PAIRED_CANARY_REQUEST_SHA256 == (
        "6963d2cc0f3c02ee9a4e5f2f3c3718a027bbbfe36a97e884c48e83e94122be28"
    )
    assert promotion.validate_paired_checkpoint_canary_request(request) == request
    assert request["eval_scale"] == "canary"
    assert request["task"] == "DROID-FoodBussing"
    assert request["rollouts_per_checkpoint"] == 1
    assert request["total_rollouts"] == 2
    assert request["native_joint_position_or_pi05_authorized"] is False
    assert request["shared_train_eval_contract"]["policy_client"] == ("EgoLAPEefPose")
    for checkpoint in request["checkpoints"].values():
        assert checkpoint["policy_type"] == "flow"
        assert checkpoint["flow_num_steps"] == 10
        assert checkpoint["response_horizon"] == 16
        assert checkpoint["execution_horizon"] == 8
        assert checkpoint["native_image_resolution"] == [720, 1280]
        assert checkpoint["model_image_resolution"] == [224, 224]
    normalization = request["shared_train_eval_contract"]["normalization"]
    assert normalization == {
        "source": "checkpoint_assets",
        "type": "bounds_q99",
        "scope": "global",
        "configured_policy_category": "single_arm",
        "effective_selected_category": None,
        "compute_dtype": "float32",
        "formula_profile": "q99_train_matched_v1",
        "input_formula_id": "q99_input_eps1e-8_clip_zero0_v1",
        "output_formula_id": "q99_output_eps1e-8_zeroq01_extrapolate_v1",
    }


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("eval_scale",), "smoke_suite"),
        (("eval_scale",), "standard"),
        (("task",), "DROID-PanClean"),
        (("checkpoint_roles",), ["official_lap3b"]),
        (("rollouts_per_checkpoint",), 2),
        (("total_rollouts",), 1),
        (("control_mode",), "joint_position"),
        (("native_joint_position_or_pi05_authorized",), True),
        (("shared_train_eval_contract", "policy_client"), "DroidJointPos"),
        (("checkpoints", "official_lap3b", "policy_type"), "ar"),
        (
            ("checkpoints", "official_lap3b", "model_image_order"),
            ["wrist", "external"],
        ),
        (
            ("checkpoints", "reasoning_43075", "model_image_order"),
            ["external", "wrist", "blank"],
        ),
        (
            ("checkpoints", "reasoning_43075", "model_image_resolution"),
            [256, 256],
        ),
        (
            ("shared_train_eval_contract", "normalization", "scope"),
            "category",
        ),
        (
            (
                "shared_train_eval_contract",
                "normalization",
                "effective_selected_category",
            ),
            "single_arm",
        ),
        (
            (
                "shared_train_eval_contract",
                "wrist_image_preprocessing",
                "operation_order",
            ),
            ["rotate_180", "resize_with_pad"],
        ),
    ],
)
def test_request_rejects_scope_checkpoint_image_normalization_and_pi05_drift(
    path: tuple[str, ...], replacement: object
) -> None:
    request = promotion.canonical_paired_checkpoint_canary_request()
    _set_nested(request, path, replacement)

    with pytest.raises(ValueError, match="request drift"):
        promotion.validate_paired_checkpoint_canary_request(request)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("schema_version",), 1),
        (("lineage", "controller_implementation", "commit"), "0" * 40),
        (("lineage", "controller_evidence", "finalizer_sha256"), "0" * 64),
        (("lineage", "image_implementation", "commit"), "0" * 40),
        (("lineage", "image_smoke_producer", "commit"), "0" * 40),
        (("controller_smoke", "attestation_identity", "sha256"), "0" * 64),
        (("image_smoke", "expected_contract", "native_shape"), [360, 640, 3]),
        (("authorization", "smoke_suite_authorized"), True),
        (("authorization", "standard_authorized"), True),
        (("authorization", "native_joint_position_or_pi05_authorized"), True),
    ],
)
def test_promotion_manifest_rejects_nested_mutation(
    path: tuple[str, ...], replacement: object
) -> None:
    evidence = promotion.canonical_eef_concurrent_arm_gripper_v6_promotion_evidence()
    _set_nested(evidence, path, replacement)

    with pytest.raises(ValueError, match="manifest drift"):
        promotion.validate_eef_concurrent_arm_gripper_v6_promotion_evidence(evidence)


def test_exact_promotion_source_identity_passes() -> None:
    result = promotion.validate_v6_promotion_source_identity(REPO_ROOT)

    assert result["controller_implementation_source_sha256"] == dict(
        promotion.CONTROLLER_IMPLEMENTATION_SOURCE_SHA256
    )
    assert result["image_implementation_source_sha256"] == dict(
        promotion.IMAGE_IMPLEMENTATION_SOURCE_SHA256
    )
    assert result["image_smoke_producer_source_sha256"] == dict(
        promotion.IMAGE_SMOKE_PRODUCER_SOURCE_SHA256
    )
    assert result["controller_evidence"]["finalizer_sha256"] == (
        promotion.CONTROLLER_FINALIZER_SHA256
    )
    assert result["image_evidence"]["finalizer_sha256"] == (
        promotion.IMAGE_EVIDENCE_FINALIZER_SHA256
    )


@pytest.mark.parametrize(
    "relative",
    [
        "src/polaris/eef_controller_repair.py",
        "src/polaris/splat_image_contract.py",
        "src/polaris/splat_renderer/splat_renderer.py",
        "src/polaris/policy/droid_jointpos_client.py",
        "scripts/smoke_splat_image_contract.py",
        promotion.CONTROLLER_FINALIZER_PATH,
        "src/polaris/eef_velocity_recovery_standard_promotion.py",
    ],
)
def test_inherited_source_identity_rejects_each_lineage_class_mutation(
    tmp_path: Path, relative: str
) -> None:
    _materialize_inherited_sources(tmp_path)
    path = tmp_path / relative
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="digest drift"):
        promotion.validate_v6_inherited_source_identity(tmp_path)


def test_inherited_source_identity_rejects_symlink_and_hardlink(
    tmp_path: Path,
) -> None:
    _materialize_inherited_sources(tmp_path)
    path = tmp_path / "scripts/eval.py"
    replacement = tmp_path / "replacement.py"
    replacement.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(replacement)
    with pytest.raises(ValueError, match="not a regular file"):
        promotion.validate_v6_inherited_source_identity(tmp_path)

    path.unlink()
    path.write_bytes(replacement.read_bytes())
    os.link(path, tmp_path / "hardlink.py")
    with pytest.raises(ValueError, match="one hard link"):
        promotion.validate_v6_inherited_source_identity(tmp_path)


def test_exact_controller_attestation_mirror_passes(tmp_path: Path) -> None:
    path = _write_controller_attestation(tmp_path)

    value = promotion.validate_controller_smoke_attestation(
        path, allow_content_addressed_mirror=True
    )
    assert value["producer"]["polaris_commit"] == (
        promotion.CONTROLLER_IMPLEMENTATION_COMMIT
    )


@pytest.mark.parametrize("mutation", ["content", "size", "mode", "hardlink", "symlink"])
def test_controller_attestation_rejects_file_identity_mutation(
    tmp_path: Path, mutation: str
) -> None:
    path = _write_controller_attestation(tmp_path)
    candidate = path
    if mutation == "content":
        data = bytearray(path.read_bytes())
        data[0] = ord("[")
        path.chmod(0o644)
        path.write_bytes(data)
        path.chmod(0o444)
    elif mutation == "size":
        path.chmod(0o644)
        path.write_bytes(path.read_bytes() + b"\n")
        path.chmod(0o444)
    elif mutation == "mode":
        path.chmod(0o640)
    elif mutation == "hardlink":
        candidate = tmp_path / "hardlink.json"
        os.link(path, candidate)
    elif mutation == "symlink":
        candidate = tmp_path / "symlink.json"
        candidate.symlink_to(path)
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match="Controller-smoke attestation"):
        promotion.validate_controller_smoke_attestation(
            candidate, allow_content_addressed_mirror=True
        )


def test_exact_image_attestation_mirror_and_semantics_pass(tmp_path: Path) -> None:
    path = _write_image_attestation(tmp_path)
    data = path.read_bytes()

    assert len(data) == 28_839
    assert hashlib.sha256(data).hexdigest() == promotion.IMAGE_ATTESTATION_SHA256
    value = promotion.validate_image_smoke_attestation(
        path, allow_content_addressed_mirror=True
    )
    assert value["producer"]["commit"] == promotion.IMAGE_SMOKE_PRODUCER_COMMIT
    assert value["reviewer"]["commit"] == promotion.IMAGE_EVIDENCE_COMMIT
    assert value["job"]["job_id"] == "1098982"
    assert (
        value["semantic_evidence"]["contracts"]["renderer_conversion"]["channel_order"]
        == "RGB"
    )
    assert value["authorizations"]["checkpoint_evaluation"] is False

    mutated = json.loads(data)
    mutated["authorizations"]["checkpoint_evaluation"] = True
    with pytest.raises(ValueError, match="standalone-evidence authorization drift"):
        promotion._validate_image_attestation_semantics(mutated)


@pytest.mark.parametrize("mutation", ["content", "size", "mode", "hardlink", "symlink"])
def test_image_attestation_rejects_file_identity_mutation(
    tmp_path: Path, mutation: str
) -> None:
    path = _write_image_attestation(tmp_path)
    candidate = path
    if mutation == "content":
        data = bytearray(path.read_bytes())
        data[0] = ord("[")
        path.chmod(0o644)
        path.write_bytes(data)
        path.chmod(0o444)
    elif mutation == "size":
        path.chmod(0o644)
        path.write_bytes(path.read_bytes() + b"\n")
        path.chmod(0o444)
    elif mutation == "mode":
        path.chmod(0o640)
    elif mutation == "hardlink":
        candidate = tmp_path / "hardlink-image.json"
        os.link(path, candidate)
    elif mutation == "symlink":
        candidate = tmp_path / "symlink-image.json"
        candidate.symlink_to(path)
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match="Image-smoke attestation"):
        promotion.validate_image_smoke_attestation(
            candidate, allow_content_addressed_mirror=True
        )


def test_finalized_c5_identity_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(promotion, "IMAGE_EVIDENCE_COMMIT", "g" * 40)

    assert promotion.image_evidence_finalized() is False
    assert promotion.eef_concurrent_arm_gripper_v6_eval_scale_allowed("canary") is False
    with pytest.raises(ValueError, match="identity drift"):
        promotion.validate_image_smoke_attestation(Path("/missing.json"))
    with pytest.raises(ValueError, match="identity drift"):
        promotion.validate_v6_promotion_source_identity(REPO_ROOT)
    with pytest.raises(RuntimeError, match="identity drift"):
        promotion.canonical_eef_concurrent_arm_gripper_v6_promotion_evidence()


def test_finalized_authorizer_validates_both_attestations_and_only_pair(
    tmp_path: Path,
) -> None:
    controller = _write_controller_attestation(tmp_path)
    image = _write_image_attestation(tmp_path)
    evidence = promotion.canonical_eef_concurrent_arm_gripper_v6_promotion_evidence()
    request = promotion.canonical_paired_checkpoint_canary_request()

    result = promotion.validate_and_authorize_paired_checkpoint_canaries(
        evidence,
        request,
        controller_attestation_path=controller,
        image_attestation_path=image,
        repo_root=REPO_ROOT,
        allow_content_addressed_attestation_mirrors=True,
    )
    assert result["authorized"] is True
    assert result["eval_scale"] == "canary"
    assert result["validation_claims"] == {
        "camera_image_contract_validated": True,
        "image_order_or_resolution_validated": True,
        "checkpoint_loaded": False,
        "normalization_validated": False,
        "policy_serving_validated": False,
        "task_success_metric_validated": False,
    }
    assert result["native_joint_position_or_pi05_authorized"] is False


@pytest.mark.parametrize(
    "payload",
    [
        b'{"duplicate": 1, "duplicate": 1}',
        b'{"nonfinite": NaN}',
        b"[]",
    ],
)
def test_strict_json_rejects_duplicate_nonfinite_and_nonobject(
    payload: bytes,
) -> None:
    with pytest.raises(ValueError):
        promotion._strict_json(payload, "test payload")
