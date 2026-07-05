from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest

from polaris import eef_concurrent_arm_gripper_v6_promotion as promotion


REPO_ROOT = Path(__file__).parents[1]


def _set_nested(
    value: dict[str, object], path: tuple[str, ...], replacement: object
) -> None:
    cursor: object = value
    for part in path[:-1]:
        assert type(cursor) is dict
        cursor = cursor[part]
    assert type(cursor) is dict
    cursor[path[-1]] = replacement


def _write_exact_attestation(root: Path) -> Path:
    path = root / "smoke-1098922.promotion-attestation.json"
    path.write_bytes(promotion.canonical_controller_smoke_attestation_bytes())
    path.chmod(0o444)
    return path


def _materialize_source_identity(root: Path) -> None:
    specs = (
        *promotion.PRODUCER_SOURCE_SHA256,
        (promotion.FINALIZER_PATH, promotion.FINALIZER_SHA256),
        *promotion.PRESERVED_V5_SOURCE_SHA256,
    )
    for relative, _ in specs:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPO_ROOT / relative).read_bytes())


def test_embedded_attestation_has_exact_reviewed_byte_and_semantic_identity() -> None:
    data = promotion.canonical_controller_smoke_attestation_bytes()
    value = json.loads(data)

    assert len(data) == 10_423
    assert hashlib.sha256(data).hexdigest() == (
        "c359e978bf4aede7555fd3d6118a2abf5f7f4c2e5cf058326d7c3304bda2305a"
    )
    assert value["producer"]["polaris_commit"] == (
        "6e4b7c5be5ff6db670970774be3250c5d5ffa4d2"
    )
    assert value["reviewer"]["evidence_commit"] == (
        "f4a27ce2bdbbaf2b87a38b4850390f9697ce8f9e"
    )
    assert value["reviewer"]["evidence_tree"] == (
        "4c4ce225bdfd57564e2e90db7657f9dc807a93f8"
    )
    assert value["reviewer"]["finalizer"]["sha256"] == (
        "f9ab24398286d5e4db2af816cfa86c9b0b355c13eeb246e307331b5e14720c4c"
    )
    assert value["sealed_provenance"] == promotion._expected_sealed_provenance()
    assert value["validation_summary"] == promotion._expected_validation_summary()
    assert value["coverage_limits"] == promotion._expected_coverage_limits()


def test_promotion_manifest_has_exact_closed_identity_and_negative_claims() -> None:
    evidence = promotion.canonical_eef_concurrent_arm_gripper_v6_promotion_evidence()

    assert promotion.PROMOTION_EVIDENCE_SHA256 == (
        "714b22a185ff06135cdc84d03a17347943c405b3d782f3a0141455f0194eb937"
    )
    assert (
        promotion.validate_eef_concurrent_arm_gripper_v6_promotion_evidence(evidence)
        == evidence
    )
    assert set(evidence) == {
        "schema_version",
        "profile",
        "status",
        "lineage",
        "controller_smoke",
        "authorization",
    }
    identity = evidence["controller_smoke"]["attestation_identity"]
    assert identity == {
        "path": promotion.CONTROLLER_SMOKE_ATTESTATION_PATH,
        "sha256": promotion.CONTROLLER_SMOKE_ATTESTATION_SHA256,
        "size_bytes": 10_423,
        "mode": "0444",
        "nlink": 1,
    }
    authorization = evidence["authorization"]
    assert authorization["allowed_eval_scales"] == ["canary"]
    assert authorization["canary_authorized"] is True
    assert authorization["smoke_suite_authorized"] is False
    assert authorization["standard_authorized"] is False
    assert not any(authorization["controller_smoke_validation_claims"].values())
    request = authorization["paired_checkpoint_canary_request"]
    assert request["checkpoints"]["official_lap3b"]["revision"] == (
        "601db9c1ab4bcaf6dddb160c7b2dec589a67b730"
    )
    assert request["checkpoints"]["reasoning_43075"]["policy_type"] == "flow"
    normalization = request["shared_train_eval_contract"]["normalization"]
    assert normalization["scope"] == "global"
    assert normalization["policy_category"] == "single_arm"
    assert normalization["effective_selected_category"] is None


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("lineage", "producer_commit"), "0" * 40),
        (("lineage", "evidence_tree"), "0" * 40),
        (("lineage", "finalizer_sha256"), "0" * 64),
        (
            ("lineage", "producer_source_sha256", "scripts/eval.py"),
            "0" * 64,
        ),
        (
            (
                "lineage",
                "preserved_v5_source_sha256",
                "src/polaris/eef_velocity_recovery_promotion.py",
            ),
            "0" * 64,
        ),
        (("controller_smoke", "attestation_identity", "mode"), "0644"),
        (
            (
                "controller_smoke",
                "attestation",
                "sealed_provenance",
                "sacct",
                "sha256",
            ),
            "0" * 64,
        ),
        (
            (
                "controller_smoke",
                "attestation",
                "producer",
                "controller_profile",
            ),
            "wrong-profile",
        ),
        (
            ("controller_smoke", "validation_summary", "total_controller_apply_calls"),
            5855,
        ),
        (
            (
                "controller_smoke",
                "attestation",
                "validation_summary",
                "recovery_events",
            ),
            1,
        ),
        (
            ("controller_smoke", "coverage_limits", "checkpoint_loaded"),
            True,
        ),
        (
            (
                "controller_smoke",
                "attestation",
                "coverage_limits",
                "normalization_validated",
            ),
            True,
        ),
        (("authorization", "allowed_eval_scales"), ["canary", "smoke_suite"]),
        (("authorization", "smoke_suite_authorized"), True),
        (("authorization", "standard_authorized"), True),
        (
            (
                "authorization",
                "controller_smoke_validation_claims",
                "image_order_or_resolution_validated",
            ),
            True,
        ),
    ],
)
def test_promotion_manifest_rejects_nested_mutation(
    path: tuple[str, ...], replacement: object
) -> None:
    evidence = promotion.canonical_eef_concurrent_arm_gripper_v6_promotion_evidence()
    _set_nested(evidence, path, replacement)

    with pytest.raises(ValueError, match="promotion .*drift"):
        promotion.validate_eef_concurrent_arm_gripper_v6_promotion_evidence(evidence)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("schema_version",), True),
        (
            ("controller_smoke", "validation_summary", "ordinary_pose_cases_passed"),
            13.0,
        ),
        (("authorization", "allowed_eval_scales"), ("canary",)),
    ],
)
def test_promotion_manifest_is_type_strict(
    path: tuple[str, ...], replacement: object
) -> None:
    evidence = promotion.canonical_eef_concurrent_arm_gripper_v6_promotion_evidence()
    _set_nested(evidence, path, replacement)

    with pytest.raises(ValueError):
        promotion.validate_eef_concurrent_arm_gripper_v6_promotion_evidence(evidence)


def test_validator_returns_an_independent_copy() -> None:
    evidence = promotion.canonical_eef_concurrent_arm_gripper_v6_promotion_evidence()
    validated = promotion.validate_eef_concurrent_arm_gripper_v6_promotion_evidence(
        evidence
    )
    validated["authorization"]["standard_authorized"] = True

    assert evidence["authorization"]["standard_authorized"] is False
    assert (
        promotion.canonical_eef_concurrent_arm_gripper_v6_promotion_evidence()[
            "authorization"
        ]["standard_authorized"]
        is False
    )


def test_exact_source_finalizer_and_preserved_v5_identities_pass() -> None:
    result = promotion.validate_v6_promotion_source_identity(REPO_ROOT)

    assert result["producer_source_sha256"] == dict(promotion.PRODUCER_SOURCE_SHA256)
    assert result["finalizer_sha256"] == promotion.FINALIZER_SHA256
    assert result["preserved_v5_source_sha256"] == dict(
        promotion.PRESERVED_V5_SOURCE_SHA256
    )


@pytest.mark.parametrize(
    "relative",
    [
        "src/polaris/eef_controller_repair.py",
        promotion.FINALIZER_PATH,
        "src/polaris/eef_velocity_recovery_standard_promotion.py",
    ],
)
def test_source_identity_rejects_each_lineage_class_mutation(
    tmp_path: Path, relative: str
) -> None:
    _materialize_source_identity(tmp_path)
    path = tmp_path / relative
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="digest drift"):
        promotion.validate_v6_promotion_source_identity(tmp_path)


def test_source_identity_rejects_symlink(tmp_path: Path) -> None:
    _materialize_source_identity(tmp_path)
    relative = "scripts/eval.py"
    path = tmp_path / relative
    target = tmp_path / "replacement.py"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)

    with pytest.raises(ValueError, match="not a regular file"):
        promotion.validate_v6_promotion_source_identity(tmp_path)


def test_source_identity_rejects_hardlink(tmp_path: Path) -> None:
    _materialize_source_identity(tmp_path)
    path = tmp_path / "scripts/eval.py"
    os.link(path, tmp_path / "scripts/eval-hardlink.py")

    with pytest.raises(ValueError, match="exactly one hard link"):
        promotion.validate_v6_promotion_source_identity(tmp_path)


def test_exact_immutable_attestation_file_passes(tmp_path: Path) -> None:
    path = _write_exact_attestation(tmp_path)

    assert promotion.validate_controller_smoke_attestation(
        path, allow_content_addressed_mirror=True
    ) == json.loads(promotion.canonical_controller_smoke_attestation_bytes())


def test_attestation_defaults_to_literal_pinned_result_path(tmp_path: Path) -> None:
    path = _write_exact_attestation(tmp_path)

    with pytest.raises(ValueError, match="path drift"):
        promotion.validate_controller_smoke_attestation(path)


@pytest.mark.parametrize("mutation", ["content", "size", "mode", "hardlink", "symlink"])
def test_attestation_file_rejects_identity_mutation(
    tmp_path: Path, mutation: str
) -> None:
    path = _write_exact_attestation(tmp_path)
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
    else:  # pragma: no cover - parameter list is closed above
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match="attestation"):
        promotion.validate_controller_smoke_attestation(
            candidate, allow_content_addressed_mirror=True
        )


@pytest.mark.parametrize(
    "payload",
    [
        b'{"duplicate": 1, "duplicate": 1}',
        b'{"nonfinite": NaN}',
        b"[]",
    ],
)
def test_strict_json_rejects_duplicate_nonfinite_and_nonobject(payload: bytes) -> None:
    with pytest.raises(ValueError):
        promotion._strict_json(payload, "test payload")


def test_only_exact_paired_one_rollout_foodbussing_request_is_valid() -> None:
    request = promotion.canonical_paired_checkpoint_canary_request()

    assert promotion.validate_paired_checkpoint_canary_request(request) == request
    assert request == {
        "eval_scale": "canary",
        "stage": "paired_official_and_reasoning_foodbussing_canaries",
        "benchmark": "polaris_droid_suite_v1",
        "task": "DROID-FoodBussing",
        "checkpoint_roles": ["official_lap3b", "reasoning_43075"],
        "checkpoints": {
            "official_lap3b": {
                "uri": ("hf://lihzha/LAP-3B@601db9c1ab4bcaf6dddb160c7b2dec589a67b730"),
                "revision": "601db9c1ab4bcaf6dddb160c7b2dec589a67b730",
                "content_manifest_sha256": (
                    "567cc3ff7d20f3f03913a6f11c3fa151f789e1c0118ed5af0eea24d9cc48f20e"
                ),
                "checkpoint_profile": "original_lap_public_3b_v1",
                "policy_type": "flow",
                "flow_num_steps": 10,
                "response_horizon": 16,
                "execution_horizon": 8,
                "model_image_keys": ["base_0_rgb", "left_wrist_0_rgb"],
                "model_image_order": ["external", "wrist"],
                "legacy_image_order": True,
                "image_resolution": [224, 224],
                "state_encoding": "EEF_R6",
                "state_layout": "xyz+r6_first_two_rows+gripper_open",
                "state_layout_mode": "public_lap_train_matched_rows_v1",
                "frame_description": "robot base frame",
            },
            "reasoning_43075": {
                "uri": (
                    "gs://v6_east1d/checkpoints/lap_oxe_magic_soup_reasoning_full/"
                    "oxe_magic_soup_reasoning_full_v2_flow_pred0_cf0_ckpt25_"
                    "v6_32_b512_s42_20260630/43075"
                ),
                "step": 43075,
                "inference_subset_profile": "policy-inference-params-assets-v1",
                "inference_subset_sha256": (
                    "bb9ea5bb041f689a08f914cac7dfe5d061c822ddbe87e292f9c7878a9d3bfc4d"
                ),
                "checkpoint_profile": "manifest_v1_canonical",
                "policy_type": "flow",
                "flow_num_steps": 10,
                "response_horizon": 16,
                "execution_horizon": 8,
                "model_image_keys": [
                    "camera_0_rgb",
                    "camera_1_rgb",
                    "camera_2_rgb",
                ],
                "model_image_order": ["wrist", "external", "blank"],
                "legacy_image_order": False,
                "image_resolution": [224, 224],
                "state_encoding": "EEF_R6",
                "state_layout": "xyz+r6_first_two_columns+gripper_open",
                "state_layout_mode": "manifest_train_matched_columns_v1",
                "frame_description": "egocentric frame",
            },
        },
        "shared_train_eval_contract": {
            "image_color_space": "RGB",
            "image_dtype": "uint8",
            "wrist_image_preprocessing": {
                "operation_order": ["resize_with_pad", "rotate_180"],
                "rotation_degrees": 180,
            },
            "normalization": {
                "source": "checkpoint_assets",
                "type": "bounds_q99",
                "scope": "global",
                "policy_category": "single_arm",
                "effective_selected_category": None,
                "compute_dtype": "float32",
                "formula_profile": "q99_train_matched_v1",
                "input_formula_id": "q99_input_eps1e-8_clip_zero0_v1",
                "output_formula_id": ("q99_output_eps1e-8_zeroq01_extrapolate_v1"),
            },
            "response_semantics": "cumulative_delta_targets",
            "numeric_action_frame": "robot_base",
        },
        "rollouts_per_checkpoint": 1,
        "total_rollouts": 2,
        "control_mode": "absolute_end_effector_pose",
        "eef_frame": "panda_link8_relative_to_panda_link0",
        "controller_profile": promotion.CONTROLLER_PROFILE,
        "ik_safety_profile": promotion.IK_SAFETY_PROFILE,
    }


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("eval_scale", "smoke_suite"),
        ("eval_scale", "standard"),
        ("task", "DROID-PanClean"),
        ("checkpoint_roles", ["official_lap3b"]),
        ("checkpoint_roles", ["reasoning_43075", "official_lap3b"]),
        ("rollouts_per_checkpoint", 2),
        ("rollouts_per_checkpoint", True),
        ("total_rollouts", 1),
        ("control_mode", "joint_position"),
    ],
)
def test_canary_request_rejects_scale_pair_task_count_and_type_drift(
    field: str, replacement: object
) -> None:
    request = promotion.canonical_paired_checkpoint_canary_request()
    request[field] = replacement

    with pytest.raises(ValueError, match="request drift"):
        promotion.validate_paired_checkpoint_canary_request(request)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("checkpoints", "official_lap3b", "revision"), "0" * 40),
        (
            ("checkpoints", "official_lap3b", "content_manifest_sha256"),
            "0" * 64,
        ),
        (
            ("checkpoints", "official_lap3b", "model_image_order"),
            ["wrist", "external"],
        ),
        (("checkpoints", "official_lap3b", "legacy_image_order"), False),
        (
            ("checkpoints", "official_lap3b", "state_layout_mode"),
            "manifest_train_matched_columns_v1",
        ),
        (("checkpoints", "reasoning_43075", "uri"), "gs://wrong/43075"),
        (
            ("checkpoints", "reasoning_43075", "inference_subset_sha256"),
            "0" * 64,
        ),
        (("checkpoints", "reasoning_43075", "policy_type"), "ar"),
        (
            ("checkpoints", "reasoning_43075", "model_image_order"),
            ["external", "wrist", "blank"],
        ),
        (("checkpoints", "reasoning_43075", "image_resolution"), [256, 256]),
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
            ("shared_train_eval_contract", "normalization", "formula_profile"),
            "q99_legacy_upstream_v1",
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
def test_canary_request_rejects_checkpoint_and_train_eval_contract_drift(
    path: tuple[str, ...], replacement: object
) -> None:
    request = promotion.canonical_paired_checkpoint_canary_request()
    _set_nested(request, path, replacement)

    with pytest.raises(ValueError, match="request drift"):
        promotion.validate_paired_checkpoint_canary_request(request)


@pytest.mark.parametrize(
    ("eval_scale", "allowed"),
    [
        ("canary", True),
        ("smoke_suite", False),
        ("standard", False),
        ("unknown", False),
        (True, False),
    ],
)
def test_eval_scale_gate_explicitly_rejects_smoke_suite_and_standard(
    eval_scale: object, allowed: bool
) -> None:
    assert (
        promotion.eef_concurrent_arm_gripper_v6_eval_scale_allowed(eval_scale)
        is allowed
    )


def test_fresh_checks_authorize_only_the_next_canary_pair(tmp_path: Path) -> None:
    attestation_path = _write_exact_attestation(tmp_path)
    evidence = promotion.canonical_eef_concurrent_arm_gripper_v6_promotion_evidence()
    request = promotion.canonical_paired_checkpoint_canary_request()

    result = promotion.validate_and_authorize_paired_checkpoint_canaries(
        evidence,
        request,
        attestation_path=attestation_path,
        repo_root=REPO_ROOT,
        allow_content_addressed_attestation_mirror=True,
    )

    assert result["authorized"] is True
    assert result["eval_scale"] == "canary"
    assert result["stage"] == promotion.NEXT_STAGE
    assert result["request"] == request
    assert result["content_addressed_attestation_mirror_used"] is True
    assert result["verified_attestation_path"] == str(attestation_path)
    assert result["checkpoint_image_normalization_and_task_validation_claimed"] is False


def test_authorizer_rejects_mutated_evidence_before_launch(tmp_path: Path) -> None:
    attestation_path = _write_exact_attestation(tmp_path)
    evidence = promotion.canonical_eef_concurrent_arm_gripper_v6_promotion_evidence()
    request = promotion.canonical_paired_checkpoint_canary_request()
    mutated = deepcopy(evidence)
    mutated["authorization"]["standard_authorized"] = True

    with pytest.raises(ValueError, match="promotion .*drift"):
        promotion.validate_and_authorize_paired_checkpoint_canaries(
            mutated,
            request,
            attestation_path=attestation_path,
            repo_root=REPO_ROOT,
            allow_content_addressed_attestation_mirror=True,
        )
