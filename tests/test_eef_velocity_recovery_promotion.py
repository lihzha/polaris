from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from polaris import eef_velocity_recovery_promotion as promotion


def test_promotion_manifest_has_exact_closed_identity() -> None:
    evidence = promotion.canonical_eef_velocity_recovery_v5_promotion_evidence()

    assert promotion.PROMOTION_EVIDENCE_SHA256 == (
        "9576a178253741571a50cd23fe8a16b75b9a386ced5bc43ee416348fa52454f7"
    )
    assert (
        promotion.validate_eef_velocity_recovery_v5_promotion_evidence(evidence)
        == evidence
    )
    assert set(evidence) == {
        "schema_version",
        "profile",
        "status",
        "producer",
        "canaries",
        "inspection",
        "authorization",
    }
    assert evidence["producer"]["ego_lap_commit"] == (
        "74bb225d07ccdd2408bc568fe900709e633047e6"
    )
    assert evidence["producer"]["polaris_commit"] == (
        "f11ae45a64b2f839dcb3325459ab06776d1dd81a"
    )
    assert evidence["producer"]["polaris_tree"] == (
        "7d68beea046e485dfe622a4a41c9e03d3a423ef2"
    )
    assert evidence["producer"]["controller_semantics_changed_by_promotion"] is False


def test_exact_producer_sources_remain_unchanged() -> None:
    repo_root = Path(__file__).parents[1]

    assert promotion.validate_producer_source_identity(repo_root) == (
        promotion.PRODUCER_SOURCE_SHA256
    )


def test_official_and_reasoning_canaries_bind_full_horizon_artifacts() -> None:
    canaries = promotion.canonical_eef_velocity_recovery_v5_promotion_evidence()[
        "canaries"
    ]
    official = canaries["official_lap3b"]
    reasoning = canaries["reasoning_43075"]

    for canary in canaries.values():
        assert canary["policy_steps"] == 450
        assert canary["apply_calls"] == 3600
        assert canary["numerical_failures"] == 0
        assert canary["raw_successes"] == 0
        assert canary["task_valid_successes"] == 0
        assert set(canary["artifacts"]) == {
            "runtime_contract_sha256",
            "episode_sidecar_sha256",
            "finalized_trace_sha256",
            "rollout_video_sha256",
            "completion_audit_sha256",
            "registry_candidate_sha256",
            "suite_summary_sha256",
            "summary_video_sha256",
        }

    assert official["recovery"]["counters"] == {
        "current_hard_limit_aborts": 0,
        "hold_target_applies": 0,
        "lower_endpoint_transition_aborts": 0,
        "predicted_limit_aborts": 0,
        "recovered_events": 0,
        "recovery_active_substeps": 0,
        "recovery_events": 0,
        "release_ramp_target_applies": 0,
        "residual_events": 0,
        "residual_joints": 0,
        "sustained_aborts": 0,
        "transaction_aborts": 0,
    }
    assert official["recovery"]["events"] == []
    assert official["recovery"]["maxima"] == {
        "abs_velocity_residual_excess_rad_s": [0.0] * 7,
        "abs_velocity_to_limit_ratio": 0.9338167309761047,
        "consecutive_recovery_substeps": 0,
    }
    assert reasoning["recovery"]["counters"] == {
        "current_hard_limit_aborts": 0,
        "hold_target_applies": 4,
        "lower_endpoint_transition_aborts": 0,
        "predicted_limit_aborts": 0,
        "recovered_events": 1,
        "recovery_active_substeps": 4,
        "recovery_events": 1,
        "release_ramp_target_applies": 16,
        "residual_events": 2,
        "residual_joints": 2,
        "sustained_aborts": 0,
        "transaction_aborts": 0,
    }
    assert reasoning["recovery"]["maxima"] == {
        "abs_velocity_residual_excess_rad_s": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.14167523384094238,
        ],
        "abs_velocity_to_limit_ratio": 1.0542817115783691,
        "consecutive_recovery_substeps": 4,
    }
    assert reasoning["recovery"]["events"] == [
        {
            "event_index": 0,
            "start_apply_index": 2386,
            "start_policy_step": 298,
            "start_physics_substep": 2,
            "start_reason": "measured_velocity_above_float32_envelope",
            "start_velocity_to_limit_ratio": 1.0542817115783691,
            "start_velocity_residual_excess_rad_s": 0.14167523384094238,
            "end_apply_index": 2404,
            "end_policy_step": 300,
            "end_physics_substep": 4,
            "end_reason": "clean2_release_ramp_complete",
            "recovery_completed_apply_index": 2404,
        }
    ]
    assert official["protocol_variant"] != reasoning["protocol_variant"]
    assert "r6rows-q99repair" in official["protocol_variant"]
    assert "canary1-q99repair" in reasoning["protocol_variant"]


@pytest.mark.parametrize("canary", ["official_lap3b", "reasoning_43075"])
def test_promotion_manifest_rejects_single_canary_omission(canary: str) -> None:
    evidence = promotion.canonical_eef_velocity_recovery_v5_promotion_evidence()
    evidence["canaries"].pop(canary)

    with pytest.raises(ValueError, match="promotion evidence drift"):
        promotion.validate_eef_velocity_recovery_v5_promotion_evidence(evidence)


@pytest.mark.parametrize(
    "field",
    [
        "runtime_contract_sha256",
        "episode_sidecar_sha256",
        "finalized_trace_sha256",
        "rollout_video_sha256",
        "completion_audit_sha256",
        "registry_candidate_sha256",
        "suite_summary_sha256",
        "summary_video_sha256",
    ],
)
def test_promotion_manifest_rejects_cross_job_artifact_swap(field: str) -> None:
    evidence = promotion.canonical_eef_velocity_recovery_v5_promotion_evidence()
    official = evidence["canaries"]["official_lap3b"]["artifacts"]
    reasoning = evidence["canaries"]["reasoning_43075"]["artifacts"]
    official[field], reasoning[field] = reasoning[field], official[field]

    with pytest.raises(ValueError, match="promotion evidence drift"):
        promotion.validate_eef_velocity_recovery_v5_promotion_evidence(evidence)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("producer", "ego_lap_commit", "0" * 40),
        ("inspection", "videos_fully_decoded", False),
        ("authorization", "standard_authorized", True),
    ],
)
def test_promotion_manifest_rejects_top_level_section_drift(
    section: str,
    field: str,
    value: object,
) -> None:
    evidence = promotion.canonical_eef_velocity_recovery_v5_promotion_evidence()
    evidence[section][field] = value

    with pytest.raises(ValueError, match="promotion evidence drift"):
        promotion.validate_eef_velocity_recovery_v5_promotion_evidence(evidence)


@pytest.mark.parametrize(
    ("canary", "field"),
    [
        ("official_lap3b", "completion_audit_sha256"),
        ("official_lap3b", "registry_candidate_sha256"),
        ("official_lap3b", "suite_summary_sha256"),
        ("official_lap3b", "summary_video_sha256"),
        ("reasoning_43075", "completion_audit_sha256"),
        ("reasoning_43075", "registry_candidate_sha256"),
        ("reasoning_43075", "suite_summary_sha256"),
        ("reasoning_43075", "summary_video_sha256"),
    ],
)
def test_promotion_manifest_rejects_each_completion_identity_drift(
    canary: str,
    field: str,
) -> None:
    evidence = promotion.canonical_eef_velocity_recovery_v5_promotion_evidence()
    evidence["canaries"][canary]["artifacts"][field] = "0" * 64

    with pytest.raises(ValueError, match="promotion evidence drift"):
        promotion.validate_eef_velocity_recovery_v5_promotion_evidence(evidence)


def test_validator_returns_an_independent_copy() -> None:
    evidence = promotion.canonical_eef_velocity_recovery_v5_promotion_evidence()
    validated = promotion.validate_eef_velocity_recovery_v5_promotion_evidence(evidence)
    mutated = deepcopy(validated)
    mutated["authorization"]["smoke_suite_rollouts_per_task"] = 50

    assert validated != mutated
    assert evidence == validated


@pytest.mark.parametrize(
    ("eval_scale", "allowed"),
    [
        ("canary", True),
        ("smoke_suite", True),
        ("standard", False),
        ("unknown", False),
    ],
)
def test_promotion_authorizes_only_canary_and_smoke_suite(
    eval_scale: str,
    allowed: bool,
) -> None:
    assert promotion.eef_velocity_recovery_v5_eval_scale_allowed(eval_scale) is allowed


def test_standard_stays_blocked_pending_inspected_smoke_suite() -> None:
    authorization = promotion.canonical_eef_velocity_recovery_v5_promotion_evidence()[
        "authorization"
    ]

    assert authorization["next_eval_scale"] == "smoke_suite"
    assert authorization["smoke_suite_rollouts_per_task"] == 1
    assert authorization["smoke_suite_tasks"] == list(
        promotion.CANONICAL_SMOKE_SUITE_TASKS
    )
    assert authorization["standard_authorized"] is False
    assert authorization["standard_blocker"] == (
        "requires_completed_and_inspected_six_task_one_rollout_smoke_suite"
    )
