from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from polaris import eef_velocity_recovery_promotion as predecessor
from polaris import eef_velocity_recovery_standard_promotion as promotion


EXPECTED_TASK_ARTIFACTS = {
    "authoritative_attempt",
    "completion_audit",
    "episode_sidecar",
    "eval_success",
    "finalized_trace",
    "registry_candidate",
    "rollout_video",
    "runtime_contract",
    "summary_sidecar",
    "summary_video",
    "task_complete_verified",
}


def _evidence() -> dict[str, object]:
    return promotion.canonical_eef_velocity_recovery_v5_standard_promotion_evidence()


def test_standard_promotion_manifest_has_exact_closed_identity() -> None:
    evidence = _evidence()

    assert promotion.PROMOTION_EVIDENCE_SHA256 == (
        "7fd7a390da9dbe61531bdc8de75f83867a2011a6685fa2cfe761b1d965aba458"
    )
    assert (
        promotion.validate_eef_velocity_recovery_v5_standard_promotion_evidence(
            evidence
        )
        == evidence
    )
    assert set(evidence) == {
        "schema_version",
        "profile",
        "status",
        "lineage",
        "smoke_suites",
        "inspection",
        "scientific_scope",
        "authorization",
    }


def test_predecessor_promotion_remains_unchanged_and_smoke_only() -> None:
    prior = predecessor.canonical_eef_velocity_recovery_v5_promotion_evidence()

    assert predecessor.PROMOTION_EVIDENCE_SHA256 == (
        promotion.PREDECESSOR_EVIDENCE_SHA256
    )
    assert predecessor.validate_eef_velocity_recovery_v5_promotion_evidence(prior)
    assert prior["authorization"]["standard_authorized"] is False
    assert predecessor.eef_velocity_recovery_v5_eval_scale_allowed("standard") is False


def test_exact_predecessor_and_controller_sources_remain_unchanged() -> None:
    repo_root = Path(__file__).parents[1]

    identity = promotion.validate_promotion_lineage_source_identity(repo_root)
    assert identity["predecessor_source_sha256"] == (
        promotion.PREDECESSOR_SOURCE_SHA256
    )
    assert identity["producer_source_sha256"] == predecessor.PRODUCER_SOURCE_SHA256


def test_lineage_binds_runtime_parent_and_sealed_attestation() -> None:
    lineage = _evidence()["lineage"]

    assert lineage["predecessor_promotion_commit"] == (
        "0142e8518769d386c0a8227778767800b30c7e83"
    )
    assert lineage["runtime_parent_commit"] == (
        "9ab844b3fcaac6d29b51bc9fb2c2758c125201f3"
    )
    assert lineage["runtime_parent_tree"] == (
        "2063a7d091ee9d1c6e0646a60ee501a8abd395e8"
    )
    assert lineage["smoke_ego_lap_commit"] == (
        "5ad7da3057829d5a90cb5da78197bb3bd21f969f"
    )
    runtime = lineage["runtime_descendant_attestation"]
    assert runtime["attestation"]["job_id"] == 1098834
    assert runtime["attestation"]["sha256"] == (
        "efded6682bce983a4d773b038990f9e9fd5968cd05efe42b204063c6b4c7b0c5"
    )
    assert lineage["controller_or_evaluator_semantics_changed_by_promotion"] is False


def test_smoke_suites_bind_exact_roots_and_jobs() -> None:
    suites = _evidence()["smoke_suites"]

    reasoning = suites["reasoning_43075"]
    official = suites["official_lap3b"]
    assert reasoning["root"] == promotion.REASONING_ROOT
    assert official["root"] == promotion.OFFICIAL_ROOT
    assert reasoning["watcher_job_id"] == 1098870
    assert official["watcher_job_id"] == 1098873
    assert reasoning["watcher_scheduler_state"] == "COMPLETED"
    assert official["watcher_scheduler_exit_code"] == "0:0"
    assert [
        reasoning["tasks"][task]["worker_job_id"] for task in promotion.CANONICAL_TASKS
    ] == [
        1098871,
        1098872,
        1098878,
        1098879,
        1098882,
        1098884,
    ]
    assert [
        official["tasks"][task]["worker_job_id"] for task in promotion.CANONICAL_TASKS
    ] == [
        1098874,
        1098875,
        1098876,
        1098877,
        1098881,
        1098883,
    ]


def test_all_twelve_tasks_bind_full_horizon_authoritative_artifacts() -> None:
    suites = _evidence()["smoke_suites"]

    tasks_seen = 0
    for suite in suites.values():
        assert tuple(suite["tasks"]) == promotion.CANONICAL_TASKS
        for task in suite["tasks"].values():
            tasks_seen += 1
            assert task["scheduler_state"] == "COMPLETED"
            assert task["scheduler_exit_code"] == "0:0"
            assert task["policy_steps"] == 450
            assert task["physics_apply_calls"] == 3600
            assert task["numerical_failures"] == 0
            assert task["controller_aborts"] == 0
            assert task["dropped_diagnostics"] == 0
            assert task["raw_rubric_successes"] == 0
            assert task["task_valid_successes"] == 0
            assert set(task["artifacts"]) == EXPECTED_TASK_ARTIFACTS
            for artifact in task["artifacts"].values():
                assert artifact["relative_path"].startswith(
                    suite["step_root"].removeprefix(f"{suite['root']}/")
                )
                assert len(artifact["sha256"]) == 64
                int(artifact["sha256"], 16)
    assert tasks_seen == 12


def test_suite_aggregates_preserve_zero_success_and_progress() -> None:
    suites = _evidence()["smoke_suites"]

    assert suites["reasoning_43075"]["metrics"] == {
        "tasks": 6,
        "episodes_completed": 6,
        "raw_rubric_successes": 0,
        "task_valid_successes": 0,
        "mean_progress": 3 / 28,
        "numerical_failures": 0,
    }
    assert suites["official_lap3b"]["metrics"] == {
        "tasks": 6,
        "episodes_completed": 6,
        "raw_rubric_successes": 0,
        "task_valid_successes": 0,
        "mean_progress": 5 / 36,
        "numerical_failures": 0,
    }
    assert (
        suites["reasoning_43075"]["tasks"]["DROID-PanClean"]["normalized_progress"]
        == 1 / 3
    )
    assert (
        suites["official_lap3b"]["tasks"]["DROID-TapeIntoContainer"][
            "normalized_progress"
        ]
        == 2 / 3
    )


def test_visual_inspection_closes_all_raw_and_summary_videos() -> None:
    inspection = _evidence()["inspection"]

    assert inspection["fresh_full_task_complete_rechecks"]["all_passed"] is True
    assert inspection["fresh_full_task_complete_rechecks"]["validated_attempts"] == 12
    assert inspection["rollout_pairs_inspected"] == 12
    assert inspection["raw_and_summary_video_files_fully_decoded"] == 24
    assert inspection["expected_video_files"] == 24
    assert inspection["video_codec"] == "h264"
    assert inspection["video_pixel_format"] == "yuv420p"
    assert inspection["video_fps"] == 15
    assert inspection["video_frames"] == 450
    assert inspection["video_duration_seconds"] == 30.0
    assert inspection["raw_video_dimensions"] == [448, 224]
    assert inspection["summary_video_dimensions"] == [960, 608]
    assert inspection["correct_task_and_camera_views"] is True
    assert inspection["blank_or_corrupt_views"] == 0
    assert inspection["physics_explosions"] == 0
    assert inspection["motion_stable_and_physically_plausible"] is True
    assert inspection["raw_positive_rollouts"] == 0


def test_standard_authorization_is_exactly_the_canonical_protocol() -> None:
    evidence = _evidence()
    authorization = evidence["authorization"]

    assert authorization["allowed_eval_scales"] == [
        "canary",
        "smoke_suite",
        "standard",
    ]
    assert authorization["next_eval_scale"] == "standard"
    assert authorization["standard_authorized"] is True
    assert authorization["standard_protocol"] == {
        "benchmark": "polaris_droid_suite_v1",
        "tasks": list(promotion.CANONICAL_TASKS),
        "rollouts_per_task": 50,
        "policy_steps": 450,
        "policy_hz": 15,
        "control_mode": "absolute_end_effector_pose",
        "eef_frame": "panda_link8_relative_to_panda_link0",
        "environments": 1,
    }
    assert authorization["requires_exact_manifest_validation"] is True
    assert authorization["controller_or_evaluator_behavior_change_authorized"] is False
    assert (
        evidence["scientific_scope"]["smoke_establishes_standard_success_rate"] is False
    )


@pytest.mark.parametrize(
    ("eval_scale", "allowed"),
    [
        ("canary", True),
        ("smoke_suite", True),
        ("standard", True),
        ("unknown", False),
        (1, False),
    ],
)
def test_scale_authorization_requires_exact_evidence(
    eval_scale: object, allowed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    fresh_calls: list[dict[str, object]] = []

    def _fresh_authorization(evidence: dict[str, object]) -> dict[str, object]:
        fresh_calls.append(evidence)
        return {"standard_authorized": True}

    monkeypatch.setattr(
        promotion,
        "validate_and_authorize_eef_velocity_recovery_v5_standard",
        _fresh_authorization,
    )
    assert (
        promotion.eef_velocity_recovery_v5_standard_eval_scale_allowed(
            _evidence(), eval_scale
        )
        is allowed
    )
    assert len(fresh_calls) == (1 if eval_scale == "standard" else 0)


def _first_pinned_artifact(evidence: dict[str, object]) -> tuple[str, str]:
    artifact = evidence["smoke_suites"]["reasoning_43075"]["suite_artifacts"][
        "eval_success"
    ]
    return artifact["relative_path"], artifact["sha256"]


def _all_root_overrides(root: Path) -> dict[str, Path]:
    return {"reasoning_43075": root, "official_lap3b": root}


def test_fresh_artifact_validator_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Pinned artifact .*missing"):
        promotion.validate_eef_velocity_recovery_v5_standard_promotion_artifacts(
            _evidence(), root_overrides=_all_root_overrides(tmp_path)
        )


def test_fresh_artifact_validator_rejects_tampered_file(tmp_path: Path) -> None:
    evidence = _evidence()
    relative_path, expected_sha256 = _first_pinned_artifact(evidence)
    artifact_path = tmp_path / relative_path
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"tampered")
    assert expected_sha256 != hashlib.sha256(b"tampered").hexdigest()

    with pytest.raises(ValueError, match="Pinned artifact SHA-256 drift"):
        promotion.validate_eef_velocity_recovery_v5_standard_promotion_artifacts(
            evidence, root_overrides=_all_root_overrides(tmp_path)
        )


def test_fresh_artifact_validator_rejects_symlink(tmp_path: Path) -> None:
    evidence = _evidence()
    relative_path, _ = _first_pinned_artifact(evidence)
    artifact_path = tmp_path / relative_path
    artifact_path.parent.mkdir(parents=True)
    target = tmp_path / "symlink-target"
    target.write_bytes(b"not trusted through an alias")
    artifact_path.symlink_to(target)

    with pytest.raises(ValueError, match="symlink or cannot be securely opened"):
        promotion.validate_eef_velocity_recovery_v5_standard_promotion_artifacts(
            evidence, root_overrides=_all_root_overrides(tmp_path)
        )


def test_fresh_artifact_validator_rejects_non_regular_file(tmp_path: Path) -> None:
    evidence = _evidence()
    relative_path, _ = _first_pinned_artifact(evidence)
    artifact_path = tmp_path / relative_path
    artifact_path.mkdir(parents=True)

    with pytest.raises(ValueError, match="not a regular file"):
        promotion.validate_eef_velocity_recovery_v5_standard_promotion_artifacts(
            evidence, root_overrides=_all_root_overrides(tmp_path)
        )


def test_fresh_artifact_validator_rejects_hardlink(tmp_path: Path) -> None:
    evidence = _evidence()
    relative_path, _ = _first_pinned_artifact(evidence)
    artifact_path = tmp_path / relative_path
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"hardlink alias")
    alias = tmp_path / "second-name-for-same-inode"
    alias.hardlink_to(artifact_path)

    with pytest.raises(ValueError, match="must have exactly one hard link"):
        promotion.validate_eef_velocity_recovery_v5_standard_promotion_artifacts(
            evidence, root_overrides=_all_root_overrides(tmp_path)
        )


def test_fresh_artifact_validator_rejects_root_symlink(tmp_path: Path) -> None:
    evidence = _evidence()
    physical = tmp_path / "physical"
    physical.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(physical, target_is_directory=True)

    with pytest.raises(ValueError, match="root must not be a symlink"):
        promotion.validate_eef_velocity_recovery_v5_standard_promotion_artifacts(
            evidence, root_overrides=_all_root_overrides(alias)
        )


def test_fresh_artifact_validator_rejects_path_schema_drift(tmp_path: Path) -> None:
    evidence = _evidence()
    artifact = evidence["smoke_suites"]["reasoning_43075"]["suite_artifacts"][
        "eval_success"
    ]
    artifact["relative_path"] = "../escape"

    with pytest.raises(ValueError, match="standard promotion evidence drift"):
        promotion.validate_eef_velocity_recovery_v5_standard_promotion_artifacts(
            evidence, root_overrides=_all_root_overrides(tmp_path)
        )


def test_standard_authorizer_requires_fresh_artifact_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "artifact_count": 138,
        "verification_sha256": "1" * 64,
    }
    calls: list[dict[str, object]] = []

    def _fresh(evidence: dict[str, object]) -> dict[str, object]:
        calls.append(evidence)
        return expected

    monkeypatch.setattr(
        promotion,
        "validate_eef_velocity_recovery_v5_standard_promotion_artifacts",
        _fresh,
    )
    result = promotion.validate_and_authorize_eef_velocity_recovery_v5_standard(
        _evidence()
    )

    assert len(calls) == 1
    assert result == {
        "standard_authorized": True,
        "promotion_evidence_sha256": promotion.EXPECTED_PROMOTION_EVIDENCE_SHA256,
        "artifact_verification": expected,
    }


@pytest.mark.skipif(
    not Path(promotion.REASONING_RESOLVED_ROOT).is_dir()
    or not Path(promotion.OFFICIAL_RESOLVED_ROOT).is_dir(),
    reason="Pinned l401 NFS evidence roots are unavailable on this host",
)
def test_fresh_artifact_validator_accepts_exact_pinned_nfs_trees() -> None:
    result = promotion.validate_eef_velocity_recovery_v5_standard_promotion_artifacts(
        _evidence()
    )

    assert result["suite_count"] == 2
    assert result["task_count"] == 12
    assert result["artifact_count"] == 138
    assert result["suite_artifact_counts"] == {
        "reasoning_43075": 69,
        "official_lap3b": 69,
    }
    assert result["root_overrides_used"] is False
    assert len(result["artifact_inventory_sha256"]) == 64
    assert len(result["verification_sha256"]) == 64


@pytest.mark.parametrize(
    ("suite_name", "field"),
    [
        ("reasoning_43075", "suite_summary"),
        ("reasoning_43075", "registry_candidates"),
        ("official_lap3b", "suite_summary"),
        ("official_lap3b", "registry_candidates"),
    ],
)
def test_manifest_rejects_suite_artifact_drift(suite_name: str, field: str) -> None:
    evidence = _evidence()
    evidence["smoke_suites"][suite_name]["suite_artifacts"][field]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="standard promotion evidence drift"):
        promotion.validate_eef_velocity_recovery_v5_standard_promotion_evidence(
            evidence
        )


@pytest.mark.parametrize("suite_name", ["reasoning_43075", "official_lap3b"])
@pytest.mark.parametrize("artifact_name", sorted(EXPECTED_TASK_ARTIFACTS))
def test_manifest_rejects_every_task_artifact_drift(
    suite_name: str, artifact_name: str
) -> None:
    evidence = _evidence()
    task = evidence["smoke_suites"][suite_name]["tasks"]["DROID-BlockStackKitchen"]
    task["artifacts"][artifact_name]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="standard promotion evidence drift"):
        promotion.validate_eef_velocity_recovery_v5_standard_promotion_evidence(
            evidence
        )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("lineage", "runtime_parent_commit", "0" * 40),
        ("inspection", "correct_task_and_camera_views", False),
        ("inspection", "raw_and_summary_video_files_fully_decoded", 23),
        ("authorization", "standard_authorized", False),
    ],
)
def test_manifest_rejects_authorization_prerequisite_drift(
    section: str, field: str, value: object
) -> None:
    evidence = _evidence()
    evidence[section][field] = value

    with pytest.raises(ValueError, match="standard promotion evidence drift"):
        promotion.validate_eef_velocity_recovery_v5_standard_promotion_evidence(
            evidence
        )
    with pytest.raises(ValueError, match="standard promotion evidence drift"):
        promotion.eef_velocity_recovery_v5_standard_eval_scale_allowed(
            evidence, "standard"
        )


@pytest.mark.parametrize("suite_name", ["reasoning_43075", "official_lap3b"])
def test_manifest_rejects_suite_or_task_omission(suite_name: str) -> None:
    suite_missing = _evidence()
    suite_missing["smoke_suites"].pop(suite_name)
    with pytest.raises(ValueError, match="standard promotion evidence drift"):
        promotion.validate_eef_velocity_recovery_v5_standard_promotion_evidence(
            suite_missing
        )

    task_missing = _evidence()
    task_missing["smoke_suites"][suite_name]["tasks"].pop("DROID-TapeIntoContainer")
    with pytest.raises(ValueError, match="standard promotion evidence drift"):
        promotion.validate_eef_velocity_recovery_v5_standard_promotion_evidence(
            task_missing
        )


def test_validator_returns_independent_copy() -> None:
    evidence = _evidence()
    validated = promotion.validate_eef_velocity_recovery_v5_standard_promotion_evidence(
        evidence
    )
    mutated = deepcopy(validated)
    mutated["authorization"]["standard_protocol"]["rollouts_per_task"] = 1

    assert validated != mutated
    assert evidence == validated
