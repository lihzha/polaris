"""Immutable evidence gate for the v5 measured-velocity recovery smoke suite.

This module changes no controller behavior.  It closes the two inspected
full-horizon canaries over the exact v5 producer and authorizes only the next
bounded lifecycle stage: one rollout on each of the six canonical DROID tasks.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROMOTION_PROFILE = "measured_velocity_recovery_v5_full_horizon_canaries_v1"
PROMOTION_STATUS = "validated_on_two_full_horizon_canaries_pending_smoke_suite"
PRODUCER_EGO_LAP_COMMIT = "74bb225d07ccdd2408bc568fe900709e633047e6"
PRODUCER_POLARIS_COMMIT = "f11ae45a64b2f839dcb3325459ab06776d1dd81a"
PRODUCER_POLARIS_TREE = "7d68beea046e485dfe622a4a41c9e03d3a423ef2"
AUTHORIZED_EVAL_SCALES = ("canary", "smoke_suite")
NEXT_EVAL_SCALE = "smoke_suite"
CANONICAL_SMOKE_SUITE_TASKS = (
    "DROID-BlockStackKitchen",
    "DROID-FoodBussing",
    "DROID-PanClean",
    "DROID-MoveLatteCup",
    "DROID-OrganizeTools",
    "DROID-TapeIntoContainer",
)

PRODUCER_SOURCE_SHA256 = {
    "scripts/smoke_eef_pose_canary_trace_replay.py": (
        "b7f236453d2cc832ddfcc1b2dafbbb1b57299d05bbc8eb36f8e0bdc5cf586e8d"
    ),
    "src/polaris/config.py": (
        "5242bb54847f6aa014d11e947870b491e805c1d37392aaa6ba2789b7edb7e4c7"
    ),
    "src/polaris/eef_controller_profile.py": (
        "a37a2a2129978f713326cce71b82168d51e11e9d2ef9c65694f6a6b51386f2a1"
    ),
    "src/polaris/eef_controller_repair.py": (
        "63423f7ee6b3d7057004afcc7a7f1118f655601207e6e90cac175d19d84be508"
    ),
    "src/polaris/eef_ik_safety.py": (
        "fdfda76337b944452005b0b2cfc816eafac3bf145ab7e271a20179712dba380e"
    ),
    "src/polaris/eef_runtime_contract.py": (
        "b21632a5c0417a759cdf4d82cbdd2276ee1972f8eb786cb56a6d1a88947efeb0"
    ),
    "src/polaris/robust_differential_ik.py": (
        "1b3be5e8949d3428fcff2fe307cf93ae5ec65da9c79c5c07ad9342687424ed4f"
    ),
}


def _artifact_evidence(
    *,
    runtime: str,
    sidecar: str,
    trace: str,
    rollout: str,
    completion_audit: str,
    registry_candidate: str,
    suite_summary: str,
    summary_video: str,
) -> dict[str, str]:
    return {
        "runtime_contract_sha256": runtime,
        "episode_sidecar_sha256": sidecar,
        "finalized_trace_sha256": trace,
        "rollout_video_sha256": rollout,
        "completion_audit_sha256": completion_audit,
        "registry_candidate_sha256": registry_candidate,
        "suite_summary_sha256": suite_summary,
        "summary_video_sha256": summary_video,
    }


def canonical_eef_velocity_recovery_v5_promotion_evidence() -> dict[str, Any]:
    """Return the closed evidence manifest for the smoke-suite promotion."""

    return {
        "schema_version": 1,
        "profile": PROMOTION_PROFILE,
        "status": PROMOTION_STATUS,
        "producer": {
            "ego_lap_commit": PRODUCER_EGO_LAP_COMMIT,
            "polaris_commit": PRODUCER_POLARIS_COMMIT,
            "polaris_tree": PRODUCER_POLARIS_TREE,
            "controller_profile": (
                "arm_slew_0p95_gripper_rate0p25_fixed_anchor86_release_ramp16_"
                "velocity_recovery8_clean2_mimic100_damping1p2_v5"
            ),
            "ik_safety_profile": (
                "panda_velocity_physxlimit_solveriter1_residual_recovery8_clean2_v5"
            ),
            "controller_semantics_changed_by_promotion": False,
            "source_sha256": dict(PRODUCER_SOURCE_SHA256),
        },
        "canaries": {
            "official_lap3b": {
                "checkpoint": {
                    "uri": (
                        "hf://lihzha/LAP-3B@601db9c1ab4bcaf6dddb160c7b2dec589a67b730"
                    ),
                    "content_manifest_sha256": (
                        "567cc3ff7d20f3f03913a6f11c3fa151f789e1c0118ed5af0eea24d9cc48f20e"
                    ),
                },
                "task": "DROID-FoodBussing",
                "protocol_variant": (
                    "polaris-droid-link8-eefpose-canary1-r6rows-q99repair-"
                    "controllercandidate-v5-iksafety-v5"
                ),
                "watcher_job_id": 1098707,
                "worker_job_id": 1098708,
                "policy_steps": 450,
                "apply_calls": 3600,
                "numerical_failures": 0,
                "raw_successes": 0,
                "task_valid_successes": 0,
                "normalized_progress": 1 / 6,
                "recovery": {
                    "counters": {
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
                    },
                    "events": [],
                    "maxima": {
                        "abs_velocity_residual_excess_rad_s": [0.0] * 7,
                        "abs_velocity_to_limit_ratio": 0.9338167309761047,
                        "consecutive_recovery_substeps": 0,
                    },
                },
                "artifacts": _artifact_evidence(
                    runtime=(
                        "7fb9d06c37f42cc07bd10ad758919080cf57bc9b228bdd16dfc55725d40fc682"
                    ),
                    sidecar=(
                        "9f4e4111ee9bd272e985649ea448ebc38de1d1c9decedb349fe388dfabf2692f"
                    ),
                    trace=(
                        "a94deec730578cff731160544c1ca160ddf510b6cdbda361589abc2502482217"
                    ),
                    rollout=(
                        "5d953ee5bd2117ab9dd46217f77a93b4d50aa60d3169963c6d139cba46c17336"
                    ),
                    completion_audit=(
                        "679119c85befb2060c796b9c7c235bfe542d6003367434e47ba9be7ba28b7bdf"
                    ),
                    registry_candidate=(
                        "5dee54dea0358535fad8dff7d1802d0716e9d7c74c23ac121bce5fa4341412a5"
                    ),
                    suite_summary=(
                        "6c346cb7abea349a8fdd54a1b8630f79fa7d86e72c2c8e66cd05d5b34ea998f7"
                    ),
                    summary_video=(
                        "53be891a25b16042de78dec8bb4a22e5949de27cc8b785aa2e6fe2e51854f5ce"
                    ),
                ),
                "registry_revision": 25,
                "visual_finding": (
                    "approached_and_stalled_in_yellow_bowl_without_stable_"
                    "grasp_transport_or_bussing"
                ),
            },
            "reasoning_43075": {
                "checkpoint": {
                    "uri": (
                        "gs://v6_east1d/checkpoints/lap_oxe_magic_soup_reasoning_full/"
                        "oxe_magic_soup_reasoning_full_v2_flow_pred0_cf0_ckpt25_"
                        "v6_32_b512_s42_20260630/43075"
                    ),
                    "inference_subset_profile": "policy-inference-params-assets-v1",
                    "inference_subset_sha256": (
                        "bb9ea5bb041f689a08f914cac7dfe5d061c822ddbe87e292f9c7878a9d3bfc4d"
                    ),
                },
                "task": "DROID-FoodBussing",
                "protocol_variant": (
                    "polaris-droid-link8-eefpose-canary1-q99repair-"
                    "controllercandidate-v5-iksafety-v5"
                ),
                "watcher_job_id": 1098709,
                "worker_job_id": 1098710,
                "policy_steps": 450,
                "apply_calls": 3600,
                "numerical_failures": 0,
                "raw_successes": 0,
                "task_valid_successes": 0,
                "normalized_progress": 1 / 6,
                "recovery": {
                    "counters": {
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
                    },
                    "events": [
                        {
                            "event_index": 0,
                            "start_apply_index": 2386,
                            "start_policy_step": 298,
                            "start_physics_substep": 2,
                            "start_reason": "measured_velocity_above_float32_envelope",
                            "start_velocity_to_limit_ratio": 1.0542817115783691,
                            "start_velocity_residual_excess_rad_s": (
                                0.14167523384094238
                            ),
                            "end_apply_index": 2404,
                            "end_policy_step": 300,
                            "end_physics_substep": 4,
                            "end_reason": "clean2_release_ramp_complete",
                            "recovery_completed_apply_index": 2404,
                        }
                    ],
                    "maxima": {
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
                    },
                },
                "artifacts": _artifact_evidence(
                    runtime=(
                        "57b6f4720d853e03c52456f6dd2c11640b7f496443a5a30fc20ed459d442c13e"
                    ),
                    sidecar=(
                        "9f991a98eafa26aeefd1edac98c3d572e3d99e8b5c6dc1a289f31ad84e04f6a2"
                    ),
                    trace=(
                        "33de921ba18318041c362048dab5006853080702f18ef22c87706bc9ab5a5f2a"
                    ),
                    rollout=(
                        "ba44ad8642adf846b250cb6c3fe23bc6d65029e82d5274b71f59177745557c51"
                    ),
                    completion_audit=(
                        "3bb4dc554d48e1b3622a6d7eba3ca1e6136a26465a2b1961c9a6637f2643aa18"
                    ),
                    registry_candidate=(
                        "67d1ccc00b8f1d8a6ca5e7051256d354f4a4e8bcdf8f593514823d9b786b0af8"
                    ),
                    suite_summary=(
                        "d6597b27bf476b1dfae2a0f05f7c363e70c76026eac7289ea6cc6c41a631e212"
                    ),
                    summary_video=(
                        "bc99208ab2c4479806f230579c4643e20301ec86e37f4a4718dafa8b746898ea"
                    ),
                ),
                "registry_revision": 68,
                "visual_finding": "moved_cup_without_completing_transport_or_bussing",
            },
        },
        "inspection": {
            "fresh_remote_task_complete_recheck_passed": True,
            "videos_fully_decoded": True,
            "video_codec": "h264",
            "video_pixel_format": "yuv420p",
            "video_fps": 15,
            "video_frames": 450,
            "video_duration_seconds": 30.0,
            "summary_video_width": 960,
            "summary_video_height": 608,
            "rollout_video_width": 448,
            "rollout_video_height": 224,
            "first_middle_last_frames_inspected": True,
            "reasoning_recovery_window_inspected": True,
            "correct_external_and_wrist_views": True,
            "correct_task_labels": True,
            "blank_or_corrupt_cells": 0,
            "physics_explosions": 0,
            "raw_positive_rollouts_requiring_adjudication": 0,
        },
        "authorization": {
            "allowed_eval_scales": list(AUTHORIZED_EVAL_SCALES),
            "next_eval_scale": NEXT_EVAL_SCALE,
            "smoke_suite_tasks": list(CANONICAL_SMOKE_SUITE_TASKS),
            "smoke_suite_rollouts_per_task": 1,
            "standard_authorized": False,
            "standard_blocker": (
                "requires_completed_and_inspected_six_task_one_rollout_smoke_suite"
            ),
        },
    }


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


PROMOTION_EVIDENCE_SHA256 = _canonical_sha256(
    canonical_eef_velocity_recovery_v5_promotion_evidence()
)
EXPECTED_PROMOTION_EVIDENCE_SHA256 = (
    "9576a178253741571a50cd23fe8a16b75b9a386ced5bc43ee416348fa52454f7"
)
if PROMOTION_EVIDENCE_SHA256 != EXPECTED_PROMOTION_EVIDENCE_SHA256:
    raise RuntimeError("Canonical v5 measured-velocity promotion evidence drift")


def validate_eef_velocity_recovery_v5_promotion_evidence(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless ``value`` is the exact canonical promotion manifest."""

    expected = canonical_eef_velocity_recovery_v5_promotion_evidence()
    if type(value) is not dict or value != expected:
        raise ValueError("V5 measured-velocity recovery promotion evidence drift")
    if _canonical_sha256(value) != EXPECTED_PROMOTION_EVIDENCE_SHA256:
        raise ValueError("V5 measured-velocity recovery promotion digest drift")
    return deepcopy(expected)


def eef_velocity_recovery_v5_eval_scale_allowed(eval_scale: str) -> bool:
    """Return the closed post-canary lifecycle authorization."""

    return type(eval_scale) is str and eval_scale in AUTHORIZED_EVAL_SCALES


def validate_producer_source_identity(repo_root: Path) -> dict[str, str]:
    """Hash every controller source and reject any drift from the v5 producer."""

    actual: dict[str, str] = {}
    for relative_path, expected_sha256 in PRODUCER_SOURCE_SHA256.items():
        path = repo_root / relative_path
        if not path.is_file():
            raise ValueError(f"Missing v5 producer source: {relative_path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError(f"V5 producer source digest drift: {relative_path}")
        actual[relative_path] = digest
    return actual
