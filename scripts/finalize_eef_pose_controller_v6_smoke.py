#!/usr/bin/env python3
"""Finalize the exact cadence-correct concurrent-arm/gripper v6 smoke.

This evidence-only host finalizer changes no controller or simulator behavior.
It validates the immutable job-1098975 capture, its saved launch surface, the
exact producer checkout, image, FoodBussing scene, and terminal Slurm record,
then publishes one non-overwriting mode-0444 promotion attestation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys
from typing import Any, Mapping


class VerificationError(RuntimeError):
    """Raised when any immutable smoke-evidence invariant drifts."""


JOB_ID = 1098975
JOB_NAME = "pol_v6_cad_3941840"
PRODUCER_COMMIT = "39418400493cdcf8cd8272608980a798f7929a20"
PRODUCER_TREE = "7fc1ff24053e3aeab5ed3e06068089b5aa596bc6"
PRODUCER_PARENT = "ee6d09351bed75e32db93ecf59c039a8e99fac9f"
CONTROLLER_PROFILE = (
    "arm_slew_0p95_gripper_rate0p25_concurrent_arm_velocity_recovery8_"
    "clean2_mimic100_damping1p2_v6"
)
IK_SAFETY_PROFILE = (
    "panda_velocity_physxlimit_solveriter1_residual_recovery8_clean2_"
    "concurrent_arm_gripper_v6"
)
PROMOTION_PROFILE = "concurrent_arm_gripper_v6_cadence_smoke_job1098975_v1"
PROMOTION_STATUS = (
    "validated_controller_smoke_pending_corrected_camera_image_contract_smoke_"
    "then_two_checkpoint_canaries"
)
PROMOTION_SCOPE = "standalone_controller_smoke_only_no_checkpoint_or_task_metric"
NEXT_REQUIRED_GATE = (
    "corrected_camera_image_contract_smoke_before_paired_official_and_reasoning_"
    "foodbussing_canaries"
)
IMAGE_SHA256 = "ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a"
SCENE_SHA256 = "82cd641e422935b394ce7ea7b6be55214c9952a2544000222921e544c409b489"
SRUN_START_EPOCH_NS = 1_783_229_839_000_000_000
IMAGE_METADATA = {
    "size_bytes": 7_183_130_624,
    "mtime_ns": 1_782_886_704_000_000_000,
    "ctime_ns": 1_782_886_704_000_000_000,
}
SCENE_METADATA = {
    "size_bytes": 14_914,
    "mtime_ns": 1_782_866_941_000_000_000,
    "ctime_ns": 1_782_884_937_000_000_000,
}
L401_LITERAL_USER_ROOT = Path("/lustre/fsw/portfolios/nvr/users/lzha")
L401_CANONICAL_USER_ROOT = Path(
    "/lustre/fs11/portfolios/nvr/projects/nvr_lpr_rvp/users/lzha"
)

RESULT_ROOT = (
    "/lustre/fsw/portfolios/nvr/users/lzha/results/polaris_eval/"
    "controller_concurrent_v6_cadence_smoke/3941840-20260705T053510Z"
)
RAW_PATH = f"{RESULT_ROOT}/smoke-{JOB_ID}.raw.json"
READY_PATH = f"{RAW_PATH}.ready.json"
INLINE_ATTESTATION_PATH = f"{RESULT_ROOT}/smoke-{JOB_ID}.host-attestation.json"
SOURCE_IDENTITY_PATH = f"{RESULT_ROOT}/source-identity-{JOB_ID}.sha256"
PROMOTION_ATTESTATION_PATH = f"{RESULT_ROOT}/smoke-{JOB_ID}.promotion-attestation.json"
PROVENANCE_ROOT = f"{RESULT_ROOT}/promotion-provenance-job-{JOB_ID}"
SEALED_SOURCE_IDENTITY_PATH = f"{PROVENANCE_ROOT}/source-identity.sha256"
SEALED_JOB_SCRIPT_PATH = f"{PROVENANCE_ROOT}/saved-job-script.sbatch"
SEALED_SLURM_LOG_PATH = f"{PROVENANCE_ROOT}/slurm.out"
SEALED_SACCT_PATH = f"{PROVENANCE_ROOT}/sacct.json"
SAVED_JOB_SCRIPT_PATH = (
    "/lustre/fsw/portfolios/nvr/users/lzha/launchers/polaris_eval/"
    "polaris_v6_cadence_smoke_3941840_20260705T053510Z.sbatch"
)
SLURM_LOG_PATH = (
    "/lustre/fsw/portfolios/nvr/users/lzha/slurm_logs/polaris_eval/"
    "pol_v6_cad_3941840-1098975.out"
)
PRODUCER_REPO_PATH = (
    "/lustre/fsw/portfolios/nvr/users/lzha/src/"
    "PolaRiS-concurrent-v6-cadence-3941840-20260705T053510Z"
)
CONTAINER_IMAGE_PATH = (
    "/lustre/fsw/portfolios/nvr/users/lzha/cache/polaris/"
    "polaris-eval-cuda13-fd00a51.sqsh"
)
SCENE_PATH = (
    "/lustre/fsw/portfolios/nvr/users/lzha/data/PolaRiS-Hub/food_bussing/scene.usda"
)

ARTIFACT_SPECS = {
    "raw": {
        "size_bytes": 793098,
        "sha256": "393f0a57f409beb249635214ab2d7efb66783625048ddb18a5dc57426eaef2a5",
        "mode": "0444",
    },
    "ready": {
        "size_bytes": 380,
        "sha256": "9e0a6826601a9d7019f6a4836a6524e259551bdc048b11e6184de0cf6dafc576",
        "mode": "0444",
    },
    "inline_attestation": {
        "size_bytes": 1923,
        "sha256": "dfb0d40593241b85ea2af261e3de70d3c4d75fc6331109f38d32c305adefee42",
        "mode": "0444",
    },
    "source_identity": {
        "size_bytes": 800,
        "sha256": "d18a718f402d539d031ae699da1230144e8f9f016874530189d31916f508e2d1",
        "mode": "0644",
    },
    "saved_job_script": {
        "size_bytes": 10396,
        "sha256": "2215a73434d5c0f76368238932a8a18ebfd18125afb3b447f9396b4187fa18d4",
        "mode": "0444",
    },
    "slurm_log": {
        "size_bytes": 43539,
        "sha256": "115a7d83b887a3403138626cd85429615955ab99a050e07e9c710f093b772b56",
        "mode": "0644",
    },
}

EXPECTED_SCHEDULER_EVIDENCE = {
    "allocation": {
        "job_id": "1098975",
        "job_name": "pol_v6_cad_3941840",
        "account": "nvr_lpr_rvp",
        "partition": "batch",
        "state": "COMPLETED",
        "exit_code": "0:0",
        "elapsed_seconds": 310,
        "start": "2026-07-04T22:37:09",
        "end": "2026-07-04T22:42:19",
        "node": "pool0-00010",
        "allocated_tres": "billing=1,cpu=16,gres/gpu=1,mem=96G,node=1",
        "requested_tres": "billing=1,cpu=16,gres/gpu=1,mem=96G,node=1",
    },
    "batch": {
        "job_id": "1098975.batch",
        "job_name": "batch",
        "account": "nvr_lpr_rvp",
        "partition": "",
        "state": "COMPLETED",
        "exit_code": "0:0",
        "elapsed_seconds": 310,
        "start": "2026-07-04T22:37:09",
        "end": "2026-07-04T22:42:19",
        "node": "pool0-00010",
        "allocated_tres": "cpu=16,gres/gpu=1,mem=96G,node=1",
        "requested_tres": "",
    },
    "extern": {
        "job_id": "1098975.extern",
        "job_name": "extern",
        "account": "nvr_lpr_rvp",
        "partition": "",
        "state": "COMPLETED",
        "exit_code": "0:0",
        "elapsed_seconds": 310,
        "start": "2026-07-04T22:37:09",
        "end": "2026-07-04T22:42:19",
        "node": "pool0-00010",
        "allocated_tres": "billing=1,cpu=16,gres/gpu=1,mem=96G,node=1",
        "requested_tres": "",
    },
    "srun": {
        "job_id": "1098975.0",
        "job_name": "env",
        "account": "nvr_lpr_rvp",
        "partition": "",
        "state": "COMPLETED",
        "exit_code": "0:0",
        "elapsed_seconds": 300,
        "start": "2026-07-04T22:37:19",
        "end": "2026-07-04T22:42:19",
        "node": "pool0-00010",
        "allocated_tres": "cpu=16,gres/gpu=1,mem=96G,node=1",
        "requested_tres": "",
    },
    "monitoring_1": {
        "job_id": "1098975.1",
        "job_name": "nvidia-smi",
        "account": "nvr_lpr_rvp",
        "partition": "",
        "state": "COMPLETED",
        "exit_code": "0:0",
        "elapsed_seconds": 1,
        "start": "2026-07-04T22:38:49",
        "end": "2026-07-04T22:38:50",
        "node": "pool0-00010",
        "allocated_tres": "cpu=1,gres/gpu=1,mem=96G,node=1",
        "requested_tres": "",
    },
    "monitoring_2": {
        "job_id": "1098975.2",
        "job_name": "nvidia-smi",
        "account": "nvr_lpr_rvp",
        "partition": "",
        "state": "COMPLETED",
        "exit_code": "0:0",
        "elapsed_seconds": 0,
        "start": "2026-07-04T22:39:11",
        "end": "2026-07-04T22:39:11",
        "node": "pool0-00010",
        "allocated_tres": "cpu=1,gres/gpu=1,mem=96G,node=1",
        "requested_tres": "",
    },
}
SACCT_SNAPSHOT_PAYLOAD = {
    "schema_version": 1,
    "profile": "slurm_job_srun_and_read_only_monitoring_terminal_identity_v1",
    "scheduler": EXPECTED_SCHEDULER_EVIDENCE,
}
SACCT_SNAPSHOT_BYTES = (
    json.dumps(SACCT_SNAPSHOT_PAYLOAD, indent=2, sort_keys=True) + "\n"
).encode()
SEALED_PROVENANCE_SPECS = {
    "source_identity": {
        **ARTIFACT_SPECS["source_identity"],
        "mode": "0444",
    },
    "saved_job_script": {
        **ARTIFACT_SPECS["saved_job_script"],
        "mode": "0444",
    },
    "slurm_log": {
        **ARTIFACT_SPECS["slurm_log"],
        "mode": "0444",
    },
    "sacct": {
        "size_bytes": len(SACCT_SNAPSHOT_BYTES),
        "sha256": hashlib.sha256(SACCT_SNAPSHOT_BYTES).hexdigest(),
        "mode": "0444",
    },
}
EVIDENCE_COMMIT_CHANGED_PATHS = {
    "WORKLOG.v6.md",
    "scripts/finalize_eef_pose_controller_v6_smoke.py",
    "tests/test_finalize_eef_pose_controller_v6_smoke.py",
}

CAPTURE_SOURCE_SHA256 = {
    "scripts/smoke_eef_pose_controller.py": (
        "b5b1b621041b74247dbce6488483cdf7d33d6fa7c4002e821cf50ed26609f6a8"
    ),
    "src/polaris/config.py": (
        "47f1a5af67e680e7b5762848697bd4c51b3b0f31132968d3551e0b28b0c889b6"
    ),
    "src/polaris/eef_controller_profile.py": (
        "fa55f0b1fc1bb9600c5d2d11d39bd670980791374de5a0dfba955e90929496a4"
    ),
    "src/polaris/eef_controller_repair.py": (
        "b2a4df4cccf5c7a4efadd9f6ca990e9b9c9eca8230024787c519c94b284d0f76"
    ),
    "src/polaris/eef_gripper_runtime.py": (
        "0687434bc2c61bb09739be473d5477f7b92d7b7b846a800298a89225f5b4f220"
    ),
    "src/polaris/eef_ik_safety.py": (
        "bc34d745705227c1154bdb266a5e7f937739a90dad00e4802eebf2da8c6cd978"
    ),
    "src/polaris/eef_runtime_contract.py": (
        "fb7094a37a1b6c676c61cce1a371d1f146db69183d55fd46d9db84c2a8739a8b"
    ),
    "src/polaris/robust_differential_ik.py": (
        "a07a62f0ef5aebfb69e214e2c2d11bac197ab1a377686a72b02024e8285b16f4"
    ),
}
PRODUCER_SOURCE_SHA256 = {
    **CAPTURE_SOURCE_SHA256,
    "scripts/eval.py": (
        "b5464158b8cc996bffd55ee744133ba2d0d3708cca2288059076c829cee8a86f"
    ),
    "src/polaris/eef_gripper_failure_trace.py": (
        "f66af5001f8636333f6db00948a64214909a43b7d6afd1af968397dea33280b0"
    ),
}

EXPECTED_CASES = [
    "hold",
    "translate +x",
    "translate -x",
    "translate +y",
    "translate -y",
    "translate +z",
    "translate -z",
    "rotate +x",
    "rotate -x",
    "rotate +y",
    "rotate -y",
    "rotate +z",
    "rotate -z",
]
CRITICAL_COUNTERS = {
    "post_clamp_target_violations",
    "current_joint_limit_aborts",
    "invariant_aborts",
    "nonfinite_aborts",
    "dls_fallbacks",
    "guard_diagnostics_dropped",
}
RECOVERY_ZERO_COUNTERS = {
    "residual_events",
    "residual_joints",
    "recovery_events",
    "recovery_active_substeps",
    "recovered_events",
    "hold_target_applies",
    "release_ramp_target_applies",
    "sustained_aborts",
    "current_hard_limit_aborts",
    "predicted_limit_aborts",
    "transaction_aborts",
    "lower_endpoint_transition_aborts",
}
INLINE_CHECKS = {
    "arm_observed_endpoint_change_count": 2,
    "pose_cases_passed": 13,
    "concurrent_apply_calls": 168,
    "closed_endpoint_fresh_dls_target_applies": 80,
    "closed_endpoint_distinct_desired_pose_count": 10,
    "recovery_owned_target_applies": 0,
    "deferred_endpoint_transition_count": 0,
    "driver_endpoint_change_count": 2,
    "stored_target_replay_count": 0,
    "open_endpoint_samples": 99,
    "follower_threshold_crossing_samples": 0,
    "interlock_control_counters_zero": True,
    "coupled_impulse_failure_samples": 0,
}
RAW_FIELDS = {
    "case",
    "close_failures",
    "concurrent_arm_gripper_discriminator",
    "eef_controller_profile",
    "eef_frame",
    "environment",
    "exit_code",
    "failure",
    "finalized",
    "frame_position_tolerance_m",
    "frame_rotation_tolerance_deg",
    "gripper_close_velocity_headroom",
    "gripper_delayed_close_replay",
    "hold_steps",
    "ik_safety_adversarial",
    "ik_safety_episodes",
    "passed",
    "persistence_failures",
    "position_delta_m",
    "position_tolerance_m",
    "raw_ik_safety_capture",
    "results",
    "rotation_delta_deg",
    "rotation_tolerance_deg",
    "schema_version",
    "stage",
    "terminal_failure_evidence",
}
SAFETY_FIELDS = {
    "apply_actions_cadence",
    "arm_velocity_target_profile",
    "arm_velocity_target_rad_s",
    "articulation_solver_profile",
    "articulation_solver_readback",
    "control_dt",
    "counters",
    "current_joint_soft_limit_tolerance_rad",
    "current_joint_velocity_abort",
    "current_joint_velocity_recovery",
    "decimation",
    "eef_quaternion_unit_norm_tolerance",
    "episode_index",
    "gripper_runtime_dynamic",
    "gripper_runtime_static",
    "guard_diagnostics",
    "joint_effort_limits",
    "joint_names",
    "joint_slew_float32_tolerance_rad",
    "joint_velocity_limit_tolerance_rad_s",
    "joint_velocity_limits_rad_s",
    "max_delta_joint_pos_rad",
    "max_raw_delta_diagnostic",
    "maxima",
    "physics_dt",
    "physx_derived_soft_joint_pos_limits_float32_sha256",
    "physx_derived_soft_joint_pos_limits_rad",
    "physx_derived_soft_limit_profile",
    "physx_hard_joint_pos_limits_float32_sha256",
    "physx_hard_joint_pos_limits_rad",
    "physx_hard_limit_profile",
    "physx_hard_limit_write_count",
    "physx_solver_type",
    "profile",
    "soft_joint_pos_limit_factor",
    "soft_joint_pos_limits_float32_sha256",
    "soft_joint_pos_limits_rad",
    "solver_position_iteration_count",
    "solver_velocity_iteration_count",
    "target_joint_pos_limits_float32_sha256",
    "target_joint_pos_limits_rad",
    "target_soft_limit_guard_band_profile",
    "target_soft_limit_margin_rad",
}
GRIPPER_DYNAMIC_FIELDS = {
    "apply_entry_samples",
    "driver_target_slew",
    "dropped_diagnostics",
    "joint_indices",
    "joint_names",
    "max_abs_joint_acceleration_rad_s2",
    "max_abs_joint_velocity_rad_s",
    "max_velocity_diagnostic",
    "nonfinite_samples",
    "open_endpoint_contact_mimic_impulse",
    "post_policy_step_samples",
    "profile",
    "terminal_state",
}
TELEMETRY_FIELDS = {
    "arm_threshold_profile",
    "arm_velocity_envelopes_rad_s",
    "coupled_impulse_failure_samples",
    "enabled",
    "endpoint",
    "failure_predicate",
    "first_coupled_impulse_failure_diagnostic",
    "follower_threshold_crossing_samples",
    "follower_threshold_rad_s_float32",
    "follower_threshold_semantics",
    "max_abs_arm_joint_velocity_rad_s",
    "max_abs_follower_joint_acceleration_rad_s2",
    "max_abs_follower_joint_velocity_rad_s",
    "maximum_follower_diagnostic",
    "nonfinite_open_endpoint_samples",
    "open_endpoint_samples",
    "passed",
    "profile",
}
RECOVERY_FIELDS = {"contract", "counters", "events", "maxima", "state"}
RECOVERY_CONTRACT_FIELDS = {
    "clean_samples_required",
    "envelope_formula_profile",
    "hard_joint_position_limits_little_endian_float32_sha256",
    "hard_joint_position_limits_rad",
    "hard_limit_profile",
    "hold_profile",
    "joint_names",
    "maximum_active_substeps",
    "physics_dt_float32",
    "predicted_position_profile",
    "profile",
    "relative_envelope_float32",
    "release_ramp_profile",
    "schema_version",
    "transaction_profile",
    "velocity_envelopes_rad_s",
    "velocity_limits_rad_s",
}
RECOVERY_STATE_FIELDS = {
    "active",
    "consecutive_active_substeps",
    "consecutive_clean_samples",
    "phase",
    "release_ramp_next_index",
}
RECOVERY_MAXIMA_FIELDS = {
    "abs_velocity_residual_excess_rad_s",
    "abs_velocity_to_limit_ratio",
    "consecutive_recovery_substeps",
}
CONTROLLER_REPORT_FIELDS = {
    "arm_slew_headroom",
    "concurrent_arm_gripper",
    "current_joint_velocity_recovery",
    "gripper_close_arm_interlock",
}
ARM_SLEW_FIELDS = {
    "enabled",
    "nominal_max_delta_joint_pos_rad",
    "physical_max_delta_joint_pos_rad",
    "profile",
    "ratio",
}
INTERLOCK_FIELDS = {
    "activation_count",
    "active_apply_count",
    "anchor_capture_count",
    "anchor_completion_count",
    "anchor_first_exact_target_count",
    "anchor_open_cancel_count",
    "anchor_position_limit_event_count",
    "anchor_position_limited_joint_count",
    "anchor_refresh_count",
    "anchor_slew_limit_event_count",
    "anchor_slew_limited_joint_count",
    "anchor_target_apply_count",
    "anchor_valid",
    "configured_substeps",
    "enabled",
    "endpoint_observed",
    "last_activation_apply_index",
    "last_anchor_joint_pos_rad",
    "last_anchor_little_endian_float32_sha256",
    "max_abs_active_delta_joint_pos_rad",
    "max_abs_current_anchor_residual_rad",
    "max_abs_released_delta_joint_pos_rad",
    "max_abs_target_anchor_residual_rad",
    "observed_endpoint_change_count",
    "profile",
    "released_apply_count",
    "remaining_substeps",
}

RESULT_FIELDS = {
    "actual_position",
    "actual_quaternion_wxyz",
    "case",
    "final_frame_position_error_m",
    "final_frame_rotation_error_rad",
    "passed",
    "position_error_m",
    "reset_frame_position_error_m",
    "reset_frame_rotation_error_rad",
    "rotation_error_rad",
    "target_position",
    "target_quaternion_wxyz",
}
COUNTER_FIELDS = {
    "apply_calls",
    "current_joint_limit_aborts",
    "dls_fallbacks",
    "environment_substeps",
    "guard_diagnostics_dropped",
    "invariant_aborts",
    "nonfinite_aborts",
    "position_limit_events",
    "position_limited_joints",
    "post_clamp_target_violations",
    "slew_limit_events",
    "slew_limited_joints",
}
MAXIMA_FIELDS = {
    "abs_joint_vel_rad_s",
    "applied_delta_joint_pos_rad",
    "current_joint_soft_limit_violation_rad",
    "current_physx_hard_limit_violation_rad",
    "minimum_outer_joint_clearance_rad",
    "post_clamp_target_guard_band_violation_rad",
    "post_clamp_target_soft_limit_violation_rad",
    "raw_delta_joint_pos_rad",
    "raw_target_soft_limit_violation_rad",
}
DIAGNOSTIC_FIELDS = {
    "eef_quaternion_norm",
    "episode_index",
    "jacobian_finite",
    "jacobian_max_abs",
    "joint_pos_rad",
    "kind",
    "physics_substep",
    "policy_step",
    "pose_error_norm",
    "raw_delta_joint_pos_rad",
    "raw_joint_pos_target_rad",
    "safe_joint_pos_target_rad",
}
DELAYED_CLOSE_FIELDS = {
    "arm_abort_count",
    "case",
    "close_policy_steps",
    "close_transition_substeps",
    "episode_index",
    "ik_safety",
    "open_policy_steps",
    "passed",
    "profile",
    "terminated",
    "truncated",
}
DISCRIMINATOR_FIELDS = {
    "controller_report",
    "distinct_policy_targets",
    "episode_index",
    "expected_apply_calls",
    "expected_closed_endpoint_applies",
    "ik_safety",
    "open_endpoint_contact_mimic_impulse",
    "passed",
    "profile",
    "transition_policy_steps",
    "transition_substeps",
}
ADVERSARIAL_FIELDS = {
    "case",
    "eef_state_is_finite",
    "guard_error",
    "guard_evidence",
    "ik_safety",
    "joint_pos_within_captured_soft_limits",
    "joint_state",
    "joint_state_is_finite",
    "passed",
    "state_is_finite",
    "terminated",
    "truncated",
}
JOINT_STATE_FIELDS = {
    "joint_names",
    "joint_pos_rad",
    "joint_vel_rad_s",
    "position_within_captured_soft_limits",
    "soft_limit_tolerance_rad",
    "soft_limit_violation_rad",
}
GUARD_EVIDENCE_FIELDS = {
    "abort_count",
    "applied_within_bounds",
    "apply_calls",
    "post_clamp_target_violations",
    "slew_limit_events",
}
CLOSE_HEADROOM_FIELDS = {
    "completion_gate_applied",
    "delayed_close_replay",
    "immediate_close_hold",
    "passed",
    "profile",
    "threshold_ratio",
}
CLOSE_HEADROOM_ENTRY_FIELDS = {
    "episode_index",
    "joint_names",
    "joint_velocity_limit_rad_s",
    "max_abs_joint_velocity_rad_s",
    "maximum_ratio",
    "passed",
    "threshold_ratio",
    "velocity_to_limit_ratio",
}
DRIVER_TARGET_SLEW_FIELDS = {
    "apply_calls",
    "endpoint_change_count",
    "endpoint_reached_apply_count",
    "initial_anchor_rad",
    "initialization_count",
    "last_applied_target_rad",
    "last_requested_endpoint_rad",
    "live_limit_validation_count",
    "max_abs_endpoint_error_after_step_rad",
    "max_abs_endpoint_error_before_step_rad",
    "max_abs_target_step_rad",
    "process_action_calls",
    "profile",
    "repeated_endpoint_process_count",
    "slew_limited_apply_count",
}
GRIPPER_DIAGNOSTIC_FIELDS = {
    "joint_acceleration_rad_s2",
    "joint_position_rad",
    "joint_position_target_rad",
    "joint_velocity_rad_s",
    "joint_velocity_target_rad_s",
    "sample_index",
    "sample_phase",
}
GRIPPER_TERMINAL_FIELDS = GRIPPER_DIAGNOSTIC_FIELDS - {"sample_phase"}
FOLLOWER_DIAGNOSTIC_FIELDS = {
    "arm_joint_velocity_rad_s",
    "arm_recovery_envelope_crossed",
    "coupled_impulse_failure",
    "follower_joint_acceleration_rad_s2",
    "follower_joint_velocity_rad_s",
    "follower_threshold_crossed",
    "sample_index",
    "sample_phase",
}

EXPECTED_JOINT_NAMES = [f"panda_joint{index}" for index in range(1, 8)]
EXPECTED_GRIPPER_JOINT_NAMES = [
    "finger_joint",
    "right_outer_knuckle_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
    "left_inner_finger_knuckle_joint",
    "right_inner_finger_knuckle_joint",
]
EXPECTED_VELOCITY_LIMITS = [2.174999952316284] * 4 + [2.609999895095825] * 3
EXPECTED_VELOCITY_ENVELOPES = [2.175217390060425] * 4 + [2.6102609634399414] * 3
EXPECTED_EFFORT_LIMITS = [87.0] * 4 + [12.0] * 3
EXPECTED_PHYSICAL_MAX_DELTA = [0.018125001341104507] * 4 + [0.02174999937415123] * 3
EXPECTED_NOMINAL_MAX_DELTA = [0.017218751832842827] * 4 + [0.020662499591708183] * 3
EXPECTED_SOFT_LIMITS = [
    [-2.8973000049591064, 2.8973000049591064],
    [-1.7627999782562256, 1.7627999782562256],
    [-2.8973000049591064, 2.8973000049591064],
    [-3.0717999935150146, -0.06979990005493164],
    [-2.8973000049591064, 2.8973000049591064],
    [-0.017499923706054688, 3.752500057220459],
    [-2.8973000049591064, 2.8973000049591064],
]
EXPECTED_HARD_LIMITS = [
    [-2.8791749477386475, 2.8791749477386475],
    [-1.7446749210357666, 1.7446749210357666],
    [-2.8791749477386475, 2.8791749477386475],
    [-3.0536749362945557, -0.08792489767074585],
    [-2.8755500316619873, 2.8755500316619873],
    [0.004250075668096542, 3.73075008392334],
    [-2.8755500316619873, 2.8755500316619873],
]
EXPECTED_PHYSX_DERIVED_SOFT_LIMITS = [
    [-2.8791749477386475, 2.8791749477386475],
    [-1.7446749210357666, 1.7446749210357666],
    [-2.8791749477386475, 2.8791749477386475],
    [-3.0536749362945557, -0.08792495727539062],
    [-2.8755500316619873, 2.8755500316619873],
    [0.004250049591064453, 3.73075008392334],
    [-2.8755500316619873, 2.8755500316619873],
]
SOFT_LIMIT_DIGEST = "fbf7535901c042fea5d901812ecd02c5fd81ade06c23c1499c32d66a859104de"
HARD_LIMIT_DIGEST = "09b20ab18c35d6dc22a3edbc2beca2edff419e242dd07d74cd1d65df9ce67e0f"
PHYSX_DERIVED_SOFT_LIMIT_DIGEST = (
    "dd7865f59efb23e96d7d4cbb5e129906b04a42b5e5c0941459bfc8866dd7ecd0"
)
GRIPPER_STATIC_CANONICAL_SHA256 = (
    "48e4e60a53989d0aa4fe5ddbc94fac9cd11c432699fa509e7c27dcf75c0e4495"
)
RECOVERY_CONTRACT_CANONICAL_SHA256 = (
    "870964e7722e99e431b6561d54d77ec00ab62a61d2a146595a3bbb11b9d1717a"
)
GRIPPER_RUNTIME_PROFILE = "implicit_gripper_physx_velocity_limit5_followers5_cuda_actuator_cpu_static_physx_v1"
GRIPPER_TARGET_SLEW_PROFILE = (
    "eef_binary_driver_target_slew_rate1p25_from_live_limit5_"
    "per_120hz_substep_candidate_v1"
)
GRIPPER_CLOSED_TARGET = 0.7853981852531433
GRIPPER_MAX_TARGET_STEP = 0.010416666977107525
FOLLOWER_THRESHOLD = 5.000999927520752


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _reject_constant(token: str) -> None:
    raise VerificationError(f"Nonfinite JSON constant {token!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_json(data: bytes, field: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"Invalid {field}: {error}") from error
    _require(type(value) is dict, f"{field} must be an object")
    return value


def _read_regular_file(path: Path, field: str) -> tuple[bytes, os.stat_result]:
    try:
        before = os.lstat(path)
    except OSError as error:
        raise VerificationError(f"Cannot inspect {field}: {error}") from error
    _require(stat.S_ISREG(before.st_mode), f"{field} is not a regular file")
    _require(before.st_nlink == 1, f"{field} must have exactly one hard link")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise VerificationError(f"Cannot open {field}: {error}") from error
    try:
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(opened.st_mode)
            and (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
            f"{field} changed during secure open",
        )
        metadata_fields = (
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_nlink",
            "st_mode",
        )
        _require(
            all(
                getattr(opened, name) == getattr(before, name)
                for name in metadata_fields
            ),
            f"{field} metadata changed during secure open",
        )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        _require(
            all(
                getattr(after, name) == getattr(opened, name)
                for name in metadata_fields
            ),
            f"{field} changed while being read",
        )
        try:
            linked = os.lstat(path)
        except OSError as error:
            raise VerificationError(f"Cannot re-inspect {field}: {error}") from error
        path_fields = (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_nlink",
            "st_mode",
        )
        _require(
            stat.S_ISREG(linked.st_mode)
            and linked.st_nlink == 1
            and all(
                getattr(linked, name) == getattr(after, name) for name in path_fields
            ),
            f"{field} path changed while being read",
        )
    finally:
        os.close(descriptor)
    data = b"".join(chunks)
    _require(len(data) == after.st_size, f"{field} read size drift")
    return data, after


def _identity(
    path: Path, field: str, spec: Mapping[str, Any]
) -> tuple[dict[str, Any], bytes]:
    data, status = _read_regular_file(path, field)
    actual = {
        "path": str(path),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "mode": f"{stat.S_IMODE(status.st_mode):04o}",
    }
    _require(
        {key: actual[key] for key in ("size_bytes", "sha256", "mode")} == spec,
        f"{field} immutable identity drift",
    )
    return actual, data


def _finite_tree(value: Any, field: str) -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        _require(math.isfinite(value), f"{field} is nonfinite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{field}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{field}.{key}")
        return
    raise VerificationError(f"Unexpected {field} type {type(value).__name__}")


def _typed_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return set(actual) == set(expected) and all(
            _typed_equal(actual[key], expected[key]) for key in expected
        )
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _typed_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _object(value: Any, field: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{field} must be an object")
    return value


def _array(value: Any, field: str, *, length: int | None = None) -> list[Any]:
    _require(type(value) is list, f"{field} must be an array")
    if length is not None:
        _require(len(value) == length, f"{field} must contain {length} entries")
    return value


def _finite_number(value: Any, field: str) -> float:
    _require(
        type(value) is float and math.isfinite(value),
        f"{field} must be a finite JSON float",
    )
    return value


def _exact_int(value: Any, expected: int, field: str) -> int:
    _require(
        type(value) is int and value == expected, f"{field} must be int {expected}"
    )
    return value


def _exact_float(value: Any, expected: float, field: str) -> float:
    actual = _finite_number(value, field)
    _require(actual == expected, f"{field} mismatch")
    return actual


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _finite_vector(value: Any, field: str, *, length: int) -> list[float]:
    return [
        _finite_number(item, f"{field}[{index}]")
        for index, item in enumerate(_array(value, field, length=length))
    ]


def _exact_vector(value: Any, expected: list[float], field: str) -> list[float]:
    actual = _finite_vector(value, field, length=len(expected))
    _require(actual == expected, f"{field} mismatch")
    return actual


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _normalize_quaternion(quaternion: list[float]) -> list[float]:
    norm = math.sqrt(sum(item * item for item in quaternion))
    _require(norm > 0.0, "quaternion norm must be positive")
    return [item / norm for item in quaternion]


def _quaternion_distance(left: list[float], right: list[float]) -> float:
    lhs = _normalize_quaternion(left)
    rhs = _normalize_quaternion(right)
    dot = abs(sum(a * b for a, b in zip(lhs, rhs, strict=True)))
    return 2.0 * math.acos(min(1.0, max(0.0, dot)))


def _quaternion_multiply(left: list[float], right: list[float]) -> list[float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return [
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ]


def _vector_evidence(value: Any, field: str) -> list[float]:
    evidence = _object(value, field)
    _require(
        set(evidence) == {"values", "finite_mask", "finite_count", "max_abs"},
        f"{field} schema drift",
    )
    values = _finite_vector(evidence.get("values"), f"{field}.values", length=7)
    _require(evidence.get("finite_mask") == [True] * 7, f"{field}.finite_mask")
    _exact_int(evidence.get("finite_count"), 7, f"{field}.finite_count")
    maximum = _finite_number(evidence.get("max_abs"), f"{field}.max_abs")
    _require(
        math.isclose(maximum, max(abs(item) for item in values), abs_tol=1e-12),
        f"{field}.max_abs binding",
    )
    return values


def _diagnostic_vector(value: Any, field: str) -> list[float]:
    evidence = _object(value, field)
    _require(
        set(evidence) == {"values", "finite_mask", "finite_count"},
        f"{field} schema drift",
    )
    values = _finite_vector(evidence.get("values"), f"{field}.values", length=7)
    _require(evidence.get("finite_mask") == [True] * 7, f"{field}.finite_mask")
    _exact_int(evidence.get("finite_count"), 7, f"{field}.finite_count")
    return values


def _validate_diagnostic(
    value: Any, *, field: str, episode: int, applies: int, kind: str
) -> dict[str, Any]:
    diagnostic = _object(value, field)
    _require(set(diagnostic) == DIAGNOSTIC_FIELDS, f"{field} schema drift")
    _require(diagnostic.get("kind") == kind, f"{field}.kind")
    _exact_int(diagnostic.get("episode_index"), episode, f"{field}.episode")
    policy_step = diagnostic.get("policy_step")
    physics_substep = diagnostic.get("physics_substep")
    _require(type(policy_step) is int and policy_step >= 0, f"{field}.policy_step")
    _require(
        type(physics_substep) is int and 0 <= physics_substep < 8,
        f"{field}.physics_substep",
    )
    _require(policy_step * 8 + physics_substep < applies, f"{field} cadence")
    for name in (
        "joint_pos_rad",
        "raw_delta_joint_pos_rad",
        "raw_joint_pos_target_rad",
        "safe_joint_pos_target_rad",
    ):
        _diagnostic_vector(diagnostic.get(name), f"{field}.{name}")
    _require(diagnostic.get("jacobian_finite") is True, f"{field}.jacobian_finite")
    for name in ("pose_error_norm", "jacobian_max_abs"):
        _require(_finite_number(diagnostic.get(name), f"{field}.{name}") >= 0.0, name)
    _require(diagnostic.get("eef_quaternion_norm") is None, f"{field}.quaternion")
    return diagnostic


def _validate_gripper_static(value: Any, field: str) -> None:
    contract = _object(value, field)
    expected_fields = {
        "actuator_joint_ownership",
        "device_partition",
        "driver_actuator",
        "driver_joint_index",
        "driver_joint_name",
        "driver_target_slew",
        "follower_joint_indices",
        "follower_joint_names",
        "gripper_joint_indices",
        "gripper_joint_names",
        "joint_names",
        "measured_velocity_is_hard_bounded_by_limit",
        "mimic_compliance",
        "mimic_joint_contract",
        "profile",
        "velocity_limit_write_contract",
        "velocity_limits_after_write",
        "velocity_limits_before_write",
    }
    _require(set(contract) == expected_fields, f"{field} schema drift")
    _require(
        _canonical_sha256(contract) == GRIPPER_STATIC_CANONICAL_SHA256,
        f"{field} canonical contract drift",
    )
    _require(
        contract.get("profile") == GRIPPER_RUNTIME_PROFILE
        and contract.get("joint_names")
        == [*EXPECTED_JOINT_NAMES, *EXPECTED_GRIPPER_JOINT_NAMES]
        and contract.get("gripper_joint_names") == EXPECTED_GRIPPER_JOINT_NAMES
        and contract.get("gripper_joint_indices") == list(range(7, 13))
        and contract.get("driver_joint_name") == "finger_joint"
        and contract.get("driver_joint_index") == 7
        and contract.get("follower_joint_names") == EXPECTED_GRIPPER_JOINT_NAMES[1:]
        and contract.get("follower_joint_indices") == list(range(8, 13))
        and contract.get("measured_velocity_is_hard_bounded_by_limit") is False,
        f"{field} identity drift",
    )


def _validate_driver_target_slew(
    value: Any, *, field: str, applies: int, mode: str
) -> None:
    report = _object(value, field)
    _require(set(report) == DRIVER_TARGET_SLEW_FIELDS, f"{field} schema drift")
    _require(report.get("profile") == GRIPPER_TARGET_SLEW_PROFILE, f"{field}.profile")
    process_calls = applies // 8
    expected_changes = {
        "empty": 0,
        "open": 0,
        "close": 0,
        "delayed_close": 1,
        "roundtrip": 2,
    }[mode]
    expected_limited = {
        "empty": 0,
        "open": 0,
        "close": 75,
        "delayed_close": 75,
        "roundtrip": 150,
    }[mode]
    integer_expectations = {
        "process_action_calls": process_calls,
        "apply_calls": applies,
        "initialization_count": int(applies > 0),
        "endpoint_change_count": expected_changes,
        "repeated_endpoint_process_count": max(process_calls - 1 - expected_changes, 0),
        "slew_limited_apply_count": expected_limited,
        "endpoint_reached_apply_count": applies - expected_limited,
        "live_limit_validation_count": applies,
    }
    for name, expected in integer_expectations.items():
        _exact_int(report.get(name), expected, f"{field}.{name}")
    maxima = [
        _finite_number(report.get(name), f"{field}.{name}")
        for name in (
            "max_abs_target_step_rad",
            "max_abs_endpoint_error_before_step_rad",
            "max_abs_endpoint_error_after_step_rad",
        )
    ]
    _require(all(item >= 0.0 for item in maxima), f"{field}.maxima")
    if mode in {"empty", "open"}:
        _require(maxima == [0.0, 0.0, 0.0], f"{field}.open maxima")
    else:
        _require(
            struct.pack("<f", maxima[0]) == struct.pack("<f", GRIPPER_MAX_TARGET_STEP)
            and struct.pack("<f", maxima[1]) == struct.pack("<f", GRIPPER_CLOSED_TARGET)
            and 0.0 <= maxima[2] <= maxima[1],
            f"{field}.transition maxima",
        )
    if mode == "empty":
        _require(
            report.get("initial_anchor_rad") is None
            and report.get("last_requested_endpoint_rad") is None
            and report.get("last_applied_target_rad") is None,
            f"{field}.empty state",
        )
    else:
        _exact_float(report.get("initial_anchor_rad"), 0.0, f"{field}.anchor")
        expected_endpoint = (
            GRIPPER_CLOSED_TARGET if mode in {"close", "delayed_close"} else 0.0
        )
        _exact_float(
            report.get("last_requested_endpoint_rad"),
            expected_endpoint,
            f"{field}.request",
        )
        _exact_float(
            report.get("last_applied_target_rad"), expected_endpoint, f"{field}.applied"
        )


def _validate_open_telemetry(
    value: Any, *, field: str, total_samples: int, expected_open_samples: int
) -> dict[str, Any]:
    telemetry = _object(value, field)
    _require(set(telemetry) == TELEMETRY_FIELDS, f"{field} schema drift")
    _require(
        telemetry.get("enabled") is True
        and telemetry.get("profile")
        == "open_endpoint_follower5p001_and_arm_float32_recovery_envelope_v1"
        and telemetry.get("endpoint") == "open"
        and telemetry.get("follower_threshold_semantics")
        == "passive_follower_crossing_is_telemetry_only_v1"
        and telemetry.get("arm_threshold_profile")
        == "per_joint_float32_physical_limit_plus_limit_times_float32_1e_4_v1"
        and telemetry.get("failure_predicate")
        == "open_and_follower_gt_5p001_and_any_arm_gt_its_recovery_envelope_v1",
        f"{field} identity drift",
    )
    _exact_float(
        telemetry.get("follower_threshold_rad_s_float32"), FOLLOWER_THRESHOLD, field
    )
    _exact_vector(
        telemetry.get("arm_velocity_envelopes_rad_s"),
        EXPECTED_VELOCITY_ENVELOPES,
        f"{field}.arm_envelopes",
    )
    _exact_int(
        telemetry.get("open_endpoint_samples"),
        expected_open_samples,
        f"{field}.samples",
    )
    _require(0 <= expected_open_samples <= total_samples, f"{field}.sample cadence")
    for name in (
        "nonfinite_open_endpoint_samples",
        "follower_threshold_crossing_samples",
        "coupled_impulse_failure_samples",
    ):
        _exact_int(telemetry.get(name), 0, f"{field}.{name}")
    _require(
        telemetry.get("passed") is True
        and telemetry.get("first_coupled_impulse_failure_diagnostic") is None,
        f"{field}.failure state",
    )
    arm_max = _finite_vector(
        telemetry.get("max_abs_arm_joint_velocity_rad_s"), f"{field}.arm_max", length=7
    )
    follower_max = _finite_vector(
        telemetry.get("max_abs_follower_joint_velocity_rad_s"),
        f"{field}.follower_max",
        length=5,
    )
    acceleration_max = _finite_vector(
        telemetry.get("max_abs_follower_joint_acceleration_rad_s2"),
        f"{field}.acceleration_max",
        length=5,
    )
    _require(
        all(item >= 0.0 for item in [*arm_max, *follower_max, *acceleration_max])
        and all(
            actual <= limit
            for actual, limit in zip(arm_max, EXPECTED_VELOCITY_ENVELOPES, strict=True)
        )
        and all(item <= FOLLOWER_THRESHOLD for item in follower_max),
        f"{field}.aggregate maxima",
    )
    maximum = telemetry.get("maximum_follower_diagnostic")
    if expected_open_samples == 0:
        _require(maximum is None, f"{field}.zero-sample maximum")
        _require(
            arm_max == [0.0] * 7
            and follower_max == [0.0] * 5
            and acceleration_max == [0.0] * 5,
            f"{field}.zero-sample maxima",
        )
        return telemetry
    maximum = _object(maximum, f"{field}.maximum")
    _require(set(maximum) == FOLLOWER_DIAGNOSTIC_FIELDS, f"{field}.maximum schema")
    _require(maximum.get("sample_phase") in {"apply_entry", "post_policy_step"}, field)
    sample_index = maximum.get("sample_index")
    _require(type(sample_index) is int and 0 <= sample_index < total_samples, field)
    arm = _finite_vector(
        maximum.get("arm_joint_velocity_rad_s"), f"{field}.maximum.arm", length=7
    )
    follower = _finite_vector(
        maximum.get("follower_joint_velocity_rad_s"),
        f"{field}.maximum.follower",
        length=5,
    )
    _finite_vector(
        maximum.get("follower_joint_acceleration_rad_s2"),
        f"{field}.maximum.acceleration",
        length=5,
    )
    follower_crossed = any(abs(item) > FOLLOWER_THRESHOLD for item in follower)
    arm_crossed = any(
        abs(item) > limit
        for item, limit in zip(arm, EXPECTED_VELOCITY_ENVELOPES, strict=True)
    )
    _require(
        maximum.get("follower_threshold_crossed") is follower_crossed
        and maximum.get("arm_recovery_envelope_crossed") is arm_crossed
        and maximum.get("coupled_impulse_failure")
        is (follower_crossed and arm_crossed),
        f"{field}.maximum predicates",
    )
    _require(
        math.isclose(
            max(abs(item) for item in follower),
            max(follower_max),
            rel_tol=0.0,
            abs_tol=1e-6,
        ),
        f"{field}.maximum/aggregate binding",
    )
    return telemetry


def _validate_gripper_dynamic(
    value: Any,
    *,
    field: str,
    applies: int,
    driver_mode: str,
    expected_open_samples: int,
) -> dict[str, Any]:
    report = _object(value, field)
    _require(set(report) == GRIPPER_DYNAMIC_FIELDS, f"{field} schema drift")
    _require(
        report.get("profile") == GRIPPER_RUNTIME_PROFILE
        and report.get("joint_names") == EXPECTED_GRIPPER_JOINT_NAMES
        and report.get("joint_indices") == list(range(7, 13)),
        f"{field}.identity",
    )
    post_samples = applies // 8
    for name, expected in (
        ("apply_entry_samples", applies),
        ("post_policy_step_samples", post_samples),
        ("nonfinite_samples", 0),
        ("dropped_diagnostics", 0),
    ):
        _exact_int(report.get(name), expected, f"{field}.{name}")
    physical_maxima = {}
    for name in ("max_abs_joint_velocity_rad_s", "max_abs_joint_acceleration_rad_s2"):
        vector = _finite_vector(report.get(name), f"{field}.{name}", length=6)
        _require(all(item >= 0.0 for item in vector), f"{field}.{name}")
        physical_maxima[name] = vector
    diagnostic = report.get("max_velocity_diagnostic")
    terminal = report.get("terminal_state")
    if applies == 0:
        _require(diagnostic is None and terminal is None, f"{field}.empty diagnostics")
        _require(
            all(item == 0.0 for vector in physical_maxima.values() for item in vector),
            f"{field}.empty maxima",
        )
    else:
        diagnostic = _object(diagnostic, f"{field}.max_velocity_diagnostic")
        _require(
            set(diagnostic) == GRIPPER_DIAGNOSTIC_FIELDS, f"{field}.diagnostic schema"
        )
        _require(
            diagnostic.get("sample_phase") in {"apply_entry", "post_policy_step"}, field
        )
        sample_index = diagnostic.get("sample_index")
        _require(
            type(sample_index) is int and 0 <= sample_index < applies + post_samples,
            field,
        )
        diagnostic_vectors = {
            name: _finite_vector(
                diagnostic.get(name), f"{field}.diagnostic.{name}", length=6
            )
            for name in GRIPPER_DIAGNOSTIC_FIELDS - {"sample_phase", "sample_index"}
        }
        _require(
            math.isclose(
                max(abs(item) for item in diagnostic_vectors["joint_velocity_rad_s"]),
                max(physical_maxima["max_abs_joint_velocity_rad_s"]),
                rel_tol=0.0,
                abs_tol=1e-6,
            ),
            f"{field}.diagnostic/velocity maximum",
        )
        terminal = _object(terminal, f"{field}.terminal")
        _require(set(terminal) == GRIPPER_TERMINAL_FIELDS, f"{field}.terminal schema")
        _exact_int(
            terminal.get("sample_index"),
            (applies // 8) * 9 - 1,
            f"{field}.terminal sample",
        )
        for name in GRIPPER_TERMINAL_FIELDS - {"sample_index"}:
            _finite_vector(terminal.get(name), f"{field}.terminal.{name}", length=6)
    _validate_driver_target_slew(
        report.get("driver_target_slew"),
        field=f"{field}.driver",
        applies=applies,
        mode=driver_mode,
    )
    return _validate_open_telemetry(
        report.get("open_endpoint_contact_mimic_impulse"),
        field=f"{field}.open_endpoint",
        total_samples=applies + post_samples,
        expected_open_samples=expected_open_samples,
    )


def _validate_recovery(
    value: Any, *, field: str, safety_velocity_maxima: list[float]
) -> dict[str, Any]:
    recovery = _object(value, field)
    _require(set(recovery) == RECOVERY_FIELDS, f"{field} schema drift")
    contract = _object(recovery.get("contract"), f"{field}.contract")
    _require(set(contract) == RECOVERY_CONTRACT_FIELDS, f"{field}.contract schema")
    _require(
        _canonical_sha256(contract) == RECOVERY_CONTRACT_CANONICAL_SHA256,
        f"{field}.contract canonical drift",
    )
    _require(
        contract.get("schema_version") == 4
        and contract.get("profile")
        == "current_joint_velocity_residual_hold_concurrent_resume_v2"
        and contract.get("maximum_active_substeps") == 8
        and contract.get("clean_samples_required") == 2
        and contract.get("release_ramp_profile") is None
        and contract.get("joint_names") == EXPECTED_JOINT_NAMES
        and contract.get("hard_joint_position_limits_little_endian_float32_sha256")
        == HARD_LIMIT_DIGEST,
        f"{field}.contract identity",
    )
    _exact_vector(
        contract.get("velocity_limits_rad_s"),
        EXPECTED_VELOCITY_LIMITS,
        f"{field}.limits",
    )
    _exact_vector(
        contract.get("velocity_envelopes_rad_s"),
        EXPECTED_VELOCITY_ENVELOPES,
        f"{field}.envelopes",
    )
    state = _object(recovery.get("state"), f"{field}.state")
    _require(
        set(state) == RECOVERY_STATE_FIELDS
        and _typed_equal(
            state,
            {
                "phase": "inactive",
                "active": False,
                "consecutive_active_substeps": 0,
                "consecutive_clean_samples": 0,
                "release_ramp_next_index": None,
            },
        ),
        f"{field}.state drift",
    )
    counters = _object(recovery.get("counters"), f"{field}.counters")
    _require(set(counters) == RECOVERY_ZERO_COUNTERS, f"{field}.counter schema")
    _require(
        all(type(item) is int and item == 0 for item in counters.values()),
        f"{field}.counter nonzero",
    )
    _require(recovery.get("events") == [], f"{field}.events")
    maxima = _object(recovery.get("maxima"), f"{field}.maxima")
    _require(set(maxima) == RECOVERY_MAXIMA_FIELDS, f"{field}.maxima schema")
    _exact_int(maxima.get("consecutive_recovery_substeps"), 0, f"{field}.consecutive")
    _exact_vector(
        maxima.get("abs_velocity_residual_excess_rad_s"), [0.0] * 7, f"{field}.residual"
    )
    ratio = _finite_number(maxima.get("abs_velocity_to_limit_ratio"), f"{field}.ratio")
    expected_ratio = max(
        _float32(_float32(velocity) / _float32(limit))
        for velocity, limit in zip(
            safety_velocity_maxima, EXPECTED_VELOCITY_LIMITS, strict=True
        )
    )
    _require(ratio == expected_ratio, f"{field}.velocity ratio binding")
    return recovery


def _validate_safety(
    report: Any,
    *,
    episode: int | None,
    applies: int,
    driver_mode: str,
    expected_open_samples: int,
) -> tuple[dict[str, int], dict[str, list[float]], dict[str, Any]]:
    report = _object(report, "safety report")
    _require(set(report) == SAFETY_FIELDS, "safety report schema drift")
    _require(report.get("profile") == IK_SAFETY_PROFILE, "IK-safety profile drift")
    if episode is None:
        _require(report.get("episode_index") is None, "IK-safety episode drift")
    else:
        _exact_int(report.get("episode_index"), episode, "IK-safety episode")
    for name, expected in {
        "apply_actions_cadence": "physics_substep",
        "target_soft_limit_guard_band_profile": "eef_physx_inner_hardlimit_one_substep_v2",
        "physx_hard_limit_profile": "outer_minus_one_velocity_substep_v1",
        "physx_derived_soft_limit_profile": "isaaclab_midpoint_range_factor1_float32_v1",
        "arm_velocity_target_profile": "zero_per_physics_substep_v1",
        "articulation_solver_profile": "tgs_position64_velocity1_eef_only_v1",
        "articulation_solver_readback": "composed_usd_physx_articulation_api_all_env_roots_v1",
        "target_joint_pos_limits_float32_sha256": HARD_LIMIT_DIGEST,
        "physx_hard_joint_pos_limits_float32_sha256": HARD_LIMIT_DIGEST,
        "physx_derived_soft_joint_pos_limits_float32_sha256": PHYSX_DERIVED_SOFT_LIMIT_DIGEST,
        "soft_joint_pos_limits_float32_sha256": SOFT_LIMIT_DIGEST,
    }.items():
        _require(type(report.get(name)) is str and report.get(name) == expected, name)
    for name, expected in {
        "decimation": 8,
        "physx_hard_limit_write_count": 1,
        "physx_solver_type": 1,
        "solver_position_iteration_count": 64,
        "solver_velocity_iteration_count": 1,
    }.items():
        _exact_int(report.get(name), expected, name)
    for name, expected in {
        "physics_dt": 1 / 120,
        "control_dt": 1 / 15,
        "current_joint_soft_limit_tolerance_rad": 1e-5,
        "eef_quaternion_unit_norm_tolerance": 1e-3,
        "joint_slew_float32_tolerance_rad": 1e-6,
        "joint_velocity_limit_tolerance_rad_s": 1e-5,
        "soft_joint_pos_limit_factor": 1.0,
    }.items():
        _exact_float(report.get(name), expected, name)
    _require(report.get("joint_names") == EXPECTED_JOINT_NAMES, "joint names")
    _exact_vector(
        report.get("joint_velocity_limits_rad_s"),
        EXPECTED_VELOCITY_LIMITS,
        "velocity limits",
    )
    _exact_vector(
        report.get("joint_effort_limits"), EXPECTED_EFFORT_LIMITS, "effort limits"
    )
    _exact_vector(
        report.get("max_delta_joint_pos_rad"), EXPECTED_PHYSICAL_MAX_DELTA, "max delta"
    )
    _exact_vector(
        report.get("target_soft_limit_margin_rad"),
        EXPECTED_PHYSICAL_MAX_DELTA,
        "target margin",
    )
    for name, expected in (
        ("target_joint_pos_limits_rad", EXPECTED_HARD_LIMITS),
        ("physx_hard_joint_pos_limits_rad", EXPECTED_HARD_LIMITS),
        ("physx_derived_soft_joint_pos_limits_rad", EXPECTED_PHYSX_DERIVED_SOFT_LIMITS),
        ("soft_joint_pos_limits_rad", EXPECTED_SOFT_LIMITS),
    ):
        matrix = _array(report.get(name), name, length=7)
        for index, (actual, wanted) in enumerate(zip(matrix, expected, strict=True)):
            _exact_vector(actual, wanted, f"{name}[{index}]")
    _exact_vector(report.get("arm_velocity_target_rad_s"), [0.0] * 7, "velocity target")
    _require(
        report.get("current_joint_velocity_abort") is None, "unexpected velocity abort"
    )
    _validate_gripper_static(report.get("gripper_runtime_static"), "gripper static")

    counters = _object(report.get("counters"), "safety counters")
    _require(set(counters) == COUNTER_FIELDS, "safety counter schema drift")
    _require(
        all(type(item) is int and item >= 0 for item in counters.values()),
        "safety counters",
    )
    _exact_int(counters.get("apply_calls"), applies, "apply calls")
    _exact_int(counters.get("environment_substeps"), applies, "environment substeps")
    for event, joints in (
        ("slew_limit_events", "slew_limited_joints"),
        ("position_limit_events", "position_limited_joints"),
    ):
        _require(
            counters[event] <= applies
            and counters[event] <= counters[joints] <= 7 * counters[event],
            f"{event}/{joints} impossible",
        )
    _require(all(counters[name] == 0 for name in CRITICAL_COUNTERS), "critical counter")
    _require(
        counters["position_limit_events"] == 0
        and counters["position_limited_joints"] == 0,
        "position limiter unexpectedly active",
    )

    maxima_value = _object(report.get("maxima"), "safety maxima")
    _require(set(maxima_value) == MAXIMA_FIELDS, "safety maxima schema drift")
    maxima = {
        name: _finite_vector(vector, f"maxima.{name}", length=7)
        for name, vector in maxima_value.items()
    }
    _require(
        all(
            item >= 0.0
            for name, vector in maxima.items()
            if name != "minimum_outer_joint_clearance_rad"
            for item in vector
        ),
        "negative safety maximum",
    )
    _require(
        maxima["post_clamp_target_soft_limit_violation_rad"] == [0.0] * 7
        and maxima["post_clamp_target_guard_band_violation_rad"] == [0.0] * 7
        and maxima["current_joint_soft_limit_violation_rad"] == [0.0] * 7
        and maxima["current_physx_hard_limit_violation_rad"] == [0.0] * 7,
        "limit maximum nonzero",
    )
    _require(
        all(
            actual <= limit + 1e-5
            for actual, limit in zip(
                maxima["abs_joint_vel_rad_s"], EXPECTED_VELOCITY_LIMITS, strict=True
            )
        )
        and all(
            actual <= bound + 1e-6
            for actual, bound in zip(
                maxima["applied_delta_joint_pos_rad"],
                EXPECTED_PHYSICAL_MAX_DELTA,
                strict=True,
            )
        ),
        "velocity/slew maximum exceeded",
    )
    _require(
        all(item >= 0.0 for item in maxima["minimum_outer_joint_clearance_rad"]),
        "negative joint clearance",
    )
    raw_slew_activated = any(
        raw > bound
        for raw, bound in zip(
            maxima["raw_delta_joint_pos_rad"], EXPECTED_NOMINAL_MAX_DELTA, strict=True
        )
    )
    _require(
        (counters["slew_limit_events"] > 0) is raw_slew_activated,
        "slew counter/maxima mismatch",
    )
    _require(report.get("guard_diagnostics") == [], "unexpected guard diagnostics")
    maximum_diagnostic = report.get("max_raw_delta_diagnostic")
    if applies == 0:
        _require(maximum_diagnostic is None, "initial max diagnostic")
        _require(all(item == 0 for item in counters.values()), "initial counters")
        _require(
            all(item == 0.0 for vector in maxima.values() for item in vector),
            "initial maxima",
        )
    else:
        _require(episode is not None, "nonempty safety report lacks episode")
        diagnostic = _validate_diagnostic(
            maximum_diagnostic,
            field="max raw diagnostic",
            episode=episode,
            applies=applies,
            kind="max_raw_delta",
        )
        raw_delta = _diagnostic_vector(
            diagnostic["raw_delta_joint_pos_rad"], "diagnostic raw delta"
        )
        _require(
            math.isclose(
                max(abs(item) for item in raw_delta),
                max(maxima["raw_delta_joint_pos_rad"]),
                rel_tol=0.0,
                abs_tol=1e-6,
            ),
            "max diagnostic binding",
        )
        q = _diagnostic_vector(diagnostic["joint_pos_rad"], "diagnostic q")
        raw_target = _diagnostic_vector(
            diagnostic["raw_joint_pos_target_rad"], "diagnostic raw target"
        )
        safe_target = _diagnostic_vector(
            diagnostic["safe_joint_pos_target_rad"], "diagnostic safe target"
        )
        for index, (position, delta, raw, safe, bound, limits) in enumerate(
            zip(
                q,
                raw_delta,
                raw_target,
                safe_target,
                EXPECTED_PHYSICAL_MAX_DELTA,
                EXPECTED_HARD_LIMITS,
                strict=True,
            )
        ):
            _require(
                math.isclose(raw, position + delta, abs_tol=1e-6), f"raw target {index}"
            )
            _require(abs(safe - position) <= bound + 1e-6, f"safe slew {index}")
            _require(
                limits[0] - 1e-5 <= safe <= limits[1] + 1e-5, f"safe limit {index}"
            )

    _validate_recovery(
        report.get("current_joint_velocity_recovery"),
        field="velocity recovery",
        safety_velocity_maxima=maxima["abs_joint_vel_rad_s"],
    )
    telemetry = _validate_gripper_dynamic(
        report.get("gripper_runtime_dynamic"),
        field="gripper dynamic",
        applies=applies,
        driver_mode=driver_mode,
        expected_open_samples=expected_open_samples,
    )
    return counters, maxima, telemetry


def _validate_ordinary_result(
    value: Any,
    *,
    index: int,
    position_tolerance: float,
    rotation_tolerance: float,
    frame_position_tolerance: float,
    frame_rotation_tolerance: float,
) -> None:
    field = f"ordinary[{index}]"
    result = _object(value, field)
    _require(set(result) == RESULT_FIELDS, f"{field} schema drift")
    _require(
        result.get("case") == EXPECTED_CASES[index] and result.get("passed") is True,
        f"{field} identity",
    )
    position_error = _finite_number(
        result.get("position_error_m"), f"{field}.position_error"
    )
    rotation_error = _finite_number(
        result.get("rotation_error_rad"), f"{field}.rotation_error"
    )
    _require(0.0 <= position_error <= position_tolerance, f"{field} position error")
    _require(0.0 <= rotation_error <= rotation_tolerance, f"{field} rotation error")
    target_position = _finite_vector(
        result.get("target_position"), f"{field}.target", length=3
    )
    actual_position = _finite_vector(
        result.get("actual_position"), f"{field}.actual", length=3
    )
    _require(
        math.isclose(
            math.dist(target_position, actual_position), position_error, abs_tol=1e-8
        ),
        f"{field} position error binding",
    )
    quaternions = {}
    for name in ("target_quaternion_wxyz", "actual_quaternion_wxyz"):
        quaternion = _finite_vector(result.get(name), f"{field}.{name}", length=4)
        _require(
            abs(math.sqrt(sum(item * item for item in quaternion)) - 1.0) <= 1e-3,
            f"{field}.{name} norm",
        )
        quaternions[name] = quaternion
    _require(
        math.isclose(
            _quaternion_distance(
                quaternions["target_quaternion_wxyz"],
                quaternions["actual_quaternion_wxyz"],
            ),
            rotation_error,
            abs_tol=1e-7,
        ),
        f"{field} rotation error binding",
    )
    for name, tolerance in (
        ("reset_frame_position_error_m", frame_position_tolerance),
        ("reset_frame_rotation_error_rad", frame_rotation_tolerance),
        ("final_frame_position_error_m", frame_position_tolerance),
        ("final_frame_rotation_error_rad", frame_rotation_tolerance),
    ):
        error = _finite_number(result.get(name), f"{field}.{name}")
        _require(0.0 <= error <= tolerance, f"{field}.{name}")


def _validate_case_geometry(ordinary: list[Any]) -> None:
    hold = _object(ordinary[0], "ordinary hold")
    hold_position = _finite_vector(
        hold.get("target_position"), "hold position", length=3
    )
    hold_quaternion = _finite_vector(
        hold.get("target_quaternion_wxyz"), "hold quaternion", length=4
    )
    for case_index, axis, sign in (
        (1, 0, 1.0),
        (2, 0, -1.0),
        (3, 1, 1.0),
        (4, 1, -1.0),
        (5, 2, 1.0),
        (6, 2, -1.0),
    ):
        case = _object(ordinary[case_index], f"ordinary[{case_index}]")
        target = _finite_vector(
            case.get("target_position"), "translation target", length=3
        )
        expected = hold_position.copy()
        expected[axis] += sign * 0.04
        _require(
            all(
                math.isclose(actual, wanted, abs_tol=1e-6)
                for actual, wanted in zip(target, expected, strict=True)
            ),
            f"ordinary[{case_index}] translation geometry",
        )
        quaternion = _finite_vector(
            case.get("target_quaternion_wxyz"), "translation quaternion", length=4
        )
        _require(
            _quaternion_distance(quaternion, hold_quaternion) <= 1e-6,
            f"ordinary[{case_index}] translation rotation",
        )
    half_angle = math.radians(15.0) / 2.0
    for case_index, axis, sign in (
        (7, 0, 1.0),
        (8, 0, -1.0),
        (9, 1, 1.0),
        (10, 1, -1.0),
        (11, 2, 1.0),
        (12, 2, -1.0),
    ):
        case = _object(ordinary[case_index], f"ordinary[{case_index}]")
        target = _finite_vector(
            case.get("target_position"), "rotation target", length=3
        )
        _require(
            all(
                math.isclose(actual, wanted, abs_tol=1e-6)
                for actual, wanted in zip(target, hold_position, strict=True)
            ),
            f"ordinary[{case_index}] rotation position",
        )
        delta = [math.cos(half_angle), 0.0, 0.0, 0.0]
        delta[axis + 1] = sign * math.sin(half_angle)
        expected = _quaternion_multiply(hold_quaternion, delta)
        quaternion = _finite_vector(
            case.get("target_quaternion_wxyz"), "rotation quaternion", length=4
        )
        _require(
            _quaternion_distance(quaternion, expected) <= 2e-6,
            f"ordinary[{case_index}] rotation geometry",
        )


def _validate_discriminator_geometry(
    value: Any, hold: Mapping[str, Any]
) -> list[list[float]]:
    targets = _array(value, "discriminator targets", length=21)
    vectors = [
        _finite_vector(item, f"discriminator target[{index}]", length=7)
        for index, item in enumerate(targets)
    ]
    hold_position = _finite_vector(
        hold.get("target_position"), "hold position", length=3
    )
    hold_quaternion = _finite_vector(
        hold.get("target_quaternion_wxyz"), "hold quaternion", length=4
    )
    for index, target in enumerate(vectors):
        if index == 0:
            expected_position = hold_position
        elif index <= 10:
            expected_position = [
                hold_position[0] + 0.002 * index,
                hold_position[1],
                hold_position[2],
            ]
        else:
            step = index - 10
            expected_position = [
                hold_position[0] + 0.002 * step,
                hold_position[1] + 0.0004 * step,
                hold_position[2],
            ]
        _require(
            all(
                math.isclose(actual, wanted, abs_tol=1e-6)
                for actual, wanted in zip(target[:3], expected_position, strict=True)
            )
            and _quaternion_distance(target[3:], hold_quaternion) <= 1e-6,
            f"discriminator target[{index}] geometry",
        )
    _require(len({tuple(item) for item in vectors}) == 21, "discriminator diversity")
    return vectors


def _validate_headroom_entry(
    value: Any, *, field: str, episode: int, safety_maxima: list[float]
) -> None:
    entry = _object(value, field)
    _require(set(entry) == CLOSE_HEADROOM_ENTRY_FIELDS, f"{field} schema drift")
    _exact_int(entry.get("episode_index"), episode, f"{field}.episode")
    _require(entry.get("joint_names") == EXPECTED_JOINT_NAMES, f"{field}.joints")
    _exact_vector(
        entry.get("max_abs_joint_velocity_rad_s"), safety_maxima, f"{field}.maxima"
    )
    _exact_vector(
        entry.get("joint_velocity_limit_rad_s"),
        EXPECTED_VELOCITY_LIMITS,
        f"{field}.limits",
    )
    ratios = _finite_vector(
        entry.get("velocity_to_limit_ratio"), f"{field}.ratios", length=7
    )
    expected = [
        maximum / limit
        for maximum, limit in zip(safety_maxima, EXPECTED_VELOCITY_LIMITS, strict=True)
    ]
    _require(
        all(
            math.isclose(a, b, abs_tol=1e-12)
            for a, b in zip(ratios, expected, strict=True)
        ),
        f"{field}.ratio binding",
    )
    maximum = _finite_number(entry.get("maximum_ratio"), f"{field}.maximum")
    _exact_float(entry.get("threshold_ratio"), 0.95, f"{field}.threshold")
    _require(
        math.isclose(maximum, max(expected), abs_tol=1e-12)
        and maximum <= 0.95
        and entry.get("passed") is True,
        f"{field}.gate",
    )


def _validate_adversarial_joint_state(value: Any) -> tuple[list[float], list[float]]:
    state = _object(value, "adversarial joint state")
    _require(set(state) == JOINT_STATE_FIELDS, "adversarial joint-state schema")
    _require(
        state.get("joint_names") == EXPECTED_JOINT_NAMES
        and state.get("position_within_captured_soft_limits") is True,
        "adversarial joint-state identity",
    )
    q = _vector_evidence(state.get("joint_pos_rad"), "adversarial q")
    dq = _vector_evidence(state.get("joint_vel_rad_s"), "adversarial dq")
    violations = _vector_evidence(
        state.get("soft_limit_violation_rad"), "adversarial soft violations"
    )
    tolerance = _exact_float(
        state.get("soft_limit_tolerance_rad"), 1e-5, "adversarial tolerance"
    )
    _require(
        all(
            abs(velocity) <= limit + 1e-6
            for velocity, limit in zip(dq, EXPECTED_VELOCITY_LIMITS, strict=True)
        )
        and all(item <= tolerance for item in violations),
        "adversarial finite-state limits",
    )
    for index, (position, limits) in enumerate(
        zip(q, EXPECTED_SOFT_LIMITS, strict=True)
    ):
        _require(
            limits[0] - tolerance <= position <= limits[1] + tolerance,
            f"adversarial q[{index}] outside soft limits",
        )
    return q, dq


def _semantic_summary(raw: Mapping[str, Any]) -> dict[str, Any]:
    _finite_tree(raw, "raw")
    raw = _object(raw, "raw")
    _require(set(raw) == RAW_FIELDS, "raw top-level schema drift")
    _exact_int(raw.get("schema_version"), 3, "raw schema")
    _require(
        raw.get("finalized") is False
        and raw.get("passed") is False
        and raw.get("stage") == "simulation_app_close_pending"
        and raw.get("case") is None,
        "raw lifecycle drift",
    )
    _exact_int(raw.get("exit_code"), 0, "raw exit code")
    _require(
        raw.get("failure") is None
        and raw.get("terminal_failure_evidence") is None
        and raw.get("close_failures") == []
        and raw.get("persistence_failures") == [],
        "raw failure evidence present",
    )
    _require(
        raw.get("environment") == "DROID-FoodBussing"
        and raw.get("eef_frame") == "panda_link8"
        and raw.get("eef_controller_profile") == CONTROLLER_PROFILE,
        "raw environment/controller identity drift",
    )
    _exact_int(raw.get("hold_steps"), 45, "raw hold steps")
    for name, expected in (
        ("position_delta_m", 0.04),
        ("rotation_delta_deg", 15.0),
        ("position_tolerance_m", 0.01),
        ("rotation_tolerance_deg", 5.0),
        ("frame_position_tolerance_m", 1e-5),
        ("frame_rotation_tolerance_deg", 0.01),
    ):
        _exact_float(raw.get(name), expected, f"raw {name}")

    initial_counters, initial_maxima, _ = _validate_safety(
        raw.get("raw_ik_safety_capture"),
        episode=None,
        applies=0,
        driver_mode="empty",
        expected_open_samples=0,
    )
    _require(
        all(item == 0 for item in initial_counters.values())
        and all(item == 0.0 for vector in initial_maxima.values() for item in vector),
        "initial safety capture is not empty",
    )

    ordinary = _array(raw.get("results"), "ordinary results", length=13)
    for index, item in enumerate(ordinary):
        _validate_ordinary_result(
            item,
            index=index,
            position_tolerance=raw["position_tolerance_m"],
            rotation_tolerance=math.radians(raw["rotation_tolerance_deg"]),
            frame_position_tolerance=raw["frame_position_tolerance_m"],
            frame_rotation_tolerance=math.radians(raw["frame_rotation_tolerance_deg"]),
        )
    _validate_case_geometry(ordinary)

    reports = _array(raw.get("ik_safety_episodes"), "ordinary safety", length=13)
    ordinary_safety: list[
        tuple[dict[str, int], dict[str, list[float]], dict[str, Any]]
    ] = []
    for index, report in enumerate(reports):
        ordinary_safety.append(
            _validate_safety(
                report,
                episode=index,
                applies=360,
                driver_mode="close" if index == 0 else "open",
                expected_open_samples=0 if index == 0 else 405,
            )
        )

    delayed = _object(raw.get("gripper_delayed_close_replay"), "delayed close")
    _require(set(delayed) == DELAYED_CLOSE_FIELDS, "delayed-close schema drift")
    _require(
        delayed.get("profile") == "eef_open115_then_close10_same_arm_pose_v2"
        and delayed.get("case")
        == "open 115 policy steps then close at the same arm pose"
        and delayed.get("passed") is True
        and delayed.get("terminated") is False
        and delayed.get("truncated") is False,
        "delayed-close identity/outcome drift",
    )
    for name, expected in (
        ("episode_index", 13),
        ("open_policy_steps", 115),
        ("close_policy_steps", 10),
        ("close_transition_substeps", 76),
        ("arm_abort_count", 0),
    ):
        _exact_int(delayed.get(name), expected, f"delayed {name}")
    delayed_counters, delayed_maxima, _ = _validate_safety(
        delayed.get("ik_safety"),
        episode=13,
        applies=1000,
        driver_mode="delayed_close",
        expected_open_samples=1035,
    )
    _require(
        delayed["arm_abort_count"]
        == sum(
            delayed_counters[name]
            for name in (
                "current_joint_limit_aborts",
                "invariant_aborts",
                "nonfinite_aborts",
            )
        ),
        "delayed-close abort binding",
    )

    close_headroom = _object(
        raw.get("gripper_close_velocity_headroom"), "close velocity headroom"
    )
    _require(
        set(close_headroom) == CLOSE_HEADROOM_FIELDS
        and close_headroom.get("profile") == "arm_velocity_max_over_limit_le_0p95_v1"
        and close_headroom.get("passed") is True
        and close_headroom.get("completion_gate_applied") is False,
        "v6 close-headroom identity drift",
    )
    _exact_float(close_headroom.get("threshold_ratio"), 0.95, "headroom threshold")
    _validate_headroom_entry(
        close_headroom.get("immediate_close_hold"),
        field="immediate-close headroom",
        episode=0,
        safety_maxima=ordinary_safety[0][1]["abs_joint_vel_rad_s"],
    )
    _validate_headroom_entry(
        close_headroom.get("delayed_close_replay"),
        field="delayed-close headroom",
        episode=13,
        safety_maxima=delayed_maxima["abs_joint_vel_rad_s"],
    )

    discriminator = _object(
        raw.get("concurrent_arm_gripper_discriminator"), "concurrent discriminator"
    )
    _require(
        set(discriminator) == DISCRIMINATOR_FIELDS,
        "concurrent discriminator schema drift",
    )
    _require(
        discriminator.get("profile")
        == "moving_eef_close_reopen_fresh_dls_every_apply_v1"
        and discriminator.get("passed") is True,
        "concurrent discriminator identity/outcome drift",
    )
    for name, expected in (
        ("episode_index", 14),
        ("transition_policy_steps", 10),
        ("transition_substeps", 76),
        ("expected_apply_calls", 168),
        ("expected_closed_endpoint_applies", 80),
    ):
        _exact_int(discriminator.get(name), expected, f"discriminator {name}")
    targets = _validate_discriminator_geometry(
        discriminator.get("distinct_policy_targets"), ordinary[0]
    )
    _, _, discriminator_telemetry = _validate_safety(
        discriminator.get("ik_safety"),
        episode=14,
        applies=168,
        driver_mode="roundtrip",
        expected_open_samples=99,
    )
    controller = _object(discriminator.get("controller_report"), "controller report")
    _require(
        set(controller) == CONTROLLER_REPORT_FIELDS,
        "controller report schema drift",
    )
    arm_slew = _object(controller.get("arm_slew_headroom"), "arm slew")
    _require(
        set(arm_slew) == ARM_SLEW_FIELDS
        and arm_slew.get("enabled") is True
        and arm_slew.get("profile")
        == "panda_nominal_target_slew_0p95_physical_limit_v1",
        "arm-slew identity drift",
    )
    _exact_float(arm_slew.get("ratio"), 0.95, "arm-slew ratio")
    _exact_vector(
        arm_slew.get("physical_max_delta_joint_pos_rad"),
        EXPECTED_PHYSICAL_MAX_DELTA,
        "arm-slew physical bounds",
    )
    _exact_vector(
        arm_slew.get("nominal_max_delta_joint_pos_rad"),
        EXPECTED_NOMINAL_MAX_DELTA,
        "arm-slew nominal bounds",
    )
    _require(
        controller.get("current_joint_velocity_recovery")
        == discriminator["ik_safety"].get("current_joint_velocity_recovery"),
        "controller/safety recovery mismatch",
    )
    concurrent = _object(
        controller.get("concurrent_arm_gripper"), "concurrent controller"
    )
    expected_concurrent = {
        "enabled": True,
        "profile": "fresh_dls_every_normal_apply_no_gripper_target_replay_v1",
        "fresh_dls_target_applies": 168,
        "normal_target_setter_applies": 168,
        "closed_endpoint_fresh_dls_target_applies": 80,
        "closed_endpoint_distinct_desired_pose_count": 10,
        "recovery_owned_target_applies": 0,
        "deferred_endpoint_transition_count": 0,
        "stored_target_replay_count": 0,
    }
    _require(
        _typed_equal(concurrent, expected_concurrent),
        "concurrent controller accounting drift",
    )
    _require(
        concurrent["fresh_dls_target_applies"]
        == discriminator["expected_apply_calls"]
        == discriminator["ik_safety"]["counters"]["apply_calls"]
        and concurrent["closed_endpoint_fresh_dls_target_applies"]
        == discriminator["expected_closed_endpoint_applies"]
        and concurrent["closed_endpoint_distinct_desired_pose_count"]
        == len(targets[11:]),
        "concurrent controller cross-binding",
    )
    interlock = _object(
        controller.get("gripper_close_arm_interlock"), "close interlock"
    )
    discriminator_driver = _object(
        discriminator["ik_safety"]["gripper_runtime_dynamic"].get("driver_target_slew"),
        "discriminator driver target slew",
    )
    _require(
        set(interlock) == INTERLOCK_FIELDS
        and interlock.get("enabled") is False
        and interlock.get("profile") == "concurrent_arm_no_close_interlock_v1"
        and interlock.get("anchor_valid") is False
        and interlock.get("endpoint_observed") is False,
        "close interlock identity drift",
    )
    for field in (
        "last_activation_apply_index",
        "last_anchor_joint_pos_rad",
        "last_anchor_little_endian_float32_sha256",
    ):
        _require(interlock.get(field) is None, f"close interlock {field}")
    for field, value in interlock.items():
        if field in {
            "enabled",
            "profile",
            "anchor_valid",
            "endpoint_observed",
            "last_activation_apply_index",
            "last_anchor_joint_pos_rad",
            "last_anchor_little_endian_float32_sha256",
            "observed_endpoint_change_count",
        }:
            continue
        if type(value) is int:
            _require(value == 0, f"close interlock {field} counter")
        elif type(value) is list:
            _require(value == [0.0] * 7, f"close interlock {field} maxima")
        else:
            raise VerificationError(f"unexpected close interlock field: {field}")
    _exact_int(
        interlock.get("observed_endpoint_change_count"),
        discriminator_driver.get("endpoint_change_count"),
        # This is cadence evidence, not disabled interlock control state.  The
        # corrected producer must report the exact two-transition round trip
        # on both the arm-side cursor and the validated finger driver.
        "close interlock/discriminator driver endpoint-change cadence",
    )
    _require(
        discriminator.get("open_endpoint_contact_mimic_impulse")
        == discriminator_telemetry,
        "discriminator/open-telemetry binding",
    )

    adversarial = _object(raw.get("ik_safety_adversarial"), "adversarial")
    _require(set(adversarial) == ADVERSARIAL_FIELDS, "adversarial schema drift")
    _require(
        adversarial.get("case") == "oversized absolute +x target for one policy step"
        and adversarial.get("passed") is True
        and adversarial.get("state_is_finite") is True
        and adversarial.get("eef_state_is_finite") is True
        and adversarial.get("joint_state_is_finite") is True
        and adversarial.get("joint_pos_within_captured_soft_limits") is True
        and adversarial.get("terminated") is False
        and adversarial.get("truncated") is False
        and adversarial.get("guard_error") == "",
        "adversarial identity/outcome drift",
    )
    _validate_adversarial_joint_state(adversarial.get("joint_state"))
    adversarial_counters, adversarial_maxima, _ = _validate_safety(
        adversarial.get("ik_safety"),
        episode=15,
        applies=8,
        driver_mode="open",
        expected_open_samples=9,
    )
    guard = _object(adversarial.get("guard_evidence"), "adversarial guard")
    _require(set(guard) == GUARD_EVIDENCE_FIELDS, "adversarial guard schema")
    for name, expected in (
        ("apply_calls", 8),
        ("slew_limit_events", adversarial_counters["slew_limit_events"]),
        (
            "abort_count",
            sum(
                adversarial_counters[item]
                for item in (
                    "current_joint_limit_aborts",
                    "invariant_aborts",
                    "nonfinite_aborts",
                )
            ),
        ),
        (
            "post_clamp_target_violations",
            adversarial_counters["post_clamp_target_violations"],
        ),
    ):
        _exact_int(guard.get(name), expected, f"adversarial guard {name}")
    _require(
        guard.get("applied_within_bounds") is True
        and adversarial_counters["slew_limit_events"] == 8,
        "adversarial guard outcome",
    )
    saturated = [
        index
        for index, (raw_delta, bound) in enumerate(
            zip(
                adversarial_maxima["raw_delta_joint_pos_rad"],
                EXPECTED_NOMINAL_MAX_DELTA,
                strict=True,
            )
        )
        if raw_delta > bound + 1e-6
    ]
    _require(saturated, "adversarial target never genuinely saturated")
    for index in saturated:
        _require(
            adversarial_maxima["applied_delta_joint_pos_rad"][index]
            >= EXPECTED_NOMINAL_MAX_DELTA[index] - 1e-6,
            f"adversarial joint {index} lacks saturated applied maximum",
        )

    all_safety_reports = [
        raw["raw_ik_safety_capture"],
        *reports,
        delayed["ik_safety"],
        discriminator["ik_safety"],
        adversarial["ik_safety"],
    ]
    _require(len(all_safety_reports) == 17, "total safety-report count drift")
    total_apply_calls = sum(
        item["counters"]["apply_calls"] for item in all_safety_reports
    )
    total_post_samples = sum(
        item["gripper_runtime_dynamic"]["post_policy_step_samples"]
        for item in all_safety_reports
    )
    total_open_samples = sum(
        item["gripper_runtime_dynamic"]["open_endpoint_contact_mimic_impulse"][
            "open_endpoint_samples"
        ]
        for item in all_safety_reports
    )
    _require(total_apply_calls == 5856, "total controller apply count drift")
    _require(total_post_samples == 732, "total post-policy sample count drift")
    _require(total_open_samples == 6003, "total open-endpoint sample count drift")
    return {
        "safety_report_count": 17,
        "total_controller_apply_calls": total_apply_calls,
        "total_post_policy_step_samples": total_post_samples,
        "total_open_endpoint_samples": total_open_samples,
        "ordinary_pose_cases_passed": 13,
        "ordinary_apply_calls": 13 * 360,
        "maximum_pose_position_error_m": max(
            item["position_error_m"] for item in ordinary
        ),
        "maximum_pose_rotation_error_deg": math.degrees(
            max(item["rotation_error_rad"] for item in ordinary)
        ),
        "delayed_close_apply_calls": 1000,
        "concurrent_apply_calls": 168,
        "concurrent_closed_fresh_dls_applies": 80,
        "concurrent_closed_distinct_desired_poses": 10,
        "open_endpoint_samples": 99,
        "maximum_follower_velocity_rad_s": max(
            discriminator_telemetry["max_abs_follower_joint_velocity_rad_s"]
        ),
        "coupled_impulse_failure_samples": 0,
        "adversarial_apply_calls": 8,
        "adversarial_slew_events": 8,
        "recovery_events": 0,
        "controller_aborts": 0,
        "nonfatal_log_findings": [
            "headless_glfw_no_display_warnings",
            "optional_ngx_context_errors",
            "missing_viewport_camera_mesh_asset_warning",
            "eight_active_actuators_plus_five_passive_mimic_dofs_warning",
        ],
    }


def validate_capture_artifacts(
    *,
    raw_result: Path,
    inline_attestation: Path,
    source_identity: Path,
    saved_job_script: Path,
    slurm_log: Path,
) -> dict[str, Any]:
    """Validate the exact immutable capture and return its semantic summary."""

    ready_marker = raw_result.with_name(raw_result.name + ".ready.json")
    raw_identity, raw_bytes = _identity(raw_result, "raw result", ARTIFACT_SPECS["raw"])
    ready_identity, ready_bytes = _identity(
        ready_marker, "ready marker", ARTIFACT_SPECS["ready"]
    )
    inline_identity, inline_bytes = _identity(
        inline_attestation,
        "inline host attestation",
        ARTIFACT_SPECS["inline_attestation"],
    )
    source_identity_entry, source_bytes = _identity(
        source_identity, "source identity", ARTIFACT_SPECS["source_identity"]
    )
    job_script_identity, _ = _identity(
        saved_job_script,
        "saved job script",
        ARTIFACT_SPECS["saved_job_script"],
    )
    slurm_log_identity, log_bytes = _identity(
        slurm_log, "Slurm log", ARTIFACT_SPECS["slurm_log"]
    )
    identities = {
        "raw": raw_identity,
        "ready": ready_identity,
        "inline_attestation": inline_identity,
        "source_identity": source_identity_entry,
        "saved_job_script": job_script_identity,
        "slurm_log": slurm_log_identity,
    }
    raw = _strict_json(raw_bytes, "raw result")
    ready = _strict_json(ready_bytes, "ready marker")
    inline = _strict_json(inline_bytes, "inline host attestation")
    summary = _semantic_summary(raw)
    _require(
        _typed_equal(
            ready,
            {
                "schema_version": 1,
                "stage": "simulation_app_close_pending",
                "raw_result": {
                    "path": RAW_PATH,
                    "size_bytes": ARTIFACT_SPECS["raw"]["size_bytes"],
                    "sha256": ARTIFACT_SPECS["raw"]["sha256"],
                    "mode": "0444",
                },
            },
        ),
        "ready marker content drift",
    )
    _require(
        inline.get("schema_version") == 1
        and inline.get("status") == "validated_after_zero_srun_exit"
        and inline.get("slurm_job_id") == str(JOB_ID)
        and inline.get("polaris_commit") == PRODUCER_COMMIT
        and inline.get("polaris_tree") == PRODUCER_TREE
        and inline.get("eef_controller_profile") == CONTROLLER_PROFILE
        and inline.get("container_sha256") == IMAGE_SHA256
        and _typed_equal(inline.get("checks"), INLINE_CHECKS),
        "inline host attestation content drift",
    )
    for name, expected_path, identity_name in (
        ("raw", RAW_PATH, "raw"),
        ("ready", READY_PATH, "ready"),
        ("source_identity", SOURCE_IDENTITY_PATH, "source_identity"),
    ):
        entry = inline.get(name)
        _require(
            type(entry) is dict and entry.get("path") == expected_path,
            f"inline {name} path",
        )
        _require(
            entry.get("size_bytes") == ARTIFACT_SPECS[identity_name]["size_bytes"]
            and entry.get("sha256") == ARTIFACT_SPECS[identity_name]["sha256"],
            f"inline {name} identity",
        )

    expected_source = "".join(
        f"{digest}  {relative}\n" for relative, digest in CAPTURE_SOURCE_SHA256.items()
    ).encode()
    _require(source_bytes == expected_source, "source identity content drift")
    try:
        log = log_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise VerificationError(f"Slurm log is not UTF-8: {error}") from error
    for forbidden in (
        "Traceback (most recent call last)",
        "RuntimeError:",
        "ValueError:",
        "CUDA out of memory",
        "Out Of Memory",
        "Segmentation fault",
        "core dumped",
    ):
        _require(forbidden not in log, f"Slurm log contains {forbidden!r}")
    _require(
        log.count("NVIDIA L40S") == 1 and "| 0   | NVIDIA L40S" in log,
        "Slurm log does not prove exactly one active L40S",
    )
    for required in (
        f"JOB_SCRIPT_SHA256={ARTIFACT_SPECS['saved_job_script']['sha256']}",
        f"POLARIS_COMMIT={PRODUCER_COMMIT}",
        f"POLARIS_TREE={PRODUCER_TREE}",
        f"POLARIS_PARENT={PRODUCER_PARENT}",
        "EEF pose smoke: 16/16 passed",
        "concurrent close/reopen: PASS fresh_dls=168 closed_fresh=80 "
        "distinct_closed=10 open_samples=99 coupled_failures=0",
        "adversarial: PASS apply_calls=8 slew_events=8",
        f"POLARIS_SMOKE_HOST_ATTESTATION_SHA256={ARTIFACT_SPECS['inline_attestation']['sha256']}",
        "SMOKE_EXIT_CODE=0",
        "GLFW initialization failed",
        "Failed to create NGX context",
        "Could not open asset @/lustre/fsw/portfolios/nvr/users/lzha/real2simeval/",
        "Total number of actuated joints not equal to number of joints available: 8 != 13",
    ):
        _require(required in log, f"Slurm log lacks {required!r}")
    return {"identities": identities, "validation_summary": summary}


def _git(repo: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise VerificationError(f"Git provenance failed: {error}") from error
    return completed.stdout.strip()


def _validate_standalone_detached_repository(repo: Path, field: str) -> Path:
    try:
        repo_status = os.lstat(repo)
        git_status = os.lstat(repo / ".git")
        resolved = repo.resolve(strict=True)
    except OSError as error:
        raise VerificationError(f"Cannot inspect {field} layout: {error}") from error
    _require(stat.S_ISDIR(repo_status.st_mode), f"{field} root is not a directory")
    _require(stat.S_ISDIR(git_status.st_mode), f"{field} .git is not a directory")
    _require(
        Path(_git(repo, "rev-parse", "--show-toplevel")) == resolved,
        f"{field} is not a top-level checkout",
    )
    expected_git_directory = (resolved / ".git").resolve(strict=True)
    _require(
        (resolved / _git(repo, "rev-parse", "--git-dir")).resolve(strict=True)
        == expected_git_directory,
        f"{field} git-dir is not in-root",
    )
    _require(
        (resolved / _git(repo, "rev-parse", "--git-common-dir")).resolve(strict=True)
        == expected_git_directory,
        f"{field} git common-dir is not in-root",
    )
    _require(
        _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD",
        f"{field} is not detached",
    )
    return resolved


def validate_producer_source_identity(repo_root: Path) -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative, expected in PRODUCER_SOURCE_SHA256.items():
        data, _ = _read_regular_file(
            repo_root / relative, f"producer source {relative}"
        )
        digest = hashlib.sha256(data).hexdigest()
        _require(digest == expected, f"producer source digest drift: {relative}")
        actual[relative] = digest
    return actual


def _scheduler_evidence() -> dict[str, Any]:
    sacct = shutil.which("sacct")
    _require(sacct is not None, "sacct is unavailable")
    try:
        completed = subprocess.run(
            [
                sacct,
                "-j",
                str(JOB_ID),
                "--format=JobIDRaw,JobName%30,Account,Partition,State,ExitCode,"
                "ElapsedRaw,Start,End,NodeList,AllocTRES%100,ReqTRES%100",
                "-n",
                "-P",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise VerificationError(f"Slurm accounting query failed: {error}") from error
    rows = [line.split("|") for line in completed.stdout.splitlines() if line.strip()]
    _require(
        len(rows) == 6 and all(len(row) == 12 for row in rows),
        "unexpected Slurm accounting rows",
    )
    names = {
        "1098975": "allocation",
        "1098975.batch": "batch",
        "1098975.extern": "extern",
        "1098975.0": "srun",
        "1098975.1": "monitoring_1",
        "1098975.2": "monitoring_2",
    }
    actual: dict[str, dict[str, Any]] = {}
    for row in rows:
        (
            job_id,
            job_name,
            account,
            partition,
            state,
            exit_code,
            elapsed,
            start,
            end,
            node,
            allocated_tres,
            requested_tres,
        ) = row
        _require(
            job_id in names and names[job_id] not in actual, "unexpected Slurm job step"
        )
        actual[names[job_id]] = {
            "job_id": job_id,
            "job_name": job_name,
            "account": account,
            "partition": partition,
            "state": state,
            "exit_code": exit_code,
            "elapsed_seconds": int(elapsed),
            "start": start,
            "end": end,
            "node": node,
            "allocated_tres": allocated_tres,
            "requested_tres": requested_tres,
        }
    _require(actual == EXPECTED_SCHEDULER_EVIDENCE, "Slurm terminal lifecycle drift")
    return actual


def _publish_bytes_nonoverwriting(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_or_validate_bytes(path: Path, data: bytes, field: str) -> None:
    if not path.exists():
        _publish_bytes_nonoverwriting(path, data)
    actual, status = _read_regular_file(path, field)
    _require(stat.S_IMODE(status.st_mode) == 0o444, f"{field} mode drift")
    _require(actual == data, f"{field} content drift")


def _seal_provenance(args: argparse.Namespace) -> None:
    for path, expected, field in (
        (
            Path(SEALED_SOURCE_IDENTITY_PATH),
            SEALED_SOURCE_IDENTITY_PATH,
            "sealed source identity",
        ),
        (
            Path(SEALED_JOB_SCRIPT_PATH),
            SEALED_JOB_SCRIPT_PATH,
            "sealed saved job script",
        ),
        (Path(SEALED_SLURM_LOG_PATH), SEALED_SLURM_LOG_PATH, "sealed Slurm log"),
        (Path(SEALED_SACCT_PATH), SEALED_SACCT_PATH, "sealed sacct snapshot"),
    ):
        _exact_cli_path(path, expected, field)
    _, source_bytes = _identity(
        args.source_identity,
        "source identity",
        ARTIFACT_SPECS["source_identity"],
    )
    _, wrapper_bytes = _identity(
        args.saved_job_script,
        "saved job script",
        ARTIFACT_SPECS["saved_job_script"],
    )
    _, log_bytes = _identity(args.slurm_log, "Slurm log", ARTIFACT_SPECS["slurm_log"])
    _require(
        _scheduler_evidence() == EXPECTED_SCHEDULER_EVIDENCE,
        "scheduler evidence drift before sealing",
    )
    for path, data, field in (
        (Path(SEALED_SOURCE_IDENTITY_PATH), source_bytes, "sealed source identity"),
        (Path(SEALED_JOB_SCRIPT_PATH), wrapper_bytes, "sealed saved job script"),
        (Path(SEALED_SLURM_LOG_PATH), log_bytes, "sealed Slurm log"),
        (Path(SEALED_SACCT_PATH), SACCT_SNAPSHOT_BYTES, "sealed sacct snapshot"),
    ):
        _publish_or_validate_bytes(path, data, field)


def _validate_sealed_provenance() -> dict[str, dict[str, Any]]:
    paths = {
        "source_identity": Path(SEALED_SOURCE_IDENTITY_PATH),
        "saved_job_script": Path(SEALED_JOB_SCRIPT_PATH),
        "slurm_log": Path(SEALED_SLURM_LOG_PATH),
        "sacct": Path(SEALED_SACCT_PATH),
    }
    identities: dict[str, dict[str, Any]] = {}
    sacct_bytes: bytes | None = None
    for name, path in paths.items():
        _require_l401_canonical_path(path, f"sealed {name}")
        identity, data = _identity(
            path, f"sealed {name}", SEALED_PROVENANCE_SPECS[name]
        )
        identities[name] = identity
        if name == "sacct":
            sacct_bytes = data
    _require(sacct_bytes is not None, "sealed sacct snapshot missing")
    _require(
        _typed_equal(
            _strict_json(sacct_bytes, "sealed sacct snapshot"),
            SACCT_SNAPSHOT_PAYLOAD,
        ),
        "sealed sacct snapshot content drift",
    )
    return identities


def _hash_external(
    path: Path,
    field: str,
    expected_sha256: str,
    *,
    required_mode: int,
    expected_metadata: Mapping[str, int] | None = None,
    must_predate_ns: int | None = None,
) -> dict[str, Any]:
    try:
        before = os.lstat(path)
    except OSError as error:
        raise VerificationError(f"Cannot inspect {field}: {error}") from error
    _require(stat.S_ISREG(before.st_mode), f"{field} is not a regular file")
    _require(before.st_nlink == 1, f"{field} must have exactly one hard link")
    _require(
        stat.S_IMODE(before.st_mode) == required_mode,
        f"{field} mode must be {required_mode:04o}",
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise VerificationError(f"Cannot open {field}: {error}") from error
    hasher = hashlib.sha256()
    size_bytes = 0
    try:
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(opened.st_mode)
            and (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
            f"{field} changed during secure open",
        )
        metadata_fields = (
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_nlink",
            "st_mode",
        )
        _require(
            all(
                getattr(opened, name) == getattr(before, name)
                for name in metadata_fields
            ),
            f"{field} metadata changed during secure open",
        )
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
            size_bytes += len(chunk)
        after = os.fstat(descriptor)
        _require(
            all(
                getattr(after, name) == getattr(opened, name)
                for name in metadata_fields
            ),
            f"{field} changed while being hashed",
        )
        try:
            linked = os.lstat(path)
        except OSError as error:
            raise VerificationError(f"Cannot re-inspect {field}: {error}") from error
        path_fields = (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_nlink",
            "st_mode",
        )
        _require(
            stat.S_ISREG(linked.st_mode)
            and linked.st_nlink == 1
            and all(
                getattr(linked, name) == getattr(after, name) for name in path_fields
            ),
            f"{field} path changed while being hashed",
        )
    finally:
        os.close(descriptor)
    _require(size_bytes == after.st_size, f"{field} size changed while being hashed")
    if expected_metadata is not None:
        actual_metadata = {
            "size_bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns,
            "ctime_ns": after.st_ctime_ns,
        }
        _require(actual_metadata == expected_metadata, f"{field} metadata drift")
    if must_predate_ns is not None:
        _require(
            max(after.st_mtime_ns, after.st_ctime_ns) <= must_predate_ns,
            f"{field} does not predate the smoke srun",
        )
    digest = hasher.hexdigest()
    _require(digest == expected_sha256, f"{field} digest drift")
    return {
        "path": str(path),
        "size_bytes": size_bytes,
        "sha256": digest,
        "mode": f"{stat.S_IMODE(after.st_mode):04o}",
        "nlink": 1,
        "mtime_ns": after.st_mtime_ns,
        "ctime_ns": after.st_ctime_ns,
    }


def _require_l401_canonical_path(actual: Path, field: str) -> None:
    try:
        relative = actual.relative_to(L401_LITERAL_USER_ROOT)
    except ValueError as error:
        raise VerificationError(
            f"{field} is outside the literal l401 user root"
        ) from error
    expected_canonical = L401_CANONICAL_USER_ROOT / relative
    _require(
        actual.resolve(strict=False) == expected_canonical,
        f"{field} canonical path drift",
    )


def _exact_cli_path(actual: Path, expected: str, field: str) -> None:
    _require(str(actual) == expected, f"{field} path drift")
    _require_l401_canonical_path(actual, field)


def _build_expected(args: argparse.Namespace) -> dict[str, Any]:
    for actual, expected, field in (
        (args.raw_result, RAW_PATH, "raw result"),
        (args.inline_attestation, INLINE_ATTESTATION_PATH, "inline attestation"),
        (args.source_identity, SOURCE_IDENTITY_PATH, "source identity"),
        (args.saved_job_script, SAVED_JOB_SCRIPT_PATH, "saved job script"),
        (args.slurm_log, SLURM_LOG_PATH, "Slurm log"),
        (args.producer_repo, PRODUCER_REPO_PATH, "producer repo"),
        (args.container_image, CONTAINER_IMAGE_PATH, "container image"),
        (args.scene_usda, SCENE_PATH, "scene"),
        (args.attestation, PROMOTION_ATTESTATION_PATH, "promotion attestation"),
    ):
        _exact_cli_path(actual, expected, field)
    _require_l401_canonical_path(args.evidence_repo, "evidence repo")
    capture = validate_capture_artifacts(
        raw_result=args.raw_result,
        inline_attestation=args.inline_attestation,
        source_identity=args.source_identity,
        saved_job_script=args.saved_job_script,
        slurm_log=args.slurm_log,
    )
    producer_resolved = _validate_standalone_detached_repository(
        args.producer_repo, "producer repo"
    )
    _require(
        _git(args.producer_repo, "rev-parse", "HEAD") == PRODUCER_COMMIT,
        "producer commit",
    )
    _require(
        _git(args.producer_repo, "rev-parse", "HEAD^{tree}") == PRODUCER_TREE,
        "producer tree",
    )
    _require(
        _git(args.producer_repo, "rev-parse", "HEAD^") == PRODUCER_PARENT,
        "producer parent",
    )
    _require(
        _git(args.producer_repo, "status", "--porcelain") == "", "producer repo dirty"
    )
    producer_sources = validate_producer_source_identity(args.producer_repo)

    evidence_resolved = _validate_standalone_detached_repository(
        args.evidence_repo, "evidence repo"
    )
    _require(evidence_resolved != producer_resolved, "producer/evidence repo collision")
    evidence_commit = _git(args.evidence_repo, "rev-parse", "HEAD")
    evidence_tree = _git(args.evidence_repo, "rev-parse", "HEAD^{tree}")
    evidence_parent = _git(args.evidence_repo, "rev-parse", "HEAD^")
    _require(
        evidence_commit == args.expected_evidence_commit, "evidence commit mismatch"
    )
    _require(evidence_tree == args.expected_evidence_tree, "evidence tree mismatch")
    _require(
        evidence_parent == PRODUCER_COMMIT,
        "evidence commit is not direct producer descendant",
    )
    _require(
        _git(args.evidence_repo, "status", "--porcelain") == "", "evidence repo dirty"
    )
    changed_paths = set(
        _git(
            args.evidence_repo,
            "diff",
            "--name-only",
            f"{PRODUCER_COMMIT}..{evidence_commit}",
        ).splitlines()
    )
    _require(
        changed_paths == EVIDENCE_COMMIT_CHANGED_PATHS,
        "evidence commit changed-path allowlist drift",
    )
    finalizer_path = Path(__file__).resolve()
    _require(
        finalizer_path
        == (args.evidence_repo / "scripts" / Path(__file__).name).resolve(),
        "finalizer is outside evidence repo",
    )
    finalizer = _hash_external(
        finalizer_path,
        "finalizer",
        args.expected_finalizer_sha256,
        required_mode=0o644,
    )
    image = _hash_external(
        args.container_image,
        "container image",
        IMAGE_SHA256,
        required_mode=0o644,
        expected_metadata=IMAGE_METADATA,
        must_predate_ns=SRUN_START_EPOCH_NS,
    )
    scene = _hash_external(
        args.scene_usda,
        "FoodBussing scene",
        SCENE_SHA256,
        required_mode=0o640,
        expected_metadata=SCENE_METADATA,
        must_predate_ns=SRUN_START_EPOCH_NS,
    )
    sealed_provenance = _validate_sealed_provenance()
    return {
        "schema_version": 1,
        "profile": PROMOTION_PROFILE,
        "status": PROMOTION_STATUS,
        "finalized": True,
        "passed": True,
        "scope": PROMOTION_SCOPE,
        "producer": {
            "polaris_repo": str(args.producer_repo),
            "polaris_commit": PRODUCER_COMMIT,
            "polaris_tree": PRODUCER_TREE,
            "polaris_parent": PRODUCER_PARENT,
            "controller_profile": CONTROLLER_PROFILE,
            "ik_safety_profile": IK_SAFETY_PROFILE,
            "source_sha256": producer_sources,
        },
        "runtime": {
            "scheduler": deepcopy(EXPECTED_SCHEDULER_EVIDENCE),
            "srun_start_epoch_ns": SRUN_START_EPOCH_NS,
            "container_image": image,
            "food_bussing_scene": scene,
        },
        "artifacts": capture["identities"],
        "sealed_provenance": sealed_provenance,
        "validation_summary": capture["validation_summary"],
        "coverage_limits": {
            "checkpoint_loaded": False,
            "policy_serving_validated": False,
            "camera_image_contract_validated": False,
            "normalization_validated": False,
            "task_success_metric_validated": False,
            "scene_digest_logged_by_smoke_job": False,
            "scene_post_job_digest_and_pre_srun_metadata_validated": True,
            "live_recovery_event_observed": False,
            "live_follower_threshold_crossing_observed": False,
            "next_required_gate": NEXT_REQUIRED_GATE,
        },
        "reviewer": {
            "evidence_repo": str(args.evidence_repo),
            "evidence_commit": evidence_commit,
            "evidence_tree": evidence_tree,
            "evidence_parent": evidence_parent,
            "changed_paths": sorted(changed_paths),
            "finalizer": finalizer,
        },
    }


def _serialized(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _validate_exact_serialized_payload(
    data: bytes, expected: Mapping[str, Any], field: str
) -> dict[str, Any]:
    actual = _strict_json(data, field)
    _require(data == _serialized(expected), f"{field} byte content drift")
    return actual


def _publish_nonoverwriting(path: Path, payload: Mapping[str, Any]) -> None:
    _publish_bytes_nonoverwriting(path, _serialized(payload))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("finalize", "verify"))
    parser.add_argument("--attestation", required=True, type=Path)
    parser.add_argument("--raw-result", required=True, type=Path)
    parser.add_argument("--inline-attestation", required=True, type=Path)
    parser.add_argument("--source-identity", required=True, type=Path)
    parser.add_argument("--saved-job-script", required=True, type=Path)
    parser.add_argument("--slurm-log", required=True, type=Path)
    parser.add_argument("--producer-repo", required=True, type=Path)
    parser.add_argument("--evidence-repo", required=True, type=Path)
    parser.add_argument("--expected-evidence-commit", required=True)
    parser.add_argument("--expected-evidence-tree", required=True)
    parser.add_argument("--expected-finalizer-sha256", required=True)
    parser.add_argument("--container-image", required=True, type=Path)
    parser.add_argument("--scene-usda", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.mode == "finalize":
            _seal_provenance(args)
        expected = _build_expected(args)
        if args.mode == "finalize":
            _publish_nonoverwriting(args.attestation, expected)
        data, status = _read_regular_file(args.attestation, "promotion attestation")
        _require(stat.S_IMODE(status.st_mode) == 0o444, "promotion attestation mode")
        _validate_exact_serialized_payload(data, expected, "promotion attestation")
    except (OSError, VerificationError, ValueError) as error:
        print(
            f"V6_SMOKE_PROMOTION_ATTESTATION_FAIL={error}", file=sys.stderr, flush=True
        )
        return 1
    print(f"V6_SMOKE_PROMOTION_ATTESTATION_PASS={args.attestation}", flush=True)
    print(f"V6_SMOKE_PROMOTION_ATTESTATION_SIZE_BYTES={len(data)}", flush=True)
    print(
        "V6_SMOKE_PROMOTION_ATTESTATION_SHA256=" + hashlib.sha256(data).hexdigest(),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
