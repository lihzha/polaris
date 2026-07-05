"""Fail-closed serving and execution contract for official ``pi05_droid``.

This profile preserves the released checkpoint, global DROID statistics,
OpenPI FLOW inference, state, cameras, and response chunk.  It changes only
the simulator-side actuator adaptation from the historical (incorrect) direct
rad/s path to the position command actually used by the official DROID stack.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any

from polaris.pi05_droid_jointvelocity_contract import (
    NATIVE_GRIPPER_DRIVE_PROFILE,
    NATIVE_GRIPPER_EFFORT_LIMIT,
    NATIVE_GRIPPER_MEASURED_VELOCITY_TOLERANCE,
    NATIVE_GRIPPER_PRECONDITION_POSITION_TOLERANCE,
    NATIVE_GRIPPER_PRECONDITION_STEPS,
    NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S,
    PANDA_ARM_EFFORT_LIMITS,
    PANDA_ARM_JOINT_NAMES,
    PANDA_ARM_VELOCITY_LIMITS,
    PI05_DROID_CHECKPOINT_BYTES,
    PI05_DROID_CHECKPOINT_MANIFEST_SHA256,
    PI05_DROID_CHECKPOINT_OBJECT_COUNT,
    PI05_DROID_CHECKPOINT_URI,
    PI05_DROID_CONTRACT_FILENAME,
    PI05_DROID_CONTRACT_METADATA_KEY,
    PI05_DROID_GRIPPER_OBSERVATION_BOUND_TOLERANCE,
    PI05_DROID_GRIPPER_OBSERVATION_CONTRACT,
    PI05_DROID_ISAACLAB_REVISION,
    PI05_DROID_ISAACLAB_SOURCE_SHA256,
    PI05_DROID_ISAACLAB_VERSION,
    PI05_DROID_NORM_STATS_SHA256,
    PI05_DROID_OPENPI_INFERENCE_COMPATIBILITY_COMMIT,
    attest_imported_openpi_modules,
    expected_pi05_droid_jointvelocity_contract,
    reference_openpi_runtime_attestation,
    validate_openpi_runtime_attestation,
    verify_openpi_git_checkout,
    verify_pi05_droid_checkpoint,
    verify_profile_manifest,
    verify_profile_source_files,
)
from polaris.pi05_droid_position_adapter import (
    OFFICIAL_DROID_COMMIT,
    OFFICIAL_DROID_CONTROL_SOURCES,
    PI05_DROID_ACTION_DIM,
    PI05_DROID_ARM_COMMAND_SCALE_RAD,
    PI05_DROID_EXECUTION_HORIZON,
    PI05_DROID_PHYSICS_FREQUENCY_HZ,
    PI05_DROID_PHYSICS_SUBSTEPS,
    PI05_DROID_POLICY_FREQUENCY_HZ,
    PI05_DROID_POSITION_ADAPTER_PROFILE,
    PI05_DROID_RESPONSE_HORIZON,
    canonical_json_bytes,
    expected_position_limit_contract,
    official_droid_source_contract,
)


PI05_DROID_POSITION_DRIVE_STIFFNESS = 400.0
PI05_DROID_POSITION_DRIVE_DAMPING = 80.0
PI05_DROID_POSITION_CONTRACT_MARKER = "POLARIS_PI05_DROID_POSITION_CONTRACT="
PI05_DROID_POSITION_TRANSFORM_RUNTIME_CONTRACT = {
    "asset_id": "droid",
    "use_quantile_norm": True,
    "repack_inputs": [],
    "repack_outputs": [],
    "data_inputs": ["openpi.policies.droid_policy.DroidInputs"],
    "data_outputs": ["openpi.policies.droid_policy.DroidOutputs"],
    "model_inputs": [
        "openpi.transforms.InjectDefaultPrompt",
        "openpi.transforms.ResizeImages",
        "openpi.transforms.TokenizePrompt",
        "openpi.transforms.PadStatesAndActions",
    ],
    "model_outputs": [],
    "sequence_types": {
        "repack_inputs": "tuple",
        "repack_outputs": "tuple",
        "data_inputs": "list",
        "data_outputs": "list",
        "model_inputs": "list",
        "model_outputs": "tuple",
    },
    "droid_input_model_type": "pi05",
    "resize": [224, 224],
    "tokenizer": "openpi.models.tokenizer.PaligemmaTokenizer",
    "discrete_state_input": True,
    "model_action_dim": 32,
    "forbidden_transforms_absent": [
        "openpi.transforms.DeltaActions",
        "openpi.transforms.AbsoluteActions",
    ],
    "output_projection": "DroidOutputs_leading8",
}
PI05_DROID_POSITION_MODEL_EVAL_CONTRACT = {
    "schema_version": 1,
    "profile": PI05_DROID_POSITION_ADAPTER_PROFILE,
    "checkpoint": {
        "uri": PI05_DROID_CHECKPOINT_URI,
        "content_manifest_sha256": PI05_DROID_CHECKPOINT_MANIFEST_SHA256,
    },
    "normalization": {
        "asset_id": "droid",
        "scope": "checkpoint_global_droid",
        "path": "assets/droid/norm_stats.json",
        "sha256": PI05_DROID_NORM_STATS_SHA256,
        "category_override": "forbidden",
        "rejected_category_substitutions": ["single_arm", "single-arm", "single arm"],
    },
    "policy_input": {
        "state": "7_panda_joint_positions_radians_plus_closed_positive_gripper",
        "state_width": 8,
        "request_state_dtype": "float32",
        "native_images": [
            {"source": "external_cam", "shape": [720, 1280, 3], "dtype": "uint8"},
            {"source": "wrist_cam", "shape": [720, 1280, 3], "dtype": "uint8"},
        ],
        "image_order": ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb_masked"],
        "resize": "openpi_image_tools_resize_with_pad_224_v1",
        "wrist_rotation_degrees": 0,
    },
    "inference": {
        "objective": "flow",
        "sampler": "flow_euler_t1_to_t0_num_steps10_rng_key0_v1",
        "response_shape": [15, 8],
        "execute_first": 8,
    },
    "policy_output": {
        "server_semantics": "normalized_droid_joint_velocity_command",
        "server_action_transform": "none_before_DroidOutputs_leading8_projection",
        "client_processing": [
            "binarize_absolute_closed_positive_gripper",
            "clip_all_dimensions_minus1_plus1",
            "read_fresh_live_measured_panda_q",
            "q_target_equals_measured_q_plus_0p2_times_clipped_arm_command",
        ],
        "simulator_command": "absolute_joint_position_target",
        "target_hold_physics_substeps": 8,
        "simulator_target_guard": expected_position_limit_contract(),
    },
}


def contract_sha256(contract: dict[str, Any]) -> str:
    payload = copy.deepcopy(contract)
    payload.pop("contract_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def expected_pi05_droid_position_contract(
    openpi_runtime_attestation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the only accepted official-DROID position-adapter contract."""

    if openpi_runtime_attestation is None:
        openpi_runtime_attestation = reference_openpi_runtime_attestation()
    attestation = validate_openpi_runtime_attestation(openpi_runtime_attestation)
    inherited = expected_pi05_droid_jointvelocity_contract(attestation)

    contract = copy.deepcopy(inherited)
    contract.pop("contract_sha256")
    contract["profile"] = PI05_DROID_POSITION_ADAPTER_PROFILE
    contract["official_droid"] = official_droid_source_contract()
    contract["policy_input"]["native_images"] = [
        {"source": "external_cam", "shape": [720, 1280, 3], "dtype": "uint8"},
        {"source": "wrist_cam", "shape": [720, 1280, 3], "dtype": "uint8"},
    ]
    contract["policy_output"] = {
        "response_shape": [PI05_DROID_RESPONSE_HORIZON, PI05_DROID_ACTION_DIM],
        "response_dtype": "float64_after_checkpoint_unnormalize",
        "execute_first": PI05_DROID_EXECUTION_HORIZON,
        "arm": "normalized_droid_joint_velocity_command_not_direct_rad_per_second",
        "gripper": "absolute_closed_positive_binarize_gt_0p5_else_open",
        "processing_order": [
            "binarize_gripper_gt_0p5",
            "clip_all_8_dimensions_minus1_plus1",
            "read_fresh_measured_panda_joint1_through_7",
            "multiply_clipped_arm_command_by_0p2_radians",
            "add_delta_to_fresh_measurement",
            "apply_absolute_joint_position_target_for_8_physics_substeps",
        ],
    }
    inherited_gripper = inherited["control"]["gripper_drive"]
    contract["control"] = {
        "mode": "fresh_jointdelta_to_absolute_joint_position",
        "isaaclab_version": PI05_DROID_ISAACLAB_VERSION,
        "isaaclab_revision": PI05_DROID_ISAACLAB_REVISION,
        "isaaclab_source_sha256": copy.deepcopy(PI05_DROID_ISAACLAB_SOURCE_SHA256),
        "policy_frequency_hz": PI05_DROID_POLICY_FREQUENCY_HZ,
        "physics_frequency_hz": PI05_DROID_PHYSICS_FREQUENCY_HZ,
        "decimation": PI05_DROID_PHYSICS_SUBSTEPS,
        "joint_names": list(PANDA_ARM_JOINT_NAMES),
        "command_anchor": "fresh_measured_joint_position_at_each_executed_step",
        "chunk_query_anchor": False,
        "normalized_command_clip": [-1.0, 1.0],
        "joint_delta_scale_rad": PI05_DROID_ARM_COMMAND_SCALE_RAD,
        "target_formula": "q_target_t_equals_measured_q_t_plus_0p2_times_clipped_command_t",
        "target_clipping": None,
        "action_cfg": "polaris_AuditedDroidDeltaJointPositionActionCfg",
        "action_cfg_base": "isaaclab_JointPositionActionCfg",
        "scale": 1.0,
        "offset": 0.0,
        "use_default_offset": False,
        "action_manager_clip": None,
        "target_mode": "absolute_joint_position",
        "target_hold": {
            "setter": "Articulation.set_joint_position_target",
            "apply_calls_per_policy_step": PI05_DROID_PHYSICS_SUBSTEPS,
            "same_target_each_substep": True,
        },
        "position_drive": {
            "actuator": "implicit_physx_pd",
            "semantic_role": (
                "existing_polaris_NVIDIA_DROID_simulator_analogue_of_"
                "hardware_cartesian_impedance_update_desired_joint_positions"
            ),
            "claims_exact_hardware_controller_gains": False,
            "position_stiffness": PI05_DROID_POSITION_DRIVE_STIFFNESS,
            "velocity_damping": PI05_DROID_POSITION_DRIVE_DAMPING,
            "effort_limit_sim": list(PANDA_ARM_EFFORT_LIMITS),
            "velocity_limit_sim": list(PANDA_ARM_VELOCITY_LIMITS),
        },
        "position_limits": expected_position_limit_contract(),
        "gripper_drive": copy.deepcopy(inherited_gripper),
    }
    contract["contract_sha256"] = contract_sha256(contract)
    return contract


def expected_pi05_droid_position_server_metadata(
    openpi_runtime_attestation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        PI05_DROID_CONTRACT_METADATA_KEY: expected_pi05_droid_position_contract(
            openpi_runtime_attestation
        )
    }


def validate_pi05_droid_position_server_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        PI05_DROID_CONTRACT_METADATA_KEY
    }:
        raise ValueError("pi0.5-DROID position handshake schema mismatch")
    contract = value[PI05_DROID_CONTRACT_METADATA_KEY]
    if not isinstance(contract, dict):
        raise ValueError("pi0.5-DROID position contract must be an object")
    if contract.get("contract_sha256") != contract_sha256(contract):
        raise ValueError("pi0.5-DROID position contract SHA-256 is invalid")
    runtime = validate_openpi_runtime_attestation(
        contract.get("openpi", {}).get("runtime_attestation")
    )
    expected = expected_pi05_droid_position_server_metadata(runtime)
    if value != expected:
        raise ValueError("pi0.5-DROID position-adapter contract mismatch")
    return copy.deepcopy(contract)


def serving_contract_bytes(metadata: dict[str, Any]) -> bytes:
    validate_pi05_droid_position_server_metadata(metadata)
    return canonical_json_bytes(metadata) + b"\n"


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_immutable_position_serving_contract(
    path: Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    path = Path(path)
    if path.name != PI05_DROID_CONTRACT_FILENAME:
        raise ValueError(f"contract must be named {PI05_DROID_CONTRACT_FILENAME}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = serving_contract_bytes(metadata)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return validate_persisted_position_serving_contract(path, metadata)


def validate_persisted_position_serving_contract(
    path: Path, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    path = Path(path)
    if path.name != PI05_DROID_CONTRACT_FILENAME or path.is_symlink():
        raise ValueError("persisted position contract path is invalid")
    metadata_stat = path.stat()
    if (
        not stat.S_ISREG(metadata_stat.st_mode)
        or metadata_stat.st_nlink != 1
        or stat.S_IMODE(metadata_stat.st_mode) != 0o444
    ):
        raise ValueError("persisted position contract must be one mode-0444 file")
    payload = path.read_bytes()
    try:
        persisted = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("persisted position contract is not strict JSON") from error
    contract = validate_pi05_droid_position_server_metadata(persisted)
    if payload != serving_contract_bytes(persisted):
        raise ValueError("persisted position contract is not canonical JSON")
    if metadata is not None and payload != serving_contract_bytes(metadata):
        raise ValueError("persisted position contract differs from live handshake")
    return {
        "path": str(path.resolve()),
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "contract_sha256": contract["contract_sha256"],
        "mode": "0444",
        "nlink": 1,
    }


def verify_official_droid_git_checkout(droid_dir: Path) -> dict[str, Any]:
    """Verify the exact official DROID revision and three control blobs."""

    requested = Path(droid_dir)
    if requested.is_symlink():
        raise ValueError("official DROID checkout path must not be a symlink")
    root = requested.resolve()
    if not root.is_dir() or not (root / ".git").exists():
        raise ValueError("official DROID source must be a Git checkout")

    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError("cannot inspect official DROID checkout") from error

    if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("official DROID path must name the checkout root")
    if git("rev-parse", "HEAD") != OFFICIAL_DROID_COMMIT:
        raise ValueError("official DROID revision mismatch")
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("official DROID checkout must be completely clean")
    sources: dict[str, dict[str, str]] = {}
    for relative_path, expected in OFFICIAL_DROID_CONTROL_SOURCES.items():
        path = root / relative_path
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing official DROID control source: {relative_path}")
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        blob = git("rev-parse", f"HEAD:{relative_path}")
        if sha256 != expected["sha256"] or blob != expected["git_blob_sha1"]:
            raise ValueError(f"official DROID source identity mismatch: {relative_path}")
        sources[relative_path] = {"sha256": sha256, "git_blob_sha1": blob}
    return {"root": str(root), "revision": OFFICIAL_DROID_COMMIT, "sources": sources}


__all__ = [
    "NATIVE_GRIPPER_DRIVE_PROFILE",
    "NATIVE_GRIPPER_EFFORT_LIMIT",
    "NATIVE_GRIPPER_MEASURED_VELOCITY_TOLERANCE",
    "NATIVE_GRIPPER_PRECONDITION_POSITION_TOLERANCE",
    "NATIVE_GRIPPER_PRECONDITION_STEPS",
    "NATIVE_GRIPPER_VELOCITY_LIMIT_RAD_S",
    "PANDA_ARM_EFFORT_LIMITS",
    "PANDA_ARM_JOINT_NAMES",
    "PANDA_ARM_VELOCITY_LIMITS",
    "PI05_DROID_CHECKPOINT_BYTES",
    "PI05_DROID_CHECKPOINT_MANIFEST_SHA256",
    "PI05_DROID_CHECKPOINT_OBJECT_COUNT",
    "PI05_DROID_CHECKPOINT_URI",
    "PI05_DROID_CONTRACT_FILENAME",
    "PI05_DROID_CONTRACT_METADATA_KEY",
    "PI05_DROID_GRIPPER_OBSERVATION_BOUND_TOLERANCE",
    "PI05_DROID_GRIPPER_OBSERVATION_CONTRACT",
    "PI05_DROID_NORM_STATS_SHA256",
    "PI05_DROID_OPENPI_INFERENCE_COMPATIBILITY_COMMIT",
    "PI05_DROID_POSITION_ADAPTER_PROFILE",
    "PI05_DROID_POSITION_MODEL_EVAL_CONTRACT",
    "PI05_DROID_POSITION_TRANSFORM_RUNTIME_CONTRACT",
    "attest_imported_openpi_modules",
    "expected_pi05_droid_position_contract",
    "expected_pi05_droid_position_server_metadata",
    "publish_immutable_position_serving_contract",
    "reference_openpi_runtime_attestation",
    "serving_contract_bytes",
    "validate_persisted_position_serving_contract",
    "validate_pi05_droid_position_server_metadata",
    "verify_official_droid_git_checkout",
    "verify_openpi_git_checkout",
    "verify_pi05_droid_checkpoint",
    "verify_profile_manifest",
    "verify_profile_source_files",
]
