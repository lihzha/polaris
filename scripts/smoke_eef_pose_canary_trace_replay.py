#!/usr/bin/env python3
"""Replay the two exact PolaRiS canary prefixes without a policy server.

Gate 0 is a baseline identity test, not a controller candidate.  It installs
the production ``EgoLapEefPoseActionCfg`` and EEF gripper runtime, resets
FoodBussing IC0, then feeds byte-pinned absolute actions from job 1098292 or
1098294.  Passing means reproducing the source failure at the exact policy
step, physics substep, joint, and evidence digest while preserving the full
64-substep arm causal ring and an all-six-joint gripper tail.

Heavy Isaac imports are intentionally confined to the live runner.  Fixture,
capture, and tail validators remain host-testable with the standard library.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import sys
import traceback
from typing import Any
import zlib


ENVIRONMENT = "DROID-FoodBussing"
PROFILE = "polaris_eef_canary_gate0_model_free_replay_v1"
FIXTURE_PROFILE = "polaris_eef_canary_gate0_trace_replay_v1"
GRIPPER_TAIL_PROFILE = "polaris_eef_all_six_gripper_substep_tail_v1"
BASE_COMMIT = "712240cbb215ecb31830cdb2ee65e91704160372"
INITIAL_CONDITION_INDEX = 0
DECIMATION = 8
GRIPPER_TAIL_CAPACITY = 64
GRIPPER_JOINT_NAMES = [
    "finger_joint",
    "right_outer_knuckle_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
    "left_inner_finger_knuckle_joint",
    "right_inner_finger_knuckle_joint",
]
GRIPPER_JOINT_INDICES = list(range(7, 13))
EXPECTED_ROBOT_USD_SHA256 = (
    "d8379925b103963dbf3e7c85bcc4ae101b81b7c1d7dabe7d2e964f41d069ec44"
)

SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCTION_EVAL_PATH = SCRIPT_DIR / "eval.py"
EXPECTED_PRODUCTION_EVAL_SIZE_BYTES = 20549
EXPECTED_PRODUCTION_EVAL_SHA256 = (
    "dc115ad9721f7fcc8e15073a871f8b7da97dc96de38093b6e84a761f60b2ddd8"
)
PRODUCTION_RESET_CALL = (
    "obs, info = env.reset(object_positions=initial_conditions[episode])"
)
PRODUCTION_RESET_PROFILE = "production_environment_seed_none_expensive_default_ic0_v1"
PRODUCTION_POLICY_CONFIG_PATH = SCRIPT_DIR.parent / "src" / "polaris" / "config.py"
EXPECTED_PRODUCTION_POLICY_CONFIG_SIZE_BYTES = 4430
EXPECTED_PRODUCTION_POLICY_CONFIG_SHA256 = (
    "1f5fd2b966f49fa116002031cb0b70210a0c41a058652e1fc13fda6585aa8401"
)
PRODUCTION_LAP_CLIENT_PATH = (
    SCRIPT_DIR.parent / "src" / "polaris" / "policy" / "lap_eef_pose_client.py"
)
EXPECTED_PRODUCTION_LAP_CLIENT_SIZE_BYTES = 47564
EXPECTED_PRODUCTION_LAP_CLIENT_SHA256 = (
    "b853182c2cac34e5a25edc09113c49f49537f2b43dacc784104ee7167473aa5e"
)
PRODUCTION_RENDER_DEFAULT_SOURCE = "render_every_step: bool = True"
PRODUCTION_STEP_CALL_SOURCE = "expensive=policy_client.rerender,"
PRODUCTION_RERENDER_SOURCE = (
    "if self.args.render_every_step:\n"
    "            # Full-step videos must use the same splat-composited cameras as\n"
    "            # query frames, rather than raw simulator cameras between queries.\n"
    "            return True"
)
PRODUCTION_STEP_RENDER_PROFILE = "ego_lap_render_every_step_expensive_true_v1"
BOUNDARY_HELPER_PATH = SCRIPT_DIR / "smoke_eef_pose_boundary_replay.py"
EXPECTED_BOUNDARY_HELPER_SIZE_BYTES = 112869
EXPECTED_BOUNDARY_HELPER_SHA256 = (
    "edc62b33f6e5edb7737e121fb60cf801cb9964cbd62a92ccf26d292cc3937209"
)

EXPECTED_ASSET_CONTRACT = {
    "environment": ENVIRONMENT,
    "initial_condition_index": INITIAL_CONDITION_INDEX,
    "initial_conditions_sha256": (
        "40091faee14f692350220871d30705294f21f17ae3d2974cd3c09a34d560f5de"
    ),
    "polaris_hub_revision": "8c7e4103e266ef83d8b1ad2e9a63116edd5f155b",
    "robot_usd_sha256": EXPECTED_ROBOT_USD_SHA256,
    "scene_sha256": (
        "82cd641e422935b394ce7ea7b6be55214c9952a2544000222921e544c409b489"
    ),
}

EXPECTED_FIXTURES: dict[str, dict[str, Any]] = {
    "official_lap3b": {
        "filename": "official_lap3b_job1098292_gate0_actions.json",
        "size_bytes": 8868,
        "sha256": ("0534760269593ee5d00d92a92dfbbc424482725a0dc47556f57fbd54c8a44872"),
        "job_id": "1098292",
        "trace_sha256": (
            "490ba92f39abb1fd83c8382dd1b7a16f4e1a12e86df29cbfb484b6395474789c"
        ),
        "failure": {
            "action_event_count": 118,
            "evidence_sha256": (
                "63c061ec5a47a8bc085547f2abd8dcbc266c9616664d252e29c39ef53864a5f3"
            ),
            "execution_event_count": 117,
            "joint_name": "panda_joint5",
            "physics_substep": 6,
            "policy_step": 117,
            "query": 14,
            "query_chunk_index": 5,
        },
        "action_plan": {
            "profile": "observed_prefix_plus_recorded_query14_continuation_v1",
            "observed_action_count": 118,
            "observed_action_float32_sha256": (
                "53e0a14c089faa79162ed19f0f117843602305ab7d0993591c32e07b4e80ef38"
            ),
            "planned_continuation_action_count": 2,
            "query14_start_step": 112,
            "query14_observed_action_count": 6,
            "query14_executable_action_count": 8,
            "query14_planned_action_count": 16,
            "query14_action_float32_sha256": (
                "6b645fd57bad01ce9f5d6befcc1cddfac6585981f2e3967e8f25dd22f653ceec"
            ),
            "replay_action_count": 120,
            "continuation_semantics": (
                "recorded_query_execute_window_diagnostic_only_not_source_execution"
            ),
        },
        "action_encoding": {
            "action_count": 120,
            "action_width": 8,
            "codec": "zlib-9-base64",
            "compressed_sha256": (
                "bf867563846cb648ed3394c27645ec55f0ca0b5905734d6d8017a7218051857c"
            ),
            "compressed_size_bytes": 3261,
            "dtype": "little-endian-float32",
            "uncompressed_sha256": (
                "6d004a243f613488433a4a422a1bc7d688ef84ce4ade9df8098739d4a9e39d8a"
            ),
            "uncompressed_size_bytes": 3840,
        },
    },
    "reasoning_43075": {
        "filename": "reasoning_43075_job1098294_gate0_actions.json",
        "size_bytes": 8927,
        "sha256": ("22c6a5b73b59aaccb54d0644be13059fb620d6bff81b3fd8731c873c94833527"),
        "job_id": "1098294",
        "trace_sha256": (
            "a48107f2268a38f507bae5848f194f2680ccc52eedd042039cea5fd3cbebd948"
        ),
        "failure": {
            "action_event_count": 113,
            "evidence_sha256": (
                "3c6242a645b40fe29f7223dc4a146cdb7ee04fe661b136098929bb9b973580b8"
            ),
            "execution_event_count": 112,
            "joint_name": "panda_joint7",
            "physics_substep": 2,
            "policy_step": 112,
            "query": 14,
            "query_chunk_index": 0,
        },
        "action_plan": {
            "profile": "observed_prefix_plus_recorded_query14_continuation_v1",
            "observed_action_count": 113,
            "observed_action_float32_sha256": (
                "66dce518684b001e6b03e7c01e7a07a7ea43d801312a4c09e79e713f339e738e"
            ),
            "planned_continuation_action_count": 7,
            "query14_start_step": 112,
            "query14_observed_action_count": 1,
            "query14_executable_action_count": 8,
            "query14_planned_action_count": 16,
            "query14_action_float32_sha256": (
                "8b3b6d77106c2f418e8a14fa6686f3e87622532d3b322ed2d0e65910f2978837"
            ),
            "replay_action_count": 120,
            "continuation_semantics": (
                "recorded_query_execute_window_diagnostic_only_not_source_execution"
            ),
        },
        "action_encoding": {
            "action_count": 120,
            "action_width": 8,
            "codec": "zlib-9-base64",
            "compressed_sha256": (
                "ca7c8dc0cdbff60fc811fe005e84a341262bec0a23f177f4776a0b515d8d4e33"
            ),
            "compressed_size_bytes": 3260,
            "dtype": "little-endian-float32",
            "uncompressed_sha256": (
                "e0b94b8455d0e1ce98eff2ec42ce5fc71a6a8bc925690bbf80b69aeb845f7bdf"
            ),
            "uncompressed_size_bytes": 3840,
        },
    },
}

FIXTURE_FIELDS = {
    "schema_version",
    "fixture_profile",
    "variant",
    "polaris_base_commit",
    "source",
    "asset_contract",
    "action_plan",
    "action_encoding",
    "actions_zlib_base64_chunks",
}
SNAPSHOT_FIELDS = {
    "joint_pos_rad",
    "joint_vel_rad_s",
    "joint_acc_rad_s2",
    "joint_pos_target_rad",
    "joint_vel_target_rad_s",
    "joint_effort_target_nm",
}
GRIPPER_ENTRY_FIELDS = {
    "apply_index",
    "policy_step",
    "physics_substep",
    "raw_action",
    "requested_endpoint_rad",
    "pre",
    "target_after_setter_rad",
    "post",
}
GRIPPER_TAIL_FIELDS = {
    "schema_version",
    "profile",
    "capacity",
    "decimation",
    "joint_names",
    "joint_indices",
    "process_action_calls",
    "total_apply_entries",
    "dropped_entries",
    "entries",
    "failure_snapshot",
}
CAPTURE_FIELDS = {
    "schema_version",
    "profile",
    "finalized",
    "passed",
    "stage",
    "environment",
    "variant",
    "lifecycle",
    "repository",
    "production_eval",
    "fixture",
    "boundary_helper",
    "assets",
    "runtime_protocol",
    "runtime_frame",
    "gripper_runtime_contract",
    "initial_ik_safety",
    "outcome",
    "failure_exception",
    "arm_failure_runtime_evidence",
    "all_six_gripper_tail",
    "close_failures",
}

FAILURE_PATTERN = re.compile(
    r"joint='(?P<joint_name>panda_joint[1-7])'.*"
    r"policy_step=(?P<policy_step>[0-9]+), "
    r"physics_substep=(?P<physics_substep>[0-7]), "
    r"evidence_sha256=(?P<evidence_sha256>[0-9a-f]{64})\)"
)


class Gate0ReplayValidationError(ValueError):
    """A fixture, runtime capture, or publication violated Gate 0."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Gate0ReplayValidationError(message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing/linked file: {path}")
    data = path.read_bytes()
    metadata = path.stat()
    _require(metadata.st_nlink == 1, f"file must have one hard link: {path}")
    return {
        "path": str(path.resolve()),
        "size_bytes": len(data),
        "sha256": _sha256(data),
        "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
    }


def _reject_constant(token: str) -> None:
    raise Gate0ReplayValidationError(f"non-standard JSON constant {token!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Gate0ReplayValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def strict_json_loads(data: bytes, *, field: str) -> dict[str, Any]:
    try:
        result = json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Gate0ReplayValidationError(
            f"{field} is not strict JSON: {error}"
        ) from error
    _require(isinstance(result, dict), f"{field} must be an object")
    return result


def _finite_vector(value: Any, *, field: str, length: int) -> list[float]:
    _require(isinstance(value, list) and len(value) == length, f"{field} shape")
    result: list[float] = []
    for index, item in enumerate(value):
        _require(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item)),
            f"{field}[{index}] must be finite",
        )
        result.append(float(item))
    return result


def _load_boundary_helper() -> tuple[Any, dict[str, Any]]:
    identity = _file_identity(BOUNDARY_HELPER_PATH)
    _require(
        identity["size_bytes"] == EXPECTED_BOUNDARY_HELPER_SIZE_BYTES
        and identity["sha256"] == EXPECTED_BOUNDARY_HELPER_SHA256,
        "boundary helper identity drift",
    )
    module_name = "polaris_gate0_boundary_helper"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, BOUNDARY_HELPER_PATH)
        _require(
            spec is not None and spec.loader is not None, "cannot load boundary helper"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module, identity


def validate_production_reset_source() -> dict[str, Any]:
    identity = _file_identity(PRODUCTION_EVAL_PATH)
    _require(
        identity["size_bytes"] == EXPECTED_PRODUCTION_EVAL_SIZE_BYTES
        and identity["sha256"] == EXPECTED_PRODUCTION_EVAL_SHA256,
        "production eval source identity drift",
    )
    source = PRODUCTION_EVAL_PATH.read_text(encoding="utf-8")
    _require(source.count(PRODUCTION_RESET_CALL) == 1, "production reset call drift")
    _require(
        source.count(PRODUCTION_STEP_CALL_SOURCE) == 1, "production step call drift"
    )
    config_identity = _file_identity(PRODUCTION_POLICY_CONFIG_PATH)
    _require(
        config_identity["size_bytes"] == EXPECTED_PRODUCTION_POLICY_CONFIG_SIZE_BYTES
        and config_identity["sha256"] == EXPECTED_PRODUCTION_POLICY_CONFIG_SHA256,
        "production policy config source identity drift",
    )
    config_source = PRODUCTION_POLICY_CONFIG_PATH.read_text(encoding="utf-8")
    _require(
        config_source.count(PRODUCTION_RENDER_DEFAULT_SOURCE) == 1,
        "production render_every_step default drift",
    )
    client_identity = _file_identity(PRODUCTION_LAP_CLIENT_PATH)
    _require(
        client_identity["size_bytes"] == EXPECTED_PRODUCTION_LAP_CLIENT_SIZE_BYTES
        and client_identity["sha256"] == EXPECTED_PRODUCTION_LAP_CLIENT_SHA256,
        "production LAP client source identity drift",
    )
    client_source = PRODUCTION_LAP_CLIENT_PATH.read_text(encoding="utf-8")
    _require(
        client_source.count(PRODUCTION_RERENDER_SOURCE) == 1,
        "production LAP rerender source drift",
    )
    return {
        **identity,
        "reset_profile": PRODUCTION_RESET_PROFILE,
        "reset_call": PRODUCTION_RESET_CALL,
        "environment_seed": None,
        "reset_expensive_argument": "default_true",
        "initial_condition_index": INITIAL_CONDITION_INDEX,
        "step_render_profile": PRODUCTION_STEP_RENDER_PROFILE,
        "step_expensive_argument": "policy_client.rerender",
        "render_every_step_default": True,
        "effective_step_expensive": True,
        "policy_config_source": config_identity,
        "lap_client_source": client_identity,
    }


def decode_fixture_payload(
    payload: dict[str, Any], *, variant: str
) -> list[list[float]]:
    _require(variant in EXPECTED_FIXTURES, f"unknown variant {variant!r}")
    expected = EXPECTED_FIXTURES[variant]
    _require(set(payload) == FIXTURE_FIELDS, "fixture top-level schema drift")
    _require(payload.get("schema_version") == 1, "fixture schema version")
    _require(payload.get("fixture_profile") == FIXTURE_PROFILE, "fixture profile")
    _require(payload.get("variant") == variant, "fixture variant")
    _require(payload.get("polaris_base_commit") == BASE_COMMIT, "fixture base commit")
    _require(payload.get("asset_contract") == EXPECTED_ASSET_CONTRACT, "fixture assets")
    _require(
        payload.get("action_plan") == expected["action_plan"], "fixture action plan"
    )
    _require(
        payload.get("action_encoding") == expected["action_encoding"],
        "fixture action encoding",
    )
    source = payload.get("source")
    _require(isinstance(source, dict), "fixture source")
    _require(
        source.get("job_id") == expected["job_id"]
        and source.get("trace_sha256") == expected["trace_sha256"]
        and source.get("expected_failure") == expected["failure"],
        "fixture source identity/failure",
    )
    chunks = payload.get("actions_zlib_base64_chunks")
    _require(
        isinstance(chunks, list)
        and chunks
        and all(
            isinstance(chunk, str) and chunk and len(chunk) <= 120 and chunk.isascii()
            for chunk in chunks
        ),
        "fixture action chunks",
    )
    try:
        compressed = base64.b64decode("".join(chunks), validate=True)
    except (TypeError, ValueError) as error:
        raise Gate0ReplayValidationError("fixture action base64") from error
    encoding = expected["action_encoding"]
    _require(
        len(compressed) == encoding["compressed_size_bytes"]
        and _sha256(compressed) == encoding["compressed_sha256"],
        "fixture compressed action identity",
    )
    try:
        raw = zlib.decompress(compressed)
    except zlib.error as error:
        raise Gate0ReplayValidationError("fixture action zlib") from error
    _require(
        len(raw) == encoding["uncompressed_size_bytes"]
        and _sha256(raw) == encoding["uncompressed_sha256"],
        "fixture uncompressed action identity",
    )
    actions = [list(action) for action in struct.iter_unpack("<8f", raw)]
    _require(len(actions) == encoding["action_count"], "fixture action count")
    for step, action in enumerate(actions):
        _finite_vector(action, field=f"action {step}", length=8)
        norm = math.sqrt(sum(component * component for component in action[3:7]))
        _require(abs(norm - 1.0) <= 1e-3, f"action {step} quaternion norm")
        _require(action[7] in (0.0, 1.0), f"action {step} gripper endpoint")
    return actions


def load_replay_fixture(
    variant: str, fixture_path: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any], list[list[float]]]:
    expected = EXPECTED_FIXTURES[variant]
    path = fixture_path or SCRIPT_DIR / "fixtures" / expected["filename"]
    identity = _file_identity(path)
    _require(
        identity["size_bytes"] == expected["size_bytes"]
        and identity["sha256"] == expected["sha256"],
        "fixture file identity drift",
    )
    payload = strict_json_loads(path.read_bytes(), field="replay fixture")
    actions = decode_fixture_payload(payload, variant=variant)
    return identity, payload, actions


def parse_failure_exception(message: str) -> dict[str, Any]:
    _require(isinstance(message, str), "failure message")
    match = FAILURE_PATTERN.search(message)
    _require(match is not None, "failure message contract")
    return {
        "joint_name": match.group("joint_name"),
        "policy_step": int(match.group("policy_step")),
        "physics_substep": int(match.group("physics_substep")),
        "evidence_sha256": match.group("evidence_sha256"),
    }


def _validate_snapshot(value: Any, *, field: str) -> dict[str, Any]:
    _require(
        isinstance(value, dict) and set(value) == SNAPSHOT_FIELDS, f"{field} schema"
    )
    for name in SNAPSHOT_FIELDS:
        _finite_vector(value[name], field=f"{field}.{name}", length=6)
    return dict(value)


def validate_gripper_tail(
    value: Any, *, expected_failure: dict[str, Any]
) -> dict[str, Any]:
    _require(
        isinstance(value, dict) and set(value) == GRIPPER_TAIL_FIELDS,
        "gripper tail schema",
    )
    _require(value["schema_version"] == 1, "gripper tail schema version")
    _require(value["profile"] == GRIPPER_TAIL_PROFILE, "gripper tail profile")
    _require(value["capacity"] == GRIPPER_TAIL_CAPACITY, "gripper tail capacity")
    _require(value["decimation"] == DECIMATION, "gripper tail decimation")
    _require(value["joint_names"] == GRIPPER_JOINT_NAMES, "gripper joint names")
    _require(value["joint_indices"] == GRIPPER_JOINT_INDICES, "gripper joint indices")
    failure_step = expected_failure["policy_step"]
    failure_substep = expected_failure["physics_substep"]
    expected_apply_entries = failure_step * DECIMATION + failure_substep
    _require(
        value["process_action_calls"] == failure_step + 1,
        "gripper process/failure cadence",
    )
    _require(
        value["total_apply_entries"] == expected_apply_entries, "gripper apply cadence"
    )
    entries = value["entries"]
    _require(
        isinstance(entries, list)
        and len(entries) == min(expected_apply_entries, GRIPPER_TAIL_CAPACITY),
        "gripper tail length",
    )
    _require(
        value["dropped_entries"] == expected_apply_entries - len(entries),
        "gripper tail dropped count",
    )
    expected_first = expected_apply_entries - len(entries)
    for offset, entry in enumerate(entries):
        field = f"gripper tail entry {offset}"
        _require(
            isinstance(entry, dict) and set(entry) == GRIPPER_ENTRY_FIELDS,
            f"{field} schema",
        )
        apply_index = expected_first + offset
        _require(entry["apply_index"] == apply_index, f"{field} apply index")
        _require(
            entry["policy_step"] == apply_index // DECIMATION, f"{field} policy step"
        )
        _require(
            entry["physics_substep"] == apply_index % DECIMATION,
            f"{field} physics substep",
        )
        for scalar_name in (
            "raw_action",
            "requested_endpoint_rad",
            "target_after_setter_rad",
        ):
            scalar = entry[scalar_name]
            _require(
                isinstance(scalar, (int, float))
                and not isinstance(scalar, bool)
                and math.isfinite(float(scalar)),
                f"{field} {scalar_name}",
            )
        _require(float(entry["raw_action"]) in (0.0, 1.0), f"{field} raw endpoint")
        _validate_snapshot(entry["pre"], field=f"{field}.pre")
        _validate_snapshot(entry["post"], field=f"{field}.post")
    _require(entries, "gripper tail is empty")
    final_entry = entries[-1]
    _require(
        final_entry["policy_step"] == failure_step
        and final_entry["physics_substep"] == failure_substep - 1,
        "gripper tail terminal cadence",
    )
    failure_snapshot = _validate_snapshot(
        value["failure_snapshot"], field="gripper failure snapshot"
    )
    _require(
        failure_snapshot == final_entry["post"],
        "gripper terminal post/failure identity",
    )
    return dict(value)


def _validate_arm_failure_runtime_evidence(
    value: Any, *, expected_failure: dict[str, Any]
) -> dict[str, Any]:
    _require(isinstance(value, dict), "arm failure runtime evidence")
    _require(
        value.get("policy_step") == expected_failure["policy_step"],
        "arm failure policy step",
    )
    trace = value.get("controller_substep_trace")
    _require(
        trace is not None and value.get("controller_substep_trace_error") is None,
        "arm failure substep ring unavailable",
    )
    boundary, _ = _load_boundary_helper()
    boundary.validate_failure_substep_trace(
        trace,
        safety=value.get("ik_safety"),
        failure_policy_step=expected_failure["policy_step"],
        current_joint_pos=value.get("arm_joint_pos_rad"),
        current_joint_vel=value.get("arm_joint_vel_rad_s"),
        current_joint_pos_target=value.get("arm_joint_target_rad"),
        current_joint_vel_target=value.get("arm_joint_velocity_target_rad_s"),
        current_joint_effort_target=value.get("arm_joint_effort_target_nm"),
        current_approximate_pd_effort_preclip=value.get("arm_computed_torque"),
        current_approximate_pd_effort_postclip=value.get("arm_applied_torque"),
        physx_joint_pos=value.get("physx_arm_joint_pos_rad"),
        physx_joint_vel=value.get("physx_arm_joint_vel_rad_s"),
    )
    _require(
        trace.get("capacity") == 64 and len(trace.get("entries", [])) == 64,
        "arm failure ring is not full",
    )
    final = trace["entries"][-1]
    expected_final_apply_index = (
        expected_failure["policy_step"] * DECIMATION
        + expected_failure["physics_substep"]
        - 1
    )
    _require(
        final.get("apply_index") == expected_final_apply_index
        and final.get("policy_step") == expected_final_apply_index // DECIMATION
        and final.get("physics_substep") == expected_final_apply_index % DECIMATION,
        "arm failure ring terminal cadence",
    )
    return dict(value)


def validate_capture_payload(payload: Any) -> dict[str, Any]:
    _require(
        isinstance(payload, dict) and set(payload) == CAPTURE_FIELDS, "capture schema"
    )
    _require(payload["schema_version"] == 1, "capture schema version")
    _require(payload["profile"] == PROFILE, "capture profile")
    _require(payload["finalized"] is False, "raw capture must be unfinalized")
    _require(payload["passed"] is True, "Gate 0 capture did not pass")
    _require(payload["stage"] == "simulation_app_close_pending", "capture stage")
    _require(payload["environment"] == ENVIRONMENT, "capture environment")
    variant = payload["variant"]
    _require(variant in EXPECTED_FIXTURES, "capture variant")
    expected = EXPECTED_FIXTURES[variant]
    lifecycle = payload["lifecycle"]
    _require(
        isinstance(lifecycle, dict)
        and set(lifecycle)
        == {
            "profile",
            "launch_id",
            "job_id",
            "step_id",
            "nodelist",
            "procid",
            "localid",
            "ntasks",
        }
        and lifecycle["profile"] == "slurm_single_task_srun_lifecycle_v1"
        and re.fullmatch(r"[0-9a-f]{64}", lifecycle["launch_id"])
        and type(lifecycle["job_id"]) is int
        and lifecycle["job_id"] > 0
        and type(lifecycle["step_id"]) is int
        and lifecycle["step_id"] >= 0
        and isinstance(lifecycle["nodelist"], str)
        and bool(lifecycle["nodelist"].strip())
        and lifecycle["procid"] == 0
        and lifecycle["localid"] == 0
        and lifecycle["ntasks"] == 1,
        "capture Slurm/srun lifecycle",
    )
    repository = payload["repository"]
    _require(
        isinstance(repository, dict)
        and set(repository) == {"path", "commit", "clean_tracked"}
        and isinstance(repository["path"], str)
        and re.fullmatch(r"[0-9a-f]{40}", repository["commit"])
        and repository["clean_tracked"] is True,
        "capture repository provenance",
    )
    production_eval = payload["production_eval"]
    _require(
        isinstance(production_eval, dict)
        and production_eval.get("size_bytes") == EXPECTED_PRODUCTION_EVAL_SIZE_BYTES
        and production_eval.get("sha256") == EXPECTED_PRODUCTION_EVAL_SHA256
        and production_eval.get("reset_profile") == PRODUCTION_RESET_PROFILE
        and production_eval.get("reset_call") == PRODUCTION_RESET_CALL
        and production_eval.get("environment_seed") is None
        and production_eval.get("reset_expensive_argument") == "default_true"
        and production_eval.get("initial_condition_index") == 0,
        "capture production reset provenance",
    )
    _require(
        production_eval.get("step_render_profile") == PRODUCTION_STEP_RENDER_PROFILE
        and production_eval.get("step_expensive_argument") == "policy_client.rerender"
        and production_eval.get("render_every_step_default") is True
        and production_eval.get("effective_step_expensive") is True
        and production_eval.get("policy_config_source", {}).get("sha256")
        == EXPECTED_PRODUCTION_POLICY_CONFIG_SHA256
        and production_eval.get("lap_client_source", {}).get("sha256")
        == EXPECTED_PRODUCTION_LAP_CLIENT_SHA256,
        "capture production step-render provenance",
    )
    fixture = payload["fixture"]
    _require(
        isinstance(fixture, dict)
        and fixture.get("size_bytes") == expected["size_bytes"]
        and fixture.get("sha256") == expected["sha256"]
        and fixture.get("source_trace_sha256") == expected["trace_sha256"]
        and fixture.get("action_float32_sha256")
        == expected["action_encoding"]["uncompressed_sha256"]
        and fixture.get("action_count") == 120,
        "capture fixture identity",
    )
    helper = payload["boundary_helper"]
    _require(
        isinstance(helper, dict)
        and helper.get("size_bytes") == EXPECTED_BOUNDARY_HELPER_SIZE_BYTES
        and helper.get("sha256") == EXPECTED_BOUNDARY_HELPER_SHA256,
        "capture boundary helper identity",
    )
    assets = payload["assets"]
    _require(
        isinstance(assets, dict)
        and assets.get("contract") == EXPECTED_ASSET_CONTRACT
        and assets.get("robot_usd", {}).get("sha256") == EXPECTED_ROBOT_USD_SHA256,
        "capture asset identity",
    )
    for field in (
        "runtime_protocol",
        "runtime_frame",
        "initial_ik_safety",
    ):
        _require(
            isinstance(payload[field], dict) and payload[field], f"capture {field}"
        )
    gripper_runtime = payload["gripper_runtime_contract"]
    _require(
        isinstance(gripper_runtime, dict)
        and gripper_runtime.get("gripper_joint_names") == GRIPPER_JOINT_NAMES
        and gripper_runtime.get("gripper_joint_indices") == GRIPPER_JOINT_INDICES,
        "capture gripper runtime all-six identity",
    )
    expected_failure = expected["failure"]
    outcome = payload["outcome"]
    _require(
        isinstance(outcome, dict)
        and outcome
        == {
            "status": "expected_differential_ik_invariant_failure",
            "actions_attempted": expected_failure["policy_step"] + 1,
            "outer_steps_completed": expected_failure["policy_step"],
            "joint_name": expected_failure["joint_name"],
            "policy_step": expected_failure["policy_step"],
            "physics_substep": expected_failure["physics_substep"],
            "evidence_sha256": expected_failure["evidence_sha256"],
        },
        "capture outcome",
    )
    failure_exception = payload["failure_exception"]
    _require(
        isinstance(failure_exception, dict)
        and set(failure_exception) == {"type", "message"}
        and failure_exception["type"].endswith(".DifferentialIKInvariantError")
        and parse_failure_exception(failure_exception["message"])
        == {
            key: expected_failure[key]
            for key in (
                "joint_name",
                "policy_step",
                "physics_substep",
                "evidence_sha256",
            )
        },
        "capture failure exception",
    )
    _validate_arm_failure_runtime_evidence(
        payload["arm_failure_runtime_evidence"], expected_failure=expected_failure
    )
    validate_gripper_tail(
        payload["all_six_gripper_tail"], expected_failure=expected_failure
    )
    _require(payload["close_failures"] == [], "capture close failures")
    return dict(payload)


def _tensor_vector(tensor: Any) -> list[float]:
    values = tensor.detach().cpu().flatten().tolist()
    result = [float(value) for value in values]
    _require(
        len(result) == 6 and all(math.isfinite(value) for value in result),
        "live all-six gripper vector",
    )
    return result


def _make_tracing_gripper_class(base_class: type) -> type:
    """Wrap the production target-slew action without changing its control."""

    class TracingEefBinaryJointPositionTargetSlewAction(base_class):
        def __init__(self, cfg: Any, env: Any) -> None:
            super().__init__(cfg, env)
            live_names = list(self._asset.data.joint_names)
            _require(
                [live_names[index] for index in GRIPPER_JOINT_INDICES]
                == GRIPPER_JOINT_NAMES,
                "live all-six gripper joint ordering drift",
            )
            self._gate0_reset_trace()

        def _gate0_reset_trace(self) -> None:
            self._gate0_policy_step: int | None = None
            self._gate0_physics_substep = 0
            self._gate0_process_calls = 0
            self._gate0_apply_entries: list[dict[str, Any]] = []
            self._gate0_pending_entry: dict[str, Any] | None = None
            self._gate0_raw_action: float | None = None
            self._gate0_requested_endpoint: float | None = None
            self._gate0_failure_snapshot: dict[str, Any] | None = None

        def reset(self, env_ids: Any = None) -> None:
            super().reset(env_ids=env_ids)
            if hasattr(self, "_gate0_apply_entries"):
                self._gate0_reset_trace()

        def begin_gate0_policy_step(self, policy_step: int) -> None:
            _require(type(policy_step) is int and policy_step >= 0, "live policy step")
            # The prior policy step's eighth write is followed by physics only
            # after ``apply_actions`` returns.  The next policy boundary is the
            # first point where that entry can receive its causal post-state.
            self._gate0_finalize_pending()
            _require(
                self._gate0_policy_step is None
                or self._gate0_physics_substep == DECIMATION,
                "prior gripper policy step did not complete decimation",
            )
            self._gate0_policy_step = policy_step
            self._gate0_physics_substep = 0

        def process_actions(self, actions: Any) -> None:
            super().process_actions(actions)
            _require(
                self._gate0_policy_step is not None,
                "gripper process without policy context",
            )
            raw = float(self._raw_actions[0, 0].detach().cpu().item())
            endpoint = float(self._processed_actions[0, 0].detach().cpu().item())
            _require(
                raw in (0.0, 1.0) and math.isfinite(endpoint), "live gripper endpoint"
            )
            self._gate0_raw_action = raw
            self._gate0_requested_endpoint = endpoint
            self._gate0_process_calls += 1

        def _gate0_snapshot(self) -> dict[str, Any]:
            data = self._asset.data
            indices = GRIPPER_JOINT_INDICES
            return {
                "joint_pos_rad": _tensor_vector(data.joint_pos[:, indices][0]),
                "joint_vel_rad_s": _tensor_vector(data.joint_vel[:, indices][0]),
                "joint_acc_rad_s2": _tensor_vector(data.joint_acc[:, indices][0]),
                "joint_pos_target_rad": _tensor_vector(
                    data.joint_pos_target[:, indices][0]
                ),
                "joint_vel_target_rad_s": _tensor_vector(
                    data.joint_vel_target[:, indices][0]
                ),
                "joint_effort_target_nm": _tensor_vector(
                    data.joint_effort_target[:, indices][0]
                ),
            }

        def _gate0_finalize_pending(self) -> None:
            if self._gate0_pending_entry is None:
                return
            self._gate0_pending_entry["post"] = self._gate0_snapshot()
            self._gate0_apply_entries.append(self._gate0_pending_entry)
            self._gate0_pending_entry = None

        def apply_actions(self) -> None:
            self._gate0_finalize_pending()
            _require(
                self._gate0_policy_step is not None
                and self._gate0_raw_action is not None
                and self._gate0_requested_endpoint is not None
                and 0 <= self._gate0_physics_substep < DECIMATION,
                "gripper apply cadence/context",
            )
            pre = self._gate0_snapshot()
            super().apply_actions()
            target = float(
                self._asset.data.joint_pos_target[0, GRIPPER_JOINT_INDICES[0]]
                .detach()
                .cpu()
                .item()
            )
            self._gate0_pending_entry = {
                "apply_index": len(self._gate0_apply_entries),
                "policy_step": self._gate0_policy_step,
                "physics_substep": self._gate0_physics_substep,
                "raw_action": self._gate0_raw_action,
                "requested_endpoint_rad": self._gate0_requested_endpoint,
                "pre": pre,
                "target_after_setter_rad": target,
                "post": None,
            }
            self._gate0_physics_substep += 1

        def finalize_gate0_failure(self) -> None:
            self._gate0_finalize_pending()
            self._gate0_failure_snapshot = self._gate0_snapshot()

        def gate0_gripper_tail(self) -> dict[str, Any]:
            _require(
                self._gate0_pending_entry is None, "pending gripper entry at report"
            )
            _require(
                self._gate0_failure_snapshot is not None,
                "missing gripper failure snapshot",
            )
            total = len(self._gate0_apply_entries)
            entries = self._gate0_apply_entries[-GRIPPER_TAIL_CAPACITY:]
            return {
                "schema_version": 1,
                "profile": GRIPPER_TAIL_PROFILE,
                "capacity": GRIPPER_TAIL_CAPACITY,
                "decimation": DECIMATION,
                "joint_names": list(GRIPPER_JOINT_NAMES),
                "joint_indices": list(GRIPPER_JOINT_INDICES),
                "process_action_calls": self._gate0_process_calls,
                "total_apply_entries": total,
                "dropped_entries": total - len(entries),
                "entries": entries,
                "failure_snapshot": self._gate0_failure_snapshot,
            }

    # The production installer deliberately requires this exact class name.
    TracingEefBinaryJointPositionTargetSlewAction.__name__ = (
        "EefBinaryJointPositionTargetSlewAction"
    )
    TracingEefBinaryJointPositionTargetSlewAction.__qualname__ = (
        "EefBinaryJointPositionTargetSlewAction"
    )
    return TracingEefBinaryJointPositionTargetSlewAction


def _repository_provenance(expected_commit: str) -> dict[str, Any]:
    _require(
        re.fullmatch(r"[0-9a-f]{40}", expected_commit) is not None,
        "expected commit must be full lowercase SHA-1",
    )
    root = SCRIPT_DIR.parent
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    _require(head == expected_commit, "live repository commit drift")
    _require(status == "", "live repository has tracked modifications")
    return {"path": str(root), "commit": head, "clean_tracked": True}


def _slurm_lifecycle(launch_id: str) -> dict[str, Any]:
    _require(
        re.fullmatch(r"[0-9a-f]{64}", launch_id) is not None,
        "launch ID must be a full lowercase SHA-256 token",
    )

    def decimal_environment(name: str, *, minimum: int) -> int:
        value = os.environ.get(name)
        _require(
            isinstance(value, str) and value.isdecimal() and int(value) >= minimum,
            f"missing/invalid in-srun {name}",
        )
        return int(value)

    nodelist = os.environ.get("SLURM_NODELIST")
    _require(
        isinstance(nodelist, str) and bool(nodelist.strip()),
        "missing in-srun SLURM_NODELIST",
    )
    lifecycle = {
        "profile": "slurm_single_task_srun_lifecycle_v1",
        "launch_id": launch_id,
        "job_id": decimal_environment("SLURM_JOB_ID", minimum=1),
        "step_id": decimal_environment("SLURM_STEP_ID", minimum=0),
        "nodelist": nodelist,
        "procid": decimal_environment("SLURM_PROCID", minimum=0),
        "localid": decimal_environment("SLURM_LOCALID", minimum=0),
        "ntasks": decimal_environment("SLURM_NTASKS", minimum=1),
    }
    _require(
        lifecycle["procid"] == lifecycle["localid"] == 0 and lifecycle["ntasks"] == 1,
        "Gate 0 requires one rank in one srun step",
    )
    return lifecycle


def _validate_output_namespace(
    path: Path, *, variant: str, lifecycle: dict[str, Any]
) -> None:
    resolved = path.resolve()
    _require(
        resolved.name == f"gate0-{variant}.raw.json",
        "raw result filename/variant drift",
    )
    _require(
        resolved.parent.name == f"launch_{lifecycle['launch_id']}"
        and resolved.parent.parent.name == f"job_{lifecycle['job_id']}"
        and resolved.parent.parent.parent.name == variant,
        "raw result must use variant/job/launch output namespace",
    )


def _capture_assets(
    boundary: Any, *, scene_path: Path, robot_usd_path: Path
) -> dict[str, Any]:
    scene = boundary.validate_asset_contract(scene_path)
    robot = _file_identity(robot_usd_path)
    _require(robot["sha256"] == EXPECTED_ROBOT_USD_SHA256, "robot USD digest drift")
    return {"contract": EXPECTED_ASSET_CONTRACT, "scene": scene, "robot_usd": robot}


def _run_live(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    import gymnasium as gym  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: PLC0415

    import polaris.environments  # noqa: F401, PLC0415
    from polaris.eef_gripper_runtime import (  # noqa: PLC0415
        GRIPPER_JOINT_INDICES as PRODUCTION_GRIPPER_JOINT_INDICES,
        GRIPPER_JOINT_NAMES as PRODUCTION_GRIPPER_JOINT_NAMES,
        install_eef_gripper_runtime,
        record_eef_gripper_post_policy_step,
    )
    from polaris.eef_runtime_contract import (  # noqa: PLC0415
        begin_eef_safety_episode,
        configure_ego_lap_environment_timeout,
        validate_eef_runtime_frame,
        validate_eef_runtime_safety,
        validate_ego_lap_runtime_protocol,
    )
    from polaris.environments.droid_cfg import (  # noqa: PLC0415
        EefBinaryJointPositionTargetSlewAction,
        EgoLapEefPoseActionCfg,
    )
    from polaris.environments.robot_cfg import (  # noqa: PLC0415
        configure_eef_pose_joint_safety,
    )
    from polaris.robust_differential_ik import (  # noqa: PLC0415
        DifferentialIKInvariantError,
    )
    from polaris.utils import load_eval_initial_conditions  # noqa: PLC0415

    state["stage"] = "bind_repository"
    _require(
        list(PRODUCTION_GRIPPER_JOINT_NAMES) == GRIPPER_JOINT_NAMES
        and list(PRODUCTION_GRIPPER_JOINT_INDICES) == GRIPPER_JOINT_INDICES,
        "Gate 0/production all-six gripper identity drift",
    )
    repository = _repository_provenance(args.expected_polaris_commit)
    production_eval = validate_production_reset_source()
    lifecycle = _slurm_lifecycle(args.launch_id)
    _validate_output_namespace(
        args.output_json, variant=args.variant, lifecycle=lifecycle
    )
    state["stage"] = "load_fixture"
    fixture_identity, fixture_payload, actions = load_replay_fixture(args.variant)
    boundary, helper_identity = _load_boundary_helper()

    state["stage"] = "build_environment"
    env_cfg = parse_env_cfg(
        ENVIRONMENT,
        device=args.device,
        num_envs=1,
        use_fabric=True,
    )
    configure_ego_lap_environment_timeout(env_cfg)
    env_cfg.actions = EgoLapEefPoseActionCfg()
    env_cfg.actions.arm.enable_failure_substep_trace = True
    env_cfg.actions.arm.enable_wrist_energy_brake = False
    tracing_class = _make_tracing_gripper_class(EefBinaryJointPositionTargetSlewAction)
    env_cfg.actions.finger_joint.class_type = tracing_class
    configure_eef_pose_joint_safety(
        env_cfg.scene.robot,
        physx_cfg=env_cfg.sim.physx,
        enable_gripper_velocity_limit=True,
    )
    robot_usd_path = Path(env_cfg.scene.robot.spawn.usd_path)
    env = gym.make(ENVIRONMENT, cfg=env_cfg)
    state["env"] = env
    runtime_protocol = validate_ego_lap_runtime_protocol(env)

    state["stage"] = "validate_assets"
    assets = _capture_assets(
        boundary,
        scene_path=Path(env.unwrapped.usd_file),
        robot_usd_path=robot_usd_path,
    )
    _, initial_conditions = load_eval_initial_conditions(
        usd=env.unwrapped.usd_file,
        rollouts=1,
    )
    _require(
        isinstance(initial_conditions, list)
        and len(initial_conditions) == 1
        and isinstance(initial_conditions[INITIAL_CONDITION_INDEX], dict),
        "FoodBussing IC0 loader drift",
    )

    state["stage"] = "reset_ic0"
    # Match the production source exactly: Environment seed=None and the reset
    # render uses its default expensive=True path before runtime installation.
    observation, _ = env.reset(
        object_positions=initial_conditions[INITIAL_CONDITION_INDEX]
    )
    gripper_runtime_contract = install_eef_gripper_runtime(
        env, robot_usd_path=robot_usd_path
    )
    runtime_frame = validate_eef_runtime_frame(env, observation)
    begin_eef_safety_episode(env, 0)
    initial_ik_safety = validate_eef_runtime_safety(env, require_gripper_runtime=True)
    terms = env.unwrapped.action_manager._terms
    _require(list(terms) == ["arm", "finger_joint"], "live action order drift")
    arm_term = terms["arm"]
    finger_term = terms["finger_joint"]
    _require(type(finger_term) is tracing_class, "tracing gripper class not installed")
    _require(
        getattr(arm_term, "_failure_substep_trace_enabled", False) is True,
        "arm failure substep trace disabled",
    )

    expected_failure = EXPECTED_FIXTURES[args.variant]["failure"]
    outcome: dict[str, Any] | None = None
    failure_exception: dict[str, str] | None = None
    arm_failure: dict[str, Any] | None = None
    gripper_tail: dict[str, Any] | None = None
    state["stage"] = "replay_actions"
    for step, action_values in enumerate(actions):
        state["policy_step"] = step
        finger_term.begin_gate0_policy_step(step)
        action = torch.tensor(
            action_values, dtype=torch.float32, device=env.device
        ).reshape(1, -1)
        try:
            observation, _, terminated, truncated, _ = env.step(action, expensive=True)
        except DifferentialIKInvariantError as error:
            parsed = parse_failure_exception(str(error))
            wanted = {
                key: expected_failure[key]
                for key in (
                    "joint_name",
                    "policy_step",
                    "physics_substep",
                    "evidence_sha256",
                )
            }
            _require(parsed == wanted, "live failure cadence/digest drift")
            _require(step == expected_failure["policy_step"], "failure loop step drift")
            finger_term.finalize_gate0_failure()
            arm_failure = boundary._capture_failure_runtime_evidence(
                env, policy_step=step
            )
            gripper_tail = finger_term.gate0_gripper_tail()
            validate_gripper_tail(gripper_tail, expected_failure=expected_failure)
            _validate_arm_failure_runtime_evidence(
                arm_failure, expected_failure=expected_failure
            )
            outcome = {
                "status": "expected_differential_ik_invariant_failure",
                "actions_attempted": step + 1,
                "outer_steps_completed": step,
                **parsed,
            }
            failure_exception = {
                "type": f"{type(error).__module__}.{type(error).__qualname__}",
                "message": str(error),
            }
            break
        _require(not bool(terminated[0]), f"replay terminated at step {step}")
        _require(not bool(truncated[0]), f"replay truncated at step {step}")
        record_eef_gripper_post_policy_step(env)

    _require(outcome is not None, "replay exhausted without expected failure")
    _require(
        failure_exception is not None
        and arm_failure is not None
        and gripper_tail is not None,
        "expected failure evidence incomplete",
    )
    state["policy_step"] = None
    return {
        "lifecycle": lifecycle,
        "repository": repository,
        "production_eval": production_eval,
        "fixture": {
            **fixture_identity,
            "source_trace_sha256": fixture_payload["source"]["trace_sha256"],
            "action_float32_sha256": fixture_payload["action_encoding"][
                "uncompressed_sha256"
            ],
            "action_count": len(actions),
        },
        "boundary_helper": helper_identity,
        "assets": assets,
        "runtime_protocol": runtime_protocol,
        "runtime_frame": runtime_frame,
        "gripper_runtime_contract": gripper_runtime_contract,
        "initial_ik_safety": initial_ik_safety,
        "outcome": outcome,
        "failure_exception": failure_exception,
        "arm_failure_runtime_evidence": arm_failure,
        "all_six_gripper_tail": gripper_tail,
    }


def _strict_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _atomic_write_immutable(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    serialized = _strict_json_bytes(payload)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        path.chmod(0o444)
        file_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(file_fd)
        finally:
            os.close(file_fd)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    reread = path.read_bytes()
    _require(reread == serialized, "published JSON changed on reread")
    _require(stat.S_IMODE(path.stat().st_mode) == 0o444, "published JSON mode")
    return {
        "path": str(path),
        "size_bytes": len(reread),
        "sha256": _sha256(reread),
        "mode": "0444",
    }


def _exception_evidence(error: BaseException) -> dict[str, str]:
    return {
        "type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": str(error),
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
    }


def _parse_args() -> tuple[argparse.Namespace, Any]:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(EXPECTED_FIXTURES), required=True)
    parser.add_argument("--expected-polaris-commit", required=True)
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.headless = True
    return args, AppLauncher


def main() -> int:
    args, app_launcher_type = _parse_args()
    state: dict[str, Any] = {
        "stage": "launch_simulation_app",
        "policy_step": None,
        "env": None,
    }
    simulation_app = None
    close_failures: list[dict[str, str]] = []
    try:
        app_launcher = app_launcher_type(args)
        simulation_app = app_launcher.app
        evidence = _run_live(args, state)
        env = state.get("env")
        if env is not None:
            state["stage"] = "close_environment"
            try:
                env.close()
            except BaseException as error:
                close_failures.append(_exception_evidence(error))
                raise
        state["stage"] = "simulation_app_close_pending"
        payload = {
            "schema_version": 1,
            "profile": PROFILE,
            "finalized": False,
            "passed": True,
            "stage": state["stage"],
            "environment": ENVIRONMENT,
            "variant": args.variant,
            **evidence,
            "close_failures": close_failures,
        }
        validate_capture_payload(payload)
        identity = _atomic_write_immutable(args.output_json, payload)
        marker_path = args.output_json.resolve().with_name(
            args.output_json.name + ".ready.json"
        )
        marker = {
            "schema_version": 1,
            "profile": PROFILE,
            "stage": "simulation_app_close_pending",
            "raw_result": identity,
        }
        _atomic_write_immutable(marker_path, marker)
        print(
            "POLARIS_GATE0_RAW="
            f"{identity['path']};size={identity['size_bytes']};"
            f"sha256={identity['sha256']};mode={identity['mode']}",
            flush=True,
        )
        print(f"POLARIS_GATE0_READY_MARKER={marker_path}", flush=True)
        simulation_app.close()
        return 0
    except BaseException as error:
        traceback.print_exception(
            type(error), error, error.__traceback__, file=sys.stderr
        )
        failure_payload = {
            "schema_version": 1,
            "profile": PROFILE,
            "finalized": False,
            "passed": False,
            "stage": "failed",
            "environment": ENVIRONMENT,
            "variant": getattr(args, "variant", None),
            "policy_step": state.get("policy_step"),
            "failure": _exception_evidence(error),
            "close_failures": close_failures,
        }
        try:
            _atomic_write_immutable(args.output_json, failure_payload)
        except BaseException as persistence_error:
            traceback.print_exception(
                type(persistence_error),
                persistence_error,
                persistence_error.__traceback__,
                file=sys.stderr,
            )
        if simulation_app is not None:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
