#!/usr/bin/env python3
"""Replay the official LAP-3B boundary failure and stress the v4 EEF guard.

This is a standalone Isaac smoke.  It does not start a policy server or load a
checkpoint: the committed fixture contains the exact absolute PolaRiS actions
from the preserved official-LAP v3 canary.  The fixture and all source/assets
are content-bound before simulation starts.

The host-only fixture and evidence validators intentionally import only the
Python standard library so their fail-closed behavior can be unit-tested
without Isaac Sim.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import traceback
from typing import Any
import zlib


ENVIRONMENT = "DROID-FoodBussing"
FIXTURE_PROFILE = "official_lap3b_foodbussing_v3_boundary_actions_v1"
SMOKE_PROFILE = "panda_joint5_upper_guardband_boundary_replay_v1"
FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "official_lap3b_foodbussing_v3_boundary_actions.json"
)
EXPECTED_FIXTURE_SIZE_BYTES = 15967
EXPECTED_FIXTURE_SHA256 = (
    "640a11df435b6a8d05e924a3781c86e121dec15477211aa67f301423c539d910"
)
EXPECTED_SOURCE = {
    "checkpoint_revision": "601db9c1ab4bcaf6dddb160c7b2dec589a67b730",
    "ego_lap_commit": "c2eae469ab802f66135dc8d8e806b1454e3b5a25",
    "episode_index": 0,
    "failed_physics_substep": 5,
    "failed_policy_step": 377,
    "initial_condition_index": 0,
    "polaris_commit": "b7c8ea4218c5ef8492fe96617fae689d7a646f2d",
    "serving_contract_sha256": (
        "126a6f42ae37dcda2fdb1b34e1608353d7fc674431598eb6939fb01132937aea"
    ),
    "sidecar_sha256": (
        "0e0ab7b94c66e879475d6887a8ade1b343a6b7123d97e9ef0b702574ae4f0674"
    ),
    "task": ENVIRONMENT,
    "trace_sha256": (
        "96d158fde9151aa6f00bd7f5db75fec753aa3cc5153a5da83b953c72fc413dd0"
    ),
    "trace_size_bytes": 713719,
}
EXPECTED_ASSET_CONTRACT = {
    "initial_conditions_sha256": (
        "40091faee14f692350220871d30705294f21f17ae3d2974cd3c09a34d560f5de"
    ),
    "polaris_hub_revision": "8c7e4103e266ef83d8b1ad2e9a63116edd5f155b",
    "scene_sha256": (
        "82cd641e422935b394ce7ea7b6be55214c9952a2544000222921e544c409b489"
    ),
}
EXPECTED_ACTION_ENCODING = {
    "action_count": 378,
    "action_width": 8,
    "codec": "zlib-9-base64",
    "compressed_sha256": (
        "d68fbc6b8801a4997cf63f6168351f638eb67b5c2e6cb4548bca81ab9efdc0dc"
    ),
    "compressed_size_bytes": 10159,
    "dtype": "little-endian-float32",
    "uncompressed_sha256": (
        "d6fa051e79902532fd4e88c090a860cda59a2dc11d196e91a1111df06142e131"
    ),
    "uncompressed_size_bytes": 12096,
}

TARGET_JOINT_NAME = "panda_joint5"
TARGET_JOINT_INDEX = 4
OUTWARD_DELTA_SCALE = 0.5
ADAPTIVE_DRIVE_STEPS = 16
REQUIRED_CONSECUTIVE_DWELL_STEPS = 8
DECIMATION = 8
EXPECTED_TOTAL_POLICY_STEPS = (
    EXPECTED_ACTION_ENCODING["action_count"] + ADAPTIVE_DRIVE_STEPS
)
EXPECTED_APPLY_CALLS = EXPECTED_TOTAL_POLICY_STEPS * DECIMATION
OUTER_LOWER_LIMIT_RAD = -2.8973000049591064
OUTER_UPPER_LIMIT_RAD = 2.8973000049591064
INNER_UPPER_LIMIT_RAD = 2.8755500316619873
TARGET_LIMIT_DIGEST = "09b20ab18c35d6dc22a3edbc2beca2edff419e242dd07d74cd1d65df9ce67e0f"
SOFT_LIMIT_DIGEST = "fbf7535901c042fea5d901812ecd02c5fd81ade06c23c1499c32d66a859104de"
EXPECTED_MAX_DELTA_RAD = [0.018125001341104507] * 4 + [0.02174999937415123] * 3
EXPECTED_VELOCITY_LIMITS_RAD_S = [2.174999952316284] * 4 + [2.609999895095825] * 3
EXPECTED_EFFORT_LIMITS = [87.0] * 4 + [12.0] * 3
EXPECTED_OUTER_LIMITS_RAD = [
    [-2.8973000049591064, 2.8973000049591064],
    [-1.7627999782562256, 1.7627999782562256],
    [-2.8973000049591064, 2.8973000049591064],
    [-3.0717999935150146, -0.06979990005493164],
    [-2.8973000049591064, 2.8973000049591064],
    [-0.017499923706054688, 3.752500057220459],
    [-2.8973000049591064, 2.8973000049591064],
]
EXPECTED_TARGET_LIMITS_RAD = [
    [-2.8791749477386475, 2.8791749477386475],
    [-1.7446749210357666, 1.7446749210357666],
    [-2.8791749477386475, 2.8791749477386475],
    [-3.0536749362945557, -0.08792489767074585],
    [-2.8755500316619873, 2.8755500316619873],
    [0.004250075668096542, 3.73075008392334],
    [-2.8755500316619873, 2.8755500316619873],
]
ZERO_COUNTERS = (
    "current_joint_limit_aborts",
    "invariant_aborts",
    "nonfinite_aborts",
    "post_clamp_target_violations",
    "dls_fallbacks",
    "guard_diagnostics_dropped",
)
SAFETY_FIELDS = {
    "episode_index",
    "profile",
    "apply_actions_cadence",
    "physics_dt",
    "control_dt",
    "decimation",
    "current_joint_soft_limit_tolerance_rad",
    "target_soft_limit_guard_band_profile",
    "eef_quaternion_unit_norm_tolerance",
    "joint_slew_float32_tolerance_rad",
    "soft_joint_pos_limit_factor",
    "joint_names",
    "joint_velocity_limits_rad_s",
    "joint_effort_limits",
    "max_delta_joint_pos_rad",
    "target_soft_limit_margin_rad",
    "target_joint_pos_limits_rad",
    "target_joint_pos_limits_float32_sha256",
    "soft_joint_pos_limits_rad",
    "soft_joint_pos_limits_float32_sha256",
    "counters",
    "maxima",
    "guard_diagnostics",
    "max_raw_delta_diagnostic",
}
SAFETY_COUNTER_FIELDS = {
    "apply_calls",
    "environment_substeps",
    "slew_limit_events",
    "slew_limited_joints",
    "position_limit_events",
    "position_limited_joints",
    "post_clamp_target_violations",
    "current_joint_limit_aborts",
    "invariant_aborts",
    "nonfinite_aborts",
    "dls_fallbacks",
    "guard_diagnostics_dropped",
}
SAFETY_MAXIMA_FIELDS = {
    "raw_delta_joint_pos_rad",
    "applied_delta_joint_pos_rad",
    "raw_target_soft_limit_violation_rad",
    "post_clamp_target_soft_limit_violation_rad",
    "post_clamp_target_guard_band_violation_rad",
    "current_joint_soft_limit_violation_rad",
}
DIAGNOSTIC_FIELDS = {
    "kind",
    "episode_index",
    "policy_step",
    "physics_substep",
    "joint_pos_rad",
    "raw_delta_joint_pos_rad",
    "raw_joint_pos_target_rad",
    "safe_joint_pos_target_rad",
    "pose_error_norm",
    "jacobian_finite",
    "jacobian_max_abs",
    "eef_quaternion_norm",
}

FIXTURE_FIELDS = {
    "schema_version",
    "fixture_profile",
    "source",
    "asset_contract",
    "action_encoding",
    "actions_zlib_base64_chunks",
}
BOUNDARY_FIELDS = {
    "target_joint_name",
    "target_joint_index",
    "direction",
    "replay_action_count",
    "adaptive_drive_steps",
    "total_policy_steps",
    "expected_apply_calls",
    "outward_delta_scale",
    "required_consecutive_dwell_policy_steps",
    "observed_max_consecutive_dwell_policy_steps",
    "joint_outer_lower_limit_rad",
    "joint_outer_upper_limit_rad",
    "joint_inner_target_upper_limit_rad",
    "joint_pos_observed_min_rad",
    "joint_pos_observed_max_rad",
    "terminated",
    "truncated",
    "state_is_finite",
    "dwell_records",
}
DWELL_RECORD_FIELDS = {
    "drive_step",
    "policy_step",
    "position_limit_events_delta",
    "joint_pos_rad",
    "joint_vel_rad_s",
    "joint_target_rad",
    "predicted_outward_joint_delta_rad",
    "arm_joint_pos_rad",
    "arm_joint_vel_rad_s",
    "arm_joint_target_rad",
    "eef_position_m",
    "eef_quaternion_wxyz",
    "target_is_inner_limit",
    "within_outer_limits",
    "all_arm_joints_within_outer_limits",
    "eef_state_is_finite",
    "state_is_finite",
}
SUCCESS_PAYLOAD_FIELDS = {
    "schema_version",
    "fixture_profile",
    "smoke_profile",
    "finalized",
    "passed",
    "stage",
    "exit_code",
    "environment",
    "fixture",
    "assets",
    "runtime_protocol",
    "runtime_frame",
    "initial_ik_safety_capture",
    "boundary",
    "ik_safety",
    "failure",
    "close_failures",
}
RUNTIME_FRAME_FIELDS = {
    "eef_frame",
    "reference_frame",
    "position_error_m",
    "rotation_error_rad",
    "controlled_body",
    "body_offset",
    "command_type",
    "use_relative_mode",
    "ik_method",
    "dls_damping",
    "arm_scale",
    "arm_joint_names",
    "gripper_threshold_profile",
    "ik_safety_profile",
    "action_dim",
}


class BoundaryReplayValidationError(ValueError):
    """A fixture, asset, or smoke result does not satisfy the closed contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BoundaryReplayValidationError(message)


def _finite_number(value: Any, field: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{field} must be finite",
    )
    return float(value)


def _finite_vector(value: Any, field: str, *, length: int) -> list[float]:
    _require(isinstance(value, list) and len(value) == length, f"{field} shape")
    return [
        _finite_number(item, f"{field}[{index}]") for index, item in enumerate(value)
    ]


def _reject_constant(token: str) -> None:
    raise BoundaryReplayValidationError(f"non-standard JSON constant {token!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise BoundaryReplayValidationError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def strict_json_loads(data: bytes, *, field: str) -> dict[str, Any]:
    """Decode strict UTF-8 JSON, rejecting duplicate keys and NaN/Infinity."""

    try:
        value = json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BoundaryReplayValidationError(
            f"{field} is not strict UTF-8 JSON: {error}"
        ) from error
    _require(isinstance(value, dict), f"{field} must be a JSON object")
    return value


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing file: {path}")
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "size_bytes": len(data),
        "sha256": _sha256_bytes(data),
        "mode": f"{stat.S_IMODE(path.stat().st_mode):04o}",
    }


def _same_float32(left: float, right: float) -> bool:
    return struct.pack("<f", left) == struct.pack("<f", right)


def decode_replay_fixture(payload: dict[str, Any]) -> list[list[float]]:
    """Validate and decode the committed replay fixture entirely in memory."""

    _require(set(payload) == FIXTURE_FIELDS, "fixture top-level schema drift")
    _require(payload.get("schema_version") == 1, "fixture schema_version")
    _require(payload.get("fixture_profile") == FIXTURE_PROFILE, "fixture profile")
    _require(payload.get("source") == EXPECTED_SOURCE, "fixture source drift")
    _require(
        payload.get("asset_contract") == EXPECTED_ASSET_CONTRACT,
        "fixture asset contract drift",
    )
    _require(
        payload.get("action_encoding") == EXPECTED_ACTION_ENCODING,
        "fixture action encoding drift",
    )

    chunks = payload.get("actions_zlib_base64_chunks")
    _require(isinstance(chunks, list) and chunks, "fixture action chunks missing")
    _require(
        all(
            isinstance(chunk, str) and chunk and len(chunk) <= 120 and chunk.isascii()
            for chunk in chunks
        ),
        "fixture action chunks invalid",
    )
    try:
        compressed = base64.b64decode("".join(chunks), validate=True)
    except (ValueError, TypeError) as error:
        raise BoundaryReplayValidationError(
            "fixture action payload is not canonical base64"
        ) from error
    _require(
        len(compressed) == EXPECTED_ACTION_ENCODING["compressed_size_bytes"],
        "fixture compressed action size mismatch",
    )
    _require(
        _sha256_bytes(compressed) == EXPECTED_ACTION_ENCODING["compressed_sha256"],
        "fixture compressed action digest mismatch",
    )
    try:
        raw = zlib.decompress(compressed)
    except zlib.error as error:
        raise BoundaryReplayValidationError(
            "fixture action payload is not valid zlib"
        ) from error
    _require(
        len(raw) == EXPECTED_ACTION_ENCODING["uncompressed_size_bytes"],
        "fixture uncompressed action size mismatch",
    )
    _require(
        _sha256_bytes(raw) == EXPECTED_ACTION_ENCODING["uncompressed_sha256"],
        "fixture uncompressed action digest mismatch",
    )

    actions = [list(values) for values in struct.iter_unpack("<8f", raw)]
    _require(
        len(actions) == EXPECTED_ACTION_ENCODING["action_count"],
        "fixture action count mismatch",
    )
    for step, action in enumerate(actions):
        _require(
            all(math.isfinite(value) for value in action), f"action {step} nonfinite"
        )
        quaternion_norm = math.sqrt(sum(value * value for value in action[3:7]))
        _require(
            abs(quaternion_norm - 1.0) <= 1e-3,
            f"action {step} quaternion norm",
        )
        _require(action[7] in (0.0, 1.0), f"action {step} gripper is not binary")
    return actions


def load_replay_fixture(
    path: Path = FIXTURE_PATH,
) -> tuple[dict[str, Any], list[list[float]]]:
    """Load the one exact committed fixture and return its identity and actions."""

    identity = _file_identity(path)
    _require(
        identity["size_bytes"] == EXPECTED_FIXTURE_SIZE_BYTES,
        "fixture file size mismatch",
    )
    _require(
        identity["sha256"] == EXPECTED_FIXTURE_SHA256,
        "fixture file digest mismatch",
    )
    payload = strict_json_loads(path.read_bytes(), field="replay fixture")
    actions = decode_replay_fixture(payload)
    return {
        **identity,
        "fixture_profile": FIXTURE_PROFILE,
        "source_trace_sha256": EXPECTED_SOURCE["trace_sha256"],
        "action_float32_sha256": EXPECTED_ACTION_ENCODING["uncompressed_sha256"],
        "action_count": len(actions),
    }, actions


def validate_asset_contract(scene_path: Path) -> dict[str, Any]:
    """Validate FoodBussing scene, IC bytes, and both Hub revision records."""

    scene_path = scene_path.resolve()
    _require(scene_path.name == "scene.usda", "unexpected scene filename")
    _require(scene_path.parent.name == "food_bussing", "unexpected scene directory")
    initial_conditions = scene_path.parent / "initial_conditions.json"
    scene_identity = _file_identity(scene_path)
    initial_identity = _file_identity(initial_conditions)
    _require(
        scene_identity["sha256"] == EXPECTED_ASSET_CONTRACT["scene_sha256"],
        "FoodBussing scene digest mismatch",
    )
    _require(
        initial_identity["sha256"]
        == EXPECTED_ASSET_CONTRACT["initial_conditions_sha256"],
        "FoodBussing initial-condition digest mismatch",
    )

    data_root = scene_path.parent.parent
    metadata_root = (
        data_root / ".cache" / "huggingface" / "download" / scene_path.parent.name
    )
    metadata: dict[str, Any] = {}
    for filename in ("initial_conditions.json", "scene.usda"):
        metadata_path = metadata_root / f"{filename}.metadata"
        identity = _file_identity(metadata_path)
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
        _require(lines, f"empty Hub metadata for {filename}")
        revision = lines[0].strip()
        _require(
            revision == EXPECTED_ASSET_CONTRACT["polaris_hub_revision"],
            f"PolaRiS-Hub revision mismatch for {filename}",
        )
        metadata[filename] = {**identity, "revision": revision}
    return {
        "scene": scene_identity,
        "initial_conditions": initial_identity,
        "polaris_hub_revision": EXPECTED_ASSET_CONTRACT["polaris_hub_revision"],
        "revision_metadata": metadata,
        "initial_condition_index": 0,
    }


def _dwell_predicate(record: dict[str, Any]) -> bool:
    return (
        record["position_limit_events_delta"] == DECIMATION
        and _same_float32(record["joint_target_rad"], INNER_UPPER_LIMIT_RAD)
        and record["target_is_inner_limit"] is True
        and record["within_outer_limits"] is True
        and record["state_is_finite"] is True
        and record["predicted_outward_joint_delta_rad"] > 0.0
    )


def _validate_float_matrix(
    value: Any, field: str, expected: list[list[float]]
) -> list[list[float]]:
    _require(isinstance(value, list) and len(value) == len(expected), f"{field} shape")
    result = []
    for index, (row, expected_row) in enumerate(zip(value, expected, strict=True)):
        actual_row = _finite_vector(row, f"{field}[{index}]", length=len(expected_row))
        _require(
            all(
                _same_float32(actual, wanted)
                for actual, wanted in zip(actual_row, expected_row, strict=True)
            ),
            f"{field}[{index}] value drift",
        )
        result.append(actual_row)
    return result


def _validate_diagnostic_vector(value: Any, field: str) -> list[float]:
    _require(isinstance(value, dict), f"{field} must be an object")
    _require(
        set(value) == {"values", "finite_mask", "finite_count"},
        f"{field} schema",
    )
    values = _finite_vector(value.get("values"), f"{field}.values", length=7)
    _require(value.get("finite_mask") == [True] * 7, f"{field} finite mask")
    _require(value.get("finite_count") == 7, f"{field} finite count")
    return values


def validate_safety_static(
    report: dict[str, Any], *, episode_index: int | None
) -> None:
    """Validate the closed v4 safety schema and every immutable static field."""

    _require(isinstance(report, dict) and set(report) == SAFETY_FIELDS, "safety schema")
    _require(report.get("episode_index") == episode_index, "safety episode index")
    exact = {
        "profile": "panda_velocity_softlimit_guardband_v2",
        "apply_actions_cadence": "physics_substep",
        "physics_dt": 1.0 / 120.0,
        "control_dt": 1.0 / 15.0,
        "decimation": 8,
        "current_joint_soft_limit_tolerance_rad": 1e-5,
        "target_soft_limit_guard_band_profile": (
            "one_physics_substep_velocity_bound_v1"
        ),
        "eef_quaternion_unit_norm_tolerance": 1e-3,
        "joint_slew_float32_tolerance_rad": 1e-6,
        "soft_joint_pos_limit_factor": 1.0,
        "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
        "target_joint_pos_limits_float32_sha256": TARGET_LIMIT_DIGEST,
        "soft_joint_pos_limits_float32_sha256": SOFT_LIMIT_DIGEST,
    }
    for field, expected in exact.items():
        _require(type(report.get(field)) is type(expected), f"safety {field} type")
        _require(report[field] == expected, f"safety {field} drift")
    for field, expected in (
        ("joint_velocity_limits_rad_s", EXPECTED_VELOCITY_LIMITS_RAD_S),
        ("joint_effort_limits", EXPECTED_EFFORT_LIMITS),
        ("max_delta_joint_pos_rad", EXPECTED_MAX_DELTA_RAD),
        ("target_soft_limit_margin_rad", EXPECTED_MAX_DELTA_RAD),
    ):
        actual = _finite_vector(report.get(field), f"safety {field}", length=7)
        _require(
            all(
                _same_float32(value, wanted)
                for value, wanted in zip(actual, expected, strict=True)
            ),
            f"safety {field} value drift",
        )
    _validate_float_matrix(
        report.get("soft_joint_pos_limits_rad"),
        "safety soft limits",
        EXPECTED_OUTER_LIMITS_RAD,
    )
    _validate_float_matrix(
        report.get("target_joint_pos_limits_rad"),
        "safety target limits",
        EXPECTED_TARGET_LIMITS_RAD,
    )
    counters = report.get("counters")
    maxima = report.get("maxima")
    _require(
        isinstance(counters, dict) and set(counters) == SAFETY_COUNTER_FIELDS,
        "safety counter schema",
    )
    _require(
        all(type(value) is int and value >= 0 for value in counters.values()),
        "safety counter values",
    )
    _require(
        isinstance(maxima, dict) and set(maxima) == SAFETY_MAXIMA_FIELDS,
        "safety maxima schema",
    )
    for field, vector in maxima.items():
        _require(
            all(
                value >= 0.0
                for value in _finite_vector(vector, f"maxima {field}", length=7)
            ),
            f"maxima {field} negative",
        )
    diagnostics = report.get("guard_diagnostics")
    _require(isinstance(diagnostics, list), "safety diagnostics")


def _validate_completed_max_diagnostic(report: dict[str, Any]) -> None:
    diagnostic = report.get("max_raw_delta_diagnostic")
    _require(isinstance(diagnostic, dict), "completed safety max diagnostic")
    _require(set(diagnostic) == DIAGNOSTIC_FIELDS, "max diagnostic schema")
    _require(diagnostic.get("kind") == "max_raw_delta", "max diagnostic kind")
    _require(diagnostic.get("episode_index") == 0, "max diagnostic episode")
    policy_step = diagnostic.get("policy_step")
    substep = diagnostic.get("physics_substep")
    _require(
        type(policy_step) is int and 0 <= policy_step < EXPECTED_TOTAL_POLICY_STEPS,
        "max diagnostic policy step",
    )
    _require(type(substep) is int and 0 <= substep < 8, "max diagnostic substep")
    vectors = {
        field: _validate_diagnostic_vector(
            diagnostic.get(field), f"max diagnostic {field}"
        )
        for field in (
            "joint_pos_rad",
            "raw_delta_joint_pos_rad",
            "raw_joint_pos_target_rad",
            "safe_joint_pos_target_rad",
        )
    }
    _require(diagnostic.get("jacobian_finite") is True, "max diagnostic Jacobian")
    _require(diagnostic.get("eef_quaternion_norm") is None, "max diagnostic quat")
    _require(
        _finite_number(diagnostic.get("pose_error_norm"), "max pose error") >= 0.0,
        "max pose error",
    )
    _require(
        _finite_number(diagnostic.get("jacobian_max_abs"), "max Jacobian") >= 0.0,
        "max Jacobian",
    )
    for index, (q, raw_delta, raw_target, safe_target) in enumerate(
        zip(
            vectors["joint_pos_rad"],
            vectors["raw_delta_joint_pos_rad"],
            vectors["raw_joint_pos_target_rad"],
            vectors["safe_joint_pos_target_rad"],
            strict=True,
        )
    ):
        _require(
            math.isclose(raw_target, q + raw_delta, rel_tol=0.0, abs_tol=1e-6),
            f"max diagnostic joint {index} raw identity",
        )
        _require(
            abs(safe_target - q) <= EXPECTED_MAX_DELTA_RAD[index] + 1e-6,
            f"max diagnostic joint {index} safe slew",
        )
        lower, upper = EXPECTED_TARGET_LIMITS_RAD[index]
        _require(lower <= safe_target <= upper, f"max diagnostic joint {index} target")
    diagnostic_max = max(abs(value) for value in vectors["raw_delta_joint_pos_rad"])
    aggregate_max = max(
        _finite_vector(
            report["maxima"]["raw_delta_joint_pos_rad"],
            "aggregate raw delta maxima",
            length=7,
        )
    )
    _require(
        math.isclose(diagnostic_max, aggregate_max, rel_tol=0.0, abs_tol=1e-6),
        "max diagnostic/aggregate raw delta mismatch",
    )


def validate_boundary_result(
    boundary: dict[str, Any], safety: dict[str, Any]
) -> dict[str, Any]:
    """Host-side fail-closed validation of the complete boundary episode."""

    _require(set(boundary) == BOUNDARY_FIELDS, "boundary schema drift")
    exact_boundary = {
        "target_joint_name": TARGET_JOINT_NAME,
        "target_joint_index": TARGET_JOINT_INDEX,
        "direction": "upper",
        "replay_action_count": EXPECTED_ACTION_ENCODING["action_count"],
        "adaptive_drive_steps": ADAPTIVE_DRIVE_STEPS,
        "total_policy_steps": EXPECTED_TOTAL_POLICY_STEPS,
        "expected_apply_calls": EXPECTED_APPLY_CALLS,
        "outward_delta_scale": OUTWARD_DELTA_SCALE,
        "required_consecutive_dwell_policy_steps": REQUIRED_CONSECUTIVE_DWELL_STEPS,
        "joint_outer_lower_limit_rad": OUTER_LOWER_LIMIT_RAD,
        "joint_outer_upper_limit_rad": OUTER_UPPER_LIMIT_RAD,
        "joint_inner_target_upper_limit_rad": INNER_UPPER_LIMIT_RAD,
        "terminated": False,
        "truncated": False,
        "state_is_finite": True,
    }
    for field, expected in exact_boundary.items():
        _require(type(boundary.get(field)) is type(expected), f"boundary {field} type")
        _require(boundary[field] == expected, f"boundary {field} mismatch")
    observed_min = _finite_number(
        boundary.get("joint_pos_observed_min_rad"), "boundary observed minimum"
    )
    observed_max = _finite_number(
        boundary.get("joint_pos_observed_max_rad"), "boundary observed maximum"
    )
    _require(
        OUTER_LOWER_LIMIT_RAD <= observed_min <= observed_max <= OUTER_UPPER_LIMIT_RAD,
        "boundary observed joint position crossed an outer limit",
    )

    records = boundary.get("dwell_records")
    _require(
        isinstance(records, list) and len(records) == ADAPTIVE_DRIVE_STEPS,
        "boundary dwell record count",
    )
    max_consecutive = 0
    consecutive = 0
    for index, value in enumerate(records):
        _require(isinstance(value, dict), f"dwell record {index} is not an object")
        _require(set(value) == DWELL_RECORD_FIELDS, f"dwell record {index} schema")
        _require(value.get("drive_step") == index, f"dwell record {index} drive step")
        _require(
            value.get("policy_step")
            == EXPECTED_ACTION_ENCODING["action_count"] + index,
            f"dwell record {index} policy step",
        )
        event_delta = value.get("position_limit_events_delta")
        _require(
            type(event_delta) is int and 0 <= event_delta <= DECIMATION,
            f"dwell record {index} event delta",
        )
        q = _finite_number(value.get("joint_pos_rad"), f"dwell record {index} q")
        dq = _finite_number(value.get("joint_vel_rad_s"), f"dwell record {index} dq")
        target = _finite_number(
            value.get("joint_target_rad"), f"dwell record {index} target"
        )
        predicted = _finite_number(
            value.get("predicted_outward_joint_delta_rad"),
            f"dwell record {index} predicted delta",
        )
        _require(abs(dq) <= 2.609999895095825 + 1e-6, f"dwell record {index} dq")
        _require(predicted > 0.0, f"dwell record {index} is not outward")
        arm_q = _finite_vector(
            value.get("arm_joint_pos_rad"), f"dwell record {index} arm q", length=7
        )
        arm_dq = _finite_vector(
            value.get("arm_joint_vel_rad_s"),
            f"dwell record {index} arm dq",
            length=7,
        )
        arm_target = _finite_vector(
            value.get("arm_joint_target_rad"),
            f"dwell record {index} arm target",
            length=7,
        )
        eef_position = _finite_vector(
            value.get("eef_position_m"),
            f"dwell record {index} EEF position",
            length=3,
        )
        eef_quaternion = _finite_vector(
            value.get("eef_quaternion_wxyz"),
            f"dwell record {index} EEF quaternion",
            length=4,
        )
        _require(
            _same_float32(arm_q[TARGET_JOINT_INDEX], q)
            and _same_float32(arm_dq[TARGET_JOINT_INDEX], dq)
            and _same_float32(arm_target[TARGET_JOINT_INDEX], target),
            f"dwell record {index} joint5 scalar/vector mismatch",
        )
        all_arm_in_limits = all(
            lower <= position <= upper
            for position, (lower, upper) in zip(
                arm_q, EXPECTED_OUTER_LIMITS_RAD, strict=True
            )
        )
        _require(
            all(
                abs(velocity) <= limit + 1e-6
                for velocity, limit in zip(
                    arm_dq, EXPECTED_VELOCITY_LIMITS_RAD_S, strict=True
                )
            ),
            f"dwell record {index} arm velocity limit",
        )
        _require(
            all(
                lower <= target_value <= upper
                for target_value, (lower, upper) in zip(
                    arm_target, EXPECTED_TARGET_LIMITS_RAD, strict=True
                )
            ),
            f"dwell record {index} arm target guard limits",
        )
        _require(
            abs(math.sqrt(sum(item * item for item in eef_quaternion)) - 1.0) <= 1e-3,
            f"dwell record {index} EEF quaternion norm",
        )
        _require(
            all(math.isfinite(item) for item in (*eef_position, *eef_quaternion)),
            f"dwell record {index} EEF nonfinite",
        )
        _require(
            type(value.get("target_is_inner_limit")) is bool
            and value["target_is_inner_limit"]
            is _same_float32(target, INNER_UPPER_LIMIT_RAD),
            f"dwell record {index} target flag mismatch",
        )
        within = OUTER_LOWER_LIMIT_RAD <= q <= OUTER_UPPER_LIMIT_RAD
        _require(
            type(value.get("within_outer_limits")) is bool
            and value["within_outer_limits"] is within,
            f"dwell record {index} outer-limit flag mismatch",
        )
        _require(
            value.get("all_arm_joints_within_outer_limits") is all_arm_in_limits,
            f"dwell record {index} all-arm limit flag mismatch",
        )
        _require(all_arm_in_limits, f"dwell record {index} arm q outside limits")
        _require(
            value.get("eef_state_is_finite") is True,
            f"dwell record {index} EEF state flag",
        )
        _require(value.get("state_is_finite") is True, f"dwell record {index} state")
        if _dwell_predicate(value):
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0
    _require(
        all(_dwell_predicate(record) for record in records[-8:]),
        "last eight adaptive steps did not dwell at the joint5 inner upper limit",
    )
    _require(
        boundary.get("observed_max_consecutive_dwell_policy_steps") == max_consecutive,
        "boundary consecutive-dwell count mismatch",
    )
    _require(
        max_consecutive >= REQUIRED_CONSECUTIVE_DWELL_STEPS,
        "boundary did not sustain the required dwell",
    )

    validate_safety_static(safety, episode_index=0)
    _require(safety.get("episode_index") == 0, "boundary safety episode")
    _require(
        safety.get("profile") == "panda_velocity_softlimit_guardband_v2",
        "boundary safety profile",
    )
    _require(
        safety.get("target_soft_limit_guard_band_profile")
        == "one_physics_substep_velocity_bound_v1",
        "boundary target guard profile",
    )
    _require(safety.get("decimation") == DECIMATION, "boundary decimation")
    _require(
        safety.get("target_joint_pos_limits_float32_sha256") == TARGET_LIMIT_DIGEST,
        "boundary target-limit digest",
    )
    target_limits = safety.get("target_joint_pos_limits_rad")
    _require(
        isinstance(target_limits, list) and len(target_limits) == 7,
        "boundary target-limit shape",
    )
    _require(
        _same_float32(
            _finite_vector(
                target_limits[TARGET_JOINT_INDEX], "joint5 target limits", length=2
            )[1],
            INNER_UPPER_LIMIT_RAD,
        ),
        "boundary joint5 inner upper limit",
    )
    counters = safety.get("counters")
    _require(isinstance(counters, dict), "boundary counters")
    _require(
        counters.get("apply_calls") == EXPECTED_APPLY_CALLS, "boundary apply calls"
    )
    _require(
        counters.get("environment_substeps") == EXPECTED_APPLY_CALLS,
        "boundary environment substeps",
    )
    for field in ZERO_COUNTERS:
        _require(counters.get(field) == 0, f"boundary {field} must be zero")
    _require(
        safety.get("guard_diagnostics") == [], "boundary diagnostics must be empty"
    )
    _validate_completed_max_diagnostic(safety)
    for event_field, joint_field in (
        ("slew_limit_events", "slew_limited_joints"),
        ("position_limit_events", "position_limited_joints"),
    ):
        events = counters.get(event_field)
        joints = counters.get(joint_field)
        _require(
            type(events) is int
            and type(joints) is int
            and events <= EXPECTED_APPLY_CALLS
            and events <= joints <= 7 * events,
            f"boundary {event_field}/{joint_field} feasibility",
        )
    position_events = counters.get("position_limit_events")
    position_joints = counters.get("position_limited_joints")
    _require(
        type(position_events) is int
        and position_events >= REQUIRED_CONSECUTIVE_DWELL_STEPS * DECIMATION,
        "boundary has insufficient position-limit events",
    )
    _require(
        type(position_joints) is int and position_joints >= position_events,
        "boundary position-limited joint count",
    )
    _require(
        sum(record["position_limit_events_delta"] for record in records)
        <= position_events,
        "adaptive position-limit deltas exceed aggregate events",
    )

    maxima = safety.get("maxima")
    _require(isinstance(maxima, dict), "boundary maxima")
    raw_outer = _finite_vector(
        maxima.get("raw_target_soft_limit_violation_rad"),
        "boundary raw target violation",
        length=7,
    )
    _require(
        raw_outer[TARGET_JOINT_INDEX] > 0.0,
        "boundary never drove joint5 raw target beyond the outer upper limit",
    )
    for field in (
        "post_clamp_target_soft_limit_violation_rad",
        "post_clamp_target_guard_band_violation_rad",
        "current_joint_soft_limit_violation_rad",
    ):
        vector = _finite_vector(maxima.get(field), f"boundary maxima {field}", length=7)
        _require(all(item == 0.0 for item in vector), f"boundary {field} is nonzero")
    applied = _finite_vector(
        maxima.get("applied_delta_joint_pos_rad"),
        "boundary applied delta",
        length=7,
    )
    _require(
        all(
            actual <= bound + 1e-6
            for actual, bound in zip(applied, EXPECTED_MAX_DELTA_RAD, strict=True)
        ),
        "boundary exceeded a physics-substep slew bound",
    )
    return {
        "apply_calls": EXPECTED_APPLY_CALLS,
        "position_limit_events": position_events,
        "max_consecutive_dwell_policy_steps": max_consecutive,
        "joint5_raw_outer_violation_rad": raw_outer[TARGET_JOINT_INDEX],
        "joint5_observed_max_rad": observed_max,
    }


def validate_success_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the strict raw success object before immutable publication."""

    _require(set(payload) == SUCCESS_PAYLOAD_FIELDS, "success payload schema drift")
    exact = {
        "schema_version": 1,
        "fixture_profile": FIXTURE_PROFILE,
        "smoke_profile": SMOKE_PROFILE,
        "finalized": False,
        "passed": True,
        "stage": "simulation_app_close_pending",
        "exit_code": 0,
        "environment": ENVIRONMENT,
        "failure": None,
        "close_failures": [],
    }
    for field, expected in exact.items():
        _require(type(payload.get(field)) is type(expected), f"payload {field} type")
        _require(payload[field] == expected, f"payload {field}")
    fixture = payload.get("fixture")
    _require(isinstance(fixture, dict), "payload fixture identity")
    _require(
        set(fixture)
        == {
            "path",
            "size_bytes",
            "sha256",
            "mode",
            "fixture_profile",
            "source_trace_sha256",
            "action_float32_sha256",
            "action_count",
        },
        "payload fixture identity schema",
    )
    _require(fixture.get("size_bytes") == EXPECTED_FIXTURE_SIZE_BYTES, "fixture size")
    _require(fixture.get("sha256") == EXPECTED_FIXTURE_SHA256, "fixture SHA")
    _require(fixture.get("action_count") == 378, "fixture action count")
    assets = payload.get("assets")
    _require(isinstance(assets, dict), "payload assets")
    _require(
        set(assets)
        == {
            "scene",
            "initial_conditions",
            "polaris_hub_revision",
            "revision_metadata",
            "initial_condition_index",
        },
        "payload asset schema",
    )
    _require(
        assets.get("polaris_hub_revision")
        == EXPECTED_ASSET_CONTRACT["polaris_hub_revision"],
        "payload Hub revision",
    )
    for field, expected in (
        ("scene", EXPECTED_ASSET_CONTRACT["scene_sha256"]),
        (
            "initial_conditions",
            EXPECTED_ASSET_CONTRACT["initial_conditions_sha256"],
        ),
    ):
        identity = assets.get(field)
        _require(isinstance(identity, dict), f"payload asset {field}")
        _require(
            set(identity) == {"path", "size_bytes", "sha256", "mode"},
            f"payload asset {field} schema",
        )
        _require(identity.get("sha256") == expected, f"payload asset {field} SHA")
    metadata = assets.get("revision_metadata")
    _require(isinstance(metadata, dict), "payload asset revision metadata")
    for filename in ("initial_conditions.json", "scene.usda"):
        identity = metadata.get(filename)
        _require(isinstance(identity, dict), f"payload metadata {filename}")
        _require(
            set(identity) == {"path", "size_bytes", "sha256", "mode", "revision"},
            f"payload metadata {filename} schema",
        )
        _require(
            identity.get("revision") == EXPECTED_ASSET_CONTRACT["polaris_hub_revision"],
            f"payload metadata revision {filename}",
        )
    _require(assets.get("initial_condition_index") == 0, "payload IC index")
    protocol = payload.get("runtime_protocol")
    _require(isinstance(protocol, dict), "payload runtime protocol")
    _require(
        set(protocol)
        == {
            "episode_steps",
            "policy_hz",
            "step_dt",
            "physics_hz",
            "physics_dt",
            "decimation",
        },
        "payload runtime protocol schema",
    )
    _require(
        protocol
        == {
            "episode_steps": 450,
            "policy_hz": 15.0,
            "step_dt": 1.0 / 15.0,
            "physics_hz": 120.0,
            "physics_dt": 1.0 / 120.0,
            "decimation": 8,
        },
        "payload runtime protocol drift",
    )
    runtime_frame = payload.get("runtime_frame")
    _require(isinstance(runtime_frame, dict), "payload runtime frame")
    _require(set(runtime_frame) == RUNTIME_FRAME_FIELDS, "payload runtime frame schema")
    for field, expected in (
        ("eef_frame", "panda_link8"),
        ("reference_frame", "panda_link0"),
        ("controlled_body", "panda_link8"),
        ("body_offset", "identity"),
        ("command_type", "pose"),
        ("use_relative_mode", False),
        ("ik_method", "dls"),
        ("dls_damping", 0.01),
        ("arm_scale", 1.0),
        ("action_dim", 7),
        ("ik_safety_profile", "panda_velocity_softlimit_guardband_v2"),
    ):
        _require(runtime_frame.get(field) == expected, f"payload frame {field}")
    _require(
        _finite_number(runtime_frame.get("position_error_m"), "frame position error")
        <= 1e-5,
        "payload frame position mismatch",
    )
    _require(
        _finite_number(runtime_frame.get("rotation_error_rad"), "frame rotation error")
        <= math.radians(0.01),
        "payload frame rotation mismatch",
    )
    _require(
        runtime_frame.get("arm_joint_names")
        == [f"panda_joint{index}" for index in range(1, 8)],
        "payload frame arm joint order",
    )
    _require(
        runtime_frame.get("gripper_threshold_profile")
        == "closed_positive_ge_0p5_inverse_open_gt_0p5_v1",
        "payload frame gripper semantics",
    )
    initial = payload.get("initial_ik_safety_capture")
    _require(isinstance(initial, dict), "payload initial safety capture")
    validate_safety_static(initial, episode_index=None)
    _require(
        all(value == 0 for value in initial["counters"].values()),
        "initial safety counters are not zero",
    )
    _require(
        all(value == 0.0 for vector in initial["maxima"].values() for value in vector),
        "initial safety maxima are not zero",
    )
    _require(initial.get("guard_diagnostics") == [], "initial diagnostics")
    _require(initial.get("max_raw_delta_diagnostic") is None, "initial max diagnostic")
    return validate_boundary_result(payload["boundary"], payload["ik_safety"])


def _strict_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _atomic_write_immutable(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Non-overwriting, fsynced, mode-0444 publication of one strict raw JSON."""

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
        published_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(published_fd)
        finally:
            os.close(published_fd)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    reread = path.read_bytes()
    _require(reread == serialized, "published raw JSON changed on reread")
    _require(stat.S_IMODE(path.stat().st_mode) == 0o444, "published raw mode")
    return {
        "path": str(path),
        "size_bytes": len(reread),
        "sha256": _sha256_bytes(reread),
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


def _finite_vector_evidence(values: Any) -> dict[str, Any]:
    """Serialize a tensor-like vector without ever emitting NaN/Infinity."""

    flat_values = values.detach().cpu().flatten().tolist()
    finite_mask = [math.isfinite(float(value)) for value in flat_values]
    return {
        "values": [
            float(value) if finite else None
            for value, finite in zip(flat_values, finite_mask, strict=True)
        ],
        "finite_mask": finite_mask,
        "finite_count": sum(finite_mask),
    }


def _capture_failure_runtime_evidence(env: Any, *, policy_step: Any) -> dict[str, Any]:
    """Capture the live arm state and controller report before failure teardown."""

    from polaris.eef_runtime_contract import eef_episode_safety_report

    arm_term = env.unwrapped.action_manager._terms["arm"]
    robot = env.unwrapped.scene["robot"]
    joint_ids = arm_term._joint_ids
    return {
        "policy_step": policy_step,
        "arm_joint_names": list(arm_term._joint_names),
        "arm_joint_pos_rad": _finite_vector_evidence(
            robot.data.joint_pos[:, joint_ids][0]
        ),
        "arm_joint_vel_rad_s": _finite_vector_evidence(
            robot.data.joint_vel[:, joint_ids][0]
        ),
        "arm_joint_target_rad": _finite_vector_evidence(
            robot.data.joint_pos_target[:, joint_ids][0]
        ),
        "ik_safety": eef_episode_safety_report(env, 0),
    }


def _run_boundary_replay(
    args_cli: argparse.Namespace, state: dict[str, Any]
) -> dict[str, Any]:
    """Run the live Isaac episode.  Heavy imports remain isolated here."""

    import gymnasium as gym
    import torch
    from isaaclab.utils.math import apply_delta_pose
    from isaaclab_tasks.utils import parse_env_cfg

    import polaris.environments  # noqa: F401
    from polaris.eef_runtime_contract import begin_eef_safety_episode
    from polaris.eef_runtime_contract import eef_episode_safety_report
    from polaris.eef_runtime_contract import validate_eef_runtime_frame
    from polaris.eef_runtime_contract import validate_eef_runtime_safety
    from polaris.eef_runtime_contract import validate_ego_lap_runtime_protocol
    from polaris.environments.droid_cfg import EefPoseActionCfg
    from polaris.environments.robot_cfg import configure_eef_pose_joint_safety
    from polaris.utils import load_eval_initial_conditions

    state["stage"] = "load_fixture"
    fixture_identity, actions = load_replay_fixture()

    state["stage"] = "build_environment"
    env_cfg = parse_env_cfg(
        ENVIRONMENT,
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )
    env_cfg.actions = EefPoseActionCfg()
    configure_eef_pose_joint_safety(env_cfg.scene.robot)
    env = gym.make(ENVIRONMENT, cfg=env_cfg)
    state["env"] = env
    runtime_protocol = validate_ego_lap_runtime_protocol(env)

    state["stage"] = "validate_assets"
    assets = validate_asset_contract(Path(env.unwrapped.usd_file))
    _, initial_conditions = load_eval_initial_conditions(
        usd=env.unwrapped.usd_file,
        rollouts=1,
    )
    _require(
        isinstance(initial_conditions, list)
        and len(initial_conditions) == 1
        and isinstance(initial_conditions[0], dict),
        "production initial-condition loader did not return FoodBussing IC0",
    )

    state["stage"] = "capture_initial_safety"
    initial_safety = validate_eef_runtime_safety(env)
    state["stage"] = "reset_exact_initial_condition"
    observation, _ = env.reset(
        object_positions=initial_conditions[0],
        expensive=False,
    )
    runtime_frame = validate_eef_runtime_frame(env, observation)
    begin_eef_safety_episode(env, 0)

    arm_term = env.unwrapped.action_manager._terms["arm"]
    _require(
        list(arm_term._joint_names)[TARGET_JOINT_INDEX] == TARGET_JOINT_NAME,
        "live arm joint ordering drift",
    )
    robot = env.unwrapped.scene["robot"]
    q_min = torch.full((), float("inf"), device=env.device)
    q_max = torch.full((), float("-inf"), device=env.device)
    replay_terminated = False
    replay_truncated = False

    def update_joint_extrema() -> None:
        nonlocal q_min, q_max
        q = robot.data.joint_pos[:, arm_term._joint_ids][0, TARGET_JOINT_INDEX]
        q_min = torch.minimum(q_min, q)
        q_max = torch.maximum(q_max, q)

    state["stage"] = "replay_official_v3_actions"
    for step, action_values in enumerate(actions):
        state["policy_step"] = step
        action = torch.tensor(
            action_values,
            dtype=torch.float32,
            device=env.device,
        ).reshape(1, -1)
        observation, _, terminated, truncated, _ = env.step(
            action,
            expensive=False,
        )
        update_joint_extrema()
        replay_terminated = replay_terminated or bool(terminated[0])
        replay_truncated = replay_truncated or bool(truncated[0])
        _require(not replay_terminated, f"official replay terminated at step {step}")
        _require(not replay_truncated, f"official replay truncated at step {step}")

    state["stage"] = "adaptive_joint5_upper_dwell"
    report_before = eef_episode_safety_report(env, 0)
    previous_position_events = report_before["counters"]["position_limit_events"]
    dwell_records: list[dict[str, Any]] = []
    last_gripper = actions[-1][7]
    damping = 0.01
    for drive_step in range(ADAPTIVE_DRIVE_STEPS):
        policy_step = len(actions) + drive_step
        state["policy_step"] = policy_step
        current_position = observation["policy"]["eef_pos"]
        current_quaternion = observation["policy"]["eef_quat"]
        jacobian = arm_term._compute_frame_jacobian().clone()
        _require(
            tuple(jacobian.shape) == (1, 6, 7),
            f"adaptive Jacobian shape drift: {tuple(jacobian.shape)!r}",
        )
        _require(
            bool(torch.isfinite(jacobian).all().detach().cpu().item()),
            "adaptive Jacobian is non-finite",
        )
        delta_pose = OUTWARD_DELTA_SCALE * jacobian[:, :, TARGET_JOINT_INDEX]
        target_position, target_quaternion = apply_delta_pose(
            current_position,
            current_quaternion,
            delta_pose,
        )

        jacobian64 = jacobian.to(torch.float64)
        delta64 = delta_pose.to(torch.float64)
        normal = jacobian64 @ jacobian64.transpose(1, 2)
        normal += (damping**2) * torch.eye(
            6,
            dtype=torch.float64,
            device=env.device,
        ).unsqueeze(0)
        predicted_joint_delta = (
            jacobian64.transpose(1, 2)
            @ torch.linalg.solve(normal, delta64.unsqueeze(-1))
        ).squeeze(-1)
        predicted_outward = float(
            predicted_joint_delta[0, TARGET_JOINT_INDEX].detach().cpu().item()
        )
        _require(predicted_outward > 0.0, "adaptive Jacobian command is not outward")

        gripper = torch.tensor([[last_gripper]], dtype=torch.float32, device=env.device)
        action = torch.cat((target_position, target_quaternion, gripper), dim=-1)
        observation, _, terminated, truncated, _ = env.step(
            action,
            expensive=False,
        )
        update_joint_extrema()
        replay_terminated = replay_terminated or bool(terminated[0])
        replay_truncated = replay_truncated or bool(truncated[0])
        _require(
            not replay_terminated, f"adaptive drive terminated at step {drive_step}"
        )
        _require(not replay_truncated, f"adaptive drive truncated at step {drive_step}")

        report_after = eef_episode_safety_report(env, 0)
        position_events = report_after["counters"]["position_limit_events"]
        event_delta = position_events - previous_position_events
        previous_position_events = position_events
        arm_joint_pos = robot.data.joint_pos[:, arm_term._joint_ids][0]
        arm_joint_vel = robot.data.joint_vel[:, arm_term._joint_ids][0]
        arm_joint_target = robot.data.joint_pos_target[:, arm_term._joint_ids][0]
        joint_pos = arm_joint_pos[TARGET_JOINT_INDEX]
        joint_vel = arm_joint_vel[TARGET_JOINT_INDEX]
        joint_target = arm_joint_target[TARGET_JOINT_INDEX]
        q_value, dq_value, target_value = (
            float(value.detach().cpu().item())
            for value in (joint_pos, joint_vel, joint_target)
        )
        arm_q_values = [float(value) for value in arm_joint_pos.detach().cpu().tolist()]
        arm_dq_values = [
            float(value) for value in arm_joint_vel.detach().cpu().tolist()
        ]
        arm_target_values = [
            float(value) for value in arm_joint_target.detach().cpu().tolist()
        ]
        eef_position_values = [
            float(value)
            for value in observation["policy"]["eef_pos"][0].detach().cpu().tolist()
        ]
        eef_quaternion_values = [
            float(value)
            for value in observation["policy"]["eef_quat"][0].detach().cpu().tolist()
        ]
        all_arm_finite = all(
            math.isfinite(value)
            for value in (*arm_q_values, *arm_dq_values, *arm_target_values)
        )
        eef_state_finite = all(
            math.isfinite(value)
            for value in (*eef_position_values, *eef_quaternion_values)
        )
        all_arm_in_limits = all(
            lower <= position <= upper
            for position, (lower, upper) in zip(
                arm_q_values, EXPECTED_OUTER_LIMITS_RAD, strict=True
            )
        )
        state_finite = (
            all(
                math.isfinite(value)
                for value in (q_value, dq_value, target_value, predicted_outward)
            )
            and all_arm_finite
            and eef_state_finite
        )
        dwell_records.append(
            {
                "drive_step": drive_step,
                "policy_step": policy_step,
                "position_limit_events_delta": event_delta,
                "joint_pos_rad": q_value,
                "joint_vel_rad_s": dq_value,
                "joint_target_rad": target_value,
                "predicted_outward_joint_delta_rad": predicted_outward,
                "arm_joint_pos_rad": arm_q_values,
                "arm_joint_vel_rad_s": arm_dq_values,
                "arm_joint_target_rad": arm_target_values,
                "eef_position_m": eef_position_values,
                "eef_quaternion_wxyz": eef_quaternion_values,
                "target_is_inner_limit": _same_float32(
                    target_value, INNER_UPPER_LIMIT_RAD
                ),
                "within_outer_limits": (
                    OUTER_LOWER_LIMIT_RAD <= q_value <= OUTER_UPPER_LIMIT_RAD
                ),
                "all_arm_joints_within_outer_limits": all_arm_in_limits,
                "eef_state_is_finite": eef_state_finite,
                "state_is_finite": state_finite,
            }
        )

    state["stage"] = "validate_boundary_evidence"
    safety = validate_eef_runtime_safety(env)
    episode_safety = eef_episode_safety_report(env, 0)
    _require(safety == episode_safety, "live safety report changed during validation")
    q_min_value = float(q_min.detach().cpu().item())
    q_max_value = float(q_max.detach().cpu().item())
    max_consecutive = 0
    consecutive = 0
    for record in dwell_records:
        if _dwell_predicate(record):
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0
    boundary = {
        "target_joint_name": TARGET_JOINT_NAME,
        "target_joint_index": TARGET_JOINT_INDEX,
        "direction": "upper",
        "replay_action_count": len(actions),
        "adaptive_drive_steps": ADAPTIVE_DRIVE_STEPS,
        "total_policy_steps": len(actions) + ADAPTIVE_DRIVE_STEPS,
        "expected_apply_calls": EXPECTED_APPLY_CALLS,
        "outward_delta_scale": OUTWARD_DELTA_SCALE,
        "required_consecutive_dwell_policy_steps": REQUIRED_CONSECUTIVE_DWELL_STEPS,
        "observed_max_consecutive_dwell_policy_steps": max_consecutive,
        "joint_outer_lower_limit_rad": OUTER_LOWER_LIMIT_RAD,
        "joint_outer_upper_limit_rad": OUTER_UPPER_LIMIT_RAD,
        "joint_inner_target_upper_limit_rad": INNER_UPPER_LIMIT_RAD,
        "joint_pos_observed_min_rad": q_min_value,
        "joint_pos_observed_max_rad": q_max_value,
        "terminated": replay_terminated,
        "truncated": replay_truncated,
        "state_is_finite": (
            math.isfinite(q_min_value)
            and math.isfinite(q_max_value)
            and all(record["state_is_finite"] for record in dwell_records)
        ),
        "dwell_records": dwell_records,
    }
    validate_boundary_result(boundary, safety)
    state["policy_step"] = None
    return {
        "fixture": fixture_identity,
        "assets": assets,
        "runtime_protocol": runtime_protocol,
        "runtime_frame": runtime_frame,
        "initial_ik_safety_capture": initial_safety,
        "boundary": boundary,
        "ik_safety": safety,
    }


def _parse_args() -> tuple[argparse.Namespace, Any]:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Required non-overwriting immutable raw result path.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.enable_cameras = True
    args_cli.headless = True
    return args_cli, AppLauncher


def main() -> int:
    args_cli, app_launcher_type = _parse_args()
    state: dict[str, Any] = {
        "stage": "launch_simulation_app",
        "policy_step": None,
        "env": None,
    }
    simulation_app = None
    failure: BaseException | None = None
    close_failures: list[dict[str, str]] = []
    evidence: dict[str, Any] | None = None
    failure_runtime_evidence: dict[str, Any] | None = None
    failure_runtime_evidence_error: dict[str, str] | None = None
    try:
        app_launcher = app_launcher_type(args_cli)
        simulation_app = app_launcher.app
        evidence = _run_boundary_replay(args_cli, state)
    except BaseException as error:
        failure = error
        traceback.print_exception(
            type(error), error, error.__traceback__, file=sys.stderr
        )
        sys.stdout.flush()
        sys.stderr.flush()

    env = state.get("env")
    if env is not None and failure is not None:
        state["stage"] = "capture_failure_runtime_evidence"
        try:
            failure_runtime_evidence = _capture_failure_runtime_evidence(
                env,
                policy_step=state.get("policy_step"),
            )
        except BaseException as error:
            failure_runtime_evidence_error = _exception_evidence(error)
            traceback.print_exception(
                type(error), error, error.__traceback__, file=sys.stderr
            )
    if env is not None:
        state["stage"] = "close_environment"
        try:
            env.close()
        except BaseException as error:
            close_failures.append(_exception_evidence(error))
            if failure is None:
                failure = error
            traceback.print_exception(
                type(error), error, error.__traceback__, file=sys.stderr
            )

    if failure is None and evidence is not None and not close_failures:
        state["stage"] = "simulation_app_close_pending"
        payload = {
            "schema_version": 1,
            "fixture_profile": FIXTURE_PROFILE,
            "smoke_profile": SMOKE_PROFILE,
            "finalized": False,
            "passed": True,
            "stage": state["stage"],
            "exit_code": 0,
            "environment": ENVIRONMENT,
            **evidence,
            "failure": None,
            "close_failures": [],
        }
        try:
            validate_success_payload(payload)
            identity = _atomic_write_immutable(args_cli.output_json, payload)
            marker_path = args_cli.output_json.resolve().with_name(
                args_cli.output_json.name + ".ready.json"
            )
            marker_payload = {
                "schema_version": 1,
                "stage": "simulation_app_close_pending",
                "raw_result": identity,
            }
            expected_marker_sha256 = _sha256_bytes(_strict_json_bytes(marker_payload))
            print(
                "POLARIS_BOUNDARY_REPLAY_RAW="
                f"{identity['path']};size={identity['size_bytes']};"
                f"sha256={identity['sha256']};mode={identity['mode']}",
                flush=True,
            )
            print(f"POLARIS_BOUNDARY_REPLAY_READY_MARKER={marker_path}", flush=True)
            print(
                "POLARIS_BOUNDARY_REPLAY_READY_MARKER_EXPECTED_SHA256="
                f"{expected_marker_sha256}",
                flush=True,
            )
            sys.stdout.flush()
            sys.stderr.flush()
            _atomic_write_immutable(marker_path, marker_payload)
            simulation_app.close()
            return 0
        except BaseException as error:
            failure = error
            traceback.print_exception(
                type(error), error, error.__traceback__, file=sys.stderr
            )
            sys.stdout.flush()
            sys.stderr.flush()

    state["stage"] = "failed"
    failure_payload = {
        "schema_version": 1,
        "fixture_profile": FIXTURE_PROFILE,
        "smoke_profile": SMOKE_PROFILE,
        "finalized": False,
        "passed": False,
        "stage": state["stage"],
        "policy_step": state.get("policy_step"),
        "exit_code": 1,
        "environment": ENVIRONMENT,
        "failure": _exception_evidence(failure) if failure is not None else None,
        "failure_runtime_evidence": failure_runtime_evidence,
        "failure_runtime_evidence_error": failure_runtime_evidence_error,
        "close_failures": close_failures,
    }
    try:
        identity = _atomic_write_immutable(args_cli.output_json, failure_payload)
        print(
            "POLARIS_BOUNDARY_REPLAY_FAILURE_RAW="
            f"{identity['path']};sha256={identity['sha256']}",
            flush=True,
        )
    except BaseException as persistence_error:
        traceback.print_exception(
            type(persistence_error),
            persistence_error,
            persistence_error.__traceback__,
            file=sys.stderr,
        )
    if simulation_app is not None:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(1)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
