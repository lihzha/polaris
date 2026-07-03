"""Plan, validate, and publish the native all-six coupled controller smoke."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any

import numpy as np

from polaris.joint_velocity_runtime import validate_joint_velocity_runtime_report
from polaris.native_gripper_runtime import (
    EXPECTED_FULL_LIMITS_CAPPED,
    NATIVE_GRIPPER_ALL_SIX_PROFILE,
    PHYSX_VELOCITY_LIMIT_ABSOLUTE_TOLERANCE_RAD_S,
    validate_native_all_joint_dynamic_report,
    validate_native_gripper_mimic_contract,
    validate_native_gripper_reset_report,
)
from polaris.pi05_droid_jointvelocity_contract import (
    NATIVE_GRIPPER_DRIVE_PROFILE,
    PI05_DROID_JOINTVELOCITY_PROFILE,
)


SMOKE_PROFILE = "pi05_droid_native_all_six_coupled_controller_smoke_v1"
SCENARIO_STEPS = 12
DELAYED_TRANSITION_STEP = 8
SETTLE_STEPS = 5
PRECONDITION_STEPS = 5
MOVING_ARM_ACTION = (0.20, -0.15, 0.10, -0.20, 0.15, -0.10, 0.20)
MIN_ARM_MEASURED_VELOCITY_RAD_S = 1e-4
MIN_GRIPPER_TRANSITION_RAD = 1e-3
OPEN_PRECONDITION_MAX_DRIVER_RAD = 0.05
CLOSED_PRECONDITION_MIN_DRIVER_RAD = 0.05
MIMIC_COUPLING_RELATIVE_TOLERANCE = 0.25
# The PhysX mimic convention produces follower motion opposite the authored
# gearing sign.  This closed vector was also observed in the existing L40S
# driver transition capture: driver+, outer+, inner-left-, inner-right+,
# knuckle-left-, knuckle-right- for a close transition.
GRIPPER_CLOSE_MOTION_DIRECTIONS = (1.0, 1.0, -1.0, 1.0, -1.0, -1.0)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def coupled_scenario_plans() -> list[dict[str, Any]]:
    scenarios = (
        ("immediate_close", "open", lambda step: 1.0),
        (
            "delayed_close",
            "open",
            lambda step: 0.0 if step < DELAYED_TRANSITION_STEP else 1.0,
        ),
        ("immediate_open", "closed", lambda step: 0.0),
        (
            "delayed_open",
            "closed",
            lambda step: 1.0 if step < DELAYED_TRANSITION_STEP else 0.0,
        ),
    )
    result = []
    for label, precondition, gripper_value in scenarios:
        actions = []
        for step in range(SCENARIO_STEPS):
            sign = 1.0 if step % 2 == 0 else -1.0
            arm = [float(np.float32(sign * value)) for value in MOVING_ARM_ACTION]
            actions.append([*arm, gripper_value(step)])
        result.append(
            {
                "label": label,
                "precondition": precondition,
                "actions": actions,
            }
        )
    return result


def _finite_vector(value: Any, width: int, *, field: str) -> list[float]:
    _require(
        isinstance(value, list)
        and len(value) == width
        and all(
            type(item) in (int, float)
            and not isinstance(item, bool)
            and math.isfinite(item)
            for item in value
        ),
        f"{field} must contain {width} finite numbers",
    )
    return [float(item) for item in value]


def validate_native_all_six_smoke(
    value: Any, *, require_parent_completion: bool
) -> dict[str, Any]:
    required = {
        "schema_version",
        "profile",
        "controller_profile",
        "gripper_profile",
        "environment",
        "runtime_contract",
        "mimic_joint_contract",
        "scenario_plans",
        "scenarios",
        "lifecycle",
    }
    if require_parent_completion:
        required.add("completion")
    _require(
        isinstance(value, dict) and set(value) == required, "all-six smoke schema drift"
    )
    exact = {
        "schema_version": 1,
        "profile": SMOKE_PROFILE,
        "controller_profile": PI05_DROID_JOINTVELOCITY_PROFILE,
        "gripper_profile": NATIVE_GRIPPER_ALL_SIX_PROFILE,
        "environment": "DROID-FoodBussing",
    }
    for field, expected in exact.items():
        _require(
            value[field] == expected and type(value[field]) is type(expected),
            f"all-six smoke {field} drift",
        )
    runtime = validate_joint_velocity_runtime_report(value["runtime_contract"])
    _require(
        runtime["gripper"]["drive"]["profile"] == NATIVE_GRIPPER_DRIVE_PROFILE,
        "all-six smoke native drive profile drift",
    )
    validate_native_gripper_mimic_contract(value["mimic_joint_contract"])
    plans = coupled_scenario_plans()
    _require(value["scenario_plans"] == plans, "all-six smoke scenario plan drift")
    scenarios = value["scenarios"]
    _require(
        isinstance(scenarios, list) and len(scenarios) == len(plans),
        "all-six smoke scenario count drift",
    )
    previous_reset_count = runtime["all_six_gripper"]["reset_write"]["reset_count"]
    for scenario_index, (plan, scenario) in enumerate(
        zip(plans, scenarios, strict=True)
    ):
        _require(
            isinstance(scenario, dict)
            and set(scenario)
            == {
                "label",
                "precondition",
                "actions",
                "terminated",
                "truncated",
                "reset_write",
                "dynamic",
            },
            f"all-six scenario {scenario_index} schema drift",
        )
        for field in ("label", "precondition", "actions"):
            _require(
                scenario[field] == plan[field],
                f"all-six scenario {scenario_index} {field} drift",
            )
        _require(
            scenario["terminated"] == [False] * SCENARIO_STEPS
            and scenario["truncated"] == [False] * SCENARIO_STEPS,
            f"all-six scenario {scenario_index} ended early",
        )
        reset_write = validate_native_gripper_reset_report(scenario["reset_write"])
        _require(
            reset_write["reset_count"] == previous_reset_count + 1,
            f"all-six scenario {scenario_index} reset/write count drift",
        )
        previous_reset_count = reset_write["reset_count"]
        dynamic = validate_native_all_joint_dynamic_report(
            scenario["dynamic"], require_samples=True
        )
        _require(
            dynamic["apply_calls"] == SCENARIO_STEPS * 8
            and dynamic["post_policy_step_samples"] == SCENARIO_STEPS
            and dynamic["sample_count"] == SCENARIO_STEPS * 9,
            f"all-six scenario {scenario_index} cadence drift",
        )
        maxima_velocity = np.zeros(13, dtype=np.float64)
        maxima_acceleration = np.zeros(13, dtype=np.float64)
        for step, action in enumerate(plan["actions"]):
            samples = dynamic["samples"][step * 9 : (step + 1) * 9]
            _require(
                [sample["kind"] for sample in samples]
                == ["apply_entry"] * 8 + ["post_policy_step"],
                f"all-six scenario {scenario_index} step {step} sample kinds drift",
            )
            _require(
                [sample["physics_substep_index"] for sample in samples]
                == list(range(8)) + [8],
                f"all-six scenario {scenario_index} step {step} substeps drift",
            )
            for sample in samples:
                _require(
                    sample["policy_step_index"] == step,
                    f"all-six scenario {scenario_index} policy index drift",
                )
                velocity = np.asarray(
                    _finite_vector(
                        sample["joint_velocity"],
                        13,
                        field="all-six measured velocity",
                    ),
                    dtype=np.float64,
                )
                acceleration = np.asarray(
                    _finite_vector(
                        sample["joint_acceleration"],
                        13,
                        field="all-six measured acceleration",
                    ),
                    dtype=np.float64,
                )
                maxima_velocity = np.maximum(maxima_velocity, np.abs(velocity))
                maxima_acceleration = np.maximum(
                    maxima_acceleration, np.abs(acceleration)
                )
                target = _finite_vector(
                    sample["joint_velocity_target"],
                    13,
                    field="all-six velocity target",
                )
                _require(
                    np.array_equal(
                        np.asarray(target[:7], dtype=np.float32),
                        np.asarray(action[:7], dtype=np.float32),
                    ),
                    f"all-six scenario {scenario_index} arm target drift",
                )
                position_target = _finite_vector(
                    sample["joint_position_target"],
                    13,
                    field="all-six position target",
                )
                expected_driver = np.float32(np.pi / 4.0 if action[7] > 0.5 else 0.0)
                _require(
                    np.float32(position_target[7]) == expected_driver,
                    f"all-six scenario {scenario_index} driver target drift",
                )
        _require(
            np.array_equal(
                np.asarray(dynamic["max_abs_joint_velocity_rad_s"]), maxima_velocity
            )
            and np.array_equal(
                np.asarray(dynamic["max_abs_joint_acceleration_rad_s2"]),
                maxima_acceleration,
            ),
            f"all-six scenario {scenario_index} maxima drift",
        )
        _require(
            all(
                measured <= limit + PHYSX_VELOCITY_LIMIT_ABSOLUTE_TOLERANCE_RAD_S
                for measured, limit in zip(
                    maxima_velocity, EXPECTED_FULL_LIMITS_CAPPED, strict=True
                )
            ),
            f"all-six scenario {scenario_index} exceeded a live velocity limit",
        )
        _require(
            np.all(maxima_velocity[:7] > MIN_ARM_MEASURED_VELOCITY_RAD_S),
            f"all-six scenario {scenario_index} did not move every arm joint",
        )
        transition_step = (
            0 if plan["label"].startswith("immediate_") else DELAYED_TRANSITION_STEP
        )
        transition_sample = dynamic["samples"][transition_step * 9]
        final_sample = dynamic["samples"][-1]
        transition_position = np.asarray(
            _finite_vector(
                transition_sample["joint_position"],
                13,
                field="all-six transition position",
            ),
            dtype=np.float64,
        )
        final_position = np.asarray(
            _finite_vector(
                final_sample["joint_position"],
                13,
                field="all-six final position",
            ),
            dtype=np.float64,
        )
        driver_at_transition = transition_position[7]
        if plan["precondition"] == "open":
            _require(
                abs(driver_at_transition) <= OPEN_PRECONDITION_MAX_DRIVER_RAD,
                f"all-six scenario {scenario_index} open precondition drift",
            )
            transition_direction = 1.0
            expected_driver_endpoint = float(np.float32(np.pi / 4.0))
        else:
            _require(
                driver_at_transition >= CLOSED_PRECONDITION_MIN_DRIVER_RAD,
                f"all-six scenario {scenario_index} closed precondition drift",
            )
            transition_direction = -1.0
            expected_driver_endpoint = 0.0
        gripper_delta = final_position[7:13] - transition_position[7:13]
        expected_directions = transition_direction * np.asarray(
            GRIPPER_CLOSE_MOTION_DIRECTIONS, dtype=np.float64
        )
        _require(
            np.all(gripper_delta * expected_directions > MIN_GRIPPER_TRANSITION_RAD),
            f"all-six scenario {scenario_index} gripper motion/coupling direction drift",
        )
        _require(
            abs(final_position[7] - expected_driver_endpoint)
            < abs(driver_at_transition - expected_driver_endpoint),
            f"all-six scenario {scenario_index} driver did not approach endpoint",
        )
        driver_magnitude = abs(gripper_delta[0])
        coupling_ratios = np.abs(gripper_delta[1:]) / driver_magnitude
        _require(
            np.all(np.abs(coupling_ratios - 1.0) <= MIMIC_COUPLING_RELATIVE_TOLERANCE),
            f"all-six scenario {scenario_index} mimic coupling ratio drift",
        )

    lifecycle = value["lifecycle"]
    if require_parent_completion:
        _require(
            lifecycle
            == {
                "env_close": "complete",
                "simulation_app_close": "invoked_then_child_exited_zero",
                "publication": "stdlib_parent_after_child_exit",
            },
            "all-six parent lifecycle drift",
        )
        completion = value["completion"]
        _require(
            isinstance(completion, dict)
            and set(completion)
            == {
                "child_exit_code",
                "raw_path",
                "raw_sha256",
                "raw_size",
                "ready_path",
                "ready_sha256",
                "ready_size",
            }
            and completion["child_exit_code"] == 0
            and all(
                isinstance(completion[field], str) and completion[field]
                for field in ("raw_path", "ready_path")
            )
            and all(
                isinstance(completion[field], str)
                and len(completion[field]) == 64
                and all(
                    character in "0123456789abcdef" for character in completion[field]
                )
                for field in ("raw_sha256", "ready_sha256")
            )
            and all(
                type(completion[field]) is int and completion[field] > 0
                for field in ("raw_size", "ready_size")
            ),
            "all-six parent completion drift",
        )
    else:
        _require(
            lifecycle
            == {
                "env_close": "complete",
                "simulation_app_close": "pending_immediate_invocation",
                "publication": "kit_child_before_simulation_app_close",
            },
            "all-six child lifecycle drift",
        )
    return copy.deepcopy(value)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _read_immutable_json_artifact(
    path: Path, *, field: str
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{field} is not a readable immutable artifact") from error
    try:
        file_stat = os.fstat(descriptor)
        _require(
            stat.S_ISREG(file_stat.st_mode)
            and file_stat.st_nlink == 1
            and stat.S_IMODE(file_stat.st_mode) == 0o444,
            f"{field} identity drift",
        )
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        rendered = b"".join(chunks)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"{field} changed while it was read") from error
    finally:
        os.close(descriptor)
    try:
        current = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"{field} changed after it was read") from error
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mode",
        "st_nlink",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    identity = tuple(getattr(file_stat, name) for name in identity_fields)
    _require(
        identity == tuple(getattr(after, name) for name in identity_fields)
        and identity == tuple(getattr(current, name) for name in identity_fields),
        f"{field} changed while it was read",
    )
    try:
        value = json.loads(rendered)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} is not strict JSON") from error
    _require(isinstance(value, dict), f"{field} must contain a JSON object")
    _require(rendered == canonical_bytes(value), f"{field} is not canonical")
    return (
        value,
        rendered,
        {
            "path": str(path.resolve()),
            "size": len(rendered),
            "sha256": hashlib.sha256(rendered).hexdigest(),
            "mode": "0444",
            "nlink": 1,
            "device": format(file_stat.st_dev, "x"),
            "inode": file_stat.st_ino,
        },
    )


def _validate_child_lifecycle_artifacts(
    final_path: Path, final_value: dict[str, Any]
) -> dict[str, Any]:
    completion = final_value["completion"]
    expected_raw_path = final_path.with_name(final_path.name + ".child-close.json")
    expected_ready_path = expected_raw_path.with_name(
        expected_raw_path.name + ".ready.json"
    )
    _require(
        completion["raw_path"] == str(expected_raw_path)
        and completion["ready_path"] == str(expected_ready_path),
        "all-six child lifecycle path binding drift",
    )
    raw_value, raw_bytes, raw_identity = _read_immutable_json_artifact(
        expected_raw_path, field="all-six raw child capture"
    )
    ready_value, ready_bytes, ready_identity = _read_immutable_json_artifact(
        expected_ready_path, field="all-six ready marker"
    )
    validate_native_all_six_smoke(raw_value, require_parent_completion=False)
    _require(
        completion["raw_size"] == len(raw_bytes)
        and completion["raw_sha256"] == hashlib.sha256(raw_bytes).hexdigest()
        and completion["ready_size"] == len(ready_bytes)
        and completion["ready_sha256"] == hashlib.sha256(ready_bytes).hexdigest(),
        "all-six child lifecycle digest binding drift",
    )
    expected_ready = {
        "schema_version": 1,
        "profile": "pi05_droid_native_all_six_coupled_controller_ready_v1",
        "status": "ready_for_simulation_app_close",
        "raw_path": str(expected_raw_path),
        "raw_size": len(raw_bytes),
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }
    _require(ready_value == expected_ready, "all-six ready marker binding drift")
    expected_final = dict(raw_value)
    expected_final["lifecycle"] = {
        "env_close": "complete",
        "simulation_app_close": "invoked_then_child_exited_zero",
        "publication": "stdlib_parent_after_child_exit",
    }
    expected_final["completion"] = completion
    _require(
        final_value == expected_final,
        "all-six final artifact does not exactly derive from child capture",
    )
    return {"raw": raw_identity, "ready": ready_identity}


def publish_immutable_native_all_six_smoke(
    path: Path, value: dict[str, Any]
) -> dict[str, Any]:
    validated = validate_native_all_six_smoke(value, require_parent_completion=True)
    rendered = canonical_bytes(validated)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(rendered)
            output.flush()
            os.fsync(output.fileno())
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return validate_immutable_native_all_six_smoke(path)


def validate_immutable_native_all_six_smoke(path: Path) -> dict[str, Any]:
    value, _, identity = _read_immutable_json_artifact(
        path, field="all-six final smoke artifact"
    )
    validated = validate_native_all_six_smoke(value, require_parent_completion=True)
    child_artifacts = _validate_child_lifecycle_artifacts(Path(path), validated)
    return {
        **identity,
        "status": "pass",
        "runtime_sha256": validated["runtime_contract"]["runtime_sha256"],
        "child_artifacts": child_artifacts,
    }
