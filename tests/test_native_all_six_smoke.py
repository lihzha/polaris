import copy
import hashlib

import numpy as np
import pytest

from conftest import (
    make_joint_velocity_runtime_report,
    make_native_gripper_reset_report,
)
from polaris.native_all_six_smoke import (
    SMOKE_PROFILE,
    canonical_bytes,
    coupled_scenario_plans,
    publish_immutable_native_all_six_smoke,
    validate_immutable_native_all_six_smoke,
    validate_native_all_six_smoke,
)
from polaris.native_gripper_runtime import (
    EXPECTED_DROID_JOINT_NAMES,
    NATIVE_GRIPPER_ALL_SIX_PROFILE,
    NATIVE_GRIPPER_DYNAMIC_PROFILE,
    native_gripper_mimic_reference_contract,
)
from polaris.pi05_droid_jointvelocity_contract import (
    PI05_DROID_JOINTVELOCITY_PROFILE,
)


def _dynamic_report(plan):
    actions = plan["actions"]
    samples = []
    close_directions = np.asarray([1.0, 1.0, -1.0, 1.0, -1.0, -1.0])
    position = np.zeros(13, dtype=np.float64)
    if plan["precondition"] == "closed":
        position[7:13] = close_directions * 0.4
    max_velocity = np.zeros(13, dtype=np.float64)
    for step, action in enumerate(actions):
        for substep in range(9):
            velocity = np.zeros(13, dtype=np.float64)
            velocity[:7] = action[:7]
            transition_is_active = (
                plan["precondition"] == "open" and action[7] > 0.5
            ) or (plan["precondition"] == "closed" and action[7] <= 0.5)
            if transition_is_active:
                transition_sign = 1.0 if plan["precondition"] == "open" else -1.0
                velocity[7:13] = transition_sign * close_directions * 0.1
            max_velocity = np.maximum(max_velocity, np.abs(velocity))
            position_target = [0.0] * 13
            position_target[7] = 0.7853981852531433 if action[7] > 0.5 else 0.0
            samples.append(
                {
                    "sample_index": len(samples),
                    "kind": "apply_entry" if substep < 8 else "post_policy_step",
                    "policy_step_index": step,
                    "physics_substep_index": substep,
                    "joint_position": position.tolist(),
                    "joint_velocity": velocity.tolist(),
                    "joint_acceleration": [0.0] * 13,
                    "joint_velocity_target": [*action[:7], *([0.0] * 6)],
                    "joint_position_target": position_target,
                }
            )
            position += velocity * 0.01
    return {
        "schema_version": 1,
        "profile": NATIVE_GRIPPER_DYNAMIC_PROFILE,
        "joint_names": list(EXPECTED_DROID_JOINT_NAMES),
        "joint_indices": list(range(13)),
        "apply_calls": len(actions) * 8,
        "post_policy_step_samples": len(actions),
        "sample_count": len(actions) * 9,
        "max_abs_joint_velocity_rad_s": max_velocity.tolist(),
        "max_abs_joint_acceleration_rad_s2": [0.0] * 13,
        "samples": samples,
    }


def _child_payload():
    plans = coupled_scenario_plans()
    scenarios = []
    for index, plan in enumerate(plans, start=2):
        scenarios.append(
            {
                **plan,
                "terminated": [False] * len(plan["actions"]),
                "truncated": [False] * len(plan["actions"]),
                "reset_write": make_native_gripper_reset_report(index),
                "dynamic": _dynamic_report(plan),
            }
        )
    return {
        "schema_version": 1,
        "profile": SMOKE_PROFILE,
        "controller_profile": PI05_DROID_JOINTVELOCITY_PROFILE,
        "gripper_profile": NATIVE_GRIPPER_ALL_SIX_PROFILE,
        "environment": "DROID-FoodBussing",
        "runtime_contract": make_joint_velocity_runtime_report(),
        "mimic_joint_contract": native_gripper_mimic_reference_contract(),
        "scenario_plans": plans,
        "scenarios": scenarios,
        "lifecycle": {
            "env_close": "complete",
            "simulation_app_close": "pending_immediate_invocation",
            "publication": "kit_child_before_simulation_app_close",
        },
    }


def _publish_lifecycle_fixture(root):
    final_path = root / "smoke.json"
    raw_path = root / "smoke.json.child-close.json"
    ready_path = root / "smoke.json.child-close.json.ready.json"
    child = _child_payload()
    raw_bytes = canonical_bytes(child)
    raw_path.write_bytes(raw_bytes)
    raw_path.chmod(0o444)
    ready = {
        "schema_version": 1,
        "profile": "pi05_droid_native_all_six_coupled_controller_ready_v1",
        "status": "ready_for_simulation_app_close",
        "raw_path": str(raw_path),
        "raw_size": len(raw_bytes),
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }
    ready_bytes = canonical_bytes(ready)
    ready_path.write_bytes(ready_bytes)
    ready_path.chmod(0o444)
    final = copy.deepcopy(child)
    final["lifecycle"] = {
        "env_close": "complete",
        "simulation_app_close": "invoked_then_child_exited_zero",
        "publication": "stdlib_parent_after_child_exit",
    }
    final["completion"] = {
        "child_exit_code": 0,
        "raw_path": str(raw_path),
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "raw_size": len(raw_bytes),
        "ready_path": str(ready_path),
        "ready_sha256": hashlib.sha256(ready_bytes).hexdigest(),
        "ready_size": len(ready_bytes),
    }
    publish_immutable_native_all_six_smoke(final_path, final)
    return final_path, raw_path, ready_path


def test_all_six_smoke_validates_every_substep_plan_and_child_lifecycle():
    payload = _child_payload()
    assert (
        validate_native_all_six_smoke(payload, require_parent_completion=False)
        == payload
    )


def test_all_six_smoke_validates_parent_completion():
    payload = _child_payload()
    payload["lifecycle"] = {
        "env_close": "complete",
        "simulation_app_close": "invoked_then_child_exited_zero",
        "publication": "stdlib_parent_after_child_exit",
    }
    payload["completion"] = {
        "child_exit_code": 0,
        "raw_path": "/tmp/raw.json",
        "raw_sha256": "a" * 64,
        "raw_size": 1,
        "ready_path": "/tmp/ready.json",
        "ready_sha256": "b" * 64,
        "ready_size": 1,
    }
    assert validate_native_all_six_smoke(payload, require_parent_completion=True)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["scenarios"][0]["dynamic"]["samples"][0][
                "joint_velocity"
            ].__setitem__(8, 5.1),
            "maxima drift",
        ),
        (
            lambda payload: payload["scenarios"][0]["dynamic"]["samples"].pop(),
            "sample set",
        ),
        (
            lambda payload: payload["scenarios"][0]["dynamic"]["samples"][0][
                "joint_velocity_target"
            ].__setitem__(0, 0.0),
            "arm target",
        ),
        (
            lambda payload: payload["scenarios"][1]["reset_write"].__setitem__(
                "reset_count", 9
            ),
            "reset/write count",
        ),
        (
            lambda payload: payload["scenarios"][0]["terminated"].__setitem__(0, True),
            "ended early",
        ),
    ],
)
def test_all_six_smoke_mutations_fail_closed(mutate, message):
    payload = _child_payload()
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        validate_native_all_six_smoke(payload, require_parent_completion=False)


def test_all_six_smoke_plan_is_closed_and_deterministic():
    first = coupled_scenario_plans()
    second = coupled_scenario_plans()
    assert first == second
    assert [item["label"] for item in first] == [
        "immediate_close",
        "delayed_close",
        "immediate_open",
        "delayed_open",
    ]
    assert all(len(item["actions"]) == 12 for item in first)
    assert copy.deepcopy(first) == first


def test_all_six_smoke_rejects_frozen_or_decoupled_measured_motion():
    frozen = _child_payload()
    for scenario in frozen["scenarios"]:
        scenario["dynamic"]["max_abs_joint_velocity_rad_s"] = [0.0] * 13
        for sample in scenario["dynamic"]["samples"]:
            sample["joint_velocity"] = [0.0] * 13
            sample["joint_position"] = [0.0] * 13
    with pytest.raises(ValueError, match="move every arm joint"):
        validate_native_all_six_smoke(frozen, require_parent_completion=False)

    decoupled = _child_payload()
    for sample in decoupled["scenarios"][0]["dynamic"]["samples"]:
        sample["joint_position"][8] = 0.0
    with pytest.raises(ValueError, match="motion/coupling"):
        validate_native_all_six_smoke(decoupled, require_parent_completion=False)


def test_final_artifact_independently_reopens_and_binds_raw_and_ready(tmp_path):
    final_path, raw_path, ready_path = _publish_lifecycle_fixture(tmp_path)
    identity = validate_immutable_native_all_six_smoke(final_path)
    assert identity["child_artifacts"]["raw"]["path"] == str(raw_path)
    assert identity["child_artifacts"]["ready"]["path"] == str(ready_path)

    ready_path.chmod(0o644)
    ready_path.write_bytes(ready_path.read_bytes() + b" ")
    ready_path.chmod(0o444)
    with pytest.raises(ValueError, match="ready marker"):
        validate_immutable_native_all_six_smoke(final_path)


def test_final_artifact_rejects_missing_child_capture(tmp_path):
    final_path, raw_path, _ = _publish_lifecycle_fixture(tmp_path)
    raw_path.unlink()
    with pytest.raises(ValueError, match="raw child capture"):
        validate_immutable_native_all_six_smoke(final_path)
