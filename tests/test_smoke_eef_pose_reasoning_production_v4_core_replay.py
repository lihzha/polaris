from __future__ import annotations

import ast
import copy
from functools import lru_cache
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_module(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_path_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fixture_builder = load_module(
    "reasoning_fulltrace_fixture_builder",
    "build_reasoning_fulltrace_replay_fixture.py",
)
replay = load_module(
    "reasoning_production_v4_core_replay",
    "smoke_eef_pose_reasoning_production_v4_core_replay.py",
)
validator = load_module(
    "reasoning_production_v4_core_replay_validator",
    "validate_eef_pose_reasoning_production_v4_core_replay.py",
)
gate_io = load_module(
    "reasoning_production_v4_core_gate_io",
    "eef_pose_reasoning_production_v4_core_gate_io.py",
)


def _assert_observer_source_read_only(source: str) -> None:
    tree = ast.parse(source)
    forbidden_calls = {
        "set_joint_position_target",
        "set_joint_velocity_target",
        "set_dof_max_velocities",
        "_copy_failure_substep_trace_value",
        "_stage_failure_substep_trace",
    }
    calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in forbidden_calls
    ]
    if calls:
        raise AssertionError(f"observer contains forbidden write calls: {calls!r}")

    forbidden_store_prefixes = (
        "_arm_release_ramp_",
        "_arm_release_observed_count",
        "_gripper_close_arm_interlock_",
        "_failure_substep_trace_",
    )
    stores = [
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.ctx, (ast.Store, ast.Del))
        and node.attr.startswith(forbidden_store_prefixes)
    ]
    if stores:
        raise AssertionError(f"observer mutates production state: {stores!r}")


def test_fixture_action_and_tail_bytes_are_exact() -> None:
    identity, actions = replay.load_actions()
    tail = replay.frozen_tail_contract(actions)

    assert identity["size_bytes"] == 14_478
    assert identity["sha256"] == (
        "daf2aa682f2296a93170f842a5adb13a4fbc6b2694fa5dca28de7ac7ad83d7cb"
    )
    assert len(actions) == replay.ACTION_COUNT == 294
    assert replay.ACTION_ENCODING["uncompressed_sha256"] == (
        "0e781cd1df2d00f3496c1feb2bf079e9194ad664710ac988cc9f7e8bcde11bce"
    )
    assert tail["action_float32_sha256"] == (
        "b938c1ae7f29d0d762b48502af53789a7117e364514f3bdf9887ff5e3e36ab50"
    )
    assert tail["physics_substeps"] == 64
    assert tail["policy_steps"] == 8
    assert replay.TOTAL_APPLY_COUNT == 294 * 8 + 64 == 2_416


def test_production_v4_exact_expected_ramp_contract() -> None:
    assert replay.PRODUCTION_BASE_COMMIT == ("7fc74d648328432a7f9f06d13c0e82a03f73a0c1")
    assert replay.REPLAY_VALIDATION_FIX_COMMIT == (
        "585ab6f72098fd67118fd8b33cdd90be809bed3a"
    )
    assert replay.REPLAY_IMPLEMENTATION_COMMIT == (
        "2ebfe7db5b2a31887481781b214608976e8023db"
    )
    assert replay.REPLAY_PARENT_COMMIT == ("e18b8ebbc26fd309d8e45bd58bef9c867948098a")
    assert replay.SOURCE_TRACE_POLARIS_COMMIT == (
        "0611d384f5f26ef9bd8ff114be273e875c3fe719"
    )
    assert replay.EXPECTED_RAMP_WINDOWS == (
        (1600, 1615),
        (2176, 2191),
        (2334, 2349),
    )
    assert replay.EXPECTED_RAMP_APPLY_INDICES == [
        item
        for first, last in replay.EXPECTED_RAMP_WINDOWS
        for item in range(first, last + 1)
    ]
    assert replay.EXPECTED_RAMP_INDICES == list(range(16)) * 3
    assert replay.EXPECTED_LIMITED_APPLIES_PER_RAMP == [15, 8, 15]
    assert replay.EXPECTED_LIMITED_JOINTS_PER_RAMP == [81, 35, 105]
    assert replay.EXPECTED_CORE_RAMP_COUNTS == {
        "release_observed_count": 3,
        "ramp_started_count": 3,
        "ramp_completed_count": 3,
        "ramp_cancelled_by_reactivation_count": 0,
        "ramp_target_apply_count": 48,
        "cancelled_ramp_target_apply_count": 0,
        "ramp_limited_target_apply_count": 38,
        "ramp_limited_joint_target_count": 221,
    }


def test_arm_observer_has_no_physical_or_production_state_writes() -> None:
    source = inspect.getsource(replay.make_full_trace_arm_class)
    _assert_observer_source_read_only(source)
    tree = ast.parse(source)
    delegates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_set_targets_and_commit_gripper_close_arm_interlock"
    ]
    assert len(delegates) == 1
    assert "super()._set_targets_and_commit_gripper_close_arm_interlock(" in source
    assert "super().apply_actions()" in source


def test_gripper_observer_has_no_physical_or_controller_state_writes() -> None:
    source = inspect.getsource(replay.make_full_trace_gripper_class)
    _assert_observer_source_read_only(source)
    assert "super().apply_actions()" in source
    assert "set_joint_position_target" not in source
    assert "set_dof_max_velocities" not in source


@pytest.mark.parametrize(
    "adversarial_source",
    [
        "def bad(self):\n    self._asset.set_joint_position_target(None, None)\n",
        "def bad(self):\n    self._arm_release_ramp_phase = 'release'\n",
        (
            "def bad(self):\n"
            "    self._copy_failure_substep_trace_value(field='x', slot=0, value=None)\n"
        ),
        "def bad(self):\n    self._gripper_close_arm_interlock_remaining = 0\n",
    ],
)
def test_static_observer_gate_rejects_adversarial_writes(
    adversarial_source: str,
) -> None:
    with pytest.raises(AssertionError):
        _assert_observer_source_read_only(adversarial_source)


def _fake_release_helper(torch: Any):
    def helper(
        current: Any,
        nominal: Any,
        maximum: Any,
        *,
        ramp_index: int,
    ) -> Any:
        fraction = float(ramp_index / 15)
        if ramp_index == 0:
            target = current.detach().clone()
        elif ramp_index == 15:
            target = nominal.detach().clone()
        else:
            bound = maximum * fraction
            target = current + torch.clamp(
                nominal - current,
                min=-bound,
                max=bound,
            )
        return SimpleNamespace(
            target=target,
            fraction=float(torch.tensor(fraction, dtype=torch.float32).item()),
            limited_joint_mask=target != nominal,
        )

    return helper


@pytest.mark.parametrize("ramp_index", [0, 15])
def test_observer_behavior_delegates_one_core_setter_and_checks_endpoints(
    ramp_index: int,
) -> None:
    torch = pytest.importorskip("torch")
    release_helper = _fake_release_helper(torch)

    def bound_helper(
        current: Any,
        raw: Any,
        _maximum: Any,
        _soft: Any,
        *,
        target_guard_band_delta_joint_pos: Any,
    ) -> tuple[Any, Any, Any, Any]:
        del target_guard_band_delta_joint_pos
        zeros = torch.zeros_like(current, dtype=torch.bool)
        return raw.detach().clone(), raw - current, zeros, zeros

    class FakeProductionArm:
        def __init__(self, cfg: Any, env: Any) -> None:
            self._cfg = cfg
            self._apply_call_count = 0
            self._arm_release_ramp_target_apply_count = 0
            self._arm_release_ramp_last_target_apply_index = None
            self._arm_release_ramp_last_index = None
            self._joint_ids = list(range(7))
            self._nominal_max_delta_joint_pos = torch.full(
                (1, 7), 0.25, dtype=torch.float32
            )
            self._max_delta_joint_pos = torch.full((1, 7), 0.3, dtype=torch.float32)
            self._soft_joint_position_limits = torch.tensor(
                [[[-3.0, 3.0]] * 7], dtype=torch.float32
            )
            current = torch.tensor(
                [[0.1, 0.2, -0.1, -1.0, 0.3, 1.0, -0.2]],
                dtype=torch.float32,
            )
            nominal = current + 0.2
            data = SimpleNamespace(
                joint_pos_target=torch.zeros((1, 7), dtype=torch.float32)
            )
            self._asset = SimpleNamespace(data=data)
            self._current = current
            self._nominal = nominal
            self._failure_substep_trace_pending_slot = None
            self._failure_substep_trace_pending_apply_index = None
            self._failure_substep_trace_buffers = {
                "joint_pos_rad": torch.empty((1, 1, 7), dtype=torch.float32),
                "raw_dls_joint_pos_target_rad": torch.empty(
                    (1, 1, 7), dtype=torch.float32
                ),
                "new_joint_pos_target_rad": torch.empty((1, 1, 7), dtype=torch.float32),
            }
            self.setter_call_count = 0

        def reset(self, env_ids: Any = None) -> None:
            del env_ids

        def _set_targets_and_commit_gripper_close_arm_interlock(
            self,
            safe_target: Any,
            staged: Any,
            staged_release_ramp: Any,
            failure_trace: Any,
        ) -> None:
            del staged, staged_release_ramp, failure_trace
            self.setter_call_count += 1
            self._asset.data.joint_pos_target[:, self._joint_ids] = safe_target
            self._failure_substep_trace_buffers["new_joint_pos_target_rad"][0].copy_(
                safe_target
            )

        def apply_actions(self) -> None:
            self._apply_call_count += 1
            result = release_helper(
                self._current,
                self._nominal,
                self._nominal_max_delta_joint_pos,
                ramp_index=self._cfg.ramp_index,
            )
            self._failure_substep_trace_pending_slot = 0
            self._failure_substep_trace_pending_apply_index = self._apply_call_count - 1
            self._failure_substep_trace_buffers["joint_pos_rad"][0].copy_(self._current)
            self._failure_substep_trace_buffers["raw_dls_joint_pos_target_rad"][
                0
            ].copy_(self._nominal)
            self._set_targets_and_commit_gripper_close_arm_interlock(
                result.target,
                object(),
                object(),
                {},
            )
            self._arm_release_ramp_target_apply_count += 1
            self._arm_release_ramp_last_target_apply_index = self._apply_call_count - 1
            self._arm_release_ramp_last_index = self._cfg.ramp_index

    env = SimpleNamespace(action_manager=SimpleNamespace(_terms={"finger_joint": None}))
    observed_class = replay.make_full_trace_arm_class(
        FakeProductionArm,
        torch_module=torch,
        bound_helper=bound_helper,
        release_helper=release_helper,
    )
    arm = observed_class(SimpleNamespace(ramp_index=ramp_index), env)
    arm.apply_actions()
    report = arm.production_core_ramp_observation_report()

    assert arm.setter_call_count == 1
    assert report["entry_count"] == 1
    assert report["observer_write_contract"] == {
        "target_setter_call_count": 0,
        "failure_trace_write_count": 0,
        "release_ramp_state_write_count": 0,
        "gripper_target_or_state_write_count": 0,
    }
    entry = report["entries"][0]
    assert entry["ramp_index"] == ramp_index
    assert len(set(entry["target_little_endian_float32_sha256"].values())) == 1
    assert entry["endpoint_contract"] == {
        "index0_equals_current": True,
        "index15_equals_nominal": True,
    }


LIMITED_COUNTS = [
    [7, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
    [7, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0],
    [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0],
]


def _synthetic_core_gate_payload() -> tuple[
    list[dict[str, Any]], dict[str, Any], dict[str, Any]
]:
    records: list[dict[str, Any]] = []
    maxima = [0.0] * 7
    for ordinal, (apply_index, ramp_index) in enumerate(
        zip(
            replay.EXPECTED_RAMP_APPLY_INDICES,
            replay.EXPECTED_RAMP_INDICES,
            strict=True,
        )
    ):
        ramp_ordinal = ordinal // 16
        desired_limited = LIMITED_COUNTS[ramp_ordinal][ramp_index]
        current = [0.0] * 7
        maximum = list(replay.ARM_NOMINAL_MAX_DELTA_RAD)
        fraction = replay.float32(ramp_index / 15)
        nominal = []
        for joint_index, bound in enumerate(maximum):
            if ramp_index == 15:
                value = bound
            elif joint_index < desired_limited:
                value = bound
            else:
                value = replay.float32_multiply(
                    replay.float32_multiply(bound, fraction),
                    0.5,
                )
            nominal.append(value)
        target = [
            replay._float32_ramp_target(
                current=current_value,
                nominal=nominal_value,
                maximum_delta=maximum_value,
                index=ramp_index,
            )
            for current_value, nominal_value, maximum_value in zip(
                current, nominal, maximum, strict=True
            )
        ]
        mask = [
            not replay.float32_equal(actual, wanted)
            for actual, wanted in zip(target, nominal, strict=True)
        ]
        assert sum(mask) == desired_limited
        for joint_index, (wanted, actual) in enumerate(
            zip(nominal, target, strict=True)
        ):
            maxima[joint_index] = max(
                maxima[joint_index],
                abs(replay.float32_subtract(wanted, actual)),
            )
        target_hash = replay.little_endian_float32_sha256(
            target, field="synthetic target"
        )
        records.append(
            {
                "profile": "production_core_release_ramp_target_observation_v1",
                "apply_index": apply_index,
                "policy_step": apply_index // 8,
                "physics_substep": apply_index % 8,
                "arm_joint_ids": list(range(7)),
                "arm_joint_names": list(replay.ARM_JOINT_NAMES),
                "ramp_index": ramp_index,
                "fraction_float32": fraction,
                "nominal_max_delta_joint_pos_rad": maximum,
                "current_joint_pos_rad": current,
                "raw_dls_joint_pos_target_rad": nominal,
                "nominal_safe_target_rad": nominal,
                "production_core_target_rad": target,
                "failure_trace_target_rad": target,
                "live_setter_readback_target_rad": target,
                "limited_joint_mask": mask,
                "limited_joint_count": sum(mask),
                "max_abs_nominal_to_core_target_change_rad": max(
                    abs(replay.float32_subtract(wanted, actual))
                    for wanted, actual in zip(nominal, target, strict=True)
                ),
                "target_little_endian_float32_sha256": {
                    "production_core": target_hash,
                    "independent_helper": target_hash,
                    "failure_trace": target_hash,
                    "live_setter_readback": target_hash,
                },
                "bitwise_contract": {
                    "core_equals_independent_helper": True,
                    "core_equals_failure_trace": True,
                    "core_equals_live_setter_readback": True,
                },
                "endpoint_contract": {
                    "index0_equals_current": True,
                    "index15_equals_nominal": True,
                },
                "observer_target_setter_call_count": 0,
                "observer_failure_trace_write_count": 0,
                "observer_release_ramp_state_write_count": 0,
                "observer_gripper_target_or_state_write_count": 0,
            }
        )

    by_apply = {record["apply_index"]: record for record in records}
    entries = []
    count = 0
    for apply_index in range(replay.TOTAL_APPLY_COUNT):
        record = by_apply.get(apply_index)
        if record is not None:
            count += 1
            current = record["current_joint_pos_rad"]
            target = record["production_core_target_rad"]
        else:
            current = [0.0] * 7
            target = [0.0] * 7
        state = {
            "enabled": True,
            "target_apply_count": count,
        }
        snapshot = {
            "all_joint_pos_rad": current,
            "all_joint_pos_target_rad": target,
            "interlock": {"release_ramp": state},
        }
        entries.append(
            {
                "pre": copy.deepcopy(snapshot),
                "command_after_setters": copy.deepcopy(snapshot),
                "post": copy.deepcopy(snapshot),
            }
        )
    observation = {
        "profile": "production_v4_core_release_ramp_bitwise_observer_v1",
        "production_base_commit": replay.PRODUCTION_BASE_COMMIT,
        "controller_profile": replay.CONTROLLER_PROFILE,
        "entry_count": 48,
        "entries": records,
        "observer_write_contract": {
            "target_setter_call_count": 0,
            "failure_trace_write_count": 0,
            "release_ramp_state_write_count": 0,
            "gripper_target_or_state_write_count": 0,
        },
    }
    controller_report = {
        "arm_release_ramp": {
            "enabled": True,
            "phase": "release",
            "next_index": None,
            "last_target_apply_index": 2349,
            "last_ramp_index": 15,
            "max_abs_nominal_to_ramped_target_change_rad": maxima,
            **replay.EXPECTED_CORE_RAMP_COUNTS,
        },
        "gripper_close_arm_interlock": {
            "anchor_completion_count": 1,
            "anchor_open_cancel_count": 2,
        },
    }
    return entries, observation, controller_report


def test_core_gate_accepts_exact_48_record_contract() -> None:
    entries, observation, controller_report = _synthetic_core_gate_payload()
    result = replay.validate_core_release_ramp_trace(
        entries,
        observation=observation,
        controller_report=controller_report,
    )

    assert result["passed"] is True
    assert result["entry_count"] == 48
    assert result["limited_applies_per_ramp"] == [15, 8, 15]
    assert result["limited_joints_per_ramp"] == [81, 35, 105]
    assert result["aggregate_counts"] == replay.EXPECTED_CORE_RAMP_COUNTS


@pytest.mark.parametrize(
    ("target", "mutate"),
    [
        (
            "second setter attestation",
            lambda _entries, observation, _report: observation["entries"][
                0
            ].__setitem__("observer_target_setter_call_count", 1),
        ),
        (
            "failure-trace mismatch",
            lambda _entries, observation, _report: observation["entries"][0][
                "failure_trace_target_rad"
            ].__setitem__(0, 1.0),
        ),
        (
            "window drift",
            lambda _entries, observation, _report: observation["entries"][
                0
            ].__setitem__("apply_index", 1599),
        ),
        (
            "aggregate count drift",
            lambda _entries, _observation, report: report[
                "arm_release_ramp"
            ].__setitem__("ramp_limited_target_apply_count", 37),
        ),
    ],
)
def test_core_gate_rejects_adversarial_evidence(
    target: str,
    mutate: Any,
) -> None:
    entries, observation, controller_report = _synthetic_core_gate_payload()
    mutate(entries, observation, controller_report)
    with pytest.raises(replay.ProductionV4ReplayError, match="production"):
        replay.validate_core_release_ramp_trace(
            entries,
            observation=observation,
            controller_report=controller_report,
        )


def _mp4_box(box_type: str, payload: bytes = b"") -> bytes:
    encoded = box_type.encode("ascii")
    return (8 + len(payload)).to_bytes(4, "big") + encoded + payload


def test_mp4_faststart_layout_parser(tmp_path: Path) -> None:
    path = tmp_path / "video.mp4"
    path.write_bytes(_mp4_box("ftyp") + _mp4_box("moov") + _mp4_box("mdat"))
    assert [item["type"] for item in validator.mp4_box_layout(path)] == [
        "ftyp",
        "moov",
        "mdat",
    ]


def _gripper_snapshot(*, closed_target: bool) -> dict[str, Any]:
    target = [0.0] * 6
    if closed_target:
        target[0] = validator.gripper_runtime_contract.GRIPPER_CLOSED_TARGET_FLOAT32
    return {
        "joint_pos_rad": [0.0] * 6,
        "joint_vel_rad_s": [0.0] * 6,
        "joint_acc_rad_s2": [0.0] * 6,
        "joint_pos_target_rad": target,
        "joint_vel_target_rad_s": [0.0] * 6,
        "joint_effort_target_nm": [0.0] * 6,
    }


def _full_snapshot(gripper: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "joint_pos_rad": "all_joint_pos_rad",
        "joint_vel_rad_s": "all_joint_vel_rad_s",
        "joint_acc_rad_s2": "all_joint_acc_rad_s2",
        "joint_pos_target_rad": "all_joint_pos_target_rad",
        "joint_vel_target_rad_s": "all_joint_vel_target_rad_s",
        "joint_effort_target_nm": "all_joint_effort_target_nm",
    }
    return {
        full_field: [0.0] * 7 + list(gripper[gripper_field])
        for gripper_field, full_field in mapping.items()
    }


@lru_cache(maxsize=1)
def _synthetic_all_six_and_full_trace() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _identity, source_actions = replay.load_actions()
    policy_actions = [
        *source_actions,
        *([source_actions[-1]] * replay.TAIL_POLICY_STEPS),
    ]
    initial = _gripper_snapshot(closed_target=False)
    terminal = _gripper_snapshot(closed_target=True)
    full_trace: list[dict[str, Any]] = []
    for apply_index in range(replay.TOTAL_APPLY_COUNT):
        policy_step = apply_index // replay.DECIMATION
        raw = policy_actions[policy_step][7]
        retained = apply_index >= 2_352
        requested_endpoint = (
            validator.gripper_runtime_contract.GRIPPER_CLOSED_TARGET_FLOAT32
            if validator.float32_equal(raw, 1.0)
            else validator.gripper_runtime_contract.GRIPPER_OPEN_TARGET_FLOAT32
        )
        snapshot = terminal if retained else initial
        full_trace.append(
            {
                "apply_index": apply_index,
                "policy_step": policy_step,
                "physics_substep": apply_index % replay.DECIMATION,
                "raw_action": raw,
                "requested_endpoint_rad": requested_endpoint,
                "target_after_setter_rad": (
                    validator.gripper_runtime_contract.GRIPPER_CLOSED_TARGET_FLOAT32
                    if retained
                    else 0.0
                ),
                "pre": _full_snapshot(snapshot),
                "command_after_setters": _full_snapshot(snapshot),
                "post": _full_snapshot(snapshot),
            }
        )
    entries = [
        {
            "apply_index": apply_index,
            "policy_step": apply_index // replay.DECIMATION,
            "physics_substep": apply_index % replay.DECIMATION,
            "raw_action": 1.0,
            "requested_endpoint_rad": (
                validator.gripper_runtime_contract.GRIPPER_CLOSED_TARGET_FLOAT32
            ),
            "pre": copy.deepcopy(terminal),
            "target_after_setter_rad": (
                validator.gripper_runtime_contract.GRIPPER_CLOSED_TARGET_FLOAT32
            ),
            "post": copy.deepcopy(terminal),
        }
        for apply_index in range(2_352, 2_416)
    ]
    trace = {
        "schema_version": 1,
        "profile": validator.gripper_trace_contract.EEF_ALL_SIX_GRIPPER_TRACE_PROFILE,
        "episode_index": 0,
        "capacity": 64,
        "decimation": replay.DECIMATION,
        "joint_names": list(validator.gripper_runtime_contract.GRIPPER_JOINT_NAMES),
        "joint_indices": list(validator.gripper_runtime_contract.GRIPPER_JOINT_INDICES),
        "process_action_calls": 302,
        "total_apply_entries": 2_416,
        "dropped_entries": 2_352,
        "initial_snapshot": copy.deepcopy(initial),
        "entries": entries,
        "terminal_snapshot": copy.deepcopy(terminal),
        "numerical_failure": False,
    }
    return trace, full_trace


def test_post_kit_all_six_gate_accepts_and_retains_canonical_evidence() -> None:
    trace, full_trace = _synthetic_all_six_and_full_trace()
    result = validator.validate_production_all_six_gripper_trace(
        copy.deepcopy(trace),
        full_substep_trace=copy.deepcopy(full_trace),
    )

    assert result["episode_length"] == 302
    assert result["expected_apply_calls"] == 2_416
    assert result["validated_trace"] == trace
    assert result["canonical_json_sha256"] == validator.sha256(
        validator.canonical_json_bytes(trace)
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda trace, _full: trace.__setitem__("hidden", True),
        lambda trace, _full: trace.__setitem__("dropped_entries", 2_351),
        lambda trace, _full: trace["entries"][0].__setitem__("raw_action", 0.0),
        lambda trace, _full: trace["entries"][1]["pre"]["joint_pos_rad"].__setitem__(
            0, 1.0
        ),
        lambda _trace, full: full[2_352]["post"][
            "all_joint_pos_target_rad"
        ].__setitem__(7, 0.0),
        lambda _trace, full: full[100].__setitem__("raw_action", 1.0),
    ],
)
def test_post_kit_all_six_gate_rejects_malformed_and_cross_trace_tamper(
    mutation: Any,
) -> None:
    trace, full_trace = _synthetic_all_six_and_full_trace()
    trace = copy.deepcopy(trace)
    full_trace = copy.deepcopy(full_trace)
    mutation(trace, full_trace)

    with pytest.raises(validator.ValidationError, match="all-six|full trace"):
        validator.validate_production_all_six_gripper_trace(
            trace,
            full_substep_trace=full_trace,
        )


@lru_cache(maxsize=1)
def _production_safety_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_support = load_path_module(
        "_fulltrace_runtime_contract_test_support",
        ROOT / "tests/test_eef_runtime_contract.py",
    )
    gripper_support = load_path_module(
        "_fulltrace_gripper_runtime_test_support",
        ROOT / "tests/test_eef_gripper_runtime.py",
    )
    safety = runtime_support._episode_safety(episode=0, length=302)  # noqa: SLF001
    safety["joint_velocity_limits_rad_s"] = list(
        validator.safety_post_kit_contract.EXPECTED_VELOCITY_LIMITS
    )
    source, stage, root, _followers = gripper_support._fake_live_mimic_stage()  # noqa: SLF001
    _, followers = gripper_support.runtime._write_spawned_mimic_compliance(  # noqa: SLF001
        stage=stage,
        roots=[root],
        source_contract=source,
    )
    static = gripper_support._static_contract()  # noqa: SLF001
    static["driver_target_slew"] = gripper_support._candidate_target_slew_static()  # noqa: SLF001
    static["mimic_compliance"] = gripper_support._candidate_compliance_contract(  # noqa: SLF001
        followers
    )
    trace, _full_trace = _synthetic_all_six_and_full_trace()
    expected = validator.derive_gripper_target_slew_evidence(trace)
    dynamic = gripper_support._dynamic_evidence()  # noqa: SLF001
    dynamic["apply_entry_samples"] = 2_416
    dynamic["post_policy_step_samples"] = 302
    dynamic["terminal_state"]["sample_index"] = 2_717
    dynamic["terminal_state"]["joint_position_target_rad"][0] = (
        gripper_support.runtime.GRIPPER_CLOSED_TARGET_FLOAT32
    )
    dynamic["driver_target_slew"] = {
        "profile": validator.EXPECTED_TARGET_SLEW_PROFILE,
        "process_action_calls": expected["process_action_calls"],
        "apply_calls": expected["apply_calls"],
        "initialization_count": 1,
        "endpoint_change_count": expected["endpoint_change_count"],
        "repeated_endpoint_process_count": expected["repeated_endpoint_process_count"],
        "slew_limited_apply_count": expected["slew_limited_apply_count"],
        "endpoint_reached_apply_count": expected["endpoint_reached_apply_count"],
        "live_limit_validation_count": expected["apply_calls"],
        "max_abs_target_step_rad": expected["max_abs_target_step_rad"],
        "max_abs_endpoint_error_before_step_rad": expected[
            "max_abs_endpoint_error_before_step_rad"
        ],
        "max_abs_endpoint_error_after_step_rad": expected[
            "max_abs_endpoint_error_after_step_rad"
        ],
        "initial_anchor_rad": expected["initial_anchor_rad"],
        "last_requested_endpoint_rad": expected["last_requested_endpoint_rad"],
        "last_applied_target_rad": expected["last_applied_target_rad"],
    }
    safety["gripper_runtime_static"] = static
    safety["gripper_runtime_dynamic"] = dynamic
    report = runtime_support._candidate_controller_report(  # noqa: SLF001
        safety,
        initial=False,
        profile=replay.CONTROLLER_PROFILE,
    )
    report["gripper_close_arm_interlock"]["observed_endpoint_change_count"] = 5
    return safety, report


def test_post_kit_safety_gate_uses_exact_cumulative_target_slew_override() -> None:
    safety, report = _production_safety_fixture()
    trace, _full_trace = _synthetic_all_six_and_full_trace()

    with pytest.raises(ValueError, match="closed transition"):
        validator.safety_post_kit_contract._validate_safety_report(  # noqa: SLF001
            copy.deepcopy(safety),
            field="without cumulative override",
            episode_index=0,
            apply_calls=2_416,
            expect_closed_target=True,
            expected_endpoint_change_count=5,
            expected_gripper_target_slew_profile=(
                validator.EXPECTED_TARGET_SLEW_PROFILE
            ),
        )
    result = validator.validate_production_safety(
        copy.deepcopy(safety),
        all_six_gripper_trace=copy.deepcopy(trace),
        controller_report=copy.deepcopy(report),
    )

    assert result["episode_safety_cadence"] == {
        "apply_calls": 2_416,
        "expected_decimation": 8,
        "failed_policy_step": None,
        "failed_physics_substep": None,
        "abort_count": 0,
    }
    assert result["independent_target_slew_replay"][
        "endpoint_change_apply_indices"
    ] == [1_584, 1_600, 2_120, 2_176, 2_248]
    assert result["independent_target_slew_replay"]["slew_limited_apply_count"] == 217
    assert result["validated_safety"] == safety
    assert result["validated_controller_report"] == report


def _dictionary_field_paths(
    value: Any, path: tuple[Any, ...] = ()
) -> list[tuple[Any, ...]]:
    paths: list[tuple[Any, ...]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = (*path, key)
            paths.append(child_path)
            paths.extend(_dictionary_field_paths(child, child_path))
    elif isinstance(value, list) and value and isinstance(value[0], dict):
        paths.extend(_dictionary_field_paths(value[0], (*path, 0)))
    return paths


def _remove_path(value: Any, path: tuple[Any, ...]) -> None:
    parent = value
    for item in path[:-1]:
        parent = parent[item]
    del parent[path[-1]]


def test_post_kit_safety_gate_rejects_every_closed_schema_field_removal() -> None:
    safety, report = _production_safety_fixture()
    trace, _full_trace = _synthetic_all_six_and_full_trace()
    paths = _dictionary_field_paths(safety)
    assert len(paths) > 200
    for path in paths:
        candidate = copy.deepcopy(safety)
        _remove_path(candidate, path)
        with pytest.raises(
            validator.ValidationError,
            match="production safety",
        ):
            validator.validate_production_safety(
                candidate,
                all_six_gripper_trace=copy.deepcopy(trace),
                controller_report=copy.deepcopy(report),
            )


def test_post_kit_safety_gate_rejects_category_and_exact_counter_tamper() -> None:
    safety, report = _production_safety_fixture()
    trace, _full_trace = _synthetic_all_six_and_full_trace()
    mutations = [
        lambda value: value.__setitem__("unexpected", True),
        lambda value: value.__setitem__("physics_dt", 1.0 / 60.0),
        lambda value: value.__setitem__("profile", "wrong"),
        lambda value: value.__setitem__(
            "target_joint_pos_limits_float32_sha256", "0" * 64
        ),
        lambda value: value["counters"].__setitem__("apply_calls", 2_415),
        lambda value: value["gripper_runtime_static"]["driver_target_slew"].__setitem__(
            "profile", "wrong"
        ),
        lambda value: value["gripper_runtime_dynamic"].__setitem__(
            "post_policy_step_samples", 301
        ),
        lambda value: value["gripper_runtime_dynamic"][
            "driver_target_slew"
        ].__setitem__("process_action_calls", 301),
        lambda value: value["gripper_runtime_dynamic"][
            "driver_target_slew"
        ].__setitem__("apply_calls", 2_415),
        lambda value: value["gripper_runtime_dynamic"][
            "driver_target_slew"
        ].__setitem__("endpoint_change_count", 4),
        lambda value: value["gripper_runtime_dynamic"][
            "driver_target_slew"
        ].__setitem__("repeated_endpoint_process_count", 295),
        lambda value: value["gripper_runtime_dynamic"][
            "driver_target_slew"
        ].__setitem__("slew_limited_apply_count", 216),
        lambda value: value["gripper_runtime_dynamic"][
            "driver_target_slew"
        ].__setitem__("endpoint_reached_apply_count", 2_198),
        lambda value: value["gripper_runtime_dynamic"][
            "driver_target_slew"
        ].__setitem__("live_limit_validation_count", 2_415),
        lambda value: value["gripper_runtime_dynamic"]["terminal_state"][
            "joint_position_target_rad"
        ].__setitem__(0, 0.0),
        lambda value: value["gripper_runtime_dynamic"][
            "driver_target_slew"
        ].__setitem__("last_applied_target_rad", 0.0),
    ]
    for mutation in mutations:
        candidate = copy.deepcopy(safety)
        mutation(candidate)
        with pytest.raises(validator.ValidationError, match="production safety"):
            validator.validate_production_safety(
                candidate,
                all_six_gripper_trace=copy.deepcopy(trace),
                controller_report=copy.deepcopy(report),
            )


@pytest.mark.parametrize("overflow", ["1e999", "-1e999"])
def test_strict_json_rejects_deep_exponent_overflow_and_recursive_nonfinite(
    tmp_path: Path,
    overflow: str,
) -> None:
    path = tmp_path / "overflow.json"
    path.write_text(
        f'{{"ignored":{{"deep":[0,{overflow}]}}}}',
        encoding="utf-8",
    )
    os.chmod(path, 0o444)

    with pytest.raises(validator.ValidationError, match="non-finite JSON float"):
        validator.strict_json(path)
    with pytest.raises(validator.ValidationError, match="non-finite JSON number"):
        validator.audit_json_numbers({"ignored": {"deep": [True, math.inf]}})
    validator.audit_json_numbers({"bool_is_not_numeric": True, "integer": 1})


def test_file_identity_rejects_lexical_symlink_before_resolution(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "linked.json"
    link.symlink_to(target)

    with pytest.raises(validator.ValidationError, match="linked file"):
        validator.file_identity(link)
    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "absent.json")
    with pytest.raises(validator.ValidationError, match="linked file"):
        validator.file_identity(dangling)
    manifest_link = tmp_path / "manifest.json"
    manifest_link.symlink_to(tmp_path / "absent-manifest.json")
    with pytest.raises(validator.ValidationError, match="refusing to overwrite"):
        validator.atomic_write(manifest_link, {"passed": True})


def test_full_result_routes_all_six_then_closed_safety_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {field: None for field in validator.RESULT_FIELDS}
    payload.update(
        {
            "schema_version": 1,
            "profile": replay.PROFILE,
            "passed": True,
            "controller_replay_only": True,
            "variant": replay.VARIANT,
            "controller_profile": replay.CONTROLLER_PROFILE,
            "production_all_six_gripper_trace": {"trace": True},
            "full_substep_trace": [{"full": True}],
            "production_safety": {"safety": True},
            "production_controller_report": {"controller": True},
        }
    )
    observed: list[tuple[str, Any]] = []
    monkeypatch.setattr(validator, "validate_post_kit_validator_sources", lambda: {})

    def validate_gripper(value: Any, *, full_substep_trace: Any) -> dict[str, Any]:
        observed.append(("gripper", (value, full_substep_trace)))
        return {"validated_trace": {"validated": True}}

    def validate_safety(
        value: Any,
        *,
        all_six_gripper_trace: Any,
        controller_report: Any,
    ) -> dict[str, Any]:
        observed.append(("safety", (value, all_six_gripper_trace, controller_report)))
        raise validator.ValidationError("closed-safety-sentinel")

    monkeypatch.setattr(
        validator,
        "validate_production_all_six_gripper_trace",
        validate_gripper,
    )
    monkeypatch.setattr(validator, "validate_production_safety", validate_safety)
    with pytest.raises(validator.ValidationError, match="closed-safety-sentinel"):
        validator.validate_result(
            payload,
            expected_commit="0" * 40,
            expected_job_id=1,
            expected_launch_id="0" * 64,
            result_identity={},
            video_path=Path("unused.mp4"),
            ffprobe="unused",
            ffmpeg="unused",
            simulator_srun_exit_code=0,
        )
    assert observed == [
        ("gripper", ({"trace": True}, [{"full": True}])),
        ("safety", ({"safety": True}, {"validated": True}, {"controller": True})),
    ]


def test_validator_result_schema_matches_runner_contract() -> None:
    assert validator.RESULT_FIELDS == {
        "schema_version",
        "profile",
        "passed",
        "controller_replay_only",
        "variant",
        "controller_profile",
        "repository",
        "production_core_sources",
        "container_image",
        "lifecycle",
        "production_eval",
        "fixture",
        "source_trace_polaris_commit",
        "source_trace_sha256",
        "source_action_float32_sha256",
        "boundary_helper",
        "assets",
        "runtime_protocol",
        "runtime_frame",
        "observer_class_contract",
        "production_runtime",
        "production_gripper_contract",
        "production_safety",
        "production_controller_report",
        "production_all_six_gripper_trace",
        "production_core_ramp_observation",
        "action_count",
        "actions_completed",
        "tail_contract",
        "tail_policy_steps_completed",
        "tail_physics_substeps_completed",
        "total_apply_count",
        "numerical_failure",
        "controller_failure_evidence",
        "outcome",
        "full_substep_trace_profile",
        "full_substep_trace_cadence",
        "full_substep_trace",
        "full_substep_summary",
        "video",
        "runtime_close",
    }


def _gate_io_publish_cli(temporary: Path, marker: Path) -> list[str]:
    return [
        sys.executable,
        str(SCRIPTS / "eef_pose_reasoning_production_v4_core_gate_io.py"),
        "publish-success",
        "--temporary",
        str(temporary),
        "--marker",
        str(marker),
        "--job-id",
        "12345",
        "--variant",
        replay.VARIANT,
        "--result-sha256",
        "1" * 64,
        "--video-sha256",
        "2" * 64,
        "--manifest-sha256",
        "3" * 64,
    ]


def _gate_io_payload() -> bytes:
    return gate_io.success_payload(
        job_id="12345",
        variant=replay.VARIANT,
        result_sha256="1" * 64,
        video_sha256="2" * 64,
        manifest_sha256="3" * 64,
    )


def test_gate_io_success_marker_is_immutable_single_link(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    temporary = attempt / ".SUCCESS.12345.tmp"
    marker = attempt / "SUCCESS"

    identity = gate_io.publish_success(temporary, marker, _gate_io_payload())

    metadata = os.lstat(marker)
    payload = marker.read_bytes()
    assert not temporary.exists()
    assert payload == _gate_io_payload()
    assert stat.S_ISREG(metadata.st_mode)
    assert stat.S_IMODE(metadata.st_mode) == 0o444
    assert metadata.st_nlink == 1
    assert identity == {
        "path": str(marker.resolve()),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "mode": "0444",
        "nlink": 1,
    }


def test_gate_io_forced_post_link_failure_propagates_and_cleans(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    temporary = attempt / ".SUCCESS.12345.tmp"
    marker = attempt / "SUCCESS"
    command = [*_gate_io_publish_cli(temporary, marker), "--test-fail-after-link"]

    completed = subprocess.run(
        ["bash", "-c", 'set -Eeuo pipefail\n"$@"', "gate-io-shell", *command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "forced post-link publication failure" in completed.stderr
    assert not temporary.exists() and not temporary.is_symlink()
    assert not marker.exists() and not marker.is_symlink()


def test_gate_io_preserves_preexisting_success_inode_and_bytes(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    temporary = attempt / ".SUCCESS.12345.tmp"
    marker = attempt / "SUCCESS"
    marker.write_bytes(b"preexisting-authoritative-marker\n")
    before = os.lstat(marker)

    completed = subprocess.run(
        _gate_io_publish_cli(temporary, marker),
        check=False,
        capture_output=True,
        text=True,
    )

    after = os.lstat(marker)
    assert completed.returncode == 1
    assert marker.read_bytes() == b"preexisting-authoritative-marker\n"
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert not temporary.exists() and not temporary.is_symlink()


def test_gate_io_post_link_cleanup_preserves_replacement_inode(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    temporary = attempt / ".SUCCESS.12345.tmp"
    marker = attempt / "SUCCESS"
    replacement: dict[str, int] = {}

    def replace_marker_and_fail() -> None:
        marker.unlink()
        marker.write_bytes(b"unrelated-replacement\n")
        metadata = os.lstat(marker)
        replacement.update(dev=metadata.st_dev, ino=metadata.st_ino)
        raise gate_io.GateIoError("forced replacement failure")

    with pytest.raises(gate_io.GateIoError, match="forced replacement failure"):
        gate_io.publish_success(
            temporary,
            marker,
            _gate_io_payload(),
            after_link=replace_marker_and_fail,
        )

    after = os.lstat(marker)
    assert marker.read_bytes() == b"unrelated-replacement\n"
    assert (after.st_dev, after.st_ino) == (replacement["dev"], replacement["ino"])
    assert not temporary.exists() and not temporary.is_symlink()


def test_gate_io_cleans_link_that_succeeds_before_link_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    temporary = attempt / ".SUCCESS.12345.tmp"
    marker = attempt / "SUCCESS"
    real_link = gate_io.os.link

    def link_then_raise(*args: Any, **kwargs: Any) -> None:
        real_link(*args, **kwargs)
        raise OSError("injected post-link syscall error")

    monkeypatch.setattr(gate_io.os, "link", link_then_raise)
    with pytest.raises(OSError, match="injected post-link syscall error"):
        gate_io.publish_success(temporary, marker, _gate_io_payload())

    assert not temporary.exists() and not temporary.is_symlink()
    assert not marker.exists() and not marker.is_symlink()


def test_gate_io_cleans_temp_after_initial_metadata_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    temporary = attempt / ".SUCCESS.12345.tmp"
    marker = attempt / "SUCCESS"
    real_fstat = gate_io.os.fstat
    call_count = 0

    def first_fstat_has_extra_link(descriptor: int) -> Any:
        nonlocal call_count
        call_count += 1
        metadata = real_fstat(descriptor)
        if call_count == 1:
            return SimpleNamespace(
                st_dev=metadata.st_dev,
                st_ino=metadata.st_ino,
                st_mode=metadata.st_mode,
                st_nlink=2,
            )
        return metadata

    monkeypatch.setattr(gate_io.os, "fstat", first_fstat_has_extra_link)
    with pytest.raises(gate_io.GateIoError, match="temp metadata drift"):
        gate_io.publish_success(temporary, marker, _gate_io_payload())

    assert not temporary.exists() and not temporary.is_symlink()
    assert not marker.exists() and not marker.is_symlink()


def test_gate_io_rejects_directory_symlink_with_or_without_trailing_slash(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    cache = tmp_path / "cache"
    output.mkdir()
    cache.mkdir()
    output_link = tmp_path / "output-link"
    output_link.symlink_to(output, target_is_directory=True)

    for supplied_output in (str(output_link), f"{output_link}/"):
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "eef_pose_reasoning_production_v4_core_gate_io.py"),
                "validate-roots",
                "--output-root",
                supplied_output,
                "--cache-root",
                str(cache),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 1
        assert "must be a non-symlink directory" in completed.stderr


def test_gate_io_accepts_normal_disjoint_roots(tmp_path: Path) -> None:
    output = tmp_path / "output"
    cache = tmp_path / "cache"
    output.mkdir()
    cache.mkdir()

    assert gate_io.validate_disjoint_roots(str(output), str(cache)) == (
        str(output.resolve()),
        str(cache.resolve()),
    )
    assert gate_io.validate_disjoint_roots(f"{output}/", f"{cache}/") == (
        str(output.resolve()),
        str(cache.resolve()),
    )


def test_srun_wrapper_is_single_variant_and_launch_only() -> None:
    source = (
        SCRIPTS / "run_eef_pose_reasoning_production_v4_core_replay_srun.sh"
    ).read_text()
    gate_io_sha256 = hashlib.sha256(
        (SCRIPTS / "eef_pose_reasoning_production_v4_core_gate_io.py").read_bytes()
    ).hexdigest()
    assert "production_v4_core_ramp16" in source
    assert "cap8_abrupt_release" not in source
    assert "cap24_abrupt_release" not in source
    assert "sbatch " not in source
    assert "#SBATCH" not in source
    assert "set_dof_max_velocities" not in source
    assert "ISAAC_PYTEST_EXIT_CODE_FILE" in source
    assert "tests/test_robust_differential_ik.py" in source
    assert "tests/test_smoke_eef_pose_reasoning_production_v4_core_replay.py" in source
    assert "site-packages/isaaclab/source/isaaclab" in source
    assert "site-packages/isaaclab/source/isaaclab_tasks" in source
    assert "site-packages/isaaclab/source/isaaclab_assets" in source
    assert "trap finish EXIT" in source
    assert 'rm -rf -- "${cache}"' in source
    assert "status --porcelain=v1 --untracked-files=all" in source
    assert "FULLTRACE_SAFETY_VALIDATOR_SHA256" in source
    assert "FULLTRACE_GATE_IO_SHA256" in source
    assert "eef_pose_reasoning_production_v4_core_gate_io.py" in source
    assert f"readonly gate_io_sha256={gate_io_sha256}" in source
    assert '"${gate_io}" validate-roots' in source
    assert 'mkdir -- "${attempt}"' in source
    assert 'mkdir -- "${cache}"' in source
    assert 'if ! /usr/bin/python3 "${gate_io}" publish-success \\' in source
    assert "read -r success_size success_sha < <(" not in source
    assert 'rm -f "${success_marker}"' not in source
    assert 'mv -T "${success_temporary}"' not in source
    assert 'rev-parse HEAD^)" == "${replay_validation_fix_commit}"' in source
    assert 'rev-parse HEAD^^)" == "${replay_implementation_commit}"' in source
    assert 'rev-parse HEAD^^^)" == "${replay_parent_commit}"' in source
    assert 'rev-parse HEAD^^^^)" == "${production_base_commit}"' in source
    assert source.endswith(
        "status=0\nprintf 'FULLTRACE_PRODUCTION_V4_SUCCESS=%s\\n' "
        '"${success_marker}" || true\n'
    )


def test_fixture_builder_still_reproduces_pinned_action_payload() -> None:
    payload = json.loads(replay.FIXTURE_PATH.read_text())
    assert payload["fixture_profile"] == fixture_builder.FIXTURE_PROFILE
    assert payload["source"]["trace_sha256"] == fixture_builder.TRACE_SHA256
    assert payload["source"]["event_counts"] == fixture_builder.EXPECTED_EVENTS
    assert payload["action_encoding"] == replay.ACTION_ENCODING
