from __future__ import annotations

import ast
import copy
import importlib.util
import inspect
import json
from pathlib import Path
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


def test_srun_wrapper_is_single_variant_and_launch_only() -> None:
    source = (
        SCRIPTS / "run_eef_pose_reasoning_production_v4_core_replay_srun.sh"
    ).read_text()
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


def test_fixture_builder_still_reproduces_pinned_action_payload() -> None:
    payload = json.loads(replay.FIXTURE_PATH.read_text())
    assert payload["fixture_profile"] == fixture_builder.FIXTURE_PROFILE
    assert payload["source"]["trace_sha256"] == fixture_builder.TRACE_SHA256
    assert payload["source"]["event_counts"] == fixture_builder.EXPECTED_EVENTS
    assert payload["action_encoding"] == replay.ACTION_ENCODING
