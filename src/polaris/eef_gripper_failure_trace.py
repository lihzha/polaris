"""Observational all-six gripper tail for EEF controller canaries."""

from __future__ import annotations

import math
import struct
from typing import Any

from polaris.eef_gripper_runtime import GRIPPER_JOINT_INDICES
from polaris.eef_gripper_runtime import GRIPPER_JOINT_NAMES
from polaris.eef_gripper_runtime import GRIPPER_CLOSED_TARGET_FLOAT32
from polaris.eef_gripper_runtime import GRIPPER_OPEN_TARGET_FLOAT32
from polaris.eef_gripper_runtime import EEF_GRIPPER_TARGET_SLEW_ACTION_CLASS


EEF_ALL_SIX_GRIPPER_TRACE_PROFILE = "polaris_eef_all_six_gripper_substep_tail_v2"
EEF_ALL_SIX_GRIPPER_TRACE_CAPACITY = 64
EEF_ALL_SIX_GRIPPER_TRACE_DECIMATION = 8
EEF_ALL_SIX_GRIPPER_TRACE_SNAPSHOT_FIELDS = {
    "joint_pos_rad",
    "joint_vel_rad_s",
    "joint_acc_rad_s2",
    "joint_pos_target_rad",
    "joint_vel_target_rad_s",
    "joint_effort_target_nm",
}
EEF_ALL_SIX_GRIPPER_TRACE_ENTRY_FIELDS = {
    "apply_index",
    "policy_step",
    "physics_substep",
    "raw_action",
    "requested_endpoint_rad",
    "pre",
    "target_after_setter_rad",
    "post",
}
EEF_ALL_SIX_GRIPPER_TRACE_FIELDS = {
    "schema_version",
    "profile",
    "episode_index",
    "capacity",
    "decimation",
    "joint_names",
    "joint_indices",
    "process_action_calls",
    "total_apply_entries",
    "dropped_entries",
    "initial_snapshot",
    "entries",
    "terminal_snapshot",
    "numerical_failure",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _same_float32_bits(left: float, right: float) -> bool:
    try:
        return struct.pack("<f", left) == struct.pack("<f", right)
    except (OverflowError, struct.error):
        return False


def _tensor_vector(value: Any) -> list[float]:
    result = [float(item) for item in value.detach().cpu().tolist()]
    _require(
        len(result) == len(GRIPPER_JOINT_INDICES)
        and all(math.isfinite(item) for item in result),
        "PolaRiS EEF all-six gripper trace vector drift",
    )
    return result


def make_eef_all_six_gripper_failure_trace_class(base_class: type) -> type:
    """Wrap the production action with read-only pre/post state capture."""

    _require(isinstance(base_class, type), "PolaRiS EEF gripper trace base class")

    class TracingEefBinaryJointPositionTargetSlewAction(base_class):
        eef_all_six_gripper_trace_profile = EEF_ALL_SIX_GRIPPER_TRACE_PROFILE

        def __init__(self, cfg: Any, env: Any) -> None:
            super().__init__(cfg, env)
            live_names = list(self._asset.data.joint_names)
            _require(
                [live_names[index] for index in GRIPPER_JOINT_INDICES]
                == list(GRIPPER_JOINT_NAMES),
                "PolaRiS EEF all-six gripper trace joint ordering drift",
            )
            self._eef_trace_reset()

        def _eef_trace_reset(self) -> None:
            self._eef_trace_episode_index: int | None = None
            self._eef_trace_policy_step: int | None = None
            self._eef_trace_physics_substep = 0
            self._eef_trace_process_calls = 0
            self._eef_trace_total_apply_entries = 0
            self._eef_trace_entries: list[dict[str, Any]] = []
            self._eef_trace_pending_entry: dict[str, Any] | None = None
            self._eef_trace_raw_action: float | None = None
            self._eef_trace_requested_endpoint: float | None = None
            self._eef_trace_initial_snapshot: dict[str, Any] | None = None
            self._eef_trace_terminal_snapshot: dict[str, Any] | None = None
            self._eef_trace_numerical_failure: bool | None = None

        def reset(self, env_ids: Any = None) -> None:
            super().reset(env_ids=env_ids)
            if hasattr(self, "_eef_trace_entries"):
                self._eef_trace_reset()

        def begin_eef_policy_step(
            self, *, episode_index: int, policy_step: int
        ) -> None:
            _require(
                type(episode_index) is int
                and episode_index >= 0
                and type(policy_step) is int
                and policy_step >= 0,
                "PolaRiS EEF gripper trace policy identity drift",
            )
            self._eef_trace_finalize_pending()
            if self._eef_trace_episode_index is None:
                self._eef_trace_episode_index = episode_index
                self._eef_trace_initial_snapshot = self._eef_trace_snapshot()
            _require(
                self._eef_trace_episode_index == episode_index
                and self._eef_trace_initial_snapshot is not None
                and (
                    self._eef_trace_policy_step is None
                    or self._eef_trace_physics_substep
                    == EEF_ALL_SIX_GRIPPER_TRACE_DECIMATION
                )
                and policy_step == self._eef_trace_process_calls,
                "PolaRiS EEF gripper trace policy cadence drift",
            )
            self._eef_trace_policy_step = policy_step
            self._eef_trace_physics_substep = 0

        def process_actions(self, actions: Any) -> None:
            super().process_actions(actions)
            _require(
                self._eef_trace_policy_step is not None,
                "PolaRiS EEF gripper trace process lacks policy context",
            )
            raw = float(self._raw_actions[0, 0].detach().cpu().item())
            endpoint = float(self._processed_actions[0, 0].detach().cpu().item())
            _require(
                math.isfinite(raw) and math.isfinite(endpoint),
                "PolaRiS EEF gripper trace endpoint drift",
            )
            self._eef_trace_raw_action = raw
            self._eef_trace_requested_endpoint = endpoint
            self._eef_trace_process_calls += 1

        def _eef_trace_snapshot(self) -> dict[str, Any]:
            data = self._asset.data
            indices = list(GRIPPER_JOINT_INDICES)
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

        def _eef_trace_finalize_pending(self) -> None:
            if self._eef_trace_pending_entry is None:
                return
            self._eef_trace_pending_entry["post"] = self._eef_trace_snapshot()
            self._eef_trace_entries.append(self._eef_trace_pending_entry)
            if len(self._eef_trace_entries) > EEF_ALL_SIX_GRIPPER_TRACE_CAPACITY:
                del self._eef_trace_entries[0]
            self._eef_trace_pending_entry = None
            self._eef_trace_total_apply_entries += 1

        def apply_actions(self) -> None:
            self._eef_trace_finalize_pending()
            _require(
                self._eef_trace_policy_step is not None
                and self._eef_trace_raw_action is not None
                and self._eef_trace_requested_endpoint is not None
                and 0
                <= self._eef_trace_physics_substep
                < EEF_ALL_SIX_GRIPPER_TRACE_DECIMATION,
                "PolaRiS EEF gripper trace apply cadence drift",
            )
            pre = self._eef_trace_snapshot()
            super().apply_actions()
            target = float(
                self._asset.data.joint_pos_target[0, GRIPPER_JOINT_INDICES[0]]
                .detach()
                .cpu()
                .item()
            )
            _require(
                math.isfinite(target),
                "PolaRiS EEF gripper trace target setter drift",
            )
            self._eef_trace_pending_entry = {
                "apply_index": self._eef_trace_total_apply_entries,
                "policy_step": self._eef_trace_policy_step,
                "physics_substep": self._eef_trace_physics_substep,
                "raw_action": self._eef_trace_raw_action,
                "requested_endpoint_rad": self._eef_trace_requested_endpoint,
                "pre": pre,
                "target_after_setter_rad": target,
                "post": None,
            }
            self._eef_trace_physics_substep += 1

        def finalize_eef_rollout_trace(self, *, numerical_failure: bool) -> None:
            _require(
                type(numerical_failure) is bool
                and self._eef_trace_terminal_snapshot is None,
                "PolaRiS EEF gripper trace finalization drift",
            )
            self._eef_trace_finalize_pending()
            self._eef_trace_terminal_snapshot = self._eef_trace_snapshot()
            self._eef_trace_numerical_failure = numerical_failure

        def eef_all_six_gripper_trace(self) -> dict[str, Any]:
            _require(
                self._eef_trace_pending_entry is None
                and self._eef_trace_episode_index is not None
                and self._eef_trace_terminal_snapshot is not None
                and self._eef_trace_numerical_failure is not None,
                "PolaRiS EEF gripper trace is not finalized",
            )
            return {
                "schema_version": 1,
                "profile": EEF_ALL_SIX_GRIPPER_TRACE_PROFILE,
                "episode_index": self._eef_trace_episode_index,
                "capacity": EEF_ALL_SIX_GRIPPER_TRACE_CAPACITY,
                "decimation": EEF_ALL_SIX_GRIPPER_TRACE_DECIMATION,
                "joint_names": list(GRIPPER_JOINT_NAMES),
                "joint_indices": list(GRIPPER_JOINT_INDICES),
                "process_action_calls": self._eef_trace_process_calls,
                "total_apply_entries": self._eef_trace_total_apply_entries,
                "dropped_entries": (
                    self._eef_trace_total_apply_entries - len(self._eef_trace_entries)
                ),
                "initial_snapshot": self._eef_trace_initial_snapshot,
                "entries": list(self._eef_trace_entries),
                "terminal_snapshot": self._eef_trace_terminal_snapshot,
                "numerical_failure": self._eef_trace_numerical_failure,
            }

    # The production gripper installer requires this exact action class name.
    TracingEefBinaryJointPositionTargetSlewAction.__name__ = (
        EEF_GRIPPER_TARGET_SLEW_ACTION_CLASS
    )
    TracingEefBinaryJointPositionTargetSlewAction.__qualname__ = (
        EEF_GRIPPER_TARGET_SLEW_ACTION_CLASS
    )
    return TracingEefBinaryJointPositionTargetSlewAction


def _validate_snapshot(value: Any, *, field: str) -> dict[str, Any]:
    _require(
        isinstance(value, dict)
        and set(value) == EEF_ALL_SIX_GRIPPER_TRACE_SNAPSHOT_FIELDS,
        f"{field} schema drift",
    )
    for name in EEF_ALL_SIX_GRIPPER_TRACE_SNAPSHOT_FIELDS:
        vector = value[name]
        _require(
            isinstance(vector, list)
            and len(vector) == len(GRIPPER_JOINT_INDICES)
            and all(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
                for item in vector
            ),
            f"{field}.{name} drift",
        )
    return dict(value)


def validate_eef_all_six_gripper_trace(
    value: Any,
    *,
    episode_index: int,
    episode_length: int,
    numerical_failure: bool,
    expected_apply_calls: int,
) -> dict[str, Any]:
    """Validate one durable success/failure all-six causal tail."""

    _require(
        type(episode_index) is int
        and episode_index >= 0
        and type(episode_length) is int
        and episode_length >= 1
        and type(numerical_failure) is bool
        and type(expected_apply_calls) is int
        and expected_apply_calls >= 0,
        "PolaRiS EEF all-six gripper validation inputs drift",
    )
    _require(
        isinstance(value, dict) and set(value) == EEF_ALL_SIX_GRIPPER_TRACE_FIELDS,
        "PolaRiS EEF all-six gripper trace schema drift",
    )
    for field in (
        "schema_version",
        "episode_index",
        "capacity",
        "decimation",
        "process_action_calls",
        "total_apply_entries",
        "dropped_entries",
    ):
        _require(
            type(value.get(field)) is int and value[field] >= 0,
            f"PolaRiS EEF all-six gripper trace {field} type drift",
        )
    _require(
        value.get("schema_version") == 1
        and value.get("profile") == EEF_ALL_SIX_GRIPPER_TRACE_PROFILE
        and value.get("episode_index") == episode_index
        and value.get("capacity") == EEF_ALL_SIX_GRIPPER_TRACE_CAPACITY
        and value.get("decimation") == EEF_ALL_SIX_GRIPPER_TRACE_DECIMATION
        and value.get("joint_names") == list(GRIPPER_JOINT_NAMES)
        and value.get("joint_indices") == list(GRIPPER_JOINT_INDICES)
        and value.get("process_action_calls") == episode_length
        and value.get("total_apply_entries") == expected_apply_calls
        and value.get("numerical_failure") is numerical_failure,
        "PolaRiS EEF all-six gripper trace identity/cadence drift",
    )
    if numerical_failure:
        lower = (episode_length - 1) * EEF_ALL_SIX_GRIPPER_TRACE_DECIMATION
        upper = episode_length * EEF_ALL_SIX_GRIPPER_TRACE_DECIMATION
        _require(
            lower <= expected_apply_calls < upper,
            "PolaRiS EEF all-six gripper failure cadence drift",
        )
    else:
        _require(
            expected_apply_calls
            == episode_length * EEF_ALL_SIX_GRIPPER_TRACE_DECIMATION,
            "PolaRiS EEF all-six gripper success cadence drift",
        )
    initial = _validate_snapshot(
        value.get("initial_snapshot"), field="gripper trace initial"
    )
    entries = value.get("entries")
    expected_count = min(expected_apply_calls, EEF_ALL_SIX_GRIPPER_TRACE_CAPACITY)
    _require(
        isinstance(entries, list)
        and len(entries) == expected_count
        and value.get("dropped_entries") == expected_apply_calls - expected_count,
        "PolaRiS EEF all-six gripper trace retention drift",
    )
    first = expected_apply_calls - expected_count
    for offset, entry in enumerate(entries):
        _require(
            isinstance(entry, dict)
            and set(entry) == EEF_ALL_SIX_GRIPPER_TRACE_ENTRY_FIELDS,
            "PolaRiS EEF all-six gripper trace entry schema drift",
        )
        apply_index = first + offset
        _require(
            type(entry.get("apply_index")) is int
            and type(entry.get("policy_step")) is int
            and type(entry.get("physics_substep")) is int
            and entry.get("apply_index") == apply_index
            and entry.get("policy_step")
            == apply_index // EEF_ALL_SIX_GRIPPER_TRACE_DECIMATION
            and entry.get("physics_substep")
            == apply_index % EEF_ALL_SIX_GRIPPER_TRACE_DECIMATION,
            "PolaRiS EEF all-six gripper trace entry cadence drift",
        )
        for scalar in (
            "raw_action",
            "requested_endpoint_rad",
            "target_after_setter_rad",
        ):
            item = entry.get(scalar)
            _require(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item)),
                f"PolaRiS EEF all-six gripper trace {scalar} drift",
            )
        raw_action = float(entry["raw_action"])
        requested_endpoint = float(entry["requested_endpoint_rad"])
        raw_is_open = _same_float32_bits(raw_action, 0.0)
        raw_is_closed = _same_float32_bits(raw_action, 1.0)
        expected_endpoint = (
            GRIPPER_CLOSED_TARGET_FLOAT32
            if raw_is_closed
            else GRIPPER_OPEN_TARGET_FLOAT32
        )
        if raw_is_open is raw_is_closed or not _same_float32_bits(
            requested_endpoint, expected_endpoint
        ):
            raise ValueError("PolaRiS EEF all-six gripper binary endpoint drift")
        pre = _validate_snapshot(entry.get("pre"), field="gripper trace pre")
        post = _validate_snapshot(entry.get("post"), field="gripper trace post")
        if offset == 0 and first == 0 and pre != initial:
            raise ValueError("PolaRiS EEF all-six gripper initial identity drift")
        if offset > 0 and pre != entries[offset - 1]["post"]:
            raise ValueError("PolaRiS EEF all-six gripper transition continuity drift")
        if entry["target_after_setter_rad"] != post["joint_pos_target_rad"][0]:
            raise ValueError("PolaRiS EEF all-six gripper setter target drift")
    terminal = _validate_snapshot(
        value.get("terminal_snapshot"), field="gripper trace terminal"
    )
    if expected_apply_calls == 0 and terminal != initial:
        raise ValueError("PolaRiS EEF all-six gripper empty terminal identity drift")
    if entries and terminal != entries[-1]["post"]:
        raise ValueError("PolaRiS EEF all-six gripper terminal identity drift")
    return dict(value)
