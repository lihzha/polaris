from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import pytest

from scripts import finalize_eef_pose_boundary_replay as finalizer
from scripts import smoke_eef_pose_boundary_replay as smoke


class _FakeTensor:
    def __init__(self, values):
        self._values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def to(self, _device):
        return self

    def flatten(self):
        return self

    def tolist(self):
        return self._values

    def __getitem__(self, _index):
        return self

    def __sub__(self, other):
        return _FakeTensor(
            [
                left - right
                for left, right in zip(self._values, other._values, strict=True)
            ]
        )


def _trace_vector(values):
    values = list(values)
    return {
        "values": values,
        "finite_mask": [True] * len(values),
        "finite_count": len(values),
    }


def _failure_trace_fixture():
    q0 = [0.0] * 7
    q1 = [0.0625] * 7
    q2 = [0.125] * 7
    dq0 = [0.0] * 7
    dq1 = [0.25] * 7
    dq2 = [3.0] + [0.0] * 6
    targets = ([0.0] * 7, [0.0625] * 7, [0.125] * 7)
    entries = []
    for apply_index, (q_pre, q_post, dq_pre, dq_post) in enumerate(
        ((q0, q1, dq0, dq1), (q1, q2, dq1, dq2))
    ):
        q_target = targets[apply_index + 1]
        velocity_target = [0.0] * 7
        effort_target = [0.0] * 7
        preclip_effort = []
        postclip_effort = []
        for joint_index in range(7):
            position_error = smoke._float32_subtract(  # noqa: SLF001
                q_target[joint_index], q_pre[joint_index]
            )
            velocity_error = smoke._float32_subtract(  # noqa: SLF001
                velocity_target[joint_index], dq_pre[joint_index]
            )
            position_term = smoke._float32_multiply(  # noqa: SLF001
                smoke.EXPECTED_JOINT_DRIVE_STIFFNESS[joint_index], position_error
            )
            velocity_term = smoke._float32_multiply(  # noqa: SLF001
                smoke.EXPECTED_JOINT_DRIVE_DAMPING[joint_index], velocity_error
            )
            preclip = smoke._float32_add(  # noqa: SLF001
                smoke._float32_add(position_term, velocity_term),  # noqa: SLF001
                effort_target[joint_index],
            )
            effort_limit = smoke.EXPECTED_EFFORT_LIMITS[joint_index]
            preclip_effort.append(preclip)
            postclip_effort.append(min(max(preclip, -effort_limit), effort_limit))
        vectors = {
            "joint_pos_rad": q_pre,
            "joint_vel_rad_s": dq_pre,
            "post_joint_pos_rad": q_post,
            "post_joint_vel_rad_s": dq_post,
            "delta_joint_pos_rad": [
                smoke._float32_subtract(post, pre)  # noqa: SLF001
                for pre, post in zip(q_pre, q_post, strict=True)
            ],
            "delta_joint_vel_rad_s": [
                smoke._float32_subtract(post, pre)  # noqa: SLF001
                for pre, post in zip(dq_pre, dq_post, strict=True)
            ],
            "previous_joint_pos_target_rad": targets[apply_index],
            "raw_dls_joint_pos_target_rad": q_target,
            "new_joint_pos_target_rad": q_target,
            "new_joint_vel_target_rad_s": velocity_target,
            "new_joint_effort_target_nm": effort_target,
            "current_eef_position_m": [0.4, -0.2, 0.3],
            "current_eef_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "desired_eef_position_m": [0.41, -0.2, 0.3],
            "desired_eef_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "pose_error_position_m_axis_angle_rad": [0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
            "approximate_pd_effort_preclip_nm": preclip_effort,
            "approximate_pd_effort_postclip_nm": postclip_effort,
        }
        entries.append(
            {
                "apply_index": apply_index,
                "policy_step": 0,
                "physics_substep": apply_index,
                **{name: _trace_vector(value) for name, value in vectors.items()},
            }
        )
    return {
        "schema_version": 1,
        "profile": smoke.FAILURE_SUBSTEP_TRACE_PROFILE,
        "episode_index": 0,
        "capacity": smoke.FAILURE_SUBSTEP_TRACE_CAPACITY,
        "policy_step_capacity": smoke.FAILURE_SUBSTEP_TRACE_CAPACITY
        // smoke.DECIMATION,
        "decimation": smoke.DECIMATION,
        "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
        "joint_drive_stiffness": list(smoke.EXPECTED_JOINT_DRIVE_STIFFNESS),
        "joint_drive_damping": list(smoke.EXPECTED_JOINT_DRIVE_DAMPING),
        "joint_effort_limits": list(smoke.EXPECTED_EFFORT_LIMITS),
        "effort_semantics": smoke.FAILURE_SUBSTEP_TRACE_EFFORT_SEMANTICS,
        "phase_contract": copy.deepcopy(smoke.FAILURE_SUBSTEP_TRACE_PHASE_CONTRACT),
        "completed_entry_count": 2,
        "total_completed_entry_count": 2,
        "dropped_prefix_entry_count": 0,
        "pending_entry_count": 0,
        "pending_apply_index": None,
        "entries": entries,
    }


def _failure_trace_safety():
    return {
        "counters": {
            "apply_calls": 3,
            "invariant_aborts": 1,
            "current_joint_limit_aborts": 0,
            "nonfinite_aborts": 0,
        },
        "guard_diagnostics": [
            {
                "kind": "current_joint_velocity_limit_abort",
                "episode_index": 0,
                "policy_step": 0,
                "physics_substep": 2,
                "joint_pos_rad": _trace_vector([0.125] * 7),
                "raw_delta_joint_pos_rad": None,
                "raw_joint_pos_target_rad": None,
                "safe_joint_pos_target_rad": None,
                "pose_error_norm": None,
                "jacobian_finite": None,
                "jacobian_max_abs": None,
                "eef_quaternion_norm": None,
            }
        ],
    }


def test_failure_vector_evidence_replaces_nonfinite_values_with_null() -> None:
    evidence = smoke._finite_vector_evidence(  # noqa: SLF001
        _FakeTensor([1.0, float("nan"), float("inf"), -2.0])
    )

    assert evidence == {
        "values": [1.0, None, None, -2.0],
        "finite_mask": [True, False, False, True],
        "finite_count": 2,
    }
    assert b"NaN" not in smoke._strict_json_bytes(evidence)  # noqa: SLF001


def test_failure_runtime_capture_uses_raw_action_term_report() -> None:
    expected_report = _failure_trace_safety()
    expected_trace = _failure_trace_fixture()

    class _ArmTerm:
        _joint_ids = list(range(7))
        _joint_names = [f"panda_joint{index}" for index in range(1, 8)]

        def episode_safety_report(self, episode_index):
            assert episode_index == 0
            return expected_report

        def failure_substep_trace(self, episode_index):
            assert episode_index == 0
            return expected_trace

    class _RobotData:
        joint_pos = _FakeTensor([0.125] * 7)
        joint_vel = _FakeTensor([3.0] + [0.0] * 6)
        joint_pos_target = _FakeTensor([0.125] * 7)
        joint_vel_target = _FakeTensor([0.0] * 7)
        joint_effort_target = _FakeTensor([0.0] * 7)
        computed_torque = _FakeTensor(
            expected_trace["entries"][-1]["approximate_pd_effort_preclip_nm"]["values"]
        )
        applied_torque = _FakeTensor(
            expected_trace["entries"][-1]["approximate_pd_effort_postclip_nm"]["values"]
        )
        _sim_timestamp = 1.25

    class _RootView:
        @staticmethod
        def get_dof_positions():
            return _FakeTensor([0.125] * 7)

        @staticmethod
        def get_dof_velocities():
            return _FakeTensor([3.0] + [0.0] * 6)

        @staticmethod
        def get_dof_max_velocities():
            return _FakeTensor(list(smoke.EXPECTED_VELOCITY_LIMITS_RAD_S))

        @staticmethod
        def get_dof_max_forces():
            return _FakeTensor(list(smoke.EXPECTED_EFFORT_LIMITS))

    class _Robot:
        data = _RobotData()
        root_physx_view = _RootView()
        device = "cpu"

    class _ActionManager:
        _terms = {"arm": _ArmTerm()}

    class _Environment:
        unwrapped = None
        action_manager = _ActionManager()
        scene = {"robot": _Robot()}

    environment = _Environment()
    environment.unwrapped = environment

    evidence = smoke._capture_failure_runtime_evidence(  # noqa: SLF001
        environment,
        policy_step=0,
    )

    assert evidence["policy_step"] == 0
    assert evidence["arm_joint_vel_rad_s"]["values"] == [3.0] + [0.0] * 6
    assert evidence["physx_arm_joint_vel_rad_s"]["values"] == [3.0] + [0.0] * 6
    assert evidence["cached_minus_physx_arm_joint_vel_rad_s"]["values"] == [0.0] * 7
    assert evidence["articulation_data_sim_timestamp"] == 1.25
    assert (
        evidence["arm_computed_torque"]
        == expected_trace["entries"][-1]["approximate_pd_effort_preclip_nm"]
    )
    assert evidence["arm_joint_velocity_target_rad_s"]["values"] == [0.0] * 7
    assert evidence["arm_joint_effort_target_nm"]["values"] == [0.0] * 7
    assert evidence["ik_safety"] is expected_report
    assert evidence["controller_substep_trace"] is expected_trace
    assert evidence["controller_substep_trace_error"] is None


def test_failure_substep_trace_accepts_causal_finite_ring() -> None:
    trace = _failure_trace_fixture()

    validated = smoke.validate_failure_substep_trace(
        trace,
        safety=_failure_trace_safety(),
        failure_policy_step=0,
        current_joint_pos=_trace_vector([0.125] * 7),
        current_joint_vel=_trace_vector([3.0] + [0.0] * 6),
        current_joint_pos_target=_trace_vector([0.125] * 7),
        current_joint_vel_target=_trace_vector([0.0] * 7),
        current_joint_effort_target=_trace_vector([0.0] * 7),
        current_approximate_pd_effort_preclip=trace["entries"][-1][
            "approximate_pd_effort_preclip_nm"
        ],
        current_approximate_pd_effort_postclip=trace["entries"][-1][
            "approximate_pd_effort_postclip_nm"
        ],
        physx_joint_pos=_trace_vector([0.125] * 7),
        physx_joint_vel=_trace_vector([3.0] + [0.0] * 6),
    )

    assert validated is trace
    assert b"NaN" not in smoke._strict_json_bytes(trace)  # noqa: SLF001


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda trace: trace.__setitem__("extra", True), "schema"),
        (lambda trace: trace.__setitem__("capacity", 32), "capacity"),
        (
            lambda trace: trace.__setitem__("phase_contract", {}),
            "phase contract",
        ),
        (lambda trace: trace["entries"].reverse(), "apply index"),
        (
            lambda trace: trace["entries"][0].__setitem__("physics_substep", 7),
            "physics substep",
        ),
        (
            lambda trace: trace["entries"][0]["joint_pos_rad"]["values"].pop(),
            "width",
        ),
        (
            lambda trace: trace["entries"][0]["approximate_pd_effort_preclip_nm"][
                "finite_mask"
            ].__setitem__(0, False),
            "finite-mask count",
        ),
        (
            lambda trace: (
                trace["entries"][0]["approximate_pd_effort_preclip_nm"][
                    "values"
                ].__setitem__(0, None),
                trace["entries"][0]["approximate_pd_effort_preclip_nm"][
                    "finite_mask"
                ].__setitem__(0, False),
                trace["entries"][0]["approximate_pd_effort_preclip_nm"].__setitem__(
                    "finite_count", 6
                ),
            ),
            "must be fully finite",
        ),
        (
            lambda trace: trace["entries"][0]["delta_joint_pos_rad"][
                "values"
            ].__setitem__(0, 0.0),
            "joint_pos delta",
        ),
        (
            lambda trace: (
                trace["entries"][0]["delta_joint_pos_rad"]["values"].__setitem__(
                    0, None
                ),
                trace["entries"][0]["delta_joint_pos_rad"]["finite_mask"].__setitem__(
                    0, False
                ),
                trace["entries"][0]["delta_joint_pos_rad"].__setitem__(
                    "finite_count", 6
                ),
            ),
            "must be fully finite",
        ),
        (
            lambda trace: trace["entries"][1]["previous_joint_pos_target_rad"][
                "values"
            ].__setitem__(0, 0.0),
            "target transition continuity",
        ),
        (
            lambda trace: (
                trace.__setitem__("pending_entry_count", 1),
                trace.__setitem__("pending_apply_index", 2),
            ),
            "must not have a pending command",
        ),
    ],
)
def test_failure_substep_trace_rejects_schema_phase_and_causality_mutations(
    mutation, message
) -> None:
    trace = _failure_trace_fixture()
    mutation(trace)

    with pytest.raises(smoke.BoundaryReplayValidationError, match=message):
        smoke.validate_failure_substep_trace(
            trace,
            safety=_failure_trace_safety(),
            failure_policy_step=0,
            current_joint_pos=_trace_vector([0.125] * 7),
            current_joint_vel=_trace_vector([3.0] + [0.0] * 6),
            current_joint_pos_target=_trace_vector([0.125] * 7),
            current_joint_vel_target=_trace_vector([0.0] * 7),
            current_joint_effort_target=_trace_vector([0.0] * 7),
            current_approximate_pd_effort_preclip=trace["entries"][-1][
                "approximate_pd_effort_preclip_nm"
            ],
            current_approximate_pd_effort_postclip=trace["entries"][-1][
                "approximate_pd_effort_postclip_nm"
            ],
            physx_joint_pos=_trace_vector([0.125] * 7),
            physx_joint_vel=_trace_vector([3.0] + [0.0] * 6),
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("approximate_pd_effort_preclip_nm", "PD preclip effort"),
        ("approximate_pd_effort_postclip_nm", "PD postclip effort"),
    ],
)
def test_failure_substep_trace_rejects_arbitrary_effort(field, message) -> None:
    trace = _failure_trace_fixture()
    trace["entries"][1][field]["values"][0] += 1.0

    with pytest.raises(smoke.BoundaryReplayValidationError, match=message):
        smoke.validate_failure_substep_trace(
            trace,
            safety=_failure_trace_safety(),
            failure_policy_step=0,
            current_joint_pos=_trace_vector([0.125] * 7),
            current_joint_vel=_trace_vector([3.0] + [0.0] * 6),
            current_joint_pos_target=_trace_vector([0.125] * 7),
            current_joint_vel_target=_trace_vector([0.0] * 7),
            current_joint_effort_target=_trace_vector([0.0] * 7),
            current_approximate_pd_effort_preclip=trace["entries"][-1][
                "approximate_pd_effort_preclip_nm"
            ],
            current_approximate_pd_effort_postclip=trace["entries"][-1][
                "approximate_pd_effort_postclip_nm"
            ],
            physx_joint_pos=_trace_vector([0.125] * 7),
            physx_joint_vel=_trace_vector([3.0] + [0.0] * 6),
        )


def test_failure_substep_trace_rejects_guard_and_live_target_mismatch() -> None:
    trace = _failure_trace_fixture()
    safety = _failure_trace_safety()
    safety["guard_diagnostics"].append(copy.deepcopy(safety["guard_diagnostics"][0]))
    with pytest.raises(smoke.BoundaryReplayValidationError, match="exactly one"):
        smoke.validate_failure_substep_trace(
            trace,
            safety=safety,
            failure_policy_step=0,
            current_joint_pos=_trace_vector([0.125] * 7),
            current_joint_vel=_trace_vector([3.0] + [0.0] * 6),
            current_joint_pos_target=_trace_vector([0.125] * 7),
            current_joint_vel_target=_trace_vector([0.0] * 7),
            current_joint_effort_target=_trace_vector([0.0] * 7),
            current_approximate_pd_effort_preclip=trace["entries"][-1][
                "approximate_pd_effort_preclip_nm"
            ],
            current_approximate_pd_effort_postclip=trace["entries"][-1][
                "approximate_pd_effort_postclip_nm"
            ],
            physx_joint_pos=_trace_vector([0.125] * 7),
            physx_joint_vel=_trace_vector([3.0] + [0.0] * 6),
        )

    with pytest.raises(smoke.BoundaryReplayValidationError, match="position-target"):
        smoke.validate_failure_substep_trace(
            trace,
            safety=_failure_trace_safety(),
            failure_policy_step=0,
            current_joint_pos=_trace_vector([0.125] * 7),
            current_joint_vel=_trace_vector([3.0] + [0.0] * 6),
            current_joint_pos_target=_trace_vector([0.25] * 7),
            current_joint_vel_target=_trace_vector([0.0] * 7),
            current_joint_effort_target=_trace_vector([0.0] * 7),
            current_approximate_pd_effort_preclip=trace["entries"][-1][
                "approximate_pd_effort_preclip_nm"
            ],
            current_approximate_pd_effort_postclip=trace["entries"][-1][
                "approximate_pd_effort_postclip_nm"
            ],
            physx_joint_pos=_trace_vector([0.125] * 7),
            physx_joint_vel=_trace_vector([3.0] + [0.0] * 6),
        )


def _diagnostic_vector(values):
    return {
        "values": list(values),
        "finite_mask": [True] * 7,
        "finite_count": 7,
    }


def _safety_report(*, episode_index=0, apply_calls=smoke.EXPECTED_APPLY_CALLS):
    active = apply_calls > 0
    counters = {
        "apply_calls": apply_calls,
        "environment_substeps": apply_calls,
        "slew_limit_events": 256 if active else 0,
        "slew_limited_joints": 512 if active else 0,
        "position_limit_events": 128 if active else 0,
        "position_limited_joints": 128 if active else 0,
        "post_clamp_target_violations": 0,
        "current_joint_limit_aborts": 0,
        "invariant_aborts": 0,
        "nonfinite_aborts": 0,
        "dls_fallbacks": 0,
        "guard_diagnostics_dropped": 0,
    }
    raw_delta = [0.05] * 7 if active else [0.0] * 7
    maxima = {
        "raw_delta_joint_pos_rad": raw_delta,
        "applied_delta_joint_pos_rad": (
            list(smoke.EXPECTED_MAX_DELTA_RAD) if active else [0.0] * 7
        ),
        "raw_target_soft_limit_violation_rad": (
            [0.0, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0] if active else [0.0] * 7
        ),
        "post_clamp_target_soft_limit_violation_rad": [0.0] * 7,
        "post_clamp_target_guard_band_violation_rad": [0.0] * 7,
        "current_joint_soft_limit_violation_rad": [0.0] * 7,
        "current_physx_hard_limit_violation_rad": [0.0] * 7,
        "abs_joint_vel_rad_s": [0.1] * 7 if active else [0.0] * 7,
        "minimum_outer_joint_clearance_rad": [0.01] * 7 if active else [0.0] * 7,
    }
    q = [(lower + upper) / 2.0 for lower, upper in smoke.EXPECTED_TARGET_LIMITS_RAD]
    raw_target = [value + delta for value, delta in zip(q, raw_delta, strict=True)]
    max_diagnostic = (
        {
            "kind": "max_raw_delta",
            "episode_index": episode_index,
            "policy_step": 10,
            "physics_substep": 3,
            "joint_pos_rad": _diagnostic_vector(q),
            "raw_delta_joint_pos_rad": _diagnostic_vector(raw_delta),
            "raw_joint_pos_target_rad": _diagnostic_vector(raw_target),
            "safe_joint_pos_target_rad": _diagnostic_vector(q),
            "pose_error_norm": 0.5,
            "jacobian_finite": True,
            "jacobian_max_abs": 1.0,
            "eef_quaternion_norm": None,
        }
        if active
        else None
    )
    return {
        "episode_index": episode_index,
        "profile": "panda_velocity_physxlimit_solveriter1_v4",
        "apply_actions_cadence": "physics_substep",
        "physics_dt": 1.0 / 120.0,
        "control_dt": 1.0 / 15.0,
        "decimation": 8,
        "current_joint_soft_limit_tolerance_rad": 1e-5,
        "target_soft_limit_guard_band_profile": (
            "eef_physx_inner_hardlimit_one_substep_v2"
        ),
        "physx_hard_limit_profile": "outer_minus_one_velocity_substep_v1",
        "physx_derived_soft_limit_profile": (
            "isaaclab_midpoint_range_factor1_float32_v1"
        ),
        "physx_hard_limit_write_count": 1,
        "arm_velocity_target_profile": "zero_per_physics_substep_v1",
        "articulation_solver_profile": "tgs_position64_velocity1_eef_only_v1",
        "articulation_solver_readback": (
            "composed_usd_physx_articulation_api_all_env_roots_v1"
        ),
        "physx_solver_type": 1,
        "solver_position_iteration_count": 64,
        "solver_velocity_iteration_count": 1,
        "joint_velocity_limit_tolerance_rad_s": 1e-5,
        "eef_quaternion_unit_norm_tolerance": 1e-3,
        "joint_slew_float32_tolerance_rad": 1e-6,
        "soft_joint_pos_limit_factor": 1.0,
        "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
        "joint_velocity_limits_rad_s": list(smoke.EXPECTED_VELOCITY_LIMITS_RAD_S),
        "joint_effort_limits": list(smoke.EXPECTED_EFFORT_LIMITS),
        "max_delta_joint_pos_rad": list(smoke.EXPECTED_MAX_DELTA_RAD),
        "target_soft_limit_margin_rad": list(smoke.EXPECTED_MAX_DELTA_RAD),
        "target_joint_pos_limits_rad": copy.deepcopy(smoke.EXPECTED_TARGET_LIMITS_RAD),
        "target_joint_pos_limits_float32_sha256": smoke.TARGET_LIMIT_DIGEST,
        "physx_hard_joint_pos_limits_rad": copy.deepcopy(
            smoke.EXPECTED_TARGET_LIMITS_RAD
        ),
        "physx_hard_joint_pos_limits_float32_sha256": smoke.TARGET_LIMIT_DIGEST,
        "physx_derived_soft_joint_pos_limits_rad": copy.deepcopy(
            smoke.EXPECTED_PHYSX_DERIVED_SOFT_LIMITS_RAD
        ),
        "physx_derived_soft_joint_pos_limits_float32_sha256": (
            smoke.PHYSX_DERIVED_SOFT_LIMIT_DIGEST
        ),
        "arm_velocity_target_rad_s": [0.0] * 7,
        "soft_joint_pos_limits_rad": copy.deepcopy(smoke.EXPECTED_OUTER_LIMITS_RAD),
        "soft_joint_pos_limits_float32_sha256": smoke.SOFT_LIMIT_DIGEST,
        "counters": counters,
        "maxima": maxima,
        "guard_diagnostics": [],
        "max_raw_delta_diagnostic": max_diagnostic,
    }


def _candidate_safety_report(
    *, episode_index=0, apply_calls=smoke.EXPECTED_APPLY_CALLS
):
    report = _safety_report(
        episode_index=episode_index,
        apply_calls=apply_calls,
    )
    active = apply_calls > 0
    report["profile"] = smoke.WRIST_ENERGY_BRAKE_CANDIDATE_PROFILE
    report.update(
        {
            "wrist_energy_brake_profile": smoke.WRIST_ENERGY_BRAKE_PROFILE,
            "wrist_energy_brake_joint_names": list(
                smoke.WRIST_ENERGY_BRAKE_JOINT_NAMES
            ),
            "wrist_energy_brake_latch_substeps": (
                smoke.WRIST_ENERGY_BRAKE_LATCH_SUBSTEPS
            ),
            "wrist_energy_brake_target_shift_fraction": (
                smoke.WRIST_ENERGY_BRAKE_TARGET_SHIFT_FRACTION
            ),
            "wrist_energy_brake_target_shift_threshold_rad": [
                smoke._float32_multiply(  # noqa: SLF001
                    value,
                    smoke.WRIST_ENERGY_BRAKE_TARGET_SHIFT_FRACTION,
                )
                for value in smoke.EXPECTED_MAX_DELTA_RAD[4:]
            ],
            "wrist_energy_brake_latch_remaining_substeps": [0],
            "wrist_energy_brake_diagnostics": [],
        }
    )
    report["counters"].update(
        {
            "wrist_energy_brake_trigger_events": 1 if active else 0,
            "wrist_energy_brake_active_substeps": 2 if active else 0,
            "wrist_energy_brake_attempted_joint_targets": 5 if active else 0,
            "wrist_energy_brake_braked_joint_targets": 5 if active else 0,
            "wrist_energy_brake_diagnostics_dropped": 0,
        }
    )
    if active:
        q = [0.0] * 7
        q[5] = 1.0
        previous = [0.0] * 7
        nominal = [0.0] * 7
        applied = [0.0] * 7
        previous[4] = -0.02
        nominal[4] = 0.02
        previous[5] = nominal[5] = 1.01
        previous[6] = nominal[6] = -0.01
        joint_vel = [0.0] * 7
        joint_vel[4:] = [0.2, 0.2, -0.2]
        applied[5] = 1.0
        report["wrist_energy_brake_diagnostics"] = [
            {
                "episode_index": episode_index,
                "apply_index": 896,
                "policy_step": 112,
                "physics_substep": 0,
                "environment_index": 0,
                "reversal_detection_armed": True,
                "trigger_joint_mask": [True, False, False],
                "attempted_joint_mask": [True, True, True],
                "braked_joint_mask": [True, True, True],
                "joint_pos_rad": q,
                "joint_vel_rad_s": joint_vel,
                "previous_applied_target_rad": previous,
                "nominal_safe_target_rad": nominal,
                "applied_target_rad": applied,
                "target_shift_rad": [0.04, 0.0, 0.0],
            },
            {
                "episode_index": episode_index,
                "apply_index": 897,
                "policy_step": 112,
                "physics_substep": 1,
                "environment_index": 0,
                "reversal_detection_armed": False,
                "trigger_joint_mask": [False, False, False],
                "attempted_joint_mask": [True, True, False],
                "braked_joint_mask": [True, True, False],
                "joint_pos_rad": q,
                "joint_vel_rad_s": [0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.0],
                "previous_applied_target_rad": applied,
                "nominal_safe_target_rad": nominal,
                "applied_target_rad": [
                    applied[index] if index in (4, 5) else nominal[index]
                    for index in range(7)
                ],
                "target_shift_rad": [
                    abs(smoke._float32_subtract(nominal[index], applied[index]))
                    for index in range(4, 7)
                ],
            },
        ]
    return report


def _boundary_result():
    arm_q = [(lower + upper) / 2.0 for lower, upper in smoke.EXPECTED_OUTER_LIMITS_RAD]
    arm_q[smoke.TARGET_JOINT_INDEX] = 2.88
    arm_target = [
        (lower + upper) / 2.0 for lower, upper in smoke.EXPECTED_TARGET_LIMITS_RAD
    ]
    arm_target[smoke.TARGET_JOINT_INDEX] = smoke.INNER_UPPER_LIMIT_RAD
    records = []
    for drive_step in range(smoke.ADAPTIVE_DRIVE_STEPS):
        records.append(
            {
                "drive_step": drive_step,
                "policy_step": smoke.EXPECTED_ACTION_ENCODING["action_count"]
                + drive_step,
                "position_limit_events_delta": 8,
                "joint_pos_rad": arm_q[smoke.TARGET_JOINT_INDEX],
                "joint_vel_rad_s": 0.0,
                "joint_target_rad": smoke.INNER_UPPER_LIMIT_RAD,
                "predicted_outward_joint_delta_rad": 0.1,
                "arm_joint_pos_rad": list(arm_q),
                "arm_joint_vel_rad_s": [0.0] * 7,
                "arm_joint_target_rad": list(arm_target),
                "eef_position_m": [0.4, -0.2, 0.2],
                "eef_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "target_is_inner_limit": True,
                "within_outer_limits": True,
                "all_arm_joints_within_outer_limits": True,
                "eef_state_is_finite": True,
                "state_is_finite": True,
            }
        )
    return {
        "target_joint_name": smoke.TARGET_JOINT_NAME,
        "target_joint_index": smoke.TARGET_JOINT_INDEX,
        "direction": "upper",
        "replay_action_count": smoke.EXPECTED_ACTION_ENCODING["action_count"],
        "adaptive_drive_steps": smoke.ADAPTIVE_DRIVE_STEPS,
        "total_policy_steps": smoke.EXPECTED_TOTAL_POLICY_STEPS,
        "expected_apply_calls": smoke.EXPECTED_APPLY_CALLS,
        "outward_delta_scale": smoke.OUTWARD_DELTA_SCALE,
        "required_consecutive_dwell_policy_steps": (
            smoke.REQUIRED_CONSECUTIVE_DWELL_STEPS
        ),
        "observed_max_consecutive_dwell_policy_steps": smoke.ADAPTIVE_DRIVE_STEPS,
        "joint_outer_lower_limit_rad": smoke.OUTER_LOWER_LIMIT_RAD,
        "joint_outer_upper_limit_rad": smoke.OUTER_UPPER_LIMIT_RAD,
        "joint_inner_target_upper_limit_rad": smoke.INNER_UPPER_LIMIT_RAD,
        "joint_pos_observed_min_rad": 0.0,
        "joint_pos_observed_max_rad": 2.88,
        "terminated": False,
        "truncated": False,
        "state_is_finite": True,
        "dwell_records": records,
    }


def _identity(path: Path, *, forced_sha256=None):
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "size_bytes": len(data),
        "sha256": forced_sha256 or hashlib.sha256(data).hexdigest(),
        "mode": f"{path.stat().st_mode & 0o777:04o}",
    }


def _success_payload():
    fixture, _ = smoke.load_replay_fixture()
    dummy = {
        "path": "/pinned/food_bussing/scene.usda",
        "size_bytes": 1,
        "sha256": smoke.EXPECTED_ASSET_CONTRACT["scene_sha256"],
        "mode": "0444",
    }
    initial = {
        **dummy,
        "path": "/pinned/food_bussing/initial_conditions.json",
        "sha256": smoke.EXPECTED_ASSET_CONTRACT["initial_conditions_sha256"],
    }
    metadata = {
        filename: {
            **dummy,
            "path": f"/pinned/metadata/{filename}.metadata",
            "revision": smoke.EXPECTED_ASSET_CONTRACT["polaris_hub_revision"],
        }
        for filename in ("initial_conditions.json", "scene.usda")
    }
    return {
        "schema_version": 1,
        "fixture_profile": smoke.FIXTURE_PROFILE,
        "smoke_profile": smoke.SMOKE_PROFILE,
        "finalized": False,
        "passed": True,
        "stage": "simulation_app_close_pending",
        "exit_code": 0,
        "environment": smoke.ENVIRONMENT,
        "fixture": fixture,
        "assets": {
            "scene": dummy,
            "initial_conditions": initial,
            "polaris_hub_revision": smoke.EXPECTED_ASSET_CONTRACT[
                "polaris_hub_revision"
            ],
            "revision_metadata": metadata,
            "initial_condition_index": 0,
        },
        "runtime_protocol": {
            "episode_steps": 450,
            "policy_hz": 15.0,
            "step_dt": 1.0 / 15.0,
            "physics_hz": 120.0,
            "physics_dt": 1.0 / 120.0,
            "decimation": 8,
        },
        "runtime_frame": {
            "eef_frame": "panda_link8",
            "reference_frame": "panda_link0",
            "position_error_m": 0.0,
            "rotation_error_rad": 0.0,
            "controlled_body": "panda_link8",
            "body_offset": "identity",
            "command_type": "pose",
            "use_relative_mode": False,
            "ik_method": "dls",
            "dls_damping": 0.01,
            "arm_scale": 1.0,
            "arm_joint_names": [f"panda_joint{index}" for index in range(1, 8)],
            "gripper_threshold_profile": (
                "closed_positive_ge_0p5_inverse_open_gt_0p5_v1"
            ),
            "ik_safety_profile": "panda_velocity_physxlimit_solveriter1_v4",
            "action_dim": 7,
        },
        "initial_ik_safety_capture": _safety_report(episode_index=None, apply_calls=0),
        "boundary": _boundary_result(),
        "ik_safety": _safety_report(),
        "failure": None,
        "close_failures": [],
    }


def _candidate_success_payload():
    payload = _success_payload()
    payload["runtime_frame"]["ik_safety_profile"] = (
        smoke.WRIST_ENERGY_BRAKE_CANDIDATE_PROFILE
    )
    payload["initial_ik_safety_capture"] = _candidate_safety_report(
        episode_index=None,
        apply_calls=0,
    )
    payload["ik_safety"] = _candidate_safety_report()
    return payload


def test_fixture_exact_identity_and_decoded_action_contract():
    identity, actions = smoke.load_replay_fixture()

    assert identity["size_bytes"] == 15967
    assert identity["sha256"] == smoke.EXPECTED_FIXTURE_SHA256
    assert identity["source_trace_sha256"] == smoke.EXPECTED_SOURCE["trace_sha256"]
    assert (
        identity["action_float32_sha256"]
        == (smoke.EXPECTED_ACTION_ENCODING["uncompressed_sha256"])
    )
    assert len(actions) == 378
    assert actions[0] == pytest.approx(
        [
            0.3568142354488373,
            -0.0012728864094242454,
            0.49005618691444397,
            -0.007359412964433432,
            0.9999475479125977,
            0.005898015107959509,
            -0.003998896572738886,
            0.0,
        ],
        abs=0.0,
    )
    assert actions[-1][-1] == 1.0
    assert all(
        abs(math.sqrt(sum(v * v for v in a[3:7])) - 1.0) <= 1e-3 for a in actions
    )


def test_fixture_parser_rejects_schema_source_encoding_and_payload_mutations():
    fixture = smoke.strict_json_loads(smoke.FIXTURE_PATH.read_bytes(), field="fixture")
    mutations = []

    extra = copy.deepcopy(fixture)
    extra["extra"] = True
    mutations.append(extra)
    source = copy.deepcopy(fixture)
    source["source"]["failed_policy_step"] -= 1
    mutations.append(source)
    encoding = copy.deepcopy(fixture)
    encoding["action_encoding"]["action_count"] -= 1
    mutations.append(encoding)
    payload = copy.deepcopy(fixture)
    payload["actions_zlib_base64_chunks"][0] = (
        "A" + payload["actions_zlib_base64_chunks"][0][1:]
    )
    mutations.append(payload)

    for mutation in mutations:
        with pytest.raises(smoke.BoundaryReplayValidationError):
            smoke.decode_replay_fixture(mutation)


def test_fixture_file_identity_rejects_byte_tamper(tmp_path):
    tampered = tmp_path / smoke.FIXTURE_PATH.name
    tampered.write_bytes(smoke.FIXTURE_PATH.read_bytes() + b"\n")

    with pytest.raises(smoke.BoundaryReplayValidationError, match="file size"):
        smoke.load_replay_fixture(tampered)


def test_strict_json_rejects_duplicates_and_nonfinite_constants():
    with pytest.raises(smoke.BoundaryReplayValidationError, match="duplicate"):
        smoke.strict_json_loads(b'{"a":1,"a":2}', field="test")
    with pytest.raises(smoke.BoundaryReplayValidationError, match="constant"):
        smoke.strict_json_loads(b'{"a":NaN}', field="test")


def test_boundary_evidence_accepts_exact_full_state_dwell():
    summary = smoke.validate_boundary_result(_boundary_result(), _safety_report())

    assert summary["apply_calls"] == 3152
    assert summary["max_consecutive_dwell_policy_steps"] == 16
    assert summary["joint5_raw_outer_violation_rad"] == 0.04


def test_candidate_boundary_evidence_is_closed_and_requires_brake_activity():
    safety = _candidate_safety_report()

    summary = smoke.validate_boundary_result(_boundary_result(), safety)

    assert summary["apply_calls"] == smoke.EXPECTED_APPLY_CALLS
    assert safety["counters"]["wrist_energy_brake_trigger_events"] == 1
    assert safety["wrist_energy_brake_diagnostics"][0]["apply_index"] == 896

    for field, value in (
        ("wrist_energy_brake_trigger_events", 0),
        ("wrist_energy_brake_active_substeps", 1),
        ("wrist_energy_brake_attempted_joint_targets", 0),
        ("wrist_energy_brake_braked_joint_targets", 0),
    ):
        tampered = _candidate_safety_report()
        tampered["counters"][field] = value
        with pytest.raises(smoke.BoundaryReplayValidationError):
            smoke.validate_boundary_result(_boundary_result(), tampered)

    tampered = _candidate_safety_report()
    tampered["wrist_energy_brake_latch_remaining_substeps"] = [1]
    with pytest.raises(smoke.BoundaryReplayValidationError, match="latch"):
        smoke.validate_boundary_result(_boundary_result(), tampered)

    lossy = _candidate_safety_report()
    trigger, follow_up = lossy["wrist_energy_brake_diagnostics"]
    diagnostics = []
    for pair_index in range(16):
        trigger_copy = copy.deepcopy(trigger)
        follow_up_copy = copy.deepcopy(follow_up)
        trigger_apply_index = 896 + 4 * pair_index
        trigger_copy.update(
            {
                "apply_index": trigger_apply_index,
                "policy_step": trigger_apply_index // smoke.DECIMATION,
                "physics_substep": trigger_apply_index % smoke.DECIMATION,
            }
        )
        follow_up_apply_index = trigger_apply_index + 1
        follow_up_copy.update(
            {
                "apply_index": follow_up_apply_index,
                "policy_step": follow_up_apply_index // smoke.DECIMATION,
                "physics_substep": follow_up_apply_index % smoke.DECIMATION,
            }
        )
        diagnostics.extend((trigger_copy, follow_up_copy))
    lossy["wrist_energy_brake_diagnostics"] = diagnostics
    lossy["counters"].update(
        {
            "wrist_energy_brake_trigger_events": 17,
            "wrist_energy_brake_active_substeps": 34,
            "wrist_energy_brake_attempted_joint_targets": 85,
            "wrist_energy_brake_braked_joint_targets": 85,
            "wrist_energy_brake_diagnostics_dropped": 2,
        }
    )
    smoke.validate_safety_static(lossy, episode_index=0)
    with pytest.raises(smoke.BoundaryReplayValidationError, match="dropped causal"):
        smoke.validate_boundary_result(_boundary_result(), lossy)


def test_candidate_success_payload_binds_runtime_profile_and_initial_schema():
    payload = _candidate_success_payload()

    smoke.validate_success_payload(payload)

    tampered = copy.deepcopy(payload)
    tampered["wrist_energy_brake_extra"] = True
    with pytest.raises(smoke.BoundaryReplayValidationError, match="schema"):
        smoke.validate_success_payload(tampered)

    tampered = copy.deepcopy(payload)
    tampered["ik_safety"]["wrist_energy_brake_diagnostics"][0]["physics_substep"] = 1
    with pytest.raises(smoke.BoundaryReplayValidationError, match="apply identity"):
        smoke.validate_success_payload(tampered)

    tampered = copy.deepcopy(payload)
    tampered["initial_ik_safety_capture"][
        "wrist_energy_brake_latch_remaining_substeps"
    ] = [2]
    with pytest.raises(smoke.BoundaryReplayValidationError, match="latch"):
        smoke.validate_success_payload(tampered)

    tampered = copy.deepcopy(payload)
    tampered["initial_ik_safety_capture"] = _safety_report(
        episode_index=None,
        apply_calls=0,
    )
    with pytest.raises(smoke.BoundaryReplayValidationError, match="profile"):
        smoke.validate_success_payload(tampered)

    tampered = copy.deepcopy(payload)
    tampered["ik_safety"]["wrist_energy_brake_diagnostics"][0]["apply_index"] = 99_999
    with pytest.raises(smoke.BoundaryReplayValidationError, match="apply identity"):
        smoke.validate_success_payload(tampered)

    tampered = copy.deepcopy(payload)
    tampered["ik_safety"]["counters"]["wrist_energy_brake_active_substeps"] = 100
    with pytest.raises(smoke.BoundaryReplayValidationError):
        smoke.validate_success_payload(tampered)

    tampered = copy.deepcopy(payload)
    tampered["ik_safety"]["counters"]["wrist_energy_brake_attempted_joint_targets"] = 6
    with pytest.raises(
        smoke.BoundaryReplayValidationError,
        match="attempted-target diagnostic count",
    ):
        smoke.validate_success_payload(tampered)

    tampered = copy.deepcopy(payload)
    tampered["ik_safety"]["counters"]["wrist_energy_brake_braked_joint_targets"] = 4
    with pytest.raises(
        smoke.BoundaryReplayValidationError,
        match="effective-target diagnostic count",
    ):
        smoke.validate_success_payload(tampered)

    tampered = copy.deepcopy(payload)
    follow_up = tampered["ik_safety"]["wrist_energy_brake_diagnostics"][1]
    follow_up["apply_index"] = 1_000
    follow_up["policy_step"] = 125
    follow_up["physics_substep"] = 0
    with pytest.raises(smoke.BoundaryReplayValidationError, match="follow-up cadence"):
        smoke.validate_success_payload(tampered)

    tampered = copy.deepcopy(payload)
    trigger, follow_up = tampered["ik_safety"]["wrist_energy_brake_diagnostics"]
    trigger.update({"apply_index": 0, "policy_step": 0, "physics_substep": 0})
    follow_up.update({"apply_index": 1, "policy_step": 0, "physics_substep": 1})
    with pytest.raises(smoke.BoundaryReplayValidationError, match="arming cadence"):
        smoke.validate_success_payload(tampered)

    tampered = copy.deepcopy(payload)
    follow_up = tampered["ik_safety"]["wrist_energy_brake_diagnostics"][1]
    changed_previous = list(follow_up["previous_applied_target_rad"])
    changed_previous[4] = -0.01
    follow_up["previous_applied_target_rad"] = changed_previous
    follow_up["target_shift_rad"][0] = 0.03
    with pytest.raises(smoke.BoundaryReplayValidationError, match="previous-target"):
        smoke.validate_success_payload(tampered)

    tampered = copy.deepcopy(payload)
    tampered["ik_safety"]["counters"]["wrist_energy_brake_diagnostics_dropped"] = 1
    with pytest.raises(smoke.BoundaryReplayValidationError):
        smoke.validate_success_payload(tampered)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda boundary, safety: boundary["dwell_records"][-1].__setitem__(
                "position_limit_events_delta", 7
            ),
            "last eight",
        ),
        (
            lambda boundary, safety: boundary["dwell_records"][0][
                "arm_joint_pos_rad"
            ].__setitem__(0, 3.0),
            "all-arm limit",
        ),
        (
            lambda boundary, safety: boundary["dwell_records"][0][
                "arm_joint_target_rad"
            ].__setitem__(0, 3.0),
            "target guard",
        ),
        (
            lambda boundary, safety: boundary["dwell_records"][0][
                "eef_quaternion_wxyz"
            ].__setitem__(0, 0.0),
            "quaternion norm",
        ),
        (
            lambda boundary, safety: safety["counters"].__setitem__(
                "current_joint_limit_aborts", 1
            ),
            "must be zero",
        ),
        (
            lambda boundary, safety: safety["maxima"][
                "raw_target_soft_limit_violation_rad"
            ].__setitem__(4, 0.0),
            "never drove joint5",
        ),
        (
            lambda boundary, safety: safety.__setitem__(
                "target_joint_pos_limits_float32_sha256", "0" * 64
            ),
            "target_joint_pos_limits_float32_sha256",
        ),
        (
            lambda boundary, safety: safety["physx_derived_soft_joint_pos_limits_rad"][
                3
            ].__setitem__(1, smoke.EXPECTED_TARGET_LIMITS_RAD[3][1]),
            "PhysX-derived soft limits",
        ),
        (
            lambda boundary, safety: safety["arm_velocity_target_rad_s"].__setitem__(
                4, 1e-3
            ),
            "velocity target drift",
        ),
        (
            lambda boundary, safety: safety.__setitem__(
                "solver_velocity_iteration_count", 0
            ),
            "solver_velocity_iteration_count",
        ),
        (
            lambda boundary, safety: safety.__setitem__("physx_solver_type", 0),
            "physx_solver_type",
        ),
    ],
)
def test_boundary_evidence_rejects_mutations(mutation, message):
    boundary = _boundary_result()
    safety = _safety_report()
    mutation(boundary, safety)

    with pytest.raises(smoke.BoundaryReplayValidationError, match=message):
        smoke.validate_boundary_result(boundary, safety)


def test_success_payload_is_closed_and_binds_initial_capture():
    payload = _success_payload()
    smoke.validate_success_payload(payload)

    extra = copy.deepcopy(payload)
    extra["extra"] = True
    with pytest.raises(smoke.BoundaryReplayValidationError, match="schema"):
        smoke.validate_success_payload(extra)

    dirty_initial = copy.deepcopy(payload)
    dirty_initial["initial_ik_safety_capture"]["counters"]["apply_calls"] = 1
    with pytest.raises(smoke.BoundaryReplayValidationError, match="initial"):
        smoke.validate_success_payload(dirty_initial)


def test_immutable_raw_publication_is_nonoverwriting_and_mode_0444(tmp_path):
    path = tmp_path / "result.json"
    identity = smoke._atomic_write_immutable(path, {"schema_version": 1})

    assert identity["mode"] == "0444"
    assert path.stat().st_mode & 0o777 == 0o444
    with pytest.raises(FileExistsError):
        smoke._atomic_write_immutable(path, {"schema_version": 1})


def test_host_finalizer_reconstructs_raw_ready_and_provenance(tmp_path, monkeypatch):
    repo = tmp_path / "PolaRiS"
    (repo / "scripts" / "fixtures").mkdir(parents=True)
    runner = repo / "scripts" / "smoke_eef_pose_boundary_replay.py"
    fixture_path = (
        repo
        / "scripts"
        / "fixtures"
        / "official_lap3b_foodbussing_v3_boundary_actions.json"
    )
    shutil.copy2(Path(smoke.__file__), runner)
    shutil.copy2(smoke.FIXTURE_PATH, fixture_path)
    fixture, _ = smoke.load_replay_fixture(fixture_path)

    payload = _success_payload()
    payload["fixture"] = fixture
    job_id = 12345
    raw_path = tmp_path / f"boundary-replay-smoke-{job_id}.json"
    raw_identity = smoke._atomic_write_immutable(raw_path, payload)
    marker_path = raw_path.with_name(raw_path.name + ".ready.json")
    smoke._atomic_write_immutable(
        marker_path,
        {
            "schema_version": 1,
            "stage": "simulation_app_close_pending",
            "raw_result": raw_identity,
        },
    )
    image = tmp_path / "image.sqsh"
    image.write_bytes(b"image")
    runtime_script = tmp_path / "job.sh"
    saved_script = tmp_path / "job.saved.sh"
    runtime_script.write_text("#!/bin/bash\ntrue\n")
    saved_script.write_bytes(runtime_script.read_bytes())
    runtime_script.chmod(0o444)
    saved_script.chmod(0o444)
    attestation = tmp_path / f"boundary-replay-smoke-{job_id}.attestation.json"
    commit = "a" * 40
    monkeypatch.setenv("SLURM_JOB_ID", str(job_id))
    monkeypatch.setenv("SLURM_NODELIST", "pool0-00005")
    monkeypatch.setattr(
        finalizer,
        "_git",
        lambda _repo, *arguments: (
            "" if arguments == ("status", "--porcelain") else commit
        ),
    )
    monkeypatch.setattr(
        finalizer,
        "_validate_asset_identities",
        lambda raw: {"validated": True},
    )
    monkeypatch.setattr(finalizer.smoke, "__file__", str(runner))
    args = argparse.Namespace(
        raw_result=raw_path,
        attestation=attestation,
        srun_rc=0,
        job_id=job_id,
        runtime_job_script=runtime_script,
        saved_job_script=saved_script,
        polaris_repo=repo,
        expected_polaris_commit=commit,
        expected_safety_profile=smoke.BASE_SAFETY_PROFILE,
        expected_runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
        expected_fixture_sha256=smoke.EXPECTED_FIXTURE_SHA256,
        container_image=image,
        expected_image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        expected_finalizer_sha256=hashlib.sha256(
            Path(finalizer.__file__).read_bytes()
        ).hexdigest(),
        expected_saved_job_script_sha256=hashlib.sha256(
            saved_script.read_bytes()
        ).hexdigest(),
    )

    expected = finalizer.build_expected_attestation(args)

    assert expected["passed"] is True
    assert expected["raw_result"]["ready_marker"]["mode"] == "0444"
    assert expected["provenance"]["fixture_source"] == smoke.EXPECTED_SOURCE
    assert expected["validation_summary"]["safety_profile"] == (
        smoke.BASE_SAFETY_PROFILE
    )
    assert expected["provenance"]["expected_safety_profile"] == (
        smoke.BASE_SAFETY_PROFILE
    )
    assert expected["provenance"]["slurm"] == {
        "job_id": job_id,
        "nodelist": "pool0-00005",
    }

    cli = [
        "finalize_eef_pose_boundary_replay.py",
        "finalize",
        "--raw-result",
        str(raw_path),
        "--attestation",
        str(attestation),
        "--srun-rc",
        "0",
        "--job-id",
        str(job_id),
        "--runtime-job-script",
        str(runtime_script),
        "--saved-job-script",
        str(saved_script),
        "--polaris-repo",
        str(repo),
        "--expected-polaris-commit",
        commit,
        "--expected-safety-profile",
        smoke.BASE_SAFETY_PROFILE,
        "--expected-runner-sha256",
        args.expected_runner_sha256,
        "--expected-fixture-sha256",
        args.expected_fixture_sha256,
        "--container-image",
        str(image),
        "--expected-image-sha256",
        args.expected_image_sha256,
        "--expected-finalizer-sha256",
        args.expected_finalizer_sha256,
        "--expected-saved-job-script-sha256",
        args.expected_saved_job_script_sha256,
    ]
    monkeypatch.setattr(sys, "argv", cli)
    assert finalizer.main() == 0
    assert attestation.stat().st_mode & 0o777 == 0o444
    cli[1] = "verify"
    monkeypatch.setattr(sys, "argv", cli)
    assert finalizer.main() == 0

    candidate_job_id = job_id + 1
    candidate_payload = _candidate_success_payload()
    candidate_payload["fixture"] = fixture
    candidate_raw_path = tmp_path / f"boundary-replay-smoke-{candidate_job_id}.json"
    candidate_raw_identity = smoke._atomic_write_immutable(
        candidate_raw_path,
        candidate_payload,
    )
    candidate_marker_path = candidate_raw_path.with_name(
        candidate_raw_path.name + ".ready.json"
    )
    smoke._atomic_write_immutable(
        candidate_marker_path,
        {
            "schema_version": 1,
            "stage": "simulation_app_close_pending",
            "raw_result": candidate_raw_identity,
        },
    )
    candidate_attestation = (
        tmp_path / f"boundary-replay-smoke-{candidate_job_id}.attestation.json"
    )
    candidate_args = copy.copy(args)
    candidate_args.raw_result = candidate_raw_path
    candidate_args.attestation = candidate_attestation
    candidate_args.job_id = candidate_job_id
    candidate_args.expected_safety_profile = smoke.WRIST_ENERGY_BRAKE_CANDIDATE_PROFILE
    monkeypatch.setenv("SLURM_JOB_ID", str(candidate_job_id))

    candidate_expected = finalizer.build_expected_attestation(candidate_args)

    assert candidate_expected["validation_summary"]["safety_profile"] == (
        smoke.WRIST_ENERGY_BRAKE_CANDIDATE_PROFILE
    )
    assert candidate_expected["validation_summary"]["wrist_energy_brake"] == {
        "profile": smoke.WRIST_ENERGY_BRAKE_PROFILE,
        "trigger_events": 1,
        "active_substeps": 2,
        "attempted_joint_targets": 5,
        "braked_joint_targets": 5,
        "diagnostic_count": 2,
        "diagnostics_dropped": 0,
    }
    candidate_cli = list(cli)
    candidate_cli[1] = "finalize"
    for option, value in (
        ("--raw-result", candidate_raw_path),
        ("--attestation", candidate_attestation),
        ("--job-id", candidate_job_id),
        (
            "--expected-safety-profile",
            smoke.WRIST_ENERGY_BRAKE_CANDIDATE_PROFILE,
        ),
    ):
        candidate_cli[candidate_cli.index(option) + 1] = str(value)
    monkeypatch.setattr(sys, "argv", candidate_cli)
    assert finalizer.main() == 0
    candidate_cli[1] = "verify"
    monkeypatch.setattr(sys, "argv", candidate_cli)
    assert finalizer.main() == 0

    wrong_profile_args = copy.copy(candidate_args)
    wrong_profile_args.expected_safety_profile = smoke.BASE_SAFETY_PROFILE
    with pytest.raises(finalizer.FinalizationError, match="expected controller"):
        finalizer.build_expected_attestation(wrong_profile_args)

    monkeypatch.setenv("SLURM_JOB_ID", str(job_id))
    bad_marker = json.loads(marker_path.read_text())
    bad_marker["raw_result"]["sha256"] = "0" * 64
    marker_path.chmod(0o640)
    marker_path.write_text(json.dumps(bad_marker))
    marker_path.chmod(0o444)
    with pytest.raises(finalizer.FinalizationError, match="ready marker"):
        finalizer.build_expected_attestation(args)
