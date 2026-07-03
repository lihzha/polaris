from __future__ import annotations

import ast
import copy
import inspect
from pathlib import Path
import textwrap

import pytest

from scripts import smoke_eef_pose_canary_trace_replay as replay


def _snapshot(value: float = 0.0) -> dict[str, list[float]]:
    return {field: [value] * 6 for field in replay.SNAPSHOT_FIELDS}


def _gripper_tail(variant: str) -> dict[str, object]:
    failure = replay.EXPECTED_FIXTURES[variant]["failure"]
    total = failure["policy_step"] * replay.DECIMATION + failure["physics_substep"]
    first = total - replay.GRIPPER_TAIL_CAPACITY
    entries = []
    for apply_index in range(first, total):
        entries.append(
            {
                "apply_index": apply_index,
                "policy_step": apply_index // replay.DECIMATION,
                "physics_substep": apply_index % replay.DECIMATION,
                "raw_action": 0.0,
                "requested_endpoint_rad": 0.0,
                "pre": _snapshot(float(apply_index)),
                "target_after_setter_rad": 0.0,
                "post": _snapshot(float(apply_index + 1)),
            }
        )
    return {
        "schema_version": 1,
        "profile": replay.GRIPPER_TAIL_PROFILE,
        "capacity": replay.GRIPPER_TAIL_CAPACITY,
        "decimation": replay.DECIMATION,
        "joint_names": list(replay.GRIPPER_JOINT_NAMES),
        "joint_indices": list(replay.GRIPPER_JOINT_INDICES),
        "process_action_calls": failure["policy_step"] + 1,
        "total_apply_entries": total,
        "dropped_entries": total - len(entries),
        "entries": entries,
        "failure_snapshot": copy.deepcopy(entries[-1]["post"]),
    }


@pytest.mark.parametrize("variant", sorted(replay.EXPECTED_FIXTURES))
def test_committed_fixture_identity_and_actions(variant: str) -> None:
    identity, payload, actions = replay.load_replay_fixture(variant)
    expected = replay.EXPECTED_FIXTURES[variant]
    assert identity["size_bytes"] == expected["size_bytes"]
    assert identity["sha256"] == expected["sha256"]
    assert payload["source"]["trace_sha256"] == expected["trace_sha256"]
    assert len(actions) == 120


def test_fixture_payload_tamper_is_rejected() -> None:
    _, payload, _ = replay.load_replay_fixture("official_lap3b")
    tampered = copy.deepcopy(payload)
    tampered["action_plan"]["query14_executable_action_count"] = 16
    with pytest.raises(replay.Gate0ReplayValidationError, match="action plan"):
        replay.decode_fixture_payload(tampered, variant="official_lap3b")


@pytest.mark.parametrize("variant", sorted(replay.EXPECTED_FIXTURES))
def test_all_six_gripper_tail_exact_failure_cadence(variant: str) -> None:
    tail = _gripper_tail(variant)
    failure = replay.EXPECTED_FIXTURES[variant]["failure"]
    assert replay.validate_gripper_tail(tail, expected_failure=failure) == tail
    assert len(tail["entries"]) == 64
    assert len(tail["entries"][-1]["post"]["joint_vel_rad_s"]) == 6


def test_all_six_gripper_names_match_preserved_runtime_contract() -> None:
    assert replay.GRIPPER_JOINT_NAMES == [
        "finger_joint",
        "right_outer_knuckle_joint",
        "left_inner_finger_joint",
        "right_inner_finger_joint",
        "left_inner_finger_knuckle_joint",
        "right_inner_finger_knuckle_joint",
    ]
    assert replay.GRIPPER_JOINT_INDICES == [7, 8, 9, 10, 11, 12]


def test_tracing_wrapper_delegates_control_and_never_writes_a_target() -> None:
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(replay._make_tracing_gripper_class))
    )
    methods = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for method_name in ("process_actions", "apply_actions"):
        method = methods[method_name]
        delegated = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == method_name
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "super"
        ]
        assert len(delegated) == 1
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "set_joint_position_target"
        for node in ast.walk(tree)
    )


def test_gripper_tail_rejects_substep_and_six_joint_tamper() -> None:
    failure = replay.EXPECTED_FIXTURES["reasoning_43075"]["failure"]
    tail = _gripper_tail("reasoning_43075")
    bad_cadence = copy.deepcopy(tail)
    bad_cadence["entries"][-1]["physics_substep"] = 7
    with pytest.raises(replay.Gate0ReplayValidationError, match="physics substep"):
        replay.validate_gripper_tail(bad_cadence, expected_failure=failure)
    bad_width = copy.deepcopy(tail)
    bad_width["entries"][-1]["post"]["joint_vel_rad_s"].pop()
    with pytest.raises(replay.Gate0ReplayValidationError, match="shape"):
        replay.validate_gripper_tail(bad_width, expected_failure=failure)


@pytest.mark.parametrize(
    ("variant", "joint", "step", "substep", "digest"),
    [
        (
            "official_lap3b",
            "panda_joint5",
            117,
            6,
            "63c061ec5a47a8bc085547f2abd8dcbc266c9616664d252e29c39ef53864a5f3",
        ),
        (
            "reasoning_43075",
            "panda_joint7",
            112,
            2,
            "3c6242a645b40fe29f7223dc4a146cdb7ee04fe661b136098929bb9b973580b8",
        ),
    ],
)
def test_failure_parser_pins_source_outcomes(
    variant: str, joint: str, step: int, substep: int, digest: str
) -> None:
    message = (
        "PolaRiS EEF IK current joint velocity exceeds the live simulation limit "
        f"(joint='{joint}', velocity_rad_s=3.0, limit_rad_s=2.6, "
        f"policy_step={step}, physics_substep={substep}, evidence_sha256={digest})"
    )
    assert replay.parse_failure_exception(message) == {
        "joint_name": joint,
        "policy_step": step,
        "physics_substep": substep,
        "evidence_sha256": digest,
    }
    assert replay.EXPECTED_FIXTURES[variant]["failure"]["evidence_sha256"] == digest


def test_slurm_lifecycle_requires_a_real_single_task_srun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch_id = "a" * 64
    values = {
        "SLURM_JOB_ID": "123",
        "SLURM_STEP_ID": "4",
        "SLURM_NODELIST": "l401",
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
        "SLURM_NTASKS": "1",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    assert replay._slurm_lifecycle(launch_id) == {
        "profile": "slurm_single_task_srun_lifecycle_v1",
        "launch_id": launch_id,
        "job_id": 123,
        "step_id": 4,
        "nodelist": "l401",
        "procid": 0,
        "localid": 0,
        "ntasks": 1,
    }
    monkeypatch.setenv("SLURM_NTASKS", "2")
    with pytest.raises(replay.Gate0ReplayValidationError, match="one rank"):
        replay._slurm_lifecycle(launch_id)


def test_production_reset_source_is_seed_none_and_default_render() -> None:
    evidence = replay.validate_production_reset_source()
    assert evidence["sha256"] == replay.EXPECTED_PRODUCTION_EVAL_SHA256
    assert evidence["reset_call"] == (
        "obs, info = env.reset(object_positions=initial_conditions[episode])"
    )
    assert evidence["environment_seed"] is None
    assert evidence["reset_expensive_argument"] == "default_true"
    assert evidence["render_every_step_default"] is True
    assert evidence["step_expensive_argument"] == "policy_client.rerender"
    assert evidence["effective_step_expensive"] is True
    assert (
        evidence["policy_config_source"]["sha256"]
        == replay.EXPECTED_PRODUCTION_POLICY_CONFIG_SHA256
    )
    assert (
        evidence["lap_client_source"]["sha256"]
        == replay.EXPECTED_PRODUCTION_LAP_CLIENT_SHA256
    )


def test_capture_validator_binds_runtime_contract_tail_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variant = "official_lap3b"
    expected = replay.EXPECTED_FIXTURES[variant]
    fixture_identity, _, _ = replay.load_replay_fixture(variant)
    _, helper_identity = replay._load_boundary_helper()
    failure = expected["failure"]
    message = (
        "x "
        f"(joint='{failure['joint_name']}', velocity_rad_s=3.0, limit_rad_s=2.6, "
        f"policy_step={failure['policy_step']}, "
        f"physics_substep={failure['physics_substep']}, "
        f"evidence_sha256={failure['evidence_sha256']})"
    )
    monkeypatch.setattr(
        replay,
        "_validate_arm_failure_runtime_evidence",
        lambda value, *, expected_failure: value,
    )
    payload = {
        "schema_version": 1,
        "profile": replay.PROFILE,
        "finalized": False,
        "passed": True,
        "stage": "simulation_app_close_pending",
        "environment": replay.ENVIRONMENT,
        "variant": variant,
        "lifecycle": {
            "profile": "slurm_single_task_srun_lifecycle_v1",
            "launch_id": "b" * 64,
            "job_id": 123,
            "step_id": 0,
            "nodelist": "l401",
            "procid": 0,
            "localid": 0,
            "ntasks": 1,
        },
        "repository": {"path": "/repo", "commit": "c" * 40, "clean_tracked": True},
        "production_eval": replay.validate_production_reset_source(),
        "fixture": {
            **fixture_identity,
            "source_trace_sha256": expected["trace_sha256"],
            "action_float32_sha256": expected["action_encoding"]["uncompressed_sha256"],
            "action_count": 120,
        },
        "boundary_helper": helper_identity,
        "assets": {
            "contract": replay.EXPECTED_ASSET_CONTRACT,
            "robot_usd": {"sha256": replay.EXPECTED_ROBOT_USD_SHA256},
        },
        "runtime_protocol": {"profile": "runtime"},
        "runtime_frame": {"profile": "frame"},
        "gripper_runtime_contract": {
            "profile": "gripper",
            "gripper_joint_names": list(replay.GRIPPER_JOINT_NAMES),
            "gripper_joint_indices": list(replay.GRIPPER_JOINT_INDICES),
        },
        "initial_ik_safety": {"profile": "safety"},
        "outcome": {
            "status": "expected_differential_ik_invariant_failure",
            "actions_attempted": failure["policy_step"] + 1,
            "outer_steps_completed": failure["policy_step"],
            "joint_name": failure["joint_name"],
            "policy_step": failure["policy_step"],
            "physics_substep": failure["physics_substep"],
            "evidence_sha256": failure["evidence_sha256"],
        },
        "failure_exception": {
            "type": "polaris.robust_differential_ik.DifferentialIKInvariantError",
            "message": message,
        },
        "arm_failure_runtime_evidence": {"ring": "full"},
        "all_six_gripper_tail": _gripper_tail(variant),
        "close_failures": [],
    }
    assert replay.validate_capture_payload(payload) == payload


def test_output_namespace_is_variant_job_launch_specific(tmp_path: Path) -> None:
    lifecycle = {"launch_id": "d" * 64, "job_id": 456}
    valid = (
        tmp_path
        / "official_lap3b"
        / "job_456"
        / f"launch_{'d' * 64}"
        / "gate0-official_lap3b.raw.json"
    )
    replay._validate_output_namespace(
        valid, variant="official_lap3b", lifecycle=lifecycle
    )
    with pytest.raises(replay.Gate0ReplayValidationError, match="namespace"):
        replay._validate_output_namespace(
            Path(
                str(valid).replace("official_lap3b/job_456", "reasoning_43075/job_456")
            ),
            variant="official_lap3b",
            lifecycle=lifecycle,
        )
