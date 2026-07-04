import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from polaris import pi05_droid_native_eval_contract as native_eval_contract
from polaris.pi05_droid_jointvelocity_contract import (
    NATIVE_GRIPPER_DRIVE_PROFILE,
    PI05_DROID_GRIPPER_OBSERVATION_BOUND_TOLERANCE,
    PI05_DROID_GRIPPER_OBSERVATION_CONTRACT,
)
from polaris.pi05_droid_native_eval_contract import (
    PI05_DROID_GRIPPER_DRIVE_PROFILE,
    PI05_DROID_NATIVE_CONFIGURED_EPISODE_LENGTH_SECONDS,
    PI05_DROID_NATIVE_EPISODE_STEPS,
    PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS,
    PI05_DROID_NATIVE_MODEL_EVAL_CONTRACT,
    configure_native_environment_timeout,
    make_close_ready_artifact,
    make_episode_sidecar,
    make_environment_runtime_contract,
    make_runtime_artifact,
    publish_immutable_file_from_temporary,
    publish_immutable_json,
    should_render_expensive,
    validate_bound_artifact,
    validate_immutable_json,
    validate_native_model_eval_contract,
    validate_outer_step_flags,
    validate_terminal_rollout_evidence,
)
from polaris.native_gripper_runtime import (
    EXPECTED_DROID_JOINT_NAMES,
    NATIVE_GRIPPER_DYNAMIC_PROFILE,
)


@pytest.mark.parametrize(
    ("render_every_step", "needs_next_policy_render", "expected"),
    [
        (False, False, False),
        (False, True, True),
        (True, False, True),
        (True, True, True),
    ],
)
def test_render_every_step_is_independent_of_policy_query_cadence(
    render_every_step, needs_next_policy_render, expected
):
    assert (
        should_render_expensive(
            policy_client_name="DroidJointVelocity",
            render_every_step=render_every_step,
            needs_next_policy_render=needs_next_policy_render,
        )
        is expected
    )


@pytest.mark.parametrize("invalid", [0, 1, None, "true"])
def test_render_decision_requires_exact_booleans(invalid):
    with pytest.raises(TypeError, match="exact booleans"):
        should_render_expensive(
            policy_client_name="DroidJointVelocity",
            render_every_step=invalid,
            needs_next_policy_render=False,
        )


@pytest.mark.parametrize("policy_client_name", ["DroidJointPos", "EgoLAPEefPose"])
def test_non_native_render_decision_preserves_policy_rerender_exactly(
    policy_client_name,
):
    rerender = object()
    assert (
        should_render_expensive(
            policy_client_name=policy_client_name,
            render_every_step=True,
            needs_next_policy_render=rerender,
        )
        is rerender
    )


def test_integrated_tree_uses_promoted_native_gripper_drive_profile():
    assert PI05_DROID_GRIPPER_DRIVE_PROFILE == NATIVE_GRIPPER_DRIVE_PROFILE


def _environment_runtime():
    return make_environment_runtime_contract(
        configured_episode_length_seconds=(
            PI05_DROID_NATIVE_CONFIGURED_EPISODE_LENGTH_SECONDS
        ),
        live_max_episode_length=PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS,
    )


def _terminal_rollout(runtime):
    before = {
        "live_max_episode_length": PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS,
        "episode_length": 0,
        "sim_step_counter": 17,
        "common_step_counter": 3,
        "sensor_frame_counters": {"external_cam": 1, "wrist_cam": 1},
    }
    after = {
        "live_max_episode_length": PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS,
        "episode_length": PI05_DROID_NATIVE_EPISODE_STEPS,
        "sim_step_counter": 17 + PI05_DROID_NATIVE_EPISODE_STEPS * 8,
        "common_step_counter": 3 + PI05_DROID_NATIVE_EPISODE_STEPS,
        "sensor_frame_counters": {
            "external_cam": 1 + PI05_DROID_NATIVE_EPISODE_STEPS,
            "wrist_cam": 1 + PI05_DROID_NATIVE_EPISODE_STEPS,
        },
    }
    return {
        "schema_version": 1,
        "profile": runtime["profile"],
        "environment_runtime_sha256": runtime["sha256"],
        "outer_steps_completed": PI05_DROID_NATIVE_EPISODE_STEPS,
        "last_outer_step_index": PI05_DROID_NATIVE_EPISODE_STEPS - 1,
        "terminated_false_count": PI05_DROID_NATIVE_EPISODE_STEPS,
        "truncated_false_count": PI05_DROID_NATIVE_EPISODE_STEPS,
        "environment_before": before,
        "environment_after": after,
        "rubric": {"success": False, "progress": 0.25},
    }


def test_native_timeout_is_configured_before_construction_and_live_margin_is_closed():
    cfg = SimpleNamespace(episode_length_s=30.0)
    configured = configure_native_environment_timeout(cfg)
    assert configured == PI05_DROID_NATIVE_CONFIGURED_EPISODE_LENGTH_SECONDS
    assert configured * 15 == PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS
    runtime = _environment_runtime()
    assert runtime["outer_episode_steps"] == 450
    assert runtime["live_max_episode_length"] == 451
    assert runtime["timeout_margin_steps"] == 1
    assert runtime["sensor_liveness"]["image_hash_variation_authoritative"] is False

    with pytest.raises(ValueError, match="exceed 450"):
        make_environment_runtime_contract(
            configured_episode_length_seconds=configured,
            live_max_episode_length=450,
        )


def test_step_450_timeout_boundary_and_reset_terminal_evidence_fail_closed():
    assert validate_outer_step_flags([False], [False], outer_step_index=449) == {
        "outer_step_index": 449,
        "terminated": False,
        "truncated": False,
    }
    with pytest.raises(ValueError, match="auto-reset boundary"):
        validate_outer_step_flags([False], [True], outer_step_index=449)

    runtime = _environment_runtime()
    terminal = _terminal_rollout(runtime)
    assert validate_terminal_rollout_evidence(terminal, runtime) == terminal
    reset_terminal = copy.deepcopy(terminal)
    reset_terminal["environment_after"]["episode_length"] = 0
    reset_terminal["environment_after"]["sensor_frame_counters"] = {
        "external_cam": 1,
        "wrist_cam": 1,
    }
    with pytest.raises(ValueError, match="auto-reset"):
        validate_terminal_rollout_evidence(reset_terminal, runtime)


def test_runtime_and_close_artifacts_bind_internal_timeout_and_terminal_state(tmp_path):
    runtime = _environment_runtime()
    value = make_runtime_artifact({"runtime_sha256": "a" * 64}, runtime)
    assert value["environment_runtime_contract"] == runtime
    runtime_path = tmp_path / "runtime.json"
    identity = {
        "path": str(runtime_path.resolve()),
        "size": 1,
        "sha256": "b" * 64,
        "mode": "0444",
        "nlink": 1,
    }
    terminal = _terminal_rollout(runtime)
    result = {
        "episode": 0,
        "episode_length": 450,
        "success": False,
        "progress": 0.25,
        "numerical_failure": False,
        "numerical_failure_reason": "",
    }
    artifacts = {}
    for label in ("trace", "video"):
        temporary = tmp_path / f".{label}.partial"
        temporary.write_bytes(label.encode("ascii"))
        artifacts[label] = publish_immutable_file_from_temporary(
            temporary, tmp_path / f"{label}.bin"
        )
    dynamic = {
        "schema_version": 3,
        "profile": NATIVE_GRIPPER_DYNAMIC_PROFILE,
        "joint_names": list(EXPECTED_DROID_JOINT_NAMES),
        "joint_indices": list(range(13)),
        "apply_calls": 3600,
        "post_policy_step_samples": 450,
        "sample_count": 4050,
        "max_abs_joint_velocity_rad_s": [0.0] * 13,
        "max_abs_joint_acceleration_rad_s2": [0.0] * 13,
        "terminal_velocity_failure": None,
        "samples": None,
    }
    sidecar_path = tmp_path / "sidecar.json"
    sidecar = publish_immutable_json(
        sidecar_path,
        make_episode_sidecar(
            episode_result=result,
            terminal_outcome=terminal,
            environment_runtime_contract=runtime,
            dynamic_report=dynamic,
            trace_artifact=artifacts["trace"],
            video_artifact=artifacts["video"],
            incident_artifact=None,
        ),
    )
    sidecar_identity = {
        key: sidecar[key] for key in ("path", "size", "sha256", "mode", "nlink")
    }
    close = make_close_ready_artifact(
        runtime_artifact=identity,
        runtime_path=runtime_path,
        metrics_path=Path(tmp_path / "metrics.csv"),
        trace_path=Path(tmp_path / "trace.jsonl"),
        video_path=Path(tmp_path / "video.mp4"),
        environment_runtime_contract=runtime,
        terminal_outcome=terminal,
        episode_sidecar=sidecar_identity,
    )
    assert close["environment_runtime_contract_sha256"] == runtime["sha256"]
    assert close["terminal_outcome"] == terminal
    assert close["episode_sidecar"] == sidecar_identity


def test_close_ready_consumes_sidecar_from_stable_bound_read(tmp_path, monkeypatch):
    runtime = _environment_runtime()
    original_terminal = _terminal_rollout(runtime)
    substituted_terminal = copy.deepcopy(original_terminal)
    substituted_terminal["rubric"]["progress"] = 0.75
    original_result = {
        "episode": 0,
        "episode_length": 450,
        "success": False,
        "progress": 0.25,
        "numerical_failure": False,
        "numerical_failure_reason": "",
    }
    substituted_result = {**original_result, "progress": 0.75}
    dynamic = {
        "schema_version": 3,
        "profile": NATIVE_GRIPPER_DYNAMIC_PROFILE,
        "joint_names": list(EXPECTED_DROID_JOINT_NAMES),
        "joint_indices": list(range(13)),
        "apply_calls": 3600,
        "post_policy_step_samples": 450,
        "sample_count": 4050,
        "max_abs_joint_velocity_rad_s": [0.0] * 13,
        "max_abs_joint_acceleration_rad_s2": [0.0] * 13,
        "terminal_velocity_failure": None,
        "samples": None,
    }
    artifacts = {}
    for label in ("trace", "video"):
        temporary = tmp_path / f".{label}.partial"
        temporary.write_bytes(label.encode("ascii"))
        artifacts[label] = publish_immutable_file_from_temporary(
            temporary, tmp_path / f"{label}.bin"
        )
    original_root = tmp_path / "original"
    original_root.mkdir()
    wrong_root = tmp_path / "wrong"
    wrong_root.mkdir()
    alias_root = tmp_path / "alias"
    alias_root.symlink_to(original_root, target_is_directory=True)
    filename = "sidecar.json"
    original_sidecar = publish_immutable_json(
        original_root / filename,
        make_episode_sidecar(
            episode_result=original_result,
            terminal_outcome=original_terminal,
            environment_runtime_contract=runtime,
            dynamic_report=dynamic,
            trace_artifact=artifacts["trace"],
            video_artifact=artifacts["video"],
            incident_artifact=None,
        ),
    )
    substituted_sidecar = publish_immutable_json(
        wrong_root / filename,
        make_episode_sidecar(
            episode_result=substituted_result,
            terminal_outcome=substituted_terminal,
            environment_runtime_contract=runtime,
            dynamic_report=dynamic,
            trace_artifact=artifacts["trace"],
            video_artifact=artifacts["video"],
            incident_artifact=None,
        ),
    )
    assert original_sidecar["sha256"] != substituted_sidecar["sha256"]
    recorded = {
        key: original_sidecar[key]
        for key in ("path", "size", "sha256", "mode", "nlink")
    }
    recorded["path"] = str(alias_root / filename)
    runtime_path = tmp_path / "runtime.json"
    runtime_artifact = publish_immutable_json(runtime_path, {"runtime": "synthetic"})

    stable_validate = native_eval_contract.validate_bound_json_artifact
    retargeted = False

    def retarget_after_stable_read(*args, **kwargs):
        nonlocal retargeted
        artifact = stable_validate(*args, **kwargs)
        alias_root.unlink()
        alias_root.symlink_to(wrong_root, target_is_directory=True)
        retargeted = True
        return artifact

    monkeypatch.setattr(
        native_eval_contract,
        "validate_bound_json_artifact",
        retarget_after_stable_read,
    )
    with pytest.raises(ValueError, match="Close-ready terminal/sidecar drift"):
        make_close_ready_artifact(
            runtime_artifact=runtime_artifact,
            runtime_path=runtime_path,
            metrics_path=tmp_path / "metrics.csv",
            trace_path=tmp_path / "trace.jsonl",
            video_path=tmp_path / "video.mp4",
            environment_runtime_contract=runtime,
            terminal_outcome=substituted_terminal,
            episode_sidecar=recorded,
        )
    assert retargeted is True
    assert (alias_root / filename).resolve() == wrong_root / filename


def test_immutable_json_is_canonical_single_link_mode_0444(tmp_path):
    path = tmp_path / "artifact.json"
    identity = publish_immutable_json(path, {"z": 1, "a": [True, None]})

    assert path.read_bytes() == b'{"a":[true,null],"z":1}\n'
    assert identity["mode"] == "0444"
    assert identity["nlink"] == 1
    assert validate_immutable_json(path)["sha256"] == identity["sha256"]
    with pytest.raises(FileExistsError):
        publish_immutable_json(path, {"replacement": True})

    path.chmod(0o644)
    with pytest.raises(ValueError, match="mode 0444"):
        validate_immutable_json(path)


def test_immutable_json_rejects_noncanonical_links_and_symlinks(tmp_path):
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps({"b": 1, "a": 2}), encoding="ascii")
    noncanonical.chmod(0o444)
    with pytest.raises(ValueError, match="canonical JSON"):
        validate_immutable_json(noncanonical)

    linked = tmp_path / "linked.json"
    publish_immutable_json(linked, {"value": 1})
    os.link(linked, tmp_path / "second-link.json")
    with pytest.raises(ValueError, match="one regular link"):
        validate_immutable_json(linked)

    target = tmp_path / "target.json"
    publish_immutable_json(target, {"value": 2})
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target)
    with pytest.raises(ValueError, match="not readable"):
        validate_immutable_json(symlink)


def test_bound_artifact_accepts_parent_alias_but_rejects_target_and_hash_drift(
    tmp_path,
):
    real_root = tmp_path / "real"
    real_root.mkdir()
    alias_root = tmp_path / "alias"
    alias_root.symlink_to(real_root, target_is_directory=True)
    artifact_path = real_root / "incident.json"
    identity = publish_immutable_json(artifact_path, {"incident": "typed"})
    recorded = {
        key: identity[key] for key in ("path", "size", "sha256", "mode", "nlink")
    }
    recorded["path"] = str(alias_root / artifact_path.name)

    assert (
        validate_bound_artifact(
            recorded,
            expected_path=artifact_path,
            field="native velocity incident",
            json_artifact=True,
        )
        == recorded
    )

    wrong_root = tmp_path / "wrong"
    wrong_root.mkdir()
    wrong_path = wrong_root / "incident.json"
    publish_immutable_json(wrong_path, {"incident": "typed"})
    wrong_target = {**recorded, "path": str(wrong_path)}
    with pytest.raises(ValueError, match="artifact path drift"):
        validate_bound_artifact(
            wrong_target,
            expected_path=artifact_path,
            field="native velocity incident",
            json_artifact=True,
        )

    identity_mutations = {
        "size": recorded["size"] + 1,
        "sha256": "0" * 64,
        "mode": "0644",
        "nlink": recorded["nlink"] + 1,
    }
    for key, wrong_value in identity_mutations.items():
        drifted = {**recorded, key: wrong_value}
        with pytest.raises(ValueError, match="artifact identity drift"):
            validate_bound_artifact(
                drifted,
                expected_path=artifact_path,
                field="native velocity incident",
                json_artifact=True,
            )


@pytest.mark.parametrize("retarget_stage", ["before_open", "during_read"])
@pytest.mark.parametrize("json_artifact", [False, True])
def test_bound_artifact_rejects_parent_alias_retarget_during_validation(
    tmp_path,
    monkeypatch,
    json_artifact,
    retarget_stage,
):
    original_root = tmp_path / "original"
    original_root.mkdir()
    wrong_root = tmp_path / "wrong"
    wrong_root.mkdir()
    alias_root = tmp_path / "alias"
    alias_root.symlink_to(original_root, target_is_directory=True)
    filename = "artifact.json" if json_artifact else "artifact.bin"
    original_path = original_root / filename
    wrong_path = wrong_root / filename
    if json_artifact:
        original = publish_immutable_json(original_path, {"same": "bytes"})
        publish_immutable_json(wrong_path, {"same": "bytes"})
    else:
        for root in (original_root, wrong_root):
            temporary = root / ".artifact.partial"
            temporary.write_bytes(b"same bytes")
            published = publish_immutable_file_from_temporary(
                temporary,
                root / filename,
            )
            if root == original_root:
                original = published
    recorded = {
        key: original[key] for key in ("path", "size", "sha256", "mode", "nlink")
    }
    recorded["path"] = str(alias_root / filename)

    retargeted = False

    def retarget_alias():
        nonlocal retargeted
        if not retargeted:
            alias_root.unlink()
            alias_root.symlink_to(wrong_root, target_is_directory=True)
            retargeted = True

    original_read = native_eval_contract.os.read

    def retarget_then_read(descriptor, size):
        retarget_alias()
        return original_read(descriptor, size)

    original_open = native_eval_contract.os.open
    parent_open_count = 0

    def retarget_then_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal parent_open_count
        if dir_fd is None and flags & os.O_DIRECTORY:
            parent_open_count += 1
            if parent_open_count == 2:
                retarget_alias()
        return original_open(path, flags, mode, dir_fd=dir_fd)

    if retarget_stage == "before_open":
        monkeypatch.setattr(native_eval_contract.os, "open", retarget_then_open)
    else:
        monkeypatch.setattr(native_eval_contract.os, "read", retarget_then_read)
    with pytest.raises(ValueError, match="target changed during validation"):
        validate_bound_artifact(
            recorded,
            expected_path=original_path,
            field="retargeted artifact",
            json_artifact=json_artifact,
        )
    assert retargeted is True


@pytest.mark.parametrize(
    ("key", "invalid"),
    [
        ("path", Path("/tmp/not-a-string")),
        ("path", "relative/artifact.json"),
        ("size", True),
        ("size", 1.0),
        ("sha256", b"0" * 64),
        ("sha256", "A" * 64),
        ("mode", 0o444),
        ("nlink", True),
    ],
)
def test_bound_artifact_requires_exact_identity_types(tmp_path, key, invalid):
    path = tmp_path / "artifact.json"
    identity = publish_immutable_json(path, {"strict": True})
    recorded = {
        name: identity[name] for name in ("path", "size", "sha256", "mode", "nlink")
    }
    recorded[key] = invalid
    with pytest.raises(ValueError, match="artifact identity type drift"):
        validate_bound_artifact(
            recorded,
            expected_path=path,
            field="strict artifact",
            json_artifact=True,
        )


def test_exact_official_model_eval_contract_binds_all_train_eval_semantics():
    contract = validate_native_model_eval_contract(
        PI05_DROID_NATIVE_MODEL_EVAL_CONTRACT
    )

    assert contract["checkpoint"]["content_manifest_sha256"] == (
        "6f9ccfa5695c669962ad10dbe0dcb7d44bf903918e5fffe33e5d1ff531287922"
    )
    assert {
        key: contract["normalization"][key]
        for key in (
            "asset_id",
            "scope",
            "path",
            "sha256",
            "category_override",
            "rejected_category_substitutions",
        )
    } == {
        "asset_id": "droid",
        "scope": "checkpoint_global_droid",
        "path": "assets/droid/norm_stats.json",
        "sha256": "403b3a22f897e9ae5dd617966a3c8f7d1835ac79dfd5a8993179514be26a3b8b",
        "category_override": "forbidden",
        "rejected_category_substitutions": [
            "single_arm",
            "single-arm",
            "single arm",
        ],
    }
    assert contract["normalization"]["reference_probes"]["actions_q01_first8"][:7] == [
        -0.45799999999999996,
        -0.8076,
        -0.44719999999999993,
        -0.9268,
        -0.6456,
        -0.6459999999999999,
        -0.7616,
    ]
    assert (
        contract["normalization"]["reference_probes"]["actions_q99_first8"][7] == 0.9998
    )
    assert contract["normalization"]["reference_probes"]["state_q01_first8"][7] == 0.0
    assert contract["normalization"]["reference_probes"]["state_q99_first8"][7] == 0.991
    assert [
        (image["source"], image["model_slot"], image["rotation_degrees"])
        for image in contract["policy_input"]["images"]
    ] == [
        ("external", "base_0_rgb", 0),
        ("wrist", "left_wrist_0_rgb", 0),
        ("zero_blank", "right_wrist_0_rgb", 0),
    ]
    assert contract["policy_input"]["resize"] == (
        "openpi_image_tools_resize_with_pad_224_v1"
    )
    assert contract["policy_input"]["state"] == (
        "7_panda_joint_positions_radians_plus_closed_positive_gripper"
    )
    assert contract["policy_input"]["gripper_observation"] == (
        PI05_DROID_GRIPPER_OBSERVATION_CONTRACT
    )
    assert (
        contract["policy_input"]["gripper_observation"]["boundary_audit_tolerance"]
        == PI05_DROID_GRIPPER_OBSERVATION_BOUND_TOLERANCE
    )
    assert contract["policy_output"]["response_shape"] == [15, 8]
    assert contract["policy_output"]["execute_first"] == 8
    assert contract["policy_output"]["policy_frequency_hz"] == 15
    assert contract["policy_output"]["action_transform"] == (
        "none_before_DroidOutputs_leading8_projection"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["normalization"].update(
            {"asset_id": "single_arm", "scope": "category_single_arm"}
        ),
        lambda value: value["policy_input"]["images"][0].update(
            {"model_slot": "left_wrist_0_rgb"}
        ),
        lambda value: value["policy_input"]["images"][1].update(
            {"rotation_degrees": 180}
        ),
        lambda value: value["policy_input"]["images"][2].update(
            {"shape": [256, 256, 3]}
        ),
        lambda value: value["policy_input"].update({"state_width": 7}),
        lambda value: value["policy_input"]["gripper_observation"].update(
            {"boundary_audit_tolerance": 1e-3}
        ),
        lambda value: value["policy_output"].update({"response_shape": [16, 8]}),
        lambda value: value["policy_output"].update({"execute_first": 15}),
    ],
)
def test_official_model_eval_contract_rejects_every_semantic_substitution(mutation):
    candidate = copy.deepcopy(PI05_DROID_NATIVE_MODEL_EVAL_CONTRACT)
    mutation(candidate)
    with pytest.raises(ValueError, match="model eval contract mismatch"):
        validate_native_model_eval_contract(candidate)
