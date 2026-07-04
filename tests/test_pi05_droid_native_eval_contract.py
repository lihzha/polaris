import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from polaris.pi05_droid_jointvelocity_contract import NATIVE_GRIPPER_DRIVE_PROFILE
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
        lambda value: value["policy_output"].update({"response_shape": [16, 8]}),
        lambda value: value["policy_output"].update({"execute_first": 15}),
    ],
)
def test_official_model_eval_contract_rejects_every_semantic_substitution(mutation):
    candidate = copy.deepcopy(PI05_DROID_NATIVE_MODEL_EVAL_CONTRACT)
    mutation(candidate)
    with pytest.raises(ValueError, match="model eval contract mismatch"):
        validate_native_model_eval_contract(candidate)
