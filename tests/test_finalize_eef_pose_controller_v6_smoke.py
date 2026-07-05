from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from scripts import finalize_eef_pose_controller_v6_smoke as finalizer


CAPTURE_ROOT = Path(
    "/home/lzha/code/ego-lap/.codex_artifacts/"
    "polaris-v6-controller-smoke-6e4b7c5-job1098922-success"
)


def _capture_paths(root: Path) -> dict[str, Path]:
    result = root / "result"
    return {
        "raw_result": result / "smoke-1098922.raw.json",
        "ready": result / "smoke-1098922.raw.json.ready.json",
        "inline_attestation": result / "smoke-1098922.host-attestation.json",
        "source_identity": result / "source-identity-1098922.sha256",
        "saved_job_script": root / "remote-saved-wrapper.sbatch",
        "slurm_log": root / "pol_v6_ctrl_6e4b7c5-1098922.out",
    }


def _validate(paths: dict[str, Path]) -> dict:
    return finalizer.validate_capture_artifacts(
        raw_result=paths["raw_result"],
        inline_attestation=paths["inline_attestation"],
        source_identity=paths["source_identity"],
        saved_job_script=paths["saved_job_script"],
        slurm_log=paths["slurm_log"],
    )


def _copy_capture(tmp_path: Path) -> dict[str, Path]:
    if not CAPTURE_ROOT.is_dir():
        pytest.skip("local immutable job-1098922 capture is unavailable")
    source = _capture_paths(CAPTURE_ROOT)
    target_root = tmp_path / "capture"
    (target_root / "result").mkdir(parents=True)
    target = _capture_paths(target_root)
    for name, source_path in source.items():
        shutil.copy2(source_path, target[name])
    return target


def test_exact_capture_identity_and_semantics_are_closed() -> None:
    if not CAPTURE_ROOT.is_dir():
        pytest.skip("local immutable job-1098922 capture is unavailable")

    validated = _validate(_capture_paths(CAPTURE_ROOT))

    assert validated["validation_summary"] == {
        "safety_report_count": 17,
        "total_controller_apply_calls": 5856,
        "total_post_policy_step_samples": 732,
        "total_open_endpoint_samples": 6003,
        "ordinary_pose_cases_passed": 13,
        "ordinary_apply_calls": 4680,
        "maximum_pose_position_error_m": 0.001071291510015726,
        "maximum_pose_rotation_error_deg": 0.4384913281711513,
        "delayed_close_apply_calls": 1000,
        "concurrent_apply_calls": 168,
        "concurrent_closed_fresh_dls_applies": 80,
        "concurrent_closed_distinct_desired_poses": 10,
        "open_endpoint_samples": 99,
        "maximum_follower_velocity_rad_s": 0.7730712294578552,
        "coupled_impulse_failure_samples": 0,
        "adversarial_apply_calls": 8,
        "adversarial_slew_events": 8,
        "recovery_events": 0,
        "controller_aborts": 0,
        "nonfatal_log_findings": [
            "headless_glfw_no_display_warnings",
            "optional_ngx_context_errors",
            "missing_viewport_camera_mesh_asset_warning",
            "eight_active_actuators_plus_five_passive_mimic_dofs_warning",
        ],
    }
    assert {
        name: entry["sha256"] for name, entry in validated["identities"].items()
    } == {name: spec["sha256"] for name, spec in finalizer.ARTIFACT_SPECS.items()}


@pytest.mark.parametrize(
    "artifact",
    [
        "raw_result",
        "ready",
        "inline_attestation",
        "source_identity",
        "saved_job_script",
        "slurm_log",
    ],
)
def test_capture_validator_rejects_every_artifact_mutation(
    tmp_path: Path, artifact: str
) -> None:
    paths = _copy_capture(tmp_path)
    path = paths[artifact]
    original_mode = path.stat().st_mode & 0o7777
    path.chmod(original_mode | 0o200)
    path.write_bytes(path.read_bytes() + b"mutation")
    path.chmod(original_mode)

    with pytest.raises(finalizer.VerificationError, match="identity drift"):
        _validate(paths)


def test_strict_json_rejects_duplicate_and_nonfinite_values() -> None:
    with pytest.raises(finalizer.VerificationError, match="Duplicate JSON key"):
        finalizer._strict_json(b'{"a": 1, "a": 2}', "duplicate")
    with pytest.raises(finalizer.VerificationError, match="Nonfinite JSON constant"):
        finalizer._strict_json(b'{"a": NaN}', "nonfinite")


@pytest.mark.parametrize(
    "mutation",
    [
        "safety_schema",
        "telemetry_failure",
        "controller_recovery_mismatch",
        "interlock_activation",
        "ordinary_missing_target_position",
        "adversarial_guard_apply_calls",
        "adversarial_guard_abort_count",
        "safety_extra_counter",
        "safety_negative_position_events",
        "soft_limit_digest_wrong",
        "gripper_static_null",
        "gripper_static_nested_drift",
        "recovery_max_active_substeps",
        "recovery_ratio_wrong",
        "recovery_state_bool_int",
        "discriminator_profile_wrong",
        "concurrent_enabled_bool_int",
        "maximum_follower_diagnostic_wrong",
        "tolerance_inflation",
        "ordinary_target_geometry_wrong",
        "adversarial_joint_position_wrong",
        "adversarial_applied_slew_wrong",
        "per_episode_post_sample_shift",
        "max_raw_diagnostic_binding",
    ],
)
def test_semantic_validator_rejects_nested_evidence_drift(mutation: str) -> None:
    if not CAPTURE_ROOT.is_dir():
        pytest.skip("local immutable job-1098922 capture is unavailable")
    raw_path = _capture_paths(CAPTURE_ROOT)["raw_result"]
    raw = json.loads(raw_path.read_text())
    mutated = deepcopy(raw)
    discriminator = mutated["concurrent_arm_gripper_discriminator"]
    if mutation == "safety_schema":
        discriminator["ik_safety"].pop("decimation")
    elif mutation == "telemetry_failure":
        telemetry = discriminator["ik_safety"]["gripper_runtime_dynamic"][
            "open_endpoint_contact_mimic_impulse"
        ]
        telemetry["coupled_impulse_failure_samples"] = 1
        telemetry["passed"] = False
    elif mutation == "controller_recovery_mismatch":
        discriminator["controller_report"]["current_joint_velocity_recovery"]["maxima"][
            "abs_velocity_to_limit_ratio"
        ] = 0.5
    elif mutation == "interlock_activation":
        discriminator["controller_report"]["gripper_close_arm_interlock"][
            "activation_count"
        ] = 1
    elif mutation == "ordinary_missing_target_position":
        mutated["results"][0].pop("target_position")
    elif mutation == "adversarial_guard_apply_calls":
        mutated["ik_safety_adversarial"]["guard_evidence"]["apply_calls"] = 999
    elif mutation == "adversarial_guard_abort_count":
        mutated["ik_safety_adversarial"]["guard_evidence"]["abort_count"] = 99
    elif mutation == "safety_extra_counter":
        mutated["ik_safety_episodes"][0]["counters"]["invented"] = 99
    elif mutation == "safety_negative_position_events":
        mutated["ik_safety_episodes"][0]["counters"]["position_limit_events"] = -99
    elif mutation == "soft_limit_digest_wrong":
        mutated["ik_safety_episodes"][0]["soft_joint_pos_limits_float32_sha256"] = (
            "0" * 64
        )
    elif mutation == "gripper_static_null":
        mutated["ik_safety_episodes"][0]["gripper_runtime_static"] = None
    elif mutation == "gripper_static_nested_drift":
        mutated["ik_safety_episodes"][0]["gripper_runtime_static"]["mimic_compliance"][
            "target_natural_frequency_rad_s"
        ] = 101.0
    elif mutation == "recovery_max_active_substeps":
        mutated["ik_safety_episodes"][0]["current_joint_velocity_recovery"]["contract"][
            "maximum_active_substeps"
        ] = 999
    elif mutation == "recovery_ratio_wrong":
        mutated["ik_safety_episodes"][0]["current_joint_velocity_recovery"]["maxima"][
            "abs_velocity_to_limit_ratio"
        ] = 999.0
    elif mutation == "recovery_state_bool_int":
        mutated["ik_safety_episodes"][0]["current_joint_velocity_recovery"]["state"][
            "active"
        ] = 0
    elif mutation == "discriminator_profile_wrong":
        discriminator["profile"] = "wrong"
    elif mutation == "concurrent_enabled_bool_int":
        discriminator["controller_report"]["concurrent_arm_gripper"]["enabled"] = 1
    elif mutation == "maximum_follower_diagnostic_wrong":
        for telemetry in (
            discriminator["open_endpoint_contact_mimic_impulse"],
            discriminator["ik_safety"]["gripper_runtime_dynamic"][
                "open_endpoint_contact_mimic_impulse"
            ],
        ):
            telemetry["maximum_follower_diagnostic"]["follower_joint_velocity_rad_s"][
                0
            ] = 999.0
    elif mutation == "tolerance_inflation":
        mutated["position_tolerance_m"] = 100.0
    elif mutation == "ordinary_target_geometry_wrong":
        mutated["results"][1]["target_position"][0] += 0.01
        mutated["results"][1]["actual_position"][0] += 0.01
    elif mutation == "adversarial_joint_position_wrong":
        mutated["ik_safety_adversarial"]["joint_state"]["joint_pos_rad"]["values"][
            0
        ] = 99.0
        mutated["ik_safety_adversarial"]["joint_state"]["joint_pos_rad"]["max_abs"] = (
            99.0
        )
    elif mutation == "adversarial_applied_slew_wrong":
        mutated["ik_safety_adversarial"]["ik_safety"]["maxima"][
            "applied_delta_joint_pos_rad"
        ][1] = 1.0
    elif mutation == "per_episode_post_sample_shift":
        mutated["ik_safety_episodes"][1]["gripper_runtime_dynamic"][
            "post_policy_step_samples"
        ] += 1
        mutated["ik_safety_episodes"][2]["gripper_runtime_dynamic"][
            "post_policy_step_samples"
        ] -= 1
    elif mutation == "max_raw_diagnostic_binding":
        mutated["ik_safety_episodes"][1]["max_raw_delta_diagnostic"][
            "raw_delta_joint_pos_rad"
        ]["values"][1] = 0.0

    with pytest.raises(finalizer.VerificationError):
        finalizer._semantic_summary(mutated)


def test_scheduler_evidence_binds_exact_terminal_allocation(monkeypatch) -> None:
    monkeypatch.setattr(finalizer.shutil, "which", lambda _: "/bin/sacct")
    monkeypatch.setattr(
        finalizer.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=(
                "1098922|pol_v6_ctrl_6e4b7c5|nvr_lpr_rvp|batch|COMPLETED|0:0|317|"
                "2026-07-04T19:18:30|2026-07-04T19:23:47|pool0-00016|"
                "billing=1,cpu=16,gres/gpu=1,mem=96G,node=1|"
                "billing=1,cpu=16,gres/gpu=1,mem=96G,node=1\n"
                "1098922.batch|batch|nvr_lpr_rvp||COMPLETED|0:0|317|"
                "2026-07-04T19:18:30|2026-07-04T19:23:47|pool0-00016|"
                "cpu=16,gres/gpu=1,mem=96G,node=1|\n"
                "1098922.extern|extern|nvr_lpr_rvp||COMPLETED|0:0|317|"
                "2026-07-04T19:18:30|2026-07-04T19:23:47|pool0-00016|"
                "billing=1,cpu=16,gres/gpu=1,mem=96G,node=1|\n"
                "1098922.0|env|nvr_lpr_rvp||COMPLETED|0:0|307|"
                "2026-07-04T19:18:40|2026-07-04T19:23:47|pool0-00016|"
                "cpu=16,gres/gpu=1,mem=96G,node=1|\n"
            )
        ),
    )

    assert finalizer._scheduler_evidence() == finalizer.EXPECTED_SCHEDULER_EVIDENCE


def test_scheduler_evidence_rejects_lifecycle_drift(monkeypatch) -> None:
    monkeypatch.setattr(finalizer.shutil, "which", lambda _: "/bin/sacct")
    monkeypatch.setattr(
        finalizer.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=(
                "1098922|pol_v6_ctrl_6e4b7c5|nvr_lpr_rvp|batch|COMPLETED|0:0|317|"
                "2026-07-04T19:18:30|2026-07-04T19:23:47|pool0-00016|"
                "billing=1,cpu=16,gres/gpu=1,mem=96G,node=1|"
                "billing=1,cpu=16,gres/gpu=1,mem=96G,node=1\n"
                "1098922.batch|batch|nvr_lpr_rvp||COMPLETED|0:0|317|"
                "2026-07-04T19:18:30|2026-07-04T19:23:47|pool0-00016|"
                "cpu=16,gres/gpu=1,mem=96G,node=1|\n"
                "1098922.extern|extern|nvr_lpr_rvp||COMPLETED|0:0|317|"
                "2026-07-04T19:18:30|2026-07-04T19:23:47|pool0-00016|"
                "billing=1,cpu=16,gres/gpu=1,mem=96G,node=1|\n"
                "1098922.0|env|nvr_lpr_rvp||FAILED|1:0|307|"
                "2026-07-04T19:18:40|2026-07-04T19:23:47|pool0-00016|"
                "cpu=16,gres/gpu=1,mem=96G,node=1|\n"
            )
        ),
    )

    with pytest.raises(finalizer.VerificationError, match="lifecycle drift"):
        finalizer._scheduler_evidence()


def test_publisher_is_nonoverwriting_and_mode_0444(tmp_path: Path) -> None:
    path = tmp_path / "attestation.json"
    payload = {"schema_version": 1, "passed": True}

    finalizer._publish_nonoverwriting(path, payload)

    assert path.stat().st_mode & 0o7777 == 0o444
    assert path.stat().st_nlink == 1
    assert json.loads(path.read_text()) == payload
    with pytest.raises(FileExistsError):
        finalizer._publish_nonoverwriting(path, payload)


def test_serialized_attestation_comparison_is_bool_int_type_strict() -> None:
    expected = {"schema_version": 1, "finalized": True, "passed": True}
    mutated = finalizer._serialized(
        {"schema_version": True, "finalized": 1, "passed": 1}
    )
    assert finalizer._strict_json(mutated, "old equality leak") == expected
    with pytest.raises(finalizer.VerificationError, match="byte content drift"):
        finalizer._validate_exact_serialized_payload(
            mutated, expected, "promotion attestation"
        )


def test_sealed_provenance_resume_accepts_only_identical_bytes(tmp_path: Path) -> None:
    path = tmp_path / "sealed.bin"
    finalizer._publish_or_validate_bytes(path, b"exact", "sealed test")
    finalizer._publish_or_validate_bytes(path, b"exact", "sealed test")

    assert path.stat().st_mode & 0o7777 == 0o444
    with pytest.raises(finalizer.VerificationError, match="content drift"):
        finalizer._publish_or_validate_bytes(path, b"different", "sealed test")


def test_reader_rejects_hardlinked_evidence(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"evidence")
    second.hardlink_to(first)

    with pytest.raises(finalizer.VerificationError, match="exactly one hard link"):
        finalizer._read_regular_file(first, "hardlinked evidence")


def test_reader_rejects_symlink_and_parent_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "current"
    parent.mkdir()
    path = parent / "evidence"
    path.write_bytes(b"old-evidence")
    symlink = tmp_path / "symlink"
    symlink.symlink_to(path)
    with pytest.raises(finalizer.VerificationError, match="not a regular file"):
        finalizer._read_regular_file(symlink, "symlinked evidence")

    original_read = finalizer.os.read
    swapped = False

    def swap_parent_after_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        data = original_read(descriptor, size)
        if data and not swapped:
            swapped = True
            parent.rename(tmp_path / "old-parent")
            parent.mkdir()
            (parent / "evidence").write_bytes(b"evil-evidence")
        return data

    monkeypatch.setattr(finalizer.os, "read", swap_parent_after_read)
    with pytest.raises(finalizer.VerificationError, match="path changed"):
        finalizer._read_regular_file(path, "swapped evidence")


def test_external_hasher_rejects_mode_links_and_parent_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "current"
    parent.mkdir()
    path = parent / "external"
    data = b"external-bytes"
    path.write_bytes(data)
    path.chmod(0o644)
    digest = finalizer.hashlib.sha256(data).hexdigest()

    path.chmod(0o640)
    with pytest.raises(finalizer.VerificationError, match="mode must be 0644"):
        finalizer._hash_external(path, "external", digest, required_mode=0o644)
    path.chmod(0o644)

    hardlink = tmp_path / "hardlink"
    hardlink.hardlink_to(path)
    with pytest.raises(finalizer.VerificationError, match="exactly one hard link"):
        finalizer._hash_external(path, "external", digest, required_mode=0o644)
    hardlink.unlink()

    symlink = tmp_path / "symlink"
    symlink.symlink_to(path)
    with pytest.raises(finalizer.VerificationError, match="not a regular file"):
        finalizer._hash_external(symlink, "external", digest, required_mode=0o644)

    original_read = finalizer.os.read
    swapped = False

    def swap_parent_after_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        chunk = original_read(descriptor, size)
        if chunk and not swapped:
            swapped = True
            parent.rename(tmp_path / "old-parent")
            parent.mkdir()
            replacement = parent / "external"
            replacement.write_bytes(b"replacement")
            replacement.chmod(0o644)
        return chunk

    monkeypatch.setattr(finalizer.os, "read", swap_parent_after_read)
    with pytest.raises(finalizer.VerificationError, match="path changed"):
        finalizer._hash_external(path, "external", digest, required_mode=0o644)


def test_external_hasher_binds_metadata_and_job_time(tmp_path: Path) -> None:
    path = tmp_path / "external"
    data = b"external-bytes"
    path.write_bytes(data)
    path.chmod(0o644)
    status = path.stat()
    digest = finalizer.hashlib.sha256(data).hexdigest()
    metadata = {
        "size_bytes": status.st_size,
        "mtime_ns": status.st_mtime_ns,
        "ctime_ns": status.st_ctime_ns,
    }

    identity = finalizer._hash_external(
        path,
        "external",
        digest,
        required_mode=0o644,
        expected_metadata=metadata,
        must_predate_ns=max(status.st_mtime_ns, status.st_ctime_ns),
    )
    assert {
        name: identity[name] for name in ("size_bytes", "mtime_ns", "ctime_ns")
    } == metadata

    with pytest.raises(finalizer.VerificationError, match="metadata drift"):
        finalizer._hash_external(
            path,
            "external",
            digest,
            required_mode=0o644,
            expected_metadata={**metadata, "size_bytes": status.st_size + 1},
        )
    with pytest.raises(finalizer.VerificationError, match="does not predate"):
        finalizer._hash_external(
            path,
            "external",
            digest,
            required_mode=0o644,
            must_predate_ns=max(status.st_mtime_ns, status.st_ctime_ns) - 1,
        )


def test_l401_path_binding_allows_only_the_expected_canonical_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = tmp_path / "canonical-user"
    canonical.mkdir()
    literal = tmp_path / "literal-user"
    literal.symlink_to(canonical, target_is_directory=True)
    target = canonical / "results" / "artifact"
    target.parent.mkdir()
    target.write_bytes(b"artifact")
    aliased = literal / "results" / "artifact"
    monkeypatch.setattr(finalizer, "L401_LITERAL_USER_ROOT", literal)
    monkeypatch.setattr(finalizer, "L401_CANONICAL_USER_ROOT", canonical)

    finalizer._exact_cli_path(aliased, str(aliased), "artifact")

    outside = tmp_path / "outside"
    outside.mkdir()
    (canonical / "results" / "redirect").symlink_to(outside, target_is_directory=True)
    redirected = literal / "results" / "redirect" / "artifact"
    with pytest.raises(finalizer.VerificationError, match="canonical path drift"):
        finalizer._exact_cli_path(redirected, str(redirected), "redirected artifact")


def test_repository_layout_requires_top_level_in_root_git_and_detached_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    for command in (
        ["git", "init", "-q", str(repo)],
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
        ["git", "-C", str(repo), "config", "user.name", "Test"],
    ):
        finalizer.subprocess.run(command, check=True)
    (repo / "tracked").write_bytes(b"tracked")
    finalizer.subprocess.run(["git", "-C", str(repo), "add", "tracked"], check=True)
    finalizer.subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "test"], check=True
    )

    with pytest.raises(finalizer.VerificationError, match="not detached"):
        finalizer._validate_standalone_detached_repository(repo, "repo")
    finalizer.subprocess.run(
        ["git", "-C", str(repo), "checkout", "-q", "--detach"], check=True
    )
    assert finalizer._validate_standalone_detached_repository(repo, "repo") == (
        repo.resolve()
    )

    subdirectory = repo / "nested"
    subdirectory.mkdir()
    with pytest.raises(
        finalizer.VerificationError, match="Cannot inspect nested layout"
    ):
        finalizer._validate_standalone_detached_repository(subdirectory, "nested")

    alias = tmp_path / "repo-alias"
    alias.symlink_to(repo, target_is_directory=True)
    with pytest.raises(finalizer.VerificationError, match="root is not a directory"):
        finalizer._validate_standalone_detached_repository(alias, "alias")

    git_directory = repo / ".git"
    external_common_directory = tmp_path / "external-common"
    external_common_directory.mkdir()
    real_git = finalizer._git

    def external_common(repo_path: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "--git-common-dir"):
            return str(external_common_directory)
        return real_git(repo_path, *arguments)

    with monkeypatch.context() as scoped:
        scoped.setattr(finalizer, "_git", external_common)
        with pytest.raises(
            finalizer.VerificationError, match="common-dir is not in-root"
        ):
            finalizer._validate_standalone_detached_repository(repo, "external common")

    real_git_directory = repo / ".git-real"
    git_directory.rename(real_git_directory)
    git_directory.symlink_to(real_git_directory, target_is_directory=True)
    with pytest.raises(finalizer.VerificationError, match=r"\.git is not a directory"):
        finalizer._validate_standalone_detached_repository(repo, "redirected git")


def test_sacct_snapshot_is_exact_and_self_consistent() -> None:
    assert (
        finalizer._strict_json(finalizer.SACCT_SNAPSHOT_BYTES, "sacct snapshot")
        == finalizer.SACCT_SNAPSHOT_PAYLOAD
    )
    assert finalizer.SEALED_PROVENANCE_SPECS["sacct"] == {
        "size_bytes": len(finalizer.SACCT_SNAPSHOT_BYTES),
        "sha256": finalizer.hashlib.sha256(finalizer.SACCT_SNAPSHOT_BYTES).hexdigest(),
        "mode": "0444",
    }


def test_exact_producer_sources_remain_bound() -> None:
    repo = Path(__file__).parents[1]

    assert finalizer.validate_producer_source_identity(repo) == (
        finalizer.PRODUCER_SOURCE_SHA256
    )


def test_finalizer_is_stdlib_only_and_v5_modules_are_not_modified() -> None:
    source_path = Path(finalizer.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    allowed_roots = {
        "argparse",
        "copy",
        "hashlib",
        "json",
        "math",
        "os",
        "pathlib",
        "shutil",
        "stat",
        "struct",
        "subprocess",
        "sys",
        "typing",
        "__future__",
    }
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", 1)[0])

    assert imported_roots <= allowed_roots
    assert "eef_velocity_recovery_promotion" not in source_path.read_text()
    assert "eef_velocity_recovery_standard_promotion" not in source_path.read_text()


def test_capture_constants_pin_exact_job_and_authorize_no_checkpoint() -> None:
    assert finalizer.JOB_ID == 1098922
    assert finalizer.PRODUCER_COMMIT == ("6e4b7c5be5ff6db670970774be3250c5d5ffa4d2")
    assert finalizer.PRODUCER_PARENT == ("b6dec3d0c053066d65f6998cf1ecc33fc6e6e9ff")
    assert finalizer.PROMOTION_STATUS.endswith("pending_two_checkpoint_canaries")
    assert finalizer.PROMOTION_SCOPE == (
        "standalone_controller_smoke_only_no_checkpoint_or_task_metric"
    )
    assert (
        max(
            finalizer.IMAGE_METADATA["mtime_ns"],
            finalizer.IMAGE_METADATA["ctime_ns"],
            finalizer.SCENE_METADATA["mtime_ns"],
            finalizer.SCENE_METADATA["ctime_ns"],
        )
        < finalizer.SRUN_START_EPOCH_NS
    )
