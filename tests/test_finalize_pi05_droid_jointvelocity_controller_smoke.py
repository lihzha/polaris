import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from polaris.joint_velocity_smoke import validate_joint_velocity_smoke
from scripts import smoke_joint_velocity_controller as controller
from scripts.polaris import (
    finalize_pi05_droid_jointvelocity_controller_smoke as finalizer,
)


def _canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
    )


def _immutable(path: Path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, payload)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)


def _publish_transactional_smoke(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = path.with_name(path.name + ".child-close.json")
    ready_path = raw_path.with_name(raw_path.name + ".ready.json")
    failure_path = raw_path.with_name(raw_path.name + ".failure.json")
    child = copy.deepcopy(payload)
    child.pop("completion")
    child["lifecycle"] = {
        "env_close": "complete",
        "simulation_app_close": "pending_child_exit",
        "capture_stage": "kit_child_after_env_close_before_simulation_app_close",
    }
    validated = validate_joint_velocity_smoke(child, require_parent_completion=False)
    raw_bytes = controller._write_immutable_json(raw_path, validated)
    controller._write_immutable_json(
        ready_path, controller._child_ready_payload(raw_path, raw_bytes)
    )
    controller.finalize_child_capture(
        raw_path, ready_path, failure_path, path, child_exit_code=0
    )


def _prepare_provenance(tmp_path, monkeypatch):
    repository = tmp_path / "polaris"
    repository.mkdir()
    for relative_path in finalizer.SOURCE_PATHS:
        path = repository / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source:{relative_path}\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", repository], check=True)
    subprocess.run(["git", "-C", repository, "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", repository, "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", repository, "add", "."], check=True)
    subprocess.run(
        ["git", "-C", repository, "commit", "-qm", "test provenance"], check=True
    )
    commit = subprocess.run(
        ["git", "-C", repository, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", repository, "checkout", "--detach", "-q", commit], check=True
    )

    image = tmp_path / "runtime.sqsh"
    image.write_bytes(b"pinned-test-image")
    monkeypatch.setattr(
        finalizer, "IMAGE_SHA256", hashlib.sha256(image.read_bytes()).hexdigest()
    )

    data_dir = tmp_path / "PolaRiS-Hub"
    assets = {}
    for index, relative_path in enumerate(finalizer.ASSETS):
        asset = data_dir / relative_path
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(f"asset-{index}".encode("ascii"))
        metadata = (
            data_dir / ".cache/huggingface/download" / (relative_path + ".metadata")
        )
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(finalizer.HUB_REVISION + "\ntest-etag\n", encoding="utf-8")
        assets[relative_path] = {
            "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
            "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        }
    monkeypatch.setattr(finalizer, "ASSETS", assets)
    return repository, commit, image, data_dir


def _argv(
    *,
    mode,
    job_id,
    smoke,
    completion,
    status,
    gpu,
    saved,
    repository,
    commit,
    image,
    data_dir,
    gripper_profile=finalizer.GRIPPER_DRIVE_PROFILE,
):
    return [
        mode,
        "--job-id",
        str(job_id),
        "--smoke-artifact",
        str(smoke),
        "--completion",
        str(completion),
        "--srun-status",
        str(status),
        "--gpu-inventory",
        str(gpu),
        "--saved-job-script",
        str(saved),
        "--polaris-repo",
        str(repository),
        "--expected-polaris-commit",
        commit,
        "--expected-gripper-drive-profile",
        gripper_profile,
        "--container-image",
        str(image),
        "--data-dir",
        str(data_dir),
    ]


def test_strict_l40s_finalizer_publishes_and_reverifies_controller_only_status(
    tmp_path,
    monkeypatch,
    valid_joint_velocity_smoke_payload,
):
    job_id = 12345
    repository, commit, image, data_dir = _prepare_provenance(tmp_path, monkeypatch)
    output_dir = tmp_path / "results"
    smoke = output_dir / f"joint-velocity-smoke-{job_id}.json"
    completion = output_dir / f"controller-smoke-{job_id}.completion.json"
    status = output_dir / f"srun-{job_id}.status.json"
    gpu = output_dir / f"gpu-{job_id}.csv"
    saved = output_dir / f"job-{job_id}.sbatch"
    _publish_transactional_smoke(smoke, valid_joint_velocity_smoke_payload)
    _immutable(status, _canonical({"job_id": job_id, "srun_exit_code": 0}))
    _immutable(gpu, b"GPU-test, NVIDIA L40S, 580.105.08\n")
    _immutable(
        saved,
        (
            repository
            / "scripts/polaris/l40s_pi05_droid_jointvelocity_controller_smoke.sbatch"
        ).read_bytes(),
    )
    monkeypatch.setenv("SLURM_JOB_ID", str(job_id))
    arguments = _argv(
        mode="finalize",
        job_id=job_id,
        smoke=smoke,
        completion=completion,
        status=status,
        gpu=gpu,
        saved=saved,
        repository=repository,
        commit=commit,
        image=image,
        data_dir=data_dir,
    )

    assert finalizer.main(arguments) == 0
    attestation = json.loads(completion.read_bytes())
    assert attestation["status"] == "pass"
    assert attestation["scope"] == "controller_only_no_model_or_checkpoint"
    assert attestation["promotion"] == "forbidden_without_separate_checkpoint_canary"
    assert attestation["candidate_intent"] == {
        "expected_gripper_drive_profile": finalizer.GRIPPER_DRIVE_PROFILE
    }
    assert attestation["smoke_artifact"]["case_count"] == 20
    assert attestation["source"]["standalone_git_directory"] is True
    assert attestation["source"]["detached_head"] is True
    assert attestation["source"]["head_reference"] == "HEAD"
    assert attestation["source"]["git_directory"] == str(repository / ".git")
    assert attestation["source"]["git_common_directory"] == str(repository / ".git")
    assert (completion.stat().st_mode & 0o777) == 0o444

    verify_arguments = arguments.copy()
    verify_arguments[0] = "verify"
    assert finalizer.main(verify_arguments) == 0

    wrong_profile = verify_arguments.copy()
    profile_index = wrong_profile.index("--expected-gripper-drive-profile") + 1
    wrong_profile[profile_index] = "legacy_ignored_velocity_limit"
    with pytest.raises(ValueError, match="Expected gripper drive profile mismatch"):
        finalizer.main(wrong_profile)

    missing_profile = verify_arguments.copy()
    profile_flag = missing_profile.index("--expected-gripper-drive-profile")
    del missing_profile[profile_flag : profile_flag + 2]
    with pytest.raises(SystemExit):
        finalizer.main(missing_profile)

    completion.chmod(0o644)
    tampered = json.loads(completion.read_bytes())
    tampered["scope"] = "checkpoint"
    completion.write_bytes(_canonical(tampered))
    completion.chmod(0o444)
    with pytest.raises(ValueError, match="does not match live provenance"):
        finalizer.main(verify_arguments)


def test_finalizer_rejects_nonzero_srun_without_completion(
    tmp_path,
    monkeypatch,
    valid_joint_velocity_smoke_payload,
):
    job_id = 67890
    repository, commit, image, data_dir = _prepare_provenance(tmp_path, monkeypatch)
    output_dir = tmp_path / "failed"
    smoke = output_dir / f"joint-velocity-smoke-{job_id}.json"
    completion = output_dir / f"controller-smoke-{job_id}.completion.json"
    status = output_dir / f"srun-{job_id}.status.json"
    gpu = output_dir / f"gpu-{job_id}.csv"
    saved = output_dir / f"job-{job_id}.sbatch"
    _publish_transactional_smoke(smoke, valid_joint_velocity_smoke_payload)
    _immutable(status, _canonical({"job_id": job_id, "srun_exit_code": 1}))
    _immutable(gpu, b"GPU-test, NVIDIA L40S, 580.105.08\n")
    _immutable(
        saved,
        (
            repository
            / "scripts/polaris/l40s_pi05_droid_jointvelocity_controller_smoke.sbatch"
        ).read_bytes(),
    )
    monkeypatch.setenv("SLURM_JOB_ID", str(job_id))
    with pytest.raises(ValueError, match="srun status mismatch"):
        finalizer.main(
            _argv(
                mode="finalize",
                job_id=job_id,
                smoke=smoke,
                completion=completion,
                status=status,
                gpu=gpu,
                saved=saved,
                repository=repository,
                commit=commit,
                image=image,
                data_dir=data_dir,
            )
        )
    assert not completion.exists()


def test_source_provenance_rejects_linked_external_or_branch_git_layout(
    tmp_path, monkeypatch
):
    repository, commit, _, _ = _prepare_provenance(tmp_path, monkeypatch)
    linked = tmp_path / "linked-worktree"
    subprocess.run(
        [
            "git",
            "-C",
            repository,
            "worktree",
            "add",
            "--detach",
            linked,
            commit,
        ],
        check=True,
        capture_output=True,
    )
    assert (linked / ".git").is_file()
    with pytest.raises(ValueError, match="standalone clone"):
        finalizer._source_provenance(linked, commit)

    real_git = finalizer._git
    external_common = tmp_path / "external-common.git"
    external_common.mkdir()

    def external_common_git(path, *arguments):
        if arguments == (
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ):
            return (str(external_common) + "\n").encode("ascii")
        return real_git(path, *arguments)

    monkeypatch.setattr(finalizer, "_git", external_common_git)
    with pytest.raises(ValueError, match="common directory"):
        finalizer._source_provenance(repository, commit)

    monkeypatch.setattr(finalizer, "_git", real_git)
    subprocess.run(
        ["git", "-C", repository, "checkout", "-q", "-b", "test-branch"], check=True
    )
    with pytest.raises(ValueError, match="detached HEAD"):
        finalizer._source_provenance(repository, commit)


def test_submit_and_sbatch_reject_gitfiles_before_git_provenance_queries():
    repository = Path(__file__).parents[1]
    for relative_path in (
        "scripts/polaris/submit_pi05_droid_jointvelocity_controller_smoke.sh",
        "scripts/polaris/l40s_pi05_droid_jointvelocity_controller_smoke.sbatch",
    ):
        source = (repository / relative_path).read_text(encoding="utf-8")
        standalone_index = source.index(
            '[[ -d "${POLARIS_DIR}/.git" && ! -L "${POLARIS_DIR}/.git" ]]'
        )
        git_query_index = source.index(
            'git -C "${POLARIS_DIR}" rev-parse --absolute-git-dir'
        )
        top_level_index = source.index(
            'git -C "${POLARIS_DIR}" rev-parse --show-toplevel'
        )
        assert standalone_index < git_query_index < top_level_index
        assert "rev-parse --path-format=absolute --git-common-dir" in source
        assert "rev-parse --abbrev-ref HEAD" in source
        assert "standalone clone with an in-root .git directory" in source
        assert "checked out at a detached HEAD" in source
        lexical_index = source.index(
            'require_normalized_absolute_path POLARIS_DIR "${POLARIS_DIR}"'
        )
        realpath_index = source.index(
            'POLARIS_DIR_RESOLVED="$(realpath "${POLARIS_DIR}")"'
        )
        assert lexical_index < realpath_index < git_query_index
        assert 'POLARIS_DIR="$(realpath "${POLARIS_DIR}")"' not in source

    submit_source = (
        repository
        / "scripts/polaris/submit_pi05_droid_jointvelocity_controller_smoke.sh"
    ).read_text(encoding="utf-8")
    submit_index = submit_source.index('job_id="$(sbatch --parsable')
    assert submit_source.index("require_normalized_absolute_path()") < submit_index
    assert (
        submit_source.index('require_normalized_absolute_path RUN_DIR "${RUN_DIR}"')
        < submit_index
    )
    assert 'realpath -sm -- "${value}"' in submit_source
    assert finalizer.GRIPPER_DRIVE_PROFILE in submit_source
    assert finalizer.GRIPPER_DRIVE_PROFILE in (
        repository
        / "scripts/polaris/l40s_pi05_droid_jointvelocity_controller_smoke.sbatch"
    ).read_text(encoding="utf-8")
    sbatch_source = (
        repository
        / "scripts/polaris/l40s_pi05_droid_jointvelocity_controller_smoke.sbatch"
    ).read_text(encoding="utf-8")
    controller_source = (
        repository / "scripts/smoke_joint_velocity_controller.py"
    ).read_text(encoding="utf-8")
    assert ': "${EXPECTED_GRIPPER_DRIVE_PROFILE:?' in sbatch_source
    assert (
        '[[ "${EXPECTED_GRIPPER_DRIVE_PROFILE}" == "${GRIPPER_DRIVE_PROFILE}" ]]'
        in (sbatch_source)
    )
    assert 'parser.add_argument("--expected-gripper-drive-profile", required=True)' in (
        controller_source
    )
    assert "expected_gripper_drive_profile" in submit_source
    for field in (
        "polaris_dir_resolved",
        "run_dir_resolved",
        "sbatch_script_resolved",
        "container_image_resolved",
        "polaris_data_dir_resolved",
    ):
        assert f'"{field}"' in submit_source


def test_submitter_rejects_raw_nonnormalized_polaris_directory_spellings():
    repository = Path(__file__).parents[1]
    source = (
        repository
        / "scripts/polaris/submit_pi05_droid_jointvelocity_controller_smoke.sh"
    ).read_text(encoding="utf-8")
    helper_source = source[source.index("die() {") : source.index("command -v sbatch")]
    command = helper_source + '\nrequire_normalized_absolute_path POLARIS_DIR "$1"\n'
    for invalid in (
        "relative/repository",
        "/tmp/./repository",
        "/tmp//repository",
        "/tmp/other/../repository",
        "//tmp/repository",
        "/tmp/repository/",
    ):
        completed = subprocess.run(
            ["bash", "-c", command, "submitter-path-test", invalid],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 2, (invalid, completed)
        assert "POLARIS_DIR must" in completed.stderr

    accepted = subprocess.run(
        ["bash", "-c", command, "submitter-path-test", "/tmp/repository"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert accepted.returncode == 0, accepted


def test_finalizer_accepts_ancestor_alias_but_rejects_different_or_symlink_file(
    tmp_path,
    monkeypatch,
    valid_joint_velocity_smoke_payload,
):
    job_id = 24680
    repository, commit, image, data_dir = _prepare_provenance(tmp_path, monkeypatch)
    producer_root = tmp_path / "producer-root"
    producer_root.symlink_to(tmp_path, target_is_directory=True)
    producer_repository = producer_root / repository.name
    producer_image = producer_root / image.name
    producer_data_dir = producer_root / data_dir.name
    real_output = tmp_path / "real-results"
    real_output.mkdir()
    producer_output = tmp_path / "producer-results"
    producer_output.symlink_to(real_output, target_is_directory=True)

    smoke_name = f"joint-velocity-smoke-{job_id}.json"
    producer_smoke = producer_output / smoke_name
    host_smoke = real_output / smoke_name
    completion = real_output / f"controller-smoke-{job_id}.completion.json"
    status = real_output / f"srun-{job_id}.status.json"
    gpu = real_output / f"gpu-{job_id}.csv"
    saved = real_output / f"job-{job_id}.sbatch"
    _publish_transactional_smoke(producer_smoke, valid_joint_velocity_smoke_payload)
    _immutable(status, _canonical({"job_id": job_id, "srun_exit_code": 0}))
    _immutable(gpu, b"GPU-test, NVIDIA L40S, 580.105.08\n")
    _immutable(
        saved,
        (
            repository
            / "scripts/polaris/l40s_pi05_droid_jointvelocity_controller_smoke.sbatch"
        ).read_bytes(),
    )
    monkeypatch.setenv("SLURM_JOB_ID", str(job_id))
    assert (
        finalizer.main(
            _argv(
                mode="finalize",
                job_id=job_id,
                smoke=host_smoke,
                completion=completion,
                status=status,
                gpu=gpu,
                saved=saved,
                repository=producer_repository,
                commit=commit,
                image=producer_image,
                data_dir=producer_data_dir,
            )
        )
        == 0
    )

    attestation = json.loads(completion.read_bytes())
    child = attestation["smoke_artifact"]["child_close_capture"]
    ready = attestation["smoke_artifact"]["child_ready_marker"]
    producer_child = producer_smoke.with_name(smoke_name + ".child-close.json")
    host_child = host_smoke.with_name(smoke_name + ".child-close.json")
    assert child["path"] == str(producer_child)
    assert child["host_declared_path"] == str(host_child)
    assert child["resolved_path"] == str(host_child.resolve())
    assert child["path_alias_equivalent"] is True
    assert child["producer_host_spelling_match"] is False
    assert ready["path"] == str(producer_child) + ".ready.json"
    assert ready["host_declared_path"] == str(host_child) + ".ready.json"
    assert ready["path_alias_equivalent"] is True
    assert ready["producer_host_spelling_match"] is False
    assert attestation["smoke_artifact"]["path"] == str(host_smoke)
    assert attestation["smoke_artifact"]["resolved_path"] == str(host_smoke.resolve())
    assert attestation["smoke_artifact"]["path_alias_equivalent"] is False
    assert attestation["source"]["repository"] == str(producer_repository)
    assert attestation["source"]["resolved_repository"] == str(repository)
    hub = attestation["runtime"]["polaris_hub"]
    assert hub["root"] == str(producer_data_dir)
    assert hub["resolved_root"] == str(data_dir)
    for relative_path, asset in hub["assets"].items():
        assert asset["path"] == str(producer_data_dir / relative_path)
        assert asset["resolved_path"] == str(data_dir / relative_path)
    container = attestation["runtime"]["container_image"]
    assert container["path"] == str(producer_image)
    assert container["host_declared_path"] == str(producer_image)
    assert container["resolved_path"] == str(image)
    assert container["path_alias_equivalent"] is True
    assert container["producer_host_spelling_match"] is True

    _, _, child_stat = finalizer._read_canonical_json(host_child, "host child")
    other_dir = tmp_path / "different-results"
    other_dir.mkdir()
    different_child = other_dir / host_child.name
    _immutable(different_child, host_child.read_bytes())
    with pytest.raises(ValueError, match="same exact file"):
        finalizer._bind_artifact_path(
            str(different_child), host_child, child_stat, "different child"
        )

    symlink_dir = tmp_path / "symlink-results"
    symlink_dir.mkdir()
    symlink_child = symlink_dir / host_child.name
    symlink_child.symlink_to(host_child)
    with pytest.raises(ValueError, match="must not be a symlink"):
        finalizer._bind_artifact_path(
            str(symlink_child), host_child, child_stat, "symlink child"
        )

    with pytest.raises(ValueError, match="normalized absolute path"):
        finalizer._bind_artifact_path(
            host_child.name, host_child, child_stat, "relative child"
        )
    with pytest.raises(ValueError, match="normalized absolute path"):
        finalizer._bind_artifact_path(
            str(host_child.parent / ".." / host_child.parent.name / host_child.name),
            host_child,
            child_stat,
            "nonnormalized child",
        )
    invalid_spellings = (
        "relative-child.json",
        f"{host_child.parent}/./{host_child.name}",
        f"{host_child.parent}//{host_child.name}",
        f"//{str(host_child).lstrip('/')}",
    )
    for invalid_spelling in invalid_spellings:
        with pytest.raises(ValueError, match="normalized absolute path"):
            finalizer._bind_artifact_path(
                invalid_spelling, host_child, child_stat, "raw-spelling child"
            )
        with pytest.raises(ValueError, match="normalized absolute path"):
            controller._child_ready_payload(invalid_spelling, b"payload")
