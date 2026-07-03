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
