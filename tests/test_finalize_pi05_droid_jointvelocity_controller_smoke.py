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
    child = copy.deepcopy(payload)
    child.pop("completion")
    child["lifecycle"] = {
        "env_close": "complete",
        "simulation_app_close": "pending_child_exit",
        "capture_stage": "kit_child_after_env_close_before_simulation_app_close",
    }
    validated = validate_joint_velocity_smoke(child, require_parent_completion=False)
    controller._write_child_capture(raw_path, validated)
    controller.finalize_child_capture(raw_path, path, child_exit_code=0)


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
