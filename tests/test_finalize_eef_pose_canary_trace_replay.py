from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import subprocess
import time

import pytest

from scripts import finalize_eef_pose_canary_trace_replay as finalizer
from scripts import smoke_eef_pose_canary_trace_replay as replay
from scripts import write_eef_pose_canary_gate0_srun_status as status_writer


def _source_sha(path: Path) -> str:
    return replay._sha256(path.read_bytes())


def _build_raw_pair(
    root: Path,
    *,
    variant: str,
    launch_id: str,
    job_id: int,
    commit: str,
) -> tuple[Path, Path, dict[str, object]]:
    namespace = root / variant / f"job_{job_id}" / f"launch_{launch_id}"
    raw_path = namespace / f"gate0-{variant}.raw.json"
    fixture_identity, fixture_payload, _ = replay.load_replay_fixture(variant)
    lifecycle = {
        "profile": "slurm_single_task_srun_lifecycle_v1",
        "launch_id": launch_id,
        "job_id": job_id,
        "step_id": 0,
        "nodelist": "l401",
        "procid": 0,
        "localid": 0,
        "ntasks": 1,
    }
    raw = {
        "variant": variant,
        "lifecycle": lifecycle,
        "repository": {"commit": commit},
        "production_eval": replay.validate_production_reset_source(),
        "fixture": {
            **fixture_identity,
            "source_trace_sha256": fixture_payload["source"]["trace_sha256"],
            "action_float32_sha256": fixture_payload["action_encoding"][
                "uncompressed_sha256"
            ],
            "action_count": 120,
        },
        "outcome": replay.EXPECTED_FIXTURES[variant]["failure"],
        "arm_failure_runtime_evidence": {
            "controller_substep_trace": {"capacity": 64, "entries": [{}] * 64}
        },
        "all_six_gripper_tail": {
            "capacity": 64,
            "entries": [{}] * 64,
            "total_apply_entries": 942 if variant == "official_lap3b" else 898,
        },
        "assets": {},
    }
    raw_identity = replay._atomic_write_immutable(raw_path, raw)
    ready_path = raw_path.with_name(raw_path.name + ".ready.json")
    replay._atomic_write_immutable(
        ready_path,
        {
            "schema_version": 1,
            "profile": replay.PROFILE,
            "stage": "simulation_app_close_pending",
            "raw_result": raw_identity,
        },
    )
    return raw_path, ready_path, raw


def test_status_writer_and_finalizer_bind_exact_srun_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    variant = "official_lap3b"
    launch_id = "e" * 64
    job_id = 789
    raw_path, _, raw = _build_raw_pair(
        tmp_path,
        variant=variant,
        launch_id=launch_id,
        job_id=job_id,
        commit=commit,
    )
    monkeypatch.setenv("SLURM_JOB_ID", str(job_id))
    monkeypatch.setattr(replay, "validate_capture_payload", lambda value: value)
    now = time.time_ns()
    status_path = raw_path.with_name(f"gate0-{variant}.srun-status.json")
    status_args = argparse.Namespace(
        variant=variant,
        launch_id=launch_id,
        job_id=job_id,
        srun_rc=0,
        srun_started_at_ns=now - 2,
        srun_returned_at_ns=now - 1,
        raw_result=raw_path,
        status=status_path,
    )
    status = status_writer.build_status(status_args)
    replay._atomic_write_immutable(status_path, status)
    assert status["raw_lifecycle"] == raw["lifecycle"]
    assert status["srun_rc"] == 0

    monkeypatch.setattr(
        finalizer, "_validate_live_assets", lambda value: {"pinned": True}
    )

    def clean_git(_repo: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD"):
            return commit
        if arguments == ("status", "--porcelain", "--untracked-files=no"):
            return ""
        raise AssertionError(arguments)

    monkeypatch.setattr(finalizer, "_git", clean_git)
    container = tmp_path / "image.sqsh"
    container.write_bytes(b"container")
    job_script = tmp_path / "job.sh"
    job_script.write_bytes(b"#!/bin/bash\n")
    job_script.chmod(0o444)
    attestation = raw_path.with_name(f"gate0-{variant}.attestation.json")
    scripts = repo / "scripts"
    args = argparse.Namespace(
        variant=variant,
        launch_id=launch_id,
        job_id=job_id,
        raw_result=raw_path,
        srun_status=status_path,
        attestation=attestation,
        polaris_repo=repo,
        expected_polaris_commit=commit,
        expected_runner_sha256=_source_sha(
            scripts / "smoke_eef_pose_canary_trace_replay.py"
        ),
        expected_fixture_sha256=replay.EXPECTED_FIXTURES[variant]["sha256"],
        expected_generator_sha256=_source_sha(
            scripts / "generate_eef_pose_canary_trace_fixtures.py"
        ),
        expected_status_writer_sha256=_source_sha(
            scripts / "write_eef_pose_canary_gate0_srun_status.py"
        ),
        expected_finalizer_sha256=_source_sha(
            scripts / "finalize_eef_pose_canary_trace_replay.py"
        ),
        container_image=container,
        expected_container_sha256=_source_sha(container),
        runtime_job_script=job_script,
        saved_job_script=job_script,
        expected_saved_job_script_sha256=_source_sha(job_script),
    )
    expected = finalizer.build_attestation(args)
    finalizer._publish(attestation, expected)
    actual = replay.strict_json_loads(attestation.read_bytes(), field="attestation")
    assert actual == expected
    assert stat_mode(attestation) == "0444"
    assert expected["lifecycle"]["step_id"] == 0
    assert expected["validation_summary"]["arm_failure_ring_entries"] == 64


def stat_mode(path: Path) -> str:
    return f"{path.stat().st_mode & 0o7777:04o}"


def test_status_writer_rejects_nonzero_srun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_id = "f" * 64
    job_id = 999
    monkeypatch.setenv("SLURM_JOB_ID", str(job_id))
    args = argparse.Namespace(
        variant="reasoning_43075",
        launch_id=launch_id,
        job_id=job_id,
        srun_rc=1,
        srun_started_at_ns=1,
        srun_returned_at_ns=2,
        raw_result=tmp_path / "missing.raw.json",
        status=tmp_path / "missing.status.json",
    )
    with pytest.raises(status_writer.SrunStatusError, match="zero srun"):
        status_writer.build_status(args)


def test_finalizer_identity_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"x")
    symlink = tmp_path / "symlink"
    symlink.symlink_to(source)
    with pytest.raises(finalizer.Gate0FinalizationError, match="missing/linked"):
        finalizer._identity(symlink, field="symlink")
    hardlink = tmp_path / "hardlink"
    os.link(source, hardlink)
    with pytest.raises(finalizer.Gate0FinalizationError, match="one hard link"):
        finalizer._identity(source, field="source")


def test_identities_preserve_parent_alias_and_accept_same_inode(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    real_file = real_parent / "evidence.json"
    real_file.write_bytes(b"{}\n")
    alias_file = alias_parent / real_file.name

    status_identity = status_writer._identity(alias_file, field="status evidence")
    final_alias_identity = finalizer._identity(alias_file, field="alias evidence")
    final_real_identity = finalizer._identity(real_file, field="real evidence")

    assert status_identity["path"] == str(alias_file)
    assert final_alias_identity["path"] == str(alias_file)
    assert final_real_identity["path"] == str(real_file)
    finalizer._same_core_identity(
        final_alias_identity,
        final_real_identity,
        field="mount-alias evidence",
    )


def test_status_writer_accepts_publisher_visible_parent_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    variant = "official_lap3b"
    launch_id = "a" * 64
    job_id = 321
    real_root = tmp_path / "real-root"
    raw_path, ready_path, _ = _build_raw_pair(
        real_root,
        variant=variant,
        launch_id=launch_id,
        job_id=job_id,
        commit="b" * 40,
    )
    alias_root = tmp_path / "publisher-root"
    alias_root.symlink_to(real_root, target_is_directory=True)
    alias_raw_path = alias_root / raw_path.relative_to(real_root)
    alias_status_path = alias_raw_path.with_name(f"gate0-{variant}.srun-status.json")

    marker = replay.strict_json_loads(ready_path.read_bytes(), field="ready marker")
    marker["raw_result"]["path"] = str(alias_raw_path)
    ready_path.unlink()
    replay._atomic_write_immutable(ready_path, marker)

    monkeypatch.setenv("SLURM_JOB_ID", str(job_id))
    monkeypatch.setattr(replay, "validate_capture_payload", lambda value: value)
    now = time.time_ns()
    status = status_writer.build_status(
        argparse.Namespace(
            variant=variant,
            launch_id=launch_id,
            job_id=job_id,
            srun_rc=0,
            srun_started_at_ns=now - 2,
            srun_returned_at_ns=now - 1,
            raw_result=alias_raw_path,
            status=alias_status_path,
        )
    )
    assert status["raw_result"]["path"] == str(alias_raw_path)


def test_production_eval_evidence_allows_only_same_inode_path_aliases(
    tmp_path: Path,
) -> None:
    current = replay.validate_production_reset_source()
    repo = Path(__file__).resolve().parents[1]
    alias_repo = tmp_path / "publisher-repo"
    alias_repo.symlink_to(repo, target_is_directory=True)
    recorded = copy.deepcopy(current)
    for source in (
        recorded,
        recorded["policy_config_source"],
        recorded["lap_client_source"],
    ):
        source_path = Path(source["path"])
        source["path"] = str(alias_repo / source_path.relative_to(repo))

    finalizer._require_same_production_eval_evidence(recorded, current)

    tampered = copy.deepcopy(recorded)
    tampered["effective_step_expensive"] = False
    with pytest.raises(
        finalizer.Gate0FinalizationError,
        match="production reset/render source evidence changed",
    ):
        finalizer._require_same_production_eval_evidence(tampered, current)
