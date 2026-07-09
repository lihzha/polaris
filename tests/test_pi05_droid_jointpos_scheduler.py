import copy
from pathlib import Path

import pytest

from polaris.pi05_droid_jointpos_immutable import publish_immutable_json
from polaris.pi05_droid_jointpos_scheduler import (
    SCHEDULER_JOB_PROFILE,
    SCHEDULER_TERMINAL_PROFILE,
    build_terminal_attestation,
    parse_sacct_terminal_record,
    parse_scontrol_job_record,
    validate_persisted_scheduler_job,
)


JOB_ID = 4242
TRANSACTION = "pi05-0123456789abcdef0123456789abcdef01234567"


def _scontrol_raw(
    phase: str,
    *,
    requeue: str = "0",
    restarts: str = "0",
    transaction: str = TRANSACTION,
) -> str:
    state = "PENDING" if phase == "held" else "RUNNING"
    reason = "JobHeldUser" if phase == "held" else "None"
    return (
        f"JobId={JOB_ID} JobName=pi05-canary JobState={state} Reason={reason} "
        f"Requeue={requeue} Restarts={restarts} Comment={transaction}\n"
    )


def _publish_scheduler(path: Path, phase: str) -> dict:
    raw = _scontrol_raw(phase)
    parsed = parse_scontrol_job_record(
        raw,
        phase=phase,
        expected_job_id=JOB_ID,
        expected_transaction_id=TRANSACTION,
    )
    return publish_immutable_json(
        path,
        {
            "schema_version": 1,
            "profile": SCHEDULER_JOB_PROFILE,
            "status": f"{phase}_requeue_disabled_restart_count_zero",
            "command": ["scontrol", "show", "job", str(JOB_ID), "--oneliner"],
            "job": parsed,
        },
    )


def _sacct_raw(
    *, state: str = "COMPLETED", exit_code: str = "0:0", restarts: str = "0"
) -> str:
    return (
        f"{JOB_ID}|{state}|{exit_code}|2026-07-09T01:00:00|"
        "2026-07-09T01:01:00|2026-07-09T01:02:00|60|l401|"
        f"{restarts}\n"
    )


def test_scontrol_held_and_running_records_prove_no_requeue() -> None:
    held = parse_scontrol_job_record(
        _scontrol_raw("held"),
        phase="held",
        expected_job_id=JOB_ID,
        expected_transaction_id=TRANSACTION,
    )
    running = parse_scontrol_job_record(
        _scontrol_raw("running"),
        phase="running",
        expected_job_id=JOB_ID,
        expected_transaction_id=TRANSACTION,
    )

    assert held["state"] == "PENDING"
    assert running["state"] == "RUNNING"
    assert held["requeue"] == running["requeue"] == 0
    assert held["restarts"] == running["restarts"] == 0


@pytest.mark.parametrize(
    ("raw", "phase", "message"),
    [
        (_scontrol_raw("held", requeue="1"), "held", "permits requeue"),
        (_scontrol_raw("running", restarts="1"), "running", "already restarted"),
        (_scontrol_raw("running"), "held", "held state mismatch"),
        (
            _scontrol_raw("held", transaction="pi05-" + "0" * 40),
            "held",
            "comment mismatch",
        ),
        (
            _scontrol_raw("held").rstrip("\n") + " Requeue=0\n",
            "held",
            "duplicate scontrol field",
        ),
        (_scontrol_raw("held").rstrip("\n"), "held", "unterminated"),
        (_scontrol_raw("held") + _scontrol_raw("held"), "held", "exactly one"),
    ],
)
def test_scontrol_record_rejects_adversarial_lifecycle_evidence(
    raw: str, phase: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_scontrol_job_record(
            raw,
            phase=phase,
            expected_job_id=JOB_ID,
            expected_transaction_id=TRANSACTION,
        )


def test_persisted_scheduler_record_rejects_semantic_tampering(tmp_path: Path) -> None:
    artifact = _publish_scheduler(tmp_path / "held.json", "held")
    value = copy.deepcopy(artifact["value"])
    value["job"]["requeue"] = 1
    tampered = tmp_path / "tampered.json"
    publish_immutable_json(tampered, value)

    with pytest.raises(ValueError, match="not canonical"):
        validate_persisted_scheduler_job(
            tampered,
            phase="held",
            expected_job_id=JOB_ID,
            expected_transaction_id=TRANSACTION,
        )


def test_sacct_terminal_record_proves_clean_completion_without_restart() -> None:
    result = parse_sacct_terminal_record(_sacct_raw(), expected_job_id=JOB_ID)

    assert result["state"] == "COMPLETED"
    assert result["exit_code"] == "0:0"
    assert result["restarts"] == 0


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (_sacct_raw(restarts="1"), "job restarted"),
        (_sacct_raw(state="FAILED", exit_code="1:0"), "did not complete cleanly"),
        (_sacct_raw(exit_code="1:0"), "did not complete cleanly"),
        (_sacct_raw().replace("4242|", "4243|", 1), "job ID mismatch"),
        (_sacct_raw().rstrip("\n"), "unterminated"),
        (_sacct_raw() + _sacct_raw(), "exactly one"),
        (
            _sacct_raw().replace("2026-07-09T01:02:00", "2026-07-09T00:59:00"),
            "timestamps are invalid",
        ),
    ],
)
def test_sacct_terminal_record_rejects_invalid_scientific_completion(
    raw: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_sacct_terminal_record(raw, expected_job_id=JOB_ID)


def test_terminal_attestation_binds_held_running_and_evaluator_evidence(
    tmp_path: Path,
) -> None:
    held = _publish_scheduler(tmp_path / "held.json", "held")
    running = _publish_scheduler(tmp_path / "running.json", "running")
    running_identity = {
        key: running[key] for key in ("path", "size", "sha256", "mode", "nlink")
    }
    evidence = publish_immutable_json(
        tmp_path / "pi05_droid_jointpos_evidence_manifest.json",
        {"status": "pass", "artifacts": {"scheduler_running": running_identity}},
    )
    success = tmp_path / "SUCCESS"
    success.write_text(
        "status=success\n"
        "completed_at=2026-07-09T01:02:00-07:00\n"
        f"evidence_manifest_sha256={evidence['sha256']}\n"
    )
    success.chmod(0o444)
    command = ["sacct", "--jobs=4242"]

    value = build_terminal_attestation(
        held_record_path=Path(held["path"]),
        running_record_path=Path(running["path"]),
        evidence_manifest_path=Path(evidence["path"]),
        task_success_path=success,
        expected_job_id=JOB_ID,
        expected_transaction_id=TRANSACTION,
        sacct_raw=_sacct_raw(),
        sacct_command=command,
    )

    assert value["profile"] == SCHEDULER_TERMINAL_PROFILE
    assert value["status"] == "completed_without_requeue_or_restart"
    assert value["held_scheduler_record"]["sha256"] == held["sha256"]
    assert value["running_scheduler_record"]["sha256"] == running["sha256"]
    assert value["evidence_manifest"]["sha256"] == evidence["sha256"]
    assert value["task_success"]["artifact"]["path"] == str(success.resolve())
    assert value["task_success"]["value"]["status"] == "success"
    assert value["terminal"]["restarts"] == 0


def test_terminal_attestation_rejects_unbound_running_record(tmp_path: Path) -> None:
    held = _publish_scheduler(tmp_path / "held.json", "held")
    running = _publish_scheduler(tmp_path / "running.json", "running")
    evidence = publish_immutable_json(
        tmp_path / "pi05_droid_jointpos_evidence_manifest.json",
        {"status": "pass", "artifacts": {"scheduler_running": {}}},
    )
    success = tmp_path / "SUCCESS"
    success.write_text(
        f"status=success\nevidence_manifest_sha256={evidence['sha256']}\n"
    )
    success.chmod(0o444)

    with pytest.raises(ValueError, match="does not bind"):
        build_terminal_attestation(
            held_record_path=Path(held["path"]),
            running_record_path=Path(running["path"]),
            evidence_manifest_path=Path(evidence["path"]),
            task_success_path=success,
            expected_job_id=JOB_ID,
            expected_transaction_id=TRANSACTION,
            sacct_raw=_sacct_raw(),
            sacct_command=["sacct", "--jobs=4242"],
        )


def test_public_batch_and_submitter_both_disable_requeue() -> None:
    root = Path(__file__).resolve().parents[1]
    batch = (root / "scripts/polaris/l40s_pi05_eval_job.sbatch").read_text()
    submitter = (
        root / "scripts/polaris/submit_pi05_droid_jointpos_polaris.sh"
    ).read_text()

    assert batch.count("#SBATCH --no-requeue") == 1
    assert (
        'sbatch_argv=("${SBATCH_COMMAND}" --parsable --hold --no-requeue' in submitter
    )
    assert " --requeue" not in submitter.replace("--no-requeue", "")
