import os
from pathlib import Path
import stat
import subprocess
import textwrap


ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "scripts/polaris/submit_pi05_droid_jointpos_polaris.sh"


FAKE_SLURM = r"""
#!/usr/bin/env bash
set -euo pipefail

command_name="$(basename "$0")"
case "${command_name}" in
  sbatch)
    : > "${FAKE_STATE}/sbatch.args"
    comment=""
    held=0
    for argument in "$@"; do
      printf '%s\n' "${argument}" >> "${FAKE_STATE}/sbatch.args"
      case "${argument}" in
        --comment=*) comment="${argument#--comment=}" ;;
        --hold) held=1 ;;
      esac
    done
    [[ "${held}" == 1 && "${comment}" == pi05-* ]]
    printf '4242|%s\n' "${comment}" > "${FAKE_STATE}/active"
    if [[ "${FAKE_SBATCH_SIGNAL:-0}" == 1 ]]; then
      kill -TERM "${PPID}"
      sleep 0.05
      exit 0
    fi
    if [[ "${FAKE_SBATCH_FAIL:-0}" == 1 ]]; then
      exit 19
    fi
    printf '4242\n'
    ;;
  squeue)
    if [[ -f "${FAKE_STATE}/active" ]]; then
      cat "${FAKE_STATE}/active"
    fi
    ;;
  scontrol)
    case "${1:-}" in
      write)
        [[ "${2:-}" == batch_script && "${3:-}" == 4242 ]]
        printf 'write %s\n' "${3}" >> "${FAKE_STATE}/scontrol.log"
        if [[ "${FAKE_CAPTURE_FAIL:-0}" == 1 ]]; then
          exit 17
        fi
        printf '#!/usr/bin/env bash\necho governed\n' > "${4}"
        ;;
      release)
        [[ "${2:-}" == 4242 ]]
        manifest_row="$(
          awk -F '\t' -v id="${2}" \
            '$1 == id { print; found = 1 } END { exit !found }' \
            "${SUBMISSION_MANIFEST}"
        )"
        IFS=$'\t' read -r row_id row_mode row_task row_rollouts row_seed \
          row_namespace row_time batch_sha argv_sha provenance <<< "${manifest_row}"
        [[ "${row_id}" == 4242 && "${row_mode}" == canary ]]
        [[ "${row_task}" == DROID-FoodBussing && "${row_rollouts}" == 1 ]]
        [[ "${row_seed}" == 0 && -n "${row_namespace}" && -n "${row_time}" ]]
        [[ "$(stat -c '%a' "${provenance}/batch_script.sbatch")" == 444 ]]
        [[ "$(stat -c '%a' "${provenance}/submission_argv.sh")" == 444 ]]
        [[ "$(sha256sum "${provenance}/batch_script.sbatch" | awk '{print $1}')" == "${batch_sha}" ]]
        [[ "$(sha256sum "${provenance}/submission_argv.sh" | awk '{print $1}')" == "${argv_sha}" ]]
        printf 'release %s\n' "${2}" >> "${FAKE_STATE}/scontrol.log"
        if [[ "${FAKE_RELEASE_FAIL:-0}" == 1 ]]; then
          exit 23
        fi
        rm -f "${FAKE_STATE}/active"
        ;;
      *) exit 97 ;;
    esac
    ;;
  scancel)
    printf '%s\n' "${1}" >> "${FAKE_STATE}/scancel.log"
    if [[ "${FAKE_SCANCEL_FAIL:-0}" == 1 ]]; then
      exit 29
    fi
    rm -f "${FAKE_STATE}/active"
    ;;
  *) exit 98 ;;
esac
"""


def _write_executable(path: Path, value: str) -> None:
    path.write_text(textwrap.dedent(value).lstrip())
    path.chmod(0o755)


def _make_clean_repository(path: Path) -> Path:
    path.mkdir()
    batch_script = path / "job.sbatch"
    batch_script.write_text("#!/usr/bin/env bash\necho worker\n")
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "submit-test@example.invalid"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Submit Test"], cwd=path, check=True
    )
    subprocess.run(["git", "add", "job.sbatch"], cwd=path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "test fixture"], cwd=path, check=True
    )
    return batch_script


def _environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    repository = tmp_path / "repo"
    batch_script = _make_clean_repository(repository)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    for name in ("sbatch", "squeue", "scontrol", "scancel"):
        _write_executable(fake_bin / name, FAKE_SLURM)
    fake_state = tmp_path / "fake-state"
    fake_state.mkdir()
    manifest = tmp_path / "results" / "canary_jobs.tsv"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "POLARIS_DIR": str(repository),
            "SBATCH_SCRIPT": str(batch_script),
            "SBATCH_LOG_ROOT": str(tmp_path / "logs"),
            "SUBMISSION_MANIFEST": str(manifest),
            "RUN_NAMESPACE": "pi05-submit-host-test",
            "FAKE_STATE": str(fake_state),
            "USER": "submit-test",
        }
    )
    return env, manifest, fake_state


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SUBMITTER), "canary"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _transaction_states(manifest: Path) -> list[str]:
    transaction_root = Path(f"{manifest}.transactions")
    return sorted(
        (transaction / "state").read_text().strip()
        for transaction in transaction_root.iterdir()
    )


def test_held_job_is_released_only_after_durable_provenance_and_manifest(
    tmp_path: Path,
) -> None:
    env, manifest, fake_state = _environment(tmp_path)

    result = _run(env)

    assert result.returncode == 0, result.stderr
    assert "submitted_job_ids=4242" in result.stdout
    arguments = (fake_state / "sbatch.args").read_text().splitlines()
    assert "--hold" in arguments
    comments = [value for value in arguments if value.startswith("--comment=pi05-")]
    assert len(comments) == 1
    assert len(comments[0].removeprefix("--comment=")) == 45
    assert (fake_state / "scontrol.log").read_text().splitlines() == [
        "write 4242",
        "release 4242",
    ]
    assert not (fake_state / "scancel.log").exists()
    assert _transaction_states(manifest) == ["released"]
    rows = manifest.read_text().splitlines()
    assert len(rows) == 2
    assert rows[1].startswith("4242\tcanary\tDROID-FoodBussing\t")
    provenance = manifest.parent / "submission_provenance/job_4242"
    for name in ("batch_script.sbatch", "submission_argv.sh"):
        artifact = provenance / name
        assert artifact.is_file()
        assert stat.S_IMODE(artifact.stat().st_mode) == 0o444


def test_provenance_capture_failure_cancels_held_job(tmp_path: Path) -> None:
    env, manifest, fake_state = _environment(tmp_path)
    env["FAKE_CAPTURE_FAIL"] = "1"

    result = _run(env)

    assert result.returncode == 5
    assert "Failed to preserve submission provenance" in result.stderr
    assert (fake_state / "scancel.log").read_text().splitlines() == ["4242"]
    assert "release" not in (fake_state / "scontrol.log").read_text()
    assert _transaction_states(manifest) == ["canceled"]
    assert len(manifest.read_text().splitlines()) == 1


def test_release_failure_preserves_manifest_and_cancels_job(tmp_path: Path) -> None:
    env, manifest, fake_state = _environment(tmp_path)
    env["FAKE_RELEASE_FAIL"] = "1"

    result = _run(env)

    assert result.returncode == 5
    assert "Failed to release held job 4242" in result.stderr
    assert (fake_state / "scancel.log").read_text().splitlines() == ["4242"]
    assert (fake_state / "scontrol.log").read_text().splitlines() == [
        "write 4242",
        "release 4242",
    ]
    assert _transaction_states(manifest) == ["canceled"]
    assert len(manifest.read_text().splitlines()) == 2


def test_comment_recovers_job_when_sbatch_id_was_never_captured(tmp_path: Path) -> None:
    env, manifest, fake_state = _environment(tmp_path)
    env["FAKE_SBATCH_FAIL"] = "1"

    result = _run(env)

    assert result.returncode == 3
    assert "did not return exactly one numeric held job ID" in result.stderr
    assert (fake_state / "scancel.log").read_text().splitlines() == ["4242"]
    assert _transaction_states(manifest) == ["canceled"]
    assert len(manifest.read_text().splitlines()) == 1


def test_term_during_sbatch_capture_recovers_and_cancels_by_comment(
    tmp_path: Path,
) -> None:
    env, manifest, fake_state = _environment(tmp_path)
    env["FAKE_SBATCH_SIGNAL"] = "1"

    result = _run(env)

    assert result.returncode == 143
    assert (fake_state / "scancel.log").read_text().splitlines() == ["4242"]
    assert _transaction_states(manifest) == ["canceled"]
    assert len(manifest.read_text().splitlines()) == 1


def test_next_invocation_recovers_persisted_cleanup_pending_transaction(
    tmp_path: Path,
) -> None:
    env, manifest, fake_state = _environment(tmp_path)
    env.update({"FAKE_SBATCH_FAIL": "1", "FAKE_SCANCEL_FAIL": "1"})

    first = _run(env)

    assert first.returncode == 5
    assert _transaction_states(manifest) == ["cleanup_pending"]
    assert (fake_state / "active").is_file()

    env.pop("FAKE_SBATCH_FAIL")
    env.pop("FAKE_SCANCEL_FAIL")
    second = _run(env)

    assert second.returncode == 0, second.stderr
    assert "submitted_job_ids=4242" in second.stdout
    assert _transaction_states(manifest) == ["canceled", "released"]
    assert (fake_state / "scancel.log").read_text().splitlines() == [
        "4242",
        "4242",
    ]


def test_symlink_transaction_root_is_rejected_before_submission(tmp_path: Path) -> None:
    env, manifest, fake_state = _environment(tmp_path)
    transaction_target = tmp_path / "transaction-target"
    transaction_target.mkdir()
    manifest.parent.mkdir()
    Path(f"{manifest}.transactions").symlink_to(transaction_target)

    result = _run(env)

    assert result.returncode == 2
    assert "Transaction root must not be a symlink" in result.stderr
    assert not (fake_state / "sbatch.args").exists()
