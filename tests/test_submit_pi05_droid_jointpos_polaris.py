import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import textwrap

from polaris.pi05_droid_jointpos_consumer_binding import source_tree_sha256


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
        cp "${APPROVED_SBATCH_SCRIPT_FOR_TEST}" "${4}"
        ;;
      release)
        [[ "${2:-}" == 4242 ]]
        manifest_row="$(
          awk -F '\t' -v id="${2}" \
            '$1 == id { print; found = 1 } END { exit !found }' \
            "${SUBMISSION_MANIFEST}"
        )"
        IFS=$'\t' read -r row_id row_mode row_task row_rollouts row_seed \
          row_namespace row_source_sha row_approval_sha row_implementation \
          row_openpi_commit row_time batch_sha argv_sha provenance <<< "${manifest_row}"
        [[ "${row_id}" == 4242 && "${row_mode}" == canary ]]
        [[ "${row_task}" == DROID-FoodBussing && "${row_rollouts}" == 1 ]]
        [[ "${row_seed}" == 0 && -n "${row_namespace}" && -n "${row_time}" ]]
        [[ "${row_source_sha}" =~ ^[0-9a-f]{64}$ ]]
        [[ "${row_approval_sha}" =~ ^[0-9a-f]{64}$ ]]
        [[ "${row_implementation}" =~ ^[0-9a-f]{40}$ ]]
        [[ "${row_openpi_commit}" == bd70b8f4011e85b3f3b0f039f12113f78718e7bf ]]
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

FAKE_GIT = r"""
#!/usr/bin/env bash
set -euo pipefail
[[ "${1:-}" == -C && "${2:-}" == "${POLARIS_OPENPI_RUNTIME_DIR}" ]]
case "${3:-} ${4:-}" in
  "rev-parse HEAD")
    printf '%s\n' "${FAKE_OPENPI_COMMIT:-bd70b8f4011e85b3f3b0f039f12113f78718e7bf}"
    ;;
  "status --porcelain=v1")
    [[ "${5:-}" == --untracked-files=all ]]
    printf '%s' "${FAKE_OPENPI_STATUS:-}"
    ;;
  *) exit 98 ;;
esac
"""


def _write_executable(path: Path, value: str) -> None:
    path.write_text(textwrap.dedent(value).lstrip())
    path.chmod(0o755)


def _make_clean_repository(path: Path) -> Path:
    path.mkdir()
    module = path / "src/polaris/pi05_droid_jointpos_consumer_binding.py"
    module.parent.mkdir(parents=True)
    (module.parent / "__init__.py").write_text("")
    module.write_text(
        (ROOT / "src/polaris/pi05_droid_jointpos_consumer_binding.py").read_text()
    )
    policy = path / "src/polaris/policy"
    policy.mkdir()
    (policy / "__init__.py").write_text("")
    (policy / "droid_jointpos_client.py").write_text("FIXTURE = True\n")
    client = path / "third_party/openpi/packages/openpi-client/src/openpi_client"
    client.mkdir(parents=True)
    (client / "__init__.py").write_text("")
    (client / "image_tools.py").write_text("FIXTURE = True\n")
    (client / "websocket_client_policy.py").write_text("FIXTURE = True\n")
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
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "test fixture"], cwd=path, check=True
    )
    return batch_script


def _environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    repository = tmp_path / "repo"
    _batch_script = _make_clean_repository(repository)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    for name in ("sbatch", "squeue", "scontrol", "scancel"):
        _write_executable(fake_bin / name, FAKE_SLURM)
    _write_executable(fake_bin / "git", FAKE_GIT)
    fake_state = tmp_path / "fake-state"
    fake_state.mkdir()
    manifest = tmp_path / "results" / "canary_jobs.tsv"
    openpi_runtime = tmp_path / "openpi-runtime"
    interpreter = openpi_runtime / ".venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    source_digest = source_tree_sha256(repository)
    implementation_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    batch_path = ROOT / "scripts/polaris/l40s_pi05_eval_job.sbatch"
    approval = tmp_path / "source-approval.json"
    approval.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "openpi_pi05_droid_jointpos_source_approval_v1",
                "snapshot_path": str(repository),
                "source_tree_sha256": source_digest,
                "implementation_commit": implementation_commit,
                "polaris_base_commit": "c5b52a9cebb2c797a84e3df374b6002005d20a4f",
                "polaris_base_tree": "7fd5e1b0af26577fd323fb1d7f3595b91282e73f",
                "openpi_commit": "bd70b8f4011e85b3f3b0f039f12113f78718e7bf",
                "trusted_hasher_sha256": hashlib.sha256(
                    batch_path.read_bytes()
                ).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    approval.chmod(0o444)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "POLARIS_SOURCE_SNAPSHOT": str(repository),
            "EXPECTED_POLARIS_SOURCE_TREE_SHA256": source_digest,
            "POLARIS_SOURCE_APPROVAL": str(approval),
            "POLARIS_OPENPI_RUNTIME_DIR": str(openpi_runtime),
            "SBATCH_LOG_ROOT": str(tmp_path / "logs"),
            "SUBMISSION_MANIFEST": str(manifest),
            "RUN_NAMESPACE": "pi05-submit-host-test",
            "APPROVED_SBATCH_SCRIPT_FOR_TEST": str(
                batch_path
            ),
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


def test_overridden_batch_script_is_rejected_before_submission(tmp_path: Path) -> None:
    env, _manifest, fake_state = _environment(tmp_path)
    override = tmp_path / "override.sbatch"
    override.write_text("#!/usr/bin/env bash\n")
    env["SBATCH_SCRIPT"] = str(override)

    result = _run(env)

    assert result.returncode == 2
    assert "SBATCH_SCRIPT override is forbidden" in result.stderr
    assert not (fake_state / "sbatch.args").exists()


def test_openpi_commit_drift_is_rejected_before_submission(tmp_path: Path) -> None:
    env, _manifest, fake_state = _environment(tmp_path)
    env["FAKE_OPENPI_COMMIT"] = "0" * 40

    result = _run(env)

    assert result.returncode == 2
    assert "OpenPI runtime commit mismatch" in result.stderr
    assert not (fake_state / "sbatch.args").exists()


def test_symlinked_ancestor_inputs_are_exported_as_canonical_paths(
    tmp_path: Path,
) -> None:
    env, _manifest, fake_state = _environment(tmp_path)
    alias = tmp_path / "fsw-alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    env["POLARIS_SOURCE_SNAPSHOT"] = str(alias / "repo")
    env["POLARIS_SOURCE_APPROVAL"] = str(alias / "source-approval.json")
    env["POLARIS_OPENPI_RUNTIME_DIR"] = str(alias / "openpi-runtime")

    result = _run(env)

    assert result.returncode == 0, result.stderr
    arguments = (fake_state / "sbatch.args").read_text().splitlines()
    exported = next(value for value in arguments if value.startswith("--export="))
    assert f"POLARIS_SOURCE_SNAPSHOT={tmp_path / 'repo'}" in exported
    assert f"POLARIS_SOURCE_APPROVAL={tmp_path / 'source-approval.json'}" in exported
    assert f"POLARIS_OPENPI_RUNTIME_DIR={tmp_path / 'openpi-runtime'}" in exported
    assert "fsw-alias" not in exported


def test_existing_job_reuse_requires_identical_provenance(tmp_path: Path) -> None:
    env, manifest, fake_state = _environment(tmp_path)
    first = _run(env)
    assert first.returncode == 0, first.stderr
    (fake_state / "sbatch.args").unlink()

    exact = _run(env)
    assert exact.returncode == 0, exact.stderr
    assert "Existing canary attempt" in exact.stdout
    assert not (fake_state / "sbatch.args").exists()

    env["ROLLOUTS"] = "2"
    mismatch = _run(env)
    assert mismatch.returncode == 2
    assert "incompatible evaluation provenance" in mismatch.stderr
    assert len(manifest.read_text().splitlines()) == 2
    assert not (fake_state / "sbatch.args").exists()
