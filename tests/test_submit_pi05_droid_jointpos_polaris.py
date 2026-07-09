import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import textwrap

import pytest

from polaris.pi05_droid_jointpos_consumer_binding import source_tree_sha256
from polaris.pi05_droid_jointpos_scheduler import validate_persisted_scheduler_job


ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "scripts/polaris/submit_pi05_droid_jointpos_polaris.sh"
STALE_AUDIT_SBATCH_SHA256 = (
    "35515abaf15d7060f86ad959aaa73a53c5ed5fa9dec1e5a124fc109e91812d7e"
)
STANDARD_MANIFEST_COLUMNS = (
    "job_id",
    "mode",
    "task",
    "rollouts",
    "environment_seed",
    "run_namespace",
    "source_tree_sha256",
    "source_approval_sha256",
    "implementation_commit",
    "openpi_commit",
    "submitted_at",
    "batch_script_sha256",
    "submission_argv_sha256",
    "held_scheduler_record_sha256",
    "provenance_dir",
)
STANDARD_MANIFEST_HEADER = "\t".join(STANDARD_MANIFEST_COLUMNS)
APP_MANIFEST_COLUMNS = (*STANDARD_MANIFEST_COLUMNS, "app_runtime_provenance_sha256")
APP_MANIFEST_HEADER = "\t".join(APP_MANIFEST_COLUMNS)
RUNTIME_ROLE_PROJECTION_SHA256 = (
    "15ab50cd061a821ffbe654040a3221cb609a3c43a8b61c8eae4cb687475de6c7"
)
_PYTHON_STDLIB_RELATIVE_PATHS = (
    "__future__.py",
    "_weakrefset.py",
    "argparse.py",
    "collections/__init__.py",
    "collections/abc.py",
    "contextlib.py",
    "copyreg.py",
    "encodings/__init__.py",
    "encodings/aliases.py",
    "encodings/utf_8.py",
    "enum.py",
    "fnmatch.py",
    "functools.py",
    "gettext.py",
    "hashlib.py",
    "ipaddress.py",
    "json/__init__.py",
    "json/decoder.py",
    "json/encoder.py",
    "json/scanner.py",
    "keyword.py",
    "locale.py",
    "operator.py",
    "pathlib.py",
    "re/__init__.py",
    "re/_casefix.py",
    "re/_compiler.py",
    "re/_constants.py",
    "re/_parser.py",
    "reprlib.py",
    "selectors.py",
    "signal.py",
    "subprocess.py",
    "threading.py",
    "types.py",
    "typing.py",
    "urllib/__init__.py",
    "urllib/parse.py",
    "warnings.py",
)


def _runtime_role_paths() -> dict[str, list[str]]:
    stdlib_sources = sorted(
        f"/usr/lib/python3.12/{relative}" for relative in _PYTHON_STDLIB_RELATIVE_PATHS
    )
    stdlib_bytecode = sorted(
        str(
            Path(source).parent / "__pycache__" / f"{Path(source).stem}.cpython-312.pyc"
        )
        for source in stdlib_sources
    )
    role_paths = {
        "approval_bound_configuration": [
            "/cm/shared/apps/slurm/etc/oci-ord-cs-004/slurm.conf"
        ],
        "entrypoint": [
            "/cm/local/apps/slurm/24.11/bin/scontrol",
            "/usr/bin/python3.12",
        ],
        "python_elf_dependency": [
            "/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2",
            "/usr/lib/x86_64-linux-gnu/libc.so.6",
            "/usr/lib/x86_64-linux-gnu/libcrypto.so.3",
            "/usr/lib/x86_64-linux-gnu/libexpat.so.1.9.1",
            "/usr/lib/x86_64-linux-gnu/libm.so.6",
            "/usr/lib/x86_64-linux-gnu/libz.so.1.3",
        ],
        "python_extension_module": [
            "/usr/lib/python3.12/lib-dynload/_hashlib.cpython-312-x86_64-linux-gnu.so",
            "/usr/lib/python3.12/lib-dynload/_json.cpython-312-x86_64-linux-gnu.so",
        ],
        "python_stdlib_bytecode": stdlib_bytecode,
        "python_stdlib_resolution_landmark": ["/usr/lib/python3.12/os.py"],
        "python_stdlib_source": stdlib_sources,
        "scheduler_bootstrap_binary_drift": [
            "/cm/local/apps/slurm/24.11/bin/sacct",
            "/cm/local/apps/slurm/24.11/bin/scancel",
            "/cm/local/apps/slurm/24.11/bin/srun",
        ],
        "scontrol_elf_dependency": [
            "/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2",
            "/usr/lib/x86_64-linux-gnu/libc.so.6",
            "/usr/lib/x86_64-linux-gnu/libm.so.6",
            "/usr/lib/x86_64-linux-gnu/libmunge.so.2.0.1",
            "/usr/lib/x86_64-linux-gnu/libnss_sss.so.2",
            "/usr/lib/x86_64-linux-gnu/libreadline.so.8.2",
            "/usr/lib/x86_64-linux-gnu/libresolv.so.2",
            "/usr/lib/x86_64-linux-gnu/libtinfo.so.6.4",
        ],
        "scontrol_slurm_plugin": [
            "/cm/local/apps/slurm/24.11/lib64/slurm/accounting_storage_slurmdbd.so",
            "/cm/local/apps/slurm/24.11/lib64/slurm/auth_munge.so",
            "/cm/local/apps/slurm/24.11/lib64/slurm/cred_munge.so",
            "/cm/local/apps/slurm/24.11/lib64/slurm/hash_k12.so",
            "/cm/local/apps/slurm/24.11/lib64/slurm/tls_none.so",
        ],
        "scontrol_slurm_runtime": [
            "/cm/local/apps/slurm/24.11/lib64/slurm/libslurmfull.so"
        ],
    }
    projection = (
        json.dumps(
            role_paths,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    assert hashlib.sha256(projection).hexdigest() == RUNTIME_ROLE_PROJECTION_SHA256
    return role_paths


def _identity_record(path: str, *, roles: list[str] | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "path": path,
        "mode": (
            "0755"
            if roles and {"entrypoint", "scheduler_bootstrap_binary_drift"} & set(roles)
            else "0644"
        ),
        "uid": 0,
        "gid": 0,
        "nlink": 1,
        "size": len(path.encode("utf-8")),
        "sha256": hashlib.sha256(path.encode("utf-8")).hexdigest(),
    }
    if roles is not None:
        record["roles"] = roles
    return record


def _runtime_closure_approval() -> dict[str, object]:
    role_paths = _runtime_role_paths()
    roles_by_path: dict[str, list[str]] = {}
    for role, paths in role_paths.items():
        for path in paths:
            roles_by_path.setdefault(path, []).append(role)
    immutable_files = [
        _identity_record(path, roles=sorted(roles_by_path[path]))
        for path in sorted(roles_by_path)
    ]

    ambient_paths = {
        "nss_dns": [
            "/etc/group",
            "/etc/host.conf",
            "/etc/hosts",
            "/etc/nsswitch.conf",
            "/etc/passwd",
            "/run/systemd/resolve/resolv.conf",
            "/var/lib/sss/mc/passwd",
        ],
        "root_managed_runtime_data": [
            "/etc/ld.so.cache",
            "/etc/locale.alias",
            "/etc/ssl/openssl.cnf",
            "/usr/lib/locale/C.utf8/LC_CTYPE",
            "/usr/lib/locale/locale-archive",
            "/usr/lib/x86_64-linux-gnu/gconv/gconv-modules.cache",
            "/usr/share/zoneinfo/America/Los_Angeles",
        ],
    }
    ambient_regular_files = {
        category: [_identity_record(path) for path in paths]
        for category, paths in ambient_paths.items()
    }
    return {
        "schema_version": 1,
        "profile": "polaris_app_launcher_runtime_subclosure_approval_v1",
        "capture_scope": "allocated_pool0_compute_after_source_freeze",
        "subclosure_profile": (
            "host_python_stdlib_scontrol_and_scheduler_binary_drift_v1"
        ),
        "top_level_imports": [
            "argparse",
            "hashlib",
            "json",
            "os",
            "pathlib",
            "re",
            "signal",
            "stat",
            "subprocess",
            "sys",
            "time",
            "typing",
        ],
        "immutable_files": immutable_files,
        "symlink_bindings": [
            {
                "path": "/etc/localtime",
                "resolved_path": "/usr/share/zoneinfo/America/Los_Angeles",
                "target": "/usr/share/zoneinfo/America/Los_Angeles",
            },
            {
                "path": "/etc/resolv.conf",
                "resolved_path": "/run/systemd/resolve/resolv.conf",
                "target": "../run/systemd/resolve/resolv.conf",
            },
            {"path": "/lib", "resolved_path": "/usr/lib", "target": "usr/lib"},
            {
                "path": "/lib/x86_64-linux-gnu/libexpat.so.1",
                "resolved_path": "/usr/lib/x86_64-linux-gnu/libexpat.so.1.9.1",
                "target": "libexpat.so.1.9.1",
            },
            {
                "path": "/lib/x86_64-linux-gnu/libmunge.so.2",
                "resolved_path": "/usr/lib/x86_64-linux-gnu/libmunge.so.2.0.1",
                "target": "libmunge.so.2.0.1",
            },
            {
                "path": "/lib/x86_64-linux-gnu/libreadline.so.8",
                "resolved_path": "/usr/lib/x86_64-linux-gnu/libreadline.so.8.2",
                "target": "libreadline.so.8.2",
            },
            {
                "path": "/lib/x86_64-linux-gnu/libtinfo.so.6",
                "resolved_path": "/usr/lib/x86_64-linux-gnu/libtinfo.so.6.4",
                "target": "libtinfo.so.6.4",
            },
            {
                "path": "/lib/x86_64-linux-gnu/libz.so.1",
                "resolved_path": "/usr/lib/x86_64-linux-gnu/libz.so.1.3",
                "target": "libz.so.1.3",
            },
            {"path": "/lib64", "resolved_path": "/usr/lib64", "target": "usr/lib64"},
            {
                "path": "/usr/lib/ssl/openssl.cnf",
                "resolved_path": "/etc/ssl/openssl.cnf",
                "target": "/etc/ssl/openssl.cnf",
            },
            {
                "path": "/usr/lib64/ld-linux-x86-64.so.2",
                "resolved_path": ("/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2"),
                "target": "../lib/x86_64-linux-gnu/ld-linux-x86-64.so.2",
            },
            {
                "path": "/usr/share/locale/locale.alias",
                "resolved_path": "/etc/locale.alias",
                "target": "/etc/locale.alias",
            },
            {"path": "/var/run", "resolved_path": "/run", "target": "/run"},
        ],
        "ambient_runtime_dependencies": {
            "external_services": [
                {
                    "hosts": [
                        "oci-ord-cs-004-slurm-ctld-01",
                        "oci-ord-cs-004-slurm-ctld-02",
                    ],
                    "kind": "tcp_service",
                    "port": 6817,
                    "role": "slurm_controller_response",
                },
                {"kind": "unix_service", "role": "munge_authentication"},
                {"kind": "resolver_service", "role": "dns_resolution"},
                {
                    "conditional": True,
                    "kind": "unix_service",
                    "role": "sssd_nss_cache_miss",
                },
                {
                    "kind": "kernel_virtual_object",
                    "name": "linux-vdso.so.1",
                    "role": "elf_runtime",
                },
            ],
            "negative_resolution_assertions": ["/usr/lib/python312.zip"],
            "regular_files": ambient_regular_files,
            "scope": "point_in_time_observation_not_immutable_code_approval",
            "sockets": [
                {
                    "gid": 0,
                    "mode": "0777",
                    "path": "/run/munge/munge.socket.2",
                    "present": True,
                    "role": "munge_authentication",
                    "uid": 1,
                },
                {
                    "conditional": True,
                    "gid": 0,
                    "mode": "0666",
                    "path": "/var/lib/sss/pipes/nss",
                    "present": True,
                    "role": "sssd_nss_cache_miss",
                    "uid": 0,
                },
            ],
        },
        "trust_boundary": {
            "bootstrap_trust": [
                "kernel and dynamic loader before approval verification",
                "Slurm job launch and Pyxis/srun lifecycle",
                "the shell and hashing/verifier tools used before host Python is trusted",
                "root-owned directory traversal and symlink resolution",
            ],
            "execution_subclosure": (
                "Host Python standard-library helper plus pinned scontrol scheduler "
                "client. sacct, scancel, and srun executable identities are pinned "
                "only as bootstrap drift evidence; their dynamic closures, Bash, "
                "coreutils, Pyxis, container runtime, NVIDIA tools, kernel, and "
                "external services are not claimed as closed by this approval."
            ),
            "security_claim": (
                "This approval supports drift detection under a trusted root-managed "
                "host; it is not a self-authenticating root of trust because Python "
                "cannot establish the trustworthiness of Python before executing."
            ),
        },
    }


def _write_runtime_closure_approval(path: Path) -> str:
    # The login submitter inspects this closed schema and immutable artifact only.
    # Allocated-node tests separately exercise the batch script's live path checks.
    payload = (
        json.dumps(
            _runtime_closure_approval(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    path.write_bytes(payload)
    path.chmod(0o444)
    return hashlib.sha256(payload).hexdigest()


FAKE_SLURM = r"""
#!/usr/bin/env bash
set -euo pipefail

command_name="$(basename "$0")"
case "${command_name}" in
  sbatch)
    : > "${FAKE_STATE}/sbatch.args"
    comment=""
    held=0
    no_requeue=0
    requeue=0
    for argument in "$@"; do
      printf '%s\n' "${argument}" >> "${FAKE_STATE}/sbatch.args"
      case "${argument}" in
        --comment=*) comment="${argument#--comment=}" ;;
        --hold) held=1 ;;
        --no-requeue) no_requeue=$((no_requeue + 1)) ;;
        --requeue) requeue=$((requeue + 1)) ;;
      esac
    done
    [[ "${held}" == 1 && "${no_requeue}" == 1 && "${requeue}" == 0 ]]
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
      show)
        [[ "${2:-}" == job && "${3:-}" == 4242 && "${4:-}" == --oneliner ]]
        printf 'show %s\n' "${3}" >> "${FAKE_STATE}/scontrol.log"
        transaction="$(cut -d '|' -f 2 "${FAKE_STATE}/active")"
        if [[ "${FAKE_COMMENT_MISMATCH:-0}" == 1 ]]; then
          transaction=pi05-0000000000000000000000000000000000000000
        fi
        printf 'JobId=4242 JobState=%s Reason=%s Requeue=%s Restarts=%s Comment=%s' \
          "${FAKE_JOB_STATE:-PENDING}" "${FAKE_JOB_REASON:-JobHeldUser}" \
          "${FAKE_REQUEUE:-0}" "${FAKE_RESTARTS:-0}" "${transaction}"
        if [[ "${FAKE_DUPLICATE_REQUEUE:-0}" == 1 ]]; then
          printf ' Requeue=0'
        fi
        printf '\n'
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
          row_openpi_commit row_time batch_sha argv_sha scheduler_sha provenance \
          app_runtime_sha unexpected <<< "${manifest_row}"
        [[ -z "${unexpected}" ]]
        [[ "${row_id}" == 4242 && "${row_mode}" == "${FAKE_EXPECTED_MODE:-canary}" ]]
        [[ "${row_task}" == DROID-FoodBussing && "${row_rollouts}" == 1 ]]
        [[ "${row_seed}" == 0 && -n "${row_namespace}" && -n "${row_time}" ]]
        [[ "${row_source_sha}" =~ ^[0-9a-f]{64}$ ]]
        [[ "${row_approval_sha}" =~ ^[0-9a-f]{64}$ ]]
        [[ "${row_implementation}" =~ ^[0-9a-f]{40}$ ]]
        [[ "${row_openpi_commit}" == bd70b8f4011e85b3f3b0f039f12113f78718e7bf ]]
        [[ "$(stat -c '%a' "${provenance}/batch_script.sbatch")" == 444 ]]
        [[ "$(stat -c '%a' "${provenance}/submission_argv.sh")" == 444 ]]
        [[ "$(stat -c '%a' "${provenance}/scheduler_held.json")" == 444 ]]
        [[ "$(sha256sum "${provenance}/batch_script.sbatch" | awk '{print $1}')" == "${batch_sha}" ]]
        [[ "$(sha256sum "${provenance}/submission_argv.sh" | awk '{print $1}')" == "${argv_sha}" ]]
        [[ "$(sha256sum "${provenance}/scheduler_held.json" | awk '{print $1}')" == "${scheduler_sha}" ]]
        if [[ "${row_mode}" == app-launcher-only ]]; then
          [[ "$(head -n 1 "${SUBMISSION_MANIFEST}")" == "${FAKE_APP_MANIFEST_HEADER}" ]]
          [[ "${app_runtime_sha}" =~ ^[0-9a-f]{64}$ ]]
          [[ "$(stat -c '%a' "${provenance}/app_runtime_approval.env")" == 444 ]]
          [[ "$(stat -c '%h' "${provenance}/app_runtime_approval.env")" == 1 ]]
          [[ "$(sha256sum "${provenance}/app_runtime_approval.env" | awk '{print $1}')" == "${app_runtime_sha}" ]]
        else
          [[ "$(head -n 1 "${SUBMISSION_MANIFEST}")" == "${FAKE_STANDARD_MANIFEST_HEADER}" ]]
          [[ -z "${app_runtime_sha}" ]]
        fi
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


def _replace_exact(source: str, old: str, new: str, *, count: int) -> str:
    assert old != new
    assert source.count(old) == count
    transformed = source.replace(old, new)
    assert transformed.count(old) == 0
    assert transformed.count(new) >= count
    return transformed


def _make_submitter_harness(tmp_path: Path, fake_bin: Path) -> tuple[Path, Path]:
    harness_scripts = tmp_path / "submitter-harness/scripts/polaris"
    harness_scripts.mkdir(parents=True)
    source_batch = ROOT / "scripts/polaris/l40s_pi05_eval_job.sbatch"
    harness_batch = harness_scripts / source_batch.name
    batch_payload = source_batch.read_bytes()
    harness_batch.write_bytes(batch_payload)
    harness_batch.chmod(source_batch.stat().st_mode & 0o777)
    current_batch_sha256 = hashlib.sha256(batch_payload).hexdigest()

    source = SUBMITTER.read_text()
    pin_prefix = 'APPROVED_SBATCH_SCRIPT_SHA256="'
    pin_lines = [line for line in source.splitlines() if line.startswith(pin_prefix)]
    assert len(pin_lines) == 1
    pinned_batch_sha256 = pin_lines[0].removeprefix(pin_prefix).removesuffix('"')
    assert pinned_batch_sha256 in {
        STALE_AUDIT_SBATCH_SHA256,
        current_batch_sha256,
    }
    if pinned_batch_sha256 != current_batch_sha256:
        source = _replace_exact(
            source,
            pin_lines[0],
            f'APPROVED_SBATCH_SCRIPT_SHA256="{current_batch_sha256}"',
            count=1,
        )
    source = _replace_exact(
        source,
        "/cm/local/apps/slurm/24.11/bin/",
        f"{fake_bin}/",
        count=7,
    )
    source = _replace_exact(
        source,
        "export PATH=/cm/local/apps/slurm/24.11/bin:/usr/bin:/bin",
        f"export PATH={fake_bin}:/usr/bin:/bin",
        count=1,
    )
    source = _replace_exact(
        source,
        '"PATH": "/cm/local/apps/slurm/24.11/bin:/usr/bin:/bin",',
        f'"PATH": "{fake_bin}:/usr/bin:/bin",',
        count=1,
    )
    assert "/cm/local/apps/slurm/24.11/bin" not in source
    harness_submitter = harness_scripts / SUBMITTER.name
    harness_submitter.write_text(source)
    harness_submitter.chmod(0o755)
    return harness_submitter, harness_batch


def _make_clean_repository(path: Path) -> Path:
    path.mkdir()
    module = path / "src/polaris/pi05_droid_jointpos_consumer_binding.py"
    module.parent.mkdir(parents=True)
    (module.parent / "__init__.py").write_text("")
    module.write_text(
        (ROOT / "src/polaris/pi05_droid_jointpos_consumer_binding.py").read_text()
    )
    (module.parent / "pi05_droid_jointpos_immutable.py").write_text(
        (ROOT / "src/polaris/pi05_droid_jointpos_immutable.py").read_text()
    )
    scheduler_path = module.parent / "pi05_droid_jointpos_scheduler.py"
    scheduler_source = (
        ROOT / "src/polaris/pi05_droid_jointpos_scheduler.py"
    ).read_text()
    scheduler_signature = (
        "def validate_sacct_runtime_approval(\n"
        "    path: Path, *, expected_sha256: str, live: bool\n"
        ") -> dict[str, Any]:\n"
    )
    scheduler_test_gate = (
        scheduler_signature
        + """    if os.environ.get(
        "POLARIS_TEST_SYNTHETIC_SACCT_APPROVAL"
    ) == "1":
        artifact, _ = _stable_payload(Path(path), expected_sha256=expected_sha256)
        surface = {
            "hostname": os.uname().nodename,
            "machine_id_sha256": _sha256(Path("/etc/machine-id").read_bytes()),
            "kernel_release": os.uname().release,
            "architecture": os.uname().machine,
            "effective_uid": os.geteuid(),
            "effective_gid": os.getegid(),
        }
        identity = {
            "path": artifact["path"],
            "size": artifact["size"],
            "sha256": artifact["sha256"],
            "mode": artifact["mode"],
            "nlink": artifact["nlink"],
        }
        trace = {
            "candidate": {**identity, "path": str(Path(path).parent / "candidate.json")},
            "capture_terminal": {
                **identity,
                "path": str(Path(path).parent / "capture.json"),
            },
            "review_decision": {
                **identity,
                "path": str(Path(path).parent / "review.json"),
            },
            "closure_sha256": "a" * 64,
        }
        return {
            "artifact": artifact,
            "closure_sha256": "a" * 64,
            "execution_surface": surface,
            "query_contract": {
                "profile": "polaris_external_sacct_query_v1",
                "command_template": _expected_sacct_query_command_template(),
                "environment": {
                    "PATH": SACCT_QUERY_PATH,
                    "SLURM_CONF": str(PINNED_SLURM_CONFIG_PATH),
                    "LD_LIBRARY_PATH": SACCT_QUERY_LD_LIBRARY_PATH,
                },
                "subprocess_timeout_seconds": SACCT_SUBPROCESS_TIMEOUT_SECONDS,
            },
            "key_files": {},
            "trace_evidence": trace,
            "capture_producer": {
                "profile": SACCT_RUNTIME_CAPTURE_PRODUCER_PROFILE,
                "module": SACCT_RUNTIME_CAPTURE_PRODUCER_MODULE,
                "path": str(Path(path).parent / "capture-producer.py"),
                "sha256": "b" * 64,
                "uid": os.geteuid(),
                "host": os.uname().nodename,
            },
            "capture_job": {
                "job_id": 4242,
                "compute_node": "pool0-test",
                "transaction_id": (
                    "polaris-runtime-trace-v2-01234567-89ab-cdef-0123-456789abcdef"
                ),
            },
            "reviewer_identity": {
                "profile": "polaris_external_sacct_runtime_reviewer_identity_v2",
                "principal": "codex-agent:/root/synthetic_independent_reviewer",
                "role": "independent_agent_runtime_approval_reviewer",
            },
        }
"""
    )
    assert scheduler_source.count(scheduler_signature) == 1
    scheduler_path.write_text(
        scheduler_source.replace(scheduler_signature, scheduler_test_gate)
    )
    scripts = path / "scripts/polaris"
    scripts.mkdir(parents=True)
    (scripts / "finalize_pi05_app_launcher_only.py").write_text(
        (ROOT / "scripts/polaris/finalize_pi05_app_launcher_only.py").read_text()
    )
    for name in (
        "app_launcher_startup_diagnostic.py",
        "config.py",
        "evaluation_seed.py",
    ):
        (module.parent / name).write_text((ROOT / "src/polaris" / name).read_text())
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
    subprocess.run(["git", "config", "user.name", "Submit Test"], cwd=path, check=True)
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "test fixture"], cwd=path, check=True)
    scheduler_path.chmod(0o444)
    return batch_script


def _environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    repository = tmp_path / "repo"
    _batch_script = _make_clean_repository(repository)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    for name in ("sbatch", "squeue", "scontrol", "scancel"):
        _write_executable(fake_bin / name, FAKE_SLURM)
    _write_executable(fake_bin / "git", FAKE_GIT)
    submitter, batch_path = _make_submitter_harness(tmp_path, fake_bin)
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
    runtime_approval = tmp_path / "runtime-closure-approval.json"
    runtime_approval_sha256 = _write_runtime_closure_approval(runtime_approval)
    sacct_runtime_approval = tmp_path / "sacct-runtime-approval.json"
    sacct_runtime_approval.write_text(
        '{"profile":"reviewed-test-fixture"}\n', encoding="ascii"
    )
    sacct_runtime_approval.chmod(0o444)
    sacct_runtime_approval_sha256 = hashlib.sha256(
        sacct_runtime_approval.read_bytes()
    ).hexdigest()
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "POLARIS_SOURCE_SNAPSHOT": str(repository),
            "EXPECTED_POLARIS_SOURCE_TREE_SHA256": source_digest,
            "POLARIS_SOURCE_APPROVAL": str(approval),
            "POLARIS_OPENPI_RUNTIME_DIR": str(openpi_runtime),
            "POLARIS_RUNTIME_CLOSURE_APPROVAL": str(runtime_approval),
            "POLARIS_RUNTIME_CLOSURE_APPROVAL_SHA256": runtime_approval_sha256,
            "POLARIS_SACCT_RUNTIME_APPROVAL": str(sacct_runtime_approval),
            "POLARIS_SACCT_RUNTIME_APPROVAL_SHA256": sacct_runtime_approval_sha256,
            "POLARIS_TEST_SYNTHETIC_SACCT_APPROVAL": "1",
            "SBATCH_LOG_ROOT": str(tmp_path / "logs"),
            "SUBMISSION_MANIFEST": str(manifest),
            "OUTPUT_ROOT": str(tmp_path / "output"),
            "RUN_NAMESPACE": "pi05-submit-host-test",
            "APPROVED_SBATCH_SCRIPT_FOR_TEST": str(batch_path),
            "FAKE_STATE": str(fake_state),
            "FAKE_STANDARD_MANIFEST_HEADER": STANDARD_MANIFEST_HEADER,
            "FAKE_APP_MANIFEST_HEADER": APP_MANIFEST_HEADER,
            "TEST_SUBMITTER": str(submitter),
            "USER": "submit-test",
        }
    )
    return env, manifest, fake_state


def _run(env: dict[str, str], mode: str = "canary") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "/usr/bin/bash",
            "--noprofile",
            "--norc",
            "-p",
            env["TEST_SUBMITTER"],
            mode,
        ],
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


def _env_record(path: Path) -> dict[str, str]:
    payload = path.read_text()
    assert payload.endswith("\n")
    fields: dict[str, str] = {}
    for line in payload.splitlines():
        key, separator, value = line.partition("=")
        assert separator
        assert key not in fields
        fields[key] = value
    return fields


def test_held_job_is_released_only_after_durable_provenance_and_manifest(
    tmp_path: Path,
) -> None:
    env, manifest, fake_state = _environment(tmp_path)

    result = _run(env)

    assert result.returncode == 0, result.stderr
    assert "submitted_job_ids=4242" in result.stdout
    arguments = (fake_state / "sbatch.args").read_text().splitlines()
    assert "--hold" in arguments
    assert arguments.count("--no-requeue") == 1
    assert "--requeue" not in arguments
    comments = [value for value in arguments if value.startswith("--comment=pi05-")]
    assert len(comments) == 1
    assert len(comments[0].removeprefix("--comment=")) == 45
    assert (fake_state / "scontrol.log").read_text().splitlines() == [
        "write 4242",
        "show 4242",
        "release 4242",
    ]
    assert not (fake_state / "scancel.log").exists()
    assert _transaction_states(manifest) == ["released"]
    rows = manifest.read_text().splitlines()
    assert len(rows) == 2
    assert rows[1].startswith("4242\tcanary\tDROID-FoodBussing\t")
    provenance = manifest.parent / "submission_provenance/job_4242"
    for name in ("batch_script.sbatch", "submission_argv.sh", "scheduler_held.json"):
        artifact = provenance / name
        assert artifact.is_file()
        assert stat.S_IMODE(artifact.stat().st_mode) == 0o444
    scheduler = validate_persisted_scheduler_job(
        provenance / "scheduler_held.json",
        phase="held",
        expected_job_id=4242,
        expected_transaction_id=comments[0].removeprefix("--comment="),
    )
    assert scheduler["value"]["job"]["requeue"] == 0
    assert scheduler["value"]["job"]["restarts"] == 0


def test_app_launcher_only_submits_one_distinct_non_scientific_job(
    tmp_path: Path,
) -> None:
    env, manifest, fake_state = _environment(tmp_path)
    env["FAKE_EXPECTED_MODE"] = "app-launcher-only"

    result = _run(env, "app-launcher-only")

    assert result.returncode == 0, result.stderr
    rows = manifest.read_text().splitlines()
    assert len(rows) == 2
    assert rows[0] == APP_MANIFEST_HEADER
    fields = rows[1].split("\t")
    assert len(fields) == len(APP_MANIFEST_COLUMNS)
    assert fields[1:5] == ["app-launcher-only", "DROID-FoodBussing", "1", "0"]
    provenance = Path(fields[14])
    runtime_provenance = provenance / "app_runtime_approval.env"
    assert stat.S_IMODE(runtime_provenance.stat().st_mode) == 0o444
    assert runtime_provenance.stat().st_nlink == 1
    assert hashlib.sha256(runtime_provenance.read_bytes()).hexdigest() == fields[15]
    provenance_fields = _env_record(runtime_provenance)
    assert set(provenance_fields) == {
        "profile",
        "output_root",
        "output_namespace_parent",
        "output_namespace_parent_identity",
        "runtime_closure_approval",
        "runtime_closure_approval_sha256",
        "sacct_runtime_approval",
        "sacct_runtime_approval_sha256",
        "sacct_prelaunch_validation_receipt",
        "sacct_prelaunch_validation_receipt_sha256",
        "scheduler_query_profile",
        "scheduler_query_path",
        "scheduler_query_slurm_conf",
        "scheduler_query_ld_library_path",
        "scheduler_query_timeout_seconds",
        "expected_slurm_config_path",
        "expected_slurm_config_sha256",
        "expected_slurm_config_size",
        "expected_scontrol_sha256",
        "expected_scontrol_size",
        "expected_slurm_library_path",
        "expected_slurm_library_sha256",
        "expected_slurm_library_size",
        "expected_sacct_path",
        "expected_sacct_sha256",
        "expected_sacct_size",
        "expected_scancel_sha256",
        "expected_scancel_size",
        "expected_srun_sha256",
        "expected_srun_size",
        "approved_batch_script",
        "batch_script_sha256",
        "submission_argv_sha256",
        "held_scheduler_record_sha256",
    }
    assert provenance_fields["profile"] == "polaris_app_launcher_runtime_approval_v6"
    assert provenance_fields["scheduler_query_profile"] == (
        "polaris_app_launcher_sacct_query_v1"
    )
    assert provenance_fields["scheduler_query_path"] == "/usr/bin:/bin"
    assert provenance_fields["scheduler_query_slurm_conf"] == (
        "/cm/shared/apps/slurm/etc/oci-ord-cs-004/slurm.conf"
    )
    assert provenance_fields["scheduler_query_ld_library_path"] == (
        "/cm/local/apps/slurm/24.11/lib64:/cm/local/apps/slurm/24.11/lib64/slurm"
    )
    assert provenance_fields["scheduler_query_timeout_seconds"] == "10"
    assert (
        provenance_fields["expected_slurm_config_path"]
        == (provenance_fields["scheduler_query_slurm_conf"])
    )
    assert provenance_fields["expected_slurm_library_path"] == (
        "/cm/local/apps/slurm/24.11/lib64/slurm/libslurmfull.so"
    )
    assert provenance_fields["expected_sacct_path"] == (
        str(fake_state.parent / "fake-bin/sacct")
    )
    assert (
        provenance_fields["runtime_closure_approval"]
        == env["POLARIS_RUNTIME_CLOSURE_APPROVAL"]
    )
    assert (
        provenance_fields["runtime_closure_approval_sha256"]
        == env["POLARIS_RUNTIME_CLOSURE_APPROVAL_SHA256"]
    )
    assert (
        provenance_fields["sacct_runtime_approval"]
        == env["POLARIS_SACCT_RUNTIME_APPROVAL"]
    )
    assert (
        provenance_fields["sacct_runtime_approval_sha256"]
        == env["POLARIS_SACCT_RUNTIME_APPROVAL_SHA256"]
    )
    prelaunch_receipt = Path(provenance_fields["sacct_prelaunch_validation_receipt"])
    assert prelaunch_receipt.name == "sacct_runtime_prelaunch_validation.json"
    assert stat.S_IMODE(prelaunch_receipt.stat().st_mode) == 0o444
    assert prelaunch_receipt.stat().st_nlink == 1
    assert (
        hashlib.sha256(prelaunch_receipt.read_bytes()).hexdigest()
        == (provenance_fields["sacct_prelaunch_validation_receipt_sha256"])
    )
    receipt_value = json.loads(prelaunch_receipt.read_bytes())
    assert receipt_value["status"] == "full_live_validation_passed_before_sbatch"
    assert receipt_value["approval"]["path"] == env["POLARIS_SACCT_RUNTIME_APPROVAL"]
    assert provenance_fields["batch_script_sha256"] == fields[11]
    assert provenance_fields["submission_argv_sha256"] == fields[12]
    assert provenance_fields["held_scheduler_record_sha256"] == fields[13]
    runtime_approval = Path(env["POLARIS_RUNTIME_CLOSURE_APPROVAL"])
    assert stat.S_IMODE(runtime_approval.stat().st_mode) == 0o444
    assert runtime_approval.stat().st_nlink == 1
    assert (
        hashlib.sha256(runtime_approval.read_bytes()).hexdigest()
        == env["POLARIS_RUNTIME_CLOSURE_APPROVAL_SHA256"]
    )
    assert json.loads(runtime_approval.read_bytes()) == _runtime_closure_approval()
    arguments = (fake_state / "sbatch.args").read_text().splitlines()
    export_argument = next(item for item in arguments if item.startswith("--export="))
    assert "POLARIS_EVAL_MODE=app_launcher_only" in export_argument
    assert any(item == "--job-name=pi05-app-launcher_FoodBussing" for item in arguments)
    assert any(item == "--time=00:30:00" for item in arguments)

    before = manifest.read_bytes()
    reused = _run(env, "app-launcher-only")
    assert reused.returncode == 0, reused.stderr
    assert "Existing app-launcher-only attempt" in reused.stdout
    assert manifest.read_bytes() == before
    assert (fake_state / "scontrol.log").read_text().splitlines() == [
        "write 4242",
        "show 4242",
        "release 4242",
    ]


def test_app_launcher_reuse_rejects_mutated_runtime_provenance(
    tmp_path: Path,
) -> None:
    env, manifest, fake_state = _environment(tmp_path)
    env["FAKE_EXPECTED_MODE"] = "app-launcher-only"
    first = _run(env, "app-launcher-only")
    assert first.returncode == 0, first.stderr

    fields = manifest.read_text().splitlines()[1].split("\t")
    runtime_provenance = Path(fields[14]) / "app_runtime_approval.env"
    runtime_provenance.chmod(0o644)
    runtime_provenance.write_bytes(runtime_provenance.read_bytes() + b"tampered=1\n")
    runtime_provenance.chmod(0o444)

    second = _run(env, "app-launcher-only")
    assert second.returncode == 2
    assert "incompatible AppLauncher runtime approval" in second.stderr
    assert len(manifest.read_text().splitlines()) == 2
    assert (fake_state / "scontrol.log").read_text().splitlines() == [
        "write 4242",
        "show 4242",
        "release 4242",
    ]


def test_app_launcher_rejects_mutated_external_sacct_approval(
    tmp_path: Path,
) -> None:
    env, manifest, fake_state = _environment(tmp_path)
    env["FAKE_EXPECTED_MODE"] = "app-launcher-only"
    first = _run(env, "app-launcher-only")
    assert first.returncode == 0, first.stderr

    approval = Path(env["POLARIS_SACCT_RUNTIME_APPROVAL"])
    approval.chmod(0o644)
    approval.write_bytes(approval.read_bytes() + b"tampered\n")
    approval.chmod(0o444)

    second = _run(env, "app-launcher-only")
    assert second.returncode == 2
    assert "external sacct runtime approval identity mismatch" in second.stderr
    assert len(manifest.read_text().splitlines()) == 2
    assert (fake_state / "scontrol.log").read_text().splitlines() == [
        "write 4242",
        "show 4242",
        "release 4242",
    ]


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
        "show 4242",
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


@pytest.mark.parametrize(
    ("override", "detail"),
    [
        ({"FAKE_REQUEUE": "1"}, "permits requeue"),
        ({"FAKE_RESTARTS": "1"}, "already restarted"),
        ({"FAKE_JOB_STATE": "RUNNING"}, "held state mismatch"),
        ({"FAKE_JOB_REASON": "None"}, "not user-held"),
        ({"FAKE_COMMENT_MISMATCH": "1"}, "comment mismatch"),
        ({"FAKE_DUPLICATE_REQUEUE": "1"}, "duplicate scontrol field"),
    ],
)
def test_held_scheduler_contract_failure_cancels_before_release(
    tmp_path: Path, override: dict[str, str], detail: str
) -> None:
    env, manifest, fake_state = _environment(tmp_path)
    env.update(override)

    result = _run(env)

    assert result.returncode == 5
    assert detail in result.stderr
    assert "Failed to preserve submission provenance" in result.stderr
    assert (fake_state / "scancel.log").read_text().splitlines() == ["4242"]
    assert "release" not in (fake_state / "scontrol.log").read_text()
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
