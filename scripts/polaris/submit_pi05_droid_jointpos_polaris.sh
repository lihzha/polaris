#!/usr/bin/bash -p

# Submit ordinary, independently restartable official pi0.5 PolaRiS eval jobs.

set -Eeuo pipefail
[[ -o privileged ]] || { echo "Privileged Bash mode (-p) is required" >&2; exit 2; }
INHERITED_PATH="${PATH:-/usr/bin:/bin}"
unset BASH_ENV ENV LD_AUDIT LD_PRELOAD PYTHONHOME PYTHONPATH PYTHONUSERBASE

die() {
  echo "$*" >&2
  exit 2
}

run_bounded_host_python() {
  /usr/bin/env \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    BLIS_NUM_THREADS=1 \
    "$@"
}

write_atomic_text() {
  local path="$1"
  local mode="$2"
  local value="$3"
  local directory temporary
  directory="$(dirname "${path}")"
  temporary="${directory}/.$(basename "${path}").tmp.${BASHPID}.${RANDOM}"
  (
    umask 077
    printf '%s\n' "${value}" > "${temporary}"
  )
  chmod "${mode}" "${temporary}"
  sync -- "${temporary}"
  mv -f -- "${temporary}" "${path}"
  sync -- "${directory}"
}

append_manifest_row() {
  local row="$1"
  local directory temporary
  directory="$(dirname "${SUBMISSION_MANIFEST}")"
  temporary="${directory}/.$(basename "${SUBMISSION_MANIFEST}").tmp.${BASHPID}.${RANDOM}"
  cp -- "${SUBMISSION_MANIFEST}" "${temporary}"
  printf '%s\n' "${row}" >> "${temporary}"
  sync -- "${temporary}"
  mv -f -- "${temporary}" "${SUBMISSION_MANIFEST}"
  sync -- "${directory}"
}

prepare_app_output_namespace() {
  local raw_output_root="$1"
  local run_namespace="$2"
  /usr/bin/python3.12 -I -S - \
    "${raw_output_root}" "${run_namespace}" <<'PY'
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys


raw_root, namespace = sys.argv[1:]
pure_root = PurePosixPath(raw_root)
if (
    not pure_root.is_absolute()
    or pure_root.as_posix() != raw_root
    or any(part in {"", ".", ".."} for part in pure_root.parts)
    or any(character in raw_root for character in ",=\t\r\n")
):
    raise ValueError("OUTPUT_ROOT must use safe canonical absolute syntax")
if re.fullmatch(r"[A-Za-z0-9._-]+", namespace) is None:
    raise ValueError("RUN_NAMESPACE is unsafe")

Path(raw_root).mkdir(parents=True, exist_ok=True)
output_root = Path(raw_root).resolve(strict=True)
if not output_root.is_dir() or output_root.is_symlink():
    raise ValueError("OUTPUT_ROOT is not one real directory")
canonical_root = str(output_root)
if any(character in canonical_root for character in ",=\t\r\n"):
    raise ValueError("canonical OUTPUT_ROOT is unsafe for Slurm export")

root_fd = os.open(
    output_root,
    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
)
namespace_fd = -1
try:
    root_before = os.fstat(root_fd)
    os.mkdir(namespace, 0o755, dir_fd=root_fd)
    namespace_fd = os.open(
        namespace,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=root_fd,
    )
    os.fchmod(namespace_fd, 0o755)
    os.fsync(namespace_fd)
    os.fsync(root_fd)
    opened = os.fstat(namespace_fd)
    observed = os.stat(namespace, dir_fd=root_fd, follow_symlinks=False)
    root_after = os.fstat(root_fd)
    root_observed = os.stat(output_root, follow_symlinks=False)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or stat.S_IMODE(opened.st_mode) != 0o755
        or opened.st_dev != observed.st_dev
        or opened.st_ino != observed.st_ino
        or root_before.st_dev != root_after.st_dev
        or root_before.st_ino != root_after.st_ino
        or root_after.st_dev != root_observed.st_dev
        or root_after.st_ino != root_observed.st_ino
    ):
        raise RuntimeError("output namespace binding changed during creation")
    identity = ":".join(
        (
            str(opened.st_dev),
            str(opened.st_ino),
            str(opened.st_uid),
            str(opened.st_gid),
            f"{stat.S_IMODE(opened.st_mode):04o}",
        )
    )
    print(f"{canonical_root}\t{canonical_root}/{namespace}\t{identity}")
finally:
    if namespace_fd >= 0:
        os.close(namespace_fd)
    os.close(root_fd)
PY
}

build_app_runtime_approval_record() {
  printf -v APP_RUNTIME_APPROVAL_RECORD \
    'profile=polaris_app_launcher_runtime_approval_v6\noutput_root=%s\noutput_namespace_parent=%s\noutput_namespace_parent_identity=%s\nruntime_closure_approval=%s\nruntime_closure_approval_sha256=%s\nsacct_runtime_approval=%s\nsacct_runtime_approval_sha256=%s\nsacct_prelaunch_validation_receipt=%s\nsacct_prelaunch_validation_receipt_sha256=%s\nscheduler_query_profile=polaris_app_launcher_sacct_query_v1\nscheduler_query_path=/usr/bin:/bin\nscheduler_query_slurm_conf=/cm/shared/apps/slurm/etc/oci-ord-cs-004/slurm.conf\nscheduler_query_ld_library_path=/cm/local/apps/slurm/24.11/lib64:/cm/local/apps/slurm/24.11/lib64/slurm\nscheduler_query_timeout_seconds=10\nexpected_slurm_config_path=/cm/shared/apps/slurm/etc/oci-ord-cs-004/slurm.conf\nexpected_slurm_config_sha256=%s\nexpected_slurm_config_size=%s\nexpected_scontrol_sha256=%s\nexpected_scontrol_size=%s\nexpected_slurm_library_path=/cm/local/apps/slurm/24.11/lib64/slurm/libslurmfull.so\nexpected_slurm_library_sha256=%s\nexpected_slurm_library_size=%s\nexpected_sacct_path=/cm/local/apps/slurm/24.11/bin/sacct\nexpected_sacct_sha256=%s\nexpected_sacct_size=%s\nexpected_scancel_sha256=%s\nexpected_scancel_size=%s\nexpected_srun_sha256=%s\nexpected_srun_size=%s' \
    "${OUTPUT_ROOT}" "${OUTPUT_NAMESPACE_PARENT}" \
    "${POLARIS_OUTPUT_NAMESPACE_PARENT_IDENTITY}" \
    "${POLARIS_RUNTIME_CLOSURE_APPROVAL}" \
    "${POLARIS_RUNTIME_CLOSURE_APPROVAL_SHA256}" \
    "${POLARIS_SACCT_RUNTIME_APPROVAL}" \
    "${POLARIS_SACCT_RUNTIME_APPROVAL_SHA256}" \
    "${POLARIS_SACCT_PRELAUNCH_VALIDATION_RECEIPT}" \
    "${POLARIS_SACCT_PRELAUNCH_VALIDATION_RECEIPT_SHA256}" \
    "${POLARIS_EXPECTED_SLURM_CONFIG_SHA256}" \
    "${POLARIS_EXPECTED_SLURM_CONFIG_SIZE}" \
    "${POLARIS_EXPECTED_SCONTROL_SHA256}" \
    "${POLARIS_EXPECTED_SCONTROL_SIZE}" \
    "${POLARIS_EXPECTED_SLURM_LIBRARY_SHA256}" \
    "${POLARIS_EXPECTED_SLURM_LIBRARY_SIZE}" \
    "${POLARIS_EXPECTED_SACCT_SHA256}" \
    "${POLARIS_EXPECTED_SACCT_SIZE}" \
    "${POLARIS_EXPECTED_SCANCEL_SHA256}" \
    "${POLARIS_EXPECTED_SCANCEL_SIZE}" \
    "${POLARIS_EXPECTED_SRUN_SHA256}" \
    "${POLARIS_EXPECTED_SRUN_SIZE}"
}

validate_app_runtime_provenance() {
  local provenance_dir="$1"
  local expected_argv_sha256="$2"
  local expected_provenance_sha256="$3"
  local expected_scheduler_sha256="$4"
  local expected_job_id="$5"
  /usr/bin/python3.12 -I -S - \
    "${provenance_dir}" "${expected_argv_sha256}" \
    "${APPROVED_SBATCH_SCRIPT_SHA256}" "${expected_provenance_sha256}" \
    "${expected_scheduler_sha256}" "${expected_job_id}" \
    "${OUTPUT_ROOT}" "${RUN_NAMESPACE}" \
    "${POLARIS_SOURCE_SNAPSHOT}" \
    "${EXPECTED_POLARIS_SOURCE_TREE_SHA256}" \
    "${POLARIS_SOURCE_APPROVAL}" \
    "${POLARIS_OPENPI_RUNTIME_DIR}" \
    "${POLARIS_COMMIT}" "${HOME}" \
    "${POLARIS_RUNTIME_CLOSURE_APPROVAL}" \
    "${POLARIS_RUNTIME_CLOSURE_APPROVAL_SHA256}" \
    "${POLARIS_SACCT_RUNTIME_APPROVAL}" \
    "${POLARIS_SACCT_RUNTIME_APPROVAL_SHA256}" \
    "${POLARIS_SACCT_PRELAUNCH_VALIDATION_RECEIPT}" \
    "${POLARIS_SACCT_PRELAUNCH_VALIDATION_RECEIPT_SHA256}" \
    "${POLARIS_EXPECTED_SLURM_CONFIG_SHA256}" \
    "${POLARIS_EXPECTED_SLURM_CONFIG_SIZE}" \
    "${POLARIS_EXPECTED_SCONTROL_SHA256}" \
    "${POLARIS_EXPECTED_SCONTROL_SIZE}" \
    "${POLARIS_EXPECTED_SLURM_LIBRARY_SHA256}" \
    "${POLARIS_EXPECTED_SLURM_LIBRARY_SIZE}" \
    "${POLARIS_EXPECTED_SACCT_SHA256}" \
    "${POLARIS_EXPECTED_SACCT_SIZE}" \
    "${POLARIS_EXPECTED_SCANCEL_SHA256}" \
    "${POLARIS_EXPECTED_SCANCEL_SIZE}" \
    "${POLARIS_EXPECTED_SRUN_SHA256}" \
    "${POLARIS_EXPECTED_SRUN_SIZE}" \
    "${APPROVED_SBATCH_SCRIPT}" <<'PY'
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import sys

(
    provenance_text,
    expected_argv_sha256,
    expected_batch_sha256,
    expected_provenance_sha256,
    expected_scheduler_sha256,
    expected_job_id,
    output_root_text,
    run_namespace,
    source_snapshot,
    source_tree_sha256,
    source_approval,
    openpi_runtime,
    polaris_commit,
    home,
    runtime_approval,
    runtime_approval_sha256,
    sacct_runtime_approval,
    sacct_runtime_approval_sha256,
    sacct_prelaunch_validation_receipt,
    sacct_prelaunch_validation_receipt_sha256,
    slurm_config_sha256,
    slurm_config_size,
    scontrol_sha256,
    scontrol_size,
    slurm_library_sha256,
    slurm_library_size,
    sacct_sha256,
    sacct_size,
    scancel_sha256,
    scancel_size,
    srun_sha256,
    srun_size,
    approved_batch_script,
) = sys.argv[1:]
for digest in (
    expected_argv_sha256,
    expected_batch_sha256,
    expected_scheduler_sha256,
    runtime_approval_sha256,
    sacct_runtime_approval_sha256,
    sacct_prelaunch_validation_receipt_sha256,
    source_tree_sha256,
    slurm_config_sha256,
    scontrol_sha256,
    slurm_library_sha256,
    sacct_sha256,
    scancel_sha256,
    srun_sha256,
):
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("runtime provenance expected digest is malformed")
if re.fullmatch(r"[0-9a-f]{40}", polaris_commit) is None:
    raise ValueError("runtime provenance PolaRiS commit is malformed")
if not expected_job_id.isdecimal() or int(expected_job_id) <= 0:
    raise ValueError("runtime provenance Slurm job ID is malformed")
if expected_provenance_sha256 != "-" and re.fullmatch(
    r"[0-9a-f]{64}", expected_provenance_sha256
) is None:
    raise ValueError("runtime provenance digest is malformed")
if any(
    re.fullmatch(r"[1-9][0-9]*", size) is None
    for size in (
        slurm_config_size, scontrol_size, slurm_library_size, sacct_size,
        scancel_size, srun_size,
    )
):
    raise ValueError("runtime provenance expected size is malformed")

provenance_dir = Path(provenance_text)
output_root = Path(output_root_text).resolve(strict=True)
if (
    not provenance_dir.is_absolute()
    or PurePosixPath(provenance_text).as_posix() != provenance_text
    or provenance_dir.is_symlink()
    or provenance_dir.resolve(strict=True) != provenance_dir
    or re.fullmatch(r"[A-Za-z0-9._-]+", run_namespace) is None
):
    raise ValueError("runtime provenance path is not canonical")
directory_fd = os.open(
    provenance_dir,
    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
)


def stable_file(name):
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_nlink != 1
        ):
            raise ValueError(f"unsafe runtime provenance file: {name}")
        payload = b""
        offset = 0
        while offset < before.st_size:
            block = os.pread(descriptor, before.st_size - offset, offset)
            if not block:
                raise ValueError(f"short runtime provenance read: {name}")
            payload += block
            offset += len(block)
        after = os.fstat(descriptor)
        relative = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        absolute = os.stat(provenance_dir / name, follow_symlinks=False)
        fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink", "st_size")
        if any(getattr(before, field) != getattr(after, field) for field in fields):
            raise ValueError(f"runtime provenance file changed: {name}")
        if any(getattr(after, field) != getattr(relative, field) for field in fields):
            raise ValueError(f"runtime provenance relative binding changed: {name}")
        if any(getattr(after, field) != getattr(absolute, field) for field in fields):
            raise ValueError(f"runtime provenance absolute binding changed: {name}")
        return payload
    finally:
        os.close(descriptor)


try:
    parent_before = os.fstat(directory_fd)
    approval_payload = stable_file("app_runtime_approval.env")
    batch_payload = stable_file("batch_script.sbatch")
    argv_payload = stable_file("submission_argv.sh")
    scheduler_payload = stable_file("scheduler_held.json")
    parent_after = os.fstat(directory_fd)
    parent_public = os.stat(provenance_dir, follow_symlinks=False)
    if (
        (parent_before.st_dev, parent_before.st_ino)
        != (parent_after.st_dev, parent_after.st_ino)
        or (parent_after.st_dev, parent_after.st_ino)
        != (parent_public.st_dev, parent_public.st_ino)
    ):
        raise ValueError("runtime provenance directory binding changed")
finally:
    os.close(directory_fd)

approval_sha256 = hashlib.sha256(approval_payload).hexdigest()
if expected_provenance_sha256 != "-" and approval_sha256 != expected_provenance_sha256:
    raise ValueError("runtime provenance approval digest mismatch")
if hashlib.sha256(batch_payload).hexdigest() != expected_batch_sha256:
    raise ValueError("runtime provenance batch script digest mismatch")
if hashlib.sha256(argv_payload).hexdigest() != expected_argv_sha256:
    raise ValueError("runtime provenance submission argv digest mismatch")
if hashlib.sha256(scheduler_payload).hexdigest() != expected_scheduler_sha256:
    raise ValueError("runtime provenance held scheduler digest mismatch")

try:
    lines = approval_payload.decode("utf-8", errors="strict").splitlines()
except UnicodeDecodeError as error:
    raise ValueError("runtime provenance approval is not UTF-8") from error
if not approval_payload.endswith(b"\n"):
    raise ValueError("runtime provenance approval is not newline terminated")
fields = {}
for line in lines:
    key, separator, value = line.partition("=")
    if not separator or key in fields or re.fullmatch(r"[a-z][a-z0-9_]*", key) is None:
        raise ValueError("runtime provenance approval record is malformed")
    fields[key] = value
expected_fields = {
    "profile": "polaris_app_launcher_runtime_approval_v6",
    "output_root": str(output_root),
    "output_namespace_parent": f"{output_root}/{run_namespace}",
    "runtime_closure_approval": runtime_approval,
    "runtime_closure_approval_sha256": runtime_approval_sha256,
    "sacct_runtime_approval": sacct_runtime_approval,
    "sacct_runtime_approval_sha256": sacct_runtime_approval_sha256,
    "sacct_prelaunch_validation_receipt": sacct_prelaunch_validation_receipt,
    "sacct_prelaunch_validation_receipt_sha256": (
        sacct_prelaunch_validation_receipt_sha256
    ),
    "scheduler_query_profile": "polaris_app_launcher_sacct_query_v1",
    "scheduler_query_path": "/usr/bin:/bin",
    "scheduler_query_slurm_conf": (
        "/cm/shared/apps/slurm/etc/oci-ord-cs-004/slurm.conf"
    ),
    "scheduler_query_ld_library_path": (
        "/cm/local/apps/slurm/24.11/lib64:"
        "/cm/local/apps/slurm/24.11/lib64/slurm"
    ),
    "scheduler_query_timeout_seconds": "10",
    "expected_slurm_config_path": (
        "/cm/shared/apps/slurm/etc/oci-ord-cs-004/slurm.conf"
    ),
    "expected_slurm_config_sha256": slurm_config_sha256,
    "expected_slurm_config_size": slurm_config_size,
    "expected_scontrol_sha256": scontrol_sha256,
    "expected_scontrol_size": scontrol_size,
    "expected_slurm_library_path": (
        "/cm/local/apps/slurm/24.11/lib64/slurm/libslurmfull.so"
    ),
    "expected_slurm_library_sha256": slurm_library_sha256,
    "expected_slurm_library_size": slurm_library_size,
    "expected_sacct_path": "/cm/local/apps/slurm/24.11/bin/sacct",
    "expected_sacct_sha256": sacct_sha256,
    "expected_sacct_size": sacct_size,
    "expected_scancel_sha256": scancel_sha256,
    "expected_scancel_size": scancel_size,
    "expected_srun_sha256": srun_sha256,
    "expected_srun_size": srun_size,
    "approved_batch_script": approved_batch_script,
    "batch_script_sha256": expected_batch_sha256,
    "submission_argv_sha256": expected_argv_sha256,
    "held_scheduler_record_sha256": expected_scheduler_sha256,
}
if set(fields) != set(expected_fields) | {"output_namespace_parent_identity"}:
    raise ValueError("runtime provenance approval schema mismatch")
for key, value in expected_fields.items():
    if fields[key] != value:
        raise ValueError(f"runtime provenance approval value mismatch: {key}")
external_approval = Path(sacct_runtime_approval)
if (
    not external_approval.is_absolute()
    or PurePosixPath(sacct_runtime_approval).as_posix() != sacct_runtime_approval
    or external_approval.is_symlink()
    or external_approval.resolve(strict=True) != external_approval
):
    raise ValueError("external sacct runtime approval path is not canonical")
external_fd = os.open(
    external_approval, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
)
try:
    external_before = os.fstat(external_fd)
    if (
        not stat.S_ISREG(external_before.st_mode)
        or stat.S_IMODE(external_before.st_mode) != 0o444
        or external_before.st_nlink != 1
    ):
        raise ValueError("external sacct runtime approval metadata mismatch")
    external_payload = bytearray()
    while block := os.read(external_fd, 1024 * 1024):
        external_payload.extend(block)
    external_after = os.fstat(external_fd)
finally:
    os.close(external_fd)
external_current = os.stat(external_approval, follow_symlinks=False)
external_fields = (
    "st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink",
    "st_size", "st_mtime_ns", "st_ctime_ns",
)
if (
    any(
        getattr(external_before, name)
        != getattr(external_after, name)
        or getattr(external_after, name)
        != getattr(external_current, name)
        for name in external_fields
    )
    or hashlib.sha256(external_payload).hexdigest()
    != sacct_runtime_approval_sha256
):
    raise ValueError("external sacct runtime approval identity changed")
prelaunch_receipt = Path(sacct_prelaunch_validation_receipt)
if (
    not prelaunch_receipt.is_absolute()
    or PurePosixPath(sacct_prelaunch_validation_receipt).as_posix()
    != sacct_prelaunch_validation_receipt
    or prelaunch_receipt.is_symlink()
    or prelaunch_receipt.resolve(strict=True) != prelaunch_receipt
):
    raise ValueError("sacct prelaunch validation receipt path is not canonical")
receipt_fd = os.open(
    prelaunch_receipt, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
)
try:
    receipt_before = os.fstat(receipt_fd)
    if (
        not stat.S_ISREG(receipt_before.st_mode)
        or stat.S_IMODE(receipt_before.st_mode) != 0o444
        or receipt_before.st_nlink != 1
    ):
        raise ValueError("sacct prelaunch validation receipt metadata mismatch")
    receipt_payload = bytearray()
    while block := os.read(receipt_fd, 1024 * 1024):
        receipt_payload.extend(block)
    receipt_after = os.fstat(receipt_fd)
finally:
    os.close(receipt_fd)
receipt_current = os.stat(prelaunch_receipt, follow_symlinks=False)
if (
    any(
        getattr(receipt_before, name)
        != getattr(receipt_after, name)
        or getattr(receipt_after, name)
        != getattr(receipt_current, name)
        for name in external_fields
    )
    or hashlib.sha256(receipt_payload).hexdigest()
    != sacct_prelaunch_validation_receipt_sha256
):
    raise ValueError("sacct prelaunch validation receipt identity changed")
try:
    receipt_value = json.loads(receipt_payload)
except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError("sacct prelaunch validation receipt is not JSON") from error
canonical_receipt = (
    json.dumps(
        receipt_value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    + b"\n"
)
if (
    canonical_receipt != receipt_payload
    or not isinstance(receipt_value, dict)
    or receipt_value.get("profile")
    != "polaris_external_sacct_runtime_prelaunch_validation_receipt_v2"
    or receipt_value.get("status") != "full_live_validation_passed_before_sbatch"
    or receipt_value.get("approval", {}).get("path") != sacct_runtime_approval
    or receipt_value.get("approval", {}).get("sha256")
    != sacct_runtime_approval_sha256
):
    raise ValueError("sacct prelaunch validation receipt binding mismatch")
identity = fields["output_namespace_parent_identity"]
if re.fullmatch(r"[0-9]+:[0-9]+:[0-9]+:[0-9]+:0755", identity) is None:
    raise ValueError("runtime provenance namespace identity is malformed")

namespace = Path(fields["output_namespace_parent"])
namespace_parent_fd = os.open(
    namespace.parent,
    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
)
namespace_fd = -1
try:
    namespace_fd = os.open(
        namespace.name,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=namespace_parent_fd,
    )
    metadata = os.fstat(namespace_fd)
    observed = os.stat(namespace.name, dir_fd=namespace_parent_fd, follow_symlinks=False)
    public = os.stat(namespace, follow_symlinks=False)
    actual_identity = ":".join((
        str(metadata.st_dev), str(metadata.st_ino), str(metadata.st_uid),
        str(metadata.st_gid), f"{stat.S_IMODE(metadata.st_mode):04o}",
    ))
    if (
        actual_identity != identity
        or stat.S_IMODE(metadata.st_mode) != 0o755
        or (metadata.st_dev, metadata.st_ino) != (observed.st_dev, observed.st_ino)
        or (metadata.st_dev, metadata.st_ino) != (public.st_dev, public.st_ino)
    ):
        raise ValueError("runtime provenance namespace binding changed")
finally:
    if namespace_fd >= 0:
        os.close(namespace_fd)
    os.close(namespace_parent_fd)

argv_text = argv_payload.decode("utf-8", errors="strict")
tokens = shlex.split(argv_text)
if (
    len(tokens) != 10
    or tokens[0] != "/cm/local/apps/slurm/24.11/bin/sbatch"
    or tokens[1:4] != ["--parsable", "--hold", "--no-requeue"]
    or re.fullmatch(r"--comment=pi05-[0-9a-f]{40}", tokens[4]) is None
    or tokens[5] != "--job-name=pi05-app-launcher_FoodBussing"
    or re.fullmatch(r"--time=[0-9]{2}:[0-9]{2}:[0-9]{2}", tokens[6]) is None
    or not tokens[7].startswith("--output=/")
    or not tokens[7].endswith("/%x-%j.out")
    or tokens[9] != approved_batch_script
):
    raise ValueError("runtime provenance submission argv shape mismatch")
export_tokens = [token for token in tokens if token.startswith("--export=")]
if len(export_tokens) != 1 or export_tokens[0] != tokens[8]:
    raise ValueError("runtime provenance submission argv shape mismatch")
exports = {}
for item in export_tokens[0].removeprefix("--export=").split(","):
    key, separator, value = item.partition("=")
    if not separator or key in exports:
        raise ValueError("runtime provenance submission export is malformed")
    exports[key] = value
required_exports = {
    "PATH": "/cm/local/apps/slurm/24.11/bin:/usr/bin:/bin",
    "HOME": home,
    "POLARIS_SOURCE_SNAPSHOT": source_snapshot,
    "EXPECTED_POLARIS_SOURCE_TREE_SHA256": source_tree_sha256,
    "POLARIS_SOURCE_APPROVAL": source_approval,
    "POLARIS_OPENPI_RUNTIME_DIR": openpi_runtime,
    "EXPECTED_POLARIS_COMMIT": polaris_commit,
    "POLARIS_ENVIRONMENT": "DROID-FoodBussing",
    "ROLLOUTS": "1",
    "ENVIRONMENT_SEED": "0",
    "RUN_NAMESPACE": run_namespace,
    "SUBMISSION_TRANSACTION_ID": tokens[4].removeprefix("--comment="),
    "POLARIS_EVAL_MODE": "app_launcher_only",
    "POLARIS_RUNTIME_CLOSURE_APPROVAL": runtime_approval,
    "POLARIS_RUNTIME_CLOSURE_APPROVAL_SHA256": runtime_approval_sha256,
    "POLARIS_EXPECTED_SLURM_CONFIG_SHA256": slurm_config_sha256,
    "POLARIS_EXPECTED_SCONTROL_SHA256": scontrol_sha256,
    "POLARIS_EXPECTED_SCONTROL_SIZE": scontrol_size,
    "POLARIS_EXPECTED_SLURM_LIBRARY_SHA256": slurm_library_sha256,
    "POLARIS_EXPECTED_SLURM_LIBRARY_SIZE": slurm_library_size,
    "POLARIS_EXPECTED_SACCT_SHA256": sacct_sha256,
    "POLARIS_EXPECTED_SACCT_SIZE": sacct_size,
    "POLARIS_EXPECTED_SCANCEL_SHA256": scancel_sha256,
    "POLARIS_EXPECTED_SCANCEL_SIZE": scancel_size,
    "POLARIS_EXPECTED_SRUN_SHA256": srun_sha256,
    "POLARIS_EXPECTED_SRUN_SIZE": srun_size,
    "POLARIS_OUTPUT_NAMESPACE_PARENT_IDENTITY": identity,
    "OUTPUT_ROOT": str(output_root),
}
if exports != required_exports:
    raise ValueError("runtime provenance submission approval export mismatch")
if {"BASH_ENV", "ENV", "LD_AUDIT", "LD_PRELOAD", "PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE"} & exports.keys():
    raise ValueError("runtime provenance submission exported an unsafe variable")
scheduler = json.loads(scheduler_payload.decode("ascii"))
if (
    scheduler_payload
    != (json.dumps(scheduler, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "ascii"
    )
    or set(scheduler) != {"schema_version", "profile", "status", "command", "job"}
    or scheduler["schema_version"] != 1
    or scheduler["profile"] != "openpi_pi05_droid_jointpos_scheduler_job_v1"
    or scheduler["status"] != "held_requeue_disabled_restart_count_zero"
    or scheduler["command"]
    != ["scontrol", "show", "job", expected_job_id, "--oneliner"]
    or not isinstance(scheduler["job"], dict)
    or scheduler["job"].get("job_id") != int(expected_job_id)
    or scheduler["job"].get("phase") != "held"
    or scheduler["job"].get("state") != "PENDING"
    or scheduler["job"].get("reason") != "JobHeldUser"
    or scheduler["job"].get("requeue") != 0
    or scheduler["job"].get("restarts") != 0
    or scheduler["job"].get("transaction_id")
    != tokens[4].removeprefix("--comment=")
):
    raise ValueError("runtime provenance held scheduler contract mismatch")
print(f"{output_root}\t{namespace}\t{identity}\t{approval_sha256}")
PY
}

write_transaction_state() {
  local transaction_dir="$1"
  local state="$2"
  write_atomic_text "${transaction_dir}/state" 0600 "${state}"
}

discover_transaction_job_ids() {
  local transaction_dir="$1"
  local transaction_id="$2"
  local candidate candidate_file job_id comment queue_output
  local queue_succeeded=0
  local -a known_ids=()
  local -A seen=()

  DISCOVERED_JOB_IDS=()
  candidate="${ACTIVE_JOB_ID:-}"
  if [[ "${candidate}" =~ ^[0-9]+$ && -z "${seen[${candidate}]:-}" ]]; then
    known_ids+=("${candidate}")
    seen["${candidate}"]=1
  fi
  for candidate_file in "${transaction_dir}/job_id" "${transaction_dir}/sbatch.stdout"; do
    [[ -f "${candidate_file}" ]] || continue
    while IFS= read -r candidate; do
      if [[ "${candidate}" =~ ^[0-9]+$ && -z "${seen[${candidate}]:-}" ]]; then
        known_ids+=("${candidate}")
        seen["${candidate}"]=1
      fi
    done < "${candidate_file}"
  done

  if queue_output="$(
    "${SQUEUE_COMMAND}" --noheader \
      --user="${SUBMIT_USER}" --format='%i|%.256k' 2>/dev/null
  )"; then
    queue_succeeded=1
    while IFS='|' read -r job_id comment; do
      [[ "${job_id}" =~ ^[0-9]+$ ]] || continue
      comment="${comment#"${comment%%[![:space:]]*}"}"
      comment="${comment%"${comment##*[![:space:]]}"}"
      if [[ "${comment}" == "${transaction_id}" && -z "${seen[${job_id}]:-}" ]]; then
        known_ids+=("${job_id}")
        seen["${job_id}"]=1
      fi
    done <<< "${queue_output}"
  fi

  DISCOVERED_JOB_IDS=("${known_ids[@]}")
  (( queue_succeeded == 1 || ${#known_ids[@]} > 0 ))
}

cancel_transaction() {
  local transaction_dir="$1"
  local transaction_id="$2"
  local job_id
  local -a recovered_job_ids=()

  if ! discover_transaction_job_ids "${transaction_dir}" "${transaction_id}"; then
    write_transaction_state "${transaction_dir}" cleanup_pending || true
    echo "Could not recover held job for transaction ${transaction_id}; recovery remains pending" >&2
    return 1
  fi
  recovered_job_ids=("${DISCOVERED_JOB_IDS[@]}")
  for job_id in "${recovered_job_ids[@]}"; do
    if ! "${SCANCEL_COMMAND}" "${job_id}"; then
      write_transaction_state "${transaction_dir}" cleanup_pending || true
      echo "Could not cancel job ${job_id} for transaction ${transaction_id}" >&2
      return 1
    fi
  done
  write_transaction_state "${transaction_dir}" canceled
  if (( ${#recovered_job_ids[@]} > 0 )); then
    echo "Canceled held transaction ${transaction_id}: jobs ${recovered_job_ids[*]}" >&2
  else
    echo "Closed transaction ${transaction_id}: no submitted job was found" >&2
  fi
}

cleanup_on_exit() {
  local status=$?
  trap - EXIT HUP INT TERM
  set +e
  if [[ -n "${ACTIVE_TRANSACTION_DIR:-}" && "${ACTIVE_TRANSACTION_RELEASED:-0}" != 1 ]]; then
    if ! cancel_transaction "${ACTIVE_TRANSACTION_DIR}" "${ACTIVE_TRANSACTION_ID}"; then
      status=5
    elif (( status == 0 )); then
      status=5
    fi
  fi
  exit "${status}"
}

recover_incomplete_transactions() {
  local state transaction_dir transaction_id
  local unresolved=0
  shopt -s nullglob
  for transaction_dir in "${TRANSACTION_ROOT}"/*; do
    if [[ ! -d "${transaction_dir}" || -L "${transaction_dir}" ]]; then
      echo "Invalid transaction entry: ${transaction_dir}" >&2
      unresolved=1
      continue
    fi
    state="$(tr -d '\r\n' < "${transaction_dir}/state" 2>/dev/null || true)"
    case "${state}" in
      canceled|released) continue ;;
    esac
    transaction_id="$(tr -d '\r\n' < "${transaction_dir}/transaction_id" 2>/dev/null || true)"
    if [[ ! "${transaction_id}" =~ ^pi05-[0-9a-f]{40}$ ]]; then
      echo "Invalid incomplete transaction metadata: ${transaction_dir}" >&2
      unresolved=1
      continue
    fi
    ACTIVE_JOB_ID=""
    if ! cancel_transaction "${transaction_dir}" "${transaction_id}"; then
      unresolved=1
    fi
  done
  shopt -u nullglob
  (( unresolved == 0 )) || return 1
}

capture_submission_provenance() {
  local provenance_dir="$1"
  local job_id="$2"
  local submission_argv="$3"
  local batch_script_path submission_argv_path held_scheduler_record_path
  local batch_temporary argv_temporary scheduler_temporary

  batch_script_path="${provenance_dir}/batch_script.sbatch"
  submission_argv_path="${provenance_dir}/submission_argv.sh"
  held_scheduler_record_path="${provenance_dir}/scheduler_held.json"
  batch_temporary="${provenance_dir}/.batch_script.sbatch.tmp.${BASHPID}.${RANDOM}"
  argv_temporary="${provenance_dir}/.submission_argv.sh.tmp.${BASHPID}.${RANDOM}"
  scheduler_temporary="${provenance_dir}/.scheduler_held.json.tmp.${BASHPID}.${RANDOM}"
  mkdir -p "$(dirname "${provenance_dir}")" || return
  mkdir -m 0755 "${provenance_dir}" || return
  "${SCONTROL_COMMAND}" write batch_script \
    "${job_id}" "${batch_temporary}" || return
  [[ -f "${batch_temporary}" && ! -L "${batch_temporary}" && -s "${batch_temporary}" ]] || return
  printf '%s\n' "${submission_argv}" > "${argv_temporary}" || return
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    PYTHONPATH="${POLARIS_SOURCE_SNAPSHOT}/src" \
    run_bounded_host_python \
    "${POLARIS_OPENPI_RUNTIME_DIR}/.venv/bin/python" -B -m \
    polaris.pi05_droid_jointpos_scheduler capture-job \
    --output "${scheduler_temporary}" \
    --phase held \
    --job-id "${job_id}" \
    --transaction-id "${ACTIVE_TRANSACTION_ID}" >/dev/null || return
  chmod 0444 "${batch_temporary}" "${argv_temporary}" \
    "${scheduler_temporary}" || return
  sync -- "${batch_temporary}" || return
  sync -- "${argv_temporary}" || return
  sync -- "${scheduler_temporary}" || return
  [[ ! -e "${batch_script_path}" && ! -e "${submission_argv_path}" \
    && ! -e "${held_scheduler_record_path}" ]] || return
  mv -- "${batch_temporary}" "${batch_script_path}" || return
  mv -- "${argv_temporary}" "${submission_argv_path}" || return
  mv -- "${scheduler_temporary}" "${held_scheduler_record_path}" || return
  sync -- "${provenance_dir}" || return
  batch_script_sha256="$(sha256sum "${batch_script_path}" | awk '{print $1}')" || return
  submission_argv_sha256="$(sha256sum "${submission_argv_path}" | awk '{print $1}')" || return
  held_scheduler_record_sha256="$(sha256sum "${held_scheduler_record_path}" | awk '{print $1}')" || return
}

MODE="${1:-}"
case "${MODE}" in
  app-launcher-only|canary|foodbussing50|full) ;;
  *) echo "Usage: $0 {app-launcher-only|canary|foodbussing50|full}" >&2; exit 2 ;;
esac
if [[ "${MODE}" == app-launcher-only ]]; then
  export PATH=/cm/local/apps/slurm/24.11/bin:/usr/bin:/bin
  SBATCH_COMMAND=/cm/local/apps/slurm/24.11/bin/sbatch
  SCONTROL_COMMAND=/cm/local/apps/slurm/24.11/bin/scontrol
  SQUEUE_COMMAND=/cm/local/apps/slurm/24.11/bin/squeue
  SCANCEL_COMMAND=/cm/local/apps/slurm/24.11/bin/scancel
else
  export PATH="${INHERITED_PATH}"
  SBATCH_COMMAND="$(command -v sbatch)"
  SCONTROL_COMMAND="$(command -v scontrol)"
  SQUEUE_COMMAND="$(command -v squeue)"
  SCANCEL_COMMAND="$(command -v scancel)"
fi

SCRIPT_DIR="$(readlink -f -- "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"
APPROVED_SBATCH_SCRIPT="${SCRIPT_DIR}/l40s_pi05_eval_job.sbatch"
APPROVED_SBATCH_SCRIPT_SHA256="19a403c89e934257bf33db428ba1a9346b7605b135d86c6a6515df4c4ff57787"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${APPROVED_SBATCH_SCRIPT}}"
: "${POLARIS_SOURCE_SNAPSHOT:?Set the approved content-addressed source snapshot}"
: "${EXPECTED_POLARIS_SOURCE_TREE_SHA256:?Set the approved source-snapshot tree SHA-256}"
: "${POLARIS_SOURCE_APPROVAL:?Set the immutable source approval JSON}"
: "${POLARIS_OPENPI_RUNTIME_DIR:?Set the adopted OpenPI bd70 runtime checkout}"
RUN_NAMESPACE="${RUN_NAMESPACE:-pi05-polaris-$(date -u +%Y%m%dT%H%M%SZ)}"
NFS_ROOT="${NFS_ROOT:-/lustre/fsw/portfolios/nvr/users/lzha}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${NFS_ROOT}/results/polaris-pi05}"
SBATCH_LOG_ROOT="${SBATCH_LOG_ROOT:-${NFS_ROOT}/slurm_logs/polaris-pi05/${RUN_NAMESPACE}}"
if [[ "${MODE}" == app-launcher-only ]]; then
  SUBMISSION_MANIFEST="${SUBMISSION_MANIFEST:-${NFS_ROOT}/results/polaris-pi05-submissions/${RUN_NAMESPACE}/${MODE}_jobs.tsv}"
else
  SUBMISSION_MANIFEST="${SUBMISSION_MANIFEST:-${NFS_ROOT}/results/polaris-pi05/${RUN_NAMESPACE}/${MODE}_jobs.tsv}"
fi
TRANSACTION_ROOT=""
POLARIS_COMMIT=c5b52a9cebb2c797a84e3df374b6002005d20a4f
ALLOW_RESUBMIT="${ALLOW_RESUBMIT:-0}"
ENVIRONMENT_SEED="${ENVIRONMENT_SEED:-0}"
SUBMIT_USER="${USER:-$(id -un)}"
ACTIVE_TRANSACTION_DIR=""
ACTIVE_TRANSACTION_ID=""
ACTIVE_TRANSACTION_RELEASED=0
ACTIVE_JOB_ID=""
DISCOVERED_JOB_IDS=()
OUTPUT_NAMESPACE_PARENT=""
POLARIS_OUTPUT_NAMESPACE_PARENT_IDENTITY=""
APP_RUNTIME_APPROVAL_RECORD=""
POLARIS_SACCT_PRELAUNCH_VALIDATION_RECEIPT=""
POLARIS_SACCT_PRELAUNCH_VALIDATION_RECEIPT_SHA256=""

trap cleanup_on_exit EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

for required_command in flock git readlink sbatch scancel scontrol squeue sha256sum sync; do
  command -v "${required_command}" >/dev/null || die "Missing required command: ${required_command}"
done
[[ -f "${SBATCH_SCRIPT}" && ! -L "${SBATCH_SCRIPT}" ]] \
  || die "Missing or symlinked sbatch script: ${SBATCH_SCRIPT}"
[[ "$(readlink -f -- "${SBATCH_SCRIPT}")" == "$(readlink -f -- "${APPROVED_SBATCH_SCRIPT}")" ]] \
  || die "SBATCH_SCRIPT override is forbidden for the approved canary"
observed_sbatch_script_sha256="$(sha256sum "${SBATCH_SCRIPT}" | awk '{print $1}')"
[[ "${observed_sbatch_script_sha256}" == "${APPROVED_SBATCH_SCRIPT_SHA256}" ]] \
  || die "Approved sbatch script SHA-256 mismatch"
[[ "${RUN_NAMESPACE}" =~ ^[A-Za-z0-9._-]+$ ]] \
  || die "RUN_NAMESPACE may contain only letters, digits, dot, underscore, and dash"
[[ "${ALLOW_RESUBMIT}" == 0 || "${ALLOW_RESUBMIT}" == 1 ]] \
  || die "ALLOW_RESUBMIT must be 0 or 1"
if [[ "${MODE}" == app-launcher-only ]]; then
  : "${POLARIS_RUNTIME_CLOSURE_APPROVAL:?Set the immutable runtime-closure approval path}"
  : "${POLARIS_RUNTIME_CLOSURE_APPROVAL_SHA256:?Set the runtime-closure approval SHA-256}"
  : "${POLARIS_SACCT_RUNTIME_APPROVAL:?Set the reviewed external sacct runtime approval path}"
  : "${POLARIS_SACCT_RUNTIME_APPROVAL_SHA256:?Set the reviewed external sacct runtime approval SHA-256}"
  [[ "${POLARIS_RUNTIME_CLOSURE_APPROVAL_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
    || die "POLARIS_RUNTIME_CLOSURE_APPROVAL_SHA256 must be lowercase 64hex"
  [[ "${POLARIS_SACCT_RUNTIME_APPROVAL_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
    || die "POLARIS_SACCT_RUNTIME_APPROVAL_SHA256 must be lowercase 64hex"
  [[ -f "${POLARIS_SACCT_RUNTIME_APPROVAL}" \
    && ! -L "${POLARIS_SACCT_RUNTIME_APPROVAL}" \
    && "$(stat -c '%a' "${POLARIS_SACCT_RUNTIME_APPROVAL}")" == 444 \
    && "$(stat -c '%h' "${POLARIS_SACCT_RUNTIME_APPROVAL}")" == 1 \
    && "$(sha256sum "${POLARIS_SACCT_RUNTIME_APPROVAL}" | awk '{print $1}')" \
      == "${POLARIS_SACCT_RUNTIME_APPROVAL_SHA256}" ]] \
    || die "Reviewed external sacct runtime approval identity mismatch"
fi
[[ -d "${POLARIS_SOURCE_SNAPSHOT}" && ! -L "${POLARIS_SOURCE_SNAPSHOT}" ]] \
  || die "PolaRiS source snapshot must be one real directory"
[[ -f "${POLARIS_SOURCE_APPROVAL}" && ! -L "${POLARIS_SOURCE_APPROVAL}" ]] \
  || die "PolaRiS source approval must be one real file"
[[ -d "${POLARIS_OPENPI_RUNTIME_DIR}" && ! -L "${POLARIS_OPENPI_RUNTIME_DIR}" ]] \
  || die "Adopted OpenPI runtime must be one real directory"
POLARIS_SOURCE_SNAPSHOT="$(readlink -f -- "${POLARIS_SOURCE_SNAPSHOT}")" \
  || die "Cannot canonicalize the PolaRiS source snapshot"
POLARIS_SOURCE_APPROVAL="$(readlink -f -- "${POLARIS_SOURCE_APPROVAL}")" \
  || die "Cannot canonicalize the PolaRiS source approval"
POLARIS_OPENPI_RUNTIME_DIR="$(readlink -f -- "${POLARIS_OPENPI_RUNTIME_DIR}")" \
  || die "Cannot canonicalize the adopted OpenPI runtime"
[[ -x "${POLARIS_OPENPI_RUNTIME_DIR}/.venv/bin/python" ]] \
  || die "Adopted OpenPI runtime interpreter is missing"
observed_openpi_commit="$(git -C "${POLARIS_OPENPI_RUNTIME_DIR}" rev-parse HEAD)" \
  || die "Cannot inspect the adopted OpenPI runtime checkout"
[[ "${observed_openpi_commit}" == bd70b8f4011e85b3f3b0f039f12113f78718e7bf ]] \
  || die "Adopted OpenPI runtime commit mismatch: ${observed_openpi_commit}"
openpi_status="$(
  git -C "${POLARIS_OPENPI_RUNTIME_DIR}" status --porcelain=v1 --untracked-files=all
)" || die "Cannot inspect adopted OpenPI runtime cleanliness"
[[ -z "${openpi_status}" ]] \
  || die "Adopted OpenPI runtime checkout is not clean"
observed_source_tree_sha256="$(
  /usr/bin/bash --noprofile --norc -p "${APPROVED_SBATCH_SCRIPT}" \
    --source-digest "${POLARIS_SOURCE_SNAPSHOT}"
)" || die "Cannot hash the PolaRiS source snapshot"
[[ "${observed_source_tree_sha256}" == "${EXPECTED_POLARIS_SOURCE_TREE_SHA256}" ]] \
  || die "PolaRiS source snapshot SHA-256 mismatch"
if [[ "${MODE}" == app-launcher-only ]]; then
  snapshot_imports=(
    polaris.app_launcher_startup_diagnostic
    polaris.config
    polaris.evaluation_seed
    polaris.pi05_droid_jointpos_scheduler
  )
else
  snapshot_imports=(
    polaris.pi05_droid_jointpos_consumer_binding
    polaris.policy.droid_jointpos_client
    openpi_client.image_tools
    openpi_client.websocket_client_policy
  )
fi
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  PYTHONPATH="${POLARIS_SOURCE_SNAPSHOT}/src:${POLARIS_SOURCE_SNAPSHOT}/third_party/openpi/packages/openpi-client/src" \
  run_bounded_host_python \
  "${POLARIS_OPENPI_RUNTIME_DIR}/.venv/bin/python" -B - \
  "${POLARIS_SOURCE_SNAPSHOT}" "${snapshot_imports[@]}" <<'PY' \
  || die "Approved source snapshot import/origin smoke failed"
import importlib
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
expected = {
    "polaris.app_launcher_startup_diagnostic": (
        "src/polaris/app_launcher_startup_diagnostic.py"
    ),
    "polaris.config": "src/polaris/config.py",
    "polaris.evaluation_seed": "src/polaris/evaluation_seed.py",
    "polaris.pi05_droid_jointpos_scheduler": (
        "src/polaris/pi05_droid_jointpos_scheduler.py"
    ),
    "polaris.pi05_droid_jointpos_consumer_binding": (
        "src/polaris/pi05_droid_jointpos_consumer_binding.py"
    ),
    "polaris.policy.droid_jointpos_client": (
        "src/polaris/policy/droid_jointpos_client.py"
    ),
    "openpi_client.image_tools": (
        "third_party/openpi/packages/openpi-client/src/openpi_client/image_tools.py"
    ),
    "openpi_client.websocket_client_policy": (
        "third_party/openpi/packages/openpi-client/src/openpi_client/websocket_client_policy.py"
    ),
}
for name in sys.argv[2:]:
    relative = expected[name]
    module = importlib.import_module(name)
    if Path(module.__file__).resolve(strict=True) != root / relative:
        raise SystemExit(f"snapshot import escaped approved source: {name}")
if "polaris.app_launcher_startup_diagnostic" in sys.argv[2:]:
    forbidden_prefixes = (
        "gym", "gymnasium", "isaaclab_tasks", "openpi", "openpi_client",
        "polaris.environments", "polaris.policy", "sentencepiece", "tokenizers",
        "transformers",
    )
    forbidden_loaded = sorted(
        name for name in sys.modules
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in forbidden_prefixes
        )
    )
    if forbidden_loaded:
        raise SystemExit(
            "AppLauncher-only source smoke imported forbidden modules: "
            + ",".join(forbidden_loaded)
        )
PY
if [[ "${MODE}" == app-launcher-only ]]; then
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    run_bounded_host_python \
    "${POLARIS_OPENPI_RUNTIME_DIR}/.venv/bin/python" -I -S -B - \
    "${POLARIS_SOURCE_SNAPSHOT}" <<'PY' \
    || die "Approved AppLauncher finalizer origin/compile smoke failed"
import os
from pathlib import Path
import stat
import sys

root = Path(sys.argv[1]).resolve(strict=True)
path = root / "scripts/polaris/finalize_pi05_app_launcher_only.py"
metadata = os.stat(path, follow_symlinks=False)
if (
    path.resolve(strict=True) != path
    or not stat.S_ISREG(metadata.st_mode)
    or metadata.st_nlink != 1
):
    raise SystemExit("AppLauncher finalizer escaped approved source")
payload = path.read_bytes()
compile(payload, str(path), "exec")
PY
fi
source_approval_result="$(
  /usr/bin/bash --noprofile --norc -p "${APPROVED_SBATCH_SCRIPT}" \
    --validate-source-approval "${POLARIS_SOURCE_APPROVAL}" \
    "${POLARIS_SOURCE_SNAPSHOT}" "${EXPECTED_POLARIS_SOURCE_TREE_SHA256}" \
    "${APPROVED_SBATCH_SCRIPT_SHA256}"
)" || die "Source approval validation failed"
IFS=$'\t' read -r SOURCE_APPROVAL_SHA256 POLARIS_IMPLEMENTATION_COMMIT \
  <<< "${source_approval_result}"
[[ "${SOURCE_APPROVAL_SHA256}" =~ ^[0-9a-f]{64}$ \
  && "${POLARIS_IMPLEMENTATION_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || die "Source approval result is invalid"

if [[ "${MODE}" == app-launcher-only ]]; then
  runtime_approval_result="$(
    /usr/bin/bash --noprofile --norc -p "${APPROVED_SBATCH_SCRIPT}" \
      --inspect-runtime-closure-approval \
      "${POLARIS_RUNTIME_CLOSURE_APPROVAL}" \
      "${POLARIS_RUNTIME_CLOSURE_APPROVAL_SHA256}"
  )" || die "Runtime-closure approval validation failed"
  IFS=$'\t' read -r observed_runtime_approval_sha256 \
    POLARIS_EXPECTED_SLURM_CONFIG_SHA256 POLARIS_EXPECTED_SLURM_CONFIG_SIZE \
    POLARIS_EXPECTED_SCONTROL_SHA256 POLARIS_EXPECTED_SCONTROL_SIZE \
    POLARIS_EXPECTED_SLURM_LIBRARY_SHA256 \
    POLARIS_EXPECTED_SLURM_LIBRARY_SIZE POLARIS_EXPECTED_SACCT_SHA256 \
    POLARIS_EXPECTED_SACCT_SIZE POLARIS_EXPECTED_SCANCEL_SHA256 \
    POLARIS_EXPECTED_SCANCEL_SIZE POLARIS_EXPECTED_SRUN_SHA256 \
    POLARIS_EXPECTED_SRUN_SIZE <<< "${runtime_approval_result}"
  [[ "${observed_runtime_approval_sha256}" == \
      "${POLARIS_RUNTIME_CLOSURE_APPROVAL_SHA256}" \
    && "${POLARIS_EXPECTED_SLURM_CONFIG_SHA256}" =~ ^[0-9a-f]{64}$ \
    && "${POLARIS_EXPECTED_SLURM_CONFIG_SIZE}" =~ ^[1-9][0-9]*$ \
    && "${POLARIS_EXPECTED_SCONTROL_SHA256}" =~ ^[0-9a-f]{64}$ \
    && "${POLARIS_EXPECTED_SCONTROL_SIZE}" =~ ^[1-9][0-9]*$ \
    && "${POLARIS_EXPECTED_SLURM_LIBRARY_SHA256}" =~ ^[0-9a-f]{64}$ \
    && "${POLARIS_EXPECTED_SLURM_LIBRARY_SIZE}" =~ ^[1-9][0-9]*$ \
    && "${POLARIS_EXPECTED_SACCT_SHA256}" =~ ^[0-9a-f]{64}$ \
    && "${POLARIS_EXPECTED_SACCT_SIZE}" =~ ^[1-9][0-9]*$ \
    && "${POLARIS_EXPECTED_SCANCEL_SHA256}" =~ ^[0-9a-f]{64}$ \
    && "${POLARIS_EXPECTED_SCANCEL_SIZE}" =~ ^[1-9][0-9]*$ \
    && "${POLARIS_EXPECTED_SRUN_SHA256}" =~ ^[0-9a-f]{64}$ \
    && "${POLARIS_EXPECTED_SRUN_SIZE}" =~ ^[1-9][0-9]*$ ]] \
    || die "Runtime-closure approval result is malformed"
  POLARIS_RUNTIME_CLOSURE_APPROVAL="$(
    /usr/bin/readlink -f -- "${POLARIS_RUNTIME_CLOSURE_APPROVAL}"
  )" || die "Cannot canonicalize the runtime-closure approval"
  POLARIS_SACCT_RUNTIME_APPROVAL="$(
    /usr/bin/readlink -f -- "${POLARIS_SACCT_RUNTIME_APPROVAL}"
  )" || die "Cannot canonicalize the external sacct runtime approval"
fi

if [[ "${MODE}" == app-launcher-only ]]; then
  tasks=(DROID-FoodBussing)
  rollouts="${ROLLOUTS:-1}"
  [[ "${rollouts}" == 1 ]] || die "app-launcher-only requires ROLLOUTS=1"
  [[ "${ENVIRONMENT_SEED}" == 0 ]] \
    || die "app-launcher-only requires ENVIRONMENT_SEED=0"
  time_limit="${SBATCH_TIME:-00:30:00}"
  job_prefix="pi05-app-launcher"
  worker_eval_mode=app_launcher_only
elif [[ "${MODE}" == canary ]]; then
  tasks=(DROID-FoodBussing)
  rollouts="${ROLLOUTS:-1}"
  time_limit="${SBATCH_TIME:-01:00:00}"
  job_prefix="pi05-canary"
  worker_eval_mode=standard
elif [[ "${MODE}" == foodbussing50 ]]; then
  tasks=(DROID-FoodBussing)
  rollouts="${ROLLOUTS:-50}"
  time_limit="${SBATCH_TIME:-03:50:00}"
  job_prefix="pi05-food50-seed${ENVIRONMENT_SEED}"
  worker_eval_mode=standard
else
  tasks=(
    DROID-BlockStackKitchen
    DROID-FoodBussing
    DROID-PanClean
    DROID-MoveLatteCup
    DROID-OrganizeTools
    DROID-TapeIntoContainer
  )
  rollouts="${ROLLOUTS:-50}"
  time_limit="${SBATCH_TIME:-03:50:00}"
  job_prefix="pi05-full50"
  worker_eval_mode=standard
fi

[[ "${rollouts}" =~ ^[1-9][0-9]*$ ]] || die "ROLLOUTS must be positive"
[[ "${ENVIRONMENT_SEED}" =~ ^(0|[1-9][0-9]*)$ ]] \
  || die "ENVIRONMENT_SEED must be a non-negative integer"
(( ENVIRONMENT_SEED <= 4294967295 )) \
  || die "ENVIRONMENT_SEED must be at most 4294967295"
manifest_name="$(basename "${SUBMISSION_MANIFEST}")"
mkdir -p "${SBATCH_LOG_ROOT}" "$(dirname "${SUBMISSION_MANIFEST}")"
SBATCH_LOG_ROOT="$(readlink -f -- "${SBATCH_LOG_ROOT}")" \
  || die "Cannot canonicalize the Slurm log root"
manifest_parent="$(readlink -f -- "$(dirname "${SUBMISSION_MANIFEST}")")" \
  || die "Cannot canonicalize the submission-manifest parent"
SUBMISSION_MANIFEST="${manifest_parent}/${manifest_name}"
if [[ "${worker_eval_mode}" == app_launcher_only ]]; then
  output_namespace_syntax="$(readlink -m -- "${OUTPUT_ROOT}/${RUN_NAMESPACE}")"
  [[ "${manifest_parent}" != "${output_namespace_syntax}" \
    && "${manifest_parent}" != "${output_namespace_syntax}/"* ]] \
    || die "AppLauncher submission registry must be outside its output namespace"
fi
TRANSACTION_ROOT="${SUBMISSION_MANIFEST}.transactions"
[[ ! -L "${TRANSACTION_ROOT}" ]] || die "Transaction root must not be a symlink"
mkdir -p "${TRANSACTION_ROOT}"
[[ -d "${TRANSACTION_ROOT}" && ! -L "${TRANSACTION_ROOT}" ]] \
  || die "Transaction root is not a real directory"
[[ ! -L "${SUBMISSION_MANIFEST}" ]] || die "Submission manifest must not be a symlink"
exec 9>"${SUBMISSION_MANIFEST}.lock"
flock -n 9 || { echo "Another submitter holds ${SUBMISSION_MANIFEST}.lock" >&2; exit 4; }
recover_incomplete_transactions \
  || { echo "Unresolved prior submission transaction; refusing new work" >&2; exit 5; }

standard_header=$'job_id\tmode\ttask\trollouts\tenvironment_seed\trun_namespace\tsource_tree_sha256\tsource_approval_sha256\timplementation_commit\topenpi_commit\tsubmitted_at\tbatch_script_sha256\tsubmission_argv_sha256\theld_scheduler_record_sha256\tprovenance_dir'
app_header="${standard_header}"$'\tapp_runtime_provenance_sha256'
if [[ "${worker_eval_mode}" == app_launcher_only ]]; then
  expected_header="${app_header}"
else
  expected_header="${standard_header}"
fi
if [[ ! -e "${SUBMISSION_MANIFEST}" ]]; then
  write_atomic_text "${SUBMISSION_MANIFEST}" 0644 "${expected_header}"
else
  [[ -f "${SUBMISSION_MANIFEST}" ]] || die "Submission manifest is not a regular file"
  [[ "$(head -n 1 "${SUBMISSION_MANIFEST}")" == "${expected_header}" ]] \
    || die "Submission manifest header mismatch"
fi

job_ids=()
for task in "${tasks[@]}"; do
  existing_row="$(
    awk -F '\t' -v mode="${MODE}" -v task="${task}" \
      '$2 == mode && $3 == task {row = $0} END {print row}' \
      "${SUBMISSION_MANIFEST}"
  )"
  existing_job_id=""
  if [[ -n "${existing_row}" ]]; then
    IFS=$'\t' read -r existing_job_id existing_mode existing_task \
      existing_rollouts existing_seed existing_namespace existing_source_sha256 \
      existing_source_approval_sha256 existing_implementation_commit \
      existing_openpi_commit _existing_time _existing_batch_sha256 \
      _existing_argv_sha256 _existing_scheduler_sha256 _existing_provenance \
      _existing_app_provenance_sha256 <<< "${existing_row}"
    if [[ "${existing_mode}" != "${MODE}" \
      || "${existing_task}" != "${task}" \
      || "${existing_rollouts}" != "${rollouts}" \
      || "${existing_seed}" != "${ENVIRONMENT_SEED}" \
      || "${existing_namespace}" != "${RUN_NAMESPACE}" \
      || "${existing_source_sha256}" != "${EXPECTED_POLARIS_SOURCE_TREE_SHA256}" \
      || "${existing_source_approval_sha256}" != "${SOURCE_APPROVAL_SHA256}" \
      || "${existing_implementation_commit}" != "${POLARIS_IMPLEMENTATION_COMMIT}" \
      || "${existing_openpi_commit}" != bd70b8f4011e85b3f3b0f039f12113f78718e7bf ]]; then
      die "Existing ${MODE}/${task} row has incompatible evaluation provenance"
    fi
    if [[ "${worker_eval_mode}" == app_launcher_only ]]; then
      POLARIS_SACCT_PRELAUNCH_VALIDATION_RECEIPT="$(
        awk -F '=' '$1 == "sacct_prelaunch_validation_receipt" {print substr($0, index($0, "=") + 1)}' \
          "${_existing_provenance}/app_runtime_approval.env"
      )" || die "Cannot read existing prelaunch validation receipt path"
      POLARIS_SACCT_PRELAUNCH_VALIDATION_RECEIPT_SHA256="$(
        awk -F '=' '$1 == "sacct_prelaunch_validation_receipt_sha256" {print $2}' \
          "${_existing_provenance}/app_runtime_approval.env"
      )" || die "Cannot read existing prelaunch validation receipt digest"
      existing_app_context="$(
        validate_app_runtime_provenance \
          "${_existing_provenance}" "${_existing_argv_sha256}" \
          "${_existing_app_provenance_sha256}" \
          "${_existing_scheduler_sha256}" "${existing_job_id}"
      )" || die "Existing ${MODE}/${task} row has incompatible AppLauncher runtime approval"
      IFS=$'\t' read -r OUTPUT_ROOT OUTPUT_NAMESPACE_PARENT \
        POLARIS_OUTPUT_NAMESPACE_PARENT_IDENTITY observed_app_provenance_sha256 \
        <<< "${existing_app_context}"
      [[ "${observed_app_provenance_sha256}" == \
        "${_existing_app_provenance_sha256}" ]] \
        || die "Existing AppLauncher provenance digest changed"
      build_app_runtime_approval_record
    fi
  fi
  if [[ -n "${existing_job_id}" && "${ALLOW_RESUBMIT}" != 1 ]]; then
    echo "Existing ${MODE} attempt for ${task}: job ${existing_job_id}; set ALLOW_RESUBMIT=1 for an explicit retry"
    job_ids+=("${existing_job_id}")
    continue
  fi

  if [[ "${worker_eval_mode}" == app_launcher_only \
    && -z "${existing_job_id}" ]]; then
    namespace_result="$(
      prepare_app_output_namespace "${OUTPUT_ROOT}" "${RUN_NAMESPACE}"
    )" || die "Cannot exclusively create the AppLauncher output namespace"
    IFS=$'\t' read -r OUTPUT_ROOT OUTPUT_NAMESPACE_PARENT \
      POLARIS_OUTPUT_NAMESPACE_PARENT_IDENTITY <<< "${namespace_result}"
    [[ "${OUTPUT_NAMESPACE_PARENT}" == "${OUTPUT_ROOT}/${RUN_NAMESPACE}" \
      && "${POLARIS_OUTPUT_NAMESPACE_PARENT_IDENTITY}" =~ \
        ^[0-9]+:[0-9]+:[0-9]+:[0-9]+:0755$ ]] \
      || die "AppLauncher output namespace identity is malformed"
    build_app_runtime_approval_record
  fi

  short_task="${task#DROID-}"
  job_name="${job_prefix}_${short_task}"
  transaction_seed="${RUN_NAMESPACE}|${MODE}|${task}|${BASHPID}|$(date +%s%N)|${RANDOM}|${RANDOM}"
  transaction_digest="$(printf '%s' "${transaction_seed}" | sha256sum | awk '{print $1}')"
  ACTIVE_TRANSACTION_ID="pi05-${transaction_digest:0:40}"
  ACTIVE_TRANSACTION_DIR="${TRANSACTION_ROOT}/${ACTIVE_TRANSACTION_ID}"
  ACTIVE_TRANSACTION_RELEASED=0
  ACTIVE_JOB_ID=""
  mkdir -m 0700 "${ACTIVE_TRANSACTION_DIR}"
  write_atomic_text "${ACTIVE_TRANSACTION_DIR}/transaction_id" 0444 "${ACTIVE_TRANSACTION_ID}"
  if [[ "${worker_eval_mode}" == app_launcher_only ]]; then
    POLARIS_SACCT_PRELAUNCH_VALIDATION_RECEIPT="${ACTIVE_TRANSACTION_DIR}/sacct_runtime_prelaunch_validation.json"
    PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
      PYTHONPATH="${POLARIS_SOURCE_SNAPSHOT}/src" \
      run_bounded_host_python \
      "${POLARIS_OPENPI_RUNTIME_DIR}/.venv/bin/python" -B -m \
      polaris.pi05_droid_jointpos_scheduler validate-sacct-runtime-approval \
      --output "${POLARIS_SACCT_PRELAUNCH_VALIDATION_RECEIPT}" \
      --approval "${POLARIS_SACCT_RUNTIME_APPROVAL}" \
      --expected-approval-sha256 "${POLARIS_SACCT_RUNTIME_APPROVAL_SHA256}" \
      --source-approval "${POLARIS_SOURCE_APPROVAL}" >/dev/null \
      || die "Full live sacct approval validation failed before sbatch"
    POLARIS_SACCT_PRELAUNCH_VALIDATION_RECEIPT_SHA256="$(
      sha256sum "${POLARIS_SACCT_PRELAUNCH_VALIDATION_RECEIPT}" | awk '{print $1}'
    )" || die "Cannot hash the sacct prelaunch validation receipt"
    [[ "${POLARIS_SACCT_PRELAUNCH_VALIDATION_RECEIPT_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
      || die "Sacct prelaunch validation receipt digest is malformed"
    build_app_runtime_approval_record
  fi
  printf -v transaction_metadata \
    'mode=%s\ntask=%s\nrun_namespace=%s\npolaris_base_commit=%s\nimplementation_commit=%s\nsource_snapshot=%s\nsource_tree_sha256=%s\nsource_approval=%s\nsource_approval_sha256=%s\nopenpi_runtime=%s\nopenpi_commit=%s\nprepared_at=%s\n' \
    "${MODE}" "${task}" "${RUN_NAMESPACE}" "${POLARIS_COMMIT}" \
    "${POLARIS_IMPLEMENTATION_COMMIT}" "${POLARIS_SOURCE_SNAPSHOT}" \
    "${EXPECTED_POLARIS_SOURCE_TREE_SHA256}" "${POLARIS_SOURCE_APPROVAL}" \
    "${SOURCE_APPROVAL_SHA256}" \
    "${POLARIS_OPENPI_RUNTIME_DIR}" \
    bd70b8f4011e85b3f3b0f039f12113f78718e7bf "$(date -Iseconds)"
  transaction_metadata="${transaction_metadata%$'\n'}"
  if [[ "${worker_eval_mode}" == app_launcher_only ]]; then
    transaction_metadata+=$'\n'"${APP_RUNTIME_APPROVAL_RECORD}"
  fi
  write_atomic_text "${ACTIVE_TRANSACTION_DIR}/metadata" 0444 "${transaction_metadata}"
  write_transaction_state "${ACTIVE_TRANSACTION_DIR}" prepared

  export_vars="PATH=${PATH},HOME=${HOME},POLARIS_SOURCE_SNAPSHOT=${POLARIS_SOURCE_SNAPSHOT},EXPECTED_POLARIS_SOURCE_TREE_SHA256=${EXPECTED_POLARIS_SOURCE_TREE_SHA256},POLARIS_SOURCE_APPROVAL=${POLARIS_SOURCE_APPROVAL},POLARIS_OPENPI_RUNTIME_DIR=${POLARIS_OPENPI_RUNTIME_DIR},EXPECTED_POLARIS_COMMIT=${POLARIS_COMMIT},POLARIS_ENVIRONMENT=${task},ROLLOUTS=${rollouts},ENVIRONMENT_SEED=${ENVIRONMENT_SEED},RUN_NAMESPACE=${RUN_NAMESPACE},SUBMISSION_TRANSACTION_ID=${ACTIVE_TRANSACTION_ID}"
  if [[ "${worker_eval_mode}" == app_launcher_only ]]; then
    for slurm_export_value in \
      "${PATH}" "${HOME}" "${POLARIS_SOURCE_SNAPSHOT}" \
      "${POLARIS_SOURCE_APPROVAL}" "${POLARIS_OPENPI_RUNTIME_DIR}" \
      "${POLARIS_RUNTIME_CLOSURE_APPROVAL}" \
      "${POLARIS_SACCT_RUNTIME_APPROVAL}" "${OUTPUT_ROOT}"; do
      [[ "${slurm_export_value}" != *','* \
        && "${slurm_export_value}" != *$'\t'* \
        && "${slurm_export_value}" != *$'\r'* \
        && "${slurm_export_value}" != *$'\n'* ]] \
        || die "AppLauncher Slurm export value contains an unsafe delimiter"
    done
    export_vars+=",POLARIS_EVAL_MODE=app_launcher_only,POLARIS_RUNTIME_CLOSURE_APPROVAL=${POLARIS_RUNTIME_CLOSURE_APPROVAL},POLARIS_EXPECTED_SLURM_CONFIG_SHA256=${POLARIS_EXPECTED_SLURM_CONFIG_SHA256},POLARIS_EXPECTED_SCONTROL_SHA256=${POLARIS_EXPECTED_SCONTROL_SHA256},POLARIS_EXPECTED_SCONTROL_SIZE=${POLARIS_EXPECTED_SCONTROL_SIZE},POLARIS_EXPECTED_SLURM_LIBRARY_SHA256=${POLARIS_EXPECTED_SLURM_LIBRARY_SHA256},POLARIS_EXPECTED_SLURM_LIBRARY_SIZE=${POLARIS_EXPECTED_SLURM_LIBRARY_SIZE},POLARIS_EXPECTED_SACCT_SHA256=${POLARIS_EXPECTED_SACCT_SHA256},POLARIS_EXPECTED_SACCT_SIZE=${POLARIS_EXPECTED_SACCT_SIZE},POLARIS_EXPECTED_SCANCEL_SHA256=${POLARIS_EXPECTED_SCANCEL_SHA256},POLARIS_EXPECTED_SCANCEL_SIZE=${POLARIS_EXPECTED_SCANCEL_SIZE},POLARIS_EXPECTED_SRUN_SHA256=${POLARIS_EXPECTED_SRUN_SHA256},POLARIS_EXPECTED_SRUN_SIZE=${POLARIS_EXPECTED_SRUN_SIZE},POLARIS_RUNTIME_CLOSURE_APPROVAL_SHA256=${POLARIS_RUNTIME_CLOSURE_APPROVAL_SHA256},POLARIS_OUTPUT_NAMESPACE_PARENT_IDENTITY=${POLARIS_OUTPUT_NAMESPACE_PARENT_IDENTITY},OUTPUT_ROOT=${OUTPUT_ROOT}"
  fi
  sbatch_argv=("${SBATCH_COMMAND}" --parsable --hold --no-requeue \
    --comment="${ACTIVE_TRANSACTION_ID}" \
    --job-name="${job_name}" \
    --time="${time_limit}" \
    --output="${SBATCH_LOG_ROOT}/%x-%j.out" \
    --export="${export_vars}" \
    "${SBATCH_SCRIPT}")
  printf -v submission_argv '%q ' "${sbatch_argv[@]}"
  set +e
  "${sbatch_argv[@]}" > "${ACTIVE_TRANSACTION_DIR}/sbatch.stdout"
  sbatch_status=$?
  set -e
  chmod 0444 "${ACTIVE_TRANSACTION_DIR}/sbatch.stdout"
  sync -- "${ACTIVE_TRANSACTION_DIR}/sbatch.stdout"
  sync -- "${ACTIVE_TRANSACTION_DIR}"
  mapfile -t sbatch_lines < "${ACTIVE_TRANSACTION_DIR}/sbatch.stdout"
  if (( sbatch_status != 0 || ${#sbatch_lines[@]} != 1 )) \
    || [[ ! "${sbatch_lines[0]:-}" =~ ^[0-9]+$ ]]; then
    echo "sbatch failed or did not return exactly one numeric held job ID" >&2
    exit 3
  fi
  ACTIVE_JOB_ID="${sbatch_lines[0]}"
  write_atomic_text "${ACTIVE_TRANSACTION_DIR}/job_id" 0444 "${ACTIVE_JOB_ID}"
  write_transaction_state "${ACTIVE_TRANSACTION_DIR}" job_captured

  provenance_dir="$(dirname "${SUBMISSION_MANIFEST}")/submission_provenance/job_${ACTIVE_JOB_ID}"
  batch_script_sha256=""
  submission_argv_sha256=""
  held_scheduler_record_sha256=""
  if ! capture_submission_provenance \
    "${provenance_dir}" "${ACTIVE_JOB_ID}" "${submission_argv}"; then
    write_transaction_state "${ACTIVE_TRANSACTION_DIR}" provenance_failed || true
    echo "Failed to preserve submission provenance for held job ${ACTIVE_JOB_ID}" >&2
    exit 5
  fi
  if [[ "${batch_script_sha256}" != "${APPROVED_SBATCH_SCRIPT_SHA256}" ]]; then
    write_transaction_state "${ACTIVE_TRANSACTION_DIR}" provenance_failed || true
    echo "Slurm-spooled batch script differs from the approved script" >&2
    exit 5
  fi
  if [[ "${worker_eval_mode}" == app_launcher_only ]]; then
    final_app_runtime_record="${APP_RUNTIME_APPROVAL_RECORD}"$'\n'\
"approved_batch_script=${APPROVED_SBATCH_SCRIPT}"$'\n'\
"batch_script_sha256=${batch_script_sha256}"$'\n'\
"submission_argv_sha256=${submission_argv_sha256}"$'\n'\
"held_scheduler_record_sha256=${held_scheduler_record_sha256}"
    if ! write_atomic_text "${provenance_dir}/app_runtime_approval.env" 0444 \
      "${final_app_runtime_record}"; then
      write_transaction_state "${ACTIVE_TRANSACTION_DIR}" provenance_failed || true
      echo "AppLauncher runtime approval provenance is not durable" >&2
      exit 5
    fi
    app_context="$(
      validate_app_runtime_provenance \
        "${provenance_dir}" "${submission_argv_sha256}" - \
        "${held_scheduler_record_sha256}" "${ACTIVE_JOB_ID}"
    )" || {
      write_transaction_state "${ACTIVE_TRANSACTION_DIR}" provenance_failed || true
      echo "AppLauncher runtime approval provenance is not durable" >&2
      exit 5
    }
    IFS=$'\t' read -r validated_output_root validated_namespace \
      validated_namespace_identity app_runtime_provenance_sha256 \
      <<< "${app_context}"
    [[ "${validated_output_root}" == "${OUTPUT_ROOT}" \
      && "${validated_namespace}" == "${OUTPUT_NAMESPACE_PARENT}" \
      && "${validated_namespace_identity}" == \
        "${POLARIS_OUTPUT_NAMESPACE_PARENT_IDENTITY}" \
      && "${app_runtime_provenance_sha256}" =~ ^[0-9a-f]{64}$ ]] \
      || die "AppLauncher runtime provenance validation result is malformed"
  fi
  write_transaction_state "${ACTIVE_TRANSACTION_DIR}" provenance_durable

  if [[ "${worker_eval_mode}" == app_launcher_only ]]; then
    printf -v manifest_row '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
      "${ACTIVE_JOB_ID}" "${MODE}" "${task}" "${rollouts}" "${ENVIRONMENT_SEED}" \
      "${RUN_NAMESPACE}" "${EXPECTED_POLARIS_SOURCE_TREE_SHA256}" \
      "${SOURCE_APPROVAL_SHA256}" "${POLARIS_IMPLEMENTATION_COMMIT}" \
      bd70b8f4011e85b3f3b0f039f12113f78718e7bf \
      "$(date -Iseconds)" "${batch_script_sha256}" \
      "${submission_argv_sha256}" "${held_scheduler_record_sha256}" \
      "${provenance_dir}" \
      "${app_runtime_provenance_sha256}"
  else
    printf -v manifest_row '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
      "${ACTIVE_JOB_ID}" "${MODE}" "${task}" "${rollouts}" "${ENVIRONMENT_SEED}" \
      "${RUN_NAMESPACE}" "${EXPECTED_POLARIS_SOURCE_TREE_SHA256}" \
      "${SOURCE_APPROVAL_SHA256}" "${POLARIS_IMPLEMENTATION_COMMIT}" \
      bd70b8f4011e85b3f3b0f039f12113f78718e7bf \
      "$(date -Iseconds)" "${batch_script_sha256}" \
      "${submission_argv_sha256}" "${held_scheduler_record_sha256}" \
      "${provenance_dir}"
  fi
  append_manifest_row "${manifest_row}"
  write_transaction_state "${ACTIVE_TRANSACTION_DIR}" manifest_durable

  if [[ "${worker_eval_mode}" == app_launcher_only ]]; then
    PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
      PYTHONPATH="${POLARIS_SOURCE_SNAPSHOT}/src" \
      run_bounded_host_python \
      "${POLARIS_OPENPI_RUNTIME_DIR}/.venv/bin/python" -B -m \
      polaris.pi05_droid_jointpos_scheduler verify-sacct-prelaunch-validation \
      --receipt "${POLARIS_SACCT_PRELAUNCH_VALIDATION_RECEIPT}" \
      --expected-receipt-sha256 \
        "${POLARIS_SACCT_PRELAUNCH_VALIDATION_RECEIPT_SHA256}" \
      --approval "${POLARIS_SACCT_RUNTIME_APPROVAL}" \
      --expected-approval-sha256 \
        "${POLARIS_SACCT_RUNTIME_APPROVAL_SHA256}" >/dev/null \
      || die "Full live sacct approval validation failed before release"
  fi
  if ! "${SCONTROL_COMMAND}" release "${ACTIVE_JOB_ID}"; then
    write_transaction_state "${ACTIVE_TRANSACTION_DIR}" release_failed || true
    echo "Failed to release held job ${ACTIVE_JOB_ID}" >&2
    echo "Its durable manifest row is retained; retry requires ALLOW_RESUBMIT=1" >&2
    exit 5
  fi
  write_transaction_state "${ACTIVE_TRANSACTION_DIR}" released
  ACTIVE_TRANSACTION_RELEASED=1
  job_ids+=("${ACTIVE_JOB_ID}")
  printf '%s\n' "${manifest_row}"
  ACTIVE_TRANSACTION_DIR=""
  ACTIVE_TRANSACTION_ID=""
  ACTIVE_JOB_ID=""
  ACTIVE_TRANSACTION_RELEASED=0
done

printf 'submitted_job_ids=%s\n' "${job_ids[*]}"
printf 'submission_manifest=%s\n' "${SUBMISSION_MANIFEST}"
