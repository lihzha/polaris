#!/usr/bin/env bash
#SBATCH --account=nvr_lpr_rvp
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:50:00
#SBATCH --no-requeue
#SBATCH --job-name=pi05_polaris_eval
#SBATCH --output=/lustre/fs11/portfolios/nvr/projects/nvr_lpr_rvp/users/lzha/slurm_logs/polaris-pi05/%x-%j.out

set -Eeuo pipefail

trusted_source_tree_sha256() {
  /usr/bin/python3 -I -S - "$1" <<'PY'
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def same(left, right):
    return all(
        getattr(left, f"st_{field}") == getattr(right, f"st_{field}")
        for field in (
            "dev", "ino", "mode", "uid", "gid", "nlink", "size",
            "mtime_ns", "ctime_ns",
        )
    )


def open_directory(root_fd, parts):
    current = os.dup(root_fd)
    try:
        for part in parts:
            following = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=current,
            )
            os.close(current)
            current = following
        return current
    except BaseException:
        os.close(current)
        raise


def safe_parts(relative):
    path = PurePosixPath(relative)
    if (
        not relative or relative == "." or path.is_absolute()
        or path.as_posix() != relative
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe source path: {relative!r}")
    return path.parts


def inventory(path):
    files = set()
    directories = {"."}
    for current, names, filenames in os.walk(path, followlinks=False):
        current_path = Path(current)
        if not stat.S_ISDIR(os.stat(current_path, follow_symlinks=False).st_mode):
            raise ValueError(f"non-directory in source tree: {current_path}")
        directories.add(current_path.relative_to(path).as_posix() or ".")
        for name in names:
            child = current_path / name
            if not stat.S_ISDIR(os.stat(child, follow_symlinks=False).st_mode):
                raise ValueError(f"symlink/non-directory in source tree: {child}")
            directories.add(child.relative_to(path).as_posix())
        for name in filenames:
            child = current_path / name
            if not stat.S_ISREG(os.stat(child, follow_symlinks=False).st_mode):
                raise ValueError(f"symlink/non-file in source tree: {child}")
            files.add(child.relative_to(path).as_posix())
    return files, directories


root = Path(sys.argv[1])
if not root.is_absolute() or root.is_symlink():
    raise ValueError("source root must be absolute and non-symlinked")
filesystem_root = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
try:
    root_fd = open_directory(filesystem_root, root.parts[1:])
finally:
    os.close(filesystem_root)
try:
    root_identity = os.fstat(root_fd)
    files, directories = inventory(root)
    directory_records = []
    for relative in sorted(directories):
        descriptor = os.dup(root_fd) if relative == "." else open_directory(
            root_fd, safe_parts(relative)
        )
        try:
            opened = os.fstat(descriptor)
            observed = os.stat(
                "." if relative == "." else relative,
                dir_fd=root_fd,
                follow_symlinks=False,
            )
            if not same(opened, observed):
                raise ValueError(f"source directory changed: {relative}")
            directory_records.append({
                "relative_path": relative,
                "mode": f"{stat.S_IMODE(opened.st_mode):04o}",
            })
        finally:
            os.close(descriptor)
    file_records = []
    for relative in sorted(files):
        parts = safe_parts(relative)
        parent_fd = open_directory(root_fd, parts[:-1])
        try:
            descriptor = os.open(
                parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd
            )
        finally:
            os.close(parent_fd)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"source object is not regular: {relative}")
            digest = hashlib.sha256()
            size = before.st_size
            offset = 0
            while offset < size:
                block = os.pread(descriptor, min(16 * 1024 * 1024, size - offset), offset)
                if not block:
                    raise ValueError(f"short source object: {relative}")
                digest.update(block)
                offset += len(block)
            if os.pread(descriptor, 1, size):
                raise ValueError(f"growing source object: {relative}")
            after = os.fstat(descriptor)
            observed = os.stat(relative, dir_fd=root_fd, follow_symlinks=False)
            if not same(before, after) or not same(after, observed):
                raise ValueError(f"source file changed: {relative}")
            file_records.append({
                "relative_path": relative,
                "mode": f"{stat.S_IMODE(after.st_mode):04o}",
                "size": size,
                "sha256": digest.hexdigest(),
            })
        finally:
            os.close(descriptor)
    final_files, final_directories = inventory(root)
    if (
        files != final_files or directories != final_directories
        or not same(root_identity, os.fstat(root_fd))
        or not same(os.fstat(root_fd), os.stat(root, follow_symlinks=False))
    ):
        raise ValueError("source closure changed while hashing")
finally:
    os.close(root_fd)
print(hashlib.sha256(canonical({
    "directories": directory_records, "files": file_records,
})).hexdigest())
PY
}

trusted_source_approval() {
  /usr/bin/python3 -I -S - "$1" "$2" "$3" "$4" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

approval, expected_snapshot, expected_tree, expected_hasher = sys.argv[1:]
path = Path(approval)
before = os.stat(path, follow_symlinks=False)
if (
    not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o444
    or before.st_nlink != 1 or path.is_symlink()
):
    raise ValueError("source approval must be one mode-0444 regular link")
payload = path.read_bytes()
after = os.stat(path, follow_symlinks=False)
identity = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
if any(getattr(before, field) != getattr(after, field) for field in identity):
    raise ValueError("source approval changed while reading")
value = json.loads(payload, parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
required = {
    "schema_version", "profile", "snapshot_path", "source_tree_sha256",
    "implementation_commit", "polaris_base_commit", "polaris_base_tree",
    "openpi_commit", "trusted_hasher_sha256",
}
if set(value) != required:
    raise ValueError("source approval schema mismatch")
if (
    value["schema_version"] != 1
    or value["profile"] != "openpi_pi05_droid_jointpos_source_approval_v1"
    or value["snapshot_path"] != expected_snapshot
    or value["source_tree_sha256"] != expected_tree
    or value["polaris_base_commit"] != "c5b52a9cebb2c797a84e3df374b6002005d20a4f"
    or value["polaris_base_tree"] != "7fd5e1b0af26577fd323fb1d7f3595b91282e73f"
    or value["openpi_commit"] != "bd70b8f4011e85b3f3b0f039f12113f78718e7bf"
    or value["trusted_hasher_sha256"] != expected_hasher
    or not isinstance(value["implementation_commit"], str)
    or len(value["implementation_commit"]) != 40
    or any(character not in "0123456789abcdef" for character in value["implementation_commit"])
):
    raise ValueError("source approval identity mismatch")
canonical = json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    allow_nan=False,
).encode("ascii") + b"\n"
if payload != canonical:
    raise ValueError("source approval is not canonical JSON")
print(f"{hashlib.sha256(payload).hexdigest()}\t{value['implementation_commit']}")
PY
}

if [[ "${1:-}" == --source-digest ]]; then
  (( $# == 2 )) || { echo "Usage: $0 --source-digest ABSOLUTE_SNAPSHOT" >&2; exit 2; }
  trusted_source_tree_sha256 "$2"
  exit
fi
if [[ "${1:-}" == --validate-source-approval ]]; then
  (( $# == 5 )) || { echo "Usage: $0 --validate-source-approval FILE SNAPSHOT TREE HASHER_SHA" >&2; exit 2; }
  trusted_source_approval "$2" "$3" "$4" "$5"
  exit
fi

NFS_ROOT="${NFS_ROOT:-/lustre/fsw/portfolios/nvr/users/lzha}"
: "${POLARIS_SOURCE_SNAPSHOT:?Set the approved content-addressed PolaRiS source snapshot}"
: "${EXPECTED_POLARIS_SOURCE_TREE_SHA256:?Set the approved source-snapshot tree SHA-256}"
: "${POLARIS_OPENPI_RUNTIME_DIR:?Set the adopted OpenPI bd70 runtime checkout}"
: "${POLARIS_SOURCE_APPROVAL:?Set the immutable source approval JSON}"
: "${EXPECTED_POLARIS_COMMIT:?Set EXPECTED_POLARIS_COMMIT to the frozen launch commit}"
: "${POLARIS_ENVIRONMENT:?Set POLARIS_ENVIRONMENT to one DROID task}"
: "${RUN_NAMESPACE:?Set RUN_NAMESPACE to the immutable sweep namespace}"
: "${SUBMISSION_TRANSACTION_ID:?Set the immutable submission transaction ID}"

[[ -d "${NFS_ROOT}" && ! -L "${NFS_ROOT}" ]] \
  || { echo "NFS_ROOT must name a real directory" >&2; exit 2; }
[[ -d "${POLARIS_SOURCE_SNAPSHOT}" && ! -L "${POLARIS_SOURCE_SNAPSHOT}" ]] \
  || { echo "Source snapshot must name a real directory" >&2; exit 2; }
[[ -d "${POLARIS_OPENPI_RUNTIME_DIR}" && ! -L "${POLARIS_OPENPI_RUNTIME_DIR}" ]] \
  || { echo "OpenPI runtime must name a real directory" >&2; exit 2; }
[[ -f "${POLARIS_SOURCE_APPROVAL}" && ! -L "${POLARIS_SOURCE_APPROVAL}" ]] \
  || { echo "Source approval must name a real file" >&2; exit 2; }
NFS_ROOT="$(readlink -f -- "${NFS_ROOT}")"
POLARIS_SOURCE_SNAPSHOT="$(readlink -f -- "${POLARIS_SOURCE_SNAPSHOT}")"
POLARIS_OPENPI_RUNTIME_DIR="$(readlink -f -- "${POLARIS_OPENPI_RUNTIME_DIR}")"
POLARIS_SOURCE_APPROVAL="$(readlink -f -- "${POLARIS_SOURCE_APPROVAL}")"
export POLARIS_DIR="${POLARIS_SOURCE_SNAPSHOT}"
export OPENPI_DIR="${POLARIS_OPENPI_RUNTIME_DIR}"
export POLARIS_SOURCE_APPROVAL
export NFS_ROOT POLARIS_DIR OPENPI_DIR EXPECTED_POLARIS_COMMIT \
  EXPECTED_POLARIS_SOURCE_TREE_SHA256 POLARIS_ENVIRONMENT RUN_NAMESPACE
export POLARIS_DATA_DIR="${POLARIS_DATA_DIR:-${NFS_ROOT}/data/PolaRiS-Hub}"
export POLARIS_PYXIS_IMAGE="${POLARIS_PYXIS_IMAGE:-${NFS_ROOT}/cache/polaris/polaris-eval-cuda13-fd00a51.sqsh}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${NFS_ROOT}/cache/openpi-polaris}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${NFS_ROOT}/results/polaris-pi05}"
export POLARIS_CACHE_DIR="${POLARIS_CACHE_DIR:-${NFS_ROOT}/cache/polaris/runtime/${SLURM_JOB_ID}}"
export ROLLOUTS="${ROLLOUTS:-50}"
export ENVIRONMENT_SEED="${ENVIRONMENT_SEED:-0}"
export RUN_LABEL="official-pi05-polaris-seed${ENVIRONMENT_SEED}"
export PORT="$((20000 + SLURM_JOB_ID % 20000))"
export CHECKPOINT_URI=gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris
export POLICY_CONFIG=pi05_droid_jointpos_polaris
export DRY_RUN=0
export NVIDIA_DRIVER_CAPABILITIES=all
[[ "${EXPECTED_POLARIS_COMMIT}" == c5b52a9cebb2c797a84e3df374b6002005d20a4f ]]
[[ "$(git -C "${OPENPI_DIR}" rev-parse HEAD)" == bd70b8f4011e85b3f3b0f039f12113f78718e7bf ]]
trusted_source_tree="$(trusted_source_tree_sha256 "${POLARIS_DIR}")"
[[ "${trusted_source_tree}" == "${EXPECTED_POLARIS_SOURCE_TREE_SHA256}" ]]
export BATCH_VERIFIED_POLARIS_SOURCE_TREE_SHA256="${trusted_source_tree}"
trusted_source_hasher_path="$(readlink -f -- "${BASH_SOURCE[0]}")"
batch_hasher_sha256="$(sha256sum "${trusted_source_hasher_path}" | awk '{print $1}')"
source_approval_result="$(
  trusted_source_approval "${POLARIS_SOURCE_APPROVAL}" "${POLARIS_DIR}" \
    "${EXPECTED_POLARIS_SOURCE_TREE_SHA256}" "${batch_hasher_sha256}"
)"
IFS=$'\t' read -r SOURCE_APPROVAL_SHA256 POLARIS_IMPLEMENTATION_COMMIT \
  <<< "${source_approval_result}"
export SOURCE_APPROVAL_SHA256 POLARIS_IMPLEMENTATION_COMMIT
export TRUSTED_SOURCE_HASHER_PATH="${trusted_source_hasher_path}"
export TRUSTED_SOURCE_HASHER_SHA256="${batch_hasher_sha256}"
mapfile -t allocated_gpu_uuids < <(
  nvidia-smi --query-gpu=uuid --format=csv,noheader | sed '/^[[:space:]]*$/d'
)
(( ${#allocated_gpu_uuids[@]} == 1 )) || {
  echo "Expected exactly one allocated GPU, saw ${#allocated_gpu_uuids[@]}" >&2
  exit 5
}
export NVIDIA_VISIBLE_DEVICES="${allocated_gpu_uuids[0]}"
export PYTHONUNBUFFERED=1 SLURM_EXPORT_ENV=ALL

echo "[$(date -Iseconds)] job=${SLURM_JOB_ID} host=$(hostname) task=${POLARIS_ENVIRONMENT} rollouts=${ROLLOUTS} environment_seed=${ENVIRONMENT_SEED}"
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total --format=csv,noheader

exec /usr/bin/bash --noprofile --norc \
  "${POLARIS_DIR}/scripts/polaris/eval_pi05_droid_jointpos_polaris.sh"
