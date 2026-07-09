#!/usr/bin/bash -p
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
[[ -o privileged ]] || { echo "Privileged Bash mode (-p) is required" >&2; exit 2; }
INHERITED_PATH="${PATH:-/usr/bin:/bin}"
unset BASH_ENV ENV LD_AUDIT LD_PRELOAD PYTHONHOME PYTHONPATH PYTHONUSERBASE
POLARIS_EVAL_MODE="${POLARIS_EVAL_MODE:-standard}"
if [[ "${POLARIS_EVAL_MODE}" == app_launcher_only ]]; then
  export PATH=/cm/local/apps/slurm/24.11/bin:/usr/bin:/bin
else
  export PATH="${INHERITED_PATH}"
fi

trusted_source_tree_sha256() {
  /usr/bin/python3.12 -I -S - "$1" <<'PY'
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
  /usr/bin/python3.12 -I -S - "$1" "$2" "$3" "$4" <<'PY'
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

trusted_runtime_closure_approval() {
  /usr/bin/python3.12 -I -S - "$1" "$2" "${3:-live}" <<'PY'
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys


approval_path, expected_approval_sha256, validation_mode = sys.argv[1:]
if re.fullmatch(r"[0-9a-f]{64}", expected_approval_sha256) is None:
    raise ValueError("runtime-closure approval SHA-256 is malformed")
if validation_mode not in {"artifact", "live"}:
    raise ValueError("runtime-closure validation mode is not closed")
verify_live_runtime = validation_mode == "live"
approval = Path(approval_path)
if (
    not approval.is_absolute()
    or PurePosixPath(approval.as_posix()).as_posix() != approval.as_posix()
    or approval.is_symlink()
    or approval.resolve(strict=True) != approval
    or any(character in approval_path for character in ",=\t\r\n")
):
    raise ValueError("runtime-closure approval path is not canonical")


def stable_payload(path, *, expected_mode=None, expected_size=None):
    parent_fd = os.open(
        path.parent,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor = -1
    try:
        parent_before = os.fstat(parent_fd)
        parent_public_before = os.stat(path.parent, follow_symlinks=False)
        if (
            parent_before.st_dev != parent_public_before.st_dev
            or parent_before.st_ino != parent_public_before.st_ino
        ):
            raise ValueError(f"runtime approval parent is not publicly bound: {path.parent}")
        descriptor = os.open(
            path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (expected_mode is not None and stat.S_IMODE(before.st_mode) != expected_mode)
            or (expected_size is not None and before.st_size != expected_size)
        ):
            raise ValueError(f"runtime approval file metadata mismatch: {path}")
        payload = bytearray()
        offset = 0
        while offset < before.st_size:
            block = os.pread(descriptor, min(16 * 1024 * 1024, before.st_size - offset), offset)
            if not block:
                raise ValueError(f"short runtime approval file read: {path}")
            payload.extend(block)
            offset += len(block)
        if os.pread(descriptor, 1, before.st_size):
            raise ValueError(f"growing runtime approval file: {path}")
        after = os.fstat(descriptor)
        observed = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        absolute_observed = os.stat(path, follow_symlinks=False)
        parent_after = os.fstat(parent_fd)
        parent_public_after = os.stat(path.parent, follow_symlinks=False)
        fields = (
            "st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink",
            "st_size", "st_mtime_ns", "st_ctime_ns",
        )
        if any(getattr(before, item) != getattr(after, item) for item in fields):
            raise ValueError(f"runtime approval file changed while reading: {path}")
        if any(getattr(after, item) != getattr(observed, item) for item in fields):
            raise ValueError(f"runtime approval path binding changed: {path}")
        if any(
            getattr(after, item) != getattr(absolute_observed, item)
            for item in fields
        ):
            raise ValueError(f"runtime approval absolute binding changed: {path}")
        if any(
            getattr(parent_before, item) != getattr(parent_after, item)
            for item in ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid")
        ):
            raise ValueError(f"runtime approval parent binding changed: {path.parent}")
        if (
            parent_after.st_dev != parent_public_after.st_dev
            or parent_after.st_ino != parent_public_after.st_ino
        ):
            raise ValueError(f"runtime approval public parent changed: {path.parent}")
        return bytes(payload), after
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


payload, _ = stable_payload(approval, expected_mode=0o444)
approval_sha256 = hashlib.sha256(payload).hexdigest()
if approval_sha256 != expected_approval_sha256:
    raise ValueError("runtime-closure approval digest mismatch")
value = json.loads(
    payload,
    parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
)
expected_imports = [
    "argparse", "hashlib", "json", "os", "pathlib", "re", "signal", "stat",
    "subprocess", "sys", "time", "typing",
]
expected_trust_boundary = {
    "bootstrap_trust": [
        "kernel and dynamic loader before approval verification",
        "Slurm job launch and Pyxis/srun lifecycle",
        "the shell and hashing/verifier tools used before host Python is trusted",
        "root-owned directory traversal and symlink resolution",
    ],
    "execution_subclosure": (
        "Host Python standard-library helper plus pinned scontrol scheduler client. "
        "sacct, scancel, and srun executable identities are pinned only as bootstrap "
        "drift evidence; their dynamic closures, Bash, coreutils, Pyxis, container "
        "runtime, NVIDIA tools, kernel, and external services are not claimed as "
        "closed by this approval."
    ),
    "security_claim": (
        "This approval supports drift detection under a trusted root-managed host; "
        "it is not a self-authenticating root of trust because Python cannot "
        "establish the trustworthiness of Python before executing."
    ),
}
if (
    not isinstance(value, dict)
    or set(value) != {
        "schema_version", "profile", "capture_scope", "subclosure_profile",
        "top_level_imports", "immutable_files", "symlink_bindings",
        "ambient_runtime_dependencies", "trust_boundary",
    }
    or isinstance(value["schema_version"], bool)
    or value["schema_version"] != 1
    or value["profile"] != "polaris_app_launcher_runtime_subclosure_approval_v1"
    or value["capture_scope"] != "allocated_pool0_compute_after_source_freeze"
    or value["subclosure_profile"]
    != "host_python_stdlib_scontrol_and_scheduler_binary_drift_v1"
    or value["top_level_imports"] != expected_imports
    or not isinstance(value["immutable_files"], list)
    or not value["immutable_files"]
    or not isinstance(value["symlink_bindings"], list)
    or value["trust_boundary"] != expected_trust_boundary
):
    raise ValueError("runtime-closure approval schema mismatch")
canonical = json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
).encode("ascii") + b"\n"
if payload != canonical:
    raise ValueError("runtime-closure approval is not canonical JSON")
expected_entrypoints = {
    "/cm/local/apps/slurm/24.11/bin/scontrol",
    "/usr/bin/python3.12",
}
expected_scheduler_bootstrap_binaries = {
    "/cm/local/apps/slurm/24.11/bin/sacct",
    "/cm/local/apps/slurm/24.11/bin/scancel",
    "/cm/local/apps/slurm/24.11/bin/srun",
}
required_paths = expected_entrypoints | expected_scheduler_bootstrap_binaries | {
    "/cm/local/apps/slurm/24.11/lib64/slurm/libslurmfull.so",
    "/cm/shared/apps/slurm/etc/oci-ord-cs-004/slurm.conf",
}
allowed_roles = {
    "approval_bound_configuration",
    "entrypoint",
    "python_elf_dependency",
    "python_extension_module",
    "python_stdlib_bytecode",
    "python_stdlib_resolution_landmark",
    "python_stdlib_source",
    "scheduler_bootstrap_binary_drift",
    "scontrol_elf_dependency",
    "scontrol_slurm_plugin",
    "scontrol_slurm_runtime",
}

observed_paths = []
records_by_path = {}
role_paths = {role: [] for role in allowed_roles}
for record in value["immutable_files"]:
    if not isinstance(record, dict) or set(record) != {
        "path", "roles", "mode", "uid", "gid", "nlink", "size", "sha256"
    }:
        raise ValueError("runtime-closure file record schema mismatch")
    path_text = record["path"]
    roles = record["roles"]
    if (
        not isinstance(path_text, str)
        or not path_text.startswith("/")
        or PurePosixPath(path_text).as_posix() != path_text
        or ".." in PurePosixPath(path_text).parts
        or not isinstance(roles, list)
        or not roles
        or roles != sorted(set(roles))
        or not set(roles) <= allowed_roles
        or not isinstance(record["mode"], str)
        or re.fullmatch(r"[0-7]{4}", record["mode"]) is None
        or not isinstance(record["uid"], int)
        or isinstance(record["uid"], bool)
        or record["uid"] < 0
        or not isinstance(record["gid"], int)
        or isinstance(record["gid"], bool)
        or record["gid"] < 0
        or record["uid"] != 0
        or record["gid"] != 0
        or not isinstance(record["nlink"], int)
        or isinstance(record["nlink"], bool)
        or record["nlink"] != 1
        or not isinstance(record["size"], int)
        or isinstance(record["size"], bool)
        or record["size"] < 0
        or (
            record["size"] == 0
            and not (
                "python_stdlib_source" in roles
                and path_text == "/usr/lib/python3.12/urllib/__init__.py"
            )
        )
        or not isinstance(record["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
        or int(record["mode"], 8) & 0o022
    ):
        raise ValueError("runtime-closure file record is malformed")
    if verify_live_runtime:
        path = Path(path_text)
        if path.is_symlink() or path.resolve(strict=True) != path:
            raise ValueError(f"runtime-closure path is not canonical: {path}")
        file_payload, metadata = stable_payload(
            path,
            expected_mode=int(record["mode"], 8),
            expected_size=record["size"],
        )
        if hashlib.sha256(file_payload).hexdigest() != record["sha256"]:
            raise ValueError(f"runtime-closure file digest mismatch: {path}")
        if metadata.st_nlink != record["nlink"]:
            raise ValueError(f"runtime-closure file link count mismatch: {path}")
        if metadata.st_uid != record["uid"] or metadata.st_gid != record["gid"]:
            raise ValueError(f"runtime-closure file ownership mismatch: {path}")
    if "entrypoint" in roles and (
        path_text not in expected_entrypoints or record["mode"] != "0755"
    ):
        raise ValueError(f"runtime-closure entrypoint role mismatch: {path_text}")
    if "scheduler_bootstrap_binary_drift" in roles and (
        path_text not in expected_scheduler_bootstrap_binaries
        or record["mode"] != "0755"
    ):
        raise ValueError(
            f"runtime-closure scheduler bootstrap role mismatch: {path_text}"
        )
    if path_text == "/cm/shared/apps/slurm/etc/oci-ord-cs-004/slurm.conf" and (
        roles != ["approval_bound_configuration"] or record["mode"] != "0644"
    ):
        raise ValueError("runtime-closure Slurm config role mismatch")
    if path_text == "/cm/local/apps/slurm/24.11/lib64/slurm/libslurmfull.so" and (
        "scontrol_slurm_runtime" not in roles or record["mode"] != "0644"
    ):
        raise ValueError("runtime-closure Slurm library role mismatch")
    observed_paths.append(path_text)
    records_by_path[path_text] = record
    for role in roles:
        role_paths[role].append(path_text)
if observed_paths != sorted(observed_paths) or len(observed_paths) != len(set(observed_paths)):
    raise ValueError("runtime-closure file records are not unique path-sorted records")
role_projection = {
    role: sorted(paths) for role, paths in role_paths.items()
}
role_projection_sha256 = hashlib.sha256(
    json.dumps(
        role_projection, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii") + b"\n"
).hexdigest()
if role_projection_sha256 != (
    "15ab50cd061a821ffbe654040a3221cb609a3c43a8b61c8eae4cb687475de6c7"
):
    raise ValueError("runtime-closure role-to-path projection is not exact")
if set(role_paths["entrypoint"]) != expected_entrypoints:
    raise ValueError("runtime-closure entrypoint census is not exact")
if role_paths["approval_bound_configuration"] != [
    "/cm/shared/apps/slurm/etc/oci-ord-cs-004/slurm.conf"
]:
    raise ValueError("runtime-closure configuration census is not exact")
if role_paths["scontrol_slurm_runtime"] != [
    "/cm/local/apps/slurm/24.11/lib64/slurm/libslurmfull.so"
]:
    raise ValueError("runtime-closure Slurm runtime census is not exact")
if role_paths["python_stdlib_resolution_landmark"] != [
    "/usr/lib/python3.12/os.py"
]:
    raise ValueError("runtime-closure Python resolution landmark is not exact")
if len(role_paths["python_stdlib_source"]) != 39:
    raise ValueError("runtime-closure Python source census is not exact")
if len(role_paths["python_stdlib_bytecode"]) != 39 or any(
    not path.endswith(".cpython-312.pyc")
    for path in role_paths["python_stdlib_bytecode"]
):
    raise ValueError("runtime-closure Python bytecode census is not exact")
if set(role_paths["python_extension_module"]) != {
    "/usr/lib/python3.12/lib-dynload/_hashlib.cpython-312-x86_64-linux-gnu.so",
    "/usr/lib/python3.12/lib-dynload/_json.cpython-312-x86_64-linux-gnu.so",
}:
    raise ValueError("runtime-closure Python extension census is not exact")
required_runtime_names = {
    "libc.so.6", "libm.so.6", "libtinfo.so.6.4", "ld-linux-x86-64.so.2",
}
observed_names = {Path(path).name for path in observed_paths}
if not required_runtime_names <= observed_names:
    raise ValueError("runtime-closure approval omits a required ELF dependency")
required_slurm_plugins = {
    "accounting_storage_slurmdbd.so", "auth_munge.so", "cred_munge.so",
    "hash_k12.so", "tls_none.so",
}
plugin_names = {
    Path(path).name for path in role_paths["scontrol_slurm_plugin"]
}
if plugin_names != required_slurm_plugins:
    raise ValueError("runtime-closure approval Slurm plugin census is not exact")
if (
    not any(name.startswith("libmunge.so.2") for name in observed_names)
    or not any(name.startswith("libnss_sss.so.2") for name in observed_names)
):
    raise ValueError("runtime-closure approval omits a scheduler/NSS dependency")
required_python_suffixes = {
    "/hashlib.py", "/json/__init__.py", "/os.py", "/pathlib.py",
    "/re/__init__.py",
}
if any(
    not any(path.endswith(suffix) for path in observed_paths)
    for suffix in required_python_suffixes
):
    raise ValueError("runtime-closure approval omits a required Python source")
if not any(Path(path).name.startswith("_hashlib.") and path.endswith(".so") for path in observed_paths):
    raise ValueError("runtime-closure approval omits the _hashlib extension")
if not any(Path(path).name.startswith("_json.") and path.endswith(".so") for path in observed_paths):
    raise ValueError("runtime-closure approval omits the _json extension")
if not required_paths <= set(observed_paths):
    raise ValueError("runtime-closure approval is missing a required file")

bindings = value["symlink_bindings"]
expected_bindings = [
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
        "resolved_path": "/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2",
        "target": "../lib/x86_64-linux-gnu/ld-linux-x86-64.so.2",
    },
    {
        "path": "/usr/share/locale/locale.alias",
        "resolved_path": "/etc/locale.alias",
        "target": "/etc/locale.alias",
    },
    {"path": "/var/run", "resolved_path": "/run", "target": "/run"},
]
if bindings != expected_bindings:
    raise ValueError("runtime-closure symlink binding census is not exact")
binding_paths = []
for binding in bindings:
    if not isinstance(binding, dict) or set(binding) != {
        "path", "target", "resolved_path"
    }:
        raise ValueError("runtime-closure symlink binding schema mismatch")
    path_text = binding["path"]
    resolved_text = binding["resolved_path"]
    target = binding["target"]
    if (
        not isinstance(path_text, str)
        or not isinstance(target, str)
        or not isinstance(resolved_text, str)
        or not path_text.startswith("/")
        or not resolved_text.startswith("/")
        or PurePosixPath(path_text).as_posix() != path_text
        or PurePosixPath(resolved_text).as_posix() != resolved_text
        or ".." in PurePosixPath(path_text).parts
        or ".." in PurePosixPath(resolved_text).parts
        or not target
        or "\x00" in target
    ):
        raise ValueError("runtime-closure symlink binding is malformed")
    if verify_live_runtime:
        path = Path(path_text)
        before = os.lstat(path)
        observed_target = os.readlink(path)
        resolved = str(path.resolve(strict=True))
        after = os.lstat(path)
        public = os.stat(path, follow_symlinks=False)
        identity_fields = (
            "st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink",
            "st_size", "st_mtime_ns", "st_ctime_ns",
        )
        if (
            not stat.S_ISLNK(before.st_mode)
            or before.st_uid != 0
            or before.st_gid != 0
            or observed_target != target
            or resolved != resolved_text
            or any(
                getattr(before, item) != getattr(after, item)
                for item in identity_fields
            )
            or any(
                getattr(after, item) != getattr(public, item)
                for item in identity_fields
            )
        ):
            raise ValueError(f"runtime-closure symlink binding changed: {path}")
    binding_paths.append(path_text)
if binding_paths != sorted(binding_paths) or len(binding_paths) != len(set(binding_paths)):
    raise ValueError("runtime-closure symlink bindings are not unique path-sorted records")

ambient = value["ambient_runtime_dependencies"]
expected_regular_file_paths = {
    "nss_dns": [
        "/etc/group", "/etc/host.conf", "/etc/hosts", "/etc/nsswitch.conf",
        "/etc/passwd", "/run/systemd/resolve/resolv.conf",
        "/var/lib/sss/mc/passwd",
    ],
    "root_managed_runtime_data": [
        "/etc/ld.so.cache", "/etc/locale.alias", "/etc/ssl/openssl.cnf",
        "/usr/lib/locale/C.utf8/LC_CTYPE", "/usr/lib/locale/locale-archive",
        "/usr/lib/x86_64-linux-gnu/gconv/gconv-modules.cache",
        "/usr/share/zoneinfo/America/Los_Angeles",
    ],
}
expected_sockets = [
    {
        "gid": 0, "mode": "0777", "path": "/run/munge/munge.socket.2",
        "present": True, "role": "munge_authentication", "uid": 1,
    },
    {
        "conditional": True, "gid": 0, "mode": "0666",
        "path": "/var/lib/sss/pipes/nss", "present": True,
        "role": "sssd_nss_cache_miss", "uid": 0,
    },
]
expected_external_services = [
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
]
if (
    not isinstance(ambient, dict)
    or set(ambient) != {
        "external_services", "negative_resolution_assertions", "regular_files",
        "scope", "sockets",
    }
    or ambient["scope"] != "point_in_time_observation_not_immutable_code_approval"
    or ambient["negative_resolution_assertions"] != ["/usr/lib/python312.zip"]
    or ambient["sockets"] != expected_sockets
    or ambient["external_services"] != expected_external_services
    or not isinstance(ambient["regular_files"], dict)
    or set(ambient["regular_files"]) != set(expected_regular_file_paths)
):
    raise ValueError("runtime-closure ambient dependency schema mismatch")
if verify_live_runtime and os.path.lexists("/usr/lib/python312.zip"):
    raise ValueError("runtime-closure negative resolution assertion changed")
for category, expected_paths in expected_regular_file_paths.items():
    records = ambient["regular_files"][category]
    if not isinstance(records, list) or [
        record.get("path") if isinstance(record, dict) else None
        for record in records
    ] != expected_paths:
        raise ValueError("runtime-closure ambient regular-file census changed")
    for record in records:
        if set(record) != {
            "path", "mode", "uid", "gid", "nlink", "size", "sha256"
        }:
            raise ValueError("runtime-closure ambient file schema mismatch")
        if (
            re.fullmatch(r"[0-7]{4}", record["mode"])
            is None
            or not isinstance(record["uid"], int)
            or isinstance(record["uid"], bool)
            or record["uid"] < 0
            or not isinstance(record["gid"], int)
            or isinstance(record["gid"], bool)
            or record["gid"] < 0
            or not isinstance(record["nlink"], int)
            or isinstance(record["nlink"], bool)
            or record["nlink"] != 1
            or not isinstance(record["size"], int)
            or isinstance(record["size"], bool)
            or record["size"] <= 0
            or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
        ):
            raise ValueError("runtime-closure ambient file identity is malformed")
        if verify_live_runtime:
            ambient_path = Path(record["path"])
            if (
                ambient_path.is_symlink()
                or ambient_path.resolve(strict=True) != ambient_path
            ):
                raise ValueError("runtime-closure ambient file path is not canonical")
            ambient_payload, ambient_metadata = stable_payload(
                ambient_path,
                expected_mode=int(record["mode"], 8),
                expected_size=record["size"],
            )
            if (
                ambient_metadata.st_uid != record["uid"]
                or ambient_metadata.st_gid != record["gid"]
                or ambient_metadata.st_nlink != record["nlink"]
                or hashlib.sha256(ambient_payload).hexdigest()
                != record["sha256"]
            ):
                raise ValueError(
                    f"runtime-closure ambient file drifted: {ambient_path}"
                )
if verify_live_runtime:
    for socket_record in expected_sockets:
        socket_path = Path(socket_record["path"])
        if not socket_record["present"]:
            if os.path.lexists(socket_path):
                raise ValueError(
                    "runtime-closure conditionally absent socket appeared: "
                    f"{socket_path}"
                )
            continue
        socket_before = os.stat(socket_path, follow_symlinks=False)
        socket_after = os.stat(socket_path, follow_symlinks=False)
        socket_fields = (
            "st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink",
            "st_mtime_ns", "st_ctime_ns",
        )
        if (
            not stat.S_ISSOCK(socket_before.st_mode)
            or socket_before.st_uid != socket_record["uid"]
            or socket_before.st_gid != socket_record["gid"]
            or f"{stat.S_IMODE(socket_before.st_mode):04o}"
            != socket_record["mode"]
            or any(
                getattr(socket_before, item) != getattr(socket_after, item)
                for item in socket_fields
            )
        ):
            raise ValueError(
                f"runtime-closure ambient socket drifted: {socket_path}"
            )

def approved_identity(path):
    record = records_by_path[path]
    return record["sha256"], record["size"]


config_sha256, config_size = approved_identity(
    "/cm/shared/apps/slurm/etc/oci-ord-cs-004/slurm.conf"
)
scontrol_sha256, scontrol_size = approved_identity(
    "/cm/local/apps/slurm/24.11/bin/scontrol"
)
slurm_library_sha256, slurm_library_size = approved_identity(
    "/cm/local/apps/slurm/24.11/lib64/slurm/libslurmfull.so"
)
sacct_sha256, sacct_size = approved_identity(
    "/cm/local/apps/slurm/24.11/bin/sacct"
)
scancel_sha256, scancel_size = approved_identity(
    "/cm/local/apps/slurm/24.11/bin/scancel"
)
srun_sha256, srun_size = approved_identity(
    "/cm/local/apps/slurm/24.11/bin/srun"
)
print(
    f"{approval_sha256}\t{config_sha256}\t{config_size}"
    f"\t{scontrol_sha256}\t{scontrol_size}"
    f"\t{slurm_library_sha256}\t{slurm_library_size}"
    f"\t{sacct_sha256}\t{sacct_size}\t{scancel_sha256}\t{scancel_size}"
    f"\t{srun_sha256}\t{srun_size}"
)
PY
}

trusted_output_namespace_identity() {
  /usr/bin/python3.12 -I -S - "$1" "$2" <<'PY'
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys

path_text, expected = sys.argv[1:]
path = Path(path_text)
if (
    not path.is_absolute()
    or PurePosixPath(path_text).as_posix() != path_text
    or path.is_symlink()
    or path.resolve(strict=True) != path
    or re.fullmatch(r"[0-9]+:[0-9]+:[0-9]+:[0-9]+:0755", expected) is None
):
    raise ValueError("output namespace path or identity is malformed")
parent_fd = os.open(
    path.parent, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
)
descriptor = -1
try:
    parent_before = os.fstat(parent_fd)
    descriptor = os.open(
        path.name,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_fd,
    )
    opened = os.fstat(descriptor)
    observed = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
    parent_after = os.fstat(parent_fd)
    parent_observed = os.stat(path.parent, follow_symlinks=False)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or stat.S_IMODE(opened.st_mode) != 0o755
        or (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_uid, opened.st_gid)
        != (observed.st_dev, observed.st_ino, observed.st_mode, observed.st_uid, observed.st_gid)
        or (parent_before.st_dev, parent_before.st_ino, parent_before.st_mode)
        != (parent_after.st_dev, parent_after.st_ino, parent_after.st_mode)
        or (parent_after.st_dev, parent_after.st_ino, parent_after.st_mode)
        != (parent_observed.st_dev, parent_observed.st_ino, parent_observed.st_mode)
    ):
        raise ValueError("output namespace descriptor binding is invalid")
    actual = ":".join((
        str(opened.st_dev), str(opened.st_ino), str(opened.st_uid),
        str(opened.st_gid), f"{stat.S_IMODE(opened.st_mode):04o}",
    ))
    if actual != expected:
        raise ValueError("output namespace creator identity changed")
finally:
    if descriptor >= 0:
        os.close(descriptor)
    os.close(parent_fd)
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
if [[ "${1:-}" == --validate-runtime-closure-approval ]]; then
  (( $# == 3 )) || { echo "Usage: $0 --validate-runtime-closure-approval FILE SHA256" >&2; exit 2; }
  trusted_runtime_closure_approval "$2" "$3" live
  exit
fi
if [[ "${1:-}" == --inspect-runtime-closure-approval ]]; then
  (( $# == 3 )) || { echo "Usage: $0 --inspect-runtime-closure-approval FILE SHA256" >&2; exit 2; }
  trusted_runtime_closure_approval "$2" "$3" artifact
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
export POLARIS_EVAL_MODE
case "${POLARIS_EVAL_MODE}" in
  standard) export ROLLOUTS="${ROLLOUTS:-50}" ;;
  app_launcher_only)
    : "${POLARIS_RUNTIME_CLOSURE_APPROVAL:?AppLauncher diagnostic requires the immutable runtime-closure approval}"
    : "${POLARIS_EXPECTED_SLURM_CONFIG_SHA256:?AppLauncher diagnostic requires the approved Slurm config SHA-256}"
    : "${POLARIS_EXPECTED_SCONTROL_SHA256:?AppLauncher diagnostic requires the approved scontrol SHA-256}"
    : "${POLARIS_EXPECTED_SCONTROL_SIZE:?AppLauncher diagnostic requires the approved scontrol size}"
    : "${POLARIS_EXPECTED_SLURM_LIBRARY_SHA256:?AppLauncher diagnostic requires the approved Slurm library SHA-256}"
    : "${POLARIS_EXPECTED_SLURM_LIBRARY_SIZE:?AppLauncher diagnostic requires the approved Slurm library size}"
    : "${POLARIS_EXPECTED_SACCT_SHA256:?AppLauncher diagnostic requires the approved sacct SHA-256}"
    : "${POLARIS_EXPECTED_SACCT_SIZE:?AppLauncher diagnostic requires the approved sacct size}"
    : "${POLARIS_EXPECTED_SCANCEL_SHA256:?AppLauncher diagnostic requires the approved scancel SHA-256}"
    : "${POLARIS_EXPECTED_SCANCEL_SIZE:?AppLauncher diagnostic requires the approved scancel size}"
    : "${POLARIS_EXPECTED_SRUN_SHA256:?AppLauncher diagnostic requires the approved srun SHA-256}"
    : "${POLARIS_EXPECTED_SRUN_SIZE:?AppLauncher diagnostic requires the approved srun size}"
    : "${POLARIS_RUNTIME_CLOSURE_APPROVAL_SHA256:?AppLauncher diagnostic requires the runtime-closure approval SHA-256}"
    : "${POLARIS_OUTPUT_NAMESPACE_PARENT_IDENTITY:?AppLauncher diagnostic requires the creator-observed output namespace identity}"
    : "${OUTPUT_ROOT:?AppLauncher diagnostic requires its canonical OUTPUT_ROOT}"
    [[ "${POLARIS_EXPECTED_SLURM_CONFIG_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
      || { echo "Invalid approved Slurm config SHA-256" >&2; exit 2; }
    [[ "${POLARIS_RUNTIME_CLOSURE_APPROVAL_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
      || { echo "Invalid runtime-closure approval SHA-256" >&2; exit 2; }
    [[ "${POLARIS_EXPECTED_SCONTROL_SHA256}" =~ ^[0-9a-f]{64}$ \
      && "${POLARIS_EXPECTED_SCONTROL_SIZE}" =~ ^[1-9][0-9]*$ \
      && "${POLARIS_EXPECTED_SLURM_LIBRARY_SHA256}" =~ ^[0-9a-f]{64}$ \
      && "${POLARIS_EXPECTED_SLURM_LIBRARY_SIZE}" =~ ^[1-9][0-9]*$ \
      && "${POLARIS_EXPECTED_SACCT_SHA256}" =~ ^[0-9a-f]{64}$ \
      && "${POLARIS_EXPECTED_SACCT_SIZE}" =~ ^[1-9][0-9]*$ \
      && "${POLARIS_EXPECTED_SCANCEL_SHA256}" =~ ^[0-9a-f]{64}$ \
      && "${POLARIS_EXPECTED_SCANCEL_SIZE}" =~ ^[1-9][0-9]*$ \
      && "${POLARIS_EXPECTED_SRUN_SHA256}" =~ ^[0-9a-f]{64}$ \
      && "${POLARIS_EXPECTED_SRUN_SIZE}" =~ ^[1-9][0-9]*$ ]] \
      || { echo "Invalid approved scheduler runtime identity" >&2; exit 2; }
    runtime_approval_result="$(
      trusted_runtime_closure_approval \
        "${POLARIS_RUNTIME_CLOSURE_APPROVAL}" \
        "${POLARIS_RUNTIME_CLOSURE_APPROVAL_SHA256}"
    )" || { echo "Runtime-closure approval validation failed" >&2; exit 2; }
    IFS=$'\t' read -r batch_runtime_approval_sha256 \
      batch_slurm_config_sha256 batch_scontrol_sha256 batch_scontrol_size \
      batch_slurm_library_sha256 batch_slurm_library_size \
      batch_sacct_sha256 batch_sacct_size \
      batch_scancel_sha256 batch_scancel_size \
      batch_srun_sha256 batch_srun_size \
      <<< "${runtime_approval_result}"
    [[ "${batch_runtime_approval_sha256}" == \
        "${POLARIS_RUNTIME_CLOSURE_APPROVAL_SHA256}" \
      && "${batch_slurm_config_sha256}" == \
        "${POLARIS_EXPECTED_SLURM_CONFIG_SHA256}" \
      && "${batch_scontrol_sha256}" == "${POLARIS_EXPECTED_SCONTROL_SHA256}" \
      && "${batch_scontrol_size}" == "${POLARIS_EXPECTED_SCONTROL_SIZE}" \
      && "${batch_slurm_library_sha256}" == \
        "${POLARIS_EXPECTED_SLURM_LIBRARY_SHA256}" \
      && "${batch_slurm_library_size}" == \
        "${POLARIS_EXPECTED_SLURM_LIBRARY_SIZE}" \
      && "${batch_sacct_sha256}" == "${POLARIS_EXPECTED_SACCT_SHA256}" \
      && "${batch_sacct_size}" == "${POLARIS_EXPECTED_SACCT_SIZE}" \
      && "${batch_scancel_sha256}" == "${POLARIS_EXPECTED_SCANCEL_SHA256}" \
      && "${batch_scancel_size}" == "${POLARIS_EXPECTED_SCANCEL_SIZE}" \
      && "${batch_srun_sha256}" == "${POLARIS_EXPECTED_SRUN_SHA256}" \
      && "${batch_srun_size}" == "${POLARIS_EXPECTED_SRUN_SIZE}" ]] \
      || { echo "Runtime-closure approval values changed" >&2; exit 2; }
    POLARIS_RUNTIME_CLOSURE_APPROVAL="$(
      /usr/bin/readlink -f -- "${POLARIS_RUNTIME_CLOSURE_APPROVAL}"
    )" || { echo "Cannot canonicalize runtime-closure approval" >&2; exit 2; }
    export BATCH_VERIFIED_RUNTIME_CLOSURE_APPROVAL_SHA256="${batch_runtime_approval_sha256}"
    [[ "${POLARIS_OUTPUT_NAMESPACE_PARENT_IDENTITY}" =~ \
      ^[0-9]+:[0-9]+:[0-9]+:[0-9]+:0755$ ]] \
      || { echo "Invalid creator-observed output namespace identity" >&2; exit 2; }
    [[ "${RUN_NAMESPACE}" =~ ^[A-Za-z0-9._-]+$ ]] \
      || { echo "Invalid AppLauncher RUN_NAMESPACE" >&2; exit 2; }
    output_namespace_parent="${OUTPUT_ROOT}/${RUN_NAMESPACE}"
    trusted_output_namespace_identity \
      "${output_namespace_parent}" \
      "${POLARIS_OUTPUT_NAMESPACE_PARENT_IDENTITY}" \
      || { echo "AppLauncher output namespace validation failed" >&2; exit 2; }
    export OUTPUT_ROOT POLARIS_OUTPUT_NAMESPACE_PARENT_IDENTITY \
      POLARIS_RUNTIME_CLOSURE_APPROVAL \
      POLARIS_EXPECTED_SLURM_CONFIG_SHA256 \
      POLARIS_EXPECTED_SCONTROL_SHA256 POLARIS_EXPECTED_SCONTROL_SIZE \
      POLARIS_EXPECTED_SLURM_LIBRARY_SHA256 \
      POLARIS_EXPECTED_SLURM_LIBRARY_SIZE \
      POLARIS_EXPECTED_SACCT_SHA256 POLARIS_EXPECTED_SACCT_SIZE \
      POLARIS_EXPECTED_SCANCEL_SHA256 POLARIS_EXPECTED_SCANCEL_SIZE \
      POLARIS_EXPECTED_SRUN_SHA256 POLARIS_EXPECTED_SRUN_SIZE \
      POLARIS_RUNTIME_CLOSURE_APPROVAL_SHA256 \
      BATCH_VERIFIED_RUNTIME_CLOSURE_APPROVAL_SHA256
    export ROLLOUTS="${ROLLOUTS:-1}"
    [[ "${ROLLOUTS}" == 1 ]] \
      || { echo "app_launcher_only requires ROLLOUTS=1" >&2; exit 2; }
    ;;
  *) echo "Unsupported POLARIS_EVAL_MODE=${POLARIS_EVAL_MODE}" >&2; exit 2 ;;
esac
export ENVIRONMENT_SEED="${ENVIRONMENT_SEED:-0}"
if [[ "${POLARIS_EVAL_MODE}" == app_launcher_only ]]; then
  export RUN_LABEL="public-app-launcher-only"
else
  export RUN_LABEL="official-pi05-polaris-seed${ENVIRONMENT_SEED}"
fi
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
if [[ "${POLARIS_EVAL_MODE}" == app_launcher_only ]]; then
  [[ "${SLURM_GPUS_ON_NODE:-}" == 1 ]] \
    || { echo "AppLauncher diagnostic requires SLURM_GPUS_ON_NODE=1" >&2; exit 5; }
  [[ "${SLURM_JOB_GPUS:-}" =~ ^[0-9]+$ ]] \
    || { echo "AppLauncher diagnostic requires one numeric SLURM_JOB_GPUS index" >&2; exit 5; }
fi

echo "[$(date -Iseconds)] job=${SLURM_JOB_ID} host=$(hostname) task=${POLARIS_ENVIRONMENT} rollouts=${ROLLOUTS} environment_seed=${ENVIRONMENT_SEED}"
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total --format=csv,noheader

exec /usr/bin/bash --noprofile --norc -p \
  "${POLARIS_DIR}/scripts/polaris/eval_pi05_droid_jointpos_polaris.sh"
