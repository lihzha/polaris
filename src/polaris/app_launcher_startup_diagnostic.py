"""Policy-free public evaluator startup diagnostic.

This module is intentionally limited to the Python standard library.  Its
``exec`` entrypoint runs inside the evaluator's Pyxis ``srun`` step, captures
the step and GPU boundary, publishes one immutable pre-exec artifact, and then
replaces itself with the normal ``/.venv/bin/python scripts/eval.py`` process.
The PID is preserved across ``execve`` so the post-AppLauncher branch can prove
that it is still in the same process and Slurm step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


STARTUP_DIAGNOSTIC_MODE = "app_launcher_only"
PREEXEC_PROFILE = "polaris_public_eval_app_launcher_preexec_v1"
PRECLOSE_PROFILE = "polaris_public_eval_app_launcher_preclose_v1"
READY_PROFILE = "polaris_public_eval_app_launcher_ready_v1"
EXPECTED_NVIDIA_GPU_NAME = "NVIDIA L40S"
EXPECTED_NVIDIA_DRIVER_VERSION = "580.105.08"
NVIDIA_SMI_COMMAND = (
    "/usr/bin/nvidia-smi",
    "--query-gpu=uuid,name,driver_version,minor_number",
    "--format=csv,noheader,nounits",
)

GPU_UUID_PATTERN = re.compile(
    r"GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
)
DECIMAL_PATTERN = re.compile(r"0|[1-9][0-9]*")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

FORBIDDEN_MODULE_PREFIXES = (
    "gym",
    "gymnasium",
    "isaaclab_tasks",
    "openpi",
    "openpi_client",
    "polaris.environments",
    "polaris.policy",
    "sentencepiece",
    "tokenizers",
    "transformers",
)

ZERO_WORK_COUNTERS = {
    "checkpoint_resolutions": 0,
    "checkpoint_reads": 0,
    "tokenizer_reads": 0,
    "model_loads": 0,
    "policy_client_class_loads": 0,
    "policy_client_instances": 0,
    "policy_requests": 0,
    "action_predictions": 0,
    "environment_imports": 0,
    "environment_constructions": 0,
    "gym_make_calls": 0,
    "environment_resets": 0,
    "environment_steps": 0,
    "episodes_started": 0,
}

SOURCE_IDENTITY_ENVIRONMENT = (
    "BATCH_VERIFIED_POLARIS_SOURCE_TREE_SHA256",
    "POLARIS_IMPLEMENTATION_COMMIT",
    "SOURCE_APPROVAL_SHA256",
)
PREEXEC_GPU_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "SLURM_JOB_GPUS",
    "SLURM_STEP_GPUS",
)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the sole accepted JSON representation."""

    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _same_stat(left: os.stat_result, right: os.stat_result) -> bool:
    return all(
        getattr(left, field) == getattr(right, field)
        for field in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_uid",
            "st_gid",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    )


def _canonical_absolute_path(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise ValueError(f"{label} must be an absolute file path")
    if PurePosixPath(path.as_posix()).as_posix() != path.as_posix():
        raise ValueError(f"{label} is not lexically canonical")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError(f"{label} parent must be one real directory")
    if parent.resolve(strict=True) != parent:
        raise ValueError(f"{label} parent must use its canonical physical path")
    return path


def _read_descriptor(descriptor: int, size: int) -> bytes:
    payload = bytearray()
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not block:
            raise ValueError("short descriptor read")
        payload.extend(block)
        offset += len(block)
    if os.pread(descriptor, 1, size):
        raise ValueError("descriptor grew while reading")
    return bytes(payload)


def _artifact_identity(
    path: Path, metadata: os.stat_result, payload: bytes
) -> dict[str, Any]:
    return {
        "path": str(path),
        "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "nlink": metadata.st_nlink,
        "size": len(payload),
        "sha256": _sha256(payload),
    }


def stable_file_identity(path: Path) -> dict[str, Any]:
    """Hash one stable regular file without following a final symlink."""

    path = _canonical_absolute_path(path, label="source file")
    parent_fd = os.open(
        path.parent,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ValueError(f"source file is not one regular link: {path}")
            payload = _read_descriptor(descriptor, before.st_size)
            after = os.fstat(descriptor)
            observed = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            if not _same_stat(before, after) or not _same_stat(after, observed):
                raise ValueError(f"source file changed while reading: {path}")
            return _artifact_identity(path, after, payload)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)


def publish_immutable_json(path: Path, value: Any) -> dict[str, Any]:
    """Atomically publish one canonical, non-replacing, mode-0444 JSON file."""

    path = _canonical_absolute_path(path, label="output")
    payload = canonical_json_bytes(value)
    parent_fd = os.open(
        path.parent,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    temporary_name = f".{path.name}.partial-{os.getpid()}-{os.urandom(8).hex()}"
    descriptor = -1
    linked = False
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o400,
            dir_fd=parent_fd,
        )
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short immutable artifact write")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(
            temporary_name,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        linked = True
        os.unlink(temporary_name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        if linked:
            # A destination that was successfully linked is forensic evidence.
            # Never remove or replace it after publication.
            os.fsync(parent_fd)
        raise
    finally:
        os.close(parent_fd)

    _, identity = stable_read_immutable_json(path)
    if identity["sha256"] != _sha256(payload):
        raise RuntimeError("immutable artifact readback digest mismatch")
    return identity


def stable_read_immutable_json(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Read and validate one immutable canonical JSON artifact."""

    path = _canonical_absolute_path(path, label="immutable input")
    parent_fd = os.open(
        path.parent,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_IMODE(before.st_mode) != 0o444
                or before.st_nlink != 1
            ):
                raise ValueError("immutable input must be one mode-0444 regular link")
            payload = _read_descriptor(descriptor, before.st_size)
            after = os.fstat(descriptor)
            observed = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            if not _same_stat(before, after) or not _same_stat(after, observed):
                raise ValueError("immutable input changed while reading")
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)

    digest = _sha256(payload)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("immutable input SHA-256 mismatch")
    value = json.loads(
        payload,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if canonical_json_bytes(value) != payload:
        raise ValueError("immutable input is not canonical JSON")
    return value, _artifact_identity(path, after, payload)


def parse_nvidia_smi_output(output: str) -> dict[str, Any]:
    rows = [line for line in output.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one NVIDIA GPU row, found {len(rows)}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 4:
        raise ValueError("malformed nvidia-smi row")
    uuid, name, driver_version, minor_number = fields
    if GPU_UUID_PATTERN.fullmatch(uuid) is None:
        raise ValueError("malformed NVIDIA GPU UUID")
    if name != EXPECTED_NVIDIA_GPU_NAME:
        raise ValueError("NVIDIA GPU name mismatch")
    if driver_version != EXPECTED_NVIDIA_DRIVER_VERSION:
        raise ValueError("NVIDIA driver version mismatch")
    if DECIMAL_PATTERN.fullmatch(minor_number) is None:
        raise ValueError("malformed NVIDIA GPU minor number")
    return {
        "uuid": uuid,
        "name": name,
        "driver_version": driver_version,
        "minor_number": int(minor_number),
        "command": list(NVIDIA_SMI_COMMAND),
        "row_count": 1,
    }


def _run_nvidia_smi() -> str:
    completed = subprocess.run(
        NVIDIA_SMI_COMMAND,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.stderr:
        raise ValueError("nvidia-smi emitted unexpected stderr")
    return completed.stdout


def parse_cgroup_text(raw: str, *, job_id: str, step_id: str) -> dict[str, Any]:
    if not raw or not raw.endswith("\n"):
        raise ValueError("/proc/self/cgroup must be nonempty and newline terminated")
    records = []
    seen_lines = set()
    matching_paths = set()
    job_pattern = re.compile(rf"(?:^|/)job[_-]{re.escape(job_id)}(?:$|/)")
    step_pattern = re.compile(
        rf"(?:^|/)step[_-]{re.escape(step_id)}(?:\.scope)?(?:$|/)"
    )
    any_job_token = re.compile(r"(?:^|/)job[_-][0-9]+(?:$|/)")
    any_step_token = re.compile(
        r"(?:^|/)step[_-](?:[0-9]+|batch|extern)(?:\.scope)?(?:$|/)"
    )
    for line in raw.splitlines():
        if line in seen_lines:
            raise ValueError("duplicate cgroup record")
        seen_lines.add(line)
        fields = line.split(":", 2)
        if len(fields) != 3 or DECIMAL_PATTERN.fullmatch(fields[0]) is None:
            raise ValueError("malformed cgroup record")
        hierarchy, controllers, path = fields
        pure = PurePosixPath(path)
        if not path.startswith("/") or pure.as_posix() != path or ".." in pure.parts:
            raise ValueError("noncanonical cgroup path")
        job_match = job_pattern.search(path) is not None
        step_match = step_pattern.search(path) is not None
        if (any_job_token.search(path) is not None) != job_match:
            raise ValueError("cgroup job identity mismatch")
        if (any_step_token.search(path) is not None) != step_match:
            raise ValueError("cgroup step identity mismatch")
        if job_match != step_match:
            raise ValueError("cgroup job/step evidence is incomplete")
        if job_match:
            matching_paths.add(path)
        records.append(
            {
                "hierarchy": int(hierarchy),
                "controllers": controllers.split(",") if controllers else [],
                "path": path,
            }
        )
    if len(matching_paths) != 1:
        raise ValueError(
            f"expected exactly one cgroup job/step path, found {len(matching_paths)}"
        )
    return {
        "raw_sha256": _sha256(raw.encode("utf-8")),
        "records": records,
        "job_step_path": next(iter(matching_paths)),
    }


def capture_device_nodes() -> list[dict[str, Any]]:
    records = []
    for path in sorted(Path("/dev").glob("nvidia*"), key=lambda item: item.name):
        metadata = path.lstat()
        records.append(
            {
                "path": str(path),
                "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
                "file_type": "character" if stat.S_ISCHR(metadata.st_mode) else "other",
                "device_major": os.major(metadata.st_rdev)
                if stat.S_ISCHR(metadata.st_mode)
                else None,
                "device_minor": os.minor(metadata.st_rdev)
                if stat.S_ISCHR(metadata.st_mode)
                else None,
            }
        )
    return records


def validate_device_nodes(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_minor_number: int,
) -> dict[str, Any]:
    canonical = []
    physical = []
    seen_paths = set()
    for raw in records:
        if set(raw) != {
            "path",
            "mode",
            "file_type",
            "device_major",
            "device_minor",
        }:
            raise ValueError("device-node schema mismatch")
        path = raw["path"]
        if not isinstance(path, str) or path in seen_paths:
            raise ValueError("duplicate or malformed NVIDIA device path")
        seen_paths.add(path)
        match = re.fullmatch(r"/dev/nvidia([0-9]+)", path)
        record = dict(raw)
        canonical.append(record)
        if match is not None:
            if (
                record["file_type"] != "character"
                or record["device_major"] != 195
                or record["device_minor"] != int(match.group(1))
            ):
                raise ValueError("malformed physical NVIDIA device node")
            physical.append(record)
    if len(physical) != 1:
        raise ValueError(
            f"expected exactly one physical NVIDIA device node, found {len(physical)}"
        )
    if physical[0]["device_minor"] != expected_minor_number:
        raise ValueError("NVIDIA device node does not match nvidia-smi minor number")
    return {"all": canonical, "physical": physical, "physical_count": 1}


def _required_environment(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing required environment variable: {name}")
    return value


def _validate_source_environment(environ: Mapping[str, str]) -> dict[str, str]:
    values = {
        name: _required_environment(environ, name)
        for name in SOURCE_IDENTITY_ENVIRONMENT
    }
    if (
        SHA256_PATTERN.fullmatch(values["BATCH_VERIFIED_POLARIS_SOURCE_TREE_SHA256"])
        is None
    ):
        raise ValueError("invalid source-tree SHA-256")
    if SHA256_PATTERN.fullmatch(values["SOURCE_APPROVAL_SHA256"]) is None:
        raise ValueError("invalid source-approval SHA-256")
    if re.fullmatch(r"[0-9a-f]{40}", values["POLARIS_IMPLEMENTATION_COMMIT"]) is None:
        raise ValueError("invalid implementation commit")
    return values


def validate_target_argv(
    argv: Sequence[str],
    *,
    preexec_path: Path,
    preclose_path: Path,
    expected_gpu_uuid: str,
) -> list[str]:
    target = list(argv)
    if len(target) < 3 or target[:2] != ["/.venv/bin/python", "scripts/eval.py"]:
        raise ValueError("diagnostic target is not the public evaluator entrypoint")
    expected_pairs = {
        "--startup-diagnostic": STARTUP_DIAGNOSTIC_MODE,
        "--startup-diagnostic-preexec-path": str(preexec_path),
        "--startup-diagnostic-preclose-path": str(preclose_path),
        "--startup-diagnostic-expected-gpu-uuid": expected_gpu_uuid,
    }
    for option, expected in expected_pairs.items():
        positions = [index for index, value in enumerate(target) if value == option]
        if len(positions) != 1 or positions[0] + 1 >= len(target):
            raise ValueError(f"diagnostic target must contain exactly one {option}")
        if target[positions[0] + 1] != expected:
            raise ValueError(f"diagnostic target {option} value mismatch")
    return target


def python_argv_after_exec(target_argv: Sequence[str]) -> list[str]:
    """Return Python's real ``sys.argv`` for the approved exec target."""

    target = list(target_argv)
    if len(target) < 2 or target[0] != "/.venv/bin/python":
        raise ValueError("invalid Python exec target")
    return target[1:]


def capture_runtime_context(
    *,
    python_argv: Sequence[str],
    source_root: Path,
    expected_gpu_uuid: str,
    environ: Mapping[str, str] | None = None,
    nvidia_smi_output: str | None = None,
    cgroup_text: str | None = None,
    device_nodes: Sequence[Mapping[str, Any]] | None = None,
    pid: int | None = None,
    ppid: int | None = None,
    executable: str | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    environment = os.environ if environ is None else environ
    if GPU_UUID_PATTERN.fullmatch(expected_gpu_uuid) is None:
        raise ValueError("invalid expected batch GPU UUID")
    job_id = _required_environment(environment, "SLURM_JOB_ID")
    step_id = _required_environment(environment, "SLURM_STEP_ID")
    if DECIMAL_PATTERN.fullmatch(job_id) is None or job_id == "0":
        raise ValueError("SLURM_JOB_ID must be one positive decimal job ID")
    if DECIMAL_PATTERN.fullmatch(step_id) is None:
        raise ValueError("SLURM_STEP_ID must be one numeric srun step ID")
    job_gpus = _required_environment(environment, "SLURM_JOB_GPUS")
    step_gpus = _required_environment(environment, "SLURM_STEP_GPUS")
    if (
        DECIMAL_PATTERN.fullmatch(job_gpus) is None
        or DECIMAL_PATTERN.fullmatch(step_gpus) is None
        or job_gpus != step_gpus
    ):
        raise ValueError("Slurm job/step GPU indices must be one matching index")
    visible_gpu = _required_environment(environment, "NVIDIA_VISIBLE_DEVICES")
    if visible_gpu != expected_gpu_uuid:
        raise ValueError("NVIDIA_VISIBLE_DEVICES differs from the batch GPU UUID")
    nvidia = parse_nvidia_smi_output(
        _run_nvidia_smi() if nvidia_smi_output is None else nvidia_smi_output
    )
    if nvidia["uuid"] != expected_gpu_uuid:
        raise ValueError("nvidia-smi GPU UUID differs from the batch GPU UUID")
    if int(job_gpus) != nvidia["minor_number"]:
        raise ValueError("Slurm GPU index differs from the NVIDIA device minor number")
    if cgroup_text is None:
        cgroup_text = Path("/proc/self/cgroup").read_text(encoding="utf-8")
    cgroup = parse_cgroup_text(cgroup_text, job_id=job_id, step_id=step_id)
    nodes = validate_device_nodes(
        capture_device_nodes() if device_nodes is None else device_nodes,
        expected_minor_number=nvidia["minor_number"],
    )
    source_root = source_root.resolve(strict=True)
    if not source_root.is_dir() or source_root.is_symlink():
        raise ValueError("source root must be one canonical real directory")
    script_path = source_root / "scripts/eval.py"
    module_path = Path(__file__).resolve(strict=True)
    expected_module_path = (
        source_root / "src/polaris/app_launcher_startup_diagnostic.py"
    )
    if module_path != expected_module_path:
        raise ValueError("startup diagnostic module escaped the approved source root")
    source = {
        "root": str(source_root),
        "eval_script": stable_file_identity(script_path),
        "diagnostic_module": stable_file_identity(module_path),
        "approval": _validate_source_environment(environment),
    }
    current_cwd = Path.cwd() if cwd is None else cwd
    current_executable = sys.executable if executable is None else executable
    if current_cwd.resolve(strict=True) != source_root:
        raise ValueError("startup diagnostic process cwd differs from the source root")
    if current_executable != "/.venv/bin/python":
        raise ValueError("startup diagnostic interpreter is not /.venv/bin/python")
    selected_gpu_environment = {
        name: environment.get(name) for name in PREEXEC_GPU_ENVIRONMENT
    }
    return {
        "process": {
            "pid": os.getpid() if pid is None else pid,
            "ppid": os.getppid() if ppid is None else ppid,
            "executable": current_executable,
            "cwd": str(current_cwd),
            "python_argv": list(python_argv),
        },
        "slurm": {
            "job_id": int(job_id),
            "step_id": int(step_id),
            "job_gpu_index": int(job_gpus),
            "step_gpu_index": int(step_gpus),
            "gpu_environment": selected_gpu_environment,
        },
        "nvidia_smi": nvidia,
        "cgroup": cgroup,
        "device_nodes": nodes,
        "source": source,
    }


def build_closed_eval_environment(
    *,
    inherited: Mapping[str, str],
    source_root: Path,
    data_root: Path,
    cache_root: Path,
    preexec_sha256: str,
) -> dict[str, str]:
    if SHA256_PATTERN.fullmatch(preexec_sha256) is None:
        raise ValueError("invalid pre-exec artifact SHA-256")
    source_identity = _validate_source_environment(inherited)
    job_id = _required_environment(inherited, "SLURM_JOB_ID")
    step_id = _required_environment(inherited, "SLURM_STEP_ID")
    visible_gpu = _required_environment(inherited, "NVIDIA_VISIBLE_DEVICES")
    driver_capabilities = _required_environment(inherited, "NVIDIA_DRIVER_CAPABILITIES")
    if DECIMAL_PATTERN.fullmatch(job_id) is None or job_id == "0":
        raise ValueError("invalid inherited Slurm job ID")
    if DECIMAL_PATTERN.fullmatch(step_id) is None:
        raise ValueError("invalid inherited Slurm step ID")
    if GPU_UUID_PATTERN.fullmatch(visible_gpu) is None:
        raise ValueError("invalid inherited NVIDIA_VISIBLE_DEVICES")
    job_gpus = _required_environment(inherited, "SLURM_JOB_GPUS")
    step_gpus = _required_environment(inherited, "SLURM_STEP_GPUS")
    if (
        DECIMAL_PATTERN.fullmatch(job_gpus) is None
        or DECIMAL_PATTERN.fullmatch(step_gpus) is None
        or job_gpus != step_gpus
    ):
        raise ValueError("invalid inherited Slurm job/step GPU indices")
    return {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "NVIDIA_VISIBLE_DEVICES": visible_gpu,
        "NVIDIA_DRIVER_CAPABILITIES": driver_capabilities,
        "VK_DRIVER_FILES": "/etc/vulkan/icd.d/nvidia_icd.json",
        "ACCEPT_EULA": "Y",
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "PRIVACY_CONSENT": "Y",
        "OMNI_KIT_ALLOW_ROOT": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": (
            f"{source_root}/src:"
            f"{source_root}/third_party/openpi/packages/openpi-client/src"
        ),
        "POLARIS_DATA_PATH": str(data_root),
        "XDG_CACHE_HOME": str(cache_root),
        "HF_HOME": str(cache_root / "huggingface"),
        "HOME": str(cache_root / "home"),
        "SLURM_JOB_ID": job_id,
        "SLURM_STEP_ID": step_id,
        "SLURM_JOB_GPUS": job_gpus,
        "SLURM_STEP_GPUS": step_gpus,
        "POLARIS_STARTUP_DIAGNOSTIC_PREEXEC_SHA256": preexec_sha256,
        **source_identity,
    }


def _forbidden_loaded_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in FORBIDDEN_MODULE_PREFIXES
        )
    )


def _validate_context_continuity(
    preexec: Mapping[str, Any],
    current: Mapping[str, Any],
) -> None:
    expected_fields = {
        "schema_version",
        "profile",
        "status",
        "startup_diagnostic",
        "runtime",
        "launcher_argv",
        "target_argv",
        "zero_work_counters",
        "bounded_diagnostic_counts",
    }
    if set(preexec) != expected_fields:
        raise ValueError("pre-exec artifact schema mismatch")
    if (
        preexec.get("schema_version") != 1
        or preexec.get("profile") != PREEXEC_PROFILE
        or preexec.get("status") != "captured_before_public_eval_exec"
        or preexec.get("startup_diagnostic") != STARTUP_DIAGNOSTIC_MODE
        or preexec.get("zero_work_counters") != ZERO_WORK_COUNTERS
        or preexec.get("bounded_diagnostic_counts")
        != {
            "nvidia_smi_invocations": 1,
            "preexec_artifacts": 1,
            "preclose_artifacts": 0,
            "ready_artifacts": 0,
            "simulation_app_close_calls": 0,
        }
    ):
        raise ValueError("pre-exec artifact profile mismatch")
    launcher_argv = preexec.get("launcher_argv")
    target_argv = preexec.get("target_argv")
    if (
        not isinstance(launcher_argv, list)
        or not launcher_argv
        or not all(isinstance(value, str) for value in launcher_argv)
        or not isinstance(target_argv, list)
        or not all(isinstance(value, str) for value in target_argv)
    ):
        raise ValueError("pre-exec argv schema mismatch")
    before = preexec.get("runtime")
    if not isinstance(before, dict):
        raise ValueError("pre-exec runtime record is missing")
    if before.get("process", {}).get("python_argv") != python_argv_after_exec(
        target_argv
    ):
        raise ValueError("pre-exec target/Python argv binding mismatch")
    for path in (
        ("process", "pid"),
        ("process", "ppid"),
        ("process", "executable"),
        ("process", "cwd"),
        ("process", "python_argv"),
        ("slurm", "job_id"),
        ("slurm", "step_id"),
        ("slurm", "job_gpu_index"),
        ("slurm", "step_gpu_index"),
        ("nvidia_smi", "uuid"),
        ("nvidia_smi", "name"),
        ("nvidia_smi", "driver_version"),
        ("nvidia_smi", "minor_number"),
        ("cgroup", "raw_sha256"),
        ("cgroup", "job_step_path"),
        ("source", "root"),
        ("source", "eval_script", "sha256"),
        ("source", "diagnostic_module", "sha256"),
        ("source", "approval"),
        ("device_nodes", "physical"),
    ):
        left: Any = before
        right: Any = current
        for component in path:
            if not isinstance(left, dict) or not isinstance(right, dict):
                raise ValueError(f"runtime continuity schema mismatch at {path}")
            left = left.get(component)
            right = right.get(component)
        if left != right:
            raise ValueError(f"runtime continuity mismatch at {'.'.join(path)}")


def ready_path_for(preclose_path: Path) -> Path:
    return preclose_path.with_name(f"{preclose_path.stem}.ready{preclose_path.suffix}")


def run_app_launcher_only_diagnostic(
    *,
    simulation_app: Any,
    preexec_path: Path,
    preclose_path: Path,
    expected_gpu_uuid: str,
) -> dict[str, Any]:
    """Finalize the policy-free branch and close the SimulationApp exactly once."""

    expected_preexec_sha256 = _required_environment(
        os.environ, "POLARIS_STARTUP_DIAGNOSTIC_PREEXEC_SHA256"
    )
    if SHA256_PATTERN.fullmatch(expected_preexec_sha256) is None:
        raise ValueError("invalid pre-exec digest handoff")
    preexec, preexec_identity = stable_read_immutable_json(
        preexec_path,
        expected_sha256=expected_preexec_sha256,
    )
    source_root = Path(preexec["runtime"]["source"]["root"])
    current = capture_runtime_context(
        python_argv=sys.argv,
        source_root=source_root,
        expected_gpu_uuid=expected_gpu_uuid,
    )
    _validate_context_continuity(preexec, current)
    forbidden_modules = _forbidden_loaded_modules()
    if forbidden_modules:
        raise RuntimeError(
            "policy-free AppLauncher boundary imported forbidden modules: "
            + ",".join(forbidden_modules)
        )
    preclose_value = {
        "schema_version": 1,
        "profile": PRECLOSE_PROFILE,
        "status": "simulation_app_close_pending",
        "startup_diagnostic": STARTUP_DIAGNOSTIC_MODE,
        "preexec": preexec_identity,
        "runtime": current,
        "forbidden_module_prefixes": list(FORBIDDEN_MODULE_PREFIXES),
        "forbidden_loaded_modules": [],
        "zero_work_counters": dict(ZERO_WORK_COUNTERS),
        "bounded_diagnostic_counts": {
            "nvidia_smi_invocations": 2,
            "preexec_artifacts": 1,
            "preclose_artifacts": 1,
            "ready_artifacts": 0,
            "simulation_app_close_calls": 0,
        },
    }
    preclose_identity = publish_immutable_json(preclose_path, preclose_value)
    ready_path = ready_path_for(preclose_path)
    ready_value = {
        "schema_version": 1,
        "profile": READY_PROFILE,
        "status": "ready_for_simulation_app_close",
        "startup_diagnostic": STARTUP_DIAGNOSTIC_MODE,
        "preexec": preexec_identity,
        "preclose": preclose_identity,
        "zero_work_counters": dict(ZERO_WORK_COUNTERS),
        "bounded_diagnostic_counts": {
            "nvidia_smi_invocations": 2,
            "preexec_artifacts": 1,
            "preclose_artifacts": 1,
            "ready_artifacts": 1,
            "simulation_app_close_calls": 0,
        },
    }
    ready_identity = publish_immutable_json(ready_path, ready_value)
    try:
        simulation_app.close()
    except BaseException as error:
        print(
            "POLARIS_STARTUP_DIAGNOSTIC_CLOSE_ERROR="
            f"{type(error).__module__}.{type(error).__qualname__}",
            file=sys.stderr,
            flush=True,
        )
        raise RuntimeError(
            "AppLauncher diagnostic SimulationApp.close() failed"
        ) from error
    return {
        "preexec": preexec_identity,
        "preclose": preclose_identity,
        "ready": ready_identity,
    }


def prepare_public_eval_exec(
    *,
    target_argv: Sequence[str],
    preexec_path: Path,
    preclose_path: Path,
    expected_gpu_uuid: str,
    source_root: Path,
    data_root: Path,
    cache_root: Path,
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    target = validate_target_argv(
        target_argv,
        preexec_path=preexec_path,
        preclose_path=preclose_path,
        expected_gpu_uuid=expected_gpu_uuid,
    )
    runtime = capture_runtime_context(
        python_argv=python_argv_after_exec(target),
        source_root=source_root,
        expected_gpu_uuid=expected_gpu_uuid,
    )
    preexec_value = {
        "schema_version": 1,
        "profile": PREEXEC_PROFILE,
        "status": "captured_before_public_eval_exec",
        "startup_diagnostic": STARTUP_DIAGNOSTIC_MODE,
        "runtime": runtime,
        "launcher_argv": list(sys.argv),
        "target_argv": target,
        "zero_work_counters": dict(ZERO_WORK_COUNTERS),
        "bounded_diagnostic_counts": {
            "nvidia_smi_invocations": 1,
            "preexec_artifacts": 1,
            "preclose_artifacts": 0,
            "ready_artifacts": 0,
            "simulation_app_close_calls": 0,
        },
    }
    preexec_identity = publish_immutable_json(preexec_path, preexec_value)
    closed_environment = build_closed_eval_environment(
        inherited=os.environ,
        source_root=source_root,
        data_root=data_root,
        cache_root=cache_root,
        preexec_sha256=preexec_identity["sha256"],
    )
    return target, closed_environment, preexec_identity


def _parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a policy-free public evaluator pre-exec boundary"
    )
    parser.add_argument("--preexec-output", type=Path, required=True)
    parser.add_argument("--preclose-output", type=Path, required=True)
    parser.add_argument("--expected-batch-gpu-uuid", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("target", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.target[:1] == ["--"]:
        args.target = args.target[1:]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_cli(argv)
    target, environment, _ = prepare_public_eval_exec(
        target_argv=args.target,
        preexec_path=args.preexec_output,
        preclose_path=args.preclose_output,
        expected_gpu_uuid=args.expected_batch_gpu_uuid,
        source_root=args.source_root,
        data_root=args.data_root,
        cache_root=args.cache_root,
    )
    os.execve(target[0], target, environment)
    raise AssertionError("os.execve unexpectedly returned")


if __name__ == "__main__":
    raise SystemExit(main())
