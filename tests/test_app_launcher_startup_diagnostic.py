from __future__ import annotations

import argparse
import hashlib
import json
import io
import importlib.util
import os
from pathlib import Path
import signal
import shlex
import stat
import subprocess
import sys
import threading
import time

import pytest

from polaris import app_launcher_startup_diagnostic as diagnostic
from polaris import pi05_droid_jointpos_scheduler as job_scheduler


ROOT = Path(__file__).parents[1]
APP_FINALIZER_PATH = ROOT / "scripts/polaris/finalize_pi05_app_launcher_only.py"
GPU_UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"
SOURCE_TREE_SHA256 = "1" * 64
SOURCE_APPROVAL_SHA256 = "2" * 64
IMPLEMENTATION_COMMIT = "3" * 40
SUBMISSION_TRANSACTION_ID = "pi05-0123456789abcdef0123456789abcdef01234567"
EXPECTED_SCONTROL_SHA256 = "4" * 64
EXPECTED_SCONTROL_SIZE = 880968
EXPECTED_SLURM_LIBRARY_SHA256 = "5" * 64
EXPECTED_SLURM_LIBRARY_SIZE = 9886048
EXPECTED_SLURM_CONFIG_SHA256 = "6" * 64
RUNTIME_CLOSURE_APPROVAL_SHA256 = "7" * 64
PYXIS_IMAGE_SHA256 = "8" * 64
EXPECTED_SACCT_SHA256 = "9" * 64
EXPECTED_SACCT_SIZE = 101
EXPECTED_SCANCEL_SHA256 = "a" * 64
EXPECTED_SCANCEL_SIZE = 102
EXPECTED_SRUN_SHA256 = "b" * 64
EXPECTED_SRUN_SIZE = 103

VALID_PHASE_LOG = b"".join(
    f"POLARIS_EVAL_PHASE={phase}\n".encode()
    for phase in (
        "before_app_launcher",
        "after_app_launcher",
        "before_app_launcher_diagnostic_close",
        "after_app_launcher_diagnostic_close",
    )
)


def runtime_environment(**updates: str) -> dict[str, str]:
    environment = {
        "SLURM_JOB_ID": "12345",
        "SLURM_STEP_ID": "0",
        "SLURM_JOB_GPUS": "2",
        "SLURM_STEP_GPUS": "2",
        "SLURM_GPUS_ON_NODE": "1",
        "SLURM_GPUS_PER_TASK": "1",
        "SLURM_TRES_PER_TASK": "cpu=16,gres/gpu:1",
        "SLURM_CPUS_PER_TASK": "16",
        "SLURM_NTASKS": "1",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_MEM_PER_NODE": "131072",
        "SLURM_JOB_ACCOUNT": "nvr_lpr_rvp",
        "SLURM_JOB_PARTITION": "batch",
        "SLURM_JOB_QOS": "normal",
        "SLURM_JOB_USER": "lzha",
        "SUBMISSION_TRANSACTION_ID": SUBMISSION_TRANSACTION_ID,
        "POLARIS_EVAL_MODE": "app_launcher_only",
        "CUDA_VISIBLE_DEVICES": "0",
        "NVIDIA_VISIBLE_DEVICES": GPU_UUID,
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        "BATCH_VERIFIED_POLARIS_SOURCE_TREE_SHA256": SOURCE_TREE_SHA256,
        "SOURCE_APPROVAL_SHA256": SOURCE_APPROVAL_SHA256,
        "POLARIS_IMPLEMENTATION_COMMIT": IMPLEMENTATION_COMMIT,
        "POLARIS_EXPECTED_SCONTROL_SHA256": EXPECTED_SCONTROL_SHA256,
        "POLARIS_EXPECTED_SCONTROL_SIZE": str(EXPECTED_SCONTROL_SIZE),
        "POLARIS_EXPECTED_SLURM_LIBRARY_SHA256": EXPECTED_SLURM_LIBRARY_SHA256,
        "POLARIS_EXPECTED_SLURM_LIBRARY_SIZE": str(EXPECTED_SLURM_LIBRARY_SIZE),
        "POLARIS_EXPECTED_SLURM_CONFIG_SHA256": EXPECTED_SLURM_CONFIG_SHA256,
        "POLARIS_EXPECTED_SACCT_SHA256": EXPECTED_SACCT_SHA256,
        "POLARIS_EXPECTED_SACCT_SIZE": str(EXPECTED_SACCT_SIZE),
        "POLARIS_EXPECTED_SCANCEL_SHA256": EXPECTED_SCANCEL_SHA256,
        "POLARIS_EXPECTED_SCANCEL_SIZE": str(EXPECTED_SCANCEL_SIZE),
        "POLARIS_EXPECTED_SRUN_SHA256": EXPECTED_SRUN_SHA256,
        "POLARIS_EXPECTED_SRUN_SIZE": str(EXPECTED_SRUN_SIZE),
        "POLARIS_RUNTIME_CLOSURE_APPROVAL_SHA256": (RUNTIME_CLOSURE_APPROVAL_SHA256),
        "POLARIS_PYXIS_IMAGE_PATH": "/synthetic/polaris.sqsh",
        "POLARIS_EXPECTED_PYXIS_IMAGE_SHA256": PYXIS_IMAGE_SHA256,
        "POLARIS_OBSERVED_PYXIS_IMAGE_SHA256": PYXIS_IMAGE_SHA256,
        "POLARIS_OBSERVED_PYXIS_IMAGE_MODE": "0444",
        "POLARIS_OBSERVED_PYXIS_IMAGE_NLINK": "1",
        "POLARIS_OBSERVED_PYXIS_IMAGE_SIZE": "1",
    }
    environment.update(updates)
    return environment


@pytest.fixture(autouse=True)
def trusted_helper_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in runtime_environment().items():
        monkeypatch.setenv(name, value)


def scheduler_approval_kwargs() -> dict[str, object]:
    return {
        "expected_slurm_config_sha256": EXPECTED_SLURM_CONFIG_SHA256,
        "runtime_closure_approval_sha256": RUNTIME_CLOSURE_APPROVAL_SHA256,
        "expected_scontrol_sha256": EXPECTED_SCONTROL_SHA256,
        "expected_scontrol_size": EXPECTED_SCONTROL_SIZE,
        "expected_slurm_library_sha256": EXPECTED_SLURM_LIBRARY_SHA256,
        "expected_slurm_library_size": EXPECTED_SLURM_LIBRARY_SIZE,
        "expected_sacct_sha256": EXPECTED_SACCT_SHA256,
        "expected_sacct_size": EXPECTED_SACCT_SIZE,
        "expected_scancel_sha256": EXPECTED_SCANCEL_SHA256,
        "expected_scancel_size": EXPECTED_SCANCEL_SIZE,
        "expected_srun_sha256": EXPECTED_SRUN_SHA256,
        "expected_srun_size": EXPECTED_SRUN_SIZE,
    }


def job_scheduler_record(
    *, requested_gpus: int = 1, allocated_gpus: int = 1, cpus: int = 16
) -> str:
    return (
        "JobId=12345 JobName=diagnostic UserId=lzha(158351) "
        "Account=nvr_lpr_rvp QOS=normal JobState=RUNNING Reason=None Partition=batch "
        f"Requeue=0 Restarts=0 Comment={SUBMISSION_TRANSACTION_ID} "
        "NodeList=pool0-00002 BatchHost=pool0-00002 NumNodes=1 "
        f"NumCPUs={cpus} NumTasks=1 CPUs/Task=16 OverSubscribe=OK "
        "TresPerNode=gres/gpu:1 TresPerTask=cpu=16 "
        f"ReqTRES=billing=1,cpu={cpus},gres/gpu={requested_gpus},mem=128G,node=1 "
        f"AllocTRES=billing=1,cpu={cpus},gres/gpu={allocated_gpus},mem=128G,node=1"
    )


def step_scheduler_record(*, gpus: int = 1, cpus: int = 16) -> str:
    return (
        "StepId=12345.0 Name=python UserId=lzha(158351) State=RUNNING "
        "Partition=batch Nodes=1 NodeList=pool0-00002 "
        f"CPUs={cpus} Tasks=1 "
        f"TRES=cpu={cpus},gres/gpu={gpus},mem=128G,node=1"
    )


def synthetic_scheduler_client() -> dict[str, object]:
    return {
        "profile": "polaris_approved_scontrol_24_11_v2",
        "runtime_closure_approval_sha256": RUNTIME_CLOSURE_APPROVAL_SHA256,
        "scontrol": {
            "path": str(diagnostic.EXPECTED_SCONTROL_PATH),
            "mode": "0755",
            "nlink": 1,
            "size": EXPECTED_SCONTROL_SIZE,
            "sha256": EXPECTED_SCONTROL_SHA256,
        },
        "sacct": {
            "path": str(diagnostic.EXPECTED_SACCT_PATH),
            "mode": "0755",
            "nlink": 1,
            "size": EXPECTED_SACCT_SIZE,
            "sha256": EXPECTED_SACCT_SHA256,
        },
        "scancel": {
            "path": str(diagnostic.EXPECTED_SCANCEL_PATH),
            "mode": "0755",
            "nlink": 1,
            "size": EXPECTED_SCANCEL_SIZE,
            "sha256": EXPECTED_SCANCEL_SHA256,
        },
        "srun": {
            "path": str(diagnostic.EXPECTED_SRUN_PATH),
            "mode": "0755",
            "nlink": 1,
            "size": EXPECTED_SRUN_SIZE,
            "sha256": EXPECTED_SRUN_SHA256,
        },
        "slurm_library": {
            "path": str(diagnostic.EXPECTED_SLURM_LIBRARY_PATH),
            "mode": "0644",
            "nlink": 1,
            "size": EXPECTED_SLURM_LIBRARY_SIZE,
            "sha256": EXPECTED_SLURM_LIBRARY_SHA256,
        },
        "slurm_config": {
            "path": str(diagnostic.EXPECTED_SLURM_CONFIG_PATH),
            "mode": "0644",
            "nlink": 1,
            "size": 8671,
            "sha256": EXPECTED_SLURM_CONFIG_SHA256,
        },
        "execution_environment": {
            "PATH": "/usr/bin:/bin",
            "SLURM_CONF": str(diagnostic.EXPECTED_SLURM_CONFIG_PATH),
            "LD_LIBRARY_PATH": (
                "/cm/local/apps/slurm/24.11/lib64:"
                "/cm/local/apps/slurm/24.11/lib64/slurm"
            ),
        },
    }


def synthetic_scheduler_handoff(
    *, environment: dict[str, str] | None = None
) -> dict[str, object]:
    environment = runtime_environment() if environment is None else environment
    request = diagnostic.scheduler_request_value(
        environ=environment, expected_gpu_uuid=GPU_UUID
    )
    request_identity = {
        "path": str(ROOT / "scheduler_request.json"),
        "mode": "0444",
        "nlink": 1,
        "size": 1,
        "sha256": "5" * 64,
    }
    value = {
        "schema_version": 1,
        "profile": diagnostic.SCHEDULER_HANDOFF_PROFILE,
        "status": "host_scheduler_records_sealed",
        "request": request_identity,
        "request_value": request,
        "scheduler_client": synthetic_scheduler_client(),
        "host_binding": {
            name: environment[name]
            for name in (
                "SLURM_JOB_ID",
                "SLURM_JOB_GPUS",
                "SLURM_GPUS_ON_NODE",
                "SLURM_JOB_ACCOUNT",
                "SLURM_JOB_PARTITION",
                "SLURM_JOB_QOS",
                "SLURM_JOB_USER",
                "NVIDIA_VISIBLE_DEVICES",
                "SUBMISSION_TRANSACTION_ID",
            )
        },
        "job_record": diagnostic.parse_job_scheduler_record(
            job_scheduler_record(),
            expected_job_id=12345,
            expected_transaction_id=SUBMISSION_TRANSACTION_ID,
        ),
        "step_record": diagnostic.parse_step_scheduler_record(
            step_scheduler_record(),
            expected_job_id=12345,
            expected_step_id=0,
            requested_tres_per_task=environment["SLURM_TRES_PER_TASK"],
            expected_node="pool0-00002",
        ),
    }
    return {
        "artifact": {
            "path": str(ROOT / "scheduler_handoff.json"),
            "mode": "0444",
            "nlink": 1,
            "size": 1,
            "sha256": "6" * 64,
        },
        "value": value,
    }


def nvidia_smi_output(*, uuid: str = GPU_UUID, minor: int = 2) -> str:
    return f"{uuid}, NVIDIA L40S, 580.105.08, {minor}\n"


def cgroup_text(*, job_id: str = "12345", step_id: str = "0") -> str:
    return f"0::/slurm/uid_1000/job_{job_id}/step_{step_id}/task_0\n"


def physical_device_node(*, index: int = 2, minor: int = 2) -> dict[str, object]:
    return {
        "path": f"/dev/nvidia{index}",
        "mode": "0660",
        "file_type": "character",
        "device_major": 195,
        "device_minor": minor,
    }


def synthetic_output_directories() -> dict[str, str]:
    return {
        "namespace_parent": "/synthetic",
        "namespace_parent_identity": "1:1:3:4:0755",
        "run_dir": "/synthetic/run",
        "run_identity": "1:2:3:4:0755",
        "task_dir": "/synthetic/run/app_launcher_only",
        "task_identity": "1:3:3:4:0755",
    }


def create_output_tree(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    run_dir = tmp_path / "run"
    task_dir = run_dir / "app_launcher_only"
    namespace_parent_identity = diagnostic.capture_directory_identity(tmp_path)
    output_directories = diagnostic.create_output_directories(
        run_dir=run_dir,
        task_dir=task_dir,
        expected_parent_identity=namespace_parent_identity,
    )
    return run_dir, task_dir, output_directories


def publish_success_evidence(
    task_dir: Path,
    task_identity: str,
    *,
    log_payload: bytes = VALID_PHASE_LOG,
    environment: dict[str, str] | None = None,
) -> None:
    run_dir = task_dir.parent
    output_directories = {
        "namespace_parent": str(run_dir.parent),
        "namespace_parent_identity": diagnostic.capture_directory_identity(
            run_dir.parent
        ),
        "run_dir": str(run_dir),
        "run_identity": diagnostic.capture_directory_identity(run_dir),
        "task_dir": str(task_dir),
        "task_identity": task_identity,
    }
    environment = runtime_environment() if environment is None else environment
    request_path = task_dir / "scheduler_request.json"
    handoff_path = task_dir / "scheduler_handoff.json"
    request_identity = diagnostic.publish_scheduler_request(
        request_path,
        environ=environment,
        expected_gpu_uuid=GPU_UUID,
        expected_parent_identity=task_identity,
    )
    handoff_identity = diagnostic.seal_scheduler_handoff(
        request_path=request_path,
        output_path=handoff_path,
        job_record=job_scheduler_record(),
        step_record=step_scheduler_record(),
        scheduler_client=synthetic_scheduler_client(),
        host_environ=environment,
        expected_parent_identity=task_identity,
        **scheduler_approval_kwargs(),
    )
    handoff_value, observed_handoff_identity = diagnostic.stable_read_immutable_json(
        handoff_path,
        expected_parent_identity=task_identity,
    )
    assert observed_handoff_identity == handoff_identity
    target = canonical_target(task_dir)
    runtime = diagnostic.capture_runtime_context(
        python_argv=diagnostic.python_argv_after_exec(target),
        source_root=ROOT,
        expected_gpu_uuid=GPU_UUID,
        environ=environment,
        nvidia_smi_output=nvidia_smi_output(),
        cgroup_text=cgroup_text(),
        device_nodes=[physical_device_node()],
        pid=4444,
        ppid=3333,
        executable="/.venv/bin/python",
        cwd=ROOT,
        scheduler_handoff={
            "artifact": handoff_identity,
            "value": handoff_value,
        },
        output_directories=output_directories,
    )
    runtime["source"]["root"] = "/polaris-source"
    runtime["process"]["cwd"] = "/polaris-source"
    runtime["source"]["eval_script"]["path"] = "/polaris-source/scripts/eval.py"
    runtime["source"]["diagnostic_module"]["path"] = (
        "/polaris-source/src/polaris/app_launcher_startup_diagnostic.py"
    )
    preexec = preexec_value(runtime)
    preexec["target_argv"] = target
    preexec_identity = diagnostic.publish_immutable_json(
        task_dir / "startup_preexec.json",
        preexec,
        expected_parent_identity=task_identity,
    )
    preclose_value = {
        "schema_version": 1,
        "profile": diagnostic.PRECLOSE_PROFILE,
        "status": "simulation_app_close_pending",
        "startup_diagnostic": diagnostic.STARTUP_DIAGNOSTIC_MODE,
        "preexec": preexec_identity,
        "runtime": runtime,
        "forbidden_module_prefixes": list(diagnostic.FORBIDDEN_MODULE_PREFIXES),
        "forbidden_loaded_modules": [],
        "zero_work_counters": dict(diagnostic.ZERO_WORK_COUNTERS),
        "bounded_diagnostic_counts": {
            "nvidia_smi_invocations": 2,
            "scheduler_request_artifacts": 1,
            "scheduler_handoff_artifacts": 1,
            "job_scheduler_records": 1,
            "step_scheduler_records": 1,
            "preexec_artifacts": 1,
            "preclose_artifacts": 1,
            "ready_artifacts": 0,
            "simulation_app_close_calls": 0,
        },
    }
    preclose_identity = diagnostic.publish_immutable_json(
        task_dir / "startup_preclose.json",
        preclose_value,
        expected_parent_identity=task_identity,
    )
    diagnostic.publish_immutable_json(
        task_dir / "startup_preclose.ready.json",
        {
            "schema_version": 1,
            "profile": diagnostic.READY_PROFILE,
            "status": "ready_for_simulation_app_close",
            "startup_diagnostic": diagnostic.STARTUP_DIAGNOSTIC_MODE,
            "preexec": preexec_identity,
            "preclose": preclose_identity,
            "zero_work_counters": dict(diagnostic.ZERO_WORK_COUNTERS),
            "bounded_diagnostic_counts": {
                "nvidia_smi_invocations": 2,
                "scheduler_request_artifacts": 1,
                "scheduler_handoff_artifacts": 1,
                "job_scheduler_records": 1,
                "step_scheduler_records": 1,
                "preexec_artifacts": 1,
                "preclose_artifacts": 1,
                "ready_artifacts": 1,
                "simulation_app_close_calls": 0,
            },
        },
        expected_parent_identity=task_identity,
    )
    terminal_request_path = task_dir / "scheduler_terminal_request.json"
    terminal_request_identity = diagnostic.publish_scheduler_terminal_request(
        request_path=request_path,
        handoff_path=handoff_path,
        output_path=terminal_request_path,
        srun_exit_code=0,
        expected_parent_identity=task_identity,
    )
    request_value, _ = diagnostic.stable_read_immutable_json(request_path)
    live_cgroup_raw = "populated 1\n"
    terminal_cgroup_raw = "populated 0\n"
    cgroup_events_path = "/sys/fs/cgroup/slurm/uid_1000/job_12345/step_0/cgroup.events"
    sacct_raw = (
        "12345.0|COMPLETED|0:0|2026-07-09T00:00:00|"
        "2026-07-09T00:00:10|10|pool0-00002|\n"
    )
    diagnostic.publish_immutable_json(
        task_dir / "scheduler_terminal.json",
        {
            "schema_version": 1,
            "profile": diagnostic.SCHEDULER_TERMINAL_PROFILE,
            "status": "scheduler_step_terminal_and_cgroup_unpopulated",
            "request": request_identity,
            "handoff": handoff_identity,
            "terminal_request": terminal_request_identity,
            "scheduler_client": synthetic_scheduler_client(),
            "scontrol_absence": {
                "command": [
                    str(diagnostic.EXPECTED_SCONTROL_PATH),
                    "show",
                    "step",
                    "--oneliner",
                    "12345.0",
                ],
                "returncode": 1,
                "stdout": "",
                "stderr": (
                    "scontrol: error: scontrol_print_step: "
                    "slurm_get_job_steps(12345.0) failed: "
                    "Invalid job id specified\n"
                ),
            },
            "sacct_terminal": diagnostic.parse_sacct_terminal_record(
                sacct_raw,
                expected_job_id=request_value["job_id"],
                expected_step_id=request_value["step_id"],
                expected_node="pool0-00002",
            ),
            "cgroup_events_live": {
                "path": cgroup_events_path,
                "raw": live_cgroup_raw,
                "raw_sha256": diagnostic._sha256(live_cgroup_raw.encode()),
                "values": {"populated": 1},
            },
            "cgroup_events": {
                "path": cgroup_events_path,
                "raw": terminal_cgroup_raw,
                "raw_sha256": diagnostic._sha256(terminal_cgroup_raw.encode()),
                "values": {"populated": 0},
            },
            "scancel_invoked": False,
        },
        expected_parent_identity=task_identity,
    )
    diagnostic.capture_immutable_log(
        output_path=task_dir / "app_launcher_only.log",
        identity_path=task_dir / "app_launcher_only.log.identity.json",
        input_stream=io.BytesIO(log_payload),
        mirror_stream=io.BytesIO(),
        expected_parent_identity=task_identity,
    )


def runtime_file_record(
    path: Path, *, roles: list[str] | None = None
) -> dict[str, object]:
    path = path.resolve(strict=True)
    metadata = path.stat()
    value: dict[str, object] = {
        "path": str(path),
        "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "nlink": metadata.st_nlink,
        "size": metadata.st_size,
        "sha256": diagnostic._sha256(path.read_bytes()),
    }
    if roles is not None:
        value["roles"] = sorted(roles)
    return value


def publish_sacct_runtime_approval(
    tmp_path: Path,
    *,
    sacct_path: Path,
    slurm_config_path: Path,
    slurm_library_path: Path,
    surface_hostname: str | None = None,
) -> tuple[Path, str]:
    reviewer_identity = {
        "principal": "codex-agent:/root/synthetic_independent_reviewer",
        "profile": "polaris_external_sacct_runtime_reviewer_identity_v2",
        "role": "independent_agent_runtime_approval_reviewer",
    }
    machine_record = runtime_file_record(Path("/etc/machine-id"))
    immutable_files = sorted(
        (
            runtime_file_record(
                sacct_path,
                roles=["sacct_entrypoint"],
            ),
            runtime_file_record(
                slurm_config_path,
                roles=["approval_bound_configuration"],
            ),
            runtime_file_record(
                slurm_library_path,
                roles=[
                    "sacct_elf_dependency",
                    "sacct_slurm_plugin",
                    "sacct_slurm_runtime",
                ],
            ),
        ),
        key=lambda item: str(item["path"]),
    )
    closure = {
        "capture_scope": job_scheduler.SACCT_RUNTIME_CAPTURE_SCOPE,
        "execution_surface": {
            "hostname": (
                os.uname().nodename if surface_hostname is None else surface_hostname
            ),
            "machine_id_sha256": machine_record["sha256"],
            "kernel_release": os.uname().release,
            "architecture": os.uname().machine,
            "effective_uid": os.geteuid(),
            "effective_gid": os.getegid(),
        },
        "query_contract": {
            "profile": "polaris_external_sacct_query_v1",
            "command_template": job_scheduler._expected_sacct_query_command_template(),
            "environment": {
                "PATH": job_scheduler.SACCT_QUERY_PATH,
                "SLURM_CONF": str(job_scheduler.PINNED_SLURM_CONFIG_PATH),
                "LD_LIBRARY_PATH": job_scheduler.SACCT_QUERY_LD_LIBRARY_PATH,
            },
            "subprocess_timeout_seconds": (
                job_scheduler.SACCT_SUBPROCESS_TIMEOUT_SECONDS
            ),
        },
        "immutable_files": immutable_files,
        "symlink_bindings": [],
        "ambient_runtime_dependencies": {
            "regular_files": [machine_record],
            "sockets": [
                {
                    "path": str((tmp_path / "absent-slurm.sock").resolve()),
                    "present": False,
                    "mode": "0000",
                    "uid": 0,
                    "gid": 0,
                    "role": "slurm_accounting_socket",
                }
            ],
            "external_services": [
                {
                    "kind": "tcp_service",
                    "endpoint": "synthetic-slurmdbd:6819",
                    "role": "slurm_accounting",
                }
            ],
            "negative_resolution_assertions": [
                str((tmp_path / "absent-slurm-runtime").resolve())
            ],
        },
        "trust_boundary": {
            "closed_claim": (
                "reviewed_full_sacct_regular_file_symlink_and_declared_ambient_"
                "runtime_closure_v1"
            ),
            "trusted_but_unclosed": ["kernel"],
            "review_requirement": (
                "independent_agent_review_of_trace_candidate_and_capture_terminal"
            ),
            "reviewer_identity": reviewer_identity,
        },
    }
    closure_sha256 = diagnostic._sha256(job_scheduler._canonical_json_bytes(closure))
    candidate_value = {
        "schema_version": 2,
        "profile": job_scheduler.SACCT_RUNTIME_CANDIDATE_PROFILE,
        **closure,
        "trace_evidence": {},
    }
    candidate = diagnostic.publish_immutable_json(
        tmp_path / "external_sacct_runtime_candidate.json", candidate_value
    )
    capture_package = tmp_path / "capture-package"
    capture_package.mkdir()
    producer_source = capture_package / "finalize_runtime_trace_v2.py"
    producer_source.write_text(
        "# sealed synthetic capture producer\n", encoding="ascii"
    )
    producer_source.chmod(0o444)
    producer_source_identity = job_scheduler._identity(
        job_scheduler.validate_immutable_file(producer_source)
    )
    payload_sha256 = {
        producer_source.name: producer_source_identity["sha256"],
    }
    package_manifest = capture_package / "PACKAGE_MANIFEST.sha256"
    package_manifest.write_text(
        "".join(f"{payload_sha256[name]}  {name}\n" for name in sorted(payload_sha256)),
        encoding="ascii",
    )
    package_manifest.chmod(0o444)
    package_manifest_identity = job_scheduler._identity(
        job_scheduler.validate_immutable_file(package_manifest)
    )
    capture_package.chmod(0o555)
    package_metadata = capture_package.stat()
    producer_host = closure["execution_surface"]["hostname"]
    producer = {
        "profile": job_scheduler.SACCT_RUNTIME_CAPTURE_PRODUCER_PROFILE,
        "module": job_scheduler.SACCT_RUNTIME_CAPTURE_PRODUCER_MODULE,
        "path": producer_source_identity["path"],
        "sha256": producer_source_identity["sha256"],
        "uid": os.geteuid(),
        "host": producer_host,
    }
    capture_stdout_path = tmp_path / "capture_root_sacct.txt"
    capture_stdout_path.write_bytes(
        b"12345|COMPLETED|0:0|2026-07-09T01:00:00|"
        b"2026-07-09T01:01:00|2026-07-09T01:02:00|60|pool0-00002|0|\n"
    )
    capture_stdout_path.chmod(0o444)
    capture_stdout = job_scheduler.validate_immutable_file(capture_stdout_path)
    capture_terminal = diagnostic.publish_immutable_json(
        tmp_path / "external_sacct_runtime_capture_terminal.json",
        {
            "schema_version": 1,
            "profile": job_scheduler.SACCT_RUNTIME_CAPTURE_TERMINAL_PROFILE,
            "status": "captured_completed_root_sacct",
            "producer": producer,
            "package": {
                "root": {
                    "path": str(capture_package.resolve()),
                    "device": package_metadata.st_dev,
                    "inode": package_metadata.st_ino,
                    "mode": "0555",
                    "uid": package_metadata.st_uid,
                    "gid": package_metadata.st_gid,
                },
                "manifest": package_manifest_identity,
                "source": producer_source_identity,
                "payload_sha256": payload_sha256,
            },
            "job": {
                "job_id": 12345,
                "compute_node": "pool0-00002",
                "transaction_id": (
                    "polaris-runtime-trace-v2-01234567-89ab-cdef-0123-456789abcdef"
                ),
            },
            "query": {
                "profile": job_scheduler.SACCT_RUNTIME_CAPTURE_QUERY_PROFILE,
                "argv": [
                    token.replace("{job_id}", "12345")
                    for token in closure["query_contract"]["command_template"]
                ],
                "environment": closure["query_contract"]["environment"],
                "subprocess_timeout_seconds": (
                    job_scheduler.SACCT_SUBPROCESS_TIMEOUT_SECONDS
                ),
            },
            "surface": {
                **closure["execution_surface"],
                "capture_started_at": "2026-07-09T11:59:00Z",
                "capture_finished_at": "2026-07-09T11:59:59Z",
            },
            "sacct": {
                "executable": next(
                    item
                    for item in immutable_files
                    if item["path"] == str(sacct_path.resolve())
                ),
                "version": "slurm 24.11 synthetic",
                "runtime_closure": {
                    "profile": job_scheduler.SACCT_RUNTIME_CAPTURE_CLOSURE_PROFILE,
                    "before_sha256": closure_sha256,
                    "after_sha256": closure_sha256,
                    "identical": True,
                },
                "stdout": job_scheduler._identity(capture_stdout),
                "stderr": "",
                "returncode": 0,
                "parsed_root_row": {
                    "job_id": 12345,
                    "state": "COMPLETED",
                    "exit_code": "0:0",
                    "submit": "2026-07-09T01:00:00",
                    "start": "2026-07-09T01:01:00",
                    "end": "2026-07-09T01:02:00",
                    "elapsed_raw": 60,
                    "node": "pool0-00002",
                    "restarts": 0,
                    "raw_sha256": capture_stdout["sha256"],
                },
            },
            "dependency_census": job_scheduler._expected_dependency_census(
                {**closure}, closure_sha256=closure_sha256
            ),
            "candidate": job_scheduler._identity(candidate),
        },
    )
    approval_path = tmp_path / "external_sacct_runtime_approval.json"
    review_path = tmp_path / "external_sacct_runtime_review.json"
    review = diagnostic.publish_immutable_json(
        review_path,
        {
            "schema_version": 2,
            "profile": job_scheduler.SACCT_RUNTIME_REVIEW_PROFILE,
            "decision": "approve",
            "reviewer_identity": reviewer_identity,
            "review_scope": job_scheduler.SACCT_RUNTIME_REVIEW_SCOPE,
            "approved_at": "2026-07-09T12:00:00Z",
            "candidate_path": candidate["path"],
            "capture_terminal_path": capture_terminal["path"],
            "review_path": str(review_path.resolve()),
            "approval_path": str(approval_path.resolve()),
            "candidate_sha256": candidate["sha256"],
            "capture_terminal_sha256": capture_terminal["sha256"],
            "closure_sha256": closure_sha256,
        },
    )
    approval_value = {
        "schema_version": 2,
        "profile": job_scheduler.SACCT_RUNTIME_APPROVAL_PROFILE,
        **closure,
        "trace_evidence": {
            "candidate": job_scheduler._identity(candidate),
            "capture_terminal": job_scheduler._identity(capture_terminal),
            "review_decision": job_scheduler._identity(review),
            "closure_sha256": closure_sha256,
        },
    }
    approval = diagnostic.publish_immutable_json(approval_path, approval_value)
    return Path(approval["path"]), str(approval["sha256"])


def test_sacct_approval_validators_have_mutation_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_path = ROOT / ".codex_artifacts/runtime_trace_v2/finalize_runtime_trace_v2.py"
    if not trace_path.is_file():
        pytest.skip("external runtime-trace audit package is not present")
    specification = importlib.util.spec_from_file_location(
        "_polaris_runtime_trace_parity", trace_path
    )
    assert specification is not None and specification.loader is not None
    trace = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = trace
    previous_bytecode_policy = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(trace)
    finally:
        sys.dont_write_bytecode = previous_bytecode_policy
    assert not (trace_path.parent / "__pycache__").exists()

    sacct = tmp_path / "sacct"
    sacct.write_bytes(b"#!/usr/bin/bash\nexit 0\n")
    sacct.chmod(0o755)
    config = tmp_path / "slurm.conf"
    config.write_text("ClusterName=parity\n", encoding="ascii")
    config.chmod(0o644)
    library = tmp_path / "libslurmfull.so"
    library.write_bytes(b"parity-slurm-runtime")
    library.chmod(0o644)
    monkeypatch.setattr(job_scheduler, "PINNED_SACCT_PATH", sacct)
    monkeypatch.setattr(job_scheduler, "PINNED_SLURM_CONFIG_PATH", config)
    monkeypatch.setattr(job_scheduler, "PINNED_SLURM_LIBRARY_PATH", library)
    monkeypatch.setattr(job_scheduler, "SACCT_QUERY_LD_LIBRARY_PATH", str(tmp_path))
    approval_path, _ = publish_sacct_runtime_approval(
        tmp_path,
        sacct_path=sacct,
        slurm_config_path=config,
        slurm_library_path=library,
    )
    baseline = json.loads(approval_path.read_text(encoding="ascii"))
    candidate_path = Path(baseline["trace_evidence"]["candidate"]["path"])
    capture_path = Path(baseline["trace_evidence"]["capture_terminal"]["path"])
    review_path = Path(baseline["trace_evidence"]["review_decision"]["path"])
    originals = {
        path: path.read_bytes()
        for path in (approval_path, candidate_path, capture_path, review_path)
    }

    trace.SACCT = str(sacct)
    trace.SLURM_CONF = str(config)
    trace.SLURM_RUNTIME = str(library)
    trace.SLURM_LIBS = str(tmp_path)
    trace.SACCT_ARGV_TEMPLATE = job_scheduler._expected_sacct_query_command_template()
    trace.BROKER_ENV = {
        "PATH": job_scheduler.SACCT_QUERY_PATH,
        "SLURM_CONF": str(config),
        "LD_LIBRARY_PATH": str(tmp_path),
    }
    trace.QUERY_CONTRACT = {
        "profile": "polaris_external_sacct_query_v1",
        "command_template": trace.SACCT_ARGV_TEMPLATE,
        "environment": trace.BROKER_ENV,
        "subprocess_timeout_seconds": job_scheduler.SACCT_SUBPROCESS_TIMEOUT_SECONDS,
    }

    def rewrite(path: Path, value: dict[str, object]) -> None:
        path.chmod(0o644)
        path.write_bytes(job_scheduler._canonical_json_bytes(value))
        path.chmod(0o444)

    def restore() -> None:
        for path, payload in originals.items():
            path.chmod(0o644)
            path.write_bytes(payload)
            path.chmod(0o444)

    def source_accepts() -> bool:
        try:
            job_scheduler.validate_sacct_runtime_approval(
                approval_path,
                expected_sha256=hashlib.sha256(approval_path.read_bytes()).hexdigest(),
                live=True,
            )
        except ValueError:
            return False
        return True

    trace_errors: list[str] = []

    def trace_accepts() -> bool:
        try:
            trace.validate_external_approval(
                argparse.Namespace(
                    external_sacct_approval=str(approval_path),
                    expected_external_sacct_approval_sha256=hashlib.sha256(
                        approval_path.read_bytes()
                    ).hexdigest(),
                )
            )
        except ValueError as error:
            trace_errors.append(str(error))
            return False
        return True

    assert source_accepts() is True
    assert trace_accepts() is True, trace_errors

    def mutate_approval(mutator: object) -> None:
        value = json.loads(originals[approval_path])
        assert callable(mutator)
        mutator(value)
        rewrite(approval_path, value)

    def mutate_reference(reference: str, mutator: object) -> None:
        referenced_path = {
            "candidate": candidate_path,
            "capture_terminal": capture_path,
            "review_decision": review_path,
        }[reference]
        referenced = json.loads(originals[referenced_path])
        assert callable(mutator)
        mutator(referenced)
        rewrite(referenced_path, referenced)
        approval = json.loads(originals[approval_path])
        approval["trace_evidence"][reference] = job_scheduler._identity(
            job_scheduler.validate_immutable_file(referenced_path)
        )
        rewrite(approval_path, approval)

    unclosed_link = tmp_path / "unclosed-link"
    unclosed_link.symlink_to(approval_path)
    mutations = {
        "legacy_external_human_claim": lambda: mutate_approval(
            lambda value: value["trust_boundary"].update(
                review_requirement=(
                    "external_human_review_of_trace_candidate_and_terminal_attestation"
                )
            )
        ),
        "unpinned_reviewer": lambda: mutate_approval(
            lambda value: value["trust_boundary"]["reviewer_identity"].update(
                principal="UNPINNED_INDEPENDENT_AGENT_REVIEW_PENDING"
            )
        ),
        "duplicate_service": lambda: mutate_approval(
            lambda value: value["ambient_runtime_dependencies"][
                "external_services"
            ].append(
                dict(value["ambient_runtime_dependencies"]["external_services"][0])
            )
        ),
        "empty_sockets": lambda: mutate_approval(
            lambda value: value["ambient_runtime_dependencies"].update(sockets=[])
        ),
        "empty_negative": lambda: mutate_approval(
            lambda value: value["ambient_runtime_dependencies"].update(
                negative_resolution_assertions=[]
            )
        ),
        "extra_role": lambda: mutate_approval(
            lambda value: value["immutable_files"][0]["roles"].append(
                "unreviewed_extra_role"
            )
        ),
        "config_mode": lambda: mutate_approval(
            lambda value: next(
                item for item in value["immutable_files"] if item["path"] == str(config)
            ).update(mode="0755")
        ),
        "library_mode": lambda: mutate_approval(
            lambda value: next(
                item
                for item in value["immutable_files"]
                if item["path"] == str(library)
            ).update(mode="0755")
        ),
        "unclosed_symlink": lambda: mutate_approval(
            lambda value: value["symlink_bindings"].append(
                {
                    "path": str(unclosed_link),
                    "target": str(approval_path),
                    "resolved_path": str(approval_path),
                }
            )
        ),
        "producer": lambda: mutate_reference(
            "capture_terminal",
            lambda value: value["producer"].update(host="unreviewed-host"),
        ),
        "package": lambda: mutate_reference(
            "capture_terminal",
            lambda value: value["package"]["payload_sha256"].update(
                {
                    Path(value["producer"]["path"]).name: "0" * 64,
                }
            ),
        ),
        "job": lambda: mutate_reference(
            "capture_terminal", lambda value: value["job"].update(job_id=54321)
        ),
        "query": lambda: mutate_reference(
            "capture_terminal",
            lambda value: value["query"].update(subprocess_timeout_seconds=9.0),
        ),
        "surface": lambda: mutate_reference(
            "capture_terminal",
            lambda value: value["surface"].update(
                effective_uid=value["surface"]["effective_uid"] + 1
            ),
        ),
        "sacct": lambda: mutate_reference(
            "capture_terminal", lambda value: value["sacct"].update(stderr="error")
        ),
        "census": lambda: mutate_reference(
            "capture_terminal",
            lambda value: value["dependency_census"].update(
                total_count=value["dependency_census"]["total_count"] + 1
            ),
        ),
        "candidate": lambda: mutate_reference(
            "candidate", lambda value: value.update(profile="unreviewed-candidate")
        ),
        "reviewer": lambda: mutate_reference(
            "review_decision",
            lambda value: value["reviewer_identity"].update(
                principal="codex-agent:unreviewed"
            ),
        ),
        "path_overlap": lambda: mutate_approval(
            lambda value: value["trace_evidence"]["candidate"].update(
                path=value["trace_evidence"]["review_decision"]["path"]
            )
        ),
    }
    for name, mutate in mutations.items():
        restore()
        mutate()
        assert source_accepts() is False, name
        assert trace_accepts() is False, name
    restore()
    assert source_accepts() is True
    assert trace_accepts() is True


def test_independent_review_artifact_orders_capture_before_final_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sacct = tmp_path / "sacct"
    sacct.write_bytes(b"#!/usr/bin/bash\nexit 0\n")
    sacct.chmod(0o755)
    config = tmp_path / "slurm.conf"
    config.write_text("ClusterName=ordering\n", encoding="ascii")
    config.chmod(0o644)
    library = tmp_path / "libslurmfull.so"
    library.write_bytes(b"ordering-slurm-runtime")
    library.chmod(0o644)
    monkeypatch.setattr(job_scheduler, "PINNED_SACCT_PATH", sacct)
    monkeypatch.setattr(job_scheduler, "PINNED_SLURM_CONFIG_PATH", config)
    monkeypatch.setattr(job_scheduler, "PINNED_SLURM_LIBRARY_PATH", library)
    monkeypatch.setattr(job_scheduler, "SACCT_QUERY_LD_LIBRARY_PATH", str(tmp_path))
    approval_path, approval_sha256 = publish_sacct_runtime_approval(
        tmp_path,
        sacct_path=sacct,
        slurm_config_path=config,
        slurm_library_path=library,
    )
    approval = json.loads(approval_path.read_bytes())
    candidate_path = Path(approval["trace_evidence"]["candidate"]["path"])
    capture_path = Path(approval["trace_evidence"]["capture_terminal"]["path"])
    review_path = Path(approval["trace_evidence"]["review_decision"]["path"])
    candidate = json.loads(candidate_path.read_bytes())
    capture = json.loads(capture_path.read_bytes())
    review = json.loads(review_path.read_bytes())
    source_path = Path(capture["producer"]["path"])
    package_manifest_path = Path(capture["package"]["manifest"]["path"])
    frozen_package = {
        source_path: source_path.read_bytes(),
        package_manifest_path: package_manifest_path.read_bytes(),
    }
    assert candidate["trace_evidence"] == {}
    assert capture["candidate"] == approval["trace_evidence"]["candidate"]
    assert "review_decision" not in capture
    assert "approval" not in capture
    assert review["decision"] == "approve"
    assert review["review_scope"] == job_scheduler.SACCT_RUNTIME_REVIEW_SCOPE
    assert (
        review["candidate_sha256"] == approval["trace_evidence"]["candidate"]["sha256"]
    )
    assert (
        review["capture_terminal_sha256"]
        == approval["trace_evidence"]["capture_terminal"]["sha256"]
    )
    assert review["approved_at"] >= capture["surface"]["capture_finished_at"]
    assert review["approval_path"] == str(approval_path)
    assert (
        len(
            {
                capture["producer"]["path"],
                str(candidate_path),
                str(capture_path),
                str(review_path),
                str(approval_path),
            }
        )
        == 5
    )
    job_scheduler.validate_sacct_runtime_approval(
        approval_path, expected_sha256=approval_sha256, live=True
    )
    assert {
        source_path: source_path.read_bytes(),
        package_manifest_path: package_manifest_path.read_bytes(),
    } == frozen_package


def publish_app_submission_provenance(
    tmp_path: Path,
    *,
    source_approval_path: Path,
    source_tree_sha256: str,
    polaris_base_commit: str,
    sacct_identity: dict[str, object],
    slurm_config_identity: dict[str, object],
    slurm_library_identity: dict[str, object],
    sacct_runtime_approval_path: Path,
    sacct_runtime_approval_sha256: str,
    output_namespace_parent: Path,
) -> tuple[Path, Path, str]:
    provenance = tmp_path / "submission_provenance/job_12345"
    provenance.mkdir(parents=True)
    held_path = provenance / "scheduler_held.json"
    held_raw = (
        "JobId=12345 JobState=PENDING Reason=JobHeldUser "
        f"Requeue=0 Restarts=0 Comment={SUBMISSION_TRANSACTION_ID}\n"
    )
    held_job = job_scheduler.parse_scontrol_job_record(
        held_raw,
        phase="held",
        expected_job_id=12345,
        expected_transaction_id=SUBMISSION_TRANSACTION_ID,
    )
    held = diagnostic.publish_immutable_json(
        held_path,
        {
            "schema_version": 1,
            "profile": job_scheduler.SCHEDULER_JOB_PROFILE,
            "status": "held_requeue_disabled_restart_count_zero",
            "command": ["scontrol", "show", "job", "12345", "--oneliner"],
            "job": held_job,
        },
    )
    batch_path = provenance / "batch_script.sbatch"
    batch_path.write_bytes(b"#!/usr/bin/bash -p\n#SBATCH --no-requeue\n")
    batch_path.chmod(0o444)
    batch_sha256 = diagnostic.stable_file_identity(batch_path)["sha256"]
    source_approval_value = json.loads(source_approval_path.read_text(encoding="ascii"))
    exports = {
        "PATH": "/cm/local/apps/slurm/24.11/bin:/usr/bin:/bin",
        "HOME": "/synthetic/home",
        "POLARIS_SOURCE_SNAPSHOT": source_approval_value["snapshot_path"],
        "EXPECTED_POLARIS_SOURCE_TREE_SHA256": source_tree_sha256,
        "POLARIS_SOURCE_APPROVAL": str(source_approval_path),
        "POLARIS_OPENPI_RUNTIME_DIR": "/synthetic/openpi",
        "EXPECTED_POLARIS_COMMIT": polaris_base_commit,
        "POLARIS_ENVIRONMENT": "DROID-FoodBussing",
        "ROLLOUTS": "1",
        "ENVIRONMENT_SEED": "0",
        "RUN_NAMESPACE": "promotion-test",
        "SUBMISSION_TRANSACTION_ID": SUBMISSION_TRANSACTION_ID,
        "POLARIS_EVAL_MODE": "app_launcher_only",
    }
    export_token = "--export=" + ",".join(
        f"{name}={value}" for name, value in exports.items()
    )
    argv = [
        "/cm/local/apps/slurm/24.11/bin/sbatch",
        "--parsable",
        "--hold",
        "--no-requeue",
        f"--comment={SUBMISSION_TRANSACTION_ID}",
        "--job-name=pi05-app-launcher_FoodBussing",
        "--time=00:30:00",
        "--output=/synthetic/logs/%x-%j.out",
        export_token,
        "/approved/l40s_pi05_eval_job.sbatch",
    ]
    argv_path = provenance / "submission_argv.sh"
    argv_path.write_text(shlex.join(argv) + "\n", encoding="utf-8")
    argv_path.chmod(0o444)
    argv_sha256 = diagnostic.stable_file_identity(argv_path)["sha256"]
    prelaunch_receipt = job_scheduler.publish_sacct_prelaunch_validation_receipt(
        tmp_path / job_scheduler.SACCT_PRELAUNCH_RECEIPT_FILENAME,
        approval_path=sacct_runtime_approval_path,
        expected_approval_sha256=sacct_runtime_approval_sha256,
        source_approval_path=source_approval_path,
    )
    approval_path = provenance / "app_runtime_approval.env"
    approval_fields = {
        "profile": "polaris_app_launcher_runtime_approval_v6",
        "output_root": str(output_namespace_parent.parent),
        "output_namespace_parent": str(output_namespace_parent),
        "output_namespace_parent_identity": "1:2:3:4:0755",
        "runtime_closure_approval": "/approved/runtime.json",
        "runtime_closure_approval_sha256": "1" * 64,
        "sacct_runtime_approval": str(sacct_runtime_approval_path),
        "sacct_runtime_approval_sha256": sacct_runtime_approval_sha256,
        "sacct_prelaunch_validation_receipt": prelaunch_receipt["path"],
        "sacct_prelaunch_validation_receipt_sha256": prelaunch_receipt["sha256"],
        "scheduler_query_profile": "polaris_app_launcher_sacct_query_v1",
        "scheduler_query_path": job_scheduler.SACCT_QUERY_PATH,
        "scheduler_query_slurm_conf": str(job_scheduler.PINNED_SLURM_CONFIG_PATH),
        "scheduler_query_ld_library_path": (job_scheduler.SACCT_QUERY_LD_LIBRARY_PATH),
        "scheduler_query_timeout_seconds": str(
            int(job_scheduler.SACCT_SUBPROCESS_TIMEOUT_SECONDS)
        ),
        "expected_slurm_config_path": str(job_scheduler.PINNED_SLURM_CONFIG_PATH),
        "expected_slurm_config_sha256": str(slurm_config_identity["sha256"]),
        "expected_slurm_config_size": str(slurm_config_identity["size"]),
        "expected_scontrol_sha256": "3" * 64,
        "expected_scontrol_size": "1",
        "expected_slurm_library_path": str(job_scheduler.PINNED_SLURM_LIBRARY_PATH),
        "expected_slurm_library_sha256": str(slurm_library_identity["sha256"]),
        "expected_slurm_library_size": str(slurm_library_identity["size"]),
        "expected_sacct_path": str(job_scheduler.PINNED_SACCT_PATH),
        "expected_sacct_sha256": str(sacct_identity["sha256"]),
        "expected_sacct_size": str(sacct_identity["size"]),
        "expected_scancel_sha256": "6" * 64,
        "expected_scancel_size": "1",
        "expected_srun_sha256": "7" * 64,
        "expected_srun_size": "1",
        "approved_batch_script": "/approved/l40s_pi05_eval_job.sbatch",
        "batch_script_sha256": batch_sha256,
        "submission_argv_sha256": argv_sha256,
        "held_scheduler_record_sha256": held["sha256"],
    }
    approval_path.write_text(
        "".join(f"{name}={value}\n" for name, value in approval_fields.items()),
        encoding="utf-8",
    )
    approval_path.chmod(0o444)
    approval_sha256 = diagnostic.stable_file_identity(approval_path)["sha256"]
    return held_path, approval_path, approval_sha256


def load_app_finalizer():
    specification = importlib.util.spec_from_file_location(
        "_polaris_app_finalizer_test", APP_FINALIZER_PATH
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def synthetic_runtime(
    *,
    pid: int = 4444,
    output_directories: dict[str, str] | None = None,
) -> dict[str, object]:
    return diagnostic.capture_runtime_context(
        python_argv=["scripts/eval.py"],
        source_root=ROOT,
        expected_gpu_uuid=GPU_UUID,
        environ=runtime_environment(),
        nvidia_smi_output=nvidia_smi_output(),
        cgroup_text=cgroup_text(),
        device_nodes=[physical_device_node()],
        pid=pid,
        ppid=3333,
        executable="/.venv/bin/python",
        cwd=ROOT,
        scheduler_handoff=synthetic_scheduler_handoff(),
        output_directories=(
            synthetic_output_directories()
            if output_directories is None
            else output_directories
        ),
    )


def canonical_target(task_dir: Path, *, port: int = 32345) -> list[str]:
    preexec = task_dir / "startup_preexec.json"
    preclose = task_dir / "startup_preclose.json"
    return [
        "/.venv/bin/python",
        "scripts/eval.py",
        "--environment",
        "DROID-FoodBussing",
        "--control-mode",
        "joint-position",
        "--policy.client",
        "DroidJointPos",
        "--policy.host",
        "127.0.0.1",
        "--policy.port",
        str(port),
        "--policy.open-loop-horizon",
        "8",
        "--policy.frame-description",
        "robot base frame",
        "--policy.action-frame",
        "robot_base",
        "--policy.dataset-name",
        "droid",
        "--policy.no-rotate-wrist-180",
        "--policy.no-render-every-step",
        "--policy.state-type",
        "joint_position",
        "--policy.expected-action-horizon",
        "15",
        "--policy.expected-action-dim",
        "8",
        "--policy.trace-path",
        str(task_dir / "policy_traces.forbidden"),
        "--run-folder",
        str(task_dir),
        "--rollouts",
        "1",
        "--environment-seed",
        "0",
        "--runtime-contract-path",
        str(task_dir / "runtime_contract.forbidden"),
        "--headless",
        "--startup-diagnostic",
        "app_launcher_only",
        "--startup-diagnostic-preexec-path",
        str(preexec),
        "--startup-diagnostic-preclose-path",
        str(preclose),
        "--startup-diagnostic-expected-gpu-uuid",
        GPU_UUID,
    ]


def preexec_value(runtime: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "profile": diagnostic.PREEXEC_PROFILE,
        "status": "captured_before_public_eval_exec",
        "startup_diagnostic": diagnostic.STARTUP_DIAGNOSTIC_MODE,
        "runtime": runtime,
        "launcher_argv": ["app_launcher_startup_diagnostic.py"],
        "target_argv": ["/.venv/bin/python", "scripts/eval.py"],
        "zero_work_counters": dict(diagnostic.ZERO_WORK_COUNTERS),
        "bounded_diagnostic_counts": {
            "nvidia_smi_invocations": 1,
            "scheduler_request_artifacts": 1,
            "scheduler_handoff_artifacts": 1,
            "job_scheduler_records": 1,
            "step_scheduler_records": 1,
            "preexec_artifacts": 1,
            "preclose_artifacts": 0,
            "ready_artifacts": 0,
            "simulation_app_close_calls": 0,
        },
    }


def publish_preexec(path: Path, runtime: dict[str, object]) -> dict[str, object]:
    return diagnostic.publish_immutable_json(path, preexec_value(runtime))


def test_nvidia_smi_requires_exactly_one_expected_l40s() -> None:
    value = diagnostic.parse_nvidia_smi_output(nvidia_smi_output())
    assert value == {
        "uuid": GPU_UUID,
        "name": "NVIDIA L40S",
        "driver_version": "580.105.08",
        "minor_number": 2,
        "command": list(diagnostic.NVIDIA_SMI_COMMAND),
        "row_count": 1,
    }


@pytest.mark.parametrize(
    "output",
    [
        "",
        nvidia_smi_output()
        + nvidia_smi_output(uuid="GPU-fedcba98-7654-3210-fedc-ba9876543210", minor=3),
        f"{GPU_UUID}, NVIDIA L40S, 580.105.08\n",
        "GPU-not-a-uuid, NVIDIA L40S, 580.105.08, 2\n",
        f"{GPU_UUID}, NVIDIA A100-SXM4-80GB, 580.105.08, 2\n",
        f"{GPU_UUID}, NVIDIA L40S, 580.95.05, 2\n",
        f"{GPU_UUID}, NVIDIA L40S, 580.105.08, N/A\n",
    ],
)
def test_nvidia_smi_rejects_missing_extra_and_malformed_rows(output: str) -> None:
    with pytest.raises(ValueError):
        diagnostic.parse_nvidia_smi_output(output)


def test_cgroup_requires_one_matching_job_step_path() -> None:
    value = diagnostic.parse_cgroup_text(cgroup_text(), job_id="12345", step_id="0")
    assert value["job_step_path"] == "/slurm/uid_1000/job_12345/step_0/task_0"
    assert len(value["records"]) == 1


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "0::/\n",
        cgroup_text(job_id="99999"),
        cgroup_text(step_id="1"),
        (cgroup_text() + "1:devices:/slurm/uid_1000/job_12345/step_0/other_task\n"),
        cgroup_text() + cgroup_text(),
        "malformed\n",
    ],
)
def test_cgroup_rejects_missing_multiple_mismatched_and_duplicate_evidence(
    raw: str,
) -> None:
    with pytest.raises(ValueError):
        diagnostic.parse_cgroup_text(raw, job_id="12345", step_id="0")


def test_device_nodes_require_exactly_one_matching_physical_gpu() -> None:
    control = {
        "path": "/dev/nvidiactl",
        "mode": "0660",
        "file_type": "character",
        "device_major": 195,
        "device_minor": 255,
    }
    value = diagnostic.validate_device_nodes(
        [physical_device_node(), control], expected_minor_number=2
    )
    assert value["physical_count"] == 1
    assert value["physical"] == [physical_device_node()]


@pytest.mark.parametrize(
    "records,expected_minor",
    [
        ([], 2),
        ([physical_device_node(), physical_device_node(index=3, minor=3)], 2),
        ([physical_device_node(index=2, minor=3)], 3),
        ([physical_device_node()], 3),
        (
            [
                {
                    **physical_device_node(),
                    "file_type": "other",
                }
            ],
            2,
        ),
    ],
)
def test_device_nodes_reject_missing_extra_and_mismatched_gpu_nodes(
    records: list[dict[str, object]], expected_minor: int
) -> None:
    with pytest.raises(ValueError):
        diagnostic.validate_device_nodes(records, expected_minor_number=expected_minor)


def test_runtime_context_cross_binds_process_step_gpu_cgroup_and_source() -> None:
    value = synthetic_runtime()
    assert value["process"]["pid"] == 4444
    assert value["slurm"]["job_id"] == 12345
    assert value["slurm"]["step_id"] == 0
    assert value["nvidia_smi"]["uuid"] == GPU_UUID
    assert value["device_nodes"]["physical_count"] == 1
    assert value["slurm"]["gpus_on_node"] == 1
    assert value["slurm"]["gpus_per_task"] == 1
    assert value["slurm"]["tres_per_task_items"] == {
        "cpu": "16",
        "gres/gpu": "1",
    }
    assert (
        value["slurm"]["scheduler_handoff"]["value"]["job_record"][
            "allocated_tres_items"
        ]["gres/gpu"]
        == "1"
    )
    assert value["source"]["eval_script"]["path"] == str(ROOT / "scripts/eval.py")
    assert value["source"]["approval"] == {
        "BATCH_VERIFIED_POLARIS_SOURCE_TREE_SHA256": SOURCE_TREE_SHA256,
        "POLARIS_IMPLEMENTATION_COMMIT": IMPLEMENTATION_COMMIT,
        "SOURCE_APPROVAL_SHA256": SOURCE_APPROVAL_SHA256,
    }
    environment = runtime_environment()
    assert value["execution_approvals"] == {
        name: environment[name] for name in diagnostic.EXECUTION_APPROVAL_ENVIRONMENT
    }
    assert value["pyxis_image"]["expected_sha256"] == PYXIS_IMAGE_SHA256


@pytest.mark.parametrize(
    "updates",
    [
        {"SLURM_JOB_ID": ""},
        {"SLURM_JOB_ID": "batch"},
        {"SLURM_STEP_ID": ""},
        {"SLURM_STEP_ID": "batch"},
        {"SLURM_JOB_GPUS": ""},
        {"SLURM_STEP_GPUS": ""},
        {"SLURM_STEP_GPUS": "3"},
        {"SLURM_JOB_GPUS": "3", "SLURM_STEP_GPUS": "3"},
        {"SLURM_GPUS_ON_NODE": ""},
        {"SLURM_GPUS_ON_NODE": "2"},
        {"SLURM_GPUS_PER_TASK": ""},
        {"SLURM_GPUS_PER_TASK": "2"},
        {"SLURM_TRES_PER_TASK": ""},
        {"SLURM_TRES_PER_TASK": "cpu=16,gres/gpu:2"},
        {"SLURM_TRES_PER_TASK": "cpu=32,gres/gpu:1"},
        {"SLURM_TRES_PER_TASK": "gres/gpu:1"},
        {"SLURM_CPUS_PER_TASK": "8"},
        {"SLURM_NTASKS": "2"},
        {"SLURM_JOB_NUM_NODES": "2"},
        {"SLURM_MEM_PER_NODE": "65536"},
        {"SLURM_JOB_ACCOUNT": "other"},
        {"SLURM_JOB_PARTITION": "batch_long"},
        {"SLURM_JOB_QOS": "high"},
        {"SLURM_JOB_USER": "other"},
        {"NVIDIA_VISIBLE_DEVICES": "GPU-fedcba98-7654-3210-fedc-ba9876543210"},
    ],
)
def test_runtime_context_rejects_missing_or_mismatched_environment(
    updates: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        diagnostic.capture_runtime_context(
            python_argv=["scripts/eval.py"],
            source_root=ROOT,
            expected_gpu_uuid=GPU_UUID,
            environ=runtime_environment(**updates),
            nvidia_smi_output=nvidia_smi_output(),
            cgroup_text=cgroup_text(),
            device_nodes=[physical_device_node()],
        )


def test_external_execution_approval_environment_is_closed_and_typed() -> None:
    environment = runtime_environment()
    expected = {
        name: environment[name] for name in diagnostic.EXECUTION_APPROVAL_ENVIRONMENT
    }
    assert diagnostic._validate_execution_approval_environment(environment) == expected
    for name in diagnostic.EXECUTION_APPROVAL_ENVIRONMENT:
        missing = dict(environment)
        missing.pop(name)
        with pytest.raises(ValueError):
            diagnostic._validate_execution_approval_environment(missing)
        malformed = dict(environment)
        malformed[name] = "0" if name.endswith("_SIZE") else "not-a-digest"
        with pytest.raises(ValueError):
            diagnostic._validate_execution_approval_environment(malformed)


def test_runtime_context_rejects_nvidia_uuid_mismatch() -> None:
    with pytest.raises(ValueError, match="nvidia-smi GPU UUID"):
        diagnostic.capture_runtime_context(
            python_argv=["scripts/eval.py"],
            source_root=ROOT,
            expected_gpu_uuid=GPU_UUID,
            environ=runtime_environment(),
            nvidia_smi_output=nvidia_smi_output(
                uuid="GPU-fedcba98-7654-3210-fedc-ba9876543210"
            ),
            cgroup_text=cgroup_text(),
            device_nodes=[physical_device_node()],
        )


def test_runtime_context_rejects_diagnostic_module_origin_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    escaped = tmp_path / "app_launcher_startup_diagnostic.py"
    escaped.write_text("# adversarial shadow\n", encoding="utf-8")
    monkeypatch.setattr(diagnostic, "__file__", str(escaped))
    with pytest.raises(ValueError, match="escaped the approved source root"):
        diagnostic.capture_runtime_context(
            python_argv=["scripts/eval.py"],
            source_root=ROOT,
            expected_gpu_uuid=GPU_UUID,
            environ=runtime_environment(),
            nvidia_smi_output=nvidia_smi_output(),
            cgroup_text=cgroup_text(),
            device_nodes=[physical_device_node()],
            executable="/.venv/bin/python",
            cwd=ROOT,
        )


def test_immutable_publication_is_canonical_nonreplacing_and_mode_0444(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.json"
    identity = diagnostic.publish_immutable_json(path, {"z": 1, "a": [True, None]})
    assert path.read_bytes() == b'{"a":[true,null],"z":1}\n'
    metadata = path.stat()
    assert stat.S_IMODE(metadata.st_mode) == 0o444
    assert metadata.st_nlink == 1
    assert identity["sha256"] == diagnostic._sha256(path.read_bytes())
    with pytest.raises(FileExistsError):
        diagnostic.publish_immutable_json(path, {"replacement": True})
    assert path.read_bytes() == b'{"a":[true,null],"z":1}\n'


def test_immutable_publication_rejects_symlink_destination(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("unchanged", encoding="utf-8")
    destination = tmp_path / "artifact.json"
    destination.symlink_to(target)
    with pytest.raises(FileExistsError):
        diagnostic.publish_immutable_json(destination, {"replacement": True})
    assert target.read_text(encoding="utf-8") == "unchanged"


def test_creator_identity_rejects_parent_replacement_for_json_write_and_read(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "stable"
    moved = tmp_path / "moved"
    parent.mkdir()
    artifact = parent / "artifact.json"
    expected_identity = diagnostic.capture_directory_identity(parent)
    parent.rename(moved)
    parent.mkdir()
    with pytest.raises(RuntimeError, match="creator-observed identity mismatch"):
        diagnostic.publish_immutable_json(
            artifact,
            {"must_not": "publish"},
            expected_parent_identity=expected_identity,
        )
    assert not artifact.exists()

    moved_artifact = moved / artifact.name
    diagnostic.publish_immutable_json(moved_artifact, {"original": True})
    with pytest.raises(RuntimeError, match="creator-observed identity mismatch"):
        diagnostic.stable_read_immutable_json(
            artifact,
            expected_parent_identity=expected_identity,
        )


def test_stable_read_rejects_writable_hardlinked_and_noncanonical_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.json"
    path.write_text('{"value":1}\n', encoding="ascii")
    path.chmod(0o644)
    with pytest.raises(ValueError, match="mode is not 0444"):
        diagnostic.stable_read_immutable_json(path)
    path.chmod(0o444)
    linked = tmp_path / "linked.json"
    os.link(path, linked)
    with pytest.raises(ValueError, match="one regular link"):
        diagnostic.stable_read_immutable_json(path)
    linked.unlink()
    path.chmod(0o644)
    path.write_text('{"value": 1}\n', encoding="ascii")
    path.chmod(0o444)
    with pytest.raises(ValueError, match="canonical"):
        diagnostic.stable_read_immutable_json(path)


def test_immutable_log_uses_exclusive_writer_and_hash_bound_identity(
    tmp_path: Path,
) -> None:
    output = tmp_path / "app_launcher_only.log"
    identity = tmp_path / "app_launcher_only.log.identity.json"
    payload = b"phase-one\nphase-two\x00binary\n"
    mirror = io.BytesIO()
    result = diagnostic.capture_immutable_log(
        output_path=output,
        identity_path=identity,
        input_stream=io.BytesIO(payload),
        mirror_stream=mirror,
    )
    assert mirror.getvalue() == payload
    assert output.read_bytes() == payload
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert output.stat().st_nlink == 1
    assert result == diagnostic.validate_immutable_log_identity(identity)
    value, _ = diagnostic.stable_read_immutable_json(identity)
    assert value["log"]["sha256"] == diagnostic._sha256(payload)


def test_immutable_log_never_replaces_existing_destination_or_symlink(
    tmp_path: Path,
) -> None:
    output = tmp_path / "app_launcher_only.log"
    identity = tmp_path / "app_launcher_only.log.identity.json"
    output.write_bytes(b"prior evidence\n")
    output.chmod(0o444)
    with pytest.raises(FileExistsError):
        diagnostic.capture_immutable_log(
            output_path=output,
            identity_path=identity,
            input_stream=io.BytesIO(b"replacement\n"),
            mirror_stream=io.BytesIO(),
        )
    assert output.read_bytes() == b"prior evidence\n"
    assert not identity.exists()
    assert not list(tmp_path.glob(".*.partial-*"))

    output.unlink()
    target = tmp_path / "symlink-target.log"
    target.write_bytes(b"target remains\n")
    output.symlink_to(target)
    with pytest.raises(FileExistsError):
        diagnostic.capture_immutable_log(
            output_path=output,
            identity_path=identity,
            input_stream=io.BytesIO(b"replacement\n"),
            mirror_stream=io.BytesIO(),
        )
    assert target.read_bytes() == b"target remains\n"


def test_immutable_log_identity_collision_preserves_prior_identity_and_new_log(
    tmp_path: Path,
) -> None:
    output = tmp_path / "app_launcher_only.log"
    identity = tmp_path / "app_launcher_only.log.identity.json"
    identity.write_bytes(b"prior identity\n")
    identity.chmod(0o444)
    with pytest.raises(FileExistsError):
        diagnostic.capture_immutable_log(
            output_path=output,
            identity_path=identity,
            input_stream=io.BytesIO(b"forensic log\n"),
            mirror_stream=io.BytesIO(),
        )
    assert identity.read_bytes() == b"prior identity\n"
    assert output.read_bytes() == b"forensic log\n"
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert output.stat().st_nlink == 1
    assert not list(tmp_path.glob(".*.partial-*"))


def test_immutable_log_link_failure_cleans_only_unpublished_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "app_launcher_only.log"
    identity = tmp_path / "app_launcher_only.log.identity.json"

    def reject_link(*args, **kwargs):
        del args, kwargs
        raise OSError("injected link failure")

    monkeypatch.setattr(os, "link", reject_link)
    with pytest.raises(OSError, match="injected link failure"):
        diagnostic.capture_immutable_log(
            output_path=output,
            identity_path=identity,
            input_stream=io.BytesIO(b"unpublished\n"),
            mirror_stream=io.BytesIO(),
        )
    assert not output.exists()
    assert not identity.exists()
    assert not list(tmp_path.glob(".*.partial-*"))


def test_immutable_log_preexisting_private_temp_is_not_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "app_launcher_only.log"
    identity = tmp_path / "app_launcher_only.log.identity.json"
    monkeypatch.setattr(os, "urandom", lambda _: b"\x00" * 8)
    temporary = tmp_path / (f".{output.name}.partial-{os.getpid()}-0000000000000000")
    temporary.write_bytes(b"hostile temp\n")
    with pytest.raises(FileExistsError):
        diagnostic.capture_immutable_log(
            output_path=output,
            identity_path=identity,
            input_stream=io.BytesIO(b"new log\n"),
            mirror_stream=io.BytesIO(),
        )
    assert temporary.read_bytes() == b"hostile temp\n"
    assert not output.exists()
    assert not identity.exists()


def test_immutable_log_retries_partial_fd_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "app_launcher_only.log"
    identity = tmp_path / "app_launcher_only.log.identity.json"
    original_write = os.write
    observed_lengths: list[int] = []

    def partial_write(descriptor: int, payload: bytes) -> int:
        observed_lengths.append(len(payload))
        count = max(1, len(payload) // 2)
        return original_write(descriptor, payload[:count])

    monkeypatch.setattr(os, "write", partial_write)
    payload = b"partial-writes-must-complete\n"
    diagnostic.capture_immutable_log(
        output_path=output,
        identity_path=identity,
        input_stream=io.BytesIO(payload),
        mirror_stream=io.BytesIO(),
    )
    assert output.read_bytes() == payload
    assert len(observed_lengths) > 2
    diagnostic.validate_immutable_log_identity(identity)


def test_immutable_log_zero_write_and_fsync_failures_leave_no_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "app_launcher_only.log"
    identity = tmp_path / "app_launcher_only.log.identity.json"
    monkeypatch.setattr(os, "write", lambda _descriptor, _payload: 0)
    with pytest.raises(OSError, match="short immutable log write"):
        diagnostic.capture_immutable_log(
            output_path=output,
            identity_path=identity,
            input_stream=io.BytesIO(b"payload\n"),
            mirror_stream=io.BytesIO(),
        )
    assert not output.exists() and not identity.exists()
    assert not list(tmp_path.glob(".*.partial-*"))

    monkeypatch.undo()
    calls = 0
    original_fsync = os.fsync

    def fail_first_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_first_fsync)
    with pytest.raises(OSError, match="injected fsync failure"):
        diagnostic.capture_immutable_log(
            output_path=output,
            identity_path=identity,
            input_stream=io.BytesIO(b"payload\n"),
            mirror_stream=io.BytesIO(),
        )
    assert not output.exists() and not identity.exists()
    assert not list(tmp_path.glob(".*.partial-*"))


def test_immutable_log_parent_swap_is_detected_without_replacing_new_parent(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "stable"
    moved = tmp_path / "moved"
    parent.mkdir()
    output = parent / "app_launcher_only.log"
    identity = parent / "app_launcher_only.log.identity.json"

    class SwappingInput:
        swapped = False

        def read(self, _size: int) -> bytes:
            if self.swapped:
                return b""
            self.swapped = True
            parent.rename(moved)
            parent.mkdir()
            return b"forensic log\n"

    with pytest.raises(RuntimeError, match="parent directory binding changed"):
        diagnostic.capture_immutable_log(
            output_path=output,
            identity_path=identity,
            input_stream=SwappingInput(),
            mirror_stream=io.BytesIO(),
        )
    assert not output.exists()
    assert not identity.exists()
    assert (moved / output.name).read_bytes() == b"forensic log\n"
    assert stat.S_IMODE((moved / output.name).stat().st_mode) == 0o444


def test_creator_identity_rejects_parent_replacement_before_log_open(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "stable"
    moved = tmp_path / "moved"
    parent.mkdir()
    expected_identity = diagnostic.capture_directory_identity(parent)
    parent.rename(moved)
    parent.mkdir()
    output = parent / "app_launcher_only.log"
    identity = parent / "app_launcher_only.log.identity.json"
    with pytest.raises(RuntimeError, match="creator-observed identity mismatch"):
        diagnostic.capture_immutable_log(
            output_path=output,
            identity_path=identity,
            input_stream=io.BytesIO(b"must not publish\n"),
            mirror_stream=io.BytesIO(),
            expected_parent_identity=expected_identity,
        )
    assert not output.exists()
    assert not identity.exists()
    assert not list(parent.iterdir())
    assert not list(moved.iterdir())


def test_output_context_rejects_creator_bound_task_replacement(tmp_path: Path) -> None:
    run_dir, task_dir, output_directories = create_output_tree(tmp_path)
    moved = tmp_path / "moved-task"
    task_dir.rename(moved)
    task_dir.mkdir(mode=0o755)
    environment = {
        "POLARIS_STARTUP_DIAGNOSTIC_RUN_DIR": str(run_dir),
        "POLARIS_STARTUP_DIAGNOSTIC_TASK_DIR": str(task_dir),
        "POLARIS_OUTPUT_NAMESPACE_PARENT_IDENTITY": output_directories[
            "namespace_parent_identity"
        ],
        "POLARIS_STARTUP_DIAGNOSTIC_RUN_DIR_IDENTITY": output_directories[
            "run_identity"
        ],
        "POLARIS_STARTUP_DIAGNOSTIC_TASK_DIR_IDENTITY": output_directories[
            "task_identity"
        ],
    }
    with pytest.raises(RuntimeError, match="creator-observed identity mismatch"):
        diagnostic.capture_output_directory_context(environment)


def test_output_creation_ignores_restrictive_umask(tmp_path: Path) -> None:
    namespace = tmp_path / "namespace"
    namespace.mkdir()
    run_dir = namespace / "run"
    task_dir = run_dir / "app_launcher_only"
    parent_identity = diagnostic.capture_directory_identity(namespace)
    previous_umask = os.umask(0o077)
    try:
        created = diagnostic.create_output_directories(
            run_dir=run_dir,
            task_dir=task_dir,
            expected_parent_identity=parent_identity,
        )
    finally:
        os.umask(previous_umask)
    assert created["namespace_parent_identity"] == parent_identity
    assert stat.S_IMODE(run_dir.stat().st_mode) == 0o755
    assert stat.S_IMODE(task_dir.stat().st_mode) == 0o755


def test_output_creation_rejects_replaced_namespace_parent(tmp_path: Path) -> None:
    namespace = tmp_path / "namespace"
    moved = tmp_path / "moved-namespace"
    namespace.mkdir()
    expected_identity = diagnostic.capture_directory_identity(namespace)
    namespace.rename(moved)
    namespace.mkdir()
    with pytest.raises(RuntimeError, match="creator-observed identity mismatch"):
        diagnostic.create_output_directories(
            run_dir=namespace / "run",
            task_dir=namespace / "run/app_launcher_only",
            expected_parent_identity=expected_identity,
        )
    assert not (namespace / "run").exists()
    assert not list(moved.iterdir())


def test_in_job_evidence_tree_is_preterminal_and_has_exact_step_closure(
    tmp_path: Path,
) -> None:
    run_dir, task_dir, output_directories = create_output_tree(tmp_path)
    publish_success_evidence(task_dir, output_directories["task_identity"])
    sealed = diagnostic.seal_evidence_tree(
        task_dir=task_dir,
        run_dir=run_dir,
        outcome="success",
        expected_namespace_parent_identity=output_directories[
            "namespace_parent_identity"
        ],
        expected_task_identity=output_directories["task_identity"],
        expected_run_identity=output_directories["run_identity"],
        srun_exit_code=0,
        log_exit_code=0,
        helper_exit_code=0,
    )
    try:
        assert set(sealed["artifacts"]) == diagnostic.PRETERMINAL_TASK_ENTRIES
        assert sealed["termination_mode"] == "simulation_app_close_returned"
        assert sealed["log_sha256"] == diagnostic._sha256(VALID_PHASE_LOG)
        assert stat.S_IMODE(task_dir.stat().st_mode) == 0o555
        assert stat.S_IMODE(run_dir.stat().st_mode) == 0o555
        assert set(path.name for path in task_dir.iterdir()) == set(
            diagnostic.PRETERMINAL_TASK_ENTRIES
        )
        preterminal, _ = diagnostic.stable_read_immutable_json(
            task_dir / diagnostic.PRETERMINAL_ATTESTATION_NAME
        )
        assert preterminal["authoritative_completion"] is False
        assert preterminal["status"] == (
            "awaiting_external_allocation_terminal_attestation"
        )
        assert not (task_dir / "SUCCESS").exists()
        assert not (run_dir / "SUCCESS").exists()
        with pytest.raises(PermissionError):
            (task_dir / "unexpected").write_bytes(b"forbidden")
    finally:
        run_dir.chmod(0o755)
        task_dir.chmod(0o755)


def test_external_allocation_promotion_is_the_only_authoritative_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    polaris_base_commit = "c" * 40
    source_snapshot = tmp_path / "polaris-source"
    scheduler_source = source_snapshot / "src/polaris/pi05_droid_jointpos_scheduler.py"
    scheduler_source.parent.mkdir(parents=True)
    scheduler_source.write_bytes(
        Path(job_scheduler.__file__).resolve(strict=True).read_bytes()
    )
    scheduler_source.chmod(0o444)
    scheduler_source_identity = job_scheduler._identity(
        job_scheduler.validate_immutable_file(scheduler_source)
    )
    synthetic_producer = {
        "profile": "polaris_sacct_prelaunch_validation_producer_v1",
        "module": "polaris.pi05_droid_jointpos_scheduler",
        "path": scheduler_source_identity["path"],
        "sha256": scheduler_source_identity["sha256"],
        "uid": os.geteuid(),
        "host": os.uname().nodename,
        "principal": f"uid:{os.geteuid()}@{os.uname().nodename}",
    }
    monkeypatch.setattr(
        job_scheduler, "_scheduler_module_producer", lambda: synthetic_producer
    )
    source_approval_path = tmp_path / "polaris_source_approval.json"
    source_approval = diagnostic.publish_immutable_json(
        source_approval_path,
        {
            "schema_version": 1,
            "profile": "openpi_pi05_droid_jointpos_source_approval_v1",
            "snapshot_path": str(source_snapshot.resolve()),
            "source_tree_sha256": SOURCE_TREE_SHA256,
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "polaris_base_commit": polaris_base_commit,
            "polaris_base_tree": "d" * 40,
            "openpi_commit": "e" * 40,
            "trusted_hasher_sha256": "f" * 64,
        },
    )
    image_path = tmp_path / "polaris.sqsh"
    image_path.write_bytes(b"synthetic-pinned-pyxis-image")
    image_path.chmod(0o444)
    image = diagnostic.stable_file_identity(image_path)
    environment = runtime_environment(
        SOURCE_APPROVAL_SHA256=source_approval["sha256"],
        POLARIS_PYXIS_IMAGE_PATH=str(image_path),
        POLARIS_EXPECTED_PYXIS_IMAGE_SHA256=image["sha256"],
        POLARIS_OBSERVED_PYXIS_IMAGE_SHA256=image["sha256"],
        POLARIS_OBSERVED_PYXIS_IMAGE_MODE=image["mode"],
        POLARIS_OBSERVED_PYXIS_IMAGE_NLINK=str(image["nlink"]),
        POLARIS_OBSERVED_PYXIS_IMAGE_SIZE=str(image["size"]),
    )
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    output_namespace_parent = tmp_path / "promotion-test"
    output_namespace_parent.mkdir()
    run_dir = output_namespace_parent / (
        "promotion-test_public-app-launcher-only_DROID-FoodBussing_12345"
    )
    task_dir = run_dir / "app_launcher_only"
    output_directories = diagnostic.create_output_directories(
        run_dir=run_dir,
        task_dir=task_dir,
        expected_parent_identity=diagnostic.capture_directory_identity(
            output_namespace_parent
        ),
    )
    publish_success_evidence(
        task_dir,
        output_directories["task_identity"],
        environment=environment,
    )
    diagnostic.seal_evidence_tree(
        task_dir=task_dir,
        run_dir=run_dir,
        outcome="success",
        expected_namespace_parent_identity=output_directories[
            "namespace_parent_identity"
        ],
        expected_task_identity=output_directories["task_identity"],
        expected_run_identity=output_directories["run_identity"],
        srun_exit_code=0,
        log_exit_code=0,
        helper_exit_code=0,
    )
    sacct_path = tmp_path / "sacct"
    sacct_path.write_bytes(b"#!/usr/bin/bash\nexit 0\n")
    sacct_path.chmod(0o755)
    slurm_config_path = tmp_path / "slurm.conf"
    slurm_config_path.write_text("ClusterName=synthetic\n", encoding="utf-8")
    slurm_config_path.chmod(0o644)
    slurm_library_path = tmp_path / "libslurmfull.so"
    slurm_library_path.write_bytes(b"synthetic-slurm-library")
    slurm_library_path.chmod(0o644)
    monkeypatch.setattr(job_scheduler, "PINNED_SACCT_PATH", sacct_path)
    monkeypatch.setattr(job_scheduler, "PINNED_SLURM_CONFIG_PATH", slurm_config_path)
    monkeypatch.setattr(job_scheduler, "PINNED_SLURM_LIBRARY_PATH", slurm_library_path)
    monkeypatch.setattr(
        job_scheduler,
        "SACCT_QUERY_LD_LIBRARY_PATH",
        str(slurm_library_path.parent),
    )
    sacct_identity = job_scheduler._stable_executable(sacct_path)
    slurm_config_identity = job_scheduler._stable_regular_file(
        slurm_config_path, expected_mode=0o644, label="Slurm config"
    )
    slurm_library_identity = job_scheduler._stable_regular_file(
        slurm_library_path, expected_mode=0o644, label="Slurm library"
    )
    sacct_runtime_approval_path, sacct_runtime_approval_sha256 = (
        publish_sacct_runtime_approval(
            tmp_path,
            sacct_path=sacct_path,
            slurm_config_path=slurm_config_path,
            slurm_library_path=slurm_library_path,
        )
    )
    bad_surface_root = tmp_path / "bad-surface"
    bad_surface_root.mkdir()
    bad_surface_approval, bad_surface_sha256 = publish_sacct_runtime_approval(
        bad_surface_root,
        sacct_path=sacct_path,
        slurm_config_path=slurm_config_path,
        slurm_library_path=slurm_library_path,
        surface_hostname="wrong-finalizer-host",
    )
    with pytest.raises(ValueError, match="execution surface mismatch"):
        job_scheduler.validate_sacct_runtime_approval(
            bad_surface_approval,
            expected_sha256=bad_surface_sha256,
            live=True,
        )
    reviewed_value = json.loads(sacct_runtime_approval_path.read_text(encoding="ascii"))
    for name, mutate, message in (
        (
            "candidate-profile",
            lambda value: value.update(
                profile=job_scheduler.SACCT_RUNTIME_CANDIDATE_PROFILE
            ),
            "approval schema mismatch",
        ),
        (
            "alternate-command",
            lambda value: value["query_contract"].update(
                command_template=[str(tmp_path / "alternate-sacct")]
            ),
            "query contract mismatch",
        ),
        (
            "ambient-environment",
            lambda value: value["query_contract"]["environment"].update(
                LD_PRELOAD="/tmp/inject.so"
            ),
            "query contract mismatch",
        ),
    ):
        mutated_value = json.loads(json.dumps(reviewed_value))
        mutate(mutated_value)
        mutated_approval = diagnostic.publish_immutable_json(
            tmp_path / f"mutated-runtime-approval-{name}.json", mutated_value
        )
        with pytest.raises(ValueError, match=message):
            job_scheduler.validate_sacct_runtime_approval(
                Path(mutated_approval["path"]),
                expected_sha256=mutated_approval["sha256"],
                live=True,
            )
    held_path, approval_path, approval_sha256 = publish_app_submission_provenance(
        tmp_path,
        source_approval_path=source_approval_path,
        source_tree_sha256=SOURCE_TREE_SHA256,
        polaris_base_commit=polaris_base_commit,
        sacct_identity=sacct_identity,
        slurm_config_identity=slurm_config_identity,
        slurm_library_identity=slurm_library_identity,
        sacct_runtime_approval_path=sacct_runtime_approval_path,
        sacct_runtime_approval_sha256=sacct_runtime_approval_sha256,
        output_namespace_parent=output_namespace_parent,
    )
    allocation_raw = (
        "12345|COMPLETED|0:0|2026-07-09T01:00:00|"
        "2026-07-09T01:01:00|2026-07-09T01:02:00|60|pool0-00002|0\n"
    )

    promotion_path = (
        approval_path.parent / job_scheduler.APP_TERMINAL_PROMOTION_FILENAME
    )
    assert not promotion_path.exists()
    held_artifact = job_scheduler.validate_persisted_scheduler_job(
        held_path,
        phase="held",
        expected_job_id=12345,
        expected_transaction_id=SUBMISSION_TRANSACTION_ID,
    )
    promotion_provenance = job_scheduler._validate_app_provenance(
        approval_path,
        expected_sha256=approval_sha256,
        expected_job_id=12345,
        expected_transaction_id=SUBMISSION_TRANSACTION_ID,
        held=held_artifact,
    )
    assert not hasattr(job_scheduler, "_build_app_terminal_promotion")
    with pytest.raises(ValueError, match="job restarted"):
        job_scheduler.parse_sacct_terminal_record(
            allocation_raw[:-2] + "1\n", expected_job_id=12345
        )
    with pytest.raises(ValueError, match="approval mismatch"):
        job_scheduler._validate_app_provenance(
            approval_path,
            expected_sha256="0" * 64,
            expected_job_id=12345,
            expected_transaction_id=SUBMISSION_TRANSACTION_ID,
            held=held_artifact,
        )
    with pytest.raises(ValueError, match="comment mismatch"):
        job_scheduler.validate_persisted_scheduler_job(
            held_path,
            phase="held",
            expected_job_id=12345,
            expected_transaction_id="pi05-" + "0" * 40,
        )
    task_dir.chmod(0o755)
    with pytest.raises(ValueError, match="terminally sealed"):
        job_scheduler._validate_app_preterminal_tree(
            task_dir / diagnostic.PRETERMINAL_ATTESTATION_NAME,
            expected_job_id=12345,
            expected_transaction_id=SUBMISSION_TRANSACTION_ID,
            provenance=promotion_provenance,
        )
    task_dir.chmod(0o555)
    original_image = image_path.read_bytes()
    image_path.chmod(0o644)
    image_path.write_bytes(original_image + b"tamper")
    image_path.chmod(0o444)
    with pytest.raises(ValueError, match="Pyxis image changed"):
        job_scheduler._validate_app_preterminal_tree(
            task_dir / diagnostic.PRETERMINAL_ATTESTATION_NAME,
            expected_job_id=12345,
            expected_transaction_id=SUBMISSION_TRANSACTION_ID,
            provenance=promotion_provenance,
        )
    image_path.chmod(0o644)
    image_path.write_bytes(original_image)
    image_path.chmod(0o444)
    original_source_approval = source_approval_path.read_bytes()
    source_approval_path.chmod(0o644)
    source_approval_path.write_bytes(
        original_source_approval.replace(
            SOURCE_TREE_SHA256.encode("ascii"), ("0" * 64).encode("ascii"), 1
        )
    )
    source_approval_path.chmod(0o444)
    with pytest.raises(ValueError, match="source approval"):
        job_scheduler._validate_app_preterminal_tree(
            task_dir / diagnostic.PRETERMINAL_ATTESTATION_NAME,
            expected_job_id=12345,
            expected_transaction_id=SUBMISSION_TRANSACTION_ID,
            provenance=promotion_provenance,
        )
    source_approval_path.chmod(0o644)
    source_approval_path.write_bytes(original_source_approval)
    source_approval_path.chmod(0o444)
    step_path = task_dir / "scheduler_terminal.json"
    original_step = step_path.read_bytes()
    step_path.chmod(0o644)
    step_path.write_bytes(original_step.replace(b'"populated":0', b'"populated":1'))
    step_path.chmod(0o444)
    with pytest.raises(ValueError, match="evidence changed"):
        job_scheduler._validate_app_preterminal_tree(
            task_dir / diagnostic.PRETERMINAL_ATTESTATION_NAME,
            expected_job_id=12345,
            expected_transaction_id=SUBMISSION_TRANSACTION_ID,
            provenance=promotion_provenance,
        )
    step_path.chmod(0o644)
    step_path.write_bytes(original_step)
    step_path.chmod(0o444)
    task_dir.chmod(0o755)
    unexpected = task_dir / "SUCCESS"
    unexpected.write_text("status=success\n", encoding="utf-8")
    unexpected.chmod(0o444)
    task_dir.chmod(0o555)
    with pytest.raises(ValueError, match="success-shaped terminal marker"):
        job_scheduler._validate_app_preterminal_tree(
            task_dir / diagnostic.PRETERMINAL_ATTESTATION_NAME,
            expected_job_id=12345,
            expected_transaction_id=SUBMISSION_TRANSACTION_ID,
            provenance=promotion_provenance,
        )
    task_dir.chmod(0o755)
    unexpected.unlink()
    task_dir.chmod(0o555)
    _, approval_payload = job_scheduler._stable_payload(approval_path)
    approval_fields = job_scheduler._parse_env_record(
        approval_payload, label="AppLauncher runtime approval"
    )
    finalizer = load_app_finalizer()
    manifest_path = tmp_path / "app_jobs.tsv"
    manifest_row = (
        "\t".join(
            (
                "12345",
                "app-launcher-only",
                "DROID-FoodBussing",
                "1",
                "0",
                "promotion-test",
                SOURCE_TREE_SHA256,
                source_approval["sha256"],
                IMPLEMENTATION_COMMIT,
                "e" * 40,
                "2026-07-09T01:00:00-07:00",
                approval_fields["batch_script_sha256"],
                approval_fields["submission_argv_sha256"],
                approval_fields["held_scheduler_record_sha256"],
                str(approval_path.parent),
                approval_sha256,
            )
        )
        + "\n"
    )
    manifest_path.write_text(
        finalizer.APP_MANIFEST_HEADER + "\n" + manifest_row,
        encoding="utf-8",
    )
    manifest_path.chmod(0o644)
    assert "--captured-sacct-raw" not in APP_FINALIZER_PATH.read_text(encoding="utf-8")
    assert "attest-app-terminal" not in Path(job_scheduler.__file__).read_text(
        encoding="utf-8"
    )
    assert (
        "captured_sacct_raw" not in finalizer.finalize_manifest_job.__code__.co_varnames
    )
    assert "sacct_path" not in finalizer.finalize_manifest_job.__code__.co_varnames
    with pytest.raises(ValueError, match="Immutable JSON is not readable"):
        finalizer.finalize_manifest_job(
            manifest=manifest_path,
            job_id=12345,
            verify_only=True,
            timeout_seconds=11,
        )
    bad_mode_manifest = tmp_path / "bad_mode.tsv"
    bad_mode_manifest.write_text(
        finalizer.APP_MANIFEST_HEADER
        + "\n"
        + manifest_row.replace("\tapp-launcher-only\t", "\tcanary\t", 1),
        encoding="utf-8",
    )
    bad_mode_manifest.chmod(0o644)
    with pytest.raises(ValueError, match="closed diagnostic contract"):
        finalizer.finalize_manifest_job(
            manifest=bad_mode_manifest,
            job_id=12345,
            verify_only=False,
            timeout_seconds=11,
        )
    bad_source_manifest = tmp_path / "bad_source.tsv"
    bad_source_manifest.write_text(
        finalizer.APP_MANIFEST_HEADER
        + "\n"
        + manifest_row.replace(SOURCE_TREE_SHA256, "0" * 64, 1),
        encoding="utf-8",
    )
    bad_source_manifest.chmod(0o644)
    with pytest.raises(ValueError, match="manifest source identity disagrees"):
        finalizer.finalize_manifest_job(
            manifest=bad_source_manifest,
            job_id=12345,
            verify_only=False,
            timeout_seconds=11,
        )
    outside_provenance = tmp_path / "outside/job_12345"
    outside_provenance.mkdir(parents=True)
    outside_manifest = tmp_path / "outside.tsv"
    outside_manifest.write_text(
        finalizer.APP_MANIFEST_HEADER
        + "\n"
        + manifest_row.replace(str(approval_path.parent), str(outside_provenance), 1),
        encoding="utf-8",
    )
    outside_manifest.chmod(0o644)
    with pytest.raises(ValueError, match="canonical and job-bound"):
        finalizer.finalize_manifest_job(
            manifest=outside_manifest,
            job_id=12345,
            verify_only=False,
            timeout_seconds=11,
        )
    duplicate_manifest = tmp_path / "duplicate.tsv"
    duplicate_manifest.write_text(
        finalizer.APP_MANIFEST_HEADER + "\n" + manifest_row + manifest_row,
        encoding="utf-8",
    )
    duplicate_manifest.chmod(0o644)
    with pytest.raises(ValueError, match="exactly one requested job"):
        finalizer.finalize_manifest_job(
            manifest=duplicate_manifest,
            job_id=12345,
            verify_only=False,
            timeout_seconds=11,
        )
    for name in (
        "LD_PRELOAD",
        "LD_AUDIT",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONUSERBASE",
        "BASH_ENV",
        "ENV",
    ):
        monkeypatch.setenv(name, f"hostile-{name.lower()}")
    sacct_invocations: list[tuple[list[str], dict[str, object]]] = []

    def run_live_sacct(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        sacct_invocations.append((list(command), dict(kwargs)))
        return subprocess.CompletedProcess(command, 0, stdout=allocation_raw, stderr="")

    monkeypatch.setattr(job_scheduler.subprocess, "run", run_live_sacct)
    with pytest.raises(ValueError, match="terminal accounting timeout is invalid"):
        finalizer.finalize_manifest_job(
            manifest=manifest_path,
            job_id=12345,
            verify_only=False,
            timeout_seconds=0,
        )

    bounded_timeout_calls: list[dict[str, object]] = []

    def timeout_live_sacct(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        bounded_timeout_calls.append(dict(kwargs))
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monotonic_values = iter((100.0, 100.0, 111.0))
    with monkeypatch.context() as timeout_context:
        timeout_context.setattr(job_scheduler.subprocess, "run", timeout_live_sacct)
        timeout_context.setattr(
            job_scheduler.time, "monotonic", lambda: next(monotonic_values)
        )
        with pytest.raises(TimeoutError, match="bounded subprocess timeout"):
            finalizer.finalize_manifest_job(
                manifest=manifest_path,
                job_id=12345,
                verify_only=False,
                timeout_seconds=11,
            )
    assert len(bounded_timeout_calls) == 1
    assert bounded_timeout_calls[0]["timeout"] == 10.0
    assert bounded_timeout_calls[0]["env"] == {
        "PATH": "/usr/bin:/bin",
        "SLURM_CONF": str(slurm_config_path),
        "LD_LIBRARY_PATH": str(slurm_library_path.parent),
    }

    original_config = slurm_config_path.read_bytes()
    slurm_config_path.write_bytes(original_config + b"tamper")
    with pytest.raises(ValueError, match="slurm[.]conf differs"):
        finalizer.finalize_manifest_job(
            manifest=manifest_path,
            job_id=12345,
            verify_only=False,
            timeout_seconds=11,
        )
    slurm_config_path.write_bytes(original_config)
    original_library = slurm_library_path.read_bytes()
    slurm_library_path.write_bytes(original_library + b"tamper")
    with pytest.raises(ValueError, match="libslurmfull[.]so differs"):
        finalizer.finalize_manifest_job(
            manifest=manifest_path,
            job_id=12345,
            verify_only=False,
            timeout_seconds=11,
        )
    slurm_library_path.write_bytes(original_library)

    promoted = finalizer.finalize_manifest_job(
        manifest=manifest_path,
        job_id=12345,
        verify_only=False,
        timeout_seconds=11,
    )
    assert len(sacct_invocations) == 1
    sacct_command, sacct_kwargs = sacct_invocations[0]
    assert sacct_command == [
        str(sacct_path),
        "-X",
        "--noheader",
        "--parsable2",
        "--jobs=12345",
        "--format=JobIDRaw,State,ExitCode,Submit,Start,End,ElapsedRaw,NodeList,Restarts",
    ]
    assert sacct_kwargs["env"] == {
        "PATH": "/usr/bin:/bin",
        "SLURM_CONF": str(slurm_config_path),
        "LD_LIBRARY_PATH": str(slurm_library_path.parent),
    }
    assert not {
        "LD_PRELOAD",
        "LD_AUDIT",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONUSERBASE",
        "BASH_ENV",
        "ENV",
    } & set(sacct_kwargs["env"])
    assert sacct_kwargs["check"] is False
    assert sacct_kwargs["capture_output"] is True
    assert sacct_kwargs["text"] is True
    assert sacct_kwargs["shell"] is False
    assert sacct_kwargs["timeout"] == job_scheduler.SACCT_SUBPROCESS_TIMEOUT_SECONDS
    query_receipt = job_scheduler.validate_immutable_json(
        Path(promoted["value"]["sacct_query_receipt"]["path"])
    )
    query_value = query_receipt["value"]
    assert query_value["sacct_environment"] == sacct_kwargs["env"]
    assert query_value["sacct_command"] == sacct_command
    assert query_value["sacct_subprocess_timeout_seconds"] == 10.0
    assert query_value["slurm_config"] == slurm_config_identity
    assert query_value["slurm_library"] == slurm_library_identity
    assert "raw" not in promoted["value"]["allocation_terminal"]

    slurm_config_path.write_bytes(original_config + b"verify-tamper")
    with pytest.raises(ValueError, match="slurm[.]conf differs"):
        finalizer.finalize_manifest_job(
            manifest=manifest_path,
            job_id=12345,
            verify_only=True,
            timeout_seconds=11,
        )
    slurm_config_path.write_bytes(original_config)
    slurm_library_path.write_bytes(original_library + b"verify-tamper")
    with pytest.raises(ValueError, match="libslurmfull[.]so differs"):
        finalizer.finalize_manifest_job(
            manifest=manifest_path,
            job_id=12345,
            verify_only=True,
            timeout_seconds=11,
        )
    slurm_library_path.write_bytes(original_library)

    review_decision_path = Path(
        reviewed_value["trace_evidence"]["review_decision"]["path"]
    )
    original_review_decision = review_decision_path.read_bytes()
    review_decision_path.chmod(0o644)
    review_decision_path.write_bytes(original_review_decision + b"tamper")
    review_decision_path.chmod(0o444)
    with pytest.raises(ValueError, match="approval mismatch"):
        finalizer.finalize_manifest_job(
            manifest=manifest_path,
            job_id=12345,
            verify_only=True,
            timeout_seconds=11,
        )
    review_decision_path.chmod(0o644)
    review_decision_path.write_bytes(original_review_decision)
    review_decision_path.chmod(0o444)

    for field, mutation, message in (
        (
            "allocation_terminal",
            {**promoted["value"]["allocation_terminal"], "state": "FAILED"},
            "differs from its governed query receipt",
        ),
        (
            "sacct_query_receipt",
            {**promoted["value"]["sacct_query_receipt"], "sha256": "0" * 64},
            "query receipt identity changed",
        ),
    ):
        mutated = json.loads(json.dumps(promoted["value"]))
        mutated[field] = mutation
        mutated_path = approval_path.parent / f"mutated_{field}.json"
        diagnostic.publish_immutable_json(mutated_path, mutated)
        with pytest.raises(ValueError, match=message):
            job_scheduler.validate_app_terminal_promotion(
                mutated_path,
                expected_job_id=12345,
                expected_transaction_id=SUBMISSION_TRANSACTION_ID,
                expected_app_runtime_approval_sha256=approval_sha256,
            )
    provenance_before_verify = {
        path.name: (path.read_bytes(), stat.S_IMODE(path.stat().st_mode))
        for path in approval_path.parent.iterdir()
        if path.is_file()
    }
    validated = finalizer.finalize_manifest_job(
        manifest=manifest_path,
        job_id=12345,
        verify_only=True,
        timeout_seconds=11,
    )
    assert {
        path.name: (path.read_bytes(), stat.S_IMODE(path.stat().st_mode))
        for path in approval_path.parent.iterdir()
        if path.is_file()
    } == provenance_before_verify
    assert validated["sha256"] == promoted["sha256"]
    assert validated["value"]["authoritative_completion"] is True
    assert validated["value"]["scientific_result"] is None
    assert validated["value"]["allocation_terminal"]["restarts"] == 0
    with pytest.raises(ValueError, match="promotion already exists"):
        finalizer.finalize_manifest_job(
            manifest=manifest_path,
            job_id=12345,
            verify_only=False,
            timeout_seconds=11,
        )
    run_dir.chmod(0o755)
    task_dir.chmod(0o755)


@pytest.mark.parametrize(
    ("log_payload", "expected_mode"),
    [
        (
            VALID_PHASE_LOG,
            "simulation_app_close_returned",
        ),
        (
            VALID_PHASE_LOG.rsplit(
                b"POLARIS_EVAL_PHASE=after_app_launcher_diagnostic_close\n", 1
            )[0],
            "process_exited_zero_before_postclose_marker",
        ),
    ],
)
def test_success_seal_accepts_only_valid_phase_termination_variants(
    tmp_path: Path, log_payload: bytes, expected_mode: str
) -> None:
    run_dir, task_dir, output_directories = create_output_tree(tmp_path)
    publish_success_evidence(
        task_dir,
        output_directories["task_identity"],
        log_payload=log_payload,
    )
    sealed = diagnostic.seal_evidence_tree(
        task_dir=task_dir,
        run_dir=run_dir,
        outcome="success",
        expected_namespace_parent_identity=output_directories[
            "namespace_parent_identity"
        ],
        expected_task_identity=output_directories["task_identity"],
        expected_run_identity=output_directories["run_identity"],
        srun_exit_code=0,
        log_exit_code=0,
        helper_exit_code=0,
    )
    try:
        assert sealed["termination_mode"] == expected_mode
        assert sealed["log_sha256"] == diagnostic._sha256(log_payload)
    finally:
        run_dir.chmod(0o755)
        task_dir.chmod(0o755)


def test_success_seal_rejects_invalid_phase_sequence(tmp_path: Path) -> None:
    run_dir, task_dir, output_directories = create_output_tree(tmp_path)
    publish_success_evidence(
        task_dir,
        output_directories["task_identity"],
        log_payload=(
            b"POLARIS_EVAL_PHASE=after_app_launcher\n"
            b"POLARIS_EVAL_PHASE=before_app_launcher\n"
            b"POLARIS_EVAL_PHASE=before_app_launcher_diagnostic_close\n"
        ),
    )
    with pytest.raises(ValueError, match="phase sequence"):
        diagnostic.seal_evidence_tree(
            task_dir=task_dir,
            run_dir=run_dir,
            outcome="success",
            expected_namespace_parent_identity=output_directories[
                "namespace_parent_identity"
            ],
            expected_task_identity=output_directories["task_identity"],
            expected_run_identity=output_directories["run_identity"],
            srun_exit_code=0,
            log_exit_code=0,
            helper_exit_code=0,
        )


def test_success_seal_rejects_wrong_log_identity_schema(tmp_path: Path) -> None:
    run_dir, task_dir, output_directories = create_output_tree(tmp_path)
    publish_success_evidence(task_dir, output_directories["task_identity"])
    identity_path = task_dir / "app_launcher_only.log.identity.json"
    value, _ = diagnostic.stable_read_immutable_json(identity_path)
    identity_path.chmod(0o644)
    identity_path.unlink()
    diagnostic.publish_immutable_json(
        identity_path,
        {**value, "profile": "hostile_profile"},
        expected_parent_identity=output_directories["task_identity"],
    )
    with pytest.raises(ValueError, match="sealed task log identity mismatch"):
        diagnostic.seal_evidence_tree(
            task_dir=task_dir,
            run_dir=run_dir,
            outcome="success",
            expected_namespace_parent_identity=output_directories[
                "namespace_parent_identity"
            ],
            expected_task_identity=output_directories["task_identity"],
            expected_run_identity=output_directories["run_identity"],
            srun_exit_code=0,
            log_exit_code=0,
            helper_exit_code=0,
        )


def test_success_seal_rejects_tampered_scheduler_terminal_proof(
    tmp_path: Path,
) -> None:
    run_dir, task_dir, output_directories = create_output_tree(tmp_path)
    publish_success_evidence(task_dir, output_directories["task_identity"])
    terminal_path = task_dir / "scheduler_terminal.json"
    value, _ = diagnostic.stable_read_immutable_json(terminal_path)
    value["scontrol_absence"]["stderr"] = (
        "slurm_load_jobs error: Invalid job id specified\n"
    )
    terminal_path.unlink()
    diagnostic.publish_immutable_json(
        terminal_path,
        value,
        expected_parent_identity=output_directories["task_identity"],
    )
    with pytest.raises(ValueError, match="scontrol absence proof"):
        diagnostic.seal_evidence_tree(
            task_dir=task_dir,
            run_dir=run_dir,
            outcome="success",
            expected_namespace_parent_identity=output_directories[
                "namespace_parent_identity"
            ],
            expected_task_identity=output_directories["task_identity"],
            expected_run_identity=output_directories["run_identity"],
            srun_exit_code=0,
            log_exit_code=0,
            helper_exit_code=0,
        )


@pytest.mark.parametrize("field", ["cgroup_events_live", "cgroup_events"])
def test_success_seal_rejects_noncanonical_cgroup_values(
    tmp_path: Path, field: str
) -> None:
    run_dir, task_dir, output_directories = create_output_tree(tmp_path)
    publish_success_evidence(task_dir, output_directories["task_identity"])
    terminal_path = task_dir / "scheduler_terminal.json"
    value, _ = diagnostic.stable_read_immutable_json(terminal_path)
    value[field]["values"]["populated"] ^= 1
    terminal_path.unlink()
    diagnostic.publish_immutable_json(
        terminal_path,
        value,
        expected_parent_identity=output_directories["task_identity"],
    )
    with pytest.raises(ValueError, match="cgroup evidence is not canonical"):
        diagnostic.seal_evidence_tree(
            task_dir=task_dir,
            run_dir=run_dir,
            outcome="success",
            expected_namespace_parent_identity=output_directories[
                "namespace_parent_identity"
            ],
            expected_task_identity=output_directories["task_identity"],
            expected_run_identity=output_directories["run_identity"],
            srun_exit_code=0,
            log_exit_code=0,
            helper_exit_code=0,
        )


def test_preterminal_seal_rereads_attestation_before_directory_chmod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, task_dir, output_directories = create_output_tree(tmp_path)
    publish_success_evidence(task_dir, output_directories["task_identity"])
    original_read = diagnostic._stable_read_immutable_json_at

    def tamper_attestation_read(task_fd, *, path, **kwargs):
        value, identity = original_read(task_fd, path=path, **kwargs)
        if path.name == diagnostic.PRETERMINAL_ATTESTATION_NAME:
            value = {**value, "status": "tampered_after_publish"}
        return value, identity

    monkeypatch.setattr(
        diagnostic, "_stable_read_immutable_json_at", tamper_attestation_read
    )
    with pytest.raises(
        RuntimeError,
        match="preterminal attestation changed before terminal directory seal",
    ):
        diagnostic.seal_evidence_tree(
            task_dir=task_dir,
            run_dir=run_dir,
            outcome="success",
            expected_namespace_parent_identity=output_directories[
                "namespace_parent_identity"
            ],
            expected_task_identity=output_directories["task_identity"],
            expected_run_identity=output_directories["run_identity"],
            srun_exit_code=0,
            log_exit_code=0,
            helper_exit_code=0,
        )
    assert stat.S_IMODE(task_dir.stat().st_mode) == 0o755
    assert stat.S_IMODE(run_dir.stat().st_mode) == 0o755


@pytest.mark.parametrize(
    ("srun_code", "log_code", "helper_code"),
    [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
)
def test_success_seal_requires_worker_supplied_zero_preseal_codes(
    tmp_path: Path, srun_code: int, log_code: int, helper_code: int
) -> None:
    run_dir, task_dir, output_directories = create_output_tree(tmp_path)
    publish_success_evidence(task_dir, output_directories["task_identity"])
    with pytest.raises(ValueError, match="worker-supplied zero pre-seal exit codes"):
        diagnostic.seal_evidence_tree(
            task_dir=task_dir,
            run_dir=run_dir,
            outcome="success",
            expected_namespace_parent_identity=output_directories[
                "namespace_parent_identity"
            ],
            expected_task_identity=output_directories["task_identity"],
            expected_run_identity=output_directories["run_identity"],
            srun_exit_code=srun_code,
            log_exit_code=log_code,
            helper_exit_code=helper_code,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_success_seal_rejects_nonexact_entry_closure(
    tmp_path: Path, mutation: str
) -> None:
    run_dir, task_dir, output_directories = create_output_tree(tmp_path)
    publish_success_evidence(task_dir, output_directories["task_identity"])
    if mutation == "missing":
        (task_dir / "startup_preclose.json").unlink()
    else:
        diagnostic.publish_immutable_json(
            task_dir / "unexpected.json",
            {"unexpected": True},
            expected_parent_identity=output_directories["task_identity"],
        )
    with pytest.raises(ValueError, match="entry closure mismatch"):
        diagnostic.seal_evidence_tree(
            task_dir=task_dir,
            run_dir=run_dir,
            outcome="success",
            expected_namespace_parent_identity=output_directories[
                "namespace_parent_identity"
            ],
            expected_task_identity=output_directories["task_identity"],
            expected_run_identity=output_directories["run_identity"],
            srun_exit_code=0,
            log_exit_code=0,
            helper_exit_code=0,
        )


def test_create_output_cleanup_never_removes_from_replacement_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    task_dir = run_dir / "app_launcher_only"
    moved = tmp_path / "moved-run"
    original_validate = diagnostic._validate_child_directory_binding

    def replace_before_task_validation(parent_fd, child_fd, *, name, path):
        if path == task_dir:
            run_dir.rename(moved)
            run_dir.mkdir()
            (run_dir / "app_launcher_only").mkdir()
            raise RuntimeError("injected child binding failure")
        return original_validate(parent_fd, child_fd, name=name, path=path)

    monkeypatch.setattr(
        diagnostic, "_validate_child_directory_binding", replace_before_task_validation
    )
    with pytest.raises(RuntimeError, match="injected child binding failure"):
        diagnostic.create_output_directories(
            run_dir=run_dir,
            task_dir=task_dir,
            expected_parent_identity=diagnostic.capture_directory_identity(tmp_path),
        )
    assert (run_dir / "app_launcher_only").is_dir()
    assert (moved / "app_launcher_only").is_dir()


def test_failure_evidence_tree_binds_attested_inventory(tmp_path: Path) -> None:
    run_dir, task_dir, output_directories = create_output_tree(tmp_path)
    publish_success_evidence(task_dir, output_directories["task_identity"])
    for name in (
        "startup_preclose.json",
        "startup_preclose.ready.json",
        "app_launcher_only.log",
        "app_launcher_only.log.identity.json",
    ):
        (task_dir / name).unlink()
    attestation = diagnostic.publish_failure_attestation(
        task_dir=task_dir,
        primary_exit_code=7,
        srun_exit_code=7,
        log_exit_code=0,
        helper_exit_code=143,
        signal_name="none",
        expected_task_identity=output_directories["task_identity"],
    )
    assert attestation["mode"] == "0444"
    sealed = diagnostic.seal_evidence_tree(
        task_dir=task_dir,
        run_dir=run_dir,
        outcome="failure",
        expected_namespace_parent_identity=output_directories[
            "namespace_parent_identity"
        ],
        expected_task_identity=output_directories["task_identity"],
        expected_run_identity=output_directories["run_identity"],
    )
    try:
        assert set(sealed["artifacts"]) == {
            "startup_preexec.json",
            "scheduler_request.json",
            "scheduler_handoff.json",
            "scheduler_terminal_request.json",
            "scheduler_terminal.json",
            diagnostic.FAILURE_ATTESTATION_NAME,
        }
        assert stat.S_IMODE(task_dir.stat().st_mode) == 0o555
        assert stat.S_IMODE(run_dir.stat().st_mode) == 0o555
    finally:
        run_dir.chmod(0o755)
        task_dir.chmod(0o755)


def test_failure_attestation_rejects_terminal_proof_for_another_job_cgroup(
    tmp_path: Path,
) -> None:
    _, task_dir, output_directories = create_output_tree(tmp_path)
    publish_success_evidence(task_dir, output_directories["task_identity"])
    preexec_path = task_dir / "startup_preexec.json"
    preexec, _ = diagnostic.stable_read_immutable_json(preexec_path)
    preexec["runtime"]["cgroup"]["job_step_path"] = (
        "/slurm/uid_1000/job_99999/step_0/task_0"
    )
    preexec_path.unlink()
    diagnostic.publish_immutable_json(
        preexec_path,
        preexec,
        expected_parent_identity=output_directories["task_identity"],
    )
    with pytest.raises(ValueError, match="cgroup path is not request-bound"):
        diagnostic.publish_failure_attestation(
            task_dir=task_dir,
            primary_exit_code=7,
            srun_exit_code=7,
            log_exit_code=0,
            helper_exit_code=0,
            signal_name="none",
            expected_task_identity=output_directories["task_identity"],
        )


@pytest.mark.parametrize("mutation", ["add", "remove"])
def test_failure_seal_rejects_changes_after_attestation(
    tmp_path: Path, mutation: str
) -> None:
    run_dir, task_dir, output_directories = create_output_tree(tmp_path)
    original = task_dir / "startup_preexec.json"
    publish_success_evidence(task_dir, output_directories["task_identity"])
    for name in (
        "startup_preclose.json",
        "startup_preclose.ready.json",
        "app_launcher_only.log",
        "app_launcher_only.log.identity.json",
    ):
        (task_dir / name).unlink()
    diagnostic.publish_failure_attestation(
        task_dir=task_dir,
        primary_exit_code=7,
        srun_exit_code=7,
        log_exit_code=0,
        helper_exit_code=143,
        signal_name="none",
        expected_task_identity=output_directories["task_identity"],
    )
    if mutation == "add":
        diagnostic.publish_immutable_json(
            task_dir / "startup_preclose.json",
            {"late": True},
            expected_parent_identity=output_directories["task_identity"],
        )
    else:
        original.unlink()
    with pytest.raises(ValueError, match="changed after terminal attestation"):
        diagnostic.seal_evidence_tree(
            task_dir=task_dir,
            run_dir=run_dir,
            outcome="failure",
            expected_namespace_parent_identity=output_directories[
                "namespace_parent_identity"
            ],
            expected_task_identity=output_directories["task_identity"],
            expected_run_identity=output_directories["run_identity"],
        )


def test_target_argv_is_closed_to_exact_public_entrypoint(tmp_path: Path) -> None:
    preexec = tmp_path / "startup_preexec.json"
    preclose = tmp_path / "startup_preclose.json"
    target = canonical_target(tmp_path)
    assert (
        diagnostic.validate_target_argv(
            target,
            preexec_path=preexec,
            preclose_path=preclose,
            expected_gpu_uuid=GPU_UUID,
            expected_port=32345,
        )
        == target
    )
    assert diagnostic.python_argv_after_exec(target) == target[1:]
    mutations = [
        ["python", *target[1:]],
        [*target, "--startup-diagnostic", "app_launcher_only"],
        [*target, "--foo", "bar"],
        [*target, "--policy.client", "Fake"],
        [*target, "--instruction", "unexpected"],
    ]
    for option, replacement in (
        ("--policy.client", "Fake"),
        ("--rollouts", "999999999"),
    ):
        changed = list(target)
        changed[changed.index(option) + 1] = replacement
        mutations.append(changed)
    reordered = list(target)
    reordered[2:6] = reordered[4:6] + reordered[2:4]
    mutations.append(reordered)
    for mutation in mutations:
        with pytest.raises(ValueError):
            diagnostic.validate_target_argv(
                mutation,
                preexec_path=preexec,
                preclose_path=preclose,
                expected_gpu_uuid=GPU_UUID,
                expected_port=32345,
            )


@pytest.mark.parametrize(
    "value,expected",
    [
        ("cpu=16", {"cpu": "16"}),
        ("cpu=16,gres/gpu:1", {"cpu": "16", "gres/gpu": "1"}),
    ],
)
def test_tres_per_task_accepts_only_observed_single_gpu_shapes(
    value: str, expected: dict[str, str]
) -> None:
    assert diagnostic.parse_tres_per_task(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "cpu=16,gres/gpu:0",
        "cpu=16,gres/gpu:2",
        "cpu=16,gres/gpu:1,gres/gpu:1",
        "cpu=32,gres/gpu:1",
        "gres/gpu:1",
        "cpu=16,gres/gpu=1",
    ],
)
def test_tres_per_task_rejects_missing_mismatched_and_multigpu_shapes(
    value: str,
) -> None:
    with pytest.raises(ValueError):
        diagnostic.parse_tres_per_task(value)


def test_scheduler_records_cross_bind_job_step_requested_and_allocated_tres() -> None:
    job = diagnostic.parse_job_scheduler_record(
        job_scheduler_record(),
        expected_job_id=12345,
        expected_transaction_id=SUBMISSION_TRANSACTION_ID,
    )
    step = diagnostic.parse_step_scheduler_record(
        step_scheduler_record(),
        expected_job_id=12345,
        expected_step_id=0,
        requested_tres_per_task="cpu=16,gres/gpu:1",
        expected_node="pool0-00002",
    )
    assert job["requested_tres_items"]["gres/gpu"] == "1"
    assert job["allocated_tres_items"]["gres/gpu"] == "1"
    assert step["requested_tres_items"]["gres/gpu"] == "1"
    assert step["allocated_tres_items"]["gres/gpu"] == "1"
    assert job["requested_tres_items"]["cpu"] == "16"
    assert step["allocated_tres_items"]["cpu"] == "16"


@pytest.mark.parametrize(
    "job_record,step_record",
    [
        (job_scheduler_record(requested_gpus=2), step_scheduler_record()),
        (job_scheduler_record(allocated_gpus=2), step_scheduler_record()),
        (job_scheduler_record(cpus=32), step_scheduler_record()),
        (job_scheduler_record(), step_scheduler_record(gpus=2)),
        (job_scheduler_record(), step_scheduler_record(cpus=32)),
        (
            job_scheduler_record().replace("JobId=12345", "JobId=99999"),
            step_scheduler_record(),
        ),
        (
            job_scheduler_record(),
            step_scheduler_record().replace("StepId=12345.0", "StepId=12345.1"),
        ),
        (
            job_scheduler_record().replace(" ReqTRES=", " Requested="),
            step_scheduler_record(),
        ),
        (job_scheduler_record(), step_scheduler_record().replace(" TRES=", " Alloc=")),
    ],
)
def test_scheduler_records_reject_missing_mismatch_and_multigpu(
    job_record: str, step_record: str
) -> None:
    with pytest.raises(ValueError):
        job = diagnostic.parse_job_scheduler_record(
            job_record,
            expected_job_id=12345,
            expected_transaction_id=SUBMISSION_TRANSACTION_ID,
        )
        diagnostic.parse_step_scheduler_record(
            step_record,
            expected_job_id=12345,
            expected_step_id=0,
            requested_tres_per_task="cpu=16,gres/gpu:1",
            expected_node="pool0-00002",
        )
        assert job


def test_job_scheduler_record_rejects_every_closed_field_and_duplicate() -> None:
    baseline = job_scheduler_record()
    mutations = [
        baseline + " Account=nvr_lpr_rvp",
        baseline + " Partition=batch_long",
        baseline.replace("UserId=lzha(158351)", "UserId=other(1)"),
        baseline.replace("Account=nvr_lpr_rvp", "Account=other"),
        baseline.replace("QOS=normal", "QOS=high"),
        baseline.replace("JobState=RUNNING", "JobState=COMPLETED"),
        baseline.replace("Partition=batch", "Partition=batch_long"),
        baseline.replace("NodeList=pool0-00002", "NodeList=other-1"),
        baseline.replace("BatchHost=pool0-00002", "BatchHost=pool0-00003"),
        baseline.replace("NumNodes=1", "NumNodes=2"),
        baseline.replace("NumCPUs=16", "NumCPUs=8"),
        baseline.replace("NumTasks=1", "NumTasks=2"),
        baseline.replace("CPUs/Task=16", "CPUs/Task=8"),
        baseline.replace("OverSubscribe=OK", "OverSubscribe=NO"),
        baseline.replace("TresPerNode=gres/gpu:1", "TresPerNode=gres/gpu:2"),
        baseline.replace("TresPerTask=cpu=16", "TresPerTask=cpu=8"),
        baseline.replace("Requeue=0", "Requeue=1"),
        baseline.replace("Restarts=0", "Restarts=1"),
        baseline.replace(
            f"Comment={SUBMISSION_TRANSACTION_ID}",
            f"Comment=pi05-{'0' * 40}",
        ),
        baseline + " Requeue=0",
        baseline.replace("billing=1,cpu=16", "billing=2,cpu=16", 1),
        baseline.replace("mem=128G", "mem=64G", 1),
        baseline.replace(
            "AllocTRES=billing=1",
            "AllocTRES=unknown=1,billing=1",
        ),
    ]
    for mutation in mutations:
        with pytest.raises(ValueError):
            diagnostic.parse_job_scheduler_record(
                mutation,
                expected_job_id=12345,
                expected_transaction_id=SUBMISSION_TRANSACTION_ID,
            )


def test_step_scheduler_record_rejects_every_closed_field_and_tres_drift() -> None:
    baseline = step_scheduler_record()
    mutations = [
        baseline + " State=RUNNING",
        baseline + " TRES=cpu=16,gres/gpu=1,node=1",
        baseline.replace("Name=python", "Name=bash"),
        baseline.replace("UserId=lzha(158351)", "UserId=other(1)"),
        baseline.replace("State=RUNNING", "State=COMPLETED"),
        baseline.replace("Partition=batch", "Partition=batch_long"),
        baseline.replace("Nodes=1", "Nodes=2"),
        baseline.replace("NodeList=pool0-00002", "NodeList=pool0-00003"),
        baseline.replace("CPUs=16", "CPUs=8"),
        baseline.replace("Tasks=1", "Tasks=2"),
        baseline.replace("gres/gpu=1", "gres/gpu=2"),
        baseline.replace("mem=128G", "mem=64G"),
        baseline.replace("node=1", "node=2"),
        baseline.replace("TRES=", "TRES=unknown=1,"),
    ]
    for mutation in mutations:
        with pytest.raises(ValueError):
            diagnostic.parse_step_scheduler_record(
                mutation,
                expected_job_id=12345,
                expected_step_id=0,
                requested_tres_per_task="cpu=16,gres/gpu:1",
                expected_node="pool0-00002",
            )


@pytest.mark.parametrize(
    "value",
    [
        "cpu=16,gres/gpu=1,node=1",
        "cpu=16,gres/gpu=1,mem=128G,node=1",
        "billing=1,cpu=16,gres/gpu=1,mem=128G,node=1",
    ],
)
def test_step_tres_accepts_only_the_three_observed_exact_keysets(value: str) -> None:
    parsed = diagnostic.parse_step_allocated_tres(value, label="step")
    assert parsed["cpu"] == "16"
    assert parsed["gres/gpu"] == "1"
    assert parsed["node"] == "1"


def test_scheduler_client_identity_rejects_every_mutable_component() -> None:
    mutation_paths = [
        (("profile",), "other"),
        (("runtime_closure_approval_sha256",), "0" * 64),
        (("scontrol", "path"), "/tmp/scontrol"),
        (("scontrol", "mode"), "0777"),
        (("scontrol", "nlink"), 2),
        (("scontrol", "size"), 1),
        (("scontrol", "sha256"), "0" * 64),
        (("slurm_library", "path"), "/tmp/libslurm.so"),
        (("slurm_library", "mode"), "0666"),
        (("slurm_library", "nlink"), 2),
        (("slurm_library", "size"), 1),
        (("slurm_library", "sha256"), "0" * 64),
        (("slurm_config", "path"), "/tmp/slurm.conf"),
        (("slurm_config", "mode"), "0666"),
        (("slurm_config", "nlink"), 2),
        (("slurm_config", "size"), 0),
        (("slurm_config", "sha256"), "0" * 64),
        (("execution_environment", "PATH"), "/tmp"),
        (("execution_environment", "SLURM_CONF"), "/tmp/slurm.conf"),
        (("execution_environment", "LD_LIBRARY_PATH"), "/tmp"),
    ]
    for path, replacement in mutation_paths:
        changed = json.loads(json.dumps(synthetic_scheduler_client()))
        cursor = changed
        for component in path[:-1]:
            cursor = cursor[component]
        cursor[path[-1]] = replacement
        with pytest.raises(ValueError):
            diagnostic.validate_scheduler_client_identity(
                changed, **scheduler_approval_kwargs()
            )


def test_scheduler_broker_uses_pinned_client_and_seals_exact_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, task_dir, output_directories = create_output_tree(tmp_path)
    request_path = task_dir / "scheduler_request.json"
    output_path = task_dir / "scheduler_handoff.json"
    terminal_request_path = task_dir / "scheduler_terminal_request.json"
    terminal_output_path = task_dir / "scheduler_terminal.json"
    diagnostic.publish_scheduler_request(
        request_path,
        environ=runtime_environment(),
        expected_gpu_uuid=GPU_UUID,
        expected_parent_identity=output_directories["task_identity"],
    )
    client = synthetic_scheduler_client()

    def capture_client(**approval_kwargs):
        assert approval_kwargs == scheduler_approval_kwargs()
        return client

    monkeypatch.setattr(diagnostic, "capture_scheduler_client_identity", capture_client)
    calls: list[tuple[list[str], dict[str, str]]] = []
    step_calls = 0

    def fake_run(argv, *, env, check, capture_output, text, timeout):
        nonlocal step_calls
        assert check is False and capture_output is True and text is True
        assert timeout == 10
        calls.append((list(argv), dict(env)))
        if argv[2:5] == ["job", "--oneliner", "12345"]:
            return subprocess.CompletedProcess(
                argv, 0, stdout=job_scheduler_record() + "\n", stderr=""
            )
        elif argv[2:5] == ["step", "--oneliner", "12345.0"]:
            step_calls += 1
            if step_calls == 1:
                return subprocess.CompletedProcess(
                    argv, 0, stdout=step_scheduler_record() + "\n", stderr=""
                )
            return subprocess.CompletedProcess(
                argv,
                1,
                stdout="",
                stderr=(
                    "scontrol: error: scontrol_print_step: "
                    "slurm_get_job_steps(12345.0) failed: "
                    "Invalid job id specified\n"
                ),
            )
        if argv[0] == str(diagnostic.EXPECTED_SACCT_PATH):
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=(
                    "12345.0|COMPLETED|0:0|2026-07-09T00:00:00|"
                    "2026-07-09T00:00:10|10|pool0-00002|\n"
                ),
                stderr="",
            )
        raise AssertionError(argv)

    monkeypatch.setattr(subprocess, "run", fake_run)
    environment = runtime_environment()
    for name in (
        "SLURM_JOB_ID",
        "SLURM_JOB_GPUS",
        "SLURM_GPUS_ON_NODE",
        "SLURM_JOB_ACCOUNT",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_QOS",
        "SLURM_JOB_USER",
        "NVIDIA_VISIBLE_DEVICES",
        *diagnostic.EXECUTION_APPROVAL_ENVIRONMENT,
    ):
        monkeypatch.setenv(name, environment[name])
    diagnostic.publish_immutable_json(
        task_dir / "startup_preexec.json",
        {
            "runtime": {
                "cgroup": {"job_step_path": "/slurm/uid_1000/job_12345/step_0/task_0"}
            }
        },
        expected_parent_identity=output_directories["task_identity"],
    )
    cgroup_events_path = Path(
        "/sys/fs/cgroup/slurm/uid_1000/job_12345/step_0/cgroup.events"
    )
    sentinel_fd = 987654
    real_open = os.open
    real_close = os.close

    def fake_open(path, *args, **kwargs):
        if Path(path) == cgroup_events_path:
            return sentinel_fd
        return real_open(path, *args, **kwargs)

    def fake_close(descriptor):
        if descriptor != sentinel_fd:
            real_close(descriptor)

    cgroup_reads = 0

    def fake_read_cgroup_events(descriptor, path):
        nonlocal cgroup_reads
        assert descriptor == sentinel_fd and path == cgroup_events_path
        cgroup_reads += 1
        populated = 1 if cgroup_reads == 1 else 0
        raw = f"populated {populated}\n"
        return {
            "path": str(path),
            "raw": raw,
            "raw_sha256": diagnostic._sha256(raw.encode()),
            "values": {"populated": populated},
        }

    monkeypatch.setattr(os, "open", fake_open)
    monkeypatch.setattr(os, "close", fake_close)
    monkeypatch.setattr(diagnostic, "_read_cgroup_events", fake_read_cgroup_events)
    result: dict[str, object] = {}

    def run_broker() -> None:
        try:
            result["identity"] = diagnostic.broker_scheduler_handoff(
                request_path=request_path,
                output_path=output_path,
                terminal_request_path=terminal_request_path,
                terminal_output_path=terminal_output_path,
                expected_parent_identity=output_directories["task_identity"],
                timeout_seconds=1.0,
                terminal_timeout_seconds=1.0,
            )
        except BaseException as error:
            result["error"] = error

    broker_thread = threading.Thread(target=run_broker)
    broker_thread.start()
    deadline = time.monotonic() + 2.0
    while not output_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert output_path.exists()
    diagnostic.publish_scheduler_terminal_request(
        request_path=request_path,
        handoff_path=output_path,
        output_path=terminal_request_path,
        srun_exit_code=0,
        expected_parent_identity=output_directories["task_identity"],
    )
    broker_thread.join(timeout=3.0)
    assert not broker_thread.is_alive()
    assert "error" not in result, result.get("error")
    identity = result["identity"]
    assert isinstance(identity, dict) and identity["mode"] == "0444"
    assert len(calls) == 4
    assert all(call_env == client["execution_environment"] for _, call_env in calls)
    value, _ = diagnostic.stable_read_immutable_json(
        terminal_output_path,
        expected_parent_identity=output_directories["task_identity"],
    )
    assert value["scheduler_client"] == client
    assert value["scontrol_absence"]["stderr"].endswith(
        "slurm_get_job_steps(12345.0) failed: Invalid job id specified\n"
    )
    assert value["sacct_terminal"]["state"] == "COMPLETED"
    assert value["cgroup_events_live"]["values"]["populated"] == 1
    assert value["cgroup_events"]["values"]["populated"] == 0


@pytest.mark.parametrize(
    ("handoff_timeout", "terminal_timeout"),
    [
        (0.0, 1.0),
        (float("nan"), 1.0),
        (121.0, 1.0),
        (1.0, 0.0),
        (1.0, float("inf")),
        (1.0, 601.0),
    ],
)
def test_scheduler_broker_rejects_nonfinite_or_unbounded_timeouts(
    tmp_path: Path, handoff_timeout: float, terminal_timeout: float
) -> None:
    with pytest.raises(ValueError, match="timeout is outside the closed range"):
        diagnostic.broker_scheduler_handoff(
            request_path=tmp_path / "request.json",
            output_path=tmp_path / "handoff.json",
            terminal_request_path=tmp_path / "terminal-request.json",
            terminal_output_path=tmp_path / "terminal.json",
            expected_parent_identity=diagnostic.capture_directory_identity(tmp_path),
            timeout_seconds=handoff_timeout,
            terminal_timeout_seconds=terminal_timeout,
        )


def test_sacct_terminal_record_rejects_end_before_start() -> None:
    with pytest.raises(ValueError, match="closed step contract"):
        diagnostic.parse_sacct_terminal_record(
            "12345.0|COMPLETED|0:0|2026-07-09T00:00:10|"
            "2026-07-09T00:00:00|10|pool0-00002|\n",
            expected_job_id=12345,
            expected_step_id=0,
            expected_node="pool0-00002",
        )


def test_scheduler_handoff_is_canonical_nonreplacing_and_bound_to_request(
    tmp_path: Path,
) -> None:
    request_path = tmp_path / "scheduler_request.json"
    handoff_path = tmp_path / "scheduler_handoff.json"
    request_identity = diagnostic.publish_scheduler_request(
        request_path,
        environ=runtime_environment(),
        expected_gpu_uuid=GPU_UUID,
    )
    handoff_identity = diagnostic.seal_scheduler_handoff(
        request_path=request_path,
        output_path=handoff_path,
        job_record=job_scheduler_record(),
        step_record=step_scheduler_record(),
        scheduler_client=synthetic_scheduler_client(),
        host_environ=runtime_environment(),
        **scheduler_approval_kwargs(),
    )
    value, observed = diagnostic.stable_read_immutable_json(handoff_path)
    assert observed == handoff_identity
    assert value["request"] == request_identity
    diagnostic._validate_scheduler_handoff(
        value,
        request=diagnostic.scheduler_request_value(
            environ=runtime_environment(), expected_gpu_uuid=GPU_UUID
        ),
        request_identity=request_identity,
        **scheduler_approval_kwargs(),
    )
    with pytest.raises(FileExistsError):
        diagnostic.seal_scheduler_handoff(
            request_path=request_path,
            output_path=handoff_path,
            job_record=job_scheduler_record(),
            step_record=step_scheduler_record(),
            scheduler_client=synthetic_scheduler_client(),
            host_environ=runtime_environment(),
            **scheduler_approval_kwargs(),
        )


def test_postexec_scheduler_reads_require_creator_bound_task_directory(
    tmp_path: Path,
) -> None:
    _, task_dir, output_directories = create_output_tree(tmp_path)
    request_path = task_dir / "scheduler_request.json"
    handoff_path = task_dir / "scheduler_handoff.json"
    diagnostic.publish_scheduler_request(
        request_path,
        environ=runtime_environment(),
        expected_gpu_uuid=GPU_UUID,
        expected_parent_identity=output_directories["task_identity"],
    )
    handoff_identity = diagnostic.seal_scheduler_handoff(
        request_path=request_path,
        output_path=handoff_path,
        job_record=job_scheduler_record(),
        step_record=step_scheduler_record(),
        scheduler_client=synthetic_scheduler_client(),
        host_environ=runtime_environment(),
        expected_parent_identity=output_directories["task_identity"],
        **scheduler_approval_kwargs(),
    )
    environment = runtime_environment(
        POLARIS_STARTUP_DIAGNOSTIC_TASK_DIR_IDENTITY=output_directories[
            "task_identity"
        ],
        POLARIS_STARTUP_DIAGNOSTIC_SCHEDULER_HANDOFF_PATH=str(handoff_path),
        POLARIS_STARTUP_DIAGNOSTIC_SCHEDULER_HANDOFF_SHA256=handoff_identity["sha256"],
    )
    moved = tmp_path / "moved-task"
    task_dir.rename(moved)
    task_dir.mkdir()
    with pytest.raises(RuntimeError, match="creator-observed identity mismatch"):
        diagnostic._scheduler_handoff_runtime_value(
            environ=environment,
            expected_gpu_uuid=GPU_UUID,
        )


def test_scheduler_handoff_rejects_hostile_identity_and_record_drift(
    tmp_path: Path,
) -> None:
    request_path = tmp_path / "scheduler_request.json"
    handoff_path = tmp_path / "scheduler_handoff.json"
    request_identity = diagnostic.publish_scheduler_request(
        request_path,
        environ=runtime_environment(),
        expected_gpu_uuid=GPU_UUID,
    )
    for job_record, step_record in (
        (
            job_scheduler_record().replace("JobId=12345", "JobId=99999"),
            step_scheduler_record(),
        ),
        (
            job_scheduler_record(),
            step_scheduler_record().replace("StepId=12345.0", "StepId=12345.1"),
        ),
        (job_scheduler_record(allocated_gpus=2), step_scheduler_record()),
        (job_scheduler_record(), step_scheduler_record(gpus=2)),
    ):
        with pytest.raises(ValueError):
            diagnostic.seal_scheduler_handoff(
                request_path=request_path,
                output_path=handoff_path,
                job_record=job_record,
                step_record=step_record,
                scheduler_client=synthetic_scheduler_client(),
                host_environ=runtime_environment(),
                **scheduler_approval_kwargs(),
            )
        assert not handoff_path.exists()

    supplied = synthetic_scheduler_handoff()
    supplied["value"]["request"] = {**request_identity, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="request identity mismatch"):
        diagnostic._validate_scheduler_handoff(
            supplied["value"],
            request=diagnostic.scheduler_request_value(
                environ=runtime_environment(), expected_gpu_uuid=GPU_UUID
            ),
            request_identity=request_identity,
            **scheduler_approval_kwargs(),
        )


def test_scheduler_request_and_handoff_waits_are_bounded(tmp_path: Path) -> None:
    request_path = tmp_path / "missing_request.json"
    with pytest.raises(TimeoutError, match="scheduler request"):
        diagnostic.wait_for_scheduler_request(request_path, timeout_seconds=0)

    request = diagnostic.scheduler_request_value(
        environ=runtime_environment(), expected_gpu_uuid=GPU_UUID
    )
    request_identity = {
        "path": str(request_path),
        "mode": "0444",
        "nlink": 1,
        "size": 1,
        "sha256": "5" * 64,
    }
    with pytest.raises(TimeoutError, match="scheduler handoff"):
        diagnostic.wait_for_scheduler_handoff(
            tmp_path / "missing_handoff.json",
            request=request,
            request_identity=request_identity,
            timeout_seconds=0,
            **scheduler_approval_kwargs(),
        )


def test_closed_eval_environment_matches_public_shape_plus_step_handoff(
    tmp_path: Path,
) -> None:
    run_dir, task_dir, output_directories = create_output_tree(tmp_path)
    value = diagnostic.build_closed_eval_environment(
        inherited=runtime_environment(),
        source_root=Path("/polaris-source"),
        data_root=Path("/physical/PolaRiS-Hub"),
        cache_root=Path("/cache"),
        preexec_sha256="4" * 64,
        scheduler_handoff_path=tmp_path / "scheduler_handoff.json",
        scheduler_handoff_sha256="6" * 64,
        run_dir=run_dir,
        task_dir=task_dir,
        namespace_parent_identity=output_directories["namespace_parent_identity"],
        run_dir_identity=output_directories["run_identity"],
        task_dir_identity=output_directories["task_identity"],
    )
    assert (
        value["PATH"] == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    )
    assert value["PYTHONPATH"] == (
        "/polaris-source/src:"
        "/polaris-source/third_party/openpi/packages/openpi-client/src"
    )
    assert value["SLURM_JOB_ID"] == "12345"
    assert value["SLURM_STEP_ID"] == "0"
    assert value["SLURM_JOB_GPUS"] == "2"
    assert value["SLURM_STEP_GPUS"] == "2"
    assert value["SLURM_GPUS_ON_NODE"] == "1"
    assert value["SLURM_GPUS_PER_TASK"] == "1"
    assert value["SLURM_TRES_PER_TASK"] == "cpu=16,gres/gpu:1"
    assert value["SLURM_CPUS_PER_TASK"] == "16"
    assert value["SLURM_NTASKS"] == "1"
    assert value["SLURM_JOB_NUM_NODES"] == "1"
    assert value["SLURM_MEM_PER_NODE"] == "131072"
    assert value["SLURM_JOB_ACCOUNT"] == "nvr_lpr_rvp"
    assert value["SLURM_JOB_PARTITION"] == "batch"
    assert value["SLURM_JOB_QOS"] == "normal"
    assert value["SLURM_JOB_USER"] == "lzha"
    assert value["POLARIS_EVAL_MODE"] == "app_launcher_only"
    assert value["SUBMISSION_TRANSACTION_ID"] == SUBMISSION_TRANSACTION_ID
    assert value["POLARIS_STARTUP_DIAGNOSTIC_RUN_DIR"] == str(run_dir)
    assert value["POLARIS_STARTUP_DIAGNOSTIC_TASK_DIR"] == str(task_dir)
    assert (
        value["POLARIS_OUTPUT_NAMESPACE_PARENT_IDENTITY"]
        == output_directories["namespace_parent_identity"]
    )
    assert (
        value["POLARIS_STARTUP_DIAGNOSTIC_RUN_DIR_IDENTITY"]
        == output_directories["run_identity"]
    )
    assert (
        value["POLARIS_STARTUP_DIAGNOSTIC_TASK_DIR_IDENTITY"]
        == output_directories["task_identity"]
    )
    assert value["NVIDIA_VISIBLE_DEVICES"] == GPU_UUID
    assert "CUDA_VISIBLE_DEVICES" not in value
    assert set(value) == {
        "PATH",
        "LANG",
        "LC_ALL",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "VK_DRIVER_FILES",
        "ACCEPT_EULA",
        "OMNI_KIT_ACCEPT_EULA",
        "PRIVACY_CONSENT",
        "OMNI_KIT_ALLOW_ROOT",
        "PYTHONUNBUFFERED",
        "PYTHONPATH",
        "POLARIS_DATA_PATH",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "HOME",
        "POLARIS_EVAL_MODE",
        "SUBMISSION_TRANSACTION_ID",
        "SLURM_JOB_ID",
        "SLURM_STEP_ID",
        "SLURM_JOB_GPUS",
        "SLURM_STEP_GPUS",
        "SLURM_GPUS_ON_NODE",
        "SLURM_GPUS_PER_TASK",
        "SLURM_TRES_PER_TASK",
        "SLURM_CPUS_PER_TASK",
        "SLURM_NTASKS",
        "SLURM_JOB_NUM_NODES",
        "SLURM_MEM_PER_NODE",
        "SLURM_JOB_ACCOUNT",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_QOS",
        "SLURM_JOB_USER",
        "POLARIS_STARTUP_DIAGNOSTIC_RUN_DIR",
        "POLARIS_STARTUP_DIAGNOSTIC_TASK_DIR",
        "POLARIS_OUTPUT_NAMESPACE_PARENT_IDENTITY",
        "POLARIS_STARTUP_DIAGNOSTIC_RUN_DIR_IDENTITY",
        "POLARIS_STARTUP_DIAGNOSTIC_TASK_DIR_IDENTITY",
        "POLARIS_STARTUP_DIAGNOSTIC_PREEXEC_SHA256",
        "POLARIS_STARTUP_DIAGNOSTIC_SCHEDULER_HANDOFF_PATH",
        "POLARIS_STARTUP_DIAGNOSTIC_SCHEDULER_HANDOFF_SHA256",
        *diagnostic.EXECUTION_APPROVAL_ENVIRONMENT,
        *diagnostic.PYXIS_IMAGE_ENVIRONMENT,
        *diagnostic.SOURCE_IDENTITY_ENVIRONMENT,
    }


def test_context_continuity_rejects_process_step_gpu_cgroup_device_and_source_drift() -> (
    None
):
    runtime = synthetic_runtime()
    preexec = preexec_value(runtime)
    diagnostic._validate_context_continuity(preexec, runtime)
    closed_environment_runtime = json.loads(json.dumps(runtime))
    closed_environment_runtime["slurm"]["gpu_environment"]["CUDA_VISIBLE_DEVICES"] = (
        None
    )
    diagnostic._validate_context_continuity(preexec, closed_environment_runtime)
    mutation_paths = [
        ("process", "pid"),
        ("process", "python_argv"),
        ("slurm", "step_id"),
        ("slurm", "step_gpu_index"),
        ("slurm", "gpus_per_task"),
        ("slurm", "tres_per_task"),
        ("slurm", "scheduler_handoff"),
        ("nvidia_smi", "uuid"),
        ("cgroup", "raw_sha256"),
        ("device_nodes", "physical"),
        ("source", "eval_script", "sha256"),
    ]
    for path in mutation_paths:
        changed = json.loads(json.dumps(runtime))
        cursor = changed
        for component in path[:-1]:
            cursor = cursor[component]
        cursor[path[-1]] = "mismatch"
        with pytest.raises(ValueError, match="continuity mismatch"):
            diagnostic._validate_context_continuity(preexec, changed)


def test_preexec_publishes_bound_evidence_and_builds_exact_exec_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, task_dir, output_directories = create_output_tree(tmp_path)
    preexec_path = task_dir / "startup_preexec.json"
    preclose_path = task_dir / "startup_preclose.json"
    scheduler_request_path = task_dir / "scheduler_request.json"
    scheduler_handoff_path = task_dir / "scheduler_handoff.json"
    target = canonical_target(task_dir)
    environment = runtime_environment()
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    runtime = synthetic_runtime(pid=os.getpid(), output_directories=output_directories)
    runtime["process"]["python_argv"] = target[1:]
    captures = []

    def capture_runtime(**kwargs):
        captures.append(kwargs)
        return runtime

    monkeypatch.setattr(diagnostic, "capture_runtime_context", capture_runtime)

    def wait_handoff(
        path,
        *,
        request,
        request_identity,
        timeout_seconds=60.0,
        expected_parent_identity=None,
        **validation,
    ):
        del timeout_seconds
        assert path == scheduler_handoff_path
        assert expected_parent_identity == output_directories["task_identity"]
        assert validation == scheduler_approval_kwargs()
        supplied = synthetic_scheduler_handoff()
        supplied["value"]["request"] = request_identity
        supplied["value"]["request_value"] = request
        supplied["artifact"]["path"] = str(path)
        return supplied["value"], supplied["artifact"]

    monkeypatch.setattr(diagnostic, "wait_for_scheduler_handoff", wait_handoff)

    actual_target, closed_environment, identity = diagnostic.prepare_public_eval_exec(
        target_argv=target,
        preexec_path=preexec_path,
        preclose_path=preclose_path,
        expected_gpu_uuid=GPU_UUID,
        source_root=ROOT,
        data_root=Path("/physical/PolaRiS-Hub"),
        cache_root=Path("/cache"),
        scheduler_request_path=scheduler_request_path,
        scheduler_handoff_path=scheduler_handoff_path,
        run_dir=run_dir,
        task_dir=task_dir,
        namespace_parent_identity=output_directories["namespace_parent_identity"],
        run_dir_identity=output_directories["run_identity"],
        task_dir_identity=output_directories["task_identity"],
    )

    assert actual_target == target
    assert captures[0]["python_argv"] == target[1:]
    assert (
        closed_environment["POLARIS_STARTUP_DIAGNOSTIC_PREEXEC_SHA256"]
        == identity["sha256"]
    )
    preexec, _ = diagnostic.stable_read_immutable_json(preexec_path)
    assert preexec["runtime"]["process"]["pid"] == os.getpid()
    assert preexec["runtime"]["process"]["python_argv"] == target[1:]
    assert preexec["target_argv"] == target
    assert preexec["zero_work_counters"] == diagnostic.ZERO_WORK_COUNTERS


def test_real_execve_python_argv_matches_normalized_target(tmp_path: Path) -> None:
    capture = tmp_path / "capture_argv.py"
    capture.write_text(
        "import json, sys\nprint(json.dumps(sys.argv, separators=(',', ':')))\n",
        encoding="utf-8",
    )
    target = [sys.executable, str(capture), "scripts/eval.py", "--headless"]
    launcher = (
        "import os,sys; "
        "os.execve(sys.argv[1], sys.argv[1:], "
        "{'PATH': os.environ.get('PATH', '')})"
    )
    completed = subprocess.run(
        [sys.executable, "-c", launcher, *target],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == target[1:]


def test_cli_uses_execve_without_a_fork(monkeypatch: pytest.MonkeyPatch) -> None:
    target = ["/.venv/bin/python", "scripts/eval.py"]
    environment = {"CLOSED": "1"}
    monkeypatch.setattr(
        diagnostic,
        "_parse_cli",
        lambda _: type(
            "Args",
            (),
            {
                "target": target,
                "preexec_output": Path("/preexec.json"),
                "preclose_output": Path("/preclose.json"),
                "expected_batch_gpu_uuid": GPU_UUID,
                "source_root": Path("/polaris-source"),
                "data_root": Path("/data"),
                "cache_root": Path("/cache"),
                "scheduler_request": Path("/scheduler_request.json"),
                "scheduler_handoff": Path("/scheduler_handoff.json"),
                "run_dir": Path("/run"),
                "task_dir": Path("/run/app_launcher_only"),
                "namespace_parent_identity": "1:1:3:4:0755",
                "run_dir_identity": "1:2:3:4:0755",
                "task_dir_identity": "1:3:3:4:0755",
            },
        )(),
    )
    monkeypatch.setattr(
        diagnostic,
        "prepare_public_eval_exec",
        lambda **_: (target, environment, {"sha256": "4" * 64}),
    )
    observed = []

    class ExecveCalled(RuntimeError):
        pass

    def fake_execve(path, argv, env):
        observed.append((path, argv, env, os.getpid()))
        raise ExecveCalled

    monkeypatch.setattr(os, "execve", fake_execve)
    with pytest.raises(ExecveCalled):
        diagnostic.main([])
    assert observed == [("/.venv/bin/python", target, environment, os.getpid())]


class FakeSimulationApp:
    def __init__(self, error: BaseException | None = None):
        self.error = error
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.error is not None:
            raise self.error


@pytest.mark.parametrize("close_before_signal", [False, True])
def test_caught_signal_closes_simulation_app_once_and_restores_handlers(
    monkeypatch: pytest.MonkeyPatch,
    close_before_signal: bool,
) -> None:
    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
    }
    app = FakeSimulationApp()

    def raise_signal(*, simulation_app, **_kwargs):
        if close_before_signal:
            simulation_app.close()
        raise diagnostic._CaughtSignal(signal.SIGTERM)

    monkeypatch.setattr(diagnostic, "_run_app_launcher_only_diagnostic", raise_signal)
    with pytest.raises(diagnostic._CaughtSignal) as caught:
        diagnostic.run_app_launcher_only_diagnostic(
            simulation_app=app,
            preexec_path=Path("/unused/preexec.json"),
            preclose_path=Path("/unused/preclose.json"),
            expected_gpu_uuid=GPU_UUID,
        )
    assert caught.value.signum == signal.SIGTERM
    assert app.close_calls == int(close_before_signal)
    assert {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
    } == previous_handlers


def test_caught_signal_does_not_call_hard_exit_capable_simulation_close() -> None:
    program = """
import os
from pathlib import Path
import signal
from polaris import app_launcher_startup_diagnostic as diagnostic

class HardExitApp:
    def close(self):
        os._exit(0)

def caught_signal(**_kwargs):
    raise diagnostic._CaughtSignal(signal.SIGTERM)

diagnostic._run_app_launcher_only_diagnostic = caught_signal
try:
    diagnostic.run_app_launcher_only_diagnostic(
        simulation_app=HardExitApp(),
        preexec_path=Path('/unused/preexec.json'),
        preclose_path=Path('/unused/preclose.json'),
        expected_gpu_uuid='GPU-01234567-89ab-cdef-0123-456789abcdef',
    )
except diagnostic._CaughtSignal:
    raise SystemExit(143)
raise SystemExit(99)
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        timeout=5,
    )
    assert completed.returncode == 143


def test_main_maps_caught_signal_to_shell_exit_and_restores_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
    }

    def raise_signal(_argv):
        raise diagnostic._CaughtSignal(signal.SIGINT)

    monkeypatch.setattr(diagnostic, "_main", raise_signal)
    assert diagnostic.main([]) == 128 + signal.SIGINT
    assert {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
    } == previous_handlers


def test_app_launcher_only_publishes_preclose_ready_then_closes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, task_dir, output_directories = create_output_tree(tmp_path)
    runtime = synthetic_runtime(pid=os.getpid(), output_directories=output_directories)
    preexec_path = task_dir / "preexec.json"
    preexec_identity = publish_preexec(preexec_path, runtime)
    preclose_path = task_dir / "preclose.json"
    monkeypatch.setenv(
        "POLARIS_STARTUP_DIAGNOSTIC_PREEXEC_SHA256",
        preexec_identity["sha256"],
    )
    monkeypatch.setenv(
        "POLARIS_STARTUP_DIAGNOSTIC_TASK_DIR_IDENTITY",
        output_directories["task_identity"],
    )
    monkeypatch.setattr(diagnostic, "capture_runtime_context", lambda **_: runtime)
    monkeypatch.setattr(diagnostic, "_forbidden_loaded_modules", lambda: [])
    app = FakeSimulationApp()
    artifacts = diagnostic.run_app_launcher_only_diagnostic(
        simulation_app=app,
        preexec_path=preexec_path,
        preclose_path=preclose_path,
        expected_gpu_uuid=GPU_UUID,
    )
    assert app.close_calls == 1
    preclose, _ = diagnostic.stable_read_immutable_json(preclose_path)
    ready, _ = diagnostic.stable_read_immutable_json(
        diagnostic.ready_path_for(preclose_path)
    )
    assert preclose["status"] == "simulation_app_close_pending"
    assert ready["status"] == "ready_for_simulation_app_close"
    assert ready["preclose"] == artifacts["preclose"]
    assert set(preclose["zero_work_counters"].values()) == {0}
    assert preclose["bounded_diagnostic_counts"]["nvidia_smi_invocations"] == 2
    assert preclose["runtime"]["slurm"]["gpus_per_task"] == 1
    assert preclose["runtime"]["slurm"]["tres_per_task"] == ("cpu=16,gres/gpu:1")
    assert (
        preclose["runtime"]["slurm"]["scheduler_handoff"]
        == runtime["slurm"]["scheduler_handoff"]
    )


@pytest.mark.parametrize("close_error", [RuntimeError("close failed"), SystemExit(0)])
def test_close_failure_is_forced_nonzero_without_false_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    close_error: BaseException,
) -> None:
    _, task_dir, output_directories = create_output_tree(tmp_path)
    runtime = synthetic_runtime(pid=os.getpid(), output_directories=output_directories)
    preexec_path = task_dir / "preexec.json"
    preexec_identity = publish_preexec(preexec_path, runtime)
    preclose_path = task_dir / "preclose.json"
    monkeypatch.setenv(
        "POLARIS_STARTUP_DIAGNOSTIC_PREEXEC_SHA256",
        preexec_identity["sha256"],
    )
    monkeypatch.setenv(
        "POLARIS_STARTUP_DIAGNOSTIC_TASK_DIR_IDENTITY",
        output_directories["task_identity"],
    )
    monkeypatch.setattr(diagnostic, "capture_runtime_context", lambda **_: runtime)
    monkeypatch.setattr(diagnostic, "_forbidden_loaded_modules", lambda: [])
    app = FakeSimulationApp(close_error)
    with pytest.raises(
        RuntimeError, match="AppLauncher diagnostic SimulationApp.close"
    ) as captured:
        diagnostic.run_app_launcher_only_diagnostic(
            simulation_app=app,
            preexec_path=preexec_path,
            preclose_path=preclose_path,
            expected_gpu_uuid=GPU_UUID,
        )
    assert captured.value.__cause__ is close_error
    assert app.close_calls == 1
    assert preclose_path.is_file()
    assert diagnostic.ready_path_for(preclose_path).is_file()
    error_name = f"{type(close_error).__module__}.{type(close_error).__qualname__}"
    assert (
        f"POLARIS_STARTUP_DIAGNOSTIC_CLOSE_ERROR={error_name}"
        in capsys.readouterr().err
    )


def test_forbidden_module_boundary_fails_before_publication_or_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, task_dir, output_directories = create_output_tree(tmp_path)
    runtime = synthetic_runtime(pid=os.getpid(), output_directories=output_directories)
    preexec_path = task_dir / "preexec.json"
    preexec_identity = publish_preexec(preexec_path, runtime)
    preclose_path = task_dir / "preclose.json"
    monkeypatch.setenv(
        "POLARIS_STARTUP_DIAGNOSTIC_PREEXEC_SHA256",
        preexec_identity["sha256"],
    )
    monkeypatch.setenv(
        "POLARIS_STARTUP_DIAGNOSTIC_TASK_DIR_IDENTITY",
        output_directories["task_identity"],
    )
    monkeypatch.setattr(diagnostic, "capture_runtime_context", lambda **_: runtime)
    monkeypatch.setattr(
        diagnostic, "_forbidden_loaded_modules", lambda: ["polaris.policy"]
    )
    app = FakeSimulationApp()
    with pytest.raises(RuntimeError, match="forbidden modules"):
        diagnostic.run_app_launcher_only_diagnostic(
            simulation_app=app,
            preexec_path=preexec_path,
            preclose_path=preclose_path,
            expected_gpu_uuid=GPU_UUID,
        )
    assert app.close_calls == 0
    assert not preclose_path.exists()


@pytest.mark.parametrize("prefix", diagnostic.FORBIDDEN_MODULE_PREFIXES)
def test_every_forbidden_module_prefix_is_detected(
    prefix: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_name = f"{prefix}.adversarial_leaf"
    monkeypatch.setitem(sys.modules, module_name, object())
    assert module_name in diagnostic._forbidden_loaded_modules()
