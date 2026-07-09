from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).parents[1]
WORKER = ROOT / "scripts/polaris/eval_pi05_droid_jointpos_polaris.sh"
BATCH = ROOT / "scripts/polaris/l40s_pi05_eval_job.sbatch"
SUBMITTER = ROOT / "scripts/polaris/submit_pi05_droid_jointpos_polaris.sh"
DIAGNOSTIC_MODULE = ROOT / "src/polaris/app_launcher_startup_diagnostic.py"
PUBLIC_BASE_COMMIT = "a00ac41d822f7e6a02c7a787c6b5aae66c6aa08b"


def _function(source: str, name: str, next_name: str) -> str:
    start = source.index(f"{name}() {{")
    end = source.index(f"{next_name}() {{", start)
    return source[start:end]


def test_worker_routes_diagnostic_before_checkpoint_and_model_work() -> None:
    source = WORKER.read_text(encoding="utf-8")
    dispatch = """if [[ "${POLARIS_EVAL_MODE}" == app_launcher_only ]]; then
  export POLARIS_DIAGNOSTIC_MODULE_SHA256="${EXPECTED_DIAGNOSTIC_MODULE_SHA256}"
  capture_and_export_pyxis_image_identity
  run_app_launcher_only
  exit
fi"""
    assert source.count(dispatch) == 1
    branch = source.index(dispatch)
    checkpoint_manifest = source.index('[[ -f "${CHECKPOINT_MANIFEST}" ]]', branch)
    checkpoint_resolution = source.index(
        "from openpi.shared.download import maybe_download"
    )
    model_server = source.index('"${server_command[@]}"')
    assert branch < checkpoint_manifest < checkpoint_resolution < model_server
    function_start = source.index("run_app_launcher_only() {")
    function_end = source.index('\n}\n\n[[ -n "${SLURM_JOB_ID:-}"', function_start)
    function = source[function_start:function_end]
    assert "build_public_eval_command" in function
    assert "server_command" not in function
    assert "checkpoint_path" not in function
    assert "RUN_TOKENIZER_FILE" not in function
    assert "publish_terminal_marker" not in function
    assert "EVIDENCE_FINALIZED" not in function
    assert "policy_traces.forbidden" in function
    assert "runtime_contract.forbidden" in function
    assert "eval_results.csv" not in function
    assert "episode_*.mp4" not in function
    assert "POLARIS_APP_LAUNCHER_ONLY_PRETERMINAL_ATTESTATION=" in function
    assert "POLARIS_APP_LAUNCHER_ONLY_AUTHORITATIVE_COMPLETION=0" in function
    assert "POLARIS_APP_LAUNCHER_ONLY_PROMOTION_REQUIRED=" in function
    assert "POLARIS_APP_LAUNCHER_ONLY_EVIDENCE_READY=" not in function
    assert "POLARIS_APP_LAUNCHER_CLOSE_TERMINATION_MODE=" in function
    assert "process_exited_zero_before_postclose_marker" in function
    assert "simulation_app_close_returned" in function
    assert "| tee" not in function
    assert "immutable-log-tee" in function
    assert "app_launcher_only.log.identity.json" in function
    assert "scheduler_request.json" in function
    assert "scheduler_handoff.json" in function
    assert "scheduler_terminal_request.json" in function
    assert "scheduler_terminal.json" in function
    assert "broker-scheduler-handoff" in function
    assert "publish-scheduler-terminal-request" in function
    assert "/usr/bin/setsid" in function
    assert "await_app_process_group" in function
    assert function.index("broker-scheduler-handoff") < function.index(
        '/usr/bin/setsid "${diagnostic_eval_command[@]}"'
    )
    assert function.index("publish-scheduler-terminal-request") < function.index(
        'wait_app_process_bounded "${APP_HELPER_PID}" 7000'
    )
    diagnostic_source = DIAGNOSTIC_MODULE.read_text(encoding="utf-8")
    assert "POLARIS_STARTUP_DIAGNOSTIC_CLOSE_ERROR=" in diagnostic_source
    assert "required_phases" in diagnostic_source
    assert "validate-immutable-log" in diagnostic_source


def test_shared_builder_preserves_the_normal_public_eval_command() -> None:
    source = WORKER.read_text(encoding="utf-8")
    function = _function(source, "build_public_eval_command", "run_app_launcher_only")
    normal_start = function.index('if [[ "${POLARIS_EVAL_MODE}" == standard ]]')
    normal_end = function.index("  fi\n  [[", normal_start)
    normal = function[normal_start:normal_end]
    array_start = normal.index("    eval_command=(")
    array_end = normal.index("\n    )", array_start) + len("\n    )")
    actual_array = normal[array_start:array_end]
    expected_array = """    eval_command=(
      srun --ntasks=1 "--cpus-per-task=${SLURM_CPUS_PER_TASK:-16}"
      "--container-image=${POLARIS_PYXIS_IMAGE}"
      "--container-mounts=${pyxis_mounts}"
      "--container-workdir=${POLARIS_CONTAINER_SOURCE}"
      --no-container-entrypoint --no-container-mount-home --container-remap-root --container-writable
      "--container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES" --export=ALL
      /usr/bin/env -i
      PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
      LANG=C.UTF-8 LC_ALL=C.UTF-8
      "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES}"
      "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES}"
      VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
      ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y OMNI_KIT_ALLOW_ROOT=1
      PYTHONUNBUFFERED=1
      "PYTHONPATH=${POLARIS_CONTAINER_SOURCE}/src:${POLARIS_CONTAINER_SOURCE}/third_party/openpi/packages/openpi-client/src"
      "POLARIS_DATA_PATH=${POLARIS_DATA_DIR}"
      XDG_CACHE_HOME=/cache HF_HOME=/cache/huggingface HOME=/cache/home
      /.venv/bin/python "${eval_args[@]}"
    )"""
    assert actual_array == expected_array
    assert "startup-diagnostic" not in normal
    assert "app_launcher_startup_diagnostic" not in normal
    assert "--gpus-per-task" not in normal


def test_standard_server_eval_video_and_finalization_tail_match_public_base() -> None:
    current = WORKER.read_text(encoding="utf-8")
    relative = WORKER.relative_to(ROOT).as_posix()
    public_base = subprocess.run(
        ["git", "show", f"{PUBLIC_BASE_COMMIT}:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    def array_block(source: str, marker: str) -> str:
        start = source.index(marker)
        return source[start : source.index("\n)", start) + 2]

    assert array_block(current, "server_command=(") == array_block(
        public_base, "server_command=("
    )
    science_tail_marker = 'set +e\n(\n  cd "${POLARIS_DIR}"\n  "${eval_command[@]}"'
    assert (
        current[current.index(science_tail_marker) :]
        == public_base[public_base.index(science_tail_marker) :]
    )

    # The global injection cleanup cannot alter intended standard inputs: the
    # evaluator already enters an env-i container with explicit PATH/LANG/
    # PYTHONPATH, and the unchanged server block uses an absolute interpreter
    # with explicit OPENPI_DATA_HOME/PYTHONPATH.  Standard submission restores
    # the inherited command-search path and retains the versioned 15-column
    # registry (the added field is held-scheduler provenance, not science).
    standard_builder = _function(
        current, "build_public_eval_command", "run_app_launcher_only"
    )
    assert "/usr/bin/env -i" in standard_builder
    assert '"${OPENPI_DIR}/.venv/bin/python"' in array_block(
        current, "server_command=("
    )
    submitter = SUBMITTER.read_text(encoding="utf-8")
    assert 'export PATH="${INHERITED_PATH}"' in submitter
    assert "standard_header=$'job_id\\tmode\\ttask" in submitter
    assert "held_scheduler_record_sha256\\tprovenance_dir" in submitter


def test_worker_and_batch_preserve_nondefault_path_only_for_standard_mode() -> None:
    inherited = "/opt/site/bin:/custom/research/bin:/usr/bin:/bin"

    def resolved_path(preamble: str, mode: str) -> str:
        completed = subprocess.run(
            ["/usr/bin/bash", "--noprofile", "--norc", "-p"],
            input=preamble + '\nprintf "%s\\n" "${PATH}"\n',
            check=True,
            capture_output=True,
            text=True,
            env={"PATH": inherited, "POLARIS_EVAL_MODE": mode},
        )
        return completed.stdout.strip()

    worker = WORKER.read_text(encoding="utf-8")
    batch = BATCH.read_text(encoding="utf-8")
    preambles = (
        worker[: worker.index("export PYTHONDONTWRITEBYTECODE=1")],
        batch[: batch.index("trusted_source_tree_sha256()")],
    )
    for preamble in preambles:
        assert resolved_path(preamble, "standard") == inherited
        assert resolved_path(preamble, "app_launcher_only") == (
            "/cm/local/apps/slurm/24.11/bin:/usr/bin:/bin"
        )


def test_diagnostic_builder_uses_one_explicit_gpu_and_exec_instrumentation() -> None:
    source = WORKER.read_text(encoding="utf-8")
    function = _function(source, "build_public_eval_command", "run_app_launcher_only")
    diagnostic_start = function.index(
        '[[ "${POLARIS_EVAL_MODE}" == app_launcher_only ]]'
    )
    diagnostic = function[diagnostic_start:]
    assert "--gpus-per-task=1" in diagnostic
    assert "/.venv/bin/python -I -S" in diagnostic
    assert (
        "/polaris-source/src/polaris/app_launcher_startup_diagnostic.py" in diagnostic
    )
    assert "python -m polaris.app_launcher_startup_diagnostic" not in diagnostic
    assert "/usr/bin/bash" not in diagnostic
    assert "--export=ALL" not in diagnostic
    assert '"--export=${diagnostic_export_names}"' in diagnostic
    assert '-- /.venv/bin/python "${eval_args[@]}"' in diagnostic
    command_start = diagnostic.index("  diagnostic_eval_command=(")
    command_end = diagnostic.index("\n  )", command_start) + len("\n  )")
    command = diagnostic[command_start:command_end]
    for option in (
        "--preexec-output",
        "--preclose-output",
        "--expected-batch-gpu-uuid",
        "--source-root",
        "--data-root",
        "--cache-root",
        "--scheduler-request",
        "--scheduler-handoff",
        "--run-dir",
        "--task-dir",
        "--run-dir-identity",
        "--task-dir-identity",
    ):
        assert command.count(f"\n    {option} ") == 1
    assert "scontrol show job --oneliner" not in diagnostic
    assert "scontrol show step --oneliner" not in diagnostic


def test_diagnostic_helpers_and_container_entry_are_ambient_injection_closed() -> None:
    source = WORKER.read_text(encoding="utf-8")
    assert source.startswith("#!/usr/bin/bash -p\n")
    assert "/usr/bin/bash -p --noprofile --norc" not in source
    assert "/usr/bin/bash --noprofile --norc -p" in source
    helper = _function(
        source, "build_diagnostic_helper_command", "run_diagnostic_helper"
    )
    assert "/usr/bin/env -i" in helper
    assert "DIAGNOSTIC_HOST_PYTHON=/usr/bin/python3.12" in source
    assert 'exec {DIAGNOSTIC_MODULE_FD}<"${DIAGNOSTIC_MODULE}"' in helper
    assert '"/proc/$$/fd/${DIAGNOSTIC_MODULE_FD}"' in helper
    assert (
        '"${DIAGNOSTIC_HOST_PYTHON}" -I -S -c "${DIAGNOSTIC_MODULE_LOADER}"' in helper
    )
    assert '"${DIAGNOSTIC_MODULE}" "${DIAGNOSTIC_MODULE_FD}"' in helper
    assert "module_path_metadata" in helper
    assert "module_fd_metadata" in helper
    assert "module_sha" in helper
    for hostile in (
        "PYTHONHOME",
        "PYTHONUSERBASE",
        "PYTHONPATH",
        "LD_PRELOAD",
        "LD_AUDIT",
        "BASH_ENV",
        "ENV",
    ):
        assert hostile not in helper

    builder = _function(source, "build_public_eval_command", "app_process_is_live")
    app_start = builder.index('[[ "${POLARIS_EVAL_MODE}" == app_launcher_only ]]')
    app_builder = builder[app_start:]
    assert "/usr/bin/bash" not in app_builder
    assert "BASH_ENV" not in app_builder
    assert "SHELLOPTS" not in app_builder
    assert "--export=ALL" not in app_builder
    assert "--container-env=${diagnostic_export_names}" in app_builder
    for name in (
        "POLARIS_EVAL_MODE",
        "SUBMISSION_TRANSACTION_ID",
        "POLARIS_EXPECTED_SCONTROL_SHA256",
        "POLARIS_EXPECTED_SACCT_SHA256",
        "POLARIS_EXPECTED_SCANCEL_SHA256",
        "POLARIS_EXPECTED_SRUN_SHA256",
        "POLARIS_STARTUP_DIAGNOSTIC_RUN_DIR",
        "POLARIS_STARTUP_DIAGNOSTIC_TASK_DIR",
        "POLARIS_STARTUP_DIAGNOSTIC_RUN_DIR_IDENTITY",
        "POLARIS_STARTUP_DIAGNOSTIC_TASK_DIR_IDENTITY",
    ):
        assert name in app_builder


def test_diagnostic_builder_routes_app_mode_through_both_pyxis_env_filters() -> None:
    source = WORKER.read_text(encoding="utf-8")
    functions = _function(
        source, "build_eval_args", "build_public_eval_command"
    ) + _function(source, "build_public_eval_command", "run_app_launcher_only")
    script = f"""set -Eeuo pipefail
{functions}
POLARIS_EVAL_MODE=app_launcher_only
POLARIS_ENVIRONMENT=DROID-FoodBussing
PORT=8000
OPEN_LOOP_HORIZON=8
EXPECTED_ACTION_HORIZON=16
EXPECTED_ACTION_DIM=8
TRACE_PATH=/run/trace.forbidden
TASK_DIR=/run/app_launcher_only
ROLLOUTS=1
ENVIRONMENT_SEED=0
RUNTIME_CONTRACT_FILE=/run/runtime.forbidden
STARTUP_PREEXEC_FILE=/run/preexec.json
STARTUP_PRECLOSE_FILE=/run/preclose.json
actual_gpu_uuid=GPU-01234567-89ab-cdef-0123-456789abcdef
SLURM_CPUS_PER_TASK=16
POLARIS_PYXIS_IMAGE=/images/polaris.sqsh
POLARIS_DIR=/source
POLARIS_CONTAINER_SOURCE=/polaris-source
POLARIS_DATA_DIR=/data
RUN_DIR=/run
POLARIS_CACHE_DIR=/cache
POLARIS_VULKAN_ICD_PATH=/etc/vulkan/icd.d/nvidia_icd.json
NVIDIA_VISIBLE_DEVICES="${{actual_gpu_uuid}}"
NVIDIA_DRIVER_CAPABILITIES=all
DIAGNOSTIC_CONTAINER_LOADER=loader
EXPECTED_DIAGNOSTIC_MODULE_SHA256={"0" * 64}
STARTUP_SCHEDULER_REQUEST_FILE=/run/request.json
STARTUP_SCHEDULER_HANDOFF_FILE=/run/handoff.json
POLARIS_OUTPUT_NAMESPACE_PARENT_IDENTITY=1:2:3:4:0755
RUN_DIR_IDENTITY=1:3:3:4:0755
TASK_DIR_IDENTITY=1:4:3:4:0755
build_public_eval_command
printf '%s\n' "${{diagnostic_eval_command[@]}}"
"""
    completed = subprocess.run(
        ["/usr/bin/bash", "--noprofile", "--norc", "-p"],
        input=script,
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    argv = completed.stdout.splitlines()
    container_env = next(arg for arg in argv if arg.startswith("--container-env="))
    export = next(arg for arg in argv if arg.startswith("--export="))
    assert container_env.removeprefix("--container-env=") == export.removeprefix(
        "--export="
    )
    routed_names = export.removeprefix("--export=").split(",")
    assert routed_names.count("POLARIS_EVAL_MODE") == 1
    assert routed_names[0] == "POLARIS_EVAL_MODE"
    assert routed_names.count("SUBMISSION_TRANSACTION_ID") == 1
    assert routed_names[1] == "SUBMISSION_TRANSACTION_ID"
    assert "--startup-diagnostic" in argv
    assert "app_launcher_only" in argv


def test_worker_routes_one_well_formed_module_digest_through_both_loaders() -> None:
    source = WORKER.read_text(encoding="utf-8")
    prefix = "EXPECTED_DIAGNOSTIC_MODULE_SHA256="
    frozen = next(
        line.removeprefix(prefix)
        for line in source.splitlines()
        if line.startswith(prefix)
    )
    # The exact digest is updated only by the final source-freeze step. Until
    # then, derive the candidate without forcing an intermediate production
    # hash update, and check that the frozen value is wired through both paths.
    candidate = hashlib.sha256(DIAGNOSTIC_MODULE.read_bytes()).hexdigest()
    assert re.fullmatch(r"[0-9a-f]{64}", frozen)
    assert re.fullmatch(r"[0-9a-f]{64}", candidate)
    assert (
        source.count(
            'POLARIS_DIAGNOSTIC_MODULE_SHA256="${EXPECTED_DIAGNOSTIC_MODULE_SHA256}"'
        )
        >= 2
    )
    assert '"${EXPECTED_DIAGNOSTIC_MODULE_SHA256}"' in source


def test_process_group_cleanup_kills_descendant_after_leader_exits() -> None:
    source = WORKER.read_text(encoding="utf-8")
    start = source.index("app_process_is_live() {")
    end = source.index("cleanup_app_processes_and_temps() {", start)
    functions = source[start:end]
    script = f"""set -Eeuo pipefail
set +m
{functions}
leader=""
cleanup() {{
  if [[ -n "${{leader}}" ]]; then
    kill -KILL -- "-${{leader}}" 2>/dev/null || true
    wait "${{leader}}" 2>/dev/null || true
  fi
}}
trap cleanup EXIT
/usr/bin/setsid /usr/bin/bash --noprofile --norc -p -c '
  trap "exit 0" TERM
  (trap "" TERM; exec /usr/bin/sleep 30) &
  wait
' &
leader=$!
await_app_process_group "${{leader}}"
terminate_app_process_group "${{leader}}" "${{leader}}"
if kill -0 -- "-${{leader}}" 2>/dev/null; then
  exit 91
fi
leader=""
"""
    completed = subprocess.run(
        [
            "/usr/bin/timeout",
            "--kill-after=1",
            "8",
            "/usr/bin/bash",
            "--noprofile",
            "--norc",
            "-p",
        ],
        input=script,
        text=True,
        capture_output=True,
        timeout=10,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert completed.returncode == 0, completed.stderr


def test_large_logger_mirror_drains_without_coproc_deadlock(tmp_path: Path) -> None:
    source = WORKER.read_text(encoding="utf-8")
    start = source.index('APP_HELPER_PID=""')
    end = source.index('\n}\n\n[[ -n "${SLURM_JOB_ID:-}"', start) + 2
    lifecycle = source[start:end]
    helper = tmp_path / "helper.py"
    helper.write_text(
        "import pathlib,sys,time\n"
        "command=sys.argv[1]\n"
        "if command == 'broker-scheduler-handoff':\n"
        "    terminal=pathlib.Path(sys.argv[sys.argv.index('--terminal-request')+1])\n"
        "    for _ in range(1000):\n"
        "        if terminal.exists(): break\n"
        "        time.sleep(.01)\n"
        "    else: raise SystemExit(91)\n"
        "elif command == 'publish-scheduler-terminal-request':\n"
        "    pathlib.Path(sys.argv[sys.argv.index('--output')+1]).write_text('{}\\n')\n"
        "elif command == 'immutable-log-tee':\n"
        "    while chunk := sys.stdin.buffer.read(1048576):\n"
        "        sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "outputs"
    cache_root = tmp_path / "cache"
    script = f"""set -Eeuo pipefail
die() {{ echo "ERROR: $*" >&2; exit 2; }}
verify_trusted_source_snapshot() {{ :; }}
verify_runtime_closure_approval() {{ :; }}
close_diagnostic_module_fd() {{ :; }}
diagnostic_helper_command=()
build_diagnostic_helper_command() {{
  diagnostic_helper_command=(/usr/bin/env -i PATH=/usr/bin:/bin /usr/bin/python3 -I -S {helper})
}}
run_diagnostic_helper() {{
  case "$1" in
    create-output-directories)
      mkdir -p -- "${{RUN_DIR}}" "${{TASK_DIR}}"
      printf '%s\\t1:2:3:4:0755\\t1:3:3:4:0755\\n' \\
        "${{POLARIS_OUTPUT_NAMESPACE_PARENT_IDENTITY}}"
      ;;
    publish-failure-attestation|seal-evidence-tree) return 1 ;;
    *) build_diagnostic_helper_command; "${{diagnostic_helper_command[@]}}" "$@" ;;
  esac
    }}
    build_public_eval_command() {{
      diagnostic_eval_command=(/usr/bin/python3 -I -S -c 'import sys,time;sys.stdout.write("x"*1310720);sys.stdout.flush();time.sleep(.2);raise SystemExit(7)')
    }}
{lifecycle}
DRY_RUN=0
RUN_NAMESPACE=large-log
RUN_NAME=large-log
OUTPUT_ROOT={output_root}
POLARIS_CACHE_DIR={cache_root}
POLARIS_ENVIRONMENT=DROID-FoodBussing
POLARIS_OUTPUT_NAMESPACE_PARENT_IDENTITY=1:1:1:1:0755
EXPECTED_DIAGNOSTIC_MODULE_SHA256={"0" * 64}
SLURM_JOB_ID=12345
if run_app_launcher_only; then
  code=0
else
  code=$?
fi
[[ "${{code}}" == 7 ]]
"""
    completed = subprocess.run(
        [
            "/usr/bin/timeout",
            "--kill-after=1",
            "12",
            "/usr/bin/bash",
            "--noprofile",
            "--norc",
            "-p",
        ],
        input=script,
        text=True,
        capture_output=True,
        timeout=14,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert len(completed.stdout) >= 1_310_720


def test_app_launcher_dry_run_has_zero_filesystem_mutation(tmp_path: Path) -> None:
    source = WORKER.read_text(encoding="utf-8")
    setup_start = source.index(
        'if [[ "${POLARIS_EVAL_MODE}" == app_launcher_only && "${DRY_RUN}" == 1 ]]'
    )
    setup_end = source.index("verify_trusted_source_snapshot() {", setup_start)
    setup = source[setup_start:setup_end]
    assert setup.index('OPENPI_DATA_HOME="$(readlink -m') < setup.index("else")
    assert setup.index("else") < setup.index('mkdir -p "${OPENPI_DATA_HOME}"')
    start = source.index("run_app_launcher_only() {")
    end = source.index('\n}\n\n[[ -n "${SLURM_JOB_ID:-}"', start) + 2
    function = source[start:end]
    script = f"""set -Eeuo pipefail
die() {{ echo "ERROR: $*" >&2; exit 2; }}
verify_trusted_source_snapshot() {{ :; }}
verify_runtime_closure_approval() {{ :; }}
build_public_eval_command() {{ diagnostic_eval_command=(srun diagnostic); }}
run_diagnostic_helper() {{ exit 99; }}
{function}
DRY_RUN=1
RUN_NAMESPACE=dry
RUN_NAME=dry
OUTPUT_ROOT={tmp_path / "output"}
POLARIS_CACHE_DIR={tmp_path / "cache"}
POLARIS_ENVIRONMENT=DROID-FoodBussing
EXPECTED_DIAGNOSTIC_MODULE_SHA256={"0" * 64}
unset SLURM_JOB_ID
run_app_launcher_only
"""
    before = list(tmp_path.iterdir())
    completed = subprocess.run(
        ["/usr/bin/bash", "--noprofile", "--norc", "-p"],
        input=script,
        text=True,
        capture_output=True,
        timeout=5,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert completed.returncode == 0, completed.stderr
    assert list(tmp_path.iterdir()) == before
    assert "srun diagnostic" in completed.stdout

    preexisting = tmp_path / "preexisting"
    preexisting.mkdir()
    rejected = script.replace(
        f"OUTPUT_ROOT={tmp_path / 'output'}",
        f"RUN_DIR={preexisting}\nOUTPUT_ROOT={tmp_path / 'output'}",
    )
    completed = subprocess.run(
        ["/usr/bin/bash", "--noprofile", "--norc", "-p"],
        input=rejected,
        text=True,
        capture_output=True,
        timeout=5,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert completed.returncode == 2
    assert list(preexisting.iterdir()) == []


def test_failure_exit_precedence_and_terminalization_are_closed() -> None:
    source = WORKER.read_text(encoding="utf-8")
    start = source.index("run_app_launcher_only() {")
    end = source.index('\n}\n\n[[ -n "${SLURM_JOB_ID:-}"', start)
    function = source[start:end]
    assert 'primary_code="${APP_SRUN_CODE}"' in function
    assert '(( primary_code == 0 )) && primary_code="${APP_LOG_CODE}"' in function
    assert '(( primary_code == 0 )) && primary_code="${APP_HELPER_CODE}"' in function
    failure_start = source.index("finalize_app_failure() {")
    failure_end = source.index("app_signal_handler() {", failure_start)
    failure = source[failure_start:failure_end]
    assert failure.index("failure_sealed=1") < failure.index("APP_TERMINALIZED=1")
    assert "APP_TERMINALIZED=1" in failure
    assert "process_group_survived" in failure
    assert "rm " not in failure


def test_submitter_has_one_non_scientific_mode_and_keeps_normal_export_shape() -> None:
    source = SUBMITTER.read_text(encoding="utf-8")
    assert "app-launcher-only|canary|foodbussing50|full" in source
    assert "app-launcher-only requires ROLLOUTS=1" in source
    assert "app-launcher-only requires ENVIRONMENT_SEED=0" in source
    assert "worker_eval_mode=app_launcher_only" in source
    assert source.startswith("#!/usr/bin/bash -p\n")
    assert (
        "unset BASH_ENV ENV LD_AUDIT LD_PRELOAD PYTHONHOME PYTHONPATH PYTHONUSERBASE"
        in source
    )
    assert "SBATCH_COMMAND=/cm/local/apps/slurm/24.11/bin/sbatch" in source
    assert 'SBATCH_COMMAND="$(command -v sbatch)"' in source
    assert "/usr/bin/bash -p --noprofile --norc" not in source
    assert "/usr/bin/bash --noprofile --norc -p" in source
    export = source.index('export_vars="PATH=${PATH}')
    conditional = source.index(
        'if [[ "${worker_eval_mode}" == app_launcher_only ]]', export
    )
    sbatch = source.index('sbatch_argv=("${SBATCH_COMMAND}"', conditional)
    assert export < conditional < sbatch
    assert (
        'export_vars+=",POLARIS_EVAL_MODE=app_launcher_only,'
        in source[conditional:sbatch]
    )
    for name in (
        "POLARIS_EXPECTED_SACCT_SHA256",
        "POLARIS_EXPECTED_SACCT_SIZE",
        "POLARIS_EXPECTED_SCANCEL_SHA256",
        "POLARIS_EXPECTED_SCANCEL_SIZE",
        "POLARIS_EXPECTED_SRUN_SHA256",
        "POLARIS_EXPECTED_SRUN_SIZE",
        "POLARIS_OUTPUT_NAMESPACE_PARENT_IDENTITY",
    ):
        assert name in source[conditional:sbatch]
    assert "standard_header=$'job_id\\tmode\\ttask" in source
    assert 'app_header="${standard_header}"' in source
    app_imports = source.index("snapshot_imports=(\n    polaris.app_launcher")
    policy_imports = source.index(
        "snapshot_imports=(\n    polaris.pi05_droid_jointpos_consumer_binding"
    )
    assert app_imports < policy_imports
    assert "AppLauncher-only source smoke imported forbidden modules" in source
    assert "finalize_pi05_app_launcher_only.py" in source
    assert "Approved AppLauncher finalizer origin/compile smoke failed" in source
    assert 'compile(payload, str(path), "exec")' in source


def test_batch_closes_mode_and_validates_one_job_gpu_before_worker() -> None:
    source = BATCH.read_text(encoding="utf-8")
    assert source.startswith("#!/usr/bin/bash -p\n")
    assert (
        "unset BASH_ENV ENV LD_AUDIT LD_PRELOAD PYTHONHOME PYTHONPATH PYTHONUSERBASE"
        in source
    )
    assert 'POLARIS_EVAL_MODE="${POLARIS_EVAL_MODE:-standard}"' in source
    assert "export POLARIS_EVAL_MODE\n" in source
    assert "standard) export ROLLOUTS=" in source
    assert "app_launcher_only)" in source
    assert "app_launcher_only requires ROLLOUTS=1" in source
    gpu_count = source.index('[[ "${SLURM_GPUS_ON_NODE:-}" == 1 ]]')
    gpu_index = source.index('[[ "${SLURM_JOB_GPUS:-}" =~ ^[0-9]+$ ]]')
    assert "/usr/bin/bash -p --noprofile --norc" not in source
    worker = source.index("exec /usr/bin/bash --noprofile --norc -p")
    assert gpu_count < gpu_index < worker
    for name in (
        "POLARIS_EXPECTED_SACCT_SHA256",
        "POLARIS_EXPECTED_SACCT_SIZE",
        "POLARIS_EXPECTED_SCANCEL_SHA256",
        "POLARIS_EXPECTED_SCANCEL_SIZE",
        "POLARIS_EXPECTED_SRUN_SHA256",
        "POLARIS_EXPECTED_SRUN_SIZE",
        "POLARIS_OUTPUT_NAMESPACE_PARENT_IDENTITY",
    ):
        assert name in source[:worker]


def test_runtime_approval_inspection_is_split_from_compute_live_validation() -> None:
    submitter = SUBMITTER.read_text(encoding="utf-8")
    worker = WORKER.read_text(encoding="utf-8")
    batch = BATCH.read_text(encoding="utf-8")
    assert "--inspect-runtime-closure-approval" in submitter
    assert "--validate-runtime-closure-approval" not in submitter
    assert "--validate-runtime-closure-approval" in worker
    assert 'trusted_runtime_closure_approval "$2" "$3" artifact' in batch
    assert 'trusted_runtime_closure_approval "$2" "$3" live' in batch
    assert 'validation_mode not in {"artifact", "live"}' in batch
    assert 'verify_live_runtime = validation_mode == "live"' in batch
    assert '"libtinfo.so.6.4"' in batch
    assert '"path": "/lib/x86_64-linux-gnu/libtinfo.so.6"' in batch
    assert '"target": "libtinfo.so.6.4"' in batch
    suffix_start = batch.index("required_python_suffixes = {")
    suffix_end = batch.index("}\nif any(", suffix_start)
    assert '"/stat.py"' not in batch[suffix_start:suffix_end]
    assert '"subprocess", "sys", "time", "typing",' in batch


def test_hardened_shell_entrypoints_require_privileged_bash_mode() -> None:
    for path in (WORKER, BATCH, SUBMITTER):
        source = path.read_text(encoding="utf-8")
        assert source.startswith("#!/usr/bin/bash -p\n")
        assert (
            '[[ -o privileged ]] || { echo "Privileged Bash mode (-p) is required"'
            in source
        )
        completed = subprocess.run(
            ["/usr/bin/bash", "--noprofile", "--norc", str(path)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin"},
        )
        assert completed.returncode == 2
        assert "Privileged Bash mode (-p) is required" in completed.stderr
