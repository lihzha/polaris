from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]
WORKER = ROOT / "scripts/polaris/eval_pi05_droid_jointpos_polaris.sh"
BATCH = ROOT / "scripts/polaris/l40s_pi05_eval_job.sbatch"
SUBMITTER = ROOT / "scripts/polaris/submit_pi05_droid_jointpos_polaris.sh"


def _function(source: str, name: str, next_name: str) -> str:
    start = source.index(f"{name}() {{")
    end = source.index(f"{next_name}() {{", start)
    return source[start:end]


def test_worker_routes_diagnostic_before_checkpoint_and_model_work() -> None:
    source = WORKER.read_text(encoding="utf-8")
    dispatch = """if [[ "${POLARIS_EVAL_MODE}" == app_launcher_only ]]; then
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
    assert "*.csv" in function and "*.mp4" in function
    assert "POLARIS_APP_LAUNCHER_ONLY_EVIDENCE_READY=" in function
    assert "POLARIS_APP_LAUNCHER_CLOSE_TERMINATION_MODE=" in function
    assert "POLARIS_STARTUP_DIAGNOSTIC_CLOSE_ERROR=" in function
    assert "close_error_count" in function
    assert "required_startup_phases" in function
    assert "process_exited_zero_before_postclose_marker" in function
    assert "simulation_app_close_returned" in function
    assert "before_evaluation_imports" in function


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


def test_diagnostic_builder_uses_one_explicit_gpu_and_exec_instrumentation() -> None:
    source = WORKER.read_text(encoding="utf-8")
    function = _function(source, "build_public_eval_command", "run_app_launcher_only")
    diagnostic_start = function.index(
        '[[ "${POLARIS_EVAL_MODE}" == app_launcher_only ]]'
    )
    diagnostic = function[diagnostic_start:]
    assert "--gpus-per-task=1" in diagnostic
    assert "/.venv/bin/python -m polaris.app_launcher_startup_diagnostic" in diagnostic
    assert '-- /.venv/bin/python "${eval_args[@]}"' in diagnostic
    for option in (
        "--preexec-output",
        "--preclose-output",
        "--expected-batch-gpu-uuid",
        "--source-root",
        "--data-root",
        "--cache-root",
    ):
        assert diagnostic.count(option) == 1


def test_submitter_has_one_non_scientific_mode_and_keeps_normal_export_shape() -> None:
    source = SUBMITTER.read_text(encoding="utf-8")
    assert "app-launcher-only|canary|foodbussing50|full" in source
    assert "app-launcher-only requires ROLLOUTS=1" in source
    assert "worker_eval_mode=app_launcher_only" in source
    export = source.index('export_vars="PATH=${PATH}')
    conditional = source.index(
        'if [[ "${worker_eval_mode}" == app_launcher_only ]]', export
    )
    sbatch = source.index("sbatch_argv=(sbatch", conditional)
    assert export < conditional < sbatch
    assert (
        'export_vars+=",POLARIS_EVAL_MODE=app_launcher_only"'
        in source[conditional:sbatch]
    )
    app_imports = source.index("snapshot_imports=(\n    polaris.app_launcher")
    policy_imports = source.index(
        "snapshot_imports=(\n    polaris.pi05_droid_jointpos_consumer_binding"
    )
    assert app_imports < policy_imports
    assert "AppLauncher-only source smoke imported forbidden modules" in source


def test_batch_closes_mode_and_validates_one_job_gpu_before_worker() -> None:
    source = BATCH.read_text(encoding="utf-8")
    assert 'export POLARIS_EVAL_MODE="${POLARIS_EVAL_MODE:-standard}"' in source
    assert "standard) export ROLLOUTS=" in source
    assert "app_launcher_only)" in source
    assert "app_launcher_only requires ROLLOUTS=1" in source
    gpu_count = source.index('[[ "${SLURM_GPUS_ON_NODE:-}" == 1 ]]')
    gpu_index = source.index('[[ "${SLURM_JOB_GPUS:-}" =~ ^[0-9]+$ ]]')
    worker = source.index("exec /usr/bin/bash --noprofile --norc")
    assert gpu_count < gpu_index < worker
