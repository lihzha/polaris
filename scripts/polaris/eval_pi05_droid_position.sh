#!/usr/bin/env bash

# Run one fresh official pi0.5-DROID canary through the measured-q position adapter.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLARIS_DIR="${POLARIS_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
NFS_ROOT="${NFS_ROOT:-/lustre/fsw/portfolios/nvr/users/lzha}"
OPENPI_DIR="${OPENPI_DIR:?Set the frozen OpenPI bd70 checkout with its exact venv}"
DROID_DIR="${DROID_DIR:?Set the frozen official DROID 33ae6a checkout}"
POLARIS_DATA_DIR="${POLARIS_DATA_DIR:-${NFS_ROOT}/data/PolaRiS-Hub}"
POLARIS_PYXIS_IMAGE="${POLARIS_PYXIS_IMAGE:-${NFS_ROOT}/cache/polaris/polaris-eval-cuda13-fd00a51.sqsh}"
HOST_MEDIA_TOOLS_ROOT=/lustre/fs11/portfolios/nvr/projects/nvr_lpr_rvp/users/lzha/cache/polaris/host-media-tools/ffmpeg-7.0.2-static-amd64-abda8d77ce830914
HOST_FFPROBE_PATH="${HOST_FFPROBE_PATH:-${HOST_MEDIA_TOOLS_ROOT}/ffprobe}"
HOST_FFMPEG_PATH="${HOST_FFMPEG_PATH:-${HOST_MEDIA_TOOLS_ROOT}/ffmpeg}"
EXPECTED_HOST_MEDIA_TOOLS_MANIFEST_SHA256=09d95a1f28e9e9af1e172439806ca9c2d6b19dd661f9f5f4ee7f51185cb99be5
EXPECTED_HOST_FFPROBE_SHA256=4f231a1960d83e403d08f7971e271707bec278a9ae18e21b8b5b03186668450d
EXPECTED_HOST_FFMPEG_SHA256=e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99
POLARIS_VULKAN_ICD_PATH="${POLARIS_VULKAN_ICD_PATH:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
POLARIS_CACHE_DIR="${POLARIS_CACHE_DIR:-${NFS_ROOT}/cache/polaris/runtime/pi05-position-${SLURM_JOB_ID:-manual}}"
CHECKPOINT_MANIFEST="${SCRIPT_DIR}/pi05_droid_native_gcs_manifest.tsv"
PROTOCOL=polaris-native-droid-freshq-delta0p2-position-h8-canary1-v1
CANARY_PROFILE=openpi_pi05_droid_fresh_jointdelta_position_polaris_canary_v1
POLARIS_ENVIRONMENT=DROID-FoodBussing
REQUESTED_PORT=0
SERVER_START_TIMEOUT_SECS="${SERVER_START_TIMEOUT_SECS:-2400}"

: "${RUN_DIR:?Set one fresh pre-created canary run directory}"
: "${EXPECTED_POLARIS_COMMIT:?Set the immutable combined launch commit}"
: "${POSITION_CONTROLLER_ATTESTATION:?Set the immutable position-smoke attestation}"
: "${EXPECTED_POSITION_CONTROLLER_ATTESTATION_SHA256:?Set its exact SHA-256}"

die() { echo "ERROR: $*" >&2; exit 2; }

for forbidden in CONTROLLER_COMPLETION EXPECTED_CONTROLLER_COMPLETION_SHA256 \
  ALL_SIX_CONTROLLER_COMPLETION EXPECTED_ALL_SIX_COMPLETION_SHA256 \
  EXPECTED_ALL_SIX_PROFILE; do
  [[ -z "${!forbidden+x}" ]] \
    || die "Old direct-rad/s controller gate is forbidden: ${forbidden}"
done
[[ -n "${SLURM_JOB_ID:-}" ]] || die "A Slurm allocation is required"
[[ "${EXPECTED_POLARIS_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || die "Malformed PolaRiS commit"
[[ "${EXPECTED_POSITION_CONTROLLER_ATTESTATION_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "Malformed position-controller attestation SHA-256"
[[ -z "${PORT+x}" ]] || die "Ambient PORT is forbidden; server must bind port 0"
[[ -z "${RESUME_FROM_TASK_DIR:-}" ]] || die "Position canary forbids resume"
[[ "${SERVER_START_TIMEOUT_SECS}" =~ ^[1-9][0-9]*$ ]] \
  || die "SERVER_START_TIMEOUT_SECS must be a positive integer"

# Canonicalize after proving the lexical inputs exist.  fsw and fs11 may be
# aliases of the same Lustre root; all later comparisons use resolved identity.
for directory in POLARIS_DIR OPENPI_DIR DROID_DIR POLARIS_DATA_DIR RUN_DIR; do
  value="${!directory}"
  [[ -d "${value}" && ! -L "${value}" ]] || die "${directory} is not a regular directory"
  printf -v "${directory}" '%s' "$(realpath -e -- "${value}")"
done
for file_var in POLARIS_PYXIS_IMAGE POLARIS_VULKAN_ICD_PATH \
  POSITION_CONTROLLER_ATTESTATION HOST_FFPROBE_PATH HOST_FFMPEG_PATH; do
  value="${!file_var}"
  [[ -f "${value}" && ! -L "${value}" ]] || die "${file_var} is not a regular file"
  printf -v "${file_var}" '%s' "$(realpath -e -- "${value}")"
done
POLARIS_CACHE_DIR="$(realpath -m -- "${POLARIS_CACHE_DIR}")"

[[ -d "${POLARIS_DIR}/.git" && ! -L "${POLARIS_DIR}/.git" ]] \
  || die "PolaRiS must be a standalone clone"
[[ "$(git -C "${POLARIS_DIR}" rev-parse --show-toplevel)" == "${POLARIS_DIR}" ]] \
  || die "PolaRiS root mismatch"
[[ "$(git -C "${POLARIS_DIR}" rev-parse --absolute-git-dir)" == "${POLARIS_DIR}/.git" ]] \
  || die "PolaRiS Git metadata escaped source root"
[[ "$(git -C "${POLARIS_DIR}" rev-parse --path-format=absolute --git-common-dir)" == "${POLARIS_DIR}/.git" ]] \
  || die "PolaRiS common Git directory escaped source root"
[[ "$(git -C "${POLARIS_DIR}" rev-parse --abbrev-ref HEAD)" == HEAD ]] \
  || die "PolaRiS launch source must use detached HEAD"
[[ "$(git -C "${POLARIS_DIR}" rev-parse HEAD)" == "${EXPECTED_POLARIS_COMMIT}" ]] \
  || die "PolaRiS commit mismatch"
[[ -z "$(git -C "${POLARIS_DIR}" status --porcelain=v1 --untracked-files=all)" ]] \
  || die "PolaRiS launch source must be completely clean"
[[ -x "${OPENPI_DIR}/.venv/bin/python" ]] || die "Exact OpenPI venv is missing"
OPENPI_GIT_DIR="$(git -C "${OPENPI_DIR}" rev-parse --absolute-git-dir)"
OPENPI_GIT_COMMON_DIR="$(git -C "${OPENPI_DIR}" rev-parse --path-format=absolute --git-common-dir)"
OPENPI_GIT_DIR="$(realpath -e -- "${OPENPI_GIT_DIR}")"
OPENPI_GIT_COMMON_DIR="$(realpath -e -- "${OPENPI_GIT_COMMON_DIR}")"
[[ -d "${OPENPI_GIT_DIR}" && ! -L "${OPENPI_GIT_DIR}" \
  && -d "${OPENPI_GIT_COMMON_DIR}" && ! -L "${OPENPI_GIT_COMMON_DIR}" ]] \
  || die "OpenPI Git metadata directories are invalid"
HOST_PYTHONPATH="${POLARIS_DIR}/src:${SCRIPT_DIR}:${OPENPI_DIR}/packages/openpi-client/src"

CHECKPOINT_SNAPSHOT="${RUN_DIR}/checkpoint_snapshot"
SNAPSHOT_CREATION="${RUN_DIR}/checkpoint_snapshot_creation.json"
CHECKPOINT_PRE="${RUN_DIR}/checkpoint_pre_attestation.json"
CHECKPOINT_POST="${RUN_DIR}/checkpoint_post_attestation.json"
PREFLIGHT_RECORD="${RUN_DIR}/preflight.json"
RESOLVED_CONTRACT="${RUN_DIR}/resolved_contract.json"
SUBMISSION_RECORD="${RUN_DIR}/submission-${SLURM_JOB_ID}.json"
for required in "${CHECKPOINT_SNAPSHOT}" "${SNAPSHOT_CREATION}" \
  "${PREFLIGHT_RECORD}" "${RESOLVED_CONTRACT}" "${SUBMISSION_RECORD}"; do
  [[ -e "${required}" && ! -L "${required}" ]] || die "Missing submit-time artifact: ${required}"
done

FINALIZER="${SCRIPT_DIR}/finalize_pi05_droid_position_eval.py"
VERIFY_CHECKPOINT="${SCRIPT_DIR}/verify_pi05_droid_position_checkpoint.py"
BOUND_PORT_VALIDATOR="${SCRIPT_DIR}/validate_pi05_droid_bound_port.py"
HANDSHAKE_VALIDATOR="${SCRIPT_DIR}/validate_pi05_droid_position_handshake.py"

runtime_args=(
  --polaris-repo "${POLARIS_DIR}"
  --expected-polaris-commit "${EXPECTED_POLARIS_COMMIT}"
  --openpi-dir "${OPENPI_DIR}"
  --droid-dir "${DROID_DIR}"
  --position-controller-attestation "${POSITION_CONTROLLER_ATTESTATION}"
  --expected-position-controller-attestation-sha256 "${EXPECTED_POSITION_CONTROLLER_ATTESTATION_SHA256}"
  --container-image "${POLARIS_PYXIS_IMAGE}"
  --data-dir "${POLARIS_DATA_DIR}"
  --expected-host-media-tools-manifest-sha256 "${EXPECTED_HOST_MEDIA_TOOLS_MANIFEST_SHA256}"
  --host-ffprobe-path "${HOST_FFPROBE_PATH}"
  --expected-host-ffprobe-sha256 "${EXPECTED_HOST_FFPROBE_SHA256}"
  --host-ffmpeg-path "${HOST_FFMPEG_PATH}"
  --expected-host-ffmpeg-sha256 "${EXPECTED_HOST_FFMPEG_SHA256}"
)
TASK_DIR="${RUN_DIR}/${POLARIS_ENVIRONMENT}"
SERVER_LOG="${RUN_DIR}/policy_server.log"
EVAL_LOG="${TASK_DIR}/eval.log"
TRACE_PATH="${TASK_DIR}/policy_traces.jsonl"
RUNTIME_PATH="${TASK_DIR}/position_runtime.json"
LIFECYCLE_PATH="${TASK_DIR}/evaluator_close_ready.json"
SIDECAR_PATH="${TASK_DIR}/native_runtime/episode_000000.json"
SERVING_CONTRACT_PATH="${RUN_DIR}/ego_lap_serving_contract.json"
MODEL_RUNTIME_CONTRACT="${RUN_DIR}/pi05_droid_position_model_runtime.json"
INFERENCE_ENVIRONMENT="${RUN_DIR}/inference_environment.json"
RUN_RECORD="${RUN_DIR}/run_record.json"
COMMANDS_FILE="${RUN_DIR}/commands.sh"
BOUND_PORT_FILE="${RUN_DIR}/policy_bound_port.json"
HANDSHAKE_PATH="${RUN_DIR}/policy_handshake.json"
FAILURE_PATH="${RUN_DIR}/attempt_failed.json"
SRUN_STATUS="${RUN_DIR}/srun-${SLURM_JOB_ID}.status.json"
BOUND_PORT_TOKEN="$(printf '%s\0%s\0%s\0%s\0' \
  "${EXPECTED_POLARIS_COMMIT}" "${SLURM_JOB_ID}" "${RUN_DIR}" "${PROTOCOL}" \
  | sha256sum | awk '{print $1}')"
SERVER_PID=""
FAILURE_STAGE=checkpoint_pre_attestation

mkdir -p "${TASK_DIR}" "${POLARIS_CACHE_DIR}/home"
for output in "${CHECKPOINT_PRE}" "${CHECKPOINT_POST}" "${SERVER_LOG}" "${EVAL_LOG}" \
  "${TRACE_PATH}" "${RUNTIME_PATH}" "${LIFECYCLE_PATH}" "${SIDECAR_PATH}" \
  "${SERVING_CONTRACT_PATH}" "${MODEL_RUNTIME_CONTRACT}" "${INFERENCE_ENVIRONMENT}" \
  "${RUN_RECORD}" "${COMMANDS_FILE}" "${BOUND_PORT_FILE}" "${HANDSHAKE_PATH}" \
  "${FAILURE_PATH}" "${SRUN_STATUS}" "${TASK_DIR}/eval_results.csv" \
  "${TASK_DIR}/episode_0.mp4" "${RUN_DIR}/eval_success.txt"; do
  [[ ! -e "${output}" && ! -L "${output}" ]] || die "Refusing existing output: ${output}"
done

stop_server() {
  [[ -n "${SERVER_PID}" ]] || return 0
  kill -TERM -- "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null || true
  for _ in {1..40}; do
    kill -0 "${SERVER_PID}" 2>/dev/null || break
    sleep 0.25
  done
  kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=""
}

on_exit() {
  local code=$?
  trap - EXIT INT TERM
  set +e
  stop_server
  if (( code != 0 )) && [[ ! -e "${FAILURE_PATH}" && ! -L "${FAILURE_PATH}" ]]; then
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${HOST_PYTHONPATH}" \
      "${OPENPI_DIR}/.venv/bin/python" "${FINALIZER}" publish-failure \
      --output "${FAILURE_PATH}" --run-dir "${RUN_DIR}" --job-id "${SLURM_JOB_ID}" \
      --failure-stage "${FAILURE_STAGE:-unknown}" --exit-code "${code}" || true
  fi
  exit "${code}"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${HOST_PYTHONPATH}" \
  "${OPENPI_DIR}/.venv/bin/python" "${FINALIZER}" preflight "${runtime_args[@]}" >/dev/null

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${HOST_PYTHONPATH}" \
  "${OPENPI_DIR}/.venv/bin/python" "${VERIFY_CHECKPOINT}" attest \
  "${CHECKPOINT_SNAPSHOT}" "${CHECKPOINT_MANIFEST}" --phase pre_server \
  --output "${CHECKPOINT_PRE}"

FAILURE_STAGE=inference_environment
OPENPI_DATA_HOME="${RUN_DIR}/openpi_data_home_unused" JAX_PLATFORMS=cuda \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${HOST_PYTHONPATH}" \
  "${OPENPI_DIR}/.venv/bin/python" "${SCRIPT_DIR}/capture_pi05_droid_position_environment.py" \
  --openpi-dir "${OPENPI_DIR}" --output "${INFERENCE_ENVIRONMENT}"

server_command=(
  "${OPENPI_DIR}/.venv/bin/python" "${SCRIPT_DIR}/serve_pi05_droid_position.py"
  --checkpoint-dir "${CHECKPOINT_SNAPSHOT}"
  --openpi-dir "${OPENPI_DIR}"
  --droid-dir "${DROID_DIR}"
  --manifest "${CHECKPOINT_MANIFEST}"
  --serving-contract-output "${SERVING_CONTRACT_PATH}"
  --model-runtime-contract-output "${MODEL_RUNTIME_CONTRACT}"
  --port "${REQUESTED_PORT}"
  --bound-port-output "${BOUND_PORT_FILE}"
  --bound-port-token "${BOUND_PORT_TOKEN}"
)
FAILURE_STAGE=server_bind_and_readiness
(
  cd "${POLARIS_DIR}"
  exec setsid env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${POLARIS_DIR}/src" \
    JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 "${server_command[@]}"
) >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

elapsed=0
while [[ ! -f "${BOUND_PORT_FILE}" || -L "${BOUND_PORT_FILE}" \
  || "$(stat -c '%a' -- "${BOUND_PORT_FILE}" 2>/dev/null || true)" != 444 ]]; do
  kill -0 "${SERVER_PID}" 2>/dev/null || { tail -n 160 "${SERVER_LOG}" >&2 || true; die "Policy server exited before bind"; }
  sleep 2
  elapsed=$((elapsed + 2))
  (( elapsed < SERVER_START_TIMEOUT_SECS )) || die "Policy server bind timed out"
done

bound_port_command=(
  "${OPENPI_DIR}/.venv/bin/python" "${BOUND_PORT_VALIDATOR}"
  --artifact "${BOUND_PORT_FILE}" --expected-pid "${SERVER_PID}"
  --expected-launch-token "${BOUND_PORT_TOKEN}" --expected-requested-port 0
  --require-live-pid --output-format tsv
)
BOUND_PORT_VALIDATION="$(PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${POLARIS_DIR}/src" "${bound_port_command[@]}")"
IFS=$'\t' read -r ACTUAL_PORT BOUND_PORT_FILE_SHA256 BOUND_PORT_FILE_IDENTITY BOUND_PORT_EXTRA <<<"${BOUND_PORT_VALIDATION}"
[[ -z "${BOUND_PORT_EXTRA}" && "${ACTUAL_PORT}" =~ ^[1-9][0-9]{0,4}$ \
  && "${BOUND_PORT_FILE_SHA256}" =~ ^[0-9a-f]{64}$ \
  && "${BOUND_PORT_FILE_IDENTITY}" =~ ^[0-9]+(:[0-9]+){6}$ ]] \
  || die "Invalid bound-port artifact"
(( 10#${ACTUAL_PORT} <= 65535 )) || die "Bound port is out of range"

handshake_command=(
  "${OPENPI_DIR}/.venv/bin/python" "${HANDSHAKE_VALIDATOR}"
  --host 127.0.0.1 --port "${ACTUAL_PORT}" --expected-server-pid "${SERVER_PID}"
  --openpi-dir "${OPENPI_DIR}" --serving-contract "${SERVING_CONTRACT_PATH}"
  --timeout-seconds 3 --output "${HANDSHAKE_PATH}"
)
while true; do
  kill -0 "${SERVER_PID}" 2>/dev/null || die "Policy server exited before handshake"
  set +e
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${POLARIS_DIR}/src" "${handshake_command[@]}"
  handshake_code=$?
  set -e
  (( handshake_code == 0 )) && break
  (( handshake_code == 3 )) || die "Real WebSocket handshake failed"
  sleep 2
  elapsed=$((elapsed + 2))
  (( elapsed < SERVER_START_TIMEOUT_SECS )) || die "Real WebSocket handshake timed out"
done
[[ "$(PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${POLARIS_DIR}/src" "${bound_port_command[@]}")" == "${BOUND_PORT_VALIDATION}" ]] \
  || die "Bound-port identity changed across handshake"
for listener_artifact in "${SERVING_CONTRACT_PATH}" "${MODEL_RUNTIME_CONTRACT}" \
  "${HANDSHAKE_PATH}"; do
  [[ -f "${listener_artifact}" && ! -L "${listener_artifact}" \
    && "$(stat -c '%a' -- "${listener_artifact}")" == 444 \
    && "$(stat -c '%h' -- "${listener_artifact}")" == 1 ]] \
    || die "Listener artifact is not one immutable file: ${listener_artifact}"
done

export RUN_RECORD RUN_DIR POLARIS_DIR OPENPI_DIR DROID_DIR CHECKPOINT_SNAPSHOT \
  CHECKPOINT_MANIFEST POLARIS_PYXIS_IMAGE POLARIS_DATA_DIR SERVING_CONTRACT_PATH \
  MODEL_RUNTIME_CONTRACT BOUND_PORT_FILE HANDSHAKE_PATH POSITION_CONTROLLER_ATTESTATION \
  EXPECTED_POSITION_CONTROLLER_ATTESTATION_SHA256 RESOLVED_CONTRACT BOUND_PORT_TOKEN \
  BOUND_PORT_FILE_SHA256 BOUND_PORT_FILE_IDENTITY ACTUAL_PORT SERVER_PID \
  CANARY_PROFILE PROTOCOL OPENPI_GIT_DIR OPENPI_GIT_COMMON_DIR
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${POLARIS_DIR}/src" \
  "${OPENPI_DIR}/.venv/bin/python" - <<'PY'
import os
from pathlib import Path
from polaris.pi05_droid_native_eval_contract import publish_immutable_json, validate_immutable_json
resolved = validate_immutable_json(Path(os.environ["RESOLVED_CONTRACT"]))
publish_immutable_json(
    Path(os.environ["RUN_RECORD"]),
    {
        "schema_version": 1,
        "profile": os.environ["CANARY_PROFILE"],
        "protocol": os.environ["PROTOCOL"],
        "job_id": int(os.environ["SLURM_JOB_ID"]),
        "task": "DROID-FoodBussing",
        "rollouts": 1,
        "fresh_attempt_no_resume": True,
        "paths": {
            "run_dir": os.environ["RUN_DIR"],
            "polaris_dir": os.environ["POLARIS_DIR"],
            "openpi_dir": os.environ["OPENPI_DIR"],
            "openpi_git_dir": os.environ["OPENPI_GIT_DIR"],
            "openpi_git_common_dir": os.environ["OPENPI_GIT_COMMON_DIR"],
            "droid_dir": os.environ["DROID_DIR"],
            "checkpoint_snapshot": os.environ["CHECKPOINT_SNAPSHOT"],
            "checkpoint_manifest": os.environ["CHECKPOINT_MANIFEST"],
            "container_image": os.environ["POLARIS_PYXIS_IMAGE"],
            "polaris_data_dir": os.environ["POLARIS_DATA_DIR"],
            "serving_contract": os.environ["SERVING_CONTRACT_PATH"],
            "model_runtime_contract": os.environ["MODEL_RUNTIME_CONTRACT"],
            "bound_port": os.environ["BOUND_PORT_FILE"],
            "handshake": os.environ["HANDSHAKE_PATH"],
        },
        "listener": {
            "requested_port": 0,
            "actual_port": int(os.environ["ACTUAL_PORT"]),
            "server_pid": int(os.environ["SERVER_PID"]),
            "launch_token": os.environ["BOUND_PORT_TOKEN"],
            "bound_port_sha256": os.environ["BOUND_PORT_FILE_SHA256"],
            "bound_port_stable_identity": os.environ["BOUND_PORT_FILE_IDENTITY"],
        },
        "controller_authorization": {
            "profile": "openpi_pi05_droid_position_controller_smoke_attestation_v1",
            "path": str(Path(os.environ["POSITION_CONTROLLER_ATTESTATION"]).resolve()),
            "sha256": os.environ["EXPECTED_POSITION_CONTROLLER_ATTESTATION_SHA256"],
            "old_jointvelocity_controller_gates": "forbidden",
        },
        "resolved_contract_artifact": {
            key: resolved[key] for key in ("path", "size", "sha256", "mode", "nlink")
        },
    },
)
PY

eval_args=(
  scripts/eval.py --environment "${POLARIS_ENVIRONMENT}" --control-mode joint-position
  --policy.client DroidDeltaJointPosition --policy.host 127.0.0.1 --policy.port "${ACTUAL_PORT}"
  --policy.open-loop-horizon 8 --policy.state-type joint_position
  --policy.frame-description 'robot base frame' --policy.action-frame robot_base
  --policy.dataset-name droid --policy.no-rotate-wrist-180
  --policy.expected-action-horizon 15 --policy.expected-action-dim 8
  --policy.policy-profile openpi_pi05_droid_fresh_jointdelta_position_v1
  --policy.serving-contract-path "${SERVING_CONTRACT_PATH}"
  --policy.openpi-dir "${OPENPI_DIR}" --policy.trace-path "${TRACE_PATH}"
  --runtime-contract-path "${RUNTIME_PATH}" --lifecycle-ready-path "${LIFECYCLE_PATH}"
  --expected-gripper-drive-profile implicit_gripper_physx_velocity_limit5_followers5_every_reset_cuda_actuator_cpu_static_physx_v1
  --run-folder "${TASK_DIR}" --rollouts 1 --headless
)
mounts="/dev/shm:/dev/shm,${POLARIS_DIR}:${POLARIS_DIR}:ro,${OPENPI_DIR}:${OPENPI_DIR}:ro"
if [[ "${OPENPI_GIT_DIR}" != "${OPENPI_DIR}"/* ]]; then
  mounts+=",${OPENPI_GIT_DIR}:${OPENPI_GIT_DIR}:ro"
fi
if [[ "${OPENPI_GIT_COMMON_DIR}" != "${OPENPI_GIT_DIR}" \
  && "${OPENPI_GIT_COMMON_DIR}" != "${OPENPI_DIR}"/* ]]; then
  mounts+=",${OPENPI_GIT_COMMON_DIR}:${OPENPI_GIT_COMMON_DIR}:ro"
fi
mounts+=",${POLARIS_DATA_DIR}:${POLARIS_DATA_DIR}:ro,${RUN_DIR}:${RUN_DIR}:rw,${CHECKPOINT_SNAPSHOT}:${CHECKPOINT_SNAPSHOT}:ro,${POLARIS_CACHE_DIR}:/cache:rw,${POLARIS_VULKAN_ICD_PATH}:/etc/vulkan/icd.d/nvidia_icd.json:ro"
eval_command=(
  srun --ntasks=1 "--cpus-per-task=${SLURM_CPUS_PER_TASK:-16}"
  "--container-image=${POLARIS_PYXIS_IMAGE}" "--container-mounts=${mounts}"
  "--container-workdir=${POLARIS_DIR}" --no-container-entrypoint --no-container-mount-home
  --container-remap-root --container-writable
  "--container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES" --export=ALL
  /usr/bin/env VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json ACCEPT_EULA=Y
  OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y OMNI_KIT_ALLOW_ROOT=1
  PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
  "PYTHONPATH=${POLARIS_DIR}/src:${OPENPI_DIR}/packages/openpi-client/src"
  "POLARIS_DATA_PATH=${POLARIS_DATA_DIR}" XDG_CACHE_HOME=/cache
  HF_HOME=/cache/huggingface HOME=/cache/home /.venv/bin/python "${eval_args[@]}"
)
finalizer_command=(
  "${OPENPI_DIR}/.venv/bin/python" "${FINALIZER}" finalize
  --job-id "${SLURM_JOB_ID}" --run-dir "${RUN_DIR}" "${runtime_args[@]}"
)
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\n'
  printf 'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=%q %q %q attest %q %q --phase pre_server --output %q\n' \
    "${HOST_PYTHONPATH}" "${OPENPI_DIR}/.venv/bin/python" "${VERIFY_CHECKPOINT}" \
    "${CHECKPOINT_SNAPSHOT}" "${CHECKPOINT_MANIFEST}" "${CHECKPOINT_PRE}"
  printf 'OPENPI_DATA_HOME=%q JAX_PLATFORMS=cuda PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=%q %q %q --openpi-dir %q --output %q\n' \
    "${RUN_DIR}/openpi_data_home_unused" "${HOST_PYTHONPATH}" \
    "${OPENPI_DIR}/.venv/bin/python" "${SCRIPT_DIR}/capture_pi05_droid_position_environment.py" \
    "${OPENPI_DIR}" "${INFERENCE_ENVIRONMENT}"
  printf 'cd %q && env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=%q JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 ' \
    "${POLARIS_DIR}" "${POLARIS_DIR}/src"
  printf '%q ' "${server_command[@]}"; printf '\n'
  printf 'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=%q ' "${POLARIS_DIR}/src"
  printf '%q ' "${bound_port_command[@]}"; printf '\n'
  printf 'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=%q ' "${POLARIS_DIR}/src"
  printf '%q ' "${handshake_command[@]}"; printf '\n'
  printf '%q ' "${eval_command[@]}"; printf '\n'
  printf 'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=%q %q %q attest %q %q --phase post_server --output %q\n' \
    "${HOST_PYTHONPATH}" "${OPENPI_DIR}/.venv/bin/python" "${VERIFY_CHECKPOINT}" \
    "${CHECKPOINT_SNAPSHOT}" "${CHECKPOINT_MANIFEST}" "${CHECKPOINT_POST}"
  printf 'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=%q ' "${HOST_PYTHONPATH}"
  printf '%q ' "${finalizer_command[@]}"; printf '\n'
} >"${COMMANDS_FILE}"
chmod 0444 "${COMMANDS_FILE}"
sync -f "${COMMANDS_FILE}"
sync -f "${RUN_DIR}"

FAILURE_STAGE=evaluator_execution
set +e
(cd "${POLARIS_DIR}" && "${eval_command[@]}") 2>&1 | tee "${EVAL_LOG}"
pipeline_codes=("${PIPESTATUS[@]}")
set -e
eval_code="${pipeline_codes[0]}"
tee_code="${pipeline_codes[1]}"
set +e
BOUND_PORT_AFTER="$(PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${POLARIS_DIR}/src" "${bound_port_command[@]}")"
bound_after_code=$?
set -e
stop_server

FAILURE_STAGE=checkpoint_post_attestation
post_code=0
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${HOST_PYTHONPATH}" \
  "${OPENPI_DIR}/.venv/bin/python" "${VERIFY_CHECKPOINT}" attest \
  "${CHECKPOINT_SNAPSHOT}" "${CHECKPOINT_MANIFEST}" --phase post_server \
  --output "${CHECKPOINT_POST}" || post_code=$?

export SRUN_STATUS eval_code tee_code bound_after_code post_code BOUND_PORT_AFTER BOUND_PORT_VALIDATION
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${POLARIS_DIR}/src" \
  "${OPENPI_DIR}/.venv/bin/python" - <<'PY'
import os
from pathlib import Path
from polaris.pi05_droid_native_eval_contract import publish_immutable_json
publish_immutable_json(
    Path(os.environ["SRUN_STATUS"]),
    {
        "schema_version": 1,
        "profile": os.environ["CANARY_PROFILE"],
        "job_id": int(os.environ["SLURM_JOB_ID"]),
        "srun_exit_code": int(os.environ["eval_code"]),
        "tee_exit_code": int(os.environ["tee_code"]),
        "bound_port_unchanged_after_eval": (
            int(os.environ["bound_after_code"]) == 0
            and os.environ["BOUND_PORT_AFTER"] == os.environ["BOUND_PORT_VALIDATION"]
        ),
        "checkpoint_post_attestation_present": int(os.environ["post_code"]) == 0,
    },
)
PY
(( bound_after_code == 0 )) || die "Bound-port validation failed after evaluator"
[[ "${BOUND_PORT_AFTER}" == "${BOUND_PORT_VALIDATION}" ]] || die "Bound-port artifact changed during evaluator"
(( post_code == 0 )) || exit "${post_code}"
FAILURE_STAGE=evaluator_execution
(( eval_code == 0 )) || exit "${eval_code}"
(( tee_code == 0 )) || exit "${tee_code}"
[[ -f "${LIFECYCLE_PATH}" && ! -L "${LIFECYCLE_PATH}" \
  && "$(stat -c '%a' -- "${LIFECYCLE_PATH}")" == 444 ]] \
  || die "Evaluator returned zero without immutable close-ready evidence"

FAILURE_STAGE=host_finalization
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${HOST_PYTHONPATH}" "${finalizer_command[@]}"
echo "Official pi0.5-DROID corrected position canary complete: ${RUN_DIR}"
