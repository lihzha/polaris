#!/usr/bin/env bash

# Run one fresh, immutable official pi0.5-DROID native-velocity canary.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLARIS_DIR="${POLARIS_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
NFS_ROOT="${NFS_ROOT:-/lustre/fsw/portfolios/nvr/users/lzha}"
OPENPI_DIR="${OPENPI_DIR:-${POLARIS_DIR}/third_party/openpi}"
POLARIS_DATA_DIR="${POLARIS_DATA_DIR:-${NFS_ROOT}/data/PolaRiS-Hub}"
POLARIS_PYXIS_IMAGE="${POLARIS_PYXIS_IMAGE:-${NFS_ROOT}/cache/polaris/polaris-eval-cuda13-fd00a51.sqsh}"
POLARIS_VULKAN_ICD_PATH="${POLARIS_VULKAN_ICD_PATH:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${NFS_ROOT}/cache/openpi-pi05-droid-native-v1}"
POLARIS_CACHE_DIR="${POLARIS_CACHE_DIR:-${NFS_ROOT}/cache/polaris/runtime/pi05-native-${SLURM_JOB_ID:-manual}}"
CHECKPOINT_URI=gs://openpi-assets/checkpoints/pi05_droid
CHECKPOINT_MANIFEST="${SCRIPT_DIR}/pi05_droid_native_gcs_manifest.tsv"
EXPECTED_OPENPI_COMMIT=bd70b8f4011e85b3f3b0f039f12113f78718e7bf
EXPECTED_IMAGE_SHA256=ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a
EXPECTED_MANIFEST_SHA256=6f9ccfa5695c669962ad10dbe0dcb7d44bf903918e5fffe33e5d1ff531287922
EXPECTED_NORM_SHA256=403b3a22f897e9ae5dd617966a3c8f7d1835ac79dfd5a8993179514be26a3b8b
POLARIS_ENVIRONMENT=DROID-FoodBussing
REQUESTED_PORT=0
SERVER_START_TIMEOUT_SECS="${SERVER_START_TIMEOUT_SECS:-2400}"

: "${RUN_DIR:?Set RUN_DIR to one fresh canary attempt directory}"
: "${EXPECTED_POLARIS_COMMIT:?Set EXPECTED_POLARIS_COMMIT to the immutable launch commit}"
: "${CONTROLLER_COMPLETION:?Set CONTROLLER_COMPLETION to job1098174 completion JSON}"
: "${EXPECTED_CONTROLLER_COMPLETION_SHA256:?Set exact job1098174 completion SHA-256}"
: "${ALL_SIX_CONTROLLER_COMPLETION:?Set accepted job1098682 all-six completion JSON}"
: "${EXPECTED_ALL_SIX_COMPLETION_SHA256:?Set exact all-six completion SHA-256}"
: "${EXPECTED_ALL_SIX_PROFILE:?Set the independently reviewed all-six profile}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ -n "${SLURM_JOB_ID:-}" ]] || die "A Slurm allocation is required"
[[ "${EXPECTED_POLARIS_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || die "Malformed PolaRiS commit"
[[ "${EXPECTED_CONTROLLER_COMPLETION_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "Malformed job1098174 completion digest"
[[ "${EXPECTED_ALL_SIX_COMPLETION_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "Malformed all-six completion digest"
[[ -z "${PORT+x}" ]] \
  || die "Ambient PORT is forbidden; the WebSocket server must bind port 0"
[[ "${SERVER_START_TIMEOUT_SECS}" =~ ^[1-9][0-9]*$ ]] \
  || die "SERVER_START_TIMEOUT_SECS must be a positive integer"
[[ -z "${RESUME_FROM_TASK_DIR:-}" ]] || die "Native flow attempts forbid prefix resume"

[[ ! -L "${POLARIS_DIR}" ]] || die "POLARIS_DIR must not be a symlink"
POLARIS_DIR="$(realpath "${POLARIS_DIR}")"
[[ -d "${POLARIS_DIR}/.git" && ! -L "${POLARIS_DIR}/.git" ]] \
  || die "PolaRiS must be a standalone clone with an in-root .git directory"
git_dir="$(git -C "${POLARIS_DIR}" rev-parse --absolute-git-dir)"
git_common_dir="$(git -C "${POLARIS_DIR}" rev-parse --path-format=absolute --git-common-dir)"
[[ "${git_dir}" == "${POLARIS_DIR}/.git" && "${git_common_dir}" == "${POLARIS_DIR}/.git" ]] \
  || die "PolaRiS Git metadata escaped the source mount"
[[ "$(git -C "${POLARIS_DIR}" rev-parse --abbrev-ref HEAD)" == HEAD ]] \
  || die "PolaRiS launch checkout must use detached HEAD"
[[ "$(git -C "${POLARIS_DIR}" rev-parse HEAD)" == "${EXPECTED_POLARIS_COMMIT}" ]] \
  || die "PolaRiS commit mismatch"
[[ -z "$(git -C "${POLARIS_DIR}" status --porcelain=v1 --untracked-files=all)" ]] \
  || die "PolaRiS launch checkout must be completely clean"

[[ -f "${OPENPI_DIR}/.git" && ! -L "${OPENPI_DIR}/.git" ]] \
  || die "OpenPI submodule is not initialized"
[[ "$(git -C "${OPENPI_DIR}" rev-parse HEAD)" == "${EXPECTED_OPENPI_COMMIT}" ]] \
  || die "OpenPI commit mismatch"
[[ -z "$(git -C "${OPENPI_DIR}" status --porcelain=v1 --untracked-files=all)" ]] \
  || die "OpenPI checkout must be completely clean"
[[ -x "${OPENPI_DIR}/.venv/bin/python" ]] \
  || die "Build the exact OpenPI checkout-local venv with uv sync --frozen"
[[ -f "${POLARIS_PYXIS_IMAGE}" && ! -L "${POLARIS_PYXIS_IMAGE}" ]] \
  || die "Missing pinned Pyxis image"
[[ "$(sha256sum "${POLARIS_PYXIS_IMAGE}" | awk '{print $1}')" == "${EXPECTED_IMAGE_SHA256}" ]] \
  || die "Pyxis image SHA-256 mismatch"
[[ -d "${POLARIS_DATA_DIR}" && ! -L "${POLARIS_DATA_DIR}" ]] \
  || die "Missing regular PolaRiS-Hub root"
[[ -f "${POLARIS_VULKAN_ICD_PATH}" ]] || die "Missing Vulkan ICD"
[[ "$(sha256sum "${CHECKPOINT_MANIFEST}" | awk '{print $1}')" == "${EXPECTED_MANIFEST_SHA256}" ]] \
  || die "Native checkpoint manifest mismatch"
[[ -d "${RUN_DIR}" && ! -L "${RUN_DIR}" ]] || die "Missing regular RUN_DIR"
RUN_DIR="$(realpath -e -- "${RUN_DIR}")"

# This is intentionally the first Python action.  It blocks before checkpoint
# download or GPU work until both independently reviewed controller captures
# validate, including job1098682 all-six coupling and child lifecycle.
PYTHONPATH="${POLARIS_DIR}/src:${SCRIPT_DIR}" "${OPENPI_DIR}/.venv/bin/python" \
  "${SCRIPT_DIR}/finalize_pi05_droid_native_jointvelocity_eval.py" preflight \
  --polaris-repo "${POLARIS_DIR}" \
  --controller-completion "${CONTROLLER_COMPLETION}" \
  --expected-controller-completion-sha256 "${EXPECTED_CONTROLLER_COMPLETION_SHA256}" \
  --all-six-controller-completion "${ALL_SIX_CONTROLLER_COMPLETION}" \
  --expected-all-six-completion-sha256 "${EXPECTED_ALL_SIX_COMPLETION_SHA256}" \
  --expected-all-six-profile "${EXPECTED_ALL_SIX_PROFILE}"

TASK_DIR="${RUN_DIR}/${POLARIS_ENVIRONMENT}"
SERVER_LOG="${RUN_DIR}/policy_server.log"
EVAL_LOG="${TASK_DIR}/eval.log"
TRACE_PATH="${TASK_DIR}/policy_traces.jsonl"
RUNTIME_PATH="${TASK_DIR}/joint_velocity_runtime.json"
LIFECYCLE_PATH="${TASK_DIR}/evaluator_close_ready.json"
INCIDENT_PATH="${TASK_DIR}/native_failures/episode_000000.json"
SIDECAR_PATH="${TASK_DIR}/native_runtime/episode_000000.json"
SERVING_CONTRACT_PATH="${RUN_DIR}/ego_lap_serving_contract.json"
MODEL_RUNTIME_CONTRACT="${RUN_DIR}/pi05_droid_native_model_runtime.json"
CHECKPOINT_VERIFICATION="${RUN_DIR}/checkpoint_verification.json"
INFERENCE_ENVIRONMENT="${RUN_DIR}/inference_environment.json"
RUN_RECORD="${RUN_DIR}/run_record.json"
COMMANDS_FILE="${RUN_DIR}/commands.sh"
BOUND_PORT_FILE="${RUN_DIR}/policy_bound_port.json"
HANDSHAKE_PATH="${RUN_DIR}/policy_handshake.json"
FAILURE_PATH="${RUN_DIR}/attempt_failed.json"
BOUND_PORT_VALIDATOR="${SCRIPT_DIR}/validate_pi05_droid_bound_port.py"
HANDSHAKE_VALIDATOR="${SCRIPT_DIR}/validate_pi05_droid_handshake.py"
BOUND_PORT_TOKEN="$(
  printf '%s\0%s\0%s\0' \
    "${EXPECTED_POLARIS_COMMIT}" "${SLURM_JOB_ID}" "${RUN_DIR}" \
    | sha256sum | awk '{print $1}'
)"
[[ "${BOUND_PORT_TOKEN}" =~ ^[0-9a-f]{64}$ ]] \
  || die "Failed to derive the bound-port launch token"
SERVER_PID=""
FAILURE_STAGE=checkpoint_setup

mkdir -p "${TASK_DIR}" "${POLARIS_CACHE_DIR}/home"
for output in "${SERVER_LOG}" "${EVAL_LOG}" "${TRACE_PATH}" "${RUNTIME_PATH}" \
  "${LIFECYCLE_PATH}" "${SERVING_CONTRACT_PATH}" "${CHECKPOINT_VERIFICATION}" \
  "${MODEL_RUNTIME_CONTRACT}" "${INFERENCE_ENVIRONMENT}" "${RUN_RECORD}" "${COMMANDS_FILE}" \
  "${BOUND_PORT_FILE}" "${HANDSHAKE_PATH}" "${FAILURE_PATH}" \
  "${INCIDENT_PATH}" "${SIDECAR_PATH}" \
  "${TASK_DIR}/eval_results.csv" "${TASK_DIR}/episode_0.mp4" "${RUN_DIR}/eval_success.txt"; do
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
  if (( code != 0 )); then
    export FAILURE_PATH FAILURE_STAGE ATTEMPT_EXIT_CODE="${code}"
    export BOUND_PORT_ARTIFACT_PRESENT=false
    if [[ -f "${BOUND_PORT_FILE}" && ! -L "${BOUND_PORT_FILE}" ]]; then
      export BOUND_PORT_ARTIFACT_PRESENT=true
    fi
    PYTHONPATH="${POLARIS_DIR}/src" /usr/bin/python3 - <<'PY' || true
import os
from pathlib import Path
from polaris.pi05_droid_native_eval_contract import publish_immutable_json

publish_immutable_json(
    Path(os.environ["FAILURE_PATH"]),
    {
        "schema_version": 1,
        "profile": "openpi_pi05_droid_native_jointvelocity_polaris_canary_v1",
        "status": "failed_not_ready_for_promotion",
        "job_id": int(os.environ["SLURM_JOB_ID"]),
        "exit_code": int(os.environ["ATTEMPT_EXIT_CODE"]),
        "failure_stage": os.environ["FAILURE_STAGE"],
        "requested_port": 0,
        "bound_port_artifact_present": os.environ["BOUND_PORT_ARTIFACT_PRESENT"] == "true",
    },
)
PY
    printf 'native canary failed with exit %s at %s\n' "${code}" "$(date -Iseconds)" >&2
  fi
  exit "${code}"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

OPENPI_PYTHONPATH="${OPENPI_DIR}/src:${OPENPI_DIR}/packages/openpi-client/src"
checkpoint_path="$({
  OPENPI_DATA_HOME="${OPENPI_DATA_HOME}" PYTHONPATH="${OPENPI_PYTHONPATH}" \
    "${OPENPI_DIR}/.venv/bin/python" - "${CHECKPOINT_URI}" <<'PY'
import sys
from openpi.shared.download import maybe_download
print(maybe_download(sys.argv[1]))
PY
} | tail -n 1)"
[[ -d "${checkpoint_path}" && ! -L "${checkpoint_path}" ]] \
  || die "Official checkpoint did not resolve to one regular directory"

PYTHONPATH="${POLARIS_DIR}/src" "${OPENPI_DIR}/.venv/bin/python" \
  "${SCRIPT_DIR}/verify_pi05_droid_native_checkpoint.py" \
  "${checkpoint_path}" "${CHECKPOINT_MANIFEST}" --output "${CHECKPOINT_VERIFICATION}"
[[ "$(sha256sum "${checkpoint_path}/assets/droid/norm_stats.json" | awk '{print $1}')" == \
    "${EXPECTED_NORM_SHA256}" ]] || die "Checkpoint-local global DROID stats mismatch"

OPENPI_DATA_HOME="${OPENPI_DATA_HOME}" JAX_PLATFORMS=cuda \
  PYTHONPATH="${POLARIS_DIR}/src" "${OPENPI_DIR}/.venv/bin/python" \
  "${SCRIPT_DIR}/capture_pi05_droid_native_environment.py" \
  --openpi-dir "${OPENPI_DIR}" --output "${INFERENCE_ENVIRONMENT}"

server_command=(
  "${OPENPI_DIR}/.venv/bin/python"
  "${SCRIPT_DIR}/serve_pi05_droid_native_jointvelocity.py"
  --checkpoint-dir "${checkpoint_path}"
  --openpi-dir "${OPENPI_DIR}"
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
  exec setsid env \
    OPENPI_DATA_HOME="${OPENPI_DATA_HOME}" \
    PYTHONPATH="${POLARIS_DIR}/src" \
    JAX_PLATFORMS=cuda \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    PYTHONUNBUFFERED=1 \
    "${server_command[@]}"
) > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

elapsed=0
while [[ ! -f "${BOUND_PORT_FILE}" || -L "${BOUND_PORT_FILE}" \
  || "$(stat -c '%a' -- "${BOUND_PORT_FILE}" 2>/dev/null || true)" != 444 ]]; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    wait "${SERVER_PID}" || server_code=$?
    SERVER_PID=""
    tail -n 160 "${SERVER_LOG}" >&2 || true
    exit "${server_code:-1}"
  fi
  sleep 2
  elapsed=$((elapsed + 2))
  (( elapsed < SERVER_START_TIMEOUT_SECS )) \
    || die "Policy server did not publish a bound port before timeout"
done

bound_port_command=(
  "${OPENPI_DIR}/.venv/bin/python"
  "${BOUND_PORT_VALIDATOR}"
  --artifact "${BOUND_PORT_FILE}"
  --expected-pid "${SERVER_PID}"
  --expected-launch-token "${BOUND_PORT_TOKEN}"
  --expected-requested-port "${REQUESTED_PORT}"
  --require-live-pid
  --output-format tsv
)
set +e
BOUND_PORT_VALIDATION="$(PYTHONPATH="${POLARIS_DIR}/src" "${bound_port_command[@]}")"
bound_port_code=$?
set -e
IFS=$'\t' read -r ACTUAL_PORT BOUND_PORT_FILE_SHA256 \
  BOUND_PORT_FILE_IDENTITY BOUND_PORT_EXTRA <<< "${BOUND_PORT_VALIDATION}"
if [[ "${bound_port_code}" != 0 || -n "${BOUND_PORT_EXTRA}" \
  || ! "${ACTUAL_PORT}" =~ ^[1-9][0-9]{0,4}$ \
  || ! "${BOUND_PORT_FILE_SHA256}" =~ ^[0-9a-f]{64}$ \
  || ! "${BOUND_PORT_FILE_IDENTITY}" =~ ^[0-9]+:[0-9]+:[0-9]+:[0-9]+:[0-9]+:[0-9]+:[0-9]+$ ]] \
  || (( 10#${ACTUAL_PORT:-0} > 65535 )); then
  die "Policy server published an invalid bound-port artifact"
fi

handshake_command=(
  "${OPENPI_DIR}/.venv/bin/python"
  "${HANDSHAKE_VALIDATOR}"
  --host 127.0.0.1
  --port "${ACTUAL_PORT}"
  --expected-server-pid "${SERVER_PID}"
  --openpi-dir "${OPENPI_DIR}"
  --serving-contract "${SERVING_CONTRACT_PATH}"
  --timeout-seconds 3
  --output "${HANDSHAKE_PATH}"
)
while true; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    wait "${SERVER_PID}" || server_code=$?
    SERVER_PID=""
    tail -n 160 "${SERVER_LOG}" >&2 || true
    exit "${server_code:-1}"
  fi
  set +e
  PYTHONPATH="${POLARIS_DIR}/src" "${handshake_command[@]}"
  handshake_code=$?
  set -e
  if (( handshake_code == 0 )); then
    break
  fi
  (( handshake_code == 3 )) \
    || die "Policy server WebSocket handshake validation failed"
  sleep 2
  elapsed=$((elapsed + 2))
  (( elapsed < SERVER_START_TIMEOUT_SECS )) \
    || die "Policy server WebSocket handshake timed out"
done

BOUND_PORT_VALIDATION_AFTER_HANDSHAKE="$(
  PYTHONPATH="${POLARIS_DIR}/src" "${bound_port_command[@]}"
)" || die "Bound-port artifact failed post-handshake validation"
[[ "${BOUND_PORT_VALIDATION_AFTER_HANDSHAKE}" == "${BOUND_PORT_VALIDATION}" ]] \
  || die "Bound-port artifact changed across the real WebSocket handshake"
[[ -f "${MODEL_RUNTIME_CONTRACT}" && ! -L "${MODEL_RUNTIME_CONTRACT}" \
  && "$(stat -c '%a' -- "${MODEL_RUNTIME_CONTRACT}")" == 444 ]] \
  || die "Listener did not publish the immutable model runtime contract"
FAILURE_STAGE=evaluator_execution

eval_args=(
  scripts/eval.py
  --environment "${POLARIS_ENVIRONMENT}"
  --control-mode joint-velocity
  --policy.client DroidJointVelocity
  --policy.host 127.0.0.1
  --policy.port "${ACTUAL_PORT}"
  --policy.open-loop-horizon 8
  --policy.state-type joint_position
  --policy.frame-description 'robot base frame'
  --policy.action-frame robot_base
  --policy.dataset-name droid
  --policy.no-rotate-wrist-180
  --policy.expected-action-horizon 15
  --policy.expected-action-dim 8
  --policy.policy-profile openpi_pi05_droid_native_jointvelocity_v1
  --policy.serving-contract-path "${SERVING_CONTRACT_PATH}"
  --policy.openpi-dir "${OPENPI_DIR}"
  --policy.trace-path "${TRACE_PATH}"
  --runtime-contract-path "${RUNTIME_PATH}"
  --lifecycle-ready-path "${LIFECYCLE_PATH}"
  --expected-gripper-drive-profile implicit_gripper_physx_velocity_limit5_followers5_every_reset_cuda_actuator_cpu_static_physx_v1
  --run-folder "${TASK_DIR}"
  --rollouts 1
  --headless
)
mounts="/dev/shm:/dev/shm,${POLARIS_DIR}:${POLARIS_DIR}:ro,${POLARIS_DATA_DIR}:${POLARIS_DATA_DIR}:ro,${RUN_DIR}:${RUN_DIR}:rw,${POLARIS_CACHE_DIR}:/cache:rw,${POLARIS_VULKAN_ICD_PATH}:/etc/vulkan/icd.d/nvidia_icd.json:ro"
eval_command=(
  srun --ntasks=1 "--cpus-per-task=${SLURM_CPUS_PER_TASK:-16}"
  "--container-image=${POLARIS_PYXIS_IMAGE}"
  "--container-mounts=${mounts}"
  "--container-workdir=${POLARIS_DIR}"
  --no-container-entrypoint --no-container-mount-home --container-remap-root --container-writable
  "--container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES" --export=ALL
  /usr/bin/env
  VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
  ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y OMNI_KIT_ALLOW_ROOT=1
  PYTHONUNBUFFERED=1
  "PYTHONPATH=${POLARIS_DIR}/src:${OPENPI_DIR}/packages/openpi-client/src"
  "POLARIS_DATA_PATH=${POLARIS_DATA_DIR}"
  XDG_CACHE_HOME=/cache HF_HOME=/cache/huggingface HOME=/cache/home
  /.venv/bin/python "${eval_args[@]}"
)

export RUN_RECORD RUN_DIR CHECKPOINT_PATH="${checkpoint_path}" POLARIS_DIR OPENPI_DIR
export EXPECTED_POLARIS_COMMIT CHECKPOINT_URI CHECKPOINT_MANIFEST POLARIS_PYXIS_IMAGE
export POLARIS_DATA_DIR CONTROLLER_COMPLETION EXPECTED_CONTROLLER_COMPLETION_SHA256
export ALL_SIX_CONTROLLER_COMPLETION EXPECTED_ALL_SIX_COMPLETION_SHA256
export EXPECTED_ALL_SIX_PROFILE MODEL_RUNTIME_CONTRACT
export PORT_REQUESTED="${REQUESTED_PORT}" PORT_ACTUAL="${ACTUAL_PORT}" SERVER_PID
export BOUND_PORT_FILE BOUND_PORT_TOKEN BOUND_PORT_FILE_SHA256 BOUND_PORT_FILE_IDENTITY
export HANDSHAKE_PATH
PYTHONPATH="${POLARIS_DIR}/src" /usr/bin/python3 - <<'PY'
import os
from pathlib import Path
from polaris.pi05_droid_native_eval_contract import publish_immutable_json

keys = (
    "RUN_DIR", "CHECKPOINT_PATH", "POLARIS_DIR", "OPENPI_DIR",
    "EXPECTED_POLARIS_COMMIT", "CHECKPOINT_URI", "CHECKPOINT_MANIFEST",
    "POLARIS_PYXIS_IMAGE", "POLARIS_DATA_DIR", "CONTROLLER_COMPLETION",
    "EXPECTED_CONTROLLER_COMPLETION_SHA256", "ALL_SIX_CONTROLLER_COMPLETION",
    "EXPECTED_ALL_SIX_COMPLETION_SHA256", "EXPECTED_ALL_SIX_PROFILE",
    "PORT_REQUESTED", "PORT_ACTUAL", "SERVER_PID", "BOUND_PORT_FILE",
    "BOUND_PORT_TOKEN", "BOUND_PORT_FILE_SHA256", "BOUND_PORT_FILE_IDENTITY",
    "HANDSHAKE_PATH", "MODEL_RUNTIME_CONTRACT",
)
publish_immutable_json(
    Path(os.environ["RUN_RECORD"]),
    {
        "schema_version": 1,
        "profile": "openpi_pi05_droid_native_jointvelocity_polaris_canary_v1",
        "job_id": int(os.environ["SLURM_JOB_ID"]),
        "fresh_attempt_no_resume": True,
        "task": "DROID-FoodBussing",
        "rollouts": 1,
        "values": {key: os.environ[key] for key in keys},
    },
)
PY

{
  printf '#!/usr/bin/env bash\nset -euo pipefail\n'
  printf 'env OPENPI_DATA_HOME=%q PYTHONPATH=%q JAX_PLATFORMS=cuda ' \
    "${OPENPI_DATA_HOME}" "${POLARIS_DIR}/src"
  printf '%q ' "${server_command[@]}"
  printf '\nPYTHONPATH=%q ' "${POLARIS_DIR}/src"
  printf '%q ' "${bound_port_command[@]}"
  printf '\nPYTHONPATH=%q ' "${POLARIS_DIR}/src"
  printf '%q ' "${handshake_command[@]}"
  printf '\n'
  printf '%q ' "${eval_command[@]}"
  printf '\n'
} > "${COMMANDS_FILE}"
chmod 0444 "${COMMANDS_FILE}"
sync -f "${COMMANDS_FILE}"
sync -f "${RUN_DIR}"

set +e
(cd "${POLARIS_DIR}" && "${eval_command[@]}") 2>&1 | tee "${EVAL_LOG}"
pipeline_codes=("${PIPESTATUS[@]}")
set -e
eval_code="${pipeline_codes[0]}"
tee_code="${pipeline_codes[1]}"
set +e
BOUND_PORT_VALIDATION_AFTER_EVAL="$(
  PYTHONPATH="${POLARIS_DIR}/src" "${bound_port_command[@]}"
)"
bound_port_after_eval_code=$?
set -e
stop_server

export SRUN_STATUS="${RUN_DIR}/srun-${SLURM_JOB_ID}.status.json" SRUN_EXIT_CODE="${eval_code}"
PYTHONPATH="${POLARIS_DIR}/src" /usr/bin/python3 - <<'PY'
import os
from pathlib import Path
from polaris.pi05_droid_native_eval_contract import publish_immutable_json
publish_immutable_json(
    Path(os.environ["SRUN_STATUS"]),
    {"job_id": int(os.environ["SLURM_JOB_ID"]), "srun_exit_code": int(os.environ["SRUN_EXIT_CODE"])},
)
PY
(( bound_port_after_eval_code == 0 )) \
  || die "Bound-port artifact failed post-evaluator validation"
[[ "${BOUND_PORT_VALIDATION_AFTER_EVAL}" == "${BOUND_PORT_VALIDATION}" ]] \
  || die "Bound-port artifact changed during evaluator execution"
(( eval_code == 0 )) || exit "${eval_code}"
(( tee_code == 0 )) || exit "${tee_code}"

FAILURE_STAGE=host_finalization
PYTHONPATH="${POLARIS_DIR}/src:${SCRIPT_DIR}" "${OPENPI_DIR}/.venv/bin/python" \
  "${SCRIPT_DIR}/finalize_pi05_droid_native_jointvelocity_eval.py" finalize \
  --job-id "${SLURM_JOB_ID}" \
  --run-dir "${RUN_DIR}" \
  --polaris-repo "${POLARIS_DIR}" \
  --expected-polaris-commit "${EXPECTED_POLARIS_COMMIT}" \
  --openpi-dir "${OPENPI_DIR}" \
  --container-image "${POLARIS_PYXIS_IMAGE}" \
  --data-dir "${POLARIS_DATA_DIR}" \
  --controller-completion "${CONTROLLER_COMPLETION}" \
  --expected-controller-completion-sha256 "${EXPECTED_CONTROLLER_COMPLETION_SHA256}" \
  --all-six-controller-completion "${ALL_SIX_CONTROLLER_COMPLETION}" \
  --expected-all-six-completion-sha256 "${EXPECTED_ALL_SIX_COMPLETION_SHA256}" \
  --expected-all-six-profile "${EXPECTED_ALL_SIX_PROFILE}"

echo "Official pi0.5-DROID native canary complete: ${RUN_DIR}"
