#!/usr/bin/env bash

# Launch the exact official pi0.5-DROID-Polaris server and one bounded task eval.
# This script runs inside one ordinary one-GPU Slurm allocation.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLARIS_DIR="${POLARIS_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
OPENPI_DIR="${OPENPI_DIR:-${POLARIS_DIR}/third_party/openpi}"
NFS_ROOT="${NFS_ROOT:-/lustre/fsw/portfolios/nvr/users/lzha}"
POLARIS_DATA_DIR="${POLARIS_DATA_DIR:-${NFS_ROOT}/data/PolaRiS-Hub}"
POLARIS_PYXIS_IMAGE="${POLARIS_PYXIS_IMAGE:-${NFS_ROOT}/cache/polaris/polaris-eval-cuda13-fd00a51.sqsh}"
POLARIS_VULKAN_ICD_PATH="${POLARIS_VULKAN_ICD_PATH:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${NFS_ROOT}/cache/openpi-polaris}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${NFS_ROOT}/results/polaris-pi05}"
POLARIS_CACHE_DIR="${POLARIS_CACHE_DIR:-${NFS_ROOT}/cache/polaris/runtime/${SLURM_JOB_ID:-manual}}"

CHECKPOINT_URI="${CHECKPOINT_URI:-gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris}"
EXPECTED_CHECKPOINT_URI="gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_droid_jointpos_polaris}"
EXPECTED_OPENPI_COMMIT="${EXPECTED_OPENPI_COMMIT:-bd70b8f4011e85b3f3b0f039f12113f78718e7bf}"
EXPECTED_NORM_SHA256="${EXPECTED_NORM_SHA256:-57ce9956f9e07d65f8a8205aabec72d436a2c8927f53edb40c7a77b14a5a90c7}"
EXPECTED_PYXIS_SHA256="${EXPECTED_PYXIS_SHA256:-ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a}"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-7abd0c2294d442d429a77655783232206b2b30d95c508d435503135a5523a11c}"
CHECKPOINT_MANIFEST="${CHECKPOINT_MANIFEST:-${SCRIPT_DIR}/pi05_droid_jointpos_polaris_gcs_manifest.tsv}"
EXPECTED_ACTION_HORIZON="${EXPECTED_ACTION_HORIZON:-15}"
EXPECTED_ACTION_DIM="${EXPECTED_ACTION_DIM:-8}"
OPEN_LOOP_HORIZON="${OPEN_LOOP_HORIZON:-8}"
POLARIS_ENVIRONMENT="${POLARIS_ENVIRONMENT:-DROID-FoodBussing}"
ROLLOUTS="${ROLLOUTS:-1}"
RUN_NAMESPACE="${RUN_NAMESPACE:-pi05-polaris-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_LABEL="${RUN_LABEL:-pi05-polaris}"
PORT="${PORT:-$((20000 + ${SLURM_JOB_ID:-1} % 20000))}"
SERVER_START_TIMEOUT_SECS="${SERVER_START_TIMEOUT_SECS:-2400}"
DRY_RUN="${DRY_RUN:-0}"
RESUME_FROM_TASK_DIR="${RESUME_FROM_TASK_DIR:-}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ -n "${SLURM_JOB_ID:-}" || "${DRY_RUN}" == 1 ]] \
  || die "A Slurm allocation is required unless DRY_RUN=1"
: "${EXPECTED_POLARIS_COMMIT:?Set EXPECTED_POLARIS_COMMIT to the immutable launch commit}"
[[ "${ROLLOUTS}" =~ ^[1-9][0-9]*$ ]] || die "ROLLOUTS must be positive"
if [[ ! "${PORT}" =~ ^[1-9][0-9]*$ ]] || (( PORT > 65535 )); then
  die "Invalid PORT=${PORT}"
fi
[[ "${POLICY_CONFIG}" == pi05_droid_jointpos_polaris ]] || die "Unexpected policy config: ${POLICY_CONFIG}"
[[ "${CHECKPOINT_URI}" == "${EXPECTED_CHECKPOINT_URI}" ]] \
  || die "Unexpected checkpoint URI: ${CHECKPOINT_URI}"
[[ -x "${OPENPI_DIR}/.venv/bin/python" ]] || die "Run setup_pi05_droid_jointpos_polaris.sh first"
[[ "$(git -C "${OPENPI_DIR}" rev-parse HEAD)" == "${EXPECTED_OPENPI_COMMIT}" ]] \
  || die "OpenPI is not at ${EXPECTED_OPENPI_COMMIT}"
[[ -d "${POLARIS_DATA_DIR}" ]] || die "Missing PolaRiS data: ${POLARIS_DATA_DIR}"
[[ -f "${POLARIS_PYXIS_IMAGE}" ]] || die "Missing Pyxis image: ${POLARIS_PYXIS_IMAGE}"
[[ -f "${POLARIS_VULKAN_ICD_PATH}" ]] || die "Missing Vulkan ICD: ${POLARIS_VULKAN_ICD_PATH}"
[[ -f "${CHECKPOINT_MANIFEST}" ]] || die "Missing checkpoint manifest: ${CHECKPOINT_MANIFEST}"

case "${POLARIS_ENVIRONMENT}" in
  DROID-BlockStackKitchen)
    EXPECTED_PROMPT='Place and stack the blocks on top of the green tray'
    ;;
  DROID-FoodBussing)
    EXPECTED_PROMPT='Put all the foods in the bowl'
    ;;
  DROID-PanClean)
    EXPECTED_PROMPT='Use the yellow sponge to scrub the blue handle frying pan'
    ;;
  DROID-MoveLatteCup)
    EXPECTED_PROMPT='put the latte art cup on top of the cutting board'
    ;;
  DROID-OrganizeTools)
    EXPECTED_PROMPT='put the scissor into the large container'
    ;;
  DROID-TapeIntoContainer)
    EXPECTED_PROMPT='put the tape into the container'
    ;;
  *)
    die "Unsupported PolaRiS task: ${POLARIS_ENVIRONMENT}"
    ;;
esac

POLARIS_COMMIT="$(git -C "${POLARIS_DIR}" rev-parse HEAD)"
OPENPI_COMMIT="$(git -C "${OPENPI_DIR}" rev-parse HEAD)"
[[ "${POLARIS_COMMIT}" == "${EXPECTED_POLARIS_COMMIT}" ]] \
  || die "PolaRiS commit ${POLARIS_COMMIT} does not match ${EXPECTED_POLARIS_COMMIT}"
git -C "${POLARIS_DIR}" diff-index --quiet HEAD -- \
  || die "PolaRiS has tracked modifications; launch only from the committed revision"
RUN_NAME="${RUN_NAME:-${RUN_NAMESPACE}_${RUN_LABEL}_${POLARIS_ENVIRONMENT}_${SLURM_JOB_ID:-dryrun}}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_NAMESPACE}/${RUN_NAME}}"
TASK_DIR="${RUN_DIR}/${POLARIS_ENVIRONMENT}"
SERVER_LOG="${RUN_DIR}/policy_server.log"
EVAL_LOG="${TASK_DIR}/eval.log"
TRACE_PATH="${TASK_DIR}/policy_traces.jsonl"
TRACE_SUMMARY="${TASK_DIR}/policy_trace_summary.json"
COMMANDS_FILE="${RUN_DIR}/commands.sh"
METADATA_FILE="${RUN_DIR}/run_metadata.env"
SERVER_PID=""

mkdir -p "${TASK_DIR}" "${POLARIS_CACHE_DIR}"
if [[ -n "${RESUME_FROM_TASK_DIR}" ]]; then
  [[ ! -e "${TASK_DIR}/eval_results.csv" && ! -e "${TRACE_PATH}" ]] \
    || die "Resume destination already contains metrics or traces"
  python3 "${SCRIPT_DIR}/prepare_pi05_resume.py" \
    "${RESUME_FROM_TASK_DIR}" "${TASK_DIR}" --expected-rollouts "${ROLLOUTS}" \
    --output "${RUN_DIR}/resume_manifest.json"
  python3 "${SCRIPT_DIR}/validate_pi05_trace.py" \
    "${TRACE_PATH}" --metrics-csv "${TASK_DIR}/eval_results.csv" \
    --expected-prompt "${EXPECTED_PROMPT}" \
    --output "${RUN_DIR}/resume_trace_summary.json"
else
  [[ ! -e "${TASK_DIR}/eval_results.csv" && ! -e "${TRACE_PATH}" ]] \
    || die "Run directory already contains metrics or traces; use a new job/run directory"
fi
rm -f "${RUN_DIR}/SUCCESS" "${RUN_DIR}/FAILED"
printf 'started_at=%s\n' "$(date -Iseconds)" > "${RUN_DIR}/RUNNING"

stop_server() {
  [[ -n "${SERVER_PID}" ]] || return 0
  kill -TERM -- "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null || true
  for _ in {1..20}; do
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
  rm -f "${RUN_DIR}/RUNNING"
  if (( code == 0 )); then
    printf 'completed_at=%s\n' "$(date -Iseconds)" > "${RUN_DIR}/SUCCESS"
  else
    printf 'failed_at=%s\nexit_code=%s\n' "$(date -Iseconds)" "${code}" > "${RUN_DIR}/FAILED"
  fi
  printf 'SCRIPT_EXIT_CODE=%s\n' "${code}" >> "${METADATA_FILE}"
  exit "${code}"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

OPENPI_PYTHONPATH="${OPENPI_DIR}/src:${OPENPI_DIR}/packages/openpi-client/src"
checkpoint_path="$(
  OPENPI_DATA_HOME="${OPENPI_DATA_HOME}" PYTHONPATH="${OPENPI_PYTHONPATH}" \
    "${OPENPI_DIR}/.venv/bin/python" - "${CHECKPOINT_URI}" <<'PY'
import sys
from openpi.shared.download import maybe_download
print(maybe_download(sys.argv[1]))
PY
)"
norm_stats="${checkpoint_path}/assets/droid/norm_stats.json"
[[ -s "${norm_stats}" ]] || die "Missing checkpoint-local norm stats: ${norm_stats}"
actual_norm_sha256="$(sha256sum "${norm_stats}" | awk '{print $1}')"
[[ "${actual_norm_sha256}" == "${EXPECTED_NORM_SHA256}" ]] \
  || die "Norm SHA-256 mismatch: ${actual_norm_sha256}"
actual_manifest_sha256="$(sha256sum "${CHECKPOINT_MANIFEST}" | awk '{print $1}')"
[[ "${actual_manifest_sha256}" == "${EXPECTED_MANIFEST_SHA256}" ]] \
  || die "Checkpoint manifest SHA-256 mismatch: ${actual_manifest_sha256}"
"${OPENPI_DIR}/.venv/bin/python" "${SCRIPT_DIR}/verify_pi05_checkpoint.py" \
  "${checkpoint_path}" "${CHECKPOINT_MANIFEST}" --full-md5 \
  --output "${RUN_DIR}/checkpoint_verification.json"
PYXIS_IMAGE_SHA256="$(sha256sum "${POLARIS_PYXIS_IMAGE}" | awk '{print $1}')"
[[ "${PYXIS_IMAGE_SHA256}" == "${EXPECTED_PYXIS_SHA256}" ]] \
  || die "Pyxis image SHA-256 mismatch: ${PYXIS_IMAGE_SHA256}"

{
  printf 'RUN_START=%q\n' "$(date -Iseconds)"
  printf 'HOST=%q\n' "$(hostname)"
  printf 'SLURM_JOB_ID=%q\n' "${SLURM_JOB_ID:-dryrun}"
  printf 'RUN_NAMESPACE=%q\n' "${RUN_NAMESPACE}"
  printf 'RUN_DIR=%q\n' "${RUN_DIR}"
  printf 'POLARIS_DIR=%q\n' "${POLARIS_DIR}"
  printf 'POLARIS_COMMIT=%q\n' "${POLARIS_COMMIT}"
  printf 'OPENPI_DIR=%q\n' "${OPENPI_DIR}"
  printf 'OPENPI_COMMIT=%q\n' "${OPENPI_COMMIT}"
  printf 'CHECKPOINT_URI=%q\n' "${CHECKPOINT_URI}"
  printf 'CHECKPOINT_PATH=%q\n' "${checkpoint_path}"
  printf 'POLICY_CONFIG=%q\n' "${POLICY_CONFIG}"
  printf 'NORM_STATS_SHA256=%q\n' "${actual_norm_sha256}"
  printf 'CHECKPOINT_MANIFEST_SHA256=%q\n' "${actual_manifest_sha256}"
  printf 'POLARIS_PYXIS_IMAGE=%q\n' "${POLARIS_PYXIS_IMAGE}"
  printf 'POLARIS_PYXIS_IMAGE_SHA256=%q\n' "${PYXIS_IMAGE_SHA256}"
  printf 'POLARIS_ENVIRONMENT=%q\n' "${POLARIS_ENVIRONMENT}"
  printf 'EXPECTED_PROMPT=%q\n' "${EXPECTED_PROMPT}"
  printf 'RESUME_FROM_TASK_DIR=%q\n' "${RESUME_FROM_TASK_DIR}"
  printf 'ROLLOUTS=%q\n' "${ROLLOUTS}"
  printf 'CONTROL_MODE=joint-position\n'
  printf 'STATE_CONTRACT=7_joint_radians_plus_closed_positive_gripper\n'
  printf 'ACTION_CONTRACT=15x8_absolute_joint_targets_plus_closed_positive_gripper\n'
  printf 'OPEN_LOOP_HORIZON=%q\n' "${OPEN_LOOP_HORIZON}"
  printf 'CONTROL_FREQUENCY_HZ=15\n'
  printf 'WRIST_ROTATION_DEGREES=0\n'
  printf 'MODEL_IMAGE_SLOTS=base_0_rgb,left_wrist_0_rgb,right_wrist_0_rgb_masked\n'
} | tee "${METADATA_FILE}"

git -C "${POLARIS_DIR}" status --short --branch > "${RUN_DIR}/polaris_git_status.txt"
git -C "${POLARIS_DIR}" submodule status --recursive > "${RUN_DIR}/polaris_submodules.txt" 2>&1 || true
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total --format=csv,noheader \
  > "${RUN_DIR}/gpu_environment.csv"

server_command=(
  "${OPENPI_DIR}/.venv/bin/python"
  "${OPENPI_DIR}/scripts/serve_policy.py"
  "--port=${PORT}"
  policy:checkpoint
  "--policy.config=${POLICY_CONFIG}"
  "--policy.dir=${checkpoint_path}"
)

eval_args=(
  scripts/eval.py
  --environment "${POLARIS_ENVIRONMENT}"
  --control-mode joint-position
  --policy.client DroidJointPos
  --policy.host 127.0.0.1
  --policy.port "${PORT}"
  --policy.open-loop-horizon "${OPEN_LOOP_HORIZON}"
  --policy.state-type joint_position
  --policy.expected-action-horizon "${EXPECTED_ACTION_HORIZON}"
  --policy.expected-action-dim "${EXPECTED_ACTION_DIM}"
  --policy.trace-path "${TRACE_PATH}"
  --run-folder "${TASK_DIR}"
  --rollouts "${ROLLOUTS}"
  --headless
)

pyxis_mounts="/dev/shm:/dev/shm,${POLARIS_DIR}:${POLARIS_DIR}:ro,${POLARIS_DATA_DIR}:${POLARIS_DATA_DIR}:ro,${RUN_DIR}:${RUN_DIR}:rw,${POLARIS_CACHE_DIR}:/cache:rw,${POLARIS_VULKAN_ICD_PATH}:/etc/vulkan/icd.d/nvidia_icd.json:ro"
eval_command=(
  srun --ntasks=1 "--cpus-per-task=${SLURM_CPUS_PER_TASK:-16}"
  "--container-image=${POLARIS_PYXIS_IMAGE}"
  "--container-mounts=${pyxis_mounts}"
  "--container-workdir=${POLARIS_DIR}"
  --no-container-entrypoint --no-container-mount-home --container-remap-root --container-writable
  "--container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES" --export=ALL
  /usr/bin/env
  VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
  ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y OMNI_KIT_ALLOW_ROOT=1
  PYTHONUNBUFFERED=1
  "PYTHONPATH=${POLARIS_DIR}/src:${POLARIS_DIR}/third_party/openpi/packages/openpi-client/src"
  "POLARIS_DATA_PATH=${POLARIS_DATA_DIR}"
  XDG_CACHE_HOME=/cache HF_HOME=/cache/huggingface HOME=/cache/home
  /.venv/bin/python "${eval_args[@]}"
)

{
  printf '#!/usr/bin/env bash\nset -euo pipefail\n\n'
  printf 'cd %q\n' "${OPENPI_DIR}"
  printf 'env OPENPI_DATA_HOME=%q PYTHONPATH=%q JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 XLA_PYTHON_CLIENT_PREALLOCATE=false ' \
    "${OPENPI_DATA_HOME}" "${OPENPI_PYTHONPATH}"
  printf '%q ' "${server_command[@]}"
  printf '\ncd %q\n' "${POLARIS_DIR}"
  printf '%q ' "${eval_command[@]}"
  printf '\n'
} > "${COMMANDS_FILE}"
chmod +x "${COMMANDS_FILE}"

if [[ "${DRY_RUN}" == 1 ]]; then
  cat "${COMMANDS_FILE}"
  exit 0
fi

if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
  die "Port ${PORT} is already in use"
fi

echo "[$(date -Iseconds)] Starting official pi0.5 policy server"
(
  cd "${OPENPI_DIR}"
  exec setsid env \
    OPENPI_DATA_HOME="${OPENPI_DATA_HOME}" \
    PYTHONPATH="${OPENPI_PYTHONPATH}" \
    JAX_PLATFORMS=cuda \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    PYTHONUNBUFFERED=1 \
    "${server_command[@]}"
) > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

elapsed=0
until curl --fail --silent --show-error --max-time 2 "http://127.0.0.1:${PORT}/healthz" >/dev/null; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    wait "${SERVER_PID}" || server_code=$?
    tail -n 160 "${SERVER_LOG}" >&2 || true
    exit "${server_code:-1}"
  fi
  sleep 2
  elapsed=$((elapsed + 2))
  if (( elapsed >= SERVER_START_TIMEOUT_SECS )); then
    tail -n 160 "${SERVER_LOG}" >&2 || true
    die "Policy server did not become healthy within ${SERVER_START_TIMEOUT_SECS}s"
  fi
done
echo "[$(date -Iseconds)] Policy server healthy after ${elapsed}s"

set +e
(
  cd "${POLARIS_DIR}"
  "${eval_command[@]}"
) 2>&1 | tee "${EVAL_LOG}"
pipeline_codes=("${PIPESTATUS[@]}")
set -e
eval_code="${pipeline_codes[0]}"
tee_code="${pipeline_codes[1]}"
(( eval_code == 0 )) || exit "${eval_code}"
(( tee_code == 0 )) || exit "${tee_code}"

grep -Fq 'POLARIS_PI05_DROID_CONTRACT=' "${EVAL_LOG}" \
  || die "Missing pi0.5 client contract marker"
grep -Fq '"expected_action_horizon": 15' "${EVAL_LOG}" \
  || die "Client did not enforce a 15-step response"
grep -Fq '"expected_action_dim": 8' "${EVAL_LOG}" \
  || die "Client did not enforce an 8-D response"
grep -Fq '"wrist_rotation_degrees": 0' "${EVAL_LOG}" \
  || die "Client did not preserve the unrotated wrist contract"

csv_path="${TASK_DIR}/eval_results.csv"
[[ -s "${csv_path}" ]] || die "Missing eval metrics: ${csv_path}"
python3 "${POLARIS_DIR}/scripts/polaris/validate_pi05_trace.py" \
  "${TRACE_PATH}" --metrics-csv "${csv_path}" --expected-prompt "${EXPECTED_PROMPT}" \
  --output "${TRACE_SUMMARY}"
csv_rows="$(awk 'NR > 1 && NF {count += 1} END {print count + 0}' "${csv_path}")"
(( csv_rows == ROLLOUTS )) || die "Expected ${ROLLOUTS} CSV rows, got ${csv_rows}"
video_count="$(find "${TASK_DIR}" -maxdepth 1 -type f -name 'episode_*.mp4' -size +0c | wc -l)"
(( video_count == ROLLOUTS )) || die "Expected ${ROLLOUTS} videos, got ${video_count}"

printf 'completed_at=%s\n' "$(date -Iseconds)" > "${TASK_DIR}/SUCCESS"
echo "Evaluation complete: ${RUN_DIR}"
