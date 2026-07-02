#!/usr/bin/env bash
#SBATCH --account=nvr_lpr_rvp
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=00:45:00
#SBATCH --job-name=polaris_eef_replay
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/lzha/slurm_logs/polaris-lap3b/%x-%j.out

set -Eeuo pipefail

NFS_ROOT="${NFS_ROOT:-/lustre/fsw/portfolios/nvr/users/lzha}"
POLARIS_DIR="${POLARIS_DIR:?Set POLARIS_DIR to the exact deployed checkout}"
POLARIS_DATA_DIR="${POLARIS_DATA_DIR:-${NFS_ROOT}/data/PolaRiS-Hub}"
POLARIS_PYXIS_IMAGE="${POLARIS_PYXIS_IMAGE:-${NFS_ROOT}/cache/polaris/polaris-eval-cuda13-fd00a51.sqsh}"
POLARIS_CONTAINER_PYTHON="${POLARIS_CONTAINER_PYTHON:-/.venv/bin/python}"
POLARIS_VULKAN_ICD_PATH="${POLARIS_VULKAN_ICD_PATH:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
POLARIS_VULKAN_ICD_CONTAINER_PATH="${POLARIS_VULKAN_ICD_CONTAINER_PATH:-/etc/vulkan/icd.d/nvidia_icd.json}"
POLARIS_CACHE_DIR="${POLARIS_CACHE_DIR:-${NFS_ROOT}/cache/polaris/runtime/${SLURM_JOB_ID}}"

: "${EXPECTED_POLARIS_COMMIT:?Set EXPECTED_POLARIS_COMMIT}"
: "${EXPECTED_PYXIS_SHA256:?Set EXPECTED_PYXIS_SHA256}"
: "${POLARIS_ENVIRONMENT:?Set POLARIS_ENVIRONMENT}"
: "${REPLAY_TRACE_PATH:?Set REPLAY_TRACE_PATH}"
: "${REPLAY_EPISODE:?Set REPLAY_EPISODE}"
: "${INITIAL_CONDITIONS_FILE:?Set INITIAL_CONDITIONS_FILE}"
: "${RESULT_DIR:?Set RESULT_DIR}"

[[ "$(git -C "${POLARIS_DIR}" rev-parse HEAD)" == "${EXPECTED_POLARIS_COMMIT}" ]] || {
  echo "PolaRiS commit mismatch" >&2
  exit 2
}
[[ -z "$(git -C "${POLARIS_DIR}" status --porcelain)" ]] || {
  echo "PolaRiS checkout must be clean" >&2
  exit 2
}
[[ -f "${REPLAY_TRACE_PATH}" && -f "${INITIAL_CONDITIONS_FILE}" ]] || {
  echo "Replay trace or initial-condition fixture is missing" >&2
  exit 2
}
[[ -f "${POLARIS_PYXIS_IMAGE}" ]] || {
  echo "Pyxis image is missing: ${POLARIS_PYXIS_IMAGE}" >&2
  exit 2
}
read -r actual_pyxis_sha _ < "${POLARIS_PYXIS_IMAGE}.sha256"
[[ "${actual_pyxis_sha}" == "${EXPECTED_PYXIS_SHA256}" ]] || {
  echo "Pyxis image digest mismatch" >&2
  exit 2
}

mkdir -p "${RESULT_DIR}" "${POLARIS_CACHE_DIR}"
mapfile -t allocated_gpu_uuids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
(( ${#allocated_gpu_uuids[@]} == 1 )) || {
  echo "Expected one visible allocated GPU; found ${#allocated_gpu_uuids[@]}" >&2
  exit 2
}
export NVIDIA_VISIBLE_DEVICES="${allocated_gpu_uuids[0]}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"

{
  printf 'started_at=%q\n' "$(date -Iseconds)"
  printf 'job_id=%q\n' "${SLURM_JOB_ID}"
  printf 'host=%q\n' "$(hostname)"
  printf 'polaris_dir=%q\n' "${POLARIS_DIR}"
  printf 'polaris_commit=%q\n' "${EXPECTED_POLARIS_COMMIT}"
  printf 'pyxis_image=%q\n' "${POLARIS_PYXIS_IMAGE}"
  printf 'pyxis_sha256=%q\n' "${actual_pyxis_sha}"
  printf 'environment=%q\n' "${POLARIS_ENVIRONMENT}"
  printf 'replay_trace=%q\n' "${REPLAY_TRACE_PATH}"
  printf 'replay_episode=%q\n' "${REPLAY_EPISODE}"
  printf 'initial_conditions=%q\n' "${INITIAL_CONDITIONS_FILE}"
  printf 'result_dir=%q\n' "${RESULT_DIR}"
} > "${RESULT_DIR}/run_record.env"

mounts="/dev/shm:/dev/shm,${POLARIS_DIR}:${POLARIS_DIR}:ro,${POLARIS_DATA_DIR}:${POLARIS_DATA_DIR}:ro,${RESULT_DIR}:${RESULT_DIR}:rw,${POLARIS_CACHE_DIR}:/cache:rw"
if [[ -n "${POLARIS_VULKAN_ICD_PATH}" ]]; then
  mounts+=",${POLARIS_VULKAN_ICD_PATH}:${POLARIS_VULKAN_ICD_CONTAINER_PATH}:ro"
fi

command=(
  srun --ntasks=1 "--cpus-per-task=${SLURM_CPUS_PER_TASK:-16}"
  "--container-image=${POLARIS_PYXIS_IMAGE}"
  "--container-mounts=${mounts}"
  "--container-workdir=${POLARIS_DIR}"
  --no-container-entrypoint --no-container-mount-home
  --container-remap-root --container-writable
  --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES
  --export=ALL /usr/bin/env
)
if [[ -n "${POLARIS_VULKAN_ICD_PATH}" ]]; then
  command+=("VK_DRIVER_FILES=${POLARIS_VULKAN_ICD_CONTAINER_PATH}")
fi
command+=(
  ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y OMNI_KIT_ALLOW_ROOT=1
  PYTHONUNBUFFERED=1 "PYTHONPATH=${POLARIS_DIR}/src"
  "POLARIS_DATA_PATH=${POLARIS_DATA_DIR}"
  XDG_CACHE_HOME=/cache HF_HOME=/cache/huggingface HOME=/cache/home
  "${POLARIS_CONTAINER_PYTHON}" scripts/eval.py
  --environment "${POLARIS_ENVIRONMENT}"
  --control-mode eef-pose
  --policy.client ReplayEefPose
  --policy.eef-frame panda_link8
  --policy.replay-trace-path "${REPLAY_TRACE_PATH}"
  --policy.replay-episode "${REPLAY_EPISODE}"
  --policy.trace-path "${RESULT_DIR}/replay_observation_trace.jsonl"
  --initial-conditions-file "${INITIAL_CONDITIONS_FILE}"
  --run-folder "${RESULT_DIR}"
  --rollouts 1
  --headless
)

printf '%q ' "${command[@]}" > "${RESULT_DIR}/command.argv"
printf '\n' >> "${RESULT_DIR}/command.argv"

set +e
"${command[@]}" 2>&1 | tee "${RESULT_DIR}/eval.log"
exit_code=$?
set -e
for required_artifact in \
  "${RESULT_DIR}/eval_results.csv" \
  "${RESULT_DIR}/episode_0.mp4" \
  "${RESULT_DIR}/replay_observation_trace.jsonl"; do
  if (( exit_code == 0 )) && [[ ! -s "${required_artifact}" ]]; then
    echo "Missing replay artifact: ${required_artifact}" >&2
    exit_code=3
  fi
done
printf 'exit_code=%s\ncompleted_at=%q\n' "${exit_code}" "$(date -Iseconds)" >> "${RESULT_DIR}/run_record.env"
if (( exit_code != 0 )); then
  exit "${exit_code}"
fi

printf 'completed_at=%s\n' "$(date -Iseconds)" > "${RESULT_DIR}/SUCCESS"
