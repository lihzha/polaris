#!/usr/bin/env bash
#SBATCH --account=nvr_lpr_rvp
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --job-name=pi05_graphics_probe

set -Eeuo pipefail

: "${POLARIS_DIR:?Set POLARIS_DIR to the frozen committed source}"
: "${EXPECTED_POLARIS_COMMIT:?Set EXPECTED_POLARIS_COMMIT}"
: "${RUN_DIR:?Set a fresh absolute RUN_DIR}"

NFS_ROOT=/lustre/fsw/portfolios/nvr/users/lzha
POLARIS_PYXIS_IMAGE="${NFS_ROOT}/cache/polaris/polaris-eval-cuda13-fd00a51.sqsh"
EXPECTED_PYXIS_SHA256=ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a
POLARIS_VULKAN_ICD_PATH=/usr/share/vulkan/icd.d/nvidia_icd.json
EXPECTED_VULKAN_ICD_SHA256=7bdb6f27d35b66fc848df6f94b8773bba30ea3a7f06f114100d14154a235a34b
OUTPUT_PATH="${RUN_DIR}/graphics_environment_probe.json"

[[ "$(git -C "${POLARIS_DIR}" rev-parse HEAD)" == "${EXPECTED_POLARIS_COMMIT}" ]]
[[ -z "$(git -C "${POLARIS_DIR}" status --porcelain=v1 --untracked-files=all)" ]]
[[ ! -e "${RUN_DIR}" && ! -L "${RUN_DIR}" ]]
mkdir -p "${RUN_DIR}/cache/home"

actual_image_sha256="$(sha256sum "${POLARIS_PYXIS_IMAGE}" | awk '{print $1}')"
[[ "${actual_image_sha256}" == "${EXPECTED_PYXIS_SHA256}" ]]
actual_icd_sha256="$(sha256sum "${POLARIS_VULKAN_ICD_PATH}" | awk '{print $1}')"
[[ "${actual_icd_sha256}" == "${EXPECTED_VULKAN_ICD_SHA256}" ]]
mapfile -t gpu_rows < <(
  nvidia-smi --query-gpu=uuid,name,driver_version --format=csv,noheader,nounits \
    | sed '/^[[:space:]]*$/d'
)
(( ${#gpu_rows[@]} == 1 ))
IFS=, read -r gpu_uuid gpu_name driver_version <<<"${gpu_rows[0]}"
gpu_uuid="$(xargs <<<"${gpu_uuid}")"
gpu_name="$(xargs <<<"${gpu_name}")"
driver_version="$(xargs <<<"${driver_version}")"
[[ "${gpu_uuid}" =~ ^GPU-[0-9a-fA-F-]+$ ]]
[[ "${gpu_name}" == "NVIDIA L40S" ]]
[[ "${driver_version}" == "580.105.08" ]]

{
  printf 'job_id=%q\n' "${SLURM_JOB_ID}"
  printf 'host=%q\n' "$(hostname)"
  printf 'polaris_dir=%q\n' "${POLARIS_DIR}"
  printf 'polaris_commit=%q\n' "${EXPECTED_POLARIS_COMMIT}"
  printf 'pyxis_image=%q\n' "${POLARIS_PYXIS_IMAGE}"
  printf 'pyxis_sha256=%q\n' "${actual_image_sha256}"
  printf 'vulkan_icd_sha256=%q\n' "${actual_icd_sha256}"
  printf 'gpu_uuid=%q\n' "${gpu_uuid}"
  printf 'gpu_name=%q\n' "${gpu_name}"
  printf 'driver_version=%q\n' "${driver_version}"
  printf 'output=%q\n' "${OUTPUT_PATH}"
} >"${RUN_DIR}/run_record.env"
chmod 0444 "${RUN_DIR}/run_record.env"

mounts="/dev/shm:/dev/shm,${POLARIS_DIR}:${POLARIS_DIR}:ro,${RUN_DIR}:${RUN_DIR}:rw,${POLARIS_VULKAN_ICD_PATH}:/etc/vulkan/icd.d/nvidia_icd.json:ro"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --container-image="${POLARIS_PYXIS_IMAGE}" \
  --container-mounts="${mounts}" \
  --container-workdir="${POLARIS_DIR}" \
  --no-container-entrypoint --no-container-mount-home --container-remap-root \
  --container-writable \
  --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES --export=ALL \
  /usr/bin/env -i \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  "NVIDIA_VISIBLE_DEVICES=${gpu_uuid}" NVIDIA_DRIVER_CAPABILITIES=all \
  VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json \
  ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y OMNI_KIT_ALLOW_ROOT=1 \
  PYTHONUNBUFFERED=1 "PYTHONPATH=${POLARIS_DIR}/src" \
  XDG_CACHE_HOME="${RUN_DIR}/cache" HOME="${RUN_DIR}/cache/home" \
  /.venv/bin/python scripts/polaris/probe_pi05_graphics_environment.py \
  --output "${OUTPUT_PATH}"

[[ -f "${OUTPUT_PATH}" && ! -L "${OUTPUT_PATH}" ]]
[[ "$(stat -c %a "${OUTPUT_PATH}")" == 444 ]]
python3 -m json.tool "${OUTPUT_PATH}" >/dev/null
sha256sum "${OUTPUT_PATH}" | tee "${RUN_DIR}/graphics_environment_probe.sha256"
chmod 0444 "${RUN_DIR}/graphics_environment_probe.sha256"
