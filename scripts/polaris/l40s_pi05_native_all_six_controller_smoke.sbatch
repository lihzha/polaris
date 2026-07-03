#!/usr/bin/env bash
#SBATCH --account=nvr_lpr_rvp
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --no-requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:25:00
#SBATCH --job-name=pi05_all6_ctrl
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/lzha/slurm_logs/polaris-pi05-native-all-six/%x-%j.out

set -Eeuo pipefail

NFS_ROOT="${NFS_ROOT:-/lustre/fsw/portfolios/nvr/users/lzha}"
: "${POLARIS_DIR:?Set the frozen standalone PolaRiS checkout}"
: "${EXPECTED_POLARIS_COMMIT:?Set the exact PolaRiS commit}"
: "${RUN_DIR:?Set one fresh all-six smoke directory}"
POLARIS_PYXIS_IMAGE="${POLARIS_PYXIS_IMAGE:-${NFS_ROOT}/cache/polaris/polaris-eval-cuda13-fd00a51.sqsh}"
POLARIS_DATA_DIR="${POLARIS_DATA_DIR:-${NFS_ROOT}/data/PolaRiS-Hub}"
POLARIS_VULKAN_ICD_PATH="${POLARIS_VULKAN_ICD_PATH:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
EXPECTED_HUB_REVISION=8c7e4103e266ef83d8b1ad2e9a63116edd5f155b
CACHE_DIR="${NFS_ROOT}/cache/polaris/runtime/pi05-native-all-six-${SLURM_JOB_ID}"
JOB_SCRIPT="${POLARIS_DIR}/scripts/polaris/l40s_pi05_native_all_six_controller_smoke.sbatch"
SAVED_JOB_SCRIPT="${RUN_DIR}/job-${SLURM_JOB_ID}.sbatch"
GPU_INVENTORY="${RUN_DIR}/gpu-${SLURM_JOB_ID}.json"
SRUN_STATUS="${RUN_DIR}/srun-${SLURM_JOB_ID}.status.json"
SMOKE="${RUN_DIR}/native-all-six-smoke-${SLURM_JOB_ID}.json"
COMPLETION="${RUN_DIR}/native-all-six-smoke-${SLURM_JOB_ID}.completion.json"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ -d "${RUN_DIR}" && ! -L "${RUN_DIR}" ]] || die "missing regular RUN_DIR"
[[ -d "${POLARIS_DIR}/.git" && ! -L "${POLARIS_DIR}/.git" ]] \
  || die "PolaRiS source must have an in-root .git"
[[ "$(git -C "${POLARIS_DIR}" rev-parse --abbrev-ref HEAD)" == HEAD ]] \
  || die "PolaRiS source must use detached HEAD"
[[ "$(git -C "${POLARIS_DIR}" rev-parse HEAD)" == "${EXPECTED_POLARIS_COMMIT}" ]] \
  || die "PolaRiS source commit mismatch"
[[ -z "$(git -C "${POLARIS_DIR}" status --porcelain=v1 --untracked-files=all)" ]] \
  || die "PolaRiS source is dirty"
[[ -x "${POLARIS_DIR}/third_party/openpi/.venv/bin/python" ]] \
  || die "missing exact checkout-local host Python"
[[ -f "${POLARIS_PYXIS_IMAGE}" && ! -L "${POLARIS_PYXIS_IMAGE}" ]] \
  || die "missing pinned container"
[[ "$(sha256sum "${POLARIS_PYXIS_IMAGE}" | awk '{print $1}')" == \
  ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a ]] \
  || die "container SHA-256 mismatch"
[[ -d "${POLARIS_DATA_DIR}" && ! -L "${POLARIS_DATA_DIR}" ]] \
  || die "missing regular PolaRiS data root"
[[ -f "${POLARIS_VULKAN_ICD_PATH}" ]] || die "missing Vulkan ICD"

declare -A expected_asset_sha=(
  [food_bussing/initial_conditions.json]=40091faee14f692350220871d30705294f21f17ae3d2974cd3c09a34d560f5de
  [food_bussing/scene.usda]=82cd641e422935b394ce7ea7b6be55214c9952a2544000222921e544c409b489
  [nvidia_droid/noninstanceable.usd]=d8379925b103963dbf3e7c85bcc4ae101b81b7c1d7dabe7d2e964f41d069ec44
)
declare -A expected_metadata_sha=(
  [food_bussing/initial_conditions.json]=852dd0345afb7e4d0c7526b5c327086b5132c40624ed97ff6942962126e90534
  [food_bussing/scene.usda]=accd9b67e90e510eb4ed44a789b9169df058e71ce557164f960de2d62a840e63
  [nvidia_droid/noninstanceable.usd]=208e0f85fc16fa32ffeca972aea0fd1b33b0c6c2a582e89ff3877823291a7754
)
for relative_path in "${!expected_asset_sha[@]}"; do
  asset="${POLARIS_DATA_DIR}/${relative_path}"
  metadata="${POLARIS_DATA_DIR}/.cache/huggingface/download/${relative_path}.metadata"
  [[ -f "${asset}" && ! -L "${asset}" && -f "${metadata}" && ! -L "${metadata}" ]] \
    || die "missing regular pinned asset or metadata: ${relative_path}"
  [[ "$(sha256sum "${asset}" | awk '{print $1}')" == \
    "${expected_asset_sha[${relative_path}]}" ]] \
    || die "asset SHA-256 mismatch: ${relative_path}"
  [[ "$(sha256sum "${metadata}" | awk '{print $1}')" == \
    "${expected_metadata_sha[${relative_path}]}" ]] \
    || die "asset metadata SHA-256 mismatch: ${relative_path}"
  [[ "$(head -n 1 "${metadata}")" == "${EXPECTED_HUB_REVISION}" ]] \
    || die "asset Hub revision mismatch: ${relative_path}"
done

for output in "${SAVED_JOB_SCRIPT}" "${GPU_INVENTORY}" "${SRUN_STATUS}" \
  "${SMOKE}" "${COMPLETION}"; do
  [[ ! -e "${output}" && ! -L "${output}" ]] || die "refusing existing output ${output}"
done

scontrol write batch_script "${SLURM_JOB_ID}" "${SAVED_JOB_SCRIPT}"
chmod 0444 "${SAVED_JOB_SCRIPT}"
sync -f "${SAVED_JOB_SCRIPT}"
[[ "$(sha256sum "${SAVED_JOB_SCRIPT}" | awk '{print $1}')" == \
  "$(sha256sum "${JOB_SCRIPT}" | awk '{print $1}')" ]] \
  || die "saved Slurm script differs from committed source"

mapfile -t gpu_rows < <(
  nvidia-smi --query-gpu=uuid,name,driver_version --format=csv,noheader,nounits \
    | sed '/^[[:space:]]*$/d'
)
(( ${#gpu_rows[@]} == 1 )) || die "expected exactly one allocated GPU"
IFS=',' read -r gpu_uuid gpu_name driver_version <<<"${gpu_rows[0]}"
gpu_uuid="$(xargs <<<"${gpu_uuid}")"
gpu_name="$(xargs <<<"${gpu_name}")"
driver_version="$(xargs <<<"${driver_version}")"
[[ "${gpu_uuid}" == GPU-* && "${gpu_name}" == "NVIDIA L40S" ]] \
  || die "allocation is not one L40S"
export GPU_INVENTORY gpu_uuid gpu_name driver_version
PYTHONPATH="${POLARIS_DIR}/src" /usr/bin/python3 - <<'PY'
import os
from pathlib import Path
from polaris.pi05_droid_native_eval_contract import publish_immutable_json
publish_immutable_json(
    Path(os.environ["GPU_INVENTORY"]),
    {
        "schema_version": 1,
        "job_id": int(os.environ["SLURM_JOB_ID"]),
        "gpus": [{
            "uuid": os.environ["gpu_uuid"],
            "name": os.environ["gpu_name"],
            "driver_version": os.environ["driver_version"],
        }],
    },
)
PY

mkdir "${CACHE_DIR}"
mkdir "${CACHE_DIR}/home"
cleanup() {
  rm -rf --one-file-system "${CACHE_DIR}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export NVIDIA_VISIBLE_DEVICES="${gpu_uuid}"
export NVIDIA_DRIVER_CAPABILITIES=all
export PYTHONUNBUFFERED=1
export SLURM_EXPORT_ENV=ALL
mounts="/dev/shm:/dev/shm,${POLARIS_DIR}:${POLARIS_DIR}:ro,${POLARIS_DATA_DIR}:${POLARIS_DATA_DIR}:ro,${RUN_DIR}:${RUN_DIR}:rw,${CACHE_DIR}:/cache:rw,${POLARIS_VULKAN_ICD_PATH}:/etc/vulkan/icd.d/nvidia_icd.json:ro"

set +e
srun --ntasks=1 "--cpus-per-task=${SLURM_CPUS_PER_TASK:-16}" \
  "--container-image=${POLARIS_PYXIS_IMAGE}" \
  "--container-mounts=${mounts}" \
  "--container-workdir=${POLARIS_DIR}" \
  --no-container-entrypoint --no-container-mount-home --container-remap-root \
  --container-writable \
  "--container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES" \
  --export=ALL \
  /usr/bin/env \
  VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json \
  ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
  OMNI_KIT_ALLOW_ROOT=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
  "PYTHONPATH=${POLARIS_DIR}/src" \
  "POLARIS_DATA_PATH=${POLARIS_DATA_DIR}" \
  XDG_CACHE_HOME=/cache HF_HOME=/cache/huggingface HOME=/cache/home \
  /.venv/bin/python \
  scripts/smoke_pi05_native_all_six_controller.py \
  --environment DROID-FoodBussing \
  --output-json "${SMOKE}" \
  --headless
srun_code=$?
set -e

export SRUN_STATUS srun_code
PYTHONPATH="${POLARIS_DIR}/src" /usr/bin/python3 - <<'PY'
import os
from pathlib import Path
from polaris.pi05_droid_native_eval_contract import publish_immutable_json
publish_immutable_json(
    Path(os.environ["SRUN_STATUS"]),
    {"job_id": int(os.environ["SLURM_JOB_ID"]), "srun_exit_code": int(os.environ["srun_code"])},
)
PY
(( srun_code == 0 )) || exit "${srun_code}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${POLARIS_DIR}/src" \
  "${POLARIS_DIR}/third_party/openpi/.venv/bin/python" \
  "${POLARIS_DIR}/scripts/polaris/finalize_pi05_native_all_six_controller_smoke.py" \
  --job-id "${SLURM_JOB_ID}" \
  --polaris-repo "${POLARIS_DIR}" \
  --expected-polaris-commit "${EXPECTED_POLARIS_COMMIT}" \
  --smoke "${SMOKE}" \
  --saved-wrapper "${SAVED_JOB_SCRIPT}" \
  --srun-status "${SRUN_STATUS}" \
  --gpu-inventory "${GPU_INVENTORY}" \
  --container-image "${POLARIS_PYXIS_IMAGE}" \
  --data-dir "${POLARIS_DATA_DIR}" \
  --output "${COMPLETION}"

echo "Native all-six controller smoke complete: ${COMPLETION}"
