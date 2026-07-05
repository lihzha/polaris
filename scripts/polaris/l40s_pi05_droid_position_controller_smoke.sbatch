#!/usr/bin/env bash
#SBATCH --account=nvr_lpr_rvp
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --job-name=pi05_pos_smoke
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/lzha/slurm_logs/polaris-pi05-position/%x-%j.out

set -Eeuo pipefail

: "${POLARIS_DIR:?Set one frozen standalone PolaRiS checkout}"
: "${EXPECTED_POLARIS_COMMIT:?Set the exact PolaRiS commit}"
: "${RUN_DIR:?Set one fresh pre-created smoke directory}"
NFS_ROOT_REQUESTED="${NFS_ROOT:-/lustre/fsw/portfolios/nvr/users/lzha}"
POLARIS_DIR_REQUESTED="${POLARIS_DIR}"
RUN_DIR_REQUESTED="${RUN_DIR}"
EXPECTED_IMAGE_SHA256=ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a

die() { echo "ERROR: $*" >&2; exit 2; }
canonical_existing() {
  local field="$1" value="$2"
  [[ "${value}" == /* && "${value}" != //* ]] || die "${field} must be absolute"
  realpath -e -- "${value}" || die "cannot resolve existing ${field}"
}

[[ "${EXPECTED_POLARIS_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || die "commit must be full lowercase SHA"
NFS_ROOT="$(canonical_existing NFS_ROOT "${NFS_ROOT_REQUESTED}")"
POLARIS_DIR="$(canonical_existing POLARIS_DIR "${POLARIS_DIR_REQUESTED}")"
RUN_DIR="$(canonical_existing RUN_DIR "${RUN_DIR_REQUESTED}")"
OPENPI_DIR="$(canonical_existing OPENPI_DIR "${OPENPI_DIR:-${POLARIS_DIR}/third_party/openpi}")"
POLARIS_DATA_DIR="$(canonical_existing POLARIS_DATA_DIR "${POLARIS_DATA_DIR:-${NFS_ROOT}/data/PolaRiS-Hub}")"
IMAGE="$(canonical_existing IMAGE "${POLARIS_PYXIS_IMAGE:-${NFS_ROOT}/cache/polaris/polaris-eval-cuda13-fd00a51.sqsh}")"
VULKAN_ICD="$(canonical_existing VULKAN_ICD "${POLARIS_VULKAN_ICD_PATH:-/usr/share/vulkan/icd.d/nvidia_icd.json}")"
CACHE_ROOT="$(realpath -sm -- "${NFS_ROOT}/cache/polaris/runtime")"
CACHE_DIR="$(realpath -sm -- "${POLARIS_CACHE_DIR:-${CACHE_ROOT}/position-smoke-${SLURM_JOB_ID}}")"
RAW_SMOKE="${RUN_DIR}/position-controller-smoke-${SLURM_JOB_ID}.json"
ATTESTATION="${RUN_DIR}/position-controller-smoke-${SLURM_JOB_ID}.attestation.json"
SRUN_STATUS="${RUN_DIR}/srun-${SLURM_JOB_ID}.status.json"
GPU_INVENTORY="${RUN_DIR}/gpu-${SLURM_JOB_ID}.json"
SAVED_JOB_SCRIPT="${RUN_DIR}/job-${SLURM_JOB_ID}.sbatch"
JOB_SCRIPT="${POLARIS_DIR}/scripts/polaris/l40s_pi05_droid_position_controller_smoke.sbatch"
FINALIZER="${POLARIS_DIR}/scripts/polaris/finalize_pi05_droid_position_controller_smoke.py"
[[ -d "${RUN_DIR}" && ! -L "${RUN_DIR}" ]] || die "RUN_DIR must be one regular directory"
[[ ! -L "${POLARIS_DIR}" && -d "${POLARIS_DIR}/.git" && ! -L "${POLARIS_DIR}/.git" ]] \
  || die "standalone checkout required"
repo_root="$(realpath "${POLARIS_DIR}")"
[[ "$(git -C "${POLARIS_DIR}" rev-parse --show-toplevel)" == "${repo_root}" ]] \
  || die "POLARIS_DIR must name the exact Git root"
[[ "$(git -C "${POLARIS_DIR}" rev-parse --absolute-git-dir)" == "${repo_root}/.git" ]] \
  || die "Git directory must be checkout-local"
[[ "$(git -C "${POLARIS_DIR}" rev-parse --path-format=absolute --git-common-dir)" == "${repo_root}/.git" ]] \
  || die "Git common directory must be checkout-local"
[[ "$(git -C "${POLARIS_DIR}" rev-parse --abbrev-ref HEAD)" == HEAD ]] \
  || die "detached HEAD required"
[[ "$(git -C "${POLARIS_DIR}" rev-parse HEAD)" == "${EXPECTED_POLARIS_COMMIT}" ]] \
  || die "commit mismatch"
[[ -z "$(git -C "${POLARIS_DIR}" status --porcelain=v1 --untracked-files=all)" ]] \
  || die "checkout must be completely clean"
[[ -x "${OPENPI_DIR}/.venv/bin/python" ]] || die "checkout-local host Python missing"
[[ -f "${IMAGE}" && ! -L "${IMAGE}" ]] || die "pinned image missing"
[[ -f "${VULKAN_ICD}" && ! -L "${VULKAN_ICD}" ]] || die "Vulkan ICD missing"
[[ -d "${POLARIS_DATA_DIR}" && ! -L "${POLARIS_DATA_DIR}" ]] || die "data root missing"
[[ "${CACHE_DIR}" == "${CACHE_ROOT}/position-smoke-${SLURM_JOB_ID}" ]] \
  || die "CACHE_DIR must be this job's owned position-smoke cache"
[[ ! -e "${CACHE_DIR}" && ! -L "${CACHE_DIR}" ]] || die "CACHE_DIR must be fresh"
[[ "$(sha256sum "${IMAGE}" | awk '{print $1}')" == "${EXPECTED_IMAGE_SHA256}" ]] \
  || die "image mismatch"
for output in "${RAW_SMOKE}" "${RAW_SMOKE}.child-close.json" \
  "${RAW_SMOKE}.child-close.json.ready.json" "${RAW_SMOKE}.child-close.json.failure.json" \
  "${ATTESTATION}" "${SRUN_STATUS}" "${GPU_INVENTORY}" "${SAVED_JOB_SCRIPT}"; do
  [[ ! -e "${output}" && ! -L "${output}" ]] || die "refusing existing output ${output}"
done

scontrol write batch_script "${SLURM_JOB_ID}" "${SAVED_JOB_SCRIPT}"
chmod 0444 "${SAVED_JOB_SCRIPT}"
sync -f "${SAVED_JOB_SCRIPT}"
sync -f "${RUN_DIR}"
[[ "$(sha256sum "${SAVED_JOB_SCRIPT}" | awk '{print $1}')" == \
  "$(sha256sum "${JOB_SCRIPT}" | awk '{print $1}')" ]] \
  || die "saved spool script differs from committed launch script"

mapfile -t gpu_rows < <(
  nvidia-smi --query-gpu=uuid,name,driver_version --format=csv,noheader,nounits \
    | sed '/^[[:space:]]*$/d'
)
(( ${#gpu_rows[@]} == 1 )) || die "exactly one allocated GPU required"
IFS=',' read -r gpu_uuid gpu_name driver_version <<<"${gpu_rows[0]}"
gpu_uuid="$(xargs <<<"${gpu_uuid}")"
gpu_name="$(xargs <<<"${gpu_name}")"
driver_version="$(xargs <<<"${driver_version}")"
[[ "${gpu_uuid}" == GPU-* && "${gpu_name}" == "NVIDIA L40S" ]] || die "one L40S required"
export GPU_INVENTORY gpu_uuid gpu_name driver_version
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${POLARIS_DIR}/src" \
  "${OPENPI_DIR}/.venv/bin/python" - <<'PY'
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

mkdir -p "${CACHE_ROOT}"
[[ -d "${CACHE_ROOT}" && ! -L "${CACHE_ROOT}" ]] || die "cache root is invalid"
mkdir "${CACHE_DIR}"
mkdir "${CACHE_DIR}/home"
cleanup() { rm -rf --one-file-system "${CACHE_DIR}" 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export NVIDIA_VISIBLE_DEVICES="${gpu_uuid}"
export NVIDIA_DRIVER_CAPABILITIES=all
export PYTHONUNBUFFERED=1 SLURM_EXPORT_ENV=ALL
mounts="/dev/shm:/dev/shm,${POLARIS_DIR}:${POLARIS_DIR}:ro,${POLARIS_DATA_DIR}:${POLARIS_DATA_DIR}:ro,${RUN_DIR}:${RUN_DIR}:rw,${CACHE_DIR}:/cache:rw,${VULKAN_ICD}:/etc/vulkan/icd.d/nvidia_icd.json:ro"
set +e
srun --ntasks=1 "--cpus-per-task=${SLURM_CPUS_PER_TASK:-16}" \
  "--container-image=${IMAGE}" --container-mounts="${mounts}" \
  --container-workdir="${POLARIS_DIR}" --no-container-entrypoint \
  --no-container-mount-home --container-remap-root --container-writable \
  --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES --export=ALL \
  /usr/bin/env VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json \
  ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y OMNI_KIT_ALLOW_ROOT=1 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="${POLARIS_DIR}/src" \
  POLARIS_DATA_PATH="${POLARIS_DATA_DIR}" XDG_CACHE_HOME=/cache \
  HF_HOME=/cache/huggingface HOME=/cache/home \
  /.venv/bin/python "${POLARIS_DIR}/scripts/smoke_pi05_droid_position_controller.py" \
  --headless --device cuda:0 --environment DROID-FoodBussing \
  --command-magnitude 1.5 --output-json "${RAW_SMOKE}"
srun_exit_code=$?
set -e

export SRUN_STATUS srun_exit_code
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${POLARIS_DIR}/src" \
  "${OPENPI_DIR}/.venv/bin/python" - <<'PY'
import os
from pathlib import Path
from polaris.pi05_droid_native_eval_contract import publish_immutable_json
publish_immutable_json(
    Path(os.environ["SRUN_STATUS"]),
    {"job_id": int(os.environ["SLURM_JOB_ID"]), "srun_exit_code": int(os.environ["srun_exit_code"])},
)
PY
(( srun_exit_code == 0 )) || exit "${srun_exit_code}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${POLARIS_DIR}/src" \
  "${OPENPI_DIR}/.venv/bin/python" "${FINALIZER}" \
  --smoke "${RAW_SMOKE}" --polaris-repo "${POLARIS_DIR}" \
  --expected-polaris-commit "${EXPECTED_POLARIS_COMMIT}" --job-id "${SLURM_JOB_ID}" \
  --srun-status "${SRUN_STATUS}" --gpu-inventory "${GPU_INVENTORY}" \
  --saved-job-script "${SAVED_JOB_SCRIPT}" --container-image "${IMAGE}" \
  --data-dir "${POLARIS_DATA_DIR}" --output "${ATTESTATION}"

echo "DROID position controller-only smoke complete and non-promotable: ${ATTESTATION}"
