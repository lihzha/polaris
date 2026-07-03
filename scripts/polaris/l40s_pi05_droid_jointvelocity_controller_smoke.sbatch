#!/usr/bin/env bash
#SBATCH --account=nvr_lpr_rvp
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --job-name=pi05_jv_ctrl_smoke
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/lzha/slurm_logs/polaris-pi05-jointvelocity/%x-%j.out

set -Eeuo pipefail

NFS_ROOT="${NFS_ROOT:-/lustre/fsw/portfolios/nvr/users/lzha}"
: "${POLARIS_DIR:?Set POLARIS_DIR to the frozen committed PolaRiS worktree}"
: "${EXPECTED_POLARIS_COMMIT:?Set EXPECTED_POLARIS_COMMIT to the frozen commit}"
: "${RUN_DIR:?Set RUN_DIR to one fresh controller-smoke result directory}"

POLARIS_DATA_DIR="${POLARIS_DATA_DIR:-${NFS_ROOT}/data/PolaRiS-Hub}"
POLARIS_PYXIS_IMAGE="${POLARIS_PYXIS_IMAGE:-${NFS_ROOT}/cache/polaris/polaris-eval-cuda13-fd00a51.sqsh}"
POLARIS_VULKAN_ICD_PATH="${POLARIS_VULKAN_ICD_PATH:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
POLARIS_CACHE_DIR="${POLARIS_CACHE_DIR:-${NFS_ROOT}/cache/polaris/runtime/jointvelocity-smoke-${SLURM_JOB_ID}}"
EXPECTED_IMAGE_SHA256=ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a
EXPECTED_HUB_REVISION=8c7e4103e266ef83d8b1ad2e9a63116edd5f155b
SMOKE_ARTIFACT="${RUN_DIR}/joint-velocity-smoke-${SLURM_JOB_ID}.json"
COMPLETION="${RUN_DIR}/controller-smoke-${SLURM_JOB_ID}.completion.json"
SRUN_STATUS="${RUN_DIR}/srun-${SLURM_JOB_ID}.status.json"
GPU_INVENTORY="${RUN_DIR}/gpu-${SLURM_JOB_ID}.csv"
SAVED_JOB_SCRIPT="${RUN_DIR}/job-${SLURM_JOB_ID}.sbatch"
FINALIZER="${POLARIS_DIR}/scripts/polaris/finalize_pi05_droid_jointvelocity_controller_smoke.py"
JOB_SCRIPT="${POLARIS_DIR}/scripts/polaris/l40s_pi05_droid_jointvelocity_controller_smoke.sbatch"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_normalized_absolute_path() {
  local field="$1"
  local value="$2"
  local normalized
  [[ "${value}" == /* && "${value}" != //* ]] \
    || die "${field} must be an absolute path with one leading slash"
  normalized="$(realpath -sm -- "${value}")" \
    || die "Unable to normalize ${field}"
  [[ "${value}" == "${normalized}" ]] \
    || die "${field} must use one normalized absolute path spelling"
}

[[ "${EXPECTED_POLARIS_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || die "EXPECTED_POLARIS_COMMIT must be a full lowercase commit"
require_normalized_absolute_path POLARIS_DIR "${POLARIS_DIR}"
for path_contract in \
  "NFS_ROOT=${NFS_ROOT}" \
  "RUN_DIR=${RUN_DIR}" \
  "POLARIS_DATA_DIR=${POLARIS_DATA_DIR}" \
  "POLARIS_PYXIS_IMAGE=${POLARIS_PYXIS_IMAGE}" \
  "POLARIS_VULKAN_ICD_PATH=${POLARIS_VULKAN_ICD_PATH}" \
  "POLARIS_CACHE_DIR=${POLARIS_CACHE_DIR}"; do
  require_normalized_absolute_path \
    "${path_contract%%=*}" "${path_contract#*=}"
done
[[ ! -L "${POLARIS_DIR}" ]] || die "POLARIS_DIR must not be a symlink"
POLARIS_DIR_RESOLVED="$(realpath "${POLARIS_DIR}")"
[[ -d "${POLARIS_DIR}/.git" && ! -L "${POLARIS_DIR}/.git" ]] \
  || die "POLARIS_DIR must be a standalone clone with an in-root .git directory"
git_dir="$(git -C "${POLARIS_DIR}" rev-parse --absolute-git-dir)"
git_common_dir="$(git -C "${POLARIS_DIR}" rev-parse --path-format=absolute --git-common-dir)"
[[ "${git_dir}" == "${POLARIS_DIR_RESOLVED}/.git" && \
  "${git_common_dir}" == "${POLARIS_DIR_RESOLVED}/.git" ]] \
  || die "POLARIS_DIR Git metadata must be wholly contained in its .git directory"
[[ "$(git -C "${POLARIS_DIR}" rev-parse --abbrev-ref HEAD)" == HEAD ]] \
  || die "POLARIS_DIR must be checked out at a detached HEAD"
[[ "$(git -C "${POLARIS_DIR}" rev-parse --show-toplevel)" == \
  "${POLARIS_DIR_RESOLVED}" ]] \
  || die "POLARIS_DIR must name the exact Git root"
[[ "$(git -C "${POLARIS_DIR}" rev-parse HEAD)" == "${EXPECTED_POLARIS_COMMIT}" ]] \
  || die "PolaRiS commit mismatch"
[[ -z "$(git -C "${POLARIS_DIR}" status --porcelain=v1 --untracked-files=all)" ]] \
  || die "PolaRiS checkout must be completely clean"
[[ -f "${FINALIZER}" && -f "${JOB_SCRIPT}" ]] || die "Missing committed smoke scaffold"
[[ -f "${POLARIS_PYXIS_IMAGE}" ]] || die "Missing pinned Pyxis image"
[[ -d "${POLARIS_DATA_DIR}" ]] || die "Missing PolaRiS-Hub data"
[[ -f "${POLARIS_VULKAN_ICD_PATH}" ]] || die "Missing allocated-node Vulkan ICD"
mkdir -p "${RUN_DIR}" "${POLARIS_CACHE_DIR}" "${POLARIS_CACHE_DIR}/home"
for output in "${SMOKE_ARTIFACT}" "${COMPLETION}" "${SRUN_STATUS}" \
  "${GPU_INVENTORY}" "${SAVED_JOB_SCRIPT}"; do
  [[ ! -e "${output}" && ! -L "${output}" ]] || die "Refusing existing output: ${output}"
done

scontrol write batch_script "${SLURM_JOB_ID}" "${SAVED_JOB_SCRIPT}"
chmod 0444 "${SAVED_JOB_SCRIPT}"
sync -f "${SAVED_JOB_SCRIPT}"
sync -f "${RUN_DIR}"
[[ "$(sha256sum "${SAVED_JOB_SCRIPT}" | awk '{print $1}')" == \
    "$(sha256sum "${JOB_SCRIPT}" | awk '{print $1}')" ]] \
  || die "Saved Slurm script differs from the committed script"

[[ "$(sha256sum "${POLARIS_PYXIS_IMAGE}" | awk '{print $1}')" == \
    "${EXPECTED_IMAGE_SHA256}" ]] || die "Pyxis image SHA-256 mismatch"

declare -A expected_asset_sha=(
  [food_bussing/initial_conditions.json]=40091faee14f692350220871d30705294f21f17ae3d2974cd3c09a34d560f5de
  [food_bussing/scene.usda]=82cd641e422935b394ce7ea7b6be55214c9952a2544000222921e544c409b489
  [nvidia_droid/noninstanceable.usd]=d8379925b103963dbf3e7c85bcc4ae101b81b7c1d7dabe7d2e964f41d069ec44
)
for relative_path in "${!expected_asset_sha[@]}"; do
  asset="${POLARIS_DATA_DIR}/${relative_path}"
  metadata="${POLARIS_DATA_DIR}/.cache/huggingface/download/${relative_path}.metadata"
  [[ -f "${asset}" && -f "${metadata}" ]] || die "Missing pinned asset: ${relative_path}"
  [[ "$(sha256sum "${asset}" | awk '{print $1}')" == \
      "${expected_asset_sha[${relative_path}]}" ]] || die "Asset digest mismatch: ${relative_path}"
  [[ "$(head -n 1 "${metadata}")" == "${EXPECTED_HUB_REVISION}" ]] \
    || die "Hub revision mismatch: ${relative_path}"
done

gpu_inventory="$(nvidia-smi --query-gpu=uuid,name,driver_version --format=csv,noheader)"
[[ "$(printf '%s\n' "${gpu_inventory}" | sed '/^[[:space:]]*$/d' | wc -l)" == 1 ]] \
  || die "Expected exactly one allocated GPU"
[[ "${gpu_inventory}" == *,\ NVIDIA\ L40S,* ]] || die "Allocated GPU is not an NVIDIA L40S"
GPU_INVENTORY="${GPU_INVENTORY}" GPU_INVENTORY_TEXT="${gpu_inventory}" \
  /usr/bin/python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["GPU_INVENTORY"])
payload = (os.environ["GPU_INVENTORY_TEXT"].strip() + "\n").encode("utf-8")
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
try:
    os.write(descriptor, payload)
    os.fsync(descriptor)
    os.fchmod(descriptor, 0o444)
finally:
    os.close(descriptor)
directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY

gpu_uuid="${gpu_inventory%%,*}"
gpu_uuid="${gpu_uuid//[[:space:]]/}"
export NVIDIA_VISIBLE_DEVICES="${gpu_uuid}"
export NVIDIA_DRIVER_CAPABILITIES=all
export PYTHONUNBUFFERED=1 SLURM_EXPORT_ENV=ALL

mounts="/dev/shm:/dev/shm,${POLARIS_DIR}:${POLARIS_DIR}:ro,${POLARIS_DATA_DIR}:${POLARIS_DATA_DIR}:ro,${RUN_DIR}:${RUN_DIR}:rw,${POLARIS_CACHE_DIR}:/cache:rw,${POLARIS_VULKAN_ICD_PATH}:/etc/vulkan/icd.d/nvidia_icd.json:ro"
set +e
srun \
  --ntasks=1 \
  "--cpus-per-task=${SLURM_CPUS_PER_TASK}" \
  "--container-image=${POLARIS_PYXIS_IMAGE}" \
  "--container-mounts=${mounts}" \
  "--container-workdir=${POLARIS_DIR}" \
  --no-container-entrypoint \
  --no-container-mount-home \
  --container-remap-root \
  --container-writable \
  --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES \
  --export=ALL \
  /usr/bin/env \
    VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json \
    ACCEPT_EULA=Y \
    OMNI_KIT_ACCEPT_EULA=YES \
    PRIVACY_CONSENT=Y \
    OMNI_KIT_ALLOW_ROOT=1 \
    PYTHONUNBUFFERED=1 \
    "PYTHONPATH=${POLARIS_DIR}/src" \
    "POLARIS_DATA_PATH=${POLARIS_DATA_DIR}" \
    XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/huggingface \
    HOME=/cache/home \
    /.venv/bin/python \
      "${POLARIS_DIR}/scripts/smoke_joint_velocity_controller.py" \
      --environment DROID-FoodBussing \
      --command-magnitude 0.25 \
      --settle-steps 5 \
      --device cuda:0 \
      --headless \
      --output-json "${SMOKE_ARTIFACT}"
srun_exit_code=$?
set -e

SRUN_STATUS="${SRUN_STATUS}" SLURM_JOB_ID="${SLURM_JOB_ID}" \
  SRUN_EXIT_CODE="${srun_exit_code}" /usr/bin/python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["SRUN_STATUS"])
value = {
    "job_id": int(os.environ["SLURM_JOB_ID"]),
    "srun_exit_code": int(os.environ["SRUN_EXIT_CODE"]),
}
payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
try:
    os.write(descriptor, payload)
    os.fsync(descriptor)
    os.fchmod(descriptor, 0o444)
finally:
    os.close(descriptor)
directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY

(( srun_exit_code == 0 )) || {
  echo "Controller smoke srun failed with ${srun_exit_code}; no completion will be published" >&2
  exit "${srun_exit_code}"
}

/usr/bin/python3 "${FINALIZER}" finalize \
  --job-id "${SLURM_JOB_ID}" \
  --smoke-artifact "${SMOKE_ARTIFACT}" \
  --completion "${COMPLETION}" \
  --srun-status "${SRUN_STATUS}" \
  --gpu-inventory "${GPU_INVENTORY}" \
  --saved-job-script "${SAVED_JOB_SCRIPT}" \
  --polaris-repo "${POLARIS_DIR}" \
  --expected-polaris-commit "${EXPECTED_POLARIS_COMMIT}" \
  --container-image "${POLARIS_PYXIS_IMAGE}" \
  --data-dir "${POLARIS_DATA_DIR}"

echo "Controller-only smoke complete and non-promotable: ${COMPLETION}"
