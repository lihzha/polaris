#!/usr/bin/env bash
#SBATCH --account=nvr_lpr_rvp
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:50:00
#SBATCH --no-requeue
#SBATCH --job-name=pi05_pos_canary
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/lzha/slurm_logs/polaris-pi05-position/%x-%j.out

set -Eeuo pipefail

NFS_ROOT="${NFS_ROOT:-/lustre/fsw/portfolios/nvr/users/lzha}"
: "${POLARIS_DIR:?Set the frozen standalone PolaRiS checkout}"
: "${EXPECTED_POLARIS_COMMIT:?Set the exact combined PolaRiS commit}"
: "${OPENPI_DIR:?Set the exact external OpenPI checkout}"
: "${DROID_DIR:?Set the exact official DROID checkout}"
: "${RUN_DIR:?Set one fresh pre-created canary directory}"
: "${POSITION_CONTROLLER_ATTESTATION:?Set the position controller attestation}"
: "${EXPECTED_POSITION_CONTROLLER_ATTESTATION_SHA256:?Set its exact SHA-256}"

POLARIS_PYXIS_IMAGE="${POLARIS_PYXIS_IMAGE:-${NFS_ROOT}/cache/polaris/polaris-eval-cuda13-fd00a51.sqsh}"
POLARIS_DATA_DIR="${POLARIS_DATA_DIR:-${NFS_ROOT}/data/PolaRiS-Hub}"
POLARIS_CACHE_DIR="${POLARIS_CACHE_DIR:-${NFS_ROOT}/cache/polaris/runtime/pi05-position-${SLURM_JOB_ID}}"

die() { echo "ERROR: $*" >&2; exit 2; }

for forbidden in CONTROLLER_COMPLETION EXPECTED_CONTROLLER_COMPLETION_SHA256 \
  ALL_SIX_CONTROLLER_COMPLETION EXPECTED_ALL_SIX_COMPLETION_SHA256 \
  EXPECTED_ALL_SIX_PROFILE; do
  [[ -z "${!forbidden+x}" ]] \
    || die "Old direct-rad/s controller gate is forbidden: ${forbidden}"
done
[[ -z "${PORT+x}" ]] || die "Ambient PORT is forbidden"
[[ "${EXPECTED_POLARIS_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || die "Malformed commit"
[[ "${EXPECTED_POSITION_CONTROLLER_ATTESTATION_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "Malformed controller-attestation SHA-256"

for directory in POLARIS_DIR OPENPI_DIR DROID_DIR POLARIS_DATA_DIR RUN_DIR; do
  value="${!directory}"
  [[ -d "${value}" && ! -L "${value}" ]] || die "${directory} is invalid"
  printf -v "${directory}" '%s' "$(realpath -e -- "${value}")"
done
for file_var in POLARIS_PYXIS_IMAGE POSITION_CONTROLLER_ATTESTATION; do
  value="${!file_var}"
  [[ -f "${value}" && ! -L "${value}" ]] || die "${file_var} is invalid"
  printf -v "${file_var}" '%s' "$(realpath -e -- "${value}")"
done
POLARIS_CACHE_DIR="$(realpath -m -- "${POLARIS_CACHE_DIR}")"
JOB_SCRIPT="${POLARIS_DIR}/scripts/polaris/l40s_pi05_droid_position_canary.sbatch"
SAVED_JOB_SCRIPT="${RUN_DIR}/job-${SLURM_JOB_ID}.sbatch"
GPU_INVENTORY="${RUN_DIR}/gpu-${SLURM_JOB_ID}.json"
SLURM_JOB_RECORD="${RUN_DIR}/slurm-job-${SLURM_JOB_ID}.txt"
SUBMISSION_RECORD="${RUN_DIR}/submission-${SLURM_JOB_ID}.json"
[[ ! -e "${POLARIS_CACHE_DIR}" && ! -L "${POLARIS_CACHE_DIR}" ]] \
  || die "Refusing stale per-job cache: ${POLARIS_CACHE_DIR}"
mkdir -p "$(dirname "${POLARIS_CACHE_DIR}")"
mkdir "${POLARIS_CACHE_DIR}"
cleanup() { rm -rf --one-file-system "${POLARIS_CACHE_DIR}" 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for output in "${SAVED_JOB_SCRIPT}" "${GPU_INVENTORY}" "${SLURM_JOB_RECORD}"; do
  [[ ! -e "${output}" && ! -L "${output}" ]] || die "Refusing existing output: ${output}"
done
scontrol write batch_script "${SLURM_JOB_ID}" "${SAVED_JOB_SCRIPT}"
chmod 0444 "${SAVED_JOB_SCRIPT}"
sync -f "${SAVED_JOB_SCRIPT}"
sync -f "${RUN_DIR}"
committed_script_sha="$(git -C "${POLARIS_DIR}" show "${EXPECTED_POLARIS_COMMIT}:scripts/polaris/l40s_pi05_droid_position_canary.sbatch" | sha256sum | awk '{print $1}')"
[[ "$(sha256sum "${JOB_SCRIPT}" | awk '{print $1}')" == "${committed_script_sha}" \
  && "$(sha256sum "${SAVED_JOB_SCRIPT}" | awk '{print $1}')" == "${committed_script_sha}" ]] \
  || die "Saved Slurm spool differs from committed source"

scontrol show job "${SLURM_JOB_ID}" -dd -o >"${SLURM_JOB_RECORD}"
chmod 0444 "${SLURM_JOB_RECORD}"
sync -f "${SLURM_JOB_RECORD}"
sync -f "${RUN_DIR}"

mapfile -t gpu_rows < <(
  nvidia-smi --query-gpu=uuid,name,driver_version --format=csv,noheader,nounits \
    | sed '/^[[:space:]]*$/d'
)
(( ${#gpu_rows[@]} == 1 )) || die "Expected exactly one allocated GPU"
IFS=',' read -r gpu_uuid gpu_name driver_version <<<"${gpu_rows[0]}"
gpu_uuid="$(xargs <<<"${gpu_uuid}")"
gpu_name="$(xargs <<<"${gpu_name}")"
driver_version="$(xargs <<<"${driver_version}")"
[[ "${gpu_uuid}" == GPU-* && "${gpu_name}" == "NVIDIA L40S" ]] \
  || die "Allocation is not exactly one NVIDIA L40S"
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

for _ in {1..120}; do
  [[ -f "${SUBMISSION_RECORD}" && ! -L "${SUBMISSION_RECORD}" \
    && "$(stat -c '%a' -- "${SUBMISSION_RECORD}" 2>/dev/null || true)" == 444 ]] && break
  sleep 1
done
[[ -f "${SUBMISSION_RECORD}" && ! -L "${SUBMISSION_RECORD}" ]] \
  || die "Submitter did not publish the immutable submission record"

export POLARIS_DIR EXPECTED_POLARIS_COMMIT OPENPI_DIR DROID_DIR RUN_DIR
export POSITION_CONTROLLER_ATTESTATION EXPECTED_POSITION_CONTROLLER_ATTESTATION_SHA256
export POLARIS_PYXIS_IMAGE POLARIS_DATA_DIR POLARIS_CACHE_DIR NFS_ROOT
export NVIDIA_VISIBLE_DEVICES="${gpu_uuid}"
export NVIDIA_DRIVER_CAPABILITIES=all PYTHONUNBUFFERED=1 SLURM_EXPORT_ENV=ALL

echo "[$(date -Iseconds)] job=${SLURM_JOB_ID} host=$(hostname) protocol=polaris-native-droid-freshq-delta0p2-position-h8-canary1-v1"
"${POLARIS_DIR}/scripts/polaris/eval_pi05_droid_position.sh"
