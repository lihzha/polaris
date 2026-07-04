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
#SBATCH --job-name=pi05_native_canary
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/lzha/slurm_logs/polaris-pi05-native/%x-%j.out

set -Eeuo pipefail

NFS_ROOT="${NFS_ROOT:-/lustre/fsw/portfolios/nvr/users/lzha}"
: "${POLARIS_DIR:?Set the frozen standalone PolaRiS checkout}"
: "${EXPECTED_POLARIS_COMMIT:?Set the exact PolaRiS commit}"
: "${RUN_DIR:?Set one fresh canary attempt directory}"
: "${CONTROLLER_COMPLETION:?Set job1098174 completion path}"
: "${EXPECTED_CONTROLLER_COMPLETION_SHA256:?Set job1098174 completion digest}"
: "${ALL_SIX_CONTROLLER_COMPLETION:?Set accepted job1098682 all-six completion path}"
: "${EXPECTED_ALL_SIX_COMPLETION_SHA256:?Set accepted all-six completion digest}"
: "${EXPECTED_ALL_SIX_PROFILE:?Set accepted all-six completion profile}"

POLARIS_PYXIS_IMAGE="${POLARIS_PYXIS_IMAGE:-${NFS_ROOT}/cache/polaris/polaris-eval-cuda13-fd00a51.sqsh}"
POLARIS_DATA_DIR="${POLARIS_DATA_DIR:-${NFS_ROOT}/data/PolaRiS-Hub}"
JOB_SCRIPT="${POLARIS_DIR}/scripts/polaris/l40s_pi05_droid_native_jointvelocity_canary.sbatch"
SAVED_JOB_SCRIPT="${RUN_DIR}/job-${SLURM_JOB_ID}.sbatch"
GPU_INVENTORY="${RUN_DIR}/gpu-${SLURM_JOB_ID}.json"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ -z "${PORT+x}" ]] \
  || die "Ambient PORT is forbidden; the canary server binds an OS-assigned port"
[[ -d "${RUN_DIR}" && ! -L "${RUN_DIR}" ]] || die "Missing regular RUN_DIR"
for output in "${SAVED_JOB_SCRIPT}" "${GPU_INVENTORY}"; do
  [[ ! -e "${output}" && ! -L "${output}" ]] || die "Refusing existing output: ${output}"
done

scontrol write batch_script "${SLURM_JOB_ID}" "${SAVED_JOB_SCRIPT}"
chmod 0444 "${SAVED_JOB_SCRIPT}"
sync -f "${SAVED_JOB_SCRIPT}"
sync -f "${RUN_DIR}"
[[ "$(sha256sum "${SAVED_JOB_SCRIPT}" | awk '{print $1}')" == \
    "$(sha256sum "${JOB_SCRIPT}" | awk '{print $1}')" ]] \
  || die "Saved Slurm script differs from committed source"

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

export NVIDIA_VISIBLE_DEVICES="${gpu_uuid}"
export NVIDIA_DRIVER_CAPABILITIES=all
export PYTHONUNBUFFERED=1
export SLURM_EXPORT_ENV=ALL

echo "[$(date -Iseconds)] job=${SLURM_JOB_ID} host=$(hostname) profile=official-pi05-droid-native-canary"
exec "${POLARIS_DIR}/scripts/polaris/eval_pi05_droid_native_jointvelocity.sh"
