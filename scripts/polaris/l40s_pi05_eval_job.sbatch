#!/usr/bin/env bash
#SBATCH --account=nvr_lpr_rvp
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:50:00
#SBATCH --job-name=pi05_polaris_eval
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/lzha/slurm_logs/polaris-pi05/%x-%j.out

set -Eeuo pipefail

NFS_ROOT="${NFS_ROOT:-/lustre/fsw/portfolios/nvr/users/lzha}"
: "${POLARIS_DIR:?Set POLARIS_DIR to the frozen committed PolaRiS worktree}"
: "${EXPECTED_POLARIS_COMMIT:?Set EXPECTED_POLARIS_COMMIT to the frozen launch commit}"
: "${POLARIS_ENVIRONMENT:?Set POLARIS_ENVIRONMENT to one DROID task}"
: "${RUN_NAMESPACE:?Set RUN_NAMESPACE to the immutable sweep namespace}"

export NFS_ROOT POLARIS_DIR EXPECTED_POLARIS_COMMIT POLARIS_ENVIRONMENT RUN_NAMESPACE
export OPENPI_DIR="${OPENPI_DIR:-${POLARIS_DIR}/third_party/openpi}"
export POLARIS_DATA_DIR="${POLARIS_DATA_DIR:-${NFS_ROOT}/data/PolaRiS-Hub}"
export POLARIS_PYXIS_IMAGE="${POLARIS_PYXIS_IMAGE:-${NFS_ROOT}/cache/polaris/polaris-eval-cuda13-fd00a51.sqsh}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${NFS_ROOT}/cache/openpi-polaris}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${NFS_ROOT}/results/polaris-pi05}"
export POLARIS_CACHE_DIR="${POLARIS_CACHE_DIR:-${NFS_ROOT}/cache/polaris/runtime/${SLURM_JOB_ID}}"
export ROLLOUTS="${ROLLOUTS:-50}"
export ENVIRONMENT_SEED="${ENVIRONMENT_SEED:-0}"
export RUN_LABEL=official-pi05-polaris
export PORT="$((20000 + SLURM_JOB_ID % 20000))"
export CHECKPOINT_URI=gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris
export POLICY_CONFIG=pi05_droid_jointpos_polaris
export DRY_RUN=0
export NVIDIA_DRIVER_CAPABILITIES=all
mapfile -t allocated_gpu_uuids < <(
  nvidia-smi --query-gpu=uuid --format=csv,noheader | sed '/^[[:space:]]*$/d'
)
(( ${#allocated_gpu_uuids[@]} == 1 )) || {
  echo "Expected exactly one allocated GPU, saw ${#allocated_gpu_uuids[@]}" >&2
  exit 5
}
export NVIDIA_VISIBLE_DEVICES="${allocated_gpu_uuids[0]}"
export PYTHONUNBUFFERED=1 SLURM_EXPORT_ENV=ALL

echo "[$(date -Iseconds)] job=${SLURM_JOB_ID} host=$(hostname) task=${POLARIS_ENVIRONMENT} rollouts=${ROLLOUTS} environment_seed=${ENVIRONMENT_SEED}"
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total --format=csv,noheader

exec "${POLARIS_DIR}/scripts/polaris/eval_pi05_droid_jointpos_polaris.sh"
