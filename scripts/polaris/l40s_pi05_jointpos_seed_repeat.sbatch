#!/usr/bin/env bash
#SBATCH --account=nvr_lpr_rvp
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --job-name=pi05_seed_repeat
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/lzha/slurm_logs/polaris-pi05/%x-%j.out

set -Eeuo pipefail

NFS_ROOT="${NFS_ROOT:-/lustre/fsw/portfolios/nvr/users/lzha}"
: "${POLARIS_DIR:?Set POLARIS_DIR to the frozen committed PolaRiS worktree}"
: "${EXPECTED_POLARIS_COMMIT:?Set EXPECTED_POLARIS_COMMIT}"
: "${RUN_NAMESPACE_BASE:?Set RUN_NAMESPACE_BASE}"

export NFS_ROOT POLARIS_DIR EXPECTED_POLARIS_COMMIT
export OPENPI_DIR="${OPENPI_DIR:-${POLARIS_DIR}/third_party/openpi}"
export POLARIS_DATA_DIR="${POLARIS_DATA_DIR:-${NFS_ROOT}/data/PolaRiS-Hub}"
export POLARIS_PYXIS_IMAGE="${POLARIS_PYXIS_IMAGE:-${NFS_ROOT}/cache/polaris/polaris-eval-cuda13-fd00a51.sqsh}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${NFS_ROOT}/cache/openpi-polaris}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${NFS_ROOT}/results/polaris-pi05}"
export POLARIS_ENVIRONMENT=DROID-FoodBussing
export ROLLOUTS="${ROLLOUTS:-2}"
export ENVIRONMENT_SEED="${ENVIRONMENT_SEED:-0}"
export CHECKPOINT_URI=gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris
export POLICY_CONFIG=pi05_droid_jointpos_polaris
export DRY_RUN=0
export NVIDIA_DRIVER_CAPABILITIES=all
export PYTHONUNBUFFERED=1 SLURM_EXPORT_ENV=ALL

[[ "$(git -C "${POLARIS_DIR}" rev-parse HEAD)" == "${EXPECTED_POLARIS_COMMIT}" ]]
[[ -z "$(git -C "${POLARIS_DIR}" status --porcelain=v1 --untracked-files=all)" ]]
mapfile -t allocated_gpu_uuids < <(
  nvidia-smi --query-gpu=uuid --format=csv,noheader | sed '/^[[:space:]]*$/d'
)
(( ${#allocated_gpu_uuids[@]} == 1 )) || {
  echo "Expected exactly one allocated GPU, saw ${#allocated_gpu_uuids[@]}" >&2
  exit 5
}
export NVIDIA_VISIBLE_DEVICES="${allocated_gpu_uuids[0]}"

manifest_dir="${OUTPUT_ROOT}/${RUN_NAMESPACE_BASE}"
manifest="${manifest_dir}/seed_repeat_runs.tsv"
mkdir -p "${manifest_dir}"
[[ ! -e "${manifest}" ]] || { echo "Repeat manifest already exists: ${manifest}" >&2; exit 2; }
printf 'repeat_id\trun_namespace\tenvironment_seed\trollouts\tstarted_at\tcompleted_at\n' > "${manifest}"

port_offset=0
for repeat_id in a b; do
  export RUN_NAMESPACE="${RUN_NAMESPACE_BASE}-repeat${repeat_id}"
  export RUN_LABEL="official-pi05-polaris-seed${ENVIRONMENT_SEED}-repeat${repeat_id}"
  export POLARIS_CACHE_DIR="${NFS_ROOT}/cache/polaris/runtime/${SLURM_JOB_ID}/repeat${repeat_id}"
  export PORT="$((20000 + SLURM_JOB_ID % 19000 + port_offset))"
  started_at="$(date -Iseconds)"
  "${POLARIS_DIR}/scripts/polaris/eval_pi05_droid_jointpos_polaris.sh"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${repeat_id}" "${RUN_NAMESPACE}" "${ENVIRONMENT_SEED}" "${ROLLOUTS}" \
    "${started_at}" "$(date -Iseconds)" >> "${manifest}"
  port_offset=$((port_offset + 1))
done

echo "Seed-repeat evaluation complete: ${manifest}"
