#!/usr/bin/env bash

# Submit ordinary, independently restartable official pi0.5 PolaRiS eval jobs.

set -Eeuo pipefail

MODE="${1:-}"
case "${MODE}" in
  canary|full) ;;
  *) echo "Usage: $0 {canary|full}" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${SCRIPT_DIR}/l40s_pi05_eval_job.sbatch}"
POLARIS_DIR="${POLARIS_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUN_NAMESPACE="${RUN_NAMESPACE:-pi05-polaris-$(date -u +%Y%m%dT%H%M%SZ)}"
NFS_ROOT="${NFS_ROOT:-/lustre/fsw/portfolios/nvr/users/lzha}"
SBATCH_LOG_ROOT="${SBATCH_LOG_ROOT:-${NFS_ROOT}/slurm_logs/polaris-pi05/${RUN_NAMESPACE}}"
SUBMISSION_MANIFEST="${SUBMISSION_MANIFEST:-${NFS_ROOT}/results/polaris-pi05/${RUN_NAMESPACE}/${MODE}_jobs.tsv}"
POLARIS_COMMIT="$(git -C "${POLARIS_DIR}" rev-parse HEAD)"
ALLOW_RESUBMIT="${ALLOW_RESUBMIT:-0}"

if [[ "${MODE}" == canary ]]; then
  tasks=(DROID-FoodBussing)
  rollouts="${ROLLOUTS:-1}"
  time_limit="${SBATCH_TIME:-01:00:00}"
  job_prefix="pi05-canary"
else
  tasks=(
    DROID-BlockStackKitchen
    DROID-FoodBussing
    DROID-PanClean
    DROID-MoveLatteCup
    DROID-OrganizeTools
    DROID-TapeIntoContainer
  )
  rollouts="${ROLLOUTS:-50}"
  time_limit="${SBATCH_TIME:-03:50:00}"
  job_prefix="pi05-full50"
fi

[[ "${rollouts}" =~ ^[1-9][0-9]*$ ]] || { echo "ROLLOUTS must be positive" >&2; exit 2; }
[[ -f "${SBATCH_SCRIPT}" ]] || { echo "Missing sbatch script: ${SBATCH_SCRIPT}" >&2; exit 2; }
mkdir -p "${SBATCH_LOG_ROOT}" "$(dirname "${SUBMISSION_MANIFEST}")"
exec 9>"${SUBMISSION_MANIFEST}.lock"
flock -n 9 || { echo "Another submitter holds ${SUBMISSION_MANIFEST}.lock" >&2; exit 4; }
if [[ ! -e "${SUBMISSION_MANIFEST}" ]]; then
  printf 'job_id\tmode\ttask\trollouts\trun_namespace\tsubmitted_at\n' > "${SUBMISSION_MANIFEST}"
fi

job_ids=()
for task in "${tasks[@]}"; do
  existing_job_id="$(
    awk -F '\t' -v mode="${MODE}" -v task="${task}" \
      '$2 == mode && $3 == task {job_id = $1} END {print job_id}' \
      "${SUBMISSION_MANIFEST}"
  )"
  if [[ -n "${existing_job_id}" && "${ALLOW_RESUBMIT}" != 1 ]]; then
    echo "Existing ${MODE} attempt for ${task}: job ${existing_job_id}; set ALLOW_RESUBMIT=1 for an explicit retry"
    job_ids+=("${existing_job_id}")
    continue
  fi
  short_task="${task#DROID-}"
  job_name="${job_prefix}_${short_task}"
  export_vars="PATH=${PATH},HOME=${HOME},POLARIS_DIR=${POLARIS_DIR},EXPECTED_POLARIS_COMMIT=${POLARIS_COMMIT},POLARIS_ENVIRONMENT=${task},ROLLOUTS=${rollouts},RUN_NAMESPACE=${RUN_NAMESPACE}"
  job_id="$(sbatch --parsable \
    --job-name="${job_name}" \
    --time="${time_limit}" \
    --output="${SBATCH_LOG_ROOT}/%x-%j.out" \
    --export="${export_vars}" \
    "${SBATCH_SCRIPT}")"
  [[ "${job_id}" =~ ^[0-9]+$ ]] || { echo "Invalid job ID: ${job_id}" >&2; exit 3; }
  job_ids+=("${job_id}")
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${job_id}" "${MODE}" "${task}" "${rollouts}" "${RUN_NAMESPACE}" "$(date -Iseconds)" \
    | tee -a "${SUBMISSION_MANIFEST}"
done

printf 'submitted_job_ids=%s\n' "${job_ids[*]}"
printf 'submission_manifest=%s\n' "${SUBMISSION_MANIFEST}"
