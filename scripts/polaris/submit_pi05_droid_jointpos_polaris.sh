#!/usr/bin/env bash

# Submit ordinary, independently restartable official pi0.5 PolaRiS eval jobs.

set -Eeuo pipefail

MODE="${1:-}"
case "${MODE}" in
  canary|foodbussing50|full) ;;
  *) echo "Usage: $0 {canary|foodbussing50|full}" >&2; exit 2 ;;
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
ENVIRONMENT_SEED="${ENVIRONMENT_SEED:-0}"

[[ -z "$(git -C "${POLARIS_DIR}" status --porcelain=v1 --untracked-files=all)" ]] \
  || { echo "PolaRiS source has tracked or untracked modifications" >&2; exit 2; }

if [[ "${MODE}" == canary ]]; then
  tasks=(DROID-FoodBussing)
  rollouts="${ROLLOUTS:-1}"
  time_limit="${SBATCH_TIME:-01:00:00}"
  job_prefix="pi05-canary"
elif [[ "${MODE}" == foodbussing50 ]]; then
  tasks=(DROID-FoodBussing)
  rollouts="${ROLLOUTS:-50}"
  time_limit="${SBATCH_TIME:-03:50:00}"
  job_prefix="pi05-food50-seed${ENVIRONMENT_SEED}"
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
[[ "${ENVIRONMENT_SEED}" =~ ^(0|[1-9][0-9]*)$ ]] \
  || { echo "ENVIRONMENT_SEED must be a non-negative integer" >&2; exit 2; }
(( ENVIRONMENT_SEED <= 4294967295 )) \
  || { echo "ENVIRONMENT_SEED must be at most 4294967295" >&2; exit 2; }
[[ -f "${SBATCH_SCRIPT}" ]] || { echo "Missing sbatch script: ${SBATCH_SCRIPT}" >&2; exit 2; }
mkdir -p "${SBATCH_LOG_ROOT}" "$(dirname "${SUBMISSION_MANIFEST}")"
exec 9>"${SUBMISSION_MANIFEST}.lock"
flock -n 9 || { echo "Another submitter holds ${SUBMISSION_MANIFEST}.lock" >&2; exit 4; }
expected_header=$'job_id\tmode\ttask\trollouts\tenvironment_seed\trun_namespace\tsubmitted_at\tbatch_script_sha256\tsubmission_argv_sha256\tprovenance_dir'
if [[ ! -e "${SUBMISSION_MANIFEST}" ]]; then
  printf '%s\n' "${expected_header}" > "${SUBMISSION_MANIFEST}"
else
  [[ "$(head -n 1 "${SUBMISSION_MANIFEST}")" == "${expected_header}" ]] \
    || { echo "Submission manifest header mismatch" >&2; exit 2; }
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
  export_vars="PATH=${PATH},HOME=${HOME},POLARIS_DIR=${POLARIS_DIR},EXPECTED_POLARIS_COMMIT=${POLARIS_COMMIT},POLARIS_ENVIRONMENT=${task},ROLLOUTS=${rollouts},ENVIRONMENT_SEED=${ENVIRONMENT_SEED},RUN_NAMESPACE=${RUN_NAMESPACE}"
  sbatch_argv=(sbatch --parsable \
    --job-name="${job_name}" \
    --time="${time_limit}" \
    --output="${SBATCH_LOG_ROOT}/%x-%j.out" \
    --export="${export_vars}" \
    "${SBATCH_SCRIPT}")
  printf -v submission_argv '%q ' "${sbatch_argv[@]}"
  job_id="$("${sbatch_argv[@]}")"
  [[ "${job_id}" =~ ^[0-9]+$ ]] || { echo "Invalid job ID: ${job_id}" >&2; exit 3; }
  provenance_dir="$(dirname "${SUBMISSION_MANIFEST}")/submission_provenance/job_${job_id}"
  batch_script_path="${provenance_dir}/batch_script.sbatch"
  submission_argv_path="${provenance_dir}/submission_argv.sh"
  capture_submission_provenance() {
    mkdir -p "$(dirname "${provenance_dir}")" || return
    mkdir -m 0755 "${provenance_dir}" || return
    scontrol write batch_script "${job_id}" "${batch_script_path}" || return
    printf '%s\n' "${submission_argv}" > "${submission_argv_path}" || return
    chmod 0444 "${batch_script_path}" "${submission_argv_path}" || return
    batch_script_sha256="$(sha256sum "${batch_script_path}" | awk '{print $1}')" || return
    submission_argv_sha256="$(sha256sum "${submission_argv_path}" | awk '{print $1}')" || return
  }
  if ! capture_submission_provenance; then
    scancel "${job_id}" || true
    echo "Canceled job ${job_id}: failed to preserve submission provenance" >&2
    exit 5
  fi
  job_ids+=("${job_id}")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${job_id}" "${MODE}" "${task}" "${rollouts}" "${ENVIRONMENT_SEED}" "${RUN_NAMESPACE}" "$(date -Iseconds)" \
    "${batch_script_sha256}" "${submission_argv_sha256}" "${provenance_dir}" \
    | tee -a "${SUBMISSION_MANIFEST}"
done

printf 'submitted_job_ids=%s\n' "${job_ids[*]}"
printf 'submission_manifest=%s\n' "${SUBMISSION_MANIFEST}"
