#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLARIS_DIR="${POLARIS_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${SCRIPT_DIR}/l40s_pi05_jointpos_seed_repeat.sbatch}"
NFS_ROOT="${NFS_ROOT:-/lustre/fsw/portfolios/nvr/users/lzha}"
RUN_NAMESPACE_BASE="${RUN_NAMESPACE_BASE:-pi05-jointpos-seed-repeat-$(date -u +%Y%m%dT%H%M%SZ)}"
SBATCH_LOG_ROOT="${SBATCH_LOG_ROOT:-${NFS_ROOT}/slurm_logs/polaris-pi05/${RUN_NAMESPACE_BASE}}"
ROLLOUTS="${ROLLOUTS:-2}"
ENVIRONMENT_SEED="${ENVIRONMENT_SEED:-0}"
SBATCH_TIME="${SBATCH_TIME:-02:00:00}"
POLARIS_COMMIT="$(git -C "${POLARIS_DIR}" rev-parse HEAD)"

[[ "${ROLLOUTS}" =~ ^[1-9][0-9]*$ ]] || { echo "ROLLOUTS must be positive" >&2; exit 2; }
[[ "${ENVIRONMENT_SEED}" =~ ^(0|[1-9][0-9]*)$ ]] \
  || { echo "ENVIRONMENT_SEED must be a non-negative integer" >&2; exit 2; }
[[ -f "${SBATCH_SCRIPT}" ]] || { echo "Missing sbatch script: ${SBATCH_SCRIPT}" >&2; exit 2; }
[[ -z "$(git -C "${POLARIS_DIR}" status --porcelain=v1 --untracked-files=all)" ]] \
  || { echo "PolaRiS source has tracked or untracked modifications" >&2; exit 2; }
mkdir -p "${SBATCH_LOG_ROOT}"

export_vars="PATH=${PATH},HOME=${HOME},POLARIS_DIR=${POLARIS_DIR},EXPECTED_POLARIS_COMMIT=${POLARIS_COMMIT},RUN_NAMESPACE_BASE=${RUN_NAMESPACE_BASE},ROLLOUTS=${ROLLOUTS},ENVIRONMENT_SEED=${ENVIRONMENT_SEED}"
sbatch_argv=(sbatch --parsable \
  --time="${SBATCH_TIME}" \
  --output="${SBATCH_LOG_ROOT}/%x-%j.out" \
  --export="${export_vars}" \
  "${SBATCH_SCRIPT}")
printf -v submission_argv '%q ' "${sbatch_argv[@]}"
job_id="$("${sbatch_argv[@]}")"
[[ "${job_id}" =~ ^[0-9]+$ ]] || { echo "Invalid job ID: ${job_id}" >&2; exit 3; }
provenance_dir="${NFS_ROOT}/results/polaris-pi05/${RUN_NAMESPACE_BASE}/submission_provenance/job_${job_id}"
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
printf 'job_id=%s\nrun_namespace_base=%s\nbatch_script_sha256=%s\nsubmission_argv_sha256=%s\nprovenance_dir=%s\n' \
  "${job_id}" "${RUN_NAMESPACE_BASE}" "${batch_script_sha256}" \
  "${submission_argv_sha256}" "${provenance_dir}"
