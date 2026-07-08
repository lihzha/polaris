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
POLARIS_COMMIT="$(git -C "${POLARIS_DIR}" rev-parse HEAD)"

[[ "${ROLLOUTS}" =~ ^[1-9][0-9]*$ ]] || { echo "ROLLOUTS must be positive" >&2; exit 2; }
[[ "${ENVIRONMENT_SEED}" =~ ^(0|[1-9][0-9]*)$ ]] \
  || { echo "ENVIRONMENT_SEED must be a non-negative integer" >&2; exit 2; }
[[ -f "${SBATCH_SCRIPT}" ]] || { echo "Missing sbatch script: ${SBATCH_SCRIPT}" >&2; exit 2; }
git -C "${POLARIS_DIR}" diff-index --quiet HEAD -- \
  || { echo "PolaRiS source has tracked modifications" >&2; exit 2; }
mkdir -p "${SBATCH_LOG_ROOT}"

export_vars="PATH=${PATH},HOME=${HOME},POLARIS_DIR=${POLARIS_DIR},EXPECTED_POLARIS_COMMIT=${POLARIS_COMMIT},RUN_NAMESPACE_BASE=${RUN_NAMESPACE_BASE},ROLLOUTS=${ROLLOUTS},ENVIRONMENT_SEED=${ENVIRONMENT_SEED}"
job_id="$(sbatch --parsable \
  --output="${SBATCH_LOG_ROOT}/%x-%j.out" \
  --export="${export_vars}" \
  "${SBATCH_SCRIPT}")"
[[ "${job_id}" =~ ^[0-9]+$ ]] || { echo "Invalid job ID: ${job_id}" >&2; exit 3; }
printf 'job_id=%s\nrun_namespace_base=%s\n' "${job_id}" "${RUN_NAMESPACE_BASE}"
