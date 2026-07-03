#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLARIS_DIR="${POLARIS_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
NFS_ROOT="${NFS_ROOT:-/lustre/fsw/portfolios/nvr/users/lzha}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${SCRIPT_DIR}/l40s_pi05_native_all_six_controller_smoke.sbatch}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${NFS_ROOT}/results/polaris-pi05-native/all-six-controller-smoke}"
LOG_ROOT="${LOG_ROOT:-${NFS_ROOT}/slurm_logs/polaris-pi05-native-all-six}"
POLARIS_PYXIS_IMAGE="${POLARIS_PYXIS_IMAGE:-${NFS_ROOT}/cache/polaris/polaris-eval-cuda13-fd00a51.sqsh}"
POLARIS_DATA_DIR="${POLARIS_DATA_DIR:-${NFS_ROOT}/data/PolaRiS-Hub}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

command -v sbatch >/dev/null || die "run this submitter on l401/l402/l403"
[[ -d "${POLARIS_DIR}/.git" && ! -L "${POLARIS_DIR}/.git" ]] \
  || die "PolaRiS source must have an in-root .git"
POLARIS_DIR="$(realpath "${POLARIS_DIR}")"
[[ "$(git -C "${POLARIS_DIR}" rev-parse --abbrev-ref HEAD)" == HEAD ]] \
  || die "PolaRiS source must use detached HEAD"
POLARIS_COMMIT="$(git -C "${POLARIS_DIR}" rev-parse HEAD)"
[[ -z "$(git -C "${POLARIS_DIR}" status --porcelain=v1 --untracked-files=all)" ]] \
  || die "PolaRiS source is dirty"
[[ -f "${SBATCH_SCRIPT}" && ! -L "${SBATCH_SCRIPT}" ]] \
  || die "missing committed sbatch script"
[[ -x "${POLARIS_DIR}/third_party/openpi/.venv/bin/python" ]] \
  || die "missing exact checkout-local host Python"

RUN_NAMESPACE="${RUN_NAMESPACE:-$(date -u +%Y%m%dT%H%M%SZ)-${POLARIS_COMMIT:0:12}}"
[[ "${RUN_NAMESPACE}" =~ ^[A-Za-z0-9._-]+$ ]] || die "unsafe RUN_NAMESPACE"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_NAMESPACE}}"
[[ ! -e "${RUN_DIR}" && ! -L "${RUN_DIR}" ]] \
  || die "refusing existing attempt directory"
mkdir -p "${LOG_ROOT}" "$(dirname "${RUN_DIR}")"
mkdir "${RUN_DIR}"

job_id="$(sbatch --parsable \
  --output="${LOG_ROOT}/pi05-all-six-controller-%j.out" \
  --export="ALL,NFS_ROOT=${NFS_ROOT},POLARIS_DIR=${POLARIS_DIR},EXPECTED_POLARIS_COMMIT=${POLARIS_COMMIT},RUN_DIR=${RUN_DIR},POLARIS_PYXIS_IMAGE=${POLARIS_PYXIS_IMAGE},POLARIS_DATA_DIR=${POLARIS_DATA_DIR}" \
  "${SBATCH_SCRIPT}")"
[[ "${job_id}" =~ ^[0-9]+$ ]] || die "invalid Slurm job ID"
printf 'submitted_job_id=%s\nrun_dir=%s\nprofile=pi05-native-all-six-controller-smoke-v1\n' \
  "${job_id}" "${RUN_DIR}"
