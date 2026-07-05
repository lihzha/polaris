#!/usr/bin/env bash
set -Eeuo pipefail

: "${POLARIS_DIR:?Set one frozen standalone PolaRiS checkout}"
: "${EXPECTED_POLARIS_COMMIT:?Set the exact committed PolaRiS revision}"
: "${RUN_DIR:?Set one fresh pre-created result directory}"
NFS_ROOT_REQUESTED="${NFS_ROOT:-/lustre/fsw/portfolios/nvr/users/lzha}"
POLARIS_DIR_REQUESTED="${POLARIS_DIR}"
RUN_DIR_REQUESTED="${RUN_DIR}"

die() { echo "ERROR: $*" >&2; exit 2; }
canonical_existing() {
  local field="$1" value="$2"
  [[ "${value}" == /* && "${value}" != //* ]] || die "${field} must be absolute"
  realpath -e -- "${value}" || die "cannot resolve existing ${field}"
}

[[ "${EXPECTED_POLARIS_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || die "commit must be full lowercase SHA"
NFS_ROOT="$(canonical_existing NFS_ROOT "${NFS_ROOT_REQUESTED}")"
POLARIS_DIR="$(canonical_existing POLARIS_DIR "${POLARIS_DIR_REQUESTED}")"
RUN_DIR="$(canonical_existing RUN_DIR "${RUN_DIR_REQUESTED}")"
LOG_DIR="${NFS_ROOT}/slurm_logs/polaris-pi05-position"
[[ ! -L "${POLARIS_DIR}" && -d "${POLARIS_DIR}/.git" && ! -L "${POLARIS_DIR}/.git" ]] \
  || die "standalone checkout required"
[[ -d "${RUN_DIR}" && ! -L "${RUN_DIR}" ]] || die "RUN_DIR must be one regular directory"
[[ -z "$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
  || die "RUN_DIR must be empty"
repo_root="$(realpath "${POLARIS_DIR}")"
[[ "$(git -C "${POLARIS_DIR}" rev-parse --show-toplevel)" == "${repo_root}" ]] \
  || die "POLARIS_DIR must name the exact Git root"
[[ "$(git -C "${POLARIS_DIR}" rev-parse --absolute-git-dir)" == "${repo_root}/.git" ]] \
  || die "Git metadata must be checkout-local"
[[ "$(git -C "${POLARIS_DIR}" rev-parse --path-format=absolute --git-common-dir)" == "${repo_root}/.git" ]] \
  || die "Git common directory must be checkout-local"
[[ "$(git -C "${POLARIS_DIR}" rev-parse --abbrev-ref HEAD)" == HEAD ]] \
  || die "detached HEAD required"
[[ "$(git -C "${POLARIS_DIR}" rev-parse HEAD)" == "${EXPECTED_POLARIS_COMMIT}" ]] \
  || die "commit mismatch"
[[ -z "$(git -C "${POLARIS_DIR}" status --porcelain=v1 --untracked-files=all)" ]] \
  || die "checkout must be completely clean"
mkdir -p "${LOG_DIR}"
[[ -d "${LOG_DIR}" && ! -L "${LOG_DIR}" ]] || die "Slurm log directory is invalid"

job_id="$(sbatch --parsable \
  --export=ALL,POLARIS_DIR="${POLARIS_DIR}",EXPECTED_POLARIS_COMMIT="${EXPECTED_POLARIS_COMMIT}",RUN_DIR="${RUN_DIR}" \
  "${POLARIS_DIR}/scripts/polaris/l40s_pi05_droid_position_controller_smoke.sbatch")"
[[ "${job_id}" =~ ^[0-9]+$ ]] || die "sbatch did not return one numeric job id"
printf '%s\n' "${job_id}"
