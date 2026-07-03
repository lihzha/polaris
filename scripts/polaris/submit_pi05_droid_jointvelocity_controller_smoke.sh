#!/usr/bin/env bash

# Submit one pinned L40S controller-only smoke. This never launches a model or checkpoint.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLARIS_DIR="${POLARIS_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
NFS_ROOT="${NFS_ROOT:-/lustre/fsw/portfolios/nvr/users/lzha}"
POLARIS_DATA_DIR="${POLARIS_DATA_DIR:-${NFS_ROOT}/data/PolaRiS-Hub}"
POLARIS_PYXIS_IMAGE="${POLARIS_PYXIS_IMAGE:-${NFS_ROOT}/cache/polaris/polaris-eval-cuda13-fd00a51.sqsh}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${SCRIPT_DIR}/l40s_pi05_droid_jointvelocity_controller_smoke.sbatch}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${NFS_ROOT}/results/polaris-pi05-jointvelocity/controller-smoke}"
LOG_ROOT="${LOG_ROOT:-${NFS_ROOT}/slurm_logs/polaris-pi05-jointvelocity}"
EXPECTED_IMAGE_SHA256=ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a
EXPECTED_GRIPPER_DRIVE_PROFILE=implicit_gripper_physx_velocity_limit5_followers5_every_reset_cuda_actuator_cpu_static_physx_v1

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_normalized_absolute_path() {
  local field="$1"
  local value="$2"
  local normalized
  [[ "${value}" == /* && "${value}" != //* ]] \
    || die "${field} must be an absolute path with one leading slash"
  normalized="$(realpath -sm -- "${value}")" \
    || die "Unable to normalize ${field}"
  [[ "${value}" == "${normalized}" ]] \
    || die "${field} must use one normalized absolute path spelling"
}

command -v sbatch >/dev/null || die "sbatch is required; run this on l401/l402/l403"
require_normalized_absolute_path POLARIS_DIR "${POLARIS_DIR}"
[[ ! -L "${POLARIS_DIR}" ]] || die "POLARIS_DIR must not be a symlink"
POLARIS_DIR_RESOLVED="$(realpath "${POLARIS_DIR}")"
for path_contract in \
  "NFS_ROOT=${NFS_ROOT}" \
  "POLARIS_DATA_DIR=${POLARIS_DATA_DIR}" \
  "POLARIS_PYXIS_IMAGE=${POLARIS_PYXIS_IMAGE}" \
  "SBATCH_SCRIPT=${SBATCH_SCRIPT}" \
  "OUTPUT_ROOT=${OUTPUT_ROOT}" \
  "LOG_ROOT=${LOG_ROOT}"; do
  require_normalized_absolute_path \
    "${path_contract%%=*}" "${path_contract#*=}"
done
[[ -d "${POLARIS_DIR}/.git" && ! -L "${POLARIS_DIR}/.git" ]] \
  || die "POLARIS_DIR must be a standalone clone with an in-root .git directory"
git_dir="$(git -C "${POLARIS_DIR}" rev-parse --absolute-git-dir)"
git_common_dir="$(git -C "${POLARIS_DIR}" rev-parse --path-format=absolute --git-common-dir)"
[[ "${git_dir}" == "${POLARIS_DIR_RESOLVED}/.git" && \
  "${git_common_dir}" == "${POLARIS_DIR_RESOLVED}/.git" ]] \
  || die "POLARIS_DIR Git metadata must be wholly contained in its .git directory"
[[ "$(git -C "${POLARIS_DIR}" rev-parse --abbrev-ref HEAD)" == HEAD ]] \
  || die "POLARIS_DIR must be checked out at a detached HEAD"
[[ "$(git -C "${POLARIS_DIR}" rev-parse --show-toplevel)" == \
  "${POLARIS_DIR_RESOLVED}" ]] \
  || die "POLARIS_DIR must name the exact Git root"
POLARIS_COMMIT="$(git -C "${POLARIS_DIR}" rev-parse HEAD)"
[[ "${POLARIS_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || die "Malformed PolaRiS commit"
[[ -z "$(git -C "${POLARIS_DIR}" status --porcelain=v1 --untracked-files=all)" ]] \
  || die "PolaRiS checkout must be completely clean"
git -C "${POLARIS_DIR}" ls-files --error-unmatch \
  "scripts/polaris/l40s_pi05_droid_jointvelocity_controller_smoke.sbatch" \
  "scripts/polaris/finalize_pi05_droid_jointvelocity_controller_smoke.py" \
  "scripts/smoke_joint_velocity_controller.py" >/dev/null \
  || die "Controller-smoke scaffold must be committed"
[[ -f "${SBATCH_SCRIPT}" && ! -L "${SBATCH_SCRIPT}" ]] || die "Missing sbatch script"
[[ -f "${POLARIS_PYXIS_IMAGE}" && ! -L "${POLARIS_PYXIS_IMAGE}" ]] \
  || die "Missing pinned Pyxis image"
[[ "$(sha256sum "${POLARIS_PYXIS_IMAGE}" | awk '{print $1}')" == \
    "${EXPECTED_IMAGE_SHA256}" ]] || die "Pyxis image SHA-256 mismatch"
[[ -d "${POLARIS_DATA_DIR}" && ! -L "${POLARIS_DATA_DIR}" ]] \
  || die "Missing regular PolaRiS-Hub data root"

RUN_NAMESPACE="${RUN_NAMESPACE:-$(date -u +%Y%m%dT%H%M%SZ)-${POLARIS_COMMIT:0:12}}"
[[ "${RUN_NAMESPACE}" =~ ^[A-Za-z0-9._-]+$ ]] || die "Unsafe RUN_NAMESPACE"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_NAMESPACE}}"
require_normalized_absolute_path RUN_DIR "${RUN_DIR}"
[[ ! -e "${RUN_DIR}" && ! -L "${RUN_DIR}" ]] \
  || die "Refusing existing RUN_DIR: ${RUN_DIR}"
mkdir -p "${LOG_ROOT}"
mkdir "${RUN_DIR}"

job_id="$(sbatch --parsable \
  --output="${LOG_ROOT}/pi05_jv_ctrl_smoke-%j.out" \
  --export="ALL,NFS_ROOT=${NFS_ROOT},POLARIS_DIR=${POLARIS_DIR},EXPECTED_POLARIS_COMMIT=${POLARIS_COMMIT},EXPECTED_GRIPPER_DRIVE_PROFILE=${EXPECTED_GRIPPER_DRIVE_PROFILE},POLARIS_DATA_DIR=${POLARIS_DATA_DIR},POLARIS_PYXIS_IMAGE=${POLARIS_PYXIS_IMAGE},RUN_DIR=${RUN_DIR}" \
  "${SBATCH_SCRIPT}")"
[[ "${job_id}" =~ ^[0-9]+$ ]] || die "Invalid sbatch job ID: ${job_id}"
printf 'submitted_job_id=%s\n' "${job_id}"

(
export SUBMISSION_RECORD="${RUN_DIR}/submission-${job_id}.json"
export JOB_ID="${job_id}"
export POLARIS_COMMIT POLARIS_DIR RUN_DIR SBATCH_SCRIPT
export POLARIS_PYXIS_IMAGE POLARIS_DATA_DIR EXPECTED_GRIPPER_DRIVE_PROFILE
/usr/bin/python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

path = Path(os.environ["SUBMISSION_RECORD"])
job_script = Path(os.environ["SBATCH_SCRIPT"])
polaris_dir = Path(os.environ["POLARIS_DIR"])
run_dir = Path(os.environ["RUN_DIR"])
container_image = Path(os.environ["POLARIS_PYXIS_IMAGE"])
data_dir = Path(os.environ["POLARIS_DATA_DIR"])
value = {
    "schema_version": 1,
    "scope": "controller_only_no_model_or_checkpoint",
    "promotion": "forbidden_without_separate_checkpoint_canary",
    "job_id": int(os.environ["JOB_ID"]),
    "polaris_commit": os.environ["POLARIS_COMMIT"],
    "expected_gripper_drive_profile": os.environ[
        "EXPECTED_GRIPPER_DRIVE_PROFILE"
    ],
    "polaris_dir": str(polaris_dir),
    "polaris_dir_resolved": str(polaris_dir.resolve()),
    "run_dir": str(run_dir),
    "run_dir_resolved": str(run_dir.resolve()),
    "sbatch_script": str(job_script),
    "sbatch_script_resolved": str(job_script.resolve()),
    "sbatch_script_sha256": hashlib.sha256(job_script.read_bytes()).hexdigest(),
    "container_image": str(container_image),
    "container_image_resolved": str(container_image.resolve()),
    "polaris_data_dir": str(data_dir),
    "polaris_data_dir_resolved": str(data_dir.resolve()),
}
payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
try:
    os.write(descriptor, payload)
    os.fsync(descriptor)
    os.fchmod(descriptor, 0o444)
finally:
    os.close(descriptor)
directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY
)

printf 'job_id=%s\nrun_dir=%s\nscope=controller_only_no_model_or_checkpoint\n' \
  "${job_id}" "${RUN_DIR}"
