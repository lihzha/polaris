#!/usr/bin/env bash

# Submit one ordinary, fresh official pi0.5-DROID native-velocity canary job.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLARIS_DIR="${POLARIS_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
NFS_ROOT="${NFS_ROOT:-/lustre/fsw/portfolios/nvr/users/lzha}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${SCRIPT_DIR}/l40s_pi05_droid_native_jointvelocity_canary.sbatch}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${NFS_ROOT}/results/polaris-pi05-native/canary}"
LOG_ROOT="${LOG_ROOT:-${NFS_ROOT}/slurm_logs/polaris-pi05-native}"
POLARIS_PYXIS_IMAGE="${POLARIS_PYXIS_IMAGE:-${NFS_ROOT}/cache/polaris/polaris-eval-cuda13-fd00a51.sqsh}"
POLARIS_DATA_DIR="${POLARIS_DATA_DIR:-${NFS_ROOT}/data/PolaRiS-Hub}"

: "${CONTROLLER_COMPLETION:?Set job1098174 completion path}"
: "${EXPECTED_CONTROLLER_COMPLETION_SHA256:?Set exact job1098174 completion SHA-256}"
: "${ALL_SIX_CONTROLLER_COMPLETION:?Set accepted job1098682 all-six completion path}"
: "${EXPECTED_ALL_SIX_COMPLETION_SHA256:?Set exact all-six completion SHA-256}"
: "${EXPECTED_ALL_SIX_PROFILE:?Set independently reviewed all-six profile}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

command -v sbatch >/dev/null || die "Run this submitter on l401/l402/l403"
[[ -z "${PORT+x}" ]] \
  || die "Ambient PORT is forbidden; the canary server binds an OS-assigned port"
[[ -z "${RESUME_FROM_TASK_DIR:-}" ]] || die "Native flow canaries forbid prefix resume"
[[ "${EXPECTED_CONTROLLER_COMPLETION_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "Malformed job1098174 completion digest"
[[ "${EXPECTED_ALL_SIX_COMPLETION_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "Malformed all-six completion digest"
[[ -n "${EXPECTED_ALL_SIX_PROFILE}" ]] || die "Empty all-six profile"

[[ ! -L "${POLARIS_DIR}" ]] || die "POLARIS_DIR must not be a symlink"
POLARIS_DIR="$(realpath "${POLARIS_DIR}")"
[[ -d "${POLARIS_DIR}/.git" && ! -L "${POLARIS_DIR}/.git" ]] \
  || die "PolaRiS must be a standalone clone with in-root .git"
git_dir="$(git -C "${POLARIS_DIR}" rev-parse --absolute-git-dir)"
git_common_dir="$(git -C "${POLARIS_DIR}" rev-parse --path-format=absolute --git-common-dir)"
[[ "${git_dir}" == "${POLARIS_DIR}/.git" && "${git_common_dir}" == "${POLARIS_DIR}/.git" ]] \
  || die "PolaRiS Git metadata escaped the source root"
[[ "$(git -C "${POLARIS_DIR}" rev-parse --abbrev-ref HEAD)" == HEAD ]] \
  || die "PolaRiS submit checkout must use detached HEAD"
POLARIS_COMMIT="$(git -C "${POLARIS_DIR}" rev-parse HEAD)"
[[ "${POLARIS_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || die "Malformed PolaRiS commit"
[[ -z "$(git -C "${POLARIS_DIR}" status --porcelain=v1 --untracked-files=all)" ]] \
  || die "PolaRiS submit checkout must be completely clean"
[[ -f "${SBATCH_SCRIPT}" && ! -L "${SBATCH_SCRIPT}" ]] || die "Missing sbatch script"
[[ -x "${POLARIS_DIR}/third_party/openpi/.venv/bin/python" ]] \
  || die "Missing exact checkout-local OpenPI venv"

# The official OpenPI commit imports pytest.Cache from a module reached by the
# production server, despite declaring pytest only in its dev group.  Reject a
# no-dev environment without the exact hash-pinned runtime overlay before an
# L40S allocation is submitted.
PYTHONPATH="${POLARIS_DIR}/src:${SCRIPT_DIR}" \
  "${POLARIS_DIR}/third_party/openpi/.venv/bin/python" \
  "${SCRIPT_DIR}/capture_pi05_droid_native_environment.py" \
  --openpi-dir "${POLARIS_DIR}/third_party/openpi" \
  --runtime-package-preflight

# Block submission before any allocation until both external controller gates
# validate, including job1098682 all-six coupling and child lifecycle.
PYTHONPATH="${POLARIS_DIR}/src:${SCRIPT_DIR}" \
  "${POLARIS_DIR}/third_party/openpi/.venv/bin/python" \
  "${SCRIPT_DIR}/finalize_pi05_droid_native_jointvelocity_eval.py" preflight \
  --polaris-repo "${POLARIS_DIR}" \
  --controller-completion "${CONTROLLER_COMPLETION}" \
  --expected-controller-completion-sha256 "${EXPECTED_CONTROLLER_COMPLETION_SHA256}" \
  --all-six-controller-completion "${ALL_SIX_CONTROLLER_COMPLETION}" \
  --expected-all-six-completion-sha256 "${EXPECTED_ALL_SIX_COMPLETION_SHA256}" \
  --expected-all-six-profile "${EXPECTED_ALL_SIX_PROFILE}"

RUN_NAMESPACE="${RUN_NAMESPACE:-$(date -u +%Y%m%dT%H%M%SZ)-${POLARIS_COMMIT:0:12}}"
[[ "${RUN_NAMESPACE}" =~ ^[A-Za-z0-9._-]+$ ]] || die "Unsafe RUN_NAMESPACE"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_NAMESPACE}}"
[[ ! -e "${RUN_DIR}" && ! -L "${RUN_DIR}" ]] \
  || die "Refusing existing fresh-attempt directory: ${RUN_DIR}"
mkdir -p "${LOG_ROOT}"
mkdir -p "$(dirname "${RUN_DIR}")"
mkdir "${RUN_DIR}"

export_vars="ALL,NFS_ROOT=${NFS_ROOT},POLARIS_DIR=${POLARIS_DIR},EXPECTED_POLARIS_COMMIT=${POLARIS_COMMIT},RUN_DIR=${RUN_DIR},POLARIS_PYXIS_IMAGE=${POLARIS_PYXIS_IMAGE},POLARIS_DATA_DIR=${POLARIS_DATA_DIR},CONTROLLER_COMPLETION=${CONTROLLER_COMPLETION},EXPECTED_CONTROLLER_COMPLETION_SHA256=${EXPECTED_CONTROLLER_COMPLETION_SHA256},ALL_SIX_CONTROLLER_COMPLETION=${ALL_SIX_CONTROLLER_COMPLETION},EXPECTED_ALL_SIX_COMPLETION_SHA256=${EXPECTED_ALL_SIX_COMPLETION_SHA256},EXPECTED_ALL_SIX_PROFILE=${EXPECTED_ALL_SIX_PROFILE}"
job_id="$(sbatch --parsable \
  --output="${LOG_ROOT}/pi05_native_canary-%j.out" \
  --export="${export_vars}" \
  "${SBATCH_SCRIPT}")"
[[ "${job_id}" =~ ^[0-9]+$ ]] || die "Invalid Slurm job ID: ${job_id}"

export SUBMISSION_RECORD="${RUN_DIR}/submission-${job_id}.json" job_id RUN_DIR
export POLARIS_DIR POLARIS_COMMIT SBATCH_SCRIPT POLARIS_PYXIS_IMAGE POLARIS_DATA_DIR
PYTHONPATH="${POLARIS_DIR}/src" /usr/bin/python3 - <<'PY'
import os
from pathlib import Path
from polaris.pi05_droid_native_eval_contract import file_sha256, publish_immutable_json
publish_immutable_json(
    Path(os.environ["SUBMISSION_RECORD"]),
    {
        "schema_version": 1,
        "profile": "openpi_pi05_droid_native_jointvelocity_polaris_canary_v1",
        "job_id": int(os.environ["job_id"]),
        "run_dir": os.environ["RUN_DIR"],
        "polaris_dir": os.environ["POLARIS_DIR"],
        "polaris_commit": os.environ["POLARIS_COMMIT"],
        "sbatch_script": os.environ["SBATCH_SCRIPT"],
        "sbatch_script_sha256": file_sha256(Path(os.environ["SBATCH_SCRIPT"])),
        "container_image": os.environ["POLARIS_PYXIS_IMAGE"],
        "polaris_data_dir": os.environ["POLARIS_DATA_DIR"],
        "fresh_attempt_no_resume": True,
        "task": "DROID-FoodBussing",
        "rollouts": 1,
    },
)
PY

printf 'submitted_job_id=%s\nrun_dir=%s\nprofile=official-pi05-droid-native-canary\n' \
  "${job_id}" "${RUN_DIR}"
