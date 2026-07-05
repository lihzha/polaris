#!/usr/bin/env bash

# Submit one ordinary, fresh official pi0.5-DROID corrected position canary.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLARIS_DIR="${POLARIS_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
NFS_ROOT="${NFS_ROOT:-/lustre/fsw/portfolios/nvr/users/lzha}"
: "${OPENPI_DIR:?Set the frozen OpenPI bd70 checkout with exact venv}"
: "${DROID_DIR:?Set the frozen official DROID 33ae6a checkout}"
: "${POSITION_CONTROLLER_ATTESTATION:?Set the immutable position-smoke attestation}"
: "${EXPECTED_POSITION_CONTROLLER_ATTESTATION_SHA256:?Set its exact SHA-256}"

NFS_ROOT="$(realpath -m -- "${NFS_ROOT}")"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${SCRIPT_DIR}/l40s_pi05_droid_position_canary.sbatch}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${NFS_ROOT}/results/polaris-pi05-position/canary}"
LOG_ROOT="${LOG_ROOT:-${NFS_ROOT}/slurm_logs/polaris-pi05-position}"
POLARIS_PYXIS_IMAGE="${POLARIS_PYXIS_IMAGE:-${NFS_ROOT}/cache/polaris/polaris-eval-cuda13-fd00a51.sqsh}"
POLARIS_DATA_DIR="${POLARIS_DATA_DIR:-${NFS_ROOT}/data/PolaRiS-Hub}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${NFS_ROOT}/cache/openpi-pi05-droid-native-v1}"
HOST_MEDIA_TOOLS_ROOT=/lustre/fs11/portfolios/nvr/projects/nvr_lpr_rvp/users/lzha/cache/polaris/host-media-tools/ffmpeg-7.0.2-static-amd64-abda8d77ce830914
HOST_FFPROBE_PATH="${HOST_FFPROBE_PATH:-${HOST_MEDIA_TOOLS_ROOT}/ffprobe}"
HOST_FFMPEG_PATH="${HOST_FFMPEG_PATH:-${HOST_MEDIA_TOOLS_ROOT}/ffmpeg}"
EXPECTED_HOST_MEDIA_TOOLS_MANIFEST_SHA256=09d95a1f28e9e9af1e172439806ca9c2d6b19dd661f9f5f4ee7f51185cb99be5
EXPECTED_HOST_FFPROBE_SHA256=4f231a1960d83e403d08f7971e271707bec278a9ae18e21b8b5b03186668450d
EXPECTED_HOST_FFMPEG_SHA256=e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99
CHECKPOINT_URI=gs://openpi-assets/checkpoints/pi05_droid
CHECKPOINT_MANIFEST="${SCRIPT_DIR}/pi05_droid_native_gcs_manifest.tsv"
CANARY_PROFILE=openpi_pi05_droid_fresh_jointdelta_position_polaris_canary_v1
PROTOCOL=polaris-native-droid-freshq-delta0p2-position-h8-canary1-v1

die() { echo "ERROR: $*" >&2; exit 2; }

command -v sbatch >/dev/null || die "Run this submitter on l401/l402/l403"
for forbidden in CONTROLLER_COMPLETION EXPECTED_CONTROLLER_COMPLETION_SHA256 \
  ALL_SIX_CONTROLLER_COMPLETION EXPECTED_ALL_SIX_COMPLETION_SHA256 \
  EXPECTED_ALL_SIX_PROFILE; do
  [[ -z "${!forbidden+x}" ]] \
    || die "Old direct-rad/s controller gate is forbidden: ${forbidden}"
done
[[ -z "${PORT+x}" ]] || die "Ambient PORT is forbidden"
[[ -z "${RESUME_FROM_TASK_DIR:-}" ]] || die "Position canary forbids resume"
[[ "${EXPECTED_POSITION_CONTROLLER_ATTESTATION_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "Malformed position-controller attestation SHA-256"

for directory in POLARIS_DIR OPENPI_DIR DROID_DIR POLARIS_DATA_DIR; do
  value="${!directory}"
  [[ -d "${value}" && ! -L "${value}" ]] || die "${directory} is invalid"
  printf -v "${directory}" '%s' "$(realpath -e -- "${value}")"
done
for file_var in SBATCH_SCRIPT POLARIS_PYXIS_IMAGE POSITION_CONTROLLER_ATTESTATION \
  HOST_FFPROBE_PATH HOST_FFMPEG_PATH; do
  value="${!file_var}"
  [[ -f "${value}" && ! -L "${value}" ]] || die "${file_var} is invalid"
  printf -v "${file_var}" '%s' "$(realpath -e -- "${value}")"
done
OUTPUT_ROOT="$(realpath -m -- "${OUTPUT_ROOT}")"
LOG_ROOT="$(realpath -m -- "${LOG_ROOT}")"
OPENPI_DATA_HOME="$(realpath -m -- "${OPENPI_DATA_HOME}")"

[[ -d "${POLARIS_DIR}/.git" && ! -L "${POLARIS_DIR}/.git" ]] \
  || die "PolaRiS submit source must be a standalone clone"
[[ "$(git -C "${POLARIS_DIR}" rev-parse --show-toplevel)" == "${POLARIS_DIR}" \
  && "$(git -C "${POLARIS_DIR}" rev-parse --absolute-git-dir)" == "${POLARIS_DIR}/.git" \
  && "$(git -C "${POLARIS_DIR}" rev-parse --path-format=absolute --git-common-dir)" == "${POLARIS_DIR}/.git" ]] \
  || die "PolaRiS standalone Git layout mismatch"
[[ "$(git -C "${POLARIS_DIR}" rev-parse --abbrev-ref HEAD)" == HEAD ]] \
  || die "PolaRiS submit source must use detached HEAD"
POLARIS_COMMIT="$(git -C "${POLARIS_DIR}" rev-parse HEAD)"
[[ "${POLARIS_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || die "Malformed PolaRiS commit"
[[ -z "$(git -C "${POLARIS_DIR}" status --porcelain=v1 --untracked-files=all)" ]] \
  || die "PolaRiS submit source must be completely clean"
[[ -x "${OPENPI_DIR}/.venv/bin/python" ]] || die "Exact OpenPI venv is unavailable"
[[ "${SBATCH_SCRIPT}" == "${POLARIS_DIR}/scripts/polaris/l40s_pi05_droid_position_canary.sbatch" ]] \
  || die "SBATCH_SCRIPT must be the committed position canary entrypoint"
HOST_PYTHONPATH="${POLARIS_DIR}/src:${SCRIPT_DIR}:${OPENPI_DIR}/packages/openpi-client/src"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${HOST_PYTHONPATH}" \
  "${OPENPI_DIR}/.venv/bin/python" \
  "${SCRIPT_DIR}/capture_pi05_droid_position_environment.py" \
  --openpi-dir "${OPENPI_DIR}" --runtime-package-preflight >/dev/null

RUN_NAMESPACE="${RUN_NAMESPACE:-$(date -u +%Y%m%dT%H%M%SZ)-${POLARIS_COMMIT:0:12}}"
[[ "${RUN_NAMESPACE}" =~ ^[A-Za-z0-9._-]+$ ]] || die "Unsafe RUN_NAMESPACE"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_NAMESPACE}}"
RUN_DIR="$(realpath -m -- "${RUN_DIR}")"
[[ ! -e "${RUN_DIR}" && ! -L "${RUN_DIR}" ]] \
  || die "Refusing existing fresh-attempt directory: ${RUN_DIR}"
mkdir -p "${LOG_ROOT}" "$(dirname "${RUN_DIR}")"
mkdir "${RUN_DIR}"

runtime_args=(
  --polaris-repo "${POLARIS_DIR}"
  --expected-polaris-commit "${POLARIS_COMMIT}"
  --openpi-dir "${OPENPI_DIR}"
  --droid-dir "${DROID_DIR}"
  --position-controller-attestation "${POSITION_CONTROLLER_ATTESTATION}"
  --expected-position-controller-attestation-sha256 "${EXPECTED_POSITION_CONTROLLER_ATTESTATION_SHA256}"
  --container-image "${POLARIS_PYXIS_IMAGE}"
  --data-dir "${POLARIS_DATA_DIR}"
  --expected-host-media-tools-manifest-sha256 "${EXPECTED_HOST_MEDIA_TOOLS_MANIFEST_SHA256}"
  --host-ffprobe-path "${HOST_FFPROBE_PATH}"
  --expected-host-ffprobe-sha256 "${EXPECTED_HOST_FFPROBE_SHA256}"
  --host-ffmpeg-path "${HOST_FFMPEG_PATH}"
  --expected-host-ffmpeg-sha256 "${EXPECTED_HOST_FFMPEG_SHA256}"
)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${HOST_PYTHONPATH}" \
  "${OPENPI_DIR}/.venv/bin/python" "${SCRIPT_DIR}/finalize_pi05_droid_position_eval.py" \
  preflight "${runtime_args[@]}" --output "${RUN_DIR}/preflight.json" >/dev/null
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${HOST_PYTHONPATH}" \
  "${OPENPI_DIR}/.venv/bin/python" "${SCRIPT_DIR}/finalize_pi05_droid_position_eval.py" \
  write-resolved-contract --output "${RUN_DIR}/resolved_contract.json" >/dev/null

OPENPI_PYTHONPATH="${OPENPI_DIR}/src:${OPENPI_DIR}/packages/openpi-client/src"
checkpoint_cache="$({
  OPENPI_DATA_HOME="${OPENPI_DATA_HOME}" PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${OPENPI_PYTHONPATH}" "${OPENPI_DIR}/.venv/bin/python" \
    - "${CHECKPOINT_URI}" <<'PY'
import sys
from openpi.shared.download import maybe_download
print(maybe_download(sys.argv[1]))
PY
} | tail -n 1)"
[[ -d "${checkpoint_cache}" && ! -L "${checkpoint_cache}" ]] \
  || die "Official checkpoint did not resolve to one regular cache directory"
checkpoint_cache="$(realpath -e -- "${checkpoint_cache}")"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${HOST_PYTHONPATH}" \
  "${OPENPI_DIR}/.venv/bin/python" \
  "${SCRIPT_DIR}/verify_pi05_droid_position_checkpoint.py" snapshot \
  --source "${checkpoint_cache}" --destination "${RUN_DIR}/checkpoint_snapshot" \
  --manifest "${CHECKPOINT_MANIFEST}" \
  --output "${RUN_DIR}/checkpoint_snapshot_creation.json"

export_vars="ALL,NFS_ROOT=${NFS_ROOT},POLARIS_DIR=${POLARIS_DIR},EXPECTED_POLARIS_COMMIT=${POLARIS_COMMIT},OPENPI_DIR=${OPENPI_DIR},DROID_DIR=${DROID_DIR},RUN_DIR=${RUN_DIR},POLARIS_PYXIS_IMAGE=${POLARIS_PYXIS_IMAGE},POLARIS_DATA_DIR=${POLARIS_DATA_DIR},POSITION_CONTROLLER_ATTESTATION=${POSITION_CONTROLLER_ATTESTATION},EXPECTED_POSITION_CONTROLLER_ATTESTATION_SHA256=${EXPECTED_POSITION_CONTROLLER_ATTESTATION_SHA256},HOST_FFPROBE_PATH=${HOST_FFPROBE_PATH},HOST_FFMPEG_PATH=${HOST_FFMPEG_PATH}"
job_id="$(sbatch --parsable --output="${LOG_ROOT}/pi05_pos_canary-%j.out" \
  --export="${export_vars}" "${SBATCH_SCRIPT}")"
[[ "${job_id}" =~ ^[0-9]+$ ]] || die "Invalid Slurm job ID: ${job_id}"

export SUBMISSION_RECORD="${RUN_DIR}/submission-${job_id}.json" job_id RUN_DIR
export POLARIS_DIR POLARIS_COMMIT OPENPI_DIR DROID_DIR SBATCH_SCRIPT
export POLARIS_PYXIS_IMAGE POLARIS_DATA_DIR POSITION_CONTROLLER_ATTESTATION
export EXPECTED_POSITION_CONTROLLER_ATTESTATION_SHA256 HOST_FFPROBE_PATH HOST_FFMPEG_PATH
export EXPECTED_HOST_MEDIA_TOOLS_MANIFEST_SHA256 EXPECTED_HOST_FFPROBE_SHA256
export EXPECTED_HOST_FFMPEG_SHA256 CANARY_PROFILE PROTOCOL
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${POLARIS_DIR}/src" \
  "${OPENPI_DIR}/.venv/bin/python" - <<'PY'
import os
from pathlib import Path
from polaris.pi05_droid_native_eval_contract import file_sha256, publish_immutable_json
publish_immutable_json(
    Path(os.environ["SUBMISSION_RECORD"]),
    {
        "schema_version": 1,
        "profile": os.environ["CANARY_PROFILE"],
        "protocol": os.environ["PROTOCOL"],
        "job_id": int(os.environ["job_id"]),
        "run_dir": os.environ["RUN_DIR"],
        "polaris_dir": os.environ["POLARIS_DIR"],
        "polaris_commit": os.environ["POLARIS_COMMIT"],
        "openpi_dir": os.environ["OPENPI_DIR"],
        "droid_dir": os.environ["DROID_DIR"],
        "sbatch_script": os.environ["SBATCH_SCRIPT"],
        "sbatch_script_sha256": file_sha256(Path(os.environ["SBATCH_SCRIPT"])),
        "container_image": os.environ["POLARIS_PYXIS_IMAGE"],
        "polaris_data_dir": os.environ["POLARIS_DATA_DIR"],
        "position_controller_attestation": os.environ["POSITION_CONTROLLER_ATTESTATION"],
        "position_controller_attestation_sha256": os.environ[
            "EXPECTED_POSITION_CONTROLLER_ATTESTATION_SHA256"
        ],
        "host_media": {
            "manifest_sha256": os.environ["EXPECTED_HOST_MEDIA_TOOLS_MANIFEST_SHA256"],
            "ffprobe_path": os.environ["HOST_FFPROBE_PATH"],
            "ffprobe_sha256": os.environ["EXPECTED_HOST_FFPROBE_SHA256"],
            "ffmpeg_path": os.environ["HOST_FFMPEG_PATH"],
            "ffmpeg_sha256": os.environ["EXPECTED_HOST_FFMPEG_SHA256"],
        },
        "checkpoint_snapshot": str(Path(os.environ["RUN_DIR"]) / "checkpoint_snapshot"),
        "fresh_attempt_no_resume": True,
        "task": "DROID-FoodBussing",
        "rollouts": 1,
    },
)
PY

printf 'submitted_job_id=%s\nrun_dir=%s\nprotocol=%s\n' \
  "${job_id}" "${RUN_DIR}" "${PROTOCOL}"
