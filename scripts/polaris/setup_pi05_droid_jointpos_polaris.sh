#!/usr/bin/env bash

# Prepare the exact official OpenPI runtime and checkpoint used by PolaRiS.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLARIS_DIR="${POLARIS_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
OPENPI_DIR="${OPENPI_DIR:-${POLARIS_DIR}/third_party/openpi}"
OPENPI_COMMIT="${OPENPI_COMMIT:-bd70b8f4011e85b3f3b0f039f12113f78718e7bf}"
CHECKPOINT_URI="${CHECKPOINT_URI:-gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris}"
EXPECTED_CHECKPOINT_URI="gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris"
EXPECTED_NORM_SHA256="${EXPECTED_NORM_SHA256:-57ce9956f9e07d65f8a8205aabec72d436a2c8927f53edb40c7a77b14a5a90c7}"
NFS_ROOT="${NFS_ROOT:-/lustre/fsw/portfolios/nvr/users/lzha}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${NFS_ROOT}/cache/openpi-polaris}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${NFS_ROOT}/cache/uv}"
SETUP_RECORD_DIR="${SETUP_RECORD_DIR:-${NFS_ROOT}/results/polaris-pi05/setup}"
CHECKPOINT_MANIFEST="${CHECKPOINT_MANIFEST:-${SCRIPT_DIR}/pi05_droid_jointpos_polaris_gcs_manifest.tsv}"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-7abd0c2294d442d429a77655783232206b2b30d95c508d435503135a5523a11c}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

command -v uv >/dev/null 2>&1 || die "uv is required"
[[ "${CHECKPOINT_URI}" == "${EXPECTED_CHECKPOINT_URI}" ]] \
  || die "Unexpected checkpoint URI: ${CHECKPOINT_URI}"
git -C "${POLARIS_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
  || die "POLARIS_DIR is not a Git checkout: ${POLARIS_DIR}"
[[ -f "${OPENPI_DIR}/pyproject.toml" ]] || die "OpenPI submodule is not initialized: ${OPENPI_DIR}"
[[ -f "${CHECKPOINT_MANIFEST}" ]] || die "Missing checkpoint manifest: ${CHECKPOINT_MANIFEST}"
actual_manifest_sha256="$(sha256sum "${CHECKPOINT_MANIFEST}" | awk '{print $1}')"
[[ "${actual_manifest_sha256}" == "${EXPECTED_MANIFEST_SHA256}" ]] \
  || die "Checkpoint manifest SHA-256 mismatch: ${actual_manifest_sha256}"
actual_openpi_commit="$(git -C "${OPENPI_DIR}" rev-parse HEAD)"
[[ "${actual_openpi_commit}" == "${OPENPI_COMMIT}" ]] \
  || die "OpenPI commit ${actual_openpi_commit} does not match ${OPENPI_COMMIT}"

mkdir -p "${OPENPI_DATA_HOME}" "${UV_CACHE_DIR}" "${SETUP_RECORD_DIR}"

(
  cd "${OPENPI_DIR}"
  GIT_LFS_SKIP_SMUDGE=1 UV_CACHE_DIR="${UV_CACHE_DIR}" uv sync --frozen
  GIT_LFS_SKIP_SMUDGE=1 UV_CACHE_DIR="${UV_CACHE_DIR}" uv pip install -e .
)

OPENPI_DATA_HOME="${OPENPI_DATA_HOME}" PYTHONPATH="${OPENPI_DIR}/src" \
  "${OPENPI_DIR}/.venv/bin/python" - "${CHECKPOINT_URI}" "${SETUP_RECORD_DIR}/checkpoint_path.txt" <<'PY'
import pathlib
import sys

from openpi.shared.download import maybe_download

checkpoint = pathlib.Path(maybe_download(sys.argv[1])).resolve()
pathlib.Path(sys.argv[2]).write_text(str(checkpoint) + "\n")
print(checkpoint)
PY

checkpoint_path="$(<"${SETUP_RECORD_DIR}/checkpoint_path.txt")"
"${OPENPI_DIR}/.venv/bin/python" "${SCRIPT_DIR}/verify_pi05_checkpoint.py" \
  "${checkpoint_path}" "${CHECKPOINT_MANIFEST}" --full-md5 \
  --output "${SETUP_RECORD_DIR}/checkpoint_verification.json"
norm_stats="${checkpoint_path}/assets/droid/norm_stats.json"
[[ -s "${norm_stats}" ]] || die "Missing checkpoint-local norm stats: ${norm_stats}"
actual_norm_sha256="$(sha256sum "${norm_stats}" | awk '{print $1}')"
[[ "${actual_norm_sha256}" == "${EXPECTED_NORM_SHA256}" ]] \
  || die "Norm SHA-256 ${actual_norm_sha256} does not match ${EXPECTED_NORM_SHA256}"

OPENPI_DATA_HOME="${OPENPI_DATA_HOME}" PYTHONPATH="${OPENPI_DIR}/src" \
  "${OPENPI_DIR}/.venv/bin/python" - <<'PY'
from openpi.training import config

resolved = config.get_config("pi05_droid_jointpos_polaris")
assert resolved.model.action_horizon == 15
assert resolved.model.action_dim == 32
assert resolved.data.assets.asset_id == "droid"
print("resolved_config=pi05_droid_jointpos_polaris action_horizon=15 model_action_dim=32 asset_id=droid")
PY

{
  printf 'setup_completed_at=%s\n' "$(date -Iseconds)"
  printf 'host=%s\n' "$(hostname)"
  printf 'polaris_dir=%s\n' "${POLARIS_DIR}"
  printf 'polaris_commit=%s\n' "$(git -C "${POLARIS_DIR}" rev-parse HEAD)"
  printf 'openpi_dir=%s\n' "${OPENPI_DIR}"
  printf 'openpi_commit=%s\n' "${actual_openpi_commit}"
  printf 'checkpoint_uri=%s\n' "${CHECKPOINT_URI}"
  printf 'checkpoint_path=%s\n' "${checkpoint_path}"
  printf 'checkpoint_manifest_sha256=%s\n' "${actual_manifest_sha256}"
  printf 'checkpoint_verification=%s\n' "${SETUP_RECORD_DIR}/checkpoint_verification.json"
  printf 'checkpoint_bytes=%s\n' "12434530837"
  printf 'norm_stats_sha256=%s\n' "${actual_norm_sha256}"
  printf 'openpi_data_home=%s\n' "${OPENPI_DATA_HOME}"
  printf 'uv_version=%s\n' "$(uv --version)"
} | tee "${SETUP_RECORD_DIR}/setup_manifest.env"
