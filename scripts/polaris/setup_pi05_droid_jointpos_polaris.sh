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
SETUP_RECORD_DIR="${SETUP_RECORD_DIR:-}"
CHECKPOINT_MANIFEST="${CHECKPOINT_MANIFEST:-${SCRIPT_DIR}/pi05_droid_jointpos_polaris_gcs_manifest.tsv}"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-7abd0c2294d442d429a77655783232206b2b30d95c508d435503135a5523a11c}"
TOKENIZER_VERIFICATION_FILE="${SETUP_RECORD_DIR}/paligemma_tokenizer_verification.json"
PACKAGE_PRESEAL_VERIFICATION_FILE="${SETUP_RECORD_DIR}/openpi_package_environment_preseal.json"
PACKAGE_VERIFICATION_FILE="${SETUP_RECORD_DIR}/openpi_package_environment.json"
PACKAGE_FINAL_VERIFICATION_FILE="${SETUP_RECORD_DIR}/openpi_package_environment_final.json"
NUMPYDANTIC_STUB_WARNING_FILTER='ignore:ndarray.pyi stub file could not be generated:ImportWarning:numpydantic.meta'
export PYTHONWARNINGS="${NUMPYDANTIC_STUB_WARNING_FILTER}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

command -v uv >/dev/null 2>&1 || die "uv is required"
: "${SETUP_RECORD_DIR:?Set a unique, previously unused SETUP_RECORD_DIR}"
[[ "${SETUP_RECORD_DIR}" == /* ]] \
  || die "SETUP_RECORD_DIR must be an absolute path"
[[ ! -e "${SETUP_RECORD_DIR}" && ! -L "${SETUP_RECORD_DIR}" ]] \
  || die "SETUP_RECORD_DIR must be a unique, previously unused path"
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

mkdir -p "${OPENPI_DATA_HOME}"
mkdir -p "$(dirname "${SETUP_RECORD_DIR}")"
mkdir "${SETUP_RECORD_DIR}"

if [[ -e "${OPENPI_DIR}/.venv" || -L "${OPENPI_DIR}/.venv" ]]; then
  [[ -d "${OPENPI_DIR}/.venv" && ! -L "${OPENPI_DIR}/.venv" ]] \
    || die "Refusing unsafe OpenPI environment path: ${OPENPI_DIR}/.venv"
  rm -rf -- "${OPENPI_DIR}/.venv"
fi
(
  cd "${OPENPI_DIR}"
  GIT_LFS_SKIP_SMUDGE=1 uv sync \
    --frozen --no-cache --reinstall --link-mode copy
  # OpenPI and LeRobot lock the full and headless OpenCV wheels, which claim
  # three different bytes at the same cv2 paths.  Reinstall the direct OpenPI
  # dependency last from this exact frozen lock so the active provider is
  # deterministic before the package-overlap attestation runs.
  GIT_LFS_SKIP_SMUDGE=1 uv sync \
    --frozen --no-cache --reinstall-package opencv-python --link-mode copy
  GIT_LFS_SKIP_SMUDGE=1 uv pip install \
    --no-cache --reinstall --link-mode copy --no-deps -e .
)

package_identity="$(
  PYTHONPATH="${POLARIS_DIR}/src" \
    "${OPENPI_DIR}/.venv/bin/python" - \
    "${OPENPI_DIR}" "${PACKAGE_PRESEAL_VERIFICATION_FILE}" \
    "${PACKAGE_VERIFICATION_FILE}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

from polaris.pi05_droid_jointpos_serving_contract import (
    canonical_json_bytes,
    seal_openpi_import_generated_stubs,
    verify_openpi_package_environment,
)

preseal_report = verify_openpi_package_environment(
    Path(sys.argv[1]), require_import_generated_stub_seals=False
)
Path(sys.argv[2]).write_text(
    json.dumps(preseal_report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
seal_report = seal_openpi_import_generated_stubs(Path(sys.argv[1]))
report = verify_openpi_package_environment(Path(sys.argv[1]))
if report["import_generated_stub_seals"] != seal_report:
    raise RuntimeError("Import-generated stub seal changed before package attestation")
if {
    key: value
    for key, value in preseal_report.items()
    if key != "import_generated_stub_seals"
} != {
    key: value
    for key, value in report.items()
    if key != "import_generated_stub_seals"
}:
    raise RuntimeError("Package environment changed while sealing typing stub")
Path(sys.argv[3]).write_text(
    json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
print(
    hashlib.sha256(canonical_json_bytes(preseal_report)).hexdigest(),
    hashlib.sha256(canonical_json_bytes(report)).hexdigest(),
    len(report["installed_distributions"]),
    len(report["record_verified_distributions"]),
    hashlib.sha256(canonical_json_bytes(seal_report)).hexdigest(),
)
PY
)" || die "Hermetic OpenPI package RECORD/uv.lock verification failed"
read -r PACKAGE_PRESEAL_ENVIRONMENT_SHA256 PACKAGE_ENVIRONMENT_SHA256 \
  PACKAGE_COUNT RECORD_VERIFIED_COUNT \
  IMPORT_GENERATED_STUB_SEAL_SHA256 \
  <<<"${package_identity}"
[[ "${PACKAGE_ENVIRONMENT_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "Invalid OpenPI package-environment identity"
[[ "${PACKAGE_PRESEAL_ENVIRONMENT_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "Invalid pre-seal OpenPI package-environment identity"
(( RECORD_VERIFIED_COUNT == PACKAGE_COUNT - 2 )) \
  || die "OpenPI package RECORD verification did not cover every noneditable package"
[[ "${IMPORT_GENERATED_STUB_SEAL_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "Invalid import-generated stub seal identity"

tokenizer_identity="$(
  OPENPI_DATA_HOME="${OPENPI_DATA_HOME}" \
    PYTHONPATH="${POLARIS_DIR}/src:${OPENPI_DIR}/src" \
    "${OPENPI_DIR}/.venv/bin/python" - "${TOKENIZER_VERIFICATION_FILE}" <<'PY'
import json
from pathlib import Path
import sys

from openpi.models import tokenizer as openpi_tokenizer
from polaris.pi05_droid_jointpos_serving_contract import (
    attest_loaded_tokenizer_sentencepiece,
    verify_paligemma_tokenizer_artifact,
)

artifact = verify_paligemma_tokenizer_artifact(openpi_tokenizer.download)
loaded = attest_loaded_tokenizer_sentencepiece(
    openpi_tokenizer.PaligemmaTokenizer(max_len=200)
)
report = {
    "schema_version": 1,
    "status": "pass",
    "artifact": artifact,
    "loaded_sentencepiece": loaded,
}
Path(sys.argv[1]).write_text(
    json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
print(
    artifact["remote"]["generation"],
    artifact["local"]["size"],
    artifact["local"]["md5_base64"],
    artifact["local"]["sha256"],
)
PY
)" || die "PaliGemma tokenizer generation/bytes/proto verification failed"
read -r TOKENIZER_GENERATION TOKENIZER_SIZE TOKENIZER_MD5_BASE64 \
  TOKENIZER_SHA256 <<<"${tokenizer_identity}"
[[ "${TOKENIZER_GENERATION}" == 1711547605575873 ]] \
  || die "Unexpected PaliGemma tokenizer generation: ${TOKENIZER_GENERATION}"
[[ "${TOKENIZER_SIZE}" == 4264023 ]] \
  || die "Unexpected PaliGemma tokenizer size: ${TOKENIZER_SIZE}"
[[ "${TOKENIZER_MD5_BASE64}" == 'FCCtyYVnIKVZ6KhyhLGV4g==' ]] \
  || die "Unexpected PaliGemma tokenizer MD5: ${TOKENIZER_MD5_BASE64}"
[[ "${TOKENIZER_SHA256}" == 8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6 ]] \
  || die "Unexpected PaliGemma tokenizer SHA-256: ${TOKENIZER_SHA256}"

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

package_final_identity="$(
  PYTHONPATH="${POLARIS_DIR}/src" \
    "${OPENPI_DIR}/.venv/bin/python" - \
    "${OPENPI_DIR}" "${PACKAGE_VERIFICATION_FILE}" \
    "${PACKAGE_FINAL_VERIFICATION_FILE}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

from polaris.pi05_droid_jointpos_serving_contract import (
    canonical_json_bytes,
    verify_openpi_package_environment,
)

sealed_report = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
final_report = verify_openpi_package_environment(Path(sys.argv[1]))
if final_report != sealed_report:
    raise RuntimeError(
        "OpenPI package environment changed during tokenizer/checkpoint/config imports"
    )
Path(sys.argv[3]).write_text(
    json.dumps(final_report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
print(hashlib.sha256(canonical_json_bytes(final_report)).hexdigest())
PY
)" || die "Final OpenPI package/seal re-attestation failed"
read -r PACKAGE_FINAL_ENVIRONMENT_SHA256 <<<"${package_final_identity}"
[[ "${PACKAGE_FINAL_ENVIRONMENT_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "Invalid final OpenPI package-environment identity"
[[ "${PACKAGE_FINAL_ENVIRONMENT_SHA256}" == "${PACKAGE_ENVIRONMENT_SHA256}" ]] \
  || die "Final OpenPI package environment differs from sealed setup identity"
cmp -s "${PACKAGE_VERIFICATION_FILE}" "${PACKAGE_FINAL_VERIFICATION_FILE}" \
  || die "Final OpenPI package report differs byte-for-byte from sealed report"

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
  printf 'tokenizer_verification=%s\n' "${TOKENIZER_VERIFICATION_FILE}"
  printf 'tokenizer_generation=%s\n' "${TOKENIZER_GENERATION}"
  printf 'tokenizer_size=%s\n' "${TOKENIZER_SIZE}"
  printf 'tokenizer_md5_base64=%s\n' "${TOKENIZER_MD5_BASE64}"
  printf 'tokenizer_sha256=%s\n' "${TOKENIZER_SHA256}"
  printf 'package_verification=%s\n' "${PACKAGE_VERIFICATION_FILE}"
  printf 'package_preseal_verification=%s\n' \
    "${PACKAGE_PRESEAL_VERIFICATION_FILE}"
  printf 'package_preseal_environment_sha256=%s\n' \
    "${PACKAGE_PRESEAL_ENVIRONMENT_SHA256}"
  printf 'package_environment_sha256=%s\n' "${PACKAGE_ENVIRONMENT_SHA256}"
  printf 'package_final_verification=%s\n' \
    "${PACKAGE_FINAL_VERIFICATION_FILE}"
  printf 'package_final_environment_sha256=%s\n' \
    "${PACKAGE_FINAL_ENVIRONMENT_SHA256}"
  printf 'package_environment_unchanged_after_imports=true\n'
  printf 'package_count=%s\n' "${PACKAGE_COUNT}"
  printf 'record_verified_count=%s\n' "${RECORD_VERIFIED_COUNT}"
  printf 'import_generated_stub_seal_sha256=%s\n' \
    "${IMPORT_GENERATED_STUB_SEAL_SHA256}"
  printf 'pythonwarnings=%s\n' "${NUMPYDANTIC_STUB_WARNING_FILTER}"
  printf 'openpi_data_home=%s\n' "${OPENPI_DATA_HOME}"
  printf 'uv_version=%s\n' "$(uv --version)"
  printf 'uv_cache_mode=%s\n' no-cache
} | tee "${SETUP_RECORD_DIR}/setup_manifest.env"
