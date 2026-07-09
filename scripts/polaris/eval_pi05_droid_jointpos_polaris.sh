#!/usr/bin/env bash

# Launch the exact official pi0.5-DROID-Polaris server and one bounded task eval.
# This script runs inside one ordinary one-GPU Slurm allocation.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLARIS_DIR="${POLARIS_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
OPENPI_DIR="${OPENPI_DIR:-${POLARIS_DIR}/third_party/openpi}"
NFS_ROOT="${NFS_ROOT:-/lustre/fsw/portfolios/nvr/users/lzha}"
POLARIS_DATA_DIR="${POLARIS_DATA_DIR:-${NFS_ROOT}/data/PolaRiS-Hub}"
POLARIS_PYXIS_IMAGE="${POLARIS_PYXIS_IMAGE:-${NFS_ROOT}/cache/polaris/polaris-eval-cuda13-fd00a51.sqsh}"
POLARIS_VULKAN_ICD_PATH="${POLARIS_VULKAN_ICD_PATH:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${NFS_ROOT}/cache/openpi-polaris}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${NFS_ROOT}/results/polaris-pi05}"
POLARIS_CACHE_DIR="${POLARIS_CACHE_DIR:-${NFS_ROOT}/cache/polaris/runtime/${SLURM_JOB_ID:-manual}}"

CHECKPOINT_URI="${CHECKPOINT_URI:-gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris}"
EXPECTED_CHECKPOINT_URI="gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_droid_jointpos_polaris}"
EXPECTED_OPENPI_COMMIT="${EXPECTED_OPENPI_COMMIT:-bd70b8f4011e85b3f3b0f039f12113f78718e7bf}"
EXPECTED_NORM_SHA256="${EXPECTED_NORM_SHA256:-57ce9956f9e07d65f8a8205aabec72d436a2c8927f53edb40c7a77b14a5a90c7}"
EXPECTED_PYXIS_SHA256="${EXPECTED_PYXIS_SHA256:-ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a}"
EXPECTED_VULKAN_ICD_SHA256="7bdb6f27d35b66fc848df6f94b8773bba30ea3a7f06f114100d14154a235a34b"
EXPECTED_NVIDIA_DRIVER_VERSION="580.105.08"
EXPECTED_NVIDIA_GPU_NAME="NVIDIA L40S"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-7abd0c2294d442d429a77655783232206b2b30d95c508d435503135a5523a11c}"
CHECKPOINT_MANIFEST="${CHECKPOINT_MANIFEST:-${SCRIPT_DIR}/pi05_droid_jointpos_polaris_gcs_manifest.tsv}"
EXPECTED_ACTION_HORIZON="${EXPECTED_ACTION_HORIZON:-15}"
EXPECTED_ACTION_DIM="${EXPECTED_ACTION_DIM:-8}"
OPEN_LOOP_HORIZON="${OPEN_LOOP_HORIZON:-8}"
ENVIRONMENT_SEED="${ENVIRONMENT_SEED:-0}"
POLARIS_ENVIRONMENT="${POLARIS_ENVIRONMENT:-DROID-FoodBussing}"
ROLLOUTS="${ROLLOUTS:-1}"
RUN_NAMESPACE="${RUN_NAMESPACE:-pi05-polaris-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_LABEL="${RUN_LABEL:-pi05-polaris}"
PORT="${PORT:-$((20000 + ${SLURM_JOB_ID:-1} % 20000))}"
SERVER_START_TIMEOUT_SECS="${SERVER_START_TIMEOUT_SECS:-2400}"
DRY_RUN="${DRY_RUN:-0}"
RESUME_FROM_TASK_DIR="${RESUME_FROM_TASK_DIR:-}"
ENVIRONMENT_SEED_SCHEME=base_plus_episode_index_v1
ENVIRONMENT_DETERMINISM_CLAIM=rng_bound_not_bitwise
POLARIS_DATA_REVISION=8c7e4103e266ef83d8b1ad2e9a63116edd5f155b
EXPECTED_ROBOT_ASSET_SHA256=d8379925b103963dbf3e7c85bcc4ae101b81b7c1d7dabe7d2e964f41d069ec44
EXPECTED_ROBOT_METADATA_SHA256=208e0f85fc16fa32ffeca972aea0fd1b33b0c6c2a582e89ff3877823291a7754
NUMPYDANTIC_STUB_WARNING_FILTER='ignore:ndarray.pyi stub file could not be generated:ImportWarning:numpydantic.meta'

die() {
  echo "ERROR: $*" >&2
  exit 2
}

capture_gpu_runtime() {
  local -a rows
  mapfile -t rows < <(
    nvidia-smi --query-gpu=uuid,name,driver_version \
      --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d'
  )
  (( ${#rows[@]} == 1 )) || die "Expected exactly one allocated NVIDIA GPU"
  IFS=, read -r actual_gpu_uuid actual_gpu_name \
    actual_nvidia_driver_version <<<"${rows[0]}"
  actual_gpu_uuid="$(xargs <<<"${actual_gpu_uuid}")"
  actual_gpu_name="$(xargs <<<"${actual_gpu_name}")"
  actual_nvidia_driver_version="$(xargs <<<"${actual_nvidia_driver_version}")"
  [[ "${actual_gpu_uuid}" =~ ^GPU-[0-9a-fA-F-]+$ ]] \
    || die "Invalid allocated NVIDIA GPU UUID: ${actual_gpu_uuid}"
  [[ "${actual_gpu_name}" == "${EXPECTED_NVIDIA_GPU_NAME}" ]] \
    || die "NVIDIA GPU name mismatch: ${actual_gpu_name}"
  [[ "${actual_nvidia_driver_version}" == "${EXPECTED_NVIDIA_DRIVER_VERSION}" ]] \
    || die "NVIDIA driver version mismatch: ${actual_nvidia_driver_version}"
  [[ "${NVIDIA_VISIBLE_DEVICES:-}" == "${actual_gpu_uuid}" ]] \
    || die "NVIDIA_VISIBLE_DEVICES does not identify the allocated GPU"
}

capture_package_environment_sha256() {
  env \
    PYTHONWARNINGS="${NUMPYDANTIC_STUB_WARNING_FILTER}" \
    PYTHONPATH="${POLARIS_DIR}/src" \
    "${OPENPI_DIR}/.venv/bin/python" - "${OPENPI_DIR}" <<'PY'
import hashlib
import sys
from pathlib import Path

from polaris.pi05_droid_jointpos_serving_contract import (
    canonical_json_bytes,
    verify_openpi_package_environment,
)

report = verify_openpi_package_environment(Path(sys.argv[1]))
print(hashlib.sha256(canonical_json_bytes(report)).hexdigest())
PY
}

[[ -n "${SLURM_JOB_ID:-}" || "${DRY_RUN}" == 1 ]] \
  || die "A Slurm allocation is required unless DRY_RUN=1"
: "${EXPECTED_POLARIS_COMMIT:?Set EXPECTED_POLARIS_COMMIT to the immutable launch commit}"
[[ "${ROLLOUTS}" =~ ^[1-9][0-9]*$ ]] || die "ROLLOUTS must be positive"
[[ "${ENVIRONMENT_SEED}" =~ ^(0|[1-9][0-9]*)$ ]] \
  || die "ENVIRONMENT_SEED must be a non-negative integer"
(( ENVIRONMENT_SEED <= 4294967295 )) \
  || die "ENVIRONMENT_SEED must be at most 4294967295"
if ! python3 - "${ENVIRONMENT_SEED}" "${ROLLOUTS}" <<'PY'
import sys

base_seed = int(sys.argv[1])
rollouts = int(sys.argv[2])
if base_seed + rollouts - 1 > 2**32 - 1:
    raise SystemExit(1)
PY
then
  die "ENVIRONMENT_SEED + ROLLOUTS - 1 exceeds the uint32 seed range"
fi
EXPECTED_POLICY_REQUESTS=$((ROLLOUTS * 57))
[[ -z "${RESUME_FROM_TASK_DIR}" ]] \
  || die "Seed-bound native evaluation forbids resume until policy RNG restoration is supported"
if [[ ! "${PORT}" =~ ^[1-9][0-9]*$ ]] || (( PORT > 65535 )); then
  die "Invalid PORT=${PORT}"
fi
[[ "${POLICY_CONFIG}" == pi05_droid_jointpos_polaris ]] || die "Unexpected policy config: ${POLICY_CONFIG}"
[[ "${CHECKPOINT_URI}" == "${EXPECTED_CHECKPOINT_URI}" ]] \
  || die "Unexpected checkpoint URI: ${CHECKPOINT_URI}"
[[ -x "${OPENPI_DIR}/.venv/bin/python" ]] || die "Run setup_pi05_droid_jointpos_polaris.sh first"
[[ "$(git -C "${OPENPI_DIR}" rev-parse HEAD)" == "${EXPECTED_OPENPI_COMMIT}" ]] \
  || die "OpenPI is not at ${EXPECTED_OPENPI_COMMIT}"
[[ -d "${POLARIS_DATA_DIR}" && ! -L "${POLARIS_DATA_DIR}" ]] \
  || die "Missing regular PolaRiS data: ${POLARIS_DATA_DIR}"
[[ -f "${POLARIS_PYXIS_IMAGE}" ]] || die "Missing Pyxis image: ${POLARIS_PYXIS_IMAGE}"
[[ -f "${POLARIS_VULKAN_ICD_PATH}" && ! -L "${POLARIS_VULKAN_ICD_PATH}" ]] \
  || die "Vulkan ICD must be a regular non-symlink file: ${POLARIS_VULKAN_ICD_PATH}"
actual_vulkan_icd_sha256="$(sha256sum "${POLARIS_VULKAN_ICD_PATH}" | awk '{print $1}')"
[[ "${actual_vulkan_icd_sha256}" == "${EXPECTED_VULKAN_ICD_SHA256}" ]] \
  || die "Vulkan ICD SHA-256 mismatch: ${actual_vulkan_icd_sha256}"
capture_gpu_runtime
preflight_vulkan_icd_sha256="${actual_vulkan_icd_sha256}"
preflight_gpu_uuid="${actual_gpu_uuid}"
preflight_gpu_name="${actual_gpu_name}"
preflight_nvidia_driver_version="${actual_nvidia_driver_version}"
[[ -f "${CHECKPOINT_MANIFEST}" ]] || die "Missing checkpoint manifest: ${CHECKPOINT_MANIFEST}"

case "${POLARIS_ENVIRONMENT}" in
  DROID-BlockStackKitchen)
    EXPECTED_PROMPT='Place and stack the blocks on top of the green tray'
    ASSET_SUBDIR=block_stack_kitchen
    EXPECTED_INITIAL_CONDITIONS_SHA256=eebd5052254c1d56681592129960412d1c4c9efbc33555213c330213025f63e7
    EXPECTED_SCENE_SHA256=eb13abf802c16a8bff9b05151ffd0ffc26feb8fac8c76aa7a57b5d9468be3363
    EXPECTED_INITIAL_CONDITIONS_METADATA_SHA256=19ac250464ccc7b723ab0a02f9c6345987b2ce74d4a8539db0df1145f8d3c306
    EXPECTED_SCENE_METADATA_SHA256=39ba7b391d90d1e8eb11759ad9240155f34c6d92589a20647687c56e00c199fc
    ;;
  DROID-FoodBussing)
    EXPECTED_PROMPT='Put all the foods in the bowl'
    ASSET_SUBDIR=food_bussing
    EXPECTED_INITIAL_CONDITIONS_SHA256=40091faee14f692350220871d30705294f21f17ae3d2974cd3c09a34d560f5de
    EXPECTED_SCENE_SHA256=82cd641e422935b394ce7ea7b6be55214c9952a2544000222921e544c409b489
    EXPECTED_INITIAL_CONDITIONS_METADATA_SHA256=852dd0345afb7e4d0c7526b5c327086b5132c40624ed97ff6942962126e90534
    EXPECTED_SCENE_METADATA_SHA256=accd9b67e90e510eb4ed44a789b9169df058e71ce557164f960de2d62a840e63
    ;;
  DROID-PanClean)
    EXPECTED_PROMPT='Use the yellow sponge to scrub the blue handle frying pan'
    ASSET_SUBDIR=pan_clean
    EXPECTED_INITIAL_CONDITIONS_SHA256=a47debf4fdd0f3562a331380a125640b12fc0f5786aff849e71388f07756b8c8
    EXPECTED_SCENE_SHA256=c10140794bc3a46fa2a713f4560cf9651c163c9337697c9aafbdccc26762856e
    EXPECTED_INITIAL_CONDITIONS_METADATA_SHA256=c9bfa54d0fd1de261ddcdc9c9ebdec6690ad20d33d3cfe4cef1e43e92f2affeb
    EXPECTED_SCENE_METADATA_SHA256=efea8aa97431bc8b8a61a52af2b756cce670b271f7fff53bdeb1121c6309d840
    ;;
  DROID-MoveLatteCup)
    EXPECTED_PROMPT='put the latte art cup on top of the cutting board'
    ASSET_SUBDIR=move_latte_cup
    EXPECTED_INITIAL_CONDITIONS_SHA256=44d73616b396abfc1ca03e37cd4de26e2f02845967a0c89bbd9a4c3a1e800421
    EXPECTED_SCENE_SHA256=cec74210b92155782ad0e2a911c3227c8bc251c986ba15a56be4f4e5f382529b
    EXPECTED_INITIAL_CONDITIONS_METADATA_SHA256=aa493d82bc28d748fe5e1ad542fcff9531d1844022465c38e57681f3d74a2a2b
    EXPECTED_SCENE_METADATA_SHA256=22563e7f5a132094f1072edbf69089307c249dbc4cf915ceb3164ac58b8ae8b4
    ;;
  DROID-OrganizeTools)
    EXPECTED_PROMPT='put the scissor into the large container'
    ASSET_SUBDIR=organize_tools
    EXPECTED_INITIAL_CONDITIONS_SHA256=2f2dba117c834b0137bed5a07fee3c421a49dbdb0f5b36697f080c6693b6bf54
    EXPECTED_SCENE_SHA256=41ab252b02766aa6bd3b763d3feeaa9f5749b6984cf0f4a9f3c32d6c5db96c81
    EXPECTED_INITIAL_CONDITIONS_METADATA_SHA256=8e7677d03cf57b1c163657fdcbdd674a436d5396e8b8ebcd2af7e03bff5be672
    EXPECTED_SCENE_METADATA_SHA256=08d9c737be826da7243a62e8e8249b9b879fb6b839792d0f0eba3da7c1a666f2
    ;;
  DROID-TapeIntoContainer)
    EXPECTED_PROMPT='put the tape into the container'
    ASSET_SUBDIR=tape_into_container
    EXPECTED_INITIAL_CONDITIONS_SHA256=0e8ea9329812709f194324dd19b7e64cbdbd0905aa31d2ed5f6bdc9a32a5bfec
    EXPECTED_SCENE_SHA256=18f86c02ab6dec5ea31a67706458b9fccd80be2dcb8e0035858e2ab69f4cfad7
    EXPECTED_INITIAL_CONDITIONS_METADATA_SHA256=bcaa11e0e598149c9da03cbcf3b8f30eb9d3cce975622528495f82305acc4a99
    EXPECTED_SCENE_METADATA_SHA256=c634e51498d67db91835bd9fb5d071640408737920bdec4bc1192beeb826d052
    ;;
  *)
    die "Unsupported PolaRiS task: ${POLARIS_ENVIRONMENT}"
    ;;
esac

initial_conditions_path="${POLARIS_DATA_DIR}/${ASSET_SUBDIR}/initial_conditions.json"
scene_path="${POLARIS_DATA_DIR}/${ASSET_SUBDIR}/scene.usda"
robot_asset_path="${POLARIS_DATA_DIR}/nvidia_droid/noninstanceable.usd"
[[ -f "${initial_conditions_path}" && ! -L "${initial_conditions_path}" ]] \
  || die "Missing regular initial conditions: ${initial_conditions_path}"
[[ -f "${scene_path}" && ! -L "${scene_path}" ]] \
  || die "Missing regular scene: ${scene_path}"
[[ -f "${robot_asset_path}" && ! -L "${robot_asset_path}" ]] \
  || die "Missing regular robot asset: ${robot_asset_path}"
actual_initial_conditions_sha256="$(sha256sum "${initial_conditions_path}" | awk '{print $1}')"
actual_scene_sha256="$(sha256sum "${scene_path}" | awk '{print $1}')"
actual_robot_asset_sha256="$(sha256sum "${robot_asset_path}" | awk '{print $1}')"
[[ "${actual_initial_conditions_sha256}" == "${EXPECTED_INITIAL_CONDITIONS_SHA256}" ]] \
  || die "Initial-conditions SHA-256 mismatch: ${actual_initial_conditions_sha256}"
[[ "${actual_scene_sha256}" == "${EXPECTED_SCENE_SHA256}" ]] \
  || die "Scene SHA-256 mismatch: ${actual_scene_sha256}"
[[ "${actual_robot_asset_sha256}" == "${EXPECTED_ROBOT_ASSET_SHA256}" ]] \
  || die "Robot asset SHA-256 mismatch: ${actual_robot_asset_sha256}"

robot_metadata_path="${POLARIS_DATA_DIR}/.cache/huggingface/download/nvidia_droid/noninstanceable.usd.metadata"
[[ -f "${robot_metadata_path}" && ! -L "${robot_metadata_path}" ]] \
  || die "Missing regular robot Hub metadata: ${robot_metadata_path}"
actual_robot_metadata_sha256="$(sha256sum "${robot_metadata_path}" | awk '{print $1}')"
[[ "${actual_robot_metadata_sha256}" == "${EXPECTED_ROBOT_METADATA_SHA256}" ]] \
  || die "Robot Hub metadata SHA-256 mismatch: ${actual_robot_metadata_sha256}"
[[ "$(head -n 1 "${robot_metadata_path}")" == "${POLARIS_DATA_REVISION}" ]] \
  || die "Robot asset Hub revision mismatch"

initial_conditions_metadata_path="${POLARIS_DATA_DIR}/.cache/huggingface/download/${ASSET_SUBDIR}/initial_conditions.json.metadata"
scene_metadata_path="${POLARIS_DATA_DIR}/.cache/huggingface/download/${ASSET_SUBDIR}/scene.usda.metadata"
[[ -f "${initial_conditions_metadata_path}" && ! -L "${initial_conditions_metadata_path}" ]] \
  || die "Missing regular task initial-conditions Hub metadata"
[[ -f "${scene_metadata_path}" && ! -L "${scene_metadata_path}" ]] \
  || die "Missing regular task scene Hub metadata"
actual_initial_conditions_metadata_sha256="$(sha256sum "${initial_conditions_metadata_path}" | awk '{print $1}')"
actual_scene_metadata_sha256="$(sha256sum "${scene_metadata_path}" | awk '{print $1}')"
[[ "${actual_initial_conditions_metadata_sha256}" == \
  "${EXPECTED_INITIAL_CONDITIONS_METADATA_SHA256}" ]] \
  || die "Task initial-conditions Hub metadata SHA-256 mismatch"
[[ "${actual_scene_metadata_sha256}" == "${EXPECTED_SCENE_METADATA_SHA256}" ]] \
  || die "Task scene Hub metadata SHA-256 mismatch"
[[ "$(head -n 1 "${initial_conditions_metadata_path}")" == \
  "${POLARIS_DATA_REVISION}" ]] \
  || die "Task initial-conditions Hub revision mismatch"
[[ "$(head -n 1 "${scene_metadata_path}")" == "${POLARIS_DATA_REVISION}" ]] \
  || die "Task scene Hub revision mismatch"

POLARIS_COMMIT="$(git -C "${POLARIS_DIR}" rev-parse HEAD)"
OPENPI_COMMIT="$(git -C "${OPENPI_DIR}" rev-parse HEAD)"
[[ "${POLARIS_COMMIT}" == "${EXPECTED_POLARIS_COMMIT}" ]] \
  || die "PolaRiS commit ${POLARIS_COMMIT} does not match ${EXPECTED_POLARIS_COMMIT}"
[[ "${EXPECTED_POLARIS_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || die "EXPECTED_POLARIS_COMMIT must be one full lowercase commit"
[[ -z "$(git -C "${POLARIS_DIR}" status --porcelain=v1 --untracked-files=all)" ]] \
  || die "PolaRiS has tracked or untracked modifications; launch only from a clean committed revision"
RUN_NAME="${RUN_NAME:-${RUN_NAMESPACE}_${RUN_LABEL}_${POLARIS_ENVIRONMENT}_${SLURM_JOB_ID:-dryrun}}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_NAMESPACE}/${RUN_NAME}}"
TASK_DIR="${RUN_DIR}/${POLARIS_ENVIRONMENT}"
SERVER_LOG="${RUN_DIR}/policy_server.log"
EVAL_LOG="${TASK_DIR}/eval.log"
TRACE_PATH="${TASK_DIR}/policy_traces.jsonl"
TRACE_SUMMARY="${TASK_DIR}/policy_trace_summary.json"
RUNTIME_CONTRACT_FILE="${TASK_DIR}/pi05_droid_jointpos_runtime.json"
SERVING_CONTRACT_FILE="${RUN_DIR}/pi05_droid_jointpos_serving_contract.json"
MODEL_RUNTIME_CONTRACT_FILE="${RUN_DIR}/pi05_droid_jointpos_model_runtime.json"
RNG_STREAM_FILE="${RUN_DIR}/pi05_droid_jointpos_rng_stream.json"
REQUEST_PROOF_FILE="${RUN_DIR}/pi05_droid_jointpos_request_proof.json"
ASSET_MANIFEST_FILE="${RUN_DIR}/polaris_asset_dependency_manifest.json"
VIDEO_VALIDATION_FILE="${TASK_DIR}/pi05_droid_jointpos_video_validation.json"
VIDEO_VALIDATION_LOG="${TASK_DIR}/video_validation.log"
COMMANDS_FILE="${RUN_DIR}/commands.sh"
METADATA_FILE="${RUN_DIR}/run_metadata.env"
SERVER_PID=""
EVIDENCE_FINALIZED=0
EVIDENCE_MANIFEST_SHA256=""

mkdir -p "${TASK_DIR}" "${POLARIS_CACHE_DIR}"
[[ ! -e "${SERVING_CONTRACT_FILE}" && ! -L "${SERVING_CONTRACT_FILE}" ]] \
  || die "Serving-contract output already exists"
[[ ! -e "${MODEL_RUNTIME_CONTRACT_FILE}" && ! -L "${MODEL_RUNTIME_CONTRACT_FILE}" ]] \
  || die "Model-runtime output already exists"
[[ ! -e "${RNG_STREAM_FILE}" && ! -L "${RNG_STREAM_FILE}" ]] \
  || die "Policy RNG-stream output already exists"
[[ ! -e "${REQUEST_PROOF_FILE}" && ! -L "${REQUEST_PROOF_FILE}" ]] \
  || die "Policy request-proof output already exists"
[[ ! -e "${RUNTIME_CONTRACT_FILE}" && ! -L "${RUNTIME_CONTRACT_FILE}" ]] \
  || die "Live joint-position runtime output already exists"
[[ ! -e "${ASSET_MANIFEST_FILE}" && ! -L "${ASSET_MANIFEST_FILE}" ]] \
  || die "PolaRiS asset dependency manifest output already exists"
[[ ! -e "${VIDEO_VALIDATION_FILE}" && ! -L "${VIDEO_VALIDATION_FILE}" ]] \
  || die "Video-validation output already exists"
[[ ! -e "${VIDEO_VALIDATION_LOG}" && ! -L "${VIDEO_VALIDATION_LOG}" ]] \
  || die "Video-validation log already exists"
if [[ -n "${RESUME_FROM_TASK_DIR}" ]]; then
  [[ ! -e "${TASK_DIR}/eval_results.csv" && ! -e "${TRACE_PATH}" ]] \
    || die "Resume destination already contains metrics or traces"
  python3 "${SCRIPT_DIR}/prepare_pi05_resume.py" \
    "${RESUME_FROM_TASK_DIR}" "${TASK_DIR}" --expected-rollouts "${ROLLOUTS}" \
    --output "${RUN_DIR}/resume_manifest.json"
  PYTHONPATH="${POLARIS_DIR}/src" "${OPENPI_DIR}/.venv/bin/python" \
    "${SCRIPT_DIR}/validate_pi05_trace.py" \
    "${TRACE_PATH}" --metrics-csv "${TASK_DIR}/eval_results.csv" \
    --expected-prompt "${EXPECTED_PROMPT}" \
    --expected-environment-seed "${ENVIRONMENT_SEED}" \
    --output "${RUN_DIR}/resume_trace_summary.json"
else
  [[ ! -e "${TASK_DIR}/eval_results.csv" && ! -e "${TRACE_PATH}" ]] \
    || die "Run directory already contains metrics or traces; use a new job/run directory"
fi
for marker in "${RUN_DIR}/SUCCESS" "${RUN_DIR}/FAILED" "${RUN_DIR}/DRY_RUN"; do
  [[ ! -e "${marker}" && ! -L "${marker}" ]] \
    || die "Run terminal marker already exists: ${marker}"
done
[[ ! -e "${TASK_DIR}/SUCCESS" && ! -L "${TASK_DIR}/SUCCESS" ]] \
  || die "Task SUCCESS marker already exists"
[[ ! -e "${TASK_DIR}/FAILED" && ! -L "${TASK_DIR}/FAILED" ]] \
  || die "Task FAILED marker already exists"
printf 'started_at=%s\n' "$(date -Iseconds)" > "${RUN_DIR}/RUNNING"

stop_server() {
  [[ -n "${SERVER_PID}" ]] || return 0
  kill -TERM -- "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null || true
  for _ in {1..20}; do
    kill -0 "${SERVER_PID}" 2>/dev/null || break
    sleep 0.25
  done
  kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=""
}

publish_terminal_marker() {
  local destination="$1"
  shift
  "${OPENPI_DIR}/.venv/bin/python" - "${destination}" "$@" <<'PY'
import os
from pathlib import Path
import re
import secrets
import stat
import sys

destination = Path(sys.argv[1])
items = sys.argv[2:]
if destination.name not in {"SUCCESS", "FAILED", "DRY_RUN"}:
    raise SystemExit("unexpected terminal marker name")
if destination.exists() or destination.is_symlink():
    raise SystemExit(f"terminal marker already exists: {destination}")
lines = []
keys = set()
for item in items:
    key, separator, value = item.partition("=")
    if (
        not separator
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None
        or key in keys
        or "\n" in value
        or "\r" in value
    ):
        raise SystemExit("invalid terminal marker field")
    keys.add(key)
    lines.append(f"{key}={value}\n")
payload = "".join(lines).encode("utf-8")
if not payload:
    raise SystemExit("terminal marker payload is empty")
destination.parent.mkdir(parents=True, exist_ok=True)
temporary = destination.with_name(
    f".{destination.name}.partial-{os.getpid()}-{secrets.token_hex(8)}"
)
linked = False
descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
try:
    with os.fdopen(descriptor, "wb", closefd=False) as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.fchmod(descriptor, 0o444)
    os.fsync(descriptor)
    os.close(descriptor)
    descriptor = -1
    os.link(temporary, destination, follow_symlinks=False)
    linked = True
    temporary.unlink()
    directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    reader = os.open(destination, flags)
    try:
        metadata = os.fstat(reader)
        observed = b""
        while chunk := os.read(reader, 64 * 1024):
            observed += chunk
    finally:
        os.close(reader)
    current = os.stat(destination, follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o444
        or observed != payload
        or (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mode)
        != (current.st_dev, current.st_ino, current.st_size, current.st_mode)
    ):
        raise RuntimeError("terminal marker readback validation failed")
except BaseException:
    if descriptor >= 0:
        try:
            os.close(descriptor)
        except OSError:
            pass
    if linked and (destination.exists() or destination.is_symlink()):
        destination.unlink()
    if temporary.exists() and not temporary.is_symlink():
        temporary.unlink()
    try:
        directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError:
        pass
    raise
PY
}

on_exit() {
  local original_code=$?
  local final_code="${original_code}"
  local marker_code=0
  trap - EXIT INT TERM
  set +e
  stop_server
  if ! rm -f "${RUN_DIR}/RUNNING"; then
    echo "ERROR: Could not remove RUNNING marker" >&2
    (( final_code == 0 )) && final_code=91
  fi
  if [[ -f "${METADATA_FILE}" && -w "${METADATA_FILE}" ]]; then
    if ! printf 'SCRIPT_EXIT_CODE=%s\n' "${final_code}" >> "${METADATA_FILE}"; then
      echo "ERROR: Could not append script exit code to run metadata" >&2
      (( final_code == 0 )) && final_code=92
    fi
  fi
  if (( final_code == 0 )) && [[ "${DRY_RUN}" == 1 ]]; then
    publish_terminal_marker "${RUN_DIR}/DRY_RUN" \
      "status=dry_run" "completed_at=$(date -Iseconds)" \
      "evidence_finalized=0" || marker_code=$?
  elif (( final_code == 0 )) && [[ "${EVIDENCE_FINALIZED}" == 1 ]]; then
    publish_terminal_marker "${RUN_DIR}/SUCCESS" \
      "status=success" "completed_at=$(date -Iseconds)" \
      "evidence_manifest_sha256=${EVIDENCE_MANIFEST_SHA256}" \
      || marker_code=$?
  elif (( final_code == 0 )); then
    echo "ERROR: Refusing terminal success without finalized evidence" >&2
    marker_code=93
  fi
  if (( marker_code != 0 )); then
    echo "ERROR: Terminal success-marker publication failed (${marker_code})" >&2
    final_code=94
  fi
  if (( final_code != 0 )); then
    if ! publish_terminal_marker "${RUN_DIR}/FAILED" \
      "status=failed" "failed_at=$(date -Iseconds)" \
      "exit_code=${final_code}"; then
      echo "ERROR: FAILED-marker publication also failed" >&2
      (( original_code == 0 )) && final_code=95
    fi
  fi
  exit "${final_code}"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

preflight_package_environment_sha256="$(capture_package_environment_sha256)" \
  || die "Preflight OpenPI package/seal attestation failed"
[[ "${preflight_package_environment_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "Preflight OpenPI package/seal attestation returned an invalid digest"
OPENPI_PYTHONPATH="${OPENPI_DIR}/src:${OPENPI_DIR}/packages/openpi-client/src"
checkpoint_path="$(
  OPENPI_DATA_HOME="${OPENPI_DATA_HOME}" PYTHONPATH="${OPENPI_PYTHONPATH}" \
    "${OPENPI_DIR}/.venv/bin/python" - "${CHECKPOINT_URI}" <<'PY'
import sys
from openpi.shared.download import maybe_download
print(maybe_download(sys.argv[1]))
PY
)"
norm_stats="${checkpoint_path}/assets/droid/norm_stats.json"
[[ -s "${norm_stats}" ]] || die "Missing checkpoint-local norm stats: ${norm_stats}"
actual_norm_sha256="$(sha256sum "${norm_stats}" | awk '{print $1}')"
[[ "${actual_norm_sha256}" == "${EXPECTED_NORM_SHA256}" ]] \
  || die "Norm SHA-256 mismatch: ${actual_norm_sha256}"
actual_manifest_sha256="$(sha256sum "${CHECKPOINT_MANIFEST}" | awk '{print $1}')"
[[ "${actual_manifest_sha256}" == "${EXPECTED_MANIFEST_SHA256}" ]] \
  || die "Checkpoint manifest SHA-256 mismatch: ${actual_manifest_sha256}"
"${OPENPI_DIR}/.venv/bin/python" "${SCRIPT_DIR}/verify_pi05_checkpoint.py" \
  "${checkpoint_path}" "${CHECKPOINT_MANIFEST}" --full-md5 \
  --output "${RUN_DIR}/checkpoint_verification.json"
PYXIS_IMAGE_SHA256="$(sha256sum "${POLARIS_PYXIS_IMAGE}" | awk '{print $1}')"
[[ "${PYXIS_IMAGE_SHA256}" == "${EXPECTED_PYXIS_SHA256}" ]] \
  || die "Pyxis image SHA-256 mismatch: ${PYXIS_IMAGE_SHA256}"

{
  printf 'RUN_START=%q\n' "$(date -Iseconds)"
  printf 'HOST=%q\n' "$(hostname)"
  printf 'SLURM_JOB_ID=%q\n' "${SLURM_JOB_ID:-dryrun}"
  printf 'RUN_NAMESPACE=%q\n' "${RUN_NAMESPACE}"
  printf 'RUN_DIR=%q\n' "${RUN_DIR}"
  printf 'POLARIS_DIR=%q\n' "${POLARIS_DIR}"
  printf 'POLARIS_COMMIT=%q\n' "${POLARIS_COMMIT}"
  printf 'OPENPI_DIR=%q\n' "${OPENPI_DIR}"
  printf 'OPENPI_COMMIT=%q\n' "${OPENPI_COMMIT}"
  printf 'CHECKPOINT_URI=%q\n' "${CHECKPOINT_URI}"
  printf 'CHECKPOINT_PATH=%q\n' "${checkpoint_path}"
  printf 'POLICY_CONFIG=%q\n' "${POLICY_CONFIG}"
  printf 'NORM_STATS_SHA256=%q\n' "${actual_norm_sha256}"
  printf 'CHECKPOINT_MANIFEST_SHA256=%q\n' "${actual_manifest_sha256}"
  printf 'POLARIS_PYXIS_IMAGE=%q\n' "${POLARIS_PYXIS_IMAGE}"
  printf 'POLARIS_PYXIS_IMAGE_SHA256=%q\n' "${PYXIS_IMAGE_SHA256}"
  printf 'POLARIS_VULKAN_ICD_PATH=%q\n' "${POLARIS_VULKAN_ICD_PATH}"
  printf 'POLARIS_VULKAN_ICD_SHA256=%q\n' "${actual_vulkan_icd_sha256}"
  printf 'NVIDIA_GPU_UUID=%q\n' "${actual_gpu_uuid}"
  printf 'NVIDIA_GPU_NAME=%q\n' "${actual_gpu_name}"
  printf 'NVIDIA_DRIVER_VERSION=%q\n' "${actual_nvidia_driver_version}"
  printf 'PYTHONWARNINGS=%q\n' "${NUMPYDANTIC_STUB_WARNING_FILTER}"
  printf 'PREFLIGHT_PACKAGE_ENVIRONMENT_SHA256=%q\n' \
    "${preflight_package_environment_sha256}"
  printf 'POLARIS_ENVIRONMENT=%q\n' "${POLARIS_ENVIRONMENT}"
  printf 'EXPECTED_PROMPT=%q\n' "${EXPECTED_PROMPT}"
  printf 'RESUME_FROM_TASK_DIR=%q\n' "${RESUME_FROM_TASK_DIR}"
  printf 'ROLLOUTS=%q\n' "${ROLLOUTS}"
  printf 'ENVIRONMENT_SEED=%q\n' "${ENVIRONMENT_SEED}"
  printf 'ENVIRONMENT_SEED_PROFILE=isaaclab_env_seed_base_plus_episode_v1\n'
  printf 'ENVIRONMENT_SEED_SCHEME=%q\n' "${ENVIRONMENT_SEED_SCHEME}"
  printf 'ENVIRONMENT_DETERMINISM_CLAIM=%q\n' "${ENVIRONMENT_DETERMINISM_CLAIM}"
  printf 'PHYSX_ENHANCED_DETERMINISM=false\n'
  printf 'POLARIS_DATA_REPOSITORY=owhan/PolaRiS-Hub\n'
  printf 'POLARIS_DATA_REVISION=%q\n' "${POLARIS_DATA_REVISION}"
  printf 'INITIAL_CONDITIONS_PATH=%q\n' "${initial_conditions_path}"
  printf 'INITIAL_CONDITIONS_SHA256=%q\n' "${actual_initial_conditions_sha256}"
  printf 'INITIAL_CONDITIONS_METADATA_PATH=%q\n' "${initial_conditions_metadata_path}"
  printf 'INITIAL_CONDITIONS_METADATA_SHA256=%q\n' "${actual_initial_conditions_metadata_sha256}"
  printf 'SCENE_PATH=%q\n' "${scene_path}"
  printf 'SCENE_SHA256=%q\n' "${actual_scene_sha256}"
  printf 'SCENE_METADATA_PATH=%q\n' "${scene_metadata_path}"
  printf 'SCENE_METADATA_SHA256=%q\n' "${actual_scene_metadata_sha256}"
  printf 'ROBOT_ASSET_PATH=%q\n' "${robot_asset_path}"
  printf 'ROBOT_ASSET_SHA256=%q\n' "${actual_robot_asset_sha256}"
  printf 'ROBOT_ASSET_METADATA_PATH=%q\n' "${robot_metadata_path}"
  printf 'ROBOT_ASSET_METADATA_SHA256=%q\n' "${actual_robot_metadata_sha256}"
  printf 'CONTROL_MODE=joint-position\n'
  printf 'STATE_CONTRACT=7_joint_radians_plus_closed_positive_gripper\n'
  printf 'ACTION_CONTRACT=15x8_absolute_joint_targets_plus_closed_positive_gripper\n'
  printf 'OPEN_LOOP_HORIZON=%q\n' "${OPEN_LOOP_HORIZON}"
  printf 'CONTROL_FREQUENCY_HZ=15\n'
  printf 'WRIST_ROTATION_DEGREES=0\n'
  printf 'MODEL_IMAGE_SLOTS=base_0_rgb,left_wrist_0_rgb,right_wrist_0_rgb_masked\n'
  printf 'MODEL_REQUEST_IMAGE_SHAPE=720x1280x3_uint8\n'
  printf 'CLIENT_MODEL_SPATIAL_TRANSFORM=none\n'
  printf 'SERVER_MODEL_RESIZE=openpi_transforms_ResizeImages_openpi_client_PIL_bilinear_symmetric_zero_pad_224x224\n'
  printf 'VISUALIZATION_RESIZE=PIL_bilinear_pad_224x224_non_model_only\n'
  printf 'JOINTPOS_RUNTIME_CONTRACT_FILE=%q\n' "${RUNTIME_CONTRACT_FILE}"
} | tee "${METADATA_FILE}"

git -C "${POLARIS_DIR}" status --short --branch > "${RUN_DIR}/polaris_git_status.txt"
git -C "${POLARIS_DIR}" submodule status --recursive > "${RUN_DIR}/polaris_submodules.txt" 2>&1 || true
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total --format=csv,noheader \
  > "${RUN_DIR}/gpu_environment.csv"

server_command=(
  "${OPENPI_DIR}/.venv/bin/python"
  "${SCRIPT_DIR}/serve_pi05_droid_jointpos_attested.py"
  --checkpoint-dir "${checkpoint_path}"
  --openpi-dir "${OPENPI_DIR}"
  --manifest "${CHECKPOINT_MANIFEST}"
  --serving-contract-output "${SERVING_CONTRACT_FILE}"
  --model-runtime-contract-output "${MODEL_RUNTIME_CONTRACT_FILE}"
  --rng-stream-output "${RNG_STREAM_FILE}"
  --expected-request-count "${EXPECTED_POLICY_REQUESTS}"
  --port "${PORT}"
)

asset_manifest_command=(
  "${OPENPI_DIR}/.venv/bin/python"
  "${SCRIPT_DIR}/polaris_asset_dependency_manifest.py"
  --data-root "${POLARIS_DATA_DIR}"
  --task-subdir "${ASSET_SUBDIR}"
  --output "${ASSET_MANIFEST_FILE}"
)

eval_args=(
  scripts/eval.py
  --environment "${POLARIS_ENVIRONMENT}"
  --control-mode joint-position
  --policy.client DroidJointPos
  --policy.host 127.0.0.1
  --policy.port "${PORT}"
  --policy.open-loop-horizon "${OPEN_LOOP_HORIZON}"
  --policy.frame-description "robot base frame"
  --policy.action-frame robot_base
  --policy.dataset-name droid
  --policy.no-rotate-wrist-180
  --policy.state-type joint_position
  --policy.expected-action-horizon "${EXPECTED_ACTION_HORIZON}"
  --policy.expected-action-dim "${EXPECTED_ACTION_DIM}"
  --policy.trace-path "${TRACE_PATH}"
  --run-folder "${TASK_DIR}"
  --rollouts "${ROLLOUTS}"
  --environment-seed "${ENVIRONMENT_SEED}"
  --runtime-contract-path "${RUNTIME_CONTRACT_FILE}"
  --headless
)

pyxis_mounts="/dev/shm:/dev/shm,${POLARIS_DIR}:${POLARIS_DIR}:ro,${POLARIS_DATA_DIR}:${POLARIS_DATA_DIR}:ro,${RUN_DIR}:${RUN_DIR}:rw,${POLARIS_CACHE_DIR}:/cache:rw,${POLARIS_VULKAN_ICD_PATH}:/etc/vulkan/icd.d/nvidia_icd.json:ro"
eval_command=(
  srun --ntasks=1 "--cpus-per-task=${SLURM_CPUS_PER_TASK:-16}"
  "--container-image=${POLARIS_PYXIS_IMAGE}"
  "--container-mounts=${pyxis_mounts}"
  "--container-workdir=${POLARIS_DIR}"
  --no-container-entrypoint --no-container-mount-home --container-remap-root --container-writable
  "--container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES" --export=ALL
  /usr/bin/env -i
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  LANG=C.UTF-8 LC_ALL=C.UTF-8
  "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES}"
  "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES}"
  VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
  ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y OMNI_KIT_ALLOW_ROOT=1
  PYTHONUNBUFFERED=1
  "PYTHONPATH=${POLARIS_DIR}/src:${POLARIS_DIR}/third_party/openpi/packages/openpi-client/src"
  "POLARIS_DATA_PATH=${POLARIS_DATA_DIR}"
  XDG_CACHE_HOME=/cache HF_HOME=/cache/huggingface HOME=/cache/home
  /.venv/bin/python "${eval_args[@]}"
)

video_validation_args=(
  scripts/polaris/validate_pi05_droid_jointpos_videos.py
  --task-dir "${TASK_DIR}"
  --expected-rollouts "${ROLLOUTS}"
  --container-image-sha256 "${PYXIS_IMAGE_SHA256}"
  --output "${VIDEO_VALIDATION_FILE}"
)
video_validation_command=(
  srun --ntasks=1 "--cpus-per-task=${SLURM_CPUS_PER_TASK:-16}"
  "--container-image=${POLARIS_PYXIS_IMAGE}"
  "--container-mounts=${pyxis_mounts}"
  "--container-workdir=${POLARIS_DIR}"
  --no-container-entrypoint --no-container-mount-home --container-remap-root --container-writable
  /usr/bin/env PYTHONUNBUFFERED=1 "PYTHONPATH=${POLARIS_DIR}/src"
  /.venv/bin/python "${video_validation_args[@]}"
)

{
  printf '#!/usr/bin/env bash\nset -euo pipefail\n\n'
  printf 'cd %q\n' "${POLARIS_DIR}"
  printf '%q ' "${asset_manifest_command[@]}"
  printf '\n'
  printf 'env OPENPI_DATA_HOME=%q PYTHONPATH=%q JAX_PLATFORMS=cuda PYTHONWARNINGS=%q XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 XLA_PYTHON_CLIENT_PREALLOCATE=false ' \
    "${OPENPI_DATA_HOME}" "${POLARIS_DIR}/src" "${NUMPYDANTIC_STUB_WARNING_FILTER}"
  printf '%q ' "${server_command[@]}"
  printf '\ncd %q\n' "${POLARIS_DIR}"
  printf '%q ' "${eval_command[@]}"
  printf '\n'
  printf '%q ' "${video_validation_command[@]}"
  printf '\n'
} > "${COMMANDS_FILE}"
chmod +x "${COMMANDS_FILE}"

validate_live_server_attestation() {
  timeout 60 env \
    "PYTHONPATH=${POLARIS_DIR}/src:${OPENPI_DIR}/packages/openpi-client/src" \
    "${OPENPI_DIR}/.venv/bin/python" - \
    "${PORT}" "${SERVING_CONTRACT_FILE}" "${MODEL_RUNTIME_CONTRACT_FILE}" \
    "${SERVER_PID}" <<'PY'
import sys
from pathlib import Path

from openpi_client.websocket_client_policy import WebsocketClientPolicy
from polaris.pi05_droid_jointpos_serving_contract import (
    pi05_droid_jointpos_server_contract_sha256,
    validate_persisted_pi05_droid_jointpos_model_runtime,
    validate_persisted_pi05_droid_jointpos_serving_contract,
    validate_pi05_droid_jointpos_loopback_listener,
    validate_pi05_droid_jointpos_server_metadata,
)

validate_pi05_droid_jointpos_loopback_listener(int(sys.argv[4]), int(sys.argv[1]))
client = WebsocketClientPolicy(host="127.0.0.1", port=int(sys.argv[1]))
metadata = client.get_server_metadata()
contract = validate_pi05_droid_jointpos_server_metadata(metadata)
serving = validate_persisted_pi05_droid_jointpos_serving_contract(
    Path(sys.argv[2]), metadata
)
runtime = validate_persisted_pi05_droid_jointpos_model_runtime(
    Path(sys.argv[3]), metadata
)
print(
    serving["sha256"],
    runtime["sha256"],
    pi05_droid_jointpos_server_contract_sha256(contract),
)
PY
}

if [[ "${DRY_RUN}" == 1 ]]; then
  cat "${COMMANDS_FILE}"
  exit 0
fi

asset_manifest_result="$("${asset_manifest_command[@]}")" \
  || die "PolaRiS asset dependency manifest creation failed"
asset_manifest_fields="$(
  "${OPENPI_DIR}/.venv/bin/python" - "${asset_manifest_result}" <<'PY'
import json
import sys

value = json.loads(sys.argv[1])
fields = (
    "sha256",
    "manifest_sha256",
    "tree_sha256",
    "file_count",
    "total_bytes",
)
if set(fields) - set(value) or value.get("status") != "pass":
    raise SystemExit("asset dependency manifest result is incomplete")
print(*(value[field] for field in fields))
PY
)" || die "Cannot parse PolaRiS asset dependency manifest identity"
read -r ASSET_MANIFEST_ARTIFACT_SHA256 ASSET_MANIFEST_SHA256 \
  ASSET_TREE_SHA256 ASSET_FILE_COUNT ASSET_TOTAL_BYTES <<<"${asset_manifest_fields}"
for digest in "${ASSET_MANIFEST_ARTIFACT_SHA256}" "${ASSET_MANIFEST_SHA256}" \
  "${ASSET_TREE_SHA256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] \
    || die "PolaRiS asset dependency manifest returned an invalid SHA-256"
done
{
  printf 'POLARIS_ASSET_MANIFEST_FILE=%q\n' "${ASSET_MANIFEST_FILE}"
  printf 'POLARIS_ASSET_MANIFEST_ARTIFACT_SHA256=%q\n' \
    "${ASSET_MANIFEST_ARTIFACT_SHA256}"
  printf 'POLARIS_ASSET_MANIFEST_SHA256=%q\n' "${ASSET_MANIFEST_SHA256}"
  printf 'POLARIS_ASSET_TREE_SHA256=%q\n' "${ASSET_TREE_SHA256}"
  printf 'POLARIS_ASSET_FILE_COUNT=%q\n' "${ASSET_FILE_COUNT}"
  printf 'POLARIS_ASSET_TOTAL_BYTES=%q\n' "${ASSET_TOTAL_BYTES}"
} >> "${METADATA_FILE}"

if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
  die "Port ${PORT} is already in use"
fi

echo "[$(date -Iseconds)] Starting official pi0.5 policy server"
(
  cd "${POLARIS_DIR}"
  exec setsid env \
    OPENPI_DATA_HOME="${OPENPI_DATA_HOME}" \
    PYTHONPATH="${POLARIS_DIR}/src" \
    JAX_PLATFORMS=cuda \
    PYTHONWARNINGS="${NUMPYDANTIC_STUB_WARNING_FILTER}" \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    PYTHONUNBUFFERED=1 \
    "${server_command[@]}"
) > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

elapsed=0
until curl --fail --silent --show-error --max-time 2 "http://127.0.0.1:${PORT}/healthz" >/dev/null; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    wait "${SERVER_PID}" || server_code=$?
    tail -n 160 "${SERVER_LOG}" >&2 || true
    exit "${server_code:-1}"
  fi
  sleep 2
  elapsed=$((elapsed + 2))
  if (( elapsed >= SERVER_START_TIMEOUT_SECS )); then
    tail -n 160 "${SERVER_LOG}" >&2 || true
    die "Policy server did not become healthy within ${SERVER_START_TIMEOUT_SECS}s"
  fi
done
echo "[$(date -Iseconds)] Policy server healthy after ${elapsed}s"

kill -0 "${SERVER_PID}" 2>/dev/null \
  || die "Policy server exited before live contract validation"
server_attestation_line="$(validate_live_server_attestation)" \
  || die "Policy server metadata/artifact validation failed"
read -r SERVING_CONTRACT_SHA256 MODEL_RUNTIME_CONTRACT_SHA256 \
  SERVER_CONTRACT_SHA256 <<<"${server_attestation_line}"
for digest in "${SERVING_CONTRACT_SHA256}" "${MODEL_RUNTIME_CONTRACT_SHA256}" \
  "${SERVER_CONTRACT_SHA256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] \
    || die "Policy server attestation returned an invalid digest"
done
{
  printf 'SERVING_CONTRACT_FILE=%q\n' "${SERVING_CONTRACT_FILE}"
  printf 'SERVING_CONTRACT_SHA256=%q\n' "${SERVING_CONTRACT_SHA256}"
  printf 'MODEL_RUNTIME_CONTRACT_FILE=%q\n' "${MODEL_RUNTIME_CONTRACT_FILE}"
  printf 'MODEL_RUNTIME_CONTRACT_SHA256=%q\n' "${MODEL_RUNTIME_CONTRACT_SHA256}"
  printf 'SERVER_CONTRACT_SHA256=%q\n' "${SERVER_CONTRACT_SHA256}"
} >> "${METADATA_FILE}"
echo "[$(date -Iseconds)] Policy server live contract validated"

set +e
(
  cd "${POLARIS_DIR}"
  "${eval_command[@]}"
) 2>&1 | tee "${EVAL_LOG}"
pipeline_codes=("${PIPESTATUS[@]}")
set -e
eval_code="${pipeline_codes[0]}"
tee_code="${pipeline_codes[1]}"
(( eval_code == 0 )) || exit "${eval_code}"
(( tee_code == 0 )) || exit "${tee_code}"

python3 - "${EVAL_LOG}" "${ENVIRONMENT_SEED}" "${SERVER_CONTRACT_SHA256}" <<'PY'
import json
import sys

log_lines = list(open(sys.argv[1], encoding="utf-8"))

client_marker = "POLARIS_PI05_DROID_CONTRACT="
client_lines = [line for line in log_lines if client_marker in line]
if len(client_lines) != 1:
    raise SystemExit(f"expected one client contract marker, found {len(client_lines)}")
client_payload = json.loads(client_lines[0].split(client_marker, 1)[1])
expected_client = {
    "client": "DroidJointPos",
    "profile": "openpi_pi05_droid_native_joint_position_v2",
    "serving_profile": "openpi_pi05_droid_jointpos_polaris_flow_v1",
    "server_contract_sha256": sys.argv[3],
    "state": "ordered_7_panda_joint_radians_plus_closed_positive_gripper",
    "action": "7_absolute_panda_joint_targets_plus_closed_positive_gripper",
    "image_slots": [
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb_masked",
    ],
    "request_image_shape": [720, 1280, 3],
    "request_image_dtype": "uint8",
    "client_model_spatial_transform": None,
    "server_model_resize": (
        "openpi.transforms.ResizeImages_openpi_client_PIL_bilinear_"
        "symmetric_zero_pad_224x224"
    ),
    "model_image_resolution": [224, 224],
    "visualization_image_resolution": [224, 224],
    "visualization_spatial_transform": (
        "openpi_client.image_tools.resize_with_pad_PIL_bilinear_non_model"
    ),
    "wrist_rotation_degrees": 0,
    "open_loop_horizon": 8,
    "response_horizon": 15,
    "outer_steps": 450,
    "internal_max_episode_steps": 451,
    "initial_reset_index": -1,
    "initial_global_query_index": 0,
}
if client_payload != expected_client:
    raise SystemExit(f"client contract mismatch: {client_payload!r}")

environment_marker = "POLARIS_PI05_DROID_ENVIRONMENT_CONTRACT="
environment_lines = [line for line in log_lines if environment_marker in line]
if len(environment_lines) != 1:
    raise SystemExit(
        f"expected one environment contract marker, found {len(environment_lines)}"
    )
environment_payload = json.loads(
    environment_lines[0].split(environment_marker, 1)[1]
)
expected_environment = {
    "schema_version": 1,
    "profile": "isaaclab_env_seed_base_plus_episode_v1",
    "base_seed": int(sys.argv[2]),
    "scheme": "base_plus_episode_index_v1",
    "live_cfg_seed": int(sys.argv[2]),
    "physx_enhanced_determinism": False,
    "determinism_claim": "rng_bound_not_bitwise",
    "binding": "env_cfg_seed_before_gym_make_and_reset_seed_per_episode",
}
if environment_payload != expected_environment:
    raise SystemExit(f"environment contract mismatch: {environment_payload!r}")

runtime_marker = "POLARIS_PI05_DROID_JOINTPOS_RUNTIME="
runtime_lines = [line for line in log_lines if runtime_marker in line]
if len(runtime_lines) != 1:
    raise SystemExit(f"expected one joint-position runtime marker, found {len(runtime_lines)}")
PY
! grep -Fq 'Seed not set for the environment' "${EVAL_LOG}" \
  || die "Isaac Lab reported an unset environment seed"
grep -Eq "Environment seed[[:space:]]*:[[:space:]]*${ENVIRONMENT_SEED}([[:space:]]|$)" "${EVAL_LOG}" \
  || die "Isaac Lab did not report the expected live environment seed"
csv_path="${TASK_DIR}/eval_results.csv"
[[ -s "${csv_path}" ]] || die "Missing eval metrics: ${csv_path}"
PYTHONPATH="${POLARIS_DIR}/src" "${OPENPI_DIR}/.venv/bin/python" \
  "${POLARIS_DIR}/scripts/polaris/validate_pi05_trace.py" \
  "${TRACE_PATH}" --metrics-csv "${csv_path}" --expected-prompt "${EXPECTED_PROMPT}" \
  --expected-environment-seed "${ENVIRONMENT_SEED}" \
  --expected-server-contract-sha256 "${SERVER_CONTRACT_SHA256}" \
  --runtime-contract "${RUNTIME_CONTRACT_FILE}" \
  --output "${TRACE_SUMMARY}"
runtime_attestation_line="$(
  PYTHONPATH="${POLARIS_DIR}/src" "${OPENPI_DIR}/.venv/bin/python" - \
    "${EVAL_LOG}" "${RUNTIME_CONTRACT_FILE}" "${TRACE_SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

from polaris.pi05_droid_jointpos_runtime import (
    validate_jointpos_runtime_artifact,
    validate_jointpos_runtime_report,
)

marker = "POLARIS_PI05_DROID_JOINTPOS_RUNTIME="
lines = [line for line in open(sys.argv[1], encoding="utf-8") if marker in line]
if len(lines) != 1:
    raise SystemExit(f"expected one joint-position runtime marker, found {len(lines)}")
report = validate_jointpos_runtime_report(json.loads(lines[0].split(marker, 1)[1]))
artifact = validate_jointpos_runtime_artifact(
    Path(sys.argv[2]), expected_runtime_sha256=report["runtime_sha256"]
)
summary = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
if summary.get("runtime_contract_sha256") != report["runtime_sha256"]:
    raise SystemExit("trace summary runtime contract SHA-256 mismatch")
print(artifact["sha256"], report["runtime_sha256"])
PY
)" || die "Live joint-position runtime artifact validation failed"
read -r JOINTPOS_RUNTIME_ARTIFACT_SHA256 JOINTPOS_RUNTIME_SHA256 \
  <<<"${runtime_attestation_line}"
for digest in "${JOINTPOS_RUNTIME_ARTIFACT_SHA256}" "${JOINTPOS_RUNTIME_SHA256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] \
    || die "Joint-position runtime attestation returned an invalid digest"
done
{
  printf 'JOINTPOS_RUNTIME_ARTIFACT_SHA256=%q\n' \
    "${JOINTPOS_RUNTIME_ARTIFACT_SHA256}"
  printf 'JOINTPOS_RUNTIME_SHA256=%q\n' "${JOINTPOS_RUNTIME_SHA256}"
} >> "${METADATA_FILE}"
csv_rows="$(awk 'NR > 1 && NF {count += 1} END {print count + 0}' "${csv_path}")"
(( csv_rows == ROLLOUTS )) || die "Expected ${ROLLOUTS} CSV rows, got ${csv_rows}"
video_count="$(find "${TASK_DIR}" -maxdepth 1 -type f -name 'episode_*.mp4' -size +0c | wc -l)"
(( video_count == ROLLOUTS )) || die "Expected ${ROLLOUTS} videos, got ${video_count}"
terminal_image_count="$(find "${TASK_DIR}" -maxdepth 1 -type f -name 'episode_*_terminal.png' -size +0c | wc -l)"
(( terminal_image_count == ROLLOUTS )) \
  || die "Expected ${ROLLOUTS} terminal images, got ${terminal_image_count}"

kill -0 "${SERVER_PID}" 2>/dev/null \
  || die "Policy server exited before final contract validation"
final_server_attestation_line="$(validate_live_server_attestation)" \
  || die "Final policy server metadata/artifact validation failed"
[[ "${final_server_attestation_line}" == "${server_attestation_line}" ]] \
  || die "Policy server contract or immutable artifacts changed during evaluation"

final_asset_manifest_result="$(
  "${OPENPI_DIR}/.venv/bin/python" \
    "${SCRIPT_DIR}/polaris_asset_dependency_manifest.py" \
    --data-root "${POLARIS_DATA_DIR}" --verify "${ASSET_MANIFEST_FILE}"
)" || die "Final PolaRiS asset dependency verification failed"
[[ "${final_asset_manifest_result}" == "${asset_manifest_result}" ]] \
  || die "PolaRiS asset dependency identity changed during evaluation"

rng_server_pid="${SERVER_PID}"
kill -USR1 "${rng_server_pid}" \
  || die "Cannot request final policy RNG-stream attestation"
for _ in {1..600}; do
  kill -0 "${rng_server_pid}" 2>/dev/null || break
  if [[ -r "/proc/${rng_server_pid}/stat" ]] \
    && [[ "$(awk '{print $3}' "/proc/${rng_server_pid}/stat")" == Z ]]; then
    break
  fi
  sleep 0.2
done
if kill -0 "${rng_server_pid}" 2>/dev/null \
  && { [[ ! -r "/proc/${rng_server_pid}/stat" ]] \
    || [[ "$(awk '{print $3}' "/proc/${rng_server_pid}/stat")" != Z ]]; }; then
  kill -KILL "${rng_server_pid}" 2>/dev/null || true
  wait "${rng_server_pid}" 2>/dev/null || true
  SERVER_PID=""
  die "Policy server did not finalize its RNG stream within 120 seconds"
fi
server_final_code=0
wait "${rng_server_pid}" || server_final_code=$?
SERVER_PID=""
(( server_final_code == 0 )) \
  || die "Policy server RNG-stream finalization exited ${server_final_code}"
[[ -s "${RNG_STREAM_FILE}" && ! -L "${RNG_STREAM_FILE}" ]] \
  || die "Policy server did not publish its final RNG-stream artifact"
rng_stream_result="$(
  PYTHONPATH="${POLARIS_DIR}/src" "${OPENPI_DIR}/.venv/bin/python" \
    "${SCRIPT_DIR}/verify_pi05_droid_jointpos_rng_stream.py" \
    --rng-stream "${RNG_STREAM_FILE}" \
    --trace-summary "${TRACE_SUMMARY}" \
    --serving-contract "${SERVING_CONTRACT_FILE}" \
    --model-runtime "${MODEL_RUNTIME_CONTRACT_FILE}" \
    --expected-rollouts "${ROLLOUTS}" \
    --expected-server-pid "${rng_server_pid}" \
    --output "${REQUEST_PROOF_FILE}"
)" || die "Policy RNG stream does not exactly match evaluator requests"
rng_stream_fields="$(
  "${OPENPI_DIR}/.venv/bin/python" - "${rng_stream_result}" \
    "${EXPECTED_POLICY_REQUESTS}" <<'PY'
import json
import sys

artifact = json.loads(sys.argv[1])
value = artifact.get("value", {})
if (
    value.get("status") != "pass"
    or value.get("proof")
    != "trace_requests_equal_complete_official_policy_rng_stream"
    or value.get("request_count") != int(sys.argv[2])
):
    raise SystemExit("policy RNG-stream proof is incomplete")
print(
    value["rng_stream_artifact_sha256"],
    value["request_count"],
    artifact["sha256"],
)
PY
)" || die "Cannot parse policy RNG-stream proof"
read -r RNG_STREAM_ARTIFACT_SHA256 RNG_STREAM_REQUEST_COUNT \
  REQUEST_PROOF_ARTIFACT_SHA256 \
  <<<"${rng_stream_fields}"
[[ "${RNG_STREAM_ARTIFACT_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "Policy RNG-stream artifact returned an invalid SHA-256"
[[ "${RNG_STREAM_REQUEST_COUNT}" == "${EXPECTED_POLICY_REQUESTS}" ]] \
  || die "Policy RNG-stream request count mismatch"
[[ "${REQUEST_PROOF_ARTIFACT_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "Policy request-proof artifact returned an invalid SHA-256"
{
  printf 'POLICY_RNG_STREAM_FILE=%q\n' "${RNG_STREAM_FILE}"
  printf 'POLICY_RNG_STREAM_ARTIFACT_SHA256=%q\n' \
    "${RNG_STREAM_ARTIFACT_SHA256}"
  printf 'POLICY_RNG_STREAM_REQUEST_COUNT=%q\n' "${RNG_STREAM_REQUEST_COUNT}"
  printf 'POLICY_REQUEST_PROOF_FILE=%q\n' "${REQUEST_PROOF_FILE}"
  printf 'POLICY_REQUEST_PROOF_ARTIFACT_SHA256=%q\n' \
    "${REQUEST_PROOF_ARTIFACT_SHA256}"
  printf 'POLICY_SERVER_PID=%q\n' "${rng_server_pid}"
} >> "${METADATA_FILE}"

set +e
(
  cd "${POLARIS_DIR}"
  "${video_validation_command[@]}"
) > "${VIDEO_VALIDATION_LOG}" 2>&1
video_validation_code=$?
set -e
if (( video_validation_code != 0 )); then
  tail -n 160 "${VIDEO_VALIDATION_LOG}" >&2 || true
  die "Pinned Pyxis video validation failed with exit ${video_validation_code}"
fi
video_validation_fields="$(
  PYTHONPATH="${POLARIS_DIR}/src" "${OPENPI_DIR}/.venv/bin/python" - \
    "${VIDEO_VALIDATION_FILE}" "${ROLLOUTS}" <<'PY'
import sys
from pathlib import Path

from polaris.pi05_droid_jointpos_video import validate_persisted_video_report

artifact = validate_persisted_video_report(
    Path(sys.argv[1]), expected_rollouts=int(sys.argv[2])
)
environment = artifact["value"]["execution_environment"]
print(
    artifact["sha256"],
    environment["pyxis_image_sha256"],
    environment["tools"]["ffprobe"]["sha256"],
    environment["tools"]["ffmpeg"]["sha256"],
)
PY
)" || die "Cannot validate pinned Pyxis video evidence"
read -r VIDEO_VALIDATION_SHA256 VIDEO_PYXIS_SHA256 VIDEO_FFPROBE_SHA256 \
  VIDEO_FFMPEG_SHA256 <<<"${video_validation_fields}"
for digest in "${VIDEO_VALIDATION_SHA256}" "${VIDEO_PYXIS_SHA256}" \
  "${VIDEO_FFPROBE_SHA256}" "${VIDEO_FFMPEG_SHA256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] \
    || die "Video validation returned an invalid SHA-256"
done
[[ "${VIDEO_PYXIS_SHA256}" == "${PYXIS_IMAGE_SHA256}" ]] \
  || die "Video validation used a different Pyxis image"
{
  printf 'VIDEO_VALIDATION_FILE=%q\n' "${VIDEO_VALIDATION_FILE}"
  printf 'VIDEO_VALIDATION_SHA256=%q\n' "${VIDEO_VALIDATION_SHA256}"
  printf 'VIDEO_PYXIS_SHA256=%q\n' "${VIDEO_PYXIS_SHA256}"
  printf 'VIDEO_FFPROBE_SHA256=%q\n' "${VIDEO_FFPROBE_SHA256}"
  printf 'VIDEO_FFMPEG_SHA256=%q\n' "${VIDEO_FFMPEG_SHA256}"
  printf 'TERMINAL_IMAGE_COUNT=%q\n' "${terminal_image_count}"
} >> "${METADATA_FILE}"
[[ "$(git -C "${POLARIS_DIR}" rev-parse HEAD)" == "${POLARIS_COMMIT}" ]] \
  || die "PolaRiS commit changed during evaluation"
[[ -z "$(git -C "${POLARIS_DIR}" status --porcelain=v1 --untracked-files=all)" ]] \
  || die "PolaRiS source changed during evaluation"
[[ "$(git -C "${OPENPI_DIR}" rev-parse HEAD)" == "${OPENPI_COMMIT}" ]] \
  || die "OpenPI commit changed during evaluation"
[[ -z "$(git -C "${OPENPI_DIR}" status --porcelain=v1 --untracked-files=all)" ]] \
  || die "OpenPI source changed during evaluation"
[[ -f "${POLARIS_VULKAN_ICD_PATH}" && ! -L "${POLARIS_VULKAN_ICD_PATH}" ]] \
  || die "Vulkan ICD changed type during evaluation"
actual_vulkan_icd_sha256="$(sha256sum "${POLARIS_VULKAN_ICD_PATH}" | awk '{print $1}')"
capture_gpu_runtime
postrun_package_environment_sha256="$(capture_package_environment_sha256)" \
  || die "Postrun OpenPI package/seal attestation failed"
[[ "${actual_vulkan_icd_sha256}" == "${preflight_vulkan_icd_sha256}" \
   && "${actual_vulkan_icd_sha256}" == "${EXPECTED_VULKAN_ICD_SHA256}" \
   && "${actual_gpu_uuid}" == "${preflight_gpu_uuid}" \
   && "${actual_gpu_name}" == "${preflight_gpu_name}" \
   && "${actual_nvidia_driver_version}" == "${preflight_nvidia_driver_version}" ]] \
  || die "GPU/Vulkan runtime changed during evaluation"
[[ "${postrun_package_environment_sha256}" == \
   "${preflight_package_environment_sha256}" ]] \
  || die "OpenPI package/seal environment changed during evaluation"
{
  printf 'POSTRUN_POLARIS_VULKAN_ICD_SHA256=%q\n' "${actual_vulkan_icd_sha256}"
  printf 'POSTRUN_NVIDIA_GPU_UUID=%q\n' "${actual_gpu_uuid}"
  printf 'POSTRUN_NVIDIA_GPU_NAME=%q\n' "${actual_gpu_name}"
  printf 'POSTRUN_NVIDIA_DRIVER_VERSION=%q\n' \
    "${actual_nvidia_driver_version}"
  printf 'POSTRUN_PACKAGE_ENVIRONMENT_SHA256=%q\n' \
    "${postrun_package_environment_sha256}"
} >> "${METADATA_FILE}"
git -C "${POLARIS_DIR}" status --short --branch > "${RUN_DIR}/polaris_git_status.txt"
git -C "${POLARIS_DIR}" submodule status --recursive \
  > "${RUN_DIR}/polaris_submodules.txt"
printf 'EVALUATOR_EXIT_CODE=0\n' >> "${METADATA_FILE}"
evidence_result="$(
  PYTHONPATH="${POLARIS_DIR}/src" "${OPENPI_DIR}/.venv/bin/python" \
    -m polaris.pi05_droid_jointpos_evidence \
    --run-dir "${RUN_DIR}" \
    --task-dir "${TASK_DIR}" \
    --environment "${POLARIS_ENVIRONMENT}" \
    --expected-environment-seed "${ENVIRONMENT_SEED}" \
    --expected-rollouts "${ROLLOUTS}" \
    --polaris-commit "${POLARIS_COMMIT}"
)" || die "Joint-position immutable evidence transaction failed"
EVIDENCE_MANIFEST_SHA256="$(
  "${OPENPI_DIR}/.venv/bin/python" - "${evidence_result}" <<'PY'
import json
import sys

value = json.loads(sys.argv[1])
digest = value.get("manifest", {}).get("sha256")
if not isinstance(digest, str) or len(digest) != 64:
    raise SystemExit("invalid immutable evidence manifest identity")
print(digest)
PY
)" || die "Cannot parse immutable evidence manifest identity"
[[ "${EVIDENCE_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "Immutable evidence manifest returned an invalid SHA-256"
EVIDENCE_FINALIZED=1
publish_terminal_marker "${TASK_DIR}/SUCCESS" \
  "status=success" "completed_at=$(date -Iseconds)" \
  "evidence_manifest_sha256=${EVIDENCE_MANIFEST_SHA256}"
echo "Evaluation complete: ${RUN_DIR}"
