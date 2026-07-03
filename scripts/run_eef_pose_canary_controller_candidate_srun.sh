#!/usr/bin/env bash
# Run one model-free controller-candidate replay inside an approved L40 job.
set -euo pipefail

: "${SLURM_JOB_ID:?run inside an sbatch allocation}"
: "${SLURM_NODELIST:?missing Slurm node list}"
: "${CANDIDATE_VARIANT:?official_lap3b or reasoning_43075}"
: "${CANDIDATE_LAUNCH_ID:?required 64-hex launch token}"
: "${CANDIDATE_OUTPUT_ROOT:?required durable output root}"
: "${CANDIDATE_POLARIS_REPO:?required cluster PolaRiS checkout}"
: "${CANDIDATE_POLARIS_COMMIT:?required full PolaRiS commit}"
: "${CANDIDATE_CONTAINER_IMAGE:?required immutable Pyxis image path}"
: "${CANDIDATE_CONTAINER_MOUNTS:?required comma-separated Pyxis mounts}"
: "${CANDIDATE_POLARIS_DATA_PATH:?required mounted PolaRiS-Hub path}"
: "${CANDIDATE_CACHE_ROOT:?required mounted writable runtime cache}"
: "${CANDIDATE_HOST_CACHE_ROOT:?required host side of writable runtime cache}"
: "${CANDIDATE_CONTAINER_SHA256:?required container digest}"
: "${CANDIDATE_RUNNER_SHA256:?required candidate runner digest}"
: "${CANDIDATE_VALIDATOR_SHA256:?required candidate validator digest}"
: "${CANDIDATE_FAILURE_VERIFIER_SHA256:?required failure-verifier digest}"
: "${CANDIDATE_SAFETY_VALIDATOR_SHA256:?required production-equivalent safety validator digest}"
: "${CANDIDATE_GATE0_HELPER_SHA256:?required Gate0 helper digest}"
: "${CANDIDATE_FIXTURE_SHA256:?required variant fixture digest}"
: "${CANDIDATE_STATUS_WRITER_SHA256:?required status-writer digest}"
: "${CANDIDATE_FINALIZER_SHA256:?required finalizer digest}"
: "${CANDIDATE_JOB_SCRIPT_SHA256:?required this script digest}"
CANDIDATE_CONTAINER_PYTHON="${CANDIDATE_CONTAINER_PYTHON:-/.venv/bin/python}"

case "${CANDIDATE_VARIANT}" in
  official_lap3b)
    fixture_filename="official_lap3b_job1098292_gate0_actions.json"
    ;;
  reasoning_43075)
    fixture_filename="reasoning_43075_job1098294_gate0_actions.json"
    ;;
  *) echo "invalid CANDIDATE_VARIANT=${CANDIDATE_VARIANT}" >&2; exit 2 ;;
esac
[[ "${CANDIDATE_LAUNCH_ID}" =~ ^[0-9a-f]{64}$ ]]
[[ "${CANDIDATE_POLARIS_COMMIT}" =~ ^[0-9a-f]{40}$ ]]
for digest in \
  "${CANDIDATE_CONTAINER_SHA256}" \
  "${CANDIDATE_RUNNER_SHA256}" \
  "${CANDIDATE_VALIDATOR_SHA256}" \
  "${CANDIDATE_FAILURE_VERIFIER_SHA256}" \
  "${CANDIDATE_SAFETY_VALIDATOR_SHA256}" \
  "${CANDIDATE_GATE0_HELPER_SHA256}" \
  "${CANDIDATE_FIXTURE_SHA256}" \
  "${CANDIDATE_STATUS_WRITER_SHA256}" \
  "${CANDIDATE_FINALIZER_SHA256}" \
  "${CANDIDATE_JOB_SCRIPT_SHA256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]]
done

repo_commit="$(git -C "${CANDIDATE_POLARIS_REPO}" rev-parse HEAD)"
[[ "${repo_commit}" == "${CANDIDATE_POLARIS_COMMIT}" ]]
[[ -z "$(git -C "${CANDIDATE_POLARIS_REPO}" status --porcelain --untracked-files=no)" ]]
[[ "$(sha256sum "$0" | awk '{print $1}')" == "${CANDIDATE_JOB_SCRIPT_SHA256}" ]]
[[ "$(sha256sum "${CANDIDATE_POLARIS_REPO}/scripts/smoke_eef_pose_canary_controller_candidate.py" | awk '{print $1}')" == "${CANDIDATE_RUNNER_SHA256}" ]]
[[ "$(sha256sum "${CANDIDATE_POLARIS_REPO}/scripts/validate_eef_pose_canary_controller_candidate.py" | awk '{print $1}')" == "${CANDIDATE_VALIDATOR_SHA256}" ]]
[[ "$(sha256sum "${CANDIDATE_POLARIS_REPO}/scripts/verify_eef_pose_canary_controller_candidate_failure.py" | awk '{print $1}')" == "${CANDIDATE_FAILURE_VERIFIER_SHA256}" ]]
[[ "$(sha256sum "${CANDIDATE_POLARIS_REPO}/scripts/finalize_eef_pose_smoke.py" | awk '{print $1}')" == "${CANDIDATE_SAFETY_VALIDATOR_SHA256}" ]]
[[ "$(sha256sum "${CANDIDATE_POLARIS_REPO}/scripts/smoke_eef_pose_canary_trace_replay.py" | awk '{print $1}')" == "${CANDIDATE_GATE0_HELPER_SHA256}" ]]
[[ "$(sha256sum "${CANDIDATE_POLARIS_REPO}/scripts/write_eef_pose_canary_controller_candidate_srun_status.py" | awk '{print $1}')" == "${CANDIDATE_STATUS_WRITER_SHA256}" ]]
[[ "$(sha256sum "${CANDIDATE_POLARIS_REPO}/scripts/finalize_eef_pose_canary_controller_candidate.py" | awk '{print $1}')" == "${CANDIDATE_FINALIZER_SHA256}" ]]
fixture_path="${CANDIDATE_POLARIS_REPO}/scripts/fixtures/${fixture_filename}"
[[ "$(sha256sum "${fixture_path}" | awk '{print $1}')" == "${CANDIDATE_FIXTURE_SHA256}" ]]
[[ -f "${CANDIDATE_CONTAINER_IMAGE}" && ! -L "${CANDIDATE_CONTAINER_IMAGE}" ]]
container_size_bytes="$(stat -c '%s' -- "${CANDIDATE_CONTAINER_IMAGE}")"
[[ "${container_size_bytes}" =~ ^[1-9][0-9]*$ ]]
[[ "$(sha256sum "${CANDIDATE_CONTAINER_IMAGE}" | awk '{print $1}')" == "${CANDIDATE_CONTAINER_SHA256}" ]]
[[ "${CANDIDATE_CACHE_ROOT}" == /* ]]
[[ "${CANDIDATE_HOST_CACHE_ROOT}" == /* ]]
[[ -d "${CANDIDATE_HOST_CACHE_ROOT}" && ! -L "${CANDIDATE_HOST_CACHE_ROOT}" ]]
case ",${CANDIDATE_CONTAINER_MOUNTS}," in
  *",${CANDIDATE_HOST_CACHE_ROOT}:${CANDIDATE_CACHE_ROOT},"* | \
  *",${CANDIDATE_HOST_CACHE_ROOT}:${CANDIDATE_CACHE_ROOT}:rw,"*) ;;
  *) echo "candidate host/container cache mount is absent" >&2; exit 2 ;;
esac

# Fail before allocating simulator time if the actual host interpreter cannot
# import every post-srun consumer and its transitive dependencies.
for host_consumer in \
  validate_eef_pose_canary_controller_candidate.py \
  verify_eef_pose_canary_controller_candidate_failure.py \
  write_eef_pose_canary_controller_candidate_srun_status.py \
  finalize_eef_pose_canary_controller_candidate.py; do
  PYTHONPATH="${CANDIDATE_POLARIS_REPO}/src:${CANDIDATE_POLARIS_REPO}/scripts" \
    python3 "${CANDIDATE_POLARIS_REPO}/scripts/${host_consumer}" --help >/dev/null
done

namespace="${CANDIDATE_OUTPUT_ROOT}/${CANDIDATE_VARIANT}/job_${SLURM_JOB_ID}/launch_${CANDIDATE_LAUNCH_ID}"
mkdir -p "${namespace}"
raw_result="${namespace}/candidate-${CANDIDATE_VARIANT}.raw.json"
srun_status="${namespace}/candidate-${CANDIDATE_VARIANT}.srun-status.json"
attestation="${namespace}/candidate-${CANDIDATE_VARIANT}.attestation.json"
saved_job_script="${namespace}/candidate-${CANDIDATE_VARIANT}.job.sh"
gpu_inventory="${namespace}/gpu-inventory.txt"
job_metadata="${namespace}/slurm-job.txt"
stdout_log="${namespace}/srun.stdout.log"
stderr_log="${namespace}/srun.stderr.log"

for path in \
  "${raw_result}" \
  "${raw_result}.ready.json" \
  "${srun_status}" \
  "${attestation}" \
  "${saved_job_script}" \
  "${gpu_inventory}" \
  "${job_metadata}" \
  "${stdout_log}" \
  "${stderr_log}"; do
  [[ ! -e "${path}" ]]
done
cp -- "$0" "${saved_job_script}"
chmod 0444 "${saved_job_script}"
[[ "$(sha256sum "${saved_job_script}" | awk '{print $1}')" == "${CANDIDATE_JOB_SCRIPT_SHA256}" ]]
nvidia-smi -q >"${gpu_inventory}"
scontrol show job -dd "${SLURM_JOB_ID}" >"${job_metadata}"
chmod 0444 "${gpu_inventory}" "${job_metadata}"

cache_suffix="${CANDIDATE_VARIANT}/job_${SLURM_JOB_ID}/launch_${CANDIDATE_LAUNCH_ID}"
host_cache_namespace="${CANDIDATE_HOST_CACHE_ROOT%/}/${cache_suffix}"
container_cache_namespace="${CANDIDATE_CACHE_ROOT%/}/${cache_suffix}"
[[ ! -e "${host_cache_namespace}" && ! -L "${host_cache_namespace}" ]]
mkdir -p "${host_cache_namespace}"
chmod 0700 "${host_cache_namespace}"
cleanup_cache() {
  rm -rf -- "${host_cache_namespace}"
}
trap cleanup_cache EXIT

handle_failed_srun() {
  local original_srun_rc="$1"
  local timestamp_rc=0
  local log_chmod_rc=0
  local failure_verify_rc=0

  date +%s%N >/dev/null
  timestamp_rc=$?
  if [[ "${timestamp_rc}" -ne 0 ]]; then
    echo "Controller-candidate failure timestamp failed rc=${timestamp_rc}" >&2
  fi

  chmod 0444 "${stdout_log}" "${stderr_log}"
  log_chmod_rc=$?
  if [[ "${log_chmod_rc}" -ne 0 ]]; then
    echo "Controller-candidate failure log chmod failed rc=${log_chmod_rc}" >&2
  fi

  PYTHONPATH="${CANDIDATE_POLARIS_REPO}/src:${CANDIDATE_POLARIS_REPO}/scripts" \
  python3 "${CANDIDATE_POLARIS_REPO}/scripts/verify_eef_pose_canary_controller_candidate_failure.py" \
    --variant "${CANDIDATE_VARIANT}" \
    --launch-id "${CANDIDATE_LAUNCH_ID}" \
    --job-id "${SLURM_JOB_ID}" \
    --raw-result "${raw_result}" \
    --polaris-repo "${CANDIDATE_POLARIS_REPO}" \
    --expected-polaris-commit "${CANDIDATE_POLARIS_COMMIT}" \
    --expected-runner-sha256 "${CANDIDATE_RUNNER_SHA256}" \
    --expected-validator-sha256 "${CANDIDATE_VALIDATOR_SHA256}" \
    --expected-failure-verifier-sha256 "${CANDIDATE_FAILURE_VERIFIER_SHA256}" \
    --expected-safety-validator-sha256 "${CANDIDATE_SAFETY_VALIDATOR_SHA256}" \
    --expected-gate0-helper-sha256 "${CANDIDATE_GATE0_HELPER_SHA256}" \
    --expected-fixture-sha256 "${CANDIDATE_FIXTURE_SHA256}" \
    --container-image "${CANDIDATE_CONTAINER_IMAGE}" \
    --expected-container-size-bytes "${container_size_bytes}" \
    --expected-container-sha256 "${CANDIDATE_CONTAINER_SHA256}"
  failure_verify_rc=$?
  if [[ "${failure_verify_rc}" -ne 0 ]]; then
    echo "Controller-candidate failure verification failed rc=${failure_verify_rc}" >&2
  else
    echo "Controller-candidate failure transaction verified; no promotion" >&2
  fi
  echo "Controller-candidate srun failed rc=${original_srun_rc}; returning original rc" >&2
  exit "${original_srun_rc}"
}

started_at_ns="$(date +%s%N)"
set +e
srun \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=16 \
  --gpus-per-task=1 \
  --kill-on-bad-exit=1 \
  --container-image="${CANDIDATE_CONTAINER_IMAGE}" \
  --container-mounts="${CANDIDATE_CONTAINER_MOUNTS}" \
  --container-workdir="${CANDIDATE_POLARIS_REPO}" \
  --no-container-entrypoint \
  --no-container-mount-home \
  --container-remap-root \
  --container-writable \
  --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES \
  --export=ALL \
  --output="${stdout_log}" \
  --error="${stderr_log}" \
  /usr/bin/env \
    VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json \
    ACCEPT_EULA=Y \
    OMNI_KIT_ACCEPT_EULA=YES \
    PRIVACY_CONSENT=Y \
    OMNI_KIT_ALLOW_ROOT=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${CANDIDATE_POLARIS_REPO}/src" \
    POLARIS_DATA_PATH="${CANDIDATE_POLARIS_DATA_PATH}" \
    XDG_CACHE_HOME="${container_cache_namespace}" \
    HF_HOME="${container_cache_namespace}/huggingface" \
    HOME="${container_cache_namespace}/home" \
    GIT_CONFIG_COUNT=1 \
    GIT_CONFIG_KEY_0=safe.directory \
    GIT_CONFIG_VALUE_0="${CANDIDATE_POLARIS_REPO}" \
  "${CANDIDATE_CONTAINER_PYTHON}" \
    scripts/smoke_eef_pose_canary_controller_candidate.py \
    --variant "${CANDIDATE_VARIANT}" \
    --launch-id "${CANDIDATE_LAUNCH_ID}" \
    --expected-polaris-commit "${CANDIDATE_POLARIS_COMMIT}" \
    --output-json "${raw_result}" \
    --container-image "${CANDIDATE_CONTAINER_IMAGE}" \
    --expected-container-size-bytes "${container_size_bytes}" \
    --expected-container-sha256 "${CANDIDATE_CONTAINER_SHA256}" \
    --device cuda:0
srun_rc=$?
if [[ "${srun_rc}" -ne 0 ]]; then
  handle_failed_srun "${srun_rc}"
fi
set -e
returned_at_ns="$(date +%s%N)"
chmod 0444 "${stdout_log}" "${stderr_log}"

PYTHONPATH="${CANDIDATE_POLARIS_REPO}/src:${CANDIDATE_POLARIS_REPO}/scripts" \
python3 "${CANDIDATE_POLARIS_REPO}/scripts/write_eef_pose_canary_controller_candidate_srun_status.py" \
  --variant "${CANDIDATE_VARIANT}" \
  --launch-id "${CANDIDATE_LAUNCH_ID}" \
  --job-id "${SLURM_JOB_ID}" \
  --srun-rc "${srun_rc}" \
  --srun-started-at-ns "${started_at_ns}" \
  --srun-returned-at-ns "${returned_at_ns}" \
  --raw-result "${raw_result}" \
  --status "${srun_status}"

PYTHONPATH="${CANDIDATE_POLARIS_REPO}/src:${CANDIDATE_POLARIS_REPO}/scripts" \
python3 "${CANDIDATE_POLARIS_REPO}/scripts/finalize_eef_pose_canary_controller_candidate.py" finalize \
  --variant "${CANDIDATE_VARIANT}" \
  --launch-id "${CANDIDATE_LAUNCH_ID}" \
  --job-id "${SLURM_JOB_ID}" \
  --raw-result "${raw_result}" \
  --srun-status "${srun_status}" \
  --attestation "${attestation}" \
  --polaris-repo "${CANDIDATE_POLARIS_REPO}" \
  --expected-polaris-commit "${CANDIDATE_POLARIS_COMMIT}" \
  --expected-runner-sha256 "${CANDIDATE_RUNNER_SHA256}" \
  --expected-validator-sha256 "${CANDIDATE_VALIDATOR_SHA256}" \
  --expected-failure-verifier-sha256 "${CANDIDATE_FAILURE_VERIFIER_SHA256}" \
  --expected-safety-validator-sha256 "${CANDIDATE_SAFETY_VALIDATOR_SHA256}" \
  --expected-gate0-helper-sha256 "${CANDIDATE_GATE0_HELPER_SHA256}" \
  --expected-fixture-sha256 "${CANDIDATE_FIXTURE_SHA256}" \
  --expected-status-writer-sha256 "${CANDIDATE_STATUS_WRITER_SHA256}" \
  --expected-finalizer-sha256 "${CANDIDATE_FINALIZER_SHA256}" \
  --container-image "${CANDIDATE_CONTAINER_IMAGE}" \
  --expected-container-size-bytes "${container_size_bytes}" \
  --expected-container-sha256 "${CANDIDATE_CONTAINER_SHA256}" \
  --runtime-job-script "$0" \
  --saved-job-script "${saved_job_script}" \
  --expected-saved-job-script-sha256 "${CANDIDATE_JOB_SCRIPT_SHA256}" \
  --gpu-inventory "${gpu_inventory}" \
  --job-metadata "${job_metadata}" \
  --stdout-log "${stdout_log}" \
  --stderr-log "${stderr_log}"

echo "POLARIS_CONTROLLER_CANDIDATE_COMPLETE=${attestation}"
