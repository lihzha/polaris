#!/usr/bin/env bash
set -euo pipefail

required=(
  FULLTRACE_POLARIS_REPO
  FULLTRACE_DIAGNOSTIC_COMMIT
  FULLTRACE_CONTAINER_IMAGE
  FULLTRACE_CONTAINER_SHA256
  FULLTRACE_POLARIS_DATA_PATH
  FULLTRACE_OUTPUT_ROOT
  FULLTRACE_HOST_CACHE_ROOT
  FULLTRACE_VARIANT
  FULLTRACE_LAUNCH_ID
  FULLTRACE_RUNNER_SHA256
  FULLTRACE_VALIDATOR_SHA256
  FULLTRACE_FIXTURE_SHA256
  FULLTRACE_WRAPPER_SHA256
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || {
    printf 'missing required environment variable: %s\n' "${name}" >&2
    exit 2
  }
done

case "${FULLTRACE_VARIANT}" in
  cap8_abrupt_release|cap24_abrupt_release|cap5_release_ramp16) ;;
  *)
    printf 'invalid FULLTRACE_VARIANT=%s\n' "${FULLTRACE_VARIANT}" >&2
    exit 2
    ;;
esac
[[ "${FULLTRACE_DIAGNOSTIC_COMMIT}" =~ ^[0-9a-f]{40}$ ]]
[[ "${FULLTRACE_LAUNCH_ID}" =~ ^[0-9a-f]{64}$ ]]
[[ "${SLURM_NTASKS:-}" == 1 ]]
[[ "${SLURM_PROCID:-0}" == 0 ]]
[[ "${SLURM_LOCALID:-0}" == 0 ]]

base_commit=0611d384f5f26ef9bd8ff114be273e875c3fe719
diagnostic_base_commit=26f75a1aeb2e6342d45f96d746ee101be02764f5
runner=${FULLTRACE_POLARIS_REPO}/scripts/smoke_eef_pose_reasoning_fulltrace_ablation.py
validator=${FULLTRACE_POLARIS_REPO}/scripts/validate_eef_pose_reasoning_fulltrace_ablation.py
fixture=${FULLTRACE_POLARIS_REPO}/scripts/fixtures/reasoning_43075_job1098523_fulltrace_actions.json
wrapper=$(readlink -f "$0")

[[ "$(git -C "${FULLTRACE_POLARIS_REPO}" rev-parse HEAD)" == "${FULLTRACE_DIAGNOSTIC_COMMIT}" ]]
[[ "$(git -C "${FULLTRACE_POLARIS_REPO}" rev-parse HEAD^)" == "${diagnostic_base_commit}" ]]
[[ "$(git -C "${FULLTRACE_POLARIS_REPO}" rev-parse HEAD^^)" == "${base_commit}" ]]
[[ -z "$(git -C "${FULLTRACE_POLARIS_REPO}" branch --show-current)" ]]
[[ -z "$(git -C "${FULLTRACE_POLARIS_REPO}" status --porcelain --untracked-files=all)" ]]
[[ "$(sha256sum "${FULLTRACE_CONTAINER_IMAGE}" | awk '{print $1}')" == "${FULLTRACE_CONTAINER_SHA256}" ]]
[[ "$(stat -c '%s' "${FULLTRACE_CONTAINER_IMAGE}")" == 7183130624 ]]
[[ "$(sha256sum "${runner}" | awk '{print $1}')" == "${FULLTRACE_RUNNER_SHA256}" ]]
[[ "$(sha256sum "${validator}" | awk '{print $1}')" == "${FULLTRACE_VALIDATOR_SHA256}" ]]
[[ "$(sha256sum "${fixture}" | awk '{print $1}')" == "${FULLTRACE_FIXTURE_SHA256}" ]]
[[ "$(sha256sum "${wrapper}" | awk '{print $1}')" == "${FULLTRACE_WRAPPER_SHA256}" ]]
[[ "$(sha256sum "${FULLTRACE_POLARIS_DATA_PATH}/nvidia_droid/noninstanceable.usd" | awk '{print $1}')" == d8379925b103963dbf3e7c85bcc4ae101b81b7c1d7dabe7d2e964f41d069ec44 ]]
[[ "$(sha256sum "${FULLTRACE_POLARIS_DATA_PATH}/food_bussing/initial_conditions.json" | awk '{print $1}')" == 40091faee14f692350220871d30705294f21f17ae3d2974cd3c09a34d560f5de ]]
[[ "$(sha256sum "${FULLTRACE_POLARIS_DATA_PATH}/food_bussing/scene.usda" | awk '{print $1}')" == 82cd641e422935b394ce7ea7b6be55214c9952a2544000222921e544c409b489 ]]

attempt=${FULLTRACE_OUTPUT_ROOT}/${FULLTRACE_VARIANT}/job_${SLURM_JOB_ID}_restart_${SLURM_RESTART_COUNT:-0}
cache=${FULLTRACE_HOST_CACHE_ROOT}/${FULLTRACE_VARIANT}/job_${SLURM_JOB_ID}_restart_${SLURM_RESTART_COUNT:-0}
[[ ! -e "${attempt}" && ! -L "${attempt}" ]]
mkdir -p "${attempt}" "${cache}/home" "${cache}/huggingface"
[[ ! -L "${attempt}" && ! -L "${cache}" ]]
result_json=${attempt}/result.json
video=${attempt}/video.mp4
manifest=${attempt}/validation_manifest.json
failure_marker=${attempt}/FAILED
success_marker=${attempt}/SUCCESS

status=1
write_failure_marker() {
  if [[ "${status}" != 0 ]]; then
    rm -f "${success_marker}"
  fi
  if [[ "${status}" != 0 && ! -e "${failure_marker}" ]]; then
    printf 'profile=reasoning_fulltrace_cap_release_followup_wrapper_failure_v1\njob_id=%s\nvariant=%s\nstatus=%s\n' \
      "${SLURM_JOB_ID}" "${FULLTRACE_VARIANT}" "${status}" >"${failure_marker}"
    chmod 0444 "${failure_marker}"
  fi
}
trap write_failure_marker EXIT

record=${attempt}/run_record.env
printf 'PROFILE=%s\nSLURM_JOB_ID=%s\nSLURM_RESTART_COUNT=%s\nVARIANT=%s\nLAUNCH_ID=%s\nDIAGNOSTIC_COMMIT=%s\nDIAGNOSTIC_BASE_COMMIT=%s\nPRODUCTION_BASE_COMMIT=%s\nCONTAINER_IMAGE=%s\nCONTAINER_SHA256=%s\nRESULT_JSON=%s\nVIDEO=%s\nMANIFEST=%s\n' \
  reasoning_fulltrace_cap_release_followup_srun_v1 \
  "${SLURM_JOB_ID}" "${SLURM_RESTART_COUNT:-0}" "${FULLTRACE_VARIANT}" \
  "${FULLTRACE_LAUNCH_ID}" "${FULLTRACE_DIAGNOSTIC_COMMIT}" "${diagnostic_base_commit}" "${base_commit}" \
  "${FULLTRACE_CONTAINER_IMAGE}" "${FULLTRACE_CONTAINER_SHA256}" \
  "${result_json}" "${video}" "${manifest}" >"${record}"
chmod 0444 "${record}"

mounts=/dev/shm:/dev/shm,${FULLTRACE_POLARIS_REPO}:${FULLTRACE_POLARIS_REPO}:ro,${FULLTRACE_POLARIS_DATA_PATH}:${FULLTRACE_POLARIS_DATA_PATH}:ro,${attempt}:${attempt}:rw,${cache}:/cache:rw,/usr/share/vulkan/icd.d/nvidia_icd.json:/etc/vulkan/icd.d/nvidia_icd.json:ro

srun \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
  --gpus-per-task=1 \
  --kill-on-bad-exit=1 \
  --container-image="${FULLTRACE_CONTAINER_IMAGE}" \
  --container-mounts="${mounts}" \
  --container-workdir="${FULLTRACE_POLARIS_REPO}" \
  --no-container-entrypoint \
  --no-container-mount-home \
  --container-remap-root \
  --container-writable \
  --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES \
  --export=ALL \
  /usr/bin/env \
    VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json \
    ACCEPT_EULA=Y \
    OMNI_KIT_ACCEPT_EULA=YES \
    PRIVACY_CONSENT=Y \
    OMNI_KIT_ALLOW_ROOT=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${FULLTRACE_POLARIS_REPO}/src:${FULLTRACE_POLARIS_REPO}/scripts" \
    GIT_CONFIG_COUNT=1 \
    GIT_CONFIG_KEY_0=safe.directory \
    GIT_CONFIG_VALUE_0="${FULLTRACE_POLARIS_REPO}" \
    POLARIS_DATA_PATH="${FULLTRACE_POLARIS_DATA_PATH}" \
    XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/huggingface \
    HOME=/cache/home \
    /.venv/bin/python "${runner}" \
      --variant "${FULLTRACE_VARIANT}" \
      --expected-polaris-commit "${FULLTRACE_DIAGNOSTIC_COMMIT}" \
      --launch-id "${FULLTRACE_LAUNCH_ID}" \
      --output-json "${result_json}" \
      --output-video "${video}" \
      --container-image "${FULLTRACE_CONTAINER_IMAGE}" \
      --expected-container-size-bytes 7183130624 \
      --expected-container-sha256 "${FULLTRACE_CONTAINER_SHA256}" \
      --device cuda:0

srun \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
  --gpus-per-task=1 \
  --kill-on-bad-exit=1 \
  --container-image="${FULLTRACE_CONTAINER_IMAGE}" \
  --container-mounts="${mounts}" \
  --container-workdir="${FULLTRACE_POLARIS_REPO}" \
  --no-container-entrypoint \
  --no-container-mount-home \
  --container-remap-root \
  --container-writable \
  --export=ALL \
  /usr/bin/env \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${FULLTRACE_POLARIS_REPO}/src:${FULLTRACE_POLARIS_REPO}/scripts" \
    /.venv/bin/python "${validator}" \
      --variant "${FULLTRACE_VARIANT}" \
      --expected-diagnostic-commit "${FULLTRACE_DIAGNOSTIC_COMMIT}" \
      --expected-job-id "${SLURM_JOB_ID}" \
      --expected-launch-id "${FULLTRACE_LAUNCH_ID}" \
      --result-json "${result_json}" \
      --video "${video}" \
      --output-manifest "${manifest}" \
      --simulator-srun-exit-code 0

manifest_sha=$(sha256sum "${manifest}" | awk '{print $1}')
result_sha=$(sha256sum "${result_json}" | awk '{print $1}')
video_sha=$(sha256sum "${video}" | awk '{print $1}')
success_temporary=${attempt}/.SUCCESS.${SLURM_JOB_ID}.tmp
[[ ! -e "${success_temporary}" && ! -e "${success_marker}" ]]
printf 'profile=reasoning_fulltrace_cap_release_followup_success_v1\njob_id=%s\nvariant=%s\nresult_sha256=%s\nvideo_sha256=%s\nmanifest_sha256=%s\n' \
  "${SLURM_JOB_ID}" "${FULLTRACE_VARIANT}" "${result_sha}" "${video_sha}" \
  "${manifest_sha}" >"${success_temporary}"
chmod 0444 "${success_temporary}"
sync -f "${success_temporary}"
mv -T "${success_temporary}" "${success_marker}"
status=0
trap - EXIT
printf 'FULLTRACE_SUCCESS=%s\n' "${success_marker}"
