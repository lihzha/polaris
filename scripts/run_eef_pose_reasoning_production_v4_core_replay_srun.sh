#!/usr/bin/env bash
set -Eeuo pipefail
umask 022

required=(
  FULLTRACE_POLARIS_REPO
  FULLTRACE_REPLAY_COMMIT
  FULLTRACE_CONTAINER_IMAGE
  FULLTRACE_CONTAINER_SHA256
  FULLTRACE_POLARIS_DATA_PATH
  FULLTRACE_OUTPUT_ROOT
  FULLTRACE_HOST_CACHE_ROOT
  FULLTRACE_LAUNCH_ID
  FULLTRACE_RUNNER_SHA256
  FULLTRACE_VALIDATOR_SHA256
  FULLTRACE_SAFETY_VALIDATOR_SHA256
  FULLTRACE_FIXTURE_SHA256
  FULLTRACE_FIXTURE_BUILDER_SHA256
  FULLTRACE_TEST_SHA256
  FULLTRACE_GATE_IO_SHA256
  FULLTRACE_WRAPPER_SHA256
  SLURM_JOB_ID
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || {
    printf 'missing required environment variable: %s\n' "${name}" >&2
    exit 2
  }
done

[[ "${FULLTRACE_REPLAY_COMMIT}" =~ ^[0-9a-f]{40}$ ]]
[[ "${FULLTRACE_LAUNCH_ID}" =~ ^[0-9a-f]{64}$ ]]
hash_variables=(
  FULLTRACE_CONTAINER_SHA256
  FULLTRACE_RUNNER_SHA256
  FULLTRACE_VALIDATOR_SHA256
  FULLTRACE_SAFETY_VALIDATOR_SHA256
  FULLTRACE_FIXTURE_SHA256
  FULLTRACE_FIXTURE_BUILDER_SHA256
  FULLTRACE_TEST_SHA256
  FULLTRACE_GATE_IO_SHA256
  FULLTRACE_WRAPPER_SHA256
)
for name in "${hash_variables[@]}"; do
  [[ "${!name}" =~ ^[0-9a-f]{64}$ ]]
done
[[ "${SLURM_NTASKS:-}" == 1 ]]
[[ "${SLURM_PROCID:-0}" == 0 ]]
[[ "${SLURM_LOCALID:-0}" == 0 ]]

readonly variant=production_v4_core_ramp16
readonly production_base_commit=7fc74d648328432a7f9f06d13c0e82a03f73a0c1
readonly replay_publication_fix_commit=d32115d36f2dea510dee86edeaddcc58309afc2e
readonly replay_validation_fix_commit=585ab6f72098fd67118fd8b33cdd90be809bed3a
readonly replay_implementation_commit=2ebfe7db5b2a31887481781b214608976e8023db
readonly replay_parent_commit=e18b8ebbc26fd309d8e45bd58bef9c867948098a
readonly container_size_bytes=7183130624
readonly container_sha256=ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a
readonly host_vulkan_icd=/usr/share/vulkan/icd.d/nvidia_icd.json
readonly host_vulkan_icd_sha256=7bdb6f27d35b66fc848df6f94b8773bba30ea3a7f06f114100d14154a235a34b
readonly container_vulkan_icd=/etc/vulkan/icd.d/nvidia_icd.json
readonly zero_sidecar_sha256=9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa
readonly five_sidecar_sha256=f0b5c2c2211c8d67ed15e75e656c7862d086e9245420892a7de62cd9ec582a06
readonly isaac_pytest_bootstrap_sha256=d2814cd641820e08f64daf88903c4b972efb3ae1d336d0b0c5e42924eb781dd5
readonly robust_ik_test_sha256=5af936fcd0227672f7738b6e6f45a505f942403702fc4de3b0a5af982b46ed5c
readonly safety_validator_sha256=74dccfeb25c9522e5741eb72510f3f7940abd64678be8a357aca102fe2038fc7
readonly gate_io_sha256=34b1cc6b493d2e0e078bd5769fb44a68d824a4deac8f58a54aefc41f83641cdb

readonly runner="${FULLTRACE_POLARIS_REPO}/scripts/smoke_eef_pose_reasoning_production_v4_core_replay.py"
readonly validator="${FULLTRACE_POLARIS_REPO}/scripts/validate_eef_pose_reasoning_production_v4_core_replay.py"
readonly safety_validator="${FULLTRACE_POLARIS_REPO}/scripts/finalize_eef_pose_smoke.py"
readonly fixture="${FULLTRACE_POLARIS_REPO}/scripts/fixtures/reasoning_43075_job1098523_fulltrace_actions.json"
readonly fixture_builder="${FULLTRACE_POLARIS_REPO}/scripts/build_reasoning_fulltrace_replay_fixture.py"
readonly focused_test="${FULLTRACE_POLARIS_REPO}/tests/test_smoke_eef_pose_reasoning_production_v4_core_replay.py"
readonly gate_io="${FULLTRACE_POLARIS_REPO}/scripts/eef_pose_reasoning_production_v4_core_gate_io.py"
wrapper="$(readlink -f "$0")"
readonly wrapper

[[ "$(git -C "${FULLTRACE_POLARIS_REPO}" rev-parse HEAD)" == "${FULLTRACE_REPLAY_COMMIT}" ]]
[[ "$(git -C "${FULLTRACE_POLARIS_REPO}" rev-parse HEAD^)" == "${replay_publication_fix_commit}" ]]
[[ "$(git -C "${FULLTRACE_POLARIS_REPO}" rev-parse HEAD^^)" == "${replay_validation_fix_commit}" ]]
[[ "$(git -C "${FULLTRACE_POLARIS_REPO}" rev-parse HEAD^^^)" == "${replay_implementation_commit}" ]]
[[ "$(git -C "${FULLTRACE_POLARIS_REPO}" rev-parse HEAD^^^^)" == "${replay_parent_commit}" ]]
[[ "$(git -C "${FULLTRACE_POLARIS_REPO}" rev-parse HEAD^^^^^)" == "${production_base_commit}" ]]
[[ -z "$(git -C "${FULLTRACE_POLARIS_REPO}" branch --show-current)" ]]
[[ -z "$(git -C "${FULLTRACE_POLARIS_REPO}" status --porcelain --untracked-files=all)" ]]
[[ "${FULLTRACE_CONTAINER_SHA256}" == "${container_sha256}" ]]
[[ "$(sha256sum "${FULLTRACE_CONTAINER_IMAGE}" | awk '{print $1}')" == "${container_sha256}" ]]
[[ "$(stat -c '%s' "${FULLTRACE_CONTAINER_IMAGE}")" == "${container_size_bytes}" ]]
[[ "$(sha256sum "${host_vulkan_icd}" | awk '{print $1}')" == "${host_vulkan_icd_sha256}" ]]
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)" == "NVIDIA L40S" ]]
[[ "$(sha256sum "${runner}" | awk '{print $1}')" == "${FULLTRACE_RUNNER_SHA256}" ]]
[[ "$(sha256sum "${validator}" | awk '{print $1}')" == "${FULLTRACE_VALIDATOR_SHA256}" ]]
[[ "${FULLTRACE_SAFETY_VALIDATOR_SHA256}" == "${safety_validator_sha256}" ]]
[[ "$(sha256sum "${safety_validator}" | awk '{print $1}')" == "${FULLTRACE_SAFETY_VALIDATOR_SHA256}" ]]
[[ "$(sha256sum "${fixture}" | awk '{print $1}')" == "${FULLTRACE_FIXTURE_SHA256}" ]]
[[ "$(sha256sum "${fixture_builder}" | awk '{print $1}')" == "${FULLTRACE_FIXTURE_BUILDER_SHA256}" ]]
[[ "$(sha256sum "${focused_test}" | awk '{print $1}')" == "${FULLTRACE_TEST_SHA256}" ]]
[[ "${FULLTRACE_GATE_IO_SHA256}" == "${gate_io_sha256}" ]]
[[ "$(sha256sum "${gate_io}" | awk '{print $1}')" == "${FULLTRACE_GATE_IO_SHA256}" ]]
[[ "$(sha256sum "${wrapper}" | awk '{print $1}')" == "${FULLTRACE_WRAPPER_SHA256}" ]]
[[ "$(sha256sum "${FULLTRACE_POLARIS_REPO}/scripts/run_isaac_pytest.py" | awk '{print $1}')" == "${isaac_pytest_bootstrap_sha256}" ]]
[[ "$(sha256sum "${FULLTRACE_POLARIS_REPO}/tests/test_robust_differential_ik.py" | awk '{print $1}')" == "${robust_ik_test_sha256}" ]]
[[ "$(sha256sum "${FULLTRACE_POLARIS_DATA_PATH}/nvidia_droid/noninstanceable.usd" | awk '{print $1}')" == d8379925b103963dbf3e7c85bcc4ae101b81b7c1d7dabe7d2e964f41d069ec44 ]]
[[ "$(sha256sum "${FULLTRACE_POLARIS_DATA_PATH}/food_bussing/initial_conditions.json" | awk '{print $1}')" == 40091faee14f692350220871d30705294f21f17ae3d2974cd3c09a34d560f5de ]]
[[ "$(sha256sum "${FULLTRACE_POLARIS_DATA_PATH}/food_bussing/scene.usda" | awk '{print $1}')" == 82cd641e422935b394ce7ea7b6be55214c9952a2544000222921e544c409b489 ]]

root_record=""
if ! root_record="$(
  /usr/bin/python3 "${gate_io}" validate-roots \
    --output-root "${FULLTRACE_OUTPUT_ROOT}" \
    --cache-root "${FULLTRACE_HOST_CACHE_ROOT}"
)"; then
  printf 'lexical output/cache root validation failed\n' >&2
  exit 2
fi
IFS=$'\t' read -r output_root cache_root extra_root_field <<<"${root_record}"
[[ -n "${output_root}" && -n "${cache_root}" && -z "${extra_root_field}" ]]
readonly root_record output_root cache_root extra_root_field
readonly attempt_parent="${output_root}/${variant}"
readonly cache_parent="${cache_root}/${variant}"
mkdir -p -- "${attempt_parent}" "${cache_parent}"
[[ -d "${attempt_parent}" && ! -L "${attempt_parent}" ]]
[[ -d "${cache_parent}" && ! -L "${cache_parent}" ]]
[[ "$(realpath -e -- "${attempt_parent}")" == "${attempt_parent}" ]]
[[ "$(realpath -e -- "${cache_parent}")" == "${cache_parent}" ]]
readonly namespace="job_${SLURM_JOB_ID}_restart_${SLURM_RESTART_COUNT:-0}"
[[ "${namespace}" =~ ^job_[0-9]+_restart_[0-9]+$ ]]
readonly attempt="${attempt_parent}/${namespace}"
readonly cache="${cache_parent}/${namespace}"
readonly result_json="${attempt}/result.json"
readonly video="${attempt}/video.mp4"
readonly manifest="${attempt}/validation_manifest.json"
readonly failure_marker="${attempt}/FAILED"
readonly success_marker="${attempt}/SUCCESS"
readonly negative_exit="${attempt}/isaac_pytest_negative.exit"
readonly positive_exit="${attempt}/isaac_pytest_positive.exit"
readonly success_temporary="${attempt}/.SUCCESS.${SLURM_JOB_ID}.tmp"

start_epoch="$(date +%s)"
status=1
attempt_created=0
cache_created=0
write_failure_marker() {
  if [[ "${status}" != 0 && "${attempt_created}" == 1 && -d "${attempt}" && ! -L "${attempt}" ]]; then
    if (
      set -o noclobber
      printf 'profile=production_v4_core_fulltrace_wrapper_failure_v1\njob_id=%s\nvariant=%s\nstatus=%s\n' \
        "${SLURM_JOB_ID}" "${variant}" "${status}" >"${failure_marker}"
    ) 2>/dev/null; then
      chmod 0444 "${failure_marker}"
    fi
  fi
}

cleanup_cache() {
  if [[ "${cache_created}" != 1 ]]; then
    return 0
  fi
  if [[ "$(dirname -- "${cache}")" != "${cache_parent}" ]] ||
    [[ "$(basename -- "${cache}")" != "${namespace}" ]] ||
    [[ ! -d "${cache}" ]] || [[ -L "${cache}" ]]; then
    return 1
  fi
  rm -rf -- "${cache}"
  cache_created=0
}

finish() {
  local rc="$?"
  trap - EXIT
  if [[ "${status}" != 0 ]] &&
    { [[ "$(git -C "${FULLTRACE_POLARIS_REPO}" rev-parse HEAD)" != "${FULLTRACE_REPLAY_COMMIT}" ]] ||
      [[ -n "$(git -C "${FULLTRACE_POLARIS_REPO}" status --porcelain=v1 --untracked-files=all)" ]]; }; then
    rc=1
    status=1
  fi
  cleanup_cache || {
    rc=1
    status=1
  }
  if [[ "${rc}" != 0 ]]; then
    status=1
  fi
  write_failure_marker
  printf 'FULLTRACE_END=%s\n' "$(date -Is 2>/dev/null || true)" || true
  printf 'FULLTRACE_WALLTIME_SECONDS=%s\n' "$(( $(date +%s) - start_epoch ))" || true
  printf 'FULLTRACE_WRAPPER_EXIT_CODE=%s\n' "${rc}" || true
  exit "${rc}"
}
trap finish EXIT

[[ ! -e "${attempt}" && ! -L "${attempt}" ]]
mkdir -- "${attempt}"
attempt_created=1
[[ -d "${attempt}" && ! -L "${attempt}" ]]
[[ ! -e "${cache}" && ! -L "${cache}" ]]
mkdir -- "${cache}"
cache_created=1
[[ -d "${cache}" && ! -L "${cache}" ]]
mkdir -- "${cache}/home" "${cache}/huggingface"

readonly isaaclab_root=/.venv/lib/python3.11/site-packages/isaaclab/source/isaaclab
readonly isaaclab_tasks_root=/.venv/lib/python3.11/site-packages/isaaclab/source/isaaclab_tasks
readonly isaaclab_assets_root=/.venv/lib/python3.11/site-packages/isaaclab/source/isaaclab_assets
readonly container_pythonpath="${FULLTRACE_POLARIS_REPO}/src:${FULLTRACE_POLARIS_REPO}/scripts:${isaaclab_root}:${isaaclab_tasks_root}:${isaaclab_assets_root}"
readonly record="${attempt}/run_record.env"
printf 'PROFILE=%s\nSLURM_JOB_ID=%s\nSLURM_RESTART_COUNT=%s\nVARIANT=%s\nLAUNCH_ID=%s\nREPLAY_COMMIT=%s\nREPLAY_PUBLICATION_FIX_COMMIT=%s\nREPLAY_VALIDATION_FIX_COMMIT=%s\nREPLAY_IMPLEMENTATION_COMMIT=%s\nREPLAY_PARENT_COMMIT=%s\nPRODUCTION_BASE_COMMIT=%s\nCONTAINER_IMAGE=%s\nCONTAINER_SHA256=%s\nCONTAINER_PYTHONPATH=%s\nRESULT_JSON=%s\nVIDEO=%s\nMANIFEST=%s\n' \
  production_v4_core_fulltrace_srun_v1 \
  "${SLURM_JOB_ID}" "${SLURM_RESTART_COUNT:-0}" "${variant}" \
  "${FULLTRACE_LAUNCH_ID}" "${FULLTRACE_REPLAY_COMMIT}" \
  "${replay_publication_fix_commit}" \
  "${replay_validation_fix_commit}" \
  "${replay_implementation_commit}" "${replay_parent_commit}" \
  "${production_base_commit}" "${FULLTRACE_CONTAINER_IMAGE}" \
  "${FULLTRACE_CONTAINER_SHA256}" "${container_pythonpath}" "${result_json}" "${video}" \
  "${manifest}" >"${record}"
chmod 0444 "${record}"

readonly mounts="/dev/shm:/dev/shm,${FULLTRACE_POLARIS_REPO}:${FULLTRACE_POLARIS_REPO}:ro,${FULLTRACE_POLARIS_DATA_PATH}:${FULLTRACE_POLARIS_DATA_PATH}:ro,${FULLTRACE_CONTAINER_IMAGE}:${FULLTRACE_CONTAINER_IMAGE}:ro,${attempt}:${attempt}:rw,${cache}:/cache:rw,${host_vulkan_icd}:${container_vulkan_icd}:ro"
container_base=(
  --nodes=1
  --ntasks=1
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}"
  --gpus-per-task=1
  --kill-on-bad-exit=1
  --container-image="${FULLTRACE_CONTAINER_IMAGE}"
  --container-mounts="${mounts}"
  --container-workdir="${FULLTRACE_POLARIS_REPO}"
  --no-container-entrypoint
  --no-container-mount-home
  --container-remap-root
  --container-writable
  "--container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES"
  --export=ALL
  /usr/bin/env
  VK_DRIVER_FILES="${container_vulkan_icd}"
  ACCEPT_EULA=Y
  OMNI_KIT_ACCEPT_EULA=YES
  PRIVACY_CONSENT=Y
  OMNI_KIT_ALLOW_ROOT=1
  PYTHONDONTWRITEBYTECODE=1
  PYTHONUNBUFFERED=1
  OPENBLAS_NUM_THREADS=1
  OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1
  NUMEXPR_NUM_THREADS=1
  PYTHONPATH="${container_pythonpath}"
  GIT_CONFIG_COUNT=1
  GIT_CONFIG_KEY_0=safe.directory
  GIT_CONFIG_VALUE_0="${FULLTRACE_POLARIS_REPO}"
  POLARIS_DATA_PATH="${FULLTRACE_POLARIS_DATA_PATH}"
  XDG_CACHE_HOME=/cache
  HF_HOME=/cache/huggingface
  HOME=/cache/home
)

validate_exit_sidecar() {
  local path="$1"
  local expected_code="$2"
  local expected_sha="$3"
  local temporary
  temporary="$(dirname "${path}")/.$(basename "${path}").tmp"
  [[ -f "${path}" && ! -L "${path}" && ! -e "${temporary}" ]]
  [[ "$(stat -c '%a' "${path}")" == 444 ]]
  [[ "$(stat -c '%h' "${path}")" == 1 ]]
  [[ "$(stat -c '%s' "${path}")" == 2 ]]
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected_sha}" ]]
  [[ "$(tr -d '\n' <"${path}")" == "${expected_code}" ]]
}

set +e
srun "${container_base[@]}" \
  "ISAAC_PYTEST_EXIT_CODE_FILE=${negative_exit}" \
  /.venv/bin/python scripts/run_isaac_pytest.py -q \
  tests/test_robust_differential_ik.py \
  -k __polaris_deliberate_no_tests_selected__ \
  -p no:cacheprovider
negative_srun_rc="$?"
set -e
[[ "${negative_srun_rc}" == 0 || "${negative_srun_rc}" == 5 ]]
validate_exit_sidecar "${negative_exit}" 5 "${five_sidecar_sha256}"
if validate_exit_sidecar "${negative_exit}" 0 "${zero_sidecar_sha256}"; then
  printf 'nonzero Isaac pytest sidecar passed the zero gate\n' >&2
  exit 1
fi

set +e
srun "${container_base[@]}" \
  "ISAAC_PYTEST_EXIT_CODE_FILE=${positive_exit}" \
  /.venv/bin/python scripts/run_isaac_pytest.py -q \
  tests/test_robust_differential_ik.py \
  -p no:cacheprovider
positive_srun_rc="$?"
set -e
[[ "${positive_srun_rc}" == 0 ]]
validate_exit_sidecar "${positive_exit}" 0 "${zero_sidecar_sha256}"

srun "${container_base[@]}" \
  /.venv/bin/python -m pytest -q \
  tests/test_smoke_eef_pose_reasoning_production_v4_core_replay.py \
  tests/test_run_isaac_pytest.py \
  -p no:cacheprovider

set +e
srun "${container_base[@]}" \
  /.venv/bin/python "${runner}" \
    --expected-polaris-commit "${FULLTRACE_REPLAY_COMMIT}" \
    --launch-id "${FULLTRACE_LAUNCH_ID}" \
    --output-json "${result_json}" \
    --output-video "${video}" \
    --container-image "${FULLTRACE_CONTAINER_IMAGE}" \
    --expected-container-size-bytes "${container_size_bytes}" \
    --expected-container-sha256 "${FULLTRACE_CONTAINER_SHA256}" \
    --device cuda:0
simulator_srun_rc="$?"
set -e
[[ "${simulator_srun_rc}" == 0 ]]

srun "${container_base[@]}" \
  /.venv/bin/python "${validator}" \
    --expected-replay-commit "${FULLTRACE_REPLAY_COMMIT}" \
    --expected-job-id "${SLURM_JOB_ID}" \
    --expected-launch-id "${FULLTRACE_LAUNCH_ID}" \
    --result-json "${result_json}" \
    --video "${video}" \
    --output-manifest "${manifest}" \
    --simulator-srun-exit-code "${simulator_srun_rc}"

[[ "$(git -C "${FULLTRACE_POLARIS_REPO}" rev-parse HEAD)" == "${FULLTRACE_REPLAY_COMMIT}" ]]
[[ -z "$(git -C "${FULLTRACE_POLARIS_REPO}" status --porcelain=v1 --untracked-files=all)" ]]
cleanup_cache
manifest_sha="$(sha256sum "${manifest}" | awk '{print $1}')"
result_sha="$(sha256sum "${result_json}" | awk '{print $1}')"
video_sha="$(sha256sum "${video}" | awk '{print $1}')"
readonly manifest_sha result_sha video_sha
[[ ! -e "${success_temporary}" && ! -e "${success_marker}" ]]
if ! /usr/bin/python3 "${gate_io}" publish-success \
  --temporary "${success_temporary}" \
  --marker "${success_marker}" \
  --job-id "${SLURM_JOB_ID}" \
  --variant "${variant}" \
  --result-sha256 "${result_sha}" \
  --video-sha256 "${video_sha}" \
  --manifest-sha256 "${manifest_sha}"; then
  printf 'SUCCESS publication failed\n' >&2
  exit 1
fi
status=0
printf 'FULLTRACE_PRODUCTION_V4_SUCCESS=%s\n' "${success_marker}" || true
