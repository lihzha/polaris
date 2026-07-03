#!/usr/bin/env bash
# Run inside one already-approved Slurm L40 allocation.  Submit one copy per
# variant so official/reasoning have independent job, srun, and output lives.
set -euo pipefail

: "${SLURM_JOB_ID:?run inside an sbatch allocation}"
: "${SLURM_NODELIST:?missing Slurm node list}"
: "${GATE0_VARIANT:?official_lap3b or reasoning_43075}"
: "${GATE0_LAUNCH_ID:?required 64-hex launch token}"
: "${GATE0_OUTPUT_ROOT:?required durable output root}"
: "${GATE0_POLARIS_REPO:?required cluster PolaRiS checkout}"
: "${GATE0_POLARIS_COMMIT:?required full PolaRiS commit}"
: "${GATE0_CONTAINER_IMAGE:?required immutable Pyxis image path}"
: "${GATE0_CONTAINER_MOUNTS:?required comma-separated Pyxis mounts}"
: "${GATE0_POLARIS_DATA_PATH:?required mounted PolaRiS-Hub path}"
: "${GATE0_CACHE_ROOT:?required mounted writable runtime cache}"
: "${GATE0_CONTAINER_SHA256:?required container digest}"
: "${GATE0_RUNNER_SHA256:?required runner digest}"
: "${GATE0_FIXTURE_SHA256:?required variant fixture digest}"
: "${GATE0_GENERATOR_SHA256:?required generator digest}"
: "${GATE0_STATUS_WRITER_SHA256:?required status-writer digest}"
: "${GATE0_FINALIZER_SHA256:?required finalizer digest}"
: "${GATE0_JOB_SCRIPT_SHA256:?required submitted job-script digest}"
GATE0_CONTAINER_PYTHON="${GATE0_CONTAINER_PYTHON:-/.venv/bin/python}"

case "${GATE0_VARIANT}" in
  official_lap3b)
    fixture_filename="official_lap3b_job1098292_gate0_actions.json"
    ;;
  reasoning_43075)
    fixture_filename="reasoning_43075_job1098294_gate0_actions.json"
    ;;
  *) echo "invalid GATE0_VARIANT=${GATE0_VARIANT}" >&2; exit 2 ;;
esac
[[ "${GATE0_LAUNCH_ID}" =~ ^[0-9a-f]{64}$ ]]
[[ "${GATE0_POLARIS_COMMIT}" =~ ^[0-9a-f]{40}$ ]]
for digest in \
  "${GATE0_CONTAINER_SHA256}" \
  "${GATE0_RUNNER_SHA256}" \
  "${GATE0_FIXTURE_SHA256}" \
  "${GATE0_GENERATOR_SHA256}" \
  "${GATE0_STATUS_WRITER_SHA256}" \
  "${GATE0_FINALIZER_SHA256}" \
  "${GATE0_JOB_SCRIPT_SHA256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]]
done

repo_commit="$(git -C "${GATE0_POLARIS_REPO}" rev-parse HEAD)"
[[ "${repo_commit}" == "${GATE0_POLARIS_COMMIT}" ]]
[[ -z "$(git -C "${GATE0_POLARIS_REPO}" status --porcelain --untracked-files=no)" ]]
[[ "$(sha256sum "$0" | awk '{print $1}')" == "${GATE0_JOB_SCRIPT_SHA256}" ]]
[[ "$(sha256sum "${GATE0_POLARIS_REPO}/scripts/smoke_eef_pose_canary_trace_replay.py" | awk '{print $1}')" == "${GATE0_RUNNER_SHA256}" ]]
[[ "$(sha256sum "${GATE0_POLARIS_REPO}/scripts/generate_eef_pose_canary_trace_fixtures.py" | awk '{print $1}')" == "${GATE0_GENERATOR_SHA256}" ]]
[[ "$(sha256sum "${GATE0_POLARIS_REPO}/scripts/write_eef_pose_canary_gate0_srun_status.py" | awk '{print $1}')" == "${GATE0_STATUS_WRITER_SHA256}" ]]
[[ "$(sha256sum "${GATE0_POLARIS_REPO}/scripts/finalize_eef_pose_canary_trace_replay.py" | awk '{print $1}')" == "${GATE0_FINALIZER_SHA256}" ]]
fixture_path="${GATE0_POLARIS_REPO}/scripts/fixtures/${fixture_filename}"
[[ "$(sha256sum "${fixture_path}" | awk '{print $1}')" == "${GATE0_FIXTURE_SHA256}" ]]

namespace="${GATE0_OUTPUT_ROOT}/${GATE0_VARIANT}/job_${SLURM_JOB_ID}/launch_${GATE0_LAUNCH_ID}"
mkdir -p "${namespace}"
raw_result="${namespace}/gate0-${GATE0_VARIANT}.raw.json"
srun_status="${namespace}/gate0-${GATE0_VARIANT}.srun-status.json"
attestation="${namespace}/gate0-${GATE0_VARIANT}.attestation.json"
saved_job_script="${namespace}/gate0-${GATE0_VARIANT}.job.sh"

for path in \
  "${raw_result}" \
  "${raw_result}.ready.json" \
  "${srun_status}" \
  "${attestation}" \
  "${saved_job_script}"; do
  [[ ! -e "${path}" ]]
done
cp -- "$0" "${saved_job_script}"
chmod 0444 "${saved_job_script}"
saved_job_script_sha256="$(sha256sum "${saved_job_script}" | awk '{print $1}')"
[[ "${saved_job_script_sha256}" == "${GATE0_JOB_SCRIPT_SHA256}" ]]

started_at_ns="$(date +%s%N)"
set +e
srun \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=16 \
  --gpus-per-task=1 \
  --kill-on-bad-exit=1 \
  --container-image="${GATE0_CONTAINER_IMAGE}" \
  --container-mounts="${GATE0_CONTAINER_MOUNTS}" \
  --container-workdir="${GATE0_POLARIS_REPO}" \
  --no-container-entrypoint \
  --no-container-mount-home \
  --container-remap-root \
  --container-writable \
  --container-env=NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES \
  --export=ALL \
  --output="${namespace}/srun.stdout.log" \
  --error="${namespace}/srun.stderr.log" \
  /usr/bin/env \
    VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json \
    ACCEPT_EULA=Y \
    OMNI_KIT_ACCEPT_EULA=YES \
    PRIVACY_CONSENT=Y \
    OMNI_KIT_ALLOW_ROOT=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${GATE0_POLARIS_REPO}/src" \
    POLARIS_DATA_PATH="${GATE0_POLARIS_DATA_PATH}" \
    XDG_CACHE_HOME="${GATE0_CACHE_ROOT}" \
    HF_HOME="${GATE0_CACHE_ROOT}/huggingface" \
    HOME="${GATE0_CACHE_ROOT}/home" \
    GIT_CONFIG_COUNT=1 \
    GIT_CONFIG_KEY_0=safe.directory \
    GIT_CONFIG_VALUE_0="${GATE0_POLARIS_REPO}" \
  "${GATE0_CONTAINER_PYTHON}" scripts/smoke_eef_pose_canary_trace_replay.py \
    --variant "${GATE0_VARIANT}" \
    --launch-id "${GATE0_LAUNCH_ID}" \
    --expected-polaris-commit "${GATE0_POLARIS_COMMIT}" \
    --output-json "${raw_result}" \
    --device cuda:0
srun_rc=$?
set -e
returned_at_ns="$(date +%s%N)"

if [[ "${srun_rc}" -ne 0 ]]; then
  echo "Gate 0 srun failed rc=${srun_rc}; no status/attestation promotion" >&2
  exit "${srun_rc}"
fi

python3 "${GATE0_POLARIS_REPO}/scripts/write_eef_pose_canary_gate0_srun_status.py" \
  --variant "${GATE0_VARIANT}" \
  --launch-id "${GATE0_LAUNCH_ID}" \
  --job-id "${SLURM_JOB_ID}" \
  --srun-rc "${srun_rc}" \
  --srun-started-at-ns "${started_at_ns}" \
  --srun-returned-at-ns "${returned_at_ns}" \
  --raw-result "${raw_result}" \
  --status "${srun_status}"

python3 "${GATE0_POLARIS_REPO}/scripts/finalize_eef_pose_canary_trace_replay.py" finalize \
  --variant "${GATE0_VARIANT}" \
  --launch-id "${GATE0_LAUNCH_ID}" \
  --job-id "${SLURM_JOB_ID}" \
  --raw-result "${raw_result}" \
  --srun-status "${srun_status}" \
  --attestation "${attestation}" \
  --polaris-repo "${GATE0_POLARIS_REPO}" \
  --expected-polaris-commit "${GATE0_POLARIS_COMMIT}" \
  --expected-runner-sha256 "${GATE0_RUNNER_SHA256}" \
  --expected-fixture-sha256 "${GATE0_FIXTURE_SHA256}" \
  --expected-generator-sha256 "${GATE0_GENERATOR_SHA256}" \
  --expected-status-writer-sha256 "${GATE0_STATUS_WRITER_SHA256}" \
  --expected-finalizer-sha256 "${GATE0_FINALIZER_SHA256}" \
  --container-image "${GATE0_CONTAINER_IMAGE}" \
  --expected-container-sha256 "${GATE0_CONTAINER_SHA256}" \
  --runtime-job-script "$0" \
  --saved-job-script "${saved_job_script}" \
  --expected-saved-job-script-sha256 "${GATE0_JOB_SCRIPT_SHA256}"

echo "POLARIS_GATE0_COMPLETE=${attestation}"
