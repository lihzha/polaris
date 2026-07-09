#!/usr/bin/env bash

# Submit ordinary, independently restartable official pi0.5 PolaRiS eval jobs.

set -Eeuo pipefail

die() {
  echo "$*" >&2
  exit 2
}

write_atomic_text() {
  local path="$1"
  local mode="$2"
  local value="$3"
  local directory temporary
  directory="$(dirname "${path}")"
  temporary="${directory}/.$(basename "${path}").tmp.${BASHPID}.${RANDOM}"
  (
    umask 077
    printf '%s\n' "${value}" > "${temporary}"
  )
  chmod "${mode}" "${temporary}"
  sync -- "${temporary}"
  mv -f -- "${temporary}" "${path}"
  sync -- "${directory}"
}

append_manifest_row() {
  local row="$1"
  local directory temporary
  directory="$(dirname "${SUBMISSION_MANIFEST}")"
  temporary="${directory}/.$(basename "${SUBMISSION_MANIFEST}").tmp.${BASHPID}.${RANDOM}"
  cp -- "${SUBMISSION_MANIFEST}" "${temporary}"
  printf '%s\n' "${row}" >> "${temporary}"
  sync -- "${temporary}"
  mv -f -- "${temporary}" "${SUBMISSION_MANIFEST}"
  sync -- "${directory}"
}

write_transaction_state() {
  local transaction_dir="$1"
  local state="$2"
  write_atomic_text "${transaction_dir}/state" 0600 "${state}"
}

discover_transaction_job_ids() {
  local transaction_dir="$1"
  local transaction_id="$2"
  local candidate candidate_file job_id comment queue_output
  local queue_succeeded=0
  local -a known_ids=()
  local -A seen=()

  DISCOVERED_JOB_IDS=()
  candidate="${ACTIVE_JOB_ID:-}"
  if [[ "${candidate}" =~ ^[0-9]+$ && -z "${seen[${candidate}]:-}" ]]; then
    known_ids+=("${candidate}")
    seen["${candidate}"]=1
  fi
  for candidate_file in "${transaction_dir}/job_id" "${transaction_dir}/sbatch.stdout"; do
    [[ -f "${candidate_file}" ]] || continue
    while IFS= read -r candidate; do
      if [[ "${candidate}" =~ ^[0-9]+$ && -z "${seen[${candidate}]:-}" ]]; then
        known_ids+=("${candidate}")
        seen["${candidate}"]=1
      fi
    done < "${candidate_file}"
  done

  if queue_output="$(
    squeue --noheader --user="${SUBMIT_USER}" --format='%i|%.256k' 2>/dev/null
  )"; then
    queue_succeeded=1
    while IFS='|' read -r job_id comment; do
      [[ "${job_id}" =~ ^[0-9]+$ ]] || continue
      comment="${comment#"${comment%%[![:space:]]*}"}"
      comment="${comment%"${comment##*[![:space:]]}"}"
      if [[ "${comment}" == "${transaction_id}" && -z "${seen[${job_id}]:-}" ]]; then
        known_ids+=("${job_id}")
        seen["${job_id}"]=1
      fi
    done <<< "${queue_output}"
  fi

  DISCOVERED_JOB_IDS=("${known_ids[@]}")
  (( queue_succeeded == 1 || ${#known_ids[@]} > 0 ))
}

cancel_transaction() {
  local transaction_dir="$1"
  local transaction_id="$2"
  local job_id
  local -a recovered_job_ids=()

  if ! discover_transaction_job_ids "${transaction_dir}" "${transaction_id}"; then
    write_transaction_state "${transaction_dir}" cleanup_pending || true
    echo "Could not recover held job for transaction ${transaction_id}; recovery remains pending" >&2
    return 1
  fi
  recovered_job_ids=("${DISCOVERED_JOB_IDS[@]}")
  for job_id in "${recovered_job_ids[@]}"; do
    if ! scancel "${job_id}"; then
      write_transaction_state "${transaction_dir}" cleanup_pending || true
      echo "Could not cancel job ${job_id} for transaction ${transaction_id}" >&2
      return 1
    fi
  done
  write_transaction_state "${transaction_dir}" canceled
  if (( ${#recovered_job_ids[@]} > 0 )); then
    echo "Canceled held transaction ${transaction_id}: jobs ${recovered_job_ids[*]}" >&2
  else
    echo "Closed transaction ${transaction_id}: no submitted job was found" >&2
  fi
}

cleanup_on_exit() {
  local status=$?
  trap - EXIT HUP INT TERM
  set +e
  if [[ -n "${ACTIVE_TRANSACTION_DIR:-}" && "${ACTIVE_TRANSACTION_RELEASED:-0}" != 1 ]]; then
    if ! cancel_transaction "${ACTIVE_TRANSACTION_DIR}" "${ACTIVE_TRANSACTION_ID}"; then
      status=5
    elif (( status == 0 )); then
      status=5
    fi
  fi
  exit "${status}"
}

recover_incomplete_transactions() {
  local state transaction_dir transaction_id
  local unresolved=0
  shopt -s nullglob
  for transaction_dir in "${TRANSACTION_ROOT}"/*; do
    if [[ ! -d "${transaction_dir}" || -L "${transaction_dir}" ]]; then
      echo "Invalid transaction entry: ${transaction_dir}" >&2
      unresolved=1
      continue
    fi
    state="$(tr -d '\r\n' < "${transaction_dir}/state" 2>/dev/null || true)"
    case "${state}" in
      canceled|released) continue ;;
    esac
    transaction_id="$(tr -d '\r\n' < "${transaction_dir}/transaction_id" 2>/dev/null || true)"
    if [[ ! "${transaction_id}" =~ ^pi05-[0-9a-f]{40}$ ]]; then
      echo "Invalid incomplete transaction metadata: ${transaction_dir}" >&2
      unresolved=1
      continue
    fi
    ACTIVE_JOB_ID=""
    if ! cancel_transaction "${transaction_dir}" "${transaction_id}"; then
      unresolved=1
    fi
  done
  shopt -u nullglob
  (( unresolved == 0 )) || return 1
}

capture_submission_provenance() {
  local provenance_dir="$1"
  local job_id="$2"
  local submission_argv="$3"
  local batch_script_path submission_argv_path
  local batch_temporary argv_temporary

  batch_script_path="${provenance_dir}/batch_script.sbatch"
  submission_argv_path="${provenance_dir}/submission_argv.sh"
  batch_temporary="${provenance_dir}/.batch_script.sbatch.tmp.${BASHPID}.${RANDOM}"
  argv_temporary="${provenance_dir}/.submission_argv.sh.tmp.${BASHPID}.${RANDOM}"
  mkdir -p "$(dirname "${provenance_dir}")" || return
  mkdir -m 0755 "${provenance_dir}" || return
  scontrol write batch_script "${job_id}" "${batch_temporary}" || return
  [[ -f "${batch_temporary}" && ! -L "${batch_temporary}" && -s "${batch_temporary}" ]] || return
  printf '%s\n' "${submission_argv}" > "${argv_temporary}" || return
  chmod 0444 "${batch_temporary}" "${argv_temporary}" || return
  sync -- "${batch_temporary}" || return
  sync -- "${argv_temporary}" || return
  [[ ! -e "${batch_script_path}" && ! -e "${submission_argv_path}" ]] || return
  mv -- "${batch_temporary}" "${batch_script_path}" || return
  mv -- "${argv_temporary}" "${submission_argv_path}" || return
  sync -- "${provenance_dir}" || return
  batch_script_sha256="$(sha256sum "${batch_script_path}" | awk '{print $1}')" || return
  submission_argv_sha256="$(sha256sum "${submission_argv_path}" | awk '{print $1}')" || return
}

MODE="${1:-}"
case "${MODE}" in
  canary|foodbussing50|full) ;;
  *) echo "Usage: $0 {canary|foodbussing50|full}" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${SCRIPT_DIR}/l40s_pi05_eval_job.sbatch}"
POLARIS_DIR="${POLARIS_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUN_NAMESPACE="${RUN_NAMESPACE:-pi05-polaris-$(date -u +%Y%m%dT%H%M%SZ)}"
NFS_ROOT="${NFS_ROOT:-/lustre/fsw/portfolios/nvr/users/lzha}"
SBATCH_LOG_ROOT="${SBATCH_LOG_ROOT:-${NFS_ROOT}/slurm_logs/polaris-pi05/${RUN_NAMESPACE}}"
SUBMISSION_MANIFEST="${SUBMISSION_MANIFEST:-${NFS_ROOT}/results/polaris-pi05/${RUN_NAMESPACE}/${MODE}_jobs.tsv}"
TRANSACTION_ROOT="${SUBMISSION_MANIFEST}.transactions"
POLARIS_COMMIT="$(git -C "${POLARIS_DIR}" rev-parse HEAD)"
ALLOW_RESUBMIT="${ALLOW_RESUBMIT:-0}"
ENVIRONMENT_SEED="${ENVIRONMENT_SEED:-0}"
SUBMIT_USER="${USER:-$(id -un)}"
ACTIVE_TRANSACTION_DIR=""
ACTIVE_TRANSACTION_ID=""
ACTIVE_TRANSACTION_RELEASED=0
ACTIVE_JOB_ID=""
DISCOVERED_JOB_IDS=()

trap cleanup_on_exit EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

for required_command in flock sbatch scancel scontrol squeue sha256sum sync; do
  command -v "${required_command}" >/dev/null || die "Missing required command: ${required_command}"
done
[[ "${RUN_NAMESPACE}" =~ ^[A-Za-z0-9._-]+$ ]] \
  || die "RUN_NAMESPACE may contain only letters, digits, dot, underscore, and dash"
[[ "${ALLOW_RESUBMIT}" == 0 || "${ALLOW_RESUBMIT}" == 1 ]] \
  || die "ALLOW_RESUBMIT must be 0 or 1"
[[ -z "$(git -C "${POLARIS_DIR}" status --porcelain=v1 --untracked-files=all)" ]] \
  || die "PolaRiS source has tracked or untracked modifications"

if [[ "${MODE}" == canary ]]; then
  tasks=(DROID-FoodBussing)
  rollouts="${ROLLOUTS:-1}"
  time_limit="${SBATCH_TIME:-01:00:00}"
  job_prefix="pi05-canary"
elif [[ "${MODE}" == foodbussing50 ]]; then
  tasks=(DROID-FoodBussing)
  rollouts="${ROLLOUTS:-50}"
  time_limit="${SBATCH_TIME:-03:50:00}"
  job_prefix="pi05-food50-seed${ENVIRONMENT_SEED}"
else
  tasks=(
    DROID-BlockStackKitchen
    DROID-FoodBussing
    DROID-PanClean
    DROID-MoveLatteCup
    DROID-OrganizeTools
    DROID-TapeIntoContainer
  )
  rollouts="${ROLLOUTS:-50}"
  time_limit="${SBATCH_TIME:-03:50:00}"
  job_prefix="pi05-full50"
fi

[[ "${rollouts}" =~ ^[1-9][0-9]*$ ]] || die "ROLLOUTS must be positive"
[[ "${ENVIRONMENT_SEED}" =~ ^(0|[1-9][0-9]*)$ ]] \
  || die "ENVIRONMENT_SEED must be a non-negative integer"
(( ENVIRONMENT_SEED <= 4294967295 )) \
  || die "ENVIRONMENT_SEED must be at most 4294967295"
[[ -f "${SBATCH_SCRIPT}" && ! -L "${SBATCH_SCRIPT}" ]] \
  || die "Missing or symlinked sbatch script: ${SBATCH_SCRIPT}"
mkdir -p "${SBATCH_LOG_ROOT}" "$(dirname "${SUBMISSION_MANIFEST}")"
[[ ! -L "${TRANSACTION_ROOT}" ]] || die "Transaction root must not be a symlink"
mkdir -p "${TRANSACTION_ROOT}"
[[ -d "${TRANSACTION_ROOT}" && ! -L "${TRANSACTION_ROOT}" ]] \
  || die "Transaction root is not a real directory"
[[ ! -L "${SUBMISSION_MANIFEST}" ]] || die "Submission manifest must not be a symlink"
exec 9>"${SUBMISSION_MANIFEST}.lock"
flock -n 9 || { echo "Another submitter holds ${SUBMISSION_MANIFEST}.lock" >&2; exit 4; }
recover_incomplete_transactions \
  || { echo "Unresolved prior submission transaction; refusing new work" >&2; exit 5; }

expected_header=$'job_id\tmode\ttask\trollouts\tenvironment_seed\trun_namespace\tsubmitted_at\tbatch_script_sha256\tsubmission_argv_sha256\tprovenance_dir'
if [[ ! -e "${SUBMISSION_MANIFEST}" ]]; then
  write_atomic_text "${SUBMISSION_MANIFEST}" 0644 "${expected_header}"
else
  [[ -f "${SUBMISSION_MANIFEST}" ]] || die "Submission manifest is not a regular file"
  [[ "$(head -n 1 "${SUBMISSION_MANIFEST}")" == "${expected_header}" ]] \
    || die "Submission manifest header mismatch"
fi

job_ids=()
for task in "${tasks[@]}"; do
  existing_job_id="$(
    awk -F '\t' -v mode="${MODE}" -v task="${task}" \
      '$2 == mode && $3 == task {job_id = $1} END {print job_id}' \
      "${SUBMISSION_MANIFEST}"
  )"
  if [[ -n "${existing_job_id}" && "${ALLOW_RESUBMIT}" != 1 ]]; then
    echo "Existing ${MODE} attempt for ${task}: job ${existing_job_id}; set ALLOW_RESUBMIT=1 for an explicit retry"
    job_ids+=("${existing_job_id}")
    continue
  fi

  short_task="${task#DROID-}"
  job_name="${job_prefix}_${short_task}"
  transaction_seed="${RUN_NAMESPACE}|${MODE}|${task}|${BASHPID}|$(date +%s%N)|${RANDOM}|${RANDOM}"
  transaction_digest="$(printf '%s' "${transaction_seed}" | sha256sum | awk '{print $1}')"
  ACTIVE_TRANSACTION_ID="pi05-${transaction_digest:0:40}"
  ACTIVE_TRANSACTION_DIR="${TRANSACTION_ROOT}/${ACTIVE_TRANSACTION_ID}"
  ACTIVE_TRANSACTION_RELEASED=0
  ACTIVE_JOB_ID=""
  mkdir -m 0700 "${ACTIVE_TRANSACTION_DIR}"
  write_atomic_text "${ACTIVE_TRANSACTION_DIR}/transaction_id" 0444 "${ACTIVE_TRANSACTION_ID}"
  printf -v transaction_metadata \
    'mode=%s\ntask=%s\nrun_namespace=%s\npolaris_commit=%s\nprepared_at=%s\n' \
    "${MODE}" "${task}" "${RUN_NAMESPACE}" "${POLARIS_COMMIT}" "$(date -Iseconds)"
  write_atomic_text "${ACTIVE_TRANSACTION_DIR}/metadata" 0444 "${transaction_metadata%$'\n'}"
  write_transaction_state "${ACTIVE_TRANSACTION_DIR}" prepared

  export_vars="PATH=${PATH},HOME=${HOME},POLARIS_DIR=${POLARIS_DIR},EXPECTED_POLARIS_COMMIT=${POLARIS_COMMIT},POLARIS_ENVIRONMENT=${task},ROLLOUTS=${rollouts},ENVIRONMENT_SEED=${ENVIRONMENT_SEED},RUN_NAMESPACE=${RUN_NAMESPACE}"
  sbatch_argv=(sbatch --parsable --hold \
    --comment="${ACTIVE_TRANSACTION_ID}" \
    --job-name="${job_name}" \
    --time="${time_limit}" \
    --output="${SBATCH_LOG_ROOT}/%x-%j.out" \
    --export="${export_vars}" \
    "${SBATCH_SCRIPT}")
  printf -v submission_argv '%q ' "${sbatch_argv[@]}"
  set +e
  "${sbatch_argv[@]}" > "${ACTIVE_TRANSACTION_DIR}/sbatch.stdout"
  sbatch_status=$?
  set -e
  chmod 0444 "${ACTIVE_TRANSACTION_DIR}/sbatch.stdout"
  sync -- "${ACTIVE_TRANSACTION_DIR}/sbatch.stdout"
  sync -- "${ACTIVE_TRANSACTION_DIR}"
  mapfile -t sbatch_lines < "${ACTIVE_TRANSACTION_DIR}/sbatch.stdout"
  if (( sbatch_status != 0 || ${#sbatch_lines[@]} != 1 )) \
    || [[ ! "${sbatch_lines[0]:-}" =~ ^[0-9]+$ ]]; then
    echo "sbatch failed or did not return exactly one numeric held job ID" >&2
    exit 3
  fi
  ACTIVE_JOB_ID="${sbatch_lines[0]}"
  write_atomic_text "${ACTIVE_TRANSACTION_DIR}/job_id" 0444 "${ACTIVE_JOB_ID}"
  write_transaction_state "${ACTIVE_TRANSACTION_DIR}" job_captured

  provenance_dir="$(dirname "${SUBMISSION_MANIFEST}")/submission_provenance/job_${ACTIVE_JOB_ID}"
  batch_script_sha256=""
  submission_argv_sha256=""
  if ! capture_submission_provenance \
    "${provenance_dir}" "${ACTIVE_JOB_ID}" "${submission_argv}"; then
    write_transaction_state "${ACTIVE_TRANSACTION_DIR}" provenance_failed || true
    echo "Failed to preserve submission provenance for held job ${ACTIVE_JOB_ID}" >&2
    exit 5
  fi
  write_transaction_state "${ACTIVE_TRANSACTION_DIR}" provenance_durable

  printf -v manifest_row '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
    "${ACTIVE_JOB_ID}" "${MODE}" "${task}" "${rollouts}" "${ENVIRONMENT_SEED}" \
    "${RUN_NAMESPACE}" "$(date -Iseconds)" "${batch_script_sha256}" \
    "${submission_argv_sha256}" "${provenance_dir}"
  append_manifest_row "${manifest_row}"
  write_transaction_state "${ACTIVE_TRANSACTION_DIR}" manifest_durable

  if ! scontrol release "${ACTIVE_JOB_ID}"; then
    write_transaction_state "${ACTIVE_TRANSACTION_DIR}" release_failed || true
    echo "Failed to release held job ${ACTIVE_JOB_ID}" >&2
    echo "Its durable manifest row is retained; retry requires ALLOW_RESUBMIT=1" >&2
    exit 5
  fi
  write_transaction_state "${ACTIVE_TRANSACTION_DIR}" released
  ACTIVE_TRANSACTION_RELEASED=1
  job_ids+=("${ACTIVE_JOB_ID}")
  printf '%s\n' "${manifest_row}"
  ACTIVE_TRANSACTION_DIR=""
  ACTIVE_TRANSACTION_ID=""
  ACTIVE_JOB_ID=""
  ACTIVE_TRANSACTION_RELEASED=0
done

printf 'submitted_job_ids=%s\n' "${job_ids[*]}"
printf 'submission_manifest=%s\n' "${SUBMISSION_MANIFEST}"
