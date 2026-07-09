#!/usr/bin/env bash

# Build a deterministic, read-only source snapshot and external approval record.

set -Eeuo pipefail

die() {
  echo "$*" >&2
  exit 2
}

(( $# == 3 )) || die "Usage: $0 SOURCE_CHECKOUT OUTPUT_ROOT APPROVAL_JSON"
SOURCE_CHECKOUT="$1"
OUTPUT_ROOT="$2"
APPROVAL_JSON="$3"
SCRIPT_DIR="$(readlink -f -- "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"
TRUSTED_HASHER="${SCRIPT_DIR}/l40s_pi05_eval_job.sbatch"
POLARIS_BASE_COMMIT=c5b52a9cebb2c797a84e3df374b6002005d20a4f
POLARIS_BASE_TREE=7fd5e1b0af26577fd323fb1d7f3595b91282e73f
OPENPI_COMMIT=bd70b8f4011e85b3f3b0f039f12113f78718e7bf

for command in chmod find git mkdir mktemp mv readlink sha256sum tar; do
  command -v "${command}" >/dev/null || die "Missing required command: ${command}"
done
[[ "${SOURCE_CHECKOUT}" == /* && -d "${SOURCE_CHECKOUT}" && ! -L "${SOURCE_CHECKOUT}" ]] \
  || die "SOURCE_CHECKOUT must be an absolute non-symlinked directory"
[[ "${OUTPUT_ROOT}" == /* ]] || die "OUTPUT_ROOT must be absolute"
[[ "${APPROVAL_JSON}" == /* && ! -e "${APPROVAL_JSON}" && ! -L "${APPROVAL_JSON}" ]] \
  || die "APPROVAL_JSON must be a new absolute path"
[[ -f "${TRUSTED_HASHER}" && ! -L "${TRUSTED_HASHER}" ]] \
  || die "Trusted source hasher is missing"
SOURCE_CHECKOUT="$(readlink -f -- "${SOURCE_CHECKOUT}")" \
  || die "Cannot canonicalize SOURCE_CHECKOUT"
approval_name="$(basename "${APPROVAL_JSON}")"
raw_approval_parent="$(dirname "${APPROVAL_JSON}")"
mkdir -p "${OUTPUT_ROOT}" "${raw_approval_parent}"
OUTPUT_ROOT="$(readlink -f -- "${OUTPUT_ROOT}")" \
  || die "Cannot canonicalize OUTPUT_ROOT"
approval_parent="$(readlink -f -- "${raw_approval_parent}")" \
  || die "Cannot canonicalize the approval parent"
APPROVAL_JSON="${approval_parent}/${approval_name}"
[[ ! -e "${APPROVAL_JSON}" && ! -L "${APPROVAL_JSON}" ]] \
  || die "Canonical APPROVAL_JSON already exists"
[[ "$(git -C "${SOURCE_CHECKOUT}" rev-parse --show-toplevel)" == "$(readlink -f "${SOURCE_CHECKOUT}")" ]] \
  || die "SOURCE_CHECKOUT must name the exact Git root"
implementation_commit="$(git -C "${SOURCE_CHECKOUT}" rev-parse HEAD)"
[[ "${implementation_commit}" =~ ^[0-9a-f]{40}$ ]] \
  || die "Implementation commit is invalid"
git -C "${SOURCE_CHECKOUT}" merge-base --is-ancestor \
  "${POLARIS_BASE_COMMIT}" "${implementation_commit}" \
  || die "Implementation commit is not descended from the frozen c5 base"
[[ -z "$(git -C "${SOURCE_CHECKOUT}" status --porcelain=v1 --untracked-files=all)" ]] \
  || die "SOURCE_CHECKOUT must be completely clean"
recorded_openpi_commit="$(
  git -C "${SOURCE_CHECKOUT}" ls-tree "${implementation_commit}" third_party/openpi \
    | awk '{print $3}'
)"
[[ "${recorded_openpi_commit}" == "${OPENPI_COMMIT}" ]] \
  || die "Implementation commit does not record the approved OpenPI submodule"
[[ "$(git -C "${SOURCE_CHECKOUT}/third_party/openpi" rev-parse HEAD)" == "${OPENPI_COMMIT}" ]] \
  || die "Materialized OpenPI checkout is not at bd70"
[[ -z "$(git -C "${SOURCE_CHECKOUT}/third_party/openpi" status --porcelain=v1 --untracked-files=all)" ]] \
  || die "Materialized OpenPI checkout must be completely clean"

temporary="$(mktemp -d "${OUTPUT_ROOT}/.pi05-source.partial.XXXXXXXX")"
cleanup() {
  chmod -R u+w "${temporary}" 2>/dev/null || true
  rm -rf -- "${temporary}"
}
trap cleanup EXIT HUP INT TERM

git -C "${SOURCE_CHECKOUT}" archive --format=tar "${implementation_commit}" \
  | tar -xf - -C "${temporary}"
mkdir -p "${temporary}/third_party/openpi"
git -C "${SOURCE_CHECKOUT}/third_party/openpi" archive --format=tar \
  "${OPENPI_COMMIT}" packages/openpi-client/src \
  | tar -xf - -C "${temporary}/third_party/openpi"
for required_path in \
  scripts/eval.py \
  scripts/polaris/eval_pi05_droid_jointpos_polaris.sh \
  scripts/polaris/serve_pi05_droid_jointpos_attested.py \
  src/polaris/pi05_droid_jointpos_consumer_binding.py \
  src/polaris/pi05_droid_jointpos_scheduler.py \
  src/polaris/policy/droid_jointpos_client.py \
  third_party/openpi/packages/openpi-client/src/openpi_client/image_tools.py \
  third_party/openpi/packages/openpi-client/src/openpi_client/websocket_client_policy.py; do
  [[ -f "${temporary}/${required_path}" && ! -L "${temporary}/${required_path}" ]] \
    || die "Source snapshot lacks required path: ${required_path}"
done
if find "${temporary}" -type l -print -quit | grep -q .; then
  die "Source snapshot contains a symlink"
fi
while IFS= read -r -d '' file; do
  if [[ -x "${file}" ]]; then
    chmod 0555 "${file}"
  else
    chmod 0444 "${file}"
  fi
done < <(find "${temporary}" -type f -print0)
while IFS= read -r -d '' directory; do
  chmod 0555 "${directory}"
done < <(find "${temporary}" -depth -type d -print0)

source_tree_sha256="$(
  /usr/bin/bash --noprofile --norc "${TRUSTED_HASHER}" \
    --source-digest "${temporary}"
)" || die "Trusted source snapshot hashing failed"
[[ "${source_tree_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "Trusted source snapshot digest is invalid"
snapshot_path="${OUTPUT_ROOT}/${source_tree_sha256}"
[[ ! -e "${snapshot_path}" && ! -L "${snapshot_path}" ]] \
  || die "Content-addressed source snapshot already exists: ${snapshot_path}"
mv -- "${temporary}" "${snapshot_path}"
temporary="${OUTPUT_ROOT}/.pi05-source.moved.${BASHPID}"
trusted_hasher_sha256="$(sha256sum "${TRUSTED_HASHER}" | awk '{print $1}')"

approval_temporary="$(
  mktemp "$(dirname "${APPROVAL_JSON}")/.pi05-source-approval.partial.XXXXXXXX"
)"
/usr/bin/python3 -I -S - \
  "${approval_temporary}" "${snapshot_path}" "${source_tree_sha256}" \
  "${implementation_commit}" "${POLARIS_BASE_COMMIT}" "${POLARIS_BASE_TREE}" \
  "${OPENPI_COMMIT}" "${trusted_hasher_sha256}" <<'PY'
import json
from pathlib import Path
import sys

(
    output,
    snapshot_path,
    source_tree_sha256,
    implementation_commit,
    polaris_base_commit,
    polaris_base_tree,
    openpi_commit,
    trusted_hasher_sha256,
) = sys.argv[1:]
value = {
    "schema_version": 1,
    "profile": "openpi_pi05_droid_jointpos_source_approval_v1",
    "snapshot_path": snapshot_path,
    "source_tree_sha256": source_tree_sha256,
    "implementation_commit": implementation_commit,
    "polaris_base_commit": polaris_base_commit,
    "polaris_base_tree": polaris_base_tree,
    "openpi_commit": openpi_commit,
    "trusted_hasher_sha256": trusted_hasher_sha256,
}
payload = json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    allow_nan=False,
).encode("ascii") + b"\n"
path = Path(output)
with path.open("wb") as stream:
    stream.write(payload)
    stream.flush()
PY
chmod 0444 "${approval_temporary}"
mv -- "${approval_temporary}" "${APPROVAL_JSON}"
sync -- "${APPROVAL_JSON}" "$(dirname "${APPROVAL_JSON}")" "${snapshot_path}"
trap - EXIT HUP INT TERM
printf 'snapshot_path=%s\nsource_tree_sha256=%s\napproval_json=%s\n' \
  "${snapshot_path}" "${source_tree_sha256}" "${APPROVAL_JSON}"
