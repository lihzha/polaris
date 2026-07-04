# Official pi0.5-DROID canary atomic-port repair — 2026-07-04

- `CODEX_AGENT_ID=pi05-atomic-port-fix-v1`.
- Isolated worktree:
  `/home/lzha/code/PolaRiS-worktrees/pi05-atomic-port-fix-v1-20260704`.
- Branch: `codex/pi05-atomic-port-fix-v1-20260704`.
- Exact base: `c0a8dff2d863d732bad3cfc7edb043f3de289a8c`, whose parent is the reviewed
  finalizer lineage `f5a851660cda2a9b6121b205273bcfdbeefddbd2`.
- Implementation commit: `17616d4` (`Bind pi0.5 canary to server-owned port`).
- Scope was implementation and host validation only. No deployment, package
  mutation, registry update, shared artifact mutation, GPU allocation, Slurm
  submission, simulator, checkpoint load, or model evaluation was performed.

## Trigger and repair

The canary selected `20000 + SLURM_JOB_ID % 20000`, allowed an ambient `PORT`
override, then probed and closed `/dev/tcp` before launching the WebSocket
server. That left a bind TOCTOU inside l401's ephemeral-port range. The server
also published its immutable serving and model-runtime contracts before it
attempted the real bind, so a collision consumed a fresh attempt with
misleading authoritative artifacts.

The repair makes the actual pinned OpenPI WebSocket process request port `0`.
Its listener wrapper enters the upstream `websockets.asyncio.server.serve`
context, proves that exactly one `AF_INET` socket owns a nonzero port on
`0.0.0.0`, publishes the unchanged serving and model-runtime artifacts while
that socket remains held, and publishes the immutable bound-port record last
as the readiness transition. A bind failure invokes no authoritative artifact
callback and publishes no bound-port readiness record. The evaluator also
publishes an explicit immutable `attempt_failed.json` on terminal failure.

The bound-port artifact is closed-schema canonical JSON containing its own
canonical path, requested port `0`, actual port, server PID, launch token,
host, and socket family. It is created with `O_EXCL|O_NOFOLLOW`, fsynced,
changed to mode `0444`, fsynced again, and bound to a single-link regular-file
identity. The reader holds stable parent and file descriptors, rejects path,
mode, link, schema, type, token, PID, and requested-port drift, checks the live
server PID before and after the read, and returns the content SHA-256 plus
device/inode/size/mtime/ctime/mode/link identity.

The shell rejects even an empty ambient `PORT`. It validates the server-owned
record, performs a real WebSocket metadata handshake against the unchanged
persisted serving contract, requires an identical record recheck immediately
after that handshake, passes only the actual port to `DroidJointVelocity`, and
requires another identical live-PID recheck after evaluator execution. The
host finalizer independently rereads and binds the port and handshake
artifacts after server shutdown. Its preflight pins all five handoff sources
before checkpoint download.

## Scientific contract preservation

The official model contract remains unchanged: global DROID statistics,
external/wrist/blank 224-pixel image routing, no wrist rotation, native
joint-velocity action semantics, 15-action flow chunk with eight open-loop
executions, and the existing OpenPI pi0.5 transform/runtime validations.

The following scientific/controller paths are byte-identical to `c0a8dff`:
`scripts/eval.py`, the checkpoint manifest, config and DROID controller/runtime
sources, `pi05_droid_jointvelocity_contract.py`,
`pi05_droid_native_eval_contract.py`, lifecycle code, and
`DroidJointVelocityClient`. The finalizer also compares every server statement
between argument parsing and metadata construction against the job-1098682
attested server source. Only the post-metadata listener/publication
orchestration may differ, and the complete replacement server source is pinned
to SHA-256
`d0eb961a5068400fa0d4772bf6ddb6921d8d6477e978948c0215d82435a3f581`.

Pinned atomic-port runtime source SHA-256 values:

- eval shell: `dde53327f23cd7772c6219ccb74d14eeecdbc38a48805d133b6b401e83572e39`
- server: `d0eb961a5068400fa0d4772bf6ddb6921d8d6477e978948c0215d82435a3f581`
- port validator: `ba9e4510f86bd0aea0b396990a30e31e0ad4c7d64eae8da5f5451427987ee559`
- handshake validator: `736493f6729bb51598367f39c1d14bbc10788df6fcffde9195af4a0bd5d5daa6`
- listener/record module: `2e5c559aaf3dcd5fdb41c21ef919bf7eb23e6fb52f855195fa528d3102af2e3b`

## Validation

- Focused listener/server/finalizer/all-six tests: `71 passed`.
- Broad host suite, all repository tests except the real-Isaac-only
  `test_robust_differential_ik.py`: `307 passed`, `5 subtests passed`, with
  three external dependency warnings.
- Hostile coverage includes symlink, hardlink, wrong mode, duplicate and
  noncanonical JSON, exact-type drift, wrong PID/token/path, nonlive PID,
  same-size directory-entry replacement during read, parent-directory
  replacement during read, preexisting output, fixed-port rejection, and
  synthetic bind failure with no readiness/model callback.
- A real localhost WebSocket test proved that the listener is connectable
  inside the publication callback before the port record exists, then completed
  a WebSocket handshake through the published actual port.
- A standalone compatibility probe used the exact pinned OpenPI server and
  `websockets==15.0.1`: pass; the OS assigned port `46303` in that disposable
  probe and the real metadata handshake succeeded.
- Exact committed-source preflight: five pinned runtime sources passed against
  `HEAD`.
- Ruff 0.15.16 format and lint, in-memory Python compilation, Bash syntax,
  ShellCheck, and `git diff --check`: pass.

## Handoff and residual gate

The preexisting c0a8dff launch package embeds the vulnerable source and must
not be launched. Integration must cherry-pick the implementation and worklog
commits into a clean PolaRiS integration worktree, preserve the five exact
source pins, rerun the host suite, and obtain independent review before
building a new immutable package. A single fresh official FoodBussing canary
on one L40S remains mandatory because this agent was explicitly prohibited
from launching. Inspect its server log, bound-port and handshake artifacts,
trace/runtime/sidecar, metrics, full video decode, completion, explicit source
identities, and terminal scheduler state before any wider evaluation.

There are no owned processes, jobs, monitors, deployment checkouts, packages,
or registry drafts to clean up or transfer.

## Official OpenPI runtime dependency recovery — 2026-07-04

- Recovery agent: `CODEX_AGENT_ID=pi05-runtime-dependency-fix`; same isolated
  branch and worktree. The workstation Ego-LAP checkout remained the primary
  clean `main` worktree apart from its preexisting ignored artifact/cache
  directories. No Ego-LAP integration worktree or shared registry record was
  modified by this recovery.
- Root launched the previously sealed source `84a3407a5912...` as Slurm job
  `1098869`. It ran on L40S node `pool0-00021` from
  `2026-07-04T11:21:40-07:00` through `11:22:48-07:00` and terminated
  `FAILED`, exit `1:0`, elapsed `00:01:08`. The exact run directory is
  `/lustre/fsw/portfolios/nvr/users/lzha/results/polaris-pi05-native/canary/20260704T174000Z-84a3407-atomic-port-canary1`.
- The external controller attestations, one-L40S inventory, official checkpoint
  content/MD5 gate, global-DROID normalization digest/probes, and inference
  environment capture all passed. Server import then failed at
  `third_party/openpi/src/openpi/models_pytorch/gemma_pytorch.py:3` with
  `ModuleNotFoundError: No module named 'pytest'`, before port bind, WebSocket
  handshake, simulator startup, or any policy action. Immutable
  `attempt_failed.json` records `failure_stage=server_bind_and_readiness`,
  `bound_port_artifact_present=false`, and SHA-256
  `4dc8e6764dbf402ecae129007274fa431e978534f323806239101f4a0d913e92`.
  The checkpoint-verification SHA-256 is
  `3a2e6479f488b422c565ed70f3f93c29d8e091d15addbbc7371a29a3c0c557e4`;
  inference-environment SHA-256 is
  `9e5564047f869cd0d2dba14e5ec0c24790d797e53026b247ca39e794268e74a1`;
  server-log SHA-256 is
  `2bafe8e53da6210def34165612fb71bd5b04cc54374bf0e6efe5c24b5b546fe2`.
  A byte-identical local evidence copy is under
  `/home/lzha/code/ego-lap/.codex_artifacts/polaris-pi05/job1098869-failed-server-import`.
- The failed source, bundle, wrapper, and job evidence were not mutated. The
  original source remains detached/clean at `84a3407...`, tree `5927d7d...`,
  with no writable source entries. Its mode-0444 Git bundle remains SHA-256
  `7f1a08959a9de8c9b7da131338aa096986600660e75f5e8d42b14df931d34bf3`;
  its mode-0444 wrapper remains SHA-256
  `ddbda137d45b0bb672877250136b341fa6f89f896e09ca25a7c44ca7e3d616ee`.

### Diagnosis and minimal repair

- Official OpenPI commit `bd70b8f4011e85b3f3b0f039f12113f78718e7bf`
  imports `pytest` at module scope and evaluates the annotation `pytest.Cache`
  in production `gemma_pytorch.py`, while declaring `pytest>=8.3.4` only in
  its `dev` dependency group. The intentional `uv sync --frozen --no-dev`
  therefore removed `pytest==8.3.5` and its missing runtime dependencies
  `iniconfig==2.1.0` and `pluggy==1.6.0`. Existing `packaging==25.0` already
  satisfied pytest's remaining Linux dependency.
- Preserve the official OpenPI source and submodule SHA exactly. Add one
  requirements overlay containing the exact OpenPI-lock wheel versions and
  SHA-256 hashes for pytest and its three Linux requirements. Install it after
  the no-dev sync with `uv pip install --no-config --require-hashes`; the
  `--no-config` gate prevents PolaRiS's unrelated torch override from entering
  this resolver transaction.
- The native inference-environment contract now requires all four overlay
  distributions to be present at their exact locked versions, verifies each
  wheel hash against committed OpenPI `uv.lock`, imports `pytest.Cache`, and
  proves the imported pytest module is inside the exact checkout-local venv.
  The public submitter runs this check before `sbatch`; the GPU environment
  capture reruns it before the server import. The finalizer binds the overlay
  requirements as a critical source file.
- No checkpoint, data transform, image order/resolution, normalization,
  sampler, policy I/O, controller, physics, task, asset, or scientific model
  source changed. The atomic-port server and eval shell remain byte-identical
  to the already reviewed `17616d4` implementation.

### Host validation before deployment

- Exact overlay install into an empty CPython 3.11.15 venv with
  `uv pip install --no-config --require-hashes`: pass; exact versions
  `iniconfig 2.1.0`, `packaging 25.0`, `pluggy 1.6.0`, and `pytest 8.3.5`;
  `pytest.Cache` resolves to `_pytest.cacheprovider.Cache`; `uv pip check`
  passes.
- Preflight against an existing exact-commit full OpenPI venv: pass. Direct
  production import of `openpi.models.model` and
  `openpi.models_pytorch.gemma_pytorch`: pass from exact official source.
- New dependency/provenance regressions: `4 passed`.
- Native server/atomic-port/finalizer focused suite: `67 passed`.
- Broad repository host suite scoped to `tests/`, excluding only the real-Isaac
  `test_robust_differential_ik.py`: `309 passed, 1 skipped`.
- Ruff 0.15.16 format/lint, in-memory compilation, Bash syntax, ShellCheck,
  and `git diff --check`: pass. A fresh detached deployment, exact no-dev plus
  overlay environment validation, immutable package manifest, and root-owned
  one-rollout relaunch remain the next gates.

### Sealed root-launch handoff

- Dependency-repair implementation commit
  `014a7d5c0bd74fd1e25af35f49dd471fa4fde22c`, tree
  `57ead78fe6294b31f67147d36bea21f48468d929`, is pushed to
  `lihzha/codex/pi05-atomic-port-fix-v1-20260704`.
- A full Git bundle was used to create a fresh standalone, detached remote
  source at
  `/lustre/fsw/portfolios/nvr/users/lzha/src/PolaRiS-pi05-runtime-dependency-014a7d5-20260704T183503Z-standalone`.
  OpenPI's configured SSH submodule URL initially failed authentication before
  populating content; a command-scoped HTTPS URL rewrite then initialized the
  public submodule at the same exact `bd70b8f...` Git SHA. The first uv attempt
  failed before environment creation because `$HOME/.cache/uv` exceeded quota;
  the retry used dedicated NFS `UV_CACHE_DIR`/`XDG_CACHE_HOME` roots.
- Fresh environment construction performed `uv sync --frozen --no-dev` for
  204 packages, then the exact hash-required overlay. It installed only the
  three missing distributions, producing 207 total compatible packages.
  Canonical installed-inventory SHA-256 is
  `82e8f99ba721abce4311589a26158339fab022a222d9e863a61b5baba50931b4`;
  sorted `uv pip freeze` SHA-256 is
  `f424a7a7b5d851259244360e6064b692e1eb10abf10335d4da2553b32189029f`.
  `uv sync --frozen --no-dev --inexact --dry-run` reports no changes.
- The first direct model import on the login host inherited 64 BLAS threads and
  made slow progress for five minutes; it was terminated without source or
  artifact output. Repeating with OpenBLAS/OMP/MKL threads pinned to one passed
  in under a minute and resolved both production modules from the exact
  OpenPI source. Fresh-venv dependency regressions passed `4/4`, and the full
  native controller plus atomic-port preflight passed against jobs
  `1098174/1098723`.
- The complete source plus checkout-local venv is read-only: 45,852 regular
  files, zero writable regular files or directories, detached/clean exact Git
  identities, and a successful post-seal package preflight. A global `sync`
  used during sealing blocked in Lustre `super_lock`; independent path checks
  established the zero-writable and Git identities, and only path-scoped fsync
  was used for the final package manifest.
- Root-owned immutable stage:
  `/lustre/fsw/portfolios/nvr/users/lzha/staging/pi05-runtime-dependency-014a7d5-20260704T183503Z`.
  Bundle SHA-256:
  `ce55ec8e6b78b2b6a43a7b5bc276c9c0a8227522c9c0398ef2a7a59f8e638648`;
  launch-wrapper SHA-256:
  `62ccc47a86427c9e66f780c8fbf9490186c629a4ef90a7c4b5f5cd0d7eb07bd5`;
  package-manifest SHA-256:
  `70cc66f768a4225195cc52bdbeada2bf25c8b6ca8906bbc7a1f078f5192a3ab3`.
  All three are mode `0444`, one link, re-read after sealing. The wrapper's
  fresh target
  `.../20260704T183503Z-014a7d5-runtime-overlay-canary2` did not exist at
  handoff. This recovery agent did not execute the wrapper, submit Slurm,
  mutate the registry, or claim a rollout result.
