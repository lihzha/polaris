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
