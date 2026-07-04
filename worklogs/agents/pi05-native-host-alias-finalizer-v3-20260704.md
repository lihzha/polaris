# pi0.5 native host-alias finalizer repair v3 — 2026-07-04

- Agent: `pi05_official_model_canary`.
- Base: `f9c82d75cc3dd880c53ae9a3d196f7c355527f10`.
- Trigger: official model canary job `1098704`, produced under the then-current
  `f9c82d7` runtime schema, produced a complete typed-failure transaction but
  the host finalizer rejected the valid incident descriptor after resolving
  container path `/lustre/fsw/...` to host alias `/lustre/fs11/...`.
- Scope: artifact identity/path comparison; correction of stale user-visible
  all-six provenance from job `1098349` to the already-bound recovery smoke job
  `1098682`; and an explicit raw DROID gripper-observation boundary audit. No
  checkpoint, normalization, image order/resolution, action, sampler,
  controller, evaluator physics, or model-input transform changes.
- Required behavior: bind resolved path target plus exact size/SHA/mode/nlink while retaining the recorded lexical path so enclosing immutable contracts rebuild byte-for-byte. Reject a different resolved target and any content/identity drift.
- Rerun gate: host tests, static checks, and read-only selective validation of
  job `1098704` close-ready/trace/video evidence only as far as its producing
  `f9c82d7` runtime schema remains scientifically applicable. This is not a
  full exact-current runtime replay. Commit/push and independent review remain
  mandatory before a fresh GPU attempt.

## Local validation

- Focused contract/finalizer tests: `33 passed`.
- Broader pi0.5/native tests with the exact pinned OpenPI submodule first on
  `PYTHONPATH`: `127 passed`; only external dependency deprecation warnings.
- Ruff format/check, `bash -n`, ShellCheck, and `git diff --check`: pass.
- Alias regressions bind the resolved target and exact size/SHA-256/mode/nlink,
  preserve the recorded lexical path, and reject wrong targets or identity
  drift.
- Gate-provenance regressions require job `1098682`, reject stale job
  `1098349`, and ensure the base-controller descendant authority names the
  current all-six job.

## Masked raw-gripper audit defect

- Selective validation of preserved job `1098704` under its producing
  `f9c82d7` schema reached the trace audit and exposed a second masked host
  defect: PhysX produced a minimum raw normalized open-gripper value of
  `-1.701161989053901e-9`, while the audit required an exact mathematical
  `[0, 1]` bound. This observation is not evidence of a full exact-current
  runtime replay.
- Pinned OpenPI `docs/norm_stats.md` defines open/closed gripper state as
  `[0, 1]`, but pinned official DROID commit
  `33ae6a67274f36d2e29525b86f23a56616ef43a7` computes observation state as
  `1 - width / max_width` without clipping. Pinned OpenPI `DroidInputs` also
  forwards it unchanged before checkpoint quantile normalization. Only DROID
  gripper commands are clipped.
- The repair therefore preserves the raw float32 value and adds a named bound
  tolerance of exactly eight float32 epsilons (`2^-20`,
  `9.5367431640625e-7`). The client rejects anything outside that envelope
  before the server; serving, live runtime, model-eval, and trace validators all
  bind the same contract.
- The all-six source gate permits only this exact additive, non-transforming
  guard and still compares the remaining request/image/action semantics to the
  integrated official-model base AST.
- Updated broader host suite: `163 passed`; Ruff format/check and
  `git diff --check` pass. A fresh no-model all-six L40S gate and rebind are
  mandatory because the contract, runtime, gate finalizer, and client are
  all-six-critical sources.

## Additive stable parent-alias binding closure

- Agent branch:
  `codex/pi05-native-bound-artifact-toctou-v4-20260704` in a fresh worktree
  created exactly from `3e2d773c34c3dbdf6f377dfe446aa4fae39a2a26`.
  The frozen `3e2d773` worktree, canonical checkout, and registries were not
  edited.
- `validate_bound_artifact` now resolves the lexical parent alias once, opens
  that resolved directory with `O_DIRECTORY|O_NOFOLLOW`, opens the basename
  relative to its stable dirfd with `O_NOFOLLOW`, and retains both the recorded
  and expected-target handles across the read. Parent fd, directory entry, and
  file fd identities are checked before and after; the lexical parent and
  expected target are rebound after the read. JSON canonicality and ordinary
  file hashes are computed from the already-open descriptor. Only the original
  lexical path is returned for enclosing canonical bytes.
- Artifact identity fields now require exact path/string, size/integer,
  lowercase SHA-256/string, octal-mode/string, link-count/integer, and
  JSON-selector/boolean types. Final-component symlinks, multiple links,
  non-0444 modes, parent/file replacement, different expected targets, and
  all nonpath identity drift remain fail-closed.
- Deterministic tests retarget a parent alias from one immutable inode to a
  byte-identical different inode both between resolved-parent selection and
  `openat`, and during `os.read`; canonical JSON and ordinary files are rejected
  in both windows. Separate trace and video tests reproduce the fsw/fs11 alias,
  accept the exact target, and reject a byte-identical wrong target or wrong
  digest.
- Loading the exact `3e2d773` validator in isolation confirmed the formal
  NO-GO: it accepted both the JSON and ordinary-file byte-identical retargets
  when the alias changed after the expected-path resolve and before its reopen.
- The host finalizer no longer compares trace/video descriptor dictionaries
  byte-for-byte. It validates each sidecar lexical descriptor against the exact
  sealed target through the stable alias-aware contract, then compares
  size/SHA-256/mode/nlink exactly.

### Preserved job 1098704 evidence scope

Job `1098704` was produced by `f9c82d7` and is not a full exact-current runtime
replay. Read-only validation on `l401` was deliberately split by scientific
scope:

- the producing `f9c82d7` runtime schema plus the alias-only `34da75c` host fix
  validated runtime SHA-256
  `91ed3cb3f357648ee8259b9f2b8050e88c075f4a5b512997d7d311a241a09cc2`,
  environment-runtime SHA-256
  `637b8470e167ab6051a7017358574fc7303d7f8dfa3c8541b922db84e65f051c`,
  close-ready evidence, and the typed all-joint velocity-limit terminal;
- the current `3e2d773` trace audit selectively validated 199 records, episode
  length 93, trace SHA-256
  `8d9d893c002953bd10dd6375520196ca5a3f22e2ec0eccb6088b4d662fba49a2`,
  and raw normalized gripper range
  `[-1.701161989053901e-9, 8.525632438249886e-6]`;
- the preserved video remains exact single-link mode-0444 sidecar content:
  94,956 bytes, SHA-256
  `9838a008d77d48dcec6d49a2fed5bcec317ee398a3136406f04a1c2f038b3d6b`,
  with top-level boxes `ftyp,free,mdat,moov`. The login host had no `ffprobe`,
  so no new full-decode claim is made;
- applying the current `3e2d773` runtime validator directly to the old artifact
  rejects it with `Joint-velocity runtime contract schema mismatch`. This is
  expected and explicitly prevents promotion of the selective checks into a
  false exact-current replay claim.

### Host validation

- Focused contract/finalizer suite: `48 passed`.
- Broad pi0.5/native host suite with both pinned OpenPI source roots first on
  `PYTHONPATH`: `157 passed`, with three external dependency deprecation
  warnings.
- Ruff 0.15.16 lint and format, Python byte compilation, Bash syntax,
  ShellCheck, and `git diff --check`: passed.

No GPU, Slurm allocation, simulator, evaluation, model server, registry
publication, or shared-document write was launched. The preserved artifact
inspection used only login-host read operations.
