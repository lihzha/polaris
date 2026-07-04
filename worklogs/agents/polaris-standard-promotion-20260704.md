# PolaRiS V5 Repaired-Smoke Standard Promotion

Agent: `polaris-standard-promotion-20260704`

Branch: `codex/polaris-standard-promotion-20260704`

Base: `9ab844b3fcaac6d29b51bc9fb2c2758c125201f3` (tree
`2063a7d091ee9d1c6e0646a60ee501a8abd395e8`)

## Goal and scope

Promote the sealed measured-velocity recovery v5 runtime from its repaired
six-task smoke stage to the canonical PolaRiS 50-rollout-per-task standard
stage. This branch is evidence only. It does not modify evaluator,
environment, policy-client, controller, or asset behavior, and it launches no
job or registry mutation.

## Lineage and evidence

The new closed gate is
`src/polaris/eef_velocity_recovery_standard_promotion.py`, manifest SHA-256
`7fd7a390da9dbe61531bdc8de75f83867a2011a6685fa2cfe761b1d965aba458`.
It preserves and validates the predecessor evidence at promotion commit
`0142e8518769d386c0a8227778767800b30c7e83`, manifest SHA-256
`9576a178253741571a50cd23fe8a16b75b9a386ced5bc43ee416348fa52454f7`,
and source SHA-256
`f98f6d3ae6eb06f0127e3ec686fa70e3bb524ea892582b0ee3461b0dd6d84df4`.
It also pins the sealed `9ab844b...` runtime attestation from job `1098834`,
SHA-256
`efded6682bce983a4d773b038990f9e9fd5968cd05efe42b204063c6b4c7b0c5`.

The repaired reasoning suite is rooted at
`/lustre/fsw/portfolios/nvr/users/lzha/results/polaris_eval/reasoning-full-43075-main-5ad7da3-polaris-9ab844b-smoke6-rerun-20260704`.
Watcher `1098870` and workers `1098871`, `1098872`, `1098878`, `1098879`,
`1098882`, and `1098884` completed `0:0`. Its suite summary SHA-256 is
`80a0c2e5f7439af989e8d57e6ddf9decdac5cd28ab611fc20ce4ed1e05a3790b`
and combined candidates SHA-256 is
`0b78776aa43f417199500a1b2949285a7de86d1380c2e0f7cbbc8ae1154040ec`.

The repaired official LAP-3B suite is rooted at
`/lustre/fsw/portfolios/nvr/users/lzha/results/polaris_eval/official-lap3b-601db9c1-main-5ad7da3-polaris-9ab844b-smoke6-rerun-20260704`.
Watcher `1098873` and workers `1098874`, `1098875`, `1098876`, `1098877`,
`1098881`, and `1098883` completed `0:0`. Its suite summary SHA-256 is
`8100e9e6f9e216208f1628d49d13dcaebd3f6c2ab17535c0828c27c1a761e058`
and combined candidates SHA-256 is
`10eccd6c015aba14d608dd797cfa35e7a6434aa1215b4492c77bd6741e8a7f0c`.

The manifest binds 138 authoritative artifacts: three suite-level artifacts
per suite and 11 task-level artifacts per task across 12 tasks. These include
every completion/authoritative marker, candidate, audit, runtime contract,
episode sidecar, finalized trace, raw video, summary sidecar, and summary
video.

## Fresh artifact and behavior audit

On l401, the exact Ego-LAP `5ad7da3...` `task-complete` validator was replayed
against all 12 authoritative attempt directories. All passed. Each rollout had
450 policy steps, 3,600 physics applies, and zero numerical failure,
controller abort, DLS fallback, dropped diagnostic, or post-clamp violation.

All 24 locally fetched raw/summary videos were independently passed through
`ffprobe` and a full `ffmpeg -f null -` decode. Every raw video is H.264
yuv420p, 448x224, 15 fps, 450 frames, and 30 seconds; every summary is H.264
yuv420p, 960x608, 15 fps, 450 frames, and 30 seconds. The hashes matched the
remote pinned artifacts. Previously generated nine-frame contact sheets for
all 12 rollouts were inspected at original resolution: correct tasks and
views, stable plausible motion, no blank/corrupt views, and no physics
explosion. There were no raw positives requiring adjudication. Task-specific
failure observations are closed in the manifest.

Both suites were raw/task-valid `0/6`; reasoning mean progress was `3/28` and
official mean progress was `5/36`. This is promotion/wiring evidence and not a
standard success-rate estimate.

## Strict authorization design

`validate_eef_velocity_recovery_v5_standard_promotion_artifacts` validates the
closed manifest, resolves each canonical NFS alias to its pinned physical root,
and walks every artifact with descriptor-relative `O_NOFOLLOW` opens. It
rejects missing files, symlinks, hard-link counts other than one, non-regular
files, path/schema/duplicate drift, SHA-256 mismatch, and mutation during
hashing, then returns canonical inventory and verification digests. The standard scale path invokes this
fresh verifier; equality to an in-code boolean is insufficient.

The authorization is exactly six canonical DROID tasks, 50 rollouts per task,
450 policy steps at 15 Hz, one environment, absolute EEF pose control, and
physical `panda_link8` relative to `panda_link0`.

## Local validation

Focused predecessor plus standard-promotion tests:

```bash
PYTHONPATH=$PWD/src /home/lzha/code/ego-lap/.venv/bin/python -m pytest -q \
  tests/test_eef_velocity_recovery_promotion.py \
  tests/test_eef_velocity_recovery_standard_promotion.py
```

passed `85`, skipped one NFS-only success test on the workstation.

The complete host-safe suite:

```bash
PYTHONPATH=$PWD/src /home/lzha/code/ego-lap/.venv/bin/python -m pytest -q \
  tests --ignore=tests/test_robust_differential_ik.py
```

passed `1033` tests plus `30` subtests, with one expected NFS-only skip. The
excluded module imports unavailable host `isaaclab` at collection time. Ruff
check/format, byte compilation, and `git diff --check` passed.

Negative coverage explicitly exercises missing, tampered, symlinked,
hard-linked, non-regular, path-drifted, task-omitted, suite-omitted,
artifact-drifted, and authorization-prerequisite-drifted evidence.

## L401 fresh standard-authorization verification

Implementation commit `92fbca47a56b50848a75fbd74e31d2e9d01eda2d` (tree
`d162a3ce223214ca953218e5ffb32a05c57bea17`) was sealed in a complete-history
Git bundle, SHA-256
`d490d900c42d8723c8f3c850767df4579679244aec698855795f4f44f8899fda`,
and checked out detached and clean at
`/lustre/fsw/portfolios/nvr/users/lzha/src/PolaRiS-standard-promotion-preflight-92fbca4-20260704`.

The public
`validate_and_authorize_eef_velocity_recovery_v5_standard` entrypoint was then
run on the l401 login host with no root override. It securely reopened and
rehashed all `138/138` pinned artifacts: `69` per suite, `12` tasks, two exact
canonical roots. It returned:

- artifact inventory SHA-256
  `b2649ef53c4a64e27ccb9e0e84c9cc594ab1c5f2dd8bb16b60ba4e46dda94226`;
- verification SHA-256
  `467a4933a54f38a395233c569ed4aa18709fffe521917df49cdea82c9a7161cf`;
- promotion evidence SHA-256
  `7fd7a390da9dbe61531bdc8de75f83867a2011a6685fa2cfe761b1d965aba458`;
- `standard_authorized: true` and `root_overrides_used: false`.

No Slurm job, local long-running process, registry write, or evaluation launch
was created by this implementation. The final handoff package is sealed after
this append-only evidence update; its exact final commit/tree/bundle/manifest
identities are reported to the orchestrator.
