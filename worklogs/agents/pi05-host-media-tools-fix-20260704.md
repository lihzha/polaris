# pi05-host-media-tools-fix-20260704

- Agent: `pi05-host-media-tools-fix-20260704`
- Base: `014a7d5c0bd74fd1e25af35f49dd471fa4fde22c`
- Implementation commit: `04814580d6a2fe1d6ec1813393aa28e4c6493fa5`
- Goal: repair job 1098880's host-only `ffprobe`/`ffmpeg` dependency without
  changing model, controller, checkpoint, image, normalization, sampler, or
  rollout behavior.

## Change

- Pinned the upstream FFmpeg 7.0.2 static amd64 archive and exact executable
  identities in a committed closed manifest.
- Added fail-closed submit, compute-node preflight, and post-`srun` finalizer
  validation for canonical absolute paths, manifest/tool SHA-256 values,
  executable mode, static ELF identity, version output, and every package-root
  directory component.
- Added explicit finalizer arguments and run/submission/commands/completion
  provenance. The finalizer records and compares pre/post path, inode, size,
  mode, hash, version, ELF, and package-root identities around all media
  subprocesses.
- Preserved the separate reviewed host-finalizer lifecycle and all existing
  video contract checks.

## Package

- Root:
  `/lustre/fs11/portfolios/nvr/projects/nvr_lpr_rvp/users/lzha/cache/polaris/host-media-tools/ffmpeg-7.0.2-static-amd64-abda8d77ce830914`
- Archive SHA-256:
  `abda8d77ce8309141f83ab8edf0596834087c52467f6badf376a6a2a4c87cf67`
- `ffprobe` SHA-256:
  `4f231a1960d83e403d08f7971e271707bec278a9ae18e21b8b5b03186668450d`
- `ffmpeg` SHA-256:
  `e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99`
- Manifest SHA-256:
  `09d95a1f28e9e9af1e172439806ca9c2d6b19dd661f9f5f4ee7f51185cb99be5`

The package was atomically published with mode `0555` on its directory and
executables and mode `0444` on its manifest/license/readme. `namei` confirmed
that the canonical path has no symlink component. Both tools executed on the
l401 host and fully probed/decoded job 1098880's preserved raw video.

## Validation

- Focused finalizer tests: `47 passed`.
- Broad relevant CPU suite: `240 passed, 1 deselected`; the deselected
  real-OpenPI import test requires the exact OpenPI runtime and is scheduled for
  the prepared remote standalone package.
- Ruff format/lint, `py_compile`, `bash -n`, ShellCheck, and `git diff --check`
  passed.
- No GPU evaluation, Slurm job, registry write, shared worklog update, or
  canonical-checkout mutation was performed. Job 1098880 artifacts remain
  unchanged.

## Next gate

Push the agent branch, build a fresh detached standalone l401 checkout at the
final branch commit, recreate its exact OpenPI venv/runtime overlay, run the
real-OpenPI and host-media preflights plus a CPU-only media smoke, and prepare a
fresh absent run path and launch wrapper without submitting it.
