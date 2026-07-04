# Reasoning full-trace gripper/release ablation — 2026-07-03

## Objective

Diagnose the controller-candidate numerical abort from reasoning checkpoint
step 43075 at policy step 293 / physics substep 2 without another model-server
variable.  The source is the finalized job-1098523 trace produced by PolaRiS
commit `0611d384f5f26ef9bd8ff114be273e875c3fe719`.

## Closed replay contract

- Source trace SHA-256:
  `db2436639cd2ddf2c9252346c837b9a081a6563048d8005d1d8b7cf2957aea80`.
- All 294 absolute 8-wide PolaRiS actions are encoded as little-endian
  float32.  Uncompressed action SHA-256:
  `0e781cd1df2d00f3496c1feb2bf079e9194ad664710ac988cc9f7e8bcde11bce`.
- The fixture validates the 37-query serving contract, including flow mode,
  10 Euler steps, 8-action execution horizon, egocentric language frame,
  robot-base numeric frame, `panda_link8`, global train-matched q99
  normalization, and the train-matched R6 column layout.
- Reset, FoodBussing IC0, asset hashes, camera preprocessing, controller
  candidate, mimic-compliance overlay, and 120 Hz / decimation-8 cadence are
  inherited unchanged from the production eval profile.

## Pre-registered variants

1. `baseline`: exact current stack; must reproduce the step-293/substep-2
   joint-7 abort before other variants are interpreted.
2. `force_open`: changes only action dimension 7 to open; all requested EEF
   arm poses remain byte-identical.  This disables both gripper mechanics and
   the close-triggered arm interlock, so it isolates the whole close-command
   path rather than passive mechanical coupling alone.
3. `follower_default_limit`: after the normal installer, restores only the
   five passive mimic follower max-velocity values from 5 to their source
   default 174.5329285 rad/s.
4. `hold_close_anchor`: changes only the final-close fixed-anchor duration
   from 86 to 102 substeps.  The additional 16 substeps span the observed
   12-released-apply failure latency and keep the original failure boundary
   inside the hold.

Every completed physics substep records a closed three-phase state for all 13
joints: state after the arm setter/before the gripper setter, state after both
action setters, and causal post-physics state captured before the next arm
apply.  The fields include q, dq, ddq, position/velocity/effort targets,
computed/applied torque, all-six gripper state, and close-interlock lifecycle.
Each variant also writes a fully decoded 224x448 model-view MP4.  The final
contract uses an H.264 fast-start container and proves progressive scan from
every decoded frame flag, so it remains valid when the pinned ffprobe omits
the optional stream-level `field_order` key.

The runner publishes its result JSON after environment close and immediately
before the terminal SimulationApp close call.  The wrapper invokes a separate
post-Kit validator only after the simulator `srun` exits zero; that validator
re-derives the trace summary/cadence, checks the one-variable intervention,
and requires H.264, yuv420p, progressive 15 fps video with a full ffmpeg decode
before the wrapper creates `SUCCESS`.

Launch sequencing is fail-closed: run `baseline` alone first and require the
exact source abort plus 2,346 contiguous completed substeps.  Only after that
runtime gate may the three novel variants launch in parallel.

## Host validation before launch

- Ruff: pass.
- Python compile: pass.
- New focused tests: 15 passed, including lifecycle-v2 zero-exit promotion,
  nonzero-exit rejection, legacy-close-schema rejection, mismatch rejection,
  wrapper ordering, field-order compatibility, progressive-frame rejection,
  and MP4 fast-start rejection.
- All host-compatible tests (excluding the Isaac-only robust-DIK module):
  848 passed, 30 subtests passed.
- Full Isaac runtime validation and L40S results: pending immutable commit and
  independent review.

## Runtime launch corrections

- Job `1098565` did not enter the container: Slurm's generated `--wrap` script
  used `/bin/sh`, which rejected `pipefail`.  The retry launcher dispatches an
  explicit `/bin/bash -c`; the failed scheduler/log evidence is preserved.
- Job `1098566` reached the terminal 295-frame video, but demonstrated that
  `SimulationApp.close()` is terminal in the pinned Kit runtime.  A result
  publication placed after that call was unreachable, so the post-Kit
  validator rejected the missing JSON and no `SUCCESS` was created.
- Runtime contract v2 publishes the immutable raw result after environment
  close and before the terminal SimulationApp close.  The independent
  validator can run only after the simulator `srun` exits zero and records that
  zero exit in its final manifest.  This preserves full evidence while keeping
  completion fail-closed.
- Job `1098568` verified runtime contract v2: simulator step `.0` completed
  zero and published an immutable 32,555,231-byte result plus 295-frame video.
  It exactly reproduced the source abort after 2,346 contiguous substeps:
  panda joint 7 at policy step 293 / substep 2, 2.9236657619 rad/s versus the
  2.6099998951 limit, with evidence SHA
  `81ab9fb0cf1b74d67abbafb75ecc2ded5e606547fb46eec3e2b5a06acadd2959`.
  Validator step `.1` then failed closed because the pinned ffprobe reported
  H.264/yuv420p/448x224/15 fps/295 frames but omitted stream `field_order`.
  Newer ffprobe identifies the exact video as progressive and full decoding
  succeeds.  The v3 validator therefore requires every decoded frame's
  `interlaced_frame` flag to be zero, accepts only absent or `progressive`
  stream metadata, and additionally requires `moov` before `mdat`.  The runner
  now remuxes with `-movflags +faststart` before immutable publication.
- Exact-container probe job `1098569` completed `0:0`.  The pinned Ubuntu
  ffprobe 4.4.2 again omitted stream `field_order`, enumerated exactly 295
  frames, and reported `interlaced_frame=0` and `top_field_first=0` for all
  295; pinned ffmpeg completed a full `-xerror` decode.  This directly
  validates the v3 compatibility proof in the runtime image rather than only
  against the newer host ffprobe.
