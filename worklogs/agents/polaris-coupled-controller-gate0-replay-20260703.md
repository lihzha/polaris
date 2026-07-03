# PolaRiS coupled-controller Gate 0 replay — 2026-07-03

- Agent: `coupled_controller_diagnosis`
- Branch: `codex/coupled-controller-gate0-replay-20260703`
- Worktree:
  `/home/lzha/code/PolaRiS-worktrees/coupled-controller-gate0-replay-20260703`
- Exact base: `712240cbb215ecb31830cdb2ee65e91704160372`
- Scope: baseline-only implementation, host validation, and reviewed launch
  surface. No controller candidate, GPU, simulator, Slurm job, registry write,
  deployment, or canonical-checkout mutation was performed.

## Gate definition

Gate 0 is a model-free identity replay for the two preserved FoodBussing IC0
canaries. It must reproduce, not repair, these exact outcomes:

- official LAP-3B job `1098292`: attempted policy step `117`, physics substep
  `6`, `panda_joint5`, evidence SHA-256
  `63c061ec5a47a8bc085547f2abd8dcbc266c9616664d252e29c39ef53864a5f3`;
- reasoning-43075 job `1098294`: attempted policy step `112`, physics substep
  `2`, `panda_joint7`, evidence SHA-256
  `3c6242a645b40fe29f7223dc4a146cdb7ee04fe661b136098929bb9b973580b8`.

Any continuation, earlier/later abort, different joint/substep/digest,
termination, truncation, incomplete ring/tail, lifecycle mismatch, or source
drift fails the gate. An expected `DifferentialIKInvariantError` is the only
passing simulator outcome.

## Exact source and fixtures

The deterministic generator accepts only these mirrored JSONL bytes:

- job `1098292`: 338377 bytes, SHA-256
  `490ba92f39abb1fd83c8382dd1b7a16f4e1a12e86df29cbfb484b6395474789c`;
- job `1098294`: 331875 bytes, SHA-256
  `a48107f2268a38f507bae5848f194f2680ccc52eedd042039cea5fd3cbebd948`.

It checks strict JSONL, duplicate/nonfinite rejection, exact event counts,
query cadence, every observed action against its recorded query plan, serving
contract, flow sampler, frame, state-layout, normalization stats/formulas,
execution horizon, and terminal failure fields. Assets are pinned to
FoodBussing scene `82cd641e...`, IC file `40091fae...`, PolaRiS-Hub revision
`8c7e410...`, robot USD `d8379925...`, and IC index zero.

Both fixtures contain exactly actions 0–119. Query 14 begins at step 112 and
contains 16 planned actions, but its production execute window is only chunks
0–7. The fixture retains the full 16-action plan digest as provenance while
replaying only the eight executable actions:

- official: observed 0–117 plus recorded/planned chunks 6–7 at 118–119;
- reasoning: observed 0–112 plus recorded/planned chunks 1–7 at 113–119.

There is no synthetic action 120 and no claim that post-failure planned bytes
were executed by the source jobs. The official close begins at step 115; its
38th 120-Hz slew write occurs during step 119, so the recorded execute window
is sufficient for endpoint-arrival diagnostics if a future candidate reaches
it.

## Production lifecycle and evidence

The runner uses `EgoLapEefPoseActionCfg`, enables the existing 64-entry arm
failure-substep trace, disables the wrist-energy brake, configures production
gripper velocity limits, and calls `install_eef_gripper_runtime` after the
first explicit reset. It validates FoodBussing IC0 and the runtime frame before
beginning episode accounting.

Reset/render behavior is byte-bound to production sources:

- `scripts/eval.py` calls
  `env.reset(object_positions=initial_conditions[episode])` with environment
  seed `None` and no `expensive` override, so reset uses the default expensive
  render;
- `PolicyArgs.render_every_step` defaults to `True`;
- `LapEefPoseClient.rerender` returns `True` in that mode;
- production eval passes `expensive=policy_client.rerender`, and Gate 0 passes
  `expensive=True` for every attempted step.

The trace-only gripper subclass calls the unchanged production target-slew
`process_actions`/`apply_actions` and records state around it; it does not write
an additional target. It live-compares its constants with
`polaris.eef_gripper_runtime` and asserts `robot.data.joint_names[7:13]` is:

1. `finger_joint`
2. `right_outer_knuckle_joint`
3. `left_inner_finger_joint`
4. `right_inner_finger_joint`
5. `left_inner_finger_knuckle_joint`
6. `right_inner_finger_knuckle_joint`

The failure artifact contains the independently validated full 64-entry arm
causal ring and the last 64 all-six gripper substeps, including position,
velocity, acceleration, position/velocity target, effort target, raw binary
action, requested endpoint, and driver target after the production setter.
Official/reasoning require respectively 942/898 completed finger applies and
118/113 process calls.

## Lifecycle, namespace, and immutable promotion

The runner must execute inside one numeric Slurm `srun` step with one task,
rank zero, and a caller-provided 64-hex launch ID. Output is restricted to:

`<root>/<variant>/job_<job>/launch_<launch-id>/gate0-<variant>.*`

The raw and ready files are non-overwriting, fsynced, mode 0444, single-link
files. After `srun` returns exactly zero, the host status writer revalidates
the raw capture and writes an immutable record binding launch ID, job ID,
in-srun step ID, timestamps, raw/ready bytes and mtimes. The finalizer then
binds that status to the clean exact commit, runner/generator/status/finalizer
sources, variant fixture, saved submitted script, container image, live asset
files/revision metadata, production reset/render sources, runtime contract,
arm ring, and gripper tail before non-overwriting attestation publication.

The submitted wrapper prechecks every frozen source/script/fixture hash before
starting `srun`, uses the established Pyxis/Isaac environment, saves itself
mode 0444, and finalizes only after a zero outer `srun` return. Official and
reasoning must use separate `sbatch` allocations and launch IDs.

## Host validation

No CUDA/Isaac/Slurm process was launched.

- New generator/replay/status/finalizer tests plus existing boundary-ring
  tests: `67 passed`.
- Existing EEF gripper runtime, target-slew, and runtime-contract suites:
  `134 passed` (one environment `pynvml` deprecation warning).
- Full host suite excluding only the real-Isaac robust-controller module:
  `646 passed`, `30 subtests passed` (the same warning).
- Deterministic generator `--check`: both committed fixtures reproduced
  byte-for-byte.
- Ruff check and format check, Python byte compilation, Bash syntax,
  ShellCheck, and `git diff --check`: passed.
- `tests/test_robust_differential_ik.py` cannot collect in the host shell
  because Isaac Lab is unavailable. The requested next step is the separately
  approved real-Isaac Gate 0 itself; it has not been launched.

## Frozen candidate identities

- generator:
  `d742192c3579593199fdba16c1e7bced5283e18f65ea26d71dead8a070e6c50e`
- runner:
  `4c62872713145adf4e82bc6061c6c25965df49aacb7540889c5547e08c7ddf8f`
- status writer:
  `3c88d80c12baf294146aca8f4695a0d260a49e9755443849fcaaa5bc26f2c00d`
- finalizer:
  `2654a2c1863ef093ce675e2778eb4743de263e324a179cb7bd5c8c20f2765160`
- submitted wrapper:
  `1982befe6d7eb911623aa398a54811d6005914642ba71c0dbae979add75d1f36`
- official fixture:
  `0534760269593ee5d00d92a92dfbbc424482725a0dc47556f57fbd54c8a44872`
- reasoning fixture:
  `22c6a5b73b59aaccb54d0644be13059fb620d6bff81b3fd8731c873c94833527`
- established image `polaris-eval-cuda13-fd00a51.sqsh`:
  `ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a`

These are pre-commit candidate identities. Re-run all hashes after any review
edit; do not launch with a stale value.

## Proposed parallel L40S submissions (not executed)

After commit/push, deploy that exact commit as a detached clean cluster clone,
confirm the image hash above, and prepare two independent launch IDs (for
example, `openssl rand -hex 32`). Submit the same frozen wrapper twice with
`--no-requeue`, one L40S GPU, 16 CPUs, 64 GB memory, and a 10-minute limit.
Comparable Isaac steps reached about 31 GB MaxRSS, so 32 GB is not an
acceptable promotion margin for 118 expensive-render steps plus finalization:

```text
sbatch --parsable --no-requeue --time=00:10:00 --cpus-per-task=16 \
  --mem=64G --gres=gpu:l40s:1 --job-name=pol-g0-off \
  --export=ALL,GATE0_VARIANT=official_lap3b,<frozen Gate0 variables> \
  scripts/run_eef_pose_canary_gate0_srun.sh

sbatch --parsable --no-requeue --time=00:10:00 --cpus-per-task=16 \
  --mem=64G --gres=gpu:l40s:1 --job-name=pol-g0-rsn \
  --export=ALL,GATE0_VARIANT=reasoning_43075,<frozen Gate0 variables> \
  scripts/run_eef_pose_canary_gate0_srun.sh
```

`<frozen Gate0 variables>` must include the exact commit and hashes above,
distinct launch IDs, cluster repo/output paths, image path, a same-path
read-only repo mount, same-path read-only PolaRiS-Hub mount, same-path
read-write output mount, `/dev/shm`, a per-job host cache mounted at `/cache`,
and the NVIDIA Vulkan ICD mount. Do not submit until an independent reviewer
accepts the exact commit/diff, variables, mounts, and node/partition routing.

## Git handoff

- Implementation commit: pending.
- Push: pending.
- Real-Isaac Gate 0: pending independent approval; no job launched.
