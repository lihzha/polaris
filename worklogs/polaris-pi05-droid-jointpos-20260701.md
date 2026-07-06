# Official pi0.5-DROID PolaRiS evaluation

## 2026-07-01 — plan and contract gate

- Agent: `codex-pi05-polaris-20260701`.
- Goal: evaluate the official PolaRiS-ready pi0.5 checkpoint across all six
  standard tasks, with one FoodBussing canary before a six-job 50-rollout
  fan-out.
- PolaRiS base: `c32b28732cda59eaf76a04a83e78aca1feffd092`.
- OpenPI submodule: `bd70b8f4011e85b3f3b0f039f12113f78718e7bf`.
- Config: `pi05_droid_jointpos_polaris`.
- Checkpoint:
  `gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris`.
- Checkpoint-local normalization SHA-256:
  `57ce9956f9e07d65f8a8205aabec72d436a2c8927f53edb40c7a77b14a5a90c7`.
- Contract: external RGB to `base_0_rgb`, unrotated wrist RGB to
  `left_wrist_0_rgb`, masked blank `right_wrist_0_rgb`; 224x224; seven joint
  radians plus closed-positive gripper state; 15x8 response converted by
  OpenPI to absolute joint targets; execute eight actions at 15 Hz.
- The LAP `panda_link8`/Robotiq-base mismatch does not affect this joint-space
  baseline.
- Shared Ego-LAP training registry: not applicable because this is an external
  OpenPI checkpoint and creates no Ego-LAP checkpoint lineage. This worklog and
  per-run manifests are the authoritative run records.
- Initial validation: the new trace-only client instrumentation and existing
  LAP client tests pass (`11 passed`); Ruff lint passes. No job has been
  launched yet.

## 2026-07-01 — implementation and independent review

- Added fail-closed setup, one-task worker, ordinary-job submitter, trace
  auditor, and a committed 27-object GCS checkpoint manifest.
- Setup and every GPU worker verify the complete 12,434,530,837-byte checkpoint
  against every public GCS object size and MD5. The verifier rejects missing or
  extra files (including an unexpected `model.safetensors`) and all symlinks.
- The worker pins the exact checkpoint URI, config, parent commit, OpenPI
  submodule commit, normalization SHA, checkpoint-manifest SHA, Pyxis-image
  SHA, prompts, image slots, unrotated wrist convention, action shape, and one
  allocated GPU UUID.
- Trace records now distinguish a predicted/planned chunk from each action
  actually emitted by the client. The auditor requires duplicate-free,
  contiguous reset/query/action indices and reconciles emitted actions and
  query counts against each CSV episode length. Reusing a run directory is
  rejected.
- The original client behavior is preserved: its strict `> 0.5` gripper rule
  and NumPy float64 promotion are covered by tests.
- Submission uses an explicit environment whitelist, a file lock, append-only
  attempt rows, duplicate-task protection, ordinary `sbatch` jobs, and forces
  `DRY_RUN=0` in production.
- Independent launcher and trace reviews found no remaining launch blocker
  after these changes.
- Validation: Ruff clean; `bash -n` clean; ShellCheck clean; `git diff --check`
  clean; 19 focused tests pass. The full suite additionally reaches a
  pre-existing IsaacLab controller test that cannot import IsaacLab in the
  workstation environment; that test is deferred to the validated L40S
  container/canary.
- No cluster job has been launched yet.

## 2026-07-01 — deployment, setup, and canary

- Immutable launch commit: `b02decad1d9abb7518755e7667a058085f464462`.
- GitHub push to the upstream PolaRiS repository was rejected with the expected
  permission error. The tracked source was therefore deployed with a complete
  Git bundle rather than an untracked copy:
  `polaris-b02deca.bundle`, SHA-256
  `bd4dd770a2d4c72f723326663473a05e4816c697a2fb174e273cf92e259cda22`.
- Frozen remote checkout:
  `/lustre/fsw/portfolios/nvr/users/lzha/src/PolaRiS-pi05-polaris-v1-20260701T222047Z`.
  It was detached and clean at `b02deca`, with OpenPI detached at `bd70b8f`.
- Run namespace: `pi05-polaris-v1-20260701T222047Z`.
- One-time setup installed the 242-package pinned OpenPI environment and fully
  verified all 27 checkpoint objects (12,434,530,837 bytes). Checkpoint object
  manifest SHA-256:
  `7abd0c2294d442d429a77655783232206b2b30d95c508d435503135a5523a11c`.
  Pyxis image SHA-256:
  `ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a`.
- FoodBussing canary job `1090783` completed `0:0` on one L40S. Its single
  rollout was a strict failure but achieved `2/3` progress with no numerical
  failure. Trace validation passed with 57 queries, 450 emitted actions, and
  SHA-256
  `1dd6616c19be1890ffc772d952640069cb52f69977f61c328a36532db2738a96`.
- Visual inspection found two sharp, correctly oriented camera streams and
  coherent multi-object grasp/place behavior. Canary video:
  `/home/lzha/code/shared_artifacts/polaris-pi05-20260701/canary/DROID-FoodBussing/episode_0.mp4`.

## 2026-07-01 — full sweep launch and recovery

- Six ordinary one-L40S jobs were launched in parallel, 50 rollouts each:
  `1090805` BlockStack, `1090806` FoodBussing, `1090807` PanClean,
  `1090808` MoveLatteCup, `1090809` OrganizeTools, and `1090810` Tape.
- Every worker independently passed the full 27-object checkpoint MD5 audit,
  restored the 6.2-GiB Orbax parameters, loaded checkpoint-local `assets/droid`
  normalization, and emitted the exact joint/image contract marker.
- BlockStack attempt `1090805` completed four rows, then a finite-to-nonfinite
  PhysX state divergence caused an uncaught observation exception and hung
  teardown. The allocation was canceled after preserving its artifacts.
- Commit `edc399141714fb3fd62734f4b155ba22b8f5534b` introduced a typed
  `JointPositionObservationNumericalError` and records such simulator failures
  in CSV/video/trace rather than aborting. The 50-seed BlockStack task was
  rerun from seed zero as job `1090902` in namespace
  `pi05-polaris-blockstack-retry-v2-20260701T230527Z`. Its deployment bundle
  SHA-256 was
  `45cad8e9f8baa6de6ee6c1853e9745ca8234686f7f470c55d082b2c42b9fcdd1`.
- OrganizeTools attempt `1090809` preserved 16 complete rows/videos, then Isaac
  hung inside episode 16 with a live process/GPU but no trace progress. It was
  canceled and its process/server cleanup was verified.
- Commit `40df043ccf5d57eefb108e32c56294ddf345f10c` added fail-closed prefix
  resume. It retained exactly 16 rows/videos and 8,112 completed trace records,
  discarded 440 records from the hung partial reset, independently validated
  the prefix, initialized the next reset at 16, and continued in fresh job
  `1091046`. Deployment bundle SHA-256:
  `4bb8daa7dd4c88b7318ed18354954db5274631c6333c123bbef73e879d6d862f`.
- Resume continuity was verified before continuing: the first appended reset
  contained 57 queries and 450 actions, produced row/video 16, and the final
  reconstructed trace contained 50 contiguous resets.
- Focused validation after both fixes: 14 tests pass in the exact remote OpenPI
  environment; Ruff, `bash -n`, ShellCheck, and `git diff --check` pass.

## 2026-07-01 — final official metrics

All six authoritative jobs completed with top-level Slurm state `COMPLETED`,
exit `0:0`, task/run `SUCCESS` markers, exact CSV/video counts, and final trace
validator status `pass`.

| Task | Success | Mean progress | Recorded numerical | State-OOB lower bound | State-valid success |
| --- | ---: | ---: | ---: | ---: | ---: |
| BlockStack | 4/50 (8%) | 0.5114 | 5 | 12 | 3/50 (6%) |
| FoodBussing | 12/50 (24%) | 0.6333 | 0 | 1 | 12/50 (24%) |
| PanClean | 24/50 (48%) | 0.7800 | 0 | 1 | 24/50 (48%) |
| MoveLatteCup | 4/50 (8%) | 0.1933 | 0 | 0 | 4/50 (8%) |
| OrganizeTools | 1/50 (2%) | 0.3667 | 0 | 1 | 1/50 (2%) |
| TapeIntoContainer | 8/50 (16%) | 0.2067 | 0 | 0 | 8/50 (16%) |

- Official aggregate: `53/300 = 17.6667%` success; mean progress `0.44857`.
- Recorded numerical failures: `5/300`, all in BlockStack. They are included
  as failures; their videos and exact truncated action counts are preserved.
- State-bound audit: 15 episodes have at least one query state outside standard
  Panda limits (lower bound because state is sampled every eight actions).
  Ten were not caught by the original numerical-failure predicate. One nominal
  BlockStack success (episode 30) was state-invalid.
- State-valid aggregate, keeping invalid seeds in the denominator as failures:
  `52/300 = 17.3333%`.
- Aggregate accounting: 300 metrics rows, 300 nonempty videos, 16,947 policy
  queries, and 133,798 emitted actions.
- Per-task trace SHA-256:
  - BlockStack:
    `814adb09e94360a421438d7db131e9e2825f8be36b5863154059bebdd6858146`
  - FoodBussing:
    `06e577fc56d8317fd6c22e77eb941e912ff7114a2402b8e8a5079945ae1e6f9d`
  - PanClean:
    `6183722b6e2e6fcae8a3d1888963268e8cb9e6449aa8bef9b37193ae91e202df`
  - MoveLatteCup:
    `8543ed0257b15e646839a4f1c22a52f952c31fdfcfa1411e212a264b8c5cc98b`
  - OrganizeTools:
    `11e2b618320b8fbcb753707b43252b68337249f917a41e7336dc9a104fd8a01d`
  - TapeIntoContainer:
    `4ec5de14ac0bf2e62c1f4fe8bff61f91301f6d95eef8671516b1f72b9f4318f6`

## 2026-07-01 — train/eval and physical-contract audit

- No static normalization, image-order, resolution, wrist-rotation, state, or
  action-semantic mismatch was found. Training and inference both use the
  checkpoint-local DROID quantile statistics, delta-first-seven joint training,
  absolute-action reconstruction from current state, 15x8 output, eight-action
  execution, external/wrist/masked-third image order, unrotated 224x224 RGB,
  and closed-positive gripper semantics.
- There is no explicit seven-joint target clipping in the pinned OpenPI output
  transforms, PolaRiS client, or Isaac Lab `JointPositionAction` path. PhysX
  receives raw targets and constrains realized state only through articulation
  limits and effort saturation. This is a real actuation/safety gap; real robot
  hardware clipping must not be assumed.
- The severe state divergences were not initiated by infeasible immediately
  preceding chunks. Every first BlockStack divergence, the Food divergence, and
  the Organize divergence followed an eight-action chunk within Panda limits.
  This implicates simulator/controller/contact behavior rather than a decoding
  or normalization error.
- State-OOB lower-bound episode sets:
  - BlockStack: `3,5,6,16,21,24,25,28,30,34,40,49`
  - FoodBussing: `3`
  - PanClean: `36`
  - MoveLatteCup: none
  - OrganizeTools: `12`
  - TapeIntoContainer: none
- Target-only excursions are logged separately and do not reduce the primary
  state-valid metric because the simulated articulation may saturate them.
  They do, however, require explicit bounds/slew enforcement before any real
  hardware deployment.

## 2026-07-01 — local artifacts and visualization

- Final local root:
  `/home/lzha/code/shared_artifacts/polaris-pi05-20260701/final`.
- Aggregate JSON SHA-256:
  `7168f1362dfed5eb9df063c0ee1b69bae5c94efdf8cfd9e1a635ea850c70ce49`.
- Aggregate TSV SHA-256:
  `e7c3ea338232d162bad05bacd202a7069a36fcdefeb16163aa24146030abf00d`.
- One visually inspected, state-valid success from every task:
  `final/gallery/successes`. Storyboard SHA-256:
  `d0daf5ad97baa67782b1847ba74421528d73375e8047db9624569b0b46d9e421`.
- Diagnostic finite-divergence videos and boundary storyboards:
  `final/gallery/diagnostics`. Combined boundary storyboard SHA-256:
  `143c010ad1ac3bf50189c58c86cd3d415208092c34a51ce29aac45ce01c5226e`.
- Success gallery URL:
  `http://localhost:8765/view?path=shared_artifacts/polaris-pi05-20260701/final/gallery/successes`.
- Diagnostic gallery URL:
  `http://localhost:8765/view?path=shared_artifacts/polaris-pi05-20260701/final/gallery/diagnostics`.

## 2026-07-01 — cleanup

- No listed canary/full/retry/resume job remains in `squeue`.
- Authoritative jobs `1090806`, `1090807`, `1090808`, `1090810`, `1090902`,
  and `1091046` all report `COMPLETED 0:0` in `sacct`.
- Superseded attempts `1090805` and `1090809` remain preserved as canceled
  evidence; neither contributes rows to the authoritative aggregate.
- Policy-server processes and job ports were gone after allocation cleanup.
- Deleted only the nine job-scoped Isaac runtime caches for this evaluation
  (`1090783`, `1090805`–`1090810`, `1090902`, `1091046`), reclaiming about
  274 MiB. Checkpoint cache, frozen source, logs, metrics, traces, and videos
  remain intact.

## 2026-07-06 — latest-tree native joint-position canary

- The co-trained native checkpoint was rerun through the latest PolaRiS tree,
  rather than through the new delta-position adapter. Frozen source commit
  `4b137a67363d85b0bca9174ac8eb7d15da0c5eee` contains runtime commit
  `25563f0b99ff03191aa7cc28c6947c60b4e6cafc`; OpenPI remained pinned at
  `bd70b8f4011e85b3f3b0f039f12113f78718e7bf`.
- Setup job `1100640` completed `0:0` on a tracked CPU allocation. Evaluation
  job `1100641` then completed `0:0` on one L40S in 7 minutes 58 seconds.
  The immutable 27-object, 12,434,530,837-byte checkpoint passed full MD5
  verification. Checkpoint-manifest SHA-256 is
  `7abd0c2294d442d429a77655783232206b2b30d95c508d435503135a5523a11c`;
  checkpoint-owned DROID normalization SHA-256 is
  `57ce9956f9e07d65f8a8205aabec72d436a2c8927f53edb40c7a77b14a5a90c7`.
- The first two pinned FoodBussing conditions both reached 450 steps without a
  numerical failure. Episode 0 failed with progress `1/6`; episode 1 failed
  with progress `1/3`. This bounded `0/2` canary is not a replacement estimate
  for the historical 50-condition `12/50` FoodBussing result.
- Trace validation passed with exactly 57 queries and 450 emitted actions per
  episode, 15x8 model responses, execute-eight chunks, and SHA-256
  `7a19fdcb27b80b5e62715af2008c30665b9b6a48dac7971b4ffef14cd990c2c2`.
  Both H.264 High/yuv420p videos are 448x224, 15 fps, 450 frames, and 30
  seconds; full decode passed. Visual inspection found coherent manipulation:
  episode 0 moved one item into or around the bowl before unsuccessful work
  near the grape/cup area, while episode 1 repeatedly manipulated the grape
  and orange objects without filling the bowl.
- No train/eval contract drift was found. The run used config
  `pi05_droid_jointpos_polaris`, checkpoint-local `assets/droid` statistics,
  external then unrotated wrist images padded to 224x224 plus a masked third
  slot, seven joint positions plus closed-positive gripper state, absolute
  joint targets reconstructed by OpenPI, and the historical 15x8/execute-eight
  FLOW path. The current fresh-run `DroidJointPos` client and launcher are
  byte-identical to commit `40df043`; changes relative to the original
  FoodBussing commit `b02deca` add resume indexing and typed nonfinite-state
  handling without changing a finite fresh trajectory. The shared evaluator's
  render decision also reduces to the historical `policy_client.rerender` for
  this client.
- Replay forensics found bit-identical initial joint/gripper state for both
  conditions, but different first external and wrist pixel hashes before the
  first policy query. The episode-1 first frame remains visually equivalent
  (PSNR 52.4 dB), and the first emitted arm target differs by only about 0.0011
  rad; the closed-loop state then diverges past 0.01 rad by query 2 and 0.1 rad
  by query 3. Historical same-source jobs `1090783` and `1090806` exhibit
  first-frame variation of the same magnitude and different progress on
  condition 0. Together with both logs' `Seed not set for the environment`
  warning, this supports unseeded Isaac/RTX and contact sensitivity rather than
  a new normalization, state, image-order, or action-order mismatch.
- Sampled query states remained within standard Panda joint limits in both new
  episodes. Episode 0 nevertheless received two executed joint-4 targets just
  beyond the standard lower limit, and both episodes had out-of-range values
  somewhere in the full unexecuted 15-step responses. Historical runs show the
  same target-only behavior. This remains a controller-safety caveat because
  the native joint-position path has no explicit target clamp; it is not the
  first source of the historical-versus-current divergence.
- Durable remote run:
  `/lustre/fsw/portfolios/nvr/users/lzha/results/polaris-pi05/pi05-jointpos-latest-4b137a6-20260706T200023Z/pi05-jointpos-latest-4b137a6-20260706T200023Z_official-pi05-polaris_DROID-FoodBussing_1100641`.
  Mirrored local inspection root:
  `/home/lzha/code/cluster_results/l401/polaris_pi05_jointpos_latest_4b137a6_1100641`.
