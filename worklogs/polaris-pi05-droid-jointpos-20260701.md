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
