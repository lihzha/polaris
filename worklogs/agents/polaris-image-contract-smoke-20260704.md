# PolaRiS image-contract smoke producer — 2026-07-04

- Owner: `CODEX_AGENT_ID=polaris_image_smoke_impl`
- Branch: `codex/image-contract-smoke-20260704`
- Base: `42e266353df71d5906e98975165f8aa021020dad`
- Producer revision: the commit containing this worklog (resolve with
  `git rev-parse HEAD`); it must be a direct child of the base above.
- Scope: implementation and host validation only. No local or cluster job was
  launched, no shared registry or canonical checkout was modified, and no
  checkpoint/policy/action/task-metric/canary claim is made.

## Goal and hypothesis

Close the simulator-to-Ego-LAP image boundary after removal of the accidental
OpenCV half-downsample/upsample. A real FoodBussing smoke should be able to
prove the native renderer conversion, robot compositing, and exact client
preprocessing/request serialization without loading or contacting a model.

## Implementation

- Added `scripts/smoke_splat_image_contract.py`.
- Added `tests/test_smoke_splat_image_contract.py`.
- The live smoke wraps the bound production `render_splat`,
  `SplatRenderer.render`, and `get_robot_from_sim` methods for one real
  `custom_render(True)` transaction and requires the exact two-camera event
  sequence.
- It independently recomputes `(clip(raw, 0, 1) * 255).astype(uint8)`, verifies
  exact native 720x1280 RGB pixels, recomputes `np.where` compositing, and uses
  `EgoLAPEefPoseClient.__new__` plus the real `_extract_observation` and
  `_build_request` methods without opening a network connection.
- It requires the pinned FoodBussing scene, initial-condition file/index 0,
  Hub metadata revision and digests; validates 126x224 content plus 49/49
  padding and resize-before-wrist-rotation; and exercises a noncommuting odd
  5x8-to-7x7 operation-order sentinel.
- It saves immutable exact `.npy` arrays, lossless PNGs/contact sheet, and an
  exact `openpi_client.msgpack_numpy` request. The removed OpenCV resize is
  executed only as a labeled counterfactual and must measurably differ.
- A closed strict-JSON raw result and ready marker are published mode 0444
  after `env.close()` and immediately before `SimulationApp.close()`. The raw
  result explicitly denies promotion and requires a later host finalizer after
  zero `srun`/Slurm accounting.

## Host validation

Executed from the isolated worktree using the existing Ego-LAP host venv
because the PolaRiS simulator venv intentionally lacks host NumPy:

```text
PYTHONPATH=$PWD/src:/home/lzha/code/PolaRiS/third_party/openpi/packages/openpi-client/src \
  /home/lzha/code/ego-lap/.venv/bin/python -m pytest -q \
  tests/test_smoke_splat_image_contract.py \
  tests/test_splat_image_contract.py \
  tests/test_lap_image_resize.py \
  tests/test_lap_eef_pose_client.py
72 passed, 30 subtests passed

/home/lzha/code/ego-lap/.venv/bin/ruff check \
  scripts/smoke_splat_image_contract.py tests/test_smoke_splat_image_contract.py
All checks passed!

/home/lzha/code/ego-lap/.venv/bin/ruff format --check \
  scripts/smoke_splat_image_contract.py tests/test_smoke_splat_image_contract.py
2 files already formatted

/home/lzha/code/ego-lap/.venv/bin/python -m py_compile \
  scripts/smoke_splat_image_contract.py tests/test_smoke_splat_image_contract.py
passed
```

The tests mock the simulator boundary, reject closed-schema tampering, verify
non-overwriting artifact publication and raw/ready binding, and AST-check the
actual production environment and Ego-LAP client call paths.

## Handoff and remaining gate

This producer has not been run under Isaac Sim. The orchestrator must deploy
the exact committed revision to a clean agent-owned l401 checkout, launch one
FoodBussing L40S smoke from the pinned simulator image, inspect every saved
array/PNG/contact sheet, and add a separate evidence-only host finalizer commit
that binds successful `srun` and complete Slurm accounting. Until then the
result is not promotable and cannot authorize a checkpoint canary.
