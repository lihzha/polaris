# PolaRiS splat image-contract repair

## Scope

The Ego-LAP PolaRiS evaluation path rendered both `external_cam` and
`wrist_cam` at the configured simulator resolution, converted them from float
RGB to uint8, and then unconditionally downsampled and upsampled every splat
image with OpenCV before the policy client's training-matched preprocessing.
That extra resampling was not present in training and blurred even-sized images;
for odd dimensions it also changed the image shape.

This implementation branch starts from the endpoint-cadence repair commit
`39418400493cdcf8cd8272608980a798f7929a20`.  It is deliberately limited to
the renderer image boundary and does not authorize checkpoint evaluation.

## Change

- Removed the unconditional OpenCV half-resolution/down-up pass from
  `ManagerBasedRLSplatEnv.render_splat` and the now-unused `cv2` import.
- Factored the existing clip-to-[0,1], multiply-by-255, truncating uint8
  conversion into a pure host-testable helper.  The helper performs no spatial
  operation and is applied to every image returned by `SplatRenderer.render`,
  which includes both the static external and moving wrist camera entries.
- Added exact pixel regressions, including an odd `5 x 7` image that must retain
  its shape, plus a source-structure gate that rejects any reintroduced resize
  inside `render_splat`.

## Audit notes

`SplatRenderer` constructs each camera at the Isaac sensor's native
`image_shape` and its renderer emits that height and width directly.  Its
`render` method returns all configured cameras; `render_splat` updates wrist
extrinsics and then converts every returned mapping entry, so the removed
down-up pass affected both external and wrist observations.  The Ego-LAP EEF
client subsequently performs the intended training-matched 224x224
resize-with-pad for external images and resize-with-pad-then-rotate-180 for
wrist images.

Other resizes found during the audit are outside this mismatch: `FakeClient`
uses OpenCV only for its own 224x224 visualization; `DroidJointPosClient` uses
its policy-specific 224x224 preprocessing; and the unreferenced
`PILtoTorch` utility contains a generic PIL resize.  None is on the
`EgoLAPEefPoseClient` observation path before its contracted preprocessing.

## Validation

Host validation used the already-provisioned Python 3.11 environment at
`/home/lzha/code/ego-lap/.venv/bin/python`, with this worktree's `src` first on
`PYTHONPATH`.

```bash
PYTHONPATH=$PWD/src /home/lzha/code/ego-lap/.venv/bin/python -m pytest -q \
  tests/test_splat_image_contract.py \
  tests/test_lap_image_resize.py \
  tests/test_lap_eef_pose_client.py \
  -p no:cacheprovider
```

passed `52` tests plus `30` subtests.  This includes all three new conversion
and integration regressions.

The broad host-safe command was:

```bash
PYTHONPATH=$PWD/src /home/lzha/code/ego-lap/.venv/bin/python -m pytest -q \
  tests --ignore=tests/test_robust_differential_ik.py \
  -p no:cacheprovider
```

It passed `1,154`, skipped one expected NFS-only test, and retained 11 closed
historical/evidence identity failures.  A temporary clean detached worktree of
the exact base commit was run with the same command: it passed `1,151`, skipped
one, and produced the identical 11 failure node IDs.  Thus this patch adds
three passing tests and no broad-suite regression.  The failures bind old v5,
reset-replay, and not-yet-finalized v6 producer identities; none references a
changed image-contract path.  The excluded `test_robust_differential_ik.py`
requires Isaac Lab, which is unavailable in the host runtime.

Ruff check and format-check passed for all changed Python files.  Python byte
compilation and `git diff --check` also passed.  No GPU process, simulator,
Slurm allocation, checkpoint server, or evaluation was launched.  The image
fix still requires a fresh real-PolaRiS image-contract smoke before any
checkpoint canary or standard evaluation is authorized.
