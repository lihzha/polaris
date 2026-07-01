# PolaRiS LAP train-matched resize worklog

## 2026-07-01 — Implementation and validation

- Agent: `polaris-resize-fix-20260701`.
- Goal: replace the Ego-LAP PolaRiS client's OpenPI/Pillow image downsampling
  with the preprocessing contract used by Ego-LAP training, without adding
  TensorFlow to the PolaRiS runtime or changing RGB ordering, wrist rotation,
  visualization, request keys, or control behavior.
- Isolation and provenance:
  - verified the source checkout was clean and that local `main` and
    `origin/main` both resolved to
    `2f4046bfe9e0b6a7ce5f86f76c7964e16c3238b4`;
  - created worktree
    `/home/lzha/code/PolaRiS-worktrees/polaris-trainmatch-resize-20260701`
    and branch `codex/polaris-trainmatch-resize-20260701` from that main commit;
  - explicitly merged prior EEF-evaluation branch
    `codex/ego-lap-eef-eval-20260630` at
    `9c50a8f28a1f3c41236945fcf69d490a3ec56183` in merge commit
    `675d78e0c65f39008cf4bd9fb058c734cfd7ec34`;
  - an initial merge command was accidentally evaluated in the clean source
    checkout, where it was a no-op (`Already up to date`); the deliberate
    merge was then run and verified in the dedicated worktree.
- Implementation commits:
  - `58373081137dcee31f69b7a6bdd7c4537026f22a` introduced the training-contract
    resize, runtime markers, and golden tests with Torch interpolation;
  - `1347b0c02916ccd08dee7e0782beee94135d5196` replaced the nearly equivalent
    Torch kernel with a bit-exact NumPy implementation after the audit below.

### Change

- Added `resize_lap_image`, which mirrors Ego-LAP training's uint8
  `_tf_resize_with_pad` implementation:
  - aspect-preserving resized dimensions use float32 ratio arithmetic and
    `floor`;
  - vectorized NumPy reproduces TensorFlow's half-pixel coordinates and exact
    float32 bilinear interpolation operation order, without antialiasing;
  - samples are rounded, clipped to `[0, 255]`, and cast to uint8;
  - zero padding is split symmetrically, with an odd remainder placed on the
    bottom or right exactly as in training.
- Replaced both external- and wrist-camera OpenPI/Pillow resize calls. The
  existing 180-degree wrist rotation still happens before wrist resizing.
- Added strict input validation and kept output at `224 x 224 x 3` uint8.
- Added stable one-time runtime records:
  - exact contract marker:
    `POLARIS_LAP_IMAGE_PREPROCESSOR=tf_bilinear_half_pixel_antialias_false_uint8_round_symmetric_zero_pad_224x224_numpy_float32_exact_v2`;
  - `POLARIS_LAP_IMAGE_IO=<compact JSON>` on the first model-image call, with
    external/wrist input/output shapes and dtypes. A standard run should report
    `720 x 1280 x 3` uint8 inputs and `224 x 224 x 3` uint8 outputs.
- Added no runtime dependency, CLI option, or trace schema field.

### TensorFlow parity evidence

The oracle is a direct test copy of Ego-LAP
`src/lap/datasets/utils/image_utils.py::_tf_resize_with_pad` using TensorFlow
2.15.0. The production candidate was also run inside the exact local evaluation
image `polaris-eval:cuda13` (digest
`sha256:e32265e25ae2d61b88cf0c530d5561d3fd48e6b2655f48cb583d18fb18851006`,
NumPy 1.26.0), and its bytes were compared outside the container with the
TensorFlow oracle:

| Case | Maximum error | Mean absolute error | Differing channels |
| --- | ---: | ---: | ---: |
| synthetic PolaRiS 720x1280 | 0 | 0 | 0 / 150,528 |
| synthetic DROID 180x320 | 0 | 0 | 0 / 150,528 |
| synthetic odd landscape 17x29 | 0 | 0 | 0 / 150,528 |
| synthetic odd portrait 29x17 | 0 | 0 | 0 / 150,528 |
| natural `docs/images/Teaser Figure.png` | 0 | 0 | 0 / 150,528 |

The final implementation is byte-identical across all 752,640 checked output
channels: maximum and mean error are both zero, with no residual differing
pixels or channels. A broader local audit over 48 fixed/random input and target
shapes also found zero float32 bit mismatches before rounding and zero uint8
mismatches after rounding/padding. The first Torch candidate had a 1-LSB error
in two of the 752,640 target-runtime channels; it was rejected despite its
algorithmic parity. The exact NumPy path benchmarks at 2.288 ms per
720x1280-to-224 image (437 images/s, 200 calls after warmup) in the target
container, which is practical for two images at the policy query cadence.

### Validation

- TensorFlow-oracle suite:
  `TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES= PYTHONPATH="$PWD/src:/home/lzha/code/ego-lap/third_party/openpi/packages/openpi-client/src" /home/lzha/code/ego-lap/.venv/bin/python -m unittest discover -s tests -p 'test_lap*.py' -v`
  — 12/12 passed, including the static TensorFlow golden and synthetic/natural
  oracle parity test.
- Target runtime suite:
  `docker run --rm --name codex-polaris-resize-exact-test-20260701 -v "$PWD:/workspace/polaris:ro" -w /workspace/polaris polaris-eval:cuda13 /bin/bash -lc 'PYTHONPATH=/workspace/polaris/src /.venv/bin/python -m unittest discover -s tests -p "test_lap*.py" -v'`
  — 12 tests successful (11 passed and the optional TensorFlow-oracle test
  skipped because TensorFlow is intentionally absent from the runtime image).
- Ruff lint and format checks passed for all three owned Python files.
- `py_compile` and `git diff --check` passed.
- The bounded validation containers were removed automatically. No simulator,
  policy-server, GPU evaluation, Slurm job, monitor, or other owned process is
  active.

### Handoff

- The orchestrator should assert both `POLARIS_LAP_IMAGE_PREPROCESSOR=...` and
  the parsed `POLARIS_LAP_IMAGE_IO=...` record in every task's `eval.log`
  before accepting a rerun.
- Remaining work is orchestration-owned integration, deployment, canary
  evaluation, and full rerun; this implementation agent launched none of them.
