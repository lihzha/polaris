# Official pi0.5 joint-position historical-control rerun

## 2026-07-08 — purpose and preserved control

- Goal: rerun the prior FoodBussing seed-0 implementation as a historical
  control after fixing the action-450 autoreset/scoring boundary and adding
  full train/eval/evidence attestation. This branch starts from historical
  seed-control commit `4645b4e092c7f3264f07e4b8216cd1641b289765`.
- The control intentionally retains its older direct evaluator/Isaac shutdown
  lifecycle and excludes the newer joint-velocity, delta-position, and
  `NativeEvaluatorLifecycle` integration. It is therefore not a copy of the
  current evaluator.
- Portable contract commit
  `5a575e6194b8bdf3a3a25e6b5d39bf8802d0e0eb` carries the exact official checkpoint/config, global checkpoint
  DROID normalization, native-image/server-PIL resize probe, tokenizer and
  package attestation, joint-position client/controller/runtime checks,
  schema-4 trace/RNG closure, pinned asset/media validation, terminal PNG, and
  immutable evidence transaction used by the current arm.
- The branch-local hand-port adds only `runtime_contract_path` plus the native
  joint-position action/observation config, internal-451/outer-450 timeout,
  live runtime publication, per-rollout seed/reset binding, exact 450-step
  execution recording, render-every-step composite frames, and the trace-bound
  post-action-450 PNG. It preserves the historical direct `env.close()` and
  `SimulationApp.close()` behavior.
- Validation: the complete historical host-runnable suite passes 103 tests plus
  5 subtests with only the Isaac-dependent test excluded; the focused
  joint-position/evidence surface passes 65 tests. Ruff format/lint, Python
  compilation, and Git whitespace checks pass. The portable shell surface was
  already validated with Bash syntax and ShellCheck before cherry-pick. No GPU
  job has yet been launched from this final control candidate.
- Branch-local integration commit:
  `94c48f525a53a67fb8c03841682b9e9bf16db3a2`.

## 2026-07-08 — shared schema-4 physical audit

- CPU setup job `1101795` was intentionally canceled before evaluation after
  the final prelaunch review found that the current tree's optional physical
  analyzer had not yet learned the schema-4 execution records. No L40S job was
  launched from that attempt.
- This historical control now carries the same byte-identical schema-4
  joint-bound analyzer, aggregate summarizer, and tests as the current arm.
  The analyzer requires one execution record per emitted action and audits
  every measured post-action state, including action 450; legacy query-only
  traces are retained only with an explicit lower-bound label. These files do
  not participate in rollout execution, so the intended historical evaluator
  lifecycle remains unchanged while paired raw and state-valid results use the
  same committed analysis implementation.

- The subsequent adversarial closure is also copied byte-identically from the
  reviewed current arm: homogeneous schema and complete query/action/execution
  ordering, query/post-action state continuity, canonical Panda limits and
  `1e-3` rad tolerance, trace/CSV SHA-256 binding, sealed-evidence metrics
  identity, and fail-closed aggregation of success, numerical, state, and
  target episode sets. It rejects execution deletion, schema downgrade,
  reordered pairs, missing queries, swapped steps, stale metrics, noncanonical
  tolerance, and impossible state-valid counts. These additions remain
  analysis/evidence-only and do not change the historical rollout lifecycle.

- Clean setup `1101806` exposed a shared Lustre path-alias false reject after
  package installation; historical setup `1101807` was canceled once the
  common cause was known. The paired interpreter attestation now proves the
  resolved virtual-environment prefix plus executable inode, rejects a foreign
  venv even when it shares the same base Python, and then persists the
  canonical Git-root executable spelling for downstream runtime validation.
  The serving contract and alias tests are byte-identical to the independently
  reviewed current arm; rollout behavior remains unchanged.

## 2026-07-08 — v4 setup cancellation after shared integrity-gate failure

- Historical setup job `1101809`, from immutable PolaRiS commit
  `4cc64030d575723ae0ba7194e0cc72bb87cf7ca4` and OpenPI commit
  `bd70b8f4011e85b3f3b0f039f12113f78718e7bf`, ran from
  `2026-07-08T20:05:46Z` until user cancellation at
  `2026-07-08T20:09:37Z`. It prepared 242 packages but did not finish
  installation or independently reach the package-RECORD verifier.
- The cancellation followed current setup `1101808` failing on the shared
  pinned `augmax==0.4.1` integrity gate. The historical log is
  `/lustre/fsw/portfolios/nvr/users/lzha/slurm_logs/polaris-pi05/pi05-confidence-final-v4-20260708T200401Z/setup-historical-1101809.out`,
  5,498 bytes, SHA-256
  `7ab30ee0db94b54effeee9529f1bc3fb8aaeefbd3da55adcfd3e92147003f4e8`.
  Its setup-record directory is empty and no GPU evaluation launched.
- The current-arm forensic result identifies an upstream RECORD-encoding
  defect in the exact locked wheel, not an observed historical-only outcome.
  The same fail-closed, exact-entry correction will be copied byte-identically
  before another paired setup. There are no active jobs or namespace
  processes from this attempt.
- The final package-integrity correction is copied byte-identically from the
  independently reviewed current arm. It pins the exact Augmax and both
  OpenCV Linux x86_64 wheel artifacts; requires all three distributions to be
  installed at their exact versions; permits the malformed hexadecimal hash
  encoding only for Augmax's exact 11 path/size/digest tuples; and explicitly
  seals the three `opencv-python-headless` claims whose live files are the
  exact co-installed `opencv-python` winner bytes. Missing/extra entries,
  reversed install order, path/version/size/digest drift, or any other overlap
  fails closed. The complete clean-environment census found no discrepancy
  outside those pinned cases.
- The historical source and contract-test blobs exactly match current commit
  `9e6a7ce60494594d54d268db03ea939480297833`. This package-only attestation
  correction does not touch the intentionally historical evaluator lifecycle.
  Its complete host-safe suite passes 131 tests with one unrelated skip; Ruff
  format/lint, Python compilation, and Git whitespace checks pass.

## 2026-07-08 — deterministic OpenCV provider setup

- Parallel v5 setup proved that the full/headless OpenCV last writer was not
  deterministic. Historical job `1101811` ended with the locked full
  `opencv-python` files active, passed the complete 242-package/240-RECORD
  audit, tokenizer generation/bytes/proto audit, all 27 checkpoint objects and
  full MD5 verification, global DROID norm hash, and official config check.
  It completed `0:0` in 9m11s. Its package environment canonical digest is
  `3f8b25f9cc6472f3879d6c0f9996ca9e676ed33ba3ce5071dc1e91581bcc8f04`;
  the JSON file SHA-256 is
  `f15dfbe85464600152ef3beea89c4d7be8ba8ca7afa8b28c7babf20c2cabff86`.
  The terminal log SHA-256 is
  `a94113a89b3a011c2e232fa52c80b5b00b97d855bcb098496c87413eec6c6afa`.
- Simultaneous current job `1101810`, using the byte-identical verifier and
  same OpenPI lock, ended with the headless files active and correctly failed
  the required full-provider profile. This paired evidence establishes an
  installer-order ambiguity rather than checkpoint, branch, or content drift.
- The setup script now performs a frozen, no-cache, copy-mode reinstall of the
  direct OpenPI dependency `opencv-python` after the full sync and before the
  editable OpenPI reinstall and strict audit. A target-surface diagnostic
  transformed the headless-active failed environment into the exact required
  state: 130 full-wheel direct claims plus 95 headless direct claims and the
  three pinned loser-to-winner resolutions. The full and headless packages
  both remain installed at `4.11.0.86`; no dependency is removed or exempted.
- Although historical v5 setup artifacts are valid, the paired evaluation will
  use a fresh tree and setup from this deterministic script so both arms have
  the same setup provenance. The setup/test blobs remain byte-identical to the
  current arm, and the evaluator lifecycle remains intentionally historical.
- Current-arm canary `1101815` then exposed a separate pre-rollout runtime pin:
  the allocated L40S node uses NVIDIA driver `580.105.08` and Vulkan ICD
  `1.4.312`, SHA-256
  `7bdb6f27d35b66fc848df6f94b8773bba30ea3a7f06f114100d14154a235a34b`,
  while the joint-position wrapper still expected the older login-host
  `1.4.303` file. The job failed after five seconds before server, model, or
  simulator startup; no scientific result was produced.
- The paired evaluator now requires exactly one visible driver row equal to
  `580.105.08` plus the exact upgraded ICD bytes, records both in its sealed run
  evidence, and mounts the ICD read-only. This matches existing successful
  PolaRiS L40S controller/image/runtime evidence and changes no model,
  observation, action, normalization, or controller semantics. An independent
  adversarial review found no blocking issue. The eval/test blobs are again
  byte-identical to the current arm; the historical evaluator lifecycle remains
  the only intended execution difference.

## 2026-07-08 — final paired clean setup retry

- Historical v6 setup job `1101813` was externally canceled at 14:00:20 PDT
  after 14:10, before setup completion. It consumed no GPU and produced no
  policy rollout; no cancellation reason is claimed by this record.
- Independent L40S runtime probe `1101818` then completed successfully on
  `pool0-00016`: the exact upgraded ICD was mounted into the pinned image and
  `SimulationApp` started and closed cleanly. Its 44,079-byte log has SHA-256
  `82bedb51d14d8b6c268438d4e5c5e601397759ca0d461b36603cac45c0330c7f`.
- Setup jobs `1101819` and `1101820` failed closed in 3 and 2 seconds because
  the orchestrator pre-created the supposedly fresh `SETUP_RECORD_DIR` paths.
  The setup script rejected them before package work, simulator use, or GPU
  rollout; this submission-only error is retained in the attempt history.
- The corrected untouched-path pair is current job `1101822` and historical
  job `1101823` under namespace
  `pi05-confidence-final-v8-20260708T210352Z`. Historical uses exact commit
  `68899317fa8dfe8c86abe47a59e2f75e37abfd6a`, tree
  `c6abf07fb21d111d1fc26ab332caf09ca9afa1ff`, and OpenPI
  `bd70b8f4011e85b3f3b0f039f12113f78718e7bf`; current uses commit
  `1dcdde6fe702a3966110abb77e6022a79de53871`, tree
  `9ed22d52879e162b0c0ef20a1b5f534115d17cd5`. The immutable submission
  manifest SHA-256 is
  `4f0d56d0670f56fa0b27f75bcdee36dd42ba1fde62681f5cdd795bb69bca535e`.

## 2026-07-08 — v8 external cancellation

- Current setup `1101822` and historical setup `1101823` were canceled
  simultaneously by user UID `158351` after 3:46. The orchestrator and both
  assigned agents deny any cancellation or other mutating cluster command, and
  no matching Codex tool call was found at the exact timestamp. Historical had
  prepared 242 packages but had not installed or attested them. Its setup-record
  directory is empty, it used no GPU, and it produced no rollout. Historical
  log SHA-256 is
  `e8496038f3c53ef4a0d7df8bf347a2e0fb77b4a2dc80152ce93b3f5b4d538e2e`
  (5,498 bytes); current log SHA-256 is
  `7a581e5bd9bbaeb4c463d35c1b6e9c7e43b126c4dbd9ff6074b49189927b85df`
  (12,280 bytes).
- The paired setup was serialized to isolate cancellation ownership.
  Current-only `1101824` completed `0:0` in 8:51 under
  `pi05-confidence-final-v9-20260708T211229Z`, validating all 242 packages,
  checkpoint, tokenizer, DROID normalization, and policy config. It remains
  setup-only evidence from pre-attestation commit `1dcdde6` and cannot support
  final evaluation. Historical clean setup still requires a fresh launch from
  the final commit. Registry revision 17 preserves all failed/canceled history
  and leaves scientific result counts unchanged.

## 2026-07-08 — mapped graphics-runtime parity

- Source commit `2d0597f` applies the exact seven source/test blobs from current
  commit `1684777`; the historical evaluator lifecycle remains the only intended
  execution difference. Runtime schema 3 now seals the exact L40S GPU, driver,
  Vulkan ICD, and the 14 mapped Vulkan/NVIDIA ELF objects identified by bounded
  probe `1101825`. The stable graphics-runtime digest is
  `cd0ae19f2ea2cbdd0b8371796acad34c6d1b36d38c26aca68e8715b663c2f9f5`.
- A fresh historical setup and live full-horizon canary are mandatory. No result
  from the earlier source revisions will be reused as final evidence.

## 2026-07-08 — final mapped-runtime setup launch

- Independently reviewed source/test commit `2d0597f99cb2e50fe0f05c7c0e6fb17abbdf1650`
  is launched through documentation tip `bcc2df988ee0cfaca388bf5eaa157d834cafdab3`
  (tree `1ec2d26c5a1f64132e48e0f4ae26eea977bfcc57`) from the clean frozen
  checkout
  `/lustre/fsw/portfolios/nvr/users/lzha/src/PolaRiS-pi05-confidence-historical-bcc2df9-20260708T213230Z`.
- Fresh CPU setup job `1101828` is running under namespace
  `pi05-confidence-final-v10-20260708T213230Z` with immutable submission and
  ownership records. Current job `1101827` runs in parallel from its exact
  paired commit. Both jobs prepared all 242 locked packages; neither is a GPU
  evaluation and neither has produced a rollout result.
- Shared registry revision 20 records both running jobs and the exact
  mapped-runtime contract. The source/test runtime blobs remain byte-identical
  to current; the historical evaluator lifecycle is still the only intended
  paired factor.

## 2026-07-08 — mapped-runtime closure port and final setup gate

- Current-only setup `1101824` completed `0:0` and revalidated the complete
  242-package OpenPI environment, official checkpoint, tokenizer, global DROID
  normalization, and config. It is setup-only and superseded for evaluation
  because its source commit predates mapped-library attestation; it produced no
  rollout.
- L40S closure probe `1101825` completed `0:0` and sealed all 778 mapped ELF
  objects. The canonical 375,900-byte report has SHA-256
  `9c870463dfad67b23526a64ab044e7203a60a7730d3aaa25ffbc6369aa058207`.
  The final runtime contract pins the image-bundled Vulkan loader and all 13
  injected NVIDIA driver objects by path, size, SHA-256, and GNU build ID,
  checks live map device/inode identity, and runs the simulator under an
  explicit `env -i` baseline with loader/layer/preload overrides forbidden.
- Commit `2d0597f` ports the reviewed closure to the historical arm. All seven
  evaluator/runtime/evidence/test blobs are byte-identical to current commit
  `1684777`; the historical evaluator lifecycle remains the only intended
  execution difference. The stable graphics-runtime digest is
  `cd0ae19f2ea2cbdd0b8371796acad34c6d1b36d38c26aca68e8715b663c2f9f5`.
  Historical host-safe regression is 176 passed with all eight subtests
  passed; Ruff, Bash syntax, compilation, and whitespace checks pass.

## 2026-07-08 — final setup completion and paired canary launch

- Historical setup job `1101828` completed `0:0` in 8:14 on
  `cpu-large-0002`. It revalidated 242 installed distributions, all 240
  noneditable RECORD inventories, the 27-object/12,434,530,837-byte checkpoint
  with full MD5, tokenizer, global DROID normalization SHA-256
  `57ce9956f9e07d65f8a8205aabec72d436a2c8927f53edb40c7a77b14a5a90c7`,
  and `pi05_droid_jointpos_polaris`. Its package-environment canonical
  SHA-256 is
  `2acc79a45c9cdd7ef2242f95fbcbbac9ca60fd7f82723492bc1ff01de0db4a1c`;
  the JSON, setup manifest, and terminal log SHA-256 values are respectively
  `7ed2524d452991afedfc3e57ce1a4c3c1e4c62867c6759c5b4f4241d4ffbd4ef`,
  `aa21cdd7d30d8daf13e78b990f7576f78d7d292823c863499ac354bb5c9407ec`,
  and `d9db8511449ff7f7cf9b3bf6e48ad13fa6a84dc602ec54f589c81bf841fc32f3`.
  Current setup `1101827` independently completed `0:0` in 6:15 with the same
  scientific gates.
- A byte-level parity audit found the same 242 distributions and required
  versions in both environments. Of 72 differing RECORD paths, 65 console
  scripts differed only by their exact virtual-environment prefix, four
  editable OpenPI/OpenPI-client files differed only by their expected frozen
  checkout paths, and three locally compiled `evdev` objects differed only in
  debug sections and GNU build IDs. Their executable and data sections are
  identical, and neither evaluated OpenPI nor PolaRiS source imports `evdev`.
  This closes the clean-environment parity gate without changing the policy.
- Public-submitter full-horizon seed-0 historical canary `1101830` launched
  from the exact clean frozen historical checkout on one L40S under namespace
  `pi05-confidence-historical-canary-v10-20260708T213230Z`. Current canary
  `1101829` runs on a separate L40S. Both request one rollout and use the same
  official checkpoint, global DROID normalization, three image-slot contract,
  720x1280 client images with server-side 224x224 resize, 15x8 absolute
  joint-position output, eight-action execution horizon, and seed 0. They are
  runtime-validation canaries, not final scientific results.

## 2026-07-08 — v10 canary failure and import-time mutation root cause

- Historical canary `1101830` and current canary `1101829` both failed before
  simulator/container launch and produced zero episodes, traces, or videos.
  They restored the official checkpoint and loaded the global DROID
  normalization successfully, then failed complete package RECORD
  re-attestation. This is an infrastructure failure, not a model result.
- Locked `numpydantic==1.6.9` deterministically rewrote its RECORD-tracked
  `numpydantic/ndarray.pyi` during OpenPI import: the original wheel file is
  705 bytes with SHA-256
  `36e9708637fe45a17da721ff308ba7ba5f4f1ac7dda1ce7eeec615b29097ee00`,
  while the generated file is 452 bytes with SHA-256
  `954af1cf45ab82657347f8155fb26eef2ed6996e4a9dda915f9f83186a314cc4`.
  No other RECORD-tracked environment file changed. Because v10 setup itself
  imported OpenPI after its initial package report, both v10 environments are
  retired and will not be reused.
- Shared registry revision 23 preserves both failed v10 objects and their
  exact artifacts. They contribute no rollout or success-rate observation.

## 2026-07-08 — byte-exact v11 package-integrity closure

- Historical fix commit
  `4fa1d165def1f049a196c16eef6e976bc25b53c5` is byte-identical across all
  seven evaluator/setup/serving/evidence/test blobs to current commit
  `3178c03ba8632c2eb650e605460707bcdc010400`. The historical evaluator
  lifecycle remains the only intended paired factor.
- The fix pins the exact numpydantic wheel, RECORD claim, original stub bytes,
  and observed fresh-install modes `0640`/`0644`, then seals only the typing
  stub to canonical `0444`. No-follow descriptor reads, path/inode stability,
  one-link ownership, and non-root execution are required. Setup reattests the
  full package environment before sealing, after sealing, and after all
  tokenizer/checkpoint/config imports. Serving and the outer evaluator add
  matching pre/post checks, bound by model-runtime v3 and evidence profile v6.
- Historical validation is `183 passed` with all eight subtests passing for
  the host-safe suite and `50 passed` for focused serving/evidence tests. Ruff
  lint/format, Bash syntax, Python compilation, and whitespace checks pass.
  The sole excluded test needs Isaac and remains assigned to the paired live
  L40S canary.
- Only fresh v11 frozen source trees, unique setup/result namespaces, and
  rebuilt virtual environments are valid. Paired one-rollout canaries gate the
  paired FoodBussing50 evaluation.

## 2026-07-08 — fresh v11 setup launch

- Shared registry revision 24 records the exact reviewed v11 commits and
  contracts before launch. Clean frozen checkout
  `/lustre/fsw/portfolios/nvr/users/lzha/src/PolaRiS-pi05-confidence-historical-a126bb3-20260708T221428Z`
  has launch commit `a126bb3fc66de2afed4a091b2b620abfca52084e`, tree
  `cb8703904e808e32b02b4b4d321e76583637e25e`, OpenPI
  `bd70b8f4011e85b3f3b0f039f12113f78718e7bf`, and no preexisting `.venv`.
- Fresh CPU setup job `1101832` is running on `cpu-large-0002` under namespace
  `pi05-confidence-final-v11-20260708T221428Z`; paired current setup `1101831`
  runs independently on `cpu-large-0001`. The exact immutable submission
  script SHA-256 is
  `4357677926fbe337d006e93b5f564e2aae389aa74eeb11927fd5458eee72b506`.
  These jobs perform package/checkpoint setup only and contribute zero rollout
  episodes. GPU canaries remain gated on both setup records completing.

## 2026-07-08 — v11 setup completion and pre-rollout graphics rejection

- Historical setup `1101832` completed `0:0` in 13:07 and current setup
  `1101831` completed `0:0` in 10:27. Both verified 242 distributions, all
  240 noneditable RECORD inventories, the 27-object/12,434,530,837-byte
  checkpoint with full MD5, tokenizer, global DROID normalization, and the
  official `pi05_droid_jointpos_polaris` config. Their final package canonical
  SHA-256 values are respectively
  `560f9a7f1650a4c4cc0a02b8ceb24e86c0b1b4693709ae274276f6930f22224c`
  and
  `989bc5a3ad4b7c3d5bebeb81b47cec51170a7b9e8f50827d72672ce934bedd62`;
  each sealed report is byte-identical to its final report.
- Paired one-rollout canaries `1101836` (historical) and `1101835` (current)
  passed package, checkpoint, tokenizer, model-runtime v3, serving, GPU,
  driver, and Vulkan gates, then failed closed after DROID environment reset
  but before the first policy request. The old exact mapped-library table was
  missing only
  `/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.580.105.08`.
  Both jobs were subsequently canceled by another UID-158351 process while
  Isaac teardown was pending. Each remains a non-authoritative zero-episode
  infrastructure failure: no runtime JSON, policy trace, CSV, RNG proof,
  video, terminal PNG, evidence-v6 manifest, or success marker exists.
- The historical failure log is 19,737 bytes with SHA-256
  `e2a9dcd9759235c5022f1d1378cb804ce625d613bc73460bf6888dad952c6fb4`.
  The current arm independently produced the same exception; its 19,601-byte
  log has SHA-256
  `78bca931f646403f3613e893216908f9f826a43b24978dd431c46c2b90dc8d19`.
  External cancellation bypassed normal marker finalization, so stale
  `RUNNING` files are explicitly invalid and do not denote active or
  successful evaluation.

## 2026-07-08 — lazy PTX runtime closure and fresh v12 setup launch

- The source probe for the prior 14-library table constructed only a bare
  `SimulationApp`. A real DROID reset performs Warp/CUDA PTX work and lazily
  maps the driver-matched PTX JIT compiler. Live process maps and the mounted
  file bind that fifteenth object to 39,422,584 bytes, SHA-256
  `1ed129c4f703547fe5f8961dada7d53cb2981404fabdbfa9b9b3e3d83a04f6ac`,
  and GNU build ID `6257a5b3887eab41edd54343ea3623c373ab8e8e`.
- Historical fix commit `449ab6465450a871221803d529f3febb92e5d62b` retains
  exact-set rejection, adds only that pinned identity, advances the graphics
  profile to `l401_pyxis_nvidia_580_105_08_mapped_graphics_v2`, and binds
  canonical digest
  `f3ee6c8027f0cfea3c0f4875c2d3c0aba4c8cf41f8bde040a0bf236b81133a84`.
  Historical and current runtime/test blobs are byte-identical at
  `41e679ae754fa2037190ec9613a95c386aca5baf` and
  `c65935ff5da3623bda75236a733e250589ef61e9`. Host validation remains 183
  passed plus eight subtests for historical and 487 plus eight for current;
  the focused runtime/evidence/serving set passes 89/89 on each arm.
- Shared registry revision 30 records both v11 failures and the v12 prelaunch
  contract. Fresh checkout
  `/lustre/fsw/portfolios/nvr/users/lzha/src/PolaRiS-pi05-confidence-historical-449ab64-20260708T225010Z`
  is clean at tree `e25b051d18268081c91a61b11bd2519c7fd6e2bc`; the paired
  current checkout is clean at commit
  `61cfddc5773e35ee355eccfdf76cdc3f9aebf1f1`. CPU-only setup jobs
  `1101842` and `1101841` were submitted under
  `pi05-confidence-final-v12-20260708T225010Z` using immutable script SHA-256
  `a31ebd3c0ffc687bcbe51fcc0d5e77d4aca2cdf4259dd826f8454de83dd0cd9b`.
  No v12 GPU evaluation is launched until both setup records pass.
