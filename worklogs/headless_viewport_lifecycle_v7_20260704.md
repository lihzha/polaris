# Headless viewport recovery lifecycle v7 — 2026-07-04

Owner: `codex-root-polaris-recovery-v7`

## Goal

Close the post-`SimulationApp.close()` evidence bug found by the exact L40S forced-recovery smoke without changing the production camera guard or any evaluation semantics.

## Failed runtime evidence

- PolaRiS producer commit: `8db95a83cdfffde358893af1aced433f1029b65c`
- Ego-LAP wrapper commit: `76dbfcadbe7262e91d660e1d45461e7c345b5d83`
- Slurm job: `1098821`, one L40S on `pool0-00029`
- Result: producer `srun` completed `0:0`, but the batch job failed `2:0` after 125 seconds because no raw JSON existed.
- Exact log terminal: `VIEWPORT_SMOKE_SRUN_RC=0` followed by `ERROR: successful producer did not publish one sealed raw result`.
- Root cause: the smoke called `simulation_app.close()` before `_publish(...)`. In the pinned Isaac runtime, `SimulationApp.close()` hard-exits the process, so publication after it is unreachable.

## Change

- Publish the success raw JSON as immutable mode `0444`, with content and directory fsync plus immediate readback.
- Bind the raw path, size, SHA-256, and mode in a separate immutable `.ready.json` marker.
- Mark the raw lifecycle stage `simulation_app_close_pending`.
- Flush producer diagnostics, publish the ready marker, and call `SimulationApp.close()` immediately afterward.
- Hard-exit nonzero on any producer, environment-close, or publication failure so the pinned runtime cannot turn an error into a successful `srun`.
- Add source-order and live filesystem publication tests, including non-overwrite behavior.

## Validation

- `python3 -m py_compile scripts/smoke_headless_viewport_camera_recovery.py`: passed.
- `ruff check` and `ruff format --check` on changed Python: passed.
- `tests/test_headless_viewport.py`: 10 passed.
- Broad host suite excluding the one Isaac-Lab-only import test: 973 passed, 30 subtests passed, one pre-existing warning.
- Full unfiltered host collection is unavailable because host Python has no `isaaclab`; this is covered by the exact L40S runtime smoke after the matching host-side wrapper/finalizer is updated.

## Next gate

Update and independently test the Ego-LAP wrapper/finalizer to require the ready marker, deploy both commits by Git bundle, and rerun one model-free L40S smoke. No checkpoint evaluation is authorized until the post-exit attestation finalizes and re-verifies byte-for-byte.
