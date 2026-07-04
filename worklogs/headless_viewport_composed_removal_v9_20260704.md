# Headless viewport composed-removal smoke v9 — 2026-07-04

Owner: `codex-root-polaris-composed-removal-v9`

## Runtime evidence

- Exact model-free smoke job `1098823` ran PolaRiS `38153722deecdd4af34d0926f6d095988c9a2306` and Ego-LAP `cf287750c9e5991514bba67e19ac83a972754901` on one L40S.
- The corrected lifecycle worked: the previously hidden simulator exception was printed and preserved in a sealed failure raw JSON; the inner and outer jobs correctly ended nonzero.
- Failure: `UsdStage.RemovePrim('/OmniverseKit_Persp')` removed only the current edit-target opinion. USD immediately recomposed the camera from another contributing layer, so the smoke's deliberate-absence assertion fired before the production recovery guard inspected the prim.
- This matches OpenUSD's documented `RemovePrim` semantics: it is layer-local and does not necessarily remove a composed prim.
- Follow-up L40S smoke job `1098827` ran the first composed-spec implementation and failed closed before the production guard: the pinned Isaac Sim/OpenUSD Python binding does not expose `Sdf.Layer.RemoveRootPrim`.
- Exact-image probe job `1098833`, launched with the same Pyxis image, remap flags, and `AppLauncher`, proved the available mutation API: `Sdf.BatchNamespaceEdit`, `Sdf.NamespaceEdit.Remove`, and `Sdf.Layer.Apply` are present; `Sdf.Layer.RemoveRootPrim` is absent. The probe also proved that applying one remove edit to a contributing root spec changes its layer's `GetPrimAtPath` result from valid to absent.

## Change

- Change only the model-free smoke's forced-removal mechanism.
- Capture the complete composed prim stack for the root camera prim. For each exact contributing `SdfPrimSpec`, build a one-edit `Sdf.BatchNamespaceEdit` with `Sdf.NamespaceEdit.Remove(prim_spec.path)` and apply it to `prim_spec.layer`.
- Require every `Layer.Apply` result to be exactly `True` and require the composed stage camera to be absent after all edits.
- Fail if the camera was initially absent, the prim stack is empty, a spec path differs, or any composed camera opinion remains afterward.
- Preserve the production viewport guard, camera definition, environment, evaluator, controller, assets, and all policy/checkpoint semantics byte-for-byte.
- Emit the removed-spec count to the Slurm log for diagnosis; successful raw/ready schemas remain unchanged.

## Validation

- Focused host tests: 14 passed; broad host suite excluding the unavailable host-only Isaac-Lab import: 978 passed plus 30 subtests with one pre-existing warning.
- Tests prove all composed root specs are removed, a failed layer application is rejected, and a lingering opinion fails closed.
- Python compilation, Ruff, formatting, and `git diff --check`: passed.

## Next gate

Commit this exact-container-compatible follow-up, update the exact Ego-LAP provenance pins/result namespace, obtain independent review, deploy both commits through verified Git bundles, and rerun one fresh model-free L40S smoke. No checkpoint rollout is authorized.

## Terminal runtime evidence

- Independent review returned GO for one model-free smoke. Job `1098834` ran the exact producer commit on one L40S and all Slurm components plus inner `srun` completed `0:0`.
- The bound namespace-edit path removed exactly one contributing `/OmniverseKit_Persp` spec. The production guard recovered it exactly once, and the sealed raw records forced absence plus a valid camera afterward.
- Finalize and replay verification returned the identical mode-`0444` attestation SHA-256 `efded6682bce983a4d773b038990f9e9fd5968cd05efe42b204063c6b4c7b0c5`, status `sealed_validated_forced_l40s_recovery_v1`.
- Checkpoint rollout still requires a separate reviewed Ego-LAP integration commit that pins this exact evidence.
