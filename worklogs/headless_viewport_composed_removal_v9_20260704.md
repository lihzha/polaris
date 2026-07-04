# Headless viewport composed-removal smoke v9 — 2026-07-04

Owner: `codex-root-polaris-composed-removal-v9`

## Runtime evidence

- Exact model-free smoke job `1098823` ran PolaRiS `38153722deecdd4af34d0926f6d095988c9a2306` and Ego-LAP `cf287750c9e5991514bba67e19ac83a972754901` on one L40S.
- The corrected lifecycle worked: the previously hidden simulator exception was printed and preserved in a sealed failure raw JSON; the inner and outer jobs correctly ended nonzero.
- Failure: `UsdStage.RemovePrim('/OmniverseKit_Persp')` removed only the current edit-target opinion. USD immediately recomposed the camera from another contributing layer, so the smoke's deliberate-absence assertion fired before the production recovery guard inspected the prim.
- This matches OpenUSD's documented `RemovePrim` semantics: it is layer-local and does not necessarily remove a composed prim.

## Change

- Change only the model-free smoke's forced-removal mechanism.
- Capture the complete composed prim stack for the root camera prim and call each contributing layer's `RemoveRootPrim` on its exact root `SdfPrimSpec`.
- Fail if the camera was initially absent, the prim stack is empty, a spec path differs, or any composed camera opinion remains afterward.
- Preserve the production viewport guard, camera definition, environment, evaluator, controller, assets, and all policy/checkpoint semantics byte-for-byte.
- Emit the removed-spec count to the Slurm log for diagnosis; successful raw/ready schemas remain unchanged.

## Validation

- Focused host tests: 13 passed; broad host suite: 977 passed plus 30 subtests with one pre-existing warning.
- Tests prove all composed root specs are removed and that a lingering opinion fails closed.
- Python compilation, Ruff, formatting, and `git diff --check`: passed.

## Next gate

Run the broad PolaRiS host suite, commit, update the exact Ego-LAP provenance pins/result namespace, obtain independent review, deploy both commits through verified Git bundles, and rerun one fresh model-free L40S smoke. No checkpoint rollout is authorized.
