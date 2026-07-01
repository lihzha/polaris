# PolaRiS development worklog

## 2026-06-30 — Ego-LAP absolute EEF-pose evaluation integration

- Agent: `implement_polaris_eef`; branch
  `codex/ego-lap-eef-eval-20260630`; base commit `2f4046b`.
- Goal: add a selectable absolute end-effector pose evaluation path for
  Ego-LAP without changing the existing joint-position default.
- Controller/observation change: added Isaac Lab absolute differential IK for
  Robotiq `base_link` (`pose`, `use_relative_mode=False`, DLS, scale 1) and
  exposed `panda_link0 -> base_link` position/quaternion observations.
- Client change: registered `EgoLAPEefPose`; added DROID external/wrist image
  mapping with 180-degree wrist rotation, 10-D `xyz + rot6d + open-gripper`
  state, strict finite `T x 7` response validation, one-anchor chunk conversion,
  SciPy `xyzw` to Isaac `wxyz` conversion, open-to-closed gripper inversion,
  full-step splat visualization, and optional JSONL traces.
- Evaluation change: controller selection now occurs before `gym.make`; known
  policy/controller mismatches fail early; instruction overrides are honored;
  initial-condition indexing no longer repeats an episode; and videos request
  one frame per executed step.
- Supporting change: added a headless scripted axis/rotation controller smoke,
  pure unit tests, dependency metadata, and usage documentation.
- CPU validation:
  - `PYTHONPATH=... /home/lzha/code/ego-lap/.venv/bin/python -m unittest discover -s tests -v`:
    6 tests passed.
  - `ruff check` on all changed Python files: passed.
  - `ruff format --check` on all changed Python files: passed.
  - `python3 -m py_compile` on all changed Python files: passed.
  - `git diff --check`: passed.
- Dependency note: SciPy 1.15.3 was already resolved in `uv.lock`; it is now a
  direct PolaRiS dependency for quaternion and rotation conversion. A fresh
  `uv lock` remains blocked by upstream packaging: the broad Python matrix
  cannot resolve Isaac Lab 2.3 on unsupported interpreters, and a 3.11-only
  attempt reaches `flatdict==4.0.1`'s undeclared `pkg_resources` build need.
  The existing resolved versions were therefore preserved and only the
  PolaRiS-to-SciPy lock edges were added.
- Execution status: no Isaac Sim, policy-server, GPU, or cluster job was
  launched, as requested. The controller smoke and an end-to-end rollout remain
  target-runtime validation steps.
