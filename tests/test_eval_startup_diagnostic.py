from __future__ import annotations

import ast
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Iterator

import pytest

from polaris.config import EvalArgs, PolicyArgs


ROOT = Path(__file__).parents[1]
EVAL_SCRIPT = ROOT / "scripts/eval.py"
GPU_UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"


class FakeSimulationApp:
    pass


class FakeAppLauncher:
    instances: list["FakeAppLauncher"] = []

    def __init__(self, namespace):
        self.namespace = namespace
        self.app = FakeSimulationApp()
        self.instances.append(self)


@contextmanager
def evaluator_module(*, eval_mode: str | None = None) -> Iterator[ModuleType]:
    replacements = {
        name: ModuleType(name) for name in ("mediapy", "pandas", "torch", "tqdm")
    }
    isaaclab = ModuleType("isaaclab")
    isaaclab.__path__ = []
    isaaclab_app = ModuleType("isaaclab.app")
    isaaclab_app.AppLauncher = FakeAppLauncher
    replacements["isaaclab"] = isaaclab
    replacements["isaaclab.app"] = isaaclab_app
    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    module_name = "_polaris_eval_startup_diagnostic_test"
    previous_eval = sys.modules.get(module_name)
    previous_eval_mode = os.environ.get("POLARIS_EVAL_MODE")
    try:
        if eval_mode is None:
            os.environ.pop("POLARIS_EVAL_MODE", None)
        else:
            os.environ["POLARIS_EVAL_MODE"] = eval_mode
        specification = importlib.util.spec_from_file_location(module_name, EVAL_SCRIPT)
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        specification.loader.exec_module(module)
        yield module
    finally:
        if previous_eval_mode is None:
            os.environ.pop("POLARIS_EVAL_MODE", None)
        else:
            os.environ["POLARIS_EVAL_MODE"] = previous_eval_mode
        if previous_eval is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_eval
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def diagnostic_args(tmp_path: Path) -> EvalArgs:
    return EvalArgs(
        policy=PolicyArgs(
            client="DroidJointPos",
            trace_path=str(tmp_path / "policy_trace.forbidden"),
        ),
        environment="DROID-FoodBussing",
        run_folder=str(tmp_path),
        rollouts=1,
        environment_seed=0,
        control_mode="joint-position",
        runtime_contract_path=str(tmp_path / "runtime.forbidden"),
        startup_diagnostic="app_launcher_only",
        startup_diagnostic_preexec_path=str(tmp_path / "preexec.json"),
        startup_diagnostic_preclose_path=str(tmp_path / "preclose.json"),
        startup_diagnostic_expected_gpu_uuid=GPU_UUID,
    )


def test_diagnostic_branch_calls_public_app_launcher_then_returns_before_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    FakeAppLauncher.instances.clear()
    forbidden_prefixes = (
        "gym",
        "gymnasium",
        "isaaclab_tasks",
        "openpi",
        "openpi_client",
        "polaris.environments",
        "polaris.policy",
    )
    forbidden_before = {
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in forbidden_prefixes
        )
    }
    with evaluator_module(eval_mode="app_launcher_only") as evaluator:
        calls = []

        def run_diagnostic(**kwargs):
            calls.append(kwargs)

        import polaris.app_launcher_startup_diagnostic as diagnostic

        monkeypatch.setattr(
            diagnostic, "run_app_launcher_only_diagnostic", run_diagnostic
        )
        monkeypatch.setattr(
            evaluator,
            "_run_evaluation",
            lambda *_: pytest.fail("diagnostic branch called _run_evaluation"),
        )
        evaluator.main(diagnostic_args(tmp_path))
    forbidden_after = {
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in forbidden_prefixes
        )
    }
    assert forbidden_after == forbidden_before
    assert len(FakeAppLauncher.instances) == 1
    namespace = FakeAppLauncher.instances[0].namespace
    assert vars(namespace) == {"enable_cameras": True, "headless": True}
    assert len(calls) == 1
    assert calls[0]["simulation_app"] is FakeAppLauncher.instances[0].app
    assert calls[0]["expected_gpu_uuid"] == GPU_UUID
    assert capsys.readouterr().out.splitlines() == [
        "POLARIS_EVAL_PHASE=before_app_launcher",
        "POLARIS_EVAL_PHASE=after_app_launcher",
        "POLARIS_EVAL_PHASE=before_app_launcher_diagnostic_close",
        "POLARIS_EVAL_PHASE=after_app_launcher_diagnostic_close",
    ]


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("startup_diagnostic", "unknown", "unsupported startup diagnostic"),
        ("startup_diagnostic_preexec_path", None, "requires every"),
        (
            "startup_diagnostic_preclose_path",
            "relative.json",
            "paths must be absolute",
        ),
        (
            "startup_diagnostic_expected_gpu_uuid",
            "GPU-not-a-uuid",
            "expected GPU UUID is malformed",
        ),
    ],
)
def test_invalid_diagnostic_arguments_fail_before_app_launcher(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    FakeAppLauncher.instances.clear()
    args = diagnostic_args(tmp_path)
    setattr(args, field, value)
    with (
        evaluator_module(eval_mode="app_launcher_only") as evaluator,
        pytest.raises(ValueError, match=message),
    ):
        evaluator.main(args)
    assert FakeAppLauncher.instances == []


def test_diagnostic_arguments_are_rejected_when_mode_is_off(tmp_path: Path) -> None:
    FakeAppLauncher.instances.clear()
    args = EvalArgs(
        policy=PolicyArgs(client="Fake"),
        environment="DROID-FoodBussing",
        run_folder=str(tmp_path),
        startup_diagnostic_preexec_path=str(tmp_path / "unexpected.json"),
    )
    with evaluator_module() as evaluator, pytest.raises(ValueError):
        evaluator.main(args)
    assert FakeAppLauncher.instances == []


@pytest.mark.parametrize(
    "process_mode,argument_mode",
    [
        ("app_launcher_only", None),
        ("standard", "app_launcher_only"),
    ],
)
def test_process_and_argument_diagnostic_modes_must_match(
    tmp_path: Path,
    process_mode: str,
    argument_mode: str | None,
) -> None:
    FakeAppLauncher.instances.clear()
    args = diagnostic_args(tmp_path)
    args.startup_diagnostic = argument_mode
    with (
        evaluator_module(eval_mode=process_mode) as evaluator,
        pytest.raises(
            ValueError,
            match="must be selected together",
        ),
    ):
        evaluator.main(args)
    assert FakeAppLauncher.instances == []


def test_default_path_still_constructs_lifecycle_runs_evaluation_and_closes(
    tmp_path: Path,
) -> None:
    FakeAppLauncher.instances.clear()
    lifecycle_module = ModuleType("polaris.pi05_droid_native_lifecycle")
    events = []

    class FakeLifecycle:
        def __init__(self, app):
            events.append(("lifecycle", app))

        def close(self):
            events.append(("close", None))

    lifecycle_module.NativeEvaluatorLifecycle = FakeLifecycle
    previous = sys.modules.get("polaris.pi05_droid_native_lifecycle")
    sys.modules["polaris.pi05_droid_native_lifecycle"] = lifecycle_module
    try:
        with evaluator_module(eval_mode="standard") as evaluator:
            evaluator._run_evaluation = lambda args, lifecycle: events.append(
                ("run", lifecycle)
            )
            result = evaluator.main(
                EvalArgs(
                    policy=PolicyArgs(client="Fake"),
                    environment="DROID-FoodBussing",
                    run_folder=str(tmp_path),
                )
            )
    finally:
        if previous is None:
            sys.modules.pop("polaris.pi05_droid_native_lifecycle", None)
        else:
            sys.modules["polaris.pi05_droid_native_lifecycle"] = previous
    assert result is None
    assert [event[0] for event in events] == ["lifecycle", "run", "close"]
    assert len(FakeAppLauncher.instances) == 1


def test_eval_ast_preserves_standard_gym_import_boundary() -> None:
    source = EVAL_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_gym_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and any(alias.name in {"gym", "gymnasium"} for alias in node.names)
    ]
    assert len(top_level_gym_imports) == 1
    gym_import = top_level_gym_imports[0]
    mode_guard = next(
        node
        for node in tree.body
        if isinstance(node, ast.If) and node.body == [gym_import]
    )
    assert mode_guard.body == [gym_import]
    process_mode_binding = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_PROCESS_EVAL_MODE"
            for target in node.targets
        )
    )
    assert "POLARIS_EVAL_MODE" in ast.unparse(process_mode_binding.value)
    assert "_PROCESS_EVAL_MODE" in ast.unparse(mode_guard.test)
    imports = {
        alias.name: node.lineno
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    # These are the original evaluator's adjacent scientific import slots.
    assert imports["tqdm"] < gym_import.lineno < imports["torch"]


def test_default_startup_validation_is_a_pure_noop(tmp_path: Path) -> None:
    args = EvalArgs(
        policy=PolicyArgs(client="Fake"),
        environment="DROID-FoodBussing",
        run_folder=str(tmp_path),
    )
    before = vars(args).copy()
    with evaluator_module(eval_mode="standard") as evaluator:
        assert evaluator._validate_startup_diagnostic_args(args) is None
    assert vars(args) == before


def test_eval_ast_places_closed_branch_at_exact_import_boundary() -> None:
    source = EVAL_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    app_launcher_line = next(
        node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "AppLauncher"
    )
    diagnostic_branch = next(
        node
        for node in main.body
        if isinstance(node, ast.If) and "startup_diagnostic" in ast.unparse(node.test)
    )
    lifecycle_import_line = next(
        node.lineno
        for node in main.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "polaris.pi05_droid_native_lifecycle"
    )
    run_call_line = next(
        node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_run_evaluation"
    )
    assert app_launcher_line < diagnostic_branch.lineno < lifecycle_import_line
    assert diagnostic_branch.end_lineno < run_call_line
    assert any(isinstance(node, ast.Return) for node in diagnostic_branch.body)


def test_diagnostic_module_has_only_stdlib_imports() -> None:
    path = ROOT / "src/polaris/app_launcher_startup_diagnostic.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots <= {
        "__future__",
        "argparse",
        "hashlib",
        "json",
        "os",
        "pathlib",
        "re",
        "signal",
        "stat",
        "subprocess",
        "sys",
        "time",
        "typing",
    }


def test_forbidden_work_names_do_not_appear_in_diagnostic_branch() -> None:
    tree = ast.parse(EVAL_SCRIPT.read_text(encoding="utf-8"))
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    branch = next(
        node
        for node in main.body
        if isinstance(node, ast.If) and "startup_diagnostic" in ast.unparse(node.test)
    )
    branch_source = ast.unparse(branch)
    for forbidden in (
        "NativeEvaluatorLifecycle",
        "_run_evaluation",
        "gym.make",
        "InferenceClient",
        "checkpoint",
        "tokenizer",
        "env.step",
        "env.reset",
    ):
        assert forbidden not in branch_source
